#!/usr/bin/env python3
"""Standalone summary quality audit.

Scans all non-NULL summaries in the database, runs the smell test on each,
and reports issues. Can optionally fix fixable issues in-place (commentary
leaks, transcription artifacts, trailing incomplete items) and nullify
unfixable summaries (missing sections, mid-sentence truncation) for
regeneration on the next build.

Usage:
    uv run python scripts/audit_summaries.py            # report only
    uv run python scripts/audit_summaries.py --fix      # fix + nullify
    uv run python scripts/audit_summaries.py --json     # machine-readable output

Intended to be run manually or via cron before/after a build batch.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Ensure resources/ is on sys.path for sibling imports.
_RESOURCES = Path(__file__).resolve().parent.parent / "resources"
if str(_RESOURCES) not in sys.path:
    sys.path.insert(0, str(_RESOURCES))

from summarization import (
    _fix_transcription_artifacts,
    _strip_commentary,
    _trim_trailing_incomplete,
    smell_test,
)

DB_PATH = Path(__file__).resolve().parent.parent / "zeeker-judgements.db"


def audit(db_path: str = str(DB_PATH), fix: bool = False) -> list[dict]:
    """Run the smell test over all summaries. Returns a list of issue records.

    When *fix* is True, also applies the post-processing fixes (commentary
    strip, artifact fix, trailing trim) and nullifies summaries that still
    fail the smell test with structural issues.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, citation, case_name, court, fragment_count, summary
        FROM judgments
        WHERE summary IS NOT NULL
        ORDER BY citation
    """)
    rows = cur.fetchall()

    results = []
    fixed_count = 0
    nullified_count = 0

    for row in rows:
        summary = row["summary"]
        frag_count = row["fragment_count"] or 0
        citation = row["citation"] or row["id"]

        # Run the smell test first.
        result = smell_test(summary, fragment_count=frag_count)
        if result["passed"]:
            continue

        record = {
            "id": row["id"],
            "citation": citation,
            "court": row["court"],
            "fragment_count": frag_count,
            "word_count": len(summary.split()),
            "issues": result["issues"],
        }
        results.append(record)

        if not fix:
            continue

        # Apply post-processing fixes.
        modified = _fix_transcription_artifacts(_strip_commentary(summary))
        modified = _trim_trailing_incomplete(modified)

        if modified != summary:
            # Re-test after fixes.
            retest = smell_test(modified, fragment_count=frag_count)
            if retest["passed"]:
                cur.execute(
                    "UPDATE judgments SET summary = ? WHERE id = ?",
                    [modified, row["id"]],
                )
                fixed_count += 1
                record["action"] = "fixed"
                continue
            # Still failing after fixes — fall through to nullify.

        # Nullify for regeneration if the issue is structural.
        # "too short", "no structural heading", "ends without terminal punctuation"
        # are structural — can't be fixed by post-processing.
        structural_issues = [
            "too short", "no structural heading",
            "ends without terminal punctuation",
        ]
        if any(s in "; ".join(result["issues"]) for s in structural_issues):
            cur.execute(
                "UPDATE judgments SET summary = NULL, summary_generated_at = NULL WHERE id = ?",
                [row["id"]],
            )
            cache_path = (
                Path(__file__).resolve().parent.parent
                / ".cache" / "judgments_summaries" / f"{row['id']}.json"
            )
            if cache_path.exists():
                cache_path.unlink()
            nullified_count += 1
            record["action"] = "nullified"
        else:
            record["action"] = "unfixed"

    if fix:
        conn.commit()

    conn.close()

    if fix:
        print(f"Fixed {fixed_count} summaries, nullified {nullified_count} for regeneration.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Audit summary quality.")
    parser.add_argument(
        "--db", default=str(DB_PATH), help="Path to the SQLite database."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply post-processing fixes and nullify unfixable summaries.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (machine-readable).",
    )
    args = parser.parse_args()

    issues = audit(db_path=args.db, fix=args.fix)

    if args.json:
        print(json.dumps(issues, ensure_ascii=False, indent=2))
    else:
        if not issues:
            print("All summaries pass the smell test.")
        else:
            print(f"Found {len(issues)} summaries with issues:\n")
            for r in issues:
                action = f" → {r.get('action', 'unfixed')}" if args.fix else ""
                print(f"  {r['citation']} ({r['court']}, {r['word_count']} words){action}")
                for issue in r["issues"]:
                    print(f"    • {issue}")
                print()

    # Exit code: 0 if no issues, 1 if issues found (useful for CI).
    sys.exit(0 if not issues else 1)


if __name__ == "__main__":
    main()