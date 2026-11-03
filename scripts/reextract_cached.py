"""Re-run Phase 2 extraction over cached HTML, without any HTTP.

Phase 2 only extracts rows whose ``content_text IS NULL``, so a fix to
``resources/extraction.py`` never reaches judgments that were already
extracted. This script re-parses each judgment's cached HTML under
``.cache/judgments_html/``, rewrites its extraction cache entry, updates
the ``judgments`` row and replaces its ``judgments_fragments`` rows.

Fragment ids are ``{judgment_id}_{ordinal}``, so they stay stable as long
as the fragment count does not change; the script reports any judgment
whose count moves. Phase 3 summaries are left alone.

Run:  uv run python scripts/reextract_cached.py [--dry-run] [--db PATH] [ID ...]
      (no IDs = every judgment with cached HTML)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import sqlite_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resources import extraction, extraction_cache  # noqa: E402
from resources.judgments import (  # noqa: E402
    FRAGMENTS_TABLE_NAME,
    _insert_fragments,
    _update_row,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="judgment ids (default: all cached)")
    parser.add_argument("--db", default=str(ROOT / "zeeker-judgements.db"))
    parser.add_argument("--dry-run", action="store_true", help="parse only, write nothing")
    args = parser.parse_args()

    ids = args.ids or sorted(
        p.name.split(".")[0] for p in extraction_cache.HTML_DIR.glob("*.html.gz")
    )
    db = sqlite_utils.Database(args.db)
    table = db["judgments"]
    known = {r["id"] for r in db.query("SELECT id FROM judgments")}

    done = moved = skipped = 0
    for jid in ids:
        html = extraction_cache.read_html(jid)
        if html is None or jid not in known:
            print(f"{jid}: skipped ({'no cached HTML' if html is None else 'no judgments row'})")
            skipped += 1
            continue
        try:
            extracted = extraction.extract_judgment(html, jid)
        except extraction.ExtractionError as exc:
            print(f"{jid}: skipped (extraction error: {exc})")
            skipped += 1
            continue

        old_count = db.execute(
            f"SELECT count(*) FROM {FRAGMENTS_TABLE_NAME} WHERE judgment_id = ?", [jid]
        ).fetchone()[0]
        new_count = len(extracted.fragments)
        if old_count != new_count:
            moved += 1
            print(f"{jid}: fragment count {old_count} -> {new_count}")
        if args.dry_run:
            done += 1
            continue

        now_iso = datetime.now().isoformat(timespec="seconds")
        extraction_cache.write_extraction_atomic(
            jid,
            {
                "judgment_id": jid,
                "extracted_at": now_iso,
                "content_text": extracted.content_text,
                "court_summary": extracted.court_summary,
                "has_content": extracted.has_content,
                "has_court_summary": extracted.has_court_summary,
                "fragments": extracted.fragments,
                "extraction_status": "ok",
            },
        )
        with db.conn:
            _update_row(
                table,
                jid,
                {
                    "content_text": extracted.content_text,
                    "court_summary": extracted.court_summary,
                    "has_content": 1 if extracted.has_content else 0,
                    "has_court_summary": 1 if extracted.has_court_summary else 0,
                    "fragment_count": new_count,
                    "extracted_at": now_iso,
                },
            )
            db.execute(f"DELETE FROM {FRAGMENTS_TABLE_NAME} WHERE judgment_id = ?", [jid])
            _insert_fragments(db, extracted.fragments)
        done += 1

    verb = "parsed" if args.dry_run else "re-extracted"
    print(f"{verb} {done}, skipped {skipped}, fragment count changed for {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
