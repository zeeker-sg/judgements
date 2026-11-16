"""One-off repair for rows poisoned before the 2026-11 fetch hardening.

A 2026-11 data survey (see RUNBOOK § "Maintenance-stub poisoning and the
legacy-template gap") found several classes of damaged rows in the live DB.
This script scans for them and applies the repairs. Everything is a plan
printout unless ``--apply`` is passed.

Repairs:
  stub-poisoned  has_content=0 rows whose cached HTML is eLitigation's
                 ":: Maintenance Notice ::" page (HTTP 200 stubs served
                 during maintenance windows). Reset the row so the next
                 build refetches it (purge both cache files).
  thin-stale     has_content=1 rows with <1000 chars of body text that are
                 not LawNet "citation reassigned" admin notes — the judgment
                 text was published after extraction (e.g. "Judgment
                 reserved"). Reset row + fragments + summary so the next
                 build refetches and re-summarises.
  duplicates     exact same-id duplicate rows (2026-04/06 double-inserts;
                 keep the newest created_at) and cross-id duplicates where
                 eLitigation serves one judgment under two slugs (keep the
                 non-zero-padded slug, delete the loser row + fragments +
                 cache files).
  empty-court    court='' rows whose citation still carries the court token
                 (e.g. "[2013] SGHC115" → SGHC).

Legacy-template rows (has_content=0, extraction status "ok", cached HTML
has content but no Judg-* classes) are NOT reset here: they are repaired by
re-parsing their cached HTML with the fixed extractor —
``uv run python scripts/reextract_cached.py --from-file <ids-file>``. Pass
``--emit-legacy-ids PATH`` to write that id file.

Run between cron build windows (the live DB is locked while a build runs):
  uv run python scripts/repair_poisoned_rows.py [--db PATH] [--apply]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# resources/judgments.py uses sibling imports (`import extraction`) that
# resolve only when resources/ itself is on sys.path — same as conftest.py
# and the zeeker resource loader do.
sys.path.insert(0, str(ROOT / "resources"))

from resources.judgments import CITATION_RE  # noqa: E402

STUB_MARKER = "eLitigation - Maintenance Notice"
THIN_MAX_CHARS = 1000
ADMIN_NOTE_MARKER = "LawNet Admin Note"
ZERO_PADDED_SLUG_RE = re.compile(r"_(0\d+)$")


def _cache_paths(jid: str) -> list[Path]:
    return [
        ROOT / ".cache" / "judgments_html" / f"{jid}.html.gz",
        ROOT / ".cache" / "judgments_extractions" / f"{jid}.json",
        ROOT / ".cache" / "judgments_summaries" / f"{jid}.json",
    ]


def _cached_html_is_stub(jid: str) -> bool:
    import gzip

    path = ROOT / ".cache" / "judgments_html" / f"{jid}.html.gz"
    if not path.exists():
        return False
    with gzip.open(path, "rt", errors="replace") as fh:
        return STUB_MARKER in fh.read(4000)


def _cached_status(jid: str) -> str:
    path = ROOT / ".cache" / "judgments_extractions" / f"{jid}.json"
    if not path.exists():
        return ""
    try:
        with open(path) as fh:
            return str(json.load(fh).get("extraction_status", ""))
    except (json.JSONDecodeError, OSError):
        return "corrupt"


def _reset_row_sql(jid: str, include_summary: bool) -> tuple[str, tuple]:
    cols = [
        "content_text = NULL",
        "court_summary = NULL",
        "has_content = NULL",
        "has_court_summary = NULL",
        "fragment_count = NULL",
        "extracted_at = NULL",
    ]
    if include_summary:
        cols += ["summary = NULL", "summary_generated_at = NULL"]
    return f"UPDATE judgments SET {', '.join(cols)} WHERE id = ?", (jid,)


def scan(db: sqlite3.Connection) -> dict:
    empty_rows = db.execute(
        "SELECT id, citation, source_url, created_at FROM judgments WHERE has_content = 0"
    ).fetchall()
    stub_ids = [r[0] for r in empty_rows if _cached_html_is_stub(r[0])]

    thin_rows = db.execute(
        """SELECT id, citation FROM judgments
           WHERE has_content = 1 AND length(content_text) < ?
             AND content_text NOT LIKE ?""",
        (THIN_MAX_CHARS, f"%{ADMIN_NOTE_MARKER}%"),
    ).fetchall()

    same_id_dups: list[tuple[str, list[tuple]]] = []
    for jid, *_ in db.execute(
        """SELECT id, COUNT(*) c FROM judgments GROUP BY id HAVING c > 1"""
    ).fetchall():
        rows = db.execute(
            "SELECT rowid, citation, created_at FROM judgments WHERE id = ? ORDER BY created_at",
            (jid,),
        ).fetchall()
        same_id_dups.append((jid, rows))

    cross_id_dups: list[tuple[str, list[tuple]]] = []
    for (citation,) in db.execute(
        "SELECT citation FROM judgments GROUP BY citation HAVING COUNT(DISTINCT id) > 1"
    ).fetchall():
        rows = db.execute(
            "SELECT id, source_url, created_at FROM judgments WHERE citation = ?",
            (citation,),
        ).fetchall()
        cross_id_dups.append((citation, rows))

    empty_court = [
        (rid, citation)
        for rid, citation in db.execute(
            "SELECT id, citation FROM judgments WHERE court IS NULL OR court = ''"
        ).fetchall()
        if CITATION_RE.search(citation or "")
    ]
    legacy_ids = [r[0] for r in empty_rows if r[0] not in stub_ids and _cached_status(r[0]) == "ok"]
    return {
        "stub_ids": stub_ids,
        "thin_rows": thin_rows,
        "same_id_dups": same_id_dups,
        "cross_id_dups": cross_id_dups,
        "empty_court": empty_court,
        "legacy_ids": legacy_ids,
    }


def _losers_of_cross_id_group(rows: list[tuple]) -> list:
    """Keep the non-zero-padded slug (2015_SGHCR_2 over 2015_SGHCR_02); all
    other rows in the group are losers."""

    def sort_key(row):
        url = row[1] or ""
        zero_padded = 1 if ZERO_PADDED_SLUG_RE.search(url) else 0
        return (zero_padded, row[2] or "")

    return [row[0] for row in sorted(rows, key=sort_key)[1:]]


def apply_repair(db: sqlite3.Connection, found: dict, apply: bool) -> None:
    def do(sql: str, args: tuple, desc: str) -> None:
        if apply:
            db.execute(sql, args)
        print(f"  {'DONE' if apply else 'PLAN'}: {desc}")

    print(f"== stub-poisoned rows to reset (refetch on next build): {len(found['stub_ids'])}")
    for jid in found["stub_ids"]:
        do(*_reset_row_sql(jid, include_summary=False), desc=f"{jid} reset")
        for p in _cache_paths(jid)[:2]:
            if apply and p.exists():
                p.unlink()
    print(f"== thin-stale rows to reset (refetch + re-summarise): {len(found['thin_rows'])}")
    for jid, citation in found["thin_rows"]:
        do(*_reset_row_sql(jid, include_summary=True), desc=f"{jid} ({citation}) reset")
        do(
            "DELETE FROM judgments_fragments WHERE judgment_id = ?",
            (jid,),
            desc=f"{jid} old fragments deleted",
        )
        for p in _cache_paths(jid):
            if apply and p.exists():
                p.unlink()
    print(
        f"== same-id duplicate rows to delete (keep newest created_at): "
        f"{sum(len(rows) - 1 for _, rows in found['same_id_dups'])}"
    )
    for jid, rows in found["same_id_dups"]:
        for row in rows[:-1]:
            do(
                "DELETE FROM judgments WHERE rowid = ?",
                (row[0],),
                desc=f"delete dup of {jid} ({row[1]}, created {row[2]})",
            )
    print(f"== cross-id slug duplicates to delete: {len(found['cross_id_dups'])}")
    for citation, rows in found["cross_id_dups"]:
        for loser in _losers_of_cross_id_group(rows):
            do("DELETE FROM judgments WHERE id = ?", (loser,), desc=f"delete {loser} ({citation})")
            do(
                "DELETE FROM judgments_fragments WHERE judgment_id = ?",
                (loser,),
                desc=f"delete fragments of {loser}",
            )
            for p in _cache_paths(loser):
                if apply and p.exists():
                    p.unlink()
    print(f"== empty-court rows to fix from citation: {len(found['empty_court'])}")
    for jid, citation in found["empty_court"]:
        court = CITATION_RE.search(citation).group(2).upper()
        do(
            "UPDATE judgments SET court = ? WHERE id = ?",
            (court, jid),
            desc=f"{jid} ({citation}) court={court}",
        )
    if apply:
        db.commit()
    print(
        f"\n== legacy-template rows repaired separately via reextract_cached.py: "
        f"{len(found['legacy_ids'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(ROOT / "zeeker-judgements.db"))
    parser.add_argument("--apply", action="store_true", help="execute the plan (default: dry-run)")
    parser.add_argument(
        "--emit-legacy-ids",
        metavar="PATH",
        help="write the legacy-template id list for scripts/reextract_cached.py --from-file",
    )
    args = parser.parse_args()

    db = sqlite3.connect(args.db, timeout=30)
    try:
        found = scan(db)
        apply_repair(db, found, args.apply)
    finally:
        db.close()

    if args.emit_legacy_ids:
        Path(args.emit_legacy_ids).write_text("\n".join(found["legacy_ids"]) + "\n")
        print(f"legacy ids written to {args.emit_legacy_ids}")

    mode = "APPLIED" if args.apply else "DRY RUN (pass --apply to execute)"
    print(f"\n{mode}. Next steps:")
    print("  1. uv run python scripts/reextract_cached.py --from-file <legacy-ids-file>")
    print("  2. let the 6-hourly builds drain the refetch/summarise backlog,")
    print("     or run manual builds with JUDGMENTS_EXTRACT_MAX_PER_RUN raised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
