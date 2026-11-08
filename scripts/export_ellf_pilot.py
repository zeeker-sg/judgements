"""Export judgment text as Prodigy/Ellf span tasks, pre-filled by the rules.

One task per paragraph and one per footnote, in document order, with the
spans the rules find attached so the annotator corrects rather than starts
from blank. The rules run over the whole judgment in reading order
(``detect_judgment``), as production will, so a short name defined in an
earlier paragraph is already marked. Every paragraph is exported, with or
without detections, so the gold set can measure what the rules miss.

Output lines look like:
  {"text": "...", "spans": [{"start": 4, "end": 20, "label": "CITATION"}],
   "meta": {"judgment": "[2026] SGDC 136", "fragment_id": "…_0012",
            "part": "content" | "footnote 1", "detector": "rules-4",
            "context": "judgment"}}

Run:  uv run python scripts/export_ellf_pilot.py OUT.jsonl [ID ...]
      (no IDs = every judgment in the local DB that has content)

With ``--units PATH`` it exports the sampled gold-set units written by
``scripts/sample_gold_fragments.py`` instead. Units are interleaved round
robin across court-level/era strata, so however far the annotator gets,
the labelled part stays spread across strata. Each unit's judgment is
re-extracted from the sampler's HTML cache for context; a unit whose
judgment is missing there falls back to the passage alone, marked
``"context": "passage"``. ``--skip PATH`` leaves out units already answered
in an Ellf export.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import sqlite_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resources import extraction  # noqa: E402
from resources.citation_rules import DETECTOR_VERSION, detect, detect_parts  # noqa: E402

GOLD_HTML = ROOT / ".cache" / "gold_sample" / "html"


def task(text: str, mentions, meta: dict) -> dict:
    return {
        "text": text,
        "spans": [{"start": m.start, "end": m.end, "label": m.label} for m in mentions],
        "meta": {**meta, "detector": DETECTOR_VERSION},
    }


def unit_mentions(unit: dict, cache: dict) -> tuple:
    """Mentions for a sampled unit, run over its whole judgment when the HTML is cached."""
    jid = unit["judgment_id"]
    if jid not in cache:
        path = GOLD_HTML / f"{jid}.html.gz"
        try:
            html = gzip.decompress(path.read_bytes()).decode("utf-8")
            cache[jid] = detect_parts(extraction.extract_judgment(html, jid).fragments)
        except (OSError, extraction.ExtractionError):
            cache[jid] = {}
    found = cache[jid].get((unit["fragment_id"], unit["part"]))
    if found and found[0] == unit["text"]:
        return found[1], "judgment"
    return detect(unit["text"]), "passage"


def answered_keys(path: Path) -> set:
    """``(fragment_id, part)`` of every task answered in an Ellf dataset export."""
    keys = set()
    for line in path.open(encoding="utf-8"):
        meta = json.loads(line).get("meta", {})
        keys.add((meta.get("fragment_id"), meta.get("part")))
    return keys


def export_units(units_path: Path, out_path: Path, seed: int, skip: set = frozenset()) -> tuple:
    rng = random.Random(seed)
    strata = defaultdict(list)
    for line in units_path.open(encoding="utf-8"):
        u = json.loads(line)
        strata[(u["level"], u["era"])].append(u)
    queues = []
    for key in sorted(strata):
        rng.shuffle(strata[key])
        queues.append(strata[key])
    rng.shuffle(queues)
    n_tasks = n_spans = n_passage = 0
    cache: dict = {}
    with out_path.open("w", encoding="utf-8") as out:
        for row in zip_longest(*queues):
            for u in row:
                if u is None or (u["fragment_id"], u["part"]) in skip:
                    continue
                meta = {k: u[k] for k in ("citation", "fragment_id", "part", "level", "era")}
                meta.update(judgment=meta.pop("citation"), split=u["split"], signal=u["signal"])
                mentions, context = unit_mentions(u, cache)
                t = task(u["text"], mentions, {**meta, "context": context})
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
                n_tasks += 1
                n_spans += len(t["spans"])
                n_passage += context == "passage"
    return len(strata), n_tasks, n_spans, n_passage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("ids", nargs="*", help="judgment ids (default: all with content)")
    parser.add_argument("--db", default=str(ROOT / "zeeker-judgements.db"))
    parser.add_argument("--units", type=Path, help="sampled units.jsonl to export instead")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--skip", type=Path, help="Ellf export whose answered units to leave out")
    args = parser.parse_args()

    if args.units:
        skip = answered_keys(args.skip) if args.skip else set()
        n_strata, n_tasks, n_spans, n_passage = export_units(args.units, args.out, args.seed, skip)
        print(
            f"{n_strata} strata, {n_tasks} tasks ({len(skip)} answered left out), "
            f"{n_spans} pre-filled spans, {n_passage} without judgment context -> {args.out}"
        )
        return 0

    db = sqlite_utils.Database(args.db)
    if args.ids:
        marks = ", ".join("?" for _ in args.ids)
        judgments = list(
            db.query(f"SELECT id, citation FROM judgments WHERE id IN ({marks})", args.ids)
        )
    else:
        judgments = list(db.query("SELECT id, citation FROM judgments WHERE has_content = 1"))

    n_tasks = n_spans = 0
    with args.out.open("w", encoding="utf-8") as out:
        for j in judgments:
            fragments = list(
                db.query(
                    "SELECT id, paragraph_number, content_text, footnote_text"
                    " FROM judgments_fragments WHERE judgment_id = ? ORDER BY ordinal",
                    [j["id"]],
                )
            )
            paragraph = {f["id"]: f["paragraph_number"] for f in fragments}
            for (fid, part), (text, mentions) in detect_parts(fragments).items():
                if not text.strip():
                    continue
                meta = {
                    "judgment": j["citation"],
                    "fragment_id": fid,
                    "paragraph": paragraph[fid],
                    "part": part,
                    "context": "judgment",
                }
                t = task(text, mentions, meta)
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
                n_tasks += 1
                n_spans += len(t["spans"])

    print(f"{len(judgments)} judgments, {n_tasks} tasks, {n_spans} pre-filled spans -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
