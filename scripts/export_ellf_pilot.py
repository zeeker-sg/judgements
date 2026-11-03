"""Export judgment text as Prodigy/Ellf span tasks, pre-filled by the rules.

One task per paragraph and one per footnote, in document order, with the
spans from ``resources.citation_rules.detect`` attached so the annotator
corrects rather than starts from blank. Every paragraph is exported, with
or without detections, so the gold set can measure what the rules miss.

Output lines look like:
  {"text": "...", "spans": [{"start": 4, "end": 20, "label": "CITATION"}],
   "meta": {"judgment": "[2026] SGDC 136", "fragment_id": "…_0012",
            "part": "content" | "footnote 1", "detector": "rules-1"}}

Run:  uv run python scripts/export_ellf_pilot.py OUT.jsonl [ID ...]
      (no IDs = every judgment in the local DB that has content)

With ``--units PATH`` it exports the sampled gold-set units written by
``scripts/sample_gold_fragments.py`` instead. Units are interleaved round
robin across court-level/era strata, so however far the annotator gets,
the labelled part stays spread across strata.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import sqlite_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resources.citation_rules import DETECTOR_VERSION, detect  # noqa: E402


def task(text: str, meta: dict) -> dict:
    return {
        "text": text,
        "spans": [{"start": m.start, "end": m.end, "label": m.label} for m in detect(text)],
        "meta": {**meta, "detector": DETECTOR_VERSION},
    }


def export_units(units_path: Path, out_path: Path, seed: int) -> tuple:
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
    n_tasks = n_spans = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in zip_longest(*queues):
            for u in row:
                if u is None:
                    continue
                meta = {k: u[k] for k in ("citation", "fragment_id", "part", "level", "era")}
                meta.update(judgment=meta.pop("citation"), split=u["split"], signal=u["signal"])
                t = task(u["text"], meta)
                out.write(json.dumps(t, ensure_ascii=False) + "\n")
                n_tasks += 1
                n_spans += len(t["spans"])
    return len(strata), n_tasks, n_spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("ids", nargs="*", help="judgment ids (default: all with content)")
    parser.add_argument("--db", default=str(ROOT / "zeeker-judgements.db"))
    parser.add_argument("--units", type=Path, help="sampled units.jsonl to export instead")
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    if args.units:
        n_strata, n_tasks, n_spans = export_units(args.units, args.out, args.seed)
        print(f"{n_strata} strata, {n_tasks} tasks, {n_spans} pre-filled spans -> {args.out}")
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
            fragments = db.query(
                "SELECT id, paragraph_number, content_text, footnote_text"
                " FROM judgments_fragments WHERE judgment_id = ? ORDER BY ordinal",
                [j["id"]],
            )
            for f in fragments:
                base = {
                    "judgment": j["citation"],
                    "fragment_id": f["id"],
                    "paragraph": f["paragraph_number"],
                }
                parts = [("content", f["content_text"])]
                notes = json.loads(f["footnote_text"]) if f["footnote_text"] else []
                parts += [(f"footnote {i}", note) for i, note in enumerate(notes, 1)]
                for part, text in parts:
                    if not text or not text.strip():
                        continue
                    t = task(text, {**base, "part": part})
                    out.write(json.dumps(t, ensure_ascii=False) + "\n")
                    n_tasks += 1
                    n_spans += len(t["spans"])

    print(f"{len(judgments)} judgments, {n_tasks} tasks, {n_spans} pre-filled spans -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
