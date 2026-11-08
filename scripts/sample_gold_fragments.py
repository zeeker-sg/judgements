"""Draw the citation gold-set sample: long fragments from many judgments.

Labelling whole judgments spends most of the time on headings, counsel
lists and dates. This script instead:

1. downloads the judgments catalogue from data.zeeker.sg (no full text);
2. draws judgments stratified by court level and citation-style era;
3. fetches each judgment's HTML from eLitigation, politely, into its own
   cache under ``.cache/gold_sample/`` (never the Phase 2 cache);
4. keeps body paragraphs, quotes and list items of at least
   ``--min-chars`` characters, and footnotes of at least ``--min-footnote``;
5. picks up to ``--per-judgment`` units per judgment, about 60% from units
   with a loose sign of a citation and 40% at random, so the gold set can
   still show what the rules miss;
6. assigns whole judgments to ``tune`` or ``test`` so no judgment is in both.

Writes ``.cache/gold_sample/units.jsonl``, one unit per line, each with its
stratum, split and whether it carried the loose signal. Re-running reuses
the catalogue and fetched HTML; ``--seed`` makes the draw repeatable.

A held-out linker set (2026-11-07) is drawn with ``--cases-only`` (only
units that carry an SLR or neutral case citation), ``--exclude`` (skip
every judgment in an earlier units file) and ``--out`` (a separate file):

  uv run python scripts/sample_gold_fragments.py --cases-only --tune-share 0 \\
      --exclude .cache/gold_sample/units.jsonl \\
      --out .cache/gold_sample/heldout/units.jsonl --seed 20261107

Run:  uv run python scripts/sample_gold_fragments.py [--judgments 200]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resources import extraction  # noqa: E402
from resources.judgments import (  # noqa: E402
    EXTRACT_DELAY_BASE,
    EXTRACT_DELAY_JITTER,
    _fetch_detail,
    create_client,
    make_id,
    polite_sleep,
)

OUT_DIR = ROOT / ".cache" / "gold_sample"
CATALOGUE_URL = (
    "https://data.zeeker.sg/zeeker-judgements/judgments.json"
    "?_size=1000&_col=citation&_col=court&_col=decision_date&_col=source_url"
)

COURT_LEVEL = {
    "SGCA": "court_of_appeal",
    "SGCAI": "court_of_appeal",
    "SGHC": "high_court",
    "SGHCF": "high_court",
    "SGHCR": "high_court",
    "SGHCI": "high_court",
    "SGHCA": "high_court",
    "SGAD": "high_court",
    "SGDC": "state_courts",
    "SGFC": "state_courts",
    "SGMC": "state_courts",
    "SGYC": "state_courts",
    "SGPHC": "state_courts",
    "SGSCT": "tribunals",
    "SGECT": "tribunals",
    "SGCDT": "tribunals",
}
# Share of the sample per court level; tribunals are small, so they take what exists.
LEVEL_SHARE = {"court_of_appeal": 0.25, "high_court": 0.4, "state_courts": 0.25, "tribunals": 0.1}


def era(decision_date: str) -> str:
    """Citation-style era: original SLR, current SLR, 2020 Revised Edition."""
    year = int(decision_date[:4])
    if year <= 2009:
        return "slr_original"
    if year <= 2021:
        return "slr_current"
    return "rev_ed_2020"


# Body classes worth labelling; headings, counsel, dates and signatures are not.
BODY_CLASS_RE = re.compile(r"^Judg-(?:1|1-firstpara|2|3|4|Quote.*|List.*)$")
# A loose sign of a citation, deliberately wider than the rules.
SIGNAL_RE = re.compile(
    r"[\[(]\d{4}[\])]|\s v\s|\bss?\s?\d|\b(?:Act|Code|Rules|Regulations)\b|\b[A-Z]{2,}\s?\d+/\d{4}"
)
# A Singapore case citation: an SLR / SLR(R) report or a neutral citation.
CASE_RE = re.compile(r"\[\d{4}\]\s*(?:\d+\s+SLR(?:\s*\(R\))?\s+\d|SG[A-Z]+\s+\d)")


def load_catalogue() -> list[dict]:
    path = OUT_DIR / "catalogue.json"
    if path.exists():
        return json.loads(path.read_text())
    rows: list[dict] = []
    url = CATALOGUE_URL
    with httpx.Client(
        timeout=60, follow_redirects=True, headers={"User-Agent": "zeeker-judgements gold sampler"}
    ) as c:
        while url:
            data = c.get(url).raise_for_status().json()
            cols = data["columns"]
            rows += [dict(zip(cols, r)) for r in data["rows"]]
            url = data.get("next_url")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def draw(catalogue: list[dict], n: int, rng: random.Random) -> list[dict]:
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in catalogue:
        level = COURT_LEVEL.get(row["court"])
        if level and row.get("decision_date") and row.get("source_url"):
            strata[(level, era(row["decision_date"]))].append(row)
    chosen: list[dict] = []
    for level, share in LEVEL_SHARE.items():
        target = round(n * share)
        eras = [e for e in ("slr_original", "slr_current", "rev_ed_2020") if strata[(level, e)]]
        # Spread the level's quota over its eras; an era short of rows hands
        # its remainder to the eras after it.
        remaining = target
        for i, e in enumerate(eras):
            pool = strata[(level, e)]
            quota = min(len(pool), remaining // (len(eras) - i))
            for row in rng.sample(pool, quota):
                chosen.append({**row, "level": level, "era": e})
            remaining -= quota
    return chosen


def fetch_html(client, row: dict) -> str | None:
    jid = make_id(row["source_url"])
    path = OUT_DIR / "html" / f"{jid}.html.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    try:
        html = _fetch_detail(client, row["source_url"])
    except Exception as exc:  # network or HTTP error: skip this judgment
        print(f"  {row['citation']}: fetch failed ({type(exc).__name__})")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(html.encode("utf-8")))
    polite_sleep(EXTRACT_DELAY_BASE, EXTRACT_DELAY_JITTER)
    return html


def units_of(extracted, min_chars: int, min_footnote: int) -> list[dict]:
    units = []
    for f in extracted.fragments:
        if BODY_CLASS_RE.match(f["class_name"] or "") and len(f["content_text"]) >= min_chars:
            units.append({"fragment_id": f["id"], "part": "content", "text": f["content_text"]})
        notes = json.loads(f["footnote_text"]) if f["footnote_text"] else []
        for i, note in enumerate(notes, 1):
            if len(note) >= min_footnote:
                units.append({"fragment_id": f["id"], "part": f"footnote {i}", "text": note})
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judgments", type=int, default=200)
    parser.add_argument("--per-judgment", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--min-footnote", type=int, default=40)
    parser.add_argument("--tune-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "units.jsonl")
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="units.jsonl whose judgments to leave out (repeatable)",
    )
    parser.add_argument(
        "--cases-only",
        action="store_true",
        help="pick only units with an SLR or neutral case citation",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    catalogue = load_catalogue()
    excluded = {json.loads(line)["judgment_id"] for path in args.exclude for line in path.open()}
    if excluded:
        catalogue = [r for r in catalogue if make_id(r["source_url"] or "") not in excluded]
    sample = draw(catalogue, args.judgments, rng)
    print(f"catalogue {len(catalogue)} judgments ({len(excluded)} excluded); drew {len(sample)}")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_units = n_signal = n_skipped = 0
    with create_client() as client, out_path.open("w", encoding="utf-8") as out:
        for k, row in enumerate(sample, 1):
            html = fetch_html(client, row)
            if html is None:
                n_skipped += 1
                continue
            jid = make_id(row["source_url"])
            try:
                extracted = extraction.extract_judgment(html, jid)
            except extraction.ExtractionError:
                n_skipped += 1
                continue
            units = units_of(extracted, args.min_chars, args.min_footnote)
            if args.cases_only:
                units = [u for u in units if CASE_RE.search(u["text"])]
                if not units:
                    n_skipped += 1
                    continue
            signal = [u for u in units if SIGNAL_RE.search(u["text"])]
            plain = [u for u in units if not SIGNAL_RE.search(u["text"])]
            n_signal_pick = min(len(signal), round(args.per_judgment * 0.6 + 0.4))
            picks = rng.sample(signal, n_signal_pick)
            picks += rng.sample(plain, min(len(plain), args.per_judgment - len(picks)))
            split = "tune" if rng.random() < args.tune_share else "test"
            for u in picks:
                out.write(
                    json.dumps(
                        {
                            **u,
                            "judgment_id": jid,
                            "citation": row["citation"],
                            "court": row["court"],
                            "level": row["level"],
                            "era": row["era"],
                            "split": split,
                            "signal": u in signal,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_units += 1
                n_signal += u in signal
            if k % 20 == 0:
                print(f"  {k}/{len(sample)} judgments, {n_units} units")
    print(
        f"{n_units} units ({n_signal} with a citation signal) from "
        f"{len(sample) - n_skipped} judgments, {n_skipped} skipped -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
