"""Score the citation rules against labelled gold spans from Ellf.

Reads a ``citations_gold`` dataset export (``ellf datasets export
citations_gold -o FILE``) and compares, per label:

- the spans the annotator was shown (``server_anns``, the detector named in
  each task's meta), and
- the current rules, run over each passage's whole judgment from the
  sampler's HTML cache (``detect_judgment``), as production will.

Prodigy stores no ``user_anns`` when a task is accepted untouched, so the
shown spans are the gold for those. Rejected tasks are left out.

Scores are exact span matches. "lenient" also forgives edge punctuation and
a leading "the", which the annotation UI's token snapping often adds. R~
counts a gold span as found when a same-label prediction overlaps it.

Run:  uv run python scripts/score_gold.py EXPORT.jsonl [--diff OUT.md]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resources import extraction  # noqa: E402
from resources.citation_rules import DETECTOR_VERSION, detect, detect_parts  # noqa: E402

GOLD_HTML = ROOT / ".cache" / "gold_sample" / "html"
LABELS = ["CITATION", "CASE_NUMBER", "CASE_NAME", "SHORT_NAME", "INSTRUMENT", "PROVISION"]
TRIM = " .,;:()“”\"‘’'|[]\n"


def spans(lst) -> set:
    return {(s["start"], s["end"], s["label"]) for s in lst or []}


def lenient(text: str, spanset: set) -> set:
    """Trim edge punctuation and a leading "the" so token snapping doesn't count."""
    out = set()
    for a, b, label in spanset:
        while a < b and text[a] in TRIM:
            a += 1
        while b > a and text[b - 1] in TRIM:
            b -= 1
        if text[a : a + 4].lower() == "the ":
            a += 4
        out.add((a, b, label))
    return out


class Judgments:
    """Rules over whole judgments, re-extracted from the sampler's HTML cache."""

    def __init__(self):
        self.cache: dict = {}

    def mentions(self, meta: dict, text: str):
        jid = meta["fragment_id"].rsplit("_", 1)[0]
        if jid not in self.cache:
            try:
                html = gzip.decompress((GOLD_HTML / f"{jid}.html.gz").read_bytes()).decode()
                self.cache[jid] = detect_parts(extraction.extract_judgment(html, jid).fragments)
            except (OSError, extraction.ExtractionError):
                self.cache[jid] = {}
        found = self.cache[jid].get((meta["fragment_id"], meta["part"]))
        if found and found[0] == text:
            return found[1], True
        return detect(text), False


def load(path: Path) -> list:
    judgments = Judgments()
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["answer"] != "accept":
            continue
        text = r["input"]["text"]
        mentions, in_context = judgments.mentions(r["meta"], text)
        rows.append(
            {
                "text": text,
                "meta": r["meta"],
                # No user_anns: accepted without edits, so the shown spans are gold.
                "gold": spans((r.get("user_anns") or r["server_anns"]).get("spans")),
                "shown": spans(r["server_anns"].get("spans")),
                "now": {(m.start, m.end, m.label) for m in mentions},
                "in_context": in_context,
            }
        )
    return rows


def score(rows: list, key: str, mode: str = "strict") -> list:
    tp, fp, fn, overlap = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        gold, pred = r["gold"], r[key]
        if mode == "lenient":
            gold, pred = lenient(r["text"], gold), lenient(r["text"], pred)
        for s in gold & pred:
            tp[s[2]] += 1
        for s in pred - gold:
            fp[s[2]] += 1
        for s in gold - pred:
            fn[s[2]] += 1
        for g in gold:
            if any(p[2] == g[2] and p[0] < g[1] and g[0] < p[1] for p in pred):
                overlap[g[2]] += 1
    out = []
    for label in LABELS + ["ALL"]:
        pick = (lambda c: sum(c.values())) if label == "ALL" else (lambda c: c[label])
        t, f, n, o = pick(tp), pick(fp), pick(fn), pick(overlap)
        p = t / (t + f) if t + f else 0
        r = t / (t + n) if t + n else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        out.append((label, t + n, t + f, p, r, f1, o / (t + n) if t + n else 0))
    return out


def table(title: str, result: list) -> None:
    print(f"\n{title}\n{'label':12} {'gold':>5} {'pred':>5} {'P':>6} {'R':>6} {'F1':>6} {'R~':>6}")
    for label, g, pr, p, r, f1, ov in result:
        print(f"{label:12} {g:5} {pr:5} {p:6.2f} {r:6.2f} {f1:6.2f} {ov:6.2f}")


def context(text: str, span: tuple) -> str:
    a, b, _ = span
    return (text[max(0, a - 40) : a] + "⟦" + text[a:b] + "⟧" + text[b : b + 40]).replace("\n", " ")


def write_diff(rows: list, path: Path) -> int:
    lines, n = [], 0
    for i, r in enumerate(rows):
        gold, pred = lenient(r["text"], r["gold"]), lenient(r["text"], r["now"])
        if gold == pred:
            continue
        n += 1
        m = r["meta"]
        lines.append(f"## {i} {m['judgment']} {m['part']} [{m['split']}]\n")
        lines += [f"- MISSED {s[2]}: …{context(r['text'], s)}…" for s in sorted(gold - pred)]
        lines += [f"- EXTRA  {s[2]}: …{context(r['text'], s)}…" for s in sorted(pred - gold)]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export", type=Path)
    parser.add_argument("--diff", type=Path, help="write disagreements (current rules) here")
    args = parser.parse_args()

    rows = load(args.export)
    n_spans = sum(len(r["gold"]) for r in rows)
    print(f"{len(rows)} accepted tasks, {n_spans} gold spans")
    print(f"untouched by the annotator: {sum(r['gold'] == r['shown'] for r in rows)}")
    print(f"whole-judgment context found: {sum(r['in_context'] for r in rows)} of {len(rows)}")

    table("shown to the annotator", score(rows, "shown"))
    table(f"{DETECTOR_VERSION}, whole judgment", score(rows, "now"))
    table(f"{DETECTOR_VERSION}, whole judgment, lenient", score(rows, "now", "lenient"))
    for split in ("tune", "test"):
        rs = [r for r in rows if r["meta"]["split"] == split]
        table(f"{split} split, lenient (n={len(rs)})", score(rs, "now", "lenient"))
    for key in ("era", "level"):
        groups = defaultdict(list)
        for r in rows:
            groups[r["meta"][key]].append(r)
        print(f"\nall labels, lenient, by {key}:")
        for g, rs in sorted(groups.items()):
            _, gold, _, p, r, f1, _ = score(rs, "now", "lenient")[-1]
            print(f"  {g:18} n={len(rs):3} gold={gold:4} P={p:.2f} R={r:.2f} F1={f1:.2f}")

    if args.diff:
        n = write_diff(rows, args.diff)
        print(f"\n{n} tasks disagree -> {args.diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
