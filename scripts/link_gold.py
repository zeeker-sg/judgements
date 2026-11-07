"""Link the gold set's case citations and count how each one fared.

Uses the gold CITATION and CASE_NAME spans (not the rules' output), so the
numbers measure linking alone. With ``--detected`` it links what rules-5
finds in the same passages instead, which measures the two steps together.

    uv run python scripts/link_gold.py GOLD.jsonl [--catalog .cache/catalog.json]
        [--html DIR ...] [--review OUT.md] [--detected]

``--html`` adds directories of ``{id}.html.gz`` pages used to break ties
from the candidates' own text; the extraction and gold-sample caches are
always searched.
"""

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resources import extraction  # noqa: E402
from resources.citation_linker import (  # noqa: E402
    Catalog,
    Context,
    break_tie_by_cues,
    break_tie_by_pinpoint,
    break_tie_by_text,
    cited_name_mentions,
    link_neutral,
    link_report,
)
from resources.citation_rules import (  # noqa: E402
    CITATION_RE,
    Mention,
    _citation_kind,
    detect,
    judgment_parts,
)

HTML_DIRS = [Path(".cache/judgments_html"), Path(".cache/gold_sample/html")]


def gold_mentions(row: dict) -> list:
    text = row["input"]["text"]
    spans = (row.get("user_anns") or row["server_anns"]).get("spans") or []
    out = []
    for s in sorted(spans, key=lambda s: s["start"]):
        span = text[s["start"] : s["end"]]
        kind = None
        if s["label"] == "CITATION":
            m = CITATION_RE.search(span)
            kind = _citation_kind(m.group("abbr")) if m else "other_report"
        out.append(Mention(s["start"], s["end"], s["label"], span, kind=kind))
    return out


class Texts:
    """Candidate judgments' strings and numbered paragraphs, from cached detail pages."""

    def __init__(self, dirs):
        self.dirs = dirs
        self.cache = {}

    def fragments(self, jid):
        if jid not in self.cache:
            self.cache[jid] = None
            for d in self.dirs:
                p = d / f"{jid}.html.gz"
                if p.exists():
                    html = gzip.decompress(p.read_bytes()).decode()
                    try:
                        self.cache[jid] = extraction.extract_judgment(html, jid).fragments
                    except extraction.ExtractionError:
                        pass
                    break
        return self.cache[jid]

    def __call__(self, jid):
        frags = self.fragments(jid)
        return None if frags is None else [t for _, _, t in judgment_parts(frags)]

    def paragraphs(self, jid):
        frags = self.fragments(jid)
        if frags is None:
            return None
        out = collections.defaultdict(str)
        for f in frags:
            if f["paragraph_number"] is not None:
                out[f["paragraph_number"]] += " " + (f["content_text"] or "")
        return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--catalog", default=".cache/catalog.json")
    ap.add_argument("--html", nargs="*", default=[])
    ap.add_argument("--review")
    ap.add_argument("--detected", action="store_true")
    args = ap.parse_args()

    catalog = Catalog(json.loads(Path(args.catalog).read_text()))
    texts = Texts(HTML_DIRS + [Path(d) for d in args.html])
    counts = collections.Counter()
    review = []

    for line in open(args.gold, encoding="utf-8"):
        row = json.loads(line)
        if row["answer"] != "accept":
            continue
        text = row["input"]["text"]
        citing_id = row["meta"]["fragment_id"].rsplit("_", 1)[0]
        citing = catalog.by_id.get(citing_id)
        citing_date = citing.decision_date if citing else None
        mentions = detect(text) if args.detected else gold_mentions(row)
        names = cited_name_mentions(mentions, text)
        for i, m in enumerate(mentions):
            if m.label != "CITATION" or m.kind not in ("sg_neutral", "sg_report"):
                continue
            if m.kind == "sg_neutral":
                link = link_neutral(catalog, m.text)
                counts[f"neutral {link.tier}"] += 1
                continue
            name_m = names.get(i)
            name = name_m.text if name_m else None
            year = int(re.search(r"\d{4}", m.text).group())
            link = link_report(catalog, name, year, citing_id, citing_date)
            first = link.tier
            if name_m:
                ctx = Context(text, name_m.start, m.end, name, citing.court if citing else "")
                link = break_tie_by_cues(catalog, link, ctx)
                link = break_tie_by_text(catalog, link, m.text, name, texts)
                link = break_tie_by_pinpoint(catalog, link, ctx, texts.paragraphs)
            if first == "tied" and link.tier == "resolved":
                counts[f"slr resolved by {link.note.split(':')[0].split(' (')[0]}"] += 1
            counts[f"slr {link.tier}"] += 1
            if first == "tied":
                counts["slr tied before text"] += 1
            review.append((m.text, name, link))

    for k in sorted(counts):
        print(f"{counts[k]:4d}  {k}")

    if args.review:
        lines = [
            "| SLR citation | Name in text | Tier | Linked to / candidates |",
            "| --- | --- | --- | --- |",
        ]
        for cit, name, link in sorted(review, key=lambda r: (r[2].tier, r[0])):
            cands = "; ".join(
                f"{catalog.by_id[j].citation} {catalog.by_id[j].case_name} ({s})"
                for j, s in link.candidates[:2]
            )
            if link.target_id:
                t = catalog.by_id[link.target_id]
                cands = f"**{t.citation} {t.case_name}**" + (f" — {link.note}" if link.note else "")
            lines.append(f"| {cit} | {name or ''} | {link.tier} | {cands} |")
        Path(args.review).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
