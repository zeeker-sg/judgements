"""Relabel a citations_gold export to the v3 labelling guide (decisions 17 and 18).

Run:
  uv run python scripts/relabel_gold_v3.py EXPORT.jsonl OUT.jsonl

Decision 17: an Act name defined in the judgment (`(the “Women’s Charter”)`) is
SHORT_NAME at the definition and every later use; one never defined stays
INSTRUMENT. Decision 18: "new", "old", "current" … in front of a short name are
part of it. The raw export is left untouched; changed rows get user_anns.
"""

import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resources import extraction  # noqa: E402
from resources.citation_rules import judgment_parts  # noqa: E402

# As the rules' DEFINITION_RE, but a ’ inside a word ("Women’s") does not close the name.
DEFINITION_RE = re.compile(
    r"\s*\(\s*(?:the\s+|hereafter\s+|hereinafter\s+)?[“\"‘](?P<name>(?:[^”\"’]|’(?=\w)){1,80})[”\"’]\s*\)"
)

HTML = ROOT / ".cache" / "gold_sample" / "html"
KEYWORD = re.compile(
    r"\b(?:Act|Code|Charter|Rules|Regulations|Constitution|Convention|Law|Treaty|Ordinance|Bill)\b"
)
GENERIC_LEAD = {"Act", "Code", "Rules", "Amendment", "This", "Principal"}
MODIFIER = re.compile(r"\b(new|old|current|amended|repealed|former|present|previous)\s+$")
# An edition, year or chapter marks a full title, which stays INSTRUMENT.
FULL_TITLE = re.compile(r"\d{4}|\bCap\b|Rev Ed")

_defs: dict = {}


def definitions(jid: str):
    """[(part_index, offset, name)] for every bracketed definition in a judgment."""
    if jid not in _defs:
        out, index = [], {}
        try:
            html = gzip.decompress((HTML / f"{jid}.html.gz").read_bytes()).decode()
            parts = judgment_parts(extraction.extract_judgment(html, jid).fragments)
        except (OSError, extraction.ExtractionError):
            parts = []
        for i, (fid, part, text) in enumerate(parts):
            index[(fid, part)] = (i, text)
            for d in DEFINITION_RE.finditer(text):
                out.append((i, d.start("name"), re.sub(r"^[Tt]he\s+", "", d.group("name").strip())))
        _defs[jid] = (out, index)
    return _defs[jid]


def defined_before(meta: dict, text: str, start: int, name: str):
    jid = meta["fragment_id"].rsplit("_", 1)[0]
    defs, index = definitions(jid)
    here = index.get((meta["fragment_id"], meta["part"]))
    if here is None or here[1] != text:
        # No whole-judgment context: fall back to this passage alone.
        return (
            any(
                re.sub(r"^[Tt]he\s+", "", d.group("name").strip()) == name
                and d.start("name") <= start
                for d in DEFINITION_RE.finditer(text)
            ),
            False,
        )
    return any(n == name and (i, o) <= (here[0], start) for i, o, n in defs), True


def main(src: str, dst: str) -> None:
    changes, no_context = [], 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            if r["answer"] != "accept":
                out.write(line)
                continue
            text = r["input"]["text"]
            old = (r.get("user_anns") or r["server_anns"]).get("spans") or []
            new = []
            for s in old:
                s = dict(s)
                label, a, b = s["label"], s["start"], s["end"]
                span = text[a:b]
                if label in ("SHORT_NAME", "INSTRUMENT"):
                    mod = MODIFIER.search(text, 0, a)
                    if label == "SHORT_NAME" and mod and mod.end() == a:
                        s["start"] = a = mod.start(1)
                        changes.append(("rule16", r["meta"]["fragment_id"], span, text[a:b]))
                        span = text[a:b]
                    bare = re.sub(MODIFIER.pattern.replace(r"\s+$", r"\s+"), "", span)
                    words = bare.split()
                    if (
                        KEYWORD.search(bare)
                        and len(words) >= 2
                        and words[0] not in GENERIC_LEAD
                        and words[0][0].isupper()
                        and not FULL_TITLE.search(bare)
                    ):
                        defined, ctx = defined_before(r["meta"], text, a, bare)
                        no_context += not ctx
                        want = "SHORT_NAME" if defined else "INSTRUMENT"
                        if want != label:
                            s["label"] = want
                            changes.append(
                                (
                                    f"rule15 {label}->{want}",
                                    r["meta"]["fragment_id"],
                                    span,
                                    "edited" if r.get("user_anns") else "untouched",
                                )
                            )
                new.append(s)
            if new != old:
                r["user_anns"] = {**(r.get("user_anns") or {}), "spans": new}
                r.setdefault("meta", {})["relabelled"] = "v3"
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    for c in changes:
        print(*c, sep=" | ")
    print(f"\n{len(changes)} changes; {no_context} checks without whole-judgment context")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
