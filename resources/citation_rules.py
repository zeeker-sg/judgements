"""Rule-based detection of citations in judgment text (citation linker, step 1).

Pure functions, no IO: ``detect(text)`` returns the ``Mention`` spans found
in one string — a fragment's ``content_text`` or one entry of its
``footnote_text`` array. It only finds spans; deciding which mentions refer
to the same case (short names, abbreviations) is step 2, and linking to
``judgments`` rows is step 3.

Labels follow Blackstone's scheme, adapted (it spells ``CASENAME``):

- ``CITATION``    a law-report or neutral citation, ``[1999] 2 SLR(R) 462``;
                  law journal articles are not citations
- ``CASE_NUMBER`` a court file number, ``CA/CCA 20/2022``, ``Suit No 123 of 2019``
- ``CASE_NAME``   the case name written right before a citation
- ``SHORT_NAME``  a name defined in brackets right after a citation or Act,
                  ``(“KS v LHY”)``; the span is the quoted name only
- ``INSTRUMENT``  an Act, Code, Charter, Rules or Regulations title
- ``PROVISION``   a section, rule, order or article reference, ``s 52(14)``,
                  or a paragraph of a Schedule (``para 4`` of the First Schedule)

The edge cases are settled in the design doc's "Labelling guide" tab.

Phase 0 (``extraction.element_text``) is assumed: no U+00A0, no padded
quotes, no words split across inline tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

DETECTOR_VERSION = "rules-2"

# ---------------------------------------------------------------------------
# Citations: [year] vol ABBR page, one pattern for neutral and reported forms
# ---------------------------------------------------------------------------

# A report or court abbreviation is 1-3 capitalised tokens ("SLR(R)",
# "All ER", "Lloyd’s Law Rep", "EWCA Civ", "W.L.R."). The page is required,
# which keeps "(2020 Rev Ed)" and bare years out.
_ABBR_TOKEN = r"[A-Z][A-Za-z’'.&]*(?:\s?\([A-Z]{1,3}\))?"
CITATION_RE = re.compile(
    r"(?P<open>[\[(])(?P<year>\d{4}(?:\s*[–-]\s*\d{4})?)[\])]"
    r"\s*(?:(?P<vol>\d{1,3})\s*)?"
    rf"(?P<abbr>{_ABBR_TOKEN}(?:\s+(?:{_ABBR_TOKEN}|Cas|Rep|Civ|Crim)){{0,2}})"
    r"\s*(?P<page>\d{1,5})(?!\d)"
    # UK neutral citations carry a division suffix: [2011] EWHC 123 (Ch)
    r"(?:\s\((?P<div>Ch|QB|KB|Comm|Fam|TCC|Admin|Pat|IPEC)\))?"
)

# Law journals cite like reports but are not judgments (guide, decision 3).
# MLJ and CLJ are law reports, so the list is explicit.
JOURNAL_ABBR_RE = re.compile(
    r"^(?:SAcLJ|SJLS|Sing JLS|LQR|MLR|OJLS|JBL|LMCLQ|HKLJ|CLP|Asian JCL|[A-Z][\w.]*\s?L\s?Rev)$"
)

# ---------------------------------------------------------------------------
# Court file numbers: CA/CCA 20/2022, HC/SUM 1629/2025, Suit No 123 of 2019
# ---------------------------------------------------------------------------

CASE_NUMBER_RE = re.compile(
    r"(?<![\w/])(?P<num>"
    r"(?:[A-Z]{2,4}/)?[A-Z]{1,6}\s\d{1,5}/\d{4}(?![\w/])"
    r"|(?:[A-Z][\w’']*\s+){1,4}No\.?\s*\d{1,5}\s+of\s+\d{4}(?!\d)"
    r")"
)
_CASE_NUMBER_LEAD_STOP = {"In", "The", "See", "Under", "By", "And", "Of", "On", "For"}
# "Act No 5 of 2010" numbers legislation, not a court file.
_NOT_A_CASE_NUMBER_RE = re.compile(
    r"\b(?:Act|Bill|Ordinance|Notification|Gazette|Order|Rules?|Regulations?)\s+No\b"
)

SG_NEUTRAL_ABBR_RE = re.compile(r"^SG[A-Z]+(?:\([A-Z]\))?$")
SG_REPORT_ABBR_RE = re.compile(r"^SLR(?:\s?\(R\))?$")

# Pinpoint after a citation: "at [47]", "at [14]–[15]", "at [38] and [136]".
_PINPOINT = r"at\s*\[[^\]]{1,12}\](?:\s*(?:–|-|to|and|,)\s*\[[^\]]{1,12}\])*"
PINPOINT_RE = re.compile(r"\s*(?P<pin>" + _PINPOINT + r")")

# A defined name in brackets: (“KS v LHY”), (the “Default Judgment”).
DEFINITION_RE = re.compile(
    r"\s*\(\s*(?:the\s+|hereafter\s+|hereinafter\s+)?[“\"‘](?P<name>[^”\"’]{1,80})[”\"’]\s*\)"
)

# ---------------------------------------------------------------------------
# Case names: walk back from the citation over name-like tokens
# ---------------------------------------------------------------------------

# Lower-case words that can sit inside a case name.
CASE_NAME_CONNECTORS = {
    "v", "and", "another", "others", "anor", "ors", "of", "the", "de", "von",
    "van", "der", "bin", "binti", "binte", "s/o", "d/o", "a/l", "a/p", "alias",
    "(alias", "&", "matter", "matters", "appeal", "appeals", "suit", "suits",
    "application", "applications", "re", "ex", "parte", "for", "on", "behalf", "an",
    "a", "trading", "as", "nee", "formerly", "deceased", "estate",
}  # fmt: skip

# Connectors that can also open a name ("de Dampierre v de Dampierre").
NAME_PARTICLES = {"de", "von", "van", "der", "bin", "binti", "binte"}

# Capitalised words that open a sentence or clause, not a case name.
CASE_NAME_LEAD_STOP = {
    "In", "See", "Cf", "Also", "And", "The", "As", "Following", "Applying",
    "Per", "Unlike", "Like", "Similarly", "Thus", "Hence", "Further",
}  # fmt: skip

_TOKEN_RE = re.compile(r"\S+")
_NAME_WINDOW = 300


def _case_name_before(text: str, end: int) -> Optional[tuple]:
    """Return ``(start, end)`` of a case name ending at ``end``, or None.

    A case name is a run of capitalised tokens and connectors that contains
    `` v `` or opens with Re / In re / Ex parte. The walk stops at the first
    lower-case word that is not a connector, at a token starting with a
    digit, or at clause punctuation (``;`` ``:`` ``[`` ``]``).
    """
    window_start = max(0, end - _NAME_WINDOW)
    window = text[window_start:end].rstrip()
    tokens = list(_TOKEN_RE.finditer(window))
    kept: List[re.Match] = []
    for tok in reversed(tokens):
        word = tok.group(0)
        if any(ch in word for ch in ";:[]") or word.endswith("."):
            # "Ltd." or "Co." could be in a name; ". " usually ends a sentence.
            if not re.fullmatch(r"(?:Co|Ltd|Inc|Bhd|Corp|No|St)\.,?", word):
                break
        bare = word.strip('(),“”"')
        if not bare:
            break
        if bare[0].isdigit():
            break
        if bare[0].isupper() or word in CASE_NAME_CONNECTORS or bare in CASE_NAME_CONNECTORS:
            kept.append(tok)
            continue
        break
    kept.reverse()
    # Trim leading clause openers and connectors.
    while kept:
        first = kept[0].group(0).strip('(“"')
        if first in ("In", "Ex") and len(kept) > 1:
            nxt = kept[1].group(0)
            if (first, nxt) in (("In", "re"), ("Ex", "parte")):
                break
        if first in CASE_NAME_LEAD_STOP or (
            first in CASE_NAME_CONNECTORS and first not in NAME_PARTICLES
        ):
            kept.pop(0)
            continue
        break
    if not kept:
        return None
    words = [t.group(0) for t in kept]
    opens_re = words[0] in ("Re", "In", "Ex") and (
        words[0] == "Re" or (len(words) > 1 and words[1] in ("re", "parte"))
    )
    if "v" not in words and not opens_re:
        return None
    start = window_start + kept[0].start()
    stop = window_start + kept[-1].end()
    # A case name never starts with an opening bracket or quote.
    while start < stop and text[start] in '(“"':
        start += 1
    return start, stop


# ---------------------------------------------------------------------------
# Statutes
# ---------------------------------------------------------------------------

_STATUTE_KEYWORD = (
    r"(?:Act|Code|Charter|Ordinance|Rules|Regulations|Constitution|Treaty|Convention|Protocol)"
)
INSTRUMENT_RE = re.compile(
    # The lookbehind stops a title starting inside "s 11A of the …".
    r"(?<![\w’'])(?P<title>"
    r"(?:[A-Z][\w’'-]*\s+)"
    r"(?:(?:[A-Z][\w’'()-]*|of|and|the|for|in|on|to|&)\s+){0,10}?"
    # A second keyword continues the title: "Energy Charter Treaty".
    rf"{_STATUTE_KEYWORD}(?:\s+{_STATUTE_KEYWORD})?\b"
    r"(?:\s+of\s+(?:the\s+)?[A-Z][\w’']*(?:\s+(?:of\s+)?[A-Z][\w’']*)*)?"
    r"|Rules\s+of\s+Court"
    r")"
    r"(?:\s+(?P<year>\d{4}))?"
    r"(?:\s*\((?P<edition>(?:Cap\s*[\w.]+,\s*)?(?:R\s*\d+,\s*)?\d{4}\s+Rev\s+Ed|Cap\s*[\w.]+)\))?"
)
# Capitalised function words that open a clause before a statute title.
INSTRUMENT_LEAD_STOP = {
    "The", "In", "Under", "By", "See", "Pursuant", "As", "For", "Of", "And",
    "This", "That", "These", "Since", "While", "If", "Unlike", "Like", "Per",
    "Following", "Section", "Sections", "Part", "Both", "Whereas", "To",
}  # fmt: skip
SCHEDULE_PREFIX_RE = re.compile(
    r"(?:\w+\s+)?(?:Schedule|Part)\s+(?:\w+\s+)?(?:of|to)\s+(?:the\s+)?"
)
_INSTRUMENT_TOO_GENERIC = {"Act", "Code", "Rules", "Regulations", "Charter", "Constitution"}
# A title never contains these: "What Can They Act" is a sentence, not a statute.
_INSTRUMENT_NOT_TITLE_WORDS = {
    "I", "We", "You", "He", "She", "It", "They", "Can", "Could", "Will", "Would",
    "Shall", "Should", "May", "Might", "Must", "Did", "Does", "Do", "What", "How",
}  # fmt: skip

PROVISION_RE = re.compile(
    # Not after an apostrophe: "Respondent’s 25%" is a possessive, not s 25.
    r"(?<![\w(’'])(?P<prov>"
    # A paragraph of a Schedule: "para 4 of the First Schedule"
    r"paras?\s*\d+[A-Z]*(?:\(\w{1,5}\))*"
    r"(?:(?:\s+and\s+|,\s*|\s*[–-]\s*)\d+[A-Z]*(?:\(\w{1,5}\))*)*"
    r"(?=\s+(?:of|to|in)\s+(?:the\s+)?(?:\w+\s+)?Schedule\b)"
    # Rules of Court: O 3 r 2(8)
    r"|"
    r"O\s*\d+[A-Z]?\s+rr?\s*\d+[A-Z]*(?:\(\w{1,5}\))*"
    r"|(?:ss?|sections?|regs?|rr?|Arts?|arts?)\s*\d+(?:\.\d+)*[A-Z]*(?:\(\w{1,5}\))*"
    r"(?:(?:\s*[–-]\s*|\s+to\s+|\s+and\s+|,\s*)(?:\d+[A-Z]*(?:\(\w{1,5}\))*|(?:\(\w{1,5}\))+))*"
    r")(?![\w%])"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class Mention:
    start: int
    end: int
    label: str
    text: str
    kind: Optional[str] = None  # sg_neutral | sg_report | other_report | defined
    normalised: Optional[str] = None
    pinpoint: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _normalise_citation(m: re.Match) -> str:
    abbr = re.sub(r"\s+(?=\()", "", re.sub(r"\s+", " ", m.group("abbr")))
    year = re.sub(r"\s*[–-]\s*", "–", m.group("year"))
    parts = [f"{m.group('open')}{year}{']' if m.group('open') == '[' else ')'}"]
    if m.group("vol"):
        parts.append(m.group("vol"))
    parts += [abbr, m.group("page")]
    if m.group("div"):
        parts.append(f"({m.group('div')})")
    return " ".join(parts)


def _citation_kind(abbr: str) -> str:
    abbr = re.sub(r"\s+", " ", abbr)
    if SG_NEUTRAL_ABBR_RE.match(abbr):
        return "sg_neutral"
    if SG_REPORT_ABBR_RE.match(abbr):
        return "sg_report"
    return "other_report"


def _trailing(text: str, pos: int):
    """Pinpoint and/or defined name right after ``pos``, in either order."""
    pinpoint = definition = None
    for _ in range(2):
        m = PINPOINT_RE.match(text, pos)
        if m and pinpoint is None:
            pinpoint = re.sub(r"^at\s*", "", m.group("pin"))
            pos = m.end()
            continue
        d = DEFINITION_RE.match(text, pos)
        if d and definition is None:
            definition = d
            pos = d.end()
            continue
        break
    return pinpoint, definition


def detect(text: str) -> List[Mention]:
    """Find citation mentions in one string, sorted by start offset.

    Overlaps are resolved by priority: citations, then case names and
    defined names, then statute titles, then provisions.
    """
    if not text:
        return []
    found: List[Mention] = []

    for m in CITATION_RE.finditer(text):
        if JOURNAL_ABBR_RE.match(re.sub(r"\s+", " ", m.group("abbr"))):
            continue
        pinpoint, definition = _trailing(text, m.end())
        found.append(
            Mention(
                m.start(),
                m.end(),
                "CITATION",
                m.group(0),
                kind=_citation_kind(m.group("abbr")),
                normalised=_normalise_citation(m),
                pinpoint=pinpoint,
            )
        )
        name = _case_name_before(text, m.start())
        if name:
            found.append(Mention(name[0], name[1], "CASE_NAME", text[name[0] : name[1]]))
        if definition:
            found.append(
                Mention(
                    definition.start("name"),
                    definition.end("name"),
                    "SHORT_NAME",
                    definition.group("name"),
                    kind="defined",
                    meta={"defined_after": m.start()},
                )
            )

    for m in CASE_NUMBER_RE.finditer(text):
        if _NOT_A_CASE_NUMBER_RE.search(m.group("num")):
            continue
        start = m.start("num")
        # "In Suit No 123 of 2019" -> "Suit No 123 of 2019"
        for tok in _TOKEN_RE.finditer(m.group("num")):
            if tok.group(0) not in _CASE_NUMBER_LEAD_STOP:
                start = m.start("num") + tok.start()
                break
        found.append(
            Mention(
                start,
                m.end("num"),
                "CASE_NUMBER",
                text[start : m.end("num")],
                normalised=re.sub(r"\s+", " ", text[start : m.end("num")]),
            )
        )

    for m in INSTRUMENT_RE.finditer(text):
        start = m.start("title")
        tokens = list(_TOKEN_RE.finditer(m.group("title")))
        # Drop clause openers ("Under the Moneylenders Act" -> "Moneylenders Act").
        while tokens and (
            tokens[0].group(0) in INSTRUMENT_LEAD_STOP or tokens[0].group(0)[0].islower()
        ):
            tokens.pop(0)
        if not tokens:
            continue
        start = m.start("title") + tokens[0].start()
        # "First Schedule of the Stamp Duties Act": the Schedule is a part of
        # the Act, not part of its title.
        part = SCHEDULE_PREFIX_RE.match(text, start, m.end("title"))
        if part:
            start = part.end()
            tokens = list(_TOKEN_RE.finditer(text[start : m.end("title")]))
        title = text[start : m.end("title")]
        if (
            not tokens
            or title in _INSTRUMENT_TOO_GENERIC
            or any(t.group(0) in _INSTRUMENT_NOT_TITLE_WORDS for t in tokens)
        ):
            continue
        end = m.end()
        found.append(
            Mention(
                start,
                end,
                "INSTRUMENT",
                text[start:end],
                normalised=" ".join(
                    p
                    for p in (
                        title,
                        m.group("year"),
                        m.group("edition") and f"({m.group('edition')})",
                    )
                    if p
                ),
                meta={"title": title, "year": m.group("year"), "edition": m.group("edition")},
            )
        )
        definition = DEFINITION_RE.match(text, end)
        if definition:
            found.append(
                Mention(
                    definition.start("name"),
                    definition.end("name"),
                    "SHORT_NAME",
                    definition.group("name"),
                    kind="defined",
                    meta={"defined_after": start},
                )
            )

    for m in PROVISION_RE.finditer(text):
        found.append(
            Mention(
                m.start("prov"),
                m.end("prov"),
                "PROVISION",
                m.group("prov"),
                normalised=re.sub(r"\s+", " ", m.group("prov")),
            )
        )

    return _resolve_overlaps(found)


_PRIORITY = {
    "CITATION": 0,
    "CASE_NUMBER": 0,
    "CASE_NAME": 1,
    "SHORT_NAME": 1,
    "INSTRUMENT": 2,
    "PROVISION": 3,
}


def _resolve_overlaps(mentions: List[Mention]) -> List[Mention]:
    kept: List[Mention] = []
    for m in sorted(mentions, key=lambda x: (_PRIORITY[x.label], x.start, -(x.end - x.start))):
        if any(m.start < k.end and k.start < m.end for k in kept):
            continue
        kept.append(m)
    return sorted(kept, key=lambda x: x.start)
