"""Rule-based detection of citations in judgment text (citation linker, step 1).

Pure functions, no IO: ``detect(text)`` returns the ``Mention`` spans found
in one string — a fragment's ``content_text`` or one entry of its
``footnote_text`` array. ``detect_judgment(texts)`` runs over a judgment's
strings in reading order and carries the short names defined so far, so a
name defined in one paragraph is found in the next. It only finds spans;
deciding which mention a short name stands for is step 2, and linking to
``judgments`` rows is step 3.

Labels follow Blackstone's scheme, adapted (it spells ``CASENAME``):

- ``CITATION``    a law-report or neutral citation, ``[1999] 2 SLR(R) 462``;
                  law journal articles are not citations
- ``CASE_NUMBER`` a court file number, ``CA/CCA 20/2022``, ``Suit No 123 of 2019``
- ``CASE_NAME``   a case name written out in full, with or without a citation
- ``SHORT_NAME``  a name defined in brackets right after a citation or Act,
                  ``(“KS v LHY”)`` (the span is the quoted name only), an
                  Act's name defined anywhere (``(“Penal Code”)``), a supra
                  reference (``Bayoil ([36] supra)``), an Act abbreviation or
                  "Act" after a provision (``s 52 of the TCA``), and every
                  later use of a name defined earlier, with a word or year in
                  front (``new LTA``, ``2019 RTA``)
- ``INSTRUMENT``  an Act, Bill, Code, Charter, Rules or Regulations title, with
                  its year, edition and number
- ``PROVISION``   a section, rule, order or article reference, ``s 52(14)``,
                  a Schedule, or a provision of one (``para 4 of the First
                  Schedule``)

The edge cases are settled in the design doc's "Labelling guide" tab.

Phase 0 (``extraction.element_text``) is assumed: no U+00A0, no padded
quotes, no words split across inline tags.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

DETECTOR_VERSION = "rules-5"

# ---------------------------------------------------------------------------
# Citations: [year] vol ABBR page, one pattern for neutral and reported forms
# ---------------------------------------------------------------------------

# A report or court abbreviation is 1-3 capitalised tokens ("SLR(R)",
# "All ER", "Lloyd’s Law Rep", "EWCA Civ", "W.L.R."). The page is required,
# which keeps "(2020 Rev Ed)" and bare years out.
_ABBR_TOKEN = r"[A-Z][A-Za-z’'.&]*(?:\s?\([A-Z]{1,3}\))?"
CITATION_RE = re.compile(
    r"(?P<open>[\[(])(?P<year>\d{4}(?:\s*[–-]\s*(?:\d{4}|\d{2}))?)[\])]"
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
    # An old file number may end in a letter: "CA 83/2002/A", "Suit 755/2001/A".
    # A State Courts charge has 6 digits ("DAC 921470/2021"); an appeal from
    # one adds "/01" ("HC/MA 9089/2025/01").
    r"(?:[A-Z]{2,4}/)?(?:[A-Z]{1,6}|Suit)\s\d{1,6}/\d{4}(?:/(?:[A-Z]|\d{2}))?(?![\w/])"
    # "Admiralty in Rem No 181 of 2009" keeps its lower-case "in".
    r"|(?:[A-Z][\w’']*\s+(?:in\s+(?=Rem\b|Personam\b))?){1,4}No\.?\s*\d{1,5}\s+of\s+\d{4}(?!\d)"
    r")"
)
_CASE_NUMBER_LEAD_STOP = {"In", "The", "See", "Under", "By", "And", "Of", "On", "For"}
# "Act No 5 of 2010" numbers legislation, not a court file.
_NOT_A_CASE_NUMBER_RE = re.compile(
    r"\b(?:Act|Bill|Ordinance|Notification|Gazette|Order|Rules?|Regulations?)\s+No\b"
)
# "(GN No S 98/1963)" numbers a Gazette notification.
_GAZETTE_BEFORE_RE = re.compile(r"\b(?:GN|G\.\s?N\.)\s*No\.?\s*$")

SG_NEUTRAL_ABBR_RE = re.compile(r"^SG[A-Z]+(?:\([A-Z]\))?$")
SG_REPORT_ABBR_RE = re.compile(r"^SLR(?:\s?\(R\))?$")

# Pinpoint after a citation: "at [47]", "at [14]–[15]", "at [38] and [136]".
_PINPOINT = r"at\s*\[[^\]]{1,12}\](?:\s*(?:–|-|to|and|,)\s*\[[^\]]{1,12}\])*"
PINPOINT_RE = re.compile(r"\s*(?P<pin>" + _PINPOINT + r")")

# A defined name in brackets: (“KS v LHY”), (the “Default Judgment”). A ’
# inside a word ("Women’s Charter") does not close the name.
DEFINITION_RE = re.compile(
    r"\s*\(\s*(?:the\s+|hereafter\s+|hereinafter\s+)?[“\"‘]"
    r"(?P<name>(?:[^”\"’]|’(?=\w)){1,80})[”\"’]\s*\)"
)

# Decision 18: a word or year in front of a short name is part of it
# ("the new LTA", "the old SCJA", "the 2019 RTA").
_MODIFIER = r"(?:new|old|current|amended|repealed|former|present|previous|\d{4})"

# Decision 20: a case name used as the name of a test is not labelled
# ("the Saudi Eagle test", "the Evans v Bartlam test").
TEST_AFTER_RE = re.compile(
    r"\s+(?:test|doctrine|principle|approach|framework|formulation|threshold)s?\b"
)

# A supra reference names a case cited earlier: "Bayoil ([36] supra)",
# "DNG ([27] above)". The name is 1-6 capitalised tokens. "(see [21] above)"
# after a party or a document is a cross-reference (decision 13), so "see"
# and "the" before the name rule it out.
_SHORT_TOKEN = r"(?:[A-Z][\w’'&.-]*|\((?:No\s*\d+|[A-Z]{1,4})\))"
SUPRA_RE = re.compile(
    rf"(?<![\w’'])(?<!the )(?P<name>{_SHORT_TOKEN}(?:\s+(?:{_SHORT_TOKEN}|v|of|and|&)){{0,5}})"
    r"\s*\(\s*(?:at\s+)?\[\d+\]\s*,?\s*(?:supra|above)\s*\)"
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
    "a", "trading", "as", "nee", "formerly", "deceased", "estate", "other",
    "la", "le", "du", "di", "da", "del", "des", "bankrupt", "decd", "v.",
}  # fmt: skip

# Connectors that can also open a name ("de Dampierre v de Dampierre").
NAME_PARTICLES = {"de", "von", "van", "der", "bin", "binti", "binte"}

# Capitalised words that open a sentence or clause, not a case name.
CASE_NAME_LEAD_STOP = {
    "In", "See", "Cf", "Also", "And", "The", "As", "Following", "Applying",
    "Per", "Unlike", "Like", "Similarly", "Thus", "Hence", "Further", "Finally",
    "However", "Moreover", "Indeed", "Accordingly", "Therefore", "Again", "Here",
    "First", "Second", "Third", "Lastly", "Notably", "Nevertheless", "Conversely",
    "Distinguishing", "Citing", "Contrast", "Compare", "Both", "Neither", "Either",
    "Likewise", "Instead", "Recently", "Earlier", "Later", "Since", "Although",
}  # fmt: skip

# A bracketed phrase inside a party's name: "(formerly known as …)", "(in
# liquidation)", "(administrator of the estate of …, deceased)", "(S)", "(Aust.)".
_NAME_BRACKET_RE = re.compile(
    r"\((?:(?:formerly|in|administrator|administratrix|executor|executrix|trading|suing|sued"
    r"|as|by|a|an|previously|now|f\.?k\.?a\.?)\b[^()\[\];]{0,160}|[A-Z][A-Za-z.]{0,5})\)"
)
# Tokens with a full stop that sit inside a name: initials ("Y.E.S.", "C.M."),
# company words ("Pty.", "Ltd.", "Co.") and "v.".
_DOTTED_NAME_TOKEN_RE = re.compile(
    r"\(?(?:[A-Z]\.){1,5}[A-Z]?\.?,?|\(?(?:Co|Ltd|Inc|Bhd|Corp|No|St|Pty|Aust|Bros|Mfg|Plc|Pte"
    r"|Ltda|Assn|Intl)\.\)?,?|v\."
)

# "(No 2)" ends a case name: "Carl Zeiss Stiftung v Rayner & Keeler Ltd (No 2)".
_NUMBER_SUFFIX_RE = re.compile(r"\s*\(No\.?\s*\d+\)$")
# A ship's name stands for an admiralty case: The “Jian He”, The Playa Larga.
SHIP_NAME_RE = re.compile(
    r"(?<![\w’'])The\s+[“\"]?[A-Z][\w’'-]*(?:\s+[A-Z0-9][\w’'-]*){0,3}[”\"]?$"
)

_TOKEN_RE = re.compile(r"\S+")
# "Ms How ([318] above)" points back at a person, not a case.
_PERSONAL_TITLES = {"Mr", "Ms", "Mrs", "Mdm", "Miss", "Dr", "Prof", "Justice", "Judge"}
_NAME_WINDOW = 300


def _case_name_before(text: str, end: int, ships: bool = True) -> Optional[tuple]:
    """Return ``(start, end)`` of a case name ending at ``end``, or None.

    A case name is a run of capitalised tokens and connectors that contains
    `` v `` or opens with Re / In re / Ex parte. The walk stops at the first
    lower-case word that is not a connector, at a number that is not inside
    a party's name, at clause punctuation (``;`` ``:`` ``[`` ``]``), or at the
    "and" before a second `` v `` (two names in a row). A ship's name
    (``The “Jian He”``) counts when ``ships`` is set, i.e. before a citation.
    """
    window_start = max(0, end - _NAME_WINDOW)
    window = text[window_start:end].rstrip()
    suffix = _NUMBER_SUFFIX_RE.search(window)
    if suffix:
        name = _case_name_before(text, window_start + suffix.start(), ships)
        return (name[0], window_start + len(window)) if name else None
    tokens = list(_TOKEN_RE.finditer(window))
    kept: List[re.Match] = []
    seen_v = False
    i = len(tokens)
    while i > 0:
        i -= 1
        tok = tokens[i]
        word = tok.group(0)
        # A bracketed phrase inside a name is taken whole: "(in liquidation)".
        if word.endswith(")") and "(" not in word:
            j = next((k for k in range(i - 1, max(-1, i - 25), -1)
                      if tokens[k].group(0).startswith("(")), None)  # fmt: skip
            if j is not None and _NAME_BRACKET_RE.fullmatch(window, tokens[j].start(), tok.end()):
                kept.extend(reversed(tokens[j : i + 1]))
                i = j
                continue
        if word.startswith("[") and len(word) > 1 and word[1].isupper() and "]" not in word:
            # "[Hyundai Shipbuilding v Pournaras [1978] …": the name opens here.
            kept.append(tok)
            break
        if any(ch in word for ch in ";:[]") or word.endswith("."):
            # "Ltd.", "Pty." or "Y.E.S." can be in a name; ". " usually ends a
            # sentence.
            if not _DOTTED_NAME_TOKEN_RE.fullmatch(word):
                break
        bare = word.strip('(),“”"')
        # A closing definition, (“ETA”), ends the clause before the name.
        if not bare or word.endswith(("”)", '")', "’)")):
            break
        if word in ("v", "v."):
            if seen_v:
                # A second name: cut at the "and" that joins the two.
                ands = [k for k, t in enumerate(kept) if t.group(0) == "and"]
                kept = kept[: ands[-1]] if ands else []
                break
            seen_v = True
        if bare[0].isdigit():
            # "BTI 2014 LLC": a number between two capitalised words; "6DM",
            # a name that opens with a digit; "Plan No 301", a numbered name.
            inside = (
                word.isalnum()
                and kept
                and kept[-1].group(0)[0].isupper()
                and i > 0
                and tokens[i - 1].group(0)[0].isupper()
            )
            numbered = i > 0 and tokens[i - 1].group(0) in ("No", "No.") and bare.isdigit()
            lettered = bool(re.fullmatch(r"\d+[A-Z][A-Za-z]*", bare)) and bool(kept)
            if not (inside or numbered or lettered):
                break
            kept.append(tok)
            continue
        if bare[0].isupper() or word in CASE_NAME_CONNECTORS or bare in CASE_NAME_CONNECTORS:
            kept.append(tok)
            continue
        break
    kept.reverse()
    if ships and not seen_v:
        ship = SHIP_NAME_RE.search(window)
        if ship and " v " not in ship.group(0):
            return window_start + ship.start(), window_start + ship.end()
    # Trim leading clause openers and connectors.
    while kept:
        first = kept[0].group(0).strip('(“",[')
        # A list marker: "(a) In WGE v WGF".
        if len(kept) > 1 and re.fullmatch(r"\(\w{1,4}\)", kept[0].group(0)):
            kept.pop(0)
            continue
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
    words = ["v" if t.group(0) == "v." else t.group(0) for t in kept]
    opens_re = words[0] in ("Re", "In", "Ex") and (
        words[0] == "Re" or (len(words) > 1 and words[1] in ("re", "parte"))
    )
    if "v" not in words and not opens_re:
        return None
    start = window_start + kept[0].start()
    stop = window_start + kept[-1].end()
    # A case name never starts with an opening bracket or quote, nor ends
    # with a comma.
    while start < stop and text[start] in '(“"[':
        start += 1
    while stop > start and text[stop - 1] in ",":
        stop -= 1
    return start, stop


_COMPANY_SUFFIXES = {"Co", "Ltd", "Inc", "Bhd", "Corp", "LLC", "Limited", "Pte", "plc", "PLC"}
# Forward from " v ": the second party's tokens, up to a lower-case word.
_PARTY_STOP_PUNCT = re.compile(r"[,;:.)\]]$")


def _case_name_around(text: str, v_pos: int) -> Optional[tuple]:
    """A case name written without a citation, found from its `` v ``.

    Walks forward over the second party (capitalised tokens and connectors,
    up to a lower-case word, clause punctuation or another `` v ``), then
    back over the first party with ``_case_name_before``.
    """
    v_end = v_pos + (3 if text.startswith("v.", v_pos) else 2)
    after = list(_TOKEN_RE.finditer(text, v_end, min(len(text), v_end + _NAME_WINDOW)))
    if not after or not after[0].group(0).lstrip('(“"')[:1].isupper():
        return None
    stop = None
    for k, tok in enumerate(after):
        word = tok.group(0)
        bare = word.strip('(),“”"’')
        if word in ("v", "v."):
            # Two names in a row: this one ends before the "and" joining them.
            ands = [j for j in range(k) if after[j].group(0) == "and"]
            stop = after[ands[-1] - 1] if ands and ands[-1] > 0 else None
            break
        if not bare:
            break
        if bare[0].isupper() or bare[0].isdigit() or bare in CASE_NAME_CONNECTORS:
            # "BTI 2014 LLC" has a number inside; "Co 78 Mass App Ct" is a
            # reporter citation after the name.
            if bare[0].isdigit() and not (
                k + 1 < len(after)
                and after[k + 1].group(0)[:1].isupper()
                and after[k - 1].group(0).strip(".,") not in _COMPANY_SUFFIXES
            ):
                break
            stop = tok
            if _PARTY_STOP_PUNCT.search(word) and not word.endswith(")") or word.endswith("."):
                break
            continue
        break
    if stop is None:
        return None
    # Trailing connectors are not part of the name ("… Ltd and Fraser …"),
    # nor an unclosed bracket ("PP v AOP (a minor)" ends at "AOP").
    kept = [t for t in after if t.start() <= stop.start()]
    while kept and (
        kept[-1].group(0).strip(",.") in CASE_NAME_CONNECTORS
        or (kept[-1].group(0).startswith("(") and ")" not in kept[-1].group(0))
    ):
        kept.pop()
    if not kept:
        return None
    end = kept[-1].end()
    while end > v_pos and text[end - 1] in ',.;:’”"':
        if text[end - 1] == "." and re.search(r"(?:Co|Ltd|Inc|Bhd|Corp)\.$", text[:end]):
            break
        end -= 1
    name = _case_name_before(text, end, ships=False)
    if name and name[0] < v_pos:
        return name
    return None


# ---------------------------------------------------------------------------
# Statutes
# ---------------------------------------------------------------------------

_STATUTE_KEYWORD = (
    r"(?:Act|Bill|Code|Charter|Ordinance|Rules|Regulations|Constitution|Treaty|Convention"
    r"|Protocol)"
)
# A defined name that names an Act (decision 17): "Penal Code", "SIAC Rules".
ACT_NAME_RE = re.compile(rf"\b(?:{_STATUTE_KEYWORD}|Law)\b")
INSTRUMENT_RE = re.compile(
    # The lookbehind stops a title starting inside "s 11A of the …".
    r"(?<![\w’'])(?P<title>"
    r"(?:[A-Z][\w’'-]*\s+)"
    # A bracketed word can sit inside: "Administration of Justice (Protection) Act".
    r"(?:(?:\(?[A-Z][\w’'()-]*|of|and|the|for|in|on|to|by|&)\s+){0,10}?"
    # A second keyword continues the title: "Energy Charter Treaty".
    rf"{_STATUTE_KEYWORD}(?:\s+{_STATUTE_KEYWORD})?\b"
    r"(?:\s+of\s+(?:the\s+)?[A-Z][\w’']*(?:\s+(?:of\s+)?[A-Z][\w’']*)*)?"
    r"|Rules\s+of\s+Court"
    # A title can open with its keyword: "Constitution of the Republic of Singapore".
    r"|(?:Constitution|Charter|Convention|Treaty|Rules)\s+of\s+(?:the\s+)?[A-Z][\w’']*"
    r"(?:\s+(?:of\s+)?[A-Z][\w’']*)*"
    r")"
)
# A title goes on past a keyword through bracketed words to another keyword:
# "Women’s Charter (Family Violence and Other Matters) (Amendment) Act",
# "Carriage by Air (Montreal Convention, 1999) Act".
_TITLE_END_KEYWORD = r"\s+(?:Act|Bill|Rules|Regulations|Code|Order)\b"
TITLE_GOES_ON_RE = re.compile(r"(?:\s+\([^()]{1,60}\))+" + _TITLE_END_KEYWORD)
TITLE_BRACKET_CLOSES_RE = re.compile(r"[^()]{0,30}\)(?:\s+\([^()]{1,60}\))*" + _TITLE_END_KEYWORD)
# Year, edition and number after a title: "1996 (Act 11 of 1996)",
# "(Cap 224, 2008 Rev Ed)", "(1985 Rev Ed, 1999 Reprint)", "Cap 354".
INSTRUMENT_TAIL_RE = re.compile(
    r"(?:\s+(?P<year>\d{4}))?"
    r"(?:\s*\((?P<edition>(?:Cap\s*[\w.]+,\s*)?(?:R\s*\d+,\s*)?\d{4}\s+Rev\s+Ed"
    r"(?:,\s*\d{4}\s+Reprint)?|Cap\s*[\w.]+|UK|Cth|NSW|India)\)|\s+(?P<cap>Cap\s*\d+[A-Z]?)\b)?"
    r"(?:\s*\((?P<number>(?:Act|Bill)\s+(?:No\.?\s*)?\d+\s+of\s+\d{4})\))?"
)
# "Second Reading of the Road Traffic (Amendment) Act 1996": the debate is
# not part of the title.
READING_PREFIX_RE = re.compile(
    r"(?:(?:First|Second|Third)\s+Reading|Opening\s+Speech)\s+of\s+(?:the\s+)?"
)
# Capitalised function words that open a clause before a statute title.
INSTRUMENT_LEAD_STOP = {
    "The", "In", "Under", "By", "See", "Pursuant", "As", "For", "Of", "And",
    "This", "That", "These", "Since", "While", "If", "Unlike", "Like", "Per",
    "Following", "Section", "Sections", "Part", "Both", "Whereas", "To", "An", "A",
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

_NUM = r"\d+(?:\.\d+)*[A-Z]*(?:\(\w{1,5}\))*"
_SUBNUM = r"(?:\(\w{1,5}\))+"
_LIST_JOIN = r"(?:\s*[–-]\s*|\s+to\s+|\s+and\s+|\s+or\s+|,\s*)"
PROVISION_RE = re.compile(
    # Not after an apostrophe: "Respondent’s 25%" is a possessive, not s 25.
    r"(?<![\w’'])(?P<prov>"
    # A paragraph of a Schedule: "para 4 of the First Schedule"
    r"(?:[Pp]ara(?:graph)?s?)\s*\d+[A-Z]*(?:\(\w{1,5}\))*"
    r"(?:(?:\s+and\s+|,\s*|\s*[–-]\s*)\d+[A-Z]*(?:\(\w{1,5}\))*)*"
    r"(?=\s+(?:of|to|in)\s+(?:the\s+)?(?:\w+\s+)?Schedule\b)"
    # A paragraph or subsection of a provision: "para (e)", "subsection (2)",
    # "Subsection (5) of s 33"
    rf"|(?:[Pp]ara(?:graph)?s?|[Ss]ub-?sections?|subss?)\s*{_SUBNUM}"
    rf"(?:{_LIST_JOIN}{_SUBNUM})*"
    rf"(?:\s+of\s+(?:s|section)\s*{_NUM})?"
    # Rules of Court: O 3 r 2(8), Order 89B r 2(2), O 2 rr 1(1) and 1(2),
    # Order 110, Rule 46(1), and a whole Order: O 24, Order 89
    rf"|(?:O|Order)\s*(?!(?:19|20)\d\d\b)\d+[A-Z]?(?:,?\s+(?:rr?|[Rr]ules?)\s*{_NUM}(?:{_LIST_JOIN}{_NUM})*)?"
    rf"|(?:ss?|[Ss]ections?|regs?|[Rr]egulations?|rr?|Arts?|arts?|Articles?)\s*{_NUM}"
    rf"(?:{_LIST_JOIN}(?:{_NUM}|{_SUBNUM}))*"
    # "Rules 1 and 2 of that Order", but not the year in "Family Justice Rules 2014"
    rf"|Rules?\s+(?!(?:19|20)\d\d\b){_NUM}(?:{_LIST_JOIN}{_NUM})*"
    # Decision 9: a Schedule on its own, "First Schedule", "Schedule 2"
    r"|(?:(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh"
    r"|Twelfth)\s+Schedule|Schedule\s+\d+[A-Z]?)"
    r")(?![\w%])"
    # Decision 9: "of the Schedule" belongs to the provision: "s 1A of the
    # Schedule", "para 1 of the Second Schedule".
    r"(?:\s+(?:of|to|in)\s+(?:the\s+(?:(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh"
    r"|Eighth|Ninth|Tenth|Eleventh|Twelfth)\s+)?Schedule\b(?!\s+\d)|Schedule\s+\d+[A-Z]?\b))?"
)

# Decision 14: a contract is not an instrument, so "Art 26.1 of the Contract"
# is not a PROVISION. The contract word must end the title, so "s 3 of the
# Unfair Contract Terms Act" still counts.
CONTRACT_AFTER_RE = re.compile(
    r"\s+(?:of|in|to|under)\s+(?:the\s+|this\s+|that\s+|its\s+|their\s+)?"
    r"(?:(?:[A-Z][\w’'-]*|and|of|for)\s+){0,4}?"
    r"(?:Contract|Subcontract|Sub-[Cc]ontract|Agreement|Deed|Lease|Tenancy|Charterparty"
    r"|Charter\s?[Pp]arty|SPA|Policy|Terms|Conditions|Guarantee|Bond|Memorandum)s?\b"
    r"(?![ \t]+[A-Z])"
)
# Decision 15: in "his section 122(6) statement" or "the s 23 statement" the
# number is part of the document's name, not a reference to the section.
NAMED_DOCUMENT_AFTER_RE = re.compile(r"\s+(?:[A-Z]{2,6}\s+)?statements?\b")

# An Act abbreviation after a provision is a short name even where the
# definition is out of sight: "s 52 of the TCA". So is "the Act" standing
# for one Act (decision 16); the span is "Act" alone.
ABBR_AFTER_PROVISION_RE = re.compile(
    rf"\s+of\s+(?:(?P<the>the)\s+)?(?:(?P<mod>{_MODIFIER})\s+)?"
    r"(?P<abbr>[A-Z][A-Z0-9]{1,7}|Act|Code)\b"
    r"(?![ \t]+(?:[A-Z(]|\d))(?!\s*\(\s*(?:Cap|\d))"
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
    # CITATION: sg_neutral | sg_report | other_report; CASE_NAME: cited |
    # uncited; SHORT_NAME: defined | supra | after_provision | use
    kind: Optional[str] = None
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


def defined_names(mentions: Iterable[Mention]) -> Set[str]:
    """Short names that later text may use: those defined or found by rule.

    A name defined with "the" (“the Rules”) is kept with it, so only "the
    Rules" counts as a later use, not every "Rules".
    """
    return {
        m.meta.get("name", m.text) for m in mentions if m.label == "SHORT_NAME" and m.kind != "use"
    }


def _short_name(text: str, start: int, end: int, kind: str, **meta) -> Mention:
    """A SHORT_NAME span, leaving out a leading "the" (guide, SHORT_NAME row)."""
    name = text[start:end]
    if re.match(r"[Tt]he\s", name):
        meta["name"] = "the " + name[4:]
        start += 4
    return Mention(start, end, "SHORT_NAME", text[start:end], kind=kind, meta=meta)


def detect_judgment(texts: Iterable[str]) -> List[List[Mention]]:
    """``detect`` over a judgment's strings in reading order.

    A short name defined in one string is found in every later one, so
    "Bayoil" or "the TCA" is labelled where the judgment uses it again.
    """
    known: Set[str] = set()
    out: List[List[Mention]] = []
    for text in texts:
        mentions = detect(text, known)
        known |= defined_names(mentions)
        out.append(mentions)
    return out


def judgment_parts(fragments: Iterable[dict]) -> List[tuple]:
    """``(fragment_id, part, text)`` for each string of a judgment, in reading order.

    ``fragments`` are ``judgments_fragments`` rows (or ``extract_judgment``
    fragments) in ordinal order; a fragment's footnotes follow its content.
    ``part`` is "content" or "footnote N", as in the Ellf task meta.
    """
    parts = []
    for f in fragments:
        parts.append((f["id"], "content", f["content_text"] or ""))
        notes = json.loads(f["footnote_text"]) if f["footnote_text"] else []
        parts += [(f["id"], f"footnote {i}", note) for i, note in enumerate(notes, 1)]
    return parts


def detect_parts(fragments: Iterable[dict]) -> dict:
    """``{(fragment_id, part): (text, mentions)}`` for a whole judgment."""
    parts = judgment_parts(fragments)
    mentions = detect_judgment(text for _, _, text in parts)
    return {(fid, part): (text, ms) for (fid, part, text), ms in zip(parts, mentions)}


def _uses(text: str, names: Iterable[str]) -> List[Mention]:
    """Later uses of known short names, as whole words.

    A name kept with "the" ("the Act") matches only after "the", and not
    where a title goes on ("the Act 1967", "the Rules of Court" is left to
    the INSTRUMENT rule, which outranks a use).
    """
    found = []
    for name in names:
        with_the = name.startswith("the ")
        bare = name[4:] if with_the else name
        if len(bare) < 2 or not any(ch.isupper() for ch in bare):
            continue
        pattern = (
            r"(?<![\w’'])"
            + (r"[Tt]he\s+" if with_the else "")
            + rf"(?:(?P<mod>{_MODIFIER})\s+)?"
            + re.escape(bare)
        )
        for m in re.finditer(pattern + r"(?![\w(])", text):
            after = text[m.end() :]
            if NAMED_DOCUMENT_AFTER_RE.match(after) or (
                with_the and re.match(r"[ \t]+(?:[A-Z(]|\d)", after)
            ):
                continue
            # Decision 18: "the new LTA" is labelled "new LTA".
            start = m.start("mod") if m.group("mod") else m.end() - len(bare)
            found.append(Mention(start, m.end(), "SHORT_NAME", text[start : m.end()], kind="use"))
    return found


def detect(text: str, known: Optional[Iterable[str]] = None) -> List[Mention]:
    """Find citation mentions in one string, sorted by start offset.

    ``known`` holds short names defined earlier in the judgment; their uses
    here are labelled too (``detect_judgment`` passes them). Overlaps are
    resolved by priority: citations and case numbers, then case names and
    defined names, then statute titles, then provisions, then uses of
    known names.
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
            found.append(
                Mention(name[0], name[1], "CASE_NAME", text[name[0] : name[1]], kind="cited")
            )
        if definition:
            found.append(
                _short_name(
                    text,
                    definition.start("name"),
                    definition.end("name"),
                    "defined",
                    defined_after=m.start(),
                )
            )

    # Decision 10: a case name written out in full, with no citation after it.
    for vm in re.finditer(r"(?<=\s)v\.?(?=\s)", text):
        name = _case_name_around(text, vm.start())
        if name:
            found.append(
                Mention(name[0], name[1], "CASE_NAME", text[name[0] : name[1]], kind="uncited")
            )

    for m in SUPRA_RE.finditer(text):
        tokens = list(_TOKEN_RE.finditer(m.group("name")))
        while tokens and tokens[0].group(0) in CASE_NAME_LEAD_STOP | {"v", "of", "and", "&"}:
            tokens.pop(0)
        if not tokens or tokens[0].group(0) in _PERSONAL_TITLES:
            continue
        start = m.start("name") + tokens[0].start()
        found.append(
            Mention(start, m.end("name"), "SHORT_NAME", text[start : m.end("name")], kind="supra")
        )

    for m in CASE_NUMBER_RE.finditer(text):
        if _NOT_A_CASE_NUMBER_RE.search(m.group("num")):
            continue
        if _GAZETTE_BEFORE_RE.search(text, 0, m.start("num")):
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
        title_end = m.end("title")
        for prefix in (SCHEDULE_PREFIX_RE, READING_PREFIX_RE):
            part = prefix.match(text, start, title_end)
            if part:
                start = part.end()
                tokens = list(_TOKEN_RE.finditer(text[start:title_end]))
        while True:
            unclosed = text.count("(", start, title_end) > text.count(")", start, title_end)
            more = (TITLE_BRACKET_CLOSES_RE if unclosed else TITLE_GOES_ON_RE).match(
                text, title_end
            )
            if not more:
                break
            title_end = more.end()
        title = text[start:title_end]
        if (
            not tokens
            or title in _INSTRUMENT_TOO_GENERIC
            or any(t.group(0) in _INSTRUMENT_NOT_TITLE_WORDS for t in tokens)
        ):
            continue
        tail = INSTRUMENT_TAIL_RE.match(text, title_end)
        end = tail.end()
        edition = tail.group("edition") or tail.group("cap")
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
                        tail.group("year"),
                        edition and f"({edition})",
                        tail.group("number") and f"({tail.group('number')})",
                    )
                    if p
                ),
                meta={
                    "title": title,
                    "year": tail.group("year"),
                    "edition": edition,
                    "number": tail.group("number"),
                },
            )
        )
        definition = DEFINITION_RE.match(text, end)
        if definition:
            found.append(
                _short_name(
                    text,
                    definition.start("name"),
                    definition.end("name"),
                    "defined",
                    defined_after=start,
                )
            )

    for m in PROVISION_RE.finditer(text):
        end = m.end()
        if CONTRACT_AFTER_RE.match(text, end) or NAMED_DOCUMENT_AFTER_RE.match(text, end):
            continue
        # "Rule 5", "Order 2" or "Schedule 3" after a capitalised word is part
        # of a title or a name ("Family Justice Rule 5", "Transfer Order 2").
        if re.match(r"(?:Rules?|Order|Schedule)\s", m.group("prov")) and re.search(
            r"(?<![\w’'])[A-Z][\w’'-]*\s+$", text[max(0, m.start() - 30) : m.start()]
        ):
            continue
        found.append(
            Mention(
                m.start("prov"),
                end,
                "PROVISION",
                text[m.start("prov") : end],
                normalised=re.sub(r"\s+", " ", text[m.start("prov") : end]),
            )
        )
        abbr = ABBR_AFTER_PROVISION_RE.match(text, end)
        if abbr and (abbr.group("abbr") not in ("Act", "Code") or abbr.group("the")):
            generic = abbr.group("abbr") in ("Act", "Code")
            if generic:
                start = abbr.start("the")
            else:
                start = abbr.start("mod") if abbr.group("mod") else abbr.start("abbr")
            short = _short_name(text, start, abbr.end("abbr"), "after_provision")
            # A later use is the name without its modifier: "the Act", "SCJA".
            short.meta["name"] = ("the " if generic else "") + abbr.group("abbr")
            found.append(short)

    # Decision 17: an Act's name defined in brackets is a short name wherever the
    # definition sits ("… 1999) Act 2007 (the “Montreal Convention”)").
    for d in DEFINITION_RE.finditer(text):
        name = d.group("name").strip()
        if ACT_NAME_RE.search(name) and name[0].isupper() and name not in _INSTRUMENT_TOO_GENERIC:
            found.append(_short_name(text, d.start("name"), d.end("name"), "defined"))

    names = set(known or ()) | defined_names(found)
    uses = _uses(text, names)
    use_spans = {(u.start, u.end) for u in uses}
    found = [
        m
        for m in found
        # A defined Act name is a short name, not a title (decision 17), and a
        # case name defined earlier is a use of its short name.
        if not (
            (m.start, m.end) in use_spans
            and (
                (m.label == "INSTRUMENT" and not (m.meta.get("year") or m.meta.get("edition")))
                or m.kind == "uncited"
            )
        )
    ]
    found += uses
    found = [
        m
        for m in found
        if not (
            (m.kind == "uncited" or m.label == "SHORT_NAME") and TEST_AFTER_RE.match(text, m.end)
        )
    ]
    return _resolve_overlaps(found)


_PRIORITY = {
    "CITATION": 0,
    "CASE_NUMBER": 0,
    "CASE_NAME": 1,
    "SHORT_NAME": 1,
    "INSTRUMENT": 2,
    "PROVISION": 3,
}


def _priority(m: Mention) -> float:
    # A name found without a citation, or a use of a known name, gives way
    # to anything the stronger rules found.
    if m.kind == "uncited":
        return 1.5
    if m.kind == "use":
        return 4
    return _PRIORITY[m.label]


def _resolve_overlaps(mentions: List[Mention]) -> List[Mention]:
    kept: List[Mention] = []
    for m in sorted(mentions, key=lambda x: (_priority(x), x.start, -(x.end - x.start))):
        if any(m.start < k.end and k.start < m.end for k in kept):
            continue
        kept.append(m)
    return sorted(kept, key=lambda x: x.start)
