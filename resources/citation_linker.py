"""Link detected case citations to rows in ``judgments``.

Detection (``citation_rules``) only finds spans; this module decides which
judgment a citation refers to, following eyecite's split. It never fetches:
the caller passes the ``judgments`` rows (in the build, the table itself).

Two kinds of citation can be linked:

- **Neutral** (``[2026] SGHC 62``): exact match on ``judgments.citation``.
  A citation missing from the catalog is usually a State Courts or tribunal
  decision that eLitigation never published.
- **SLR** (``[2012] 4 SLR 308``, ``[2001] 2 SLR(R) 435``): there is no SLR to
  neutral map, so the case name written before the citation is matched
  against judgments decided in the report year or the year before (option
  (a) in the design doc). The High Court and Court of Appeal decisions in
  one case usually share a name and fall in that window, so a match can tie.
  A trial and its appeal are separate decisions, so a tie is never settled
  by a default: ``break_tie_by_cues`` (the citing passage),
  ``break_tie_by_text`` (the later candidate's text) and
  ``break_tie_by_pinpoint`` (the paragraph cited) settle some on evidence,
  and the rest stay unlinked.

Every result carries a tier, so callers can decide how much to trust it:

=============  ==============================================================
``exact``      neutral citation found once in the catalog
``unique``     one judgment's name matches clearly
``resolved``   a tie settled by evidence: the cited name ("and another
               appeal"), the citing passage naming the court or the date,
               the later candidate's own text, or the pinpointed paragraph
``tied``       two or more judgments match about equally; not linked
``weak``       best match falls below the threshold; not linked
``none``       nothing matched
``too_early``  report year is before the catalog starts (pre-2000 cases)
=============  ==============================================================
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from resources.citation_rules import CITATION_RE, Mention, _citation_kind, _normalise_citation

LINKER_VERSION = "link-1"

# A name match at or above this share links; below it is "weak". On the gold
# set the lowest right match is 0.75 (a misspelt party name) and the highest
# wrong one 0.33 (a different company in the same liquidation).
MATCH_THRESHOLD = 0.6
# Candidates within this of the best score count as tied with it.
TIE_MARGIN = 0.1
# A tied candidate wins outright when its name has this much less padding,
# but only against candidates from courts of the same level.
CLOSENESS_MARGIN = 0.15

# Court of Appeal, its international bench, and the Appellate Division.
APPELLATE_COURTS = {"SGCA", "SGCAI", "SGHCA"}
APPEALS_RE = re.compile(r"\b(?:another|other)\s+appeals?\b", re.IGNORECASE)

LINKED_TIERS = {"exact", "unique", "resolved"}

# ---------------------------------------------------------------------------
# Case names
# ---------------------------------------------------------------------------

# Words that say nothing about which case it is: company forms, joinder
# phrases ("and another appeal", "& 4 Ors"), and procedural words.
_FILLER = {
    "a",
    "an",
    "and",
    "another",
    "others",
    "other",
    "anor",
    "ors",
    "appeal",
    "appeals",
    "matter",
    "matters",
    "suit",
    "suits",
    "the",
    "of",
    "in",
    "on",
    "for",
    "by",
    "as",
    "re",
    "ex",
    "parte",
    "pte",
    "pvt",
    "ltd",
    "limited",
    "private",
    "co",
    "company",
    "inc",
    "corp",
    "corporation",
    "bhd",
    "sdn",
    "berhad",
    "llc",
    "llp",
    "plc",
    "liquidation",
    "compulsory",
    "creditors",
    "voluntary",
    "judicial",
    "management",
    "formerly",
    "known",
    "sued",
    "firm",
    "trading",
    "doing",
    "business",
    "under",
    "name",
    "style",
    "no",
}
_V_RE = re.compile(r"\s+v\.?\s+", re.IGNORECASE)


def name_tokens(name: str) -> frozenset:
    """Distinctive lower-case words of a case name, or one side of it.

    "PP" becomes "public prosecutor"; dotted abbreviations close up
    ("S.R.L." → "srl", "Pte. Ltd." → "pte ltd"); possessives and
    single letters (from "s/o", "a/l") drop out.
    """
    n = name.lower().replace("&", " and ").replace("’", "'")
    n = re.sub(r"\bpp\b", "public prosecutor", n)
    n = re.sub(r"\b(\w)\.(?=\w\.)", r"\1", n)  # s.r.l. → sr.l. → srl.
    n = re.sub(r"(?<=\w)\.(?=\w)", "", n)
    n = re.sub(r"'s\b", "", n)
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return frozenset(w for w in n.split() if len(w) > 1 and w not in _FILLER)


def name_sides(name: str) -> Tuple[frozenset, ...]:
    """Tokens of each party side: two for "A v B", one for "Re A" or a ship."""
    return tuple(name_tokens(part) for part in _V_RE.split(name, maxsplit=1))


def name_score(query: str, candidate: str) -> float:
    """How well a cited name matches a catalog name, 0 to 1.

    The share of the cited name's words found in the catalog name, because
    catalog names often add parties ("& 4 Ors") or a former name. When both
    names have two sides, each side must share a word with the same side of
    the other, so "Public Prosecutor v Tan" never matches "Public Prosecutor
    v Lim".
    """
    q, c = name_tokens(query), name_tokens(candidate)
    if not q or not c:
        return 0.0
    qs, cs = name_sides(query), name_sides(candidate)
    if len(qs) == 2 and len(cs) == 2:
        if not (qs[0] & cs[0] and qs[1] & cs[1]):
            return 0.0
    return len(q & c) / len(q)


def name_closeness(query: str, candidate: str) -> float:
    """Shared words over all words (Jaccard): breaks ties between full matches.

    "Muhammad bin Abdullah" is in both "Muhammad bin Abdullah" and
    "Muhammad Nur bin Abdullah"; only the first has no extra words.
    """
    q, c = name_tokens(query), name_tokens(candidate)
    return len(q & c) / len(q | c) if q and c else 0.0


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Judgment:
    id: str
    citation: str
    case_name: str
    decision_date: str
    court: str = ""


def _citation_key(citation: str) -> str:
    return re.sub(r"\s+", "", citation or "")


class Catalog:
    """``judgments`` rows indexed by neutral citation and decision year."""

    def __init__(self, rows: Iterable[dict]):
        seen = set()
        self.by_citation = defaultdict(list)
        self.by_year = defaultdict(list)
        for row in rows:
            if row["id"] in seen or not row.get("decision_date"):
                continue
            seen.add(row["id"])
            j = Judgment(
                row["id"],
                row.get("citation") or "",
                row.get("case_name") or "",
                row["decision_date"],
                row.get("court") or "",
            )
            self.by_citation[_citation_key(j.citation)].append(j)
            self.by_year[int(j.decision_date[:4])].append(j)
        self.by_id = {j.id: j for js in self.by_year.values() for j in js}
        self.first_year = min(self.by_year) if self.by_year else 0

    @classmethod
    def from_db(cls, db) -> "Catalog":
        """From a sqlite-utils ``Database`` holding the ``judgments`` table."""
        return cls(db.query("select id, citation, case_name, decision_date, court from judgments"))


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


@dataclass
class Link:
    tier: str
    target_id: Optional[str] = None
    # (judgment id, score) for every candidate considered, best first.
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    note: str = ""

    @property
    def linked(self) -> bool:
        return self.tier in LINKED_TIERS


def link_neutral(catalog: Catalog, citation: str) -> Link:
    """A Singapore neutral citation, by exact match."""
    found = catalog.by_citation.get(_citation_key(citation), [])
    if len(found) == 1:
        return Link("exact", found[0].id, [(found[0].id, 1.0)])
    if found:
        return Link("tied", None, [(j.id, 1.0) for j in found], "citation listed twice")
    return Link("none", note="not in catalog")


def link_report(
    catalog: Catalog,
    case_name: Optional[str],
    report_year: int,
    citing_id: Optional[str] = None,
    citing_date: Optional[str] = None,
) -> Link:
    """An SLR citation, by the case name cited with it and the report year.

    Candidates are judgments decided in ``report_year`` or the year before.
    The citing judgment itself, and judgments decided after it, are left
    out: a judgment cannot cite itself or a later one.
    """
    if report_year < catalog.first_year:
        return Link("too_early")
    if not case_name:
        return Link("none", note="no case name before the citation")
    scored = []
    for year in (report_year, report_year - 1):
        for j in catalog.by_year.get(year, []):
            if j.id == citing_id or (citing_date and j.decision_date > citing_date):
                continue
            s = name_score(case_name, j.case_name)
            if s > 0:
                scored.append((s, j))
    scored.sort(key=lambda sj: (-sj[0], sj[1].decision_date))
    candidates = [(j.id, round(s, 3)) for s, j in scored[:5]]
    if not scored:
        return Link("none")
    best = scored[0][0]
    if best < MATCH_THRESHOLD:
        return Link("weak", None, candidates)
    top = [j for s, j in scored if s >= best - TIE_MARGIN]
    if len(top) > 1 and len({j.court in APPELLATE_COURTS for j in top}) == 1:
        # Among full matches in courts of one level, one with clearly fewer
        # extra words wins. A trial and its appeal are different decisions,
        # and the name the report uses says nothing about which one it is.
        close = sorted(
            ((name_closeness(case_name, j.case_name), j) for j in top), key=lambda cj: -cj[0]
        )
        if close[0][0] - close[1][0] >= CLOSENESS_MARGIN:
            top = [close[0][1]]
    if len(top) == 1:
        return Link("unique", top[0].id, candidates)
    # "X v Y and another appeal": consolidated appeals point to the appellate
    # court when only one tied candidate sits there.
    if APPEALS_RE.search(case_name):
        appellate = [j for j in top if j.court in APPELLATE_COURTS]
        if len(appellate) == 1:
            return Link("resolved", appellate[0].id, candidates, "name says appeal")
    return Link("tied", None, candidates)


def _report_citations(text: str) -> Iterable[str]:
    for m in CITATION_RE.finditer(text):
        if _citation_kind(m.group("abbr")) == "sg_report":
            yield _normalise_citation(m)


def break_tie_by_text(
    catalog: Catalog,
    link: Link,
    citation: str,
    case_name: str,
    texts_of: Callable[[str], Optional[Sequence[str]]],
) -> Link:
    """Settle a two-way tie from the later candidate's own text.

    The usual tie is a High Court decision and the Court of Appeal's
    decision on appeal from it. The later judgment often cites the earlier
    one by its SLR citation, and a judgment never cites itself, so:

    - if the later judgment cites this SLR citation, it belongs to the
      earlier one;
    - if the later judgment cites a different SLR citation under the same
      name, reported in the earlier judgment's year or the year after, that
      one is the earlier judgment's report, so this one is the later's.

    Only the later candidate's text counts: an earlier judgment that cites
    a report of the same name is citing some older decision in the dispute.

    ``texts_of(judgment_id)`` returns the candidate's strings (paragraphs
    and footnotes), or None when its text is not available.
    """
    if link.tier != "tied":
        return link
    best = link.candidates[0][1]
    tied = [catalog.by_id[jid] for jid, s in link.candidates if s >= best - TIE_MARGIN]
    if len(tied) != 2:
        return link
    earlier, later = sorted(tied, key=lambda j: j.decision_date)
    if earlier.decision_date == later.decision_date:
        return link
    texts = texts_of(later.id)
    if not texts:
        return link
    want = _citation_key(citation)
    earlier_year = int(earlier.decision_date[:4])
    cites_other = False
    for text in texts:
        for found in _report_citations(text):
            if _citation_key(found) == want:
                return Link("resolved", earlier.id, link.candidates, "the later judgment cites it")
            year = int(re.search(r"\d{4}", found).group())
            if year in (earlier_year, earlier_year + 1) and _same_case_cited_before(
                text, found, case_name
            ):
                cites_other = True
    if cites_other:
        return Link(
            "resolved",
            later.id,
            link.candidates,
            "the later judgment cites the earlier one's report",
        )
    return link


def _same_case_cited_before(text: str, citation: str, case_name: str) -> bool:
    """Whether ``citation`` in ``text`` follows a name matching ``case_name``."""
    key = _citation_key(citation)
    for m in CITATION_RE.finditer(text):
        if _citation_key(_normalise_citation(m)) != key:
            continue
        window = text[max(0, m.start() - 250) : m.start()]
        # The name ends right before the citation; try each "X v Y" ending there.
        for v in re.finditer(r"\s+v\.?\s+", window):
            start = window.rfind(". ", 0, v.start()) + 1
            if name_score(case_name, window[start:]) >= MATCH_THRESHOLD:
                return True
    return False


# ---------------------------------------------------------------------------
# Ties settled from the citing passage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """Where an SLR citation sits in the citing judgment."""

    text: str
    name_start: int  # start of the case name cited with it
    end: int  # end of the citation
    case_name: str
    citing_court: str = ""


COURT_OF_APPEAL = {"SGCA", "SGCAI"}
_MONTHS = (
    "January February March April May June July August September October November December"
).split()

# A court named just before the case name: "the Court of Appeal in", "the
# decision of this court in", "This Court's decision in", "our remarks in".
_COURT_BEFORE_RE = re.compile(
    r"\b(Appellate Division(?: of the High Court)?|Court of Appeal|High Court"
    r"|[Tt]his [Cc]ourt|[Oo]ur(?=\s+(?:decision|remarks|observations?|judgment|holding)))"
    r"(?:['’]s)?(?:\s+[a-z]+){0,4}?\s+in\s*$"
)
# Or just after the citation: "(“Glaziers”), the Court of Appeal held".
_COURT_AFTER_RE = re.compile(
    r"^\s*(?:\([^)]{0,40}\)\s*)?,?\s*(?:the|this)\s+"
    r"(Appellate Division|Court of Appeal|High Court|[Cc]ourt)\s+"
    r"(?:held|said|stated|observed|found|noted|explained|emphasi[sz]ed|recognised|recognized)\b"
)
# The cited decision was the one appealed from.
_TRIAL_AFTER_RE = re.compile(r"^[^.;]{0,60}?\b(?:affirmed|upheld|reversed) (?:on appeal|in)\b")
_DATE_RE = re.compile(r"\bon (\d{1,2}) (" + "|".join(_MONTHS) + r") (\d{4})\b")


def _tied(catalog: Catalog, link: Link) -> List[Judgment]:
    best = link.candidates[0][1]
    return [catalog.by_id[jid] for jid, s in link.candidates if s >= best - TIE_MARGIN]


def _court_test(phrase: str, citing_court: str) -> Optional[Callable[[str], bool]]:
    """Which candidate courts a phrase naming the court allows, if it says."""
    p = phrase.lower()
    if p.startswith("appellate division"):
        return lambda c: c == "SGHCA"
    if p == "court of appeal":
        return lambda c: c in COURT_OF_APPEAL
    if p == "high court":
        return lambda c: c not in APPELLATE_COURTS and c.startswith("SGHC")
    if not citing_court:
        return None
    # "this court" and "our decision" are the citing court's own level.
    if citing_court in APPELLATE_COURTS:
        return lambda c: c in APPELLATE_COURTS
    return lambda c: c not in APPELLATE_COURTS


def break_tie_by_cues(catalog: Catalog, link: Link, ctx: Context) -> Link:
    """Settle a tie from what the citing passage says about the decision.

    A trial and its appeal are separate decisions, so a tie is settled only
    when the passage names the court ("the Court of Appeal in", "this court"
    in an appellate judgment), says the decision went on appeal ("affirmed
    on appeal"), or gives the decision date ("dismissed … on 2 December
    2021"), and exactly one tied candidate fits.
    """
    if link.tier != "tied":
        return link
    tied = _tied(catalog, link)
    before = ctx.text[max(0, ctx.name_start - 120) : ctx.name_start]
    before = before[before.rfind(". ") + 1 :]
    after = ctx.text[ctx.end : ctx.end + 120]

    tests = []
    m = _COURT_BEFORE_RE.search(before) or _COURT_AFTER_RE.match(after)
    if m:
        test = _court_test(m.group(1), ctx.citing_court)
        if test:
            tests.append((test, f"text names the court: {m.group(1)!r}"))
    if _TRIAL_AFTER_RE.match(after):
        tests.append((lambda c: c not in APPELLATE_COURTS, "text says it went on appeal"))
    for test, note in tests:
        fits = [j for j in tied if test(j.court)]
        if len(fits) == 1:
            return Link("resolved", fits[0].id, link.candidates, note)

    for d in _DATE_RE.finditer(before + " " + after.split(". ")[0]):
        date = f"{int(d.group(3)):04d}-{_MONTHS.index(d.group(2)) + 1:02d}-{int(d.group(1)):02d}"
        fits = [j for j in tied if j.decision_date == date]
        if len(fits) == 1:
            return Link("resolved", fits[0].id, link.candidates, f"text gives the date {date}")
    return link


# The paragraph a citation points to: "at [55]", "(“Liton”) at [38]",
# "at [27]–[31]".
_PINPOINT_RE = re.compile(
    r"^\s*(?:\((?:“[^”]{1,40}”|\"[^\"]{1,40}\")\)\s*)?,?\s*\(?(?:see\s+)?at\s+"
    r"\[(\d+)\](?:\s*(?:–|-|to)\s*\[(\d+)\])?"
)
_STOP = set(
    "the a an of to in and or that is was be by for on as at it this with which not are from "
    "has have had its their there were been any such may if would should could under see also "
    "court".split()
)
# The pinpointed paragraph of the right candidate shares at least this share
# of the citing passage's words, and this much more than any other candidate.
PINPOINT_MIN = 0.2
PINPOINT_MARGIN = 0.15


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def break_tie_by_pinpoint(
    catalog: Catalog,
    link: Link,
    ctx: Context,
    paragraphs_of: Callable[[str], Optional[dict]],
) -> Link:
    """Settle a tie from the paragraph the citation points to ("at [55]").

    A candidate whose numbered paragraphs stop before the pinpoint cannot
    be the one cited. Among those that have it, one wins only when its
    paragraph shares clearly more words with the citing passage than any
    other's. A candidate with no numbered text (often 2000–2002 pages)
    leaves the tie standing: there is no evidence either way.

    ``paragraphs_of(judgment_id)`` returns ``{paragraph number: text}``, or
    None when the judgment's text is not available.
    """
    if link.tier != "tied":
        return link
    pin = _PINPOINT_RE.match(ctx.text[ctx.end : ctx.end + 80])
    if not pin:
        return link
    lo = int(pin.group(1))
    hi = max(lo, min(int(pin.group(2) or lo), lo + 10))
    paras = {j.id: paragraphs_of(j.id) for j in _tied(catalog, link)}
    left = [jid for jid, p in paras.items() if not p or max(p) >= lo]
    if not left:
        return link
    if len(left) == 1:
        return Link("resolved", left[0], link.candidates, f"only one reaches [{lo}]")
    if not all(paras[jid] and lo in paras[jid] for jid in left):
        return link
    claim = _words(
        ctx.text[max(0, ctx.name_start - 300) : ctx.name_start]
        + " "
        + ctx.text[ctx.end : ctx.end + 200]
    ) - _words(ctx.case_name)
    if not claim:
        return link
    sims = sorted(
        (
            (
                len(claim & _words(" ".join(paras[jid].get(n, "") for n in range(lo, hi + 1))))
                / len(claim),
                jid,
            )
            for jid in left
        ),
        reverse=True,
    )
    (best, jid), (second, _) = sims[0], sims[1]
    if best >= PINPOINT_MIN and best - second >= PINPOINT_MARGIN:
        return Link(
            "resolved", jid, link.candidates, f"[{lo}] matches the citing text ({best:.2f})"
        )
    return link


# ---------------------------------------------------------------------------
# Pairing names with citations in a detected string
# ---------------------------------------------------------------------------

# Between a name and its citation: spaces, a comma or semicolon, a closing
# quote or bracket left outside the name span (The “Bunga Melati 5”,
# "(formerly known as E Merck)"), or a short bracketed note.
_BETWEEN_RE = re.compile(r"[\s,;”\"’)]*(?:\([^)]{0,40}\)\s*)?")


def cited_names(mentions: Sequence[Mention], text: str) -> dict:
    """``{citation index: case name}`` for each CITATION written after a CASE_NAME.

    A name carries over a chain of citations ("Name [2012] SGCA 5;
    [2012] 4 SLR 308"), but not past other words.
    """
    return {i: m.text for i, m in cited_name_mentions(mentions, text).items()}


def cited_name_mentions(mentions: Sequence[Mention], text: str) -> dict:
    """``{citation index: CASE_NAME mention}``, as ``cited_names`` pairs them."""
    out = {}
    ordered = sorted(range(len(mentions)), key=lambda i: mentions[i].start)
    name, last_end = None, None
    for i in ordered:
        m = mentions[i]
        if m.label == "CASE_NAME":
            name, last_end = m, m.end
        elif m.label == "CITATION" and name is not None:
            gap = text[last_end : m.start]
            if _BETWEEN_RE.fullmatch(gap):
                out[i] = name
                # The pinpoint is part of the citation's trailing text.
                last_end = m.end + _pinpoint_len(text, m.end)
            else:
                name = None
        else:
            name = None
    return out


def _pinpoint_len(text: str, pos: int) -> int:
    m = re.match(r"\s*at\s*\[[^\]]{1,12}\](?:\s*(?:–|-|to|and|,)\s*\[[^\]]{1,12}\])*", text[pos:])
    return m.end() if m else 0


def link_mention(
    catalog: Catalog,
    mention: Mention,
    case_name: Optional[str] = None,
    citing_id: Optional[str] = None,
    citing_date: Optional[str] = None,
) -> Optional[Link]:
    """Link one CITATION mention, or None for a kind we never link (foreign reports)."""
    if mention.label != "CITATION":
        return None
    if mention.kind == "sg_neutral":
        return link_neutral(catalog, mention.normalised or mention.text)
    if mention.kind == "sg_report":
        year = int(re.search(r"\d{4}", mention.text).group())
        return link_report(catalog, case_name, year, citing_id, citing_date)
    return None
