"""Pure-function HTML extraction for eLitigation judgment detail pages.

No IO, no HTTP — input is an HTML string, output is an
``ExtractedJudgment`` with the main-table fields plus a list of
per-paragraph fragment dicts that map 1:1 to rows in
``judgments_fragments``.

See CLAUDE.md "Phase 2 design notes" for the site structure (container
IDs, ``Judg-*`` class map, footnote handling, table/figure attachment
rules) that this module encodes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class-name detection is prefix-based, not a static whitelist, because
# eLitigation emits variants across eras that aren't worth enumerating:
# Judg-1-firstpara, Judg-List-1 (bare), Judg-Quote-List, Judg-Hearing-Date,
# etc. Any Judg-* class becomes a fragment unless excluded below.
JUDG_PREFIX = "Judg-"
HEADING_PREFIX = "Judg-Heading-"

# Judg-EOF is the "end of document" ornament — skip entirely.
EXCLUDED_CLASSES = {"Judg-EOF"}

# Classes whose leading token is an arabic paragraph number. Judg-2 uses
# alpha letters "(a)" and Judg-3 uses roman numerals "(i)" — still
# fragments, but no parsed paragraph_number.
NUMBERED_CLASSES = {"Judg-1", "Judg-1-firstpara"}

# Classes that anchor the backward-attachment rule for standalone
# tables/figures. Matches CLAUDE.md ("backward-attach to most recent
# Judg-1/Judg-2") plus the Judg-3 sibling tier that exists in practice.
ANCHOR_CLASSES = {"Judg-1", "Judg-1-firstpara", "Judg-2", "Judg-3"}

# Separator between paragraph number and text: non-breaking space on older
# docs, em-space on newer ones, plus any regular whitespace.
PARA_NUM_RE = re.compile(r"^(\d+)[\s\xa0\u2003]+(.*)$", re.DOTALL)

TABLE_OPEN = "\n\n---table---\n"
TABLE_CLOSE = "\n---end-table---\n"

# eLitigation's mobile/web conversion preserves U+00A0 (non-breaking
# space) and U+2003 (em-space) liberally — between paragraph numbers and
# text, inside case names ("Smith\xa0v\xa0Jones"), inside dates, etc.
# BS4's ``get_text`` does not normalize these, so they leak into
# ``content_text`` and produce awkward copy-paste output plus subtly
# different FTS tokenization.
_WS_TRANSLATION = str.maketrans({"\xa0": " ", "\u2003": " "})


def _normalize_ws(text: str) -> str:
    return text.translate(_WS_TRANSLATION) if text else text


# Tags that render as their own block. Only these boundaries become a
# space when flattening to text: inline markup must add nothing, because
# eLitigation splits words across spans mid-word ("Lee H</span><span>sien
# Loong") and wraps italic short names inside quotes ("(“<span>KS v
# LHY</span>”)"). BS4's ``get_text(" ")`` put a space at every tag edge,
# which produced "Lee H sien Loong" and "(“ KS v LHY ”)".
BLOCK_TAGS = [
    "div", "p", "li", "ul", "ol", "tr", "td", "th", "table", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6",
]  # fmt: skip

_WS_RUN_RE = re.compile(r"\s+")


def element_text(element: Tag) -> str:
    """Flatten ``element`` to one line of text the way a browser shows it.

    Block boundaries and ``<br>`` become a single space; inline tags join
    their text with nothing added. All whitespace runs (including U+00A0
    and U+2003) collapse to one ASCII space. Works on a copy, so the
    caller's tree is untouched.
    """
    root = copy.copy(element)
    for br in root.find_all("br"):
        br.replace_with(" ")
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before(" ")
        tag.insert_after(" ")
    return _WS_RUN_RE.sub(" ", _normalize_ws(root.get_text())).strip()


def _is_footnote_id(value: Optional[str]) -> bool:
    return bool(value) and value.startswith("fn")


def strip_footnote_markup(root: Tag) -> None:
    """Remove footnote modals and their ``<sup>`` triggers from ``root``.

    Newer judgments nest each footnote's modal ``div#fn-…`` inside the
    paragraph that cites it, so the paragraph text used to carry "Foot
    Note 2 × Affidavit of …" plus the marker digit. The footnote's text
    already lands in ``footnote_text``. Bare ``<sup>`` tags with no
    resolvable link (``<sup>[note: 1]</sup>``) are kept, as before.
    """
    for modal in root.find_all("div", id=_is_footnote_id):
        modal.decompose()
    for sup in root.find_all("sup"):
        for trigger in sup.find_all(["a", "button"]):
            target = (trigger.get("data-target") or trigger.get("href") or "").strip()
            if target.startswith("#fn"):
                sup.decompose()
                break


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when a judgment page lacks the required container."""


@dataclass
class ExtractedJudgment:
    judgment_id: str
    content_text: str
    court_summary: str
    has_content: bool
    has_court_summary: bool
    fragments: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Primitives (unit-testable in isolation)
# ---------------------------------------------------------------------------


def parse_paragraph_number(text: str) -> Tuple[Optional[int], str]:
    """Split a leading paragraph number from the rest of the text.

    Handles the separator quirk documented in CLAUDE.md: newer judgments
    use U+2003 (em-space), older ones use U+00A0 (non-breaking space).
    """
    m = PARA_NUM_RE.match(text)
    if m is None:
        return None, text
    return int(m.group(1)), m.group(2)


def absorb_table(table: Tag) -> str:
    """Render a ``<table>`` as pipe-separated text, one row per line.

    Returns an empty string for tables with no meaningful rows. Wraps the
    rendered rows in ``---table---``/``---end-table---`` markers so the
    block is visible in ``content_text`` for display and still
    searchable via FTS.
    """
    lines: List[str] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        line = " | ".join(element_text(cell) for cell in cells)
        if line.replace("|", "").strip():
            lines.append(line)
    if not lines:
        return ""
    return TABLE_OPEN + "\n".join(lines) + TABLE_CLOSE


def absorb_figure(img: Tag) -> Tuple[str, str]:
    """Derive a ``(src, alt)`` pair for an ``<img>``.

    ``src`` is the raw ``src`` attribute for URL-based images, or
    ``sha256:<hex>`` for ``data:`` URIs (the base64 payload is hashed to
    avoid dragging potentially-megabytes-per-image blobs into the DB).
    ``alt`` falls back to a stable placeholder built from dimensions so
    FTS still has something to match on even when the markup omits alt
    text.
    """
    src = (img.get("src") or "").strip()
    alt = _normalize_ws((img.get("alt") or "").strip())
    if src.startswith("data:"):
        digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
        src = f"sha256:{digest}"
    if not alt:
        w = (img.get("width") or "").strip()
        h = (img.get("height") or "").strip()
        if w and h:
            alt = f"[Figure, {w}x{h}]"
        else:
            alt = "[Figure]"
    return src, alt


def extract_footnotes(soup: BeautifulSoup) -> Dict[str, str]:
    """Collect footnote contents from ``div[id^='fn']``.

    Keys are the raw element ids (e.g. ``fn1`` or
    ``fn-041fe0fc-fb3c-…``); values are the stripped text contents.
    Excludes the main judgment container, which also happens to have an
    id starting with ``"f"`` in some themes. Bootstrap modals keep only
    their ``.modal-body``, dropping the "Foot Note n" title and the ×
    close button.
    """
    result: Dict[str, str] = {}
    for el in soup.find_all("div", id=True):
        fid = el.get("id", "")
        if not fid.startswith("fn"):
            continue
        # Guard: the top-level judgment container is "divJudgement"; the
        # fn* prefix is specific enough that false positives are unlikely,
        # but be explicit.
        if fid in {"divJudgement", "divCaseSummary", "divHeadMessage"}:
            continue
        body = el.find("div", class_="modal-body") or el
        text = element_text(body)
        if text:
            result[fid] = text
    return result


def collect_footnote_refs(element: Tag) -> List[str]:
    """Return the ordered list of footnote ids referenced inside ``element``.

    Handles two markup variants seen on eLitigation:
      - ``<sup><button data-target="#fn…">n</button></sup>`` — Bootstrap
        modal trigger used on newer judgments.
      - ``<sup><a href="#fn…">n</a></sup>`` — plain anchor used on
        older judgments.
    ``<sup>`` tags without either link kind are skipped because we
    can't reliably map a bare superscript number to a footnote id.
    """
    ids: List[str] = []
    for sup in element.find_all("sup"):
        for descendant in sup.find_all(["a", "button"]):
            target = (descendant.get("data-target") or descendant.get("href") or "").strip()
            if target.startswith("#"):
                fid = target[1:]
                if fid.startswith("fn") and fid not in ids:
                    ids.append(fid)
    return ids


# ---------------------------------------------------------------------------
# Element content extraction
# ---------------------------------------------------------------------------


@dataclass
class _ElementContent:
    text: str
    has_table: bool
    has_figure: bool
    figure_srcs: List[str] = field(default_factory=list)
    figure_alts: List[str] = field(default_factory=list)


def _extract_element_content(element: Tag) -> _ElementContent:
    """Extract text + absorbed tables/figures from a fragment element.

    Works on a re-parsed copy so destructive decomposition doesn't mutate
    the caller's tree (we still need the original for ``html_raw``).
    """
    # Round-trip through BS4 to get a deep copy. The lxml parser will wrap
    # in <html><body>; find the first tag of the same name as the original
    # (at any depth) to recover the root.
    copy_soup = BeautifulSoup(str(element), "lxml")
    root = copy_soup.find(element.name)
    if root is None:
        # Defensive fallback — shouldn't happen for normal Judg-* divs.
        return _ElementContent(
            text=element_text(element),
            has_table=False,
            has_figure=False,
        )

    strip_footnote_markup(root)

    tables = list(root.find_all("table"))
    imgs = list(root.find_all("img"))

    # Pull renderings out BEFORE decomposing so the tags are still attached.
    table_renderings = [absorb_table(t) for t in tables]
    figures = [absorb_figure(i) for i in imgs]

    for t in tables:
        t.decompose()
    for i in imgs:
        i.decompose()

    text = element_text(root)

    for rendering in table_renderings:
        if rendering:
            text = text + rendering

    fig_srcs: List[str] = []
    fig_alts: List[str] = []
    for src, alt in figures:
        fig_srcs.append(src)
        fig_alts.append(alt)
        text = text + "\n\n" + alt

    return _ElementContent(
        text=text,
        has_table=bool(tables) and any(r for r in table_renderings),
        has_figure=bool(imgs),
        figure_srcs=fig_srcs,
        figure_alts=fig_alts,
    )


# ---------------------------------------------------------------------------
# Fragment assembly
# ---------------------------------------------------------------------------


def _make_fragment(
    *,
    judgment_id: str,
    ordinal: int,
    html_raw: str,
    class_name: str,
    paragraph_number: Optional[int],
    section_heading: Optional[str],
    text: str,
    has_table: bool,
    has_figure: bool,
    figure_srcs: List[str],
    figure_alts: List[str],
    footnote_texts: List[str],
) -> Dict[str, Any]:
    has_footnotes = bool(footnote_texts)
    return {
        "id": f"{judgment_id}_{ordinal:04d}",
        "judgment_id": judgment_id,
        "ordinal": ordinal,
        "paragraph_number": paragraph_number,
        "class_name": class_name,
        "section_heading": section_heading,
        "content_text": text,
        "html_raw": html_raw,
        "footnote_text": (
            json.dumps(footnote_texts, ensure_ascii=False) if has_footnotes else None
        ),
        "has_footnotes": has_footnotes,
        "has_table": has_table,
        "has_figure": has_figure,
        "figure_src": (json.dumps(figure_srcs, ensure_ascii=False) if figure_srcs else None),
        "figure_descriptions": (
            json.dumps(figure_alts, ensure_ascii=False) if figure_alts else None
        ),
    }


def classify_element(tag: Tag) -> Tuple[Optional[str], Optional[str]]:
    """Categorise a Tag by its Judg-* class.

    Returns ``(kind, class_name)`` where ``kind`` is one of
    ``"heading"``, ``"paragraph"``, ``None`` (not a Judg-* fragment).
    ``class_name`` is the specific Judg-* class that triggered the
    match, preserved for the fragment row.
    """
    classes = tag.get("class") or []
    # Heading check first — headings match the JUDG_PREFIX too.
    for cls in classes:
        if cls.startswith(HEADING_PREFIX):
            return "heading", cls
    for cls in classes:
        if cls in EXCLUDED_CLASSES:
            continue
        if cls.startswith(JUDG_PREFIX):
            return "paragraph", cls
    return None, None


def walk_judgment_body(container: Tag):
    """Yield interesting elements in document order, Judg-* as leaves.

    The detail pages wrap content in arbitrary levels of non-Judg-*
    divs (e.g. a custom ``<content>`` element with inner wrappers).
    This walker descends through those wrappers and yields:
      - Judg-* elements (paragraphs and headings) without recursing
        further — their inline tables/images are absorbed downstream.
      - Standalone ``<table>``/``<img>`` nodes that are NOT nested
        inside a Judg-* element (orphan exhibits between paragraphs).
    """
    stack: List[Any] = list(reversed(list(container.children)))
    while stack:
        node = stack.pop()
        if isinstance(node, NavigableString):
            continue
        if not isinstance(node, Tag):
            continue
        kind, _ = classify_element(node)
        if kind is not None:
            yield node
            continue  # leaf — do not recurse into its children
        if node.name in ("table", "img"):
            yield node
            continue
        # Non-Judg wrapper — descend
        for child in reversed(list(node.children)):
            stack.append(child)


def extract_paragraphs(
    div_judgement: Tag,
    footnote_map: Dict[str, str],
    judgment_id: str,
) -> List[Dict[str, Any]]:
    """Walk ``#divJudgement`` and emit fragments in document order.

    Handles the CLAUDE.md attachment rules:
      - ``<table>``/``<img>`` nested inside a Judg-* element → absorbed
        into that element's fragment (via ``_extract_element_content``).
      - ``<table>``/``<img>`` as a sibling of Judg-* elements (not
        nested inside one) → backward-attach to the most recent Judg-1/
        Judg-2/Judg-3 fragment, or forward-attach to the next one if no
        anchor precedes.
      - Never creates a standalone fragment for a table or image.
    """
    fragments: List[Dict[str, Any]] = []
    current_heading: Optional[str] = None
    # Buffers for elements that arrive before any anchor paragraph.
    pending_table_renderings: List[str] = []
    pending_figures: List[Tuple[str, str]] = []  # (src, alt)
    last_anchor_idx: Optional[int] = None

    def attach_orphan_table(rendering: str) -> None:
        if not rendering:
            return
        if last_anchor_idx is not None:
            f = fragments[last_anchor_idx]
            f["content_text"] = f["content_text"] + rendering
            f["has_table"] = True
        else:
            pending_table_renderings.append(rendering)

    def attach_orphan_figure(src: str, alt: str) -> None:
        if last_anchor_idx is not None:
            f = fragments[last_anchor_idx]
            f["content_text"] = f["content_text"] + "\n\n" + alt
            f["has_figure"] = True
            existing_srcs = json.loads(f["figure_src"]) if f["figure_src"] else []
            existing_alts = json.loads(f["figure_descriptions"]) if f["figure_descriptions"] else []
            existing_srcs.append(src)
            existing_alts.append(alt)
            f["figure_src"] = json.dumps(existing_srcs, ensure_ascii=False)
            f["figure_descriptions"] = json.dumps(existing_alts, ensure_ascii=False)
        else:
            pending_figures.append((src, alt))

    def drain_forward_attachments(fragment: Dict[str, Any]) -> None:
        if not pending_table_renderings and not pending_figures:
            return
        prefix_text = ""
        for rendering in pending_table_renderings:
            prefix_text += rendering
        fig_srcs: List[str] = list(
            json.loads(fragment["figure_src"]) if fragment["figure_src"] else []
        )
        fig_alts: List[str] = list(
            json.loads(fragment["figure_descriptions"]) if fragment["figure_descriptions"] else []
        )
        for src, alt in pending_figures:
            prefix_text += "\n\n" + alt
            fig_srcs.append(src)
            fig_alts.append(alt)
        if pending_table_renderings:
            fragment["has_table"] = True
            pending_table_renderings.clear()
        if pending_figures:
            fragment["has_figure"] = True
            fragment["figure_src"] = json.dumps(fig_srcs, ensure_ascii=False)
            fragment["figure_descriptions"] = json.dumps(fig_alts, ensure_ascii=False)
            pending_figures.clear()
        fragment["content_text"] = fragment["content_text"] + prefix_text

    for node in walk_judgment_body(div_judgement):
        kind, class_name = classify_element(node)
        html_raw = str(node)

        if kind == "heading":
            heading_content = _extract_element_content(node)
            heading_text = heading_content.text
            if heading_text:
                current_heading = heading_text
            fragments.append(
                _make_fragment(
                    judgment_id=judgment_id,
                    ordinal=len(fragments),
                    html_raw=html_raw,
                    class_name=class_name or "",
                    paragraph_number=None,
                    section_heading=None,
                    text=heading_text,
                    has_table=heading_content.has_table,
                    has_figure=heading_content.has_figure,
                    figure_srcs=heading_content.figure_srcs,
                    figure_alts=heading_content.figure_alts,
                    footnote_texts=[
                        footnote_map[fid]
                        for fid in collect_footnote_refs(node)
                        if fid in footnote_map
                    ],
                )
            )
            continue

        if kind == "paragraph":
            content = _extract_element_content(node)
            text = content.text
            paragraph_number: Optional[int] = None
            if class_name in NUMBERED_CLASSES:
                num, rest = parse_paragraph_number(text)
                if num is not None:
                    paragraph_number = num
                    text = rest
            footnote_texts = [
                footnote_map[fid] for fid in collect_footnote_refs(node) if fid in footnote_map
            ]
            fragment = _make_fragment(
                judgment_id=judgment_id,
                ordinal=len(fragments),
                html_raw=html_raw,
                class_name=class_name or "",
                paragraph_number=paragraph_number,
                section_heading=current_heading,
                text=text,
                has_table=content.has_table,
                has_figure=content.has_figure,
                figure_srcs=content.figure_srcs,
                figure_alts=content.figure_alts,
                footnote_texts=footnote_texts,
            )
            if class_name in ANCHOR_CLASSES:
                drain_forward_attachments(fragment)
            fragments.append(fragment)
            if class_name in ANCHOR_CLASSES:
                last_anchor_idx = len(fragments) - 1
            continue

        # Orphan <table> / <img>
        if node.name == "table":
            attach_orphan_table(absorb_table(node))
            continue
        if node.name == "img":
            src, alt = absorb_figure(node)
            attach_orphan_figure(src, alt)
            continue

    return fragments


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _compose_content_text(fragments: List[Dict[str, Any]]) -> str:
    """Concatenate fragments into a single plain-text judgment body.

    Numbered paragraphs are prefixed with their number (so
    ``content_text`` reads natively); headings are separated by blank
    lines. The fragments table is the authoritative per-paragraph
    representation — this blob is for whole-judgment display and FTS.
    """
    out: List[str] = []
    for f in fragments:
        if not f["content_text"]:
            continue
        if f["paragraph_number"] is not None:
            out.append(f"{f['paragraph_number']} {f['content_text']}")
        else:
            out.append(f["content_text"])
    return "\n\n".join(out)


def extract_court_summary(soup: BeautifulSoup) -> str:
    div = soup.find("div", id="divCaseSummary")
    if div is None:
        return ""
    return element_text(div)


def extract_judgment(html: str, judgment_id: str) -> ExtractedJudgment:
    """Top-level: HTML → ExtractedJudgment (main fields + fragments)."""
    soup = BeautifulSoup(html, "lxml")
    div_j = soup.find("div", id="divJudgement")
    if div_j is None:
        raise ExtractionError("div#divJudgement not found on page")

    footnote_map = extract_footnotes(soup)
    fragments = extract_paragraphs(div_j, footnote_map, judgment_id)
    content_text = _compose_content_text(fragments)
    court_summary = extract_court_summary(soup)

    has_content = bool(content_text.strip())
    has_court_summary = bool(court_summary.strip())

    return ExtractedJudgment(
        judgment_id=judgment_id,
        content_text=content_text,
        court_summary=court_summary,
        has_content=has_content,
        has_court_summary=has_court_summary,
        fragments=fragments,
    )
