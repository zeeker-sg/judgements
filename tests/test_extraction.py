"""Unit + fixture tests for resources.extraction.

Unit tests exercise each helper with hand-crafted HTML so edge cases
(separator variants, empty containers, decompose semantics) are pinned
independently of real fixtures. Fixture tests then confirm end-to-end
extraction on real eLitigation pages saved under ``tests/fixtures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from resources.extraction import (
    ExtractionError,
    absorb_figure,
    absorb_table,
    collect_footnote_refs,
    extract_footnotes,
    extract_judgment,
    parse_paragraph_number,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# parse_paragraph_number
# ---------------------------------------------------------------------------


class TestParseParagraphNumber:
    def test_non_breaking_space_separator(self):
        # Older judgments use U+00A0 between the number and text.
        num, rest = parse_paragraph_number("1\xa0The appellants appeal against…")
        assert num == 1
        assert rest == "The appellants appeal against…"

    def test_em_space_separator(self):
        # Newer judgments use U+2003.
        num, rest = parse_paragraph_number("42\u2003In these reasons…")
        assert num == 42
        assert rest == "In these reasons…"

    def test_regular_space_separator(self):
        num, rest = parse_paragraph_number("7 Plain ASCII space works too.")
        assert num == 7
        assert rest == "Plain ASCII space works too."

    def test_multi_digit(self):
        num, rest = parse_paragraph_number("128\xa0A deeply buried paragraph.")
        assert num == 128

    def test_no_leading_number(self):
        # Headings, front-matter etc. have no leading number.
        num, rest = parse_paragraph_number("Background")
        assert num is None
        assert rest == "Background"

    def test_alpha_enumeration_is_not_a_number(self):
        # Judg-2 uses "(a)" — must not be parsed as a number.
        num, rest = parse_paragraph_number("(a)\xa0the first element…")
        assert num is None


# ---------------------------------------------------------------------------
# absorb_table
# ---------------------------------------------------------------------------


class TestAbsorbTable:
    def _table(self, html: str):
        return BeautifulSoup(html, "lxml").find("table")

    def test_simple_table_renders_pipe_separated(self):
        t = self._table("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>")
        out = absorb_table(t)
        assert "---table---" in out
        assert "---end-table---" in out
        assert "A | B" in out
        assert "1 | 2" in out

    def test_empty_table_returns_empty_string(self):
        t = self._table("<table></table>")
        assert absorb_table(t) == ""

    def test_table_with_only_blank_cells_returns_empty(self):
        t = self._table("<table><tr><td>  </td><td></td></tr></table>")
        assert absorb_table(t) == ""

    def test_headers_and_body_both_captured(self):
        t = self._table("<table><tr><th>Col</th></tr><tr><td>Val</td></tr></table>")
        out = absorb_table(t)
        assert "Col" in out
        assert "Val" in out


# ---------------------------------------------------------------------------
# absorb_figure
# ---------------------------------------------------------------------------


class TestAbsorbFigure:
    def _img(self, html: str):
        return BeautifulSoup(html, "lxml").find("img")

    def test_url_src_kept_verbatim(self):
        img = self._img('<img src="https://example.com/x.png" alt="diagram A">')
        src, alt = absorb_figure(img)
        assert src == "https://example.com/x.png"
        assert alt == "diagram A"

    def test_data_uri_hashed_to_sha256(self):
        img = self._img('<img src="data:image/png;base64,AAAA" alt="">')
        src, alt = absorb_figure(img)
        assert src.startswith("sha256:")
        assert len(src) == len("sha256:") + 64

    def test_missing_alt_gets_placeholder_with_dims(self):
        img = self._img('<img src="x.png" width="1024" height="768">')
        _, alt = absorb_figure(img)
        assert alt == "[Figure, 1024x768]"

    def test_missing_alt_and_dims_gets_plain_placeholder(self):
        img = self._img('<img src="x.png">')
        _, alt = absorb_figure(img)
        assert alt == "[Figure]"


# ---------------------------------------------------------------------------
# collect_footnote_refs
# ---------------------------------------------------------------------------


class TestFootnoteRefs:
    def test_bootstrap_modal_button_variant(self):
        # Newer judgments wrap the marker in a Bootstrap modal trigger.
        p = BeautifulSoup(
            '<div class="Judg-1">Text'
            '<sup><button data-target="#fn-abc" data-toggle="modal">1</button></sup>'
            "</div>",
            "lxml",
        ).find("div")
        assert collect_footnote_refs(p) == ["fn-abc"]

    def test_anchor_href_variant(self):
        p = BeautifulSoup(
            '<div class="Judg-1">Text<sup><a href="#fn1">1</a></sup></div>',
            "lxml",
        ).find("div")
        assert collect_footnote_refs(p) == ["fn1"]

    def test_dedupes_same_id(self):
        p = BeautifulSoup(
            '<div class="Judg-1">A<sup><a href="#fn1">1</a></sup>'
            'and B<sup><a href="#fn1">1</a></sup>.</div>',
            "lxml",
        ).find("div")
        assert collect_footnote_refs(p) == ["fn1"]

    def test_bare_sup_is_ignored(self):
        # A plain superscript number has no resolvable target.
        p = BeautifulSoup('<div class="Judg-1">Text<sup>[note: 1]</sup></div>', "lxml").find("div")
        assert collect_footnote_refs(p) == []


# ---------------------------------------------------------------------------
# extract_footnotes
# ---------------------------------------------------------------------------


class TestExtractFootnotes:
    def test_fn_prefix_divs_are_collected(self):
        soup = BeautifulSoup(
            '<div id="divJudgement">body</div>'
            '<div id="fn1">Footnote one</div>'
            '<div id="fn-uuid">Footnote two</div>',
            "lxml",
        )
        fns = extract_footnotes(soup)
        assert fns == {"fn1": "Footnote one", "fn-uuid": "Footnote two"}

    def test_non_fn_divs_ignored(self):
        soup = BeautifulSoup(
            '<div id="divJudgement">x</div>' '<div id="sidebar">noise</div>',
            "lxml",
        )
        assert extract_footnotes(soup) == {}


# ---------------------------------------------------------------------------
# extract_judgment — error path
# ---------------------------------------------------------------------------


def test_missing_div_judgement_raises():
    with pytest.raises(ExtractionError):
        extract_judgment("<html><body>no container</body></html>", "abc123")


# ---------------------------------------------------------------------------
# Synthetic attachment rule tests
# ---------------------------------------------------------------------------


def _extract_from_divj(inner_html: str, judgment_id: str = "t1"):
    html = f'<html><body><div id="divJudgement">{inner_html}</div></body></html>'
    return extract_judgment(html, judgment_id)


class TestAttachmentRule:
    def test_inline_table_absorbed_into_paragraph(self):
        ej = _extract_from_divj(
            '<div class="Judg-1">1\xa0See figures: '
            "<table><tr><td>A</td><td>B</td></tr></table></div>"
        )
        assert len(ej.fragments) == 1
        f = ej.fragments[0]
        assert f["has_table"] is True
        assert "A | B" in f["content_text"]

    def test_orphan_table_backward_attaches_to_previous_paragraph(self):
        ej = _extract_from_divj(
            '<div class="Judg-1">1\xa0First paragraph.</div>'
            "<table><tr><td>X</td><td>Y</td></tr></table>"
            '<div class="Judg-1">2\xa0Second paragraph.</div>'
        )
        # Two numbered paragraph fragments — no standalone table fragment.
        assert len(ej.fragments) == 2
        assert ej.fragments[0]["has_table"] is True
        assert "X | Y" in ej.fragments[0]["content_text"]
        assert ej.fragments[1]["has_table"] is False

    def test_orphan_table_before_any_paragraph_forward_attaches(self):
        ej = _extract_from_divj(
            "<table><tr><td>A</td></tr></table>"
            '<div class="Judg-1">1\xa0The only paragraph.</div>'
        )
        assert len(ej.fragments) == 1
        f = ej.fragments[0]
        assert f["has_table"] is True
        assert "A" in f["content_text"]

    def test_orphan_figure_backward_attaches(self):
        ej = _extract_from_divj(
            '<div class="Judg-1">1\xa0Para.</div>'
            '<img src="http://x/y.png" alt="diagram X">'
            '<div class="Judg-1">2\xa0Next.</div>'
        )
        assert len(ej.fragments) == 2
        f = ej.fragments[0]
        assert f["has_figure"] is True
        srcs = json.loads(f["figure_src"])
        alts = json.loads(f["figure_descriptions"])
        assert srcs == ["http://x/y.png"]
        assert alts == ["diagram X"]


class TestHeadingContext:
    def test_section_heading_propagates_to_following_paragraphs(self):
        ej = _extract_from_divj(
            '<div class="Judg-Heading-1">Background</div>'
            '<div class="Judg-1">1\xa0Para under background.</div>'
            '<div class="Judg-Heading-1">Analysis</div>'
            '<div class="Judg-1">2\xa0Para under analysis.</div>'
        )
        headings = [f for f in ej.fragments if f["class_name"].startswith("Judg-Heading-")]
        paras = [f for f in ej.fragments if f["class_name"] in ("Judg-1",)]
        assert len(headings) == 2
        assert len(paras) == 2
        assert paras[0]["section_heading"] == "Background"
        assert paras[1]["section_heading"] == "Analysis"


class TestCaseSummary:
    def test_empty_div_case_summary_marks_has_court_summary_false(self):
        html = (
            "<html><body>"
            '<div id="divJudgement">'
            '<div class="Judg-1">1\xa0Body.</div>'
            "</div>"
            '<div id="divCaseSummary"></div>'
            "</body></html>"
        )
        ej = extract_judgment(html, "x")
        assert ej.has_court_summary is False
        assert ej.court_summary == ""

    def test_populated_div_case_summary_captured(self):
        html = (
            "<html><body>"
            '<div id="divJudgement"><div class="Judg-1">1\xa0Body.</div></div>'
            '<div id="divCaseSummary">Important headnote text.</div>'
            "</body></html>"
        )
        ej = extract_judgment(html, "x")
        assert ej.has_court_summary is True
        assert "Important headnote text." in ej.court_summary


# ---------------------------------------------------------------------------
# Fixture integration tests
# ---------------------------------------------------------------------------

FIXTURE_FILES = sorted(FIXTURES.glob("*.html"))


@pytest.mark.parametrize("fixture_path", FIXTURE_FILES, ids=lambda p: p.name)
def test_fixture_extracts_cleanly(fixture_path: Path):
    """Every fixture must produce a non-empty extraction with numbered paras."""
    html = fixture_path.read_text(encoding="utf-8")
    ej = extract_judgment(html, fixture_path.stem)
    assert ej.has_content is True, f"{fixture_path.name} produced no content"
    assert ej.content_text.strip(), f"{fixture_path.name} content_text is blank"
    assert any(
        f["paragraph_number"] == 1 for f in ej.fragments
    ), f"{fixture_path.name} has no '1' paragraph"
    # Ordinals must be strictly ascending and unique.
    ords = [f["ordinal"] for f in ej.fragments]
    assert ords == list(range(len(ej.fragments)))
    # IDs must follow the deterministic scheme.
    for f in ej.fragments:
        assert f["id"] == f"{fixture_path.stem}_{f['ordinal']:04d}"


def test_newer_fixture_has_footnotes():
    """The 2026 SGHC 85 fixture has 21 footnotes per the source page."""
    path = FIXTURES / "recent_2026_sghc_85.html"
    ej = extract_judgment(path.read_text(encoding="utf-8"), path.stem)
    total_refs = sum(
        len(json.loads(f["footnote_text"])) for f in ej.fragments if f["footnote_text"]
    )
    assert total_refs >= 10, f"expected many footnote refs, got {total_refs}"


def test_sgdc_fixture_has_figures():
    """The 2026 SGDC 136 fixture is large; we observed 2 figures on it."""
    path = FIXTURES / "recent_2026_sgdc_136.html"
    ej = extract_judgment(path.read_text(encoding="utf-8"), path.stem)
    with_fig = [f for f in ej.fragments if f["has_figure"]]
    assert len(with_fig) >= 1


def test_sgca_fixture_has_court_summary():
    """The recent SGCA fixture surfaced a non-empty court-authored summary."""
    path = FIXTURES / "recent_2026_sgca_19.html"
    ej = extract_judgment(path.read_text(encoding="utf-8"), path.stem)
    assert ej.has_court_summary is True
    assert len(ej.court_summary) > 100


def test_fragment_html_raw_preserved():
    """html_raw should contain the original element markup verbatim."""
    path = FIXTURES / "recent_2026_sghc_85.html"
    ej = extract_judgment(path.read_text(encoding="utf-8"), path.stem)
    para1 = next(f for f in ej.fragments if f["paragraph_number"] == 1)
    assert para1["html_raw"].startswith("<div")
    assert "Judg-1" in para1["html_raw"]
    assert "1" in para1["html_raw"]


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


class TestWhitespaceNormalization:
    """Fragment content_text must not leak U+00A0 / U+2003 from the source."""

    def test_nbsp_stripped_from_content_text(self):
        ej = _extract_from_divj('<div class="Judg-1">1\xa0Smith\xa0v\xa0Jones\xa0(2025).</div>')
        assert "\xa0" not in ej.fragments[0]["content_text"]
        assert "Smith v Jones" in ej.fragments[0]["content_text"]

    def test_em_space_stripped(self):
        ej = _extract_from_divj(
            '<div class="Judg-Date-Reserved">16 April 2026\u2003Judgment reserved.</div>'
        )
        assert "\u2003" not in ej.fragments[0]["content_text"]

    def test_court_summary_normalized(self):
        html = (
            "<html><body>"
            '<div id="divJudgement"><div class="Judg-1">1\xa0Body.</div></div>'
            '<div id="divCaseSummary">Headnote\xa0with\xa0nbsp.</div>'
            "</body></html>"
        )
        ej = extract_judgment(html, "x")
        assert "\xa0" not in ej.court_summary
        assert ej.court_summary == "Headnote with nbsp."

    def test_footnote_text_normalized(self):
        html = (
            "<html><body>"
            '<div id="divJudgement">'
            '<div class="Judg-1">1\xa0Ref<sup><a href="#fn1">1</a></sup>.</div>'
            "</div>"
            '<div id="fn1">Foot Note\xa01\xa0\xd7\xa0Source document.</div>'
            "</body></html>"
        )
        ej = extract_judgment(html, "x")
        para = ej.fragments[0]
        fn_texts = json.loads(para["footnote_text"])
        assert all("\xa0" not in t for t in fn_texts)


# ---------------------------------------------------------------------------
# Inline markup and nested footnote modals
# ---------------------------------------------------------------------------


class TestInlineText:
    """Inline tags add no whitespace; block boundaries add exactly one space."""

    def test_word_split_across_spans_is_rejoined(self):
        # Seen in [2026] SGDC: "Lee H</span><span>sien Loong".
        ej = _extract_from_divj(
            '<div class="Judg-1">1 see <span class="font-italic ">Lee H</span>'
            '<span class="font-italic ">sien Loong v Leong Sze Hian </span>'
            "<span>[2021] 4 SLR 1128 at [45]. </span></div>"
        )
        assert ej.fragments[0]["content_text"] == (
            "see Lee Hsien Loong v Leong Sze Hian [2021] 4 SLR 1128 at [45]."
        )

    def test_italic_short_name_inside_quotes_not_padded(self):
        ej = _extract_from_divj(
            '<div class="Judg-1">1 <span>[2024] 5 SLR 194 (“</span>'
            '<span class="font-italic ">KS v LHY</span><span>”) at [21].</span></div>'
        )
        assert "(“KS v LHY”)" in ej.fragments[0]["content_text"]

    def test_block_children_and_br_become_single_spaces(self):
        ej = _extract_from_divj(
            '<div class="Judg-Lawyers"><p>Counsel A</p><p>Counsel B<br/>for the claimant</p></div>'
        )
        assert ej.fragments[0]["content_text"] == "Counsel A Counsel B for the claimant"

    def test_table_cells_rejoin_split_words(self):
        t = BeautifulSoup(
            "<table><tr><td><span>Dam</span><span>ages</span></td><td>$10,000</td></tr></table>",
            "lxml",
        ).find("table")
        assert "Damages | $10,000" in absorb_table(t)


class TestNestedFootnoteModal:
    """Newer judgments nest each footnote modal inside the citing paragraph."""

    HTML = (
        '<html><body><div id="divJudgement">'
        '<div class="Judg-1">5 Not much is known about the Defendant.'
        "<sup><button type='button' data-toggle='modal' data-target='#fn-abc'>2</button></sup>"
        "<div class='modal fade' id='fn-abc'><div class='modal-dialog'><div class='modal-content'>"
        "<div class='modal-header'><h5 class='modal-title'>Foot Note 2</h5>"
        "<button type='button' class='close'><span>&times;</span></button></div>"
        "<div class='modal-body'><div class='FootnoteRef'> Affidavit of Ng dated 26 December"
        " 2025 (“RN AEIC”), Exhibit A, at 39.</div></div>"
        "</div></div></div> </div>"
        "</div></body></html>"
    )

    def test_modal_and_marker_removed_from_paragraph_text(self):
        para = extract_judgment(self.HTML, "x").fragments[0]
        assert para["content_text"] == "Not much is known about the Defendant."

    def test_footnote_text_keeps_only_modal_body(self):
        para = extract_judgment(self.HTML, "x").fragments[0]
        assert json.loads(para["footnote_text"]) == [
            "Affidavit of Ng dated 26 December 2025 (“RN AEIC”), Exhibit A, at 39."
        ]

    def test_bare_sup_without_link_is_kept(self):
        ej = _extract_from_divj('<div class="Judg-1">1\xa0A claim<sup>[note: 1]</sup>.</div>')
        assert ej.fragments[0]["content_text"] == "A claim[note: 1]."


# ---------------------------------------------------------------------------
# extraction_cache corruption handling
# ---------------------------------------------------------------------------


class TestExtractionCacheCorruption:
    """Corrupt cache files must not raise — they get quarantined + treated as miss."""

    def _with_temp_cache(self, monkeypatch, tmp_path):
        """Point extraction_cache at a temp directory for test isolation."""
        import resources.extraction_cache as ec

        monkeypatch.setattr(ec, "CACHE_ROOT", tmp_path)
        monkeypatch.setattr(ec, "HTML_DIR", tmp_path / "judgments_html")
        monkeypatch.setattr(ec, "EXTRACTIONS_DIR", tmp_path / "judgments_extractions")
        return ec

    def test_corrupt_json_quarantined_and_returns_none(self, monkeypatch, tmp_path):
        ec = self._with_temp_cache(monkeypatch, tmp_path)
        ec.EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = ec.extraction_path("abc123")
        path.write_text("{ not valid json", encoding="utf-8")

        assert ec.read_extraction("abc123") is None
        # Original file renamed out of the way; a corrupt-* file exists.
        assert not path.exists()
        siblings = list(ec.EXTRACTIONS_DIR.iterdir())
        assert any(".corrupt-" in s.name for s in siblings)

    def test_corrupt_gzip_quarantined_and_returns_none(self, monkeypatch, tmp_path):
        ec = self._with_temp_cache(monkeypatch, tmp_path)
        ec.HTML_DIR.mkdir(parents=True, exist_ok=True)
        path = ec.html_path("abc123")
        # Not a gzip — gzip.open will raise OSError on read.
        path.write_bytes(b"not actually gzipped")

        assert ec.read_html("abc123") is None
        assert not path.exists()
        siblings = list(ec.HTML_DIR.iterdir())
        assert any(".corrupt-" in s.name for s in siblings)

    def test_missing_file_returns_none_without_quarantine(self, monkeypatch, tmp_path):
        ec = self._with_temp_cache(monkeypatch, tmp_path)
        assert ec.read_extraction("never-existed") is None
        assert ec.read_html("never-existed") is None
