"""Unit tests for Phase 3 — scoring, packing, cache, and the full
_summarise_row flow against a mock OpenAI client + in-memory DB.

Keeps the openai SDK out of the test path entirely by injecting a stub
client into ``summarization.summarise``. That way the test suite runs in
the same ~0.3s as the rest of the repo and doesn't need network or a
real model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlite_utils

from resources import summarization, summary_cache
from resources.summarization import (
    compose_summary_input,
    score_fragment,
)


def _frag(
    *,
    ordinal: int,
    class_name: str,
    text: str,
    paragraph_number=None,
    section_heading=None,
    has_footnotes: bool = False,
    has_table: bool = False,
    judgment_id: str = "abc123",
) -> dict:
    return {
        "id": f"{judgment_id}_{ordinal:04d}",
        "judgment_id": judgment_id,
        "ordinal": ordinal,
        "paragraph_number": paragraph_number,
        "class_name": class_name,
        "section_heading": section_heading,
        "content_text": text,
        "html_raw": "",
        "footnote_text": None,
        "has_footnotes": has_footnotes,
        "has_table": has_table,
        "has_figure": False,
        "figure_src": None,
        "figure_descriptions": None,
    }


# ---------------------------------------------------------------------------
# score_fragment
# ---------------------------------------------------------------------------


class TestScoreFragment:
    def test_plain_short_paragraph_scores_low(self):
        f = _frag(ordinal=0, class_name="Judg-1", text="Short.", paragraph_number=1)
        # No signals except a tiny length bonus.
        assert score_fragment(f) == pytest.approx(0.1 * 6 / 100, abs=1e-6)

    def test_footnotes_add_two(self):
        f = _frag(ordinal=0, class_name="Judg-1", text="x" * 100, has_footnotes=True)
        # 2.0 (footnotes) + 0.1 * 100 / 100 = 2.1
        assert score_fragment(f) == pytest.approx(2.1, abs=1e-6)

    def test_dispositive_heading_adds_three(self):
        f = _frag(
            ordinal=0,
            class_name="Judg-1",
            text="",
            section_heading="Conclusion and disposition",
        )
        # 3.0 (dispositive) + 0 length. Headings match "conclusion".
        assert score_fragment(f) == pytest.approx(3.0, abs=1e-6)

    def test_analysis_heading_adds_one_point_five(self):
        f = _frag(
            ordinal=0,
            class_name="Judg-1",
            text="",
            section_heading="Analysis of the first issue",
        )
        assert score_fragment(f) == pytest.approx(1.5, abs=1e-6)

    def test_dispositive_wins_over_analysis(self):
        # Heading mentions both "decision" (dispositive) and "analysis";
        # the dispositive weight should win because it matches first.
        f = _frag(
            ordinal=0,
            class_name="Judg-1",
            text="",
            section_heading="Decision and analysis",
        )
        assert score_fragment(f) == pytest.approx(3.0, abs=1e-6)

    def test_has_table_adds_half(self):
        f = _frag(ordinal=0, class_name="Judg-1", text="", has_table=True)
        assert score_fragment(f) == pytest.approx(0.5, abs=1e-6)

    def test_length_bonus_caps_at_half(self):
        # 10,000-char paragraph should still only contribute 0.5 (cap).
        f = _frag(ordinal=0, class_name="Judg-1", text="x" * 10_000)
        assert score_fragment(f) == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# compose_summary_input — always-keep membership + budget + order
# ---------------------------------------------------------------------------


class TestComposeSummaryInput:
    def _sample_row(self, court_summary: str = "") -> dict:
        return {
            "id": "abc123",
            "court_summary": court_summary,
            "content_text": "full content",
        }

    def _build_fragments(self) -> list:
        # 1 heading, 5 numbered paragraphs, 2 unnumbered fragments.
        return [
            _frag(ordinal=0, class_name="Judg-Heading-1", text="Introduction"),
            _frag(
                ordinal=1,
                class_name="Judg-1",
                text="First numbered para.",
                paragraph_number=1,
                section_heading="Introduction",
            ),
            _frag(
                ordinal=2,
                class_name="Judg-1",
                text="Second numbered para.",
                paragraph_number=2,
                section_heading="Introduction",
            ),
            _frag(ordinal=3, class_name="Judg-Quote-1", text="Quoted text."),
            _frag(
                ordinal=4,
                class_name="Judg-1",
                text="Third numbered para.",
                paragraph_number=3,
                section_heading="Analysis",
            ),
            _frag(
                ordinal=5,
                class_name="Judg-1",
                text="Fourth numbered para.",
                paragraph_number=4,
                section_heading="Analysis",
            ),
            _frag(
                ordinal=6,
                class_name="Judg-1",
                text="Fifth numbered para.",
                paragraph_number=5,
                section_heading="Conclusion",
            ),
            _frag(ordinal=7, class_name="Judg-List-1", text="List entry."),
        ]

    def test_always_keeps_court_summary_headings_first_and_last_three(self):
        row = self._sample_row(court_summary="Court's own summary.")
        out = compose_summary_input(row, self._build_fragments(), max_chars=4000)
        # Court summary goes at the top.
        assert out.startswith("## Court Summary")
        assert "Court's own summary." in out
        # Every heading is present.
        assert "## Introduction" in out
        # First numbered paragraph (paragraph_number=1).
        assert "[1] First numbered para." in out
        # Last three numbered paragraphs (3, 4, 5).
        assert "[3] Third numbered para." in out
        assert "[4] Fourth numbered para." in out
        assert "[5] Fifth numbered para." in out

    def test_empty_court_summary_is_omitted(self):
        row = self._sample_row(court_summary="")
        out = compose_summary_input(row, self._build_fragments(), max_chars=4000)
        assert "## Court Summary" not in out

    def test_budget_is_respected(self):
        # Very tight budget — output must be ≤ max_chars.
        row = self._sample_row(court_summary="Court's own summary.")
        out = compose_summary_input(row, self._build_fragments(), max_chars=120)
        assert len(out) <= 120

    def test_document_order_is_preserved(self):
        # Numbered paras should be emitted in paragraph_number order in
        # the output, mirroring document order.
        row = self._sample_row()
        out = compose_summary_input(row, self._build_fragments(), max_chars=4000)
        pos1 = out.find("[1] First")
        pos3 = out.find("[3] Third")
        pos5 = out.find("[5] Fifth")
        assert 0 < pos1 < pos3 < pos5

    def test_fallback_when_no_fragments(self):
        row = {"id": "abc123", "court_summary": "", "content_text": "Raw body text."}
        out = compose_summary_input(row, [], max_chars=20)
        assert out == "Raw body text."[:20]

    def test_scored_remainder_fills_remaining_budget(self):
        # Add a high-scoring remainder (has_footnotes + dispositive
        # heading) and check it gets picked when there's room.
        frags = self._build_fragments()
        frags.append(
            _frag(
                ordinal=8,
                class_name="Judg-Quote-0",
                text="Pivotal quote about the holding.",
                section_heading="Decision",
                has_footnotes=True,
            )
        )
        row = self._sample_row()
        out = compose_summary_input(row, frags, max_chars=4000)
        assert "Pivotal quote about the holding." in out


# ---------------------------------------------------------------------------
# summary_cache — round-trip and quarantine
# ---------------------------------------------------------------------------


class TestSummaryCache:
    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        payload = {
            "judgment_id": "abc123",
            "generated_at": "2026-04-18T12:00:00",
            "model": "llama3.1:8b",
            "endpoint": "http://localhost:11434/v1",
            "input_chars": 1234,
            "frags_kept": 5,
            "summary": "A single paragraph summary.",
        }
        summary_cache.write_summary_atomic("abc123", payload)
        assert summary_cache.read_summary("abc123") == payload

    def test_corrupt_json_is_quarantined(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Write a corrupt file directly.
        summary_cache.SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        path = summary_cache.summary_path("abc123")
        path.write_text("{not-valid-json")
        assert summary_cache.read_summary("abc123") is None
        # Corrupt file is renamed out of the way.
        assert not path.exists()
        quarantined = list(summary_cache.SUMMARIES_DIR.glob("*.corrupt-*"))
        assert len(quarantined) == 1


# ---------------------------------------------------------------------------
# _summarise_row — mock LLM + in-memory DB integration
# ---------------------------------------------------------------------------


class TestSummariseRow:
    def test_end_to_end_with_mock_client(self, tmp_path, monkeypatch):
        # Run in a temp dir so cache + checkpoint writes don't pollute the repo.
        monkeypatch.chdir(tmp_path)

        # Build an in-memory sqlite_utils DB mirroring Phase 1/2 schema
        # well enough for the row-update path. We only need id, summary,
        # summary_generated_at, court_summary, content_text,
        # has_content on the main table, plus the fragments table.
        db = sqlite_utils.Database(":memory:")
        db["judgments"].insert(
            {
                "id": "abc123",
                "citation": "[2026] SGHC 1",
                "court": "SGHC",
                "decision_date": "2026-01-01",
                "court_summary": "Court's own summary.",
                "content_text": "Body.",
                "summary": None,
                "summary_generated_at": None,
                "has_content": 1,
            }
        )
        db["judgments_fragments"].insert(
            {
                "id": "abc123_0000",
                "judgment_id": "abc123",
                "ordinal": 0,
                "paragraph_number": 1,
                "class_name": "Judg-1",
                "section_heading": None,
                "content_text": "First numbered paragraph.",
                "html_raw": "",
                "footnote_text": None,
                "has_footnotes": 0,
                "has_table": 0,
                "has_figure": 0,
                "figure_src": None,
                "figure_descriptions": None,
            },
            pk="id",
        )

        # Import judgments module via resources/ path (sibling-import hack).
        import importlib.util
        import sys

        resources_dir = Path(__file__).resolve().parent.parent / "resources"
        if str(resources_dir) not in sys.path:
            sys.path.insert(0, str(resources_dir))
        spec = importlib.util.spec_from_file_location(
            "_judgments_under_test", resources_dir / "judgments.py"
        )
        judgments_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(judgments_mod)

        # Stub the LLM call — we don't want openai as a test-time dep.
        # NOTE: judgments_mod's sibling-import hack loads summarization.py
        # as the top-level module "summarization". That is a DIFFERENT
        # module object from resources.summarization. Patch the one the
        # code under test actually calls into: judgments_mod.summarization.
        # _summarise_row calls rolling_summarise(row, fragments, model, client),
        # so that is the function to stub.
        monkeypatch.setattr(
            judgments_mod.summarization,
            "rolling_summarise",
            lambda row, fragments, model, client, **kwargs: ("Stub summary paragraph of the judgment."),
        )
        # summary_cache is likewise a separate module object; use the
        # one judgments_mod imported so its writes land in our tmp dir.
        sc_mod = judgments_mod.summary_cache

        row = dict(db["judgments"].rows_where("id = ?", ["abc123"]).__next__())
        status, detail = judgments_mod._summarise_row(
            row, db["judgments"], client=MagicMock(), model="stub-model"
        )

        assert status == "ok"
        updated = dict(db["judgments"].rows_where("id = ?", ["abc123"]).__next__())
        assert updated["summary"] == "Stub summary paragraph of the judgment."
        assert updated["summary_generated_at"]  # ISO timestamp set.
        # Cache file written via the module judgments_mod imported.
        cached = sc_mod.read_summary("abc123")
        assert cached is not None
        assert cached["summary"] == "Stub summary paragraph of the judgment."
        assert cached["model"] == "stub-model"

    def test_uses_cached_summary_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        db = sqlite_utils.Database(":memory:")
        db["judgments"].insert(
            {
                "id": "abc123",
                "court_summary": "",
                "content_text": "Body.",
                "summary": None,
                "summary_generated_at": None,
                "has_content": 1,
            }
        )

        # Prime the cache.
        summary_cache.write_summary_atomic(
            "abc123",
            {
                "judgment_id": "abc123",
                "generated_at": "2026-04-18T00:00:00",
                "model": "cached-model",
                "endpoint": "",
                "input_chars": 0,
                "frags_kept": 0,
                "summary": "Cached paragraph.",
            },
        )

        import importlib.util
        import sys

        resources_dir = Path(__file__).resolve().parent.parent / "resources"
        if str(resources_dir) not in sys.path:
            sys.path.insert(0, str(resources_dir))
        spec = importlib.util.spec_from_file_location(
            "_judgments_under_test2", resources_dir / "judgments.py"
        )
        judgments_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(judgments_mod)

        # summarise should NOT be called — raise loudly if it is.
        def _boom(*a, **kw):
            raise AssertionError("summarise() was called despite cache hit")

        monkeypatch.setattr(summarization, "summarise", _boom)

        row = dict(db["judgments"].rows_where("id = ?", ["abc123"]).__next__())
        status, detail = judgments_mod._summarise_row(
            row, db["judgments"], client=MagicMock(), model="unused"
        )
        assert status == "cached"
        updated = dict(db["judgments"].rows_where("id = ?", ["abc123"]).__next__())
        assert updated["summary"] == "Cached paragraph."
