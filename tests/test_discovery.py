"""Phase 1 discovery walk tests: the JUDGMENTS_MIN_PAGES_PER_RUN floor.

Late-published judgments slot into eLitigation's date-sorted listing at their
decision-date position, BELOW newer arrivals. Before the floor existed, the
first run of INCREMENTAL_STOP_THRESHOLD known IDs stopped the walk — leaving
those rows permanently unseen. These tests pin the contract:

- the walk never stops on known-ID runs within the first MIN_PAGES_PER_RUN pages;
- the steady-state threshold applies fresh from page MIN+1 onward;
- MIN_PAGES_PER_RUN=0 restores the original first-trip stop.
"""

import contextlib
import json

import pytest

import judgments


class FakeTable:
    """Stand-in for a sqlite_utils Table: only .rows is consulted."""

    def __init__(self, ids):
        self.rows = [{"id": i} for i in ids]


def make_item(judgment_id):
    return {"id": judgment_id}


class FakeResponse:
    def __init__(self, page):
        self.status_code = 200
        # parse_listing_page (monkeypatched) reads the page number from here
        self.text = f"PAGE-{page}"


def install_pages(monkeypatch, pages):
    """pages: dict page_number -> list of item dicts; pages['_checkpoint'] is
    the checkpoint Path. Returns the fetch log."""
    fetch_log = []

    class FakeClient:
        def get(self, url, params=None):
            page = int(params["CurrentPage"])
            fetch_log.append(page)
            return FakeResponse(page)

    monkeypatch.setattr(
        judgments, "create_client", lambda: contextlib.nullcontext(FakeClient())
    )
    monkeypatch.setattr(
        judgments,
        "parse_listing_page",
        lambda html: pages[int(html.split("-")[1])],
    )
    monkeypatch.setattr(judgments, "detect_total_pages", lambda html: None)
    monkeypatch.setattr(judgments, "polite_sleep", lambda: None)
    monkeypatch.setattr(judgments, "CHECKPOINT_FILE", pages["_checkpoint"])
    return fetch_log


def page_items(page_no, known_ids, new_ids):
    return [make_item(i) for i in known_ids] + [make_item(i) for i in new_ids]


@pytest.fixture(autouse=True)
def _pacing_and_phases(monkeypatch):
    monkeypatch.setattr(judgments, "REQUEST_DELAY_BASE", 0.0)
    monkeypatch.setattr(judgments, "MIN_PAGES_PER_RUN", 10)
    monkeypatch.setattr(judgments, "MAX_PAGES_PER_RUN", 50)
    monkeypatch.setattr(judgments, "INCREMENTAL_STOP_THRESHOLD", 5)
    # Neutralise Phases 2+3 — they are covered by their own test modules.
    monkeypatch.setattr(
        judgments,
        "_run_phase2",
        lambda client, table, breaker: {
            "extracted": 0, "empty": 0, "backlog": 0,
            "failed": 0, "quarantined": 0,
        },
    )
    monkeypatch.setattr(
        judgments,
        "_run_phase3",
        lambda table: {"summarised": 0, "cached": 0, "backlog": 0},
    )


def test_floor_walks_past_known_runs_and_catches_late_published(tmp_path, monkeypatch):
    """The regression scenario: everything on pages 1-2 is known, and page 3
    holds two late-published judgments between known rows. The walk must pass
    through the known runs on the guaranteed pages and stage the page-3 rows."""
    monkeypatch.setattr(judgments, "MIN_PAGES_PER_RUN", 3)
    checkpoint = tmp_path / "checkpoint.json"
    pages = {
        1: page_items(1, [f"known-1-{n}" for n in range(10)], []),
        2: page_items(2, [f"known-2-{n}" for n in range(10)], []),
        # 4 known + 2 new + 4 known: the run of 5 known IDs completes within
        # page 3, but pages_this_run (2) < MIN (3), so the walk continues.
        3: (
            [make_item(f"known-3-{n}") for n in range(4)]
            + [make_item("late-a"), make_item("late-b")]
            + [make_item(f"known-3-{n}") for n in range(4, 8)]
        ),
        4: page_items(4, [f"known-4-{n}" for n in range(10)], []),
        "_checkpoint": checkpoint,
    }
    fetch_log = install_pages(monkeypatch, pages)

    known_all = [f"known-{p}-{n}" for p in (1, 2, 3, 4) for n in range(10)]

    staged = judgments.fetch_data(FakeTable(known_all))

    assert staged and {s["id"] for s in staged} == {"late-a", "late-b"}
    # walked the three floor pages, then page 4 where the threshold could fire
    assert fetch_log == [1, 2, 3, 4]
    # steady-state exit cleared the checkpoint
    assert not checkpoint.exists()


def test_floor_stops_on_first_unknown_run_beyond_min(tmp_path, monkeypatch):
    """Beyond the floor the original behaviour is unchanged: the first run of
    5 known IDs stops the walk."""
    monkeypatch.setattr(judgments, "MIN_PAGES_PER_RUN", 2)
    checkpoint = tmp_path / "checkpoint.json"
    pages = {
        p: page_items(p, [f"known-{p}-{n}" for n in range(10)], [])
        for p in (1, 2, 3, 4)
    }
    pages["_checkpoint"] = checkpoint
    fetch_log = install_pages(monkeypatch, pages)

    staged = judgments.fetch_data(
        FakeTable([f"known-{p}-{n}" for p in (1, 2, 3, 4) for n in range(10)])
    )

    assert staged == []
    # pages 1-2 are guaranteed (no stop possible), page 3 is the first page
    # where the threshold may fire — and it does, mid-page, on the 5th ID
    assert fetch_log == [1, 2, 3]
    assert not checkpoint.exists()


def test_min_zero_restores_original_first_trip_stop(tmp_path, monkeypatch):
    """JUDGMENTS_MIN_PAGES_PER_RUN=0 reproduces the pre-floor behaviour:
    stop inside page 1 at the 5th consecutive known ID."""
    monkeypatch.setattr(judgments, "MIN_PAGES_PER_RUN", 0)
    checkpoint = tmp_path / "checkpoint.json"
    items = [make_item(f"known-{n}") for n in range(5)] + [
        make_item("unseen-below-the-trip-point")
    ]
    pages = {1: items, "_checkpoint": checkpoint}
    fetch_log = install_pages(monkeypatch, pages)

    staged = judgments.fetch_data(FakeTable([f"known-{n}" for n in range(5)]))

    assert staged == []
    assert fetch_log == [1]
    assert not checkpoint.exists()


def test_floor_resets_threshold_after_new_row(tmp_path, monkeypatch):
    """A new ID inside the floor resets the known-run counter: 5 known before
    and after a new row must not stop the walk within the guaranteed pages."""
    monkeypatch.setattr(judgments, "MIN_PAGES_PER_RUN", 1)
    checkpoint = tmp_path / "checkpoint.json"
    page1 = (
        [make_item(f"known-{n}") for n in range(5)]
        + [make_item("fresh-1")]
        + [make_item(f"known-{n}") for n in range(5, 10)]
    )
    page2 = [make_item(f"known-2-{n}") for n in range(5)]
    pages = {1: page1, 2: page2, "_checkpoint": checkpoint}
    fetch_log = install_pages(monkeypatch, pages)

    known = (
        [f"known-{n}" for n in range(10)]
        + [f"known-2-{n}" for n in range(5)]
    )
    staged = judgments.fetch_data(FakeTable(known))

    assert [s["id"] for s in staged] == ["fresh-1"]
    # page 1 fully scanned (floor), page 2 stops at the 5th known ID
    assert fetch_log == [1, 2]
    assert not checkpoint.exists()


def test_checkpoint_preserved_when_batch_cap_hits_mid_floor(tmp_path, monkeypatch):
    """If MAX_PAGES_PER_RUN ends the walk mid-floor, staged items survive in
    the checkpoint for the next run (existing batch-crawl semantics)."""
    monkeypatch.setattr(judgments, "MIN_PAGES_PER_RUN", 10)
    monkeypatch.setattr(judgments, "MAX_PAGES_PER_RUN", 2)
    checkpoint = tmp_path / "checkpoint.json"
    pages = {
        p: page_items(p, [f"known-{p}-{n}" for n in range(9)], [f"new-{p}"])
        for p in (1, 2, 3)
    }
    pages["_checkpoint"] = checkpoint
    fetch_log = install_pages(monkeypatch, pages)

    staged = judgments.fetch_data(
        FakeTable([f"known-{p}-{n}" for p in (1, 2, 3) for n in range(9)])
    )

    assert staged and {s["id"] for s in staged} == {"new-1", "new-2"}
    assert fetch_log == [1, 2]
    # batch cap hit before the threshold could fire — checkpoint retains
    # position and staged items for the next run
    saved = json.loads(checkpoint.read_text())
    assert saved["last_page"] == 2
    assert {i["id"] for i in saved["items_collected"]} == {"new-1", "new-2"}


def test_checkpoint_staged_items_not_restaged_on_rediscovery(tmp_path, monkeypatch):
    """A staged checkpoint item must not be re-staged when the walk also
    finds it on the listing (observed live 2026-09-01: SGHC 178 inserted
    twice — once from a hand-staged checkpoint, once from page-1 discovery).
    Staged IDs join the known set before the walk starts."""
    monkeypatch.setattr(judgments, "MIN_PAGES_PER_RUN", 0)
    checkpoint = tmp_path / "checkpoint.json"
    # The checkpoint stages 'staged-1'; page 1's listing contains it too.
    checkpoint.write_text(
        json.dumps(
            {
                "last_page": 0,
                "items_collected": [dict(make_item("staged-1"), extra="ctx")],
                "total_pages": None,
            }
        )
    )
    pages = {
        1: [make_item("staged-1"), make_item("staged-2"), make_item("fresh-1")]
        + [make_item("old-1")] * 5,
        "_checkpoint": checkpoint,
    }
    install_pages(monkeypatch, pages)

    staged = judgments.fetch_data(FakeTable(["old-1"]))

    # staged-1 stays single-staged (its checkpoint copy), staged-2 and
    # fresh-1 come from the walk; nothing appears twice.
    ids = sorted(s["id"] for s in staged)
    assert ids == ["fresh-1", "staged-1", "staged-2"]
    assert len(ids) == len(set(ids))