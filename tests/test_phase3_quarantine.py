"""Unit tests for the Phase 3 escalating quarantine (issue #3).

The quarantine has three tiers keyed off the checkpoint's per-doc failure
count:

- below SUMMARY_MAX_RETRIES: retried with the primary model every build;
- from SUMMARY_MAX_RETRIES up to (exclusive) SUMMARY_HARD_FAIL_LIMIT:
  the alt-model priority pool — deliberately bypasses any TTL;
- at or past SUMMARY_HARD_FAIL_LIMIT: hard-quarantined for
  SUMMARY_RETRY_AFTER_HARD after the last attempt, so a doc that fails on
  both endpoints (like SGHC 249) stops burning a slot every daily build.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from resources import judgments

NOW = datetime(2026, 7, 16, 12, 0, 0)


def _state(count: int, last_attempt: str) -> dict:
    return {
        "failures": {"doc1": {"count": count, "last_error": "boom", "last_attempt": last_attempt}}
    }


class TestIsSummaryQuarantined:
    def test_unknown_doc_is_not_quarantined(self):
        assert not judgments._is_summary_quarantined({"failures": {}}, "doc1", NOW)

    def test_fresh_failures_retry_every_build(self):
        state = _state(judgments.SUMMARY_MAX_RETRIES - 1, NOW.isoformat())
        assert not judgments._is_summary_quarantined(state, "doc1", NOW)

    def test_alt_model_pool_bypasses_ttl(self):
        # Just reached the primary cap — promoted to alt-model retries,
        # never quarantined even seconds after the last failure.
        state = _state(judgments.SUMMARY_MAX_RETRIES, NOW.isoformat())
        assert not judgments._is_summary_quarantined(state, "doc1", NOW)

        # Last count before the hard limit still bypasses.
        state = _state(judgments.SUMMARY_HARD_FAIL_LIMIT - 1, NOW.isoformat())
        assert not judgments._is_summary_quarantined(state, "doc1", NOW)

    def test_hard_limit_with_recent_attempt_is_quarantined(self):
        last = (NOW - timedelta(days=1)).isoformat()
        state = _state(judgments.SUMMARY_HARD_FAIL_LIMIT, last)
        assert judgments._is_summary_quarantined(state, "doc1", NOW)

    def test_hard_limit_retries_after_window_expires(self):
        last = (NOW - timedelta(seconds=judgments.SUMMARY_RETRY_AFTER_HARD + 1)).isoformat()
        state = _state(judgments.SUMMARY_HARD_FAIL_LIMIT, last)
        assert not judgments._is_summary_quarantined(state, "doc1", NOW)

    def test_persistent_failure_stays_quarantined_past_24h(self):
        # The SGHC 249 scenario: 9 failures, last attempt yesterday. Under
        # the old 24h TTL (which the priority pool bypassed anyway) this
        # would be retried every daily build; now it waits out the hard
        # window.
        last = (NOW - timedelta(hours=25)).isoformat()
        state = _state(9, last)
        assert judgments._is_summary_quarantined(state, "doc1", NOW)

    def test_malformed_last_attempt_is_not_quarantined(self):
        state = _state(judgments.SUMMARY_HARD_FAIL_LIMIT, "not-a-timestamp")
        assert not judgments._is_summary_quarantined(state, "doc1", NOW)

    def test_missing_last_attempt_is_not_quarantined(self):
        state = _state(judgments.SUMMARY_HARD_FAIL_LIMIT, "")
        assert not judgments._is_summary_quarantined(state, "doc1", NOW)
