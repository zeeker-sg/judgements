"""Tests for the 2026-11 fetch hardening: maintenance-stub detection and
the citation-derived court fallback.

During eLitigation maintenance windows every detail-page request returns an
~818-byte ":: eLitigation - Maintenance Notice ::" page with HTTP 200. Before
detection existed, 16 windows (Jul–Oct 2026) poisoned 525 rows as permanently
has_content=0. The stub must now raise ``MaintenancePageError`` so the
existing failure-counter/quarantine machinery retries later — and must NOT
be retried by the tenacity decorator (the window outlives a retry cycle).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import judgments
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

STUB_HTML = """<html><head><title>:: eLitigation - Maintenance Notice ::</title>
<style>.default_text { font-family:"Century Gothic"; }</style></head>
<body>:: eLitigation - Maintenance Notice ::<p>The site is down for
maintenance. Please try again later.</p></body></html>"""


def _client(body: str, status: int = 200, calls: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url)
        return httpx.Response(status, text=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestMaintenanceStubDetection:
    def test_stub_raises_maintenance_page_error(self):
        with _client(STUB_HTML) as client:
            with pytest.raises(judgments.MaintenancePageError):
                judgments._fetch_detail(client, "https://www.elitigation.sg/gd/s/2026_SGHC_234")

    def test_stub_is_not_retried(self):
        # MaintenancePageError is not an httpx.RequestError, so the tenacity
        # decorator must propagate it on the first attempt.
        calls: list = []
        with _client(STUB_HTML, calls=calls) as client:
            with pytest.raises(judgments.MaintenancePageError):
                judgments._fetch_detail(client, "https://www.elitigation.sg/gd/s/2026_SGHC_234")
        assert len(calls) == 1

    def test_real_page_passes_through(self):
        html = (FIXTURES / "legacy" / "legacy_2000_sghc_275.html").read_text()
        with _client(html) as client:
            out = judgments._fetch_detail(client, "https://www.elitigation.sg/gd/s/2000_SGHC_275")
        assert "Grounds of Decision" in out

    def test_marker_deep_in_body_is_not_a_stub(self):
        # A real judgment mentioning the notice phrase mid-body must pass;
        # only the stub carries it inside the document head.
        html = (
            "<html><head><title>[2000] SGHC 275</title></head><body>"
            + "x" * 3000
            + "<p>eLitigation - Maintenance Notice was mentioned in argument.</p>"
            + "</body></html>"
        )
        with _client(html) as client:
            out = judgments._fetch_detail(client, "https://www.elitigation.sg/gd/s/2000_SGHC_275")
        assert "mentioned in argument" in out

    def test_404_still_raises_http_status_error(self):
        with _client("gone", status=404) as client:
            with pytest.raises(httpx.HTTPStatusError):
                judgments._fetch_detail(client, "https://www.elitigation.sg/gd/s/2000_SGHC_275")


class TestCourtFallback:
    def test_url_match_still_wins(self):
        url = "https://www.elitigation.sg/gd/s/2013_SGHC_288"
        assert judgments.parse_court_from_url(url, "[2013] SGHC 288") == "SGHC"

    def test_sicc_codes_round_trip(self):
        assert (
            judgments.parse_court_from_url(
                "https://www.elitigation.sg/gd/s/2017_SGHCI_1", "[2017] SGHC(I) 1"
            )
            == "SGHCI"
        )

    def test_malformed_slug_falls_back_to_citation(self):
        # Source-side citation typo "[2013] SGHC115" produced the slug
        # 2013_SGHC115_, which defeats the URL regex.
        url = "https://www.elitigation.sg/gd/s/2013_SGHC115_"
        assert judgments.parse_court_from_url(url, "[2013] SGHC115") == "SGHC"

    def test_no_citation_and_bad_slug_yields_empty(self):
        assert judgments.parse_court_from_url("https://www.elitigation.sg/gd/s/2013_SGHC115_") == ""

    def test_citation_fallback_ignores_division_suffix(self):
        # "[2017] SGHC(I) 1" has no bare digits after the court token, so
        # CITATION_RE cannot rescue it — URL regex is the only path there.
        assert (
            judgments.parse_court_from_url("https://www.elitigation.sg/gd/x", "[2017] SGHC(I) 1")
            == ""
        )
