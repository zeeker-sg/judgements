"""
Singapore court judgments from eLitigation.

Source:   https://www.elitigation.sg/gd/
Scope:    ~10,588 judgments across SGCA, SGHC, SGHCA, SGHCF, SGHCR, SGDC,
          SGFC, SGMC + tribunals like SGCDT, SGSCT (back to 2000). The
          court code is parsed permissively from the URL — any SG[A-Z]+
          token is accepted.
Cadence:  Tier 4 one-shot batch for initial backfill; transitions to Tier 1
          daily incremental once the archive is fully discovered.

Two-phase pipeline runs in one fetch_data invocation per build:

Phase 1 (discovery): scrape listing pages, persist catalog metadata
  (citation, case name, court, date, subject tags, source_url, pdf_url).
  ``content_text``, ``court_summary``, fragment rows stay NULL.

Phase 2 (enrichment): for each existing row with ``content_text IS NULL``,
  fetch the detail HTML at source_url, parse paragraphs/tables/footnotes
  via resources.extraction, archive raw HTML under ``.cache/judgments_html/``
  and parsed output under ``.cache/judgments_extractions/``, then
  ``existing_table.update()`` the row with content_text, court_summary,
  has_content, has_court_summary, fragment_count, extracted_at. The extraction
  cache is the source of truth that ``fetch_fragments_data`` reads from —
  zeeker reloads this module between main-table and fragment phases, so any
  in-memory state would be lost. Disk cache survives the reload and any
  crash mid-run (atomic writes).

Crawl strategy (shared)
-----------------------
- Single ``httpx.Client`` with connection pool, jittered 1-2s delay between
  requests, 3-attempt exponential-backoff retry on transient errors, and a
  circuit breaker that bails out with a saved checkpoint after 5 consecutive
  failures.

Phase-1 specifics
-----------------
- Pagination: URL-parameter Pattern A (?CurrentPage=N), 10 items/page.
- Checkpoint/resume: state in ``checkpoint_judgments_discovery.json``.
- Batch cap: ``MAX_PAGES_PER_RUN`` (~50 pages = a few minutes per run).
- Incremental stop: ``INCREMENTAL_STOP_THRESHOLD`` consecutive known IDs.

Phase-2 specifics
-----------------
- Batch cap: ``EXTRACT_MAX_PER_RUN`` (~15 docs per run, conservative given
  the source is flaky).
- Failure tracking: ``checkpoint_judgments_extraction.json`` records per-id
  failure count + last error + last attempt timestamp. After
  ``EXTRACT_MAX_RETRIES`` failures a doc is quarantined until
  ``EXTRACT_RETRY_AFTER`` seconds have passed, giving the source time to
  recover without blocking the rest of the backlog.
- Set ``JUDGMENTS_EXTRACT_ENABLED=0`` to run discovery only (useful during
  the initial Phase-1 archive crawl).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin

import click
import httpx
from bs4 import BeautifulSoup
from sqlite_utils.db import Table
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Zeeker loads resource files via importlib.util.spec_from_file_location,
# which bypasses package imports — ``from resources import ...`` fails at
# build time. Make sibling modules importable by adding this file's
# directory to sys.path before importing them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction  # noqa: E402
import extraction_cache  # noqa: E402
import summarization  # noqa: E402
import summary_cache  # noqa: E402
from extraction import ExtractionError  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_URL = "https://www.elitigation.sg"
INDEX_PATH = "/gd/Home/Index"
INDEX_PARAMS = {
    "Filter": "SUPCT",
    "YearOfDecision": "All",
    "SortBy": "Date",
    "SortAscending": "False",
}
USER_AGENT = "ZeekerBot/1.0 (+https://data.zeeker.sg)"

# Crawl pacing (env-overridable so smoke tests don't require code edits)
MAX_PAGES_PER_RUN = int(os.environ.get("JUDGMENTS_MAX_PAGES_PER_RUN", "50"))
INCREMENTAL_STOP_THRESHOLD = int(os.environ.get("JUDGMENTS_INCREMENTAL_STOP", "5"))
REQUEST_DELAY_BASE = float(os.environ.get("JUDGMENTS_DELAY_BASE", "1.5"))
REQUEST_DELAY_JITTER = float(os.environ.get("JUDGMENTS_DELAY_JITTER", "0.5"))
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 3

# Circuit breaker
MAX_CONSECUTIVE_FAILURES = 5
CIRCUIT_BREAKER_COOLDOWN = 60.0

# State
CHECKPOINT_FILE = Path("checkpoint_judgments_discovery.json")
EXTRACTION_CHECKPOINT_FILE = Path("checkpoint_judgments_extraction.json")
SUMMARY_CHECKPOINT_FILE = Path("checkpoint_judgments_summary.json")

# Phase 2 knobs
EXTRACT_ENABLED = os.environ.get("JUDGMENTS_EXTRACT_ENABLED", "1") == "1"
EXTRACT_MAX_PER_RUN = int(os.environ.get("JUDGMENTS_EXTRACT_MAX_PER_RUN", "15"))
EXTRACT_MAX_RETRIES = int(os.environ.get("JUDGMENTS_EXTRACT_MAX_RETRIES", "3"))
EXTRACT_RETRY_AFTER = int(os.environ.get("JUDGMENTS_EXTRACT_RETRY_AFTER", "86400"))
EXTRACT_DELAY_BASE = float(os.environ.get("JUDGMENTS_EXTRACT_DELAY_BASE", "1.5"))
EXTRACT_DELAY_JITTER = float(os.environ.get("JUDGMENTS_EXTRACT_DELAY_JITTER", "0.5"))

# Phase 3 knobs
SUMMARY_ENABLED = os.environ.get("JUDGMENTS_SUMMARY_ENABLED", "1") == "1"
SUMMARY_MAX_PER_RUN = int(os.environ.get("JUDGMENTS_SUMMARY_MAX_PER_RUN", "15"))
SUMMARY_MAX_RETRIES = int(os.environ.get("JUDGMENTS_SUMMARY_MAX_RETRIES", "3"))
SUMMARY_RETRY_AFTER = int(os.environ.get("JUDGMENTS_SUMMARY_RETRY_AFTER", "86400"))
SUMMARY_MAX_INPUT_CHARS = int(os.environ.get("JUDGMENTS_SUMMARY_MAX_INPUT_CHARS", "32000"))

# Per-process sentinel so Phase 2 runs ONCE per build even though zeeker
# reloads this module and re-calls fetch_data to populate fragment context.
# Module-level vars don't survive the reload; env vars do (same process).
_PHASE2_SENTINEL_ENV = "_JUDGMENTS_PHASE2_RAN_PID"
_PHASE3_SENTINEL_ENV = "_JUDGMENTS_PHASE3_RAN_PID"
_PHASE3_PROGRESS_FILE = Path(__file__).parent.parent / "phase3_progress.jsonl"


def _phase3_log(entry: dict) -> None:
    """Append one JSON line to the Phase 3 progress file (for real-time monitoring)."""
    try:
        with open(_PHASE3_PROGRESS_FILE, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
            fh.flush()
    except OSError:
        pass

# Per-process cache — zeeker's fragment build flow may call fetch_data twice
# (once for main-table insert, once to provide main_data_context for fragments).
# Caching here keeps the crawl idempotent within a single process so we don't
# double the HTTP cost or advance the checkpoint past work zeeker will ignore.
_FETCH_CACHE: Optional[List[Dict[str, Any]]] = None

# Parsing — court alternation is permissive because eLitigation emits codes
# beyond the "canonical six" (SGHCA appellate division, SGHCR registrar
# appeals, SGCDT community disputes tribunal, SGSCT small claims tribunal,
# etc). Any /gd/s/YYYY_SGXXX_N URL yields SGXXX as the court.
CITATION_RE = re.compile(r"\[(\d{4})\]\s*(SG[A-Z]+)\s*(\d+)")
COURT_FROM_URL_RE = re.compile(r"/gd/s/\d{4}_(SG[A-Z]+)_\d+", re.IGNORECASE)
TOTAL_PAGES_RE = re.compile(r"CurrentPage=(\d+)")
DATE_PARAM_RE = re.compile(r'DecisionDate:"(\d{4}-\d{2}-\d{2})"')
VISIBLE_DATE_RE = re.compile(r"Decision Date:\s*(\d{1,2}\s+\w+\s+\d{4})")


# =============================================================================
# HELPERS
# =============================================================================
def make_id(*elements: str) -> str:
    joined = "|".join(str(e) for e in elements)
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def polite_sleep(base: float = REQUEST_DELAY_BASE, jitter: float = REQUEST_DELAY_JITTER) -> None:
    delay = base + random.uniform(-jitter, jitter)
    time.sleep(max(0.1, delay))


class CircuitBreaker:
    def __init__(
        self,
        max_failures: int = MAX_CONSECUTIVE_FAILURES,
        cooldown: float = CIRCUIT_BREAKER_COOLDOWN,
    ):
        self.max_failures = max_failures
        self.cooldown = cooldown
        self.consecutive_failures = 0
        self.total_failures = 0
        self.total_successes = 0

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_successes += 1

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_failures += 1

    @property
    def is_open(self) -> bool:
        return self.consecutive_failures >= self.max_failures

    def wait_if_needed(self) -> None:
        if self.is_open:
            click.echo(
                f"Circuit breaker: {self.consecutive_failures} consecutive failures. "
                f"Cooling down for {self.cooldown:.0f}s.",
                err=True,
            )
            time.sleep(self.cooldown)
            self.consecutive_failures = 0

    def summary(self) -> str:
        total = self.total_successes + self.total_failures
        return f"{self.total_successes}/{total} pages succeeded, {self.total_failures} failed"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        click.echo(
            f"Resuming discovery: next page is {data.get('last_page', 0) + 1}, "
            f"{len(data.get('items_collected', []))} items pending from prior runs"
        )
        return data
    return {"last_page": 0, "items_collected": [], "total_pages": None}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2, default=str))


def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        click.echo("Checkpoint cleared — discovery complete")


def create_client() -> httpx.Client:
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        follow_redirects=True,
    )


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    reraise=True,
)
def _fetch_listing(client: httpx.Client, url: str, params: dict) -> httpx.Response:
    response = client.get(url, params=params)
    # Retry 5xx and network errors; let 4xx surface without retry.
    if 500 <= response.status_code < 600:
        response.raise_for_status()
    return response


def detect_total_pages(html: str) -> Optional[int]:
    """Parse the pagination 'Last' link to discover total page count."""
    soup = BeautifulSoup(html, "lxml")
    last_link = soup.select_one(".pagination .PagedList-skipToLast a")
    if last_link is None or not last_link.get("href"):
        return None
    match = TOTAL_PAGES_RE.search(last_link["href"])
    return int(match.group(1)) if match else None


def parse_decision_date(card) -> str:
    """Extract ISO-8601 decision date from the card, preferring the machine-readable attribute."""
    date_link = card.select_one("a.decision-date-link")
    if date_link is None:
        return ""
    dsp = date_link.get("data-searchparam", "")
    match = DATE_PARAM_RE.search(dsp)
    if match:
        return match.group(1)
    # Fallback: parse visible "Decision Date: 17 Apr 2026" text
    txt = date_link.get_text(" ", strip=True)
    vis = VISIBLE_DATE_RE.search(txt)
    if vis:
        try:
            return datetime.strptime(vis.group(1), "%d %b %Y").date().isoformat()
        except ValueError:
            return ""
    return ""


def parse_court_from_url(source_url: str) -> str:
    match = COURT_FROM_URL_RE.search(source_url)
    return match.group(1).upper() if match else ""


def build_pdf_url(citation: str) -> str:
    """Build authoritative PDF URL from citation like '[2026] SGDC 136'.

    The PDF endpoint is /gd/gd/<citation>/pdf — the citation segment contains
    brackets and spaces, which must be percent-encoded for the URL to be
    valid for a plain HTTP client.
    """
    cm = CITATION_RE.search(citation)
    if cm is None:
        return ""
    year, court, number = cm.group(1), cm.group(2), cm.group(3)
    cit_slug = f"[{year}] {court} {number}"
    return urljoin(BASE_URL, f"/gd/gd/{quote(cit_slug, safe='')}/pdf")


def parse_listing_page(html: str) -> List[Dict[str, Any]]:
    """Extract judgment records from a listing page's HTML."""
    soup = BeautifulSoup(html, "lxml")
    items: List[Dict[str, Any]] = []
    now = datetime.now().isoformat()

    for card in soup.select("div.card.col-12"):
        title_link = card.select_one("a.gd-heardertext")
        if title_link is None or not title_link.get("href"):
            continue

        href = title_link["href"].strip()
        source_url = urljoin(BASE_URL, href)
        case_name = title_link.get_text(strip=True)
        if not case_name:
            continue

        citation_el = card.select_one("a.citation-num-link span.gd-addinfo-text")
        citation = ""
        if citation_el is not None:
            citation = citation_el.get_text(strip=True).rstrip("|").strip()

        decision_date = parse_decision_date(card)

        case_numbers = " | ".join(
            el.get_text(strip=True)
            for el in card.select("a.case-num-link")
            if el.get_text(strip=True)
        )

        subject_tags: List[str] = []
        for cw in card.select(".gd-catchword-container a.gd-cw"):
            text = cw.get_text(strip=True)
            if not text:
                continue
            # Catchwords are shown like "[Foo — Bar]"; strip the outer brackets.
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1].strip()
            subject_tags.append(text)

        items.append(
            {
                "id": make_id(source_url),
                "citation": citation,
                "case_name": case_name,
                "case_numbers": case_numbers,
                "decision_date": decision_date,
                "court": parse_court_from_url(source_url),
                "subject_tags": json.dumps(subject_tags, ensure_ascii=False),
                "source_url": source_url,
                "pdf_url": build_pdf_url(citation),
                "content_text": None,
                "court_summary": None,
                "summary": None,
                "created_at": now,
            }
        )

    return items


# =============================================================================
# PHASE 2 — ENRICHMENT (helpers)
# =============================================================================
# The extraction checkpoint records per-judgment failure state so the
# source's flakiness doesn't spin forever on the same broken doc.
# Shape:
#   {
#     "failures": {
#       "<judgment_id>": {
#         "count": int,
#         "last_error": str,
#         "last_attempt": "YYYY-MM-DDTHH:MM:SS"
#       }, ...
#     }
#   }


def load_extraction_state() -> dict:
    if EXTRACTION_CHECKPOINT_FILE.exists():
        try:
            return json.loads(EXTRACTION_CHECKPOINT_FILE.read_text())
        except json.JSONDecodeError:
            click.echo("Extraction checkpoint was corrupt — starting fresh.", err=True)
    return {"failures": {}}


def save_extraction_state(state: dict) -> None:
    tmp = EXTRACTION_CHECKPOINT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, EXTRACTION_CHECKPOINT_FILE)


def _record_extraction_failure(state: dict, jid: str, err: Exception) -> None:
    failures = state.setdefault("failures", {})
    entry = failures.setdefault(jid, {"count": 0, "last_error": "", "last_attempt": ""})
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["last_error"] = f"{type(err).__name__}: {err}"[:500]
    entry["last_attempt"] = datetime.now().isoformat(timespec="seconds")


def _clear_extraction_failure(state: dict, jid: str) -> None:
    failures = state.get("failures", {})
    if jid in failures:
        del failures[jid]


def _is_quarantined(state: dict, jid: str, now: datetime) -> bool:
    entry = state.get("failures", {}).get(jid)
    if entry is None:
        return False
    if int(entry.get("count", 0)) < EXTRACT_MAX_RETRIES:
        return False
    last_attempt_str = entry.get("last_attempt") or ""
    try:
        last_attempt = datetime.fromisoformat(last_attempt_str)
    except ValueError:
        return False
    age_seconds = (now - last_attempt).total_seconds()
    return age_seconds < EXTRACT_RETRY_AFTER


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    reraise=True,
)
def _fetch_detail(client: httpx.Client, url: str) -> str:
    """Fetch a detail-page HTML with the same retry policy as Phase 1."""
    response = client.get(url)
    if 500 <= response.status_code < 600:
        response.raise_for_status()
    if response.status_code >= 400:
        # 404 on a source_url means the judgment was removed — raise so
        # the caller can record a structural failure and quarantine.
        response.raise_for_status()
    return response.text


PHASE2_ADDED_COLUMNS = {
    "has_content": int,
    "has_court_summary": int,
    "fragment_count": int,
    "extracted_at": str,
}

# Phase 3 column additions. ``summary`` is already present from Phase 1
# (initialized to NULL in ``parse_listing_page``), so only the timestamp
# is genuinely new.
PHASE3_ADDED_COLUMNS = {
    "summary_generated_at": str,
}

FRAGMENT_COLUMNS = {
    "id": str,
    "judgment_id": str,
    "ordinal": int,
    "paragraph_number": int,
    "class_name": str,
    "section_heading": str,
    "content_text": str,
    "html_raw": str,
    "footnote_text": str,
    "has_footnotes": int,
    "has_table": int,
    "has_figure": int,
    "figure_src": str,
    "figure_descriptions": str,
}
FRAGMENTS_TABLE_NAME = "judgments_fragments"


def _ensure_phase2_columns(table: Table) -> None:
    """Idempotently add Phase-2 columns to the judgments table.

    Zeeker builds the initial schema without a declared primary key
    (just the implicit rowid), so sqlite_utils ``table.update`` by the
    ``id`` value fails with NotFoundError. We do our own column
    management + ``UPDATE … WHERE id = ?`` in raw SQL.
    """
    existing = set(table.columns_dict)
    for name, col_type in PHASE2_ADDED_COLUMNS.items():
        if name not in existing:
            table.add_column(name, col_type)


def _ensure_fragments_table(db) -> Table:
    """Create ``judgments_fragments`` with an explicit schema if missing.

    Done here (not via a fetch_fragments_data first-row inference) so
    Phase 2 can write fragments in the same transaction as the main-row
    UPDATE — zeeker skips the fragment-fetch pipeline when fetch_data
    returns no new rows, which happens on every steady-state daily
    backfill run.
    """
    if FRAGMENTS_TABLE_NAME not in db.table_names():
        db[FRAGMENTS_TABLE_NAME].create(FRAGMENT_COLUMNS, pk="id")
    db.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{FRAGMENTS_TABLE_NAME}_judgment_id"
        f" ON {FRAGMENTS_TABLE_NAME}(judgment_id)"
    )
    return db[FRAGMENTS_TABLE_NAME]


def _update_row(table: Table, jid: str, updates: Dict[str, Any]) -> None:
    """``UPDATE judgments SET <cols> WHERE id = ?`` — works regardless of PK."""
    if not updates:
        return
    cols = list(updates.keys())
    placeholders = ", ".join(f'"{c}" = ?' for c in cols)
    params = [updates[c] for c in cols] + [jid]
    table.db.execute(f'UPDATE "{table.name}" SET {placeholders} WHERE id = ?', params)


def _insert_fragments(db, fragments: List[Dict[str, Any]]) -> int:
    """Insert fragment rows, ignoring duplicates by PK.

    Returns the number of rows actually inserted (sqlite_utils reports
    total_changes, which reflects ignored rows as 0 changes).
    """
    if not fragments:
        return 0
    table = _ensure_fragments_table(db)
    before = db.conn.total_changes
    # ignore=True → INSERT OR IGNORE, so re-runs after a crash don't
    # collide on the deterministic PKs we compose in extraction.py.
    table.insert_all(fragments, ignore=True, alter=False)
    return db.conn.total_changes - before


def _push_cached_extraction_to_db(existing_table: Table, jid: str, cached: Dict[str, Any]) -> None:
    """Update main row + insert fragments from an already-cached extraction."""
    if cached.get("extraction_status", "").startswith("empty"):
        _update_row(
            existing_table,
            jid,
            {
                "content_text": "",
                "court_summary": "",
                "has_content": 0,
                "has_court_summary": 0,
                "fragment_count": 0,
                "extracted_at": cached.get("extracted_at") or "",
            },
        )
        return
    _update_row(
        existing_table,
        jid,
        {
            "content_text": cached.get("content_text") or "",
            "court_summary": cached.get("court_summary") or "",
            "has_content": 1 if cached.get("has_content") else 0,
            "has_court_summary": 1 if cached.get("has_court_summary") else 0,
            "fragment_count": len(cached.get("fragments") or []),
            "extracted_at": cached.get("extracted_at") or "",
        },
    )
    _insert_fragments(existing_table.db, cached.get("fragments") or [])


def _enrich_row(
    client: httpx.Client,
    row: Dict[str, Any],
    existing_table: Table,
    breaker: CircuitBreaker,
) -> Tuple[str, Optional[str]]:
    """Extract + update one judgment row. Returns ``(status, detail)``.

    Status is one of: ``ok`` (content populated), ``empty`` (extraction
    ran but divJudgement was missing/empty; row marked has_content=False
    so we won't retry), or ``http_error`` (transient fetch failure;
    caller bumps failure counter).
    """
    jid = row["id"]
    source_url = row["source_url"]

    # Fast path: cached extraction from an earlier run that crashed
    # before the DB write. Push it to the DB without re-parsing.
    cached = extraction_cache.read_extraction(jid)
    if cached is not None:
        _push_cached_extraction_to_db(existing_table, jid, cached)
        if cached.get("extraction_status", "").startswith("empty"):
            return "empty", "cached"
        frag_n = len(cached.get("fragments") or [])
        return "ok", f"{frag_n} frags (cached)"

    # 1. Load HTML — either from disk cache or by fetching.
    html = extraction_cache.read_html(jid)
    if html is None:
        breaker.wait_if_needed()
        try:
            html = _fetch_detail(client, source_url)
            breaker.record_success()
        except Exception as exc:
            breaker.record_failure()
            return "http_error", f"{type(exc).__name__}: {exc}"
        extraction_cache.write_html_atomic(jid, html)
        polite_sleep(EXTRACT_DELAY_BASE, EXTRACT_DELAY_JITTER)

    # 2. Parse. Empty body is a structural failure we want to remember.
    try:
        extracted = extraction.extract_judgment(html, jid)
    except ExtractionError as exc:
        empty_iso = datetime.now().isoformat(timespec="seconds")
        empty_payload = {
            "judgment_id": jid,
            "extracted_at": empty_iso,
            "content_text": "",
            "court_summary": "",
            "has_content": False,
            "has_court_summary": False,
            "fragments": [],
            "extraction_status": f"empty: {exc}",
        }
        extraction_cache.write_extraction_atomic(jid, empty_payload)
        _update_row(
            existing_table,
            jid,
            {
                "content_text": "",
                "court_summary": "",
                "has_content": 0,
                "has_court_summary": 0,
                "fragment_count": 0,
                "extracted_at": empty_iso,
            },
        )
        return "empty", str(exc)

    now_iso = datetime.now().isoformat(timespec="seconds")
    payload = {
        "judgment_id": jid,
        "extracted_at": now_iso,
        "content_text": extracted.content_text,
        "court_summary": extracted.court_summary,
        "has_content": extracted.has_content,
        "has_court_summary": extracted.has_court_summary,
        "fragments": extracted.fragments,
        "extraction_status": "ok",
    }
    extraction_cache.write_extraction_atomic(jid, payload)

    _update_row(
        existing_table,
        jid,
        {
            "content_text": extracted.content_text,
            "court_summary": extracted.court_summary,
            "has_content": 1 if extracted.has_content else 0,
            "has_court_summary": 1 if extracted.has_court_summary else 0,
            "fragment_count": len(extracted.fragments),
            "extracted_at": now_iso,
        },
    )
    _insert_fragments(existing_table.db, extracted.fragments)
    return "ok", (
        f"{len(extracted.fragments)} frags, "
        f"{sum(1 for f in extracted.fragments if f['has_footnotes'])} w/fn"
    )


def _run_phase2(
    client: httpx.Client,
    existing_table: Optional[Table],
    breaker: CircuitBreaker,
) -> None:
    if not EXTRACT_ENABLED:
        click.echo("Phase 2: disabled (JUDGMENTS_EXTRACT_ENABLED=0) — skipping.")
        return
    if existing_table is None:
        # Fresh DB with no Phase 1 rows yet — nothing to enrich.
        return
    if breaker.is_open:
        click.echo("Phase 2: circuit breaker open from Phase 1 — skipping.")
        return

    # Sentinel: zeeker will re-invoke fetch_data after module reload to
    # build fragment context. Skip Phase 2 on the second call so we don't
    # double the enrichment budget within one build.
    if os.environ.get(_PHASE2_SENTINEL_ENV) == str(os.getpid()):
        click.echo("Phase 2: already ran this build (fragment-context pass) — skipping.")
        return
    os.environ[_PHASE2_SENTINEL_ENV] = str(os.getpid())

    _ensure_phase2_columns(existing_table)

    state = load_extraction_state()
    now = datetime.now()

    # Query all rows with NULL content_text, ordered most recent first
    # (fresh judgments matter more to users; backfill oldest last).
    candidates: List[Dict[str, Any]] = list(
        existing_table.rows_where("content_text IS NULL", order_by="decision_date DESC")
    )
    remaining_total = len(candidates)
    if remaining_total == 0:
        click.echo("Phase 2: no rows need enrichment.")
        return

    click.echo(
        f"Phase 2: enriching up to {EXTRACT_MAX_PER_RUN} / {remaining_total} "
        f"remaining (EXTRACT_MAX_RETRIES={EXTRACT_MAX_RETRIES}, "
        f"EXTRACT_RETRY_AFTER={EXTRACT_RETRY_AFTER}s)"
    )

    successes = 0
    structurally_empty = 0
    transient_failures = 0
    skipped_quarantined = 0
    attempted = 0

    try:
        for row in candidates:
            if attempted >= EXTRACT_MAX_PER_RUN:
                break
            if breaker.is_open:
                click.echo("Phase 2: circuit breaker tripped — stopping early.")
                break
            jid = row["id"]
            if _is_quarantined(state, jid, now):
                skipped_quarantined += 1
                continue
            attempted += 1
            label = f"{row.get('court') or '?'}] {row.get('citation') or jid}"
            try:
                status, detail = _enrich_row(client, row, existing_table, breaker)
            except Exception as exc:  # defensive — should be rare
                transient_failures += 1
                _record_extraction_failure(state, jid, exc)
                click.echo(
                    f"  {attempted}/{EXTRACT_MAX_PER_RUN} [{label} → UNEXPECTED: {exc}", err=True
                )
                save_extraction_state(state)
                continue

            if status == "ok":
                successes += 1
                _clear_extraction_failure(state, jid)
                click.echo(f"  {attempted}/{EXTRACT_MAX_PER_RUN} [{label} → {detail}")
            elif status == "empty":
                structurally_empty += 1
                _clear_extraction_failure(state, jid)  # structural, not transient
                click.echo(f"  {attempted}/{EXTRACT_MAX_PER_RUN} [{label} → empty ({detail})")
            else:  # http_error
                transient_failures += 1
                fake = RuntimeError(detail or "http error")
                _record_extraction_failure(state, jid, fake)
                click.echo(
                    f"  {attempted}/{EXTRACT_MAX_PER_RUN} [{label} → fetch failed: {detail}",
                    err=True,
                )
            save_extraction_state(state)
    except KeyboardInterrupt:
        click.echo("Phase 2 interrupted — state saved", err=True)
        save_extraction_state(state)
        raise

    click.echo(
        f"Phase 2 complete: {successes} extracted, {structurally_empty} empty-body, "
        f"{transient_failures} fetch-failed, {skipped_quarantined} quarantined; "
        f"{remaining_total - successes - structurally_empty} NULL-content rows "
        f"remain in DB."
    )


# =============================================================================
# PHASE 3 — AI SUMMARIES (helpers)
# =============================================================================
# Summary checkpoint mirrors the Phase 2 shape so the helper code stays
# symmetric. Duplication (load/save/record/clear/is_quarantined) is
# preferred over a generic abstraction because the two phases already
# diverge on a few small things (error sources, quarantine messages,
# column names) and the extra shared-code ceremony outweighs the savings.


def load_summary_state() -> dict:
    if SUMMARY_CHECKPOINT_FILE.exists():
        try:
            return json.loads(SUMMARY_CHECKPOINT_FILE.read_text())
        except json.JSONDecodeError:
            click.echo("Summary checkpoint was corrupt — starting fresh.", err=True)
    return {"failures": {}}


def save_summary_state(state: dict) -> None:
    tmp = SUMMARY_CHECKPOINT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, SUMMARY_CHECKPOINT_FILE)


def _record_summary_failure(state: dict, jid: str, err: Exception) -> None:
    failures = state.setdefault("failures", {})
    entry = failures.setdefault(jid, {"count": 0, "last_error": "", "last_attempt": ""})
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["last_error"] = f"{type(err).__name__}: {err}"[:500]
    entry["last_attempt"] = datetime.now().isoformat(timespec="seconds")


def _clear_summary_failure(state: dict, jid: str) -> None:
    failures = state.get("failures", {})
    if jid in failures:
        del failures[jid]


def _is_summary_quarantined(state: dict, jid: str, now: datetime) -> bool:
    entry = state.get("failures", {}).get(jid)
    if entry is None:
        return False
    if int(entry.get("count", 0)) < SUMMARY_MAX_RETRIES:
        return False
    last_attempt_str = entry.get("last_attempt") or ""
    try:
        last_attempt = datetime.fromisoformat(last_attempt_str)
    except ValueError:
        return False
    return (now - last_attempt).total_seconds() < SUMMARY_RETRY_AFTER


def _ensure_phase3_columns(table: Table) -> None:
    existing = set(table.columns_dict)
    for name, col_type in PHASE3_ADDED_COLUMNS.items():
        if name not in existing:
            table.add_column(name, col_type)


def _summarise_row(
    row: Dict[str, Any],
    existing_table: Table,
    client,
    model: str,
    *,
    endpoint: str = "",
) -> Tuple[str, Optional[str]]:
    """Generate + persist a summary for one row. Returns ``(status, detail)``.

    Status is one of: ``ok``, ``cached`` (re-used disk cache from a
    previous crashed-mid-run), or ``error`` (LLM failure; caller bumps
    failure counter).
    """
    jid = row["id"]

    # Fast path: previous run wrote the cache but crashed before the DB
    # UPDATE. Push to DB without calling the LLM again.
    cached = summary_cache.read_summary(jid)
    if cached is not None and cached.get("summary"):
        _update_row(
            existing_table,
            jid,
            {
                "summary": cached["summary"],
                "summary_generated_at": cached.get("generated_at")
                or datetime.now().isoformat(timespec="seconds"),
            },
        )
        existing_table.db.conn.commit()
        return "cached", None

    fragments = (
        list(
            existing_table.db[FRAGMENTS_TABLE_NAME].rows_where(
                "judgment_id = ?", [jid], order_by="ordinal"
            )
        )
        if FRAGMENTS_TABLE_NAME in existing_table.db.table_names()
        else []
    )

    try:
        summary_text = summarization.rolling_summarise(row, fragments, model, client)
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"

    if not summary_text:
        return "error", "empty response"

    now_iso = datetime.now().isoformat(timespec="seconds")
    endpoint = endpoint or os.environ.get("LLM_BASE_URL", "")
    summary_cache.write_summary_atomic(
        jid,
        {
            "judgment_id": jid,
            "generated_at": now_iso,
            "model": model,
            "endpoint": endpoint,
            "frags_total": len(fragments),
            "summary": summary_text,
        },
    )
    _update_row(
        existing_table,
        jid,
        {"summary": summary_text, "summary_generated_at": now_iso},
    )
    existing_table.db.conn.commit()
    return "ok", f"{len(summary_text)} chars"


def _run_phase3(existing_table: Optional[Table]) -> None:
    if not SUMMARY_ENABLED:
        click.echo("Phase 3: disabled (JUDGMENTS_SUMMARY_ENABLED=0) — skipping.")
        return
    if existing_table is None:
        return
    if os.environ.get(_PHASE3_SENTINEL_ENV) == str(os.getpid()):
        click.echo("Phase 3: already ran this build (fragment-context pass) — skipping.")
        return
    os.environ[_PHASE3_SENTINEL_ENV] = str(os.getpid())

    client = summarization.make_client()
    if client is None:
        click.echo("Phase 3: LLM not configured (LLM_BASE_URL unset) — skipping.")
        return

    client_alt = summarization.make_client_alt()
    _ensure_phase3_columns(existing_table)
    model = summarization.resolve_model()
    model_alt = summarization.resolve_model_alt()

    state = load_summary_state()
    now = datetime.now()

    all_candidates: List[Dict[str, Any]] = list(
        existing_table.rows_where(
            "has_content = 1 AND summary IS NULL",
            order_by="decision_date DESC",
        )
    )
    remaining_total = len(all_candidates)
    if remaining_total == 0:
        click.echo("Phase 3: no rows need summarisation.")
        return

    # Priority queue: quarantined docs (fail_count >= SUMMARY_MAX_RETRIES) fill first,
    # then date-ordered fresh docs. Quarantined docs bypass the TTL check — they're
    # here specifically to be retried on the (potentially updated) endpoint.
    quarantined_ids = {
        jid for jid, info in state.get("failures", {}).items()
        if info.get("count", 0) >= SUMMARY_MAX_RETRIES
    }
    priority_docs = [r for r in all_candidates if r["id"] in quarantined_ids]
    fresh_docs = [r for r in all_candidates if r["id"] not in quarantined_ids]
    candidates = priority_docs + fresh_docs

    alt_label = f", alt_model={model_alt}" if model_alt != model else ""
    click.echo(
        f"Phase 3: summarising up to {SUMMARY_MAX_PER_RUN} / {remaining_total} "
        f"remaining (model={model}{alt_label}, max_chars={SUMMARY_MAX_INPUT_CHARS}, "
        f"priority={len(priority_docs)})"
    )
    _phase3_log({"event": "start", "ts": datetime.now().isoformat(timespec="seconds"),
                 "remaining": remaining_total, "model": model, "model_alt": model_alt,
                 "priority_quarantined": len(priority_docs)})

    successes = 0
    cached_hits = 0
    failures = 0
    skipped_quarantined = 0
    attempted = 0

    try:
        for row in candidates:
            if attempted >= SUMMARY_MAX_PER_RUN:
                break
            jid = row["id"]
            fail_count = state.get("failures", {}).get(jid, {}).get("count", 0)
            # Quarantined docs (fail_count >= SUMMARY_MAX_RETRIES) bypass TTL —
            # they were promoted to priority slots to be retried. Fresh docs still
            # respect TTL to avoid hammering on transient failures.
            if fail_count < SUMMARY_MAX_RETRIES and _is_summary_quarantined(state, jid, now):
                skipped_quarantined += 1
                continue
            attempted += 1
            is_priority = jid in quarantined_ids
            use_model = model_alt if is_priority else model
            use_client = client_alt if (is_priority and client_alt is not None) else client
            endpoint = os.environ.get("LLM_BASE_URL_2" if (is_priority and client_alt is not None) else "LLM_BASE_URL", "")
            label = f"{row.get('court') or '?'}] {row.get('citation') or jid}"
            priority_tag = " [alt-model]" if is_priority else ""
            try:
                status, detail = _summarise_row(row, existing_table, use_client, use_model, endpoint=endpoint)
            except Exception as exc:  # defensive
                failures += 1
                _record_summary_failure(state, jid, exc)
                click.echo(
                    f"  {attempted}/{SUMMARY_MAX_PER_RUN} [{label}{priority_tag} → UNEXPECTED: {exc}",
                    err=True,
                )
                _phase3_log({
                    "event": "attempt", "ts": datetime.now().isoformat(timespec="seconds"),
                    "n": attempted, "id": jid,
                    "citation": row.get("citation"), "court": row.get("court"),
                    "status": "error", "detail": f"UNEXPECTED: {exc}",
                    "model": use_model, "alt": is_priority,
                })
                save_summary_state(state)
                continue

            if status == "ok":
                successes += 1
                _clear_summary_failure(state, jid)
                click.echo(f"  {attempted}/{SUMMARY_MAX_PER_RUN} [{label}{priority_tag} → {detail}")
            elif status == "cached":
                cached_hits += 1
                _clear_summary_failure(state, jid)
                click.echo(f"  {attempted}/{SUMMARY_MAX_PER_RUN} [{label}{priority_tag} → cached")
            else:  # error
                failures += 1
                fake = RuntimeError(detail or "llm error")
                _record_summary_failure(state, jid, fake)
                click.echo(
                    f"  {attempted}/{SUMMARY_MAX_PER_RUN} [{label}{priority_tag} → llm failed: {detail}",
                    err=True,
                )
            _phase3_log({
                "event": "attempt", "ts": datetime.now().isoformat(timespec="seconds"),
                "n": attempted, "id": jid,
                "citation": row.get("citation"), "court": row.get("court"),
                "status": status, "detail": detail,
                "model": use_model, "alt": is_priority,
            })
            save_summary_state(state)
    except KeyboardInterrupt:
        click.echo("Phase 3 interrupted — state saved", err=True)
        save_summary_state(state)
        raise

    save_summary_state(state)
    click.echo(
        f"Phase 3 complete: {successes} summarised, {cached_hits} from cache, "
        f"{failures} failed, {skipped_quarantined} quarantined; "
        f"{remaining_total - successes - cached_hits} NULL-summary rows remain."
    )


# =============================================================================
# PHASE 1 — DISCOVERY
# =============================================================================
def fetch_data(existing_table: Optional[Table]) -> List[Dict[str, Any]]:
    """Discover judgment catalog records from listing pages.

    Respects ``MAX_PAGES_PER_RUN`` (batch limit) and
    ``INCREMENTAL_STOP_THRESHOLD`` (steady-state early exit).
    State persisted to ``checkpoint_judgments_discovery.json``.
    """
    global _FETCH_CACHE
    if _FETCH_CACHE is not None:
        click.echo(
            f"fetch_data: returning {len(_FETCH_CACHE)} cached records "
            f"(already fetched this process)"
        )
        return _FETCH_CACHE

    existing_ids: set[str] = set()
    if existing_table is not None:
        existing_ids = {row["id"] for row in existing_table.rows}
    click.echo(f"Existing records in database: {len(existing_ids)}")

    checkpoint = load_checkpoint()
    start_page = checkpoint.get("last_page", 0) + 1
    # Drop any staged items that were persisted in a prior run's build.
    staged: List[Dict[str, Any]] = [
        r for r in checkpoint.get("items_collected", []) if r["id"] not in existing_ids
    ]
    total_pages: Optional[int] = checkpoint.get("total_pages")

    breaker = CircuitBreaker()
    consecutive_known = 0
    pages_this_run = 0
    page = start_page
    url = urljoin(BASE_URL, INDEX_PATH)
    exhausted = False

    def _snapshot(last_completed_page: int) -> dict:
        return {
            "last_page": last_completed_page,
            "items_collected": staged,
            "total_pages": total_pages,
        }

    with create_client() as client:
        try:
            while True:
                if MAX_PAGES_PER_RUN > 0 and pages_this_run >= MAX_PAGES_PER_RUN:
                    click.echo(
                        f"Batch limit reached ({pages_this_run} pages). "
                        f"{len(staged)} new records staged this run. "
                        f"Re-run `uv run zeeker build judgments` to continue from page {page}."
                    )
                    save_checkpoint(_snapshot(page - 1))
                    break

                breaker.wait_if_needed()
                params = {**INDEX_PARAMS, "CurrentPage": str(page)}
                progress = f"({len(staged)} new staged, {pages_this_run}/{MAX_PAGES_PER_RUN} batch)"
                if total_pages:
                    click.echo(f"Fetching page {page}/{total_pages} {progress}")
                else:
                    click.echo(f"Fetching page {page} {progress}")

                try:
                    response = _fetch_listing(client, url, params)
                except Exception as exc:
                    breaker.record_failure()
                    click.echo(f"  → Fetch failed after retries: {exc}", err=True)
                    save_checkpoint(_snapshot(page - 1))
                    click.echo("Aborting run — checkpoint saved. Try again later.")
                    break

                if response.status_code == 404:
                    click.echo("Reached end of pagination (404)")
                    exhausted = True
                    break
                if response.status_code >= 400:
                    breaker.record_failure()
                    click.echo(
                        f"  → HTTP {response.status_code} on page {page}; aborting.",
                        err=True,
                    )
                    save_checkpoint(_snapshot(page - 1))
                    break

                breaker.record_success()
                html = response.text

                if total_pages is None:
                    detected = detect_total_pages(html)
                    if detected:
                        total_pages = detected
                        click.echo(f"  → Total pages detected: {total_pages}")

                items = parse_listing_page(html)
                if not items:
                    click.echo("No items on this page — end of results")
                    exhausted = True
                    break

                page_new = 0
                steady_state = False
                for item in items:
                    if item["id"] in existing_ids:
                        consecutive_known += 1
                        if consecutive_known >= INCREMENTAL_STOP_THRESHOLD:
                            click.echo(
                                f"Stopping: {consecutive_known} consecutive known IDs — "
                                f"steady-state mode."
                            )
                            # Steady-state runs must always start from page 1
                            # next time to catch newly-published judgments,
                            # so drop any stale checkpoint from earlier batch
                            # crawls. Fall through to Phase 2 — daily runs
                            # need it to drain the enrichment backlog.
                            clear_checkpoint()
                            steady_state = True
                            break
                    else:
                        consecutive_known = 0
                        staged.append(item)
                        existing_ids.add(item["id"])
                        page_new += 1

                click.echo(f"  → {page_new}/{len(items)} new on page {page}")
                if not steady_state:
                    # Steady-state already cleared the checkpoint above; don't
                    # immediately recreate it with the current page or the
                    # next run resumes mid-archive instead of from page 1.
                    save_checkpoint(_snapshot(page))

                if steady_state:
                    break

                if total_pages and page >= total_pages:
                    click.echo("Reached last page of archive")
                    exhausted = True
                    break

                page += 1
                pages_this_run += 1
                polite_sleep()
        except KeyboardInterrupt:
            click.echo("Interrupted — saving checkpoint", err=True)
            save_checkpoint(_snapshot(page - 1))
            raise

        if exhausted:
            clear_checkpoint()

        click.echo(f"Discovery run complete: {len(staged)} new records, {breaker.summary()}")

        # Phase 2 runs inside the same client context so it reuses the
        # connection pool. Uses the breaker state from Phase 1 — if the
        # source was flaky during discovery we skip enrichment entirely.
        _run_phase2(client, existing_table, breaker)

    # Phase 3 lives outside the httpx client block — the LLM call goes
    # through its own OpenAI-compatible client, not the eLitigation
    # connection pool.
    _run_phase3(existing_table)

    _FETCH_CACHE = staged
    return staged


def fetch_fragments_data(
    existing_fragments_table: Optional[Table],
    main_data_context: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """No-op: Phase 2 writes fragments directly during ``fetch_data``.

    Zeeker skips the fragment-fetch pipeline entirely when fetch_data
    returns no new main-table rows — which is the steady-state case for
    this project once discovery is complete. To stay robust against
    that, we insert fragments transactionally with the main-row UPDATE
    in ``_enrich_row`` rather than relying on zeeker to call us here.

    Returning ``[]`` keeps ``fragments = true`` in zeeker.toml valid
    without duplicating any work.
    """
    return []


def transform_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return raw_data


def transform_fragments_data(raw_fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return raw_fragments
