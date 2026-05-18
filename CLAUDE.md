# CLAUDE.md - Zeeker-Judgements Project Development Guide

This file provides Claude Code with project-specific context and guidance for developing this project.

## Project Overview

**Project Name:** zeeker-judgements
**Database:** zeeker-judgements.db
**Purpose:** Database project for zeeker-judgements data management

## Development Environment

This project uses **uv** for dependency management with an isolated virtual environment:

- `pyproject.toml` - Project dependencies and metadata
- `.venv/` - Isolated virtual environment (auto-created)
- All commands should be run with `uv run` prefix

### Dependency Management
- **Add dependencies:** `uv add package_name` (e.g., `uv add requests pandas`)
- **Install dependencies:** `uv sync` (automatically creates .venv if needed)
- **Common packages:** requests, beautifulsoup4, pandas, lxml, pdfplumber, openpyxl

### Environment Variables
Zeeker automatically loads `.env` files when running build, deploy, and asset commands:

- **Create `.env` file:** Store sensitive credentials and configuration
- **Auto-loaded:** Environment variables are available in your resources during `zeeker build`
- **S3 deployment:** Required for `zeeker deploy` and `zeeker assets deploy`

**Example `.env` file:**
```
# S3 deployment credentials
S3_BUCKET=my-datasette-bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_ENDPOINT_URL=https://s3.amazonaws.com

# API keys for data resources
JINA_API_TOKEN=your_jina_token
OPENAI_API_KEY=your_openai_key
```

**Usage in resources:**
```python
import os

def fetch_data(existing_table):
    api_key = os.getenv("MY_API_KEY")  # Loaded from .env automatically
    # ... rest of your code
```

## Development Commands

### Quick Commands
- `uv run zeeker add RESOURCE_NAME` - Add new resource to this project
- `uv run zeeker add RESOURCE_NAME --fragments` - Add resource with document fragments support
- `uv run zeeker build` - Build database from all resources in this project
- `uv run zeeker deploy` - Deploy this project's database to S3

### Code Formatting
- `uv run black .` - Format code with black
- `uv run ruff check .` - Lint code with ruff
- `uv run ruff check --fix .` - Auto-fix ruff issues

### Testing This Project
- `uv run pytest` - Run tests (if added to project)
- Check generated `zeeker-judgements.db` after build
- Verify metadata.json structure

### Working with Dependencies
When implementing resources that need external libraries:
1. **First add the dependency:** `uv add library_name`
2. **Then use in your resource:** `import library_name` in `resources/resource_name.py`
3. **Build works automatically:** `uv run zeeker build` uses the isolated environment

## Resources in This Project

### `judgments` Resource
- **Description:** Singapore court judgments from eLitigation with paragraph-level search
- **File:** `resources/judgments.py` (+ `extraction.py`, `extraction_cache.py`)
- **Facets:** court, decision_date, has_content
- **Default Sort:** decision_date desc
- **Page Size:** 25
- **Type:** Main table + `judgments_fragments` for paragraph-level search (populated by Phase 2)
- **Schema:** see `zeeker.toml` `[resource.judgments.columns]` and `[resource.judgments_fragments.columns]`

## Project-Specific Notes

### Data Source
- **URL:** https://www.elitigation.sg/gd/
- **License:** `© Government of Singapore` (per the site footer).
- **Scale:** ~10,588 judgments across ~1,059 listing pages (10/page). Covers
  all Singapore courts back to 2000 — SGCA, SGHC, SGHCA, SGHCF, SGHCR, SGDC,
  SGFC, SGMC — plus tribunals like SGCDT (Community Disputes) and SGSCT
  (Small Claims). The `court` regex is permissive: any `SG[A-Z]+` token in
  the URL path is captured verbatim.
- **Cadence:** Tier 4 one-shot batch crawl for the initial archive;
  transitions to Tier 1 daily incremental once the archive is complete.
- **Update frequency:** New judgments appear regularly (typically several
  per week day). Sort-by-date-desc + `INCREMENTAL_STOP_THRESHOLD=5` makes
  daily catch-up cheap.

### Roadmap (phased)
- **Phase 1 (DONE):** Discovery crawler — scrapes listing pages, persists
  catalog metadata to `judgments`. Content columns stay NULL until Phase 2
  backfill runs.
- **Phase 2 (DONE):** HTML content extraction from the detail page at
  `source_url` (NOT the PDF — see "Phase 2 design" below). Populates
  `content_text` + `court_summary` on the main row and emits per-paragraph
  rows into `judgments_fragments`. Runs in the **same** `fetch_data()`
  invocation as Phase 1 (after discovery, before return), so there is one
  resource, one build command, two phases per run. Default cap is 15 docs
  per build — with a 10,588-doc backlog, expect hundreds of runs to drain.
  Fragments are inserted inline from `fetch_data` (not via
  `fetch_fragments_data`, which is a no-op stub); see the "Implementation
  notes" subsection below for the why.
- **Phase 3 (DONE):** AI summaries — populates `summary` with a single
  ≤100-word paragraph via an OpenAI-compatible LLM endpoint. Runs in the
  **same** `fetch_data()` invocation after Phase 2, so one
  `zeeker build judgments` drains all three backlogs on a single pass.
  Skips gracefully when `LLM_BASE_URL` is unset (local-first — no cloud
  dependency required). Default cap is 15 docs per build; see "Phase 3
  implementation notes" below.
- **Deployment (deferred):** No S3 workflow wired up yet. The
  auto-generated `.github/workflows/deploy.yml` is inert until secrets are
  configured.

### Phase 2 design notes (HTML over PDF)

The initial plan was "PDF → Docling Serve → markdown". After inspecting the
detail pages we confirmed a much simpler path: the HTML served at
`source_url` is the same court-approved mobile/web conversion, with stable
semantic CSS classes across 20+ years and every court tier (verified SGCA,
SGHC, SGHCF, SGMC, SGSCT; 2005 → 2026). We keep `pdf_url` in the catalog
so users can reach the authoritative PDF, but extraction uses BeautifulSoup
on HTML — no Docling infrastructure required.

**Source containers on each detail page:**
- `div#divJudgement` — full judgment body (required, always present)
- `div#divCaseSummary` — court-authored summary (often empty; capture only
  when non-empty)
- `div#divHeadMessage` — standard disclaimer:
  *"This judgment text has undergone conversion so that it is mobile and
  web-friendly. This may have created formatting or a[...]"* — surface this
  in the UI/README so readers know the PDF is authoritative.

**`Judg-*` class map (stable across years/courts):** classification is
**prefix-based** in code (`resources/extraction.py`) rather than a static
whitelist, because eLitigation emits era-specific variants that aren't
worth enumerating (observed: `Judg-1-firstpara`, `Judg-List-1` bare,
`Judg-Quote-List`, `Judg-Hearing-Date`). Any class starting with `Judg-`
becomes a fragment unless it's in `EXCLUDED_CLASSES` (currently just
`Judg-EOF`, the end-of-document ornament).

| Class | Role |
|---|---|
| `Judg-1`, `Judg-1-firstpara` | Top-level numbered paragraph (arabic number is first token, e.g. `"1 The Claimant..."`). The **only** classes where `paragraph_number` is parseable. |
| `Judg-2` | Sub-paragraph — uses alpha enumeration `(a)`, `(b)` in the text; no parseable paragraph number. |
| `Judg-3` | Sub-sub-paragraph — roman numerals `(i)`, `(ii)`. |
| `Judg-Heading-1` … `Judg-Heading-5` | Nested section headings. The most recent heading above a fragment becomes its `section_heading`. |
| `Judg-Quote-0/1/2`, `Judg-Quote-List`, `Judg-QuoteList-*` | Block quotations. |
| `Judg-List-1`, `Judg-List-1-No`, `Judg-List-1-Item` | Numbered-list entries. |
| `Judg-Author`, `Judg-Date-Reserved`, `Judg-Hearing-Date`, `Judg-Sign` | Front/back-matter. |
| `Judg-Lawyers` | Counsel block. |
| `Judg-EOF` | Visual end marker — **excluded**, never produces a fragment. |

**Backward-attachment anchors:** standalone tables/figures between
paragraphs attach to the most recent Judg-1, Judg-1-firstpara, Judg-2,
or Judg-3 fragment (not just Judg-1/2).

**Walk strategy:** detail pages wrap content in a custom `<content>`
element (and sometimes further wrappers) inside `div#divJudgement`, so
Judg-* elements are at depth 3+, not direct children. The walker in
`extraction.py` descends through non-Judg-* wrappers and yields Judg-*
elements as leaves.

**Paragraph-number parsing quirk:** the separator between the number and
the paragraph text is a non-breaking space `\xa0` on older docs and an
em-space `\u2003` on newer ones. Use `re.match(r"(\d+)[\s\xa0\u2003]+", text)`.

**Footnotes:** inline references are `<sup>` tags. Two markup variants in
the wild:
- Newer (2020+): `<sup><button data-target="#fn-<uuid>" data-toggle="modal">N</button></sup>`
  — Bootstrap modal trigger; href lives in `data-target`.
- Older: `<sup><a href="#fn1">N</a></sup>` — plain anchor.
Both variants resolve to `div[id^="fn"]` elsewhere in the document (ids
may be `fn1`, `fn2`, … or UUID-form `fn-041fe0fc-...`). Bare `<sup>`
tags without a resolvable link (e.g. `<sup>[note: 1]</sup>` in ~2015-era
docs) are kept in the paragraph text but not captured into
`footnote_text`. Volume varies — 0 in older judgments, 100+ in long
family-court ones.

**Images and tables — attachment rule:** judgments embed tables and images
both *inside* numbered paragraphs (e.g. a screenshot referenced mid-text)
and *between* numbered paragraphs as standalone exhibits. Phase 2 must
handle both placements consistently:

- **Tables** (`<table>`) — convert cells to a pipe-separated text
  representation so the content is searchable via FTS. Append the text to
  the **parent paragraph's** `content_text` (backward attachment: attach
  to the most recent `Judg-1`/`Judg-2`). Flag with `has_table = true`.
  If a table appears before any numbered paragraph in a section, attach
  forward to the next one. Normalise to a consistent separator (e.g.
  `\n\n---table---\n`).
- **Images** (`<img>`) — can be remote URLs OR base64 `data:` URIs (we've
  seen embedded screenshots as base64 in at least one recent judgment).
  Store the `alt` text when present, otherwise derive a placeholder
  (`"[Figure: screenshot, 1024x768]"` etc.) so FTS can hit on captions /
  alt text. Flag with `has_figure = true`; persist the image URL or a
  stable identifier on the fragment (`figure_src`) without downloading
  binary content.
- **Inline vs block**: if the element is nested inside a `Judg-1` paragraph,
  just append the text/placeholder to that fragment's `content_text`. If it
  sits at the top level of `div#divJudgement` between paragraphs, apply the
  backward-attachment rule. Never create a separate fragment for a figure
  or table — they belong to their parent paragraph for search/display
  purposes.

**Extension to the fragment schema for Phase 2:**
- `has_table` (bool) — paragraph contains or absorbed a table
- `has_figure` (bool) — paragraph contains or absorbed an image
- `figure_src` (text, JSON array) — list of image URLs / data URI hashes
- `figure_descriptions` (text, JSON array) — alt-text per figure, aligned
  to `figure_src` by position

These are useful Datasette facets later (filter to paragraphs containing
tables, e.g. "show me every judgment that has a damages schedule").

### Phase 2 implementation notes (design deltas from the plan)

A handful of choices diverged from the original "plan first, build later"
sketch once we hit real code. Future-you should know about these:

1. **Fragments are inserted inline from `fetch_data`, not via
   `fetch_fragments_data`.** Zeeker skips the fragment pipeline entirely
   when `fetch_data` returns no new main-table rows — which is the
   steady-state case every daily build (Phase 1 finds nothing new; Phase 2
   still wants to drain the enrichment backlog). So `_enrich_row` now
   calls `_insert_fragments(db, extracted.fragments)` directly, same
   transaction as the main-row `UPDATE`. `fetch_fragments_data` is a
   `return []` no-op stub left in place to satisfy `fragments = true` in
   `zeeker.toml`.

2. **Direct SQL `UPDATE … WHERE id = ?` rather than
   `sqlite_utils.Table.update()`.** Zeeker creates the `judgments` table
   without a declared primary key (uses implicit rowid), so
   `sqlite_utils`' `update(id_value, …)` raises `NotFoundError` — it
   treats the arg as a rowid, not as our `id` column. We manage columns
   (`_ensure_phase2_columns`) and updates (`_update_row`) in raw SQL.

3. **`judgments_fragments` is created with an explicit schema.** Because
   `fetch_fragments_data` is a stub, Zeeker's first-row schema inference
   never fires. `_ensure_fragments_table` creates the table with
   `pk="id"` up front (see `FRAGMENT_COLUMNS` in `resources/judgments.py`).

4. **Env-var sentinel guards against zeeker's module reload.** When
   `fragments = true`, zeeker re-imports the resource module and calls
   `fetch_data` a second time to build `main_data_context`. Module-level
   variables don't survive the reload, so `_JUDGMENTS_PHASE2_RAN_PID`
   lives in `os.environ` (process-scoped, survives the reload, naturally
   cleared across builds). Second call sees the matching PID and skips
   Phase 2 entirely. Without this, every build would enrich
   `2 × EXTRACT_MAX_PER_RUN` docs instead of the budgeted number.

5. **Two-layer disk cache under `.cache/`** (gitignored):
   - `.cache/judgments_html/{id}.html.gz` — raw detail-page HTML, gzipped.
     Source of truth for re-extraction. If parsing rules change, delete
     `.cache/judgments_extractions/` and re-run — no server traffic.
   - `.cache/judgments_extractions/{id}.json` — parsed extraction output.
     `_enrich_row` short-circuits on this: if the extraction JSON exists
     (e.g. from a previous run that crashed after parsing but before the
     DB write), it pushes straight to the DB without re-fetching or
     re-parsing. Atomic writes (`tmp` + `os.replace`) prevent half-
     written files on crash.

6. **Failure quarantine.** `checkpoint_judgments_extraction.json` tracks
   per-judgment failure count, last error, and last-attempt timestamp.
   After `EXTRACT_MAX_RETRIES` failures a doc is quarantined for
   `EXTRACT_RETRY_AFTER` seconds before the next attempt. This prevents
   a single broken page from burning the whole daily budget.

7. **Sibling-module import hack in `resources/judgments.py`.** Zeeker
   loads resource files via `importlib.util.spec_from_file_location`,
   which bypasses package imports — `from resources import extraction`
   fails at build time. We prepend `Path(__file__).parent` to `sys.path`
   before importing sibling modules.

**Env vars (Phase 2):** set `JUDGMENTS_EXTRACT_ENABLED=0` to skip Phase 2
entirely (useful during the initial Phase-1 archive crawl). The rest are
in the knob table below.

### Phase 3 implementation notes

Phase 3 mirrors Phase 2's shape: batch-limited backfill, checkpointed
failure tracking, disk cache for crash recovery, env-var sentinel to
survive zeeker's module reload. The deltas worth knowing:

1. **The skill's naive `text[:4000]` truncation throws away the
   holding.** The zeeker-source-creator skill's suggested summariser
   template truncates the content blindly. The holding of a judgment
   usually lives at the end, so we replaced truncation with
   fragment-weighted sampling in `resources/summarization.py`:
   - **Always keep:** `court_summary` (if non-empty), every
     `Judg-Heading-*`, the first numbered paragraph (smallest
     `paragraph_number`), and the last three numbered paragraphs
     (largest).
   - **Scored remainder:** `+2` for `has_footnotes`, `+3` for a
     dispositive heading (conclusion/decision/holding/disposition/
     order), `+1.5` for analysis/issue/reasoning, `+0.5` for
     `has_table`, plus a capped length bonus (up to `+0.5`).
   - Pack highest-scored fragments into whatever char budget remains
     after the always-keep set, then re-emit everything in document
     order. Fallback for rows with zero fragments: send
     `content_text[:SUMMARY_MAX_INPUT_CHARS]`.

2. **Local-first by default.** If `LLM_BASE_URL` is unset,
   `summarization.make_client()` returns `None` and `_run_phase3` logs
   "LLM not configured — skipping" without importing openai. `LLM_API_KEY`
   defaults to `"not-needed"` so Ollama / vLLM work out of the box.

3. **Direct `UPDATE` + ad-hoc column add (same pattern as Phase 2).**
   `_ensure_phase3_columns` only adds `summary_generated_at`; the
   `summary` column is created during Phase 1 (initialised to NULL in
   `parse_listing_page`). Writes go through `_update_row` because zeeker
   didn't declare a primary key on `judgments`.

4. **Env-var sentinel (`_JUDGMENTS_PHASE3_RAN_PID`) guards the module
   reload.** Identical mechanism to Phase 2 — without it every build
   would summarise `2 × SUMMARY_MAX_PER_RUN` docs.

5. **Cache under `.cache/judgments_summaries/{id}.json`.** Survives
   crashes after the LLM call but before the DB `UPDATE`. If parsing/
   prompting rules change, delete the JSONs and reset the rows' `summary`
   column to NULL to force regeneration. Atomic writes via tmp +
   `os.replace`; corrupt JSON is quarantined (renamed aside) so the next
   run regenerates.

6. **Query candidates via `has_content = 1 AND summary IS NULL`.** Don't
   summarise rows that Phase 2 hasn't enriched yet (no body text = no
   useful input). The real progress signal is exactly this SQL, same as
   Phase 2's `content_text IS NULL` signal.

7. **Phase 3 lives outside the `with create_client() as client:` block**
   in `fetch_data`. The LLM call goes through its own OpenAI-compatible
   client, not the eLitigation connection pool — keeping them separate
   means a flaky Phase 2 source doesn't break summarisation and vice
   versa.

### Environment variables
Phase 1 + 2 need **none** for correctness — both are public-HTML crawls
reusing the same `httpx.Client`. Phase 3 needs `LLM_BASE_URL` to do
anything; without it Phase 3 skips gracefully. See `.env.example`.

The resource has a handful of operational knobs that default via env
vars (handy for smoke tests — no code edits required):

| Env var | Default | What it controls |
|---|---|---|
| `JUDGMENTS_MAX_PAGES_PER_RUN` | `50` | Phase 1: batch cap on listing pages per invocation. Set to `2` for smoke tests. |
| `JUDGMENTS_INCREMENTAL_STOP` | `5` | Phase 1: consecutive already-known IDs before early exit. |
| `JUDGMENTS_DELAY_BASE` | `1.5` | Phase 1: base sleep (s) between listing-page fetches. |
| `JUDGMENTS_DELAY_JITTER` | `0.5` | Phase 1: +/- jitter added to the base sleep. |
| `JUDGMENTS_EXTRACT_ENABLED` | `1` | Phase 2: set to `0` to run discovery only (skip enrichment). |
| `JUDGMENTS_EXTRACT_MAX_PER_RUN` | `15` | Phase 2: max docs enriched per `zeeker build`. |
| `JUDGMENTS_EXTRACT_MAX_RETRIES` | `3` | Phase 2: failures before a doc is quarantined. |
| `JUDGMENTS_EXTRACT_RETRY_AFTER` | `86400` | Phase 2: quarantine TTL in seconds (default 24h). |
| `JUDGMENTS_EXTRACT_DELAY_BASE` | `1.5` | Phase 2: base sleep (s) between detail-page fetches. |
| `JUDGMENTS_EXTRACT_DELAY_JITTER` | `0.5` | Phase 2: +/- jitter on the sleep. |
| `JUDGMENTS_SUMMARY_ENABLED` | `1` | Phase 3: set to `0` to skip summarisation. |
| `JUDGMENTS_SUMMARY_MAX_PER_RUN` | `15` | Phase 3: max docs summarised per `zeeker build`. |
| `JUDGMENTS_SUMMARY_MAX_BATCHES` | `20` | Phase 3: max rolling-pass batches per doc (wider batch_size for large docs). |
| `JUDGMENTS_SUMMARY_MAX_RETRIES` | `3` | Phase 3: failures before a doc is quarantined. |
| `JUDGMENTS_SUMMARY_RETRY_AFTER` | `86400` | Phase 3: quarantine TTL in seconds (default 24h). |
| `JUDGMENTS_SUMMARY_MAX_INPUT_CHARS` | `32000` | Phase 3: char budget for composed LLM input (~8K tokens). |
| `LLM_BASE_URL` | *unset* | Phase 3: OpenAI-compatible endpoint. Unset → Phase 3 skips. |
| `LLM_API_KEY` | `not-needed` | Phase 3: optional for local servers; required for cloud. |
| `LLM_MODEL` | `llama3.1:8b` | Phase 3: any model the endpoint accepts. |
| `LLM_BASE_URL_2` | *unset* | Phase 3: alt endpoint for docs that hit the primary failure cap (count≥3). Unset → quarantine behaviour unchanged. |
| `LLM_API_KEY_2` | falls back to `LLM_API_KEY` | Phase 3: API key for the alt endpoint. |
| `LLM_MODEL_2` | same as `LLM_MODEL` | Phase 3: model name on the alt endpoint. |
| `JUDGMENTS_SUMMARY_MAX_BATCHES_ALT` | `5` | Phase 3: rolling-pass cap for alt-model calls. Fewer, wider batches (vs 20 for primary) — alt model handles more fragments per call. |
| `JUDGMENTS_SUMMARY_MAX_TOKENS_ALT` | `8192` | Phase 3: output token floor for alt-model calls. Raised above 4096 to handle wide-batch intermediate summaries and any thinking tokens the alt model uses. |
| `JUDGMENTS_SUMMARY_TEMPERATURE` | `0.0` | Phase 3: sampling temperature. 0.0 = deterministic extraction, no creativity. |
| `JUDGMENTS_SUMMARY_TOP_P` | `0.9` | Phase 3: nucleus sampling ceiling. Limits generation to top 90% probability mass. |
| `JUDGMENTS_SUMMARY_FREQUENCY_PENALTY` | `0.15` | Phase 3: reduces repetition of the same phrases across the summary. |
| `JUDGMENTS_SUMMARY_PRESENCE_PENALTY` | `0.1` | Phase 3: discourages restating the same concept in different wording. |
| `JUDGMENTS_SUMMARY_SEED` | `42` | Phase 3: fixed random seed. Re-run the same doc → identical output (for debugging). |
| `JUDGMENTS_SUMMARY_REPEAT_PENALTY` | `1.2` | Phase 3: Ollama-specific repetition suppression (via `extra_body`). |
| `JUDGMENTS_SUMMARY_TOP_K` | `20` | Phase 3: Ollama-specific token restriction (via `extra_body`). |

### Operational notes
- **Phase 1 checkpointing:** state lives in
  `checkpoint_judgments_discovery.json` (gitignored). Resumes mid-archive
  across many runs; cleared automatically when the archive is exhausted
  OR when the daily incremental stop fires.
- **Phase 2 checkpointing:** failure state in
  `checkpoint_judgments_extraction.json` (gitignored). The **real**
  progress signal for enrichment is the DB query
  `SELECT COUNT(*) FROM judgments WHERE content_text IS NULL` — whatever
  isn't enriched yet still has NULL content. The checkpoint only tracks
  what's been *tried and failed*.
- **Batch-crawl pacing (Phase 1):** at defaults, ~50 pages ≈ 75s sleep +
  fetch/parse. Each run lands ~500 new records; ~22 runs cover the full
  archive.
- **Enrichment pacing (Phase 2):** at defaults, ~15 docs ≈ 25s per run.
  With a 10,588-doc backlog expect ~700 runs to drain. Set
  `JUDGMENTS_EXTRACT_MAX_PER_RUN` higher if the source is cooperating,
  lower (or `0` via `JUDGMENTS_EXTRACT_ENABLED=0`) if it's flaky.
- **Politeness:** single `httpx.Client` connection pool shared across
  phases, jittered 1–2s delay, 3-retry tenacity backoff, 5-failure
  circuit breaker with 60s cooldown. User-Agent identifies the bot. If
  Phase 1 trips the breaker, Phase 2 is skipped for that run (no point
  hammering a source that's already failing).
- **Build:** `uv run zeeker build judgments`. Re-invoke the same command
  to continue a batch crawl (checkpoint drives resume) and/or drain
  more of the enrichment backlog.
- **zeeker quirk (dodged in Phase 2):** when `fragments = true` in
  `zeeker.toml`, zeeker reloads the module and calls `fetch_data` a
  second time to build fragment context. Our env-var sentinel
  (`_JUDGMENTS_PHASE2_RAN_PID`) makes the second call skip Phase 2.

### Smoke test playbook
**Phase 1 (discovery only):**
1. `rm -f zeeker-judgements.db checkpoint_judgments_discovery.json && rm -rf .cache/`
2. `JUDGMENTS_EXTRACT_ENABLED=0 JUDGMENTS_MAX_PAGES_PER_RUN=2 JUDGMENTS_DELAY_BASE=1.0 uv run zeeker build judgments`
   — expect 20 records staged from pages 1–2, no `.cache/` created.
3. Re-run the same command — should resume from page 3, add 20 more
   records (40 total in DB), advance checkpoint to `last_page=4`.
4. Delete the checkpoint and re-run — the crawler should hit 5 known IDs
   on page 1 and exit in under a second ("steady-state mode"), then clear
   the checkpoint automatically.

**Phase 2 (enrichment) — run on top of step 4:**
1. `JUDGMENTS_EXTRACT_MAX_PER_RUN=3 uv run zeeker build judgments`
   — expect Phase 1 to hit steady-state on page 1, then Phase 2 to
   extract 3 judgments. `.cache/judgments_html/` and
   `.cache/judgments_extractions/` each get 3 files; `judgments` has 3
   rows with `has_content=1`; `judgments_fragments` is created and
   populated.
2. Re-run — picks 3 **different** docs (the NULL-content query skips
   the already-enriched rows).
3. Resume test: `rm -rf .cache/judgments_html/` then re-run — expect HTML
   to be re-fetched from server but DB state unchanged (extraction cache
   still pushes already-parsed rows).
4. Rule-change test: `rm -rf .cache/judgments_extractions/` (keep the
   HTML archive), reset a few rows with `UPDATE judgments SET
   content_text=NULL WHERE id IN (…)`, re-run — expect parsing from
   archived HTML, no server traffic.

**Unit + fixture tests:** `uv run pytest tests/` (39 tests, ~0.3s).

### Data source notes
- Catchwords (`subject_tags`) are hierarchical: `Subject — Topic — Sub —
  Question`. Stored as a JSON array per row so Datasette can facet / full-
  text search them. Use `json_extract(subject_tags, '$[0]')` or similar for
  queries.
- Case numbers can be multi-valued (e.g. `DC/OC 1154/2025 ( DC/AD 16/2026 )`
  has an embedded secondary reference). Stored verbatim from the listing,
  pipe-separated when multiple `a.case-num-link` elements exist.
- PDF URLs are stored percent-encoded because the underlying eLitigation
  endpoint embeds `[`, `]` and spaces from the citation into the path.

---

This file is automatically created by Zeeker and can be customized for your project's needs.
The main Zeeker development guide is in the repository root CLAUDE.md file.
