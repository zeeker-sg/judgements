"""Phase 3 summarisation — rolling fragment approach with sanity-check pass.

Two-pass design
---------------
Pass 1 (rolling): reads ALL fragments in order, processing ``batch_size``
at a time. Each call receives the running summary plus the next batch and
produces an updated summary. This sidesteps context-window overflow entirely
— no fragment is ever skipped or deprioritised by a weighted sampler.

Pass 2 (sanity check): one final call on the completed rolling summary.
Strips rolling artefacts (meta-text, duplicate headers), checks internal
coherence, and compresses to the dynamic character limit.

Batch cap
---------
``_MAX_BATCHES`` (env ``JUDGMENTS_SUMMARY_MAX_BATCHES``, default 20) caps the
number of rolling calls per document. For large judgments the effective
batch_size widens via ceiling-division so exactly ``_MAX_BATCHES`` passes
cover the full fragment list. A 957-fragment judgment that would need 96
batches at batch_size=10 is handled in 20 passes at batch_size=48 instead,
keeping per-document wall-time proportional to _MAX_BATCHES.

Dynamic length limit
--------------------
``max_summary_chars(fragment_count)`` returns:
  4,000 chars for ≤ 100 fragments
  +1,000 chars per additional 100 fragments beyond that

So a 265-fragment judgment gets 6,000 chars; a 464-fragment judgment gets
7,000 chars. Configured via ``JUDGMENTS_SUMMARY_BASE_CHARS`` (default 4000)
and ``JUDGMENTS_SUMMARY_CHARS_PER_100`` (default 1000).

Legacy helpers
--------------
``compose_summary_input``, ``summarise``, and their supporting functions are
retained for reference. They are no longer called from ``_summarise_row``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import httpx

# ── Dynamic length limit ─────────────────────────────────────────────────────

_SUMMARY_BASE_CHARS = int(os.environ.get("JUDGMENTS_SUMMARY_BASE_CHARS", "4000"))
_SUMMARY_CHARS_PER_100 = int(os.environ.get("JUDGMENTS_SUMMARY_CHARS_PER_100", "1000"))

# ── Sampling parameters ──────────────────────────────────────────────────────
#
# Tuned for factual extraction from legal documents.
#   temperature=0.0   → deterministic, no creativity
#   top_p=0.9         → focused sampling
#   frequency/presence penalty → reduce repetitive legal boilerplate
#   seed (fixed)      → reproducible output for debugging
#   repeat_penalty    → Ollama-specific repetition suppression
#   top_k             → Ollama-specific token restriction
#
_SUMMARY_TEMPERATURE = float(os.environ.get("JUDGMENTS_SUMMARY_TEMPERATURE", "0.0"))
_SUMMARY_TOP_P = float(os.environ.get("JUDGMENTS_SUMMARY_TOP_P", "0.9"))
_SUMMARY_FREQUENCY_PENALTY = float(
    os.environ.get("JUDGMENTS_SUMMARY_FREQUENCY_PENALTY", "0.15")
)
_SUMMARY_PRESENCE_PENALTY = float(
    os.environ.get("JUDGMENTS_SUMMARY_PRESENCE_PENALTY", "0.1")
)
_SUMMARY_SEED = int(os.environ.get("JUDGMENTS_SUMMARY_SEED", "42"))
_SUMMARY_REPEAT_PENALTY = float(
    os.environ.get("JUDGMENTS_SUMMARY_REPEAT_PENALTY", "1.2")
)
_SUMMARY_TOP_K = int(os.environ.get("JUDGMENTS_SUMMARY_TOP_K", "20"))


def max_summary_chars(fragment_count: int) -> int:
    """Return the character budget for a rolling summary given fragment count.

    4,000 for ≤100 fragments; +1,000 per additional 100 fragments beyond that.
    """
    extra = max(0, (fragment_count - 100) // 100)
    return _SUMMARY_BASE_CHARS + extra * _SUMMARY_CHARS_PER_100


# ── Rolling prompts ──────────────────────────────────────────────────────────

ROLLING_SYSTEM_PROMPT = """You are a Singapore lawyer reading a court judgment. \
Build a structured summary covering three sections:

**Facts** — the key facts, parties, and nature of the dispute
**Holding** — what the court decided
**Reasons** — the main legal reasoning, principles applied, and cases cited

Write in plain prose under those three headings. Be concise and information-dense.

Do not conflate cited cases with the case before the court. Other named cases \
and their parties are precedents the court discusses; their facts, holdings, and \
monetary awards belong under **Reasons** (as authority applied, distinguished, or \
overruled), never under **Facts** or **Holding**. **Facts** and **Holding** \
describe only the dispute and decision in the present judgment.

Transcribe party names, statute titles, and legal terms exactly as they appear \
in the excerpts. Do not abbreviate, paraphrase, or alter names. Do not include \
any meta-commentary about the excerpts or the summarisation process — output \
only the three-section summary.

If the new excerpts add nothing to the existing summary, output the existing \
summary unchanged without any preamble.{anchor}"""

# Per-row anchor appended to the system prompt so every batch call knows which
# case is "the present case" — the model cannot otherwise tell the current
# dispute from a heavily-discussed precedent. See Issue #1 (JGP v JGQ).
_ROLLING_ANCHOR = """

The present case you are summarising is: {case}. Any other case name or party \
you encounter in the excerpts is a cited precedent, not the present dispute."""

_ROLLING_FIRST = """\
Here are the opening excerpts from a Singapore court judgment. Begin building your summary.
You will receive further excerpts to refine it. Keep the summary under {limit} characters.

<excerpts>
{text}
</excerpts>

Summary (Facts · Holding · Reasons):"""

_ROLLING_CONTINUE = """\
You are building a running summary of a Singapore court judgment.

<current_summary>
{summary}
</current_summary>

Here are the next excerpts. Update the summary to incorporate any new facts, holding, \
or reasons. Preserve what you already established unless the new excerpts correct it.
Keep the total summary under {limit} characters — synthesise rather than append \
if it is getting long.

<new_excerpts>
{text}
</new_excerpts>

Updated summary:"""

# ── Sanity-check prompts ─────────────────────────────────────────────────────

_SANITY_SYSTEM = """\
You are a senior Singapore lawyer reviewing a draft case summary for a legal research database."""

_SANITY_PROMPT = """\
The following draft summary of a Singapore court judgment is too long. \
Condense it to under {limit} characters while preserving the three-section \
structure (Facts, Holding, Reasons). Remove less important detail; \
never truncate mid-sentence.

<draft_summary>
{summary}
</draft_summary>

Condensed summary:"""

# Regex to strip common rolling-pass artefacts from Pass 1 output.
_META_PREFIX_RE = re.compile(
    r"^\s*(Updated summary|Summary so far|Draft summary|"
    r"Summary\s*\(Facts[^)]*\))\s*:?\s*",
    re.IGNORECASE,
)

# Rolling-pass commentary that leaks into the model's output as prose.
# The model sometimes prefaced its response with meta-text about what the
# new excerpts contained and why the summary was unchanged. These are NOT
# simple prefixes — they are full sentences the model writes to itself.
# Pattern: zero or more commentary paragraphs followed by the real content
# (which starts with **Facts** or ## Facts or similar heading). We strip
# everything before the first structural heading.
_COMMENTARY_LEAK_RE = re.compile(
    r"""^                          # start of text
    (?:
        (?:The\s+provided\s+(?:new\s+)?excerpts?   # "The provided excerpts..."
           (?:contain|consist\s+of|are\s+empty|are\s+only)[^.]*\.
        )
        |(?:As\s+(?:there\s+is\s+no|these\s+do\s+not)[^.]*\.)
        |(?:The\s+summary\s+remains\s+unchanged\.?)
        |(?:As\s+these\s+do\s+not\s+introduce[^.]*\.)
        |(?:Based\s+on\s+the\s+provided\s+(?:headings|excerpts)[^.]*\.)
        |(?:I\s+have\s+initiated\s+the\s+summary\.?)
        |(?:These\s+details\s+are\s+already\s+captured[^.]*\.)
        |.*?currently\s+placeholder.*
    )
    \s*                           # optional whitespace
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Structural heading that marks the start of real summary content.
_HEADING_LINE_RE = re.compile(
    r"^\s*\*{0,2}\s*(Facts|Factual\s+Background|Background|Holding|Decision|Reasons|Analysis)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_commentary(text: str) -> str:
    """Remove rolling-pass commentary that leaked into the summary output.

    The model sometimes wrote prose about the excerpts it was processing
    ("The provided new excerpts contain only the names of the legal
    counsel...") before the actual summary. We strip everything before
    the first structural heading (**Facts**, ## Holding, etc.) if the
    text before that heading looks like meta-commentary.
    """
    if not text:
        return text
    # Fast path: if the text starts with a heading, there's no commentary.
    if _HEADING_LINE_RE.match(text):
        return text
    # Find the first structural heading.
    m = _HEADING_LINE_RE.search(text)
    if m and m.start() > 0:
        prefix = text[: m.start()].strip()
        # Only strip if the prefix looks like commentary, not actual content.
        # Commentary is short (a few sentences, no markdown structure).
        if len(prefix) < 500 and not prefix.startswith("**"):
            return text[m.start():].lstrip()
    return text


# ── LLM transcription-artifact cleanup ──────────────────────────────────────

# Underscores inside words are tokenization artifacts (e.g. "E_ik" for "Eik").
# Only fix when the underscore is between two letters — never in URLs, code,
# or markdown formatting. We use word boundaries to avoid touching snake_case
# identifiers that may legitimately appear.
_UNDERSCORE_IN_NAME_RE = re.compile(r"(?<=[A-Za-z])_(?=[A-Za-z])")

# Dollar sign before a Capital letter — common artifact where the model
# drops an apostrophe or space (e.g. "Women's $Charter" → "Women's Charter").
# Only replace when preceded by a word char + optional space/apostrophe.
_DOLLAR_BEFORE_WORD_RE = re.compile(r"(?<=[a-z'\s])\$Charter\b", re.IGNORECASE)

# Extra trailing consonants on names (e.g. "Khoon" → "Khoont") are harder
# to detect generically; we handle the known cases via a small lookup.
# This is intentionally conservative — only fixing observed artifacts.
# NOTE: "Broadly" is NOT a transcription error — it is the judge's own
# shorthand for "Broadley Engineering" in the judgment body text.
_KNOWN_NAME_FIXES = {
    "Khoont": "Khoon",
    "Emplain": "Claim",
}


def _fix_transcription_artifacts(text: str) -> str:
    """Fix common LLM tokenization artifacts in party names and legal terms.

    These are NOT extraction errors — the source HTML and fragment text are
    correct. The artifacts arise from the LLM's tokenization of proper nouns
    and special characters, producing underscores, dollar signs, and extra
    consonants in names.
    """
    if not text:
        return text
    # Fix underscores between letters in names (e.g. "E_ik" → "Eik").
    text = _UNDERSCORE_IN_NAME_RE.sub("", text)
    # Fix "$Charter" → "Charter" (dollar sign replacing apostrophe/space).
    text = _DOLLAR_BEFORE_WORD_RE.sub("Charter", text)
    # Fix HTML entities that leaked through from source markup.
    text = _HTML_ENTITY_RE.sub(_decode_html_entity, text)
    # Fix known name transcription errors.
    for wrong, right in _KNOWN_NAME_FIXES.items():
        text = text.replace(wrong, right)
    return text


# HTML entities that can appear when the LLM copies text containing
# unresolved entities from the source HTML (e.g. &amp;, &ge;, &le;).
_HTML_ENTITY_RE = re.compile(r"&(amp|lt|gt|ge|le|ne|quot|apos|nbsp);")
_HTML_ENTITY_MAP = {
    "amp": "&", "lt": "<", "gt": ">", "ge": "≥", "le": "≤",
    "ne": "≠", "quot": '"', "apos": "'", "nbsp": " ",
}


def _decode_html_entity(m: re.Match) -> str:
    return _HTML_ENTITY_MAP.get(m.group(1), m.group(0)) or m.group(0)


# ── Summary smell test ───────────────────────────────────────────────────────
#
# Lightweight post-generation quality gate. Runs after every LLM call to
# reject obviously bad summaries before they reach the DB. The checks are
# all regex / string-based (no LLM call) so they add negligible latency.
#
# For the standalone audit (scripts/audit_summaries.py), the same function
# is applied to every existing summary in the DB.

# Minimum acceptable word count. Summaries shorter than this are almost
# certainly truncated or empty LLM responses.
_SMELL_MIN_WORDS = 50

# Maximum prefix (before the first structural heading) that we tolerate.
# Longer than this and it's likely real content, not commentary.
_SMELL_MAX_PREFIX_CHARS = 500

# Structural headings — the summary must have at least one.
_SMELL_HEADING_RE = re.compile(
    r"^\s*\*{0,2}\s*(Facts|Factual\s+Background|Background|Holding|Decision|Reasons|Analysis)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Commentary leak patterns — any of these at the start of the summary
# means the rolling loop's self-talk leaked into the output.
_SMELL_COMMENTARY_RE = re.compile(
    r"^\s*(?:"
    r"The\s+provided\s+(?:new\s+)?excerpts?"
    r"|As\s+there\s+is\s+no\s+new"
    r"|The\s+summary\s+remains\s+unchanged"
    r"|Based\s+on\s+the\s+provided"
    r"|I\s+have\s+initiated\s+the\s+summary"
    r"|These\s+details\s+are\s+already\s+captured"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Mid-sentence truncation: summary doesn't end with terminal punctuation.
_SMELL_TERMINAL_RE = re.compile(r"[.!?:)\]\"'`]$")

# HTML entities that shouldn't appear in clean output.
_SMELL_ENTITY_RE = re.compile(r"&(amp|lt|gt|ge|le|ne|quot|apos|nbsp|#\d+);")

# Underscore between letters (tokenization artifact).
_SMELL_UNDERSCORE_RE = re.compile(r"[A-Za-z]_[A-Za-z]")


def smell_test(summary: str, *, fragment_count: int = 0) -> Dict[str, Any]:
    """Lightweight quality gate for a generated summary.

    Returns a dict with:
        ``passed`` (bool): True if all checks pass.
        ``issues`` (list[str]): Human-readable issue descriptions.
        ``severity`` (str): "pass", "warn", or "fail".

    A ``fail`` means the summary should not be persisted — treat it as an
    LLM error and retry. A ``warn`` means the summary is acceptable but
    has a minor issue worth logging.

    All checks are regex/string-based — no LLM call, so this adds
    negligible latency to the build pipeline.
    """
    issues: List[str] = []
    if not summary or not summary.strip():
        return {"passed": False, "issues": ["empty summary"], "severity": "fail"}

    s = summary.strip()
    wc = len(s.split())

    # 1. Too short — likely truncated or empty LLM response.
    # For judgments with few fragments, a short summary is expected.
    # For larger judgments, scale the minimum up proportionally.
    if fragment_count <= 20:
        min_words = 20  # Floor for very short judgments
    else:
        min_words = max(_SMELL_MIN_WORDS, min(fragment_count // 5, 200))
    if wc < min_words:
        issues.append(f"too short: {wc} words (need ≥{min_words} for {fragment_count} fragments)")

    # 2. No structural heading — missing Facts/Holding/Reasons.
    if not _SMELL_HEADING_RE.search(s):
        issues.append("no structural heading (Facts/Holding/Reasons)")

    # 3. Commentary leak at the start.
    if _SMELL_COMMENTARY_RE.search(s[:500]):
        issues.append("commentary leak at start")

    # 4. Mid-sentence truncation — doesn't end with terminal punctuation.
    if not _SMELL_TERMINAL_RE.search(s.rstrip()[-1:]):
        issues.append("ends without terminal punctuation (possible truncation)")

    # 5. Trailing incomplete list marker.
    if _INCOMPLETE_TRAIL_RE.search(s.rstrip()):
        issues.append("trailing incomplete list marker")

    # 6. HTML entities in the text.
    if _SMELL_ENTITY_RE.search(s):
        issues.append("HTML entity in text")

    # 7. Underscore-in-name artifact.
    if _SMELL_UNDERSCORE_RE.search(s):
        issues.append("underscore in name (tokenization artifact)")

    severity = "fail" if issues else "pass"
    return {"passed": not issues, "issues": issues, "severity": severity}


# Trailing list-item / heading patterns that indicate incomplete content.
# Key: must be a standalone line ending with a bare number+period, NOT a
# decimal/dollar amount at the end of a sentence. We require the number to
# be at the START of its line (preceded by start-of-line or whitespace only)
# and followed by nothing but whitespace.
_INCOMPLETE_TRAIL_RE = re.compile(
    r"(?:"
    r"(?:^|\n)\s*\d+\.\s*$"       # bare "2." on its own line at end
    r"|(?:^|\n)\s*\d+\.\s*\*{0,2}\s*$"  # "2. **" on its own line at end
    r"|(?:^|\n)\s*\*\*\d+\.\s*$"  # "**2." — truncated bold heading at end
    r"|(?:^|\n)\s*\(\w+\)\s*$"    # bare "(a)" on its own line at end
    r"|(?:^|\n)\s*[•\-\*]\s*$"    # trailing bullet on its own line
    r")",
)


def _trim_to_complete(text: str, limit: int) -> str:
    """Trim text to *limit* chars, ending at a complete sentence or list item.

    Improves on the old ``rsplit(". ", 1)`` approach by also handling:
    - Trailing list markers ("2." with nothing after) — trim back to the
      previous complete item.
    - Trailing markdown heading/bold openers ("**2." ) — trim back.
    Returns the trimmed text with a closing period if the cut point was
    mid-sentence.
    """
    if len(text) <= limit:
        # Even when within limit, check for trailing incomplete items.
        return _trim_trailing_incomplete(text)
    chunk = text[:limit]

    # If the chunk ends with an incomplete list item (e.g. "2." or "(a)"),
    # trim back to the start of that item's line.
    stripped = chunk.rstrip()
    m = _INCOMPLETE_TRAIL_RE.search(stripped)
    if m:
        result = stripped[: m.start()].rstrip()
        if result and not result.endswith((".", ":", "!", "?")):
            result += "."
        if result and len(result) < len(stripped):
            return result

    # Default: trim to last complete sentence within the limit.
    trimmed = chunk.rsplit(". ", 1)
    if len(trimmed) > 1 and len(trimmed[0]) > limit * 0.5:
        return trimmed[0] + "."
    # Fallback: look for the last period, newline, or list marker.
    for sep in ["\n\n", "\n", ". ", ".\n"]:
        idx = chunk.rfind(sep)
        if idx > limit * 0.5:
            result = chunk[:idx].rstrip()
            if not result.endswith((".", ":", "!", "?")):
                result += "."
            return result
    return chunk.rstrip() + "."


def _trim_trailing_incomplete(text: str) -> str:
    """Remove trailing incomplete list items (e.g. bare "2." or "(a)").

    Unlike _trim_to_complete, this does NOT enforce a character limit —
    it only strips trailing incomplete content, leaving the rest intact.
    Used in the backfill to clean summaries that were truncated mid-list
    by the LLM's token limit.
    """
    stripped = text.rstrip()
    if not _INCOMPLETE_TRAIL_RE.search(stripped):
        return text
    # Find the start of the trailing incomplete item and trim there.
    # The trailing item is a bare number/letter marker on its own line.
    m = _INCOMPLETE_TRAIL_RE.search(stripped)
    if m:
        # Cut everything from the start of the trailing item's line.
        # m.start() points to the \n before the bare number, or start of text.
        cut = m.start()
        # If the match starts with \n, we want to keep up to the \n.
        # If it starts at position 0, trim everything (edge case).
        result = stripped[:cut].rstrip()
        if result and not result.endswith((".", ":", "!", "?")):
            result += "."
        if result and len(result) < len(stripped):
            return result
    return text


# ── Token usage tracking ─────────────────────────────────────────────────────

import json
from datetime import datetime, timezone

_TOKEN_LOG_PATH = os.environ.get(
    "ZEEKER_TOKEN_LOG", "/workspace/agent/token_usage.jsonl"
)


def _log_token_usage(
    *,
    endpoint: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    call_type: str = "summary",
) -> None:
    """Append a token-usage record to the shared JSONL log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent": "zeeker-judgements",
        "endpoint": endpoint,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "call_type": call_type,
    }
    try:
        os.makedirs(os.path.dirname(_TOKEN_LOG_PATH), exist_ok=True)
        with open(_TOKEN_LOG_PATH, "a") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass  # Fail silently — token tracking is non-critical


# ── LLM helpers ──────────────────────────────────────────────────────────────

def make_client():
    """Build an OpenAI-compatible client, or return None when unconfigured."""
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    if not base_url:
        return None
    from openai import OpenAI

    api_key = os.environ.get("LLM_API_KEY", "").strip() or "not-needed"
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=0)


def make_client_alt():
    """Build an alt OpenAI-compatible client, or return None when unconfigured."""
    base_url = os.environ.get("LLM_BASE_URL_2", "").strip()
    if not base_url:
        return None
    from openai import OpenAI

    api_key = os.environ.get("LLM_API_KEY_2", "").strip() or os.environ.get("LLM_API_KEY", "").strip() or "not-needed"
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=0)


def resolve_model(default: str = "llama3.1:8b") -> str:
    return os.environ.get("LLM_MODEL", "").strip() or default


def resolve_model_alt() -> str:
    """Return LLM_MODEL_2 if set, otherwise fall back to LLM_MODEL."""
    primary = os.environ.get("LLM_MODEL", "").strip() or "llama3.1:8b"
    return os.environ.get("LLM_MODEL_2", "").strip() or primary


def _call_once_native_ollama(
    messages: List[Dict[str, str]],
    model: str,
    base_url: str,
    *,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> str:
    """Native Ollama /api/chat call.

    Bypasses the OpenAI-compatible layer so ``num_ctx`` and ``num_predict``
    are respected directly. Uses a hard wall-clock timeout (total timeout in
    httpx) so a stalled stream cannot hang the build.
    """
    url = base_url.rstrip("/").removesuffix("/v1") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "num_ctx": 128000,
            "num_predict": max_tokens,
            "temperature": _SUMMARY_TEMPERATURE,
            "top_p": _SUMMARY_TOP_P,
            "repeat_penalty": _SUMMARY_REPEAT_PENALTY,
            "top_k": _SUMMARY_TOP_K,
            "seed": _SUMMARY_SEED,
        },
        "think": False,
        "stream": False,
    }
    with httpx.Client(timeout=timeout) as http:
        resp = http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = (data.get("message", {}).get("content") or "").strip()
    if not content:
        done_reason = data.get("done_reason", "unknown")
        raise ValueError(f"LLM returned empty content (done_reason={done_reason})")
    _log_token_usage(
        endpoint="ollama",
        model=model,
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
        call_type="summary",
    )
    return content


def _call_once(
    messages: List[Dict[str, str]],
    model: str,
    client,
    *,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> str:
    """Single LLM call.

    Dispatches to the native Ollama API when the endpoint looks like an Ollama
    server (base URL ends with ``/v1``), otherwise uses the OpenAI-compatible
    client. Raises ValueError on empty content.

    Sampling tuned for factual extraction: low temperature, constrained top-p/top-k,
    and penalties to suppress repetitive legal boilerplate.
    """
    base_url = str(getattr(client, "base_url", ""))
    ollama_url = os.environ.get("LLM_BASE_URL", "").strip()
    is_ollama = base_url.rstrip("/").endswith("/v1") and base_url == ollama_url
    if is_ollama:
        return _call_once_native_ollama(
            messages, model, base_url, max_tokens=max_tokens, timeout=timeout
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=_SUMMARY_TEMPERATURE,
        top_p=_SUMMARY_TOP_P,
        frequency_penalty=_SUMMARY_FREQUENCY_PENALTY,
        presence_penalty=_SUMMARY_PRESENCE_PENALTY,
        seed=_SUMMARY_SEED,
        timeout=timeout,
        extra_body={
            "think": False,
            "repeat_penalty": _SUMMARY_REPEAT_PENALTY,
            "top_k": _SUMMARY_TOP_K,
        },
    )
    usage = getattr(response, "usage", None)
    _log_token_usage(
        endpoint="openai-compatible",
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        call_type="summary",
    )
    choice = response.choices[0]
    content = getattr(choice.message, "content", "") or ""
    if not content:
        finish_reason = getattr(choice, "finish_reason", "unknown")
        raise ValueError(f"LLM returned empty content (finish_reason={finish_reason})")
    return content.strip()


# ── Fragment rendering (shared with legacy path) ─────────────────────────────

_HEADING_PREFIX = "Judg-Heading-"
_NUMBERED_CLASSES = {"Judg-1", "Judg-1-firstpara"}


def _is_heading(frag: Dict[str, Any]) -> bool:
    return (frag.get("class_name") or "").startswith(_HEADING_PREFIX)


def _is_numbered(frag: Dict[str, Any]) -> bool:
    return frag.get("class_name") in _NUMBERED_CLASSES and frag.get("paragraph_number") is not None


def _render_fragment(frag: Dict[str, Any]) -> str:
    text = (frag.get("content_text") or "").strip()
    if not text:
        return ""
    if _is_heading(frag):
        return f"## {text}"
    pn = frag.get("paragraph_number")
    if pn is not None:
        return f"[{pn}] {text}"
    return text


# ── Rolling summariser ────────────────────────────────────────────────────────

_MAX_BATCHES = int(os.environ.get("JUDGMENTS_SUMMARY_MAX_BATCHES", "20"))


def rolling_summarise(
    row: Dict[str, Any],
    fragments: List[Dict[str, Any]],
    model: str,
    client,
    *,
    batch_size: int = 10,
    max_batches: int = _MAX_BATCHES,
    timeout: float = 300.0,
) -> str:
    """Two-pass rolling summariser. See module docstring for design notes.

    ``row`` must contain at minimum ``id`` and optionally ``fragment_count``
    (used for the dynamic length limit). ``fragments`` are the full ordered
    list of fragment dicts from the fragments table.

    Raises on LLM failure — the caller (``_summarise_row``) handles retry /
    quarantine.
    """
    frags_ordered = sorted(fragments, key=lambda f: f.get("ordinal") or 0)
    frag_texts = [_render_fragment(f) for f in frags_ordered]
    frag_texts = [t for t in frag_texts if t.strip()]

    if not frag_texts:
        fallback = (row.get("content_text") or "").strip()
        if not fallback:
            raise ValueError("no fragment text and no content_text fallback")
        frag_texts = [fallback]

    # Dynamic limit based on actual rendered fragment count.
    n_frags = row.get("fragment_count") or len(frag_texts)
    limit = max_summary_chars(n_frags)
    # Scale token budget with the char limit — ~2 chars/token + 1024 for thinking overhead.
    call_max_tokens = max(4096, limit // 2 + 1024)

    # Cap at _MAX_BATCHES by widening batch_size for large docs.  For a
    # 957-frag judgment the default batch_size=10 yields 96 batches; with the
    # cap we stride at 48 frags/batch (20 batches) instead, keeping wall-time
    # proportional to max_batches rather than doc length. The alt model uses
    # max_batches=5 so each pass sees more fragments (fewer, wider passes).
    effective_batch = max(batch_size, -(-len(frag_texts) // max_batches))  # ceiling div

    batches = [frag_texts[i : i + effective_batch] for i in range(0, len(frag_texts), effective_batch)]
    summary = ""

    # Build the per-row system prompt with a case anchor so every batch knows
    # which case is "the present case" (defends against conflating a heavily
    # cited precedent into Facts/Holding — Issue #1, JGP v JGQ).
    case_name = (row.get("case_name") or "").strip()
    citation = (row.get("citation") or "").strip()
    case_label = " ".join(p for p in (case_name, citation) if p)
    anchor = _ROLLING_ANCHOR.format(case=case_label) if case_label else ""
    system_prompt = ROLLING_SYSTEM_PROMPT.format(anchor=anchor)

    # Pass 1: rolling
    for i, batch in enumerate(batches):
        text = "\n\n---\n\n".join(batch)
        if i == 0:
            user_msg = _ROLLING_FIRST.format(text=text, limit=limit)
        else:
            user_msg = _ROLLING_CONTINUE.format(summary=summary, text=text, limit=limit)

        try:
            summary = _call_once(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                model=model,
                client=client,
                max_tokens=call_max_tokens,
                timeout=timeout,
            )
        except ValueError:
            if i == 0:
                raise
            # Continuation failed (model returns empty content on this prompt).
            # Return the best summary we have so far — first-batch coverage is usually enough.
            break

        # Strip rolling-pass commentary that leaked into the output.
        summary = _strip_commentary(summary)

        # Prevent uncapped growth: _call_once returns partial content on finish_reason=length.
        # Without this guard, accumulated summary can reach 5k–20k tokens → overflows num_ctx
        # in the sanity-check pass (input=0 output tokens, content="", finish_reason=length).
        if len(summary) > limit:
            summary = _trim_to_complete(summary, limit)

    # Pass 2: sanity check — strip Python-side artefacts first.
    cleaned = _strip_commentary(_META_PREFIX_RE.sub("", summary).strip())
    cleaned = _fix_transcription_artifacts(cleaned)

    # Skip the LLM call when Pass 1 already produced a clean, within-limit result.
    if len(cleaned) <= limit:
        return cleaned

    # Summary exceeds the limit — one focused LLM call to condense it.
    sanity_user = _SANITY_PROMPT.format(summary=cleaned, limit=limit)
    final = _call_once(
        messages=[
            {"role": "system", "content": _SANITY_SYSTEM},
            {"role": "user", "content": sanity_user},
        ],
        model=model,
        client=client,
        max_tokens=call_max_tokens,
        timeout=timeout,
    )

    return _fix_transcription_artifacts(_strip_commentary(final))


# ── Legacy single-pass path (retained for reference) ─────────────────────────

SUMMARY_SYSTEM_PROMPT = """You are a legal research expert summarising Singapore court judgments.

Produce a single paragraph of at most 100 words that emphasises:
- the court and the level of the decision (e.g. Court of Appeal, High Court)
- the parties and the nature of their dispute
- the key legal issues the court had to decide
- the court's holding and its reasoning in brief
- any precedents cited or distinguished

Write in a plain, information-dense style suitable for a legal-research
search index. Use terms a legal researcher would search for. Do not
include disclaimers, speculation, editorial commentary, or text beyond
the summary paragraph.
"""

_DISPOSITIVE_RE = re.compile(r"conclusion|decision|holding|disposition|order", re.IGNORECASE)
_ANALYSIS_RE = re.compile(r"issue|analysis|reasoning", re.IGNORECASE)


def score_fragment(frag: Dict[str, Any]) -> float:
    score = 0.0
    if frag.get("has_footnotes"):
        score += 2.0
    heading = (frag.get("section_heading") or "").strip()
    if heading:
        if _DISPOSITIVE_RE.search(heading):
            score += 3.0
        elif _ANALYSIS_RE.search(heading):
            score += 1.5
    if frag.get("has_table"):
        score += 0.5
    text_len = len(frag.get("content_text") or "")
    score += 0.1 * min(text_len, 500) / 100
    return score


def compose_summary_input(
    row: Dict[str, Any],
    fragments: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    """Fragment-weighted single-pass input builder. No longer the primary path."""
    fragments = sorted(fragments, key=lambda f: f.get("ordinal") or 0)

    if not fragments:
        fallback = (row.get("content_text") or "").strip()
        return fallback[:max_chars]

    keep_ordinals: set = set()
    headings = [f for f in fragments if _is_heading(f)]
    for f in headings:
        keep_ordinals.add(f["ordinal"])

    numbered = [f for f in fragments if _is_numbered(f)]
    if numbered:
        numbered_sorted = sorted(numbered, key=lambda f: f["paragraph_number"])
        keep_ordinals.add(numbered_sorted[0]["ordinal"])
        for f in numbered_sorted[-3:]:
            keep_ordinals.add(f["ordinal"])

    def _length_of(fragment: Dict[str, Any]) -> int:
        return len(_render_fragment(fragment)) + 2

    court_summary = (row.get("court_summary") or "").strip()

    def _total_kept_chars() -> int:
        total = sum(_length_of(f) for f in fragments if f["ordinal"] in keep_ordinals)
        if court_summary:
            total += len(court_summary) + 2
        return total

    remainder = [
        f
        for f in fragments
        if f["ordinal"] not in keep_ordinals and not _is_heading(f) and not _is_numbered(f)
    ]
    remainder.sort(key=score_fragment, reverse=True)

    for f in remainder:
        if _total_kept_chars() + _length_of(f) > max_chars:
            continue
        keep_ordinals.add(f["ordinal"])

    if _total_kept_chars() > max_chars and numbered:
        keep_last = sorted(numbered, key=lambda f: f["paragraph_number"])[-3:]
        for f in keep_last:
            if _total_kept_chars() <= max_chars:
                break
            keep_ordinals.discard(f["ordinal"])

    parts: List[str] = []
    if court_summary:
        parts.append(f"## Court Summary\n{court_summary}")
    for f in fragments:
        if f["ordinal"] not in keep_ordinals:
            continue
        rendered = _render_fragment(f)
        if rendered:
            parts.append(rendered)

    return "\n\n".join(parts)[:max_chars]


def summarise(
    input_text: str,
    model: str,
    client,
    *,
    timeout: float = 120.0,
    max_tokens: int = 4096,
    temperature: float = _SUMMARY_TEMPERATURE,
) -> str:
    """Single-pass LLM call. No longer the primary path; retained for reference."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=_SUMMARY_TOP_P,
        frequency_penalty=_SUMMARY_FREQUENCY_PENALTY,
        presence_penalty=_SUMMARY_PRESENCE_PENALTY,
        seed=_SUMMARY_SEED,
        timeout=timeout,
        extra_body={
            "think": False,
            "repeat_penalty": _SUMMARY_REPEAT_PENALTY,
            "top_k": _SUMMARY_TOP_K,
        },
    )
    choice = response.choices[0]
    content = getattr(choice.message, "content", "") or ""
    if not content:
        finish_reason = getattr(choice, "finish_reason", "unknown")
        raise ValueError(f"LLM returned empty content (finish_reason={finish_reason})")
    return content.strip()
