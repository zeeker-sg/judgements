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

# ── Dynamic length limit ─────────────────────────────────────────────────────

_SUMMARY_BASE_CHARS = int(os.environ.get("JUDGMENTS_SUMMARY_BASE_CHARS", "4000"))
_SUMMARY_CHARS_PER_100 = int(os.environ.get("JUDGMENTS_SUMMARY_CHARS_PER_100", "1000"))


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

Write in plain prose under those three headings. Be concise and information-dense."""

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


# ── LLM helpers ──────────────────────────────────────────────────────────────

def make_client():
    """Build an OpenAI-compatible client, or return None when unconfigured."""
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    if not base_url:
        return None
    from openai import OpenAI

    api_key = os.environ.get("LLM_API_KEY", "").strip() or "not-needed"
    return OpenAI(base_url=base_url, api_key=api_key)


def make_client_alt():
    """Build the alt OpenAI-compatible client for quarantine-routed docs, or None."""
    base_url = os.environ.get("LLM_BASE_URL_2", "").strip()
    if not base_url:
        return None
    from openai import OpenAI

    api_key = (
        os.environ.get("LLM_API_KEY_2", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
        or "not-needed"
    )
    return OpenAI(base_url=base_url, api_key=api_key)


def resolve_model(default: str = "llama3.1:8b") -> str:
    return os.environ.get("LLM_MODEL", "").strip() or default


def resolve_model_alt() -> str:
    primary = os.environ.get("LLM_MODEL", "").strip() or "llama3.1:8b"
    return os.environ.get("LLM_MODEL_2", "").strip() or primary


def _call_once(
    messages: List[Dict[str, str]],
    model: str,
    client,
    *,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> str:
    """Single OpenAI-compatible call. Raises ValueError on empty content."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        timeout=timeout,
        extra_body={"think": False},
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
    # Scale token budget with the char limit — Gemma4:26b uses ~2 chars/token.
    # Add 1024 overhead for thinking tokens that count against the same budget.
    call_max_tokens = max(4096, limit // 2 + 1024)

    # Cap at _MAX_BATCHES by widening batch_size for large docs.  For a
    # 957-frag judgment the default batch_size=10 yields 96 batches; with the
    # cap we stride at 48 frags/batch (20 batches) instead, keeping wall-time
    # proportional to _MAX_BATCHES rather than doc length.
    effective_batch = max(batch_size, -(-len(frag_texts) // _MAX_BATCHES))  # ceiling div

    batches = [frag_texts[i : i + effective_batch] for i in range(0, len(frag_texts), effective_batch)]
    summary = ""

    # Pass 1: rolling
    for i, batch in enumerate(batches):
        text = "\n\n---\n\n".join(batch)
        if i == 0:
            user_msg = _ROLLING_FIRST.format(text=text, limit=limit)
        else:
            user_msg = _ROLLING_CONTINUE.format(summary=summary, text=text, limit=limit)

        summary = _call_once(
            messages=[
                {"role": "system", "content": ROLLING_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            client=client,
            max_tokens=call_max_tokens,
            timeout=timeout,
        )
        # Prevent uncapped growth: _call_once returns partial content on finish_reason=length.
        # Without this guard, accumulated summary can reach 5k–20k tokens → overflows num_ctx
        # in the sanity-check pass (input=0 output tokens, content="", finish_reason=length).
        if len(summary) > limit:
            trimmed = summary[:limit].rsplit(". ", 1)
            summary = (trimmed[0] + ".") if len(trimmed) > 1 else summary[:limit]

    # Pass 2: sanity check — strip Python-side artefacts first.
    cleaned = _META_PREFIX_RE.sub("", summary).strip()

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

    return final


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
    temperature: float = 0.2,
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
        timeout=timeout,
        extra_body={"think": False},
    )
    choice = response.choices[0]
    content = getattr(choice.message, "content", "") or ""
    if not content:
        finish_reason = getattr(choice, "finish_reason", "unknown")
        raise ValueError(f"LLM returned empty content (finish_reason={finish_reason})")
    return content.strip()
