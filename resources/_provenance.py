"""Per-record provenance tracking for Zeeker databases.

Drop this file into any zeeker-builds project's ``resources/`` directory.
It creates and writes to a hidden ``_zeeker_provenance`` table — the
``_zeeker_*`` prefix makes Datasette hide it from the UI automatically.

Schema
------
The table has a composite PK on (table_name, record_id) so each row in
every resource table can carry its own provenance record:

    table_name        — which resource table this record belongs to
    record_id         — the PK value of the row in that table
    build_id          — the zeeker build that produced/modified this record
    source_url        — URL of the original source document
    source_hash       — hash of the raw source content (for change detection)
    extractor         — e.g. "BeautifulSoup", "extraction.py v2", "docling"
    model             — LLM model name (e.g. "kimi-k2.6:cloud")
    prompt_hash       — SHA-256[:12] of the system prompt used
    llm_endpoint      — base URL of the LLM API
    confidence        — extraction confidence (0.0–1.0), if available
    processing_notes  — errors, retries, smell_test issues, etc.
    created_at        — when this provenance row was first written
    updated_at        — when it was last updated

Usage in a resource module
--------------------------
At the top of your resource file (e.g. ``resources/judgments.py``)::

    from _provenance import record_provenance, prompt_hash, ensure_provenance_table

Then inside ``fetch_data()``, after you produce or enrich a record::

    record_provenance(
        existing_table.db,
        table_name="judgments",
        record_id=row_id,
        model=model_name,
        prompt_hash=PHASH,
        llm_endpoint=os.environ.get("LLM_BASE_URL", ""),
        processing_notes="summary: ok, 4200 chars",
    )

The helper is idempotent: calling it again for the same (table_name, record_id)
performs an upsert, updating ``updated_at`` and any changed fields.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Optional

import sqlite_utils

PROVENANCE_TABLE = "_zeeker_provenance"

_SCHEMA = {
    "table_name": str,
    "record_id": str,
    "build_id": str,
    "source_url": str,
    "source_hash": str,
    "extractor": str,
    "model": str,
    "prompt_hash": str,
    "llm_endpoint": str,
    "confidence": float,
    "processing_notes": str,
    "created_at": str,
    "updated_at": str,
}


def ensure_provenance_table(db: sqlite_utils.Database) -> None:
    """Create the ``_zeeker_provenance`` table if it doesn't exist."""
    if not db[PROVENANCE_TABLE].exists():
        db[PROVENANCE_TABLE].create(_SCHEMA, pk=("table_name", "record_id"))


def get_build_id(db: sqlite_utils.Database) -> Optional[str]:
    """Read the most recent build_id from ``_zeeker_updates``.

    Returns None if the table doesn't exist or is empty.
    """
    try:
        if not db["_zeeker_updates"].exists():
            return None
        rows = list(
            db["_zeeker_updates"].rows_where(
                order_by="last_updated desc", limit=1
            )
        )
        return rows[0]["build_id"] if rows else None
    except Exception:
        return None


def prompt_hash(prompt_text: str) -> str:
    """Return a stable 12-char hash of a prompt string.

    Use this to track which prompt version produced a given LLM output::

        PHASH = prompt_hash(ROLLING_SYSTEM_PROMPT)
    """
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:12]


def source_hash(content: str) -> str:
    """Return a stable 12-char hash of source content (for change detection)."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def record_provenance(
    db: sqlite_utils.Database,
    *,
    table_name: str,
    record_id: str,
    build_id: Optional[str] = None,
    source_url: Optional[str] = None,
    source_hash: Optional[str] = None,
    extractor: Optional[str] = None,
    model: Optional[str] = None,
    prompt_hash: Optional[str] = None,
    llm_endpoint: Optional[str] = None,
    confidence: Optional[float] = None,
    processing_notes: Optional[str] = None,
) -> None:
    """Upsert a provenance row for a specific record.

    Safe to call multiple times for the same record — subsequent calls
    update ``updated_at`` and any changed fields. The ``build_id`` is
    auto-derived from ``_zeeker_updates`` if not provided.

    This function never raises — provenance is best-effort metadata and
    must not break the build pipeline.
    """
    try:
        ensure_provenance_table(db)

        if build_id is None:
            build_id = get_build_id(db)

        now = datetime.now().isoformat(timespec="seconds")

        # Check if row exists (to set created_at vs updated_at)
        existing = None
        try:
            existing = db[PROVENANCE_TABLE].get((table_name, record_id))
        except (KeyError, sqlite_utils.db.NotFoundError):
            pass
        except Exception:
            pass

        row = {
            "table_name": table_name,
            "record_id": record_id,
            "build_id": build_id,
            "source_url": source_url,
            "source_hash": source_hash,
            "extractor": extractor,
            "model": model,
            "prompt_hash": prompt_hash,
            "llm_endpoint": llm_endpoint,
            "confidence": confidence,
            "processing_notes": processing_notes,
        }

        if existing:
            row["created_at"] = existing.get("created_at", now)
            row["updated_at"] = now
            # Only update fields that are explicitly provided (not None),
            # so a Phase 3 summary call doesn't clobber Phase 2 extraction
            # fields (extractor, source_hash) that were set earlier.
            update = {"updated_at": now}
            for k, v in row.items():
                if k in ("created_at", "updated_at"):
                    continue
                if v is not None:
                    update[k] = v
            db[PROVENANCE_TABLE].update((table_name, record_id), update)
        else:
            row["created_at"] = now
            row["updated_at"] = now
            db[PROVENANCE_TABLE].insert(row, replace=True)
    except Exception:
        # Provenance is best-effort. Never break the build.
        pass


def batch_record_provenance(
    db: sqlite_utils.Database,
    rows: list[dict[str, Any]],
) -> None:
    """Insert/update multiple provenance rows at once.

    Each dict in ``rows`` must contain ``table_name`` and ``record_id``;
    all other fields are optional (same keys as ``record_provenance``).
    """
    try:
        ensure_provenance_table(db)
        now = datetime.now().isoformat(timespec="seconds")
        build_id = get_build_id(db)

        for row in rows:
            row.setdefault("build_id", build_id)
            row.setdefault("created_at", now)
            row.setdefault("updated_at", now)

        db[PROVENANCE_TABLE].insert_all(rows, replace=True)
    except Exception:
        pass