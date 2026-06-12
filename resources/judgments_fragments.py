"""Stub resource for the judgments_fragments table — real rows come from judgments.py.

The fragments table is populated by Phase 2 inside ``judgments.fetch_data``
(``_insert_fragments`` writes rows transactionally with each main-row
UPDATE; ``judgments.fetch_fragments_data`` is itself a deliberate no-op).
This file exists only because zeeker's builder treats every
``[resource.X]`` section in zeeker.toml as a buildable resource and fails
the build when ``resources/X.py`` is missing — and the
``[resource.judgments_fragments]`` section must stay, because it carries
the Datasette column descriptions and facets for the fragments table.

Returning an empty list makes zeeker record this resource as
``[SKIP] no data returned`` without touching the table.
"""

from typing import Any, Dict, List, Optional

from sqlite_utils.db import Table


def fetch_data(existing_table: Optional[Table]) -> List[Dict[str, Any]]:
    """No-op: judgments_fragments rows are written by judgments.py Phase 2."""
    return []
