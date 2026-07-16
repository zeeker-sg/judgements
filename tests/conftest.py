"""Shared test setup.

Zeeker >= 0.9.0 puts the resources/ directory on sys.path while it loads a
resource module, so resource files use plain top-level sibling imports
(``import extraction``). Tests that import resource modules through the
``resources`` package need the same directory on sys.path for those sibling
imports to resolve.
"""

import sys
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
if str(RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(RESOURCES_DIR))
