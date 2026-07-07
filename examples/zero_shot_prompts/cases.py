"""Compat shim — the case corpus now lives in ``zsp/cases/`` (one module
per capability axis) with the schema in ``zsp/schema.py``. Old imports
(``from cases import CASES, Case``) keep working."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from zsp.registry import CASES  # noqa: E402,F401
from zsp.schema import Case  # noqa: E402,F401

__all__ = ["CASES", "Case"]
