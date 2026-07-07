"""Compat shim — the recorder lives in ``zsp/state_log.py`` (trace
schema 2: contacts, articulated joints, cavities, trace_meta.json)."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from zsp.state_log import (  # noqa: E402,F401
    SCHEMA_VERSION,
    StateRecorder,
    hang_site,
    load_jsonl,
)

__all__ = ["SCHEMA_VERSION", "StateRecorder", "hang_site", "load_jsonl"]
