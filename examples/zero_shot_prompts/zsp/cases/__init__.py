"""Case corpus: one module per capability area, aggregated here.

Each module exports ``CASES: list[Case]``. Duplicate ids are a hard
error at import time.
"""

from __future__ import annotations

from zsp.schema import Case

_MODULES = (
    "classic0",
    "grounding",
    "spatial",
    "temporal",
    "quantity",
    "coding",
    "report",
    "robustness",
    "refusal",
    "classic_libero",
    "sweeps",
)

ALL_CASES: list[Case] = []
_seen: set[str] = set()
for _mod_name in _MODULES:
    _mod = __import__(f"zsp.cases.{_mod_name}", fromlist=["CASES"])
    for _case in _mod.CASES:
        if _case.id in _seen:
            raise ValueError(
                f"duplicate case id {_case.id!r} (module {_mod_name})")
        _seen.add(_case.id)
        ALL_CASES.append(_case)
