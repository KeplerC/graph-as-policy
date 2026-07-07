"""Compat shim — criteria are now declarative DSL specs on each case
(see ``zsp/dsl/``), evaluated by ``zsp.evaluate.evaluate_seed``. The v1
per-case evaluator functions and checker dispatch are gone; this module
keeps the old import surface working."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from zsp.diagnostics import (  # noqa: E402,F401
    max_node_iterations,
    workflow_structure,
)
from zsp.dsl.score import Check, SeedEvaluation  # noqa: E402,F401
from zsp.evaluate import evaluate_seed  # noqa: E402,F401

__all__ = [
    "Check",
    "SeedEvaluation",
    "evaluate_seed",
    "max_node_iterations",
    "workflow_structure",
]
