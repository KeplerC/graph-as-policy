"""Criteria DSL: declarative, JSON-serializable success conditions.

A criterion is a dict::

    {"checks": [
        {"name": "picked_up", "weight": 0.3, "gating": true,
         "expr": {"picked_up": {"target": "alphabet_soup"}}},
        ...],
     "options": {"container": "basket", "require_picked_up": true}}

evaluated over the recorded trace (``state.jsonl`` records converted to
:class:`gap.runtime.predicates.goal_eval.SimState`). Leaf predicates
resolve, in order, against:

1. :data:`zsp.dsl.trace_predicates.TRACE_REGISTRY` — cross-time
   predicates over the whole trace (grasp-and-lift, ordering,
   displacement-from-start, report scanning, ...);
2. :data:`zsp.dsl.state_predicates.STATE_EXTRAS` — per-state predicates
   this benchmark adds (v1-parity containment, metadata-based held/
   settled, named-joint thresholds, ...);
3. the core :data:`gap.runtime.predicates.predicates.REGISTRY` — the
   same leaf vocabulary GaP's rehearsal/goal system uses.

Composition ops: ``and or not eventually always at_end never`` (a
per-state leaf at the top of a check defaults to ``at_end``, mirroring
``goal_eval``). ``{"$t0": {...}}`` value nodes resolve derived referents
(closest_to, extreme, largest, midpoint, ...) against the first
snapshot. Verdict/scoring and the auto-injected anti-gaming invariants
live in :mod:`zsp.dsl.score`.
"""

from __future__ import annotations

from zsp.dsl.ast import (
    evaluate_check_expr,
    resolve_t0_refs,
    validate_criterion,
)
from zsp.dsl.convert import EvalContext, states_from_records
from zsp.dsl.score import Check, SeedEvaluation, evaluate_criterion

__all__ = [
    "Check",
    "EvalContext",
    "SeedEvaluation",
    "evaluate_check_expr",
    "evaluate_criterion",
    "resolve_t0_refs",
    "states_from_records",
    "validate_criterion",
]
