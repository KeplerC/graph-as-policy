"""Case schema v2: taxonomy, tiers, the Case dataclass, templates.

A case is one (prompt, scene, criterion) triple. The criterion is a
declarative DSL spec (see :mod:`zsp.dsl`) — no per-case evaluator
functions. Fields the v1 schema wired into checker dispatch
(``checker/targets/forbidden/base/reference/support``) are gone: they
are plain predicate args inside ``criterion`` now.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

TAXONOMY = (
    "grounding",     # referents, attributes, categories, negation, noise
    "spatial",       # precision placement, relations, stacking, motion-only
    "temporal",      # ordering, sequencing, memory
    "quantity",      # counting, cardinality constraints
    "articulated",   # drawers, knobs, doors (classic scenes)
    "coding",        # iteration, branching, computation, verify loops
    "report",        # perception + answer, no manipulation required
    "robustness",    # seed sweeps, variance/swap/permutation/perturbed
    "refusal",       # absent referents, infeasible/unsafe, fail-safe
    "classic",       # classic-LIBERO native-language generalization
)
TIERS = ("T0", "T1", "T2")   # regression / core / frontier
SUCCESS_SOURCES = ("trace", "env", "both", "gen")
AGGREGATES = ("all_seeds", "any_seed")

DEFAULT_SIM_TASK = "libero_object_all_variance/0"


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    category: str
    tier: str
    criterion: Mapping[str, Any]
    intent: str                     # human prose: what the criterion means
    baseline: str = ""              # tester-observed outcome, verbatim
    expected: str = ""              # author expectation notes (frontier)
    tags: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (0,)
    timeout_min: int = 45
    sim_task: str = DEFAULT_SIM_TASK
    success_source: str = "trace"   # trace | env | both | gen
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    env_consumes_packed: bool = False
    requires_schema: int = 1        # 2 = needs contacts/joints in the trace
    vacuous_ok: bool = False        # opt out of the no-op selftest guard
    layout: str = ""                # layouts/ key override
    aggregate: str = "all_seeds"
    template: str = ""              # set by expand() for permutation sweeps

    def layout_key(self, seed: int | None = None) -> str:
        """The layouts/ stem for this case (first seed by default)."""
        if self.layout:
            return self.layout
        suite, _, task = self.sim_task.partition("/")
        s = self.seeds[0] if seed is None else seed
        return f"{suite}__task{int(task)}__seed{s}"

    def validate(self) -> None:
        errs: list[str] = []
        if not self.id or not self.id.replace("_", "").isalnum():
            errs.append(f"bad id {self.id!r}")
        if not self.prompt.strip():
            errs.append("empty prompt")
        if self.category not in TAXONOMY:
            errs.append(f"category {self.category!r} not in {TAXONOMY}")
        if self.tier not in TIERS:
            errs.append(f"tier {self.tier!r} not in {TIERS}")
        if self.success_source not in SUCCESS_SOURCES:
            errs.append(f"success_source {self.success_source!r} invalid")
        if self.aggregate not in AGGREGATES:
            errs.append(f"aggregate {self.aggregate!r} invalid")
        if self.success_source != "gen" and not self.seeds:
            errs.append("no seeds")
        if self.timeout_min <= 0:
            errs.append("timeout_min must be positive")
        if self.requires_schema not in (1, 2):
            errs.append("requires_schema must be 1 or 2")
        if "/" not in self.sim_task:
            errs.append(f"sim_task {self.sim_task!r} must be 'suite/task_id'")
        if not isinstance(self.criterion, Mapping) or not self.criterion.get("checks"):
            errs.append("criterion must be a dict with non-empty checks")
        if not self.intent.strip():
            errs.append("empty intent")
        if errs:
            raise ValueError(f"case {self.id!r}: " + "; ".join(errs))


def expand(
    template: str,
    make: Callable[[Any], Case],
    params: Iterable[Any],
) -> list[Case]:
    """Stamp one capability template over permuted parameters.

    ``make(param)`` returns a fully-formed Case; ``expand`` marks each
    with the template name (report grouping / per-object breakdown) and
    enforces the ``<template>__<param-ish>`` id convention.
    """
    out: list[Case] = []
    for p in params:
        case = dataclasses.replace(make(p), template=template)
        if not case.id.startswith(f"{template}__"):
            raise ValueError(
                f"template case id {case.id!r} must start with "
                f"{template + '__'!r}"
            )
        out.append(case)
    return out
