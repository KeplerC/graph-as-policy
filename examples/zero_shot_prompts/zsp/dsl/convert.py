"""Convert recorded ``state.jsonl`` dicts into goal-eval ``SimState``s.

Accepts both trace schemas:

- **v1** (no ``schema`` key): bodies carry ``p q lo hi v g`` only —
  contacts become empty frozensets, joints empty arrays, cavities None.
  Predicates that need the missing data degrade per their own ``auto``
  modes (see :mod:`zsp.dsl.state_predicates`).
- **v2**: adds per-body ``c`` (contacts), ``j`` (named joint qpos),
  ``cav`` (cavity AABB) and top-level stray ``joints``.

Everything :class:`BodyView` can't carry (held body, gripper fraction,
EE position, per-body speed / grasped flag, named joints) rides in
``SimState.metadata`` — the benchmark's own predicates read it there;
core predicates never look at metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gap.runtime.predicates.goal_eval import BodyView, SimState

#: Bodies never treated as task movables in any scene (v1 parity).
ALWAYS_STATIC = ("table",)
DEFAULT_CONTAINER = "basket"


def state_from_record(rec: dict[str, Any]) -> SimState:
    """One ``state.jsonl`` line -> ``SimState`` (empty on snapshot_error)."""
    bodies: dict[str, BodyView] = {}
    vel: dict[str, float] = {}
    grasped: dict[str, bool] = {}
    joints: dict[str, dict[str, float]] = {}
    for name, b in (rec.get("bodies") or {}).items():
        j = b.get("j") or {}
        cav = b.get("cav")
        bodies[name] = BodyView(
            name=name,
            position=np.asarray(b["p"], dtype=np.float64),
            quaternion_wxyz=np.asarray(b["q"], dtype=np.float64),
            aabb_lower=np.asarray(b["lo"], dtype=np.float64),
            aabb_upper=np.asarray(b["hi"], dtype=np.float64),
            joint_qpos=np.asarray(list(j.values()), dtype=np.float64),
            contacts=frozenset(b.get("c") or ()),
            cavity_lower=(
                np.asarray(cav[0], dtype=np.float64) if cav else None
            ),
            cavity_upper=(
                np.asarray(cav[1], dtype=np.float64) if cav else None
            ),
        )
        vel[name] = float(b.get("v", 0.0))
        grasped[name] = bool(b.get("g", False))
        if j:
            joints[name] = {k: float(v) for k, v in j.items()}
    for jname, q in (rec.get("joints") or {}).items():
        joints.setdefault("", {})[jname] = float(q)
    return SimState(
        bodies=bodies,
        time_s=float(rec.get("t", 0.0)),
        metadata={
            "schema": int(rec.get("schema", 1)),
            "held": rec.get("held"),
            "grip": rec.get("grip"),
            "ee": (
                np.asarray(rec["ee"], dtype=np.float64)
                if rec.get("ee") is not None else None
            ),
            "vel": vel,
            "grasped": grasped,
            "joints": joints,
            "tool": rec.get("tool"),
            "snapshot_error": rec.get("snapshot_error"),
        },
    )


def states_from_records(records: list[dict]) -> list[SimState]:
    return [state_from_record(r) for r in records]


@dataclass
class EvalContext:
    """Everything a trace-level predicate may consult."""

    states: list[SimState]
    case: Any = None
    events: list[dict] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    t0_cache: dict[str, Any] = field(default_factory=dict)

    # -- scene vocabulary ------------------------------------------------

    @property
    def container(self) -> str:
        return (
            self.options.get("container")
            or getattr(self.case, "container", None)  # legacy field
            or DEFAULT_CONTAINER
        )

    def movables(self) -> list[str]:
        """Task movables observed in the trace: every body minus the
        container, declared statics, and the always-static set (v1
        ``scene_objects`` semantics)."""
        names: set[str] = set()
        for s in self.states:
            names |= set(s.bodies.keys())
        statics = set(self.options.get("statics") or ()) | set(ALWAYS_STATIC)
        statics.add(self.container)
        return sorted(n for n in names if n not in statics)

    # -- state access ----------------------------------------------------

    @property
    def t0(self) -> SimState | None:
        """First snapshot that actually carries bodies."""
        for s in self.states:
            if s.bodies:
                return s
        return None

    @property
    def final(self) -> SimState | None:
        for s in reversed(self.states):
            if s.bodies:
                return s
        return None

    def initial_position(self, name: str) -> np.ndarray | None:
        """v1 convention: the body's position in the first snapshot that
        contains it."""
        for s in self.states:
            b = s.bodies.get(name)
            if b is not None:
                return b.position
        return None

    def initial_body(self, name: str) -> BodyView | None:
        for s in self.states:
            b = s.bodies.get(name)
            if b is not None:
                return b
        return None

    def final_body(self, name: str) -> BodyView | None:
        for s in reversed(self.states):
            if s.bodies:
                return s.bodies.get(name)
        return None

    def env_consumes_packed(self) -> bool:
        return bool(
            self.options.get(
                "env_consumes_packed",
                getattr(self.case, "env_consumes_packed", False),
            )
        )
