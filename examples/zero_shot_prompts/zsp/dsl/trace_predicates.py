"""Trace-level predicates: cross-time checks over the whole recorded run.

Evaluator contract ``(EvalContext, resolved_args) -> (bool, diagnostics)``
— same :class:`PredicateSpec` descriptor as per-state leaves, separate
registry, dispatched first by the walker. These consume the WHOLE trace
(plus events/execution/extras) and therefore reject temporal wrappers at
validation time.

Semantics marked "v1" reproduce ``criteria.py`` exactly (parity gate).
A ``score`` key in diagnostics is the leaf's fractional partial credit;
the check-level scorer picks it up when the check fails.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gap.runtime.predicates.predicates import PredicateSpec
from zsp.dsl.convert import EvalContext
from zsp.dsl.state_predicates import (
    GRIP_CLOSED_MAX,
    LIFT_MIN,
    resting_in_container,
)

#: v1: the VAB packing suites' pack_all_into predicate TELEPORTS a
#: contained object to a far-away graveyard (measured (50.6, 50.0)).
GRAVEYARD_ABS = 5.0

# v1 workspace sanity envelope (floor arena; classic scenes override via
# criterion options — probed layouts are the source for those numbers).
SANE_MIN_Z = -0.15
SANE_MAX_ABS_Y = 0.70
SANE_X = (-0.10, 1.25)

TRACE_REGISTRY: dict[str, PredicateSpec] = {}


def register_trace(
    name: str,
    *,
    body_args: Sequence[str] = (),
    optional_args: Mapping[str, Any] | None = None,
    description: str = "",
) -> Callable:
    def deco(fn: Callable) -> Callable:
        if name in TRACE_REGISTRY:
            raise ValueError(f"trace predicate {name!r} already registered")
        TRACE_REGISTRY[name] = PredicateSpec(
            name=name,
            body_args=tuple(body_args),
            optional_args=dict(optional_args or {}),
            evaluator=fn,
            description=description,
        )
        return fn

    return deco


# ---------------------------------------------------------------------------
# Shared index helpers (v1 semantics)
# ---------------------------------------------------------------------------


def grasp_lift_index(
    ctx: EvalContext, name: str,
    *, min_lift_m: float = LIFT_MIN, grip_max: float = GRIP_CLOSED_MAX,
) -> int | None:
    """v1: first snapshot where `name` is held (engine held_body), the
    gripper is closed on it, and it has risen >= min_lift above its
    initial height. None = never picked up."""
    z0: float | None = None
    for s in ctx.states:
        b = s.bodies.get(name)
        if b is not None:
            z0 = float(b.position[2])
            break
    if z0 is None:
        return None
    for i, s in enumerate(ctx.states):
        if s.metadata.get("held") != name:
            continue
        grip = s.metadata.get("grip")
        if grip is not None and float(grip) >= grip_max:
            continue
        b = s.bodies.get(name)
        if b is not None and float(b.position[2]) >= z0 + min_lift_m:
            return i
    return None


def first_in_container_index(
    ctx: EvalContext, name: str, container: str | None = None,
) -> int | None:
    container = container or ctx.container
    for i, s in enumerate(ctx.states):
        if resting_in_container(s, name, container):
            return i
    return None


def at_graveyard(p: np.ndarray, bound: float = GRAVEYARD_ABS) -> bool:
    return abs(float(p[0])) > bound or abs(float(p[1])) > bound


def _quat_rotmat(q: Any) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=float)[:4])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_from_initial_up_deg(q0: Any, q1: Any) -> float:
    """v1: angle between world-up and the body direction that pointed up
    at t0. 0 = same lean as at start; ~90 = knocked onto its side."""
    up0_local = _quat_rotmat(q0).T @ np.array([0.0, 0.0, 1.0])
    now = _quat_rotmat(q1) @ up0_local
    c = float(np.clip(now[2] / (np.linalg.norm(now) or 1.0), -1.0, 1.0))
    return math.degrees(math.acos(c))


# ---------------------------------------------------------------------------
# Manipulation-event leaves
# ---------------------------------------------------------------------------


@register_trace(
    "picked_up",
    body_args=("target",),
    optional_args={"min_lift_m": LIFT_MIN, "grip_max": GRIP_CLOSED_MAX},
    description="v1 grasp-and-lift: at some snapshot the engine reports "
                "the object held, the gripper closed (< grip_max open), "
                "and the object risen >= min_lift_m above its initial "
                "height. Rejects nudges and teleports.",
)
def picked_up(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    idx = grasp_lift_index(
        ctx, args["target"],
        min_lift_m=float(args["min_lift_m"]), grip_max=float(args["grip_max"]),
    )
    return idx is not None, {"grasp_snapshot": idx}


@register_trace(
    "never_held",
    body_args=("target",),
    optional_args={},
    description="No snapshot reports the object as the held body.",
)
def never_held(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    hits = [i for i, s in enumerate(ctx.states)
            if s.metadata.get("held") == args["target"]]
    return not hits, {"held_snapshots": hits[:10]}


@register_trace(
    "before",
    body_args=(),
    optional_args={"first": None, "then": None},
    description="Event ordering: both events occur and `first`'s snapshot "
                "index is strictly smaller. Event specs: "
                "{'enters_container': {'target': X, 'container'?: C}} "
                "(first settled-inside index, v1) or "
                "{'picked_up': {'target': X}} (grasp-and-lift index).",
)
def before(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    def event_index(spec: Any) -> int | None:
        if not isinstance(spec, dict) or len(spec) != 1:
            raise ValueError(f"bad event spec for before: {spec!r}")
        [(kind, payload)] = spec.items()
        payload = dict(payload)
        if kind == "enters_container":
            return first_in_container_index(
                ctx, payload["target"], payload.get("container"))
        if kind == "picked_up":
            return grasp_lift_index(ctx, payload["target"])
        raise ValueError(f"unknown before event kind {kind!r}")

    if args["first"] is None or args["then"] is None:
        raise ValueError("before requires `first` and `then` event specs")
    ia, ib = event_index(args["first"]), event_index(args["then"])
    ok = ia is not None and ib is not None and ia < ib
    return bool(ok), {"first_index": ia, "then_index": ib}


# ---------------------------------------------------------------------------
# Displacement / orientation leaves
# ---------------------------------------------------------------------------


@register_trace(
    "displaced",
    body_args=("target",),
    optional_args={"axis": "xy", "lo": 0.0, "hi": None, "signed": False},
    description="Final-vs-initial displacement along an axis (x|y|z, "
                "optionally signed) or a norm (xy|xyz) within [lo, hi].",
)
def displaced(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    name = args["target"]
    b0, b1 = ctx.initial_body(name), ctx.final_body(name)
    if b0 is None or b1 is None:
        return False, {"missing": name}
    d = b1.position - b0.position
    axis = str(args["axis"])
    if axis in ("x", "y", "z"):
        val = float(d["xyz".index(axis)])
        if not args["signed"]:
            val = abs(val)
    elif axis == "xy":
        val = float(np.linalg.norm(d[:2]))
    elif axis == "xyz":
        val = float(np.linalg.norm(d))
    else:
        raise ValueError(f"displaced axis must be x|y|z|xy|xyz; got {axis!r}")
    hi = float("inf") if args["hi"] is None else float(args["hi"])
    ok = float(args["lo"]) <= val <= hi
    return bool(ok), {
        "value_m": round(val, 4),
        "d_m": [round(float(v), 4) for v in d],
    }


@register_trace(
    "relocated",
    body_args=("target",),
    optional_args={
        "min_disp_m": 0.20,
        "flip_axis": "y",
        "flip_min_m": 0.20,
        "cross_axis": "x",
        "cross_min_m": 0.30,
    },
    description="v1 basket_moved_to_opposite_side: xy displacement >= "
                "min_disp AND (sign flip along flip_axis with |delta| >= "
                "flip_min, OR |delta| along cross_axis >= cross_min). "
                "Partial credit min(1, disp/min)*0.5 when failed.",
)
def relocated(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    name = args["target"]
    b0, b1 = ctx.initial_body(name), ctx.final_body(name)
    if b0 is None or b1 is None:
        return False, {"missing": name, "score": 0.0}
    d = b1.position[:2] - b0.position[:2]
    disp = float(np.linalg.norm(d))
    fi = "xyz".index(str(args["flip_axis"]))
    ci = "xyz".index(str(args["cross_axis"]))
    a0, a1 = float(b0.position[fi]), float(b1.position[fi])
    lateral_flip = (a0 * a1 < 0) and abs(a1 - a0) >= float(args["flip_min_m"])
    depth_cross = abs(float(b1.position[ci] - b0.position[ci])) >= float(
        args["cross_min_m"])
    ok = disp >= float(args["min_disp_m"]) and (lateral_flip or depth_cross)
    return bool(ok), {
        "displacement_m": round(disp, 4),
        "lateral_flip": bool(lateral_flip),
        "depth_cross": bool(depth_cross),
        "score": 1.0 if ok else min(1.0, disp / float(args["min_disp_m"])) * 0.5,
    }


@register_trace(
    "tilt_from_initial",
    body_args=("target",),
    optional_args={"max_deg": 30.0, "min_deg": None},
    description="v1 uprightness: the body direction that pointed world-up "
                "at t0 is within max_deg of vertical at the end (0 = same "
                "lean as start, ~90 = on its side). Set min_deg too for a "
                "band (lay-on-side: 60..120; flipped: min_deg=150).",
)
def tilt_from_initial(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    name = args["target"]
    b0, b1 = ctx.initial_body(name), ctx.final_body(name)
    if b0 is None or b1 is None:
        return False, {"missing": name}
    tilt = tilt_from_initial_up_deg(b0.quaternion_wxyz, b1.quaternion_wxyz)
    lo = float(args["min_deg"]) if args["min_deg"] is not None else 0.0
    ok = lo <= tilt <= float(args["max_deg"])
    return bool(ok), {"tilt_deg": round(tilt, 1)}


@register_trace(
    "untouched",
    body_args=("target",),
    optional_args={"tol_m": 0.02, "allow_held": False},
    description="The object never strays more than tol_m from its initial "
                "position (max over ALL snapshots, 3D) and (unless "
                "allow_held) is never the held body.",
)
def untouched(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    name = args["target"]
    p0 = ctx.initial_position(name)
    if p0 is None:
        return False, {"missing": name}
    max_d, max_i = 0.0, None
    for i, s in enumerate(ctx.states):
        b = s.bodies.get(name)
        if b is None:
            continue
        d = float(np.linalg.norm(b.position - p0))
        if d > max_d:
            max_d, max_i = d, i
    was_held = any(s.metadata.get("held") == name for s in ctx.states)
    ok = max_d <= float(args["tol_m"]) and (args["allow_held"] or not was_held)
    return bool(ok), {
        "max_excursion_m": round(max_d, 4), "at_snapshot": max_i,
        "was_held": bool(was_held),
    }


@register_trace(
    "all_untouched",
    body_args=(),
    optional_args={"targets": "movables", "except": (), "tol_m": 0.03},
    description="Every listed movable (default: all scene movables minus "
                "`except`) stays within tol_m of its initial position and "
                "is never held.",
)
def all_untouched(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    targets = list(args["targets"])
    skip = set(args["except"] or ())
    violators: dict[str, float] = {}
    for name in targets:
        if name in skip:
            continue
        ok, diag = untouched(ctx, {
            "target": name, "tol_m": args["tol_m"], "allow_held": False,
        })
        if not ok:
            violators[name] = diag.get("max_excursion_m", -1.0)
    n = len([t for t in targets if t not in skip])
    return not violators, {
        "violators": violators,
        "score": 1.0 - len(violators) / max(1, n),
    }


# ---------------------------------------------------------------------------
# Packing leaves
# ---------------------------------------------------------------------------


@register_trace(
    "no_new_packing",
    body_args=(),
    optional_args={"container": None, "allowed": ()},
    description="v1 no_unrequested_packing: no movable outside `allowed` "
                "is settled in the container at the end unless it already "
                "was at snapshot 0.",
)
def no_new_packing(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    container = args["container"] or ctx.container
    allowed = set(args["allowed"] or ())
    first, final = ctx.t0, ctx.final
    if first is None or final is None:
        return False, {"missing": "trace"}
    stray = sorted(
        n for n in ctx.movables()
        if n not in allowed
        and resting_in_container(final, n, container)
        and not resting_in_container(first, n, container)
    )
    return not stray, {"stray": stray}


@register_trace(
    "kept_out",
    body_args=(),
    optional_args={"targets": (), "container": None},
    description="v1 excluded_objects_stay_out: every listed object is NOT "
                "settled in the container at the end (or already was at "
                "snapshot 0 — pre-packed doesn't count against the run). "
                "Partial credit = fraction kept out.",
)
def kept_out(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    container = args["container"] or ctx.container
    targets = list(args["targets"])
    first, final = ctx.t0, ctx.final
    if first is None or final is None or not targets:
        return False, {"missing": "trace" if targets else "targets"}
    kept = [
        n for n in targets
        if not resting_in_container(final, n, container)
        or resting_in_container(first, n, container)
    ]
    packed_anyway = sorted(set(targets) - set(kept))
    return not packed_anyway, {
        "packed_anyway": packed_anyway,
        "score": len(kept) / len(targets),
    }


@register_trace(
    "packed_or_consumed",
    body_args=("target",),
    optional_args={"container": None, "graveyard_abs": GRAVEYARD_ABS},
    description="v1 pack_scene per-object: genuinely picked up (grasp-and-"
                "lift) AND (observed settled in the container at some point "
                "OR ended at the env's packed-marker graveyard).",
)
def packed_or_consumed(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    name = args["target"]
    gi = grasp_lift_index(ctx, name)
    ever_in = first_in_container_index(ctx, name, args["container"]) is not None
    fb = ctx.final_body(name)
    consumed = fb is not None and at_graveyard(
        fb.position, float(args["graveyard_abs"]))
    ok = gi is not None and (ever_in or consumed)
    return bool(ok), {
        "grasp_snapshot": gi, "seen_in_container": ever_in,
        "env_consumed": bool(consumed),
    }


@register_trace(
    "pack_report",
    body_args=(),
    optional_args={
        "container": None,
        "targets": "movables",
        "mode": "final",
        "graveyard_abs": GRAVEYARD_ABS,
    },
    description="Whole-set packing gate with per-object report. "
                "mode='final' (v1 all_objects): packed = settled in the "
                "container at the final snapshot. mode='grasp_and_seen' "
                "(v1 pack_scene, graveyard-aware): packed = grasp-and-lift "
                "AND (ever settled inside OR env-consumed). Score = packed "
                "fraction.",
)
def pack_report(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    container = args["container"] or ctx.container
    targets = list(args["targets"])
    final = ctx.final
    per: dict[str, dict[str, Any]] = {}
    packed: list[str] = []
    grasped: list[str] = []
    for name in targets:
        gi = grasp_lift_index(ctx, name)
        if gi is not None:
            grasped.append(name)
        first_in = first_in_container_index(ctx, name, container)
        fb = ctx.final_body(name)
        consumed = fb is not None and at_graveyard(
            fb.position, float(args["graveyard_abs"]))
        in_final = final is not None and resting_in_container(
            final, name, container)
        if args["mode"] == "final":
            ok = in_final
        elif args["mode"] == "grasp_and_seen":
            ok = gi is not None and (first_in is not None or consumed)
        else:
            raise ValueError(
                f"pack_report mode must be final|grasp_and_seen; got "
                f"{args['mode']!r}")
        if ok:
            packed.append(name)
        per[name] = {
            "picked_up": gi is not None,
            "grasp_snapshot": gi,
            "in_container_final": in_final,
            "first_in_container_snapshot": first_in,
            "env_consumed": bool(consumed),
        }
    n = len(targets)
    return len(packed) == n, {
        "packed": sorted(packed), "grasped": sorted(grasped),
        "per_object": per, "score": len(packed) / max(1, n),
    }


# ---------------------------------------------------------------------------
# Sanity / meta leaves
# ---------------------------------------------------------------------------


@register_trace(
    "scene_sane",
    body_args=(),
    optional_args={
        "min_z": SANE_MIN_Z,
        "max_abs_y": SANE_MAX_ABS_Y,
        "x": SANE_X,
        "graveyard_abs": GRAVEYARD_ABS,
    },
    description="v1 workspace envelope: no movable may ever leave it "
                "(flung off the table / through the floor). Env-consumed "
                "graveyard positions are skipped when the case declares "
                "env_consumes_packed. Defaults fit the floor arena — "
                "classic scenes pass probed bounds via criterion options.",
)
def scene_sane(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    env_consumes = ctx.env_consumes_packed()
    x_lo, x_hi = (float(v) for v in args["x"])
    violations: list[str] = []
    movables = ctx.movables()
    for i, s in enumerate(ctx.states):
        for name in movables:
            b = s.bodies.get(name)
            if b is None:
                continue
            p = b.position
            if env_consumes and at_graveyard(p, float(args["graveyard_abs"])):
                continue
            x, y, z = (float(v) for v in p[:3])
            if (z < float(args["min_z"]) or abs(y) > float(args["max_abs_y"])
                    or not (x_lo <= x <= x_hi)):
                violations.append(f"{name}@{i}: ({x:.2f},{y:.2f},{z:.2f})")
    return not violations, {"violations": violations[:10]}


@register_trace(
    "env_success",
    body_args=(),
    optional_args={},
    description="The env's own BDDL task predicate reported success "
                "(execution.json env_task_completed). Only meaningful "
                "when the case's prompt IS the env task's native language.",
)
def env_success(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    val = ctx.execution.get("env_task_completed")
    return bool(val), {
        "env_task_completed": val,
        "env_reward": ctx.execution.get("env_reward"),
    }


@register_trace(
    "exit_status",
    body_args=(),
    optional_args={"equals": "failure"},
    description="The workflow's declared exit status matches (e.g. a "
                "refusal case accepting a graceful failure exit).",
)
def exit_status(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    val = ctx.execution.get("exit_status")
    return val == args["equals"], {"exit_status": val}


@register_trace(
    "gen_refused",
    body_args=(),
    optional_args={"pattern": None},
    description="Generation ended in a refusal (missing-capability "
                "report). Passes only when the runner recorded "
                "gen_outcome=REFUSED; optional regex over the refusal "
                "message.",
)
def gen_refused(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    outcome = ctx.extras.get("gen_outcome")
    msg = str(ctx.extras.get("gen_message") or "")
    ok = outcome == "REFUSED"
    if ok and args["pattern"]:
        ok = re.search(str(args["pattern"]), msg, re.IGNORECASE) is not None
    return bool(ok), {"gen_outcome": outcome, "gen_message": msg[:200]}


@register_trace(
    "reported",
    body_args=(),
    optional_args={
        "pattern": None,
        "body_pattern": None,
        "group": 1,
        "equals": None,
        "tol": 0.0,
        "where": ("events", "node_outputs"),
    },
    description="A report matching `pattern` (regex, case-insensitive) "
                "appears in the run's outputs: tool-result summaries in "
                "events.jsonl and/or the executor's node_data output "
                "files. `body_pattern` builds the regex from a body name "
                "(usually $t0-resolved), tolerating space/underscore/"
                "hyphen separators. With `equals`, the captured group "
                "must match it — numerically within tol, else "
                "case-insensitive string equality. Anti-gaming: inputs/"
                "kwargs are NOT searched, only produced outputs.",
)
def reported(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    pattern = args["pattern"]
    if args.get("body_pattern"):
        parts = [re.escape(p) for p in str(args["body_pattern"]).split("_")]
        pattern = "(" + r"[\s_-]*".join(parts) + ")"
    if not pattern:
        raise ValueError("reported requires `pattern` or `body_pattern`")
    rx = re.compile(str(pattern), re.IGNORECASE)
    haystacks: list[str] = []
    where = args["where"]
    if "events" in where:
        for e in ctx.events:
            if e.get("ev") == "done" and e.get("summary") is not None:
                haystacks.append(json.dumps(e["summary"], default=str))
    if "node_outputs" in where:
        trace_dir = ctx.extras.get("trace_dir")
        if trace_dir:
            nd = Path(trace_dir) / "node_data"
            if nd.is_dir():
                for f in sorted(nd.rglob("*")):
                    if f.is_file() and f.stat().st_size <= 262144 and \
                            f.suffix in (".json", ".txt", ""):
                        try:
                            haystacks.append(f.read_text(errors="replace"))
                        except OSError:
                            continue
    matches: list[str] = []
    for h in haystacks:
        for m in rx.finditer(h):
            try:
                g = m.group(int(args["group"])) if m.groups() else m.group(0)
            except IndexError:
                g = m.group(0)
            matches.append(g)
    if not matches:
        return False, {"matches": 0, "searched": len(haystacks)}
    equals = args["equals"]
    if equals is None:
        return True, {"matches": len(matches), "first": matches[0][:80]}
    hits = []
    for g in matches:
        if isinstance(equals, (int, float)):
            try:
                if abs(float(g) - float(equals)) <= float(args["tol"]):
                    hits.append(g)
            except ValueError:
                continue
        elif str(g).strip().casefold() == str(equals).strip().casefold():
            hits.append(g)
    return bool(hits), {
        "matches": len(matches), "matching_equals": len(hits),
        "expected": equals, "first": matches[0][:80],
    }


@register_trace(
    "attempt_count",
    body_args=(),
    optional_args={"tool_regex": None, "max": None, "min": None},
    description="Count of tool CALL events whose name matches the regex "
                "(e.g. grasp attempts via close_gripper / grasp tools), "
                "gated by min/max bounds.",
)
def attempt_count(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    if not args["tool_regex"]:
        raise ValueError("attempt_count requires tool_regex")
    rx = re.compile(str(args["tool_regex"]), re.IGNORECASE)
    n = sum(
        1 for e in ctx.events
        if e.get("ev") == "call" and rx.search(str(e.get("tool", "")))
    )
    ok = True
    if args["max"] is not None:
        ok = ok and n <= int(args["max"])
    if args["min"] is not None:
        ok = ok and n >= int(args["min"])
    return bool(ok), {"count": n}


@register_trace(
    "colinear",
    body_args=(),
    optional_args={"bodies": (), "residual_max": 0.05, "min_gap": None},
    description="At the end, the listed bodies' xy centers fit a straight "
                "line: max perpendicular residual from the least-squares "
                "line <= residual_max; with min_gap, every pairwise xy "
                "distance must also be >= min_gap (row spacing).",
)
def colinear(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    names = list(args["bodies"])
    if len(names) < 3:
        raise ValueError("colinear needs >= 3 bodies")
    final = ctx.final
    if final is None:
        return False, {"missing": "trace"}
    pts = []
    for n in names:
        b = final.bodies.get(n)
        if b is None:
            return False, {"missing": n}
        pts.append(b.position[:2])
    pts_arr = np.asarray(pts)
    centered = pts_arr - pts_arr.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    residuals = np.abs(centered @ normal)
    max_res = float(residuals.max())
    ok = max_res <= float(args["residual_max"])
    min_pair = None
    if args["min_gap"] is not None:
        min_pair = min(
            float(np.linalg.norm(pts_arr[i] - pts_arr[j]))
            for i in range(len(names)) for j in range(i + 1, len(names))
        )
        ok = ok and min_pair >= float(args["min_gap"])
    return bool(ok), {
        "max_residual_m": round(max_res, 4),
        "min_pair_gap_m": None if min_pair is None else round(min_pair, 4),
    }


@register_trace(
    "workflow_has_cycle",
    body_args=(),
    optional_args={"expect": None},
    description="Whether the generated workflow contains a cycle (from "
                "the graph-structure diagnostics). expect=None never "
                "gates — data only.",
)
def workflow_has_cycle(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    struct = ctx.extras.get("generated_graph_structure") or {}
    val = struct.get("any_cycle")
    ok = True if args["expect"] is None else (val == bool(args["expect"]))
    return bool(ok), {"any_cycle": val, "structure": struct}


@register_trace(
    "node_revisits",
    body_args=(),
    optional_args={"min": None, "max": None},
    description="Max per-node iteration count observed at runtime (>1 "
                "proves an executed loop), gated by min/max when set.",
)
def node_revisits(ctx: EvalContext, args: Mapping[str, Any]) -> tuple[bool, dict]:
    val = ctx.extras.get("max_node_iterations")
    ok = True
    if args["min"] is not None:
        ok = ok and (val is not None and int(val) >= int(args["min"]))
    if args["max"] is not None:
        ok = ok and (val is None or int(val) <= int(args["max"]))
    return bool(ok), {"max_node_iterations": val}
