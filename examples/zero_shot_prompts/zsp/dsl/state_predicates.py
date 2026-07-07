"""Per-state predicates this benchmark adds on top of the core registry.

Same evaluator contract as :mod:`gap.runtime.predicates.predicate_evaluators`
(``(SimState, resolved_args) -> (bool, diagnostics)``) and the same
:class:`PredicateSpec` descriptor — registered in a LOCAL registry so the
core vocabulary used by LLM goal authoring stays unpolluted. Semantics
marked "v1" reproduce ``criteria.py`` exactly (the migration parity gate
depends on it).

Data the core ``BodyView`` can't carry (held body, gripper fraction, EE
position, per-body speed, named joints) is read from
``SimState.metadata`` (see :mod:`zsp.dsl.convert`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from gap.runtime.predicates.goal_eval import SimState
from gap.runtime.predicates.predicates import PredicateSpec

# v1 constants (criteria.py), each measured or an engine default.
CONTAIN_TOL_XY = 0.02      # Body.contains default tol_xy_m
CONTAIN_TOL_Z = 0.02       # Body.contains default tol_z_m
SETTLED_SPEED = 0.08       # Body.is_settled default
GRIP_CLOSED_MAX = 0.9      # gripper open-fraction while actually holding
LIFT_MIN = 0.03            # center must rise >= 3 cm to count as picked up
ON_TABLE_MAX_BOTTOM_Z = 0.03

STATE_EXTRAS: dict[str, PredicateSpec] = {}


def register_extra(
    name: str,
    *,
    body_args: Sequence[str],
    optional_args: Mapping[str, Any] | None = None,
    description: str = "",
) -> Callable:
    def deco(fn: Callable) -> Callable:
        if name in STATE_EXTRAS:
            raise ValueError(f"state extra {name!r} already registered")
        STATE_EXTRAS[name] = PredicateSpec(
            name=name,
            body_args=tuple(body_args),
            optional_args=dict(optional_args or {}),
            evaluator=fn,
            description=description,
        )
        return fn

    return deco


# ---------------------------------------------------------------------------
# Shared geometry helpers (also used by trace predicates)
# ---------------------------------------------------------------------------


def center_in_container(
    state: SimState,
    target: str,
    container: str,
    *,
    tol_xy: float = CONTAIN_TOL_XY,
    tol_z: float = CONTAIN_TOL_Z,
    use_cavity: bool = True,
) -> bool:
    """v1 ``in_basket``: center inside the container's AABB with +/-2 cm
    slack — against the interior cavity when the trace carries one."""
    t = state.bodies.get(target)
    c = state.bodies.get(container)
    if t is None or c is None:
        return False
    if use_cavity and c.cavity_lower is not None and c.cavity_upper is not None:
        lo, hi = c.cavity_lower, c.cavity_upper
    else:
        lo, hi = c.aabb_lower, c.aabb_upper
    p = t.position
    return bool(
        lo[0] - tol_xy <= p[0] <= hi[0] + tol_xy
        and lo[1] - tol_xy <= p[1] <= hi[1] + tol_xy
        and lo[2] - tol_z <= p[2] <= hi[2] + tol_z
    )


def is_settled(state: SimState, name: str,
               max_speed: float = SETTLED_SPEED) -> bool:
    return float(state.metadata.get("vel", {}).get(name, 0.0)) < max_speed


def is_held(state: SimState, name: str) -> bool:
    return state.metadata.get("held") == name


def container_contact(state: SimState, target: str, container: str) -> bool:
    """Target rests IN the container: it touches the container directly, OR
    it touches another body that is itself inside the container (a
    previously-packed item). Transitive containment — resting on already-
    packed objects still counts as packed, matching how a basket fills up
    (and VAB's contact-free ``contained_in``). Rejects only objects that
    touch nothing inside the container (e.g. perched only on the table)."""
    t = state.bodies.get(target)
    if t is None:
        return False
    if container in t.contacts:
        return True
    for other in t.contacts:
        if other == container or other not in state.bodies:
            continue
        # a supporting body that is itself centered inside the container
        if center_in_container(state, other, container):
            return True
    return False


def resting_in_container(
    state: SimState, target: str, container: str,
    *,
    require_contact: str | bool = "auto",
) -> bool:
    """v1 ``resting_in_basket`` (+ optional schema-2 transitive-contact check)."""
    if target not in state.bodies:
        return False
    if not center_in_container(state, target, container):
        return False
    if is_held(state, target):
        return False
    if not is_settled(state, target):
        return False
    want_contact = (
        int(state.metadata.get("schema", 1)) >= 2
        if require_contact == "auto" else bool(require_contact)
    )
    if want_contact:
        return container_contact(state, target, container)
    return True


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------


@register_extra(
    "in_container",
    body_args=("target", "container"),
    optional_args={
        "tol_xy_m": CONTAIN_TOL_XY,
        "tol_z_m": CONTAIN_TOL_Z,
        "require_settled": True,
        "require_released": True,
        "require_contact": "auto",
    },
    description="v1 resting_in_basket: center inside the container's cavity "
                "(or outer AABB when no cavity is recorded) with +/-2 cm "
                "slack, released and settled. require_contact='auto' also "
                "demands target-container contact on schema-2 traces "
                "(physically implied for a settled contained object); v1 "
                "traces carry no contacts so the check is skipped there.",
)
def in_container(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    target, container = args["target"], args["container"]
    t = state.bodies.get(target)
    c = state.bodies.get(container)
    if t is None or c is None:
        return False, {"missing": target if t is None else container}
    contain = center_in_container(
        state, target, container,
        tol_xy=float(args["tol_xy_m"]), tol_z=float(args["tol_z_m"]),
    )
    held = is_held(state, target)
    settled = is_settled(state, target)
    want_contact = (
        int(state.metadata.get("schema", 1)) >= 2
        if args["require_contact"] == "auto" else bool(args["require_contact"])
    )
    contact = container_contact(state, target, container) if want_contact else True
    ok = (
        contain
        and (not args["require_released"] or not held)
        and (not args["require_settled"] or settled)
        and contact
    )
    return bool(ok), {
        "contain": bool(contain), "held": bool(held),
        "settled": bool(settled),
        "contact": bool(contact) if want_contact else "not_required",
    }


@register_extra(
    "resting_on",
    body_args=("target", "support"),
    optional_args={
        "max_above": 0.035,
        "min_below": -0.015,
        "require_settled": True,
        "require_released": True,
    },
    description="v1 resting_on: target bottom within [min_below, max_above] "
                "of the support's top (is_on_strict-style asymmetric window), "
                "xy centroid inside the support's xy AABB (+2 cm), released "
                "and settled. Raise max_above for nesting (bowls into bowls).",
)
def resting_on(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    target, support = args["target"], args["support"]
    t = state.bodies.get(target)
    sup = state.bodies.get(support)
    if t is None or sup is None:
        return False, {"missing": target if t is None else support}
    dz = float(t.aabb_lower[2]) - float(sup.aabb_upper[2])
    p = t.position
    lo, hi = sup.aabb_lower, sup.aabb_upper
    inside_xy = bool(
        lo[0] - CONTAIN_TOL_XY <= p[0] <= hi[0] + CONTAIN_TOL_XY
        and lo[1] - CONTAIN_TOL_XY <= p[1] <= hi[1] + CONTAIN_TOL_XY
    )
    held = is_held(state, target)
    settled = is_settled(state, target)
    ok = (
        float(args["min_below"]) <= dz <= float(args["max_above"])
        and inside_xy
        and (not args["require_released"] or not held)
        and (not args["require_settled"] or settled)
    )
    return bool(ok), {
        "dz_m": round(dz, 4), "inside_xy": inside_xy,
        "held": bool(held), "settled": bool(settled),
    }


@register_extra(
    "on_table",
    body_args=("target",),
    optional_args={
        "container": "basket",
        "max_bottom_z": ON_TABLE_MAX_BOTTOM_Z,
        "require_settled": False,
        "require_released": False,
    },
    description="v1 on_table: bottom face within max_bottom_z of the "
                "tabletop (z~0 in the floor arena — classic scenes must "
                "pass a probed threshold) and NOT inside the container.",
)
def on_table(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    target = args["target"]
    t = state.bodies.get(target)
    if t is None:
        return False, {"missing": target}
    bottom = float(t.aabb_lower[2])
    in_cont = center_in_container(state, target, args["container"])
    held = is_held(state, target)
    settled = is_settled(state, target)
    ok = (
        bottom <= float(args["max_bottom_z"])
        and not in_cont
        and (not args["require_released"] or not held)
        and (not args["require_settled"] or settled)
    )
    return bool(ok), {
        "bottom_z": round(bottom, 4), "in_container": bool(in_cont),
        "held": bool(held), "settled": bool(settled),
    }


@register_extra(
    "settled",
    body_args=("target",),
    optional_args={"max_speed": SETTLED_SPEED},
    description="Linear speed below max_speed (v1/Body.is_settled default "
                "0.08 m/s), from the recorded per-body velocity.",
)
def settled(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    v = float(state.metadata.get("vel", {}).get(args["target"], 0.0))
    ok = v < float(args["max_speed"])
    return ok, {"speed": round(v, 4)}


@register_extra(
    "held",
    body_args=("target",),
    optional_args={},
    description="The engine's held_body() reports this object (recorded "
                "as the snapshot's top-level `held`).",
)
def held(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    h = state.metadata.get("held")
    return h == args["target"], {"held": h}


@register_extra(
    "near_point",
    body_args=("target",),
    optional_args={"point": None, "radius_m": 0.05, "axis": "xy"},
    description="Target center within radius_m of an explicit point "
                "(usually a $t0-resolved value), measured in the given "
                "axis subset (xy default, or xyz).",
)
def near_point(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    if args["point"] is None:
        raise ValueError("near_point requires a `point` (e.g. a $t0 value)")
    t = state.bodies.get(args["target"])
    if t is None:
        return False, {"missing": args["target"]}
    point = np.asarray(args["point"], dtype=np.float64)
    axis = str(args["axis"])
    if axis == "xy":
        dist = float(np.linalg.norm(t.position[:2] - point[:2]))
    elif axis == "xyz":
        dist = float(np.linalg.norm(t.position[:3] - point[:3]))
    else:
        i = "xyz".index(axis)
        dist = abs(float(t.position[i] - point[i]))
    ok = dist <= float(args["radius_m"])
    return ok, {"dist_m": round(dist, 4), "radius_m": args["radius_m"]}


@register_extra(
    "offset_along",
    body_args=("target", "reference"),
    optional_args={
        "axis": "y",
        "lo": 0.0,
        "hi": None,
        "signed": False,
        "perp_max": None,
    },
    description="v1 lateral_of geometry: target-minus-reference offset "
                "along one axis within [lo, hi] (absolute value unless "
                "signed), optionally bounding the perpendicular xy drift.",
)
def offset_along(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.bodies.get(args["target"])
    r = state.bodies.get(args["reference"])
    if t is None or r is None:
        return False, {"missing": args["target"] if t is None else args["reference"]}
    axis = str(args["axis"])
    i = "xyz".index(axis)
    d = float(t.position[i] - r.position[i])
    val = d if args["signed"] else abs(d)
    hi = float("inf") if args["hi"] is None else float(args["hi"])
    ok = float(args["lo"]) <= val <= hi
    diag: dict[str, Any] = {f"d{axis}_m": round(d, 4)}
    if args["perp_max"] is not None and axis in ("x", "y"):
        j = 1 - i  # the other horizontal axis
        perp = abs(float(t.position[j] - r.position[j]))
        diag[f"d{'xy'[j]}_m"] = round(perp, 4)
        ok = ok and perp <= float(args["perp_max"])
    return bool(ok), diag


@register_extra(
    "joint_open",
    body_args=("body",),
    optional_args={"joint": None, "threshold": 0.0, "mode": "lt"},
    description="A named (or any) non-robot joint recorded for `body` "
                "satisfies the threshold comparison (gt|ge|lt|le|abs_gt|"
                "abs_lt). LIBERO drawers OPEN toward negative qpos "
                "(probed: wooden_cabinet slide range [-0.16, 0.01]) — "
                "hence the lt default. Requires a schema-2 trace.",
)
def joint_open(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    body = args["body"]
    joints = dict(state.metadata.get("joints", {}).get(body, {}))
    joints.update(state.metadata.get("joints", {}).get("", {}))
    if args["joint"] is not None:
        joints = {k: v for k, v in joints.items() if k == args["joint"]}
    if not joints:
        return False, {
            "reason": "no recorded joints for body (schema-2 trace needed?)",
            "body": body, "joint": args["joint"],
        }
    th = float(args["threshold"])
    mode = str(args["mode"])
    cmp = {
        "gt": lambda q: q > th, "ge": lambda q: q >= th,
        "lt": lambda q: q < th, "le": lambda q: q <= th,
        "abs_gt": lambda q: abs(q) > th, "abs_lt": lambda q: abs(q) < th,
    }.get(mode)
    if cmp is None:
        raise ValueError(f"joint_open mode must be gt|ge|lt|le|abs_gt|abs_lt; got {mode!r}")
    hits = {k: v for k, v in joints.items() if cmp(float(v))}
    return bool(hits), {"joints": joints, "satisfying": sorted(hits),
                        "threshold": th, "mode": mode}


@register_extra(
    "count_in",
    body_args=("container",),
    optional_args={
        "targets": "movables",
        "count": None,
        "min_count": None,
        "max_count": None,
        "require_contact": "auto",
    },
    description="How many of `targets` are settled inside `container` "
                "(v1 resting_in_basket per target). Gate with count=<int> "
                "(exact), count='all', or min_count/max_count bounds. "
                "count='all' emits a fractional `score` diagnostic "
                "(v1 all_objects partial credit).",
)
def count_in(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    targets = list(args["targets"]) if not isinstance(args["targets"], str) else []
    if isinstance(args["targets"], str):
        raise ValueError(
            "count_in targets must be a list (the walker expands 'movables')"
        )
    container = args["container"]
    packed = [
        t for t in targets
        if resting_in_container(
            state, t, container, require_contact=args["require_contact"],
        )
    ]
    n = len(packed)
    count = args["count"]
    ok = True
    diag: dict[str, Any] = {
        "count": n, "packed": sorted(packed), "of": len(targets),
    }
    if count == "all":
        ok = n == len(targets)
        diag["score"] = n / max(1, len(targets))
    elif count is not None:
        ok = n == int(count)
    if args["min_count"] is not None:
        ok = ok and n >= int(args["min_count"])
    if args["max_count"] is not None:
        ok = ok and n <= int(args["max_count"])
    return bool(ok), diag


@register_extra(
    "ee_near",
    body_args=(),
    optional_args={
        "point": None,
        "body": None,
        "offset": (0.0, 0.0, 0.0),
        "tol_m": 0.05,
        "tol_xy_m": None,
        "tol_z_m": None,
    },
    description="The recorded end-effector position is within tol of an "
                "explicit point or of `body`'s center plus `offset` "
                "(evaluated against the SAME snapshot — pass a $t0 point "
                "to target the initial pose instead). tol_xy_m/tol_z_m "
                "split the tolerance anisotropically when set.",
)
def ee_near(state: SimState, args: Mapping[str, Any]) -> tuple[bool, dict]:
    ee = state.metadata.get("ee")
    if ee is None:
        return False, {"reason": "no ee recorded"}
    if args["point"] is not None:
        target = np.asarray(args["point"], dtype=np.float64)
    elif args["body"] is not None:
        b = state.bodies.get(str(args["body"]))
        if b is None:
            return False, {"missing": args["body"]}
        target = b.position + np.asarray(args["offset"], dtype=np.float64)
    else:
        raise ValueError("ee_near requires `point` or `body`")
    ee = np.asarray(ee, dtype=np.float64)
    d_xy = float(np.linalg.norm(ee[:2] - target[:2]))
    d_z = abs(float(ee[2] - target[2]))
    d = float(np.linalg.norm(ee - target))
    if args["tol_xy_m"] is not None or args["tol_z_m"] is not None:
        ok = (
            d_xy <= float(args["tol_xy_m"] or float("inf"))
            and d_z <= float(args["tol_z_m"] or float("inf"))
        )
    else:
        ok = d <= float(args["tol_m"])
    return bool(ok), {
        "dist_m": round(d, 4), "dist_xy_m": round(d_xy, 4),
        "dist_z_m": round(d_z, 4), "ee": [round(float(v), 4) for v in ee],
    }
