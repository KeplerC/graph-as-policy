"""Criterion expression walker, ``$t0`` resolver, and validation.

Expression grammar (single-key dicts, mirroring ``goal_eval``)::

    expr := {"and"|"or": [expr, ...]}
          | {"not"|"eventually"|"always"|"at_end"|"never": expr}
          | {<leaf>: payload}

Leaf resolution order: trace predicates -> benchmark state extras ->
core registry. Trace leaves consume the whole run and are only legal at
the top (trace) scope; per-state leaves at trace scope default to
``at_end``. Temporal ops nested inside a state scope collapse to the
active state (goal_eval behaviour).

``{"$t0": {<resolver>: payload}}`` value nodes compute derived referents
against the first snapshot (closest_to, extreme, largest/smallest,
midpoint, pos_of, count_within, distance); resolutions are cached on the
context and echoed into the evaluation diagnostics.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from gap.runtime.predicates.goal_eval import SimState
from gap.runtime.predicates.predicates import (
    REGISTRY as CORE_REGISTRY,
)
from gap.runtime.predicates.predicates import (
    PredicateSpec,
    _ensure_builtins_loaded,
    resolve_predicate_args,
)
from zsp.dsl.convert import EvalContext
from zsp.dsl.state_predicates import STATE_EXTRAS
from zsp.dsl.trace_predicates import TRACE_REGISTRY

OPS = frozenset({"and", "or", "not", "eventually", "always", "at_end", "never"})

#: Arg names whose value "movables" expands to the scene's movable set.
_MOVABLE_KEYS = frozenset({"targets", "among", "bodies", "allowed", "except"})

#: Leaves that must be given specific optional args to mean anything.
_REQUIRED_OPTIONALS: dict[str, tuple[str, ...]] = {
    "before": ("first", "then"),
    "attempt_count": ("tool_regex",),
    "colinear": ("bodies",),
    "near_point": ("point",),
}

_ALLOWED_OPTIONS = frozenset({
    "container", "statics", "require_picked_up", "push_ok",
    "env_consumes_packed", "scene_sane",
})

_checked_disjoint = False


def _leaf_kind(name: str) -> tuple[str, PredicateSpec] | None:
    """Resolve a leaf name -> ("trace"|"state"|"core", spec)."""
    global _checked_disjoint
    _ensure_builtins_loaded()
    if not _checked_disjoint:
        overlap = (
            (set(TRACE_REGISTRY) & set(STATE_EXTRAS))
            | (set(TRACE_REGISTRY) & set(CORE_REGISTRY))
            | (set(STATE_EXTRAS) & set(CORE_REGISTRY))
        )
        assert not overlap, f"predicate name collision across registries: {overlap}"
        _checked_disjoint = True
    if name in TRACE_REGISTRY:
        return "trace", TRACE_REGISTRY[name]
    if name in STATE_EXTRAS:
        return "state", STATE_EXTRAS[name]
    if name in CORE_REGISTRY:
        return "core", CORE_REGISTRY[name]
    return None


def _single_key(node: Any, path: str) -> tuple[str, Any]:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(
            f"criterion node at {path} must be a single-key mapping; "
            f"got {node!r}"
        )
    [(op, payload)] = node.items()
    return op, payload


def _expand_movables(args: dict[str, Any], ctx: EvalContext) -> dict[str, Any]:
    out = dict(args)
    for k, v in args.items():
        if v == "movables" and k in _MOVABLE_KEYS:
            out[k] = ctx.movables()
    return out


# ---------------------------------------------------------------------------
# $t0 resolvers
# ---------------------------------------------------------------------------


def _among(state: SimState, spec: Any, ctx: EvalContext) -> list[str]:
    names = ctx.movables() if (spec is None or spec == "movables") else list(spec)
    return [n for n in names if n in state.bodies]


def _r_closest_to(state: SimState, args: dict, ctx: EvalContext) -> str:
    anchor = state.bodies[args["anchor"]]
    cands = [n for n in _among(state, args.get("among"), ctx)
             if n != args["anchor"]]
    if not cands:
        raise ValueError(f"closest_to: no candidates among {args.get('among')}")
    return min(cands, key=lambda n: float(
        np.linalg.norm(state.bodies[n].position[:2] - anchor.position[:2])))


def _r_farthest_from(state: SimState, args: dict, ctx: EvalContext) -> str:
    anchor = state.bodies[args["anchor"]]
    cands = [n for n in _among(state, args.get("among"), ctx)
             if n != args["anchor"]]
    if not cands:
        raise ValueError("farthest_from: no candidates")
    return max(cands, key=lambda n: float(
        np.linalg.norm(state.bodies[n].position[:2] - anchor.position[:2])))


def _r_extreme(state: SimState, args: dict, ctx: EvalContext) -> str:
    axis = "xyz".index(str(args.get("axis", "y")))
    mode = str(args.get("mode", "min"))
    cands = _among(state, args.get("among"), ctx)
    if not cands:
        raise ValueError("extreme: no candidates")
    key = lambda n: float(state.bodies[n].position[axis])  # noqa: E731
    return min(cands, key=key) if mode == "min" else max(cands, key=key)


def _r_middle_of(state: SimState, args: dict, ctx: EvalContext) -> str:
    """Median body along an axis (ordinal 'the middle one' of a row)."""
    axis = "xyz".index(str(args.get("axis", "y")))
    cands = _among(state, args.get("among"), ctx)
    if len(cands) % 2 == 0 or len(cands) < 3:
        raise ValueError(f"middle_of needs an odd number >= 3; got {cands}")
    ordered = sorted(cands,
                     key=lambda n: float(state.bodies[n].position[axis]))
    return ordered[len(ordered) // 2]


def _size(state: SimState, name: str, by: str) -> float:
    ext = state.bodies[name].aabb_upper - state.bodies[name].aabb_lower
    if by == "volume":
        return float(np.prod(ext))
    if by == "height":
        return float(ext[2])
    raise ValueError(f"size measure must be volume|height; got {by!r}")


def _r_largest(state: SimState, args: dict, ctx: EvalContext) -> str:
    cands = _among(state, args.get("among"), ctx)
    return max(cands, key=lambda n: _size(state, n, args.get("by", "volume")))


def _r_smallest(state: SimState, args: dict, ctx: EvalContext) -> str:
    cands = _among(state, args.get("among"), ctx)
    return min(cands, key=lambda n: _size(state, n, args.get("by", "volume")))


def _r_midpoint(state: SimState, args: dict, ctx: EvalContext) -> list[float]:
    a = state.bodies[args["a"]].position
    b = state.bodies[args["b"]].position
    return [round(float(v), 5) for v in (a + b) / 2.0]


def _r_pos_of(state: SimState, args: dict, ctx: EvalContext) -> list[float]:
    p = state.bodies[args["body"]].position
    off = np.asarray(args.get("offset", (0.0, 0.0, 0.0)), dtype=np.float64)
    return [round(float(v), 5) for v in p + off]


def _r_count_within(state: SimState, args: dict, ctx: EvalContext) -> int:
    anchor = state.bodies[args["anchor"]]
    r = float(args["radius_m"])
    cands = [n for n in _among(state, args.get("among"), ctx)
             if n != args["anchor"]]
    return sum(
        1 for n in cands
        if float(np.linalg.norm(
            state.bodies[n].position[:2] - anchor.position[:2])) <= r
    )


def _r_distance(state: SimState, args: dict, ctx: EvalContext) -> float:
    a = state.bodies[args["a"]].position
    b = state.bodies[args["b"]].position
    axis = str(args.get("axis", "xyz"))
    scale = float(args.get("scale", 1.0))   # e.g. 100 -> centimeters
    if axis == "xy":
        d = float(np.linalg.norm(a[:2] - b[:2]))
    elif axis == "xyz":
        d = float(np.linalg.norm(a - b))
    else:
        d = abs(float(a["xyz".index(axis)] - b["xyz".index(axis)]))
    return round(d * scale, 5)


RESOLVERS = {
    "closest_to": _r_closest_to,
    "farthest_from": _r_farthest_from,
    "extreme": _r_extreme,
    "middle_of": _r_middle_of,
    "largest": _r_largest,
    "smallest": _r_smallest,
    "midpoint": _r_midpoint,
    "pos_of": _r_pos_of,
    "count_within": _r_count_within,
    "distance": _r_distance,
}


def resolve_t0_refs(node: Any, ctx: EvalContext) -> Any:
    """Deep-copy ``node`` with every ``{"$t0": ...}`` value resolved."""
    if isinstance(node, dict):
        if len(node) == 1 and "$t0" in node:
            name, payload = _single_key(node["$t0"], "$t0")
            fn = RESOLVERS.get(name)
            if fn is None:
                raise ValueError(
                    f"unknown $t0 resolver {name!r}; "
                    f"known: {sorted(RESOLVERS)}"
                )
            key = json.dumps(node["$t0"], sort_keys=True, default=str)
            if key not in ctx.t0_cache:
                t0 = ctx.t0
                if t0 is None:
                    raise ValueError(
                        f"$t0 resolver {name!r} needs a snapshot but the "
                        "trace has none"
                    )
                ctx.t0_cache[key] = fn(t0, dict(payload), ctx)
            return ctx.t0_cache[key]
        return {k: resolve_t0_refs(v, ctx) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_t0_refs(v, ctx) for v in node]
    return node


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_check_expr(
    expr: Any, ctx: EvalContext,
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate one check expression; returns (passed, leaf records)."""
    leaves: list[dict[str, Any]] = []
    ok = _eval_trace_node(expr, ctx, path="$", leaves=leaves)
    return bool(ok), leaves


def _record_leaf(
    leaves: list[dict], path: str, name: str, args: Any,
    success: bool, diagnostics: dict,
) -> bool:
    leaves.append({
        "path": path, "predicate": name,
        "args": {k: v for k, v in (args or {}).items() if k != "ctx"},
        "success": bool(success), "diagnostics": diagnostics or {},
    })
    return bool(success)


def _eval_leaf(
    name: str, payload: Any, ctx: EvalContext,
    state: SimState | None, path: str, leaves: list[dict],
) -> bool:
    kind_spec = _leaf_kind(name)
    if kind_spec is None:
        raise ValueError(f"unknown predicate {name!r} at {path}")
    kind, spec = kind_spec
    args = _expand_movables(resolve_predicate_args(spec, payload), ctx)
    if kind == "trace":
        if state is not None:
            raise ValueError(
                f"trace predicate {name!r} at {path} cannot appear inside "
                "a temporal scope (it already consumes the whole trace)"
            )
        success, diag = spec.evaluator(ctx, args)
        return _record_leaf(leaves, path, name, args, success, diag)
    # Per-state leaf: default to the final snapshot at trace scope.
    target_state = state
    if target_state is None:
        target_state = ctx.final
        path = f"{path}@end"
    if target_state is None:
        return _record_leaf(
            leaves, path, name, args, False, {"error": "empty trace"})
    try:
        success, diag = spec.evaluator(target_state, args)
    except KeyError as exc:
        success, diag = False, {
            "error": "missing_body",
            "message": str(exc).strip("'")[:200],
        }
    return _record_leaf(leaves, path, name, args, success, diag)


def _states_with_bodies(ctx: EvalContext) -> list[tuple[int, SimState]]:
    return [(i, s) for i, s in enumerate(ctx.states) if s.bodies]


def _eval_trace_node(
    node: Any, ctx: EvalContext, *, path: str, leaves: list[dict],
) -> bool:
    op, payload = _single_key(node, path)
    if op == "and":
        return all(
            _eval_trace_node(c, ctx, path=f"{path}.and[{i}]", leaves=leaves)
            for i, c in enumerate(payload)
        )
    if op == "or":
        return any(
            _eval_trace_node(c, ctx, path=f"{path}.or[{i}]", leaves=leaves)
            for i, c in enumerate(payload)
        )
    if op == "not":
        return not _eval_trace_node(payload, ctx, path=f"{path}.not",
                                    leaves=leaves)
    if op == "eventually":
        return any(
            _eval_state_node(payload, ctx, s, path=f"{path}.eventually[t={i}]",
                             leaves=leaves)
            for i, s in _states_with_bodies(ctx)
        )
    if op == "never":
        return not any(
            _eval_state_node(payload, ctx, s, path=f"{path}.never[t={i}]",
                             leaves=leaves)
            for i, s in _states_with_bodies(ctx)
        )
    if op == "always":
        pairs = _states_with_bodies(ctx)
        if not pairs:
            return False
        return all(
            _eval_state_node(payload, ctx, s, path=f"{path}.always[t={i}]",
                             leaves=leaves)
            for i, s in pairs
        )
    if op == "at_end":
        final = ctx.final
        if final is None:
            return _record_leaf(leaves, f"{path}.at_end", "at_end", {},
                                False, {"error": "empty trace"})
        return _eval_state_node(payload, ctx, final, path=f"{path}.at_end",
                                leaves=leaves)
    return _eval_leaf(op, payload, ctx, None, path, leaves)


def _eval_state_node(
    node: Any, ctx: EvalContext, state: SimState, *,
    path: str, leaves: list[dict],
) -> bool:
    op, payload = _single_key(node, path)
    if op == "and":
        return all(
            _eval_state_node(c, ctx, state, path=f"{path}.and[{i}]",
                             leaves=leaves)
            for i, c in enumerate(payload)
        )
    if op == "or":
        return any(
            _eval_state_node(c, ctx, state, path=f"{path}.or[{i}]",
                             leaves=leaves)
            for i, c in enumerate(payload)
        )
    if op == "not":
        return not _eval_state_node(payload, ctx, state, path=f"{path}.not",
                                    leaves=leaves)
    if op in ("eventually", "always", "at_end", "never"):
        # Re-entrant temporal op collapses to the active state (goal_eval).
        inner = _eval_state_node(payload, ctx, state, path=f"{path}.{op}",
                                 leaves=leaves)
        return not inner if op == "never" else inner
    return _eval_leaf(op, payload, ctx, state, path, leaves)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_T0_DUMMY = "\x00t0"


def _substitute_t0_dummies(node: Any) -> Any:
    if isinstance(node, dict):
        if len(node) == 1 and "$t0" in node:
            return _T0_DUMMY
        return {k: _substitute_t0_dummies(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_t0_dummies(v) for v in node]
    return node


def _iter_t0_nodes(node: Any):
    if isinstance(node, dict):
        if len(node) == 1 and "$t0" in node:
            yield node["$t0"]
            return
        for v in node.values():
            yield from _iter_t0_nodes(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_t0_nodes(v)


def validate_criterion(
    criterion: Any, known_bodies: set[str] | None = None,
) -> None:
    """Raise ``ValueError`` on a malformed criterion spec.

    With ``known_bodies`` (from a probed layout), also checks that every
    literal body name referenced by a leaf or a $t0 resolver exists.
    """
    errors: list[str] = []
    if not isinstance(criterion, dict):
        raise ValueError(f"criterion must be a dict; got {type(criterion).__name__}")
    checks = criterion.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("criterion.checks must be a non-empty list")
    options = criterion.get("options") or {}
    unknown_opts = set(options) - _ALLOWED_OPTIONS
    if unknown_opts:
        errors.append(f"unknown options: {sorted(unknown_opts)}")
    unknown_keys = set(criterion) - {"checks", "options"}
    if unknown_keys:
        errors.append(f"unknown criterion keys: {sorted(unknown_keys)}")

    names_seen: set[str] = set()
    for i, c in enumerate(checks):
        where = f"checks[{i}]"
        if not isinstance(c, dict):
            errors.append(f"{where}: must be a dict")
            continue
        name = c.get("name")
        if not name or not isinstance(name, str):
            errors.append(f"{where}: missing/blank name")
        elif name in names_seen:
            errors.append(f"{where}: duplicate check name {name!r}")
        else:
            names_seen.add(name)
        w = c.get("weight", 0)
        if not isinstance(w, (int, float)) or w < 0:
            errors.append(f"{where} ({name}): weight must be >= 0")
        if not isinstance(c.get("gating", True), bool):
            errors.append(f"{where} ({name}): gating must be a bool")
        unknown = set(c) - {"name", "weight", "gating", "expr"}
        if unknown:
            errors.append(f"{where} ({name}): unknown keys {sorted(unknown)}")
        if "expr" not in c:
            errors.append(f"{where} ({name}): missing expr")
            continue
        try:
            _validate_node(c["expr"], scope="trace", path=f"{where}.expr",
                           known_bodies=known_bodies, errors=errors)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(
            "invalid criterion:\n  - " + "\n  - ".join(errors[:20]))


def _validate_node(
    node: Any, *, scope: str, path: str,
    known_bodies: set[str] | None, errors: list[str],
) -> None:
    op, payload = _single_key(node, path)
    if op in ("and", "or"):
        if not isinstance(payload, list) or not payload:
            errors.append(f"{path}: {op} requires a non-empty list")
            return
        for i, c in enumerate(payload):
            _validate_node(c, scope=scope, path=f"{path}.{op}[{i}]",
                           known_bodies=known_bodies, errors=errors)
        return
    if op == "not":
        _validate_node(payload, scope=scope, path=f"{path}.not",
                       known_bodies=known_bodies, errors=errors)
        return
    if op in ("eventually", "always", "at_end", "never"):
        _validate_node(payload, scope="state", path=f"{path}.{op}",
                       known_bodies=known_bodies, errors=errors)
        return

    kind_spec = _leaf_kind(op)
    if kind_spec is None:
        errors.append(
            f"{path}: unknown predicate {op!r} (known: trace="
            f"{sorted(TRACE_REGISTRY)[:8]}..., state={sorted(STATE_EXTRAS)}, "
            f"core={sorted(CORE_REGISTRY)})"
        )
        return
    kind, spec = kind_spec
    if kind == "trace" and scope != "trace":
        errors.append(
            f"{path}: trace predicate {op!r} cannot appear inside a "
            "temporal scope"
        )
        return
    try:
        args = resolve_predicate_args(spec, _substitute_t0_dummies(payload))
    except ValueError as exc:
        errors.append(f"{path}: {exc}")
        return
    for req in _REQUIRED_OPTIONALS.get(op, ()):
        if args.get(req) is None:
            errors.append(f"{path}: {op!r} requires {req!r}")
    if op == "ee_near" and args.get("point") is None and args.get("body") is None:
        errors.append(f"{path}: ee_near requires point or body")
    if op == "reported" and args.get("pattern") is None \
            and args.get("body_pattern") is None:
        errors.append(f"{path}: reported requires pattern or body_pattern")
    if op == "before":
        for k in ("first", "then"):
            ev = args.get(k)
            if isinstance(ev, dict) and len(ev) == 1:
                [(ekind, epayload)] = ev.items()
                if ekind not in ("enters_container", "picked_up"):
                    errors.append(f"{path}: before.{k} kind {ekind!r} unknown")
                elif not isinstance(epayload, dict) or "target" not in epayload:
                    errors.append(f"{path}: before.{k} needs a target")
                elif known_bodies is not None:
                    t = epayload["target"]
                    if isinstance(t, str) and t != _T0_DUMMY \
                            and t not in known_bodies:
                        errors.append(
                            f"{path}: before.{k} target {t!r} not in layout")
            elif ev != _T0_DUMMY:
                errors.append(f"{path}: before.{k} must be an event spec")

    if known_bodies is not None:
        for arg_name in spec.body_args:
            val = args.get(arg_name)
            if isinstance(val, str) and val != _T0_DUMMY \
                    and val not in known_bodies:
                errors.append(
                    f"{path}: {op}.{arg_name}={val!r} not in layout bodies")
        for key in ("targets", "allowed", "except", "bodies"):
            val = args.get(key)
            if isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, str) and v != _T0_DUMMY \
                            and v not in known_bodies:
                        errors.append(
                            f"{path}: {op}.{key} entry {v!r} not in layout")

    # Validate $t0 payloads referenced anywhere in this leaf.
    for t0_node in _iter_t0_nodes(payload):
        try:
            rname, rpayload = _single_key(t0_node, f"{path}.$t0")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if rname not in RESOLVERS:
            errors.append(
                f"{path}: unknown $t0 resolver {rname!r} "
                f"(known: {sorted(RESOLVERS)})"
            )
            continue
        if not isinstance(rpayload, dict):
            errors.append(f"{path}: $t0.{rname} payload must be a dict")
            continue
        if known_bodies is not None:
            for key in ("anchor", "a", "b", "body"):
                v = rpayload.get(key)
                if isinstance(v, str) and v not in known_bodies:
                    errors.append(
                        f"{path}: $t0.{rname}.{key}={v!r} not in layout")
            among = rpayload.get("among")
            if isinstance(among, (list, tuple)):
                for v in among:
                    if isinstance(v, str) and v not in known_bodies:
                        errors.append(
                            f"{path}: $t0.{rname}.among entry {v!r} "
                            "not in layout")
