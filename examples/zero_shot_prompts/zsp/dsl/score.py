"""Check/verdict/score assembly for one criterion evaluation.

Verdict = every gating check passes (authored ``gating: true`` — the
default — plus the auto-injected invariants below). Score = the
weight-normalized sum over authored checks with ``weight > 0``; a failed
check contributes its leaf's fractional ``score`` diagnostic when the
expression is a single leaf that emits one (count fractions, relocation
progress), else 0. Partial credit is a per-primitive concern — never
expression algebra — so scores stay explainable.

Auto-injected anti-gaming invariants (reported as checks with
``"injected": true`` in their data):

1. ``scene_sane`` — always, gating, weight 0; no opt-out. Bounds come
   from ``options.scene_sane`` when a scene needs probed overrides.
2. A gating, weight-0 ``picked_up`` for every literal target of a
   gating containment/stack/on leaf that has no explicit ``picked_up``
   anywhere in the spec — unless the target is listed in
   ``options.push_ok`` or the case sets
   ``options.require_picked_up: false`` (push/slide tasks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zsp.dsl.ast import evaluate_check_expr, resolve_t0_refs
from zsp.dsl.convert import EvalContext

#: Gating containment leaves -> the arg naming the object that must have
#: been genuinely picked up for the claim to be non-gamed.
_CONTAINMENT_TARGET_ARG = {
    "in_container": "target",
    "resting_on": "target",
    "stack": "top",
    "on": "target",
    "in": "target",
}


@dataclass
class Check:
    name: str
    passed: bool
    score: float
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "score": round(float(self.score), 4),
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class SeedEvaluation:
    verdict: str            # PASS | FAIL | TIMEOUT | CRASH | NO_TRACE | ...
    score: float
    summary: str
    checks: list[Check]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": round(float(self.score), 4),
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Detail / data formatting
# ---------------------------------------------------------------------------


def _fmt_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)) and len(v) > 6:
        return f"[{len(v)} items]"
    return str(v)


def _fmt_diag(diag: dict[str, Any], limit: int = 5) -> str:
    parts = []
    for k, v in diag.items():
        if k in ("per_object", "leaves", "structure"):
            continue
        parts.append(f"{k}={_fmt_val(v)}")
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def _detail_from_leaves(leaves: list[dict], passed: bool) -> str:
    if not leaves:
        return "no leaves evaluated"
    if len(leaves) == 1:
        lf = leaves[0]
        return (
            f"{lf['predicate']}: {'ok' if lf['success'] else 'FAIL'} "
            f"({_fmt_diag(lf['diagnostics'])})"
        )
    n_ok = sum(1 for lf in leaves if lf["success"])
    head = f"{n_ok}/{len(leaves)} leaves ok"
    if passed:
        return head
    first_bad = next((lf for lf in leaves if not lf["success"]), None)
    if first_bad is None:
        return head
    return (
        f"{head}; first failure {first_bad['predicate']}@"
        f"{first_bad['path']} ({_fmt_diag(first_bad['diagnostics'])})"
    )


def _leaf_partial_score(leaves: list[dict]) -> float | None:
    """Fractional credit for a failed check: only from a single-leaf
    expression whose diagnostics carry a ``score``."""
    if len(leaves) == 1 and "score" in leaves[0]["diagnostics"]:
        try:
            return max(0.0, min(1.0, float(leaves[0]["diagnostics"]["score"])))
        except (TypeError, ValueError):
            return None
    return None


def _trim_leaves(leaves: list[dict], cap: int = 12) -> list[dict]:
    if len(leaves) <= cap:
        return leaves
    kept = [lf for lf in leaves if not lf["success"]][: cap - 1]
    return kept + [{"note": f"{len(leaves) - len(kept)} more leaves omitted"}]


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------


def _iter_leaves(node: Any):
    """Yield (leaf_name, payload) for every leaf in an expression."""
    if not isinstance(node, dict) or len(node) != 1:
        return
    [(op, payload)] = node.items()
    if op in ("and", "or"):
        for c in payload if isinstance(payload, list) else ():
            yield from _iter_leaves(c)
    elif op in ("not", "eventually", "always", "at_end", "never"):
        yield from _iter_leaves(payload)
    else:
        yield op, payload


def _containment_targets(expr: Any) -> set[str]:
    out: set[str] = set()
    for name, payload in _iter_leaves(expr):
        arg = _CONTAINMENT_TARGET_ARG.get(name)
        if arg is None:
            continue
        if isinstance(payload, dict):
            val = payload.get(arg)
        elif isinstance(payload, list) and payload:
            val = payload[0]
        else:
            val = None
        if isinstance(val, str):
            out.add(val)
    return out


def _explicit_pickups(checks: list[dict]) -> set[str]:
    out: set[str] = set()
    for c in checks:
        for name, payload in _iter_leaves(c.get("expr", {})):
            if name in ("picked_up", "packed_or_consumed"):
                val = payload.get("target") if isinstance(payload, dict) \
                    else (payload[0] if payload else None)
                if isinstance(val, str):
                    out.add(val)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def evaluate_criterion(
    case: Any, criterion: dict[str, Any], ctx: EvalContext,
) -> SeedEvaluation:
    ctx.options = dict(criterion.get("options") or {})
    try:
        resolved = resolve_t0_refs(criterion.get("checks") or [], ctx)
    except (ValueError, KeyError) as exc:
        return SeedEvaluation(
            "FAIL", 0.0,
            f"criterion $t0 resolution failed: {str(exc)[:200]}",
            [], {"t0_error": str(exc)[:300]},
        )

    checks: list[Check] = []
    weighted: list[tuple[float, float]] = []
    gating_ok = True

    for spec in resolved:
        name = spec.get("name", "?")
        try:
            passed, leaves = evaluate_check_expr(spec.get("expr"), ctx)
        except ValueError as exc:
            passed, leaves = False, [{
                "path": "$", "predicate": "?", "args": {},
                "success": False, "diagnostics": {"error": str(exc)[:300]},
            }]
        if passed:
            score = 1.0
        else:
            score = _leaf_partial_score(leaves) or 0.0
        chk = Check(
            name, passed, score, _detail_from_leaves(leaves, passed),
            {"leaves": _trim_leaves(leaves)},
        )
        checks.append(chk)
        w = float(spec.get("weight", 0) or 0)
        if w > 0:
            weighted.append((w, score))
        if spec.get("gating", True):
            gating_ok = gating_ok and passed

    # -- injected invariants ------------------------------------------------

    authored_leaf_names = {
        n for spec in resolved for n, _ in _iter_leaves(spec.get("expr", {}))
    }
    injected: list[Check] = []

    if "scene_sane" not in authored_leaf_names:
        from zsp.dsl.trace_predicates import TRACE_REGISTRY

        sane_args = dict(ctx.options.get("scene_sane") or {})
        spec_ss = TRACE_REGISTRY["scene_sane"]
        args = {**spec_ss.optional_args, **sane_args}
        ok, diag = spec_ss.evaluator(ctx, args)
        injected.append(Check(
            "scene_sane", ok, 1.0 if ok else 0.0,
            "no object left the workspace" if ok
            else f"violation(s): {diag.get('violations', [])[:2]}",
            {"injected": True, **diag},
        ))
        gating_ok = gating_ok and ok

    if ctx.options.get("require_picked_up", True):
        push_ok = set(ctx.options.get("push_ok") or ())
        explicit = _explicit_pickups(resolved)
        need = set()
        for spec in resolved:
            if spec.get("gating", True):
                need |= _containment_targets(spec.get("expr", {}))
        for target in sorted(need - explicit - push_ok):
            from zsp.dsl.trace_predicates import TRACE_REGISTRY

            spec_pu = TRACE_REGISTRY["picked_up"]
            ok, diag = spec_pu.evaluator(
                ctx, dict(spec_pu.optional_args, target=target))
            injected.append(Check(
                f"{target}_picked_up", ok, 1.0 if ok else 0.0,
                f"{target} grasp-and-lift at snapshot "
                f"{diag.get('grasp_snapshot')}" if ok else
                f"{target} was never picked up (containment claim "
                "requires a real grasp)",
                {"injected": True, **diag},
            ))
            gating_ok = gating_ok and ok

    checks = injected + checks

    # -- verdict / score / summary -------------------------------------

    if weighted:
        total_w = sum(w for w, _ in weighted)
        score = sum(w * s for w, s in weighted) / total_w
    else:
        score = 1.0 if gating_ok else 0.0

    failed = [c.name for c in checks if not c.passed]
    if gating_ok:
        summary = f"all {len(checks)} checks passed"
    else:
        first = next(c for c in checks if not c.passed)
        summary = (
            f"{len(checks) - len(failed)}/{len(checks)} checks passed; "
            f"failed: {', '.join(failed[:4])}"
            f"{'...' if len(failed) > 4 else ''} — {first.detail[:160]}"
        )

    diagnostics: dict[str, Any] = {}
    if ctx.t0_cache:
        diagnostics["t0"] = dict(ctx.t0_cache)
    return SeedEvaluation(
        "PASS" if gating_ok else "FAIL",
        score, summary, checks, diagnostics,
    )
