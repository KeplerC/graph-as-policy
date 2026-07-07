"""Seed-level evaluation entry point (v2 of ``criteria.evaluate_seed``).

Same call shape as v1 — ``evaluate_seed(case, states, events, execution,
extras)`` over raw ``state.jsonl``/``events.jsonl`` dicts — but the
criterion comes from the case's declarative DSL spec instead of a
checker-name dispatch table.

``success_source`` routing:

- ``trace`` (default): the DSL criterion is the verdict; the env's own
  reward stays advisory metadata.
- ``env``: the env's BDDL ``check_success`` IS the task predicate (the
  prompt is the env task's native language) — an ``env_success`` check
  gates, authored checks are demoted to weighted diagnostics (their
  weights still produce the partial-progress score).
- ``both``: env_success AND the authored gating checks.
- ``gen``: no execution — the criterion is evaluated against the
  generation record (``gen_refused`` leaves) with an empty trace.
"""

from __future__ import annotations

from typing import Any

from zsp.dsl import EvalContext, evaluate_criterion, states_from_records
from zsp.dsl.score import SeedEvaluation
from zsp.state_log import hang_site


def _routed_criterion(case: Any) -> dict[str, Any]:
    criterion = dict(getattr(case, "criterion", None) or {})
    checks = [dict(c) for c in (criterion.get("checks") or [])]
    src = getattr(case, "success_source", "trace")
    if src == "env":
        for c in checks:
            c["gating"] = False
        checks.insert(0, {
            "name": "env_success", "weight": 0.0, "gating": True,
            "expr": {"env_success": {}},
        })
    elif src == "both":
        checks.insert(0, {
            "name": "env_success", "weight": 0.0, "gating": True,
            "expr": {"env_success": {}},
        })
    criterion["checks"] = checks
    return criterion


def evaluate_seed(
    case: Any,
    states: list[dict],
    events: list[dict],
    execution: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> SeedEvaluation:
    extras = dict(extras or {})
    src = getattr(case, "success_source", "trace")
    timed_out = bool(execution.get("timed_out"))

    if src != "gen" and len(states) < 2:
        verdict = "TIMEOUT" if timed_out else (
            "CRASH" if execution.get("harness_crash") else "NO_TRACE")
        ev = SeedEvaluation(
            verdict, 0.0,
            f"no usable state trace ({verdict.lower()}); "
            f"{str(execution.get('harness_crash') or execution.get('error') or '')[:200]}",
            [],
        )
    else:
        ctx = EvalContext(
            states=states_from_records(states),
            case=case,
            events=events,
            execution=execution,
            extras=extras,
        )
        ev = evaluate_criterion(case, _routed_criterion(case), ctx)
        if timed_out:
            ev.verdict = "TIMEOUT"
            ev.summary = "run killed at timeout — " + ev.summary
        elif execution.get("harness_crash"):
            ev.verdict = "CRASH"
            ev.summary = (
                f"harness crash ({str(execution['harness_crash'])[:120]}) — "
                + ev.summary
            )

    site = hang_site(events)
    if timed_out and site is not None:
        ev.diagnostics["hang"] = {
            "tool": site.get("tool"),
            "node": site.get("node"),
            "kwargs": site.get("kw"),
            "started_at_s": site.get("t"),
        }
        ev.summary += (
            f" | hung in tool {site.get('tool')!r}"
            f" (node {site.get('node')!r})"
        )
    tool_errors = [
        {"tool": e.get("tool"), "node": e.get("node"),
         "err": e.get("err", "")[:200]}
        for e in events if e.get("ev") == "error"
    ]
    if tool_errors:
        ev.diagnostics["tool_errors"] = tool_errors[:20]
    planner_errors = [e for e in tool_errors
                      if "curobo" in str(e.get("tool", ""))
                      or "plan" in str(e.get("tool", ""))]
    if planner_errors:
        ev.diagnostics["planner_errors"] = planner_errors[:10]

    # Perception-vs-ground-truth diagnostics for cases that ask for them
    # (tag "geometry_diag"; v1 hardcoded this for cream_cheese).
    if "geometry_diag" in (getattr(case, "tags", ()) or ()):
        geo = [
            {"tool": e.get("tool"), "summary": e.get("summary")}
            for e in events
            if e.get("ev") == "done"
            and str(e.get("tool", "")).startswith("geometry.")
        ]
        if geo:
            ev.diagnostics["geometry_outputs"] = geo[:30]
        gt = (getattr(case, "expected", "") or "").strip()
        if gt:
            ev.diagnostics["expected_note"] = gt[:300]

    if extras:
        ev.diagnostics.update(extras)
    return ev
