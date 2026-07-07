"""Run one benchmark case end-to-end: generate -> execute -> score.

    MUJOCO_GL=egl uv run python \
        examples/zero_shot_prompts/run_prompt.py \
        --case soup_basket --out outputs/zero_shot_prompts/dev

Pipeline per case:

1. ``gap.agent.generate_sync(prompt)`` — the prompt verbatim, once per
   case (cached in ``<case>/gen/task_00``; ``--force-generate`` refreshes).
2. For each seed: a fresh sim connector (with the case's
   ``env_overrides`` exported first), ``StateRecorder`` (trace schema 2)
   around the tool dispatch, ``gap.execute(..., checkpoints="warn")``,
   video saved.
3. ``zsp.evaluate.evaluate_seed`` over the recorded state trace ->
   ``seed<k>/result_seed.json``; the case aggregate (all seeds must
   pass, unless the case declares ``aggregate="any_seed"``) ->
   ``<case>/result.json``.

Phases: ``--gen-only`` stops after generation (no sim, cheap to fan
out); ``--exec-only`` requires the cached generation. ``--re-eval``
re-scores existing traces from disk without sim or LLM — also how a
timed-out (hung) run gets its post-mortem verdict via ``run_all.py``.
``--graph DIR`` skips generation and executes a given workflow dir —
the reward-oracle path: score ANY graph against a case's criterion.

Refusal-aware cases (``success_source="gen"`` or a criterion using the
``gen_refused`` leaf) route a generation refusal INTO criterion
evaluation instead of short-circuiting to the GEN_REFUSED verdict.

Exit codes: 0 pass, 1 fail, 2 infra crash, 3 generation error/refusal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from zsp import diagnostics  # noqa: E402
from zsp.evaluate import evaluate_seed  # noqa: E402
from zsp.registry import CASES  # noqa: E402
from zsp.runner.gpu import pin_render_gpu, relocate_tmpdir  # noqa: E402
from zsp.schema import Case  # noqa: E402
from zsp.state_log import StateRecorder, load_jsonl  # noqa: E402

logger = logging.getLogger("zero_shot_prompts")


def _default_gcp_project() -> str | None:
    """Project for the vertex provider: the ADC quota project on this
    machine (gap.agent.llm reads $GOOGLE_CLOUD_PROJECT)."""
    adc = Path.home() / ".config/gcloud/application_default_credentials.json"
    try:
        return json.loads(adc.read_text()).get("quota_project_id") or None
    except Exception:
        return None


def env_hygiene() -> None:
    """Rendering/GPU pin (pre-set values win — run_all's round-robin
    pinning relies on that), TMPDIR off /tmp, vertex project backfill,
    and GAP_DECIDE_TRUST_ENV scrubbed (a generated decide-loop must not
    trust the env's own reward — it measures a different task)."""
    pin_render_gpu()
    relocate_tmpdir()
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        proj = _default_gcp_project()
        if proj:
            os.environ["GOOGLE_CLOUD_PROJECT"] = proj
    os.environ.pop("GAP_DECIDE_TRUST_ENV", None)
    # Make the example-local libero_zsp_* task suites discoverable by the
    # loader (GAP_EXTRA_TASK_ROOTS is searched before the vab root). Needed
    # whenever a case's sim_task is an authored zsp scene; prepended so a
    # pre-set value still wins.
    tasks_dir = str(_HERE / "tasks")
    existing = os.environ.get("GAP_EXTRA_TASK_ROOTS", "")
    parts = [p for p in existing.split(":") if p]
    if tasks_dir not in parts:
        os.environ["GAP_EXTRA_TASK_ROOTS"] = ":".join([*parts, tasks_dir])


def apply_env_overrides(case: Case) -> dict[str, str]:
    """Export the case's env overrides (e.g. GAP_LIBERO_PERTURBED).

    Each case runs in its own subprocess, so nothing to restore."""
    applied = dict(case.env_overrides or {})
    for k, v in applied.items():
        os.environ[k] = str(v)
    return applied


DEFAULT_PROVIDER = os.environ.get("GAP_LLM_PROVIDER", "vertex")
DEFAULT_MODEL = os.environ.get("GAP_LLM_MODEL", "gemini-3.1-pro-preview")


def case_accepts_refusal(case: Case) -> bool:
    """Does this case's criterion score a refusal instead of erroring?"""
    if case.success_source == "gen":
        return True
    return any("gen_refused" in json.dumps(c.get("expr", {}))
               for c in case.criterion.get("checks", []))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(case: Case, case_dir: Path, provider: str, model: str,
             force: bool) -> tuple[Path | None, dict]:
    info: dict = {"provider": provider, "model": model, "refused": False,
                  "error": None, "duration_s": 0.0, "cached": False}
    wf_dir = case_dir / "gen" / "task_00"
    if (wf_dir / "workflow.json").exists() and not force:
        info["cached"] = True
        return wf_dir, info
    from gap.agent import generate_sync

    t0 = time.time()
    try:
        graph = generate_sync(
            case.prompt, provider=provider, model=model,
            out_dir=case_dir / "gen",
        )
    except BaseException as exc:
        info["duration_s"] = round(time.time() - t0, 1)
        info["error"] = f"{type(exc).__name__}: {exc}"[:1500]
        info["refused"] = "missing capabilit" in str(exc).lower()
        return None, info
    info["duration_s"] = round(time.time() - t0, 1)
    try:
        (case_dir / "graph.txt").write_text(str(graph))
    except Exception:
        pass
    return Path(graph.path), info


def _load_workflow(wf_dir: Path | None) -> dict | None:
    if wf_dir is None:
        return None
    try:
        return json.loads((wf_dir / "workflow.json").read_text())
    except Exception:
        return None


def _gen_extras(gen_info: dict) -> dict[str, Any]:
    if gen_info.get("refused"):
        outcome = "REFUSED"
    elif gen_info.get("error"):
        outcome = "ERROR"
    else:
        outcome = "OK"
    return {"gen_outcome": outcome,
            "gen_message": str(gen_info.get("error") or "")[:500]}


# ---------------------------------------------------------------------------
# Execution (one seed)
# ---------------------------------------------------------------------------


def run_seed(case: Case, wf_dir: Path, seed_dir: Path, seed: int,
             video: bool) -> dict:
    """Execute the generated graph on one seed; returns execution meta.

    events.jsonl / state.jsonl are streamed to disk as the run goes, so
    a kill at any point leaves a scoreable partial trace.
    """
    import gap.connector
    from gap.runtime.execute import execute

    seed_dir.mkdir(parents=True, exist_ok=True)
    # Generated decide-loops keep a no-progress counter in
    # $TMPDIR/.gap_next_item_progress_<sha1(trace_dir)> (see
    # open-robot-skills perceiving-next-item/scripts/decide_next_item.py).
    # Our TMPDIR persists and trace paths repeat across re-runs of the
    # same --out, so clear the scratch before every fresh episode.
    import glob
    import tempfile
    for stale in glob.glob(os.path.join(
            tempfile.gettempdir(), ".gap_next_item_progress_*")):
        try:
            os.unlink(stale)
        except OSError:
            pass
    execution: dict = {
        "seed": seed, "ran": False, "success": False, "exit_status": None,
        "error": None, "duration_s": 0.0, "harness_crash": None,
        "env_overrides": dict(case.env_overrides or {}),
    }
    conn = gap.connector.sim(
        "libero", task=case.sim_task, record_video=video, seed=seed,
    )
    rec = StateRecorder(conn, seed_dir, meta={
        "case": case.id, "sim_task": case.sim_task, "seed": seed,
    })
    try:
        conn.reset()
        rec.install()
        rec.snapshot("__initial__")
        result = execute(
            str(wf_dir), conn,
            trace_dir=seed_dir / "trace", checkpoints="warn",
        )
        execution.update(
            ran=True,
            success=bool(result.success),
            exit_status=result.exit_status,
            error=str(result.error)[:800] if result.error else None,
            duration_s=round(result.duration_s, 1),
            checkpoints=[
                {"subgraph": c.subgraph, "name": c.name,
                 "passed": bool(c.passed)}
                for c in (result.checkpoint_results or [])
            ],
        )
        rec.snapshot("__final__")
        try:
            done, reward = conn.check_success()
            # Primary signal only for success_source env/both (native-
            # language classic cases); advisory metadata otherwise.
            execution["env_task_completed"] = bool(done)
            execution["env_reward"] = float(reward)
        except Exception:
            pass
        if video:
            try:
                saved = conn.save_video(str(seed_dir / "run_video.mp4"))
                execution["video"] = saved.get("file_path")
            except Exception as exc:
                execution["video"] = f"save failed: {exc}"
    except BaseException as exc:
        execution["harness_crash"] = f"{type(exc).__name__}: {exc}"[:800]
        execution["traceback"] = traceback.format_exc(limit=15)
        try:
            rec.snapshot("__final__")
        except Exception:
            pass
    finally:
        rec.uninstall()
        try:
            conn.close()
        except Exception:
            pass
    (seed_dir / "execution.json").write_text(
        json.dumps(execution, indent=2, default=str) + "\n")
    return execution


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def evaluate_from_disk(case: Case, case_dir: Path,
                       timed_out_case: bool) -> dict:
    """Score every seed dir on disk; aggregate to the case verdict."""
    wf_dir = case_dir / "gen" / "task_00"
    gen_path = case_dir / "generation.json"
    gen_info: dict = {}
    try:
        gen_info = json.loads(gen_path.read_text())
    except Exception:
        pass
    if not wf_dir.exists():
        override = gen_info.get("graph_override")
        wf_dir = Path(override) if override else wf_dir
    workflow = _load_workflow(wf_dir if wf_dir.exists() else None)
    structure = diagnostics.workflow_structure(workflow)

    if case.success_source == "gen":
        ev = evaluate_seed(case, [], [], {}, extras={
            **_gen_extras(gen_info),
            "generated_graph_structure": structure,
        })
        return _aggregate(case, case_dir,
                          [{"seed": None, "evaluation": ev.to_dict()}],
                          structure)

    seed_payloads = []
    for seed in case.seeds:
        seed_dir = case_dir / f"seed{seed}"
        if not seed_dir.exists():
            if timed_out_case:
                seed_payloads.append({
                    "seed": seed,
                    "evaluation": {
                        "verdict": "NOT_RUN", "score": 0.0,
                        "summary": "case timed out before this seed started",
                        "checks": [], "diagnostics": {}},
                })
            continue
        states = load_jsonl(seed_dir / "state.jsonl")
        events = load_jsonl(seed_dir / "events.jsonl")
        exec_path = seed_dir / "execution.json"
        if exec_path.exists():
            execution = json.loads(exec_path.read_text())
        else:
            # run_seed never finished for this seed: the case-level kill
            # (or a hard crash) hit while it was in flight.
            execution = {"seed": seed, "ran": True,
                         "timed_out": timed_out_case,
                         "harness_crash": None if timed_out_case
                         else "no execution.json (process died)"}
        extras = {
            "generated_graph_structure": structure,
            "max_node_iterations":
                diagnostics.max_node_iterations(seed_dir / "trace"),
            "trace_dir": str(seed_dir / "trace"),
            **_gen_extras(gen_info),
        }
        ev = evaluate_seed(case, states, events, execution, extras)
        payload = {"seed": seed, "execution": execution,
                   "evaluation": ev.to_dict()}
        (seed_dir / "result_seed.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n")
        seed_payloads.append(payload)
    if not seed_payloads:
        return {}
    return _aggregate(case, case_dir, seed_payloads, structure)


def _case_meta(case: Case) -> dict[str, Any]:
    return {
        "id": case.id, "prompt": case.prompt, "category": case.category,
        "tier": case.tier, "tags": list(case.tags),
        "template": case.template, "baseline": case.baseline,
        "intent": case.intent, "criterion": case.criterion,
        "seeds": list(case.seeds), "sim_task": case.sim_task,
        "success_source": case.success_source,
        "aggregate": case.aggregate,
        "env_overrides": dict(case.env_overrides or {}),
    }


def _aggregate(case: Case, case_dir: Path, seed_payloads: list[dict],
               structure: dict) -> dict:
    verdicts = [p["evaluation"]["verdict"] for p in seed_payloads]
    scores = [float(p["evaluation"]["score"]) for p in seed_payloads]
    if case.aggregate == "any_seed":
        passed = any(v == "PASS" for v in verdicts)
    else:
        passed = bool(verdicts) and all(v == "PASS" for v in verdicts)
    if passed:
        verdict = "PASS"
    elif "TIMEOUT" in verdicts:
        verdict = "TIMEOUT"
    else:
        verdict = "FAIL"
    gen_path = case_dir / "generation.json"
    generation = (json.loads(gen_path.read_text())
                  if gen_path.exists() else {})
    payload = {
        "case": _case_meta(case),
        "generation": generation,
        "generated_graph_structure": structure,
        "verdict": verdict,
        "score": round(sum(scores) / max(1, len(scores)), 4),
        "seeds": seed_payloads,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (case_dir / "result.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    per_seed = ", ".join(
        f"seed{p['seed']}={p['evaluation']['verdict']}"
        f"({float(p['evaluation']['score']):.2f})"
        for p in seed_payloads
    )
    print(f"CASE {case.id}: {verdict} score={payload['score']:.2f} "
          f"[{per_seed}]")
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_case(case: Case, out_root: Path, *, provider: str, model: str,
             video: bool, force_generate: bool,
             graph_override: Path | None, only_seed: int | None,
             gen_only: bool = False, exec_only: bool = False) -> dict:
    case_dir = out_root / case.id
    case_dir.mkdir(parents=True, exist_ok=True)

    if graph_override is not None:
        wf_dir: Path | None = graph_override
        gen_info = {"graph_override": str(graph_override), "refused": False}
    elif exec_only:
        wf_dir = case_dir / "gen" / "task_00"
        gen_path = case_dir / "generation.json"
        gen_info = (json.loads(gen_path.read_text())
                    if gen_path.exists() else {"cached": True})
        if not (wf_dir / "workflow.json").exists() and not case_accepts_refusal(case):
            print(f"CASE {case.id}: no cached generation for --exec-only")
            return {"verdict": "GEN_ERROR"}
        if not (wf_dir / "workflow.json").exists():
            wf_dir = None
    else:
        wf_dir, gen_info = generate(case, case_dir, provider, model,
                                    force_generate)
    (case_dir / "generation.json").write_text(
        json.dumps(gen_info, indent=2) + "\n")

    if gen_only:
        outcome = _gen_extras(gen_info)["gen_outcome"]
        print(f"CASE {case.id}: GEN_{outcome}"
              + (" (cached)" if gen_info.get("cached") else ""))
        return {"verdict": "PASS" if outcome == "OK" else f"GEN_{outcome}"}

    if case.success_source == "gen":
        return evaluate_from_disk(case, case_dir, timed_out_case=False)

    if wf_dir is None:
        if case_accepts_refusal(case):
            # The criterion itself scores the refusal (e.g. absent-object
            # probes where refusing is the CORRECT outcome).
            structure = diagnostics.workflow_structure(None)
            ev = evaluate_seed(case, [], [], {}, extras={
                **_gen_extras(gen_info),
                "generated_graph_structure": structure,
            })
            return _aggregate(case, case_dir,
                              [{"seed": None, "evaluation": ev.to_dict()}],
                              structure)
        verdict = "GEN_REFUSED" if gen_info.get("refused") else "GEN_ERROR"
        payload = {
            "case": _case_meta(case),
            "generation": gen_info,
            "verdict": verdict, "score": 0.0, "seeds": [],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (case_dir / "result.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n")
        print(f"CASE {case.id}: {verdict} — {str(gen_info.get('error', ''))[:200]}")
        return payload

    seeds = [only_seed] if only_seed is not None else list(case.seeds)
    for seed in seeds:
        logger.info("case %s: executing seed %d", case.id, seed)
        run_seed(case, wf_dir, case_dir / f"seed{seed}", seed, video)
    return evaluate_from_disk(case, case_dir, timed_out_case=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=None,
                    help="run only this seed (default: the case's list)")
    ap.add_argument("--graph", default=None,
                    help="skip generation; execute this workflow dir "
                         "(the reward-oracle path)")
    ap.add_argument("--force-generate", action="store_true")
    ap.add_argument("--gen-only", action="store_true",
                    help="stop after generation (no sim)")
    ap.add_argument("--exec-only", action="store_true",
                    help="require the cached generation; never call the LLM")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--re-eval", action="store_true",
                    help="re-score existing traces (no sim, no LLM); "
                         "honors timeout_marker.json for hang verdicts")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    env_hygiene()
    # Out-of-process tool bundles (perception VLM, ...) resolve their
    # provider from the environment; export the chosen provider/model so
    # the runtime bundles match generation.
    os.environ["GAP_LLM_PROVIDER"] = args.provider
    os.environ["GAP_LLM_MODEL"] = args.model
    case = CASES[args.case]
    apply_env_overrides(case)
    out_root = Path(args.out)
    case_dir = out_root / case.id

    if args.re_eval:
        timed_out = (case_dir / "timeout_marker.json").exists()
        payload = evaluate_from_disk(case, case_dir, timed_out_case=timed_out)
        if not payload:
            gen_path = case_dir / "generation.json"
            gen_info = (json.loads(gen_path.read_text())
                        if gen_path.exists() else {})
            verdict = ("GEN_REFUSED" if gen_info.get("refused")
                       else "TIMEOUT" if timed_out else "GEN_ERROR")
            payload = {
                "case": _case_meta(case),
                "generation": gen_info, "verdict": verdict,
                "score": 0.0, "seeds": [],
            }
            (case_dir / "result.json").write_text(
                json.dumps(payload, indent=2, default=str) + "\n")
            print(f"CASE {case.id}: {verdict}")
    else:
        payload = run_case(
            case, out_root, provider=args.provider, model=args.model,
            video=not args.no_video, force_generate=args.force_generate,
            graph_override=Path(args.graph) if args.graph else None,
            only_seed=args.seed, gen_only=args.gen_only,
            exec_only=args.exec_only,
        )

    verdict = payload.get("verdict")
    if verdict == "PASS":
        return 0
    if verdict in ("GEN_ERROR", "GEN_REFUSED"):
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
