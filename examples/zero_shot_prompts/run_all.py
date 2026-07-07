"""Run the benchmark matrix, one subprocess per case.

    MUJOCO_GL=egl uv run python \
        examples/zero_shot_prompts/run_all.py \
        --out outputs/zero_shot_prompts/run1 --jobs 4

Each case runs in a fresh subprocess (own CUDA/EGL context) under a
wall-clock timeout. On timeout the whole process GROUP is killed —
hangs are an expected, first-class outcome: a ``timeout_marker.json``
is dropped and the case is re-scored from its flushed partial trace
(``--re-eval``), which recovers the hang site from ``events.jsonl``.

Scale-out: ``--jobs N`` runs N cases concurrently, each pinned
round-robin to one of ``--gpus`` (default: all GPUs nvidia-smi reports)
via CUDA_VISIBLE_DEVICES + MUJOCO_EGL_DEVICE_ID. Phases: ``--gen-only``
fans out generation (LLM-bound, no sim); ``--exec-only`` executes
cached generations. Selection: ``--cases/--categories/--tiers/--tags/
--templates``; ``--list`` prints the manifest and exits.

Re-running with the same ``--out`` skips cases that already have a
``result.json`` (resume); ``--force`` re-runs everything;
``--retry-verdicts CRASH,NO_TRACE --retries 1`` re-runs infra failures
only (never FAIL). ``--export-manifest out.jsonl`` dumps the selected
cases as JSONL (prompt + sim_task + criterion) for external harnesses
(e.g. an RL reward oracle); ``--expand full`` adds the full template
cross-product when the corpus ships one.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from zsp.registry import CASES, select  # noqa: E402
from zsp.runner.report import load_rows, write_report  # noqa: E402
from zsp.schema import Case  # noqa: E402

REPO_ROOT = _HERE.parents[1]

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------


def _spawn(case: Case, args: argparse.Namespace, out_root: Path,
           extra: list[str], gpu: str | None) -> int:
    cmd = [
        sys.executable, str(_HERE / "run_prompt.py"),
        "--case", case.id, "--out", str(out_root),
        "--provider", args.provider, "--model", args.model,
    ] + extra
    if args.no_video:
        cmd.append("--no-video")
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["MUJOCO_EGL_DEVICE_ID"] = gpu
    log_path = out_root / case.id / "console.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n===== {' '.join(cmd)} @ {time.strftime('%F %T')}"
                  f" gpu={gpu}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=str(REPO_ROOT), env=env,
        )
        timeout_min = args.timeout_min or case.timeout_min
        if "--gen-only" in extra:
            timeout_min = min(timeout_min, 30)
        try:
            return proc.wait(timeout=60 * timeout_min)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            (out_root / case.id / "timeout_marker.json").write_text(
                json.dumps({
                    "timed_out": True,
                    "timeout_min": timeout_min,
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }) + "\n")
            return -9


def run_one(case: Case, args: argparse.Namespace, out_root: Path,
            gpu: str | None) -> str:
    case_dir = out_root / case.id
    if args.gen_only:
        wf = case_dir / "gen" / "task_00" / "workflow.json"
        gen_done = wf.exists() or (case_dir / "generation.json").exists()
        if gen_done and not args.force:
            return "cached"
        extra = ["--gen-only"] + (["--force-generate"] if args.force else [])
        code = _spawn(case, args, out_root, extra, gpu)
        return f"gen exit={code}"

    result_path = case_dir / "result.json"
    if result_path.exists() and not args.force:
        return "cached"
    marker = case_dir / "timeout_marker.json"
    if marker.exists():
        marker.unlink()

    extra = []
    if args.force:
        extra.append("--force-generate")
    if args.exec_only:
        extra.append("--exec-only")
    code = _spawn(case, args, out_root, extra, gpu)
    if code == -9 or not result_path.exists():
        # killed (hang) or died before writing the aggregate: score the
        # partial traces post-mortem in a fresh process.
        _spawn(case, args, out_root, ["--re-eval"], gpu)
        return "timeout" if code == -9 else f"exit={code} (re-evaled)"
    return f"exit={code}"


def _run_pool(cases: list[Case], args: argparse.Namespace,
              out_root: Path) -> None:
    gpus = ([g.strip() for g in args.gpus.split(",") if g.strip()]
            if args.gpus else None)
    if gpus is None and args.jobs > 1:
        gpus = _all_gpus()
    tokens: queue.Queue[str | None] = queue.Queue()
    for i in range(args.jobs):
        tokens.put(gpus[i % len(gpus)] if gpus else None)

    done = [0]
    total = len(cases)

    def worker(case: Case) -> None:
        gpu = tokens.get()
        t0 = time.time()
        try:
            status = run_one(case, args, out_root, gpu)
        except Exception as exc:  # never kill the pool
            status = f"pool error: {exc}"
        finally:
            tokens.put(gpu)
        result = {}
        rp = out_root / case.id / "result.json"
        if rp.exists() and not args.gen_only:
            try:
                result = json.loads(rp.read_text())
            except Exception:
                pass
        done[0] += 1
        say(f"[{done[0]}/{total}] {case.id}: {status} "
            f"{result.get('verdict', '')} "
            f"score={float(result.get('score', 0.0)):.2f} "
            f"({(time.time() - t0) / 60:.1f} min, gpu={gpu})")

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(worker, cases))


def _all_gpus() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return [line.strip() for line in out.strip().splitlines()]
    except Exception:
        return ["0"]


# ---------------------------------------------------------------------------
# Selection / manifest
# ---------------------------------------------------------------------------


def _select_cases(args: argparse.Namespace) -> list[Case] | None:
    ids = args.cases.split(",") if args.cases else None
    if ids:
        unknown = [c for c in ids if c not in CASES]
        if unknown:
            print(f"unknown case(s): {unknown}; have {sorted(CASES)}")
            return None
    return select(
        ids=ids,
        categories=args.categories.split(",") if args.categories else None,
        tiers=args.tiers.split(",") if args.tiers else None,
        tags=args.tags.split(",") if args.tags else None,
        templates=args.templates.split(",") if args.templates else None,
    )


def _print_manifest(cases: list[Case]) -> None:
    print(f"{'id':34s} {'cat':11s} {'tier':4s} {'src':5s} {'seeds':8s} "
          f"{'to':>3s}  sim_task")
    for c in cases:
        print(f"{c.id:34s} {c.category:11s} {c.tier:4s} "
              f"{c.success_source:5s} {str(list(c.seeds)):8s} "
              f"{c.timeout_min:3d}  {c.sim_task}"
              + (f"  [{', '.join(c.tags)}]" if c.tags else ""))
    tiers: dict[str, int] = {}
    for c in cases:
        tiers[c.tier] = tiers.get(c.tier, 0) + 1
    print(f"\n{len(cases)} cases ({', '.join(f'{k}:{v}' for k, v in sorted(tiers.items()))})")


def _case_manifest_line(c: Case) -> dict:
    return {
        "id": c.id, "template": c.template, "category": c.category,
        "tier": c.tier, "tags": list(c.tags), "prompt": c.prompt,
        "sim_task": c.sim_task, "seeds": list(c.seeds),
        "timeout_min": c.timeout_min, "success_source": c.success_source,
        "aggregate": c.aggregate, "requires_schema": c.requires_schema,
        "env_overrides": dict(c.env_overrides or {}),
        "env_consumes_packed": c.env_consumes_packed,
        "criterion": c.criterion, "intent": c.intent,
    }


def _export_manifest(cases: list[Case], path: Path, expand: str) -> None:
    lines = [_case_manifest_line(c) for c in cases]
    if expand == "full":
        try:
            from zsp.cases.expansions import full_expansion

            extra = [c for c in full_expansion()
                     if c.id not in {x["id"] for x in lines}]
            lines += [_case_manifest_line(c) for c in extra]
            print(f"expand=full: +{len(extra)} template-expanded cases")
        except ImportError:
            print("expand=full: no zsp.cases.expansions module yet — "
                  "exporting the curated set only")
    with open(path, "w") as fh:
        for line in lines:
            fh.write(json.dumps(line, default=str) + "\n")
    print(f"manifest: {len(lines)} cases -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default=None,
                    help="comma-separated ids (default: all selected)")
    ap.add_argument("--categories", default=None)
    ap.add_argument("--tiers", default=None, help="e.g. T0,T1")
    ap.add_argument("--tags", default=None)
    ap.add_argument("--templates", default=None)
    ap.add_argument("--list", action="store_true",
                    help="print the selected-case manifest and exit")
    ap.add_argument("--export-manifest", default=None, metavar="PATH",
                    help="write selected cases as JSONL and exit")
    ap.add_argument("--expand", default="curated",
                    choices=("curated", "full"),
                    help="with --export-manifest: also emit the full "
                         "template cross-product (RL task pool)")
    ap.add_argument("--provider",
                    default=os.environ.get("GAP_LLM_PROVIDER", "vertex"))
    ap.add_argument("--model",
                    default=os.environ.get("GAP_LLM_MODEL",
                                           "gemini-3.1-pro-preview"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--gpus", default=None,
                    help="comma-separated physical GPU indices for "
                         "round-robin pinning (default: all)")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--exec-only", action="store_true")
    ap.add_argument("--timeout-min", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-verdicts", default=None,
                    help="comma-separated verdicts to retry (e.g. "
                         "CRASH,NO_TRACE); FAIL is never auto-retried")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cases = _select_cases(args)
    if cases is None:
        return 2
    if args.list:
        _print_manifest(cases)
        return 0
    if args.export_manifest:
        _export_manifest(cases, Path(args.export_manifest), args.expand)
        return 0

    out_root = Path(args.out or
                    f"outputs/zero_shot_prompts/{time.strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        print(f"report: {write_report(out_root)}")
        return 0

    if not cases:
        print("no cases selected")
        return 2

    git_sha = ""
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    (out_root / "run_meta.json").write_text(json.dumps({
        "provider": args.provider, "model": args.model,
        "git_sha": git_sha, "jobs": args.jobs,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: v for k, v in vars(args).items() if v is not None},
        "case_ids": [c.id for c in cases],
    }, indent=2) + "\n")

    say(f"matrix: {len(cases)} case(s) -> {out_root} "
        f"(jobs={args.jobs}{', gen-only' if args.gen_only else ''})")
    _run_pool(cases, args, out_root)

    retry_verdicts = (set(args.retry_verdicts.split(","))
                      if args.retry_verdicts else set())
    retry_verdicts.discard("FAIL")
    for attempt in range(args.retries if retry_verdicts else 0):
        retry = []
        for c in cases:
            rp = out_root / c.id / "result.json"
            try:
                verdict = json.loads(rp.read_text()).get("verdict")
            except Exception:
                verdict = "UNREADABLE"
            if verdict in retry_verdicts:
                rp.unlink(missing_ok=True)
                retry.append(c)
        if not retry:
            break
        say(f"retry {attempt + 1}: {len(retry)} case(s) "
            f"({','.join(sorted(retry_verdicts))})")
        _run_pool(retry, args, out_root)

    if args.gen_only:
        say("generation phase complete (no report for gen-only)")
        return 0

    report = write_report(out_root)
    say(f"report: {report}")
    rows = load_rows(out_root)
    say(f"passed {sum(r.get('verdict') == 'PASS' for r in rows)}"
        f"/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
