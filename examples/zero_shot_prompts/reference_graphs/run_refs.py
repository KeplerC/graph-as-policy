"""Drive the reference graphs through the reward oracle, in parallel.

For each registered reference, run
``run_prompt.py --case <case> --graph <graph_dir>`` on one GPU from the pool
and collect the verdict/score. This is the feasibility proof: a PASS means a
correct graph exists AND the criterion accepts it; a FAIL on a frontier case
characterizes the gap (with partial-progress diagnostics).

    # all references, 4 concurrent, GPUs 4-7 (qwen holds 0-3)
    uv run python examples/zero_shot_prompts/reference_graphs/run_refs.py \
        --out outputs/zsp/reference --jobs 4 --gpus 4,5,6,7

    # a subset / a single family
    ... --cases soup_basket,place_midpoint
    ... --family spatial
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLE = _HERE.parent
_REPO = _EXAMPLE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from registry import REFS, Ref  # noqa: E402

GEN_DIR = _HERE / "_generated"
_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _graph_dir(ref: Ref) -> Path:
    if ref.kind == "pick":
        return GEN_DIR / ref.case
    return _HERE / ref.graph


def run_ref(ref: Ref, out_root: Path, model: str, provider: str,
            gpu: str, video: bool) -> dict:
    graph_dir = _graph_dir(ref)
    if not (graph_dir / "workflow.json").exists():
        return {"case": ref.case, "verdict": "NO_GRAPH", "score": 0.0,
                "family": ref.family,
                "note": f"missing {graph_dir.relative_to(_REPO)}"}
    cmd = [
        sys.executable, str(_EXAMPLE / "run_prompt.py"),
        "--case", ref.case, "--out", str(out_root),
        "--graph", str(graph_dir),
        "--provider", provider, "--model", model, "--seed", "0",
    ]
    if not video:
        cmd.append("--no-video")
    env = {"CUDA_VISIBLE_DEVICES": gpu, "MUJOCO_EGL_DEVICE_ID": gpu,
           "MUJOCO_GL": "egl"}
    import os
    full_env = {**os.environ, **env}
    log_dir = out_root / ref.case
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "ref_console.log", "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(_REPO), env=full_env)
    result = {"case": ref.case, "family": ref.family, "note": ref.note,
              "kind": ref.kind, "exit": proc.returncode}
    rp = out_root / ref.case / "result.json"
    if rp.exists():
        try:
            r = json.loads(rp.read_text())
            seeds = r.get("seeds", [])
            ev = seeds[0]["evaluation"] if seeds else {}
            result.update(verdict=r.get("verdict"), score=r.get("score", 0.0),
                          summary=ev.get("summary", "")[:200])
        except Exception as exc:
            result.update(verdict="UNREADABLE", score=0.0, summary=str(exc))
    else:
        result.update(verdict="NO_RESULT", score=0.0,
                      summary="run_prompt produced no result.json")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/zsp/reference")
    ap.add_argument("--cases", default=None, help="comma-separated case ids")
    ap.add_argument("--family", default=None)
    ap.add_argument("--kind", default=None, choices=("pick", "hand"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--provider", default="vertex")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    refs = list(REFS.values())
    if args.cases:
        want = set(args.cases.split(","))
        refs = [r for r in refs if r.case in want]
    if args.family:
        refs = [r for r in refs if r.family == args.family]
    if args.kind:
        refs = [r for r in refs if r.kind == args.kind]
    if not refs:
        print("no references selected")
        return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    tokens: queue.Queue[str] = queue.Queue()
    for i in range(args.jobs):
        tokens.put(gpus[i % len(gpus)])

    say(f"references: {len(refs)} -> {out_root} (jobs={args.jobs}, "
        f"gpus={gpus}, model={args.model})")
    results: list[dict] = []
    done = [0]

    def worker(ref: Ref) -> None:
        gpu = tokens.get()
        t0 = time.time()
        try:
            res = run_ref(ref, out_root, args.model, args.provider, gpu,
                          args.video)
        except Exception as exc:
            res = {"case": ref.case, "family": ref.family, "verdict": "ERROR",
                   "score": 0.0, "summary": str(exc)}
        finally:
            tokens.put(gpu)
        done[0] += 1
        results.append(res)
        say(f"[{done[0]}/{len(refs)}] {ref.case:32s} {res.get('verdict'):10s} "
            f"score={float(res.get('score', 0.0)):.2f} "
            f"({(time.time() - t0) / 60:.1f} min, gpu{gpu})")

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(worker, refs))

    # summary by family
    results.sort(key=lambda r: (r.get("family", ""), r["case"]))
    fam: dict[str, list[dict]] = {}
    for r in results:
        fam.setdefault(r.get("family", "?"), []).append(r)
    lines = ["", "=== reference feasibility summary ===",
             f"{'family':20s} pass/total  cases"]
    n_pass = 0
    for f, rs in sorted(fam.items()):
        p = sum(r.get("verdict") == "PASS" for r in rs)
        n_pass += p
        detail = " ".join(
            f"{r['case'].split('__')[-1]}={'PASS' if r.get('verdict') == 'PASS' else r.get('verdict')}"
            for r in rs)
        lines.append(f"{f:20s} {p}/{len(rs):<9d} {detail[:110]}")
    lines.append(f"\nTOTAL: {n_pass}/{len(results)} references PASS")
    say("\n".join(lines))
    (out_root / "reference_summary.json").write_text(
        json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
