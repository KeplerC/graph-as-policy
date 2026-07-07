"""Cross-run comparison: N benchmark output dirs -> one matrix.

    uv run python examples/zero_shot_prompts/compare.py \
        outputs/zsp/gemini outputs/zsp/qwen outputs/zsp/deepseek-v4-hosted \
        [--labels gemini,qwen,dsv4] [--out outputs/zsp/compare]

Reads each run's ``report.json`` (regenerate stale ones with
``run_all.py --out DIR --report-only``) + ``run_meta.json``, writes
``compare.md``/``compare.json``: a case x run verdict/score matrix,
per-category pass rates, and the verdict-flips section (cases where
runs disagree — the interesting rows).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _load_run(run_dir: Path) -> dict[str, dict]:
    report = run_dir / "report.json"
    if not report.exists():
        raise SystemExit(
            f"{run_dir}: no report.json — run "
            f"`run_all.py --out {run_dir} --report-only` first")
    rows = json.loads(report.read_text())
    return {r["case"]["id"]: r for r in rows if r.get("case", {}).get("id")}


def _label(run_dir: Path) -> str:
    meta = run_dir / "run_meta.json"
    try:
        m = json.loads(meta.read_text())
        return str(m.get("model") or run_dir.name)
    except Exception:
        return run_dir.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="benchmark output dirs")
    ap.add_argument("--labels", default=None,
                    help="comma-separated column labels (default: model "
                         "from run_meta.json, else dir name)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: first run dir)")
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    runs = [_load_run(d) for d in run_dirs]
    labels = (args.labels.split(",") if args.labels
              else [_label(d) for d in run_dirs])
    if len(labels) != len(runs):
        raise SystemExit("--labels count must match run dirs")

    all_ids = sorted({cid for run in runs for cid in run})

    def cell(run: dict, cid: str) -> str:
        r = run.get(cid)
        if r is None:
            return "—"
        v = r.get("verdict", "?")
        s = float(r.get("score", 0.0))
        mark = {"PASS": "✔"}.get(v, v)
        return f"{mark} {s:.2f}"

    lines = [
        "# Benchmark comparison",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        *[f"- **{lbl}**: `{d}`" for lbl, d in zip(labels, run_dirs, strict=True)],
        "",
        "## Per-category pass rates",
        "",
        "| category | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]

    cats: dict[str, list[str]] = defaultdict(list)
    for cid in all_ids:
        cat = next((run[cid]["case"].get("category", "?")
                    for run in runs if cid in run), "?")
        cats[cat].append(cid)
    for cat in sorted(cats):
        row = [cat]
        for run in runs:
            rs = [run[cid] for cid in cats[cat] if cid in run]
            n_pass = sum(r.get("verdict") == "PASS" for r in rs)
            row.append(f"{n_pass}/{len(rs)}" if rs else "—")
        lines.append("| " + " | ".join(row) + " |")
    row = ["**all**"]
    for run in runs:
        n_pass = sum(r.get("verdict") == "PASS" for r in run.values())
        row.append(f"{n_pass}/{len(run)}")
    lines += ["| " + " | ".join(row) + " |", ""]

    lines += [
        "## Case matrix",
        "",
        "| case | cat | tier | " + " | ".join(labels) + " |",
        "|---|---|---|" + "---|" * len(labels),
    ]
    flips: list[str] = []
    for cid in all_ids:
        meta = next((run[cid]["case"] for run in runs if cid in run), {})
        cells = [cell(run, cid) for run in runs]
        lines.append(
            f"| {cid} | {meta.get('category', '?')} "
            f"| {meta.get('tier', '?')} | " + " | ".join(cells) + " |")
        verdicts = {run[cid].get("verdict") for run in runs if cid in run}
        if len(verdicts) > 1:
            flips.append(f"- **{cid}**: " + ", ".join(
                f"{lbl}={run.get(cid, {}).get('verdict', '—')}"
                for lbl, run in zip(labels, runs, strict=True)))

    lines += ["", "## Verdict flips (runs disagree)", ""]
    lines += flips or ["(none — all runs agree on every case)"]
    lines.append("")

    out_dir = Path(args.out) if args.out else run_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.md").write_text("\n".join(lines))
    (out_dir / "compare.json").write_text(json.dumps({
        "labels": labels, "runs": [str(d) for d in run_dirs],
        "cases": {cid: {lbl: {
            "verdict": run.get(cid, {}).get("verdict"),
            "score": run.get(cid, {}).get("score"),
        } for lbl, run in zip(labels, runs, strict=True)} for cid in all_ids},
    }, indent=2) + "\n")
    print(f"compare: {out_dir / 'compare.md'} ({len(all_ids)} cases, "
          f"{len(flips)} flips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
