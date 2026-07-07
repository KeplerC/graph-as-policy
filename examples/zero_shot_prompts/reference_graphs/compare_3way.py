"""3-way comparison: hand-authored reference (ceiling) vs N model arms.

Re-scores every case from its recorded trace with the CURRENT criterion, so
all arms are judged identically. Usage:

    uv run python examples/zero_shot_prompts/reference_graphs/compare_3way.py \
        --ref outputs/zsp/reference \
        --arm gemini3.1-pro=outputs/zsp/gemini31pro \
        --arm qwen3.6=outputs/zsp/qwen-local \
        --out outputs/zsp/reference/compare_3way.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLE = _HERE.parent
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from zsp.evaluate import evaluate_seed  # noqa: E402
from zsp.registry import CASES  # noqa: E402
from zsp.state_log import load_jsonl  # noqa: E402


def rescore(root: str, cid: str):
    d = Path(root) / cid / "seed0"
    if not (d / "state.jsonl").exists():
        rj = Path(root) / cid / "result.json"
        if rj.exists():
            r = json.loads(rj.read_text())
            return r.get("verdict"), float(r.get("score", 0) or 0)
        return None, None
    states = load_jsonl(d / "state.jsonl")
    events = load_jsonl(d / "events.jsonl")
    execn = (json.loads((d / "execution.json").read_text())
             if (d / "execution.json").exists() else {})
    ev = evaluate_seed(CASES[cid], states, events, execn,
                       extras={"trace_dir": str(d / "trace")})
    return ev.verdict, ev.score


def executed(root: str) -> set[str]:
    return {os.path.basename(os.path.dirname(os.path.dirname(f)))
            for f in glob.glob(f"{root}/*/seed0/state.jsonl")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="outputs/zsp/reference")
    ap.add_argument("--arm", action="append", default=[],
                    help="label=path (repeatable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = [a.split("=", 1) for a in args.arm]
    labels = ["reference"] + [lbl for lbl, _ in arms]
    roots = [args.ref] + [p for _, p in arms]

    # cases executed in the reference AND every arm
    common = executed(args.ref)
    for _, p in arms:
        common &= executed(p)
    common &= set(CASES)
    cases = sorted(common)

    rows = []
    for c in cases:
        rows.append([c] + [rescore(r, c) for r in roots])

    hdr = "| case | " + " | ".join(labels) + " | model gaps |"
    sep = "|---|" + "---|" * (len(labels) + 1)
    lines = [f"# {' vs '.join(labels)}  ({len(cases)} common cases)", "", hdr, sep]
    counts = {lbl: 0 for lbl in labels}
    ref_pass = 0
    reach = {lbl: 0 for lbl in labels[1:]}
    for row in rows:
        c = row[0]
        cells = row[1:]
        for lbl, (v, _s) in zip(labels, cells, strict=True):
            if v == "PASS":
                counts[lbl] += 1
        rpass = cells[0][0] == "PASS"
        ref_pass += rpass
        gap = ""
        if rpass:
            missed = [labels[i + 1] for i, (v, _) in enumerate(cells[1:])
                      if v != "PASS"]
            for lbl, (v, _) in zip(labels[1:], cells[1:], strict=True):
                if v == "PASS":
                    reach[lbl] += 1
            gap = ("✅ all reach" if not missed
                   else "miss: " + ",".join(missed))
        cellstr = " | ".join(
            f"{v} {s:.2f}" if v else "—" for v, s in cells)
        lines.append(f"| {c} | {cellstr} | {gap} |")

    n = len(cases)
    lines += ["", "## Summary", "",
              "| arm | PASS | reaches ceiling |", "|---|---|---|"]
    lines.append(f"| reference (ceiling) | {counts['reference']}/{n} | — |")
    for lbl in labels[1:]:
        lines.append(f"| {lbl} | {counts[lbl]}/{n} | "
                     f"{reach[lbl]}/{ref_pass} feasible |")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
