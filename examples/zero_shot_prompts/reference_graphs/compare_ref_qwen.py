"""Reference (feasibility ceiling) vs a model-under-test run, per case.

Both arms are scored by the SAME criteria, so this answers: on the cases a
correct hand-authored graph can pass, how many does the model's generated
graph also pass — i.e. how close does the model get to the feasibility
ceiling.

    uv run python examples/zero_shot_prompts/reference_graphs/compare_ref_qwen.py \
        --ref outputs/zsp/reference --model outputs/zsp/qwen-local --label qwen3.6

The model dir must have EXECUTED results (result.json per case), e.g. via
``run_all.py --out <dir> --exec-only`` after generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in run_dir.glob("*/result.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        out[f.parent.name] = {"verdict": r.get("verdict"),
                              "score": float(r.get("score", 0.0) or 0.0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="outputs/zsp/reference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = _load(Path(args.ref))
    mdl = _load(Path(args.model))
    cases = sorted(set(ref) & set(mdl))
    if not cases:
        print("no overlapping cases with results in both dirs")
        return 1

    lines = [f"# Reference vs {args.label}", "",
             f"| case | reference | {args.label} | model reaches ceiling? |",
             "|---|---|---|---|"]
    ref_pass = mp = both = 0
    for c in cases:
        r, m = ref[c], mdl[c]
        rp, mpv = r["verdict"] == "PASS", m["verdict"] == "PASS"
        ref_pass += rp
        mp += mpv
        reach = "—"
        if rp:
            reach = "✅ yes" if mpv else "❌ no (gap)"
            both += rp and mpv
        elif mpv:
            reach = "⚠️ model>ref"
        lines.append(
            f"| {c} | {r['verdict']} {r['score']:.2f} | "
            f"{m['verdict']} {m['score']:.2f} | {reach} |")

    lines += ["", "## Summary",
              f"- overlapping cases: {len(cases)}",
              f"- reference PASS (feasible): {ref_pass}/{len(cases)}",
              f"- {args.label} PASS: {mp}/{len(cases)}",
              f"- **model reaches ceiling** (both PASS): {both}/{ref_pass} "
              f"of the feasible cases",
              ""]
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
