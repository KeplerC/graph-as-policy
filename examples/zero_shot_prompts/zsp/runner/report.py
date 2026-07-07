"""Report v2: capability scorecard + per-case detail.

Reads every ``<out>/<case>/result.json``, writes ``report.md`` (scorecard
by category x tier, template rollups, grouped per-case rows, detail
sections) and ``report.json`` (the raw rows, consumed by compare.py).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from zsp.registry import ALL_CASES, CASES

GEN_VERDICTS = ("GEN_ERROR", "GEN_REFUSED")


def load_rows(out_root: Path) -> list[dict]:
    rows = []
    for case in ALL_CASES:
        p = out_root / case.id / "result.json"
        if not p.exists():
            continue
        try:
            row = json.loads(p.read_text())
        except Exception:
            row = {"case": {"id": case.id}, "verdict": "UNREADABLE",
                   "score": 0.0, "seeds": []}
        # Backfill taxonomy for rows written by older runners.
        meta = row.setdefault("case", {})
        for k, v in (("category", case.category), ("tier", case.tier),
                     ("template", case.template)):
            meta.setdefault(k, v)
        rows.append(row)
    return rows


def _scorecard(rows: list[dict]) -> list[str]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        c = r.get("case", {})
        groups[(c.get("category", "?"), c.get("tier", "?"))].append(r)
    lines = [
        "| category | tier | n | gen ok | pass | timeout | mean score |",
        "|---|---|---|---|---|---|---|",
    ]
    for (cat, tier) in sorted(groups):
        rs = groups[(cat, tier)]
        n = len(rs)
        gen_ok = sum(r.get("verdict") not in GEN_VERDICTS for r in rs)
        n_pass = sum(r.get("verdict") == "PASS" for r in rs)
        n_to = sum(r.get("verdict") == "TIMEOUT" for r in rs)
        mean = sum(float(r.get("score", 0.0)) for r in rs) / max(1, n)
        lines.append(
            f"| {cat} | {tier} | {n} | {gen_ok} | {n_pass} | {n_to} "
            f"| {mean:.2f} |"
        )
    n = len(rows)
    n_pass = sum(r.get("verdict") == "PASS" for r in rows)
    mean = sum(float(r.get("score", 0.0)) for r in rows) / max(1, n)
    lines.append(f"| **all** | | {n} | "
                 f"{sum(r.get('verdict') not in GEN_VERDICTS for r in rows)} "
                 f"| {n_pass} | "
                 f"{sum(r.get('verdict') == 'TIMEOUT' for r in rows)} "
                 f"| {mean:.2f} |")
    return lines


def _template_rollup(rows: list[dict]) -> list[str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tpl = r.get("case", {}).get("template")
        if tpl:
            groups[tpl].append(r)
    if not groups:
        return []
    lines = ["## Permutation templates (per-object breakdown)", ""]
    for tpl in sorted(groups):
        rs = groups[tpl]
        n_pass = sum(r.get("verdict") == "PASS" for r in rs)
        parts = ", ".join(
            f"{r['case']['id'].split('__', 1)[1]}="
            f"{'✔' if r.get('verdict') == 'PASS' else r.get('verdict')}"
            for r in sorted(rs, key=lambda r: r["case"]["id"])
        )
        lines.append(f"- **{tpl}** {n_pass}/{len(rs)}: {parts}")
    lines.append("")
    return lines


def write_report(out_root: Path) -> Path:
    rows = load_rows(out_root)
    meta = {}
    meta_path = out_root / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    lines = [
        "# Zero-shot instruction benchmark — measured results",
        "",
        f"- output root: `{out_root}`",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- provider/model: {meta.get('provider', '?')} / {meta.get('model', '?')}",
        f"- cases with results: {len(rows)}/{len(CASES)}",
        "",
        "## Scorecard",
        "",
        *_scorecard(rows),
        "",
        *_template_rollup(rows),
        "## Cases",
        "",
        "| case | cat | tier | verdict | score | per-seed | summary |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["case"].get("category", ""),
                                         r["case"].get("tier", ""),
                                         r["case"].get("id", ""))):
        c = r.get("case", {})
        seeds = r.get("seeds", [])
        per_seed = " ".join(
            f"s{p.get('seed')}:{p.get('evaluation', {}).get('verdict', '?')}"
            for p in seeds
        ) or "—"
        summary = (seeds and str(
            seeds[-1].get("evaluation", {}).get("summary", ""))
            or str(r.get("generation", {}).get("error", ""))[:120])
        lines.append(
            f"| {c.get('id', '?')} | {c.get('category', '?')} "
            f"| {c.get('tier', '?')} | {r.get('verdict')} | "
            f"{float(r.get('score', 0.0)):.2f} | {per_seed} | "
            f"{summary.replace('|', '/')[:120]} |"
        )
    n_pass = sum(r.get("verdict") == "PASS" for r in rows)
    lines += ["", f"**{n_pass}/{len(rows)} passed**", ""]

    for r in rows:
        c = r.get("case", {})
        lines += [
            f"## {c.get('id', '?')}",
            "",
            f"- prompt: “{c.get('prompt', '')}”",
            f"- category/tier: {c.get('category')} / {c.get('tier')}"
            + (f" (template {c.get('template')})" if c.get("template") else ""),
            f"- intent: {c.get('intent', c.get('criterion', ''))}"
            if isinstance(c.get("intent", ""), str) else "",
            f"- verdict: **{r.get('verdict')}** "
            f"(score {float(r.get('score', 0.0)):.2f})",
        ]
        if c.get("baseline"):
            lines.append(f"- tester baseline: {c['baseline']}")
        gen = r.get("generation", {})
        if gen:
            lines.append(
                f"- generation: {gen.get('duration_s', '?')} s"
                + (", cached" if gen.get("cached") else "")
                + (f", ERROR: {str(gen.get('error'))[:200]}"
                   if gen.get("error") else "")
            )
        st = r.get("generated_graph_structure", {})
        if st.get("available"):
            lines.append(
                f"- generated graph: top-level cycle="
                f"{st.get('top_level_has_cycle')}, subgraph cycles="
                f"{st.get('subgraph_has_cycle')}"
            )
        for p in r.get("seeds", []):
            ev = p.get("evaluation", {})
            lines.append(
                f"- seed {p.get('seed')}: **{ev.get('verdict')}** "
                f"({float(ev.get('score', 0.0)):.2f}) — "
                f"{ev.get('summary', '')}"
            )
            for chk in ev.get("checks", []):
                mark = "✔" if chk.get("passed") else "✘"
                inj = " (injected)" if chk.get("data", {}).get("injected") \
                    else ""
                lines.append(
                    f"  - {mark} `{chk.get('name')}`{inj}: "
                    f"{chk.get('detail')}")
            diag = ev.get("diagnostics", {})
            if diag.get("hang"):
                lines.append(f"  - hang site: `{diag['hang']}`")
            if diag.get("planner_errors"):
                lines.append(
                    f"  - planner errors: {len(diag['planner_errors'])} "
                    f"(first: {diag['planner_errors'][0]})")
            if diag.get("max_node_iterations"):
                lines.append(
                    f"  - max node revisits observed at runtime: "
                    f"{diag['max_node_iterations']}")
            if diag.get("t0"):
                lines.append(f"  - $t0 referents: {diag['t0']}")
        crit = c.get("criterion")
        if isinstance(crit, dict):
            lines += [
                "", "<details><summary>criterion</summary>", "",
                "```json", json.dumps(crit, indent=1), "```",
                "", "</details>",
            ]
        lines.append("")
    report = out_root / "report.md"
    report.write_text("\n".join(lines))
    (out_root / "report.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n")
    return report
