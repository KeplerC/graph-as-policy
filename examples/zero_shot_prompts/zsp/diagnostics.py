"""Generated-graph structure diagnostics (moved verbatim from v1
``criteria.py`` — evidence for the "does GaP support cycles" question).
"""

from __future__ import annotations

from typing import Any


def _edge_pairs(graph: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for e in graph.get("edges") or []:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            pairs.append((str(e[0]), str(e[1])))
    cond = graph.get("conditional_edges") or {}
    if isinstance(cond, dict):
        for src, targets in cond.items():
            if isinstance(targets, str):
                pairs.append((str(src), targets))
            elif isinstance(targets, dict):
                # Generated graphs use {"router_field": ..., "mapping":
                # {label: target}}; older/flat forms map label -> target
                # directly.
                mapping = targets.get("mapping")
                if not isinstance(mapping, dict):
                    mapping = {k: v for k, v in targets.items()
                               if k != "router_field"}
                pairs.extend((str(src), str(t)) for t in mapping.values()
                             if isinstance(t, str))
            elif isinstance(targets, (list, tuple)):
                pairs.extend((str(src), str(t)) for t in targets
                             if isinstance(t, str))
    return pairs


def _has_cycle(pairs: list[tuple[str, str]]) -> bool:
    adj: dict[str, list[str]] = {}
    for a, b in pairs:
        adj.setdefault(a, []).append(b)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, ()):  # noqa: B023
            c = color.get(v, WHITE)
            if c == GRAY:
                return True
            if c == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(
        color.get(u, WHITE) == WHITE and dfs(u) for u in list(adj)
    )


def workflow_structure(workflow: dict | None) -> dict[str, Any]:
    """Cycle report for a generated workflow: top level + per subgraph."""
    if not workflow:
        return {"available": False}
    top = _has_cycle(_edge_pairs(workflow))
    subs = {
        name: _has_cycle(_edge_pairs(sg or {}))
        for name, sg in (workflow.get("subgraphs") or {}).items()
    }
    return {
        "available": True,
        "top_level_has_cycle": top,
        "subgraph_has_cycle": subs,
        "any_cycle": bool(top or any(subs.values())),
    }


def max_node_iterations(trace_dir) -> int | None:
    """Max per-node iteration count from the executor's own node_data
    layout (node_data/<node>/iters/<k>). >1 proves the runtime revisited
    a node (i.e. executed a loop) in this run."""
    from pathlib import Path

    nd = Path(trace_dir) / "node_data"
    if not nd.is_dir():
        return None
    best = 0
    for node in nd.iterdir():
        iters = node / "iters"
        if iters.is_dir():
            best = max(best, sum(1 for _ in iters.iterdir()))
    return best or None
