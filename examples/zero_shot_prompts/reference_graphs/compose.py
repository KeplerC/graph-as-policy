"""Compose a multi-block pick-place reference graph programmatically.

A "block" is one robust pick-and-place: perceive target -> OBB -> grasp
(grasping-with-planner) -> perceive container -> transport into it. Blocks
chain SEQUENTIALLY, and because subgraph inputs bind by name with
last-writer-wins (executor.py:_lookup_cross_subgraph_output), each block's
grasp/transport picks up its OWN block's freshly-produced target_obb/
container_obb. This composes:

- ordered sequences (soup_then_milk, order_three_strict) — same container,
  different targets, order enforced by the chain;
- multi-destination sorting (sort_cans_boxes) — per-block container.

Perceive subgraphs come from libero_quickstart (simple fixed-object perceive);
grasp/transport come from grocery_packing (planner grasp + VAB transport). The
five referenced scripts are copied into the composed graph's scripts/ so it is
self-contained.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_QUICK = _REPO / "examples" / "libero_quickstart" / "graph"
_PACK = _REPO / "examples" / "grocery_packing" / "packing_graph"


def _perceive_sg(object_name: str, kind: str) -> dict:
    """kind: 'target' (outputs target_obb/mask) or 'container' (container_obb)."""
    outputs = (
        {"target_obb": {"$ref": "filter_obb.obb"},
         "target_mask": {"$ref": "perceive.mask"}}
        if kind == "target" else
        {"container_obb": {"$ref": "filter_obb.obb"},
         "container_mask": {"$ref": "perceive.mask"}}
    )
    script = f"scripts/{kind}/perceive_dino_vlm.py"
    return {
        "skill": "perceiving-objects",
        "inputs": {},
        "outputs": outputs,
        "nodes": {
            "observe": {"type": "tool", "tool": "robot.get_observation"},
            "perceive": {"type": "script", "script": script,
                         "inputs": {"cameras": {"$ref": "observe.cameras"},
                                    "object_name": object_name}},
            "filter_obb": {"type": "tool",
                           "tool": "geometry.filter_and_compute_obb",
                           "inputs": {"points": {"$ref": "perceive.cloud"}}},
            "found": {"type": "noop"},
        },
        "edges": [["START", "observe"], ["observe", "perceive"],
                  ["perceive", "filter_obb"], ["filter_obb", "found"],
                  ["found", "END"]],
        "conditional_edges": {},
        "exit": {"router_field": None, "success_values": ["found"]},
        "on_error": "not_found",
    }


def _grasp_sg() -> dict:
    """grocery_packing's planner grasp (grasping-with-planner + grasp_move.py)."""
    return {
        "skill": "grasping-with-planner",
        "inputs": {"target_obb": "OrientedBoundingBox"},
        "outputs": {"ee_pose_at_grasp": {"$ref": "observe.arms.0.ee_pose"}},
        "nodes": {
            "open": {"type": "tool", "tool": "robot.open_gripper",
                     "inputs": {"settle_steps": 40}},
            "compute_grasp": {"type": "tool",
                              "tool": "geometry.top_down_grasp_candidates",
                              "inputs": {"obb": {"$ref": "in.target_obb"},
                                         "z_offset": -0.03}},
            "goto_grasp": {"type": "script", "script": "scripts/grasp_move.py",
                           "inputs": {
                               "grasp_pose": {"$ref": "compute_grasp.candidates.poses.0"},
                               "candidate_poses": {"$ref": "compute_grasp.candidates.poses"},
                               "target_obb": {"$ref": "in.target_obb"},
                               "hover_z": 0.2}},
            "observe": {"type": "tool", "tool": "robot.get_observation"},
            "close": {"type": "tool", "tool": "robot.close_gripper",
                      "inputs": {"settle_steps": 60}},
            "grasped": {"type": "noop"},
        },
        "edges": [["START", "open"], ["open", "compute_grasp"],
                  ["compute_grasp", "goto_grasp"], ["goto_grasp", "observe"],
                  ["observe", "close"], ["close", "grasped"],
                  ["grasped", "END"]],
        "conditional_edges": {},
        "exit": {"router_field": None, "success_values": ["grasped"]},
        "on_error": "failed", "stage": "grasp",
    }


def _transport_sg(dx: float = 0.0, dy: float = 0.0) -> dict:
    """grocery_packing's VAB transport (transport_move.py + place_release.py).

    With a nonzero (dx, dy) the container OBB is shifted before the drop so
    successive objects into the SAME container land side-by-side (on the
    basket floor, contacting the basket) rather than stacking at its center
    (where a later object rests on an earlier one and fails the container-
    contact check)."""
    move_src = "in.container_obb"
    nodes: dict = {}
    edges = [["START", "transport_move"]]
    if dx or dy:
        nodes["shift"] = {
            "type": "script", "script": "scripts/shift_container.py",
            "inputs": {"container_obb": {"$ref": "in.container_obb"},
                       "dx": dx, "dy": dy}}
        move_src = "shift.container_obb"
        edges = [["START", "shift"], ["shift", "transport_move"]]
    nodes["transport_move"] = {
        "type": "script", "script": "scripts/transport_move.py",
        "inputs": {"container_obb": {"$ref": move_src}, "place_offset": -0.06}}
    nodes["release"] = {"type": "script", "script": "scripts/place_release.py",
                        "inputs": {"place_position": {"$ref": "transport_move.place_position"}}}
    nodes["placed"] = {"type": "noop"}
    edges += [["transport_move", "release"], ["release", "placed"],
              ["placed", "END"]]
    return {
        "skill": "transporting-objects",
        "inputs": {"container_obb": "OrientedBoundingBox"},
        "outputs": {"place_position": {"$ref": "transport_move.place_position"}},
        "nodes": nodes,
        "edges": edges,
        "conditional_edges": {},
        "exit": {"router_field": None, "success_values": ["placed"]},
        "on_error": "blocked", "stage": "place",
    }


_SHIFT_SCRIPT = '''\
"""Shift a container OBB's center by (dx, dy) so successive drops spread out."""
from __future__ import annotations
from typing import TypedDict
from gap import NodeContext
from gap_core.types import OrientedBoundingBox


class Output(TypedDict):
    container_obb: OrientedBoundingBox


def run(ctx: NodeContext, container_obb: OrientedBoundingBox,
        dx: float = 0.0, dy: float = 0.0) -> Output:
    c = container_obb["center"]
    return {"container_obb": {
        "center": {"x": float(c["x"]) + float(dx),
                   "y": float(c["y"]) + float(dy), "z": float(c["z"])},
        "extent": dict(container_obb["extent"]),
        "orientation": dict(container_obb["orientation"])}}
'''


def compose(steps: list[tuple[str, str]], out_dir: Path, *,
            name: str, description: str,
            drop_offsets: list[tuple[float, float]] | None = None) -> Path:
    """steps: [(target_desc, container_desc), ...] executed in order.

    drop_offsets: optional per-step (dx, dy) applied to that step's container
    OBB before the drop, to spread multiple objects across one container."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "scripts" / "target").mkdir(parents=True, exist_ok=True)
    (out_dir / "scripts" / "container").mkdir(parents=True, exist_ok=True)
    # self-contained scripts
    shutil.copy(_QUICK / "scripts/target/perceive_dino_vlm.py",
                out_dir / "scripts/target/perceive_dino_vlm.py")
    shutil.copy(_QUICK / "scripts/container/perceive_dino_vlm.py",
                out_dir / "scripts/container/perceive_dino_vlm.py")
    for s in ("grasp_move.py", "transport_move.py", "place_release.py"):
        shutil.copy(_PACK / "scripts" / s, out_dir / "scripts" / s)
    (out_dir / "scripts" / "shift_container.py").write_text(_SHIFT_SCRIPT)

    nodes: dict = {}
    subgraphs: dict = {}
    cond: dict = {}
    n = len(steps)
    for i, (tgt, cont) in enumerate(steps):
        t, c, g, tr = f"target_{i}", f"container_{i}", f"grasp_{i}", f"transport_{i}"
        dx, dy = (drop_offsets[i] if drop_offsets else (0.0, 0.0))
        nodes[t] = {"type": "subgraph", "ref": f"{t}_sg"}
        nodes[c] = {"type": "subgraph", "ref": f"{c}_sg"}
        nodes[g] = {"type": "subgraph", "ref": f"{g}_sg"}
        nodes[tr] = {"type": "subgraph", "ref": f"{tr}_sg"}
        subgraphs[f"{t}_sg"] = _perceive_sg(tgt, "target")
        subgraphs[f"{c}_sg"] = _perceive_sg(cont, "container")
        subgraphs[f"{g}_sg"] = _grasp_sg()
        subgraphs[f"{tr}_sg"] = _transport_sg(dx, dy)
        nxt = "done" if i == n - 1 else f"target_{i + 1}"
        cond[t] = {"router_field": "exit",
                   "mapping": {"found": c, "not_found": "abort"}}
        cond[c] = {"router_field": "exit",
                   "mapping": {"found": g, "not_found": "abort"}}
        cond[g] = {"router_field": "exit",
                   "mapping": {"grasped": tr, "failed": "abort"}}
        cond[tr] = {"router_field": "exit",
                    "mapping": {"placed": nxt, "blocked": "abort"}}
    nodes["done"] = {"type": "end", "status": "success"}
    nodes["abort"] = {"type": "end", "status": "failure",
                      "recovery": [{"tool": "robot.open_gripper", "inputs": {}},
                                   {"tool": "robot.go_home", "inputs": {}}]}
    wf = {
        "version": 3,
        "meta": {"name": name, "description": description},
        "nodes": nodes,
        "edges": [["START", "target_0"]],
        "conditional_edges": cond,
        "subgraphs": subgraphs,
    }
    (out_dir / "workflow.json").write_text(json.dumps(wf, indent=1) + "\n")
    return out_dir
