"""Loop control for grocery_packing's perceive_next subgraph.

Termination is UNPRIVILEGED and layered, matching the canonical
``perceiving-next-item`` skill (and the public grocery-packing benchmark's
graph) — the same policy runs unchanged on a real robot:

  1. **VLM completion check (primary).** Every pass, the exterior
     (agentview) frame goes to the VLM: "have ALL the grocery items been
     placed inside the basket?". A confident YES exits the loop cleanly on
     "none" (success). The check only ever forces a STOP — a NO or an
     unavailable VLM never forces the loop to continue, so the guards below
     still guarantee termination.
  2. **Env verdict (secondary, sim-only).** ``sim.check_success`` is polled
     as a backstop; ``task_completed`` reads the cached success the env
     computed in its own ``step()`` (see ``FrankaLiberoEnv.task_completed``),
     never re-evaluating the stateful ``pack_all_into`` predicate (which
     teleports delivered objects on each evaluation). Wrapped so non-sim
     connectors fall through cleanly.
  3. **No-progress guard.** If the SAME perceived target (cloud centroid
     within ``_SAME_TARGET_M``) comes back ``_STUCK_LIMIT`` consecutive
     passes, the loop cannot make progress on it -> "none" rather than
     re-grasping forever. A total-pass budget backstops alternation. State
     is keyed by the workflow's trace dir (unique per trial/run) so
     benchmark workers never leak one trial's state into the next.
  4. **Perception.** Otherwise route on the perceive verdict: "found" when
     an item was returned, "none" when nothing was.

We do NOT geometrically reject items near the basket (a real item beside the
basket falls inside any reasonable radius and was being dropped); the perceive
node's ``object_description`` keeps the tournament from picking the basket
while a real item remains. ``container_obb`` is still accepted (the subgraph
wires it in) but unused. Authored by build_graph.py.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile

import numpy as np

from gap import NodeContext

logger = logging.getLogger(__name__)

# Stop a cyclic packing episode that can no longer make progress: if the SAME
# target is re-perceived this many consecutive passes (no delivery removed it,
# nothing moved), exit cleanly rather than looping.
_STUCK_LIMIT = 3
# Two perceived-cloud centroids closer than this (m, XY) are the same target.
_SAME_TARGET_M = 0.03
# Absolute pass budget — backstop for pathological alternation between two
# ungraspable targets.
_MAX_PASSES = 30

# Per-pass VLM completion check — the primary, unprivileged stop signal.
_PACKED_PROMPT = (
    "A robot is packing grocery items from a table into a basket. Looking at "
    "this image of the table and the basket, have ALL the grocery items been "
    "placed INSIDE the basket, leaving the table surface empty except for the "
    "basket itself (and the robot)? Answer YES only if there is NO grocery "
    "item left resting on the table outside the basket."
)


def _exterior_rgb(ctx: NodeContext) -> np.ndarray | None:
    """The agentview (exterior) RGB frame — the wrist eye-in-hand view is too
    close to judge the whole tabletop. Returns None if no camera is available."""
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else None
    cams = cams or []
    if isinstance(cams, dict):
        cams = list(cams.values())
    ext = [c for c in cams if "eye_in_hand" not in (c.get("name") or "")]
    cam = next(iter(ext or cams), None)
    if cam is None or cam.get("rgb") is None:
        return None
    return np.asarray(cam["rgb"])


def _vlm_all_packed(ctx: NodeContext) -> bool | None:
    """Ask the VLM whether every object is in the basket. Returns True/False,
    or None when the check could not run (no camera, or VLM error/missing
    creds) so the caller falls back to the env + guard + perception exits."""
    try:
        rgb = _exterior_rgb(ctx)
        if rgb is None:
            logger.warning("[route] VLM all-packed check skipped: no camera frame")
            return None
        resp = ctx.tool("vlm.query_yes_no", prompt=_PACKED_PROMPT, image=rgb)
        ans = bool(resp.get("answer"))
        logger.info("[route] VLM all-packed? answer=%s text=%r",
                    ans, str(resp.get("text"))[:200])
        return ans
    except Exception as exc:  # noqa: BLE001  (auth/network/no-bundle -> fall through)
        logger.warning("[route] VLM all-packed check failed: %s", exc)
        return None


def _progress_path(ctx: NodeContext | None) -> str:
    """Per-episode scratch file for the no-progress counter.

    Keyed by the workflow's trace dir (unique per trial/run), NOT just the
    PID: a benchmark worker executes many trials in one process, and a
    PID-keyed file carries the previous trial's state into the next trial's
    first pass, ending its loop before anything is grasped.
    """
    trace = getattr(ctx, "_trace", None) if ctx is not None else None
    out_dir = getattr(trace, "_output_dir", None) if trace is not None else None
    if out_dir is not None:
        tag = hashlib.sha1(str(out_dir).encode()).hexdigest()[:12]
    else:  # no-trace run: fall back to the PID (single episode per process)
        tag = f"pid{os.getpid()}"
    return os.path.join(tempfile.gettempdir(), f".gap_route_progress_{tag}")


def _no_progress(target_xy: tuple[float, float] | None,
                 ctx: NodeContext | None = None) -> bool:
    """Unprivileged stuck detector, keyed on the perceived target itself.

    A successful delivery removes the item, so the next pass perceives a
    DIFFERENT target; re-perceiving the same centroid means the last
    grasp+transport cycle changed nothing.
    """
    if target_xy is None:
        return False
    path = _progress_path(ctx)
    px, py, consec, total = 1e9, 1e9, 0.0, 0.0
    try:
        with open(path) as f:
            px, py, consec, total = (float(x) for x in f.read().split())
    except Exception:
        pass
    same = abs(target_xy[0] - px) < _SAME_TARGET_M and \
        abs(target_xy[1] - py) < _SAME_TARGET_M
    consec = consec + 1 if same else 0
    total += 1
    try:
        with open(path, "w") as f:
            f.write(f"{target_xy[0]} {target_xy[1]} {consec} {total}")
    except Exception:
        pass
    if total > _MAX_PASSES:
        logger.info("[route] pass budget exceeded (%d) -> stopping loop", _MAX_PASSES)
        return True
    if consec >= _STUCK_LIMIT:
        logger.info("[route] same target (%.3f, %.3f) for %d consecutive passes "
                    "-> stopping loop", target_xy[0], target_xy[1], _STUCK_LIMIT)
        return True
    return False


def run(ctx: NodeContext, found: bool, cloud: dict, container_obb: dict | None = None) -> dict:
    # 1: VLM completion check — the primary, unprivileged stop signal.
    if _vlm_all_packed(ctx) is True:
        logger.info("[route] VLM reports all objects packed -> stopping loop")
        return {"route": "none"}

    # 2: env verdict (secondary backstop) DISABLED for this reference.
    # The all_objects case runs on libero_object_all_variance/0, whose env
    # task is SINGLE-object (pick the alphabet soup) — so task_completed
    # fires after the first pack and would quit the loop with 1/6 packed.
    # The "pack every item" goal here is scored by our own criterion, not
    # the env's task, so we rely on the (correct) VLM completion check and
    # the no-progress guard instead. Kept as a log-only observation.
    try:
        sc = ctx.tool("sim.check_success")
        cr = sc.get("completion_rate")
        logger.info(
            "[route] (env backstop disabled) completion_rate=%s "
            "task_completed=%s perceived_item=%s",
            f"{cr:.3f}" if isinstance(cr, (int, float)) else cr,
            sc.get("task_completed"), bool(found),
        )
    except Exception:
        pass  # non-sim connector -> fall through

    # 4 (cheap verdicts first): perception.
    if not found:
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        return {"route": "none"}

    # 3: unprivileged no-progress guard on the perceived target itself.
    centroid = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
    if _no_progress(centroid, ctx):
        return {"route": "none"}
    return {"route": "found"}
