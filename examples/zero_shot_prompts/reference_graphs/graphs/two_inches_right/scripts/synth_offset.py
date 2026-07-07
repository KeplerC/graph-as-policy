"""Synthesize a placement OBB offset from the grasped object's own OBB.

For a precise free-space set-down ("place it two inches to the right"), we
reuse the normal transport (which descends into a container OBB) by feeding
it a SYNTHETIC container OBB: a copy of the target's OBB translated by
(dx, dy) in the base frame, at the target's own resting height. The transport
then lowers the held object to that spot and releases — a controlled set-down,
not a drop. 2 inches = 0.0508 m; +y is image-right in the agentview frame.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox


class Output(TypedDict):
    container_obb: OrientedBoundingBox


def run(ctx: NodeContext, target_obb: OrientedBoundingBox,
        dy: float = 0.0508, dx: float = 0.0) -> Output:
    c = target_obb["center"]
    return {
        "container_obb": {
            "center": {"x": float(c["x"]) + float(dx),
                       "y": float(c["y"]) + float(dy),
                       "z": float(c["z"])},
            "extent": dict(target_obb["extent"]),
            "orientation": dict(target_obb["orientation"]),
        }
    }
