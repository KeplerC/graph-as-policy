"""Synthesize a placement OBB at the xy-midpoint of two reference OBBs.

For "place X halfway between A and B": after grasping X, we perceive A and B,
take the midpoint of their centers, and build a synthetic container OBB there
(at X's own resting height + extents) so the normal transport sets X down at
the midpoint.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox


class Output(TypedDict):
    container_obb: OrientedBoundingBox


def run(ctx: NodeContext, ref1_obb: OrientedBoundingBox,
        ref2_obb: OrientedBoundingBox,
        target_obb: OrientedBoundingBox) -> Output:
    a, b = ref1_obb["center"], ref2_obb["center"]
    z = float(target_obb["center"]["z"])
    return {
        "container_obb": {
            "center": {"x": (float(a["x"]) + float(b["x"])) / 2.0,
                       "y": (float(a["y"]) + float(b["y"])) / 2.0,
                       "z": z},
            "extent": dict(target_obb["extent"]),
            "orientation": dict(target_obb["orientation"]),
        }
    }
