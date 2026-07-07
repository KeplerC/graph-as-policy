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
