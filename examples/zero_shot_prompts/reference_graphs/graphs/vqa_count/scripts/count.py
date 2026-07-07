"""Count grocery items on the table via a single VLM query and report it.

The returned dict is summarized into events.jsonl (and the executor's
node_data), where the ``reported`` criterion leaf scans for the number.
No manipulation — this is a perception+report reference.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    count: int
    answer_text: str


_PROMPT = (
    "Count the grocery items resting on the table in this image. Do NOT "
    "count the wicker basket, the table surface, or the robot arm — only "
    "the packaged grocery products (cans, boxes, bottles, cartons). Reply "
    "with ONLY the integer count."
)


def _exterior_rgb(cameras: Any):
    cams = cameras or []
    if isinstance(cams, dict):
        cams = list(cams.values())
    ext = [c for c in cams if "eye_in_hand" not in (c.get("name") or "")]
    cam = next(iter(ext or cams), None)
    return cam.get("rgb") if cam else None


def run(ctx: NodeContext, cameras: Any) -> Output:
    import numpy as np

    rgb = _exterior_rgb(cameras)
    img = np.asarray(rgb) if rgb is not None else None
    res = ctx.tool("vlm.query", prompt=_PROMPT, image=img)
    text = res.get("text", "") if isinstance(res, dict) else str(res)
    m = re.search(r"\d+", text)
    return {"count": int(m.group()) if m else -1, "answer_text": text[:160]}
