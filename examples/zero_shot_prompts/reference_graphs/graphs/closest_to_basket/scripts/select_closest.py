"""Select the candidate OBB whose center is closest (xy) to the anchor.

Re-emits the winner as ``target_obb`` so the downstream grasp subgraph binds
it by name. This is the computed-referent core: the chosen object is derived
from the perceived layout, so it changes with the seed.

NOTE: candidate params are DECLARED EXPLICITLY (not ``**kwargs``). The executor
filters a script's resolved inputs to the run() signature's parameter names
(``gap/runtime/nodes.py``: ``accepted = set(sig.parameters) - {"ctx"}``), so a
``**kwargs`` catch-all — whose parameter name is the catch-all's, not the keys —
would silently drop every candidate. Any candidate whose perception failed
arrives as ``None`` and is skipped.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox


class Output(TypedDict):
    target_obb: OrientedBoundingBox
    chosen: str


def run(ctx: NodeContext, anchor_obb: OrientedBoundingBox,
        cand_alphabet_soup: OrientedBoundingBox | None = None,
        cand_butter: OrientedBoundingBox | None = None,
        cand_cream_cheese: OrientedBoundingBox | None = None,
        cand_milk: OrientedBoundingBox | None = None,
        cand_salad_dressing: OrientedBoundingBox | None = None,
        cand_tomato_sauce: OrientedBoundingBox | None = None) -> Output:
    cands = {
        "alphabet_soup": cand_alphabet_soup,
        "butter": cand_butter,
        "cream_cheese": cand_cream_cheese,
        "milk": cand_milk,
        "salad_dressing": cand_salad_dressing,
        "tomato_sauce": cand_tomato_sauce,
    }
    a = anchor_obb["center"]
    ax, ay = float(a["x"]), float(a["y"])
    best_name, best_obb, best_d = None, None, float("inf")
    for name, obb in cands.items():
        if not isinstance(obb, dict) or "center" not in obb:
            continue
        c = obb["center"]
        d = ((float(c["x"]) - ax) ** 2 + (float(c["y"]) - ay) ** 2) ** 0.5
        if d < best_d:
            best_name, best_obb, best_d = name, obb, d
    if best_obb is None:
        raise ValueError("no valid candidate OBBs to select from")
    return {"target_obb": best_obb, "chosen": best_name}
