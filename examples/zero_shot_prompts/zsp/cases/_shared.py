"""Shared criterion builders for the corpus modules.

(classic0.py keeps its own local copies — the parity gate freezes those.)
"""

from __future__ import annotations

from typing import Any

TASK0_OBJECTS = (
    "alphabet_soup", "butter", "cream_cheese", "milk",
    "salad_dressing", "tomato_sauce",
)

#: scene_sane bounds for classic (mounted-table) scenes, from probed
#: layouts: bodies live at x 0.25..0.78, |y| <= 0.26, z >= -0.05 in the
#: robot-base frame (base sits at world z 0.912); a fling reads far
#: outside. The floor-arena defaults do NOT fit here.
CLASSIC_OPTIONS: dict[str, Any] = {
    "scene_sane": {"min_z": -0.30, "max_abs_y": 0.70, "x": [0.0, 1.25]},
}


def pick_checks(target: str, container: str = "basket",
                weights: tuple[float, float, float] = (0.3, 0.5, 0.2),
                ) -> list[dict[str, Any]]:
    """Single pick-into-container: grasped + settled-inside + no strays."""
    return [
        {"name": f"{target}_grasped_and_lifted", "weight": weights[0],
         "expr": {"picked_up": {"target": target}}},
        {"name": f"{target}_in_{container}_final", "weight": weights[1],
         "expr": {"in_container": {"target": target,
                                   "container": container}}},
        {"name": "no_unrequested_packing", "weight": weights[2],
         "expr": {"no_new_packing": {"container": container,
                                     "allowed": [target]}}},
    ]


def released_on_table(target: str) -> dict[str, Any]:
    return {"at_end": {"and": [
        {"on_table": {"target": target}},
        {"not": {"held": {"target": target}}},
        {"settled": {"target": target}},
    ]}}
