"""Object-permutation sweeps: one capability template stamped over
permuted objects/scenes (report groups these with a per-object
breakdown).

``pick_native``: every remaining ``libero_object_all_variance`` task
with its NATIVE target under classic task ordering (0=soup and
8-on-task-3 pudding are covered by T0 cases) — together the ten grocery
objects and their ten distinct 6-item distractor scenes are all
exercised. The prompt is the task's probed native language, so the
VAB env predicate is the same goal and co-gates (``success_source=
"both"``).

Plus permuted spatial-relation pairs on other scenes/axes."""

from __future__ import annotations

from zsp.cases._shared import pick_checks, released_on_table
from zsp.schema import Case, expand

#: task id (classic ordering, loader-verified) -> native target object.
PICK_NATIVE_TASKS: dict[int, str] = {
    1: "cream_cheese",
    2: "salad_dressing",
    4: "ketchup",
    5: "tomato_sauce",
    6: "butter",
    7: "milk",
    8: "chocolate_pudding",
    9: "orange_juice",
}


def _pick_native(item: tuple[int, str]) -> Case:
    task_id, obj = item
    spaced = obj.replace("_", " ")
    return Case(
        id=f"pick_native__{obj}",
        prompt=f"Pick the {spaced} and place it in the basket",
        category="grounding", tier="T1", tags=("native", "sweep"),
        sim_task=f"libero_object_all_variance/{task_id}",
        success_source="both",
        intent=(
            f"pick_native sweep: task {task_id}'s own language verbatim "
            f"({obj} among a distinct 6-item distractor subset, probed); "
            "the suite's contained_in predicate is the same goal and "
            "co-gates with the trace criterion."
        ),
        criterion={"checks": pick_checks(obj)},
    )


CASES: list[Case] = [
    *expand("pick_native", _pick_native, sorted(PICK_NATIVE_TASKS.items())),
    Case(
        id="lateral_of__ketchup_bbq",
        prompt="Place the ketchup to the right of the bbq sauce.",
        category="spatial", tier="T1", tags=("relation", "sweep"),
        template="lateral_of",
        sim_task="libero_object_all_variance/3",
        intent=(
            "the lateral-relation template permuted to a different "
            "object pair and scene (task-3: bbq sauce task, probed): "
            "ketchup set down 6-30 cm from the bbq sauce along y "
            "(either reading of 'right'), <= 15 cm drift in x, on the "
            "table, reference unmoved."
        ),
        criterion={"checks": [
            {"name": "ketchup_picked", "weight": 0.25,
             "expr": {"picked_up": {"target": "ketchup"}}},
            {"name": "ketchup_beside_bbq", "weight": 0.4,
             "expr": {"offset_along": {"target": "ketchup",
                                       "reference": "bbq_sauce",
                                       "axis": "y", "lo": 0.06, "hi": 0.30,
                                       "perp_max": 0.15}}},
            {"name": "ketchup_moved", "weight": 0.15,
             "expr": {"displaced": {"target": "ketchup", "axis": "xy",
                                    "lo": 0.04}}},
            {"name": "released_on_table", "weight": 0.2,
             "expr": released_on_table("ketchup")},
            {"name": "bbq_unmoved", "weight": 0.0,
             "expr": {"untouched": {"target": "bbq_sauce",
                                    "tol_m": 0.05}}},
        ]},
    ),
    Case(
        id="lateral_of__butter_front",
        prompt="Place the butter in front of the tomato sauce.",
        category="spatial", tier="T1", tags=("relation", "sweep"),
        template="lateral_of",
        sim_task="libero_object_all_variance/5",
        intent=(
            "the relation template permuted to the DEPTH axis on the "
            "task-5 scene: butter set down 6-30 cm from the tomato "
            "sauce along x (either reading of 'in front'), <= 15 cm "
            "drift in y, on the table."
        ),
        criterion={"checks": [
            {"name": "butter_picked", "weight": 0.25,
             "expr": {"picked_up": {"target": "butter"}}},
            {"name": "butter_before_tomato", "weight": 0.4,
             "expr": {"offset_along": {"target": "butter",
                                       "reference": "tomato_sauce",
                                       "axis": "x", "lo": 0.06, "hi": 0.30,
                                       "perp_max": 0.15}}},
            {"name": "butter_moved", "weight": 0.15,
             "expr": {"displaced": {"target": "butter", "axis": "xy",
                                    "lo": 0.04}}},
            {"name": "released_on_table", "weight": 0.2,
             "expr": released_on_table("butter")},
            {"name": "tomato_unmoved", "weight": 0.0,
             "expr": {"untouched": {"target": "tomato_sauce",
                                    "tol_m": 0.05}}},
        ]},
    ),
]
