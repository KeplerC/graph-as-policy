"""Report / VQA: perceive + answer, no manipulation required.

Cheapest cases in the matrix (~2-5 min: env boot + perception, no
transport). The ``reported`` leaf scans only PRODUCED outputs (tool
result summaries, executor node outputs) — never prompts/kwargs."""

from __future__ import annotations

from zsp.schema import Case

CASES: list[Case] = [
    Case(
        id="vqa_count_objects",
        prompt=("How many objects are on the table, not counting the "
                "basket? Report the number."),
        category="report", tier="T1", tags=("counting",),
        intent=(
            "perception-recall smoke for the whole report pipeline: the "
            "task-0 scene has exactly 6 movables (probed); the number 6 "
            "must appear in a produced output. The scene should ideally "
            "stay untouched (weighted, not gating)."
        ),
        criterion={
            "checks": [
                {"name": "count_reported_correctly", "weight": 0.8,
                 "expr": {"reported": {"pattern": r"\b(\d+)\b",
                                       "equals": 6}}},
                {"name": "scene_untouched", "weight": 0.2,
                 "gating": False,
                 "expr": {"all_untouched": {"tol_m": 0.03}}},
            ],
            "options": {"require_picked_up": False},
        },
        timeout_min=25,
    ),
    Case(
        id="vqa_closest_name",
        prompt="Which object is closest to the basket? Report its name.",
        category="report", tier="T1", tags=("spatial_query",),
        intent=(
            "perceive -> geometry -> language: the name of the t0-"
            "closest movable (seed-dependent) must appear in a produced "
            "output, tolerant of space/underscore separators."
        ),
        criterion={
            "checks": [
                {"name": "closest_name_reported", "weight": 1.0,
                 "expr": {"reported": {"body_pattern": {"$t0": {
                     "closest_to": {"anchor": "basket"}}}}}},
            ],
            "options": {"require_picked_up": False},
        },
        seeds=(0, 1),
        timeout_min=25,
    ),
    Case(
        id="vqa_distance_estimate",
        prompt=("Estimate the distance in centimeters between the "
                "alphabet soup and the milk, and report it."),
        category="report", tier="T1", tags=("metric_query",),
        intent=(
            "quantitative perception honesty: the reported number must "
            "be within 12 cm of the ground-truth xy distance (~39 cm at "
            "seed 0) — hallucinated numbers fail."
        ),
        criterion={
            "checks": [
                {"name": "distance_reported", "weight": 1.0,
                 "expr": {"reported": {
                     "pattern": r"\b(\d+(?:\.\d+)?)\b",
                     "equals": {"$t0": {"distance": {
                         "a": "alphabet_soup", "b": "milk",
                         "axis": "xy", "scale": 100}}},
                     "tol": 12}}},
            ],
            "options": {"require_picked_up": False},
        },
        timeout_min=25,
    ),
    Case(
        id="vqa_count_in_basket",
        prompt=("How many items are already inside the basket? Report "
                "the number."),
        category="report", tier="T1", tags=("counting", "containment"),
        sim_task="libero_zsp_prepacked/0",
        intent=(
            "containment perception (looking INTO a container): the "
            "pre-packed scene starts with exactly 2 items inside the "
            "basket across all inits (authored + probed)."
        ),
        criterion={
            "checks": [
                {"name": "in_basket_count_reported", "weight": 1.0,
                 "expr": {"reported": {"pattern": r"\b(\d+)\b",
                                       "equals": 2}}},
            ],
            "options": {"require_picked_up": False},
        },
        seeds=(0, 1),
        timeout_min=25,
    ),
    Case(
        id="vqa_size_compare",
        prompt=("Are the two soup cans the same size? Answer yes or no, "
                "and if they differ, name the color or position of the "
                "larger one."),
        category="report", tier="T1", tags=("size",),
        sim_task="libero_zsp_sizes/0",
        intent=(
            "perception-only feasibility probe for the size axis "
            "(calibrates size_bigger_soup expectations): a produced "
            "output must answer 'no' (they differ ~30% in scale)."
        ),
        criterion={
            "checks": [
                {"name": "answered_not_same", "weight": 1.0,
                 "expr": {"reported": {
                     "pattern": r"\b(no|not the same|different)\b"}}},
            ],
            "options": {"require_picked_up": False},
        },
        timeout_min=25,
    ),
]
