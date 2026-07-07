"""Robustness: one graph across genuinely different layouts/suites."""

from __future__ import annotations

from zsp.cases._shared import pick_checks
from zsp.schema import Case

CASES: list[Case] = [
    Case(
        id="posvar_ketchup",
        prompt="Pick up the ketchup and place it in the basket.",
        category="robustness", tier="T1", tags=("position_variance",),
        sim_task="libero_object_target_pos_var20x20/4",
        intent=(
            "the 20x20 position-grid suite: one generated graph executed "
            "on three very different layouts (all seeds must pass)."
        ),
        criterion={"checks": pick_checks("ketchup")},
        seeds=(0, 1, 2),
        timeout_min=90,
    ),
    Case(
        id="permutation_milk",
        prompt="Pick up the milk and place it in the basket.",
        category="robustness", tier="T1", tags=("permutation",),
        sim_task="libero_object_target_permutation_variance/7",
        intent=(
            "object-permutation suite: same slots, shuffled identities "
            "across inits — kills graphs that memorize locations instead "
            "of re-perceiving."
        ),
        criterion={"checks": pick_checks("milk")},
        seeds=(0, 1),
        timeout_min=75,
    ),
    Case(
        id="permutation_packing_native",
        prompt="Pick all the objects and place them in the basket",
        category="robustness", tier="T1", tags=("permutation", "packing"),
        sim_task="permutation_packing/0",
        success_source="both",
        env_consumes_packed=True,
        intent=(
            "packing under permuted collision-free inits, prompt "
            "VERBATIM from the suite's own language — the env's "
            "pack_all_into predicate is the same task, so it gates "
            "alongside the trace criterion (grasp-and-lift + packed-or-"
            "consumed per object)."
        ),
        criterion={"checks": [
            {"name": "all_packed_or_consumed", "weight": 1.0,
             "expr": {"pack_report": {"targets": "movables",
                                      "mode": "grasp_and_seen"}}},
        ]},
        timeout_min=120,
    ),
    Case(
        id="moving_basket",
        prompt="Pick up the alphabet soup and place it in the basket.",
        category="robustness", tier="T2", tags=("perturbed",),
        sim_task="libero_object/0",
        env_overrides={"GAP_LIBERO_PERTURBED": "1"},
        expected=(
            "the perturbed env moves the basket on a scripted circle — "
            "stale-t0 perception misses by up to the containment "
            "tolerance; graphs that re-perceive late (or track) pass"
        ),
        intent=(
            "classic libero_object/0 scene (note the _1 body suffixes) "
            "with the scripted moving basket: soup grasped-and-lifted "
            "and settled inside the basket at its FINAL position (the "
            "containment check follows the basket's live pose)."
        ),
        criterion={
            # classic libero_object is a floor-arena scene: the default
            # workspace envelope applies; only the container name changes.
            "checks": pick_checks("alphabet_soup_1", container="basket_1"),
            "options": {"container": "basket_1"},
        },
        timeout_min=75,
    ),
]
