"""Coding / control-flow: computed referents, branching, iteration,
verify loops, multi-destination routing."""

from __future__ import annotations

from zsp.cases._shared import pick_checks
from zsp.schema import Case


def _pick_resolved(target_ref: dict, weights=(0.3, 0.5, 0.2)) -> list[dict]:
    """pick_checks for a $t0-resolved referent (same ref in every slot)."""
    return [
        {"name": "target_grasped_and_lifted", "weight": weights[0],
         "expr": {"picked_up": {"target": target_ref}}},
        {"name": "target_in_basket_final", "weight": weights[1],
         "expr": {"in_container": {"target": target_ref,
                                   "container": "basket"}}},
        {"name": "no_unrequested_packing", "weight": weights[2],
         "expr": {"no_new_packing": {"allowed": [target_ref]}}},
    ]


CASES: list[Case] = [
    Case(
        id="closest_to_basket",
        prompt=("Pick up whichever object is currently closest to the "
                "basket and place it in the basket."),
        category="coding", tier="T1", tags=("computed_referent",),
        intent=(
            "computed referent: the correct object is derived from the "
            "layout at t0 (closest movable to the basket, xy) and "
            "CHANGES with the seed — a graph must compute it, not "
            "memorize it. Exactly that object packed, nothing else."
        ),
        criterion={"checks": _pick_resolved(
            {"$t0": {"closest_to": {"anchor": "basket"}}})},
        seeds=(0, 1),
        timeout_min=60,
    ),
    Case(
        id="cond_closer_soup_or_milk",
        prompt=("If the alphabet soup is closer to the basket than the "
                "milk is, put the soup in the basket. Otherwise, put the "
                "milk in the basket."),
        category="coding", tier="T1", tags=("conditional",),
        intent=(
            "flagship conditional: the criterion resolves the branch "
            "truth per seed ($t0 closest_to among {soup, milk}) — "
            "packing the wrong branch's object fails both the pick and "
            "the stray check."
        ),
        criterion={"checks": _pick_resolved(
            {"$t0": {"closest_to": {"anchor": "basket",
                                    "among": ["alphabet_soup", "milk"]}}})},
        seeds=(0, 1),
        timeout_min=75,
    ),
    Case(
        id="tallest_object",
        prompt="Put the tallest object on the table into the basket.",
        category="coding", tier="T1", tags=("computed_referent",),
        intent=(
            "superlative over a perceived physical property: tallest by "
            "AABB height resolves to the milk carton (16 cm vs <= 9 cm "
            "for everything else — unambiguous margin, probed)."
        ),
        criterion={"checks": _pick_resolved(
            {"$t0": {"largest": {"by": "height"}}})},
        timeout_min=60,
    ),
    Case(
        id="cond_count_bottles",
        prompt=("Count the bottles on the table. If there are at least "
                "two, put exactly one of them in the basket; otherwise "
                "do nothing."),
        category="coding", tier="T1", tags=("conditional", "counting"),
        sim_task="libero_object_all_variance/2",
        intent=(
            "count -> threshold -> act on the task-2 scene, which has "
            "exactly TWO bottle-shaped items (salad_dressing, ketchup — "
            "probed): exactly one of them ends in the basket, everything "
            "else stays out. Guards both under-counting (does nothing) "
            "and over-acting (packs both)."
        ),
        criterion={"checks": [
            {"name": "exactly_one_bottle_in", "weight": 0.7,
             "expr": {"count_in": {"container": "basket",
                                   "targets": ["salad_dressing", "ketchup"],
                                   "count": 1}}},
            {"name": "nothing_else_packed", "weight": 0.3,
             "expr": {"no_new_packing": {"allowed": ["salad_dressing",
                                                     "ketchup"]}}},
            {"name": "count_reported", "weight": 0.0, "gating": False,
             "expr": {"reported": {"pattern": r"\b(2|two)\b"}}},
        ]},
        timeout_min=60,
    ),
    Case(
        id="fanout_compare_cans",
        prompt=("For each of the two cans on the table, report its "
                "distance to the basket in centimeters. Then put the can "
                "that is farther from the basket into the basket."),
        category="coding", tier="T1", tags=("fanout", "computed_referent"),
        intent=(
            "per-item computation feeding a comparison decision: the "
            "farther can (resolved at t0 among the two can items) is the "
            "one that must be packed; the reported distances are "
            "recorded as diagnostics."
        ),
        criterion={"checks": [
            *_pick_resolved(
                {"$t0": {"farthest_from": {
                    "anchor": "basket",
                    "among": ["alphabet_soup", "tomato_sauce"]}}},
                weights=(0.25, 0.45, 0.1),
            ),
            {"name": "distances_reported", "weight": 0.2, "gating": False,
             "expr": {"reported": {
                 "pattern": r"(\d+(?:\.\d+)?)\s*(?:cm|centimet)"}}},
        ]},
        timeout_min=60,
    ),
    Case(
        id="grasp_retry_budget",
        prompt=("Try to pick up the cream cheese and put it in the "
                "basket. If a grasp attempt fails, retry — but make at "
                "most three grasp attempts in total, then stop."),
        category="coding", tier="T1", tags=("retry", "budget"),
        intent=(
            "bounded-retry control flow on the historically hard low "
            "object: PASS = packed within <= 3 gripper-close attempts, "
            "OR a clean stop after <= 3 attempts with nothing packed "
            "and the scene undisturbed. Unbounded retry loops fail the "
            "budget."
        ),
        criterion={
            "checks": [
                {"name": "attempts_within_budget", "weight": 0.3,
                 "expr": {"attempt_count": {
                     "tool_regex": r"close_gripper", "min": 1, "max": 3}}},
                {"name": "packed_or_clean_stop", "weight": 0.7,
                 "expr": {"or": [
                     {"and": [
                         {"picked_up": {"target": "cream_cheese"}},
                         {"in_container": {"target": "cream_cheese",
                                           "container": "basket"}}]},
                     {"no_new_packing": {"allowed": []}},
                 ]}},
            ],
            "options": {"require_picked_up": False},
        },
        timeout_min=60,
    ),
    Case(
        id="verify_until_packed",
        prompt=("Place the tomato sauce in the basket. Then check that "
                "it is really inside; if it is not, try again, up to "
                "three total attempts."),
        category="coding", tier="T1", tags=("verify_loop",),
        intent=(
            "closed-loop verify-act: the gate is simply tomato sauce "
            "settled in the basket; the interesting signal is the "
            "non-gating loop evidence (runtime node revisits, attempt "
            "count <= 3) separating single-shot graphs from checked "
            "ones."
        ),
        criterion={"checks": [
            *pick_checks("tomato_sauce"),
            {"name": "loop_evidence", "weight": 0.0, "gating": False,
             "expr": {"node_revisits": {}}},
            {"name": "attempts_bounded", "weight": 0.0, "gating": False,
             "expr": {"attempt_count": {"tool_regex": r"close_gripper",
                                        "max": 3}}},
        ]},
        timeout_min=75,
    ),
    Case(
        id="ordinal_middle_ketchup",
        prompt=("Three ketchup bottles are in a row. Pick up the middle "
                "one and put it in the basket."),
        category="coding", tier="T1", tags=("ordinal", "computed_referent"),
        sim_task="libero_zsp_colors/0",
        intent=(
            "geometric ordinal referent: the row order permutes across "
            "inits (probed), so 'the middle one' is a different bottle "
            "per seed — resolved at t0 as the median along the row axis."
        ),
        criterion={"checks": _pick_resolved(
            {"$t0": {"middle_of": {
                "axis": "y",
                "among": ["ketchup", "green_ketchup", "blue_ketchup"]}}})},
        seeds=(0, 1),
        timeout_min=60,
    ),
    # ---- new-scene sorting / retrieval -------------------------------------
    Case(
        id="sort_cans_boxes",
        prompt=("Sort the groceries: put the two cans in the basket and "
                "the two boxes on the wooden tray."),
        category="coding", tier="T1", tags=("sorting",),
        sim_task="libero_zsp_sorting/0",
        intent=(
            "two-destination categorical routing: cans (alphabet_soup, "
            "tomato_sauce) into the basket, boxes (butter, cream_cheese) "
            "onto the tray (scored center-inside the tray's rimmed AABB, "
            "probed). Cross-placed items score zero on their check."
        ),
        criterion={"checks": [
            {"name": "cans_in_basket", "weight": 0.5,
             "expr": {"count_in": {"container": "basket",
                                   "targets": ["alphabet_soup",
                                               "tomato_sauce"],
                                   "count": "all"}}},
            {"name": "boxes_on_tray", "weight": 0.5,
             "expr": {"count_in": {"container": "wooden_tray",
                                   "targets": ["butter", "cream_cheese"],
                                   "count": "all"}}},
            {"name": "no_cross_packing_basket", "weight": 0.0,
             "expr": {"no_new_packing": {"container": "basket",
                                         "allowed": ["alphabet_soup",
                                                     "tomato_sauce"]}}},
            {"name": "per_object_report", "weight": 0.0, "gating": False,
             "expr": {"pack_report": {"targets": ["alphabet_soup",
                                                  "tomato_sauce", "butter",
                                                  "cream_cheese"],
                                      "mode": "final"}}},
        ], "options": {"statics": ["wooden_tray"]}},
        timeout_min=120,
    ),
    Case(
        id="sort_by_proximity",
        prompt=("Put each grocery item into whichever container is "
                "closer to it: the basket or the wooden tray."),
        category="coding", tier="T2", tags=("sorting", "computed_referent"),
        sim_task="libero_zsp_sorting/0",
        expected=(
            "per-item computed destination with no category shortcut — "
            "the assignment differs per init (authored)"
        ),
        intent=(
            "geometric assignment: each of the four items must end in "
            "its t0-nearest container. Scored as one containment check "
            "per item against the container resolved at t0."
        ),
        criterion={"checks": [
            *[
                {"name": f"{obj}_in_nearest",
                 "weight": 0.25,
                 "expr": {"in_container": {
                     "target": obj,
                     "container": {"$t0": {"closest_to": {
                         "anchor": obj,
                         "among": ["basket", "wooden_tray"]}}}}}}
                for obj in ("alphabet_soup", "tomato_sauce", "butter",
                            "cream_cheese")
            ],
        ], "options": {"statics": ["wooden_tray"],
                       "require_picked_up": False}},
        timeout_min=120,
    ),
    Case(
        id="empty_basket",
        prompt=("Take everything out of the basket and place it on the "
                "table. Leave the basket empty."),
        category="coding", tier="T2", tags=("retrieval", "iteration"),
        sim_task="libero_zsp_prepacked/0",
        expected=(
            "in-basket retrieval is genuinely tight (12 cm cavity, "
            "cuRobo collision margins) — the physical bottleneck case"
        ),
        intent=(
            "inverse packing on the pre-packed scene (soup + butter "
            "start inside, probed): the basket ends EMPTY, the two "
            "retrieved items settle on the table, and the two table "
            "items are not disturbed into the basket."
        ),
        criterion={
            "checks": [
                {"name": "basket_empty", "weight": 0.5,
                 "expr": {"count_in": {"container": "basket",
                                       "targets": "movables",
                                       "count": 0}}},
                {"name": "retrieved_on_table", "weight": 0.5,
                 "expr": {"at_end": {"and": [
                     {"on_table": {"target": "alphabet_soup"}},
                     {"on_table": {"target": "butter"}},
                     {"settled": {"target": "alphabet_soup"}},
                     {"settled": {"target": "butter"}},
                 ]}}},
                {"name": "retrieval_grasps", "weight": 0.0, "gating": False,
                 "expr": {"and": [
                     {"picked_up": {"target": "alphabet_soup"}},
                     {"picked_up": {"target": "butter"}},
                 ]}},
            ],
            "options": {"require_picked_up": False},
        },
        timeout_min=90,
    ),
    Case(
        id="cond_look_inside_act",
        prompt=("Check whether the alphabet soup is inside the basket. "
                "If it is, take it out and set it on the table. If it is "
                "not, put it in."),
        category="coding", tier="T2", tags=("conditional", "retrieval"),
        sim_task="libero_zsp_prepacked/0",
        expected=(
            "perceive-containment -> branch -> in-basket retrieval; the "
            "soup STARTS inside (authored), so the correct branch is "
            "retrieval"
        ),
        intent=(
            "the soup starts inside the basket (probed), so PASS = soup "
            "genuinely picked up and settled on the table OUTSIDE the "
            "basket at the end."
        ),
        criterion={"checks": [
            {"name": "soup_retrieved", "weight": 0.4,
             "expr": {"picked_up": {"target": "alphabet_soup"}}},
            {"name": "soup_on_table_outside", "weight": 0.6,
             "expr": {"at_end": {"and": [
                 {"on_table": {"target": "alphabet_soup"}},
                 {"not": {"held": {"target": "alphabet_soup"}}},
                 {"settled": {"target": "alphabet_soup"}},
             ]}}},
        ]},
        timeout_min=75,
    ),
]
