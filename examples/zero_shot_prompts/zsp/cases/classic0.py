"""The original 13-case matrix (T0 regression), migrated to the DSL.

Prompts, baselines, seeds, timeouts, scenes, and check weights are the
v1 ``cases.py``/``criteria.py`` values verbatim — ``test_parity``
asserts identical verdicts and score windows on the v1 synthetic
scenarios. Scene compositions come from probed layouts (``layouts/``);
the pudding_basket docstring drift found there is corrected in its
``intent``.
"""

from __future__ import annotations

from typing import Any

from zsp.schema import Case

TASK0_OBJECTS = (
    "alphabet_soup", "butter", "cream_cheese", "milk",
    "salad_dressing", "tomato_sauce",
)


def _pick_place_checks(target: str) -> list[dict[str, Any]]:
    """v1 ``_pick_place``: 0.3 grasped + 0.5 in-basket + 0.2 no-stray."""
    return [
        {"name": f"{target}_grasped_and_lifted", "weight": 0.3,
         "expr": {"picked_up": {"target": target}}},
        {"name": f"{target}_in_basket_final", "weight": 0.5,
         "expr": {"in_container": {"target": target, "container": "basket"}}},
        {"name": "no_unrequested_packing", "weight": 0.2,
         "expr": {"no_new_packing": {"allowed": [target]}}},
    ]


_GRAPH_DIAG_CHECKS = [
    {"name": "workflow_cycles", "weight": 0.0, "gating": False,
     "expr": {"workflow_has_cycle": {}}},
    {"name": "runtime_node_revisits", "weight": 0.0, "gating": False,
     "expr": {"node_revisits": {}}},
]


CASES: list[Case] = [
    Case(
        id="soup_basket",
        prompt="Pick up the alphabet soup can and place it in the basket",
        category="grounding",
        tier="T0",
        baseline="Success, robust to variations in position",
        intent=(
            "per seed: alphabet_soup grasped-and-lifted at some point, and "
            "finishes settled inside the basket with no other object "
            "packed; case passes only if ALL seeds pass (the robustness "
            "claim)"
        ),
        criterion={"checks": _pick_place_checks("alphabet_soup")},
        seeds=(0, 1, 2),
        timeout_min=60,
    ),
    Case(
        id="all_objects",
        prompt="Pick up all objects and place them in the basket",
        category="coding",
        tier="T0",
        baseline=(
            "Failure — picks up the alphabet soup can and places it in "
            "the basket; perception fails for other objects (unable to "
            "pick up any object other than alphabet soup). Reported "
            "cause: the generated graph must contain cycles (tasks that "
            "must be repeated), but GaP seems to only support directed "
            "acyclic graphs."
        ),
        intent=(
            "all 6 objects finish settled inside the basket (score = "
            "fraction packed); per-object grasp events reported. "
            "Diagnostics (not gated): whether the generated workflow "
            "contains cycles, and how many times the executor actually "
            "revisited a node at runtime."
        ),
        criterion={"checks": [
            {"name": "all_objects_in_basket", "weight": 1.0,
             "expr": {"pack_report": {"targets": "movables",
                                      "mode": "final"}}},
            *_GRAPH_DIAG_CHECKS,
        ]},
        timeout_min=90,
        tags=("cycles",),
    ),
    Case(
        id="soup_then_milk",
        prompt=(
            "Pick up the alphabet soup and place it in the basket, then "
            "pick up the milk and place it in the basket"
        ),
        category="temporal",
        tier="T0",
        baseline=(
            "Failure — alphabet soup is successfully placed, and "
            "perception succeeds for the milk, but there seems to be a "
            "mesh loading error when grasping the milk (simulation "
            "consistently hangs at this point)."
        ),
        intent=(
            "two distinct grasp-and-lift events (soup, milk); both finish "
            "settled in the basket; soup enters the basket strictly "
            "before milk. A wall-clock timeout is classified as a HANG "
            "and the hang site is reported from the event log."
        ),
        criterion={"checks": [
            {"name": "alphabet_soup_grasped_and_lifted", "weight": 0.15,
             "expr": {"picked_up": {"target": "alphabet_soup"}}},
            {"name": "alphabet_soup_in_basket_final", "weight": 0.25,
             "expr": {"in_container": {"target": "alphabet_soup",
                                       "container": "basket"}}},
            {"name": "milk_grasped_and_lifted", "weight": 0.15,
             "expr": {"picked_up": {"target": "milk"}}},
            {"name": "milk_in_basket_final", "weight": 0.25,
             "expr": {"in_container": {"target": "milk",
                                       "container": "basket"}}},
            {"name": "soup_enters_basket_before_milk", "weight": 0.2,
             "expr": {"before": {
                 "first": {"enters_container": {"target": "alphabet_soup"}},
                 "then": {"enters_container": {"target": "milk"}}}}},
        ]},
        timeout_min=45,
        tags=("hang",),
    ),
    Case(
        id="two_inches_right",
        prompt=(
            "Pick up the alphabet soup and place it down two inches to "
            "the right"
        ),
        category="spatial",
        tier="T0",
        baseline=(
            "Failure — picks up the alphabet soup can, but knocks it "
            "down. (Later reported as success after codegen-catalog "
            "introspection, arms[0] $ref, and curobo joint-state fixes; "
            "tester unsure all were necessary.)"
        ),
        intent=(
            "soup grasped-and-lifted; finishes on the table (not in the "
            "basket, released, settled) displaced 5.1 +/- 3 cm along y "
            "with <= 5 cm drift in x; still upright (initially-up body "
            "axis within 30 degrees of vertical). 'Right' accepted in "
            "either y direction: +y is image-right in the agentview "
            "video, -y is the robot's right."
        ),
        criterion={"checks": [
            {"name": "alphabet_soup_grasped_and_lifted", "weight": 0.25,
             "expr": {"picked_up": {"target": "alphabet_soup"}}},
            {"name": "displaced_two_inches_along_y", "weight": 0.35,
             "expr": {"and": [
                 {"displaced": {"target": "alphabet_soup", "axis": "y",
                                "lo": 0.0208, "hi": 0.0808}},
                 {"displaced": {"target": "alphabet_soup", "axis": "x",
                                "lo": 0.0, "hi": 0.05}},
             ]}},
            {"name": "still_upright", "weight": 0.25,
             "expr": {"tilt_from_initial": {"target": "alphabet_soup",
                                            "max_deg": 30}}},
            {"name": "released_on_table", "weight": 0.15,
             "expr": {"at_end": {"and": [
                 {"on_table": {"target": "alphabet_soup"}},
                 {"not": {"held": {"target": "alphabet_soup"}}},
                 {"settled": {"target": "alphabet_soup"}},
             ]}}},
        ]},
        seeds=(0, 1),
        timeout_min=75,
    ),
    Case(
        id="cream_cheese",
        prompt="Pick up the cream cheese and put it in the basket",
        category="spatial",
        tier="T0",
        baseline=(
            "Failure — CuRobo planning failure, unable to calculate "
            "trajectory without collisions (possibly because the cream "
            "cheese is a lower object); geometry.top_down_grasp_candidates "
            "generated a larger than expected bounding box."
        ),
        intent=(
            "cream_cheese grasped-and-lifted and finishes settled inside "
            "the basket, nothing else packed. Diagnostics (not gated): "
            "planner tool errors and geometry.* outputs vs the "
            "ground-truth extents. Three seeds: the low-object edge is "
            "the one to test across layouts."
        ),
        expected="measured GT AABB extents: 8.3 x 4.3 x 2.0 cm — it really is a low object",
        criterion={"checks": _pick_place_checks("cream_cheese")},
        seeds=(0, 1, 2),
        timeout_min=75,
        tags=("geometry_diag",),
    ),
    Case(
        id="move_basket_ranch",
        prompt=(
            "Move the basket to the opposite side, then pick up the "
            "ranch and place it in the basket"
        ),
        category="spatial",
        tier="T0",
        baseline="(no outcome recorded in the tester matrix)",
        intent=(
            "basket displaced >= 20 cm with a lateral side-flip (or >= "
            "30 cm across the table in x); the ranch (salad_dressing) "
            "grasped-and-lifted and settled inside the RELOCATED basket. "
            "How the basket is moved (carried or pushed) is not gated. "
            "Seed 1: the basket starts clearly on the -y side."
        ),
        criterion={"checks": [
            {"name": "basket_moved_to_opposite_side", "weight": 0.45,
             "expr": {"relocated": {"target": "basket"}}},
            {"name": "salad_dressing_grasped_and_lifted", "weight": 0.2,
             "expr": {"picked_up": {"target": "salad_dressing"}}},
            {"name": "salad_dressing_in_basket_final", "weight": 0.35,
             "expr": {"in_container": {"target": "salad_dressing",
                                       "container": "basket"}}},
        ]},
        seeds=(1,),
        timeout_min=60,
        tags=("synonym",),
    ),
    Case(
        id="stack_butter_on_cream_cheese",
        prompt="Stack the butter on top of the cream cheese",
        category="spatial",
        tier="T0",
        baseline="(new case — no tester baseline)",
        intent=(
            "butter grasped-and-lifted and finishes resting ON the cream "
            "cheese (is_on_strict-style window, xy centroid inside the "
            "base), both settled, cream cheese not dragged more than "
            "10 cm. Both are ~2 cm flat boxes — a precision place."
        ),
        criterion={"checks": [
            {"name": "butter_grasped_and_lifted", "weight": 0.3,
             "expr": {"picked_up": {"target": "butter"}}},
            {"name": "butter_on_cream_cheese", "weight": 0.55,
             "expr": {"resting_on": {"target": "butter",
                                     "support": "cream_cheese"}}},
            {"name": "cream_cheese_not_dragged", "weight": 0.15,
             "expr": {"displaced": {"target": "cream_cheese", "axis": "xy",
                                    "lo": 0.0, "hi": 0.10}}},
        ]},
    ),
    Case(
        id="all_except_milk",
        prompt="Put all the objects except the milk into the basket",
        category="grounding",
        tier="T0",
        baseline="(new case — no tester baseline)",
        intent=(
            "negation probe: the five non-milk objects finish settled in "
            "the basket AND the milk stays out. Score is 0.75*packed "
            "fraction + 0.25*milk-kept-out."
        ),
        criterion={"checks": [
            {"name": "listed_targets_in_container", "weight": 0.75,
             "expr": {"count_in": {
                 "container": "basket",
                 "targets": ["alphabet_soup", "butter", "cream_cheese",
                             "salad_dressing", "tomato_sauce"],
                 "count": "all"}}},
            {"name": "excluded_objects_stay_out", "weight": 0.25,
             "expr": {"kept_out": {"targets": ["milk"]}}},
            {"name": "targets_picked_up", "weight": 0.0, "gating": False,
             "expr": {"pack_report": {
                 "targets": ["alphabet_soup", "butter", "cream_cheese",
                             "salad_dressing", "tomato_sauce"],
                 "mode": "final"}}},
        ]},
        timeout_min=90,
        tags=("negation",),
    ),
    Case(
        id="soup_left_of_milk",
        prompt="Place the alphabet soup to the left of the milk",
        category="spatial",
        tier="T0",
        baseline="(new case — no tester baseline)",
        intent=(
            "spatial-relation probe: soup grasped-and-lifted, set down "
            "settled on the table 6-30 cm from the milk along y and "
            "within 15 cm in x, not in the basket. Either reading of "
            "'left' is accepted (viewer's left is -y in the agentview "
            "video, robot's left is +y)."
        ),
        criterion={"checks": [
            {"name": "alphabet_soup_grasped_and_lifted", "weight": 0.25,
             "expr": {"picked_up": {"target": "alphabet_soup"}}},
            {"name": "alphabet_soup_beside_milk", "weight": 0.4,
             "expr": {"offset_along": {"target": "alphabet_soup",
                                       "reference": "milk", "axis": "y",
                                       "lo": 0.06, "hi": 0.30,
                                       "perp_max": 0.15}}},
            {"name": "alphabet_soup_moved", "weight": 0.15,
             "expr": {"displaced": {"target": "alphabet_soup", "axis": "xy",
                                    "lo": 0.04}}},
            {"name": "released_on_table", "weight": 0.2,
             "expr": {"at_end": {"and": [
                 {"on_table": {"target": "alphabet_soup"}},
                 {"not": {"held": {"target": "alphabet_soup"}}},
                 {"settled": {"target": "alphabet_soup"}},
             ]}}},
        ]},
    ),
    Case(
        id="pudding_basket",
        prompt="Pick up the chocolate pudding and place it in the basket",
        category="grounding",
        tier="T0",
        baseline="(new case — no tester baseline)",
        intent=(
            "single pick on a DIFFERENT variance task's scene "
            "(libero_object_all_variance/3 — probed: alphabet_soup, "
            "bbq_sauce, chocolate_pudding, cream_cheese, ketchup, "
            "salad_dressing + basket): chocolate_pudding grasped-and-"
            "lifted, settled in the basket, nothing else packed. First "
            "coverage of the pudding box and of a scene whose distractor "
            "set differs from task 0's."
        ),
        criterion={"checks": _pick_place_checks("chocolate_pudding")},
        sim_task="libero_object_all_variance/3",
    ),
    Case(
        id="oj_swap_scene",
        prompt="Pick up the orange juice and place it in the basket",
        category="robustness",
        tier="T0",
        baseline="(new case — no tester baseline)",
        intent=(
            "single pick on the basket-SWAP variance suite "
            "(libero_object_target_basket_swap_variance/5 — probed scene: "
            "bbq_sauce, butter, chocolate_pudding, milk, orange_juice, "
            "tomato_sauce + basket, with the basket relocating across "
            "inits): orange_juice grasped-and-lifted, settled in the "
            "basket, nothing else packed. Two seeds = two basket "
            "placements."
        ),
        criterion={"checks": _pick_place_checks("orange_juice")},
        sim_task="libero_object_target_basket_swap_variance/5",
        seeds=(0, 1),
        timeout_min=75,
    ),
    Case(
        id="pack_scene_all",
        prompt="Pick all the objects and place them in the basket",
        category="coding",
        tier="T0",
        baseline="(new case — no tester baseline)",
        intent=(
            "the native VAB packing suite (libero_object_packing/0), "
            "prompt VERBATIM from its task language. MEASURED env "
            "mechanics: the suite's pack_all_into predicate TELEPORTS "
            "each contained object to (50.6, 50.0) as its packed-marker, "
            "so an object counts as packed when genuinely picked up AND "
            "(observed-in-basket OR env-consumed); the env's own reward "
            "is the external cross-check."
        ),
        criterion={"checks": [
            {"name": "all_objects_packed_or_consumed", "weight": 1.0,
             "expr": {"pack_report": {"targets": "movables",
                                      "mode": "grasp_and_seen"}}},
            {"name": "env_reward_cross_check", "weight": 0.0,
             "gating": False, "expr": {"env_success": {}}},
        ]},
        sim_task="libero_object_packing/0",
        env_consumes_packed=True,
        timeout_min=90,
    ),
    Case(
        id="both_boxes",
        prompt="Put both boxes in the basket",
        category="grounding",
        tier="T0",
        baseline="(new case — no tester baseline)",
        intent=(
            "category grounding within the probed task-0 scene: the only "
            "two box-shaped items — butter (7.8x4.0x1.8 cm) and "
            "cream_cheese (8.3x4.3x2.0 cm) — finish settled in the "
            "basket; the can/bottle/carton items all stay out."
        ),
        criterion={"checks": [
            {"name": "listed_targets_in_container", "weight": 0.75,
             "expr": {"count_in": {"container": "basket",
                                   "targets": ["butter", "cream_cheese"],
                                   "count": "all"}}},
            {"name": "excluded_objects_stay_out", "weight": 0.25,
             "expr": {"kept_out": {"targets": [
                 "alphabet_soup", "milk", "salad_dressing",
                 "tomato_sauce"]}}},
            {"name": "targets_picked_up", "weight": 0.0, "gating": False,
             "expr": {"pack_report": {"targets": ["butter", "cream_cheese"],
                                      "mode": "final"}}},
        ]},
        timeout_min=60,
        tags=("category",),
    ),
]
