"""Temporal: ordering, sequencing, memory of earlier state."""

from __future__ import annotations

from zsp.schema import Case

CASES: list[Case] = [
    Case(
        id="order_three_strict",
        prompt=("Put the butter, then the milk, then the tomato sauce "
                "into the basket — in exactly that order."),
        category="temporal", tier="T1", tags=("ordering",),
        intent=(
            "3-step strict sequencing (extends soup_then_milk to three "
            "objects with explicit order language): all three packed, "
            "butter entering the basket before milk before tomato sauce."
        ),
        criterion={"checks": [
            {"name": "butter_in", "weight": 0.2,
             "expr": {"in_container": {"target": "butter",
                                       "container": "basket"}}},
            {"name": "milk_in", "weight": 0.2,
             "expr": {"in_container": {"target": "milk",
                                       "container": "basket"}}},
            {"name": "tomato_sauce_in", "weight": 0.2,
             "expr": {"in_container": {"target": "tomato_sauce",
                                       "container": "basket"}}},
            {"name": "butter_before_milk", "weight": 0.2,
             "expr": {"before": {
                 "first": {"enters_container": {"target": "butter"}},
                 "then": {"enters_container": {"target": "milk"}}}}},
            {"name": "milk_before_tomato_sauce", "weight": 0.2,
             "expr": {"before": {
                 "first": {"enters_container": {"target": "milk"}},
                 "then": {"enters_container": {"target": "tomato_sauce"}}}}},
        ]},
        timeout_min=90,
    ),
    Case(
        id="then_reverse",
        prompt=("Pick up the milk and place it in the basket, then pick "
                "up the alphabet soup and place it in the basket"),
        category="temporal", tier="T1", tags=("ordering", "permutation"),
        intent=(
            "soup_then_milk with the roles permuted — the ordering "
            "criterion must flip with the prompt (memorized-order guard). "
            "Same structure and weights as the T0 case."
        ),
        criterion={"checks": [
            {"name": "milk_grasped_and_lifted", "weight": 0.15,
             "expr": {"picked_up": {"target": "milk"}}},
            {"name": "milk_in_basket_final", "weight": 0.25,
             "expr": {"in_container": {"target": "milk",
                                       "container": "basket"}}},
            {"name": "alphabet_soup_grasped_and_lifted", "weight": 0.15,
             "expr": {"picked_up": {"target": "alphabet_soup"}}},
            {"name": "alphabet_soup_in_basket_final", "weight": 0.25,
             "expr": {"in_container": {"target": "alphabet_soup",
                                       "container": "basket"}}},
            {"name": "milk_enters_basket_before_soup", "weight": 0.2,
             "expr": {"before": {
                 "first": {"enters_container": {"target": "milk"}},
                 "then": {"enters_container": {"target": "alphabet_soup"}}}}},
        ]},
    ),
    Case(
        id="put_back_exact",
        prompt=("Pick up the alphabet soup, hold it up for a moment, "
                "then put it back exactly where it was."),
        category="temporal", tier="T1", tags=("memory",),
        intent=(
            "memory probe: the graph must persist the pre-grasp pose "
            "across nodes — soup genuinely picked up, then back within "
            "5 cm (xy) of its t0 position, upright, settled on the "
            "table."
        ),
        criterion={"checks": [
            {"name": "soup_picked", "weight": 0.3,
             "expr": {"picked_up": {"target": "alphabet_soup"}}},
            {"name": "returned_to_start", "weight": 0.5,
             "expr": {"near_point": {
                 "target": "alphabet_soup",
                 "point": {"$t0": {"pos_of": {"body": "alphabet_soup"}}},
                 "radius_m": 0.05}}},
            {"name": "still_upright", "weight": 0.2,
             "expr": {"tilt_from_initial": {"target": "alphabet_soup",
                                            "max_deg": 30}}},
            {"name": "released_and_settled", "weight": 0.0,
             "expr": {"at_end": {"and": [
                 {"not": {"held": {"target": "alphabet_soup"}}},
                 {"settled": {"target": "alphabet_soup"}},
             ]}}},
        ]},
    ),
    Case(
        id="order_alphabetical_all",
        prompt=("Put every object into the basket, one at a time, in "
                "alphabetical order of their names."),
        category="temporal", tier="T2", tags=("ordering", "computed"),
        expected=(
            "full iteration + a derived schedule — combines the "
            "all_objects loop with computed ordering; prefix-correct "
            "partial credit via the chain of before checks"
        ),
        intent=(
            "all six task-0 objects packed (alphabet_soup, butter, "
            "cream_cheese, milk, salad_dressing, tomato_sauce) with "
            "first-entered-basket order matching their alphabetical "
            "order — each adjacent-pair ordering is a weighted check, "
            "so a correct prefix earns proportional score."
        ),
        criterion={"checks": [
            {"name": "all_packed", "weight": 0.5,
             "expr": {"pack_report": {"targets": "movables",
                                      "mode": "final"}}},
            *[
                {"name": f"{a}_before_{b}", "weight": 0.1,
                 "expr": {"before": {
                     "first": {"enters_container": {"target": a}},
                     "then": {"enters_container": {"target": b}}}}}
                for a, b in zip(
                    ("alphabet_soup", "butter", "cream_cheese", "milk",
                     "salad_dressing"),
                    ("butter", "cream_cheese", "milk", "salad_dressing",
                     "tomato_sauce"), strict=True)
            ],
        ]},
        timeout_min=120,
    ),
    Case(
        id="swap_two",
        prompt="Swap the positions of the butter and the milk.",
        category="temporal", tier="T2", tags=("memory",),
        expected=(
            "the classic three-move puzzle (needs a temporary spot) — "
            "pure planning/state-bookkeeping quality"
        ),
        intent=(
            "at the end the butter sits within 8 cm (xy) of where the "
            "milk started and vice versa, both genuinely picked up and "
            "settled."
        ),
        criterion={"checks": [
            {"name": "butter_at_milks_start", "weight": 0.4,
             "expr": {"near_point": {
                 "target": "butter",
                 "point": {"$t0": {"pos_of": {"body": "milk"}}},
                 "radius_m": 0.08}}},
            {"name": "milk_at_butters_start", "weight": 0.4,
             "expr": {"near_point": {
                 "target": "milk",
                 "point": {"$t0": {"pos_of": {"body": "butter"}}},
                 "radius_m": 0.08}}},
            {"name": "butter_picked", "weight": 0.1,
             "expr": {"picked_up": {"target": "butter"}}},
            {"name": "milk_picked", "weight": 0.1,
             "expr": {"picked_up": {"target": "milk"}}},
            {"name": "both_settled", "weight": 0.0,
             "expr": {"at_end": {"and": [
                 {"settled": {"target": "butter"}},
                 {"settled": {"target": "milk"}},
             ]}}},
        ]},
        timeout_min=90,
    ),
]
