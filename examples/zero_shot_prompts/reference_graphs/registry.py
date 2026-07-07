"""Reference-graph registry: case_id -> the oracle graph that solves it.

Two kinds:
- ``pick`` cases are STAMPED from the libero_quickstart pick-place template
  (perceive target -> OBB -> grasp -> transport-into-container), swapping the
  target/container descriptions. ``build.py`` materializes them under
  ``_generated/``.
- ``hand`` cases point at a hand-authored graph dir (multi-step, spatial,
  sorting, computed-referent, report, refusal, frontier).

Each entry records the target/container BODY names (for stamping + docs) and
the family, so ``run_refs.py`` can drive them and the feasibility report can
group by family. Not every one of the 97 cases has a reference — the goal is
to prove each TEMPLATE FAMILY feasible and characterize the frontier, plus
broad pick-single coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ref:
    case: str
    family: str
    kind: str                      # "pick" | "hand"
    target: str = ""               # grasp-target body (pick)
    container: str = "basket"      # place-container body (pick)
    graph: str = ""                # hand-authored graph dir (relative to reference_graphs/)
    note: str = ""                 # feasibility expectation
    tags: tuple[str, ...] = field(default_factory=tuple)


# -- pick-single family: stamped from the quickstart template ---------------
# (case, target-body, container-body). Perception uses descriptions.describe().
_PICK: list[tuple[str, str, str]] = [
    ("soup_basket", "alphabet_soup", "basket"),
    ("pudding_basket", "chocolate_pudding", "basket"),
    ("oj_swap_scene", "orange_juice", "basket"),
    ("pick_native__cream_cheese", "cream_cheese", "basket"),
    ("pick_native__salad_dressing", "salad_dressing", "basket"),
    ("pick_native__ketchup", "ketchup", "basket"),
    ("pick_native__tomato_sauce", "tomato_sauce", "basket"),
    ("pick_native__butter", "butter", "basket"),
    ("pick_native__milk", "milk", "basket"),
    ("pick_native__chocolate_pudding", "chocolate_pudding", "basket"),
    ("pick_native__orange_juice", "orange_juice", "basket"),
    ("wk_salad_dressing", "salad_dressing", "basket"),
    ("noise_polite", "alphabet_soup", "basket"),
    ("noise_typos", "tomato_sauce", "basket"),
    ("noise_multilingual_zh", "butter", "basket"),
    ("object_native_classic", "tomato_sauce_1", "basket_1"),
    ("color_green_ketchup", "green_ketchup", "basket"),
    ("color_blue_bottle", "blue_ketchup", "basket"),
    ("size_bigger_soup", "bigger_alphabet_soup", "basket"),
    ("size_smaller_milk", "milk", "basket"),
    ("vocab_mac_cheese", "macaroni_and_cheese", "basket"),
    ("vocab_yellow_cookies", "yellow_cookies", "basket"),
    ("wk_popcorn_microwave", "popcorn", "basket"),
    ("kitchen_red_mug_plate", "red_coffee_mug", "plate"),
    ("kitchen_bowl_frypan", "white_bowl", "chefmate_8_frypan"),
]

_PICK_NOTES = {
    "size_bigger_soup": "perception may not separate size variants (P6) — expected-hard",
    "size_smaller_milk": "perception may not separate size variants (P6) — expected-hard",
    "kitchen_red_mug_plate": "place-onto-plate via the descend-into transport — flat receptacle",
    "kitchen_bowl_frypan": "shallow-cavity receptacle",
}

REFS: dict[str, Ref] = {}
for _case, _tgt, _cont in _PICK:
    REFS[_case] = Ref(
        case=_case, family="pick", kind="pick", target=_tgt, container=_cont,
        note=_PICK_NOTES.get(_case, "single pick-and-place"),
    )


def _hand(case: str, family: str, graph: str, note: str = "",
          tags: tuple[str, ...] = ()) -> None:
    REFS[case] = Ref(case=case, family=family, kind="hand", graph=graph,
                     note=note, tags=tags)


# -- hand-authored exemplars, one per remaining family ----------------------
_hand("all_objects", "multi_pack", "graphs/pack_all",
      "grocery_packing loop until only the basket remains")
_hand("both_boxes", "multi_pack", "graphs/both_boxes",
      "two pick-place blocks: butter + cream cheese into the basket")
_hand("soup_then_milk", "ordered", "graphs/soup_then_milk",
      "two pick-place blocks chained in order")
_hand("then_reverse", "ordered", "graphs/then_reverse",
      "role-permuted: milk then soup")
_hand("order_three_strict", "ordered", "graphs/order_three_strict",
      "three pick-place blocks: butter -> milk -> tomato sauce")
_hand("place_midpoint", "spatial", "graphs/place_midpoint",
      "pick + place at the computed midpoint of two reference objects")
_hand("two_inches_right", "spatial", "graphs/two_inches_right",
      "pick + place offset 2 inches along +y from the grasp point")
_hand("sort_cans_boxes", "sorting", "graphs/sort_cans_boxes",
      "four pick-place blocks routing cans->basket, boxes->tray")
_hand("closest_to_basket", "computed_ref", "graphs/closest_to_basket",
      "compute the closest object to the basket, then pick-place it")
_hand("vqa_count_objects", "report", "graphs/vqa_count",
      "perceive the scene, count objects, report the number (no manipulation)")
_hand("do_nothing", "refusal", "graphs/do_nothing",
      "a graph that leaves the table untouched")
_hand("absent_banana", "refusal", "graphs/absent_object",
      "perceive the named object, find none, exit gracefully without moving anything")
_hand("drawer_open_middle", "frontier_articulated", "graphs/drawer_open",
      "FRONTIER: perceive the drawer handle, approach, grip, pull along the drawer axis")
_hand("push_slide_butter", "frontier_push", "graphs/push_slide",
      "FRONTIER: closed-gripper cartesian sweep to slide the butter without lifting")


def pick_refs() -> list[Ref]:
    return [r for r in REFS.values() if r.kind == "pick"]


def hand_refs() -> list[Ref]:
    return [r for r in REFS.values() if r.kind == "hand"]
