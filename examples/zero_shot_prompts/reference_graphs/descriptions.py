"""VLM-friendly object descriptions for perceiving-objects.

The perception tournament disambiguates DINO detections by a natural-
language description, so these are chosen to be distinctive (color/shape/
size cues where a scene has near-identical distractors). The criterion
still checks the ground-truth BODY name — perception only needs to find
the right thing to grasp. Classic ``_1``-suffixed bodies share the base
description.
"""

from __future__ import annotations

DESCRIPTIONS: dict[str, str] = {
    # HOPE grocery
    "alphabet_soup": "blue and yellow alphabet soup can",
    "cream_cheese": "small cream cheese box",
    "salad_dressing": "salad dressing bottle",
    "bbq_sauce": "bbq sauce bottle",
    "ketchup": "red ketchup bottle",
    "tomato_sauce": "tomato sauce can",
    "butter": "small butter box",
    "milk": "milk carton",
    "chocolate_pudding": "chocolate pudding box",
    "orange_juice": "orange juice carton",
    "popcorn": "popcorn box",
    "macaroni_and_cheese": "macaroni and cheese box",
    "cookies": "cookie box",
    "yellow_cookies": "yellow cookie box",
    # color / size variants
    "green_ketchup": "green ketchup bottle",
    "blue_ketchup": "blue ketchup bottle",
    "red_alphabet_soup": "red alphabet soup can",
    "bigger_alphabet_soup": "large alphabet soup can",
    "bigger_milk": "large milk carton",
    # containers / receptacles
    "basket": "wicker basket",
    "red_basket": "red basket",
    "wooden_tray": "wooden tray",
    "plate": "white plate",
    "chefmate_8_frypan": "frying pan",
    # kitchenware
    "red_coffee_mug": "red coffee mug",
    "porcelain_mug": "white porcelain mug",
    "white_bowl": "white bowl",
    # classic scenes
    "akita_black_bowl": "black bowl",
    "wooden_cabinet": "wooden cabinet",
    "flat_stove": "stove",
    "microwave": "microwave",
}


def describe(body: str) -> str:
    """Description for a body name, stripping a classic ``_N`` suffix."""
    if body in DESCRIPTIONS:
        return DESCRIPTIONS[body]
    base = body
    while base and base.rsplit("_", 1)[-1].isdigit():
        base = base.rsplit("_", 1)[0]
    return DESCRIPTIONS.get(base, base.replace("_", " "))
