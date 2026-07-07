"""The 7 example-local VAB scene definitions (libero_zsp_* suites).

Data only — ``author_inits.py`` turns these into settled task YAMLs
under ``examples/zero_shot_prompts/tasks/``. All coordinates are VAB
WORLD-frame (floor arena; usable envelope measured from the shipped
grocery YAMLs: x in [-0.21, 0.17], y in [-0.26, 0.26]; the robot base
sits at world (-0.6, 0, 0)).

Init-permutation rule (benchmark requirement): across a scene's inits,
object IDENTITIES permute over slots wherever physically sensible — a
graph that memorizes seed-0 placements must fail on other seeds.

``objects``: [(id, asset)]. ``inits``: list of {id: (x, y)} nominal
placements; z + resting quat come from measured rest poses (or a
drop-and-settle for assets without one). ``special`` per-init overrides:
``drop_z`` (spawn height for stacked/contained items), ``lying`` (start
the object on its side).
"""

from __future__ import annotations

from typing import Any

# Measured rest poses (z, quat xyzw) scraped from the shipped VAB
# grocery YAMLs; assets absent here are dropped from DROP_Z and settled.
REST: dict[str, tuple[float, list[float]]] = {
    "alphabet_soup": (0.038, [0.0023, 0.7077, 0.7065, -0.0023]),
    "basket": (0.035, [0.0, 0.0, 0.7071, 0.7071]),
    "bbq_sauce": (0.055, [0.0, 0.0, 0.7071, 0.7071]),
    "butter": (0.009, [0.0, 0.0, 1.0, 0.0]),
    "chocolate_pudding": (0.013, [0.0, 0.0, 1.0, 0.0]),
    "cream_cheese": (0.009, [0.0, 0.0, 1.0, 0.0]),
    "ketchup": (0.072, [0.5, 0.5, 0.5, 0.5]),
    "milk": (0.070, [0.5, 0.5, 0.5, 0.5]),
    "orange_juice": (0.070, [0.5, 0.5, 0.5, 0.5]),
    "salad_dressing": (0.073, [0.5, 0.5, 0.5, 0.5]),
    "tomato_sauce": (0.038, [0.0023, 0.7077, 0.7065, -0.0023]),
}
#: Same-shape color/size variants share the base asset's rest pose.
REST_ALIASES = {
    "green_ketchup": "ketchup", "blue_ketchup": "ketchup",
    "red_alphabet_soup": "alphabet_soup",
    "bigger_alphabet_soup": "alphabet_soup",
    "bigger_milk": "milk",
    "red_basket": "basket",
    "yellow_cookies": "cookies",
}
DROP_Z = 0.06          # spawn height for assets without a measured rest
FLAT_DROP_Z = 0.04     # plates / trays / pans

FLAT_ASSETS = {"plate", "wooden_tray", "chefmate_8_frypan"}


def rest_pose(asset: str) -> tuple[float, list[float] | None]:
    asset = REST_ALIASES.get(asset, asset)
    if asset in REST:
        z, q = REST[asset]
        return z + 0.01, list(q)   # small drop margin, settled out
    if asset in FLAT_ASSETS:
        return FLAT_DROP_Z, None   # None -> identity, settle finds rest
    return DROP_Z, None


Scene = dict[str, Any]

SCENES: dict[str, Scene] = {
    # -- color discrimination: 3 identical-shape ketchups, 2 baskets ------
    "libero_zsp_colors": {
        "file": "color_row",
        "language": "Pick up the green ketchup bottle and place it in the basket",
        "objects": [
            ("ketchup", "ketchup"),
            ("green_ketchup", "green_ketchup"),
            ("blue_ketchup", "blue_ketchup"),
            ("alphabet_soup", "alphabet_soup"),
            ("red_alphabet_soup", "red_alphabet_soup"),
            ("basket", "basket"),
            ("red_basket", "red_basket"),
        ],
        # ketchup row slots at x=-0.06, y in {-0.12, 0, +0.12}; row order
        # permutes so the "middle one" identity differs per init.
        "inits": [
            {"ketchup": (-0.06, -0.12), "green_ketchup": (-0.06, 0.0),
             "blue_ketchup": (-0.06, 0.12),
             "alphabet_soup": (0.08, 0.20), "red_alphabet_soup": (0.08, 0.26),
             "basket": (0.14, -0.20), "red_basket": (-0.19, -0.22)},
            {"green_ketchup": (-0.06, -0.12), "blue_ketchup": (-0.06, 0.0),
             "ketchup": (-0.06, 0.12),
             "red_alphabet_soup": (0.08, 0.20), "alphabet_soup": (0.08, 0.26),
             "red_basket": (0.14, -0.20), "basket": (-0.19, -0.22)},
            {"blue_ketchup": (-0.06, -0.12), "ketchup": (-0.06, 0.0),
             "green_ketchup": (-0.06, 0.12),
             "alphabet_soup": (0.08, 0.26), "red_alphabet_soup": (0.08, 0.20),
             "basket": (0.14, -0.20), "red_basket": (-0.19, -0.22)},
        ],
        "success": {"predicate": "contained_in",
                    "args": {"obj": "green_ketchup", "container": "basket",
                             "xy_tol": 0.1, "z_low": -0.05, "z_high": 0.25}},
    },
    # -- size discrimination: two size pairs -------------------------------
    "libero_zsp_sizes": {
        "file": "size_pairs",
        "language": "Pick up the larger of the two soup cans and place it in the basket",
        "objects": [
            ("alphabet_soup", "alphabet_soup"),
            ("bigger_alphabet_soup", "bigger_alphabet_soup"),
            ("milk", "milk"),
            ("bigger_milk", "bigger_milk"),
            ("basket", "basket"),
        ],
        "inits": [
            {"alphabet_soup": (-0.10, -0.15), "bigger_alphabet_soup": (-0.01, -0.15),
             "milk": (-0.10, 0.15), "bigger_milk": (-0.01, 0.15),
             "basket": (0.14, 0.0)},
            {"bigger_alphabet_soup": (-0.10, 0.15), "alphabet_soup": (-0.01, 0.15),
             "bigger_milk": (-0.10, -0.15), "milk": (-0.01, -0.15),
             "basket": (0.14, 0.0)},
            {"alphabet_soup": (-0.10, 0.15), "bigger_alphabet_soup": (-0.01, 0.15),
             "milk": (-0.01, -0.15), "bigger_milk": (-0.10, -0.15),
             "basket": (0.10, -0.22)},
        ],
        "success": {"predicate": "contained_in",
                    "args": {"obj": "bigger_alphabet_soup",
                             "container": "basket",
                             "xy_tol": 0.1, "z_low": -0.05, "z_high": 0.25}},
    },
    # -- two-container sorting ---------------------------------------------
    "libero_zsp_sorting": {
        "file": "two_containers",
        "language": "Sort the groceries: put the two cans in the basket and the two boxes on the wooden tray",
        "objects": [
            ("basket", "basket"),
            ("wooden_tray", "wooden_tray"),
            ("alphabet_soup", "alphabet_soup"),
            ("tomato_sauce", "tomato_sauce"),
            ("butter", "butter"),
            ("cream_cheese", "cream_cheese"),
        ],
        # containers ~40 cm apart in y; item slots shuffle so per-item
        # nearest-container assignment differs per init; init 1 swaps
        # the container sides.
        "inits": [
            {"basket": (0.12, -0.20), "wooden_tray": (0.12, 0.20),
             "alphabet_soup": (-0.16, -0.10), "tomato_sauce": (-0.06, 0.10),
             "butter": (-0.16, 0.10), "cream_cheese": (-0.06, -0.10)},
            {"wooden_tray": (0.12, -0.20), "basket": (0.12, 0.20),
             "tomato_sauce": (-0.16, -0.10), "alphabet_soup": (-0.06, 0.10),
             "cream_cheese": (-0.16, 0.10), "butter": (-0.06, -0.10)},
            {"basket": (0.12, -0.20), "wooden_tray": (0.12, 0.20),
             "butter": (-0.16, -0.10), "cream_cheese": (-0.06, 0.12),
             "alphabet_soup": (-0.16, 0.12), "tomato_sauce": (-0.06, -0.10)},
        ],
        "success": {"predicate": "contained_in",
                    "args": {"obj": "alphabet_soup", "container": "basket",
                             "xy_tol": 0.1, "z_low": -0.05, "z_high": 0.25}},
    },
    # -- pre-packed basket (retrieval / look-inside) ------------------------
    "libero_zsp_prepacked": {
        "file": "prepacked_basket",
        "language": "Take everything out of the basket and place it on the table",
        "objects": [
            ("basket", "basket"),
            ("alphabet_soup", "alphabet_soup"),
            ("butter", "butter"),
            ("milk", "milk"),
            ("cream_cheese", "cream_cheese"),
        ],
        # soup + butter start INSIDE the basket (dropped from above its
        # rim and settled); milk + cream_cheese on the table. Basket
        # position and the two table items permute across inits.
        "inits": [
            {"basket": (0.10, 0.0), "alphabet_soup": (0.10, -0.03),
             "butter": (0.10, 0.04),
             "milk": (-0.16, -0.15), "cream_cheese": (-0.16, 0.15)},
            {"basket": (0.06, -0.18), "alphabet_soup": (0.06, -0.21),
             "butter": (0.06, -0.14),
             "cream_cheese": (-0.16, -0.10), "milk": (-0.14, 0.16)},
            {"basket": (0.06, 0.18), "alphabet_soup": (0.06, 0.15),
             "butter": (0.06, 0.22),
             "milk": (-0.18, 0.02), "cream_cheese": (-0.08, -0.18)},
        ],
        "special": {
            "alphabet_soup": {"drop_z": 0.12},
            "butter": {"drop_z": 0.12},
        },
        "success": {"predicate": "contained_in",
                    "args": {"obj": "alphabet_soup", "container": "basket",
                             "xy_tol": 0.1, "z_low": -0.05, "z_high": 0.25}},
    },
    # -- kitchenware (non-grocery vocabulary) --------------------------------
    "libero_zsp_kitchenware": {
        "file": "mug_plate",
        "language": "Put the red mug on the plate",
        "objects": [
            ("plate", "plate"),
            ("white_bowl", "white_bowl"),
            ("porcelain_mug", "porcelain_mug"),
            ("red_coffee_mug", "red_coffee_mug"),
            ("chefmate_8_frypan", "chefmate_8_frypan"),
        ],
        "inits": [
            {"plate": (0.10, 0.0), "chefmate_8_frypan": (0.10, -0.22),
             "porcelain_mug": (-0.15, -0.15), "white_bowl": (-0.18, 0.0),
             "red_coffee_mug": (-0.15, 0.15)},
            {"plate": (0.10, 0.0), "chefmate_8_frypan": (0.10, 0.22),
             "red_coffee_mug": (-0.15, -0.15), "porcelain_mug": (-0.18, 0.0),
             "white_bowl": (-0.15, 0.15)},
            {"plate": (0.08, -0.14), "chefmate_8_frypan": (0.10, 0.20),
             "white_bowl": (-0.15, -0.15), "red_coffee_mug": (-0.18, 0.02),
             "porcelain_mug": (-0.15, 0.17)},
        ],
        "success": {"predicate": "on_top_of",
                    "args": {"obj": "red_coffee_mug", "surface": "plate",
                             "xy_tol": 0.15, "z_min": 0.005, "z_max": 0.20}},
    },
    # -- dense clutter + novel vocabulary + a can lying on its side ---------
    "libero_zsp_clutter": {
        "file": "dense_grocery",
        "language": "Put the popcorn in the basket",
        "objects": [
            ("popcorn", "popcorn"),
            ("macaroni_and_cheese", "macaroni_and_cheese"),
            ("cookies", "cookies"),
            ("yellow_cookies", "yellow_cookies"),
            ("orange_juice", "orange_juice"),
            ("tomato_sauce", "tomato_sauce"),
            ("basket", "basket"),
        ],
        # 2x3 grid, ~9 cm gaps; the grid slots permute; tomato_sauce
        # always starts LYING ON ITS SIDE (stand_up_tomato case).
        "inits": [
            {"popcorn": (-0.11, -0.09), "macaroni_and_cheese": (-0.11, 0.0),
             "cookies": (-0.11, 0.09), "yellow_cookies": (-0.01, -0.09),
             "orange_juice": (-0.01, 0.0), "tomato_sauce": (-0.01, 0.10),
             "basket": (0.14, 0.21)},
            {"yellow_cookies": (-0.11, -0.09), "orange_juice": (-0.11, 0.0),
             "popcorn": (-0.11, 0.09), "tomato_sauce": (-0.01, -0.10),
             "macaroni_and_cheese": (-0.01, 0.0), "cookies": (-0.01, 0.09),
             "basket": (0.14, 0.21)},
            {"cookies": (-0.11, -0.09), "popcorn": (-0.11, 0.0),
             "macaroni_and_cheese": (-0.11, 0.09),
             "orange_juice": (-0.01, -0.09), "yellow_cookies": (-0.01, 0.0),
             "tomato_sauce": (-0.01, 0.10), "basket": (0.14, -0.21)},
        ],
        "special": {"tomato_sauce": {"lying": True}},
        "success": {"predicate": "contained_in",
                    "args": {"obj": "popcorn", "container": "basket",
                             "xy_tol": 0.1, "z_low": -0.05, "z_high": 0.25}},
    },
    # -- flat-box stacking (butter pre-stacked on cream cheese) -------------
    "libero_zsp_stack": {
        "file": "flat_boxes",
        "language": "Stack the chocolate pudding on the cream cheese",
        "objects": [
            ("butter", "butter"),
            ("cream_cheese", "cream_cheese"),
            ("chocolate_pudding", "chocolate_pudding"),
            ("plate", "plate"),
            ("basket", "basket"),
        ],
        "inits": [
            {"cream_cheese": (-0.05, 0.0), "butter": (-0.05, 0.0),
             "chocolate_pudding": (-0.16, -0.15), "plate": (0.10, 0.15),
             "basket": (0.13, -0.18)},
            {"cream_cheese": (-0.08, -0.10), "butter": (-0.08, -0.10),
             "chocolate_pudding": (-0.16, 0.12), "plate": (0.10, 0.18),
             "basket": (0.13, -0.20)},
            {"cream_cheese": (-0.02, 0.12), "butter": (-0.02, 0.12),
             "chocolate_pudding": (-0.18, -0.05), "plate": (0.10, -0.16),
             "basket": (0.13, 0.20)},
        ],
        "special": {"butter": {"drop_z": 0.05}},   # lands on cream_cheese
        "success": {"predicate": "on_top_of",
                    "args": {"obj": "butter", "surface": "cream_cheese",
                             "xy_tol": 0.15, "z_min": 0.005, "z_max": 0.20}},
    },
}
