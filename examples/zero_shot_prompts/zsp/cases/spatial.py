"""Movement & spatial: precision placement, relations, stacking,
pushing, reorientation, motion-only EE goals."""

from __future__ import annotations

from zsp.cases._shared import pick_checks, released_on_table
from zsp.schema import Case

CASES: list[Case] = [
    Case(
        id="offset_toward_robot",
        prompt=("Pick up the milk and set it down 10 centimeters closer "
                "to the robot."),
        category="spatial", tier="T1", tags=("metric",),
        intent=(
            "metric offset along the robot's depth axis (complement of "
            "two_inches_right: cm units, x axis, no left/right "
            "ambiguity): milk displaced -10 +/- 4 cm in x (toward the "
            "base), <= 5 cm drift in y, still upright, released on the "
            "table."
        ),
        criterion={"checks": [
            {"name": "milk_picked", "weight": 0.25,
             "expr": {"picked_up": {"target": "milk"}}},
            {"name": "moved_10cm_toward_robot", "weight": 0.35,
             "expr": {"and": [
                 {"displaced": {"target": "milk", "axis": "x",
                                "lo": -0.14, "hi": -0.06, "signed": True}},
                 {"displaced": {"target": "milk", "axis": "y",
                                "lo": 0.0, "hi": 0.05}},
             ]}},
            {"name": "still_upright", "weight": 0.25,
             "expr": {"tilt_from_initial": {"target": "milk",
                                            "max_deg": 30}}},
            {"name": "released_on_table", "weight": 0.15,
             "expr": released_on_table("milk")},
        ]},
        timeout_min=60,
    ),
    Case(
        id="place_midpoint",
        prompt=("Pick up the butter and place it exactly halfway between "
                "the milk and the tomato sauce."),
        category="spatial", tier="T1", tags=("computed",),
        intent=(
            "geometric computation -> precision place: butter within "
            "6 cm (xy) of the milk/tomato-sauce midpoint at t0, both "
            "reference objects unmoved."
        ),
        criterion={"checks": [
            {"name": "butter_picked", "weight": 0.25,
             "expr": {"picked_up": {"target": "butter"}}},
            {"name": "butter_at_midpoint", "weight": 0.55,
             "expr": {"near_point": {
                 "target": "butter",
                 "point": {"$t0": {"midpoint": {"a": "milk",
                                                "b": "tomato_sauce"}}},
                 "radius_m": 0.06}}},
            {"name": "references_unmoved", "weight": 0.2,
             "expr": {"and": [
                 {"untouched": {"target": "milk", "tol_m": 0.05}},
                 {"untouched": {"target": "tomato_sauce", "tol_m": 0.05}},
             ]}},
        ]},
        timeout_min=60,
    ),
    Case(
        id="ee_hover_no_grasp",
        prompt=("Move the gripper until it hovers about 15 centimeters "
                "directly above the milk carton, then stop. Do not grasp "
                "or move anything."),
        category="spatial", tier="T1", tags=("motion_only",),
        intent=(
            "motion-only EE goal — pure perceive->pose->move, no grasp/"
            "transport skill: final EE within 5 cm (xy) / 6 cm (z) of "
            "15 cm above the milk; every object untouched."
        ),
        criterion={
            "checks": [
                {"name": "ee_above_milk", "weight": 0.7,
                 "expr": {"ee_near": {"body": "milk",
                                      "offset": [0.0, 0.0, 0.15],
                                      "tol_xy_m": 0.05, "tol_z_m": 0.06}}},
                {"name": "nothing_touched", "weight": 0.3,
                 "expr": {"all_untouched": {"tol_m": 0.03}}},
            ],
            "options": {"require_picked_up": False},
        },
        timeout_min=30,
    ),
    Case(
        id="stack_reverse",
        prompt="Stack the cream cheese on top of the butter.",
        category="spatial", tier="T1", tags=("permutation",),
        template="stack_reverse",   # documented pair of the T0 stack case
        intent=(
            "role-permuted inverse of stack_butter_on_cream_cheese: the "
            "criterion must flip with the prompt (catches memorized "
            "roles and asymmetric-criterion bugs)."
        ),
        criterion={"checks": [
            {"name": "cream_cheese_picked", "weight": 0.3,
             "expr": {"picked_up": {"target": "cream_cheese"}}},
            {"name": "cream_cheese_on_butter", "weight": 0.55,
             "expr": {"resting_on": {"target": "cream_cheese",
                                     "support": "butter"}}},
            {"name": "butter_not_dragged", "weight": 0.15,
             "expr": {"displaced": {"target": "butter", "axis": "xy",
                                    "lo": 0.0, "hi": 0.10}}},
        ]},
    ),
    # ---- frontier: pushing ------------------------------------------------
    Case(
        id="push_slide_butter",
        prompt=("Slide the butter about 10 centimeters to the right, "
                "keeping it on the table the whole time. Do not lift it."),
        category="spatial", tier="T2", tags=("push",),
        expected=(
            "no pushing skill exists — tests cartesian-sweep composition; "
            "open-loop planar pushes sometimes work"
        ),
        intent=(
            "planar push: butter displaced 5-18 cm along y (either "
            "reading of 'right', logged), never grasped (a pick-and-"
            "place registers held_body and fails), finishes on the "
            "table."
        ),
        criterion={
            "checks": [
                {"name": "butter_slid_along_y", "weight": 0.5,
                 "expr": {"displaced": {"target": "butter", "axis": "y",
                                        "lo": 0.05, "hi": 0.18}}},
                {"name": "never_grasped", "weight": 0.3,
                 "expr": {"never_held": {"target": "butter"}}},
                {"name": "still_on_table", "weight": 0.2,
                 "expr": {"at_end": {"and": [
                     {"on_table": {"target": "butter"}},
                     {"settled": {"target": "butter"}},
                 ]}}},
            ],
            "options": {"push_ok": ["butter"],
                        "require_picked_up": False},
        },
        timeout_min=60,
    ),
    Case(
        id="push_to_basket",
        prompt=("Push the tomato sauce along the table until it rests "
                "against the side of the basket. Do not lift it."),
        category="spatial", tier="T2", tags=("push",),
        expected=(
            "closed-loop push toward a target — needs a re-perceive "
            "loop; expected to separate perceive-once graphs from "
            "iterating ones"
        ),
        intent=(
            "tomato sauce pushed to within 17 cm (xy, center-to-center "
            "~= touching the basket wall) of the basket, NOT inside it, "
            "never grasped; basket itself not shoved more than 10 cm."
        ),
        criterion={
            "checks": [
                {"name": "resting_against_basket", "weight": 0.5,
                 "expr": {"at_end": {"and": [
                     {"xy_within": {"target": "tomato_sauce",
                                    "reference": "basket",
                                    "radius_m": 0.17}},
                     {"not": {"in_container": {"target": "tomato_sauce",
                                               "container": "basket"}}},
                     {"settled": {"target": "tomato_sauce"}},
                 ]}}},
                {"name": "moved_at_all", "weight": 0.2,
                 "expr": {"displaced": {"target": "tomato_sauce",
                                        "axis": "xy", "lo": 0.05}}},
                {"name": "never_grasped", "weight": 0.2,
                 "expr": {"never_held": {"target": "tomato_sauce"}}},
                {"name": "basket_not_shoved", "weight": 0.1,
                 "expr": {"displaced": {"target": "basket", "axis": "xy",
                                        "lo": 0.0, "hi": 0.10}}},
            ],
            "options": {"push_ok": ["tomato_sauce"],
                        "require_picked_up": False},
        },
        timeout_min=75,
    ),
    # ---- frontier: reorientation -------------------------------------------
    Case(
        id="lay_milk_on_side",
        prompt="Lay the milk carton down on its side on the table.",
        category="spatial", tier="T2", tags=("reorientation",),
        expected=(
            "wrist-rotated place — top-down grasp + 90 deg roll "
            "approaches joint limits; release stability is the hard part"
        ),
        intent=(
            "controlled reorientation: milk picked up, ends settled on "
            "the table with its initially-up axis tilted 60-120 deg "
            "(on its side), within 30 cm of where it started."
        ),
        criterion={"checks": [
            {"name": "milk_picked", "weight": 0.3,
             "expr": {"picked_up": {"target": "milk"}}},
            {"name": "on_its_side", "weight": 0.5,
             "expr": {"tilt_from_initial": {"target": "milk",
                                            "min_deg": 60,
                                            "max_deg": 120}}},
            {"name": "settled_on_table_nearby", "weight": 0.2,
             "expr": {"and": [
                 released_on_table("milk"),
                 {"displaced": {"target": "milk", "axis": "xy",
                                "lo": 0.0, "hi": 0.30}},
             ]}},
        ]},
        timeout_min=60,
    ),
    Case(
        id="stand_up_tomato",
        prompt="The tomato sauce can is lying on its side. Stand it upright.",
        category="spatial", tier="T2", tags=("reorientation",),
        sim_task="libero_zsp_clutter/0",
        expected="side grasp (grasping-short-axis) + 90 deg reorient",
        intent=(
            "inverse reorientation (scene authored with the can on its "
            "side): at the end the initially-up axis is 60-120 deg from "
            "vertical — for a can the only stable pose in that band is "
            "STANDING — settled on the table near where it lay."
        ),
        criterion={"checks": [
            {"name": "tomato_picked", "weight": 0.3,
             "expr": {"picked_up": {"target": "tomato_sauce"}}},
            {"name": "stood_upright", "weight": 0.5,
             "expr": {"tilt_from_initial": {"target": "tomato_sauce",
                                            "min_deg": 60,
                                            "max_deg": 120}}},
            {"name": "settled_on_table_nearby", "weight": 0.2,
             "expr": {"and": [
                 released_on_table("tomato_sauce"),
                 {"displaced": {"target": "tomato_sauce", "axis": "xy",
                                "lo": 0.0, "hi": 0.25}},
             ]}},
        ]},
        timeout_min=60,
    ),
    Case(
        id="mug_flip",
        prompt="Place the red mug upside down on the plate.",
        category="spatial", tier="T2", tags=("reorientation",),
        sim_task="libero_zsp_kitchenware/0",
        expected=(
            "180 deg flip likely needs a regrasp no skill supports — "
            "the recognizable ceiling case of the reorientation family"
        ),
        intent=(
            "red_coffee_mug ends inverted (initially-up axis >= 150 deg "
            "from vertical) resting on the plate."
        ),
        criterion={"checks": [
            {"name": "mug_picked", "weight": 0.3,
             "expr": {"picked_up": {"target": "red_coffee_mug"}}},
            {"name": "upside_down", "weight": 0.45,
             "expr": {"tilt_from_initial": {"target": "red_coffee_mug",
                                            "min_deg": 150,
                                            "max_deg": 181}}},
            {"name": "on_the_plate", "weight": 0.25,
             "expr": {"resting_on": {"target": "red_coffee_mug",
                                     "support": "plate",
                                     "max_above": 0.06}}},
        ]},
        timeout_min=60,
    ),
    # ---- frontier: arrangement ---------------------------------------------
    Case(
        id="arrange_row",
        prompt=("Arrange the alphabet soup, the butter, and the salad "
                "dressing in a straight line, with at least 8 "
                "centimeters between neighbors."),
        category="spatial", tier="T2", tags=("arrangement",),
        expected="multi-object global-configuration goal — three placements",
        intent=(
            "the three named objects end colinear in xy (max residual "
            "<= 5 cm from the least-squares line) with every pairwise "
            "gap >= 8 cm, all settled on the table. Trio chosen because "
            "its spawn is clearly NOT colinear (probed residual 17 cm "
            "at seed 0 — the soup/tomato/milk trio spawns at 2 cm and "
            "would pass vacuously)."
        ),
        criterion={
            "checks": [
                {"name": "in_a_row", "weight": 0.7,
                 "expr": {"colinear": {"bodies": ["alphabet_soup",
                                                  "butter",
                                                  "salad_dressing"],
                                       "residual_max": 0.05,
                                       "min_gap": 0.08}}},
                {"name": "all_on_table", "weight": 0.3,
                 "expr": {"at_end": {"and": [
                     {"on_table": {"target": "alphabet_soup"}},
                     {"on_table": {"target": "butter"}},
                     {"on_table": {"target": "salad_dressing"}},
                 ]}}},
            ],
            # A pure-rearrangement task: pushing items into line is as
            # valid as carrying them.
            "options": {"require_picked_up": False},
        },
        timeout_min=90,
    ),
    # ---- new-scene stacking / kitchenware ----------------------------------
    Case(
        id="stack_on_plate",
        prompt=("Put the chocolate pudding on the plate, then stack the "
                "butter on top of the chocolate pudding."),
        category="spatial", tier="T1", tags=("stacking",),
        sim_task="libero_zsp_stack/0",
        intent=(
            "two-stage stack on an easy base (the T1 rung below "
            "stack_trio): pudding resting on the plate, butter resting "
            "on the pudding, placed in that order. NOTE: butter starts "
            "pre-stacked on the cream cheese in this scene."
        ),
        criterion={"checks": [
            {"name": "pudding_on_plate", "weight": 0.4,
             "expr": {"resting_on": {"target": "chocolate_pudding",
                                     "support": "plate",
                                     "max_above": 0.05}}},
            {"name": "butter_on_pudding", "weight": 0.4,
             "expr": {"resting_on": {"target": "butter",
                                     "support": "chocolate_pudding"}}},
            {"name": "pudding_placed_first", "weight": 0.2,
             "expr": {"before": {
                 "first": {"picked_up": {"target": "chocolate_pudding"}},
                 "then": {"picked_up": {"target": "butter"}}}}},
        ]},
        timeout_min=75,
    ),
    Case(
        id="unstack_top",
        prompt=("One item is stacked on top of the cream cheese. Move "
                "that item into the basket."),
        category="spatial", tier="T1", tags=("relational",),
        sim_task="libero_zsp_stack/0",
        intent=(
            "relational referent resolved from the perceived scene (the "
            "butter starts stacked on the cream cheese, probed): butter "
            "into the basket, cream cheese stays put."
        ),
        criterion={"checks": [
            *pick_checks("butter"),
            {"name": "cream_cheese_undisturbed", "weight": 0.0,
             "expr": {"displaced": {"target": "cream_cheese", "axis": "xy",
                                    "lo": 0.0, "hi": 0.05}}},
        ]},
    ),
    Case(
        id="stack_trio",
        prompt=("Build a stack: chocolate pudding on the cream cheese, "
                "then butter on top of the chocolate pudding."),
        category="spatial", tier="T2", tags=("stacking",),
        sim_task="libero_zsp_stack/0",
        expected="three-box tower — precision-place ceiling; partial credit "
                 "via the two resting checks",
        intent=(
            "3-tower: pudding resting on cream cheese AND butter resting "
            "on pudding at the end (butter must first come OFF the cream "
            "cheese), base not dragged more than 10 cm."
        ),
        criterion={"checks": [
            {"name": "pudding_on_cream_cheese", "weight": 0.4,
             "expr": {"resting_on": {"target": "chocolate_pudding",
                                     "support": "cream_cheese"}}},
            {"name": "butter_on_pudding", "weight": 0.4,
             "expr": {"resting_on": {"target": "butter",
                                     "support": "chocolate_pudding"}}},
            {"name": "base_not_dragged", "weight": 0.2,
             "expr": {"displaced": {"target": "cream_cheese", "axis": "xy",
                                    "lo": 0.0, "hi": 0.10}}},
        ]},
        timeout_min=90,
    ),
    Case(
        id="kitchen_red_mug_plate",
        prompt="Put the red mug on the plate.",
        category="spatial", tier="T1", tags=("vocabulary",),
        sim_task="libero_zsp_kitchenware/0",
        intent=(
            "non-grocery vocabulary + color pick between two mugs: "
            "red_coffee_mug resting on the plate, porcelain mug "
            "untouched."
        ),
        criterion={"checks": [
            {"name": "red_mug_picked", "weight": 0.3,
             "expr": {"picked_up": {"target": "red_coffee_mug"}}},
            {"name": "red_mug_on_plate", "weight": 0.55,
             "expr": {"resting_on": {"target": "red_coffee_mug",
                                     "support": "plate",
                                     "max_above": 0.05}}},
            {"name": "other_mug_untouched", "weight": 0.15,
             "expr": {"untouched": {"target": "porcelain_mug",
                                    "tol_m": 0.05}}},
        ]},
    ),
    Case(
        id="kitchen_bowl_frypan",
        prompt="Put the white bowl in the frying pan.",
        category="spatial", tier="T1", tags=("vocabulary",),
        sim_task="libero_zsp_kitchenware/0",
        intent=(
            "novel receptacle geometry (rimmed pan — containment scored "
            "center-inside-AABB like the tray): white_bowl inside the "
            "frypan's AABB, settled."
        ),
        criterion={"checks": [
            {"name": "bowl_picked", "weight": 0.35,
             "expr": {"picked_up": {"target": "white_bowl"}}},
            {"name": "bowl_in_frypan", "weight": 0.65,
             "expr": {"in_container": {"target": "white_bowl",
                                       "container": "chefmate_8_frypan"}}},
        ]},
    ),
]
