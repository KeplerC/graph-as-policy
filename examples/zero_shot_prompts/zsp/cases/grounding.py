"""Grounding & instruction-following: referents, attributes, categories,
constraints, prompt-noise robustness, world knowledge, ambiguity."""

from __future__ import annotations

from zsp.cases._shared import CLASSIC_OPTIONS, pick_checks
from zsp.schema import Case

CASES: list[Case] = [
    Case(
        id="category_cans",
        prompt="Put all the cans in the basket.",
        category="grounding", tier="T1", tags=("category",),
        intent=(
            "category-by-shape over the probed task-0 scene: the two can "
            "items (alphabet_soup, tomato_sauce) finish settled in the "
            "basket; the boxes, carton, and bottle all stay out."
        ),
        criterion={"checks": [
            {"name": "cans_in_basket", "weight": 0.75,
             "expr": {"count_in": {"container": "basket",
                                   "targets": ["alphabet_soup",
                                               "tomato_sauce"],
                                   "count": "all"}}},
            {"name": "non_cans_stay_out", "weight": 0.25,
             "expr": {"kept_out": {"targets": [
                 "butter", "cream_cheese", "milk", "salad_dressing"]}}},
            {"name": "cans_picked_up", "weight": 0.0, "gating": False,
             "expr": {"pack_report": {"targets": ["alphabet_soup",
                                                  "tomato_sauce"],
                                      "mode": "final"}}},
        ]},
        timeout_min=75,
    ),
    Case(
        id="constraint_no_touch",
        prompt=("Put the butter in the basket. Do not touch or bump the "
                "milk carton at any point."),
        category="grounding", tier="T1", tags=("constraint",),
        intent=(
            "negative constraint honored DURING motion: butter packed "
            "while the milk never strays more than 3 cm from its spawn "
            "and is never held — measures unsafe flailing, not just the "
            "end state."
        ),
        criterion={"checks": [
            {"name": "butter_in_basket", "weight": 0.6,
             "expr": {"in_container": {"target": "butter",
                                       "container": "basket"}}},
            {"name": "milk_never_touched", "weight": 0.4,
             "expr": {"untouched": {"target": "milk", "tol_m": 0.03}}},
        ]},
    ),
    Case(
        id="wk_salad_dressing",
        prompt="Put the item you would pour over a salad into the basket.",
        category="grounding", tier="T1", tags=("world_knowledge",),
        intent=(
            "world-knowledge referent with no color/shape hint: only the "
            "salad_dressing bottle satisfies the description."
        ),
        criterion={"checks": pick_checks("salad_dressing")},
    ),
    Case(
        id="noise_polite",
        prompt=("Hey there! Hope you're doing great. Would you be an "
                "absolute star and pop that alphabet soup can into the "
                "basket for me? No rush at all. Thanks a million!!"),
        category="grounding", tier="T1", tags=("noise",),
        intent=(
            "politeness/filler robustness: identical task to soup_basket "
            "(seed 0) under heavy conversational decoration — diff the "
            "verdict directly against soup_basket."
        ),
        criterion={"checks": pick_checks("alphabet_soup")},
    ),
    Case(
        id="noise_typos",
        prompt="pikc up teh tomatoe sause adn plase it in teh baskit",
        category="grounding", tier="T1", tags=("noise",),
        intent="typo robustness: misspelled verb, object, and container.",
        criterion={"checks": pick_checks("tomato_sauce")},
    ),
    Case(
        id="noise_multilingual_zh",
        prompt="把黄油拿起来，放进篮子里。",
        category="grounding", tier="T1", tags=("noise", "multilingual"),
        intent=(
            "multilingual grounding (Chinese: 'pick up the butter and put "
            "it in the basket') — a home-field contrast for qwen/deepseek "
            "vs gemini."
        ),
        criterion={"checks": pick_checks("butter")},
    ),
    Case(
        id="ambig_two_bowls",
        prompt="Pick up a black bowl and place it on the plate.",
        category="grounding", tier="T1", tags=("ambiguity",),
        sim_task="libero_spatial/2",
        intent=(
            "benign ambiguity on the classic spatial scene (TWO identical "
            "akita_black_bowls, probed): EITHER bowl settled on the plate "
            "passes; freezing or refusing fails. The env's own reward "
            "wants one specific bowl and stays advisory."
        ),
        criterion={
            "checks": [
                {"name": "either_bowl_on_plate", "weight": 1.0,
                 "expr": {"or": [
                     {"and": [
                         {"picked_up": {"target": "akita_black_bowl_1"}},
                         {"resting_on": {"target": "akita_black_bowl_1",
                                         "support": "plate_1",
                                         "max_above": 0.06}}]},
                     {"and": [
                         {"picked_up": {"target": "akita_black_bowl_2"}},
                         {"resting_on": {"target": "akita_black_bowl_2",
                                         "support": "plate_1",
                                         "max_above": 0.06}}]},
                 ]}},
                {"name": "env_reward_advisory", "weight": 0.0,
                 "gating": False, "expr": {"env_success": {}}},
            ],
            "options": {**CLASSIC_OPTIONS, "require_picked_up": False},
        },
        timeout_min=60,
    ),
    # ---- new-scene attribute grounding ----------------------------------
    Case(
        id="color_green_ketchup",
        prompt="Pick up the green ketchup bottle and place it in the basket.",
        category="grounding", tier="T1", tags=("color",),
        sim_task="libero_zsp_colors/0",
        intent=(
            "color-variant discrimination against two identical-shape "
            "distractor ketchups (regular red, blue): only the green one "
            "may move, into the wicker basket (a red basket distractor "
            "is also present)."
        ),
        criterion={"checks": [
            {"name": "green_ketchup_picked", "weight": 0.3,
             "expr": {"picked_up": {"target": "green_ketchup"}}},
            {"name": "green_ketchup_in_basket", "weight": 0.5,
             "expr": {"in_container": {"target": "green_ketchup",
                                       "container": "basket"}}},
            {"name": "no_unrequested_packing", "weight": 0.2,
             "expr": {"no_new_packing": {"allowed": ["green_ketchup"]}}},
            {"name": "distractor_ketchups_untouched", "weight": 0.0,
             "expr": {"and": [
                 {"untouched": {"target": "ketchup", "tol_m": 0.05}},
                 {"untouched": {"target": "blue_ketchup", "tol_m": 0.05}},
             ]}},
        ]},
        seeds=(0, 1),
    ),
    Case(
        id="color_blue_bottle",
        prompt="Put the blue bottle in the basket.",
        category="grounding", tier="T1", tags=("color",),
        sim_task="libero_zsp_colors/0",
        intent=(
            "color-only reference — the category word 'ketchup' is never "
            "said; only the blue ketchup bottle matches."
        ),
        criterion={"checks": pick_checks("blue_ketchup")},
    ),
    Case(
        id="color_cross_binding",
        prompt=("Put the red alphabet soup in the wicker basket, and the "
                "plain alphabet soup in the red basket."),
        category="grounding", tier="T1", tags=("color", "binding"),
        sim_task="libero_zsp_colors/0",
        intent=(
            "attribute binding (object color x container color): the RED "
            "soup goes to the plain wicker basket and the PLAIN soup to "
            "the RED basket — swapped placements score zero on both."
        ),
        criterion={"checks": [
            {"name": "red_soup_in_wicker_basket", "weight": 0.5,
             "expr": {"in_container": {"target": "red_alphabet_soup",
                                       "container": "basket"}}},
            {"name": "plain_soup_in_red_basket", "weight": 0.5,
             "expr": {"in_container": {"target": "alphabet_soup",
                                       "container": "red_basket"}}},
            {"name": "no_strays_wicker", "weight": 0.0,
             "expr": {"no_new_packing": {"container": "basket",
                                         "allowed": ["red_alphabet_soup"]}}},
            {"name": "no_strays_red", "weight": 0.0,
             "expr": {"no_new_packing": {"container": "red_basket",
                                         "allowed": ["alphabet_soup"]}}},
        ]},
        timeout_min=75,
    ),
    Case(
        id="size_bigger_soup",
        prompt=("Pick up the larger of the two soup cans and place it in "
                "the basket."),
        category="grounding", tier="T1", tags=("size",),
        sim_task="libero_zsp_sizes/0",
        expected=(
            "perception may struggle to separate the size variants at "
            "camera resolution (probe P6) — expected-hard, kept because "
            "vqa_size_compare isolates the perception half"
        ),
        intent=(
            "size comparative: bigger_alphabet_soup packed, the regular "
            "one untouched."
        ),
        criterion={"checks": [
            *pick_checks("bigger_alphabet_soup"),
            {"name": "smaller_soup_untouched", "weight": 0.0,
             "expr": {"untouched": {"target": "alphabet_soup",
                                    "tol_m": 0.05}}},
        ]},
    ),
    Case(
        id="size_smaller_milk",
        prompt="Put the smaller milk carton in the basket.",
        category="grounding", tier="T1", tags=("size",),
        sim_task="libero_zsp_sizes/0",
        intent=(
            "inverse comparative — guards against a bigger-word bias: "
            "the REGULAR milk is the correct referent."
        ),
        criterion={"checks": [
            *pick_checks("milk"),
            {"name": "bigger_milk_untouched", "weight": 0.0,
             "expr": {"untouched": {"target": "bigger_milk",
                                    "tol_m": 0.05}}},
        ]},
    ),
    Case(
        id="vocab_mac_cheese",
        prompt="Put the mac and cheese box in the basket.",
        category="grounding", tier="T1", tags=("vocabulary",),
        sim_task="libero_zsp_clutter/0",
        intent=(
            "colloquial name -> novel-vocabulary asset "
            "(macaroni_and_cheese) among six clutter distractors."
        ),
        criterion={"checks": pick_checks("macaroni_and_cheese")},
    ),
    Case(
        id="vocab_yellow_cookies",
        prompt=("Two cookie boxes are on the table. Put the yellow one in "
                "the basket."),
        category="grounding", tier="T1", tags=("vocabulary", "color"),
        sim_task="libero_zsp_clutter/0",
        intent=(
            "paired-instance disambiguation by color on novel vocabulary: "
            "yellow_cookies packed, the regular cookies box untouched."
        ),
        criterion={"checks": [
            *pick_checks("yellow_cookies"),
            {"name": "other_cookies_untouched", "weight": 0.0,
             "expr": {"untouched": {"target": "cookies", "tol_m": 0.05}}},
        ]},
    ),
    Case(
        id="wk_popcorn_microwave",
        prompt=("Put the snack you would cook in a microwave into the "
                "basket."),
        category="grounding", tier="T1", tags=("world_knowledge",),
        sim_task="libero_zsp_clutter/0",
        intent="world knowledge on novel vocabulary: popcorn is the referent.",
        criterion={"checks": pick_checks("popcorn")},
    ),
]
