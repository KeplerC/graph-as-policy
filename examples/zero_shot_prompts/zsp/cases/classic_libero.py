"""Classic-LIBERO generalization + articulated frontier.

Prompts are the PROBED native task languages VERBATIM (layouts/ record
what the env reports — e.g. goal/0 says 'Open the middle layer of the
drawer', not the suite-map phrasing), so ``env.check_success()`` is the
task's own predicate: these cases score with ``success_source="env"``
(env gates; authored checks become weighted partial-progress
diagnostics) or ``"both"`` where our trace check is safe to co-gate.
Articulated joint checks need schema-2 traces (drawer/knob/door qpos —
probed names and ranges in layouts/)."""

from __future__ import annotations

from zsp.cases._shared import CLASSIC_OPTIONS
from zsp.schema import Case

CASES: list[Case] = [
    Case(
        id="object_native_classic",
        prompt="Pick the tomato sauce and place it in the basket",
        category="classic", tier="T1", tags=("native",),
        sim_task="libero_object/5",
        success_source="both",
        intent=(
            "classic-pipeline anchor (BDDL env, pruned inits, _1 body "
            "suffixes): isolates fork/loader/perception issues from "
            "capability. Env goal == prompt, so both gate."
        ),
        criterion={"checks": [
            {"name": "tomato_sauce_picked", "weight": 0.4,
             "expr": {"picked_up": {"target": "tomato_sauce_1"}}},
            {"name": "tomato_sauce_in_basket", "weight": 0.6,
             "expr": {"in_container": {"target": "tomato_sauce_1",
                                       "container": "basket_1"}}},
        ], "options": {"container": "basket_1"}},
    ),
    Case(
        id="spatial_native_between",
        prompt=("Pick the akita black bowl between the plate and the "
                "ramekin and place it on the plate"),
        category="classic", tier="T1", tags=("native", "spatial_referent"),
        sim_task="libero_spatial/0",
        success_source="env",
        intent=(
            "the strongest grounding test in classic LIBERO: TWO "
            "identical bowls (probed); only the one between the plate "
            "and the ramekin counts, and the env's BDDL predicate knows "
            "which. Trace checks report which bowl moved (non-gating)."
        ),
        criterion={"checks": [
            {"name": "bowl_1_on_plate", "weight": 0.5,
             "expr": {"resting_on": {"target": "akita_black_bowl_1",
                                     "support": "plate_1",
                                     "max_above": 0.06}}},
            {"name": "bowl_2_on_plate", "weight": 0.5,
             "expr": {"resting_on": {"target": "akita_black_bowl_2",
                                     "support": "plate_1",
                                     "max_above": 0.06}}},
        ], "options": CLASSIC_OPTIONS},
        timeout_min=60,
    ),
    Case(
        id="bowl_on_plate_goal",
        prompt="Put the bowl on the plate",
        category="classic", tier="T1", tags=("native",),
        sim_task="libero_goal/8",
        success_source="both",
        intent=(
            "T1 anchor for the classic kitchen scene — separates 'scene "
            "is perceivable' from 'articulation is hard': the single "
            "bowl resting on the plate, env + trace co-gating."
        ),
        criterion={"checks": [
            {"name": "bowl_picked", "weight": 0.4,
             "expr": {"picked_up": {"target": "akita_black_bowl_1"}}},
            {"name": "bowl_on_plate", "weight": 0.6,
             "expr": {"resting_on": {"target": "akita_black_bowl_1",
                                     "support": "plate_1",
                                     "max_above": 0.06}}},
        ], "options": CLASSIC_OPTIONS},
        timeout_min=60,
    ),
    # ---- articulated frontier ----------------------------------------------
    Case(
        id="drawer_open_middle",
        prompt="Open the middle layer of the drawer",
        category="articulated", tier="T2", tags=("drawer",),
        sim_task="libero_goal/0",
        success_source="env",
        requires_schema=2,
        expected=(
            "no articulation skill exists — tests handle-perception + "
            "cartesian-pull composition; joint excursion is the partial-"
            "progress signal (probed slide range [-0.16, 0.01], open "
            "beyond -0.10)"
        ),
        intent=(
            "env BDDL Open predicate gates; the middle drawer's recorded "
            "slide qpos past -0.10 is the weighted progress diagnostic."
        ),
        criterion={"checks": [
            {"name": "middle_drawer_pulled", "weight": 1.0,
             "expr": {"joint_open": {
                 "body": "wooden_cabinet_1",
                 "joint": "wooden_cabinet_1_middle_level",
                 "threshold": -0.10, "mode": "lt"}}},
        ], "options": CLASSIC_OPTIONS},
        timeout_min=60,
    ),
    Case(
        id="stove_knob_on",
        prompt="Turn on the stove",
        category="articulated", tier="T2", tags=("knob",),
        sim_task="libero_goal/7",
        success_source="env",
        requires_schema=2,
        expected=(
            "small-knob rotation — precision beyond the drawer pull; "
            "likely fails, kept for the measured gap (probed hinge "
            "range [-0.005, 2.1], on beyond 0.5)"
        ),
        intent=(
            "env BDDL TurnOn gates; the stove button hinge past 0.5 rad "
            "is the weighted progress diagnostic."
        ),
        criterion={"checks": [
            {"name": "knob_turned", "weight": 1.0,
             "expr": {"joint_open": {
                 "body": "flat_stove_1",
                 "joint": "flat_stove_1_button",
                 "threshold": 0.5, "mode": "gt"}}},
        ], "options": CLASSIC_OPTIONS},
        timeout_min=60,
    ),
    Case(
        id="drawer_open_put_bowl",
        prompt="Open the top layer of the drawer and put the bowl inside",
        category="articulated", tier="T2", tags=("drawer", "long_horizon"),
        sim_task="libero_goal/3",
        success_source="env",
        requires_schema=2,
        expected="articulate + place-inside composition — classic long-horizon",
        intent=(
            "env gates the two-conjunct BDDL goal; weighted progress: "
            "top drawer pulled past -0.10 and the bowl's center inside "
            "the cabinet's AABB."
        ),
        criterion={"checks": [
            {"name": "top_drawer_pulled", "weight": 0.5,
             "expr": {"joint_open": {
                 "body": "wooden_cabinet_1",
                 "joint": "wooden_cabinet_1_top_level",
                 "threshold": -0.10, "mode": "lt"}}},
            {"name": "bowl_inside_cabinet", "weight": 0.5,
             "expr": {"in_container": {"target": "akita_black_bowl_1",
                                       "container": "wooden_cabinet_1"}}},
        ], "options": CLASSIC_OPTIONS},
        timeout_min=90,
    ),
    Case(
        id="push_plate_stove",
        prompt="Push the plate to the front of the stove",
        category="articulated", tier="T2", tags=("push", "native"),
        sim_task="libero_goal/5",
        success_source="env",
        expected=(
            "pushing on a classic env WITH a native oracle — cross-"
            "checks the push-family criteria against BDDL truth"
        ),
        intent=(
            "env region predicate gates; weighted progress: the plate "
            "moved >= 5 cm and was never grasped (a push, not a carry — "
            "carrying still passes the env, so this stays diagnostic)."
        ),
        criterion={"checks": [
            {"name": "plate_moved", "weight": 0.7,
             "expr": {"displaced": {"target": "plate_1", "axis": "xy",
                                    "lo": 0.05}}},
            {"name": "never_grasped", "weight": 0.3,
             "expr": {"never_held": {"target": "plate_1"}}},
        ], "options": {**CLASSIC_OPTIONS, "push_ok": ["plate_1"]}},
        timeout_min=60,
    ),
    # ---- long-horizon classic ----------------------------------------------
    Case(
        id="book_caddy",
        prompt="pick up the book and place it in the back compartment of the caddy",
        category="classic", tier="T2", tags=("native", "object_part"),
        sim_task="libero_10/5",
        success_source="env",
        expected=(
            "object-PART destination — direct fit for the "
            "perceiving-object-parts skill; the env region predicate "
            "knows the exact compartment"
        ),
        intent=(
            "env gates (back-compartment region); weighted progress: "
            "book picked up and inside the caddy's AABB."
        ),
        criterion={"checks": [
            {"name": "book_picked", "weight": 0.5,
             "expr": {"picked_up": {"target": "black_book_1"}}},
            {"name": "book_in_caddy", "weight": 0.5,
             "expr": {"in_container": {"target": "black_book_1",
                                       "container": "desk_caddy_1"}}},
        ], "options": CLASSIC_OPTIONS},
        timeout_min=90,
    ),
    Case(
        id="both_moka_pots",
        prompt="put both moka pots on the stove",
        category="classic", tier="T2", tags=("native", "iteration"),
        sim_task="libero_10/8",
        success_source="env",
        expected=(
            "two bulky-handled objects onto one small shared "
            "destination — placement collisions expected"
        ),
        intent=(
            "env gates; weighted per-pot progress: each moka pot "
            "resting on the stove slab."
        ),
        criterion={"checks": [
            {"name": "moka_1_on_stove", "weight": 0.5,
             "expr": {"resting_on": {"target": "moka_pot_1",
                                     "support": "flat_stove_1",
                                     "max_above": 0.06}}},
            {"name": "moka_2_on_stove", "weight": 0.5,
             "expr": {"resting_on": {"target": "moka_pot_2",
                                     "support": "flat_stove_1",
                                     "max_above": 0.06}}},
        ], "options": CLASSIC_OPTIONS},
        timeout_min=120,
    ),
    Case(
        id="microwave_mug",
        prompt="put the yellow and white mug in the microwave and close it",
        category="articulated", tier="T2",
        tags=("native", "door", "long_horizon"),
        sim_task="libero_10/9",
        success_source="env",
        requires_schema=2,
        expected=(
            "the hardest articulated composition (insert through an "
            "open door, then close it — probed: the door STARTS open at "
            "qpos -1.47); expected fail, measures the gap"
        ),
        intent=(
            "env gates; weighted progress: mug inside the microwave's "
            "AABB and the door hinge back near closed (>= -0.3 from a "
            "-1.47 start)."
        ),
        criterion={"checks": [
            {"name": "mug_in_microwave", "weight": 0.6,
             "expr": {"in_container": {"target": "white_yellow_mug_1",
                                       "container": "microwave_1"}}},
            {"name": "door_mostly_closed", "weight": 0.4,
             "expr": {"joint_open": {
                 "body": "microwave_1",
                 "joint": "microwave_1_microjoint",
                 "threshold": -0.3, "mode": "ge"}}},
        ], "options": CLASSIC_OPTIONS},
        timeout_min=120,
    ),
]
