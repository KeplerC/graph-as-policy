"""Per-primitive unit traces: every leaf, op, and $t0 resolver gets a
pass trace and at least one characteristic failure."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from zsp.dsl import EvalContext, states_from_records, validate_criterion
from zsp.dsl.ast import evaluate_check_expr, resolve_t0_refs
from zsp.dsl.score import evaluate_criterion
from zsp.evaluate import evaluate_seed
from zsp.selftest.synth import (
    BASKET_P,
    ROT90X_Q,
    SPREAD,
    FakeCase,
    body,
    in_basket_body,
    lifted_body,
    scene_bodies,
    snap,
)


def ctx_from(records, *, case=None, events=(), execution=None, extras=None,
             options=None):
    c = EvalContext(
        states=states_from_records(list(records)),
        case=case,
        events=list(events),
        execution=dict(execution or {}),
        extras=dict(extras or {}),
    )
    c.options = dict(options or {})
    return c


def check(t, expr, records, expect_ok, msg, **ctx_kw):
    ctx = ctx_from(records, **ctx_kw)
    ok, _ = evaluate_check_expr(expr, ctx)
    t.expect(ok == expect_ok, f"{msg}: got {ok}, wanted {expect_ok}")
    return ctx


def run(t) -> None:  # noqa: PLR0915
    S0 = snap(scene_bodies(), seq=0)

    # -- state extras ----------------------------------------------------

    packed = snap(scene_bodies({"alphabet_soup": in_basket_body("alphabet_soup")}), seq=2)
    IN = {"in_container": {"target": "alphabet_soup", "container": "basket"}}
    check(t, IN, [S0, packed], True, "in_container resting")
    check(t, IN, [S0, S0], False, "in_container: still at spawn")
    held_in = snap(scene_bodies({"alphabet_soup": in_basket_body("alphabet_soup")}),
                   seq=2, held="alphabet_soup", grip=0.4)
    check(t, IN, [S0, held_in], False, "in_container: still held")
    moving = snap(scene_bodies({"alphabet_soup": in_basket_body("alphabet_soup", v=0.5)}), seq=2)
    check(t, IN, [S0, moving], False, "in_container: not settled")
    no_contact = snap(scene_bodies({"alphabet_soup": in_basket_body("alphabet_soup", contact=False)}), seq=2)
    check(t, IN, [S0, no_contact], False,
          "in_container: schema-2 without contact fails")
    v1_no_contact = snap(scene_bodies({"alphabet_soup": in_basket_body("alphabet_soup", contact=False)}),
                         seq=2, schema=None)
    check(t, IN, [snap(scene_bodies(), schema=None), v1_no_contact], True,
          "in_container: v1 trace skips the contact check")
    # cavity-aware: center inside outer AABB but above the cavity
    cav = [[BASKET_P[0] - 0.06, BASKET_P[1] - 0.06, 0.005],
           [BASKET_P[0] + 0.06, BASKET_P[1] + 0.06, 0.12]]
    rim = snap(scene_bodies({
        "basket": body(BASKET_P, ext=(0.176, 0.183, 0.165), cav=cav),
        "alphabet_soup": body((BASKET_P[0], BASKET_P[1], 0.18), c=["basket"]),
    }), seq=2)
    check(t, IN, [S0, rim], False, "in_container: cavity rejects rim-rest")

    ON = {"resting_on": {"target": "butter", "support": "cream_cheese"}}
    stack_ok = snap(scene_bodies({
        "butter": body((0.52, -0.06, 0.030), ext=(0.078, 0.040, 0.018), c=["cream_cheese"]),
    }), seq=2)
    check(t, ON, [S0, stack_ok], True, "resting_on stacked")
    hover = snap(scene_bodies({
        "butter": body((0.52, -0.06, 0.30), ext=(0.078, 0.040, 0.018)),
    }), seq=2)
    check(t, ON, [S0, hover], False, "resting_on: hovering above")
    offset = snap(scene_bodies({
        "butter": body((0.62, -0.06, 0.030), ext=(0.078, 0.040, 0.018)),
    }), seq=2)
    check(t, ON, [S0, offset], False, "resting_on: xy off the base")

    OT = {"on_table": {"target": "alphabet_soup"}}
    check(t, OT, [S0], True, "on_table at spawn")
    check(t, OT, [packed], False, "on_table: in basket")
    up = snap(scene_bodies({"alphabet_soup": lifted_body("alphabet_soup")}), seq=1)
    check(t, OT, [up], False, "on_table: lifted")

    check(t, {"settled": {"target": "milk"}}, [S0], True, "settled")
    check(t, {"settled": {"target": "alphabet_soup"}}, [up], False,
          "settled: moving while lifted")
    check(t, {"held": {"target": "alphabet_soup"}},
          [snap(scene_bodies(), held="alphabet_soup")], True, "held")
    check(t, {"held": {"target": "milk"}},
          [snap(scene_bodies(), held="alphabet_soup")], False, "held: other")

    NP = {"near_point": {"target": "alphabet_soup",
                         "point": list(SPREAD["alphabet_soup"]), "radius_m": 0.05}}
    check(t, NP, [S0], True, "near_point at spawn")
    check(t, NP, [packed], False, "near_point: moved away")

    OA = {"offset_along": {"target": "alphabet_soup", "reference": "milk",
                           "axis": "y", "lo": 0.06, "hi": 0.60,
                           "perp_max": 0.20}}
    check(t, OA, [S0], True, "offset_along y in range")
    OA2 = {"offset_along": {"target": "alphabet_soup", "reference": "milk",
                            "axis": "y", "lo": 0.06, "hi": 0.60,
                            "perp_max": 0.05}}
    check(t, OA2, [S0], False, "offset_along: perp drift too large")
    OA3 = {"offset_along": {"target": "alphabet_soup", "reference": "milk",
                            "axis": "y", "lo": 0.0, "hi": 0.1,
                            "signed": True}}
    check(t, OA3, [S0], False, "offset_along signed: wrong side magnitude")

    cabinet = snap({
        "wooden_cabinet_1": body((0.7, -0.23, 0.0), ext=(0.26, 0.23, 0.23),
                                 j={"wooden_cabinet_1_middle_level": -0.15}),
        "basket": body(BASKET_P),
    }, seq=1)
    JO = {"joint_open": {"body": "wooden_cabinet_1", "threshold": -0.10,
                         "mode": "lt"}}
    check(t, JO, [cabinet], True, "joint_open: drawer pulled")
    closed = snap({
        "wooden_cabinet_1": body((0.7, -0.23, 0.0), ext=(0.26, 0.23, 0.23),
                                 j={"wooden_cabinet_1_middle_level": 0.0}),
        "basket": body(BASKET_P),
    }, seq=1)
    check(t, JO, [closed], False, "joint_open: drawer closed")
    check(t, JO, [snap(scene_bodies())], False, "joint_open: no joints recorded")

    two_in = snap(scene_bodies({
        "alphabet_soup": in_basket_body("alphabet_soup"),
        "milk": in_basket_body("milk", dz=0.02),
    }), seq=2)
    CI = {"count_in": {"container": "basket", "targets": "movables",
                       "count": 2}}
    check(t, CI, [S0, two_in], True, "count_in exact 2")
    check(t, CI, [S0, packed], False, "count_in: only 1")
    ctx = ctx_from([S0, packed])
    ok, leaves = evaluate_check_expr(
        {"count_in": {"container": "basket", "targets": "movables",
                      "count": "all"}}, ctx)
    t.expect(not ok and abs(leaves[0]["diagnostics"]["score"] - 1 / 6) < 1e-6,
             "count_in all: fractional score 1/6")

    EE = {"ee_near": {"point": [0.4, 0.0, 0.3], "tol_m": 0.05}}
    check(t, EE, [S0], True, "ee_near point")
    check(t, {"ee_near": {"point": [0.9, 0.4, 0.3], "tol_m": 0.05}}, [S0],
          False, "ee_near: far")
    EEB = {"ee_near": {"body": "milk", "offset": [0, 0, 0.15],
                       "tol_xy_m": 0.04, "tol_z_m": 0.06}}
    hover_ee = snap(scene_bodies(), ee=(SPREAD["milk"][0], SPREAD["milk"][1],
                                        SPREAD["milk"][2] + 0.15))
    check(t, EEB, [hover_ee], True, "ee_near body+offset hover")
    check(t, EEB, [S0], False, "ee_near body: not above")

    # -- core registry reuse through the walker ---------------------------

    check(t, {"near": {"target": "alphabet_soup", "anchor": "salad_dressing",
                       "tol_m": 0.15}}, [S0], True, "core near")
    check(t, {"stack": ["butter", "cream_cheese"]}, [S0, stack_ok], True,
          "core stack (positional args)")
    check(t, {"xy_within": {"target": "alphabet_soup", "reference": "basket",
                            "radius_m": 0.05}}, [packed], True, "core xy_within")
    check(t, {"axis_aligned": {"body": "milk"}}, [S0], True, "core axis_aligned")
    check(t, {"near": {"target": "ghost", "anchor": "milk"}}, [S0], False,
          "core leaf: missing body soft-fails")

    # -- trace leaves ------------------------------------------------------

    pick_trace = [S0, snap(scene_bodies({"alphabet_soup": lifted_body("alphabet_soup")}),
                           seq=1, held="alphabet_soup", grip=0.4),
                  packed]
    PU = {"picked_up": {"target": "alphabet_soup"}}
    check(t, PU, pick_trace, True, "picked_up")
    nudge = [S0, snap(scene_bodies(), seq=1, held="alphabet_soup", grip=0.4)]
    check(t, PU, nudge, False, "picked_up: touch without lift")
    open_grip = [S0, snap(scene_bodies({"alphabet_soup": lifted_body("alphabet_soup")}),
                          seq=1, held="alphabet_soup", grip=0.95)]
    check(t, PU, open_grip, False, "picked_up: gripper open")
    check(t, PU, [S0, packed], False, "picked_up: teleported")

    check(t, {"never_held": {"target": "milk"}}, pick_trace, True, "never_held")
    check(t, {"never_held": {"target": "alphabet_soup"}}, pick_trace, False,
          "never_held: was held")

    BF = {"before": {"first": {"picked_up": {"target": "alphabet_soup"}},
                     "then": {"enters_container": {"target": "alphabet_soup"}}}}
    check(t, BF, pick_trace, True, "before: pick then pack")
    check(t, BF, [S0, packed], False, "before: events missing")
    two_packed_trace = [
        S0,
        snap(scene_bodies({"milk": in_basket_body("milk")}), seq=1),
        snap(scene_bodies({"milk": in_basket_body("milk"),
                           "alphabet_soup": in_basket_body("alphabet_soup", dz=0.03)}), seq=2),
    ]
    BF2 = {"before": {"first": {"enters_container": {"target": "alphabet_soup"}},
                      "then": {"enters_container": {"target": "milk"}}}}
    check(t, BF2, two_packed_trace, False, "before: wrong order")

    moved = snap(scene_bodies({
        "alphabet_soup": body((SPREAD["alphabet_soup"][0],
                               SPREAD["alphabet_soup"][1] + 0.05,
                               SPREAD["alphabet_soup"][2]),
                              ext=(0.085, 0.089, 0.085)),
    }), seq=3)
    DP = {"displaced": {"target": "alphabet_soup", "axis": "y",
                        "lo": 0.021, "hi": 0.081}}
    check(t, DP, [S0, moved], True, "displaced y in band")
    check(t, DP, [S0, S0], False, "displaced: didn't move")
    check(t, {"displaced": {"target": "alphabet_soup", "axis": "xy",
                            "lo": 0.0, "hi": 0.02}}, [S0, moved], False,
          "displaced xy: exceeded hi")
    check(t, {"displaced": {"target": "alphabet_soup", "axis": "y",
                            "lo": -0.10, "hi": -0.01, "signed": True}},
          [S0, moved], False, "displaced signed: wrong direction")

    flip = snap(scene_bodies({}, basket_p=(0.71, -0.23, 0.08)), seq=3)
    RL = {"relocated": {"target": "basket"}}
    ctx = ctx_from([snap(scene_bodies({}, basket_p=(0.71, 0.23, 0.08))), flip])
    ok, leaves = evaluate_check_expr(RL, ctx)
    t.expect(ok, "relocated: lateral flip")
    ctx = ctx_from([S0, snap(scene_bodies({}, basket_p=(0.75, 0.10, 0.08)), seq=1)])
    ok, leaves = evaluate_check_expr(RL, ctx)
    t.expect(not ok and 0 < leaves[0]["diagnostics"]["score"] <= 0.5,
             "relocated: small move fails with partial score")

    tipped = snap(scene_bodies({
        "milk": body(SPREAD["milk"], q=ROT90X_Q, ext=(0.065, 0.065, 0.162)),
    }), seq=2)
    TL = {"tilt_from_initial": {"target": "milk", "max_deg": 30}}
    check(t, TL, [S0, S0], True, "tilt: unchanged")
    check(t, TL, [S0, tipped], False, "tilt: knocked over")
    check(t, {"tilt_from_initial": {"target": "milk", "min_deg": 60,
                                    "max_deg": 120}}, [S0, tipped], True,
          "tilt band: laid on side")

    UT = {"untouched": {"target": "milk", "tol_m": 0.02}}
    check(t, UT, pick_trace, True, "untouched bystander")
    check(t, UT, [S0, tipped, S0], True, "untouched: rotated in place ok")
    bumped = snap(scene_bodies({
        "milk": body((SPREAD["milk"][0], SPREAD["milk"][1] + 0.08,
                      SPREAD["milk"][2]), ext=(0.065, 0.065, 0.162)),
    }), seq=1)
    check(t, UT, [S0, bumped, S0], False, "untouched: bumped mid-run")
    check(t, {"untouched": {"target": "alphabet_soup", "tol_m": 0.02}},
          nudge, False, "untouched: was held")

    AU = {"all_untouched": {"except": ["alphabet_soup"], "tol_m": 0.03}}
    check(t, AU, pick_trace, True, "all_untouched with exception")
    check(t, {"all_untouched": {"tol_m": 0.03}}, pick_trace, False,
          "all_untouched: mover caught")

    NNP = {"no_new_packing": {"allowed": ["alphabet_soup"]}}
    check(t, NNP, [S0, packed], True, "no_new_packing: only allowed")
    check(t, NNP, [S0, two_in], False, "no_new_packing: milk stray")
    prepacked = snap(scene_bodies({"milk": in_basket_body("milk")}), seq=0)
    check(t, NNP, [prepacked, snap(scene_bodies({"milk": in_basket_body("milk"),
                                                 "alphabet_soup": in_basket_body("alphabet_soup", dz=0.03)}), seq=1)],
          True, "no_new_packing: pre-packed at t0 doesn't count")

    KO = {"kept_out": {"targets": ["milk", "butter"]}}
    check(t, KO, [S0, packed], True, "kept_out: both out")
    ctx = ctx_from([S0, two_in])
    ok, leaves = evaluate_check_expr(KO, ctx)
    t.expect(not ok and abs(leaves[0]["diagnostics"]["score"] - 0.5) < 1e-9,
             "kept_out: milk packed -> score 0.5")

    grave = snap(scene_bodies({"alphabet_soup": body((50.6, 50.0, 0.05))}), seq=3)
    PC = {"packed_or_consumed": {"target": "alphabet_soup"}}
    check(t, PC, pick_trace, True, "packed_or_consumed: seen in basket")
    check(t, PC, [S0, snap(scene_bodies({"alphabet_soup": lifted_body("alphabet_soup")}),
                           seq=1, held="alphabet_soup", grip=0.4), grave],
          True, "packed_or_consumed: grasped then env-consumed")
    check(t, PC, [S0, grave], False, "packed_or_consumed: teleport w/o grasp")

    PR = {"pack_report": {"targets": "movables", "mode": "final"}}
    all_in = snap({n: in_basket_body(n, dz=0.01 * i)
                   for i, n in enumerate(SPREAD)} | {"basket": body(BASKET_P, ext=(0.176, 0.183, 0.165))},
                  seq=2)
    check(t, PR, [S0, all_in], True, "pack_report final: all packed")
    ctx = ctx_from([S0, packed])
    ok, leaves = evaluate_check_expr(PR, ctx)
    t.expect(not ok and abs(leaves[0]["diagnostics"]["score"] - 1 / 6) < 1e-6,
             "pack_report: 1/6 partial")

    SS = {"scene_sane": {}}
    check(t, SS, pick_trace, True, "scene_sane ok")
    flung = snap(scene_bodies({"butter": body((0.5, 0.1, -0.5))}), seq=2)
    check(t, SS, [S0, flung], False, "scene_sane: below floor")
    case_consume = FakeCase({"checks": []}, env_consumes_packed=True)
    check(t, SS, [S0, grave], True, "scene_sane: graveyard ok when env consumes",
          case=case_consume)
    check(t, SS, [S0, grave], False, "scene_sane: graveyard is a fling otherwise")

    check(t, {"env_success": {}}, [S0, S0], True, "env_success true",
          execution={"env_task_completed": True})
    check(t, {"env_success": {}}, [S0, S0], False, "env_success false",
          execution={"env_task_completed": False})
    check(t, {"exit_status": {"equals": "failure"}}, [S0, S0], True,
          "exit_status failure", execution={"exit_status": "failure"})
    check(t, {"gen_refused": {}}, [S0, S0], True, "gen_refused",
          extras={"gen_outcome": "REFUSED", "gen_message": "missing capability: no banana"})
    check(t, {"gen_refused": {"pattern": "banana"}}, [S0, S0], True,
          "gen_refused with pattern",
          extras={"gen_outcome": "REFUSED", "gen_message": "no banana here"})
    check(t, {"gen_refused": {}}, [S0, S0], False, "gen_refused: generated fine",
          extras={"gen_outcome": "OK"})

    events = [
        {"ev": "call", "seq": 1, "tool": "vlm.query", "node": "count"},
        {"ev": "done", "seq": 1, "tool": "vlm.query", "node": "count",
         "summary": {"answer": "I count 6 objects on the table"}},
    ]
    RP = {"reported": {"pattern": r"count (\d+)", "equals": 6}}
    check(t, RP, [S0, S0], True, "reported: numeric equals", events=events)
    check(t, {"reported": {"pattern": r"count (\d+)", "equals": 5}},
          [S0, S0], False, "reported: wrong number", events=events)
    check(t, RP, [S0, S0], False, "reported: no events", events=[])
    with tempfile.TemporaryDirectory() as td:
        nd = Path(td) / "node_data" / "report" / "iters" / "0"
        nd.mkdir(parents=True)
        (nd / "output.json").write_text(json.dumps({"closest": "tomato sauce"}))
        check(t, {"reported": {"pattern": r"closest.{0,20}(tomato.sauce)"}},
              [S0, S0], True, "reported: node_outputs file",
              extras={"trace_dir": td})

    AC = {"attempt_count": {"tool_regex": "close_gripper|grasp", "max": 3}}
    calls = [{"ev": "call", "seq": i, "tool": "robot.close_gripper"}
             for i in range(1, 3)]
    check(t, AC, [S0, S0], True, "attempt_count under budget", events=calls)
    calls5 = [{"ev": "call", "seq": i, "tool": "robot.close_gripper"}
              for i in range(1, 6)]
    check(t, AC, [S0, S0], False, "attempt_count over budget", events=calls5)

    row = snap(scene_bodies({
        "alphabet_soup": body((0.50, -0.10, 0.04)),
        "tomato_sauce": body((0.55, 0.00, 0.04)),
        "milk": body((0.60, 0.10, 0.04), ext=(0.065, 0.065, 0.162)),
    }), seq=2)
    CL = {"colinear": {"bodies": ["alphabet_soup", "tomato_sauce", "milk"],
                       "residual_max": 0.02}}
    check(t, CL, [S0, row], True, "colinear row")
    check(t, CL, [S0, S0], False, "colinear: scattered spawn")

    check(t, {"workflow_has_cycle": {"expect": True}}, [S0, S0], True,
          "workflow_has_cycle expect",
          extras={"generated_graph_structure": {"any_cycle": True}})
    check(t, {"node_revisits": {"min": 2}}, [S0, S0], True, "node_revisits min",
          extras={"max_node_iterations": 4})
    check(t, {"node_revisits": {"min": 2}}, [S0, S0], False,
          "node_revisits: straight-line run", extras={"max_node_iterations": 1})

    # -- composition ops ---------------------------------------------------

    check(t, {"and": [PU, IN]}, pick_trace, True, "and")
    check(t, {"and": [PU, {"held": {"target": "alphabet_soup"}}]}, pick_trace,
          False, "and: at_end default fails released leaf")
    check(t, {"or": [{"held": {"target": "alphabet_soup"}}, IN]}, pick_trace,
          True, "or")
    check(t, {"not": {"held": {"target": "alphabet_soup"}}}, pick_trace, True,
          "not at_end")
    check(t, {"eventually": {"held": {"target": "alphabet_soup"}}}, pick_trace,
          True, "eventually held mid-trace")
    check(t, {"never": {"held": {"target": "milk"}}}, pick_trace, True, "never")
    check(t, {"never": {"held": {"target": "alphabet_soup"}}}, pick_trace,
          False, "never: was held")
    check(t, {"always": {"settled": {"target": "milk"}}}, pick_trace, True,
          "always settled bystander")
    check(t, {"always": {"settled": {"target": "alphabet_soup"}}}, pick_trace,
          False, "always: moved mid-trace")
    check(t, {"at_end": {"and": [IN, {"not": {"held": {"target": "alphabet_soup"}}}]}},
          pick_trace, True, "at_end nested and")

    # -- $t0 resolvers -----------------------------------------------------

    ctx = ctx_from([S0, packed])
    t.expect(resolve_t0_refs({"$t0": {"closest_to": {"anchor": "basket"}}}, ctx)
             == "salad_dressing", "$t0 closest_to basket (probed layout)")
    t.expect(resolve_t0_refs({"$t0": {"farthest_from": {"anchor": "basket"}}}, ctx)
             == "milk", "$t0 farthest_from")
    t.expect(resolve_t0_refs({"$t0": {"extreme": {"axis": "y", "mode": "min"}}}, ctx)
             == "milk", "$t0 extreme min y")
    t.expect(resolve_t0_refs({"$t0": {"largest": {}}}, ctx) == "milk",
             "$t0 largest by volume (milk carton)")
    t.expect(resolve_t0_refs({"$t0": {"smallest": {}}}, ctx) == "butter",
             "$t0 smallest by volume")
    t.expect(resolve_t0_refs({"$t0": {"largest": {"by": "height"}}}, ctx)
             == "milk", "$t0 tallest")
    mid = resolve_t0_refs({"$t0": {"midpoint": {"a": "milk", "b": "alphabet_soup"}}}, ctx)
    t.expect(abs(mid[0] - 0.54) < 0.01 and abs(mid[1] - 0.015) < 0.01,
             "$t0 midpoint")
    pos = resolve_t0_refs({"$t0": {"pos_of": {"body": "milk", "offset": [0, 0, 0.15]}}}, ctx)
    t.expect(abs(pos[2] - (SPREAD["milk"][2] + 0.15)) < 1e-6, "$t0 pos_of offset")
    t.expect(resolve_t0_refs({"$t0": {"count_within": {"anchor": "basket", "radius_m": 0.20}}}, ctx)
             == 2, "$t0 count_within (salad_dressing 0.184, tomato_sauce 0.192)")
    d = resolve_t0_refs({"$t0": {"distance": {"a": "milk", "b": "alphabet_soup", "axis": "xy"}}}, ctx)
    t.expect(0.4 < d < 0.6, "$t0 distance xy")

    # -- validation --------------------------------------------------------

    def bad(criterion, msg, bodies=None):
        try:
            validate_criterion(criterion, known_bodies=bodies)
        except ValueError:
            t.expect(True, msg)
        else:
            t.expect(False, f"{msg}: validation should have failed")

    good = {"checks": [{"name": "a", "weight": 1.0, "expr": PU}]}
    validate_criterion(good)
    t.expect(True, "validate: minimal criterion ok")
    bad({"checks": []}, "validate: empty checks")
    bad({"checks": [{"name": "a", "expr": {"nope": {}}}]}, "validate: unknown leaf")
    bad({"checks": [{"name": "a", "expr": {"eventually": PU}}]},
        "validate: trace leaf inside temporal")
    bad({"checks": [{"name": "a", "expr": PU}, {"name": "a", "expr": PU}]},
        "validate: duplicate names")
    bad({"checks": [{"name": "a", "weight": -1, "expr": PU}]},
        "validate: negative weight")
    bad({"checks": [{"name": "a", "expr": PU}], "options": {"wat": 1}},
        "validate: unknown option")
    bad({"checks": [{"name": "a", "expr": {"picked_up": {"target": "ghost"}}}]},
        "validate: unknown body vs layout", bodies={"alphabet_soup", "basket"})
    bad({"checks": [{"name": "a", "expr": {"before": {"first": {"picked_up": {"target": "x"}}}}}]},
        "validate: before missing then", bodies=None)
    bad({"checks": [{"name": "a", "expr": {"reported": {}}}]},
        "validate: reported without pattern")
    validate_criterion(
        {"checks": [{"name": "a", "expr": {"in_container": {
            "target": {"$t0": {"closest_to": {"anchor": "basket"}}},
            "container": "basket"}}}]},
        known_bodies={"alphabet_soup", "basket"})
    t.expect(True, "validate: $t0 in body-arg position ok")
    bad({"checks": [{"name": "a", "expr": {"in_container": {
        "target": {"$t0": {"wat": {}}}, "container": "basket"}}}]},
        "validate: unknown $t0 resolver")

    # -- scoring / injections ----------------------------------------------

    crit = {"checks": [
        {"name": "in", "weight": 0.5,
         "expr": {"in_container": {"target": "alphabet_soup", "container": "basket"}}},
    ]}
    case = FakeCase(crit)
    ev = evaluate_criterion(case, crit, ctx_from([S0, packed], case=case))
    t.expect(ev.verdict == "FAIL" and any(
        c.name == "alphabet_soup_picked_up" and c.data.get("injected")
        for c in ev.checks),
        "injection: containment without pickup fails via injected check")
    crit_push = {"checks": crit["checks"],
                 "options": {"push_ok": ["alphabet_soup"]}}
    ev = evaluate_criterion(case, crit_push,
                            ctx_from([S0, packed], case=case))
    t.expect(ev.verdict == "PASS", "injection: push_ok opt-out")
    crit_nopick = {"checks": crit["checks"],
                   "options": {"require_picked_up": False}}
    ev = evaluate_criterion(case, crit_nopick,
                            ctx_from([S0, packed], case=case))
    t.expect(ev.verdict == "PASS", "injection: require_picked_up=false")
    ev = evaluate_criterion(case, crit, ctx_from([S0, flung], case=case))
    t.expect(ev.verdict == "FAIL" and any(
        c.name == "scene_sane" and not c.passed for c in ev.checks),
        "injection: scene_sane always gates")

    # -- evaluate_seed shapes ------------------------------------------------

    pick_case = FakeCase({"checks": [
        {"name": "picked", "weight": 0.3, "expr": PU},
        {"name": "in", "weight": 0.5, "expr": IN},
        {"name": "stray", "weight": 0.2,
         "expr": {"no_new_packing": {"allowed": ["alphabet_soup"]}}},
    ]})
    ev = evaluate_seed(pick_case, [json.loads(json.dumps(s)) for s in pick_trace],
                       [], {})
    t.expect_eval(ev, "PASS", "evaluate_seed pick-place", score_min=0.99)
    ev = evaluate_seed(pick_case, [], [], {})
    t.expect(ev.verdict == "NO_TRACE", "evaluate_seed: empty trace NO_TRACE")
    ev = evaluate_seed(pick_case, [], [], {"timed_out": True})
    t.expect(ev.verdict == "TIMEOUT", "evaluate_seed: timeout no trace")
    ev = evaluate_seed(pick_case, pick_trace, [], {"timed_out": True})
    t.expect(ev.verdict == "TIMEOUT", "evaluate_seed: timeout override")
    ev = evaluate_seed(pick_case, pick_trace, [], {"harness_crash": "boom"})
    t.expect(ev.verdict == "CRASH", "evaluate_seed: crash override")
    hang_events = [{"ev": "call", "seq": 9, "tool": "curobo.plan", "node": "g"}]
    ev = evaluate_seed(pick_case, pick_trace, hang_events, {"timed_out": True})
    t.expect(ev.diagnostics.get("hang", {}).get("tool") == "curobo.plan",
             "evaluate_seed: hang site recovered")

    env_case = FakeCase({"checks": [
        {"name": "in", "weight": 1.0, "expr": IN},
    ]}, success_source="env")
    ev = evaluate_seed(env_case, [S0, S0], [], {"env_task_completed": True})
    t.expect(ev.verdict == "PASS",
             "success_source=env: env truth wins over authored miss")
    ev = evaluate_seed(env_case, pick_trace, [], {"env_task_completed": False})
    t.expect(ev.verdict == "FAIL" and ev.score > 0.99,
             "success_source=env: authored progress kept as score")
    both_case = FakeCase({"checks": [
        {"name": "in", "weight": 1.0, "expr": IN},
    ]}, success_source="both")
    ev = evaluate_seed(both_case, pick_trace, [], {"env_task_completed": False})
    t.expect(ev.verdict == "FAIL", "success_source=both: env miss fails")
    ev = evaluate_seed(both_case, pick_trace, [], {"env_task_completed": True})
    t.expect(ev.verdict == "PASS", "success_source=both: both pass")

    gen_case = FakeCase({"checks": [
        {"name": "refused", "weight": 1.0, "expr": {"gen_refused": {}}},
    ]}, success_source="gen")
    ev = evaluate_seed(gen_case, [], [], {}, extras={"gen_outcome": "REFUSED"})
    t.expect(ev.verdict == "PASS", "success_source=gen: refusal accepted")
    ev = evaluate_seed(gen_case, [], [], {}, extras={"gen_outcome": "OK"})
    t.expect(ev.verdict == "FAIL", "success_source=gen: generated -> fail")
