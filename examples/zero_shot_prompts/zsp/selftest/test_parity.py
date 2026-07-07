"""Migration parity gate: the v1 selftest scenarios VERBATIM.

Same synthetic traces (v1 record shape, schema 1 — no contacts), same
expected verdicts and score windows, evaluated through the migrated
cases + DSL. If this fails, the migration changed semantics.
"""

from __future__ import annotations

import math

from zsp import diagnostics
from zsp.evaluate import evaluate_seed
from zsp.registry import CASES

# Layout mirroring the probed seed-0 scene (v1 selftest constants).
SPREAD = {
    "alphabet_soup": (0.61, 0.25, 0.038),
    "butter": (0.72, -0.24, 0.009),
    "cream_cheese": (0.65, -0.10, 0.009),
    "milk": (0.40, -0.08, 0.070),
    "salad_dressing": (0.39, -0.26, 0.073),
    "tomato_sauce": (0.45, 0.06, 0.038),
}
BASKET_P = (0.75, 0.03, -0.005)
BASKET_EXT = (0.176, 0.184, 0.165)
IDQ = (1.0, 0.0, 0.0, 0.0)
ROT90X = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)

EXEC_OK = {"ran": True, "success": True, "exit_status": "success",
           "harness_crash": None}


def body(p, q=IDQ, ext=(0.06, 0.06, 0.06), v=0.0, g=False):
    half = [e / 2 for e in ext]
    return {
        "p": list(p), "q": list(q),
        "lo": [p[i] - half[i] for i in range(3)],
        "hi": [p[i] + half[i] for i in range(3)],
        "v": v, "g": g,
    }


def snap(*, held=None, grip=1.0, basket_p=BASKET_P, overrides=None):
    bodies = {"basket": body(basket_p, ext=BASKET_EXT)}
    for name, p in SPREAD.items():
        bodies[name] = body(p)
    for name, b in (overrides or {}).items():
        bodies[name] = b
    return {"seq": 0, "t": 0.0, "tool": "x", "node": None,
            "held": held, "grip": grip, "ee": [0.45, 0.0, 0.26],
            "bodies": bodies}


def in_basket_p(basket_p=BASKET_P, dx=0.0, dy=0.0, dz=0.03):
    return (basket_p[0] + dx, basket_p[1] + dy, basket_p[2] + dz)


def lifted(name, dz=0.12, q=IDQ):
    p = SPREAD[name]
    return body((p[0], p[1], p[2] + dz), q=q, g=True)


def ev(case_id, states, events=(), execution=EXEC_OK):
    return evaluate_seed(CASES[case_id], states, list(events),
                         dict(execution))


def run(t) -> None:  # noqa: PLR0915
    x = t.expect_eval
    init = snap()

    # ---- soup_basket ----
    held_soup = snap(held="alphabet_soup", grip=0.4,
                     overrides={"alphabet_soup": lifted("alphabet_soup")})
    soup_in = snap(overrides={"alphabet_soup": body(in_basket_p())})
    x(ev("soup_basket", [init, held_soup, soup_in]), "PASS",
      "parity: soup_basket pick+place", score_min=0.99)
    x(ev("soup_basket", [init, soup_in]), "FAIL",
      "parity: soup_basket teleport", score_max=0.75)
    stray_in = snap(overrides={
        "alphabet_soup": body(in_basket_p(dx=-0.04)),
        "butter": body(in_basket_p(dx=0.04), ext=(0.078, 0.04, 0.018))})
    x(ev("soup_basket", [init, held_soup, stray_in]), "FAIL",
      "parity: soup_basket unrequested packing")
    touch_only = snap(held="alphabet_soup", grip=0.4)
    x(ev("soup_basket", [init, touch_only, soup_in]), "FAIL",
      "parity: soup_basket touch-without-lift")
    flung = snap(overrides={
        "alphabet_soup": body(in_basket_p()),
        "milk": body((0.7, 0.0, -0.5), ext=(0.058, 0.06, 0.145))})
    x(ev("soup_basket", [init, held_soup, flung]), "FAIL",
      "parity: scene_sane catches flung object")

    # ---- all_objects ----
    names = list(SPREAD)
    trace = [init]
    packed: dict[str, dict] = {}
    for i, n in enumerate(names):
        trace.append(snap(held=n, grip=0.4,
                          overrides={**packed, n: lifted(n)}))
        packed[n] = body(in_basket_p(dx=0.05 * (i % 3) - 0.05,
                                     dy=0.05 * (i // 3) - 0.02,
                                     dz=0.03 + 0.02 * (i // 3)))
        trace.append(snap(overrides=dict(packed)))
    x(ev("all_objects", trace), "PASS", "parity: all_objects 6/6",
      score_min=0.99)
    only_soup = [init, held_soup, soup_in]
    x(ev("all_objects", only_soup), "FAIL",
      "parity: all_objects only-soup at 1/6", score_max=0.2)

    # ---- soup_then_milk ----
    milk_held = snap(held="milk", grip=0.4, overrides={
        "alphabet_soup": body(in_basket_p(dx=-0.04)),
        "milk": lifted("milk")})
    both_in = snap(overrides={
        "alphabet_soup": body(in_basket_p(dx=-0.04)),
        "milk": body(in_basket_p(dx=0.04, dz=0.05),
                     ext=(0.058, 0.06, 0.145))})
    good = [init, held_soup,
            snap(overrides={"alphabet_soup": body(in_basket_p(dx=-0.04))}),
            milk_held, both_in]
    x(ev("soup_then_milk", good), "PASS", "parity: soup_then_milk order",
      score_min=0.99)

    milk_first = [
        init,
        snap(held="milk", grip=0.4, overrides={"milk": lifted("milk")}),
        snap(overrides={"milk": body(in_basket_p(dx=0.04, dz=0.05),
                                     ext=(0.058, 0.06, 0.145))}),
        snap(held="alphabet_soup", grip=0.4, overrides={
            "milk": body(in_basket_p(dx=0.04, dz=0.05),
                         ext=(0.058, 0.06, 0.145)),
            "alphabet_soup": lifted("alphabet_soup")}),
        both_in,
    ]
    x(ev("soup_then_milk", milk_first), "FAIL",
      "parity: soup_then_milk wrong order", score_max=0.9)

    partial = [init, held_soup,
               snap(overrides={"alphabet_soup": body(in_basket_p(dx=-0.04))})]
    hang_events = [
        {"ev": "call", "seq": 1, "t": 1.0, "tool": "robot.get_observation",
         "node": "grasp.observe"},
        {"ev": "done", "seq": 1, "t": 2.0, "tool": "robot.get_observation",
         "node": "grasp.observe"},
        {"ev": "call", "seq": 2, "t": 3.0, "tool": "curobo.plan",
         "node": "grasp.plan", "kw": {"target": "milk"}},
    ]
    hung = ev("soup_then_milk", partial, hang_events,
              {"ran": True, "timed_out": True, "harness_crash": None})
    x(hung, "TIMEOUT", "parity: soup_then_milk hang", score_max=0.7)
    t.expect(hung.diagnostics.get("hang", {}).get("tool") == "curobo.plan",
             "parity: hang site recovered (curobo.plan)")

    # ---- two_inches_right ----
    def soup_at(dy, dx=0.0, q=IDQ):
        p = SPREAD["alphabet_soup"]
        return snap(overrides={
            "alphabet_soup": body((p[0] + dx, p[1] + dy, p[2]), q=q)})

    x(ev("two_inches_right", [init, held_soup, soup_at(+0.05)]), "PASS",
      "parity: two_inches +y", score_min=0.99)
    x(ev("two_inches_right", [init, held_soup, soup_at(-0.05)]), "PASS",
      "parity: two_inches -y", score_min=0.99)
    x(ev("two_inches_right", [init, held_soup, soup_at(+0.05, q=ROT90X)]),
      "FAIL", "parity: two_inches knocked over", score_max=0.8)
    x(ev("two_inches_right", [init, held_soup, soup_at(+0.12)]), "FAIL",
      "parity: two_inches overshoot")
    x(ev("two_inches_right", [init, soup_at(+0.05)]), "FAIL",
      "parity: two_inches never picked")

    # ---- cream_cheese ----
    cc_held = snap(held="cream_cheese", grip=0.6, overrides={
        "cream_cheese": lifted("cream_cheese", dz=0.10)})
    cc_in = snap(overrides={
        "cream_cheese": body(in_basket_p(), ext=(0.083, 0.043, 0.020))})
    x(ev("cream_cheese", [init, cc_held, cc_in]), "PASS",
      "parity: cream_cheese pick+place", score_min=0.99)
    x(ev("cream_cheese", [init, init]), "FAIL",
      "parity: cream_cheese untouched")

    # ---- move_basket_ranch ----
    b0 = (0.71, -0.23, -0.005)
    b1 = (0.71, +0.10, -0.005)
    start = snap(basket_p=b0)
    ranch_held = snap(basket_p=b1, held="salad_dressing", grip=0.4,
                      overrides={"salad_dressing":
                                 lifted("salad_dressing", dz=0.12)})
    ranch_in = snap(basket_p=b1, overrides={
        "salad_dressing": body(in_basket_p(b1, dz=0.05),
                               ext=(0.045, 0.066, 0.148))})
    x(ev("move_basket_ranch", [start, snap(basket_p=b1), ranch_held,
                               ranch_in]), "PASS",
      "parity: move_basket relocated+packed", score_min=0.99)
    unmoved_in = snap(basket_p=b0, overrides={
        "salad_dressing": body(in_basket_p(b0, dz=0.05),
                               ext=(0.045, 0.066, 0.148))})
    ranch_held_unmoved = snap(basket_p=b0, held="salad_dressing", grip=0.4,
                              overrides={"salad_dressing":
                                         lifted("salad_dressing", dz=0.12)})
    x(ev("move_basket_ranch", [start, ranch_held_unmoved, unmoved_in]),
      "FAIL", "parity: basket never moved", score_max=0.8)

    # ---- stack_butter_on_cream_cheese ----
    cc_p = SPREAD["cream_cheese"]
    butter_on_cc = snap(overrides={
        "cream_cheese": body(cc_p, ext=(0.083, 0.043, 0.020)),
        "butter": body((cc_p[0], cc_p[1], cc_p[2] + 0.019),
                       ext=(0.078, 0.040, 0.018))})
    butter_held = snap(held="butter", grip=0.4,
                       overrides={"butter": lifted("butter", dz=0.10)})
    x(ev("stack_butter_on_cream_cheese", [init, butter_held, butter_on_cc]),
      "PASS", "parity: stack passes", score_min=0.99)
    butter_beside = snap(overrides={
        "butter": body((cc_p[0] + 0.09, cc_p[1], 0.009),
                       ext=(0.078, 0.040, 0.018))})
    x(ev("stack_butter_on_cream_cheese", [init, butter_held, butter_beside]),
      "FAIL", "parity: stack beside fails")

    # ---- all_except_milk ----
    five = [n for n in SPREAD if n != "milk"]
    trace = [init]
    packed5: dict[str, dict] = {}
    for i, n in enumerate(five):
        trace.append(snap(held=n, grip=0.4,
                          overrides={**packed5, n: lifted(n)}))
        packed5[n] = body(in_basket_p(dx=0.05 * (i % 3) - 0.05,
                                      dy=0.05 * (i // 3) - 0.02,
                                      dz=0.03 + 0.02 * (i // 3)))
        trace.append(snap(overrides=dict(packed5)))
    x(ev("all_except_milk", trace), "PASS",
      "parity: all_except_milk 5-in milk-out", score_min=0.99)
    with_milk = snap(overrides={**packed5,
                                "milk": body(in_basket_p(dz=0.08),
                                             ext=(0.058, 0.06, 0.145))})
    x(ev("all_except_milk", trace + [with_milk]), "FAIL",
      "parity: milk packed anyway", score_max=0.85)

    # ---- soup_left_of_milk ----
    milk_p = SPREAD["milk"]

    def soup_beside_milk(dy, dx=0.0):
        return snap(overrides={"alphabet_soup": body(
            (milk_p[0] + dx, milk_p[1] + dy, 0.038))})

    x(ev("soup_left_of_milk", [init, held_soup, soup_beside_milk(-0.12)]),
      "PASS", "parity: left_of viewer-left", score_min=0.99)
    x(ev("soup_left_of_milk", [init, held_soup, soup_beside_milk(+0.12)]),
      "PASS", "parity: left_of robot-left", score_min=0.99)
    x(ev("soup_left_of_milk",
         [init, held_soup, soup_beside_milk(0.0, dx=0.25)]),
      "FAIL", "parity: left_of wrong axis")

    # ---- pudding_basket (variance task-3 scene, v1 probed constants) ----
    T3 = {
        "basket": body((0.400, -0.081, -0.005), ext=BASKET_EXT),
        "chocolate_pudding": body((0.750, 0.030, 0.013), ext=(0.078, 0.055, 0.028)),
        "orange_juice": body((0.700, -0.200, 0.070), ext=(0.079, 0.079, 0.145)),
        "bbq_sauce": body((0.596, 0.260, 0.055), ext=(0.045, 0.065, 0.149)),
        "ketchup": body((0.727, -0.226, 0.072), ext=(0.045, 0.065, 0.149)),
        "salad_dressing": body((0.619, -0.121, 0.073), ext=(0.045, 0.066, 0.148)),
        "alphabet_soup": body((0.481, -0.240, 0.038), ext=(0.085, 0.089, 0.085)),
    }

    def t3_snap(over=None, *, held=None, grip=1.0):
        bodies = dict(T3)
        bodies.update(over or {})
        return {"seq": 0, "t": 0.0, "tool": "x", "node": None,
                "held": held, "grip": grip, "ee": [0.45, 0.0, 0.26],
                "bodies": bodies}

    t3_basket = (0.400, -0.081, -0.005)
    t3_0 = t3_snap()
    t3_held = t3_snap({"chocolate_pudding": body(
        (0.750, 0.030, 0.15), ext=(0.078, 0.055, 0.028), g=True)},
        held="chocolate_pudding", grip=0.4)
    t3_in = t3_snap({"chocolate_pudding": body(
        (t3_basket[0], t3_basket[1], t3_basket[2] + 0.03),
        ext=(0.078, 0.055, 0.028))})
    x(ev("pudding_basket", [t3_0, t3_held, t3_in]), "PASS",
      "parity: pudding_basket", score_min=0.99)
    x(ev("pudding_basket", [t3_0, t3_0]), "FAIL",
      "parity: pudding_basket untouched")

    # ---- oj_swap_scene ----
    SW = {
        "basket": body((0.633, -0.066, -0.005), ext=BASKET_EXT),
        "bbq_sauce": body((0.396, -0.079, 0.055), ext=(0.045, 0.065, 0.149)),
        "butter": body((0.451, 0.063, 0.009), ext=(0.078, 0.04, 0.018)),
        "chocolate_pudding": body((0.750, 0.030, 0.013), ext=(0.078, 0.055, 0.028)),
        "milk": body((0.478, -0.237, 0.070), ext=(0.058, 0.06, 0.145)),
        "orange_juice": body((0.700, -0.200, 0.070), ext=(0.079, 0.079, 0.145)),
        "tomato_sauce": body((0.613, 0.260, 0.038), ext=(0.083, 0.087, 0.087)),
    }

    def sw_snap(over=None, *, held=None, grip=1.0):
        bodies = dict(SW)
        bodies.update(over or {})
        return {"seq": 0, "t": 0.0, "tool": "x", "node": None,
                "held": held, "grip": grip, "ee": [0.45, 0.0, 0.26],
                "bodies": bodies}

    sw_held = sw_snap({"orange_juice": body(
        (0.700, -0.200, 0.19), ext=(0.079, 0.079, 0.145), g=True)},
        held="orange_juice", grip=0.4)
    sw_in = sw_snap({"orange_juice": body(
        (0.633, -0.066, 0.075), ext=(0.079, 0.079, 0.145))})
    x(ev("oj_swap_scene", [sw_snap(), sw_held, sw_in]), "PASS",
      "parity: oj_swap_scene", score_min=0.99)

    # ---- both_boxes ----
    boxes_held1 = snap(held="butter", grip=0.4,
                       overrides={"butter": lifted("butter", dz=0.10)})
    boxes_in1 = snap(overrides={"butter": body(
        in_basket_p(dx=-0.04), ext=(0.078, 0.04, 0.018))})
    boxes_held2 = snap(held="cream_cheese", grip=0.4, overrides={
        "butter": body(in_basket_p(dx=-0.04), ext=(0.078, 0.04, 0.018)),
        "cream_cheese": lifted("cream_cheese", dz=0.10)})
    boxes_both = snap(overrides={
        "butter": body(in_basket_p(dx=-0.04), ext=(0.078, 0.04, 0.018)),
        "cream_cheese": body(in_basket_p(dx=0.04), ext=(0.083, 0.043, 0.02))})
    x(ev("both_boxes", [init, boxes_held1, boxes_in1, boxes_held2,
                        boxes_both]), "PASS",
      "parity: both_boxes exact pair", score_min=0.99)
    boxes_stray = snap(overrides={
        "butter": body(in_basket_p(dx=-0.04), ext=(0.078, 0.04, 0.018)),
        "cream_cheese": body(in_basket_p(dx=0.04), ext=(0.083, 0.043, 0.02)),
        "alphabet_soup": body(in_basket_p(dy=0.05, dz=0.06))})
    x(ev("both_boxes", [init, boxes_held1, boxes_in1, boxes_held2,
                        boxes_stray]), "FAIL",
      "parity: both_boxes stray soup", score_max=0.96)

    # ---- pack_scene_all (graveyard consume) ----
    GY = (50.6, 50.0, 0.01)
    trace = [init]
    consumed: dict[str, dict] = {}
    for _i, n in enumerate(names):
        trace.append(snap(held=n, grip=0.4,
                          overrides={**consumed, n: lifted(n)}))
        consumed[n] = body(GY)
        trace.append(snap(overrides=dict(consumed)))
    x(ev("pack_scene_all", trace), "PASS", "parity: pack_scene consumed",
      score_min=0.99)

    trace5 = [init]
    consumed5: dict[str, dict] = {}
    for _i, n in enumerate(names[:5]):
        trace5.append(snap(held=n, grip=0.4,
                           overrides={**consumed5, n: lifted(n)}))
        consumed5[n] = body(GY)
        trace5.append(snap(overrides=dict(consumed5)))
    x(ev("pack_scene_all", trace5), "FAIL", "parity: pack_scene 5/6",
      score_max=0.9)

    tele = [init, snap(overrides={n: body(GY) for n in names})]
    x(ev("pack_scene_all", tele), "FAIL",
      "parity: pack_scene teleport-without-grasp", score_max=0.1)

    # ---- graph-structure diagnostics ----
    dag = {"nodes": {"a": {}, "b": {}},
           "edges": [["START", "a"], ["a", "b"], ["b", "END"]],
           "subgraphs": {}}
    loop = {"nodes": {"a": {}, "b": {}},
            "edges": [["START", "a"], ["a", "b"], ["b", "a"]],
            "subgraphs": {}}
    cond_loop = {"nodes": {"a": {}, "b": {}},
                 "edges": [["START", "a"], ["a", "b"]],
                 "conditional_edges": {"b": {"again": "a", "done": "END"}},
                 "subgraphs": {}}
    routed_loop = {
        "nodes": {"perceive": {}, "grasp": {}, "transport": {}},
        "edges": [["START", "perceive"]],
        "conditional_edges": {
            "perceive": {"router_field": "exit",
                         "mapping": {"found": "grasp", "none": "done"}},
            "grasp": {"router_field": "exit",
                      "mapping": {"grasped": "transport",
                                  "failed": "abort"}},
            "transport": {"router_field": "exit",
                          "mapping": {"placed": "perceive",
                                      "blocked": "abort"}},
        },
        "subgraphs": {},
    }
    routed_chain = {
        "nodes": {"a": {}, "b": {}},
        "edges": [["START", "a"]],
        "conditional_edges": {
            "a": {"router_field": "exit", "mapping": {"ok": "b"}},
            "b": {"router_field": "exit", "mapping": {"ok": "done"}},
        },
        "subgraphs": {},
    }
    t.expect(not diagnostics.workflow_structure(dag)["any_cycle"],
             "parity: DAG has no cycle")
    t.expect(diagnostics.workflow_structure(loop)["any_cycle"],
             "parity: back-edge loop detected")
    t.expect(diagnostics.workflow_structure(cond_loop)["any_cycle"],
             "parity: conditional loop detected")
    t.expect(diagnostics.workflow_structure(routed_loop)["any_cycle"],
             "parity: router_field/mapping loop detected")
    t.expect(not diagnostics.workflow_structure(routed_chain)["any_cycle"],
             "parity: routed chain has no cycle")
