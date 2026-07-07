"""Full template cross-product for external harnesses (RL task pools).

NOT part of the curated benchmark registry — exported only via
``run_all.py --export-manifest out.jsonl --expand full``. Generates
every valid (template x object x scene) combination from the checked-in
probed layouts, so a training harness gets hundreds of distinct
(prompt, scene, criterion) triples with the same reward-oracle scoring
path (``run_prompt.py --graph`` / ``zsp.evaluate.evaluate_seed``)."""

from __future__ import annotations

from zsp.cases._shared import pick_checks, released_on_table
from zsp.schema import Case
from zsp.selftest.layouts import Layout

_VARIANCE_TASKS = range(10)
_SUITE = "libero_object_all_variance"


def _movables(task_id: int) -> list[str]:
    key = f"{_SUITE}__task{task_id}__seed0"
    if not Layout.exists(key):
        return []
    lay = Layout.load(key)
    return sorted(n for n in lay.body_names() if n != "basket")


def full_expansion() -> list[Case]:
    cases: list[Case] = []
    for task_id in _VARIANCE_TASKS:
        movables = _movables(task_id)
        if not movables:
            continue
        sim_task = f"{_SUITE}/{task_id}"
        # pick — every object in every scene.
        for obj in movables:
            spaced = obj.replace("_", " ")
            cases.append(Case(
                id=f"xpick__t{task_id}_{obj}",
                prompt=f"Pick up the {spaced} and place it in the basket.",
                category="grounding", tier="T1",
                tags=("expansion",), template="xpick",
                sim_task=sim_task, seeds=(0, 1, 2),
                intent=f"expansion: pick {obj} on {sim_task}",
                criterion={"checks": pick_checks(obj)},
            ))
        # lateral relation — adjacent (alphabetical) object pairs.
        for a, b in zip(movables, movables[1:], strict=False):
            cases.append(Case(
                id=f"xlateral__t{task_id}_{a}_{b}",
                prompt=(f"Place the {a.replace('_', ' ')} next to the "
                        f"{b.replace('_', ' ')}."),
                category="spatial", tier="T1",
                tags=("expansion",), template="xlateral",
                sim_task=sim_task, seeds=(0, 1, 2),
                intent=f"expansion: {a} beside {b} on {sim_task}",
                criterion={"checks": [
                    {"name": "picked", "weight": 0.3,
                     "expr": {"picked_up": {"target": a}}},
                    {"name": "beside", "weight": 0.5,
                     "expr": {"offset_along": {"target": a, "reference": b,
                                               "axis": "y", "lo": 0.06,
                                               "hi": 0.30,
                                               "perp_max": 0.15}}},
                    {"name": "released_on_table", "weight": 0.2,
                     "expr": released_on_table(a)},
                ]},
            ))
        # ordered pair packing — first three (alphabetical) pairs.
        for a, b in list(zip(movables, movables[1:], strict=False))[:3]:
            cases.append(Case(
                id=f"xthen__t{task_id}_{a}_{b}",
                prompt=(f"Pick up the {a.replace('_', ' ')} and place it "
                        f"in the basket, then do the same with the "
                        f"{b.replace('_', ' ')}."),
                category="temporal", tier="T1",
                tags=("expansion",), template="xthen",
                sim_task=sim_task, seeds=(0, 1, 2),
                intent=f"expansion: {a} then {b} on {sim_task}",
                criterion={"checks": [
                    {"name": "a_in", "weight": 0.3,
                     "expr": {"in_container": {"target": a,
                                               "container": "basket"}}},
                    {"name": "b_in", "weight": 0.3,
                     "expr": {"in_container": {"target": b,
                                               "container": "basket"}}},
                    {"name": "order", "weight": 0.4,
                     "expr": {"before": {
                         "first": {"enters_container": {"target": a}},
                         "then": {"enters_container": {"target": b}}}}},
                ]},
            ))
    return cases
