"""Static validation of every registered case against its probed layout."""

from __future__ import annotations

from zsp.dsl import validate_criterion
from zsp.registry import ALL_CASES
from zsp.selftest.layouts import Layout


def run(t) -> None:
    ids = [c.id for c in ALL_CASES]
    t.expect(len(ids) == len(set(ids)), "schema: case ids unique")

    for case in ALL_CASES:
        try:
            case.validate()
            t.expect(True, f"schema: {case.id} fields valid")
        except ValueError as exc:
            t.expect(False, f"schema: {case.id} fields: {exc}")
            continue

        if case.success_source == "gen":
            known = None   # no layout needed; criterion is gen-level
        else:
            key = case.layout_key()
            if not Layout.exists(key):
                t.expect(False, f"schema: {case.id} missing layout {key} "
                                "(run probes/dump_scene.py)")
                continue
            known = set()
            for seed in case.seeds:
                skey = case.layout_key(seed)
                if Layout.exists(skey):
                    known |= Layout.load(skey).body_names()
            t.expect(bool(known), f"schema: {case.id} layout has bodies")

        try:
            validate_criterion(dict(case.criterion), known_bodies=known)
            t.expect(True, f"schema: {case.id} criterion valid")
        except ValueError as exc:
            t.expect(False, f"schema: {case.id} criterion: {exc}")

        needs_joints = any(
            "joint_open" in str(c.get("expr", {}))
            for c in case.criterion.get("checks", [])
        )
        t.expect(not needs_joints or case.requires_schema >= 2,
                 f"schema: {case.id} uses joint_open but requires_schema<2")

        has_weight = any(
            float(c.get("weight", 0) or 0) > 0 and c.get("gating", True)
            for c in case.criterion.get("checks", [])
        )
        t.expect(has_weight or case.success_source in ("env", "gen"),
                 f"schema: {case.id} has no weighted gating check")
