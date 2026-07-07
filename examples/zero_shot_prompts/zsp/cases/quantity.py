"""Quantity: cardinality constraints."""

from __future__ import annotations

from zsp.schema import Case

CASES: list[Case] = [
    Case(
        id="quant_exactly_two",
        prompt=("Put exactly two objects of your choice into the basket. "
                "Not one, not three — exactly two."),
        category="quantity", tier="T1", tags=("cardinality",),
        intent=(
            "cardinality control with free identity choice: exactly two "
            "movables settled in the basket at the end. Under-packing, "
            "over-packing, and doing nothing all fail; any pair passes."
        ),
        criterion={"checks": [
            {"name": "exactly_two_in_basket", "weight": 1.0,
             "expr": {"count_in": {"container": "basket",
                                   "targets": "movables",
                                   "count": 2}}},
            {"name": "which_and_whether_grasped", "weight": 0.0,
             "gating": False,
             "expr": {"pack_report": {"targets": "movables",
                                      "mode": "final"}}},
        ]},
        timeout_min=75,
    ),
]
