"""Generate the composed multi-block reference graphs (ordered + sorting).

These reuse ``compose.py`` — each is a sequence of robust pick-place blocks.
Run after editing to regenerate:

    uv run python examples/zero_shot_prompts/reference_graphs/build_hand.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compose import compose  # noqa: E402

GRAPHS = _HERE / "graphs"

# desc shorthands (match descriptions.py where possible)
SOUP = "blue and yellow alphabet soup can"
MILK = "milk carton"
TOMATO = "tomato sauce can"
BUTTER = "small butter box"
CREAM = "small cream cheese box"
BASKET = "wicker basket"
TRAY = "wooden tray"

COMPOSED: dict[str, dict] = {
    # NOTE: drop-spread offsets were tried and REVERTED — the multi-into-basket
    # failures were not stacking (the transitive-contact criterion fix handles
    # a later object resting on a packed one) but LATER-BLOCK GRASP misses
    # (e.g. the flat cream_cheese box as block 2). Center drops are cleaner.
    "soup_then_milk": {
        "steps": [(SOUP, BASKET), (MILK, BASKET)],
        "description": "Pick the alphabet soup into the basket, then the milk.",
    },
    "then_reverse": {
        "steps": [(MILK, BASKET), (SOUP, BASKET)],
        "description": "Pick the milk into the basket, then the alphabet soup.",
    },
    "order_three_strict": {
        "steps": [(BUTTER, BASKET), (MILK, BASKET), (TOMATO, BASKET)],
        "description": "Pack butter, then milk, then tomato sauce, in that order.",
    },
    "both_boxes": {
        "steps": [(BUTTER, BASKET), (CREAM, BASKET)],
        "description": "Put the two boxes (butter, cream cheese) in the basket.",
    },
    "sort_cans_boxes": {
        "steps": [(SOUP, BASKET), (TOMATO, BASKET), (BUTTER, TRAY),
                  (CREAM, TRAY)],
        "description": "Cans (soup, tomato) into the basket; boxes (butter, "
                       "cream cheese) onto the wooden tray.",
    },
}


def main() -> None:
    for name, spec in COMPOSED.items():
        out = compose(spec["steps"], GRAPHS / name, name=name,
                      description=spec["description"],
                      drop_offsets=spec.get("offsets"))
        print(f"composed {name}: {len(spec['steps'])} blocks -> "
              f"{out.relative_to(_HERE.parents[3])}")


if __name__ == "__main__":
    main()
