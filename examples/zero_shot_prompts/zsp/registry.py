"""Case registry: aggregation + selection."""

from __future__ import annotations

from zsp.cases import ALL_CASES
from zsp.schema import Case

CASES: dict[str, Case] = {c.id: c for c in ALL_CASES}


def select(
    *,
    ids: list[str] | None = None,
    categories: list[str] | None = None,
    tiers: list[str] | None = None,
    tags: list[str] | None = None,
    templates: list[str] | None = None,
) -> list[Case]:
    """Filter cases; every given criterion must match (intersection)."""
    out = []
    for c in ALL_CASES:
        if ids is not None and c.id not in ids:
            continue
        if categories is not None and c.category not in categories:
            continue
        if tiers is not None and c.tier not in tiers:
            continue
        if tags is not None and not (set(tags) & set(c.tags)):
            continue
        if templates is not None and c.template not in templates:
            continue
        out.append(c)
    return out
