"""Synthetic state.jsonl record builders for selftests.

Mirrors the recorder's line format exactly (schema 2 unless asked for a
v1 record) so the DSL is exercised on the same shapes real runs produce.
Geometry defaults echo the probed ``libero_object_all_variance/0``
seed-0 layout (see ``layouts/``).
"""

from __future__ import annotations

from typing import Any

# Probed seed-0 layout (layouts/libero_object_all_variance__task0__seed0.json)
BASKET_P = (0.75, 0.03, 0.08)
BASKET_EXT = (0.176, 0.183, 0.165)
SPREAD: dict[str, tuple[float, float, float]] = {
    "alphabet_soup": (0.61, 0.25, 0.038),
    "butter": (0.45, 0.10, 0.009),
    "cream_cheese": (0.52, -0.06, 0.010),
    "milk": (0.47, -0.22, 0.081),
    "salad_dressing": (0.68, 0.20, 0.033),
    "tomato_sauce": (0.66, -0.14, 0.035),
}
EXT: dict[str, tuple[float, float, float]] = {
    "alphabet_soup": (0.085, 0.089, 0.085),
    "butter": (0.078, 0.040, 0.018),
    "cream_cheese": (0.083, 0.043, 0.020),
    "milk": (0.065, 0.065, 0.162),
    "salad_dressing": (0.065, 0.065, 0.065),
    "tomato_sauce": (0.070, 0.070, 0.070),
}

UP_Q = (1.0, 0.0, 0.0, 0.0)
ROT90X_Q = (0.7071, 0.7071, 0.0, 0.0)   # knocked onto its side


def body(
    p: tuple[float, float, float],
    *,
    q: tuple[float, ...] = UP_Q,
    ext: tuple[float, float, float] = (0.08, 0.08, 0.08),
    v: float = 0.0,
    g: bool = False,
    c: list[str] | None = None,
    j: dict[str, float] | None = None,
    cav: list[list[float]] | None = None,
) -> dict[str, Any]:
    lo = [p[i] - ext[i] / 2 for i in range(3)]
    hi = [p[i] + ext[i] / 2 for i in range(3)]
    out: dict[str, Any] = {
        "p": list(p), "q": list(q), "lo": lo, "hi": hi,
        "v": v, "g": g,
    }
    if c:
        out["c"] = list(c)
    if j:
        out["j"] = dict(j)
    if cav:
        out["cav"] = cav
    return out


def basket(p: tuple[float, float, float] = BASKET_P) -> dict[str, Any]:
    return body(p, ext=BASKET_EXT)


def scene_bodies(
    overrides: dict[str, dict[str, Any]] | None = None,
    *,
    objects: tuple[str, ...] = tuple(SPREAD),
    basket_p: tuple[float, float, float] = BASKET_P,
) -> dict[str, dict[str, Any]]:
    """The probed spread layout, with per-body overrides/replacements."""
    out = {
        name: body(SPREAD[name], ext=EXT[name]) for name in objects
    }
    out["basket"] = basket(basket_p)
    for name, b in (overrides or {}).items():
        out[name] = b
    return out


def snap(
    bodies: dict[str, dict[str, Any]],
    *,
    seq: int = 0,
    held: str | None = None,
    grip: float = 1.0,
    ee: tuple[float, float, float] = (0.4, 0.0, 0.3),
    schema: int | None = 2,
    tool: str = "tool.call",
    joints: dict[str, float] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "seq": seq, "t": float(seq), "tool": tool, "node": "n",
        "held": held, "grip": grip, "ee": list(ee),
        "bodies": bodies,
    }
    if schema is not None:
        rec["schema"] = schema
    if joints:
        rec["joints"] = dict(joints)
    return rec


def in_basket_body(
    name: str,
    *,
    basket_p: tuple[float, float, float] = BASKET_P,
    dz: float = 0.0,
    v: float = 0.0,
    contact: bool = True,
) -> dict[str, Any]:
    """A body resting inside the basket (contacting it, settled)."""
    p = (basket_p[0], basket_p[1], basket_p[2] + dz)
    return body(p, ext=EXT.get(name, (0.08, 0.08, 0.08)), v=v,
                c=["basket"] if contact else None)


def lifted_body(
    name: str,
    *,
    from_p: tuple[float, float, float] | None = None,
    dz: float = 0.12,
) -> dict[str, Any]:
    p0 = from_p or SPREAD[name]
    return body((p0[0], p0[1], p0[2] + dz), ext=EXT.get(name, (0.08,) * 3),
                g=True, v=0.2, c=["gripper0_finger1"])


class FakeCase:
    """Minimal case stand-in for direct DSL-level tests."""

    def __init__(self, criterion: dict[str, Any], **kw: Any) -> None:
        self.id = kw.pop("id", "synthetic")
        self.criterion = criterion
        self.success_source = kw.pop("success_source", "trace")
        self.env_consumes_packed = kw.pop("env_consumes_packed", False)
        self.tags = kw.pop("tags", ())
        self.expected = kw.pop("expected", "")
        for k, v in kw.items():
            setattr(self, k, v)
