"""Probed-layout access + synthetic trace building for selftests.

``Layout.load("libero_object_all_variance__task0__seed0")`` reads a
checked-in ``layouts/*.json`` probe dump; ``TraceBuilder`` turns it into
state.jsonl-shaped snapshot sequences (~one line per action), so a
case's pass trace reads like the plan it verifies::

    tb = TraceBuilder(Layout.load(case.layout_key()))
    tb.pick("alphabet_soup").place_in("alphabet_soup")
    ev = evaluate_seed(case, tb.states, [], EXEC_OK)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LAYOUTS_DIR = Path(__file__).resolve().parent.parent.parent / "layouts"

EXEC_OK = {"ran": True, "success": True, "exit_status": "success",
           "harness_crash": None}


class Layout:
    _cache: dict[str, Layout] = {}

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.bodies: dict[str, dict[str, Any]] = data["bodies"]
        self.joints: dict[str, dict[str, Any]] = data.get("joints", {})

    @classmethod
    def load(cls, key: str) -> Layout:
        if key not in cls._cache:
            path = LAYOUTS_DIR / f"{key}.json"
            cls._cache[key] = cls(json.loads(path.read_text()))
        return cls._cache[key]

    @classmethod
    def exists(cls, key: str) -> bool:
        return (LAYOUTS_DIR / f"{key}.json").is_file()

    def body_names(self) -> set[str]:
        return set(self.bodies)

    def p(self, name: str) -> tuple[float, float, float]:
        return tuple(self.bodies[name]["p"])

    def ext(self, name: str) -> tuple[float, float, float]:
        return tuple(self.bodies[name]["ext"])

    def record_body(self, name: str, *, p=None, q=None, v=0.0, g=False,
                    c=None, j=None) -> dict[str, Any]:
        src = self.bodies[name]
        p = list(p if p is not None else src["p"])
        q = list(q if q is not None else src["q"])
        ext = src["ext"]
        rec: dict[str, Any] = {
            "p": p, "q": q,
            "lo": [p[i] - ext[i] / 2 for i in range(3)],
            "hi": [p[i] + ext[i] / 2 for i in range(3)],
            "v": v, "g": g,
        }
        if src.get("cavity"):
            cav = src["cavity"]
            # Re-anchor the probed cavity to the body's current position.
            dp = [p[i] - src["p"][i] for i in range(3)]
            rec["cav"] = [
                [cav[0][i] + dp[i] for i in range(3)],
                [cav[1][i] + dp[i] for i in range(3)],
            ]
        joints = dict(j or {})
        for jname, jinfo in self.joints.items():
            if jname.startswith(name) and jname not in joints:
                joints[jname] = jinfo["qpos"]
        if joints:
            rec["j"] = joints
        if c:
            rec["c"] = list(c)
        return rec


class TraceBuilder:
    """Snapshot-sequence builder over a probed layout.

    Tracks each body's current pose/state; every mutating call appends
    one snapshot. All geometry (basket position, extents) comes from the
    layout — nothing hand-typed.
    """

    def __init__(self, layout: Layout, *, container: str = "basket") -> None:
        self.layout = layout
        self.container = container
        self._seq = 0
        self._cur: dict[str, dict[str, Any]] = {
            name: {"p": None, "q": None, "v": 0.0, "g": False, "c": None,
                   "j": None}
            for name in layout.bodies
        }
        self.states: list[dict[str, Any]] = []
        self._held: str | None = None
        self._grip = 1.0
        self.snapshot()

    # -- snapshot assembly -------------------------------------------------

    def snapshot(self, tool: str = "tool.call") -> TraceBuilder:
        bodies = {
            name: self.layout.record_body(name, **kw)
            for name, kw in self._cur.items()
        }
        self.states.append({
            "schema": 2, "seq": self._seq, "t": float(self._seq),
            "tool": tool, "node": "n",
            "held": self._held, "grip": self._grip,
            "ee": [0.45, 0.0, 0.26],
            "bodies": bodies,
        })
        self._seq += 1
        return self

    # -- state mutations (each appends a snapshot) ---------------------------

    def pick(self, name: str, *, dz: float = 0.12) -> TraceBuilder:
        p = self._pos(name)
        self._cur[name].update(
            p=(p[0], p[1], p[2] + dz), g=True, v=0.2,
            c=["gripper0_finger1"],
        )
        self._held, self._grip = name, 0.4
        return self.snapshot("robot.close_gripper")

    def release(self) -> TraceBuilder:
        self._held, self._grip = None, 1.0
        return self.snapshot("robot.open_gripper")

    def place_in(self, name: str, container: str | None = None, *,
                 dx: float = 0.0, dy: float = 0.0,
                 dz: float = 0.03) -> TraceBuilder:
        cont = container or self.container
        cp = self._pos(cont)
        self._cur[name].update(
            p=(cp[0] + dx, cp[1] + dy, cp[2] + dz),
            g=False, v=0.0, c=[cont],
        )
        if self._held == name:
            self._held, self._grip = None, 1.0
        return self.snapshot("robot.open_gripper")

    def place_at(self, name: str, *, p=None, dxy=(0.0, 0.0), z=None,
                 q=None, contacts=None) -> TraceBuilder:
        if p is None:
            p0 = self.layout.p(name)
            p = (p0[0] + dxy[0], p0[1] + dxy[1],
                 p0[2] if z is None else z)
        self._cur[name].update(p=tuple(p), v=0.0, g=False,
                               c=list(contacts) if contacts else None)
        if q is not None:
            self._cur[name]["q"] = tuple(q)
        if self._held == name:
            self._held, self._grip = None, 1.0
        return self.snapshot("robot.open_gripper")

    def place_on(self, name: str, support: str,
                 *, gap: float = 0.001) -> TraceBuilder:
        sp = self._pos(support)
        s_ext = self.layout.ext(support)
        n_ext = self.layout.ext(name)
        z = sp[2] + s_ext[2] / 2 + n_ext[2] / 2 + gap
        return self.place_at(name, p=(sp[0], sp[1], z), contacts=[support])

    def move(self, name: str, *, to=None, dxy=(0.0, 0.0)) -> TraceBuilder:
        p = self._pos(name)
        p = tuple(to) if to is not None else (p[0] + dxy[0], p[1] + dxy[1], p[2])
        self._cur[name].update(p=p, v=0.0)
        return self.snapshot()

    def set_joint(self, body: str, joint: str, q: float) -> TraceBuilder:
        j = dict(self._cur[body].get("j") or {})
        j[joint] = q
        self._cur[body]["j"] = j
        return self.snapshot()

    def consume(self, name: str) -> TraceBuilder:
        """Env graveyard teleport (packing suites)."""
        self._cur[name].update(p=(50.6, 50.0, 0.01), v=0.0, g=False, c=None)
        if self._held == name:
            self._held, self._grip = None, 1.0
        return self.snapshot("sim.step")

    def hold_ee(self, ee) -> TraceBuilder:
        self.states[-1]["ee"] = list(ee)
        return self

    def noop(self) -> TraceBuilder:
        return self.snapshot("robot.get_observation")

    # -- helpers -------------------------------------------------------------

    def _pos(self, name: str) -> tuple[float, float, float]:
        cur = self._cur[name]["p"]
        return tuple(cur) if cur is not None else self.layout.p(name)


def noop_trace(layout: Layout) -> list[dict[str, Any]]:
    """Two identical do-nothing snapshots (the vacuous-pass guard input)."""
    tb = TraceBuilder(layout)
    tb.noop()
    return tb.states
