"""Ground-truth state recording for one graph execution (trace schema 2).

``StateRecorder`` shadows ``conn.tool_registry.invoke`` (an instance
attribute assignment — every script/tool node call in the runtime goes
through this single dispatch point, see ``ExecutionContext.tool`` in
``gap/runtime/context.py``) and, around every tool call, appends:

- ``events.jsonl`` — one line per call/return/error, flushed
  immediately. Because the "call" line is written BEFORE dispatch, a
  hung tool leaves an unmatched "call" as the last line: the hang site
  survives a process kill and is recovered post-mortem.
- ``state.jsonl`` — one compact ground-truth snapshot per completed
  call (plus ``__initial__`` / ``__final__`` bookends), built from
  ``conn.world_snapshot()`` (a ``gap.runtime.verify.World`` in the
  robot-base frame). Criteria evaluate these dicts, so a killed run is
  still scoreable from whatever was flushed.

Schema 2 extends the v1 snapshot with (all keys omitted when empty, so
grocery-scene traces barely grow):

- per-body ``c``: contact body names (already computed by the world
  adapter for every snapshot; v1 simply never serialized them);
- per-body ``j``: ``{joint_name: qpos}`` for non-robot hinge/slide
  joints owned by that body (articulated fixtures: drawers, stove
  knobs, microwave doors) — top-level ``joints`` for any joint whose
  name matches no body;
- per-body ``cav``: interior-cavity AABB ``[lo, hi]`` in the robot-base
  frame, from a ``<body>__cavity`` MuJoCo site when one exists;
- ``schema: 2`` on every line, and a sidecar ``trace_meta.json``
  (frame, world-frame base pose, body/joint inventory, extras status).

Joints and cavities need the *raw* MuJoCo handles
(:func:`gap.connector.world_adapter.raw_mujoco_handles`); when those are
unreachable the recorder degrades to v1-shaped bodies (still ``schema:
2``) and reports why in ``trace_meta.json["extras_error"]`` — it never
crashes the run.

Snapshots are taken on the calling thread, immediately after the tool
returns — never mid-physics-step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2


def _f(v: Any) -> list[float]:
    return [round(float(x), 5) for x in np.asarray(v).ravel().tolist()]


def _quat_wxyz_to_rotmat(q: Any) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q).ravel()[:4])
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _kw_summary(kw: dict[str, Any]) -> dict[str, Any]:
    """Small, JSON-safe view of tool kwargs (no arrays, no blobs)."""
    out: dict[str, Any] = {}
    for k, v in kw.items():
        if k == "ctx":
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v if not isinstance(v, str) else v[:120]
        elif isinstance(v, np.ndarray):
            out[k] = f"<array {v.shape}>"
        elif isinstance(v, (list, tuple)):
            if len(v) <= 8 and all(isinstance(x, (int, float)) for x in v):
                out[k] = [round(float(x), 4) for x in v]
            else:
                out[k] = f"<list len={len(v)}>"
        elif isinstance(v, dict):
            out[k] = f"<dict keys={sorted(v)[:8]}>"
        else:
            out[k] = f"<{type(v).__name__}>"
    return out


def _result_summary(res: Any, depth: int = 0) -> Any:
    """Numeric-leaf summary of a tool result (capped size).

    Keeps scalars, short numeric lists, and status-ish strings so
    diagnostics like 'what bounding box did geometry return' can be
    answered from the event log without re-running anything.
    """
    if depth > 3:
        return "<depth>"
    if isinstance(res, (int, float, bool)) or res is None:
        return res
    if isinstance(res, str):
        return res[:160]
    if isinstance(res, np.ndarray):
        if res.size <= 12:
            return _f(res)
        return f"<array {res.shape}>"
    if isinstance(res, (list, tuple)):
        if len(res) <= 12 and all(
            isinstance(x, (int, float, bool)) for x in res
        ):
            return [round(float(x), 5) for x in res]
        return f"<list len={len(res)}>"
    if isinstance(res, dict):
        out = {}
        for i, (k, v) in enumerate(res.items()):
            if i >= 24:
                out["..."] = f"<{len(res) - 24} more>"
                break
            out[str(k)] = _result_summary(v, depth + 1)
        return out
    return f"<{type(res).__name__}>"


class _RawExtras:
    """Joints + cavities + base pose from the raw MuJoCo handles.

    Static tables (joint names/addresses, cavity sites, base body id) are
    built once at install time; per-snapshot reads touch only ``qpos`` and
    the site buffers. Any failure downgrades to "no extras" with the
    reason preserved — the trace stays v1-shaped rather than the run
    dying.
    """

    _ROBOT_JOINT_PREFIXES = ("robot", "gripper", "mount", "finger")

    def __init__(self, env: Any) -> None:
        self.error: str | None = None
        self.joints: list[tuple[str, int]] = []       # (joint name, qpos addr)
        self.cavities: list[tuple[int, str]] = []     # (site id, owner hint)
        self._model: Any = None
        self._data: Any = None
        self._base_bid: int | None = None
        try:
            import mujoco

            from gap.connector.world_adapter import raw_mujoco_handles

            handles = raw_mujoco_handles(env)
            if handles is None:
                self.error = "no mujoco sim reachable from env"
                return
            model, data = handles
            # Raw-handle sanity: the name APIs are the part robosuite's
            # wrappers can't serve. Probe once; degrade if incompatible.
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0)
            self._model, self._data = model, data

            for jid in range(int(model.njnt)):
                jtype = int(model.jnt_type[jid])
                if jtype not in (
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                ):
                    continue
                name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    or f"joint_{jid}"
                )
                if name.startswith(self._ROBOT_JOINT_PREFIXES):
                    continue
                self.joints.append((name, int(model.jnt_qposadr[jid])))

            for sid in range(int(model.nsite)):
                sname = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
                    or ""
                )
                if sname.endswith("__cavity"):
                    self.cavities.append((sid, sname[: -len("__cavity")]))

            for bname in ("robot0_base", "robot0_link0"):
                bid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, bname
                )
                if bid >= 0:
                    self._base_bid = int(bid)
                    break
        except Exception as exc:  # never break recording over extras
            self.error = f"{type(exc).__name__}: {exc}"[:300]
            self._model = self._data = None
            self.joints = []
            self.cavities = []

    @property
    def available(self) -> bool:
        return self._data is not None

    def _base_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """World -> robot-base ``(t, R^T)``; identity when no base body."""
        if self._base_bid is None:
            return np.zeros(3), np.eye(3)
        t = np.asarray(self._data.xpos[self._base_bid], dtype=np.float64)
        Rt = _quat_wxyz_to_rotmat(self._data.xquat[self._base_bid]).T
        return t, Rt

    def base_pose_world(self) -> dict[str, list[float]] | None:
        if not self.available or self._base_bid is None:
            return None
        return {
            "p": _f(self._data.xpos[self._base_bid]),
            "q_wxyz": _f(self._data.xquat[self._base_bid]),
        }

    def read_joints(self) -> dict[str, float]:
        if not self.available or not self.joints:
            return {}
        qpos = self._data.qpos
        return {
            name: round(float(qpos[addr]), 5) for name, addr in self.joints
        }

    def read_cavities(self) -> dict[str, list[list[float]]]:
        """Owner-hint -> cavity AABB ``[lo, hi]`` in the robot-base frame."""
        if not self.available or not self.cavities:
            return {}
        base_t, base_Rt = self._base_transform()
        out: dict[str, list[list[float]]] = {}
        for sid, owner in self.cavities:
            spos = np.asarray(self._data.site_xpos[sid], dtype=np.float64)
            smat = np.asarray(
                self._data.site_xmat[sid], dtype=np.float64
            ).reshape(3, 3)
            ssize = np.asarray(self._model.site_size[sid], dtype=np.float64)
            half = np.abs(smat) @ ssize
            corners = np.array([
                spos + half * np.array(sgn)
                for sgn in (
                    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                )
            ])
            corners = (base_Rt @ (corners - base_t).T).T
            out[owner] = [_f(corners.min(axis=0)), _f(corners.max(axis=0))]
        return out


def _attach_by_prefix(
    keys: list[str], body_names: list[str]
) -> dict[str, str | None]:
    """Map each key to the longest body name that prefixes it, else None.

    LIBERO articulated fixtures name their joints/sites after the owning
    body (``wooden_cabinet_1_top_...`` on body ``wooden_cabinet_1``).
    """
    by_len = sorted(body_names, key=len, reverse=True)
    out: dict[str, str | None] = {}
    for k in keys:
        out[k] = next((b for b in by_len if k == b or k.startswith(b)), None)
    return out


class StateRecorder:
    def __init__(
        self,
        conn: Any,
        out_dir: str | Path,
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.conn = conn
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # "w", not "a": a seed dir records exactly one run attempt — a
        # resume after a mid-run kill must not prepend the stale trace.
        self._events = open(self.out_dir / "events.jsonl", "w")
        self._states = open(self.out_dir / "state.jsonl", "w")
        self._t0 = time.monotonic()
        self._seq = 0
        self._installed = False
        self._orig: Any = None
        self._meta = dict(meta or {})
        self._extras: _RawExtras | None = None
        self._joint_owner: dict[str, str | None] = {}
        self._cavity_owner: dict[str, str | None] = {}

    # -- event / state emission ----------------------------------------

    def _emit(self, fh, payload: dict[str, Any]) -> None:
        fh.write(json.dumps(payload, default=str) + "\n")
        fh.flush()

    def _now(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def snapshot(self, tag: str, node: str | None = None) -> None:
        """Append one ground-truth state line (best-effort)."""
        try:
            w = self.conn.world_snapshot()
        except Exception as exc:
            self._emit(self._states, {
                "schema": SCHEMA_VERSION,
                "seq": self._seq, "t": self._now(), "tool": tag,
                "node": node, "snapshot_error": str(exc)[:300],
            })
            return
        try:
            held = w.held_body()
            held_name = held.name if held is not None else None
        except Exception:
            held_name = None
        rv = w.robot_view
        bodies: dict[str, dict[str, Any]] = {}
        for name in w.body_names():
            b = w.body(name)
            rec: dict[str, Any] = {
                "p": _f(b.position),
                "q": _f(b.quaternion_wxyz),
                "lo": _f(b.aabb_lower),
                "hi": _f(b.aabb_upper),
                "v": round(float(np.linalg.norm(b.linear_velocity)), 4),
                "g": bool(b.is_grasped()),
            }
            contacts = getattr(b, "contacts", None)
            if contacts:
                rec["c"] = sorted(contacts)
            bodies[name] = rec

        extras = self._extras
        stray_joints: dict[str, float] = {}
        if extras is not None and extras.available:
            joints = extras.read_joints()
            if joints and not self._joint_owner:
                self._joint_owner = _attach_by_prefix(
                    list(joints), list(bodies)
                )
            for jname, q in joints.items():
                owner = self._joint_owner.get(jname)
                if owner in bodies:
                    bodies[owner].setdefault("j", {})[jname] = q
                else:
                    stray_joints[jname] = q
            cavities = extras.read_cavities()
            if cavities and not self._cavity_owner:
                self._cavity_owner = _attach_by_prefix(
                    list(cavities), list(bodies)
                )
            for hint, aabb in cavities.items():
                owner = self._cavity_owner.get(hint)
                if owner in bodies:
                    bodies[owner]["cav"] = aabb

        line: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "seq": self._seq,
            "t": self._now(),
            "tool": tag,
            "node": node,
            "held": held_name,
            "grip": None if rv is None else round(
                float(rv.gripper_open_fraction), 4),
            "ee": None if rv is None else _f(rv.ee_position),
            "bodies": bodies,
        }
        if stray_joints:
            line["joints"] = stray_joints
        self._emit(self._states, line)

    def write_meta(self) -> None:
        """Write/refresh ``trace_meta.json`` (idempotent, best-effort)."""
        extras = self._extras
        body_names: list[str] = []
        try:
            body_names = sorted(self.conn.world_snapshot().body_names())
        except Exception:
            pass
        payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "frame": "robot_base",
            "body_names": body_names,
            "joint_names": (
                [n for n, _ in extras.joints] if extras is not None else []
            ),
            "cavity_sites": (
                [hint for _, hint in extras.cavities]
                if extras is not None else []
            ),
            "base_pose_world": (
                extras.base_pose_world() if extras is not None else None
            ),
            "extras_error": extras.error if extras is not None else "not built",
            **self._meta,
        }
        try:
            (self.out_dir / "trace_meta.json").write_text(
                json.dumps(payload, indent=2, default=str) + "\n"
            )
        except Exception:
            pass

    # -- invoke wrapping -------------------------------------------------

    def install(self) -> None:
        reg = self.conn.tool_registry
        orig = reg.invoke
        assert callable(orig), "tool_registry.invoke is not callable"
        # Guard against double-install (an instance-attr shadow already in
        # place would make uninstall restore the wrong thing).
        assert "invoke" not in vars(reg), (
            "tool_registry.invoke already shadowed — another recorder "
            "installed?"
        )
        self._orig = orig
        self._extras = _RawExtras(getattr(self.conn, "env", None))
        self.write_meta()
        recorder = self

        def wrapped(*args: Any, **kw: Any) -> Any:
            name = str(args[0]) if args else str(kw.get("name", "?"))
            ctx = kw.get("ctx")
            node = getattr(ctx, "_node_id", None)
            recorder._seq += 1
            seq = recorder._seq
            recorder._emit(recorder._events, {
                "ev": "call", "seq": seq, "t": recorder._now(),
                "tool": name, "node": node, "kw": _kw_summary(kw),
            })
            try:
                res = recorder._orig(*args, **kw)
            except BaseException as exc:
                recorder._emit(recorder._events, {
                    "ev": "error", "seq": seq, "t": recorder._now(),
                    "tool": name, "node": node,
                    "err": f"{type(exc).__name__}: {exc}"[:400],
                })
                recorder.snapshot(name, node)
                raise
            recorder._emit(recorder._events, {
                "ev": "done", "seq": seq, "t": recorder._now(),
                "tool": name, "node": node,
                "summary": _result_summary(res),
            })
            recorder.snapshot(name, node)
            return res

        reg.invoke = wrapped
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            try:
                del self.conn.tool_registry.invoke  # restore class method
            except AttributeError:
                pass
            self._installed = False
        for fh in (self._events, self._states):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass


# -- loading -------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn final line from a killed process
    return out


def hang_site(events: list[dict]) -> dict | None:
    """The last 'call' event with no matching 'done'/'error' — where a
    killed run was stuck."""
    open_calls: dict[int, dict] = {}
    for ev in events:
        if ev.get("ev") == "call":
            open_calls[int(ev.get("seq", -1))] = ev
        elif ev.get("ev") in ("done", "error"):
            open_calls.pop(int(ev.get("seq", -1)), None)
    if not open_calls:
        return None
    return open_calls[max(open_calls)]
