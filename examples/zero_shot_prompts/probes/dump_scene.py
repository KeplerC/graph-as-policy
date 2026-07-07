#!/usr/bin/env python
"""Probe a sim scene into a checked-in layout JSON.

Boots ``gap.connector.sim("libero", task=<suite>/<task>)``, resets per
seed, and dumps everything a criteria author (or the selftest trace
builder) needs — measured, never assumed:

- every world-snapshot body: pose, AABB, extents (robot-base frame,
  matching ``state.jsonl``), whether it hangs on a free joint (movable),
  cavity AABB when a ``<body>__cavity`` site exists;
- every non-robot hinge/slide joint: qpos, limits, type, owning body
  (articulated fixtures: drawers, stove knobs, microwave doors);
- cameras (world frame): position, orientation, fovy;
- env metadata: task language, t0 ``check_success`` (should be False),
  the world-frame robot-base pose.

Output: ``<out>/<suite>__task<k>__seed<n>.json`` — these files are
CHECKED IN under ``layouts/`` and become the single source for criteria
thresholds and synthetic selftest traces.

Usage::

    uv run python examples/zero_shot_prompts/probes/dump_scene.py \
        --suite libero_object_all_variance --task 0 --seeds 0,1,2

    # optional physics-settle drift check (new scenes) and a saved
    # agentview frame (perception eyeballing):
    ... --settle 200 --save-frame
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLE = _HERE.parent
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from zsp.runner.gpu import pin_render_gpu, relocate_tmpdir  # noqa: E402
from zsp.state_log import _f, _RawExtras  # noqa: E402


def _walk_attr(env: Any, name: str) -> Any:
    """First non-None ``name`` attribute along the env/handle chain."""
    seen: set[int] = set()
    frontier = [env]
    for _ in range(8):
        nxt = []
        for obj in frontier:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            val = getattr(obj, name, None)
            if val is not None:
                return val
            for attr in ("handle", "env"):
                child = getattr(obj, attr, None)
                if child is not None:
                    nxt.append(child)
        if not nxt:
            break
        frontier = nxt
    return None


def _free_joint_owners(model: Any) -> set[str]:
    """Body names hung on a free joint, ``_main`` suffix stripped."""
    import mujoco

    out: set[str] = set()
    for jid in range(int(model.njnt)):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        bid = int(model.jnt_bodyid[jid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name.endswith("_main"):
            name = name[: -len("_main")]
        if name:
            out.add(name)
    return out


def _joint_details(extras: _RawExtras) -> dict[str, dict[str, Any]]:
    import mujoco

    model, data = extras._model, extras._data
    out: dict[str, dict[str, Any]] = {}
    for name, addr in extras.joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        jtype = int(model.jnt_type[jid])
        raw_body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])
        )
        out[name] = {
            "qpos": round(float(data.qpos[addr]), 5),
            "range": (
                _f(model.jnt_range[jid])
                if bool(model.jnt_limited[jid]) else None
            ),
            "type": {2: "slide", 3: "hinge"}.get(jtype, str(jtype)),
            "raw_body": raw_body,
        }
    return out


def _cameras(extras: _RawExtras) -> dict[str, dict[str, Any]]:
    import mujoco

    model, data = extras._model, extras._data
    out: dict[str, dict[str, Any]] = {}
    for cid in range(int(model.ncam)):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
            or f"cam_{cid}"
        )
        out[name] = {
            "pos_world": _f(data.cam_xpos[cid]),
            "xmat_world": _f(data.cam_xmat[cid]),
            "fovy": round(float(model.cam_fovy[cid]), 3),
        }
    return out


def _settle_drift(conn: Any, extras: _RawExtras, steps: int) -> dict[str, float]:
    """Step physics and report per-body displacement (meters).

    Prefers the env's own zero-delta OSC stepping (controller holds the
    arm); falls back to raw ``mj_step`` (caveat: uncontrolled arm).
    """
    before = {
        n: np.asarray(conn.world_snapshot().body(n).position)
        for n in conn.world_snapshot().body_names()
    }
    stepper = _walk_attr(conn.env, "apply_policy_action")
    if callable(stepper):
        action = np.zeros(7)
        for _ in range(steps):
            stepper(action)
    else:
        import mujoco

        mujoco.mj_step(extras._model, extras._data, nstep=int(steps))
    after_w = conn.world_snapshot()
    return {
        n: round(float(np.linalg.norm(
            np.asarray(after_w.body(n).position) - before[n])), 5)
        for n in after_w.body_names() if n in before
    }


def _save_frame(conn: Any, out_path: Path) -> str | None:
    try:
        obs = conn.env.get_observation()
    except Exception:
        return None
    for key, val in obs.items():
        if "agentview" not in str(key) or "image" not in str(key):
            continue
        arr = np.asarray(val)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            try:
                import imageio.v2 as imageio

                imageio.imwrite(out_path, arr.astype(np.uint8))
                return str(out_path)
            except Exception:
                return None
    return None


def dump_seed(
    conn: Any,
    suite: str,
    task_id: int,
    seed: int,
    *,
    settle: int = 0,
    save_frame: bool = False,
    out_dir: Path,
) -> Path:
    conn.reset(seed=seed)
    extras = _RawExtras(conn.env)
    w = conn.world_snapshot()
    movable = _free_joint_owners(extras._model) if extras.available else set()
    cavities = extras.read_cavities() if extras.available else {}

    bodies: dict[str, dict[str, Any]] = {}
    for name in sorted(w.body_names()):
        b = w.body(name)
        lo, hi = np.asarray(b.aabb_lower), np.asarray(b.aabb_upper)
        rec: dict[str, Any] = {
            "p": _f(b.position),
            "q": _f(b.quaternion_wxyz),
            "lo": _f(lo),
            "hi": _f(hi),
            "ext": _f(hi - lo),
            "movable": name in movable,
        }
        cav = next(
            (aabb for hint, aabb in cavities.items()
             if name == hint or hint.startswith(name)), None,
        )
        rec["cavity"] = cav
        bodies[name] = rec

    try:
        env_success, env_reward = conn.check_success()
    except Exception:
        env_success, env_reward = None, None

    payload = {
        "schema": 1,
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "frame": "robot_base",
        "base_pose_world": extras.base_pose_world(),
        "extras_error": extras.error,
        "task_language": _walk_attr(conn.env, "task_language"),
        "bodies": bodies,
        "joints": _joint_details(extras) if extras.available else {},
        "cameras": _cameras(extras) if extras.available else {},
        "env": {
            "t0_check_success": env_success,
            "t0_reward": env_reward,
        },
    }
    if settle > 0 and extras.available:
        payload["settle_drift_m"] = _settle_drift(conn, extras, settle)

    stem = f"{suite}__task{task_id}__seed{seed}"
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    if save_frame:
        frame = _save_frame(conn, out_dir / f"{stem}.png")
        if frame:
            print(f"  frame: {frame}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--seeds", default="0",
                    help="comma-separated sim seeds (default: 0)")
    ap.add_argument("--out", default=str(_EXAMPLE / "layouts"))
    ap.add_argument("--settle", type=int, default=0,
                    help="no-op physics steps for a drift check (default off)")
    ap.add_argument("--save-frame", action="store_true",
                    help="save an agentview PNG next to the layout")
    args = ap.parse_args()

    pin_render_gpu()
    relocate_tmpdir()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    import gap.connector

    conn = gap.connector.sim(
        "libero", task=f"{args.suite}/{args.task}", record_video=False,
    )
    try:
        for seed in seeds:
            path = dump_seed(
                conn, args.suite, args.task, seed,
                settle=args.settle, save_frame=args.save_frame,
                out_dir=out_dir,
            )
            data = json.loads(path.read_text())
            n_mov = sum(1 for b in data["bodies"].values() if b["movable"])
            print(
                f"{path.name}: {len(data['bodies'])} bodies "
                f"({n_mov} movable), {len(data['joints'])} joints, "
                f"lang={data['task_language']!r}, "
                f"extras_error={data['extras_error']!r}"
            )
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
