#!/usr/bin/env python
"""Author the libero_zsp_* scene YAMLs with physics-settled inits.

    uv run python examples/zero_shot_prompts/probes/author_inits.py \
        [--suite libero_zsp_colors] [--settle 300] [--verify-only]

For each scene in ``zsp_scenes.SCENES``:

1. Compose the task YAML with NOMINAL inits (measured per-asset rest
   pose + drop margin; unknown assets drop from a safe height) and
   write it under ``examples/zero_shot_prompts/tasks/<suite>/``.
2. Boot the env through the normal gap loader (GAP_EXTRA_TASK_ROOTS)
   and, per init, settle physics with zero-delta OSC actions, then read
   the objects' free-joint world poses back from qpos.
3. Rewrite the YAML with the settled inits (xyzw quats).
4. Verification round: fresh resets must show < 1 cm drift after
   another settle — the check the benchmark's criteria depend on.

The env maps sim seed -> init index as ``(seed - 1) % n_inits``
(loader convention), so init k is exercised with seed k+1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLE = _HERE.parent
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import os  # noqa: E402

from zsp.runner.gpu import pin_render_gpu, relocate_tmpdir  # noqa: E402
from zsp_scenes import SCENES, rest_pose  # noqa: E402

TASKS_DIR = _EXAMPLE / "tasks"


def _quat_mul_xyzw(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


LYING_ROT = [0.0, 0.7071068, 0.0, 0.7071068]   # 90 deg about world y (xyzw)


def nominal_pose(scene: dict, obj: str, asset: str,
                 xy: tuple[float, float]) -> list[float]:
    z, q = rest_pose(asset)
    special = (scene.get("special") or {}).get(obj, {})
    if "drop_z" in special:
        z = float(special["drop_z"])
    if q is None:
        q = [0.0, 0.0, 0.0, 1.0]
    if special.get("lying"):
        q = _quat_mul_xyzw(LYING_ROT, q)
        z = max(z, 0.05)
    return [float(xy[0]), float(xy[1]), float(z), *[float(v) for v in q]]


def compose_yaml(suite: str, scene: dict,
                 inits: list[dict[str, list[float]]] | None = None) -> str:
    obj_ids = [o[0] for o in scene["objects"]]
    if inits is None:
        inits = [
            {obj: nominal_pose(scene, obj, asset, layout[obj])
             for obj, asset in scene["objects"]}
            for layout in scene["inits"]
        ]
    lines = [
        f"id: {suite}.{scene['file']}",
        f"language: {json.dumps(scene['language'])}",
        "arena: {name: floor}",
        "robot: {name: panda, controller: OSC_POSE}",
        "cameras: [agentview, robot0_eye_in_hand]",
        "camera_height: 128",
        "camera_width: 128",
        "camera_depth: false",
        "objects:",
        *[f"- {{id: {o}, asset: {a}}}" for o, a in scene["objects"]],
        "inits:",
    ]
    for init in inits:
        for i, obj in enumerate(obj_ids):
            # Fixed-point, never e-notation: PyYAML reads dot-less
            # exponents ("1e-06") as strings.
            pose = ", ".join(f"{float(v):.6f}" for v in init[obj])
            prefix = "- " if i == 0 else "  "
            lines.append(f"{prefix}{obj}: [{pose}]")
    lines += [
        "default_init_index: 0",
        "success:",
        f"  predicate: {scene['success']['predicate']}",
        "  args: " + json.dumps(scene["success"]["args"]),
        "horizon: 500",
        "metadata: " + json.dumps(
            {"suite": suite, "source": "authored_zsp",
             "n_inits": len(inits)}),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Env interaction
# ---------------------------------------------------------------------------


def _walk_attr(env, name: str):
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
    return None


def settle(conn, steps: int) -> None:
    stepper = _walk_attr(conn.env, "apply_policy_action")
    action = np.zeros(7)
    for _ in range(steps):
        stepper(action)


def read_world_poses(conn, obj_ids: list[str]) -> dict[str, list[float]]:
    """Free-joint qpos per object -> [x, y, z, qx, qy, qz, qw]."""
    import mujoco

    from gap.connector.world_adapter import raw_mujoco_handles

    model, data = raw_mujoco_handles(conn.env)
    out: dict[str, list[float]] = {}
    for jid in range(int(model.njnt)):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        bid = int(model.jnt_bodyid[jid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name.endswith("_main"):
            name = name[: -len("_main")]
        if name not in obj_ids:
            continue
        adr = int(model.jnt_qposadr[jid])
        x, y, z, qw, qx, qy, qz = (float(v) for v in data.qpos[adr:adr + 7])
        out[name] = [round(v, 6) for v in (x, y, z, qx, qy, qz, qw)]
    return out


def author_suite(suite: str, *, settle_steps: int, verify_steps: int,
                 verify_only: bool) -> bool:
    import gap.connector

    scene = SCENES[suite]
    obj_ids = [o[0] for o in scene["objects"]]
    n_inits = len(scene["inits"])
    suite_dir = TASKS_DIR / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = suite_dir / f"{scene['file']}.yaml"

    if not verify_only:
        yaml_path.write_text(compose_yaml(suite, scene))
        print(f"[{suite}] wrote nominal {yaml_path.name}; settling...")
        conn = gap.connector.sim("libero", task=f"{suite}/0",
                                 record_video=False)
        settled: list[dict[str, list[float]]] = []
        try:
            for k in range(n_inits):
                conn.reset(seed=k + 1)
                settle(conn, settle_steps)
                poses = read_world_poses(conn, obj_ids)
                missing = set(obj_ids) - set(poses)
                if missing:
                    print(f"[{suite}] init {k}: MISSING bodies {missing}")
                    return False
                for obj in obj_ids:
                    nominal = scene["inits"][k][obj]
                    got = poses[obj]
                    dxy = ((got[0] - nominal[0]) ** 2
                           + (got[1] - nominal[1]) ** 2) ** 0.5
                    if dxy > 0.08:
                        print(f"[{suite}] init {k}: {obj} slid "
                              f"{dxy * 100:.1f} cm from nominal "
                              f"({nominal} -> {got[:2]})")
                settled.append(poses)
                print(f"[{suite}] init {k}: settled {len(poses)} objects")
        finally:
            conn.close()
        yaml_path.write_text(compose_yaml(suite, scene, settled))
        print(f"[{suite}] rewrote {yaml_path.name} with settled inits")

    # -- verification round -------------------------------------------------
    import yaml as _yaml

    baked = _yaml.safe_load(yaml_path.read_text())["inits"]
    conn = gap.connector.sim("libero", task=f"{suite}/0", record_video=False)
    ok = True
    try:
        for k in range(n_inits):
            conn.reset(seed=k + 1)
            settle(conn, verify_steps)
            poses = read_world_poses(conn, obj_ids)
            worst = ("", 0.0)
            for obj in obj_ids:
                d = float(np.linalg.norm(
                    np.asarray(poses[obj][:3])
                    - np.asarray(baked[k][obj][:3])))
                if d > worst[1]:
                    worst = (obj, d)
            status = "ok" if worst[1] < 0.01 else "DRIFT"
            if worst[1] >= 0.01:
                ok = False
            print(f"[{suite}] verify init {k}: max drift "
                  f"{worst[1] * 100:.2f} cm ({worst[0]}) {status}")
    finally:
        conn.close()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", default=None,
                    help=f"one of {sorted(SCENES)} (default: all)")
    ap.add_argument("--settle", type=int, default=300)
    ap.add_argument("--verify-steps", type=int, default=200)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    pin_render_gpu()
    relocate_tmpdir()
    os.environ["GAP_EXTRA_TASK_ROOTS"] = str(TASKS_DIR)

    suites = [args.suite] if args.suite else sorted(SCENES)
    failures = []
    for suite in suites:
        try:
            if not author_suite(suite, settle_steps=args.settle,
                                verify_steps=args.verify_steps,
                                verify_only=args.verify_only):
                failures.append(suite)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[{suite}] FAILED: {exc}")
            failures.append(suite)
    if failures:
        print(f"\nFAILED suites: {failures}")
        return 1
    print(f"\nall {len(suites)} suite(s) authored + verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
