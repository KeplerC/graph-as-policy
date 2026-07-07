"""GPU + rendering environment hygiene for sim processes.

Every sim process (runner seed executions, probes) needs EGL rendering
pinned to a GPU with free memory. The rule everywhere: **pre-set values
win** — a parent that already exported ``CUDA_VISIBLE_DEVICES`` or
``MUJOCO_EGL_DEVICE_ID`` (e.g. ``run_all --jobs`` round-robin pinning)
is respected; only an unpinned process auto-picks the freest GPU.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def pick_free_gpu() -> str | None:
    """Index of the GPU with the most free memory, per ``nvidia-smi``."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return None
    best: tuple[int, str] | None = None
    for line in out.strip().splitlines():
        try:
            idx, free = (s.strip() for s in line.split(",")[:2])
            if best is None or int(free) > best[0]:
                best = (int(free), idx)
        except (ValueError, IndexError):
            continue
    return best[1] if best else None


def pin_render_gpu() -> str | None:
    """Ensure ``MUJOCO_GL=egl`` + an EGL device pin; also pin CUDA.

    Precedence: an existing ``MUJOCO_EGL_DEVICE_ID`` wins; else a
    pre-set single-device ``CUDA_VISIBLE_DEVICES`` pins EGL to the SAME
    physical index (EGL enumerates all GPUs regardless of CUDA
    visibility); else the freest GPU is picked for both.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
        return os.environ["MUJOCO_EGL_DEVICE_ID"]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and "," not in cvd:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = cvd.strip()
        return cvd.strip()
    gpu = pick_free_gpu()
    if gpu is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    return gpu


def relocate_tmpdir() -> None:
    """Move ``TMPDIR`` off a near-full ``/tmp`` (video frame buffers)."""
    if os.environ.get("TMPDIR", "").startswith(str(Path.home())):
        return
    tmp = Path.home() / ".cache" / "gap_zero_shot_prompts_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
