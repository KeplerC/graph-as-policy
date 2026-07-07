"""zsp — the zero-shot prompt benchmark package.

Internal package of the ``examples/zero_shot_prompts`` benchmark harness:
case corpus + criteria DSL + trace recording + runners. Import via the
example dir on ``sys.path`` (the entrypoints do this); modules are kept
import-light so ``selftest`` never drags in mujoco or an LLM client.
"""

from __future__ import annotations

__all__ = ["SCHEMA_VERSION"]

#: Trace schema written by :mod:`zsp.state_log` (state.jsonl lines).
SCHEMA_VERSION = 2
