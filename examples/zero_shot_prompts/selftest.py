"""Benchmark self-test on synthetic state traces (no sim, no LLM, ~1 s).

    uv run python examples/zero_shot_prompts/selftest.py

Runs the ``zsp.selftest`` suites: every DSL primitive/op/resolver
(pass + characteristic-fail traces), the v1-parity scenarios against the
migrated T0 cases, static schema validation of every case against its
probed layout, and the automatic vacuous-pass guard (every criterion
must FAIL a do-nothing trace).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from zsp.selftest import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
