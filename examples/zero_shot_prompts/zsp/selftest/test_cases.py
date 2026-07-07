"""Per-case behavioural guards.

The automatic **vacuous-pass guard**: every registered case's criterion
must FAIL on a do-nothing trace built from its own probed layout (two
identical snapshots, no events, successful exit). A criterion that
passes when nothing happened is broken — unless the case explicitly
declares ``vacuous_ok`` (do-nothing prompts), which flips the assertion.

Per-case PASS traces live next to the corpus: parity covers the 13 T0
cases; Wave-5 case modules register builder pass-traces here.
"""

from __future__ import annotations

from zsp.evaluate import evaluate_seed
from zsp.registry import ALL_CASES
from zsp.selftest.layouts import EXEC_OK, Layout, noop_trace


def run(t) -> None:
    for case in ALL_CASES:
        if case.success_source == "gen":
            ev = evaluate_seed(case, [], [], {},
                               extras={"gen_outcome": "OK",
                                       "gen_message": ""})
            wanted_pass = case.vacuous_ok
            t.expect(
                (ev.verdict == "PASS") == wanted_pass,
                f"vacuous guard (gen): {case.id} got {ev.verdict} "
                f"(vacuous_ok={case.vacuous_ok})",
            )
            continue

        key = case.layout_key()
        if not Layout.exists(key):
            # test_schema already reports the missing layout.
            continue
        trace = noop_trace(Layout.load(key))
        execution = dict(EXEC_OK)
        if case.success_source in ("env", "both"):
            execution["env_task_completed"] = False
        ev = evaluate_seed(case, trace, [], execution)
        if case.vacuous_ok:
            t.expect(ev.verdict == "PASS",
                     f"vacuous-ok case {case.id} should PASS a no-op "
                     f"trace; got {ev.verdict}: {ev.summary[:120]}")
        else:
            t.expect(ev.verdict != "PASS",
                     f"VACUOUS PASS: {case.id} passed a do-nothing trace "
                     f"({ev.summary[:120]})")
