"""Selftest package: no sim, no LLM, synthetic traces only (~seconds).

Suites (each a module with ``run(t) -> None`` using the shared tally):

- ``test_primitives`` — every DSL leaf, op, and $t0 resolver: one pass
  trace and at least one characteristic failure each.
- ``test_parity`` (Wave 2) — the v1 selftest scenarios verbatim against
  the migrated cases: identical verdicts and score windows.
- ``test_schema`` (Wave 2) — static validation of every registered case
  against its probed layout.
- ``test_cases`` (Wave 2) — per-case builder pass-trace + the automatic
  vacuous-pass guard over ALL cases.
"""

from __future__ import annotations

import sys


class Tally:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def expect(self, cond: bool, msg: str) -> None:
        if cond:
            self.passed += 1
        else:
            self.failures.append(msg)

    def expect_eval(self, ev, verdict: str, msg: str,
                    score_min: float | None = None,
                    score_max: float | None = None) -> None:
        ok = ev.verdict == verdict
        if score_min is not None:
            ok = ok and ev.score >= score_min - 1e-9
        if score_max is not None:
            ok = ok and ev.score <= score_max + 1e-9
        if ok:
            self.passed += 1
        else:
            detail = "; ".join(
                f"{c.name}:{'ok' if c.passed else 'FAIL'}" for c in ev.checks)
            self.failures.append(
                f"{msg}: got {ev.verdict} score={ev.score:.3f} "
                f"(wanted {verdict}"
                f"{f' score>={score_min}' if score_min is not None else ''}"
                f"{f' score<={score_max}' if score_max is not None else ''})"
                f" [{detail}] :: {ev.summary[:140]}"
            )


def main() -> int:
    t = Tally()
    from zsp.selftest import test_primitives

    test_primitives.run(t)
    for name in ("test_parity", "test_schema", "test_cases"):
        try:
            mod = __import__(f"zsp.selftest.{name}", fromlist=["run"])
        except ImportError:
            continue
        mod.run(t)

    if t.failures:
        print(f"SELFTEST FAILURES ({len(t.failures)}):")
        for f in t.failures:
            print(f"  ✘ {f}")
        print(f"\n{t.passed} passed, {len(t.failures)} FAILED")
        return 1
    print(f"selftest OK: {t.passed} assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
