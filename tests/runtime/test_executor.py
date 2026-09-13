"""WorkflowExecutor tests — stub tools only, no sim.

Each test materializes a small v3 workflow.json into ``tmp_path`` and
drives it through :class:`gap.runtime.executor.WorkflowExecutor` (or the
``gap.execute`` facade) against an in-memory :class:`gap.tools.ToolRegistry`
of stub callables.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from gap_core.errors import (
    NodeExecutionError,
    PipelineError,
    VerificationFailed,
)
from gap_core.tools import ToolRegistry, guards

import gap
from gap.runtime.execute import ExecutionResult
from gap.runtime.executor import SubgraphExitEvent, WorkflowExecutor
from gap.runtime.verify import StubWorld

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_workflow(tmp_path: Path, raw: dict) -> Path:
    """Write workflow.json into tmp_path; returns the workflow dir."""
    (tmp_path / "workflow.json").write_text(json.dumps(raw))
    return tmp_path


def _registry(
    tools: dict[str, Callable[..., Any]],
    tags: dict[str, tuple[str, ...]] | None = None,
) -> ToolRegistry:
    """Build a bare ToolRegistry from {name: callable} stubs."""
    reg = ToolRegistry()
    for name, fn in tools.items():
        reg.register_callable(
            name, fn, summary=name, tags=(tags or {}).get(name, ()),
        )
    return reg


def _executor(wf_dir: Path, reg: ToolRegistry, **kw: Any) -> WorkflowExecutor:
    return WorkflowExecutor(wf_dir, tool_registry=reg, **kw)


def _sg_workflow(recovery: list[dict] | None = None) -> dict:
    """Subgraph workflow: run_sg → done (exit "ok") / failed (on_error "boom")."""
    failed: dict[str, Any] = {"type": "end", "status": "failure"}
    if recovery:
        failed["recovery"] = recovery
    return {
        "version": 3,
        "meta": {},
        "nodes": {
            "run_sg": {"type": "subgraph", "ref": "sg_work"},
            "done": {"type": "end", "status": "success"},
            "failed": failed,
        },
        "edges": [["START", "run_sg"]],
        "conditional_edges": {
            "run_sg": {
                "router_field": "exit",
                "mapping": {"ok": "done", "boom": "failed"},
            },
        },
        "subgraphs": {
            "sg_work": {
                "skill": "stub_skill",
                "inputs": {},
                "outputs": {"result": {"$ref": "work.value"}},
                "nodes": {
                    "work": {"type": "tool", "tool": "stub.work", "inputs": {}},
                    "ok": {"type": "noop"},
                },
                "edges": [["START", "work"], ["work", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
                "on_error": "boom",
            },
        },
    }


# ---------------------------------------------------------------------------
# Linear tool chain with $ref dataflow
# ---------------------------------------------------------------------------


def test_linear_tool_chain_with_ref_dataflow(tmp_path: Path) -> None:
    calls: list[dict] = []

    def _const(value: int) -> dict:
        return {"value": value}

    def _double(x: int) -> dict:
        calls.append({"x": x})
        return {"value": x * 2}

    reg = _registry({"stub.const": _const, "stub.double": _double})
    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "first": {"type": "tool", "tool": "stub.const", "inputs": {"value": 21}},
            "second": {
                "type": "tool", "tool": "stub.double",
                "inputs": {"x": {"$ref": "first.value"}},
            },
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "first"], ["first", "second"], ["second", "done"]],
    })

    ex = _executor(wf_dir, reg, trace_dir=tmp_path / "trace")
    ex.execute()

    assert ex.exit_status == "success"
    assert calls == [{"x": 21}]
    assert (tmp_path / "trace" / "dag_trace.json").exists()


# ---------------------------------------------------------------------------
# Conditional routing on a router_field, incl. on_error path
# ---------------------------------------------------------------------------


def test_conditional_routing_success_path(tmp_path: Path) -> None:
    reg = _registry({"stub.work": lambda: {"value": 7}})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())

    ex = _executor(wf_dir, reg)
    ex.execute()

    assert ex.exit_status == "success"
    assert ex.cross_subgraph_outputs["sg_work"]["result"] == 7


def test_conditional_routing_on_error_path(tmp_path: Path) -> None:
    """A raising tool routes the subgraph through on_error to the failure end."""
    def _boom() -> dict:
        raise PipelineError("inner explosion")

    reg = _registry({"stub.work": _boom})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())

    ex = _executor(wf_dir, reg)
    with pytest.raises(PipelineError, match="failure"):
        ex.execute()
    assert ex.exit_status == "failure"
    # The output binding never resolved on the error path.
    assert ex.cross_subgraph_outputs["sg_work"] == {}


def test_node_error_message_reaches_the_trace(tmp_path: Path) -> None:
    """A routed failure still records *why* on the trace node.

    Regression: the on_error path swallowed the exception into a log line, so
    ``dag_trace.json`` carried ``status: error, error_message: null`` and no
    consumer of the trace could tell a wrong grasp from a typo'd dict key.
    """
    def _boom() -> dict:
        raise KeyError("z")

    reg = _registry({"stub.work": _boom})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())

    ex = _executor(wf_dir, reg)
    with pytest.raises(PipelineError, match="failure"):
        ex.execute()

    ex.trace.flush()
    trace = json.loads((wf_dir / "dag_trace.json").read_text())
    node = next(n for n in trace["nodes"] if n["name"] == "sg_work.work")
    assert node["status"] == "error"
    # A bare KeyError stringifies to "'z'"; the type name is what makes it
    # readable, so it has to be part of the recorded text.
    assert node["error_message"] is not None
    assert "KeyError" in node["error_message"]
    assert "'z'" in node["error_message"]


def test_terminal_failure_names_the_first_node_error(tmp_path: Path) -> None:
    """The top-level error says what broke, not just which end node it hit."""
    def _boom() -> dict:
        raise RuntimeError("the fingers closed on nothing")

    reg = _registry({"stub.work": _boom})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())

    ex = _executor(wf_dir, reg)
    with pytest.raises(PipelineError) as excinfo:
        ex.execute()
    message = str(excinfo.value)
    assert "sg_work.work" in message
    assert "the fingers closed on nothing" in message


# ---------------------------------------------------------------------------
# Parallel super-step fan-out with first-error cancellation
# ---------------------------------------------------------------------------


def test_parallel_fanout_first_error_cancels_pending(tmp_path: Path) -> None:
    ran: list[str] = []

    def _boom() -> dict:
        raise PipelineError("first error")

    def _sleeper() -> dict:
        ran.append("s")
        time.sleep(0.05)
        return {}

    tools: dict[str, Callable[..., Any]] = {"stub.boom": _boom}
    nodes: dict[str, Any] = {
        "boom": {"type": "tool", "tool": "stub.boom", "inputs": {}},
        "done": {"type": "end", "status": "success"},
    }
    edges = [["START", "boom"], ["boom", "done"]]
    for i in range(5):
        tools[f"stub.s{i}"] = _sleeper
        nodes[f"s{i}"] = {"type": "tool", "tool": f"stub.s{i}", "inputs": {}}
        edges.append(["START", f"s{i}"])
        edges.append([f"s{i}", "done"])

    reg = _registry(tools)
    wf_dir = _write_workflow(tmp_path, {
        "version": 3, "meta": {}, "nodes": nodes, "edges": edges,
    })

    # One worker: boom runs first and raises; queued sleepers get cancelled.
    ex = _executor(wf_dir, reg, max_node_workers=1)
    with pytest.raises(NodeExecutionError, match="first error"):
        ex.execute()
    assert len(ran) < 5, "pending siblings were not cancelled after first error"


# ---------------------------------------------------------------------------
# Streaming node publishing → downstream consumes via {"$ref": <node>}
# ---------------------------------------------------------------------------

_TRACKER_PY = '''\
import time


def run(ctx) -> None:
    i = 0
    while not ctx.cancel_token.is_set():
        ctx.publish({"tick": i})
        i += 1
        time.sleep(0.005)
'''

_CONSUMER_PY = '''\
from typing import TypedDict


class Out(TypedDict):
    tick: int


def run(ctx, tracker: dict) -> Out:
    return {"tick": int(tracker["tick"])}
'''


def test_streaming_node_feeds_downstream_consumer(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "tracker.py").write_text(_TRACKER_PY)
    (scripts / "consumer.py").write_text(_CONSUMER_PY)

    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "run_sg": {"type": "subgraph", "ref": "sg_stream"},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "run_sg"]],
        "conditional_edges": {
            "run_sg": {"router_field": "exit", "mapping": {"ok": "done"}},
        },
        "subgraphs": {
            "sg_stream": {
                "skill": "stub_skill",
                "inputs": {},
                "outputs": {"tick": {"$ref": "consumer.tick"}},
                "nodes": {
                    "tracker": {
                        "type": "script", "script": "scripts/tracker.py",
                        "streaming": True, "inputs": {},
                    },
                    "consumer": {
                        "type": "script", "script": "scripts/consumer.py",
                        "inputs": {"tracker": {"$ref": "tracker"}},
                    },
                    "ok": {"type": "noop"},
                },
                "edges": [
                    ["START", "tracker"],
                    ["START", "consumer"],
                    ["consumer", "ok"],
                    ["ok", "END"],
                ],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    })

    ex = _executor(wf_dir, _registry({}))
    ex.execute()

    assert ex.exit_status == "success"
    assert ex.cross_subgraph_outputs["sg_stream"]["tick"] >= 0


# ---------------------------------------------------------------------------
# Send fan-out from a router script
# ---------------------------------------------------------------------------

_ROUTER_PY = '''\
def run(ctx):
    return {"route": [
        {"to": "worker", "inputs": {"x": 1}},
        {"to": "worker", "inputs": {"x": 2}},
    ]}
'''


def test_send_fanout_from_router_script(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "router.py").write_text(_ROUTER_PY)

    calls: list[int] = []

    def _record(x: int) -> dict:
        calls.append(x)
        return {"x": x}

    reg = _registry({"stub.record": _record})
    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "route": {"type": "router", "script": "scripts/router.py"},
            "worker": {"type": "tool", "tool": "stub.record", "inputs": {"x": 0}},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "route"], ["route", "worker"], ["worker", "done"]],
    })

    ex = _executor(wf_dir, reg)
    ex.execute()

    assert ex.exit_status == "success"
    # Two Send copies (x=1, x=2) plus the frontier visit (x=0).
    assert sorted(calls) == [0, 1, 2]


# ---------------------------------------------------------------------------
# node_visit_cap cycle guard
# ---------------------------------------------------------------------------


def test_node_visit_cap_triggers(tmp_path: Path) -> None:
    reg = _registry({f"stub.n{i}": (lambda: {}) for i in range(3)})
    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "n0": {"type": "tool", "tool": "stub.n0", "inputs": {}},
            "n1": {"type": "tool", "tool": "stub.n1", "inputs": {}},
            "n2": {"type": "tool", "tool": "stub.n2", "inputs": {}},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [
            ["START", "n0"], ["n0", "n1"], ["n1", "n2"], ["n2", "done"],
        ],
    })

    ex = _executor(wf_dir, reg, node_visit_cap=2)
    with pytest.raises(PipelineError, match="super-step cap"):
        ex.execute()

    # A generous cap lets the same workflow finish.
    ex2 = _executor(wf_dir, reg, node_visit_cap=10)
    ex2.execute()
    assert ex2.exit_status == "success"


# ---------------------------------------------------------------------------
# Conditional back edges (loops)
# ---------------------------------------------------------------------------

_LOOP_ROUTER_PY = '''\
def run(ctx, n):
    return {"route": "done" if n >= 3 else "loop"}
'''

_FOREVER_ROUTER_PY = '''\
def run(ctx, n):
    return {"route": "loop"}
'''


def test_conditional_back_edge_loops_until_threshold(tmp_path: Path) -> None:
    """A conditional edge to an already-completed node re-iterates the loop
    body until the router routes to the exit."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "decide.py").write_text(_LOOP_ROUTER_PY)

    count = {"n": 0}

    def _step() -> dict:
        count["n"] += 1
        return {"n": count["n"]}

    reg = _registry({"stub.step": _step})
    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "step": {"type": "tool", "tool": "stub.step", "inputs": {}},
            "decide": {
                "type": "router", "script": "scripts/decide.py",
                "inputs": {"n": {"$ref": "step.n"}},
            },
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "step"], ["step", "decide"]],
        "conditional_edges": {
            # decide --loop--> step is the BACKWARD edge (step already ran).
            "decide": {"router_field": None,
                       "mapping": {"loop": "step", "done": "done"}},
        },
    })

    ex = _executor(wf_dir, reg)
    ex.execute()

    assert ex.exit_status == "success"
    # step re-ran on each loop iteration until n reached the threshold.
    assert count["n"] == 3


def test_a_looping_node_records_how_many_times_it_ran(tmp_path: Path) -> None:
    """A retry ladder and a livelock differ only in the count.

    Backward edges are the format's only loop and carry no bound, so a graph
    that spins reports nothing but whatever the step budget did to it. The
    count is the one number that separates the two.
    """
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "decide.py").write_text(_LOOP_ROUTER_PY)

    count = {"n": 0}

    def _step() -> dict:
        count["n"] += 1
        return {"n": count["n"]}

    reg = _registry({"stub.step": _step})
    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "step": {"type": "tool", "tool": "stub.step", "inputs": {}},
            "decide": {
                "type": "router", "script": "scripts/decide.py",
                "inputs": {"n": {"$ref": "step.n"}},
            },
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "step"], ["step", "decide"]],
        "conditional_edges": {
            "decide": {"router_field": None,
                       "mapping": {"loop": "step", "done": "done"}},
        },
    })

    ex = _executor(wf_dir, reg)
    ex.execute()
    ex.trace.flush()

    trace = json.loads((wf_dir / "dag_trace.json").read_text())
    step = next(n for n in trace["nodes"] if n["name"] == "step")
    assert step["visits"] == count["n"] > 1


def test_a_terminal_error_is_not_routed_to_on_error(tmp_path: Path) -> None:
    """"The episode ended" must stop the graph, not send it round again.

    A recovery loop answering a closed world by calling another tool raises
    the same thing forever; one study episode spun 9,186 times that way.
    """
    class WorldGone(RuntimeError):
        terminal = True

    calls = {"n": 0}

    def _boom() -> dict:
        calls["n"] += 1
        raise WorldGone("the simulation bus closed: the episode ended")

    reg = _registry({"stub.work": _boom})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())

    ex = _executor(wf_dir, reg)
    with pytest.raises(NodeExecutionError, match="episode ended"):
        ex.execute()
    # Propagated on the first raise rather than routed into the recovery path.
    assert calls["n"] == 1
    assert ex.exit_status is None


def test_an_ordinary_failure_still_routes(tmp_path: Path) -> None:
    """The terminal check must not swallow the on_error exit it guards."""
    def _boom() -> dict:
        raise PipelineError("the fingers closed on nothing")

    reg = _registry({"stub.work": _boom})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())

    ex = _executor(wf_dir, reg)
    with pytest.raises(PipelineError, match="failure"):
        ex.execute()
    assert ex.exit_status == "failure"


def test_unbounded_conditional_loop_trips_cap(tmp_path: Path) -> None:
    """A conditional back edge whose router never exits still trips the
    super-step cap — the runaway-loop guard survives loop support."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "decide.py").write_text(_FOREVER_ROUTER_PY)

    def _step() -> dict:
        return {"n": 1}

    reg = _registry({"stub.step": _step})
    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "step": {"type": "tool", "tool": "stub.step", "inputs": {}},
            "decide": {
                "type": "router", "script": "scripts/decide.py",
                "inputs": {"n": {"$ref": "step.n"}},
            },
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "step"], ["step", "decide"]],
        "conditional_edges": {
            "decide": {"router_field": None,
                       "mapping": {"loop": "step", "done": "done"}},
        },
    })

    ex = _executor(wf_dir, reg, node_visit_cap=5)
    with pytest.raises(PipelineError, match="super-step cap"):
        ex.execute()


def test_subgraph_back_edge_loops(tmp_path: Path) -> None:
    """A backward edge over SUBGRAPH nodes re-invokes the subgraph (fresh
    inner scope) each iteration — the shape the grocery_packing example uses."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "decide.py").write_text(_LOOP_ROUTER_PY)

    count = {"n": 0}

    def _bump() -> dict:
        count["n"] += 1
        return {"n": count["n"]}

    reg = _registry({"stub.bump": _bump})
    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "step": {"type": "subgraph", "ref": "step_sg"},
            "decide": {
                "type": "router", "script": "scripts/decide.py",
                "inputs": {"n": {"$ref": "step.count"}},
            },
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "step"]],
        "conditional_edges": {
            "step": {"router_field": "exit", "mapping": {"ok": "decide"}},
            # decide --loop--> step re-enters the completed subgraph node.
            "decide": {"router_field": None,
                       "mapping": {"loop": "step", "done": "done"}},
        },
        "subgraphs": {
            "step_sg": {
                "skill": "stub_skill",
                "inputs": {},
                "outputs": {"count": {"$ref": "bump.n"}},
                "nodes": {
                    "bump": {"type": "tool", "tool": "stub.bump", "inputs": {}},
                    "ok": {"type": "noop"},
                },
                "edges": [["START", "bump"], ["bump", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
                "on_error": "boom",
            },
        },
    })

    ex = _executor(wf_dir, reg)
    ex.execute()

    assert ex.exit_status == "success"
    # The subgraph re-ran each iteration until its output crossed the threshold.
    assert count["n"] == 3


# ---------------------------------------------------------------------------
# Recovery ToolCalls on a failure end node
# ---------------------------------------------------------------------------


def test_recovery_toolcalls_run_on_failure_end(tmp_path: Path) -> None:
    attempted: list[str] = []
    recovered: list[dict] = []

    def _boom() -> dict:
        raise PipelineError("inner explosion")

    def _recover_a(speed: int) -> dict:
        attempted.append("a")
        raise RuntimeError("recovery tool a is broken")

    def _recover_b() -> dict:
        recovered.append({"tool": "b"})
        return {}

    reg = _registry({
        "stub.work": _boom,
        "robot.recover_a": _recover_a,
        "robot.recover_b": _recover_b,
    })
    wf_dir = _write_workflow(tmp_path, _sg_workflow(recovery=[
        {"tool": "robot.recover_a", "inputs": {"speed": 2}},
        {"tool": "robot.recover_b", "inputs": {}},
    ]))

    ex = _executor(wf_dir, reg)
    with pytest.raises(PipelineError, match="failure"):
        ex.execute()

    # Both recovery entries ran (best-effort: a raised, b still ran;
    # neither error masked the end-node failure).
    assert attempted == ["a"]
    assert recovered == [{"tool": "b"}]


# ---------------------------------------------------------------------------
# Subgraph exit hook
# ---------------------------------------------------------------------------


def test_subgraph_exit_hook_fires(tmp_path: Path) -> None:
    events: list[SubgraphExitEvent] = []
    reg = _registry({"stub.work": lambda: {"value": 7}})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())

    ex = _executor(wf_dir, reg, subgraph_exit_hook=events.append)
    ex.execute()

    assert len(events) == 1
    ev = events[0]
    assert ev.sg_name == "sg_work"
    assert ev.visit_index == 0
    assert ev.bound_outputs == {"result": 7}
    assert ev.exit_value == "ok"
    assert ev.error_path is False
    assert ev.elapsed_s >= 0.0


def test_subgraph_exit_hook_fires_on_error_path(tmp_path: Path) -> None:
    events: list[SubgraphExitEvent] = []

    def _boom() -> dict:
        raise PipelineError("inner explosion")

    reg = _registry({"stub.work": _boom})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())

    ex = _executor(wf_dir, reg, subgraph_exit_hook=events.append)
    with pytest.raises(PipelineError, match="failure"):
        ex.execute()

    assert len(events) == 1
    assert events[0].error_path is True
    assert events[0].exit_value == "boom"


# ---------------------------------------------------------------------------
# Checkpoint enforcement (validate=True sidecar predicates)
# ---------------------------------------------------------------------------

_CHECKPOINTS_PY = '''\
from gap.runtime.verify import Checkpoint

CHECKPOINTS = [
    Checkpoint(
        name="result_is_seven",
        subgraph="sg_work",
        predicate=lambda world, outputs: outputs.get("result") == 7,
    ),
    Checkpoint(
        name="always_fails",
        subgraph="sg_work",
        predicate=lambda world: False,
    ),
    Checkpoint(
        name="probe_never_enforced",
        subgraph="sg_work",
        predicate=lambda world: False,
        validate=False,
    ),
]
'''


def _write_checkpoint_sidecar(tmp_path: Path) -> None:
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    (cp_dir / "sg_work.py").write_text(_CHECKPOINTS_PY)


def test_checkpoints_warn_mode_collects_results(tmp_path: Path) -> None:
    reg = _registry({"stub.work": lambda: {"value": 7}})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())
    _write_checkpoint_sidecar(tmp_path)

    ex = _executor(
        wf_dir, reg,
        checkpoints="warn",
        world_snapshot_fn=lambda: StubWorld(body_names=["cube"]),
    )
    ex.execute()  # warn mode never raises

    assert ex.exit_status == "success"
    by_name = {r.name: r for r in ex.checkpoint_results}
    # validate=False probes are never evaluated here.
    assert set(by_name) == {"result_is_seven", "always_fails"}
    assert by_name["result_is_seven"].passed is True
    assert by_name["always_fails"].passed is False
    assert all(r.subgraph == "sg_work" for r in ex.checkpoint_results)


def test_checkpoints_raise_mode_raises_verification_failed(tmp_path: Path) -> None:
    reg = _registry({"stub.work": lambda: {"value": 7}})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())
    _write_checkpoint_sidecar(tmp_path)

    ex = _executor(
        wf_dir, reg,
        checkpoints="raise",
        world_snapshot_fn=lambda: StubWorld(body_names=["cube"]),
    )
    with pytest.raises(VerificationFailed, match="always_fails"):
        ex.execute()
    # Results were still collected before the raise.
    assert {r.name for r in ex.checkpoint_results} == {
        "result_is_seven", "always_fails",
    }


def test_checkpoints_skipped_without_world_snapshot_fn(tmp_path: Path) -> None:
    """checkpoints != "off" with no ground truth logs once and skips."""
    reg = _registry({"stub.work": lambda: {"value": 7}})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())
    _write_checkpoint_sidecar(tmp_path)

    ex = _executor(wf_dir, reg, checkpoints="raise", world_snapshot_fn=None)
    ex.execute()
    assert ex.exit_status == "success"
    assert ex.checkpoint_results == []
    assert ex._checkpoints_warned is True


def test_checkpoints_off_ignores_sidecar(tmp_path: Path) -> None:
    reg = _registry({"stub.work": lambda: {"value": 7}})
    wf_dir = _write_workflow(tmp_path, _sg_workflow())
    _write_checkpoint_sidecar(tmp_path)

    ex = _executor(
        wf_dir, reg,
        checkpoints="off",
        world_snapshot_fn=lambda: StubWorld(body_names=["cube"]),
    )
    ex.execute()
    assert ex.checkpoint_results == []


# ---------------------------------------------------------------------------
# Guard counters reset at execute() start
# ---------------------------------------------------------------------------


def test_guards_reset_between_executes(tmp_path: Path) -> None:
    reg = _registry(
        {"vision.detect": lambda: {"found": True}},
        tags={"vision.detect": ("perception",)},
    )
    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "look": {"type": "tool", "tool": "vision.detect", "inputs": {}},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "look"], ["look", "done"]],
    })

    guards.set_limits(perception=1)
    try:
        _executor(wf_dir, reg).execute()
        # Without the reset_counters() call at execute() start, this
        # second run would trip the perception guard (2 calls > limit 1).
        _executor(wf_dir, reg).execute()
    finally:
        guards.set_limits()  # clear programmatic limits


# ---------------------------------------------------------------------------
# Script nodes (typed run(ctx, ...) -> TypedDict)
# ---------------------------------------------------------------------------

_COMPUTE_PY = '''\
from typing import TypedDict


class Out(TypedDict):
    y: int


def run(ctx, x: int) -> Out:
    return {"y": x * 3}
'''


def test_script_node_typed_run(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "compute.py").write_text(_COMPUTE_PY)

    wf_dir = _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "run_sg": {"type": "subgraph", "ref": "sg_script"},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "run_sg"]],
        "conditional_edges": {
            "run_sg": {"router_field": "exit", "mapping": {"ok": "done"}},
        },
        "subgraphs": {
            "sg_script": {
                "skill": "stub_skill",
                "inputs": {},
                "outputs": {"y": {"$ref": "compute.y"}},
                "nodes": {
                    "compute": {
                        "type": "script", "script": "scripts/compute.py",
                        # "unused" is filtered out by signature inspection.
                        "inputs": {"x": 5, "unused": 1},
                    },
                    "ok": {"type": "noop"},
                },
                "edges": [["START", "compute"], ["compute", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    })

    ex = _executor(wf_dir, _registry({}))
    ex.execute()

    assert ex.exit_status == "success"
    assert ex.cross_subgraph_outputs["sg_script"]["y"] == 15


_OPTIONAL_PY = '''\
from typing import NotRequired, TypedDict


class Out(TypedDict):
    y: int
    measured: NotRequired[float]


def run(ctx, x: int) -> Out:
    # The measurement exists only on some runs; that is what NotRequired says.
    return {"y": x * 3}
'''


_MISSING_REQUIRED_PY = '''\
from typing import NotRequired, TypedDict


class Out(TypedDict):
    y: int
    measured: NotRequired[float]


def run(ctx, x: int) -> Out:
    return {"measured": 1.5}
'''


def _typed_script_workflow(tmp_path: Path, body: str) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "compute.py").write_text(body)
    return _write_workflow(tmp_path, {
        "version": 3,
        "meta": {},
        "nodes": {
            "run_sg": {"type": "subgraph", "ref": "sg_script"},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "run_sg"]],
        "conditional_edges": {"run_sg": {"router_field": "exit", "mapping": {"ok": "done"}}},
        "subgraphs": {
            "sg_script": {
                "skill": "stub_skill",
                "inputs": {},
                "outputs": {"y": {"$ref": "compute.y"}},
                "nodes": {
                    "compute": {"type": "script", "script": "scripts/compute.py", "inputs": {"x": 5}},
                    "ok": {"type": "noop"},
                },
                "edges": [["START", "compute"], ["compute", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    })


def test_script_may_omit_a_not_required_output(tmp_path: Path) -> None:
    """``NotRequired`` declares a key that MAY be absent, and the executor honours it.

    Reading every annotation instead of ``__required_keys__`` made a script that
    returns a measurement only when it has one fail on the runs where it has
    none -- so the author's only options were to drop the annotation or return a
    placeholder. A shipped cable script had dropped it, with a comment saying
    why, and every early return in another one raised here.
    """
    ex = _executor(_typed_script_workflow(tmp_path, _OPTIONAL_PY), _registry({}))
    ex.execute()

    assert ex.exit_status == "success"
    assert ex.cross_subgraph_outputs["sg_script"]["y"] == 15


def test_script_omitting_a_required_output_still_fails(tmp_path: Path) -> None:
    """The check still binds on the keys the TypedDict requires."""
    ex = _executor(_typed_script_workflow(tmp_path, _MISSING_REQUIRED_PY), _registry({}))
    with pytest.raises(NodeExecutionError, match="missing declared key 'y'"):
        ex.execute()


# ---------------------------------------------------------------------------
# execute() facade
# ---------------------------------------------------------------------------


def test_execute_facade_dict_graph(tmp_path: Path) -> None:
    reg = _registry({"stub.work": lambda: {"value": 7}})
    conn = SimpleNamespace(tool_registry=reg)

    result = gap.execute(_sg_workflow(), conn, trace_dir=tmp_path / "trace")

    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.exit_status == "success"
    assert result.error is None
    assert result.outputs["sg_work"]["result"] == 7
    assert result.checkpoint_results == []
    assert result.duration_s > 0
    assert result.trace_path is not None
    assert (result.trace_path / "dag_trace.json").exists()


def test_execute_facade_failure_and_builder_duck_type(tmp_path: Path) -> None:
    def _boom() -> dict:
        raise PipelineError("inner explosion")

    reg = _registry({"stub.work": _boom})
    conn = SimpleNamespace(tool_registry=reg)

    class _GraphObj:  # gap.builder.Workflow duck-type
        def to_dict(self) -> dict:
            return _sg_workflow()

    result = gap.execute(_GraphObj(), conn, trace_dir=tmp_path / "trace")

    assert result.success is False
    assert result.exit_status == "failure"
    assert isinstance(result.error, PipelineError)


# ---------------------------------------------------------------------------
# A subgraph's inputs bind at the call site
# ---------------------------------------------------------------------------


def _passing_workflow(site_inputs: dict) -> dict:
    """One subgraph declaring an input; the calling node passes `site_inputs`."""
    return {
        "version": 3,
        "meta": {},
        "nodes": {
            "seed": {"type": "tool", "tool": "stub.const", "inputs": {"value": 7}},
            "run": {"type": "subgraph", "ref": "run_sg", "inputs": site_inputs},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "seed"], ["seed", "run"]],
        "conditional_edges": {
            "run": {"router_field": "exit", "mapping": {"ok": "done"}},
        },
        "subgraphs": {
            "run_sg": {
                "skill": "generic",
                "inputs": {"arm_id": "int"},
                "outputs": {},
                "nodes": {
                    "use": {
                        "type": "tool", "tool": "stub.use",
                        "inputs": {"arm_id": {"$ref": "in.arm_id"}},
                    },
                    "ok": {"type": "noop"},
                },
                "edges": [["START", "use"], ["use", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    }


def test_subgraph_input_binds_from_a_literal_at_the_call_site(tmp_path: Path) -> None:
    """`{"inputs": {"arm_id": 0}}` on the calling node is the value the
    subgraph runs with -- WORKFLOW_FORMAT's promise, and the shape that
    aborted before its first node for five self-learning iterations."""
    seen: list[int] = []
    reg = _registry({
        "stub.const": lambda value: {"value": value},
        "stub.use": lambda arm_id: seen.append(arm_id) or {"arm_id": arm_id},
    })
    wf_dir = _write_workflow(tmp_path, _passing_workflow({"arm_id": 0}))
    ex = _executor(wf_dir, reg, trace_dir=tmp_path / "trace")
    ex.execute()
    assert ex.exit_status == "success"
    assert seen == [0]


def test_subgraph_input_binds_from_a_ref_at_the_call_site(tmp_path: Path) -> None:
    """A `$ref` into the caller's scope is resolved the way script nodes do."""
    seen: list[int] = []
    reg = _registry({
        "stub.const": lambda value: {"value": value},
        "stub.use": lambda arm_id: seen.append(arm_id) or {"arm_id": arm_id},
    })
    wf_dir = _write_workflow(tmp_path, _passing_workflow({"arm_id": {"$ref": "seed.value"}}))
    ex = _executor(wf_dir, reg, trace_dir=tmp_path / "trace")
    ex.execute()
    assert seen == [7]


def test_unbound_subgraph_input_leaves_a_trace_naming_the_node(tmp_path: Path) -> None:
    """A binding failure happens before any node runs; it must still write the
    trace and blame the node, or it reads downstream as 'ran and did nothing'."""
    reg = _registry({
        "stub.const": lambda value: {"value": value},
        "stub.use": lambda arm_id: {"arm_id": arm_id},
    })
    # Passing a $ref to a field the producer never emits passes W8 (the site
    # names the input) and fails at bind time -- the one failure that happens
    # before any node runs.
    wf_dir = _write_workflow(tmp_path, _passing_workflow({"arm_id": {"$ref": "seed.missing"}}))
    ex = _executor(wf_dir, reg, trace_dir=tmp_path / "trace")
    with pytest.raises(PipelineError, match="arm_id"):
        ex.execute()
    trace = json.loads((tmp_path / "trace" / "dag_trace.json").read_text())
    errors = [e for e in trace["events"] if e["event_type"] == "error_recorded"]
    assert errors and errors[0]["node_name"] == "run"
    assert "arm_id" in errors[0]["detail"]["error"]


def test_a_stale_call_site_ref_falls_through_to_the_upstream_producer(tmp_path: Path) -> None:
    """One unresolvable input must not sink the call.

    Before call-site inputs were honoured they were ignored wholesale, so a
    graph carrying a stale `$ref` beside inputs its upstream subgraphs
    produce ran fine. The shipped expert tool-hanging graph is one of those:
    resolving the passed dict as a unit turned six of eight hung wrenches
    into a load-time abort in every episode.
    """
    seen: list[int] = []
    reg = _registry({
        "stub.const": lambda value: {"value": value},
        "stub.use": lambda arm_id: seen.append(arm_id) or {"arm_id": arm_id},
    })
    wf_dir = _write_workflow(tmp_path, _passing_workflow({"arm_id": {"$ref": "seed.missing"}}))
    ex = _executor(wf_dir, reg, trace_dir=tmp_path / "trace")
    ex.initial_inputs = {"arm_id": 3}
    ex.execute()
    assert seen == [3], "the upstream value stands in for the ref that would not resolve"


def test_an_input_no_source_answers_names_both_attempts(tmp_path: Path) -> None:
    """With nothing upstream either, the error says the ref was tried and why."""
    reg = _registry({
        "stub.const": lambda value: {"value": value},
        "stub.use": lambda arm_id: {"arm_id": arm_id},
    })
    wf_dir = _write_workflow(tmp_path, _passing_workflow({"arm_id": {"$ref": "seed.missing"}}))
    ex = _executor(wf_dir, reg, trace_dir=tmp_path / "trace")
    with pytest.raises(PipelineError) as excinfo:
        ex.execute()
    message = str(excinfo.value)
    assert "arm_id" in message and "does not resolve" in message
