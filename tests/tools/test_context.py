"""Unit tests for gap.runtime.context — NodeContext dispatch, trace, publish."""

# NOTE: no `from __future__ import annotations` — extract_schema resolves the
# TypedDict return annotations defined in this module.

from typing import TypedDict

import pytest
from gap_core.errors import GuardLimitExceeded, TaskCancelled, ToolArgumentError
from gap_core.tools import ToolRegistry, guards

from gap.runtime.context import CancelToken, NodeContext


class IouOutput(TypedDict):
    iou: float


class StubTrace:
    """Captures the exact tracer-facing calls NodeContext makes."""

    def __init__(self):
        self.subcalls = []
        self.stream_reads = []

    def record_subcall(self, node_name, seq, tool, request, response):
        self.subcalls.append((node_name, seq, tool, request, response))

    def record_stream_read(self, node_name, seq, stream_name, value, sampled_at):
        self.stream_reads.append((node_name, seq, stream_name, value, sampled_at))


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    def iou(box_a: list[float], box_b: list[float]) -> IouOutput:
        return {"iou": 0.25}

    reg.register_callable("geometry.iou", iou, summary="IoU.", tags=("geometry",))

    def segment(ctx, image: str) -> dict:
        return {"mask": [0], "ctx": ctx}

    reg.register_callable("seg.segment", segment, summary="Segment.", tags=("perception",))
    return reg


@pytest.fixture(autouse=True)
def _clean_guards(monkeypatch):
    for var in ("GAP_MAX_PERCEPTION_CALLS", "GAP_MAX_PLANNING_CALLS", "GAP_MAX_SIM_STEPS"):
        monkeypatch.delenv(var, raising=False)
    guards.set_limits()
    guards.reset_counters()
    yield
    guards.set_limits()
    guards.reset_counters()


# --- ctx.tool() dispatch + trace recording ---


def test_tool_dispatch_records_subcall():
    trace = StubTrace()
    ctx = NodeContext(_registry(), trace=trace, node_id="n_0001")

    out = ctx.tool("geometry.iou", box_a=[0, 0, 1, 1], box_b=[0, 0, 1, 1])
    assert out == {"iou": 0.25}
    assert trace.subcalls == [
        (
            "n_0001", 0, "geometry.iou",
            {"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
            {"iou": 0.25},
        ),
    ]

    # Sequence number increments per call within the same node context.
    ctx.tool("geometry.iou", box_a=[0, 0, 1, 1], box_b=[1, 1, 2, 2])
    assert [c[1] for c in trace.subcalls] == [0, 1]


def test_tool_dispatch_injects_ctx_and_refuses_unknown_kwargs():
    ctx = NodeContext(_registry())
    out = ctx.tool("seg.segment", image="img")
    assert out["mask"] == [0]
    assert out["ctx"] is ctx
    # A script naming a parameter the tool does not have is refused where it
    # is written, rather than silently running on the tool's defaults.
    with pytest.raises(ToolArgumentError, match="extraneous"):
        ctx.tool("seg.segment", image="img", extraneous="dropped")


def test_tool_dispatch_without_trace_is_silent():
    ctx = NodeContext(_registry())
    assert ctx.tool("geometry.iou", box_a=[0, 0, 1, 1], box_b=[0, 0, 1, 1]) == {"iou": 0.25}


def test_tool_dispatch_enforces_guards(monkeypatch):
    monkeypatch.setenv("GAP_MAX_PERCEPTION_CALLS", "1")
    ctx = NodeContext(_registry())
    ctx.tool("seg.segment", image="img")
    with pytest.raises(GuardLimitExceeded):
        ctx.tool("seg.segment", image="img")
    # Untagged-category tools remain unaffected by the perception limit.
    ctx.tool("geometry.iou", box_a=[0, 0, 1, 1], box_b=[0, 0, 1, 1])


def test_unknown_tool_raises_keyerror():
    ctx = NodeContext(_registry())
    with pytest.raises(KeyError, match="not found"):
        ctx.tool("no.such_tool")


# --- streaming ---


def test_publish_without_stream_slot_raises():
    ctx = NodeContext(_registry())
    with pytest.raises(RuntimeError, match="non-streaming"):
        ctx.publish({"x": 1})


def test_publish_forwards_to_stream_slot():
    class Slot:
        def __init__(self):
            self.values = []

        def publish(self, value):
            self.values.append(value)

    ctx = NodeContext(_registry())
    ctx._stream_slot = Slot()
    ctx.publish(42)
    assert ctx._stream_slot.values == [42]


def test_stream_read_records_into_trace():
    trace = StubTrace()
    ctx = NodeContext(_registry(), trace=trace, node_id="n_0002")
    ctx._stream_read("obs.camera", {"rgb": "x"}, 123.5)
    ctx._stream_read("obs.camera", {"rgb": "y"}, 124.0)
    assert trace.stream_reads == [
        ("n_0002", 0, "obs.camera", {"rgb": "x"}, 123.5),
        ("n_0002", 1, "obs.camera", {"rgb": "y"}, 124.0),
    ]


# --- CancelToken ---


def test_cancel_token():
    token = CancelToken()
    assert not token.is_set()
    token.raise_if_set()  # no-op while unset
    token.cancel()
    assert token.is_set()
    with pytest.raises(TaskCancelled):
        token.raise_if_set()
