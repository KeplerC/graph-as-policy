"""Parser + validator tests for the v3 workflow schema."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from gap_core.errors import WorkflowValidationError

from gap.runtime.validate import validate_workflow
from gap.runtime.workflow import (
    START,
    Ref,
    ToolCall,
    load_workflow,
    resolve_ref,
)


def _write_workflow(tmp_path: Path, raw: dict) -> Path:
    p = tmp_path / "workflow.json"
    p.write_text(json.dumps(raw))
    return p


def _minimal_workflow() -> dict:
    return {
        "version": 3,
        "meta": {},
        "nodes": {
            "sg":   {"type": "subgraph", "ref": "sg_def"},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "sg"]],
        "conditional_edges": {
            "sg": {"router_field": "exit", "mapping": {"ok": "done"}},
        },
        "subgraphs": {
            "sg_def": {
                "skill": "generic",
                "inputs": {},
                "outputs": {},
                "nodes": {
                    "step": {
                        "type": "tool",
                        "tool": "robot.go_home",
                        "inputs": {},
                    },
                    "ok": {"type": "noop"},
                },
                "edges": [
                    ["START", "step"],
                    ["step", "ok"],
                    ["ok", "END"],
                ],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loads_minimal_v3_workflow(tmp_path: Path) -> None:
    p = _write_workflow(tmp_path, _minimal_workflow())
    wf = load_workflow(p)
    assert wf.version == 3
    assert "sg" in wf.nodes
    assert "done" in wf.nodes
    assert wf.nodes["sg"].type == "subgraph"
    assert wf.nodes["sg"].ref == "sg_def"
    assert wf.nodes["done"].type == "end"
    assert wf.nodes["done"].status == "success"
    assert wf.edges == ((START, "sg"),)


def test_loads_subgraph_with_streaming_node(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["tracker"] = {
        "type": "tool",
        "tool": "track_object",
        "streaming": True,
        "inputs": {},
    }
    raw["subgraphs"]["sg_def"]["edges"] = [
        ["START", "tracker"],
        ["START", "step"],
        ["step", "ok"],
        ["ok", "END"],
    ]
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    sg = wf.subgraphs["sg_def"]
    assert sg.nodes["tracker"].streaming is True
    assert sg.nodes["tracker"].tool == "track_object"


@pytest.mark.parametrize("legacy", [
    {"type": "service", "service": "gripper.v1.Gripper", "method": "Open", "inputs": {}},
    {"type": "skill", "skill": "run_policy", "inputs": {}},
    {"type": "policy", "policy": "libero_pi05", "inputs": {}},
])
def test_legacy_node_types_rejected(tmp_path: Path, legacy: dict) -> None:
    """`type: service`/`skill`/`policy` were retired; the validator surfaces
    a precise migration message naming the rewrite to `type: tool`."""
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["step"] = legacy
    p = _write_workflow(tmp_path, raw)
    with pytest.raises(WorkflowValidationError, match="retired type"):
        load_workflow(p)


def test_rejects_v2_workflow(tmp_path: Path) -> None:
    raw = {
        "version": 2,
        "meta": {},
        "begin": "x",
        "subgraphs": {"x": {"type": "end", "status": "success"}},
    }
    p = _write_workflow(tmp_path, raw)
    with pytest.raises(WorkflowValidationError, match="version"):
        load_workflow(p)


def test_rejects_unknown_node_type(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["bad"] = {"type": "wat"}
    p = _write_workflow(tmp_path, raw)
    with pytest.raises(WorkflowValidationError, match="invalid type"):
        load_workflow(p)


def test_rejects_streaming_on_non_streaming_node_type(tmp_path: Path) -> None:
    """`streaming: true` is only valid on `tool` and `script` nodes."""
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["step"]["streaming"] = True
    raw["subgraphs"]["sg_def"]["nodes"]["step"]["type"] = "noop"
    raw["subgraphs"]["sg_def"]["nodes"]["step"].pop("tool", None)
    p = _write_workflow(tmp_path, raw)
    with pytest.raises(WorkflowValidationError, match="streaming"):
        load_workflow(p)


def test_resolves_ref_objects(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["consumer"] = {
        "type": "tool",
        "tool": "robot.go_home",
        "inputs": {"upstream": {"$ref": "step.field"}},
    }
    raw["subgraphs"]["sg_def"]["edges"] = [
        ["START", "step"],
        ["step", "consumer"],
        ["consumer", "ok"],
        ["ok", "END"],
    ]
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    consumer = wf.subgraphs["sg_def"].nodes["consumer"]
    assert isinstance(consumer.inputs["upstream"], Ref)
    assert consumer.inputs["upstream"].path == "step.field"


# ---------------------------------------------------------------------------
# End-node recovery (ToolCall)
# ---------------------------------------------------------------------------


def test_parses_recovery_tool_calls(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["nodes"]["done"]["recovery"] = [
        {"tool": "robot.open_gripper", "inputs": {"width": 0.08}},
        {"tool": "robot.go_home"},
    ]
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    assert wf.nodes["done"].recovery == (
        ToolCall(tool="robot.open_gripper", inputs={"width": 0.08}),
        ToolCall(tool="robot.go_home", inputs={}),
    )


def test_legacy_recovery_service_method_rejected(tmp_path: Path) -> None:
    """The proto-era `{"service": ..., "method": ...}` recovery form was
    retired; the parser surfaces the tool-name migration message."""
    raw = _minimal_workflow()
    raw["nodes"]["done"]["recovery"] = [
        {"service": "gripper.v1.Gripper", "method": "Open", "inputs": {}},
    ]
    p = _write_workflow(tmp_path, raw)
    with pytest.raises(WorkflowValidationError, match="now name a tool"):
        load_workflow(p)


def test_recovery_requires_string_tool(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["nodes"]["done"]["recovery"] = [{"inputs": {}}]
    p = _write_workflow(tmp_path, raw)
    with pytest.raises(WorkflowValidationError, match="string 'tool'"):
        load_workflow(p)


# ---------------------------------------------------------------------------
# Ref resolution (resolve_ref / _walk_field)
# ---------------------------------------------------------------------------


def test_resolve_ref_walks_plain_dicts() -> None:
    outs = {"step": {"pose": {"position": {"x": 1.5}}}}
    assert resolve_ref(Ref("step.pose.position.x"), outs) == 1.5


def test_resolve_ref_walks_lists_and_negative_indices() -> None:
    outs = {
        "traj": {
            "waypoints": [
                {"positions": [0.0, 0.1]},
                {"positions": [0.2, 0.3]},
            ],
        },
    }
    assert resolve_ref(Ref("traj.waypoints.0.positions"), outs) == [0.0, 0.1]
    assert resolve_ref(Ref("traj.waypoints.-1.positions.1"), outs) == 0.3


def test_resolve_ref_dict_key_wins_over_attribute() -> None:
    """A dict key must shadow the same-named dict method (e.g. "items")."""
    outs = {"step": {"items": [1, 2], "keys": "k"}}
    assert resolve_ref(Ref("step.items"), outs) == [1, 2]
    assert resolve_ref(Ref("step.keys"), outs) == "k"


def test_resolve_ref_attribute_access_on_objects() -> None:
    outs = {"step": SimpleNamespace(score=0.9)}
    assert resolve_ref(Ref("step.score"), outs) == 0.9


def test_resolve_ref_streaming_latest_snapshot() -> None:
    class _Slot:
        def __init__(self, value):
            self._value = value

        def latest(self):
            return self._value

    outs = {"tracker": _Slot({"pose": {"position": {"z": 0.42}}})}
    assert resolve_ref(Ref("tracker.pose.position.z"), outs) == 0.42


def test_resolve_ref_unknown_head_raises() -> None:
    with pytest.raises(WorkflowValidationError, match="unknown node"):
        resolve_ref(Ref("ghost.field"), {"step": {}})


def test_resolve_ref_index_out_of_range_raises() -> None:
    with pytest.raises(WorkflowValidationError, match="out of range"):
        resolve_ref(Ref("step.points.5"), {"step": {"points": [1, 2]}})


def test_resolve_ref_index_on_non_sequence_raises() -> None:
    with pytest.raises(WorkflowValidationError, match="non-sequence"):
        resolve_ref(Ref("step.pose.0"), {"step": {"pose": {"x": 1}}})


def test_resolve_ref_missing_field_raises() -> None:
    with pytest.raises(WorkflowValidationError, match="not found"):
        resolve_ref(Ref("step.nope"), {"step": {"pose": {}}})


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_minimal_workflow_validates(tmp_path: Path) -> None:
    p = _write_workflow(tmp_path, _minimal_workflow())
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], errors


def test_unreachable_node_is_error(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["orphan"] = {
        "type": "tool",
        "tool": "robot.go_home",
        "inputs": {},
    }
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("unreachable" in i.message for i in errors), errors


def test_streaming_node_with_outgoing_edge_is_error(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["tracker"] = {
        "type": "tool",
        "tool": "track_object",
        "streaming": True,
        "inputs": {},
    }
    raw["subgraphs"]["sg_def"]["edges"] = [
        ["START", "tracker"],
        ["tracker", "step"],          # forbidden — streaming has no outgoing
        ["step", "ok"],
        ["ok", "END"],
    ]
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("pure sources" in i.message for i in errors), errors


def test_dangling_edge_target_is_error(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["edges"].append(["step", "nonexistent"])
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("not declared" in i.message for i in errors), errors


def test_conditional_edges_non_router_requires_router_field(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    # Add a conditional edge from `step` (a non-router node) without router_field
    raw["subgraphs"]["sg_def"]["conditional_edges"] = {
        "step": {"router_field": None, "mapping": {"x": "ok"}},
    }
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("router_field" in i.message for i in errors), errors


def test_success_values_empty_is_error(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["exit"]["success_values"] = []
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("success_values" in i.message for i in errors), errors


def test_legacy_values_key_rejected_at_parse_time(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["exit"] = {
        "router_field": None,
        "values": ["ok"],   # legacy key, no longer supported
    }
    p = _write_workflow(tmp_path, raw)
    with pytest.raises(WorkflowValidationError, match="legacy 'values' key"):
        load_workflow(p)


def test_on_error_in_nodes_is_error_S9(tmp_path: Path) -> None:
    """S9: on_error must not collide with a declared node name."""
    raw = _minimal_workflow()
    # Declare a `failed` noop node and set on_error to the same name —
    # the failure exit must live only as the on_error symbol.
    raw["subgraphs"]["sg_def"]["nodes"]["failed"] = {"type": "noop"}
    raw["subgraphs"]["sg_def"]["edges"].append(["failed", "END"])
    raw["subgraphs"]["sg_def"]["edges"].insert(0, ["step", "failed"])
    raw["subgraphs"]["sg_def"]["on_error"] = "failed"
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("S9" in i.message for i in errors), errors


def test_on_error_as_cond_target_is_error_S10(tmp_path: Path) -> None:
    """S10: on_error must not appear as a conditional_edges mapping target."""
    raw = _minimal_workflow()
    # Step routes its `outcome` field to either ok (success node) or
    # `failed` (which is the on_error symbol — illegal as a target).
    raw["subgraphs"]["sg_def"]["conditional_edges"] = {
        "step": {
            "router_field": "outcome",
            "mapping": {"good": "ok", "bad": "failed"},
        },
    }
    raw["subgraphs"]["sg_def"]["on_error"] = "failed"
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("S10" in i.message for i in errors), errors


def test_success_value_missing_noop_when_router_field_null_is_error_S11(
    tmp_path: Path,
) -> None:
    """S11: when router_field is null, every success_value must be a noop node."""
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["exit"]["success_values"] = ["ok", "ghost"]
    # `ghost` is not declared as a node — must error.
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("S11" in i.message and "ghost" in i.message for i in errors), errors


def test_success_value_collides_with_node_when_router_field_set_is_error_S11(
    tmp_path: Path,
) -> None:
    """S11: when router_field is set, success_values must NOT be node names."""
    raw = _minimal_workflow()
    # Switch to router_field="status" mode and use a node-name as a success value.
    raw["subgraphs"]["sg_def"]["exit"] = {
        "router_field": "status",
        "success_values": ["step"],   # `step` is also a real node — illegal.
    }
    # Leave the existing edges alone (test only checks parse → validate path).
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any("S11" in i.message and "step" in i.message for i in errors), errors


# ---------------------------------------------------------------------------
# Declared type names (gap.schema registry)
# ---------------------------------------------------------------------------


def test_unknown_declared_input_type_is_error(tmp_path: Path) -> None:
    """A typo'd subgraph input type surfaces the gap.schema lookup error."""
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["inputs"] = {"cloud": "NotARealType"}
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert any(
        "unknown type name 'NotARealType'" in i.message for i in errors
    ), errors


def test_known_declared_input_type_resolves(tmp_path: Path) -> None:
    """Registered gap.schema names ("PointCloud", "Se3Pose") resolve cleanly;
    only the W8 missing-producer error remains."""
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["inputs"] = {"target_pose": "Se3Pose"}
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    issues = validate_workflow(wf)
    assert not any("unknown type name" in i.message for i in issues), issues
    # The input still has no upstream producer — W8 fires, type check doesn't.
    assert any("W8" in i.message for i in issues), issues


# ---------------------------------------------------------------------------
# Tool registry (schema introspection seam)
# ---------------------------------------------------------------------------


class _FakeToolRegistry:
    """Duck-typed stand-in for gap.tools registry: `in` + `.get(name)`
    returning a descriptor whose `.schema` is UnitSchema-shaped."""

    def __init__(self, tools: dict) -> None:
        self._tools = tools

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str):
        return self._tools[name]


def _unit_schema(inputs: dict, outputs: dict):
    """UnitSchema-shaped object (gap.tools.schema): FieldInfo rows carry
    the original python_type hint."""
    return SimpleNamespace(
        inputs={
            k: SimpleNamespace(name=k, python_type=v, type_str="", required=True)
            for k, v in inputs.items()
        },
        outputs={
            k: SimpleNamespace(name=k, python_type=v, type_str="", required=True)
            for k, v in outputs.items()
        },
    )


def test_unknown_tool_degrades_to_warning(tmp_path: Path) -> None:
    """Tools missing from the registry (e.g. robot.* connector tools not
    yet registered at validate time) must not hard-fail validation."""
    p = _write_workflow(tmp_path, _minimal_workflow())
    wf = load_workflow(p)
    registry = _FakeToolRegistry({})  # `robot.go_home` is not registered
    issues = validate_workflow(wf, tool_registry=registry)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], errors
    warnings = [i for i in issues if i.severity == "warning"]
    assert any("not registered" in i.message for i in warnings), issues


def test_registered_tool_schema_introspects_without_issues(tmp_path: Path) -> None:
    p = _write_workflow(tmp_path, _minimal_workflow())
    wf = load_workflow(p)
    registry = _FakeToolRegistry({
        "robot.go_home": SimpleNamespace(
            schema=_unit_schema(inputs={"speed": float}, outputs={"ok": bool}),
        ),
    })
    issues = validate_workflow(wf, tool_registry=registry)
    assert issues == [], issues


def test_no_tool_registry_skips_tool_schema_checks(tmp_path: Path) -> None:
    p = _write_workflow(tmp_path, _minimal_workflow())
    wf = load_workflow(p)
    issues = validate_workflow(wf, tool_registry=None)
    assert issues == [], issues


# ---------------------------------------------------------------------------
# Skill registry (S4 streaming contract)
# ---------------------------------------------------------------------------


class _FakeSkillRegistry:
    """Duck-typed skills loader surface: `in` + `.get(name)` returning an
    object with `.meta` carrying the streaming contract."""

    def __init__(self, skills: dict) -> None:
        self._skills = skills

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def get(self, name: str):
        return self._skills[name]


def _streaming_workflow() -> dict:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["tracker"] = {
        "type": "tool",
        "tool": "track_object",
        "streaming": True,
        "inputs": {},
    }
    raw["subgraphs"]["sg_def"]["edges"] = [
        ["START", "tracker"],
        ["START", "step"],
        ["step", "ok"],
        ["ok", "END"],
    ]
    return raw


def test_streaming_flag_matching_contract_ok_S4(tmp_path: Path) -> None:
    p = _write_workflow(tmp_path, _streaming_workflow())
    wf = load_workflow(p)
    skills = _FakeSkillRegistry({
        "track_object": SimpleNamespace(
            meta=SimpleNamespace(contract={"streaming": True}),
        ),
    })
    issues = validate_workflow(wf, skill_registry=skills)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], errors


def test_streaming_flag_contract_mismatch_is_error_S4(tmp_path: Path) -> None:
    p = _write_workflow(tmp_path, _streaming_workflow())
    wf = load_workflow(p)
    skills = _FakeSkillRegistry({
        "track_object": SimpleNamespace(
            meta=SimpleNamespace(contract={"streaming": False}),
        ),
    })
    issues = validate_workflow(wf, skill_registry=skills)
    errors = [i for i in issues if i.severity == "error"]
    assert any("S4" in i.message for i in errors), issues


def test_streaming_contract_without_flag_is_error_S4(tmp_path: Path) -> None:
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["nodes"]["step"]["tool"] = "track_object"
    p = _write_workflow(tmp_path, raw)
    wf = load_workflow(p)
    skills = _FakeSkillRegistry({
        "track_object": SimpleNamespace(
            meta=SimpleNamespace(contract={"streaming": True}),
        ),
    })
    issues = validate_workflow(wf, skill_registry=skills)
    errors = [i for i in issues if i.severity == "error"]
    assert any("S4" in i.message for i in errors), issues


# ---------------------------------------------------------------------------
# Real example workflows
# ---------------------------------------------------------------------------

EXAMPLES = [
    "examples/libero_quickstart/graph/workflow.json",
    "examples/libero_quickstart/graph_planner/workflow.json",
]


@pytest.mark.parametrize("path", EXAMPLES)
def test_real_example_validates(path: str) -> None:
    """All canonical example workflows must load and structurally validate."""
    repo_root = Path(__file__).resolve().parents[2]
    full = repo_root / path
    if not full.exists():
        pytest.skip(f"example not present: {path}")
    wf = load_workflow(full)
    issues = validate_workflow(wf)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], f"{path}: {[(i.node_id, i.message) for i in errors]}"


# ---------------------------------------------------------------------------
# W8: inputs are bound per call site, in execution order
# ---------------------------------------------------------------------------


def test_w8_accepts_an_input_passed_at_the_call_site(tmp_path: Path) -> None:
    """A literal on the calling node satisfies the input it names.

    WORKFLOW_FORMAT tells authors that literals are passed as plain JSON, and
    the executor binds them; a rule that ignored them refused graphs that run.
    """
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["inputs"] = {"arm_id": "int"}
    raw["nodes"]["sg"]["inputs"] = {"arm_id": 0}
    issues = validate_workflow(load_workflow(_write_workflow(tmp_path, raw)))
    assert not any("W8" in i.message for i in issues), issues


def test_w8_rejects_a_subgraph_that_produces_its_own_input(tmp_path: Path) -> None:
    """Self-production is not production: the executor binds from subgraphs
    that have already run, and a subgraph has not run when its inputs bind."""
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["inputs"] = {"arm_id": "int"}
    raw["subgraphs"]["sg_def"]["outputs"] = {"arm_id": {"$ref": "step.arm_id"}}
    issues = validate_workflow(load_workflow(_write_workflow(tmp_path, raw)))
    assert any("W8" in i.message for i in issues), issues


def test_w8_rejects_a_producer_that_runs_after_the_consumer(tmp_path: Path) -> None:
    """A producer downstream of the consumer satisfies "exists" but not
    "has run" -- the shape that aborted before the first node."""
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["inputs"] = {"grasp_pose": "Se3Pose"}
    raw["subgraphs"]["later_def"] = {
        "skill": "generic",
        "inputs": {},
        "outputs": {"grasp_pose": {"$ref": "step.pose"}},
        "nodes": {
            "step": {"type": "tool", "tool": "robot.go_home", "inputs": {}},
            "ok": {"type": "noop"},
        },
        "edges": [["START", "step"], ["step", "ok"], ["ok", "END"]],
        "conditional_edges": {},
        "exit": {"router_field": None, "success_values": ["ok"]},
    }
    raw["nodes"]["later"] = {"type": "subgraph", "ref": "later_def"}
    raw["conditional_edges"]["sg"]["mapping"]["ok"] = "later"
    raw["conditional_edges"]["later"] = {
        "router_field": "exit", "mapping": {"ok": "done"},
    }
    issues = validate_workflow(load_workflow(_write_workflow(tmp_path, raw)))
    assert any("W8" in i.message for i in issues), issues


def test_w8_accepts_a_producer_upstream_of_the_consumer(tmp_path: Path) -> None:
    """The same two subgraphs in the order that works."""
    raw = _minimal_workflow()
    raw["subgraphs"]["sg_def"]["inputs"] = {"grasp_pose": "Se3Pose"}
    raw["subgraphs"]["first_def"] = {
        "skill": "generic",
        "inputs": {},
        "outputs": {"grasp_pose": {"$ref": "step.pose"}},
        "nodes": {
            "step": {"type": "tool", "tool": "robot.go_home", "inputs": {}},
            "ok": {"type": "noop"},
        },
        "edges": [["START", "step"], ["step", "ok"], ["ok", "END"]],
        "conditional_edges": {},
        "exit": {"router_field": None, "success_values": ["ok"]},
    }
    raw["nodes"]["first"] = {"type": "subgraph", "ref": "first_def"}
    raw["edges"] = [["START", "first"]]
    raw["conditional_edges"] = {
        "first": {"router_field": "exit", "mapping": {"ok": "sg"}},
        "sg": {"router_field": "exit", "mapping": {"ok": "done"}},
    }
    issues = validate_workflow(load_workflow(_write_workflow(tmp_path, raw)))
    assert not any("W8" in i.message for i in issues), issues
