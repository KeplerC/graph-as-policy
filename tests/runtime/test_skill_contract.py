"""SK1-SK4: a subgraph naming a bundle is checked against the bundle.

Before these rules, exit reconciliation ran only through the codegen
pipeline's ``agent_registry`` and nothing looked at what a script node
bound against the canonical script's ``run()``: a subgraph that named a
bundle, invented an exit, and forgot the script's one required input
validated clean and failed on the node's first execution. The bundle here
is a scratch checkout loaded through the real registry loader, so the
duck-typing the validator relies on (``meta.exit_conditions``,
``canonical_scripts[...].bundle_relative/module/schema``) is the loader's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gap.runtime.validate import validate_workflow
from gap.runtime.workflow import load_workflow
from gap.skills import load_registry_set

BUNDLE = "test-mating-sk"

_SKILL_MD = f"""\
---
name: {BUNDLE}
description: Execute a supplied mating plan for a held object. Use when a validator test needs a bundle with an exit vocabulary and one canonical script.
compatibility: requires gap>=0.1
metadata: {{category: testing, tags: [fixture]}}
gap:
  allowed_tools: [robot.get_ee_pose]
  exit_conditions:
    seated: The plan completed.
    blocked: A waypoint was refused.
  required_inputs: {{plan: dict}}
  produces_outputs: {{final_pose: dict}}
  canonical_scripts:
    - execute: scripts/execute.py
  streaming: false
---

# {BUNDLE}
"""

_SCRIPT = '''\
from typing import Any, TypedDict


class Out(TypedDict):
    final_pose: dict[str, Any]


def run(ctx, plan: dict[str, Any], retries: int = 1) -> Out:
    return {"final_pose": {}}
'''


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("registry")
    bundle = root / "skills" / BUNDLE
    (bundle / "scripts").mkdir(parents=True)
    (bundle / "SKILL.md").write_text(_SKILL_MD)
    (bundle / "scripts" / "execute.py").write_text(_SCRIPT)
    return load_registry_set([root])


def _workflow(
    *,
    success: list[str] = ("seated",),
    on_error: str = "blocked",
    inputs: dict | None = None,
    script: str = "scripts/execute.py",
    outputs: dict | None = None,
    router_field: str | None = None,
) -> dict:
    """With *router_field* null (the default) each success value is a noop
    terminal node; with it set, the success values are strings the script
    returns under that field and the script itself is the terminal node."""
    success = list(success)
    nodes = {
        "execute": {"type": "script", "script": script, "inputs": {"plan": {}} if inputs is None else inputs},
    }
    if router_field is None:
        nodes.update({sv: {"type": "noop"} for sv in success})
        edges = [["START", "execute"], *[["execute", sv] for sv in success[:1]], *[[sv, "END"] for sv in success]]
    else:
        edges = [["START", "execute"], ["execute", "END"]]
    return {
        "version": 3,
        "meta": {},
        "nodes": {
            "mate": {"type": "subgraph", "ref": "mate_sg"},
            "done": {"type": "end", "status": "success"},
            "abort": {"type": "end", "status": "failure"},
        },
        "edges": [["START", "mate"]],
        "conditional_edges": {
            "mate": {"router_field": "exit", "mapping": {**{sv: "done" for sv in success}, on_error: "abort"}},
        },
        "subgraphs": {
            "mate_sg": {
                "skill": BUNDLE,
                "inputs": {},
                "outputs": {"final_pose": {"$ref": "execute.final_pose"}} if outputs is None else outputs,
                "nodes": nodes,
                "edges": edges,
                "conditional_edges": {},
                "exit": {"router_field": router_field, "success_values": success},
                "on_error": on_error,
            },
        },
    }


def _validate(tmp_path: Path, raw: dict, registry) -> tuple[list[str], list[str]]:
    (tmp_path / "workflow.json").write_text(json.dumps(raw))
    wf = load_workflow(tmp_path / "workflow.json")
    issues = validate_workflow(wf, skill_registry=registry)
    errors = [i.message for i in issues if i.severity == "error"]
    warnings = [i.message for i in issues if i.severity != "error"]
    return errors, warnings


def test_a_well_wired_bundle_subgraph_is_clean(tmp_path, registry):
    """The script is not beside the graph -- it resolves from the bundle at
    run time -- and the contract is still checked, from the registry."""
    errors, _ = _validate(tmp_path, _workflow(), registry)
    assert errors == [], errors


def test_omitting_a_required_input_is_an_error_SK3(tmp_path, registry):
    errors, _ = _validate(tmp_path, _workflow(inputs={}), registry)
    assert any("SK3" in e and "'plan'" in e for e in errors), errors


def test_binding_an_input_run_does_not_accept_is_an_error_SK3(tmp_path, registry):
    errors, _ = _validate(tmp_path, _workflow(inputs={"plan": {}, "target": {}}), registry)
    assert any("SK3" in e and "'target'" in e for e in errors), errors


def test_an_optional_parameter_may_be_left_unbound_SK3(tmp_path, registry):
    errors, _ = _validate(tmp_path, _workflow(inputs={"plan": {}}), registry)
    assert not any("SK3" in e for e in errors), errors


def test_a_returned_exit_the_bundle_does_not_declare_is_an_error_SK1(tmp_path, registry):
    """With ``router_field`` set the success values are what the bundle's
    script returns: an undeclared one is an outcome it never produces."""
    errors, _ = _validate(tmp_path, _workflow(success=["done"], router_field="status"), registry)
    assert any("SK1" in e and "'done'" in e for e in errors), errors


def test_a_terminal_node_outside_the_vocabulary_is_a_warning_SK1(tmp_path, registry):
    """With ``router_field`` null the success values are the subgraph's own
    noop nodes (S11), the graph's symbols like ``on_error``: the shipped
    tipping graph names a ``bad_start`` node its bundle's SKILL.md does not
    declare, and the parent's mapping routes it. Drift is reported, not
    fatal."""
    errors, warnings = _validate(tmp_path, _workflow(success=["done"]), registry)
    assert not any("SK1" in e for e in errors), errors
    assert any("SK1" in w and "'done'" in w for w in warnings), warnings


def test_covering_a_subset_of_the_declared_exits_is_not_an_issue_SK1(tmp_path, registry):
    """A bundle split one script per subgraph covers that script's exits."""
    errors, warnings = _validate(tmp_path, _workflow(success=["seated"], on_error="blocked"), registry)
    assert not any("SK1" in m for m in errors + warnings), (errors, warnings)


def test_an_on_error_outside_the_vocabulary_is_a_warning_SK1(tmp_path, registry):
    errors, warnings = _validate(tmp_path, _workflow(on_error="errored"), registry)
    assert not any("SK1" in e for e in errors), errors
    assert any("SK1" in w and "'errored'" in w for w in warnings), warnings


def test_a_bundle_label_on_a_subgraph_running_none_of_its_scripts_is_an_error_SK2(tmp_path, registry):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "own.py").write_text("def run(ctx, plan: dict) -> None:\n    return None\n")
    errors, _ = _validate(tmp_path, _workflow(script="scripts/own.py", outputs={}), registry)
    assert any("SK2" in e for e in errors), errors


def test_an_output_bound_to_a_field_the_script_does_not_return_is_an_error_SK4(tmp_path, registry):
    errors, _ = _validate(tmp_path, _workflow(outputs={"pose": {"$ref": "execute.pose"}}), registry)
    assert any("SK4" in e and "'pose'" in e for e in errors), errors


def test_a_script_beside_the_graph_is_the_contract_that_binds_SK3(tmp_path, registry):
    """Beside the graph first: a graph shipping its own ``execute.py`` runs
    that one, so its signature -- not the bundle's -- is what the node
    must satisfy."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "execute.py").write_text(
        "from typing import TypedDict\n\nclass Out(TypedDict):\n    pose: dict\n\n"
        "def run(ctx, target: dict) -> Out:\n    return {'pose': {}}\n"
    )
    errors, _ = _validate(
        tmp_path, _workflow(inputs={"target": {}}, outputs={"pose": {"$ref": "execute.pose"}}), registry
    )
    assert errors == [], errors
    errors, _ = _validate(tmp_path, _workflow(inputs={"plan": {}}), registry)
    assert any("SK3" in e and "'target'" in e for e in errors), errors


def test_a_generic_subgraph_gets_no_contract_check(tmp_path, registry):
    raw = _workflow(success=["done"], inputs={})
    raw["subgraphs"]["mate_sg"]["skill"] = "generic"
    raw["subgraphs"]["mate_sg"]["outputs"] = {}
    errors, warnings = _validate(tmp_path, raw, registry)
    assert not any("SK" in m for m in errors + warnings), (errors, warnings)
