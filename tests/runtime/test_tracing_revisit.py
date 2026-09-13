"""A node visited twice is closed twice, and the table shows how the last visit ended."""

from __future__ import annotations

import json

from gap.runtime.tracing import DagTrace


def _table(tmp_path) -> dict[str, dict]:
    doc = json.loads((tmp_path / "dag_trace.json").read_text())
    return {node["name"]: node for node in doc["nodes"]}


def test_a_revisited_node_that_fails_is_marked_error(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("descend", {"script": "descend.py"})
    trace.start_node("descend")
    trace.end_node("descend", True)
    # Second visit, through a retry edge: the node fails this time.
    trace.start_node("descend")
    trace.record_error("descend", "Node 'descend' failed: TypeError: boom")
    trace.end_node("descend", False)
    trace.flush()

    node = _table(tmp_path)["descend"]
    assert node["status"] == "error"
    assert node["visits"] == 2
    assert "boom" in node["error_message"]
    finished = [e for e in json.loads((tmp_path / "dag_trace.json").read_text())["events"]
                if e["event_type"] == "node_finished" and e["node_name"] == "descend"]
    assert [e["status"] for e in finished] == ["ok", "error"]


def test_a_revisited_node_that_succeeds_drops_its_earlier_error(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("grasp", {"script": "grasp.py"})
    trace.start_node("grasp")
    trace.record_error("grasp", "Node 'grasp' failed: RuntimeError: missed")
    trace.end_node("grasp", False)
    trace.start_node("grasp")
    trace.end_node("grasp", True)
    trace.flush()

    node = _table(tmp_path)["grasp"]
    assert node["status"] == "ok"
    assert node["error_message"] is None


def test_a_single_visit_is_closed_once(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("once", {"script": "once.py"})
    trace.start_node("once")
    trace.end_node("once", True)
    trace.end_node("once", False)  # a second close in the same visit is ignored
    trace.flush()
    assert _table(tmp_path)["once"]["status"] == "ok"
