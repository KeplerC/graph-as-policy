"""Tests for gap.runtime.tracing — asset extraction, payload serialization, DagTrace."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image as PILImage

from gap.runtime.tracing import DagTrace, _extract_assets, _serialize_value


def _camera_frame(name: str = "agentview", h: int = 16, w: int = 12) -> dict:
    rng = np.random.default_rng(0)
    return {
        "name": name,
        "rgb": rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8),
        "depth": rng.random((h, w), dtype=np.float32),
        "intrinsics": np.eye(3, dtype=np.float64),
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


# ---------------------------------------------------------------------------
# Per-type asset extractors
# ---------------------------------------------------------------------------


def test_image_asset_saved_as_png(tmp_path):
    arr = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)
    assets = _extract_assets(arr, tmp_path, "output_frame")
    assert assets == ["assets/output_frame_rgb.png"]
    img = np.asarray(PILImage.open(tmp_path / "output_frame_rgb.png"))
    np.testing.assert_array_equal(img, arr)


def test_image_asset_no_prefix(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    assets = _extract_assets(arr, tmp_path, "")
    assert assets == ["assets/rgb.png"]
    assert (tmp_path / "rgb.png").exists()


def test_mask_asset_saved_as_grayscale_png(tmp_path):
    arr = np.zeros((5, 4), dtype=np.uint8)
    arr[1:3, 1:3] = 255
    assets = _extract_assets(arr, tmp_path, "output_m")
    assert assets == ["assets/output_m_mask.png"]
    img = PILImage.open(tmp_path / "output_m_mask.png")
    assert img.mode == "L"
    np.testing.assert_array_equal(np.asarray(img), arr)


def test_bool_mask_scaled_to_255(tmp_path):
    arr = np.zeros((5, 4), dtype=bool)
    arr[0, 0] = True
    assets = _extract_assets(arr, tmp_path, "output_m")
    assert assets == ["assets/output_m_mask.png"]
    img = np.asarray(PILImage.open(tmp_path / "output_m_mask.png"))
    assert img[0, 0] == 255
    assert img[4, 3] == 0


def test_empty_mask_skipped(tmp_path):
    arr = np.zeros((0, 0), dtype=np.uint8)
    assert _extract_assets(arr, tmp_path, "output_m") == []
    assert list(tmp_path.iterdir()) == []


def test_depth_asset_saved_as_npy(tmp_path):
    arr = np.random.default_rng(1).random((5, 4)).astype(np.float32)
    assets = _extract_assets(arr, tmp_path, "output_d")
    assert assets == ["assets/output_d_depth.npy"]
    np.testing.assert_array_equal(np.load(tmp_path / "output_d_depth.npy"), arr)


def test_pointcloud_saved_as_npz_with_positions_and_colors(tmp_path):
    pts = np.random.default_rng(2).random((10, 3)).astype(np.float32)
    cols = np.random.default_rng(3).random((10, 3)).astype(np.float32)
    cloud = {"points": pts, "colors": cols}
    assets = _extract_assets(cloud, tmp_path, "output_pc")
    assert assets == ["assets/output_pc_cloud.npz"]
    data = np.load(tmp_path / "output_pc_cloud.npz")
    assert set(data.keys()) == {"positions", "colors"}
    np.testing.assert_array_equal(data["positions"], pts)
    assert data["colors"].dtype == np.uint8
    np.testing.assert_array_equal(
        data["colors"], (cols * 255).clip(0, 255).astype(np.uint8)
    )


def test_pointcloud_without_colors(tmp_path):
    cloud = {"points": np.ones((4, 3), dtype=np.float32)}
    assets = _extract_assets(cloud, tmp_path, "output_pc")
    assert assets == ["assets/output_pc_cloud.npz"]
    data = np.load(tmp_path / "output_pc_cloud.npz")
    assert set(data.keys()) == {"positions"}


def test_empty_pointcloud_skipped(tmp_path):
    cloud = {"points": np.zeros((0, 3), dtype=np.float32)}
    assert _extract_assets(cloud, tmp_path, "output_pc") == []


def test_camera_frame_per_field_naming(tmp_path):
    cam = _camera_frame()
    assets = _extract_assets(cam, tmp_path, "output_frame")
    assert sorted(assets) == [
        "assets/output_frame_depth_depth.npy",
        "assets/output_frame_rgb_rgb.png",
    ]
    # float64 intrinsics and the pose dict must not produce assets.
    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "output_frame_depth_depth.npy",
        "output_frame_rgb_rgb.png",
    ]


def test_camera_list_uses_camera_name_in_prefix(tmp_path):
    obs = {"cameras": [_camera_frame("agentview"), _camera_frame("wrist")], "arms": []}
    assets = _extract_assets(obs, tmp_path, "output_obs")
    assert sorted(assets) == [
        "assets/output_obs_cameras_agentview_depth_depth.npy",
        "assets/output_obs_cameras_agentview_rgb_rgb.png",
        "assets/output_obs_cameras_wrist_depth_depth.npy",
        "assets/output_obs_cameras_wrist_rgb_rgb.png",
    ]


def test_list_items_fall_back_to_index(tmp_path):
    masks = [np.full((4, 4), 255, dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)]
    assets = _extract_assets(masks, tmp_path, "output_masks")
    assert assets == [
        "assets/output_masks_0_mask.png",
        "assets/output_masks_1_mask.png",
    ]


# ---------------------------------------------------------------------------
# Payload serialization
# ---------------------------------------------------------------------------


def test_serialize_small_array_listed():
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    assert _serialize_value(arr) == [[0, 1, 2], [3, 4, 5]]


def test_serialize_large_array_summarized():
    arr = np.zeros((10, 10), dtype=np.float32)  # size 100 — not < 100
    assert _serialize_value(arr) == "<ndarray shape=(10, 10) dtype=float32>"


def test_serialize_bytes_summarized():
    assert _serialize_value(b"abcde") == "<bytes len=5>"


def test_serialize_nested_structures():
    payload = {
        "a": [1, 2.5, "x", None, True],
        "b": {"raw": b"\x00\x01", "img": np.zeros((20, 20, 3), dtype=np.uint8)},
        "c": (np.int64(7), np.float32(0.5)),
    }
    out = _serialize_value(payload)
    assert out["a"] == [1, 2.5, "x", None, True]
    assert out["b"]["raw"] == "<bytes len=2>"
    assert out["b"]["img"] == "<ndarray shape=(20, 20, 3) dtype=uint8>"
    assert out["c"][0] == 7
    assert out["c"][1] == pytest.approx(0.5)
    # The whole thing must round-trip through json.
    json.dumps(out)


# ---------------------------------------------------------------------------
# DagTrace end-to-end
# ---------------------------------------------------------------------------


def test_dag_trace_end_to_end(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("grab_frame", {"type": "tool", "tool": "camera.get_frame"})
    trace.add_node("segment", {"type": "tool", "tool": "sam.segment_box"})
    trace.add_edge("grab_frame", "segment")

    trace.start_node("grab_frame")
    trace.record_resolved_inputs("grab_frame", {"camera_name": "agentview"})
    trace.record_output("grab_frame", {"frame": _camera_frame()})
    trace.end_node("grab_frame", True)
    trace.flush()

    # dag_trace.json structure
    data = json.loads((tmp_path / "dag_trace.json").read_text())
    assert set(data.keys()) == {"nodes", "edges", "events"}

    node = data["nodes"][0]
    assert set(node.keys()) == {
        "id", "name", "node_type", "service", "method", "script",
        "started_at", "finished_at", "duration_ms", "status",
        "condition", "condition_result", "error_message",
        "has_inputs", "has_output", "visits", "assets",
    }
    assert node["id"] == "n_0000"
    assert node["name"] == "grab_frame"
    assert node["node_type"] == "tool"
    assert node["status"] == "ok"
    assert node["has_inputs"] is True
    assert node["has_output"] is True
    assert node["visits"] == 1
    assert node["duration_ms"] >= 0.0
    assert sorted(node["assets"]) == [
        "assets/output_frame_depth_depth.npy",
        "assets/output_frame_rgb_rgb.png",
    ]

    edge = data["edges"][0]
    assert edge["from_id"] == "n_0000"
    assert edge["to_id"] == "n_0001"
    assert edge["from"] == "n_0000"
    assert edge["to"] == "n_0001"

    event_types = [e["event_type"] for e in data["events"]]
    assert event_types == [
        "node_registered", "node_registered", "edge_registered",
        "node_started", "inputs_recorded", "output_recorded", "node_finished",
    ]
    assert [e["seq"] for e in data["events"]] == list(range(len(data["events"])))

    # node_data files
    node_dir = tmp_path / "node_data" / "grab_frame"
    inputs = json.loads((node_dir / "resolved_inputs.json").read_text())
    assert inputs == {"camera_name": "agentview"}
    output = json.loads((node_dir / "output.json").read_text())
    assert output["frame"]["name"] == "agentview"
    assert output["frame"]["rgb"] == "<ndarray shape=(16, 12, 3) dtype=uint8>"
    assert (node_dir / "assets" / "output_frame_rgb_rgb.png").exists()
    assert (node_dir / "assets" / "output_frame_depth_depth.npy").exists()


def test_record_request_writes_json_and_meta(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("grab_frame", {"type": "tool", "tool": "camera.get_frame"})
    trace.record_request("grab_frame", "camera.get_frame", {"camera_name": "agentview"})

    node_dir = tmp_path / "node_data" / "grab_frame"
    request = json.loads((node_dir / "request.json").read_text())
    assert request == {"camera_name": "agentview"}
    meta = json.loads((node_dir / "request.meta.json").read_text())
    assert meta == {"tool": "camera.get_frame"}


def test_record_subcall_layout(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("servo", {"type": "script", "script": "servo.py"})
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)
    trace.record_subcall(
        "servo", 0, "sam.segment_box",
        {"image": img, "box": [0, 0, 4, 4]},
        {"mask": mask, "score": 0.9},
    )

    call_dir = tmp_path / "node_data" / "servo" / "calls" / "000_segment_box"
    assert json.loads((call_dir / "request.meta.json").read_text()) == {
        "tool": "sam.segment_box"
    }
    request = json.loads((call_dir / "request.json").read_text())
    assert request["box"] == [0, 0, 4, 4]
    response = json.loads((call_dir / "response.json").read_text())
    assert response["score"] == 0.9
    assert (call_dir / "assets" / "request_image_rgb.png").exists()
    assert (call_dir / "assets" / "response_mask_mask.png").exists()


def test_record_stream_read_layout(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("servo", {"type": "script", "script": "servo.py"})
    obs = {"cameras": [_camera_frame()], "arms": []}
    trace.record_stream_read("servo", 0, "observation_stream", obs, 12.5)

    read_dir = tmp_path / "node_data" / "servo" / "stream_reads" / "000_observation_stream"
    value = json.loads((read_dir / "value.json").read_text())
    assert value["cameras"][0]["name"] == "agentview"
    meta = json.loads((read_dir / "meta.json").read_text())
    assert meta == {"stream": "observation_stream", "sampled_at": 12.5}


def test_stream_read_sampling_knob(tmp_path, monkeypatch):
    trace = DagTrace(tmp_path)
    trace.add_node("servo", {"type": "script"})

    monkeypatch.setenv("GAP_TRACE_STREAM_SAMPLE_N", "2")
    for seq in range(4):
        trace.record_stream_read("servo", seq, "obs", {"n": seq}, float(seq))
    reads_dir = tmp_path / "node_data" / "servo" / "stream_reads"
    assert sorted(d.name for d in reads_dir.iterdir()) == ["000_obs", "002_obs"]

    monkeypatch.setenv("GAP_TRACE_STREAM_SAMPLE_N", "0")
    trace.record_stream_read("servo", 4, "obs", {"n": 4}, 4.0)
    assert not (reads_dir / "004_obs").exists()


def test_skip_error_and_condition(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("check", {"type": "tool", "tool": "vlm.validate",
                             "condition": {"field": "ok", "equals": True}})
    trace.add_node("recover", {"type": "noop"})
    trace.record_condition("check", actual=False, expected=True, met=False)
    trace.record_error("check", "validation rejected")
    trace.skip_node("recover")
    trace.flush()

    data = json.loads((tmp_path / "dag_trace.json").read_text())
    check = data["nodes"][0]
    assert check["condition"] == {"field": "ok", "equals": True}
    assert check["condition_result"] == {"actual": False, "expected": True, "met": False}
    assert check["error_message"] == "validation rejected"
    assert data["nodes"][1]["status"] == "skipped"


def test_trace_dir_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("GAP_TRACE_DIR", str(tmp_path / "trace_out"))
    trace = DagTrace()
    trace.flush()
    assert (tmp_path / "trace_out" / "dag_trace.json").exists()


def test_copy_workflow(tmp_path):
    workflow_dir = tmp_path / "wf"
    workflow_dir.mkdir()
    (workflow_dir / "workflow.json").write_text('{"nodes": {}}')
    scripts = workflow_dir / "scripts"
    scripts.mkdir()
    (scripts / "servo.py").write_text("print('hi')\n")

    out = tmp_path / "out"
    trace = DagTrace(out)
    trace.copy_workflow(workflow_dir)
    assert (out / "workflow.json").read_text() == '{"nodes": {}}'
    assert (out / "scripts" / "servo.py").exists()
