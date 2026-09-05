"""Enriched DAG trace recording — captures full node I/O, conditions, and tool calls.

Outputs:
  - dag_trace.json   — enriched node metadata, timing, status, condition results
  - workflow.json     — copy of the executed workflow
  - node_data/<id>/   — per-node resolved inputs, outputs, tool requests, assets
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NodeTrace:
    id: str
    name: str
    node_type: str = ""          # "tool" | "script" | "noop" | ...
    tool: str | None = None
    script: str | None = None
    # Resolved dispatch target (only populated when a `type: tool` node
    # resolves to a specific backend; the resolved `service`+`method`
    # are recorded so the trace viewer can show the full dispatch path
    # even though the workflow.json carries only the flat tool name).
    service: str | None = None
    method: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    status: str = "pending"      # pending | running | ok | error | skipped
    condition: dict | None = None
    condition_result: dict | None = None
    error_message: str | None = None
    has_inputs: bool = False
    has_output: bool = False
    visits: int = 0
    """Times this node ran in the episode. A backward edge is the only loop
    the format has and it carries no bound, so a retry ladder and a livelock
    look identical from the outside -- until the count is printed. Runs that
    spun a graph thousands of times reported only that the simulation bus had
    closed, which reads as infrastructure trouble rather than as a policy that
    never stopped trying."""
    assets: list[str] = field(default_factory=list)


@dataclass
class TraceEvent:
    seq: int
    event_type: str
    timestamp: float
    node_id: str = ""
    node_name: str = ""
    status: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeTrace:
    from_name: str
    to_name: str
    from_id: str | None = None
    to_id: str | None = None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_value(val: Any) -> Any:
    """Serialize a value for JSON output, handling numpy and nested structures."""
    if val is None:
        return None
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if isinstance(val, bytes):
        return f"<bytes len={len(val)}>"
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, np.ndarray):
        if val.size < 100:
            return val.tolist()
        return f"<ndarray shape={val.shape} dtype={val.dtype}>"
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    return str(val)


# ---------------------------------------------------------------------------
# Asset extraction — structural type checks over the gap.types shapes
# ---------------------------------------------------------------------------

def _is_image(arr: np.ndarray) -> bool:
    """RGB image per gap.types.CameraFrame: uint8 [H, W, 3]."""
    return arr.ndim == 3 and arr.shape[-1] == 3 and arr.dtype == np.uint8


def _is_mask(arr: np.ndarray) -> bool:
    """Mask per gap.types.Mask: uint8 (or bool) [H, W]."""
    return arr.ndim == 2 and arr.dtype in (np.uint8, np.bool_)


def _is_depth(arr: np.ndarray) -> bool:
    """Depth image per gap.types.CameraFrame: float32 [H, W], meters."""
    return arr.ndim == 2 and arr.dtype == np.float32


def _is_pointcloud(obj: Any) -> bool:
    """PointCloud per gap.types: dict with "points" float [N, 3] (+ optional colors)."""
    if not isinstance(obj, dict) or "points" not in obj:
        return False
    pts = obj["points"]
    return (
        isinstance(pts, np.ndarray)
        and pts.ndim == 2
        and pts.shape[-1] == 3
        and np.issubdtype(pts.dtype, np.floating)
    )


def _camera_frame_name(obj: Any) -> str | None:
    """Return the camera name when ``obj`` looks like a gap.types.CameraFrame."""
    if (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and obj["name"]
        and (isinstance(obj.get("rgb"), np.ndarray) or isinstance(obj.get("depth"), np.ndarray))
    ):
        return obj["name"]
    return None


def _extract_assets(value: Any, assets_dir: Path, prefix: str = "") -> list[str]:
    """Walk a dict/list/ndarray payload and extract images/masks/point clouds as files.

    Returns list of relative paths (relative to node_data/<id>/).
    """
    if value is None:
        return []

    assets: list[str] = []

    try:
        if isinstance(value, np.ndarray):
            if _is_image(value):
                assets += _save_image_asset(value, assets_dir, prefix)
            elif _is_mask(value):
                assets += _save_mask_asset(value, assets_dir, prefix)
            elif _is_depth(value):
                assets += _save_depth_asset(value, assets_dir, prefix)
        elif _is_pointcloud(value):
            assets += _save_pointcloud_asset(value, assets_dir, prefix)
        elif isinstance(value, dict):
            # Recurse into sub-dicts (CameraFrame rgb/depth fields land here
            # with per-field prefixes mirroring the proto walk's naming).
            for key, child in value.items():
                child_prefix = f"{prefix}_{key}" if prefix else str(key)
                assets += _extract_assets(child, assets_dir, child_prefix)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                # CameraFrame list items are addressed by camera name so
                # asset filenames stay stable across camera reordering.
                suffix = _camera_frame_name(item) or str(i)
                child_prefix = f"{prefix}_{suffix}" if prefix else suffix
                assets += _extract_assets(item, assets_dir, child_prefix)
    except Exception:
        logger.debug("Asset extraction failed at %r", prefix, exc_info=True)

    return assets


def _save_image_asset(arr: np.ndarray, assets_dir: Path, prefix: str) -> list[str]:
    try:
        from PIL import Image as PILImage
        if arr.size == 0:
            return []
        fname = f"{prefix}_rgb.png" if prefix else "rgb.png"
        path = assets_dir / fname
        PILImage.fromarray(np.ascontiguousarray(arr), mode="RGB").save(path)
        return [f"assets/{fname}"]
    except Exception:
        logger.debug("Failed to save image asset", exc_info=True)
        return []


def _save_mask_asset(arr: np.ndarray, assets_dir: Path, prefix: str) -> list[str]:
    try:
        from PIL import Image as PILImage
        h, w = arr.shape
        # When perception finds nothing it still emits a Mask for
        # consistency, but zero-sized. PIL's PNG encoder raises "tile
        # cannot extend outside image" when given a 0-dimensional array;
        # skip the asset write rather than spamming the trace with a
        # stack trace from a benign empty mask.
        if h <= 0 or w <= 0:
            return []
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8) * 255
        fname = f"{prefix}_mask.png" if prefix else "mask.png"
        path = assets_dir / fname
        PILImage.fromarray(np.ascontiguousarray(arr), mode="L").save(path)
        return [f"assets/{fname}"]
    except Exception:
        logger.debug("Failed to save mask asset", exc_info=True)
        return []


def _save_depth_asset(arr: np.ndarray, assets_dir: Path, prefix: str) -> list[str]:
    try:
        if arr.size == 0:
            return []
        fname = f"{prefix}_depth.npy" if prefix else "depth.npy"
        path = assets_dir / fname
        np.save(path, arr)
        return [f"assets/{fname}"]
    except Exception:
        logger.debug("Failed to save depth asset", exc_info=True)
        return []


def _save_pointcloud_asset(cloud: dict, assets_dir: Path, prefix: str) -> list[str]:
    try:
        pts = np.asarray(cloud["points"], dtype=np.float32).reshape(-1, 3)
        if len(pts) == 0:
            return []
        fname = f"{prefix}_cloud.npz" if prefix else "cloud.npz"
        path = assets_dir / fname
        kw: dict[str, Any] = {"positions": pts}
        colors = cloud.get("colors")
        if colors is not None and len(colors) > 0:
            cols = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
            kw["colors"] = (cols * 255).clip(0, 255).astype(np.uint8)
        np.savez_compressed(path, **kw)
        return [f"assets/{fname}"]
    except Exception:
        logger.debug("Failed to save pointcloud asset", exc_info=True)
        return []


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for non-standard types."""
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        if obj.size < 100:
            return obj.tolist()
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    return str(obj)


# ---------------------------------------------------------------------------
# DagTrace — enriched trace recorder
# ---------------------------------------------------------------------------

class DagTrace:
    """Records enriched per-node execution data to disk."""

    def __init__(self, output_dir: str | Path | None = None):
        if output_dir is None:
            output_dir = os.environ.get("GAP_TRACE_DIR", ".")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._node_data_dir = self._output_dir / "node_data"

        self._nodes: list[NodeTrace] = []
        self._node_map: dict[str, NodeTrace] = {}
        self._edges: list[EdgeTrace] = []
        self._events: list[TraceEvent] = []
        self._counter: int = 0
        self._event_counter: int = 0
        # Per-node visit counter so a node that runs many times in a loop (e.g.
        # reperceive_basket each pack cycle) no longer overwrites its earlier
        # iterations. Top-level node_data/<name>/ keeps the LATEST visit (viz
        # back-compat); the full per-iteration history lives under .../iters/<NNN>/.
        self._visit_counts: dict[str, int] = {}

        # Lifecycle hooks — used by harnesses (rehearsal, comparison tools)
        # to capture extra state at node boundaries without modifying the
        # executor itself. Each callable receives (node_name, success).
        # ``on_node_start`` is called with success=None.
        self.on_node_start: Any = None
        self.on_node_end: Any = None

    # --- Registration ---

    def add_node(self, name: str, node_def: dict | None = None) -> str:
        """Register a node and return its trace ID."""
        if name in self._node_map:
            return self._node_map[name].id

        trace_id = f"n_{self._counter:04d}"
        self._counter += 1

        node = NodeTrace(id=trace_id, name=name)
        if node_def:
            node.node_type = node_def.get("type", "")
            node.tool = node_def.get("tool")
            node.script = node_def.get("script")
            node.condition = node_def.get("condition")

        self._nodes.append(node)
        self._node_map[name] = node
        self._record_event(
            "node_registered",
            node_name=name,
            node_id=trace_id,
            detail={"node_type": node.node_type},
        )
        return trace_id

    def add_edge(self, from_name: str, to_name: str) -> None:
        """Record a dependency edge."""
        from_node = self._node_map.get(from_name)
        to_node = self._node_map.get(to_name)
        self._edges.append(
            EdgeTrace(
                from_name=from_name,
                to_name=to_name,
                from_id=from_node.id if from_node else None,
                to_id=to_node.id if to_node else None,
            )
        )
        self._record_event(
            "edge_registered",
            node_name=from_name,
            node_id=from_node.id if from_node else "",
            detail={
                "from_name": from_name,
                "to_name": to_name,
                "to_id": to_node.id if to_node else "",
            },
        )

    # --- Lifecycle ---

    def start_node(self, name: str) -> None:
        # New visit of this node -> bump its visit index so the per-visit
        # resolved_inputs/output snapshots land in a fresh iters/<NNN>/ dir
        # instead of clobbering the previous loop iteration.
        self._visit_counts[name] = self._visit_counts.get(name, -1) + 1
        node = self._node_map.get(name)
        if node:
            node.started_at = time.time()
            # A new visit re-opens the node. end_node() refuses to close a
            # node whose earlier visit already closed (its finished_at guard),
            # so without this reset every re-visited node -- a retry ladder,
            # a loop back through a router -- stayed "running" in the table
            # forever, and a node that failed on its second visit was never
            # marked "error" anywhere but in the event stream.
            node.finished_at = 0.0
            node.error_message = None
            node.status = "running"
            node.visits += 1
            self._record_event(
                "node_started",
                node_name=name,
                node_id=node.id,
                status=node.status,
            )
        if self.on_node_start is not None:
            try:
                self.on_node_start(name, None)
            except Exception:
                pass

    def end_node(self, name: str, success: bool) -> None:
        if self.on_node_end is not None:
            try:
                self.on_node_end(name, success)
            except Exception:
                pass
        node = self._node_map.get(name)
        if node:
            if node.finished_at > 0:
                return
            node.finished_at = time.time()
            node.duration_ms = (node.finished_at - node.started_at) * 1000.0
            node.status = "ok" if success else "error"
            self._record_event(
                "node_finished",
                node_name=name,
                node_id=node.id,
                status=node.status,
                detail={"success": success, "duration_ms": round(node.duration_ms, 2)},
            )

    def skip_node(self, name: str) -> None:
        node = self._node_map.get(name)
        if node:
            node.status = "skipped"
            self._record_event(
                "node_skipped",
                node_name=name,
                node_id=node.id,
                status=node.status,
            )

    # --- I/O Recording ---

    def _node_dir(self, name: str) -> Path:
        d = self._node_data_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _assets_dir(self, name: str) -> Path:
        d = self._node_data_dir / name / "assets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _iter_dir(self, name: str) -> Path:
        """Per-visit snapshot dir for ``name``'s current visit. Mirrors the
        node's resolved_inputs/output so loop iterations are all preserved;
        the top-level node_data/<name>/ still holds the latest visit."""
        v = self._visit_counts.get(name, 0)
        d = self._node_data_dir / name / "iters" / f"{v:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record_resolved_inputs(self, name: str, resolved: dict[str, Any]) -> None:
        """Serialize resolved inputs to disk and extract visual assets."""
        node = self._node_map.get(name)
        if not node:
            return
        try:
            node_dir = self._node_dir(name)
            assets_dir = self._assets_dir(name)

            # Extract assets from input values
            all_assets: list[str] = []
            for key, val in resolved.items():
                all_assets += _extract_assets(val, assets_dir, f"input_{key}")

            # Serialize inputs as JSON. Write the latest to the top-level dir
            # (back-compat) and a per-visit copy under iters/<NNN>/ (history).
            serialized = _serialize_value(resolved)
            with open(node_dir / "resolved_inputs.json", "w") as f:
                json.dump(serialized, f, indent=2, default=_json_default)
            with open(self._iter_dir(name) / "resolved_inputs.json", "w") as f:
                json.dump(serialized, f, indent=2, default=_json_default)

            node.has_inputs = True
            node.assets.extend(all_assets)
            self._record_event(
                "inputs_recorded",
                node_name=name,
                node_id=node.id,
                detail={"asset_count": len(all_assets), "keys": sorted(resolved.keys())},
            )
        except Exception:
            logger.warning("Failed to record inputs for %s", name, exc_info=True)

    def record_output(self, name: str, output: Any) -> None:
        """Serialize node output to disk and extract visual assets."""
        node = self._node_map.get(name)
        if not node:
            return
        try:
            node_dir = self._node_dir(name)
            assets_dir = self._assets_dir(name)

            # Extract assets from output
            all_assets: list[str] = []
            if isinstance(output, dict):
                for key, val in output.items():
                    all_assets += _extract_assets(val, assets_dir, f"output_{key}")
            else:
                all_assets += _extract_assets(output, assets_dir, "output")

            # Serialize output as JSON. Latest to the top-level dir (back-compat),
            # per-visit copy under iters/<NNN>/ so loop iterations are preserved.
            serialized = _serialize_value(output)
            with open(node_dir / "output.json", "w") as f:
                json.dump(serialized, f, indent=2, default=_json_default)
            with open(self._iter_dir(name) / "output.json", "w") as f:
                json.dump(serialized, f, indent=2, default=_json_default)

            node.has_output = True
            node.assets.extend(all_assets)
            self._record_event(
                "output_recorded",
                node_name=name,
                node_id=node.id,
                detail={"asset_count": len(all_assets)},
            )
        except Exception:
            logger.warning("Failed to record output for %s", name, exc_info=True)

    def record_request(
        self,
        name: str,
        tool: str,
        request: dict[str, Any] | None,
    ) -> None:
        """Save a tool request (resolved input dict) for replay (JSON + metadata)."""
        try:
            node_dir = self._node_dir(name)

            # JSON format for viewing
            with open(node_dir / "request.json", "w") as f:
                json.dump(_serialize_value(request), f, indent=2, default=_json_default)

            # Metadata for replay dispatch
            meta = {"tool": tool}
            with open(node_dir / "request.meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            logger.warning("Failed to record request for %s", name, exc_info=True)
        else:
            node = self._node_map.get(name)
            self._record_event(
                "request_recorded",
                node_name=name,
                node_id=node.id if node else "",
                detail={"tool": tool},
            )

    def record_subcall(
        self,
        node_name: str,
        seq: int,
        tool: str,
        request: dict[str, Any] | None,
        response: Any,
    ) -> None:
        """Save a sub-call's request and response within a skill/script node.

        ``request`` is the resolved input dict the tool was invoked with;
        it may be ``None`` for calls that carry no inputs. In that case we
        skip the request side and only persist the response.
        """
        try:
            short_tool = tool.split(".")[-1] if "." in tool else tool
            call_dir = self._node_dir(node_name) / "calls" / f"{seq:03d}_{short_tool}"
            call_dir.mkdir(parents=True, exist_ok=True)
            assets_dir = call_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            # Request: JSON + metadata. Skip when no input dict was given.
            if request is not None:
                with open(call_dir / "request.json", "w") as f:
                    json.dump(_serialize_value(request), f, indent=2, default=_json_default)
                with open(call_dir / "request.meta.json", "w") as f:
                    json.dump({"tool": tool}, f, indent=2)

                # Extract visual assets from request
                _extract_assets(request, assets_dir, "request")

            # Response: JSON + visual assets
            with open(call_dir / "response.json", "w") as f:
                json.dump(_serialize_value(response), f, indent=2, default=_json_default)
            _extract_assets(response, assets_dir, "response")

            node = self._node_map.get(node_name)
            self._record_event(
                "subcall_recorded",
                node_name=node_name,
                node_id=node.id if node else "",
                detail={"seq": seq, "tool": tool},
            )
        except Exception:
            logger.warning(
                "Failed to record subcall %d for %s", seq, node_name, exc_info=True,
            )

    def record_stream_read(
        self,
        node_name: str,
        seq: int,
        stream_name: str,
        value: Any,
        sampled_at: float,
    ) -> None:
        """Save the snapshot consumed by an ObservationStreamHandle.latest() call.

        Mirrors record_subcall but for stream reads. Replay can later return
        the same value at the same (node, seq) instead of re-polling.

        Per-tick cost matters here — visual-servo loops call ``.latest()``
        every ~10 ms. The payload JSON is cheap (large arrays are summarized
        by the serializer), but the disk write rate is still tunable:

        - ``GAP_TRACE_STREAM_SAMPLE_N`` (int, default 1): only persist
          every Nth stream read. ``N=10`` cuts the disk write rate to
          1 in 10. ``N=0`` disables stream-read recording entirely.
        """
        sample_n = int(os.environ.get("GAP_TRACE_STREAM_SAMPLE_N", "1") or "1")
        if sample_n <= 0:
            return
        if sample_n > 1 and (seq % sample_n) != 0:
            return
        try:
            short = stream_name.split(".")[-1] or stream_name
            call_dir = self._node_dir(node_name) / "stream_reads" / f"{seq:03d}_{short}"
            call_dir.mkdir(parents=True, exist_ok=True)

            with open(call_dir / "value.json", "w") as f:
                json.dump(_serialize_value(value), f, indent=2, default=_json_default)
            meta = {
                "stream": stream_name,
                "sampled_at": sampled_at,
            }
            with open(call_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            node = self._node_map.get(node_name)
            self._record_event(
                "stream_read",
                node_name=node_name,
                node_id=node.id if node else "",
                detail={
                    "seq": seq,
                    "stream": stream_name,
                    "sampled_at": sampled_at,
                },
            )
        except Exception:
            logger.warning(
                "Failed to record stream read %d for %s", seq, node_name, exc_info=True,
            )

    def record_condition(
        self, name: str, actual: Any, expected: Any, met: bool
    ) -> None:
        node = self._node_map.get(name)
        if node:
            node.condition_result = {
                "actual": _serialize_value(actual),
                "expected": _serialize_value(expected),
                "met": met,
            }
            self._record_event(
                "condition_evaluated",
                node_name=name,
                node_id=node.id,
                detail={"met": met},
            )

    def record_error(self, name: str, error_msg: str) -> None:
        node = self._node_map.get(name)
        if node:
            node.error_message = error_msg
            self._record_event(
                "error_recorded",
                node_name=name,
                node_id=node.id,
                status=node.status,
                detail={"error": error_msg},
            )

    def _record_event(
        self,
        event_type: str,
        *,
        node_name: str = "",
        node_id: str = "",
        status: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(
            TraceEvent(
                seq=self._event_counter,
                event_type=event_type,
                timestamp=time.time(),
                node_id=node_id,
                node_name=node_name,
                status=status,
                detail=detail or {},
            )
        )
        self._event_counter += 1

    # --- Flush ---

    def copy_workflow(self, workflow_dir: Path) -> None:
        """Copy workflow.json and scripts/ to the trace output directory.

        Each trial may have LLM-regenerated code, so we snapshot the full
        workflow folder into the trace output for reproducibility.
        """
        try:
            # Copy workflow.json
            src_wf = workflow_dir / "workflow.json"
            if src_wf.exists():
                dest_wf = self._output_dir / "workflow.json"
                if not dest_wf.exists():
                    shutil.copy2(src_wf, dest_wf)

            # Copy scripts/ directory
            src_scripts = workflow_dir / "scripts"
            if src_scripts.is_dir():
                dest_scripts = self._output_dir / "scripts"
                if not dest_scripts.exists():
                    shutil.copytree(
                        src_scripts, dest_scripts,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
        except Exception:
            logger.debug("Failed to copy workflow artifacts", exc_info=True)

    def flush(self) -> None:
        """Write enriched dag_trace.json to disk."""
        data = {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "node_type": n.node_type,
                    "service": n.service,
                    "method": n.method,
                    "script": n.script,
                    "started_at": n.started_at,
                    "finished_at": n.finished_at,
                    "duration_ms": round(n.duration_ms, 2),
                    "status": n.status,
                    "condition": n.condition,
                    "condition_result": n.condition_result,
                    "error_message": n.error_message,
                    "has_inputs": n.has_inputs,
                    "has_output": n.has_output,
                    "visits": n.visits,
                    "assets": n.assets,
                }
                for n in self._nodes
            ],
            "edges": [
                {
                    "from": edge.from_id or edge.from_name,
                    "to": edge.to_id or edge.to_name,
                    "from_id": edge.from_id,
                    "to_id": edge.to_id,
                    "from_name": edge.from_name,
                    "to_name": edge.to_name,
                }
                for edge in self._edges
            ],
            "events": [
                {
                    "seq": event.seq,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "node_id": event.node_id,
                    "node_name": event.node_name,
                    "status": event.status,
                    "detail": event.detail,
                }
                for event in self._events
            ],
        }

        trace_path = self._output_dir / "dag_trace.json"
        with open(trace_path, "w") as f:
            json.dump(data, f, indent=2)
        # DEBUG: flush() now also runs incrementally after every node (crash
        # resilience); the end-of-run INFO line lives at the executor's
        # final flush call.
        logger.debug("DAG trace written to %s", trace_path)
