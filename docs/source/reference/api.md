# Python API Reference

The curated public Python surface of the `gap` package (distribution
`graph-as-policy`, version `0.1.0.dev0`). Every signature below is verified
against the source tree; follow the source links for full docstrings.

`import gap` is deliberately light: submodules (`gap.connector`, `gap.agent`,
`gap.builder`, ...) load lazily on first attribute access, so importing `gap`
never pulls in torch, MuJoCo, or JAX.

## Top level

Source: [gap/runtime/execute.py](gh-engine:gap/runtime/execute.py)

### gap.execute

```python
gap.execute(
    graph,
    connector=None,
    *,
    skills: str | Path | Sequence[str | Path] | None = None,
    inputs: dict[str, Any] | None = None,
    trace_dir: str | Path | None = None,
    checkpoints: str = "warn",
    max_node_workers: int = 8,
) -> ExecutionResult
```

Execute a workflow graph and return an `ExecutionResult`.

- **graph** — a workflow directory or `workflow.json` path (`str | Path`), a
  materialized workflow `dict`, or any object exposing `to_dict()` (a
  [`gap.builder.Workflow`](#building)). Dicts and builder objects are
  materialized to a temporary directory first.
- **connector** — duck-typed: anything with a `.tool_registry`
  (a `gap.tools.ToolRegistry` carrying the `robot.*` / `sim.*` tools), a
  zero-arg `.get_observation` poll function, and optionally
  `.world_snapshot` (ground truth, enables checkpoint enforcement) and
  `.capabilities`. `None` runs tools-only against the default `@tool`
  registry (tests, dry runs).
- **skills** — skill registry root(s): one path or a precedence-ordered
  sequence. When omitted, registries resolve via
  `gap.skills.resolve_registries` (`$GAP_SKILLS_PATH` list > project
  `[tool.gap]` > user config > the open-robot-skills checkout next to the
  GaP checkout); if none are found, execution proceeds without a skill
  registry.
- **inputs** — initial inputs, addressable at the top level as
  `{"$ref": "in.<name>"}` and as base producers for subgraph input binding.
- **trace_dir** — trace output directory (default: `GAP_TRACE_DIR` or the
  workflow directory). See [Traces](../running/traces.md).
- **checkpoints** — `"off" | "warn" | "raise"`, the enforcement mode for
  `validate=True` postcondition checkpoints. Only active when the connector
  exposes `world_snapshot`; `"raise"` converts a failed hard checkpoint
  into a `VerificationFailed` (which, like every execution error, lands in
  `result.error`).
- **max_node_workers** — thread budget per parallel super-step.

:::{note}
`gap.execute` **never raises on workflow failure.** Any exception thrown
during execution is caught and stored in `result.error`, with
`result.success == False`. Code after `gap.execute(...)` always runs.
(The one designed exception: `GuardLimitExceeded` is a `BaseException` —
see [Errors](#errors).)
:::

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    success: bool                 # error is None and exit_status == "success"
    exit_status: str | None       # "success" | "failure" | None
    outputs: dict[str, Any]       # {subgraph_name: bound outputs}
    trace_path: Path | None
    checkpoint_results: list[Any] # gap.runtime.verify.CheckpointResult
    error: Exception | None
    duration_s: float
```

### Other top-level exports

| Attribute | Resolves to |
|---|---|
| `gap.execute` | `gap.runtime.execute.execute` |
| `gap.connector` | the [`gap.connector`](#connectors) module |
| `gap.agent` | the [`gap.agent`](#generation) module |
| `gap.builder` | the [`gap.builder`](#building) module |
| `gap.benchmark` | the [`gap.benchmark`](#benchmarking) module |
| `gap.viz` | the [`gap.viz`](#visualization) module |
| `gap.types` | the [`gap.types`](#types) module |
| `gap.errors` | the [`gap.errors`](#errors) module |
| `gap.NodeContext` | `gap.runtime.context.NodeContext` — the `ctx` passed to script nodes (`ctx.tool(name, **kwargs)`, `ctx.publish(value)`, `ctx.cancel_token`) |
| `gap.CancelToken` | `gap.runtime.context.CancelToken` — cooperative cancellation; streaming skills call `ctx.cancel_token.raise_if_set()` per iteration |

## Connectors

Source: [gap/connector](gh-engine:gap/connector)

`gap.connector` exports `sim`, `real`, `SimConnector`, `RealConnector`,
`Connector`, `Capabilities`, and `DataCollector`. A connector owns one
environment instance and exposes:

- `tool_registry` — a fresh `ToolRegistry` with the connector's `robot.*`
  (and, for sims, `sim.*`) tools registered (see
  [Connector tools](connector-tools.md));
- `get_observation()` — the env obs dict converted to a
  [`gap.types.Observation`](#types);
- `capabilities` — what this backend can do;
- `close()` plus context-manager support (`with gap.connector.sim(...) as conn:`).

### gap.connector.sim

```python
gap.connector.sim(
    env: str = "libero",
    *,
    task="libero_object/0",
    cameras: list[str] | None = None,
    headless: bool = True,
    seed: int | None = None,
    record_video: bool = False,
    **env_kwargs,
) -> SimConnector
```

Requires the `[libero]` extra (see
[Installation](../getting-started/installation.md)); raises `ImportError`
with an install hint when `gap.envs` is unavailable.

- **env** — registry name (`"libero"`, ...) resolved via
  `gap.envs.registry.resolve`. See [Environments](../running/environments.md).
- **task** — `"suite/task_id"` (e.g. `"libero_object/0"`), a
  `(suite, task_id)` tuple, or a bare `int` task id (the env's registry
  key doubles as the suite name, e.g. `sim("libero_object", task=3)`).
- **cameras** — camera-name override (default: the env's
  `EnvConfig.default_cameras`).
- **headless** — pass `False` to enable rendering even without video
  recording.
- **seed** — stashed and consumed by the connector's first `reset()`
  (one reset, not two).
- **record_video** — construct with rendering enabled and start frame
  capture on reset.
- **env_kwargs** — extra keyword arguments forwarded to the env factory.

`SimConnector` additionally exposes `reset(seed=None) -> Observation`,
`check_success() -> tuple[bool, float]`, `get_state() -> dict` (ground-truth
arm states + object poses), `world_snapshot()`, `start_video(*, clear=True)`,
and `save_video(output_path, fps=20, clear=False) -> dict`.

### gap.connector.real

```python
gap.connector.real(
    robot: str = "franka",
    *,
    cameras: list[str] | None = None,
    rr_config: str | Path | None = None,
    rr_autostart: bool = True,
    rr_log_path: str | Path | None = None,
    port: int | None = None,
    wait_timeout_s: float = 60.0,
    **env_kwargs,
) -> RealConnector
```

Requires real hardware — read [Safety](../real-robots/safety.md) first.

- **robot** — `"franka"` (robots_realtime msgpack bridge, full motion) or
  `"ur_zed"` (UR + ZED, **perception-only**: the registry contains only
  observation/camera tools, so accidental motion is structurally
  impossible). Anything else raises `ValueError`.
- **rr_config** / **rr_autostart** / **rr_log_path** / **port** — Franka
  only: the rr-session client YAML (default
  `configs/franka/franka_robotiq_client.yaml`, resolved against the
  `third_party/robots_realtime` checkout), whether to spawn the client
  automatically (`False` restores the two-terminal debug flow), its log
  tee path, and the msgpack server port (default 9000). Passing
  `rr_config`/`port` with `robot="ur_zed"` raises `ValueError`.
- **wait_timeout_s** — the factory blocks in
  `wait_ready(timeout_s, poll_s=0.5)` until the first RGB frame arrives,
  then returns; on timeout it closes the connector and raises
  `TimeoutError` carrying a wire-stage diagnosis.
- **env_kwargs** — e.g. `host=` for franka; `robot_ip=`,
  `calibration_path=` for ur_zed.

See [Connectors](../real-robots/connectors.md) for the hardware setup these
factories assume.

### Capabilities

```python
@dataclass(frozen=True)
class Capabilities:
    reset: bool = False
    success_check: bool = False
    video: bool = False
    world_state: bool = False
```

`gap.execute` and the benchmark harness branch on these instead of
`isinstance` checks. `SimConnector` reports each flag from what the env
actually supports (e.g. `world_state` requires a reachable MuJoCo sim);
`RealConnector` reports **all False** — no scripted reset, no success check,
no ground truth on hardware, so checkpoints degrade to a logged skip.

### DataCollector

```python
DataCollector(connector, out_path: str | Path)
```

Records synchronized obs/action/reward rows from a connector into an
`.h5`/`.hdf5` file, with `start_episode()` / episode-boundary tracking.
Used by the [collect-and-train example](../examples/collect-and-train.md).

## Generation

Source: [gap/agent/\_\_init\_\_.py](gh-engine:gap/agent/__init__.py)

Requires an LLM API key (`OPENROUTER_API_KEY` by default — see
[LLM providers](../authoring/llm-providers.md)) and at least one skill
registry (resolution failure raises `FileNotFoundError`).

### gap.agent.generate / generate_sync

```python
async gap.agent.generate(
    instruction: str,
    *,
    skills: str | Path | Sequence[str | Path] | None = None,
    model: str | None = None,
    provider: str | None = None,
    out_dir: str | Path | None = None,
    config: PipelineConfig | str | Path | None = None,
) -> GeneratedGraph

gap.agent.generate_sync(...)  # same parameters; wraps asyncio.run()
```

Compiles a language instruction into a typed, verified workflow graph: a
coordinator decides the topology, one subgraph agent per subgraph fills in
structure + scripts, a whole-workflow checkpoint agent authors postcondition
sidecars, then the result is validated with a bounded LLM script-fix loop.
See [Generation](../authoring/generation.md).

- **model** — LLM model override (default: the provider default,
  `gemini-3.1-flash-lite-preview` on openrouter).
- **provider** — `"openrouter"` | `"vertex"`.
- **out_dir** — the workflow folder is written to `<out_dir>/task_00`;
  defaults to `outputs/generated_<timestamp>`.
- **config** — a `PipelineConfig` or YAML path for full control; the
  `skills`/`model`/`provider` arguments override the corresponding config
  fields.

Raises `RuntimeError` when generation fails (LLM retries exhausted, missing
capabilities reported, or the assembled workflow does not load) and
`FileNotFoundError` when no skill registry is given or discoverable.

### GeneratedGraph

```python
@dataclass
class GeneratedGraph:
    path: Path             # the written workflow folder
    workflow: dict         # parsed workflow.json contents
    code: dict[str, str]   # every generated source file, keyed by
                           # workflow-relative path (scripts + checkpoint sidecars)
```

`str(graph)` (and therefore `print(graph)`) renders the graph as
box-drawing terminal text via `gap.viz.to_text`.

### Configuration dataclasses

`PipelineConfig` (source:
[gap/agent/config.py](gh-engine:gap/agent/config.py)) — load with
`PipelineConfig.from_yaml(path)`. Key fields:

| Field | Default | Meaning |
|---|---|---|
| `task` | `""` | the task instruction (when driven from YAML) |
| `llm` | `LlmConfig()` | provider/model settings (below) |
| `composition` | `CompositionConfig()` | multi-agent knobs (below) |
| `skills` | `None` | registry root(s); `None` resolves via env/config/auto-discovery |
| `safety_limits` | 50 / 20 / 5000 | `max_perception_calls`, `max_planning_calls`, `max_sim_steps` guard limits |
| `environment.cameras` | `["agentview", "robot0_eye_in_hand"]` | cameras advertised to the agents |
| `max_retries` | `3` | LLM call retry budget |
| `trials`, `suites`, `policies`, `policy_manager` | — | benchmark-harness fields (see [Benchmark config](benchmark-config.md)) |

`LlmConfig` (source: [gap/agent/llm.py](gh-engine:gap/agent/llm.py)):

| Field | Default | Meaning |
|---|---|---|
| `provider` | `"openrouter"` | `"openrouter"` \| `"vertex"` |
| `model` | `None` | `None` uses the provider default (openrouter: `gemini-3.1-flash-lite-preview`) |
| `endpoint` | `None` | OpenAI-compatible base URL (or full `.../chat/completions` URL); set it to reach a local vLLM or other OpenAI-compatible server |
| `api_key` | `None` | falls back to `OPENROUTER_API_KEY` (openrouter) |
| `project_id`, `region` | `None` | Vertex AI only |
| `temperature` | `0.7` | omitted for models that reject sampling parameters; `None` always omits |
| `max_tokens` | `20480` | |
| `max_concurrent_requests` | `4` | per-event-loop request semaphore |
| `cache_dir` | `None` | disk response cache; falls back to `GAP_LLM_CACHE_DIR` |

`CompositionConfig`:

| Field | Default | Meaning |
|---|---|---|
| `subgraph_temperature` | `0.3` | temperature for subagent calls |
| `subgraph_model`, `coordinator_model` | `None` | per-role model overrides (default `llm.model`) |
| `max_subgraph_retries` | `2` | per-subagent retry limit on validation failure |
| `max_coordinator_retries` | `2` | coordinator retry limit |
| `max_validation_retries` | `2` | LLM fix-loop attempts for graph validation errors |
| `checkpoint_agent` | `True` | run the whole-workflow checkpoint pass |

`gap.agent` also exports the lower-level LLM calls `complete` and
`complete_with_tools` (system + messages → assistant text, with an optional
native tool-use loop).

## Building

Source: [gap/builder/core.py](gh-engine:gap/builder/core.py)

`gap.builder` exports `Workflow`, `Subgraph`, `WorkflowSpec`, `Ref`,
`START`, `END`, and `BuilderError`. The API mirrors LangGraph: `add_node`,
`add_edge`, `add_conditional_edges`. For a guided walkthrough see
[the builder guide](../authoring/builder.md); for the JSON the builder
emits see the [workflow schema](workflow-schema.md).

### Ref, START, END

```python
Ref(path: str) -> dict[str, str]   # Ref("observe.cameras") → {"$ref": "observe.cameras"}
START: str = "START"
END: str = "END"
```

Use `Ref` inline in any `inputs={...}` dict to reference another node's
output, or a subgraph-bound input as `Ref("in.<name>")`. `START`/`END` are
the virtual edge endpoints; node names may not collide with them.

### Subgraph

```python
Subgraph(*, name: str, skill: str = "generic")
```

Author one self-contained subgraph. Valid node types inside a subgraph are
`tool`, `script`, `router`, and `noop`.

```python
sg.add_node(name, *, type, tool=None, script=None, inputs=None, streaming=False)
```
Adds a node. `type="tool"` requires `tool=` (flat dispatch name);
`type="script"` and `type="router"` require `script=` (path relative to the
workflow directory). `streaming=True` is only valid on tool/script nodes.

```python
sg.add_edge(src: str, dst: str)
sg.add_conditional_edges(src, mapping: dict[str, str], *, router_field=None)
```
Static edge / conditional dispatch. `router_field` names the output field to
switch on; leave it `None` for `type="router"` nodes (the routing function
returns the target name directly). One conditional entry per source —
a second call for the same `src` raises `BuilderError`.

```python
sg.add_input(name, *, type_name: str)
```
Declares a cross-subgraph input. `type_name` is a name from
`gap.schema.TYPE_REGISTRY` (e.g. `"OrientedBoundingBox"`); reference it in
node inputs as `Ref(f"in.{name}")`.

```python
sg.add_exit(name: str)
```
Declares a success-outcome marker: creates a `noop` node and registers it
as an exit success value (the common, `router_field: null` mode). Call once
per success outcome. Mutually exclusive with `set_exit_router`.

```python
sg.set_exit_router(*, router_field: str, success_values: Sequence[str])
```
Data-dependent exit routing: the exit value is read from `router_field` on
the terminal node's output. No `noop` markers are created.

```python
sg.set_outputs(**bindings)          # each value must be a Ref(); replaces previous
sg.set_on_error(value: str | None)  # failure-path exit symbol (never a node name)
```

```python
sg.add_checkpoint(name, predicate, *, diagnostics=None, rationale="",
                  validate=True, weight=1.0)
```
Declares a postcondition checkpoint. `predicate` is evaluated against a
`gap.runtime.verify.World` snapshot taken at subgraph exit. `validate=True`
(default) marks a *hard* postcondition, enforced when the connector exposes
ground truth; `validate=False` checkpoints are probes that never gate.
More than 6 checkpoints per subgraph emits a `UserWarning`. Checkpoints are
never serialized into `workflow.json` — they live in
`checkpoints/<subgraph>.py` sidecars (see
[Checkpoints](../running/checkpoints.md)).

```python
sg.dump_checkpoints_module(path) -> Path | None
```
Writes the checkpoint sidecar (`None` if no checkpoints are declared). The
sidecar embeds the original builder source block, which is normally set by
the generation pipeline; hand-authors write sidecars by hand instead, like
[examples/libero_quickstart](gh-engine:examples/libero_quickstart/graph/checkpoints/grasp_sg.py).

```python
sg.to_dict() -> dict             # v3 subgraph JSON shape (no name field)
sg.save(path)                    # validated, name-wrapped standalone JSON
Subgraph.load(path_or_dict) -> Subgraph
```
Standalone subgraph JSON is non-canonical — executors consume subgraphs
only inside a workflow's `subgraphs` map.

### Workflow

```python
Workflow(*, name: str | None = None, description: str | None = None)
```

Author the top-level workflow. Top-level nodes are typically `subgraph` and
`end`; `tool`/`script`/`router`/`noop` are also allowed.

```python
wf.add_node(name, *, type, tool=None, script=None, ref=None, status=None,
            recovery=(), inputs=None, streaming=False)
```
All six node types. `type="subgraph"` requires `ref=` (a key in the
registered subgraphs); `type="end"` requires `status="success"` or
`"failure"` and accepts `recovery=` — a sequence of
`{"tool": ..., "inputs": {...}}` dicts (or `ToolCall` objects) run
best-effort when that end node is reached.

```python
wf.add_edge(src, dst)
wf.add_conditional_edges(src, mapping, *, router_field=None)
```
Same as `Subgraph`. Top-level conditional edges on subgraph nodes use
`router_field="exit"` (remapped internally to the subgraph's exit value).

```python
wf.add_subgraph(sg: Subgraph)    # registers under sg.name
wf.to_dict() -> dict             # full v3 workflow JSON
wf.save(path, *, validate=True)  # parse + structural validation, then write
Workflow.load(path) -> Workflow  # strict parse → mutable builder
```

`save(validate=True)` raises `GraphValidationError` on error-level issues
and resolves script paths against the JSON file's parent directory — saving
to a temp location away from your `scripts/` breaks script introspection.
`Workflow.load` → `to_dict()` round-trips byte-equivalently for any valid
`workflow.json`.

### WorkflowSpec

```python
WorkflowSpec(*, name=None, description=None)
```

Coordinator-output scaffold: workflow topology plus subgraph *metadata
stubs* (no nodes/edges inside the subgraphs — the generation pipeline's
subgraph agents fill those in afterward). Methods:

```python
spec.add_subgraph_node(name, *, ref, inputs=None)
spec.add_end(name, *, status, recovery=())
spec.declare_subgraph(name, *, skill, description, exit_success_values,
                      on_error, inputs=None, outputs=None, stage=None)
spec.to_dict() -> dict
```

`declare_subgraph`'s `inputs`/`outputs` are `{name: type_name_str}` type
declarations (not data bindings); `stage` must be `"grasp"`, `"transport"`,
`"place"`, or `None`.

## Benchmarking

Source: [gap/benchmark/\_\_init\_\_.py](gh-engine:gap/benchmark/__init__.py)

```python
gap.benchmark.run(
    config: BenchmarkConfig | str | Path,
    *,
    gate: bool = False,
    resume: bool = False,
) -> BenchmarkSummary
```

Runs a benchmark grid from a config object or YAML path (synchronous; an
async `gap.benchmark.run_benchmark` is also exported). With `gate=True`,
the returned summary's `ok` is `False` when the overall success rate falls
below `config.gate_threshold` (default `0.90`), when any cell errored, or
when no trial ran. `resume=True` reuses the latest run directory under
`config.output_dir`, skipping cells whose results already exist; the merged
summary is rebuilt over old + new cells.

`BenchmarkSummary` fields: `ok`, `gated`, `gate_threshold`, `success_rate`
(trial-pooled), `completion_rate` (trial-weighted), `n_trials`, `n_success`,
`run_dir`, `cells`, and `summary` (the merged dict also written to
`<run>/summary.json`).

The module also exports `BenchmarkConfig`, `BenchmarkModeOverride`,
`DEFAULT_GATE_THRESHOLD`, `KNOWN_FAMILIES`, `KNOWN_MODES`, `FAMILY_SUITES`,
and `POSVAR_SUITES` — see [Benchmark config](benchmark-config.md) for the
YAML shape and [Benchmarking](../benchmarks/benchmarking.md) for the
workflow.

## Visualization

Source: [gap/viz/\_\_init\_\_.py](gh-engine:gap/viz/__init__.py)

```python
gap.viz.serve(
    root: str | Path = "outputs",
    *,
    port: int = 9432,
    services: str | Path | None = None,
    host: str = "127.0.0.1",
    open_browser: bool = False,
) -> None
```

Serves the interactive trace visualizer (FastAPI + the bundled React
frontend) over `root`, scanned recursively for trials (directories
containing `dag_trace.json` or `workflow.json`). `services` points at an
open-robot-skills checkout so node tooltips show port schemas. The web UI
renders PNG/JPEG node assets inline; recorded videos are written to the
trace directory and viewed from disk, not played in the browser.

```python
gap.viz.render(
    graph: dict | str | Path,
    out: str | Path,
    *,
    data_edges: bool = False,
    legend: bool = True,
    title: str | None = None,
    also_png: bool = True,
) -> Path
```

Renders a v3 workflow to a paper-ready matplotlib figure. The `out` suffix
selects the format; when `also_png=True` and `out` is not already a PNG, a
sibling `.png` is written too. `data_edges=True` overlays dashed data-flow
(`$ref`) edges. Lazy import — `import gap.viz` does not pull in matplotlib.

```python
gap.viz.to_text(
    workflow: dict | str | Path,
    *,
    color: bool = False,
    width: int | None = None,
) -> str
```

Renders a v3 workflow as Unicode box-drawing terminal text (pure stdlib;
this is what `print(generated_graph)` and `gap generate` show). Default
width cap 76 columns, floor 24; `color=True` emits ANSI colors.

## Testing

Source: [gap/testing/\_\_init\_\_.py](gh-engine:gap/testing/__init__.py)

:::{warning}
`gap.testing` exports exactly four names: `FakeContext`, `ToolCallRecord`,
`make_test_observation`, and `assert_graph_valid`. There is no
`FakeConnector` and no `connector_contract_suite`.
:::

These are the same fixtures GaP's own suite uses, exported so a skill
bundle can be unit-tested without a robot, a GPU, or an LLM. See
[Testing bundles](../skills/testing-bundles.md) for the workflow.

### FakeContext

```python
FakeContext(tool_responses: dict[str, Any] | None = None, *, node_id: str = "test_node")
```

A `NodeContext` stand-in with scripted per-tool responses. Each
`tool_responses` value is one of:

- a **plain value** — returned on every call;
- a **callable** — invoked with the call kwargs (raise inside it to
  exercise error paths);
- a **list** — values popped front-to-back, one per call; exhausting the
  list raises `ToolError`.

Tools without a scripted response raise `ToolError`, so tests fail loudly
on unexpected calls. Surface:

```python
ctx.tool(name, **kwargs) -> Any        # dispatch; records a ToolCallRecord
ctx.publish(value) -> None             # appends to ctx.published
ctx.calls: list[ToolCallRecord]        # every dispatch, in order
ctx.calls_to(tool) -> list[ToolCallRecord]
ctx.call_count(tool) -> int
ctx.cancel_token                       # a real CancelToken
```

### ToolCallRecord

```python
@dataclass
class ToolCallRecord:
    tool: str
    kwargs: dict[str, Any]
```

### make_test_observation

```python
make_test_observation(
    objects: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] | None = None,
    *,
    camera_name: str = "test_cam",
    image_hw: tuple[int, int] = (120, 160),
    camera_eye: tuple[float, float, float] = (0.0, -0.7, 0.9),
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.1),
    table_z: float = 0.0,
    fov_deg: float = 60.0,
    arm_joints: int = 7,
) -> tuple[Observation, dict[str, Any]]
```

Renders a synthetic tabletop scene of colored axis-aligned boxes with a real
pinhole camera model: the returned `depth` + `intrinsics` + camera `pose`
reproject exactly onto the boxes' world-space surfaces, so perception math
(mask → points → OBB) can be tested against true numerics, not mocks.
`objects` entries are `(name, center_xyz, size_xyz)` with **full** extents
in meters (default: one 6 cm cube at the origin). Returns
`(observation, ground_truth)` where `ground_truth` maps each object name to
`{"center", "size", "color", "mask"}` plus `"camera_pose_mat"`
(the 4x4 camera-to-world matrix).

### assert_graph_valid

```python
assert_graph_valid(graph: dict | str | Path, *, skill_registry=None, tool_registry=None) -> None
```

Loads and structurally validates a graph (dict, `workflow.json` path, or
workflow directory); raises `AssertionError` listing every error-level
finding. Pass registries to also check tool/skill bindings.

## Types

Source: [gap/types.py](gh-engine:gap/types.py)

Plain `TypedDict`s carrying numpy arrays — no protobuf. Traces serialize
them directly and `$ref` dataflow walks them as plain dicts.

:::{important}
**Quaternions are wxyz (scalar-first) everywhere in GaP.** LIBERO/MuJoCo
and scipy use xyzw internally; env classes convert at the boundary with
`quat_xyzw_to_wxyz`. And **`OrientedBoundingBox.extent` holds
half-extents** along the box's local axes — not full sizes.
:::

| Type | Shape |
|---|---|
| `Vec3` | `{x, y, z}` floats |
| `Quaternion` | `{w, x, y, z}` — **wxyz scalar-first** |
| `Se3Pose` | `{position: Vec3, rotation: Quaternion}` |
| `OrientedBoundingBox` | `{center: Vec3, extent: Vec3, orientation: Quaternion}` — extent = **half**-extents |
| `BoundingBox2D` | `{x1, y1, x2, y2}` pixels, top-left → bottom-right |
| `Mask` | `np.ndarray`, `uint8 [H, W]`, 0 = background, 255 = foreground |
| `CameraFrame` | `{name, rgb, depth, intrinsics, pose}` — `rgb` `uint8 [H, W, 3]`; `depth` `float32 [H, W]` meters; `intrinsics` `float64 [3, 3]` pinhole K; `pose` camera-to-world |
| `PointCloud` | `{points: float32 [N, 3], colors?: float32 [N, 3] in [0, 1]}` |
| `JointState` | `{positions: float64 [dof] radians, names?}` |
| `Trajectory` | `{waypoints: list[JointState]}` |
| `GripperState` | `{position: float}` — meters, **0.0 = closed** |
| `ArmState` | `{joint_state, gripper_fraction, ee_pose, gripper_qpos?, proprio_state?}` |
| `Observation` | `{cameras: list[CameraFrame], arms: list[ArmState]}` |
| `CollisionMesh` | `{name, vertices, faces, pose}` |
| `WorldConfig` | `{meshes: list[CollisionMesh]}` |
| `GraspCandidates` | `{poses: list[Se3Pose] (best-first), scores?}` |
| `FunctionalFeature` | `{kind, pose: Se3Pose, axis: Vec3, description?, confidence?, fit_quality?, radius_inner?, radius_outer?, length?, usable_length?, insertion_depth?, seating_margin?, local_center?}` — a loop, shaft, tip, aperture, surface or region with the metric geometry a mate needs |
| `CollisionSphere` | `{center: [x, y, z], radius}` meters, in `AttachedObject.frame` |
| `AttachedObject` | `{frame, spheres: list[CollisionSphere], sphere_fit_type?, sphere_radius_shrink_m?}` — a held object's collision model in the TCP frame |
| `PoseWaypoint` | `{pose: Se3Pose, mode?, cartesian?, allow_goal_contact?, allow_start_contact?}` |
| `PoseSequence` | `{waypoints: list[PoseWaypoint], time_scale?, world_config?, attached_object?}` — an ordered TCP plan with what it was solved against |
| `Centerline` | `{points: [[x, y, z], ...], arclength_m, radius_m, nodes, ordered, visibility?, tracked?}` — an ordered centreline of a deformable linear object |

`ArmState` semantics: `gripper_fraction` runs **0.0 closed → 1.0 open**
(note the opposite anchor from `GripperState.position`, which is a width in
meters where 0.0 = closed); `ee_pose` is the end-effector in world frame;
`proprio_state` is the policy-training-exact layout (e.g. the openpi-LIBERO
`[eef_pos(3), axisangle(3), gripper_qpos(2)]`) and must **never** be
transformed by the runtime.

Helpers:

```python
identity_pose() -> Se3Pose
make_pose(xyz, quat_wxyz) -> Se3Pose
quat_xyzw_to_wxyz(q) -> tuple[float, float, float, float]
quat_wxyz_to_xyzw(q) -> tuple[float, float, float, float]
pose_to_matrix(pose: Se3Pose) -> np.ndarray   # 4x4 float64
matrix_to_pose(mat: np.ndarray) -> Se3Pose
```

Subgraph `inputs:` declarations name these types via the
`gap.schema.TYPE_REGISTRY` (aliases: `"Pose"` → `Se3Pose`,
`"CameraObservation"` → `CameraFrame`, `"ObservationResponse"` →
`Observation`, plus `str`/`int`/`float`/`bool` scalars).

## Skills

Source: [gap/skills/\_\_init\_\_.py](gh-engine:gap/skills/__init__.py)

The stable import surface for skill-bundle authors (see
[Authoring bundles](../skills/authoring-bundles.md)). Exports: `Skill`,
`SkillMeta`, `SkillRequires`, `Param`, `CanonicalScript`, `SkillsRegistry`,
`SkillInfo`, `ScriptInfo`, `RegistrySet`, `RegistrySpec`, `load_skills`,
`load_registry_set`, `resolve_registries`, `as_registry_paths`,
`find_skills_path`, `looks_like_skills_checkout`, `parse_skill_md`,
`load_prompt`, and `tool` (the `@tool` decorator, re-exported from
`gap.tools`).

### Skill

```python
class MySkill(Skill):
    meta = SkillMeta(description="...", params={...}, outputs={...})

    def run(self, ctx, **kwargs):
        ...
```

Base class for **stateful, class-based skills**. The runtime instantiates
one instance per skill per workflow execution; instance attributes persist
across all visits to the corresponding node during that execution and are
discarded at executor teardown. Function-style skills (a module-level
`run(ctx, ...)`) work unchanged — subclassing is opt-in for genuine state
needs (replan caches, trackers accumulating evidence). `meta` is a
`ClassVar[SkillMeta]`; the base `run` raises `NotImplementedError`.

### load_skills

```python
load_skills(root: str | Path, *, only: list[str] | None = None,
            disable: list[str] | None = None) -> SkillsRegistry
```

Builds a `SkillsRegistry` from one open-robot-skills checkout, discovering
both bundle roots (`<root>/tools/`, `<root>/skills/`) and assigning each
bundle its `kind` from its folder. Bundle-name collisions raise
`ValueError`. For multi-registry resolution use
`resolve_registries(...) -> RegistrySet` and
`load_registry_set(registry_set, *, only=None, disable=None)`, which merge
registries in precedence order with first-wins shadowing — see
[Registries](../skills/registries.md).

### parse_skill_md

```python
parse_skill_md(path: Path) -> SkillMeta
```

Reads a bundle's `SKILL.md` and produces a `SkillMeta`. Required
frontmatter: `name` (must equal the bundle directory name) and
`description`; all GaP extensions live under one `gap:` key. `params` and
`outputs` are *not* frontmatter — they come from Python introspection of
the bundle's callables. Raises `FileNotFoundError` for a missing file and
`ValueError` for malformed frontmatter.

`SkillMeta` carries the parsed contract: `allowed_tools`,
`exit_conditions`, `produces_outputs`, `required_inputs`,
`canonical_scripts`, `prompts`, `references`, `examples`, `errors`, `tips`,
`hard_rules`, `streaming`, `tools`, and `requires: SkillRequires | None`
(`gpu`, `env`, `env_any`, `weights` — consumed by `gap check`).

## Errors

Source: [gap/errors.py](gh-engine:gap/errors.py)

All runtime errors derive from `PipelineError(Exception)` — with one
deliberate exception:

:::{warning}
`GuardLimitExceeded` inherits from **`BaseException`**, not `Exception`.
A bare `except Exception:` in skill code cannot swallow it — safety
call-count guard violations always terminate the workflow.
:::

| Exception | Raised when |
|---|---|
| `PipelineError` | base class for pipeline execution failures |
| `PerceptionFailed` | object not detected or segmentation failed |
| `PlanningFailed` | motion planning failed after retries |
| `GraspFailed` | gripper position indicates an empty grasp or a dropped object |
| `ValidationFailed` | a VLM validation check rejected a result |
| `VerificationFailed` | a postcondition checkpoint failed (`checkpoints="raise"`) |
| `ToolError(tool, detail="")` | a tool call failed; carries `.tool` and `.detail` |
| `WorkflowValidationError` | workflow JSON is structurally invalid (strict parsing: any unknown key) |
| `GraphValidationError(issues)` | subclass of `WorkflowValidationError`; pre-execution validation found errors — carries `.issues: list[ValidationIssue]` and summarizes the first 5 |
| `NodeExecutionError(node_id, cause)` | a node failed during execution; carries `.node_id` and `.cause` |
| `TaskCancelled` | a long-running skill / parallel branch was cancelled cooperatively (`ctx.cancel_token.raise_if_set()`); catchable as a `PipelineError`, routable via `on_error` |
| `StreamUnavailable(cause=None)` | the observation stream produced no first sample within the timeout |
| `GuardLimitExceeded` | a safety call-count guard was exceeded — **`BaseException`** |

`ValidationIssue` is a dataclass `{severity, node_id, field, message}` with
`severity` either `"error"` or `"warning"`.

One generation-side error lives outside `gap.errors`:
`gap.agent.subgraph_runner.CodegenError(RuntimeError)` — raised by the
generation pipeline when the coordinator or a subgraph agent exhausts its
retries (the `gap.agent.generate` facade converts terminal failures into
`RuntimeError`).

Inside graphs, exceptions interact with control flow: a subgraph's
`on_error` declaration converts any in-subgraph exception into an exit
value; without it, the exception crashes the workflow — and lands in
`ExecutionResult.error`, since [`gap.execute`](#top-level) never raises.
See the [executor reference](executor.md) for the full semantics.
