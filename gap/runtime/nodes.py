"""Node execution — tool, script, and skill node runners.

Called one node at a time by the v3 super-step scheduler
(`gap.runtime.executor.WorkflowExecutor`). The scheduler handles
control-flow, reference resolution, and tracing; these functions just
execute the body of a single node and return its raw result.

Node bodies arrive as `gap.runtime.workflow.NodeDef` dataclasses.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gap_core.errors import NodeExecutionError, WorkflowValidationError

from .context import NodeContext
from .observation_stream import ObservationStream, ObservationStreamHandle
from .workflow import NodeDef

if TYPE_CHECKING:
    from gap.skills import SkillsRegistry

logger = logging.getLogger(__name__)


def _wrap_streams(resolved: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    """Wrap any ObservationStream in resolved inputs as a per-node handle.

    The handle carries the ctx so each .latest() call is recorded into the
    trace under the calling node's id. Mutates ``resolved`` in place and
    returns it for ergonomics.
    """
    for key, value in list(resolved.items()):
        if isinstance(value, ObservationStream):
            resolved[key] = ObservationStreamHandle(value, ctx, name=key)
    return resolved


def execute_tool_node(
    node_id: str,
    node: NodeDef,
    resolved_inputs: dict[str, Any],
    tool_registry: Any,
    policy_executor: Any = None,
    cancel_token: Any = None,
    stream_slot: Any = None,
) -> Any:
    """Execute a ``type: tool`` node.

    Dispatches via :class:`gap.tools.ToolRegistry` — the single flat
    dispatch surface for connector tools (``robot.*`` / ``sim.*``),
    bundle tools, and in-process ``@tool`` plugins. Guard enforcement and
    ctx injection are handled by ``ctx.tool``/the registry.

    ``stream_slot`` is bound by the executor for streaming nodes so the
    dispatched skill can ``ctx.publish(...)`` snapshots.

    Returns whatever the underlying tool returns (plain Python value /
    TypedDict).
    """
    tool_name = node.tool or ""
    if not tool_name:
        raise WorkflowValidationError(
            f"tool node '{node_id}' has no `tool` field"
        )
    if tool_registry is None:
        raise WorkflowValidationError(
            f"tool node '{node_id}' but the executor was constructed "
            f"without a ToolRegistry"
        )
    logger.info("[%s] invoking tool %s", node_id, tool_name)
    try:
        ctx = NodeContext(
            tool_registry, node_id=node_id,
            policy_executor=policy_executor,
            cancel_token=cancel_token,
        )
        if stream_slot is not None:
            ctx._stream_slot = stream_slot
        result = ctx.tool(tool_name, **resolved_inputs)
        logger.info("[%s] tool %s completed", node_id, tool_name)
        return result
    except Exception as e:
        raise NodeExecutionError(node_id, e) from e


def execute_script_node(
    node_id: str,
    node: NodeDef,
    resolved_inputs: dict[str, Any],
    tool_registry: Any,
    workflow_dir: Path,
    trace: Any = None,
    policy_executor: Any = None,
    cancel_token: Any = None,
    bundle_name: str = "",
    skill_registry: Any = None,
    stream_slot: Any = None,
) -> dict[str, Any]:
    """Execute a script node: import module, call run().

    Scripts must define a typed ``run(ctx: NodeContext, ...) -> Output``
    function where Output is a TypedDict (or None for scripts with no
    outputs).

    Returns:
        dict with keys matching the run() return TypedDict.
    """
    script_rel = node.script or ""
    script_path = workflow_dir / script_rel
    logger.info("[%s] running script %s", node_id, script_rel)

    module = _import_script(
        script_path, node_id,
        skill_registry=skill_registry, bundle_name=bundle_name,
    )
    run_fn = getattr(module, "run", None)
    if run_fn is None:
        raise WorkflowValidationError(
            f"Script '{script_rel}' missing run() function"
        )

    # Create context for the script
    ctx = NodeContext(
        tool_registry, trace=trace, node_id=node_id,
        policy_executor=policy_executor,
        cancel_token=cancel_token,
    )
    if stream_slot is not None:
        ctx._stream_slot = stream_slot

    # Filter inputs to only include parameters the script accepts,
    # so extra workflow inputs don't cause TypeError at call time.
    sig = inspect.signature(run_fn)
    accepted = set(sig.parameters.keys()) - {"ctx"}
    extra = set(resolved_inputs.keys()) - accepted
    if extra:
        logger.warning(
            "[%s] ignoring extra workflow inputs not in run() signature: %s",
            node_id, sorted(extra),
        )
        resolved_inputs = {k: v for k, v in resolved_inputs.items() if k in accepted}

    _wrap_streams(resolved_inputs, ctx)

    try:
        result = run_fn(ctx, **resolved_inputs)
    except Exception as e:
        raise NodeExecutionError(node_id, e) from e

    # Extract declared output keys from return type annotation. The
    # override path may have a bare callable with no enclosing module
    # for the helper to introspect — fall through to no key validation
    # in that case (the override authors are responsible).
    if module is None:
        output_keys = []
    else:
        output_keys = _get_output_keys(run_fn, module)

    if output_keys:
        # Validate outputs
        if not isinstance(result, dict):
            raise NodeExecutionError(
                node_id,
                ValueError(f"Script run() must return a dict, got {type(result).__name__}"),
            )
        for key in output_keys:
            if key not in result:
                raise NodeExecutionError(
                    node_id,
                    ValueError(f"Script output missing declared key '{key}'"),
                )

    logger.info("[%s] script completed with keys: %s", node_id,
                list(result.keys()) if isinstance(result, dict) else [])
    return result


def execute_skill_node(
    node_id: str,
    node: NodeDef,
    resolved_inputs: dict[str, Any],
    tool_registry: Any,
    skill_registry: SkillsRegistry,
    trace: Any = None,
    skill_instances: dict[str, Any] | None = None,
    policy_executor: Any = None,
    cancel_token: Any = None,
) -> dict[str, Any]:
    """Execute a callable skill bundle: look up registered skill, call run().

    Skills follow the same ``run(ctx, ...) -> Output`` contract as scripts
    but are pre-written and loaded from the skill registry. The dispatch
    name is the node's flat ``tool`` field (atomic skills register under
    their bundle name).

    The ``skill_instances`` map (owned by ``WorkflowExecutor``) carries
    per-graph instances of class-based skills so their state persists
    across multiple visits to the same skill node during one workflow
    execution.

    Returns:
        dict with keys matching the skill's Output TypedDict.
    """
    skill_name = node.tool or ""
    logger.info("[%s] calling skill '%s'", node_id, skill_name)

    ctx = NodeContext(
        tool_registry, trace=trace, node_id=node_id,
        policy_executor=policy_executor,
        cancel_token=cancel_token,
    )

    _wrap_streams(resolved_inputs, ctx)

    try:
        result = skill_registry.call(
            skill_name, ctx,
            skill_instances=skill_instances,
            **resolved_inputs,
        )
    except Exception as e:
        raise NodeExecutionError(node_id, e) from e

    if not isinstance(result, dict):
        raise NodeExecutionError(
            node_id,
            ValueError(f"Skill run() must return a dict, got {type(result).__name__}"),
        )

    # Validate output keys against schema
    skill_info = skill_registry.get(skill_name)
    for key in skill_info.schema.outputs:
        if key not in result:
            raise NodeExecutionError(
                node_id,
                ValueError(f"Skill output missing declared key '{key}'"),
            )

    logger.info("[%s] skill '%s' completed with keys: %s",
                node_id, skill_name, list(result.keys()))
    return result


def _get_output_keys(run_fn: Any, module: Any) -> set[str]:
    """The output keys a node MUST return, from run()'s return TypedDict.

    ``__required_keys__`` is the answer, not ``__annotations__``: a field
    written ``NotRequired[T]`` (or any field of a ``total=False`` TypedDict)
    is declared and optional, which is the whole point of the annotation.
    Reading every annotation made a script that returns a measurement only
    when it has one fail on the run where it has none -- so an author's
    choice was between deleting the annotation and returning a placeholder,
    and the scripts that hit it did delete it, which lost the documentation
    the type was there to give.
    """
    try:
        hints = typing.get_type_hints(run_fn, globalns=vars(module))
    except Exception:
        return set()
    return_hint = hints.get("return")
    if return_hint is None or return_hint is type(None):
        return set()
    required = getattr(return_hint, "__required_keys__", None)
    if required is not None:
        return set(required)
    if hasattr(return_hint, "__annotations__"):
        return set(return_hint.__annotations__.keys())
    return set()


def _import_script(
    path: Path,
    node_id: str,
    *,
    skill_registry: Any = None,
    bundle_name: str = "",
):
    """Dynamically import a Python script as a module.

    When *bundle_name* names a registered skill bundle, the module is
    loaded under the bundle's synthetic package
    (``gap_skills.<namespace>.<bundle>.scripts.<stem>``). This makes
    ``__package__`` resolve to a real package via importlib, which lets
    canonical scripts call ``load_prompt(__package__, ...)`` and have
    the loader walk up to find SKILL.md.

    Ad-hoc LLM-emitted scripts (no bundle) keep the synthetic name
    ``_gap_script_<node_id>_<stem>`` and cannot use ``load_prompt``
    — the correct constraint, since they don't belong to a bundle.
    """
    if (
        bundle_name
        and skill_registry is not None
        and bundle_name in skill_registry
    ):
        from gap.skills._registry import _ensure_synthetic_package
        info = skill_registry.get(bundle_name)
        scripts_pkg = f"gap_skills.{info.namespace}.{bundle_name}.scripts"
        # The script may be a per-subgraph copy outside the bundle dir;
        # set the synthetic scripts package to point at *this* file's
        # parent so any sibling imports resolve, but the load_prompt
        # walk-up still terminates at the bundle's SKILL.md (since the
        # parent of scripts is gap_skills.<ns>.<bundle> which is
        # already pinned at the bundle dir).
        _ensure_synthetic_package(
            f"gap_skills.{info.namespace}.{bundle_name}",
            info.bundle_dir,
        )
        _ensure_synthetic_package(scripts_pkg, info.bundle_dir / "scripts")
        module_name = f"{scripts_pkg}.{path.stem}"
    else:
        # Sanitize dots in node_id (e.g. "grasp_pan.check_skip_obb") so the
        # synthetic module_name stays a flat top-level name. Otherwise
        # Python's import system treats the prefix before the first dot
        # as a parent package and tries to look it up — which races
        # under parallel super-steps and crashes with
        # ``ModuleNotFoundError: No module named '_gap_script_<sg>'``.
        _safe_node_id = str(node_id).replace(".", "_")
        module_name = f"_gap_script_{_safe_node_id}_{path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise WorkflowValidationError(f"Cannot load script: {path}")

    from gap.skills._registry import _get_import_lock

    # Serialize concurrent loads of the same module name (parallel
    # super-step nodes that share a canonical-script module path) so no
    # thread observes the half-initialized shell while ``exec_module``
    # is still running.
    with _get_import_lock(module_name):
        module = importlib.util.module_from_spec(spec)
        if module_name.startswith("gap_skills."):
            module.__package__ = module_name.rsplit(".", 1)[0]
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return module
