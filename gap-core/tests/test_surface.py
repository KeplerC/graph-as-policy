"""Smoke tests for the gap-core public surface.

These pin the bundle-author API: every import a tool / policy bundle relies
on must keep working without the gap-runtime distribution installed. The
deeper behavior tests for each module live in the gap-runtime test suite
(tests/skills/, tests/tools/, tests/runtime/) where the parser, registry,
and integration paths are exercised.
"""

from __future__ import annotations


def test_tool_decorator_imports():
    from gap_core.tools import tool
    assert callable(tool)


def test_types_imports():
    # The bundle-author vocabulary — TypedDicts and ndarray aliases bundles
    # use in their @tool signatures. Just verify they import; their schemas
    # are exercised in the parser tests under gap/tests/skills/.
    from gap_core.types import (  # noqa: F401
        BoundingBox2D,
        CameraFrame,
        JointState,
        Mask,
        Observation,
        OrientedBoundingBox,
        PointCloud,
        Quaternion,
        Se3Pose,
        Trajectory,
        Vec3,
    )


def test_errors_imports():
    from gap_core.errors import (
        GuardLimitExceeded,
        PerceptionFailed,
        PipelineError,
        PlanningFailed,
        ToolError,
        ValidationFailed,
    )
    assert issubclass(PerceptionFailed, PipelineError)
    assert issubclass(PlanningFailed, PipelineError)
    # GuardLimitExceeded inherits BaseException intentionally — it must NOT be
    # caught by `except Exception:` inside skill code.
    assert issubclass(GuardLimitExceeded, BaseException)
    assert not issubclass(GuardLimitExceeded, Exception)


def test_skills_meta_dataclasses_construct():
    from gap_core.skills import (
        CanonicalScript,
        Param,
        Serving,
        Skill,
        SkillMeta,
        SkillRequires,
    )

    meta = SkillMeta(description="x. Use when testing.")
    assert meta.description.startswith("x")
    assert meta.kind == "skill"

    p = Param(description="param desc")
    assert p.description == "param desc"

    serving = Serving(command=["python", "-m", "x"])
    assert serving.command == ["python", "-m", "x"]
    assert serving.protocol == "in-process"

    req = SkillRequires(gpu=True, connector=["motion.plan_joint"])
    assert req.gpu is True
    assert req.connector == ["motion.plan_joint"]

    cs = CanonicalScript(name="n", path="scripts/n.py")
    assert cs.name == "n"

    class _MySkill(Skill):
        meta = SkillMeta(description="my skill. Use when testing.")
    assert _MySkill.meta.description.startswith("my skill")


def test_schema_type_registry_resolves():
    from gap_core.schema import TYPE_REGISTRY, resolve_type
    assert "Se3Pose" in TYPE_REGISTRY
    assert resolve_type("Se3Pose") is TYPE_REGISTRY["Se3Pose"]
    # The feature/payload/plan/curve vocabulary the manipulation skills declare.
    for name in ("FunctionalFeature", "AttachedObject", "PoseSequence", "Centerline"):
        assert name in TYPE_REGISTRY, name


def test_no_gap_runtime_import_at_load():
    """Importing every gap-core submodule MUST NOT pull in gap-runtime
    transitively — that would defeat the whole point of the split."""
    import sys
    # Fresh state: drop any cached gap-runtime modules.
    for name in list(sys.modules):
        if name == "gap" or name.startswith("gap."):
            del sys.modules[name]

    import gap_core.tools  # noqa: F401
    import gap_core.types  # noqa: F401
    import gap_core.errors  # noqa: F401
    import gap_core.schema  # noqa: F401
    import gap_core.skills  # noqa: F401

    leaked = [n for n in sys.modules if n == "gap" or n.startswith("gap.")]
    assert not leaked, (
        f"gap-core import leaked gap-runtime modules: {sorted(leaked)}"
    )
