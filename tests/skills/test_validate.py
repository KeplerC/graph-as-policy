"""Unit tests for gap.skills.validate — bundle format validation.

The same rules ``gap skills check`` enforces (and the open-robot-skills repo
test suite calls), exercised on synthetic tmp_path checkouts.
"""

from __future__ import annotations

from pathlib import Path

from gap.skills.validate import (
    BundleReport,
    connector_tool_names,
    known_tool_names,
    load_checkout_extras,
    validate_checkout,
)

_TOOL_MD = """\
---
name: good-tool
description: Compute things with a model. Use when a test needs a clean tool bundle.
gap:
  tools:
    - good-tool.compute: Compute a thing.
---

# good-tool
"""

_SKILL_MD = """\
---
name: good-skill
description: Do a manipulation. Use when a test needs a clean skill bundle.
gap:
  allowed_tools: [robot.get_observation, good-tool.compute]
  exit_conditions:
    done: It worked.
    failed: It did not.
  produces_outputs: {"<name>_obb": OrientedBoundingBox}
  required_inputs: {target_obb: OrientedBoundingBox}
  canonical_scripts:
    - main: scripts/main.py
  prompts: {greet: prompts/greet.md}
  references:
    - {title: Design, path: references/design.md}
---

# good-skill
"""


def _write(root: Path, rel: str, text: str = "stub\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _make_checkout(root: Path) -> Path:
    _write(root, "tools/good-tool/SKILL.md", _TOOL_MD)
    _write(root, "skills/good-skill/SKILL.md", _SKILL_MD)
    _write(root, "skills/good-skill/scripts/main.py", "def run(ctx):\n    return {}\n")
    _write(root, "skills/good-skill/prompts/greet.md")
    _write(root, "skills/good-skill/references/design.md")
    _write(root, "pyproject.toml", (
        '[project]\nname = "open-robot-skills"\nversion = "0"\n'
        "[project.optional-dependencies]\n"
        "good-tool = []\ngood-skill = []\n"
    ))
    return root


def _by_name(reports: list[BundleReport]) -> dict[str, BundleReport]:
    return {r.name: r for r in reports}


def test_clean_checkout_passes(tmp_path: Path):
    reports = _by_name(validate_checkout(_make_checkout(tmp_path)))
    assert set(reports) == {"good-tool", "good-skill"}
    for r in reports.values():
        assert r.status == "PASS", [str(i) for i in r.issues]
    assert reports["good-tool"].kind == "tool"
    assert reports["good-skill"].kind == "skill"


def test_connector_tools_are_known(tmp_path: Path):
    names = connector_tool_names()
    assert "robot.get_observation" in names
    assert "robot.go_to_pose" in names
    assert "sim.reset" in names
    # Bundle-declared tools extend the namespace.
    root = _make_checkout(tmp_path)
    metas = [r.meta for r in validate_checkout(root)]
    assert "good-tool.compute" in known_tool_names(metas)


def test_use_when_heuristic_warns(tmp_path: Path):
    root = _make_checkout(tmp_path)
    _write(root, "tools/terse-tool/SKILL.md", (
        "---\nname: terse-tool\ndescription: Computes things.\n"
        "gap:\n  tools:\n    - terse-tool.go: Go.\n---\n# t\n"
    ))
    _write(root, "pyproject.toml", (
        '[project]\nname = "open-robot-skills"\nversion = "0"\n'
        "[project.optional-dependencies]\n"
        "good-tool = []\ngood-skill = []\nterse-tool = []\n"
    ))
    report = _by_name(validate_checkout(root))["terse-tool"]
    assert report.status == "WARN"
    assert any("Use when" in i.message for i in report.warnings)


def test_tool_bundle_without_gap_tools_fails(tmp_path: Path):
    root = _make_checkout(tmp_path)
    _write(root, "tools/toolless/SKILL.md", (
        "---\nname: toolless\ndescription: Nothing here. Use when never.\n---\n# t\n"
    ))
    report = _by_name(validate_checkout(root))["toolless"]
    assert report.status == "FAIL"
    assert any("gap.tools" in i.message for i in report.errors)


def test_skill_without_exit_conditions_fails_unless_callable(tmp_path: Path):
    root = _make_checkout(tmp_path)
    _write(root, "skills/exitless/SKILL.md", (
        "---\nname: exitless\ndescription: No exits. Use when testing.\n---\n# s\n"
    ))
    # A callable-unit skill (gap.tools declared) is exempt, like
    # pi05-libero / tracking-objects in the real checkout.
    _write(root, "skills/callable-unit/SKILL.md", (
        "---\nname: callable-unit\ndescription: Runs as one unit. Use when testing.\n"
        "gap:\n  tools:\n    - callable-unit.run: Run it.\n---\n# s\n"
    ))
    reports = _by_name(validate_checkout(root))
    assert reports["exitless"].status == "FAIL"
    assert any("exit_conditions" in i.message for i in reports["exitless"].errors)
    assert not any("exit_conditions" in i.message for i in reports["callable-unit"].errors)


def test_missing_referenced_paths_fail(tmp_path: Path):
    root = _make_checkout(tmp_path)
    (root / "skills/good-skill/prompts/greet.md").unlink()
    (root / "skills/good-skill/scripts/main.py").unlink()
    report = _by_name(validate_checkout(root))["good-skill"]
    assert report.status == "FAIL"
    messages = " | ".join(i.message for i in report.errors)
    assert "scripts/main.py" in messages
    assert "prompts/greet.md" in messages


def test_unknown_allowed_tools_fail(tmp_path: Path):
    root = _make_checkout(tmp_path)
    md = (root / "skills/good-skill/SKILL.md").read_text().replace(
        "allowed_tools: [robot.get_observation, good-tool.compute]",
        "allowed_tools: [robot.get_observation, nonexistent.tool]",
    )
    _write(root, "skills/good-skill/SKILL.md", md)
    report = _by_name(validate_checkout(root))["good-skill"]
    assert report.status == "FAIL"
    assert any("nonexistent.tool" in i.message for i in report.errors)


def test_connector_requirement_makes_a_tool_resolvable(tmp_path: Path):
    """A skill written against a richer connector declares what it needs
    under ``gap.requires.connector``; the checker takes its word for those
    names and only those names."""
    root = _make_checkout(tmp_path)
    md = (root / "skills/good-skill/SKILL.md").read_text().replace(
        "allowed_tools: [robot.get_observation, good-tool.compute]",
        "allowed_tools: [robot.get_observation, good-tool.compute, motion.plan_joint]",
    )
    _write(root, "skills/good-skill/SKILL.md", md)
    report = _by_name(validate_checkout(root))["good-skill"]
    assert report.status == "FAIL"
    assert any("motion.plan_joint" in i.message for i in report.errors)

    declared = md.replace(
        "gap:\n", "gap:\n  requires: {connector: [motion.plan_joint]}\n", 1,
    )
    _write(root, "skills/good-skill/SKILL.md", declared)
    report = _by_name(validate_checkout(root))["good-skill"]
    assert not any("motion.plan_joint" in i.message for i in report.errors)


def test_connector_requirement_naming_a_provided_tool_warns(tmp_path: Path):
    root = _make_checkout(tmp_path)
    md = (root / "skills/good-skill/SKILL.md").read_text().replace(
        "gap:\n",
        "gap:\n  requires: {connector: [good-tool.compute, robot.get_observation]}\n",
        1,
    )
    _write(root, "skills/good-skill/SKILL.md", md)
    report = _by_name(validate_checkout(root))["good-skill"]
    warnings = " | ".join(i.message for i in report.warnings)
    assert "good-tool.compute" in warnings and "already declares" in warnings
    assert "robot.get_observation" in warnings and "every gap connector" in warnings
    assert not report.errors


def test_unnamespaced_declared_tool_fails(tmp_path: Path):
    root = _make_checkout(tmp_path)
    md = _TOOL_MD.replace("good-tool.compute", "compute")
    _write(root, "tools/good-tool/SKILL.md", md)
    report = _by_name(validate_checkout(root))["good-tool"]
    assert report.status == "FAIL"
    assert any("not namespaced" in i.message for i in report.errors)


def test_unknown_output_type_fails(tmp_path: Path):
    root = _make_checkout(tmp_path)
    md = (root / "skills/good-skill/SKILL.md").read_text().replace(
        "OrientedBoundingBox}", "NotAType}",
    )
    _write(root, "skills/good-skill/SKILL.md", md)
    report = _by_name(validate_checkout(root))["good-skill"]
    assert report.status == "FAIL"
    assert any("NotAType" in i.message for i in report.errors)


def test_missing_pip_extra_warns(tmp_path: Path):
    root = _make_checkout(tmp_path)
    _write(root, "pyproject.toml", (
        '[project]\nname = "open-robot-skills"\nversion = "0"\n'
        "[project.optional-dependencies]\ngood-tool = []\n"
    ))
    reports = _by_name(validate_checkout(root))
    assert reports["good-skill"].status == "WARN"
    assert any("pip extra" in i.message for i in reports["good-skill"].warnings)
    # Without a pyproject at all, the convention check is skipped entirely.
    (root / "pyproject.toml").unlink()
    assert load_checkout_extras(root) is None
    reports = _by_name(validate_checkout(root))
    assert reports["good-skill"].status == "PASS"


def test_rejected_skill_md_becomes_fail_report(tmp_path: Path):
    root = _make_checkout(tmp_path)
    _write(root, "skills/renamed/SKILL.md", (
        "---\nname: original-name\ndescription: Name mismatch. Use when testing.\n---\n# x\n"
    ))
    report = _by_name(validate_checkout(root))["renamed"]
    assert report.meta is None
    assert report.status == "FAIL"
    assert any("SKILL.md rejected" in i.message for i in report.errors)
