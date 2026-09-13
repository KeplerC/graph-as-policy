"""Unit tests for the ``gap.requires`` frontmatter block.

Parsing lives in ``parse_skill_md`` (hard errors — a typo'd requires
block must not silently disable a ``gap check`` probe); the softer
cross-checks (tags↔gpu drift, weights without a host module) live in
``validate_bundle_meta`` as warnings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gap_core.skills import SkillRequires

from gap.skills import parse_skill_md
from gap.skills.validate import validate_bundle_meta

FRONT = """\
---
name: {name}
description: A fixture bundle. Use when testing requires parsing.
metadata: {{category: test, tags: [{tags}]}}
gap:
  tools:
    - {name}.run: A fixture tool.
{requires}---
"""


def write_bundle(
    tmp_path: Path,
    *,
    requires: str = "",
    tags: str = "",
    name: str = "req-fixture",
    tools_py: bool = False,
) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "SKILL.md").write_text(
        FRONT.format(name=name, requires=requires, tags=tags)
    )
    if tools_py:
        (d / "tools.py").write_text("")
    return d / "SKILL.md"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_full_block_round_trips(tmp_path: Path):
    md = write_bundle(
        tmp_path,
        requires="  requires: {gpu: true, env: [MY_KEY], env_any: [A, B], weights: true}\n",
    )
    meta = parse_skill_md(md)
    assert meta.requires == SkillRequires(
        gpu=True, env=["MY_KEY"], env_any=["A", "B"], weights=True,
    )


def test_connector_entries_round_trip(tmp_path: Path):
    md = write_bundle(
        tmp_path,
        requires="  requires: {connector: [motion.plan_joint, sim.query]}\n",
    )
    meta = parse_skill_md(md)
    assert meta.requires == SkillRequires(connector=["motion.plan_joint", "sim.query"])


def test_connector_entry_shape_rejected(tmp_path: Path):
    """A connector entry is a flat tool name, ``<prefix>.<name>``; a bare
    word or an empty string would be an entry nothing could ever resolve."""
    for index, bad in enumerate(('[""]', "[noprefix]", '["Motion.Plan"]')):
        # The name is the loop position, not a hash of the string: str.hash is
        # randomised per process (pytest-randomly sets PYTHONHASHSEED), so two
        # of the three could land in one bucket and the second mkdir would
        # raise FileExistsError on some runs and not others.
        md = write_bundle(
            tmp_path, name=f"req-conn-{index}",
            requires=f"  requires: {{connector: {bad}}}\n",
        )
        with pytest.raises(ValueError, match="requires.connector entries"):
            parse_skill_md(md)


def test_absent_vs_explicitly_empty(tmp_path: Path):
    absent = parse_skill_md(write_bundle(tmp_path, name="req-absent"))
    assert absent.requires is None

    empty = parse_skill_md(write_bundle(
        tmp_path, name="req-empty", requires="  requires: {}\n",
    ))
    assert empty.requires == SkillRequires()

    # A bare `requires:` (YAML null) also counts as declared-empty.
    bare = parse_skill_md(write_bundle(
        tmp_path, name="req-bare", requires="  requires:\n",
    ))
    assert bare.requires == SkillRequires()


def test_unknown_subkey_rejected(tmp_path: Path):
    md = write_bundle(tmp_path, requires="  requires: {gpus: true}\n")
    with pytest.raises(ValueError, match="unknown keys.*gpus"):
        parse_skill_md(md)


def test_top_level_requires_rejected_with_nesting_hint(tmp_path: Path):
    d = tmp_path / "req-toplevel"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: req-toplevel\ndescription: x\nrequires: {gpu: true}\n---\n"
    )
    with pytest.raises(ValueError, match="nest them under the `gap:` key"):
        parse_skill_md(d / "SKILL.md")


def test_empty_env_entry_rejected(tmp_path: Path):
    md = write_bundle(tmp_path, requires='  requires: {env: ["  "]}\n')
    with pytest.raises(ValueError, match="non-empty"):
        parse_skill_md(md)


def test_non_mapping_requires_rejected(tmp_path: Path):
    md = write_bundle(tmp_path, requires="  requires: [gpu]\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_skill_md(md)


# ---------------------------------------------------------------------------
# validate_bundle_meta cross-checks (warnings)
# ---------------------------------------------------------------------------


def _warnings(md: Path) -> list[str]:
    meta = parse_skill_md(md)
    issues = validate_bundle_meta(meta, kind="tool", bundle_dir=md.parent)
    return [i.message for i in issues if i.severity == "warning"]


def test_gpu_requires_without_tag_warns(tmp_path: Path):
    md = write_bundle(
        tmp_path, requires="  requires: {gpu: true}\n", tools_py=True,
    )
    assert any("metadata.tags lacks 'gpu'" in w for w in _warnings(md))


def test_gpu_tag_without_requires_warns(tmp_path: Path):
    md = write_bundle(tmp_path, tags="gpu", tools_py=True)
    assert any("does not declare" in w and "gpu" in w for w in _warnings(md))

    # Consistent declarations are clean.
    consistent = write_bundle(
        tmp_path, name="req-consistent", tags="gpu",
        requires="  requires: {gpu: true}\n", tools_py=True,
    )
    warnings = _warnings(consistent)
    assert not any("gpu" in w for w in warnings)


def test_weights_without_host_module_warns(tmp_path: Path):
    md = write_bundle(tmp_path, requires="  requires: {weights: true}\n")
    assert any("weights_cached" in w for w in _warnings(md))

    hosted = write_bundle(
        tmp_path, name="req-hosted",
        requires="  requires: {weights: true}\n", tools_py=True,
    )
    assert not any("weights_cached" in w for w in _warnings(hosted))
