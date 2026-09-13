"""Unit tests for gap.skills.capability (the probes behind ``gap check``).

GPU and LLM-provider probes are exercised with monkeypatched subprocess/
env so they are deterministic on any machine (no-GPU CI included);
bundle probes run against fabricated registries in tmp_path. Bundle
names are prefixed ``rgt-cap-`` and unique per test (the ``gap_skills.*``
synthetic namespace is process-global).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gap.skills import capability as cap
from gap.skills import resolve_registries
from gap.skills.capability import (
    ProbeResult,
    build_check_report,
    dep_fix_hint,
    probe_gpu,
    probe_llm_providers,
)

_PROVIDER_VARS = (
    "OPENROUTER_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_CONFIG",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("GAP_SKILLS_PATH", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for var in _PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)
    # Point the gcloud ADC walk at an empty home so the developer
    # machine's real credentials never leak in.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def write_bundle(
    root: Path,
    name: str,
    *,
    kind: str = "tool",
    gap_block: str = "",
    tools_py: str | None = "",
) -> Path:
    folder = "tools" if kind == "tool" else "skills"
    d = root / folder / name
    d.mkdir(parents=True)
    tools_decl = (
        f"  tools:\n    - {name}.run: A fixture tool.\n" if kind == "tool"
        else "  exit_conditions: {done: ok}\n"
    )
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Fixture. Use when testing checks.\n"
        f"gap:\n{tools_decl}{gap_block}---\n"
    )
    if tools_py is not None:
        (d / "tools.py").write_text(tools_py)
    return d


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


def _fake_run(stdout: str = "", returncode: int = 0, exc: Exception | None = None):
    def fake(argv, capture_output, text, timeout):  # noqa: ARG001
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")
    return fake


def test_probe_gpu_ok(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", _fake_run("NVIDIA RTX 4090, 24564 MiB\n"),
    )
    result = probe_gpu()
    assert result.ok and "RTX 4090" in result.detail


def test_probe_gpu_no_binary(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(exc=FileNotFoundError()))
    result = probe_gpu()
    assert result.status == "missing" and "not on PATH" in result.detail


def test_probe_gpu_nonzero_and_timeout(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(returncode=9))
    assert probe_gpu().status == "missing"
    monkeypatch.setattr(
        subprocess, "run",
        _fake_run(exc=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)),
    )
    assert probe_gpu().status == "error"


def test_probe_llm_providers_all_missing():
    probes = probe_llm_providers()
    assert {p.status for p in probes.values()} == {"missing"}
    assert "export OPENROUTER_API_KEY" in probes["openrouter"].fix_hint
    assert "gcloud auth application-default login" in probes["vertex"].fix_hint


def test_probe_llm_providers_keys_and_adc(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    creds = tmp_path / "adc.json"
    creds.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
    probes = probe_llm_providers()
    assert probes["openrouter"].ok
    assert probes["vertex"].ok
    assert "GOOGLE_APPLICATION_CREDENTIALS" in probes["vertex"].detail


def test_probe_vertex_gcloud_adc_and_broken_pointer(monkeypatch, tmp_path):
    gcloud = tmp_path / "cloudsdk"
    gcloud.mkdir()
    (gcloud / "application_default_credentials.json").write_text("{}")
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(gcloud))
    assert probe_llm_providers()["vertex"].ok

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "gone.json"))
    assert probe_llm_providers()["vertex"].status == "error"


def test_dep_fix_hint_prefers_uv_lock_then_dist(tmp_path):
    (tmp_path / "uv.lock").write_text("")
    assert "uv sync --extra sam3" in dep_fix_hint(
        tmp_path, dist_name="x", bundle="sam3", has_extra=True,
    )
    plain = tmp_path / "plain"
    plain.mkdir()
    assert dep_fix_hint(
        plain, dist_name="lab-skills", bundle="sam3", has_extra=True,
    ) == "pip install 'lab-skills[sam3]'"
    assert "declare" in dep_fix_hint(
        plain, dist_name="lab-skills", bundle="sam3", has_extra=False,
    )


# ---------------------------------------------------------------------------
# build_check_report
# ---------------------------------------------------------------------------


def _report_for(monkeypatch, *paths, gpu_ok: bool = True):
    monkeypatch.setattr(
        cap, "probe_gpu",
        lambda **kw: ProbeResult("ok", "FAKE GPU") if gpu_ok
        else ProbeResult("missing", "no GPU"),
    )
    return build_check_report(resolve_registries(list(paths)))


def test_requirements_env_and_gpu(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    write_bundle(
        reg, "rgt-cap-envtool",
        gap_block="  requires: {env: [RGT_CAP_KEY], gpu: true}\n",
    )
    report = _report_for(monkeypatch, reg, gpu_ok=False)
    bundle = report.bundles[0]
    assert bundle.status == "not-ready"
    failed = {label for label, probe in bundle.requirements if not probe.ok}
    assert failed == {"gpu", "env:RGT_CAP_KEY"}

    monkeypatch.setenv("RGT_CAP_KEY", "x")
    report = _report_for(monkeypatch, reg, gpu_ok=True)
    assert report.bundles[0].status == "ready"


def test_requirements_env_any(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    write_bundle(
        reg, "rgt-cap-anytool",
        gap_block="  requires: {env_any: [RGT_CAP_A, RGT_CAP_B]}\n",
    )
    report = _report_for(monkeypatch, reg)
    assert report.bundles[0].status == "not-ready"
    label, probe = report.bundles[0].requirements[0]
    assert label == "env_any:RGT_CAP_A|RGT_CAP_B"
    assert "RGT_CAP_A" in probe.fix_hint

    monkeypatch.setenv("RGT_CAP_B", "x")
    report = _report_for(monkeypatch, reg)
    assert report.bundles[0].status == "ready"


def test_weights_hook_states(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    write_bundle(
        reg, "rgt-cap-wcached",
        gap_block="  requires: {weights: true}\n",
        tools_py="def weights_cached():\n    return True\n",
    )
    write_bundle(
        reg, "rgt-cap-wmissing",
        gap_block="  requires: {weights: true}\n",
        tools_py="def weights_cached():\n    return False\n",
    )
    write_bundle(
        reg, "rgt-cap-wnohook",
        gap_block="  requires: {weights: true}\n",
    )
    report = _report_for(monkeypatch, reg)
    by_name = {b.name: b for b in report.bundles}
    assert by_name["rgt-cap-wcached"].weights.status == "ok"
    missing = by_name["rgt-cap-wmissing"]
    assert missing.weights.status == "missing"
    assert "gap skills check --download" in missing.weights.fix_hint
    # Weights never block readiness — they download on first use.
    assert missing.status == "ready"
    assert by_name["rgt-cap-wnohook"].weights.status == "unknown"


def test_missing_deps_get_fix_hint(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    write_bundle(
        reg, "rgt-cap-broken",
        tools_py="import rgt_cap_no_such_module\n",
    )
    (reg / "pyproject.toml").write_text(
        '[project]\nname = "cap-fixture-skills"\nversion = "0"\n'
        '[project.optional-dependencies]\n"rgt-cap-broken" = []\n'
    )
    report = _report_for(monkeypatch, reg)
    bundle = report.bundles[0]
    assert bundle.status == "not-ready"
    assert "rgt_cap_no_such_module" in bundle.deps.detail
    assert "pip install 'cap-fixture-skills[rgt-cap-broken]'" in bundle.deps.fix_hint


def test_skill_rollup_blocked_and_unknown(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    write_bundle(
        reg, "rgt-cap-rolltool",
        gap_block="  requires: {env: [RGT_CAP_ROLL_KEY]}\n",
    )
    skill_dir = reg / "skills" / "rgt-cap-rollskill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: rgt-cap-rollskill\n"
        "description: Fixture. Use when testing rollups.\n"
        "gap:\n"
        "  exit_conditions: {done: ok}\n"
        "  allowed_tools:\n"
        "    - robot.get_observation\n"
        "    - rgt-cap-rolltool.run\n"
        "    - nosuch.fn\n"
        "---\n"
    )
    report = _report_for(monkeypatch, reg)
    assert len(report.skills) == 1
    skill = report.skills[0]
    assert skill.tools["robot.get_observation"] == "connector"
    assert skill.tools["rgt-cap-rolltool.run"] == "rgt-cap-rolltool"
    assert skill.blocked_by == ["rgt-cap-rolltool"]
    assert skill.unknown_tools == ["nosuch.fn"]
    assert skill.status == "blocked"

    monkeypatch.setenv("RGT_CAP_ROLL_KEY", "x")
    report = _report_for(monkeypatch, reg)
    assert report.skills[0].blocked_by == []
    assert report.skills[0].status == "ready"


def test_shadowed_bundle_is_inert_and_never_imported(monkeypatch, tmp_path):
    winner = tmp_path / "winner"
    loser = tmp_path / "loser"
    write_bundle(winner, "rgt-cap-shade", tools_py="")
    write_bundle(
        loser, "rgt-cap-shade",
        tools_py="raise RuntimeError('shadowed bundle must never import')\n",
    )
    report = _report_for(monkeypatch, winner, loser)
    by_registry = {b.registry: b for b in report.bundles}
    assert by_registry["winner"].status == "ready"
    shadowed = by_registry["loser"]
    assert shadowed.status == "shadowed"
    assert shadowed.shadowed_by == "winner"
    # Its deps probe never ran (the RuntimeError tools.py was not imported).
    assert shadowed.deps.ok


def test_connector_requirement_is_reported_not_blocking(tmp_path: Path):
    """A skill that needs a richer connector's tool says so under
    ``requires.connector``; the rollup records the need and stays ready,
    because a connector cannot be probed statically."""
    write_bundle(
        tmp_path, "needs-planner", kind="skill", tools_py=None,
        gap_block=(
            "  allowed_tools: [robot.get_observation, motion.plan_joint]\n"
            "  requires: {connector: [motion.plan_joint]}\n"
        ),
    )
    report = build_check_report(resolve_registries([tmp_path]))
    skill = report.skills[0]
    assert skill.name == "needs-planner"
    assert skill.unknown_tools == []
    assert skill.connector_required == ["motion.plan_joint"]
    assert skill.tools["motion.plan_joint"] == "connector"
    assert skill.status == "ready"
    assert skill.to_json_dict()["connector_required"] == ["motion.plan_joint"]



def test_json_schema_shape(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    write_bundle(reg, "rgt-cap-json")
    report = _report_for(monkeypatch, reg)
    data = report.to_json_dict()
    assert data["schema_version"] == 1
    assert set(data) == {
        "schema_version", "environment", "registries", "bundles", "skills",
    }
    assert data["environment"]["gpu"]["status"] == "ok"
    (bundle,) = data["bundles"]
    assert bundle["name"] == "rgt-cap-json"
    assert bundle["status"] == "ready"
    assert data["registries"][0]["source"] == "flag"
