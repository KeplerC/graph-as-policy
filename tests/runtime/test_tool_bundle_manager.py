"""Lifecycle + registration paths for ToolBundleManager.

The real ToolClient spawns a subprocess via `uv run --project`; here we
use a fake ToolClient class that records calls without spawning, so the
manager's discovery + registration + teardown paths are exercised
end-to-end without paying the uv-sync cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from gap_core.tools import ToolRegistry
from gap_core.tools._registry import RpcAdapter

from gap.runtime import tool_bundle_manager as tbm
from gap.runtime.tool_bundle_boot import (
    boot_tool_bundles,
    required_rpc_tool_bundles,
    script_tool_names,
)
from gap.runtime.tool_bundle_manager import (
    LINGER_ENV,
    ToolBundleManager,
    ToolBundleStartupError,
    close_shared_clients,
    shared_client_counts,
)


class _FakeCatalogEntry:
    def __init__(self, name, summary="", tags=(), schema_inputs=None, schema_outputs=None):
        self.name = name
        self.summary = summary
        self.tags = list(tags)
        self.metadata = {}
        self.schema_inputs = schema_inputs or {}
        self.schema_outputs = schema_outputs or {}


class _FakeToolClient:
    """Stand-in for gap_core.rpc.client.ToolClient — no subprocess, no msgpack.

    Records every call/close so the manager's behavior can be asserted. The
    constructor's signature mirrors ToolClient so the manager swap is
    transparent.
    """

    def __init__(self, bundle_name, bundle_dir, env=None, evict_grace_s=5.0,
                 call_timeout_s=None, catalog=None, on_call=None,
                 raise_on_init: bool = False):
        if raise_on_init:
            raise RuntimeError(f"bundle {bundle_name}: synthetic boot failure")
        self.bundle_name = bundle_name
        self.bundle_dir = bundle_dir
        self.call_timeout_s = call_timeout_s
        self.catalog = catalog or [
            _FakeCatalogEntry(name=f"{bundle_name}.echo", summary="echo")
        ]
        self.calls = []
        self.closed = False
        self.dead = False  # set by a test: the subprocess died under us
        self._on_call = on_call

    def call(self, tool, /, **kwargs):
        self.calls.append((tool, kwargs))
        if self._on_call is not None:
            return self._on_call(tool, **kwargs)
        return {"echoed": kwargs}

    def close(self):
        self.closed = True

    def alive(self):
        return not (self.dead or self.closed)


class _FakeSkillInfo:
    def __init__(self, *, kind="tool", protocol="stdio-msgpack",
                 bundle_dir="/tmp/fake", env=None, command=("python", "x")):
        if protocol is None:
            serving = None
        else:
            serving = SimpleNamespace(
                command=list(command), protocol=protocol,
                env=dict(env or {}), requires_gpu=False, weights_uri="",
            )
        self.kind = kind
        self.meta = SimpleNamespace(
            serving=serving, bundle_dir=Path(bundle_dir), tags=[],
        )


class _FakeSkillRegistry:
    def __init__(self, infos):
        self._infos = infos

    def get(self, name):
        if name not in self._infos:
            raise KeyError(name)
        return self._infos[name]


@pytest.fixture
def fake_client_cls(monkeypatch):
    """Patch ToolClient at the manager's import site so boot_all uses
    _FakeToolClient instead of spawning real subprocesses."""
    instances: list[_FakeToolClient] = []

    def _factory(*args, **kwargs):
        c = _FakeToolClient(*args, **kwargs)
        instances.append(c)
        return c

    monkeypatch.setattr("gap_core.rpc.client.ToolClient", _factory)
    return instances


# ---------------------------------------------------------------------------
# ToolBundleManager
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_shared_clients(monkeypatch):
    """The client cache is process-global by design, so a test that leaks a
    reference would be read by the next one as a shared server. Lingering is
    off unless a test turns it on, whatever the developer's shell says."""
    monkeypatch.delenv(LINGER_ENV, raising=False)
    tbm._SHARED.clear()
    yield
    tbm._SHARED.clear()


def _two_worlds(tmp_path, fake_client_cls):
    """Two managers, two registries, one bundle -- a wave of two worlds."""
    registry = _FakeSkillRegistry({
        "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3", protocol="stdio-msgpack"),
    })
    return [
        ToolBundleManager(skill_registry=registry, tool_registry=ToolRegistry())
        for _ in range(2)
    ]


def test_two_worlds_share_one_server(fake_client_cls, tmp_path):
    """The reason this exists. Twelve worlds booting twelve segmenters is
    6.3 GiB of GPU each and an out-of-memory crash mid-wave; they are one
    process and want one server."""
    first, second = _two_worlds(tmp_path, fake_client_cls)
    first.boot_all(["sam3"])
    second.boot_all(["sam3"])
    assert len(fake_client_cls) == 1, "a second world spawned a second subprocess"
    assert shared_client_counts() == {"sam3": 2}
    # Both registries dispatch, and to the same process.
    assert first._managed["sam3"].client is second._managed["sam3"].client


def test_the_server_outlives_the_first_world_to_finish(fake_client_cls, tmp_path):
    """Worlds of a wave end at different steps. The first to call shutdown
    must not close a server its neighbours are mid-call on."""
    first, second = _two_worlds(tmp_path, fake_client_cls)
    first.boot_all(["sam3"])
    second.boot_all(["sam3"])
    client = fake_client_cls[0]
    first.shutdown_all()
    assert not client.closed, "the first world to finish killed the shared server"
    assert shared_client_counts() == {"sam3": 1}
    second.shutdown_all()
    assert client.closed
    assert shared_client_counts() == {}


def test_shutdown_stays_idempotent_when_shared(fake_client_cls, tmp_path):
    """A double shutdown must not release a reference it does not hold."""
    first, second = _two_worlds(tmp_path, fake_client_cls)
    first.boot_all(["sam3"])
    second.boot_all(["sam3"])
    first.shutdown_all()
    first.shutdown_all()
    assert shared_client_counts() == {"sam3": 1}
    assert not fake_client_cls[0].closed


def test_a_different_environment_is_a_different_server(fake_client_cls, tmp_path):
    """Two graphs asking for one bundle under different ``serving.env`` want
    different processes; handing the second the first's would be a silent
    wrong answer."""
    one = ToolBundleManager(
        skill_registry=_FakeSkillRegistry({
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3",
                                   protocol="stdio-msgpack", env={"GAP_SAM3_DEVICE": "cuda:0"}),
        }),
        tool_registry=ToolRegistry(),
    )
    two = ToolBundleManager(
        skill_registry=_FakeSkillRegistry({
            "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3",
                                   protocol="stdio-msgpack", env={"GAP_SAM3_DEVICE": "cuda:1"}),
        }),
        tool_registry=ToolRegistry(),
    )
    one.boot_all(["sam3"])
    two.boot_all(["sam3"])
    assert len(fake_client_cls) == 2
    assert shared_client_counts() == {"sam3": 1}


def test_boot_registers_catalog_under_rpc_transport(fake_client_cls, tmp_path):
    skill_registry = _FakeSkillRegistry({
        "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3"),
    })
    tool_registry = ToolRegistry()

    mgr = ToolBundleManager(skill_registry, tool_registry)
    mgr.boot_all(["sam3"])
    try:
        assert mgr.loaded_bundles() == ["sam3"]
        descriptor = tool_registry.get("sam3.echo")
        assert descriptor.transport == "rpc"
        assert descriptor.metadata["bundle"] == "sam3"
        # Invoking through the registry must hit the fake client.
        result = tool_registry.invoke("sam3.echo", None, x=1)
        assert result == {"echoed": {"x": 1}}
        assert fake_client_cls[0].calls == [("sam3.echo", {"x": 1})]
    finally:
        mgr.shutdown_all()
    assert fake_client_cls[0].closed is True
    assert mgr.loaded_bundles() == []


def test_boot_skips_non_rpc_bundles(fake_client_cls, tmp_path):
    """Bundles whose serving is None or in-process must NOT be booted —
    they keep using the in-process @tool path."""
    skill_registry = _FakeSkillRegistry({
        "geometry": _FakeSkillInfo(protocol=None, bundle_dir=tmp_path / "geom"),
        "vlm": _FakeSkillInfo(protocol="in-process", bundle_dir=tmp_path / "vlm"),
    })
    tool_registry = ToolRegistry()
    mgr = ToolBundleManager(skill_registry, tool_registry)
    mgr.boot_all(["geometry", "vlm"])
    try:
        assert mgr.loaded_bundles() == []
        assert fake_client_cls == []
    finally:
        mgr.shutdown_all()


def test_boot_unknown_bundle_raises(fake_client_cls):
    skill_registry = _FakeSkillRegistry({})  # empty
    tool_registry = ToolRegistry()
    mgr = ToolBundleManager(skill_registry, tool_registry)
    with pytest.raises(KeyError):
        mgr.boot_all(["nope"])


def test_boot_client_init_failure_propagates(monkeypatch, tmp_path):
    """If ToolClient raises during catalog handshake, boot must surface
    ToolBundleStartupError and not leak any half-started subprocesses."""
    skill_registry = _FakeSkillRegistry({
        "bad": _FakeSkillInfo(bundle_dir=tmp_path / "bad"),
    })

    def _failing(*args, **kwargs):
        return _FakeToolClient(*args, raise_on_init=True, **kwargs)

    monkeypatch.setattr("gap_core.rpc.client.ToolClient", _failing)
    tool_registry = ToolRegistry()
    mgr = ToolBundleManager(skill_registry, tool_registry)
    with pytest.raises(ToolBundleStartupError, match="failed to start"):
        mgr.boot_all(["bad"])


def test_boot_name_collision_tears_down(fake_client_cls, tmp_path):
    """A second bundle exporting the same tool name as the first must
    surface a startup error AND tear down both clients (not leak the
    first)."""
    overlap_catalog = [_FakeCatalogEntry(name="shared.tool", summary="x")]

    def _on_factory_side_effect(name):
        infos = {
            "first": _FakeSkillInfo(bundle_dir=tmp_path / "first"),
            "second": _FakeSkillInfo(bundle_dir=tmp_path / "second"),
        }
        return infos[name]

    # Monkey-patch fake client to always return the overlapping catalog.
    instances: list[_FakeToolClient] = []

    def _factory(*args, **kwargs):
        c = _FakeToolClient(*args, **{**kwargs, "catalog": overlap_catalog})
        instances.append(c)
        return c

    import gap_core.rpc.client as client_mod
    with patch.object(client_mod, "ToolClient", _factory):
        skill_registry = _FakeSkillRegistry({
            "first": _FakeSkillInfo(bundle_dir=tmp_path / "first"),
            "second": _FakeSkillInfo(bundle_dir=tmp_path / "second"),
        })
        tool_registry = ToolRegistry()
        mgr = ToolBundleManager(skill_registry, tool_registry)
        with pytest.raises(ToolBundleStartupError, match="cannot register"):
            mgr.boot_all(["first", "second"])

    # The second client got booted but its name collided; both must be closed.
    assert all(c.closed for c in instances), [c.closed for c in instances]


def test_shutdown_is_idempotent(fake_client_cls, tmp_path):
    skill_registry = _FakeSkillRegistry({
        "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3"),
    })
    mgr = ToolBundleManager(skill_registry, ToolRegistry())
    mgr.boot_all(["sam3"])
    mgr.shutdown_all()
    mgr.shutdown_all()  # second call is a no-op
    assert fake_client_cls[0].closed is True


# ---------------------------------------------------------------------------
# GAP_TOOL_BUNDLE_LINGER: a server that outlives its episode
# ---------------------------------------------------------------------------


def _episode(tmp_path):
    """One episode's manager over a fresh ToolRegistry, the way a simulator
    connector hands ``gap.execute`` a new registry every episode."""
    registry = _FakeSkillRegistry({
        "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3", protocol="stdio-msgpack"),
    })
    return ToolBundleManager(skill_registry=registry, tool_registry=ToolRegistry())


def test_without_linger_the_last_release_still_closes(fake_client_cls, tmp_path, monkeypatch):
    """Unset (or anything but a yes) is the reference count exactly: the
    episode that lets go last closes the server and the next one boots anew."""
    monkeypatch.setenv(LINGER_ENV, "0")
    first = _episode(tmp_path)
    first.boot_all(["sam3"])
    first.shutdown_all()
    assert fake_client_cls[0].closed
    assert shared_client_counts() == {}
    _episode(tmp_path).boot_all(["sam3"])
    assert len(fake_client_cls) == 2


def test_linger_hands_the_next_episode_the_same_server(fake_client_cls, tmp_path, monkeypatch):
    """The reason it exists: thirty episodes in one process loaded the
    segmenter thirty times, ~20 s and 6.3 GiB of GPU apiece."""
    monkeypatch.setenv(LINGER_ENV, "1")
    first = _episode(tmp_path)
    first.boot_all(["sam3"])
    first.shutdown_all()
    client = fake_client_cls[0]
    assert not client.closed, "the last release closed a server lingering was asked to keep"
    assert shared_client_counts() == {"sam3": 0}

    second = _episode(tmp_path)
    second.boot_all(["sam3"])
    assert len(fake_client_cls) == 1, "the next episode booted a second server"
    assert second._managed["sam3"].client is client
    assert shared_client_counts() == {"sam3": 1}


def test_a_reused_server_registers_into_the_new_episodes_registry(fake_client_cls, tmp_path, monkeypatch):
    """Per-episode state is the registry, and it is rebuilt every episode: the
    reused server's catalog has to land in the new one, under the RPC
    transport, and dispatch to the server that is actually alive."""
    monkeypatch.setenv(LINGER_ENV, "1")
    first = _episode(tmp_path)
    first.boot_all(["sam3"])
    first.shutdown_all()

    second = _episode(tmp_path)
    second.boot_all(["sam3"])
    registry = second._tool_registry
    assert registry is not first._tool_registry
    assert registry.get("sam3.echo").transport == "rpc"
    assert registry.invoke("sam3.echo", None, x=2) == {"echoed": {"x": 2}}
    assert fake_client_cls[0].calls == [("sam3.echo", {"x": 2})]


def test_linger_respawns_a_server_that_died_while_parked(fake_client_cls, tmp_path, monkeypatch):
    """A parked server can die between episodes -- a reply timeout kills its
    process group, so can the OOM killer. Handing that corpse to the next
    episode would fail its first perception call; it is replaced instead."""
    monkeypatch.setenv(LINGER_ENV, "1")
    first = _episode(tmp_path)
    first.boot_all(["sam3"])
    first.shutdown_all()
    corpse = fake_client_cls[0]
    corpse.dead = True

    second = _episode(tmp_path)
    second.boot_all(["sam3"])
    assert len(fake_client_cls) == 2, "a dead server was handed out instead of respawned"
    fresh = fake_client_cls[1]
    assert second._managed["sam3"].client is fresh
    assert corpse.closed, "the dead server's pipes were not reaped"
    assert shared_client_counts() == {"sam3": 1}
    assert second._tool_registry.invoke("sam3.echo", None, y=1) == {"echoed": {"y": 1}}
    assert fresh.calls == [("sam3.echo", {"y": 1})]


def test_a_respawn_that_fails_still_reaps_the_dead_server(fake_client_cls, tmp_path, monkeypatch):
    """The replacement not coming up is a startup error, as any boot failure
    is -- and the corpse is still closed, not left in the cache to be handed
    out on the next try."""
    monkeypatch.setenv(LINGER_ENV, "1")
    first = _episode(tmp_path)
    first.boot_all(["sam3"])
    first.shutdown_all()
    corpse = fake_client_cls[0]
    corpse.dead = True

    def _failing(*args, **kwargs):
        return _FakeToolClient(*args, raise_on_init=True, **kwargs)

    monkeypatch.setattr("gap_core.rpc.client.ToolClient", _failing)
    with pytest.raises(ToolBundleStartupError, match="failed to start"):
        _episode(tmp_path).boot_all(["sam3"])
    assert corpse.closed
    assert shared_client_counts() == {}


def test_a_dead_server_is_replaced_without_corrupting_its_holders_count(fake_client_cls, tmp_path, monkeypatch):
    """A world still holding a server that has died keeps its reference to
    that one; the replacement a later world gets is counted separately, so the
    old holder's release closes the corpse and nothing else."""
    monkeypatch.setenv(LINGER_ENV, "1")
    holder, latecomer = _two_worlds(tmp_path, fake_client_cls)
    holder.boot_all(["sam3"])
    corpse = fake_client_cls[0]
    corpse.dead = True
    latecomer.boot_all(["sam3"])
    fresh = fake_client_cls[1]
    assert latecomer._managed["sam3"].client is fresh
    assert not corpse.closed, "a server another world holds was closed under it"

    holder.shutdown_all()
    assert corpse.closed
    assert not fresh.closed
    assert shared_client_counts() == {"sam3": 1}
    latecomer.shutdown_all()
    assert not fresh.closed, "the replacement did not linger"
    assert shared_client_counts() == {"sam3": 0}


def test_linger_keeps_the_count_for_worlds_of_one_wave(fake_client_cls, tmp_path, monkeypatch):
    """The fleet path: worlds of a wave still share one server and count it;
    lingering only changes what happens when the count reaches zero."""
    monkeypatch.setenv(LINGER_ENV, "1")
    first, second = _two_worlds(tmp_path, fake_client_cls)
    first.boot_all(["sam3"])
    second.boot_all(["sam3"])
    assert shared_client_counts() == {"sam3": 2}
    first.shutdown_all()
    first.shutdown_all()
    assert shared_client_counts() == {"sam3": 1}
    second.shutdown_all()
    assert shared_client_counts() == {"sam3": 0}
    third, fourth = _two_worlds(tmp_path, fake_client_cls)
    third.boot_all(["sam3"])
    fourth.boot_all(["sam3"])
    assert len(fake_client_cls) == 1
    assert shared_client_counts() == {"sam3": 2}


def test_lingering_servers_are_closed_at_exit(fake_client_cls, tmp_path, monkeypatch):
    """A parked server belongs to nobody, so the interpreter's exit closes it."""
    registered = []
    monkeypatch.setattr(tbm.atexit, "register", registered.append)
    monkeypatch.setattr(tbm, "_ATEXIT_REGISTERED", False)
    monkeypatch.setenv(LINGER_ENV, "1")
    first = _episode(tmp_path)
    first.boot_all(["sam3"])
    first.shutdown_all()
    assert registered == [close_shared_clients]
    registered[0]()
    assert fake_client_cls[0].closed
    assert shared_client_counts() == {}


def test_without_linger_nothing_is_registered_at_exit(fake_client_cls, tmp_path, monkeypatch):
    registered = []
    monkeypatch.setattr(tbm.atexit, "register", registered.append)
    monkeypatch.setattr(tbm, "_ATEXIT_REGISTERED", False)
    first = _episode(tmp_path)
    first.boot_all(["sam3"])
    first.shutdown_all()
    assert registered == []


def test_a_forked_child_does_not_inherit_the_parents_servers(fake_client_cls, tmp_path, monkeypatch):
    """Two processes on one pipe would interleave two conversations on a wire
    that is strictly one request, one reply. A child starts with no servers."""
    monkeypatch.setenv(LINGER_ENV, "1")
    first = _episode(tmp_path)
    first.boot_all(["sam3"])
    first.shutdown_all()
    assert shared_client_counts() == {"sam3": 0}
    lock = tbm._SHARED_LOCK
    try:
        tbm._forget_inherited()  # what os.register_at_fork runs in the child
        assert shared_client_counts() == {}
        assert not fake_client_cls[0].closed, "a child closed a server that is its parent's"
    finally:
        tbm._SHARED_LOCK = lock


_REAL_BUNDLE = """
import os
from gap_core.tools import tool


@tool(name="fake.pid", summary="The server's process id")
def pid() -> dict:
    return {"pid": os.getpid()}
"""


def test_linger_with_real_subprocesses(tmp_path, monkeypatch):
    """End to end over real stdio servers: the second episode talks to the
    first episode's process, and after that process is killed the third
    episode gets a new one without being told."""
    import os
    import signal
    import sys

    from gap_core.rpc import client as client_mod

    bundle = tmp_path / "fake"
    bundle.mkdir()
    (bundle / "tools.py").write_text(_REAL_BUNDLE)
    real = client_mod.ToolClient

    def _direct(*args, **kwargs):
        # The bundle's own venv is this one; skip `uv run --project`.
        return real(*args, command=[sys.executable, "-m", "gap_core.rpc.server", "--bundle", "fake"],
                    use_uv=False, **kwargs)

    monkeypatch.setattr(client_mod, "ToolClient", _direct)
    monkeypatch.setenv(LINGER_ENV, "1")
    registry = _FakeSkillRegistry({"fake": _FakeSkillInfo(bundle_dir=bundle)})

    def episode():
        mgr = ToolBundleManager(skill_registry=registry, tool_registry=ToolRegistry(), evict_grace_s=1.0)
        mgr.boot_all(["fake"])
        try:
            return mgr._tool_registry.invoke("fake.pid", None)["pid"], mgr._managed["fake"].client
        finally:
            mgr.shutdown_all()

    try:
        first_pid, first = episode()
        assert first.alive()
        second_pid, second = episode()
        assert second is first and second_pid == first_pid

        os.killpg(os.getpgid(first._proc.pid), signal.SIGKILL)
        first._proc.wait(timeout=10)
        assert not first.alive()

        third_pid, third = episode()
        assert third is not first
        assert third_pid != first_pid
        assert third.alive()
    finally:
        assert close_shared_clients() >= 0
    assert not third.alive()


# ---------------------------------------------------------------------------
# Discovery (required_rpc_tool_bundles + boot_tool_bundles)
# ---------------------------------------------------------------------------


def _write_workflow(tmp_path: Path, tools: list[str]) -> Path:
    wf = {
        "version": 3,
        "meta": {"name": "t"},
        "nodes": {"m": {"type": "subgraph", "ref": "m"},
                  "done": {"type": "end", "status": "success"}},
        "edges": [["START", "m"]],
        "conditional_edges": {"m": {"router_field": "exit",
                                     "mapping": {"ok": "done"}}},
        "subgraphs": {
            "m": {
                "skill": "g", "inputs": {}, "outputs": {},
                "nodes": {
                    **{f"n{i}": {"type": "tool", "tool": t, "inputs": {}}
                       for i, t in enumerate(tools)},
                    "ok": {"type": "noop"},
                },
                "edges": [["START", "n0"]]
                + [[f"n{i}", f"n{i+1}"] for i in range(len(tools) - 1)]
                + [[f"n{len(tools)-1}", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    }
    wfd = tmp_path / "wf"
    wfd.mkdir()
    (wfd / "workflow.json").write_text(json.dumps(wf))
    return wfd


def test_required_rpc_tool_bundles_filters_by_protocol(tmp_path):
    wf_dir = _write_workflow(tmp_path, ["sam3.segment", "geometry.iou",
                                         "vlm.query", "sim.check_success"])
    registry = _FakeSkillRegistry({
        "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3",
                                protocol="stdio-msgpack"),
        "geometry": _FakeSkillInfo(bundle_dir=tmp_path / "geom",
                                    protocol="in-process"),
        "vlm": _FakeSkillInfo(bundle_dir=tmp_path / "vlm",
                               protocol=None),
        # `sim.*` is a connector tool — never in the registry.
    })
    assert required_rpc_tool_bundles(wf_dir, registry) == {"sam3"}


def _write_scripted_workflow(tmp_path: Path, script_body: str, *, skill: str = "generated") -> Path:
    """A graph that names no real skill and does its work in its own Python.

    The shape every self-authored policy has: one subgraph labelled
    ``generated``, one script node, and every tool call inside the script.
    """
    wf = {
        "version": 3,
        "meta": {"name": "t"},
        "nodes": {"m": {"type": "subgraph", "ref": "m"},
                  "done": {"type": "end", "status": "success"}},
        "edges": [["START", "m"]],
        "conditional_edges": {"m": {"router_field": "exit",
                                    "mapping": {"ok": "done"}}},
        "subgraphs": {
            "m": {
                "skill": skill, "inputs": {}, "outputs": {},
                "nodes": {"look": {"type": "script", "script": "scripts/look.py", "inputs": {}},
                          "ok": {"type": "noop"}},
                "edges": [["START", "look"], ["look", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    }
    wfd = tmp_path / "wf"
    (wfd / "scripts").mkdir(parents=True)
    (wfd / "workflow.json").write_text(json.dumps(wf))
    (wfd / "scripts" / "look.py").write_text(script_body)
    return wfd


def _sam3_only(tmp_path: Path):
    return _FakeSkillRegistry({
        "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3", protocol="stdio-msgpack"),
    })


def test_a_graph_that_only_calls_a_bundle_from_its_scripts_boots_it(tmp_path):
    """The whole point. A subgraph labelled ``generated`` names no skill, so
    such a graph showed nothing, booted nothing, and died at its first call
    with ``Tool 'sam3.segment_text' not found`` -- three iterations of a
    self-learning cell went into writing fallbacks around that."""
    wf_dir = _write_scripted_workflow(
        tmp_path,
        "def run(ctx, **kw):\n"
        "    return ctx.tool('sam3.segment_text', image=kw['rgb'], query='the white sponge')\n",
    )
    assert required_rpc_tool_bundles(wf_dir, _sam3_only(tmp_path)) == {"sam3"}


def test_a_helper_beside_the_script_counts_too(tmp_path):
    """A bundle booted for the node but not for the helper it imports is the
    same failure one import deeper, so every ``*.py`` in the graph is read."""
    wf_dir = _write_scripted_workflow(
        tmp_path, "from .seg import find\n\ndef run(ctx, **kw):\n    return find(ctx)\n"
    )
    (wf_dir / "scripts" / "seg.py").write_text(
        "def find(ctx):\n    return ctx.tool_call('sam3.segment_text', query='sponge')\n"
    )
    assert required_rpc_tool_bundles(wf_dir, _sam3_only(tmp_path)) == {"sam3"}


def test_a_tool_name_that_is_not_a_call_boots_nothing(tmp_path):
    """Reading a whole directory is only safe because a string literal in first
    position of a call is the only thing that counts: prose about a tool, and a
    name assembled at run time, start no subprocess."""
    wf_dir = _write_scripted_workflow(
        tmp_path,
        "NAME = 'sam3.' + 'segment_text'\n\n"
        "def run(ctx, **kw):\n"
        "    # would call sam3.segment_text if it could\n"
        "    return ctx.tool(NAME, query='sponge')\n",
    )
    assert script_tool_names(wf_dir) == set()
    assert required_rpc_tool_bundles(wf_dir, _sam3_only(tmp_path)) == set()


def test_a_bare_tool_name_is_not_a_bundle(tmp_path):
    """``ctx.tool("noop")`` names no bundle; only a dotted name can."""
    wf_dir = _write_scripted_workflow(tmp_path, "def run(ctx, **kw):\n    return ctx.tool('noop')\n")
    assert script_tool_names(wf_dir) == set()


def test_a_script_that_does_not_parse_is_skipped(tmp_path):
    """This runs before the graph does and must never be what stops it."""
    wf_dir = _write_scripted_workflow(tmp_path, "def run(ctx:\n")
    (wf_dir / "scripts" / "ok.py").write_text(
        "def f(ctx):\n    return ctx.tool('sam3.segment_text')\n"
    )
    assert script_tool_names(wf_dir) == {"sam3.segment_text"}


def test_boot_tool_bundles_returns_none_when_no_rpc_needed(tmp_path):
    wf_dir = _write_workflow(tmp_path, ["geometry.iou"])
    registry = _FakeSkillRegistry({
        "geometry": _FakeSkillInfo(bundle_dir=tmp_path / "geom",
                                    protocol=None),
    })
    result = boot_tool_bundles(wf_dir, registry, ToolRegistry())
    assert result is None


def test_boot_tool_bundles_returns_manager_when_rpc_needed(fake_client_cls, tmp_path):
    wf_dir = _write_workflow(tmp_path, ["sam3.segment"])
    registry = _FakeSkillRegistry({
        "sam3": _FakeSkillInfo(bundle_dir=tmp_path / "sam3",
                                protocol="stdio-msgpack"),
    })
    tool_registry = ToolRegistry()
    mgr = boot_tool_bundles(wf_dir, registry, tool_registry)
    try:
        assert mgr is not None
        assert mgr.loaded_bundles() == ["sam3"]
    finally:
        mgr.shutdown_all()


# ---------------------------------------------------------------------------
# RpcAdapter direct path
# ---------------------------------------------------------------------------


def test_rpc_adapter_invoke_drops_ctx():
    """The RPC path is for ctx-free tools only — ctx must not leak into
    the wire even when the gap-runtime caller passes one."""
    adapter = RpcAdapter()
    client = _FakeToolClient("b", "/tmp/b")
    adapter.register("b.tool", client)
    adapter.invoke("b.tool", ctx=object(), x=1, y=2)
    assert client.calls == [("b.tool", {"x": 1, "y": 2})]


def test_rpc_adapter_unknown_tool_raises():
    adapter = RpcAdapter()
    with pytest.raises(KeyError, match="not registered"):
        adapter.invoke("nope", None)
