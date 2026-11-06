"""GAP_TOOL_SHARE_DIR: several processes' ToolClients reach one bundle server through a broker."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from gap_core.rpc.broker import SERVER_COMMAND_ENV
from gap_core.rpc.client import ToolClient

from gap.runtime import tool_bundle_manager as manager

FAKE_SERVER = textwrap.dedent(
    """
    import os, sys
    from gap_core.rpc.codec import decode_frame, encode_frame
    out, inp = sys.stdout.buffer, sys.stdin.buffer
    out.write(encode_frame({"id": "hs-0", "kind": "catalog", "tools": [{"name": "fake.echo", "summary": "", "tags": []}]}))
    out.flush()
    while True:
        frame = decode_frame(inp)
        if frame is None:
            break
        out.write(encode_frame({"id": frame["id"], "kind": "result", "result": {"pid": os.getpid(), "args": frame["args"]}}))
        out.flush()
    """
)


@pytest.fixture
def share(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="gap-share-", dir="/tmp") as tmp:
        server = Path(tmp) / "server.py"
        server.write_text(FAKE_SERVER)
        owner = os.getpid()
        monkeypatch.setenv(manager.SHARE_DIR_ENV, tmp)
        monkeypatch.setenv(manager.SHARE_BUNDLES_ENV, "fake")
        monkeypatch.setenv(manager.SHARE_OWNER_ENV, str(owner))
        monkeypatch.setenv(SERVER_COMMAND_ENV, json.dumps([sys.executable, str(server)]))
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(sys.path))
        yield tmp


def _client(tmp: str) -> ToolClient:
    kwargs = manager._shared_server_kwargs("fake", tmp, {})
    assert kwargs["command"][2] == "gap_core.rpc.relay"
    return ToolClient("fake", tmp, **kwargs)


def test_unshared_bundles_and_unset_dir_keep_their_own_server(monkeypatch, share):
    assert manager._shared_server_kwargs("vlm", share, {}) == {}
    monkeypatch.delenv(manager.SHARE_DIR_ENV)
    assert manager._shared_server_kwargs("fake", share, {}) == {}


def test_clients_share_one_server_and_get_their_own_answers(share):
    clients = [_client(share) for _ in range(3)]
    try:
        assert [e.name for e in clients[0].catalog] == ["fake.echo"]

        def ask(i: int) -> tuple[int, int]:
            reply = clients[i % 3].call("fake.echo", n=i)
            return reply["args"]["n"], reply["pid"]

        with ThreadPoolExecutor(6) as pool:
            answers = list(pool.map(ask, range(60)))
        assert [n for n, _ in answers] == list(range(60)), "a reply reached the wrong caller"
        assert len({pid for _, pid in answers}) == 1, "the clients did not share one server"
        assert len(list(Path(share).glob("fake-*.sock"))) == 1
    finally:
        for client in clients:
            client.close()


def test_a_new_client_after_the_others_left_reuses_the_broker(share):
    first = _client(share)
    pid = first.call("fake.echo")["pid"]
    first.close()
    second = _client(share)
    try:
        assert second.call("fake.echo")["pid"] == pid
    finally:
        second.close()


def test_the_broker_exits_with_its_owner(share, monkeypatch):
    import subprocess

    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setenv(manager.SHARE_OWNER_ENV, str(owner.pid))
    client = _client(share)
    client.call("fake.echo")
    sockets = list(Path(share).glob("fake-*.sock"))
    assert sockets
    owner.kill()
    owner.wait()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and sockets[0].exists():
        time.sleep(0.2)
    client.close()
    assert not sockets[0].exists(), "the broker outlived its owner"
