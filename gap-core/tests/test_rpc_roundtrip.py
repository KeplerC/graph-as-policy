"""End-to-end ToolClient ↔ gap_tool_server roundtrip.

A fake bundle directory is written to tmp_path with a ``tools.py`` whose
``@tool`` functions cover the protocol cases: success, numpy roundtrip,
remote exception. The client spawns the server via plain ``python -m
gap_core.rpc.server`` (no ``uv run`` — the test's own venv has gap-core)
and drives a handshake + several calls.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from gap_core.rpc.client import ToolClient, ToolClientError, ToolRemoteError


_BUNDLE_TOOLS = textwrap.dedent('''
    """Fake tool bundle used by tests."""
    import numpy as np
    from gap_core.tools import tool


    @tool(
        name="fake.echo",
        summary="Echo input back as result",
        tags=("test",),
    )
    def echo(message: str, count: int = 1) -> dict:
        return {"message": message, "count": count}


    @tool(
        name="fake.add",
        summary="Add two ints",
        tags=("test",),
    )
    def add(a: int, b: int) -> dict:
        return {"sum": a + b}


    @tool(
        name="fake.numpy_double",
        summary="Double an ndarray",
        tags=("test",),
    )
    def numpy_double(image) -> dict:
        return {"result": np.asarray(image) * 2}


    @tool(
        name="fake.raise_runtime",
        summary="Always raises",
        tags=("test",),
    )
    def raise_runtime(why: str) -> dict:
        raise RuntimeError(f"intentional: {why}")


    @tool(
        name="fake.noisy",
        summary="Pollute stdout (print + raw fd 1) then return",
        tags=("test",),
    )
    def noisy(message: str) -> dict:
        import os
        import sys
        print("stray python print to stdout")
        os.write(1, b"raw bytes written straight to fd one")
        sys.stdout.flush()
        return {"echo": message}


    @tool(
        name="fake.sleep",
        summary="Sleep (exercises the per-call timeout)",
        tags=("test",),
    )
    def sleep_long(seconds: float) -> dict:
        import time
        time.sleep(seconds)
        return {"slept": seconds}
''').lstrip()


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Fake bundle directory containing only tools.py — the server's
    ``import tools`` then drains the @tool decorators."""
    (tmp_path / "tools.py").write_text(_BUNDLE_TOOLS)
    return tmp_path


def _direct_client(bundle_dir: Path) -> ToolClient:
    """Spawn the server in the SAME venv (no `uv run --project`) so the test
    doesn't depend on uv being able to sync a fake bundle. We do this by
    handing ToolClient a custom ``command`` that bypasses the default uv-run
    prefix logic — actually our default *is* uv run, so override the whole
    argv via subprocess directly using the same shape the gap-runtime
    launcher would construct."""
    # We can't easily override the argv on the public client (its constructor
    # builds the uv-run prefix). For tests, use the protocol pieces directly
    # against a manually-spawned subprocess so we don't require uv sync of a
    # fake bundle. Implementation: monkey-patch the cwd + skip uv.
    raise NotImplementedError  # see direct subprocess fixture below


class _DirectToolClient:
    """Test helper: like ToolClient but spawns ``python -m gap_core.rpc.server``
    in the current process's interpreter instead of via ``uv run --project``.

    The real ToolClient uses ``uv run --project <bundle_dir>`` so the server
    runs inside the bundle's own venv. In tests we already have gap_core
    available in the test venv, so we can shortcut. The protocol code paths
    exercised are otherwise identical.
    """

    def __init__(self, bundle_dir: Path):
        from gap_core.rpc.codec import FrameError, decode_frame, write_frame
        self._FrameError = FrameError
        self._decode_frame = decode_frame
        self._write_frame = write_frame

        self.proc = subprocess.Popen(
            [sys.executable, "-m", "gap_core.rpc.server", "--bundle", "fake"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(bundle_dir),
            start_new_session=True,
        )
        # Handshake
        frame = self._read()
        assert frame is not None and frame.get("kind") == "catalog", frame
        self.catalog = frame["tools"]
        self._counter = 0

    def call(self, tool: str, **kwargs):
        self._counter += 1
        rid = f"req-{self._counter}"
        self._write({"id": rid, "kind": "call", "tool": tool, "args": kwargs})
        reply = self._read()
        assert reply is not None
        assert reply["id"] == rid
        if reply["kind"] == "error":
            err = reply["error"]
            raise ToolRemoteError(tool, err["type"], err["message"], err.get("traceback", ""))
        return reply["result"]

    def _write(self, payload):
        self._write_frame(self.proc.stdin, payload)

    def _read(self):
        return self._decode_frame(self.proc.stdout)

    def close(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


@pytest.fixture
def client(bundle_dir):
    c = _DirectToolClient(bundle_dir)
    yield c
    c.close()


def test_catalog_handshake_lists_every_tool(client):
    names = sorted(t["name"] for t in client.catalog)
    assert names == ["fake.add", "fake.echo", "fake.noisy", "fake.numpy_double",
                     "fake.raise_runtime", "fake.sleep"]
    echo = next(t for t in client.catalog if t["name"] == "fake.echo")
    assert echo["summary"] == "Echo input back as result"
    assert "test" in echo["tags"]
    # Schema introspection round-trips.
    assert "message" in echo["schema"]["inputs"]
    assert "count" in echo["schema"]["inputs"]


def test_call_returns_result(client):
    assert client.call("fake.add", a=2, b=40) == {"sum": 42}
    assert client.call("fake.echo", message="hi", count=3) == {"message": "hi", "count": 3}


def test_call_roundtrips_numpy(client):
    image = np.ones((4, 4), dtype=np.uint8)
    out = client.call("fake.numpy_double", image=image)
    assert "result" in out
    np.testing.assert_array_equal(out["result"], image * 2)
    assert out["result"].dtype == np.uint8


def test_remote_exception_surfaces_as_ToolRemoteError(client):
    with pytest.raises(ToolRemoteError) as exc_info:
        client.call("fake.raise_runtime", why="kaboom")
    err = exc_info.value
    assert err.remote_type == "RuntimeError"
    assert "intentional: kaboom" in str(err)
    # The remote traceback travels with the error so the user sees where
    # the failure actually occurred inside the bundle.
    assert "tools.py" in err.remote_tb or "RuntimeError" in err.remote_tb


def test_unknown_tool_surfaces_as_remote_error(client):
    with pytest.raises(ToolRemoteError) as exc_info:
        client.call("fake.does_not_exist", x=1)
    assert "not found" in str(exc_info.value).lower() or "unknown" in str(exc_info.value).lower()


def test_close_is_idempotent(client):
    """Closing twice must not raise — the gap-runtime side calls close()
    in a finally block and may also call it elsewhere on error paths."""
    client.close()
    client.close()  # second close is a no-op


def test_stdout_pollution_does_not_corrupt_frames(client):
    """A tool that writes to stdout (Python print AND raw fd-1 bytes, the
    way a CUDA/Warp C-extension would) must not desync the frame stream.

    Without the server isolating its framing fd, the stray bytes land
    mid-frame and the next read blocks forever (the real-world deadlock).
    With isolation, fd 1 is redirected to stderr, so the call returns
    normally and the channel stays usable for the next call."""
    assert client.call("fake.noisy", message="hi") == {"echo": "hi"}
    # The channel is still in sync: a follow-up call round-trips fine.
    assert client.call("fake.add", a=1, b=2) == {"sum": 3}


def test_call_timeout_terminates_hung_call(bundle_dir):
    """A call that never returns is bounded by the per-call timeout: the
    client kills the subprocess and raises ToolCallTimeout instead of
    hanging forever. Uses the real ToolClient (use_uv=False to spawn the
    server directly in the test venv)."""
    import time

    from gap_core.rpc.client import ToolCallTimeout, ToolClient

    c = ToolClient(
        bundle_name="fake",
        bundle_dir=bundle_dir,
        command=[sys.executable, "-m", "gap_core.rpc.server", "--bundle", "fake"],
        use_uv=False,
        call_timeout_s=2.0,
    )
    try:
        # A short call still works through the real client.
        assert c.call("fake.add", a=5, b=5) == {"sum": 10}

        start = time.monotonic()
        with pytest.raises(ToolCallTimeout):
            c.call("fake.sleep", seconds=30)
        elapsed = time.monotonic() - start
        # Bounded by the timeout (+ teardown grace), nowhere near 30s.
        assert elapsed < 15.0, f"timeout took {elapsed:.1f}s"
    finally:
        c.close()


def test_alive_tracks_the_subprocess(bundle_dir):
    """``alive()`` is what a caller keeping a client across episodes asks
    before reusing it: true while the server runs, false once it is killed
    out from under the client, and false after ``close()``."""
    import signal

    from gap_core.rpc.client import ToolClient

    def spawn() -> ToolClient:
        return ToolClient(
            bundle_name="fake",
            bundle_dir=bundle_dir,
            command=[sys.executable, "-m", "gap_core.rpc.server", "--bundle", "fake"],
            use_uv=False,
            evict_grace_s=1.0,
        )

    killed = spawn()
    try:
        assert killed.alive()
        os.killpg(os.getpgid(killed._proc.pid), signal.SIGKILL)
        killed._proc.wait(timeout=10)
        assert not killed.alive()
    finally:
        killed.close()

    closed = spawn()
    assert closed.alive()
    closed.close()
    assert not closed.alive()
