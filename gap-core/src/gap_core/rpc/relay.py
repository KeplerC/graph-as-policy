"""The client end of a shared tool server: stdio in, Unix socket out.

:class:`~gap_core.rpc.client.ToolClient` drives a subprocess over its stdin and
stdout. When a bundle is shared (``GAP_TOOL_SHARE_DIR``), that subprocess is
this relay instead of the server. It connects to the bundle's
:mod:`~gap_core.rpc.broker` and copies bytes both ways until either side closes,
so the client sees the same frames, catalog first, that a server of its own
would have sent.

If no broker is listening, the relay starts one, under an exclusive lock on
``<socket>.lock`` so that workers starting together start one broker between
them, and waits for its socket. The broker loads the model before it listens,
so this wait is the server's cold start, as it was before sharing.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import socket
import subprocess
import sys
import threading
import time

START_TIMEOUT_S = 900.0
"""How long a relay waits for a broker it started to begin listening (a cold SAM 3 load included)."""


def _connect(path: str) -> socket.socket | None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError:
        sock.close()
        return None
    return sock


def connect_or_start(path: str, bundle: str, bundle_dir: str, owner_pid: int) -> socket.socket:
    sock = _connect(path)
    if sock is not None:
        return sock
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sock = _connect(path)  # another relay may have started it while we waited
        if sock is not None:
            return sock
        try:
            os.unlink(path)  # a stale socket from a broker that died
        except OSError:
            pass
        with open(f"{path}.log", "ab") as log:  # the broker keeps its own descriptor
            _start_broker(path, bundle, bundle_dir, owner_pid, log)
        deadline = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < deadline:
            sock = _connect(path)
            if sock is not None:
                return sock
            time.sleep(0.2)
    raise SystemExit(
        f"[tool-relay:{bundle}] no broker at {path} after {START_TIMEOUT_S:.0f}s; see {path}.log"
    )


def _start_broker(path: str, bundle: str, bundle_dir: str, owner_pid: int, log) -> None:
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gap_core.rpc.broker",
            "--socket",
            path,
            "--bundle",
            bundle,
            "--bundle-dir",
            bundle_dir,
            "--owner-pid",
            str(owner_pid),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        env=dict(os.environ),
        start_new_session=True,  # outlives this relay; the owner or idleness ends it
    )


def _pump(src, dst_write, on_done) -> None:
    try:
        while True:
            chunk = src(65536)
            if not chunk:
                break
            dst_write(chunk)
    except OSError:
        pass
    finally:
        on_done()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--owner-pid", type=int, default=0)
    args = parser.parse_args(argv)
    sock = connect_or_start(args.socket, args.bundle, args.bundle_dir, args.owner_pid)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    done = threading.Event()

    def write_out(chunk: bytes) -> None:
        stdout.write(chunk)
        stdout.flush()

    threading.Thread(target=_pump, args=(sock.recv, write_out, done.set), daemon=True).start()
    threading.Thread(target=_pump, args=(stdin.read1, sock.sendall, done.set), daemon=True).start()
    done.wait()
    try:
        sock.close()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
