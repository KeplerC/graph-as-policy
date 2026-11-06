"""One tool server shared by several processes: the broker behind ``GAP_TOOL_SHARE_DIR``.

A stdio-msgpack bundle server has one client, the process that spawned it. On a
GPU that runs several benchmark workers, each worker then holds its own SAM 3
(about 6 GB) and GroundingDINO (about 3 GB), and those copies, not the
simulation, are what limit how many workers fit on the card. The broker owns
one real server and serves it to any number of clients over a Unix socket.

* The server is spawned exactly as :class:`~gap_core.rpc.client.ToolClient`
  spawns it (``uv run --project <bundle> -- python -m gap_core.rpc.server``),
  inheriting this process's environment, ``CUDA_VISIBLE_DEVICES`` and any MPS
  pipe directory included.
* The socket is bound only after the server's ``catalog`` frame has arrived, so
  a client that connects is never kept waiting on a cold start it cannot see.
  Every connection first receives that frame, exactly as a fresh server would
  send it.
* Calls are forwarded one at a time: a call frame is written to the server and
  its reply returned before the next client's call is sent. Request ids may
  therefore repeat across clients (each counts its own ``req-N``). The tools
  shared this way (image segmentation and detection) take everything they need
  in the call, so interleaving clients' calls changes nothing a call sees.
* A call the server has not answered after ``GAP_TOOL_CALL_TIMEOUT_S`` (the
  client's own watchdog, plus a margin) kills the server, and the broker exits:
  every client sees EOF, which the tool-bundle manager treats as a dead server
  and replaces, and the first replacement starts a new broker.
* The broker exits when the server dies, when ``--owner-pid`` exits, or after
  ``--idle-exit`` seconds without a connection.

Clients reach a broker through :mod:`gap_core.rpc.relay`, which the tool-bundle
manager runs in place of the server.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .client import _resolve_call_timeout, _terminate_group
from .codec import decode_frame, encode_frame

logger = logging.getLogger("gap_core.rpc.broker")

SERVER_COMMAND_ENV = "GAP_TOOL_BROKER_SERVER_COMMAND"
"""A JSON argv to run instead of the bundle's server (tests)."""

TIMEOUT_MARGIN_S = 30.0
"""Added to the client's own call timeout before the broker gives up on the server."""


class Broker:
    def __init__(
        self, socket_path: str, bundle: str, bundle_dir: str, *, idle_exit_s: float, owner_pid: int
    ) -> None:
        self.socket_path = socket_path
        self.bundle = bundle
        self.bundle_dir = Path(bundle_dir)
        self.idle_exit_s = idle_exit_s
        self.owner_pid = owner_pid
        self.call_lock = threading.Lock()
        self.connections = 0
        self._count_lock = threading.Lock()
        self.last_activity = time.monotonic()
        self.stopping = threading.Event()
        timeout = _resolve_call_timeout(None)
        self.call_timeout_s = None if timeout is None else timeout + TIMEOUT_MARGIN_S
        self.proc: subprocess.Popen | None = None
        self.catalog: bytes = b""

    # ---------------------------------------------------------------- server
    def start_server(self) -> None:
        override = os.environ.get(SERVER_COMMAND_ENV, "").strip()
        argv = (
            json.loads(override)
            if override
            else [
                "uv",
                "run",
                "--project",
                str(self.bundle_dir),
                "--",
                "python",
                "-m",
                "gap_core.rpc.server",
                "--bundle",
                self.bundle,
            ]
        )
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(self.bundle_dir),
            env=dict(os.environ),
            start_new_session=True,
        )
        frame = decode_frame(self.proc.stdout)
        if frame is None or frame.get("kind") != "catalog":
            raise RuntimeError(
                f"[tool-broker:{self.bundle}] server sent {frame!r} instead of its catalog"
            )
        self.catalog = encode_frame(frame)

    def forward(self, frame: dict[str, Any]) -> bytes:
        """Send one call to the server and return its reply frame, encoded."""
        assert (
            self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        )
        timer = None
        if self.call_timeout_s:
            timer = threading.Timer(
                self.call_timeout_s, self._server_hung, args=(frame.get("tool"),)
            )
            timer.daemon = True
            timer.start()
        try:
            self.proc.stdin.write(encode_frame(frame))
            self.proc.stdin.flush()
            reply = decode_frame(self.proc.stdout)
        finally:
            if timer is not None:
                timer.cancel()
        if reply is None:
            raise EOFError("the server closed its output")
        return encode_frame(reply)

    def _server_hung(self, tool: Any) -> None:
        logger.warning(
            "[tool-broker:%s] %r unanswered after %.0fs; stopping the server",
            self.bundle,
            tool,
            self.call_timeout_s,
        )
        self.stop()

    # ---------------------------------------------------------------- clients
    def serve_connection(self, conn: socket.socket) -> None:
        stream = conn.makefile("rb")
        try:
            conn.sendall(self.catalog)
            while not self.stopping.is_set():
                frame = decode_frame(stream)
                if frame is None:
                    return
                with self.call_lock:
                    if self.stopping.is_set():
                        return
                    try:
                        reply = self.forward(frame)
                    except Exception as exc:  # the server is gone: so is the broker
                        logger.warning(
                            "[tool-broker:%s] server lost (%s); exiting", self.bundle, exc
                        )
                        self.stop()
                        return
                conn.sendall(reply)
                self.last_activity = time.monotonic()
        except (OSError, EOFError):
            return
        finally:
            stream.close()
            conn.close()
            with self._count_lock:
                self.connections -= 1
            self.last_activity = time.monotonic()

    def watch(self) -> None:
        """Stop when the server, or the owner, is gone, or nobody has connected for a while."""
        while not self.stopping.wait(1.0):
            if self.proc is not None and self.proc.poll() is not None:
                logger.warning(
                    "[tool-broker:%s] server exited (%s)", self.bundle, self.proc.returncode
                )
                self.stop()
            elif self.owner_pid and not _alive(self.owner_pid):
                self.stop()
            elif (
                self.idle_exit_s
                and self.connections == 0
                and time.monotonic() - self.last_activity > self.idle_exit_s
            ):
                self.stop()

    def stop(self) -> None:
        if self.stopping.is_set():
            return
        self.stopping.set()
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
        if self.proc is not None and self.proc.poll() is None:
            _terminate_group(self.proc, 5.0)
        # Unblock accept() and every connection's read.
        os.kill(os.getpid(), signal.SIGTERM)

    def run(self) -> int:
        self.start_server()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        tmp = f"{self.socket_path}.{os.getpid()}"
        listener.bind(tmp)
        listener.listen(64)
        with open(f"{self.socket_path}.pid", "w") as pid_file:  # for whoever owns the share dir
            pid_file.write(str(os.getpid()))
        os.replace(tmp, self.socket_path)  # appears atomically, already listening
        threading.Thread(target=self.watch, daemon=True).start()
        logger.info(
            "[tool-broker:%s] serving %s (server pid %s)",
            self.bundle,
            self.socket_path,
            self.proc.pid,
        )
        while not self.stopping.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                break
            with self._count_lock:
                self.connections += 1
            self.last_activity = time.monotonic()
            threading.Thread(target=self.serve_connection, args=(conn,), daemon=True).start()
        return 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--idle-exit", type=float, default=600.0)
    parser.add_argument("--owner-pid", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    broker = Broker(
        args.socket,
        args.bundle,
        args.bundle_dir,
        idle_exit_s=args.idle_exit,
        owner_pid=args.owner_pid,
    )

    def on_term(signum, frame):  # noqa: ARG001
        broker.stopping.set()
        if broker.proc is not None and broker.proc.poll() is None:
            _terminate_group(broker.proc, 5.0)
        os._exit(0)

    signal.signal(signal.SIGTERM, on_term)
    try:
        return broker.run()
    finally:
        if broker.proc is not None and broker.proc.poll() is None:
            _terminate_group(broker.proc, 5.0)


if __name__ == "__main__":
    sys.exit(main())
