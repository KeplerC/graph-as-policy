"""Lifecycle manager for out-of-process tool bundle servers.

Parallel to :mod:`gap.runtime.policy_manager` — the policy manager owns
long-running learned-policy websocket servers; this one owns long-running
``stdio-msgpack`` tool servers (one subprocess per bundle that declares
``gap.serving.protocol: stdio-msgpack`` in its SKILL.md). The launcher
calls :meth:`boot_all` after policies are up; on workflow exit the caller
MUST invoke :meth:`shutdown_all` from a ``finally``.

A booted bundle's tools register with the gap-runtime ToolRegistry via
:meth:`gap_core.tools.ToolRegistry.register_rpc` so the executor sees the
RPC-routed tools side-by-side with the in-process @tool catalog. The LLM
catalog and the workflow validator are unaware of the transport — they
just see another ``runtime`` tool.

**One server per bundle per process, however many managers ask.** A batched
simulator runs W worlds in one process, each world its own thread, its own
graph, its own connector and its own manager — so twelve worlds booted twelve
of every bundle. Measured on one wave: 12 sam3 + 12 geometry + 12
grounding-dino, 6.3 GiB of GPU and 7.8 GiB of host RAM each for the segmenter
alone, ~20 s of model build apiece, and the whole set loaded while the
simulation sat frozen at its barrier — 138 s of a 205 s wave. It ended the way
that arithmetic has to end: ``Warp CUDA error 2: out of memory`` in the middle
of the first twelve-world iteration of a study run.

So a client is cached per process and handed to every manager that asks for
that bundle, with a reference count deciding when the subprocess dies. It is
safe by construction rather than by hope: ``ToolClient.call`` holds the
client's own lock across the whole request-and-reply, and the wire is strictly
sequential (one request, one reply, ids checked), so concurrent callers queue
instead of interleaving. What they buy for that queueing is measured too --
twelve worlds' perception serialised through one server cost ~1.2 s more per
episode than twelve parallel servers, against ~70 GiB of GPU recovered.

**Lingering across episodes: ``GAP_TOOL_BUNDLE_LINGER=1``.** The reference
count above dies at the end of every episode, so a benchmark that runs thirty
episodes in one process loaded the segmenter thirty times: ~20 s of SAM3 build
(6.3 GiB of GPU) plus ~9 s of GroundingDINO, at the top of every episode,
measured on one tool_hanging rollout. With the variable set, a server whose
count reaches zero is *parked* instead of closed, and the next acquire of the
same key -- the next episode's manager -- takes it back. Nothing about what a
graph sees changes: each episode still registers the catalog into its own
fresh ``ToolRegistry``. A parked server that has died in the meantime (a
reply-timeout kill, a crash, an OOM) is noticed at reuse and respawned in its
place, so a dead process is never handed out. Everything left at interpreter
exit is closed by an ``atexit`` hook. The price is the reason it is opt-in:
the parked servers keep their GPU memory between episodes, and between runs
if the process lives on. Unset, the behaviour is exactly the reference count's.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_SHARED_LOCK = threading.RLock()
"""Guards :data:`_SHARED`. Re-entrant because a boot failure releases under it."""

_SHARED: dict[tuple, "_SharedClient"] = {}
"""Live bundle clients, keyed by what makes two requests the same server.

Process-global on purpose: the whole point is that managers which know
nothing about each other -- one per world of a wave -- end up on one
subprocess. Nothing here is keyed by manager, and a manager that never shares
(the single-world case) sees a count of one and the old behaviour exactly.
"""


LINGER_ENV = "GAP_TOOL_BUNDLE_LINGER"
"""Set to ``1`` to keep a shared server alive after its last manager lets go."""


def linger_enabled() -> bool:
    """Whether :data:`LINGER_ENV` asks for servers to outlive their episode.

    Read at every release and acquire rather than once at import, so a caller
    (or a test) that sets it after ``gap`` was imported is still obeyed.
    """
    return os.environ.get(LINGER_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _SharedClient:
    client: Any
    refs: int


def _client_alive(client: Any) -> bool:
    """Whether *client*'s subprocess can still take a call.

    Asked of ``ToolClient.alive`` where it exists; an older gap-core without it
    is asked through its process handle, and anything with neither (a test's
    stand-in) is taken at its word.
    """
    probe = getattr(client, "alive", None)
    if callable(probe):
        try:
            return bool(probe())
        except Exception:
            return False
    proc = getattr(client, "_proc", None)
    if proc is not None and hasattr(proc, "poll"):
        try:
            return proc.poll() is None
        except Exception:
            return False
    return True


def _client_pid(client: Any) -> Any:
    """The subprocess id, for the boot/reuse log line; ``None`` when unknown."""
    return getattr(getattr(client, "_proc", None), "pid", None)


def _close_quietly(client: Any) -> None:
    try:
        client.close()
    except BaseException:
        logger.warning("closing a shared tool bundle client failed", exc_info=True)


_ATEXIT_REGISTERED = False


def _ensure_atexit() -> None:
    """Close whatever is still cached when the interpreter exits. Registered
    the first time a server is parked, so a process that never lingers runs
    exactly the code it always did."""
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(close_shared_clients)
        _ATEXIT_REGISTERED = True


def close_shared_clients() -> int:
    """Close every cached server, parked or not, and forget them. Returns how many.

    The ``atexit`` hook :data:`LINGER_ENV` installs, and the explicit way to
    give the GPU back between runs in a long-lived process. Any manager still
    holding one of these afterwards holds a closed client; it is for the end of
    the work, not the middle of it.
    """
    with _SHARED_LOCK:
        entries = list(_SHARED.values())
        _SHARED.clear()
    for entry in entries:
        _close_quietly(entry.client)
    return len(entries)


def _forget_inherited() -> None:
    """After ``fork``: the parent's servers are the parent's.

    A child that took one would write into a pipe the parent is also writing
    into, and the strictly sequential wire would interleave two conversations.
    Parked servers make an inherited cache the normal case rather than a race,
    so the child starts with an empty one -- and a fresh lock, since the
    parent's may have been held by another thread at the instant of the fork.
    Nothing is closed: the processes are not the child's to close.
    """
    global _SHARED_LOCK
    _SHARED_LOCK = threading.RLock()
    _SHARED.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_inherited)


SHARE_DIR_ENV = "GAP_TOOL_SHARE_DIR"
"""A directory for shared tool servers' sockets. Unset: every process runs its own servers."""
SHARE_BUNDLES_ENV = "GAP_TOOL_SHARE_BUNDLES"
"""Comma-separated bundles to share when :data:`SHARE_DIR_ENV` is set (default: sam3,grounding-dino)."""
SHARE_OWNER_ENV = "GAP_TOOL_SHARE_OWNER"
"""The pid whose exit stops the shared servers (a benchmark run's driver); 0 or unset: idleness only."""
DEFAULT_SHARED_BUNDLES = ("sam3", "grounding-dino")
"""The GPU-heavy, stateless-per-call bundles: about 6 GB and 3 GB a copy. Not ``vlm``, whose
calls are seconds-long network waits that a shared server would serialise."""


def _shared_server_kwargs(name: str, bundle_dir: Any, env: Any) -> dict:
    """``ToolClient`` arguments that reach a shared server through a relay, or ``{}``.

    One broker per bundle, bundle environment and GPU: the socket name hashes
    all three, so processes on different GPUs, or with different bundle
    settings, never share a server.
    """
    share_dir = os.environ.get(SHARE_DIR_ENV, "").strip()
    if not share_dir:
        return {}
    wanted = os.environ.get(SHARE_BUNDLES_ENV, "").strip()
    bundles = tuple(b.strip() for b in wanted.split(",") if b.strip()) if wanted else DEFAULT_SHARED_BUNDLES
    if name not in bundles:
        return {}
    import hashlib
    import sys

    ident = repr((name, str(bundle_dir), sorted((env or {}).items()), os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    digest = hashlib.sha1(ident.encode()).hexdigest()[:10]
    socket_path = os.path.join(share_dir, f"{name}-{digest}.sock")
    command = [
        sys.executable, "-m", "gap_core.rpc.relay",
        "--socket", socket_path, "--bundle", name, "--bundle-dir", str(bundle_dir),
        "--owner-pid", os.environ.get(SHARE_OWNER_ENV, "0").strip() or "0",
    ]
    return {"command": command, "use_uv": False}


def _client_key(name: str, bundle_dir: Any, env: Any) -> tuple:
    """What has to match for two managers to get the same server.

    The environment is part of it, not a detail: two graphs asking for the
    same bundle with different ``serving.env`` want different servers, and
    handing the second one the first's process would be a silent, wrong
    answer -- the failure mode this whole module exists to avoid.
    """
    if isinstance(env, dict):
        stamp = tuple(sorted((str(k), str(v)) for k, v in env.items()))
    else:
        stamp = (repr(env),)
    return (str(name), str(bundle_dir), stamp)


def _acquire_client(key: tuple, make: Any) -> tuple[_SharedClient, str]:
    """The cached entry for ``key``, or one built by ``make``. Returns
    ``(entry, how)`` where *how* is ``"booted"``, ``"shared"`` (another manager
    holds it right now) or ``"lingering"`` (parked by :data:`LINGER_ENV`).

    ``make`` runs while the cache lock is held, which is deliberate: eleven
    threads waiting ~20 s for one model to load is the same wall clock as
    eleven threads each loading their own, and it is the only ordering where
    the second caller cannot start a second subprocess.

    With lingering on, a cached server whose process has died is replaced
    rather than handed out. The dead entry leaves the cache; whoever still
    holds it releases it by identity (see :func:`_release_client`), so its
    count cannot be confused with the replacement's.
    """
    dead: _SharedClient | None = None
    try:
        with _SHARED_LOCK:
            shared = _SHARED.get(key)
            if shared is not None and linger_enabled() and not _client_alive(shared.client):
                proc = getattr(shared.client, "_proc", None)
                logger.warning(
                    "[tool-bundle:%s] cached server (pid %s) is dead (returncode=%s); respawning it",
                    key[0], _client_pid(shared.client), getattr(proc, "returncode", None),
                )
                _SHARED.pop(key, None)
                if shared.refs <= 0:
                    dead = shared
                shared = None
            if shared is not None:
                how = "shared" if shared.refs > 0 else "lingering"
                shared.refs += 1
                return shared, how
            entry = _SharedClient(client=make(), refs=1)
            _SHARED[key] = entry
    finally:
        if dead is not None:
            # Reaps the corpse and closes its pipes, whether or not its
            # replacement came up; nothing is listening.
            _close_quietly(dead.client)
    return entry, "booted"


def _release_client(key: tuple, entry: _SharedClient | None = None) -> None:
    """Give back one reference; close the subprocess when the last one goes.

    *entry* is the one this caller acquired. Without it the cache's current
    entry for *key* is assumed, which is the same thing unless a dead server
    was replaced in between -- the case the identity is passed for.

    With :data:`LINGER_ENV` set, a live server whose count reaches zero stays
    cached for the next acquire instead of closing.
    """
    with _SHARED_LOCK:
        current = _SHARED.get(key)
        if entry is None:
            entry = current
        if entry is None or entry.refs <= 0:
            return
        entry.refs -= 1
        if entry.refs > 0:
            return
        if current is entry:
            if linger_enabled() and _client_alive(entry.client):
                _ensure_atexit()
                logger.info(
                    "[tool-bundle:%s] last user released; server (pid %s) lingers for reuse (%s=1)",
                    key[0], _client_pid(entry.client), LINGER_ENV,
                )
                return
            _SHARED.pop(key, None)
    _close_quietly(entry.client)


class ToolBundleStartupError(RuntimeError):
    """A bundle's tool server failed to come up (or its catalog handshake
    failed). Surfaces the bundle name + the underlying error."""


@dataclass
class _Managed:
    bundle: str
    client: Any  # ToolClient
    tool_names: list[str]
    key: tuple  # into _SHARED; this manager holds one reference to it
    entry: _SharedClient | None = None  # the reference itself, released by identity


class ToolBundleManager:
    """Boot every RPC-protocol tool bundle the workflow references; shut down on exit.

    Typical use (from the launcher)::

        manager = ToolBundleManager(skill_registry=skill_registry,
                                    tool_registry=tool_registry)
        manager.boot_all(["sam3", "curobo", "geometry"])  # noqa
        try:
            executor.execute()
        finally:
            manager.shutdown_all()
    """

    def __init__(self, skill_registry: Any, tool_registry: Any,
                 *, startup_timeout_s: float = 60.0,
                 evict_grace_s: float = 5.0,
                 call_timeout_s: float | None = None) -> None:
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._timeout = float(startup_timeout_s)
        self._grace = float(evict_grace_s)
        # Per-call reply timeout passed to each ToolClient. None lets the
        # client resolve GAP_TOOL_CALL_TIMEOUT_S / its built-in default.
        self._call_timeout_s = call_timeout_s
        self._lock = threading.Lock()
        self._managed: dict[str, _Managed] = {}

    def boot_all(self, bundle_names: Iterable[str]) -> None:
        """Spawn one server per bundle, register every catalog entry.

        Bundles whose SKILL.md does NOT declare ``serving.protocol ==
        "stdio-msgpack"`` are silently skipped — they run in-process.
        """
        from gap_core.rpc.client import ToolClient

        names = sorted(set(bundle_names))
        if not names:
            return
        started: list[_Managed] = []
        try:
            for name in names:
                info = self._skill_registry.get(name)
                serving = info.meta.serving
                if serving is None or serving.protocol != "stdio-msgpack":
                    continue
                key = _client_key(name, info.meta.bundle_dir, serving.env)
                try:
                    shared, how = _acquire_client(
                        key,
                        lambda name=name, info=info, serving=serving: ToolClient(
                            bundle_name=name,
                            bundle_dir=info.meta.bundle_dir,
                            env=serving.env,
                            evict_grace_s=self._grace,
                            call_timeout_s=self._call_timeout_s,
                            **_shared_server_kwargs(name, info.meta.bundle_dir, serving.env),
                        ),
                    )
                except BaseException as exc:
                    raise ToolBundleStartupError(
                        f"tool bundle {name!r}: failed to start its server "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc
                client = shared.client
                logger.info(
                    "[tool-bundle:%s] %s (pid %s)", name,
                    {
                        "booted": "booting",
                        "shared": "reusing this process's server",
                        "lingering": "reusing this process's lingering server",
                    }[how],
                    _client_pid(client),
                )

                tool_names: list[str] = []
                for entry in client.catalog:
                    if entry.name in self._tool_registry:
                        existing = self._tool_registry.get(entry.name)
                        if existing.transport != "rpc":
                            # An in-process registration (a connector's own
                            # tool) claims this name: in-process wins, and the
                            # bundle's copy is simply not offered. A simulator
                            # connector providing geometry.* directly is the
                            # normal case, not an error.
                            logger.info(
                                "[tool-bundle:%s] %r already provided "
                                "in-process; keeping the existing registration",
                                name, entry.name,
                            )
                            continue
                    try:
                        self._tool_registry.register_rpc(
                            entry.name,
                            client,
                            summary=entry.summary,
                            tags=tuple(entry.tags),
                        )
                        tool_names.append(entry.name)
                    except ValueError as exc:
                        # Name collision between two RPC bundles exporting the
                        # same name. Surface as a startup error — silent
                        # dispatch would be confusing. Release rather than
                        # close: another world may be mid-call on this server.
                        _release_client(key, shared)
                        raise ToolBundleStartupError(
                            f"tool bundle {name!r}: cannot register "
                            f"{entry.name!r} — {exc}"
                        ) from exc
                started.append(_Managed(bundle=name, client=client,
                                         tool_names=tool_names, key=key,
                                         entry=shared))
                logger.info("[tool-bundle:%s] %d tools registered",
                            name, len(tool_names))

            with self._lock:
                for m in started:
                    self._managed[m.bundle] = m
        except BaseException:
            # Give back what we took before re-raising. A reference, not a
            # close: a sibling world may be holding the same server.
            for m in started:
                _release_client(m.key, m.entry)
            raise

    def shutdown_all(self) -> None:
        """Terminate every tool bundle subprocess we started. Idempotent."""
        with self._lock:
            managed = list(self._managed.values())
            self._managed.clear()

        if not managed:
            return
        logger.info("ToolBundleManager: releasing %d tool bundle%s",
                    len(managed), "" if len(managed) == 1 else "s")
        for m in managed:
            _release_client(m.key, m.entry)

    def loaded_bundles(self) -> list[str]:
        return sorted(self._managed.keys())


def shared_client_counts() -> dict[str, int]:
    """``{bundle: managers holding it}`` for every cached shared server.

    For tests and for anyone asking the question this module was changed to
    answer: how many servers is this process actually running? A server parked
    by :data:`LINGER_ENV` is running and counted as ``0``.
    """
    with _SHARED_LOCK:
        return {key[0]: shared.refs for key, shared in _SHARED.items()}


__all__ = [
    "LINGER_ENV",
    "ToolBundleManager",
    "ToolBundleStartupError",
    "close_shared_clients",
    "linger_enabled",
    "shared_client_counts",
]
