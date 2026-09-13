"""RayToolExecutor + registry adapter, exercised with a stub ray module.

No real ray (and no cluster) anywhere in this suite — the stub mimics
``ray.remote(cls)`` / ``handle.method.remote(...)`` / ``ray.get(ref)``.
"""

from __future__ import annotations

import pytest

from gap_core.errors import ToolArgumentError
from gap_core.tools import ToolRegistry
from gap_core.tools.ray_executor import (
    RayToolExecutor,
    bundle_for_tool,
    substitute_ray_tools,
)

# --------------------------------------------------------------------------
# Stub ray
# --------------------------------------------------------------------------


class _Ref:
    def __init__(self, value):
        self.value = value


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return _Ref(self._fn(*args, **kwargs))


class _ActorHandle:
    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name):
        return _RemoteMethod(getattr(self._instance, name))


class _RemoteClass:
    def __init__(self, cls, options=None):
        self.cls = cls
        self.options_used = options or {}

    def options(self, **kw):
        return _RemoteClass(self.cls, options=kw)

    def remote(self, *args, **kwargs):
        return _ActorHandle(self.cls(*args, **kwargs))


class StubRay:
    def __init__(self):
        self.remote_calls = 0
        self.last_options = None

    def remote(self, cls):
        self.remote_calls += 1
        return _RemoteClass(cls)

    def get(self, ref):
        assert isinstance(ref, _Ref)
        return ref.value


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------


def test_actor_created_lazily_one_per_bundle() -> None:
    ray = StubRay()
    ex = RayToolExecutor(ray_module=ray)
    ex.register_bundle("sam3", {"sam3.segment": lambda image: f"seg:{image}"})
    ex.register_bundle("dino", {"dino.detect": lambda prompt: f"det:{prompt}"})
    assert ex.live_bundles == []
    assert ray.remote_calls == 0

    assert ex.invoke("sam3", "sam3.segment", image="img") == "seg:img"
    assert ex.live_bundles == ["sam3"]
    assert ray.remote_calls == 1

    # Second invoke on the same bundle reuses the actor.
    assert ex.invoke("sam3", "sam3.segment", image="x") == "seg:x"
    assert ray.remote_calls == 1

    assert ex.invoke("dino", "dino.detect", prompt="cup") == "det:cup"
    assert ray.remote_calls == 2
    assert ex.live_bundles == ["dino", "sam3"]


def test_unknown_bundle_and_tool_raise() -> None:
    ex = RayToolExecutor(ray_module=StubRay())
    with pytest.raises(KeyError, match="no callables registered"):
        ex.invoke("nope", "x")
    ex.register_bundle("b", {"b.t": lambda: 1})
    with pytest.raises(KeyError, match="not hosted"):
        ex.invoke("b", "b.other")


def test_register_after_actor_spawn_rejected() -> None:
    ex = RayToolExecutor(ray_module=StubRay())
    ex.register_bundle("b", {"b.t": lambda: 1})
    ex.invoke("b", "b.t")
    with pytest.raises(ValueError, match="already created"):
        ex.register_bundle("b", {"b.u": lambda: 2})


def test_guarded_import_without_ray(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _no_ray(name, *args, **kwargs):
        if name == "ray":
            raise ImportError("No module named 'ray'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_ray)
    with pytest.raises(ImportError, match=r"\[ray\] extra"):
        RayToolExecutor()


# --------------------------------------------------------------------------
# Registry adapter
# --------------------------------------------------------------------------


def _bundle_fn(module: str, fn):
    """Tag *fn* as if it were imported from a bundle tools.py module."""
    fn.__module__ = module
    return fn


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def segment(image: str, prompt: str = "object") -> dict:
        return {"mask": f"{image}:{prompt}"}

    def detect(prompt: str) -> dict:
        return {"boxes": prompt}

    def local_helper(x: int) -> int:
        return x + 1

    def needs_ctx(ctx, x: int) -> int:
        return x

    reg.register_callable(
        "sam3.segment", _bundle_fn("gap_skills.tools.sam3.tools", segment),
        summary="segment", tags=("perception",),
    )
    reg.register_callable(
        "dino.detect", _bundle_fn("gap_skills.tools.grounding_dino.tools", detect),
        summary="detect", tags=("perception",),
    )
    reg.register_callable("geometry.helper", local_helper, summary="local")
    reg.register_callable(
        "sam3.ctx_tool", _bundle_fn("gap_skills.tools.sam3.tools", needs_ctx),
        summary="ctx tool",
    )
    return reg


def test_bundle_for_tool_resolution() -> None:
    reg = _make_registry()
    assert bundle_for_tool(reg.get("sam3.segment")) == "sam3"
    assert bundle_for_tool(reg.get("dino.detect")) == "grounding_dino"
    assert bundle_for_tool(reg.get("geometry.helper")) is None


def test_substitute_routes_bundle_tools_through_actors() -> None:
    ray = StubRay()
    reg = _make_registry()
    ex = substitute_ray_tools(reg, ray_module=ray)

    # Dispatch through the normal registry surface.
    out = reg.invoke("sam3.segment", image="img", prompt="cup")
    assert out == {"mask": "img:cup"}
    assert ex.live_bundles == ["sam3"]

    out = reg.invoke("dino.detect", prompt="bottle")
    assert out == {"boxes": "bottle"}
    assert sorted(ex.live_bundles) == ["grounding_dino", "sam3"]

    # Non-bundle tool untouched (no extra actors).
    assert reg.invoke("geometry.helper", x=1) == 2
    assert ray.remote_calls == 2

    # ctx-requiring bundle tool left in-process.
    assert reg.invoke("sam3.ctx_tool", ctx=None, x=5) == 5


def test_substitute_respects_bundle_filter() -> None:
    ray = StubRay()
    reg = _make_registry()
    substitute_ray_tools(reg, bundles=["sam3"], ray_module=ray)
    reg.invoke("sam3.segment", image="i")
    reg.invoke("dino.detect", prompt="p")  # stays in-process
    assert ray.remote_calls == 1


def test_substitute_checks_against_the_original_signature() -> None:
    """The proxy is judged by the ORIGINAL signature, not the proxy's own.

    Swapping in a ray proxy must not change which arguments a tool accepts:
    a declared one still dispatches, and an undeclared one is still refused
    rather than dropped on the way to the actor.
    """
    reg = _make_registry()
    substitute_ray_tools(reg, ray_module=StubRay())
    assert reg.invoke("sam3.segment", image="img") == {"mask": "img:object"}
    with pytest.raises(ToolArgumentError, match="bogus_extra"):
        reg.invoke("sam3.segment", image="img", bogus_extra=123)


def test_shared_executor_across_registries() -> None:
    """Two worker registries share one executor -> one actor per bundle."""
    ray = StubRay()
    reg_a = _make_registry()
    ex = substitute_ray_tools(reg_a, bundles=["sam3"], ray_module=ray)
    reg_a.invoke("sam3.segment", image="a")
    assert ray.remote_calls == 1

    # Same executor, second registry: dispatch swapped, NO second actor.
    reg_b = _make_registry()
    ex2 = substitute_ray_tools(reg_b, bundles=["sam3"], executor=ex)
    assert ex2 is ex
    assert reg_b.invoke("sam3.segment", image="b") == {"mask": "b:object"}
    assert ray.remote_calls == 1
    assert ex.live_bundles == ["sam3"]
