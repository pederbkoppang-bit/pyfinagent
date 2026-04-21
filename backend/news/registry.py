"""phase-6.2 news source registry -- PEP 544 Protocol + decorator.

Usage:

    from backend.news.registry import register, NewsSource

    @register("finnhub")
    class FinnhubSource:
        name = "finnhub"
        def fetch(self):
            yield {"title": "...", "url": "...", "body": "...",
                   "published_at": "2026-04-19T08:00:00Z", "source": "finnhub"}

Then:

    from backend.news.registry import get_sources
    sources = get_sources()          # all
    sources = get_sources(["finnhub"])  # subset by name

Module-level `_REGISTRY` dict maps source-name -> instance. Duplicate
registration under the same name raises `ValueError`; tests that need
to re-register (e.g. after `clear_registry()`) should call the clear
helper first.
"""
from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable


@runtime_checkable
class NewsSource(Protocol):
    """Structural type for news sources.

    Sources do not need to inherit from this class; any object with
    `name: str` + `fetch() -> Iterable[dict]` matches structurally.
    """

    name: str

    def fetch(self) -> Iterable[dict]:
        ...


_REGISTRY: dict[str, NewsSource] = {}


def register(name: str) -> Callable[[type], type]:
    """Decorator: register a source class under `name`.

    The decorator instantiates the class immediately (zero-arg init
    expected) so the registry holds ready-to-call instances. Raises
    `ValueError` on duplicate registration.
    """

    def decorator(cls: type) -> type:
        # Idempotent re-register when the SAME class re-runs the
        # decorator (happens naturally when Python double-imports a
        # package via `-m module.sub` -- first via the package
        # `__init__.py`, then as `__main__`). Different class under
        # the same name is still a ValueError.
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            # Match by class qualname because Python's `-m
            # package.module` re-executes the module as `__main__`,
            # creating a *distinct* class object with the same name
            # and body. Treat that as idempotent re-registration.
            if type(existing).__qualname__ == cls.__qualname__:
                return cls
            raise ValueError(
                f"news source {name!r} already registered to a "
                f"different class {type(existing).__qualname__!r}; "
                "call clear_registry() first if re-registering"
            )
        instance = cls()
        if getattr(instance, "name", None) != name:
            try:
                instance.name = name  # type: ignore[attr-defined]
            except Exception:
                pass
        _REGISTRY[name] = instance
        return cls

    return decorator


def get_sources(names: list[str] | None = None) -> dict[str, NewsSource]:
    """Return a name -> source dict. Optional filter by names."""
    if names is None:
        return dict(_REGISTRY)
    missing = [n for n in names if n not in _REGISTRY]
    if missing:
        raise KeyError(f"unknown news sources: {missing}")
    return {n: _REGISTRY[n] for n in names}


def clear_registry() -> None:
    """Test-only helper: wipe the registry."""
    _REGISTRY.clear()
