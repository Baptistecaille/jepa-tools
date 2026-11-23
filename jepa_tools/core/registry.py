"""Slot registry: ``@register("encoder", "vit")`` then ``get("encoder", "vit")``."""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

KINDS = ("sampler", "encoder", "target", "predictor", "distance", "regularizer")

_REGISTRY: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}


class RegistryError(ValueError):
    """Unknown kind or name, or a duplicate registration."""


def _check_kind(kind: str) -> dict[str, type]:
    if kind not in _REGISTRY:
        raise RegistryError(
            f"Unknown slot kind {kind!r}. Available kinds: {', '.join(KINDS)}"
        )
    return _REGISTRY[kind]


def register(kind: str, name: str) -> Callable[[T], T]:
    """Class decorator registering a piece under ``(kind, name)``."""
    slot = _check_kind(kind)

    def decorator(cls: T) -> T:
        existing = slot.get(name)
        if existing is not None and existing is not cls:
            raise RegistryError(
                f"{kind} {name!r} is already registered ({existing.__module__}.{existing.__qualname__})"
            )
        slot[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def get(kind: str, name: str) -> type:
    slot = _check_kind(kind)
    try:
        return slot[name]
    except KeyError:
        available = ", ".join(sorted(slot)) or "(none registered)"
        raise RegistryError(
            f"Unknown {kind} {name!r}. Available: {available}"
        ) from None


def names(kind: str) -> list[str]:
    return sorted(_check_kind(kind))
