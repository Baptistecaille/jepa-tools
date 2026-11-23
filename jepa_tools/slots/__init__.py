"""Slot implementations. Every submodule is imported on package import so that
its ``@register`` decorators run: adding a piece means adding a file here."""

from __future__ import annotations

import importlib
import pkgutil


def discover(package: str = "jepa_tools.slots") -> None:
    """Import every submodule of ``package`` (idempotent)."""
    pkg = importlib.import_module(package)
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package}."):
        importlib.import_module(info.name)


discover()
