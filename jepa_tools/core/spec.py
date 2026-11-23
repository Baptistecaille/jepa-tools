"""Run spec: one object that fully determines a run."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import yaml

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"
SECTIONS = ("data", "sampler", "encoder", "target", "predictor", "objective", "trainer")


@dataclass
class RunSpec:
    preset: str | None = None  # name of a preset to load, merged under this spec
    data: dict = field(default_factory=dict)
    sampler: dict = field(default_factory=dict)
    encoder: dict = field(default_factory=dict)
    target: dict = field(default_factory=dict)
    predictor: dict = field(default_factory=dict)
    objective: dict = field(default_factory=dict)
    trainer: dict = field(default_factory=dict)
    # Allow shared+detach targets without a regularizer (collapse risk).
    allow_collapse_risk: bool = False

    def to_dict(self) -> dict:
        return copy.deepcopy({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown key(s) in run spec: {', '.join(unknown)}")
        return cls(**copy.deepcopy(d))


def _read_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict, top: dict) -> dict:
    """Recursive merge; values in ``top`` win."""
    out = copy.deepcopy(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_override(d: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must look like 'a.b=value', got {override!r}")
    path, raw = override.split("=", 1)
    keys = path.split(".")
    if keys[0] not in {f.name for f in fields(RunSpec)}:
        raise ValueError(f"Unknown key {keys[0]!r} in override {override!r}")
    node: Any = d
    for i, key in enumerate(keys[:-1]):
        if not isinstance(node.get(key), dict):
            raise ValueError(f"Unknown key {key!r} in override {override!r}")
        node = node[key]
    leaf = keys[-1]
    if len(keys) > 1 and node and leaf not in node:
        raise ValueError(
            f"Unknown key {leaf!r} in override {override!r} "
            f"(known: {', '.join(sorted(node))})"
        )
    node[leaf] = yaml.safe_load(raw)


def load_spec(
    source: str | os.PathLike | dict,
    overrides: Sequence[str] = (),
    preset_dir: str | os.PathLike | None = None,
) -> RunSpec:
    user = copy.deepcopy(source) if isinstance(source, dict) else _read_yaml(source)
    merged = user
    if user.get("preset"):
        preset_path = Path(preset_dir or PRESET_DIR) / f"{user['preset']}.yaml"
        if not preset_path.exists():
            raise ValueError(f"Unknown preset {user['preset']!r} (looked in {preset_path.parent})")
        merged = _merge(_read_yaml(preset_path), user)
    for ov in overrides:
        _apply_override(merged, ov)
    return RunSpec.from_dict(merged)