"""Run configuration: one YAML file per experiment, loaded into a dataclass.

Only the fields below are read. Anything under `grpo:` is passed straight to
trl.GRPOConfig as keyword arguments, so new TRL knobs need no code change.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraSpec:
    enabled: bool = False
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"


@dataclass
class DataSpec:
    task: str = "gsm8k"            # key in ftlab.data.registry
    train_limit: int | None = None  # cap train rows (smoke tests)
    eval_limit: int | None = None   # cap frozen eval rows (None = full split)
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)  # task-specific (e.g. spider db dir)


@dataclass
class RunConfig:
    name: str
    model: str = "Qwen/Qwen3-1.7B"
    output_dir: str = "runs/${name}"
    data: DataSpec = field(default_factory=DataSpec)
    rewards: list[str] = field(default_factory=lambda: ["gsm8k_correct", "gsm8k_format"])
    reward_weights: list[float] | None = None
    lora: LoraSpec = field(default_factory=LoraSpec)
    system_prompt: str | None = None
    enable_thinking: bool = False
    grpo: dict[str, Any] = field(default_factory=dict)   # -> trl.GRPOConfig(**grpo)

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir.replace("${name}", self.name))


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path, depth: int = 0) -> dict:
    """Recursive `extends:` resolution (child overrides parent, dicts merge)."""
    if depth > 8:
        raise RecursionError(f"extends chain too deep at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent:
        raw = _merge(_load_raw(path.parent / parent, depth + 1), raw)
    return raw


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML config. `extends: other.yaml` pulls a base file first.

    overrides: ["grpo.max_steps=3", "data.train_limit=16"] style dotted keys.
    """
    raw = _load_raw(Path(path))
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return _from_dict(raw)


def _from_dict(raw: dict) -> RunConfig:
    raw = dict(raw)
    data = DataSpec(**raw.pop("data", {}) or {})
    lora = LoraSpec(**raw.pop("lora", {}) or {})
    return RunConfig(data=data, lora=lora, **raw)


def to_dict(cfg: RunConfig) -> dict:
    return dataclasses.asdict(cfg)
