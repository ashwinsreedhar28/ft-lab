"""Task adapters. Each exposes

    load_train(spec: DataSpec, system_prompt) -> datasets.Dataset  # columns: prompt + reward cols
    load_eval(spec: DataSpec, system_prompt)  -> datasets.Dataset  # frozen held-out split
    score(completion_text, row) -> float                            # 1.0 / 0.0 correctness

`score` is the single definition of "correct" shared by the training reward and eval.
"""
from __future__ import annotations

from importlib import import_module

TASKS = {
    "gsm8k": "ftlab.data.gsm8k",
    "spider": "ftlab.data.spider",
    "bird": "ftlab.data.bird",
    "lakehouse": "ftlab.data.lakehouse",
}


def get_task(name: str):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return import_module(TASKS[name])
