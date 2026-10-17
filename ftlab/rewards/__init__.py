"""Reward registry. A reward is `f(completions, **cols) -> list[float]`.

TRL calls each reward with `prompts`, `completions`, `completion_ids`,
`trainer_state`, `log_extra`, `log_metric` and every non-prompt dataset column
as a keyword argument (lists aligned with `completions`). We accept **kwargs and
pull what we need, so a reward only depends on the columns its task adapter emits.

The correctness reward delegates to the task adapter's `score`, so training
reward == eval metric by construction.
"""
from __future__ import annotations

from collections.abc import Callable

from ftlab.rewards import gsm8k as _gsm8k
from ftlab.rewards import sql as _sql

REWARDS: dict[str, Callable[..., list[float]]] = {
    "gsm8k_correct": _gsm8k.correct,
    "gsm8k_format": _gsm8k.format_ok,
    "sql_exec": _sql.exec_match,
    "sql_runs": _sql.runs,
    "length_penalty": _gsm8k.length_penalty,
}


def get_rewards(names: list[str]) -> list[Callable[..., list[float]]]:
    missing = [n for n in names if n not in REWARDS]
    if missing:
        raise KeyError(f"unknown rewards {missing}; known: {sorted(REWARDS)}")
    return [REWARDS[n] for n in names]
