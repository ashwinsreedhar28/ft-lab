"""GSM8K (openai/gsm8k, config "main"). Train 7,473 / test 1,319.

Reward columns: `answer` (the gold number as a string).
"""
from __future__ import annotations

import re

from datasets import load_dataset

from ftlab.config import DataSpec
from ftlab.prompting import make_messages

DEFAULT_SYSTEM = (
    "Solve the math problem. Reason step by step in a few short lines, "
    "then give the final numeric answer on its own last line in the form:\n"
    "#### <number>"
)

_GOLD_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_PRED_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
_LAST_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(answer_field: str) -> str:
    m = _GOLD_RE.search(answer_field)
    if not m:
        raise ValueError(f"no #### answer in: {answer_field[-80:]!r}")
    return _norm(m.group(1))


def _norm(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else str(f)


def extract_pred(text: str, strict: bool = False) -> str | None:
    """Return the predicted number. strict=True requires the '#### N' line."""
    m = _PRED_RE.findall(text)
    if m:
        return _norm(m[-1])
    if strict:
        return None
    nums = _LAST_NUM_RE.findall(text)
    return _norm(nums[-1]) if nums else None


def has_format(text: str) -> bool:
    return _PRED_RE.search(text) is not None


def score(text: str, row: dict) -> float:
    pred = extract_pred(text, strict=False)
    return 1.0 if pred is not None and pred == row["answer"] else 0.0


def _rows(ds, system_prompt: str | None):
    sp = system_prompt if system_prompt is not None else DEFAULT_SYSTEM
    return ds.map(
        lambda r: {"prompt": make_messages(r["question"], sp), "answer": gold_answer(r["answer"])},
        remove_columns=ds.column_names,
    )


def n_steps(answer_field: str) -> int:
    """Number of <<a*b=c>> calculator annotations in the gold solution: a proxy for difficulty."""
    return answer_field.count("<<")


def load_train(spec: DataSpec, system_prompt: str | None = None):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    min_steps = int(spec.extra.get("min_steps", 0))
    if min_steps:
        before = len(ds)
        ds = ds.filter(lambda r: n_steps(r["answer"]) >= min_steps)
        print(f"[gsm8k] hard pool: {len(ds)}/{before} train problems with >= {min_steps} calc steps", flush=True)
    ds = ds.shuffle(seed=spec.seed)
    if spec.train_limit:
        ds = ds.select(range(spec.train_limit))
    return _rows(ds, system_prompt)


def load_eval(spec: DataSpec, system_prompt: str | None = None):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if spec.eval_limit:
        ds = ds.shuffle(seed=spec.seed).select(range(spec.eval_limit))
    return _rows(ds, system_prompt)
