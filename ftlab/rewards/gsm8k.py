"""GSM8K rewards. Function __name__ becomes the metric name: rewards/<name>/mean."""
from __future__ import annotations

from ftlab.data import gsm8k as task
from ftlab.prompting import completion_text


def correct(completions, answer, log_metric=None, log_extra=None, **_):
    """1.0 if the extracted final number equals the gold answer, else 0.0."""
    texts = [completion_text(c) for c in completions]
    preds = [task.extract_pred(t) for t in texts]
    rewards = [1.0 if p is not None and p == a else 0.0 for p, a in zip(preds, answer)]
    if log_extra:
        log_extra("gold", list(answer))
        log_extra("pred", [p if p is not None else "[none]" for p in preds])
    if log_metric:
        log_metric("gsm8k/acc", sum(rewards) / max(len(rewards), 1))
    return rewards


def format_ok(completions, **_):
    """0.2 if the '#### N' answer line is present (small shaping term)."""
    return [0.2 if task.has_format(completion_text(c)) else 0.0 for c in completions]


def length_penalty(completion_ids, **_):
    """-0.1 per 256 tokens beyond 512. Off by default; a lever if lengths blow up."""
    return [-0.1 * max(0, len(ids) - 512) / 256 for ids in completion_ids]
