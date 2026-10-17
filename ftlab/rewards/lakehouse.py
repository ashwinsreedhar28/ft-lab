"""Lakehouse rewards: result match against the gold snapshot, plus runs-at-all shaping."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ftlab.data import lakehouse as task
from ftlab.data.spider import extract_sql
from ftlab.prompting import completion_text

_POOL = ThreadPoolExecutor(max_workers=8)


def _one(args):
    text, gold_sql, gold_rows, lh_dir = args
    sql = extract_sql(text)
    ok, err = task.match(sql, gold_sql, json.loads(gold_rows), Path(lh_dir))
    return sql, ok, err


def lh_match(completions, gold_sql, gold_rows, lh_dir, log_metric=None, log_extra=None, **_):
    """1.0 when the predicted query's result equals the gold snapshot."""
    texts = [completion_text(c) for c in completions]
    res = list(_POOL.map(_one, zip(texts, gold_sql, gold_rows, lh_dir)))
    rewards = [1.0 if ok else 0.0 for _, ok, _ in res]
    if log_extra:
        log_extra("pred_sql", [s or "[none]" for s, _, _ in res])
        log_extra("err", [e or "" for _, _, e in res])
    if log_metric:
        n = max(len(res), 1)
        log_metric("lh/match", sum(rewards) / n)
        log_metric("lh/failed_frac", sum(1 for _, ok, e in res if not ok and e and e != "no sql") / n)
    return rewards


def lh_runs(completions, lh_dir, **_):
    """0.1 if the query executes. Shaping only."""
    out = []
    for c, d in zip(completions, lh_dir):
        sql = extract_sql(completion_text(c))
        if not sql:
            out.append(0.0); continue
        rows, _ = task.execute(sql, Path(d))
        out.append(0.1 if rows is not None else 0.0)
    return out
