"""Spider rewards: execution match (the real one) and runs-without-error (shaping).

Both execute SQL against the row's SQLite file; a 5 s per-query timeout keeps a
pathological cross join from stalling the step. Executed in a thread pool because
sqlite releases the GIL and each query is independent.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ftlab.data import spider as task
from ftlab.prompting import completion_text

_POOL = ThreadPoolExecutor(max_workers=16)


def _one(args):
    text, gold, db_id, db_dir = args
    sql = task.extract_sql(text)
    ok, err = task.exec_match(sql, gold, db_id, Path(db_dir))
    return sql, ok, err


def _run(completions, gold_sql, db_id, db_dir):
    texts = [completion_text(c) for c in completions]
    return list(_POOL.map(_one, zip(texts, gold_sql, db_id, db_dir)))


def exec_match(completions, gold_sql, db_id, db_dir, log_metric=None, log_extra=None, **_):
    """1.0 when the predicted query returns the same rows as the gold query."""
    res = _run(completions, gold_sql, db_id, db_dir)
    rewards = [1.0 if ok else 0.0 for _, ok, _ in res]
    if log_extra:
        log_extra("pred_sql", [s or "[none]" for s, _, _ in res])
        log_extra("exec_err", [e or "" for _, _, e in res])
    if log_metric:
        n = max(len(res), 1)
        log_metric("sql/exec_acc", sum(rewards) / n)
        log_metric("sql/no_sql_frac", sum(1 for s, _, _ in res if not s) / n)
    return rewards


def runs(completions, gold_sql, db_id, db_dir, **_):
    """0.1 if the query parses and executes at all (even if wrong). Shaping only."""
    out = []
    for c, d, dd in zip(completions, db_id, db_dir):
        sql = task.extract_sql(completion_text(c))
        if not sql:
            out.append(0.0)
            continue
        rows, err = task.execute(sql, d, Path(dd))
        out.append(0.1 if rows is not None else 0.0)
    return out
