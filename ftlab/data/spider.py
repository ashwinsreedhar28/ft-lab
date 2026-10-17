"""Spider text-to-SQL with execution-match reward.

Questions: xlangai/spider (train 7,000 / validation 1,034; dev is the standard held-out).
Databases: HAL-9001/spider-databases re-hosts the official spider_data.zip
(SHA256 00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b);
`scripts/get_spider.sh` unpacks it to data/spider/database/<db_id>/<db_id>.sqlite.

Reward columns: `db_id`, `gold_sql`.
Correctness = execution match: run gold and prediction on the SQLite file, compare
result multisets (order-sensitive only when the gold query has ORDER BY). This is
Spider's "execution accuracy" and the same rule the training reward uses.
"""
from __future__ import annotations

import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset

from ftlab.config import DataSpec
from ftlab.prompting import make_messages

DEFAULT_SYSTEM = (
    "You translate questions into SQLite SQL for the given schema. "
    "Think briefly if needed, then output exactly one SQL query inside a ```sql fenced block."
)

DB_DIR = Path(os.environ.get("SPIDER_DB_DIR", "data/spider/database"))
EXEC_TIMEOUT_S = 5.0
MAX_ROWS = 5000

_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.S | re.I)
_ANY_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.S)


def db_path(db_id: str, db_dir: Path = DB_DIR) -> Path:
    return db_dir / db_id / f"{db_id}.sqlite"


@lru_cache(maxsize=512)
def schema_text(db_id: str, db_dir: str = str(DB_DIR)) -> str:
    """CREATE TABLE statements straight from sqlite_master, in one string."""
    con = sqlite3.connect(f"file:{db_path(db_id, Path(db_dir))}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    stmts = [re.sub(r"\s+", " ", r[0]).strip() for r in rows if r[0]]
    return "\n".join(s if s.endswith(";") else s + ";" for s in stmts)


def extract_sql(text: str) -> str | None:
    m = _SQL_BLOCK_RE.findall(text)
    if not m:
        m = _ANY_BLOCK_RE.findall(text)
    if not m:
        # fall back: a bare SELECT ... up to the end / first blank line
        bare = re.search(r"(SELECT\b.*?)(?:\n\s*\n|$)", text, re.S | re.I)
        return bare.group(1).strip().rstrip(";") if bare else None
    return m[-1].strip().rstrip(";")


class _Timeout(Exception):
    pass


def execute(sql: str, db_id: str, db_dir: Path = DB_DIR, timeout_s: float = EXEC_TIMEOUT_S):
    """Run read-only; return (rows | None, error | None). Rows are tuples of str()."""
    con = sqlite3.connect(f"file:{db_path(db_id, db_dir)}?mode=ro", uri=True, timeout=timeout_s)
    import time

    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        return [tuple(_cell(v) for v in r) for r in rows], None
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            return None, "timeout"
        return None, f"operational: {e}"
    except sqlite3.Error as e:
        return None, f"sqlite: {e}"
    finally:
        con.close()


def _cell(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v) if v is not None else None


def _ordered(gold_sql: str) -> bool:
    return re.search(r"\border\s+by\b", gold_sql, re.I) is not None


def exec_match(pred_sql: str | None, gold_sql: str, db_id: str, db_dir: Path = DB_DIR) -> tuple[bool, str | None]:
    if not pred_sql:
        return False, "no sql"
    gold_rows, gerr = execute(gold_sql, db_id, db_dir)
    if gold_rows is None:
        return False, f"gold failed: {gerr}"
    pred_rows, perr = execute(pred_sql, db_id, db_dir)
    if pred_rows is None:
        return False, perr
    if _ordered(gold_sql):
        return pred_rows == gold_rows, None
    return sorted(pred_rows, key=repr) == sorted(gold_rows, key=repr), None


def score(text: str, row: dict) -> float:
    ok, _ = exec_match(extract_sql(text), row["gold_sql"], row["db_id"], _db_dir(row))
    return 1.0 if ok else 0.0


def _db_dir(row: dict) -> Path:
    return Path(row.get("db_dir") or DB_DIR)


def gold_is_informative(gold_sql: str, db_id: str, db_dir: Path) -> bool:
    """Execution match is vacuous when the gold result is empty (any empty-returning
    query 'matches') or when gold itself fails. Eval splits drop these rows; train keeps
    them (an empty gold still teaches 'runs without error' via the shaping term)."""
    rows, err = execute(gold_sql, db_id, db_dir)
    return rows is not None and len(rows) > 0


def filter_informative(ds, db_dir: Path, sql_col: str = "query"):
    before = len(ds)
    ds = ds.filter(lambda r: gold_is_informative(r[sql_col], r["db_id"], db_dir))
    dropped = before - len(ds)
    if dropped:
        print(f"[eval filter] dropped {dropped}/{before} rows whose gold query returns nothing or errors", flush=True)
    return ds


def check_dbs(ds, db_dir: Path, hint: str):
    missing = sorted({r for r in ds.unique("db_id") if not db_path(r, db_dir).exists()})
    if missing:
        raise FileNotFoundError(f"{len(missing)} databases missing under {db_dir} (e.g. {missing[:3]}); run {hint}")


def build_rows(ds, system_prompt: str | None, db_dir: Path, sql_col: str = "query",
               evidence_col: str | None = None, default_system: str = DEFAULT_SYSTEM):
    """Shared by Spider and BIRD: schema + (evidence) + question -> conversational prompt."""
    sp = system_prompt if system_prompt is not None else default_system

    def build(r):
        parts = [f"Schema:\n{schema_text(r['db_id'], str(db_dir))}"]
        if evidence_col and r.get(evidence_col):
            parts.append(f"Hint: {r[evidence_col]}")
        parts.append(f"Question: {r['question']}")
        return {
            "prompt": make_messages("\n\n".join(parts), sp),
            "db_id": r["db_id"],
            "gold_sql": r[sql_col],
            "db_dir": str(db_dir),
        }

    return ds.map(build, remove_columns=ds.column_names)


def _rows(ds, system_prompt: str | None, db_dir: Path):
    check_dbs(ds, db_dir, "scripts/get_spider.sh")
    return build_rows(ds, system_prompt, db_dir)


def load_train(spec: DataSpec, system_prompt: str | None = None):
    db_dir = Path(spec.extra.get("db_dir", DB_DIR))
    ds = load_dataset("xlangai/spider", split="train").shuffle(seed=spec.seed)
    if spec.train_limit:
        ds = ds.select(range(spec.train_limit))
    return _rows(ds, system_prompt, db_dir)


def load_eval(spec: DataSpec, system_prompt: str | None = None):
    db_dir = Path(spec.extra.get("db_dir", DB_DIR))
    ds = load_dataset("xlangai/spider", split="validation")
    check_dbs(ds, db_dir, "scripts/get_spider.sh")
    ds = filter_informative(ds, db_dir)          # before the limit, so n is what's reported
    if spec.eval_limit:
        ds = ds.shuffle(seed=spec.seed).select(range(spec.eval_limit))
    return _rows(ds, system_prompt, db_dir)
