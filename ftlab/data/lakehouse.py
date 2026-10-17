"""Lakehouse text-to-SQL over the serverless-lakehouse Snowflake schema (phase 3).

Questions come from tools/gen_lakehouse_questions.py -> data/lakehouse/questions.jsonl,
each with: id, question, gold_sql, gold_rows (result snapshot, JSON), split (train/eval),
template, schema_tables. Gold rows are snapshotted at generation time so reward and eval
do not hit Snowflake per completion: the prediction is executed (Snowflake or the DuckDB
mirror, see below) and compared to the snapshot.

Execution backend, chosen by LAKEHOUSE_BACKEND:
  duckdb    (default) data/lakehouse/mirror.duckdb — a Parquet export of the gold/silver
            tables, so training rewards cost nothing and the Snowflake trial can expire.
  snowflake live, via SNOWFLAKE_* env; used once to snapshot gold and for the final eval.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from datasets import Dataset

from ftlab.config import DataSpec
from ftlab.data.spider import extract_sql
from ftlab.prompting import make_messages

LH_DIR = Path(os.environ.get("LAKEHOUSE_DIR", "data/lakehouse"))
BACKEND = os.environ.get("LAKEHOUSE_BACKEND", "duckdb")
MAX_ROWS = 2000

DEFAULT_SYSTEM = (
    "You write SQL for an analytics warehouse holding Runpod Serverless job and endpoint "
    "data in a medallion layout (bronze/silver/gold). Use only the tables in the schema. "
    "Output exactly one SQL query inside a ```sql fenced block."
)


def _schema_text(root: Path) -> str:
    return (root / "schema.sql").read_text()


def execute(sql: str, root: Path = LH_DIR):
    """Return (rows | None, err | None); rows as tuples of normalized strings."""
    try:
        if BACKEND == "duckdb":
            import duckdb

            con = duckdb.connect(str(root / "mirror.duckdb"), read_only=True)
            try:
                cur = con.execute(sql)
                rows = cur.fetchmany(MAX_ROWS)
            finally:
                con.close()
        else:
            import snowflake.connector

            con = snowflake.connector.connect(
                account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
                password=os.environ["SNOWFLAKE_PASSWORD"], warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
                database=os.environ.get("SNOWFLAKE_DATABASE", "LAKEHOUSE"),
                schema=os.environ.get("SNOWFLAKE_SCHEMA", "GOLD"),
            )
            try:
                cur = con.cursor()
                cur.execute(sql)
                rows = cur.fetchmany(MAX_ROWS)
            finally:
                con.close()
        return [tuple(_cell(v) for v in r) for r in rows], None
    except Exception as e:  # noqa: BLE001 - any backend error is "query failed"
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def _cell(v):
    if v is None:
        return None
    if isinstance(v, float):
        return f"{v:.6g}"
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _ordered(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, re.I) is not None


def match(pred_sql: str | None, gold_sql: str, gold_rows: list, root: Path = LH_DIR) -> tuple[bool, str | None]:
    if not pred_sql:
        return False, "no sql"
    pred, err = execute(pred_sql, root)
    if pred is None:
        return False, err
    gold = [tuple(r) for r in gold_rows]
    if _ordered(gold_sql):
        return pred == gold, None
    return sorted(pred, key=repr) == sorted(gold, key=repr), None


def score(text: str, row: dict) -> float:
    ok, _ = match(extract_sql(text), row["gold_sql"], json.loads(row["gold_rows"]), Path(row["lh_dir"]))
    return 1.0 if ok else 0.0


def _load(spec: DataSpec, split: str, system_prompt: str | None):
    root = Path(spec.extra.get("lh_dir", LH_DIR))
    q = root / "questions.jsonl"
    if not q.exists():
        raise FileNotFoundError(f"{q} missing; run tools/gen_lakehouse_questions.py")
    rows = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    sp = system_prompt if system_prompt is not None else DEFAULT_SYSTEM
    schema = _schema_text(root)
    ds = Dataset.from_list([
        {
            "prompt": make_messages(f"Schema:\n{schema}\n\nQuestion: {r['question']}", sp),
            "gold_sql": r["gold_sql"],
            "gold_rows": json.dumps(r["gold_rows"]),
            "lh_dir": str(root),
            "qid": r["id"],
        }
        for r in rows
    ])
    if split == "train":
        ds = ds.shuffle(seed=spec.seed)
        if spec.train_limit:
            ds = ds.select(range(spec.train_limit))
    elif spec.eval_limit:
        ds = ds.shuffle(seed=spec.seed).select(range(spec.eval_limit))
    return ds


def load_train(spec: DataSpec, system_prompt: str | None = None):
    return _load(spec, "train", system_prompt)


def load_eval(spec: DataSpec, system_prompt: str | None = None):
    return _load(spec, "eval", system_prompt)
