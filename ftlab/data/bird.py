"""BIRD dev (1,534 questions, 11 databases) as a second public text-to-SQL eval.

Why: bigger, dirtier schemas than Spider (closer to a warehouse), an `evidence`
hint per question, execution-scored like Spider. Eval only here; we don't train on it.

Data: https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip (~1.7 GB with databases).
`scripts/get_bird.sh` unpacks to data/bird/{dev.json, database/<db_id>/<db_id>.sqlite}.
dev.json fields: question_id, db_id, question, evidence, SQL, difficulty.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset

from ftlab.config import DataSpec
from ftlab.data.spider import build_rows, check_dbs, filter_informative

BIRD_DIR = Path(os.environ.get("BIRD_DIR", "data/bird"))
DEFAULT_SYSTEM = (
    "You translate questions into SQLite SQL for the given schema. A hint may give "
    "domain knowledge needed for the query; use it. Output exactly one SQL query inside a ```sql fenced block."
)


def _load(spec: DataSpec):
    root = Path(spec.extra.get("bird_dir", BIRD_DIR))
    dev = root / "dev.json"
    if not dev.exists():
        raise FileNotFoundError(f"{dev} missing; run scripts/get_bird.sh")
    rows = json.loads(dev.read_text())
    ds = Dataset.from_list([
        {"db_id": r["db_id"], "question": r["question"], "evidence": r.get("evidence") or "",
         "query": r["SQL"], "difficulty": r.get("difficulty", "")}
        for r in rows
    ])
    return ds, root / "database"


def load_eval(spec: DataSpec, system_prompt: str | None = None):
    ds, db_dir = _load(spec)
    check_dbs(ds, db_dir, "scripts/get_bird.sh")
    ds = filter_informative(ds, db_dir)
    if spec.eval_limit:
        ds = ds.shuffle(seed=spec.seed).select(range(spec.eval_limit))
    return build_rows(ds, system_prompt, db_dir, evidence_col="evidence", default_system=DEFAULT_SYSTEM)


def load_train(spec: DataSpec, system_prompt: str | None = None):
    raise NotImplementedError("BIRD is eval-only in ft-lab; train on spider or lakehouse")


# correctness is identical to Spider's execution match
from ftlab.data.spider import score  # noqa: E402,F401
