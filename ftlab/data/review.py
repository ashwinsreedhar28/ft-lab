"""Review-decision task for GRPO (phase 3).

Rows come from review/bench/tasks.jsonl (shwin-bench) and review/mine output (public repos),
same schema. The model sees the same prompt as the single-shot reviewer in review/agent/run.py
and must emit the same JSON. Columns passed to rewards: kind, gold_hunks (JSON), task_id.

Split: `split_file` in DataSpec.extra points to a JSON {"train": [task ids], "eval": [...]}
built by repo (never the same repo in both). If absent, shwin-bench rows are eval and
anything else is train.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from ftlab.config import DataSpec
from ftlab.prompting import make_messages
from review.agent.run import SYSTEM, build_prompt, parse_review
from review.bench.schema import Hunk, overlaps

DEFAULT_CONDITION = "diff+graph"


def _rows(spec: DataSpec):
    paths = spec.extra.get("tasks", ["review/bench/public_tasks.jsonl"])
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows


def _split(spec: DataSpec, rows):
    sf = spec.extra.get("split_file")
    if sf:
        s = json.loads(Path(sf).read_text())
        return [r for r in rows if r["id"] in set(s["train"])], [r for r in rows if r["id"] in set(s["eval"])]
    if all("split" in r for r in rows):          # assemble.py rows carry their split (by PR hash)
        return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "eval"]
    shwin = {"emberserve", "aether", "serverless-lakehouse"}
    return [r for r in rows if r["repo"] not in shwin], [r for r in rows if r["repo"] in shwin]


def _ds(rows, spec: DataSpec, system_prompt):
    cond = spec.extra.get("condition", DEFAULT_CONDITION)
    sp = system_prompt if system_prompt is not None else SYSTEM
    return Dataset.from_list([
        {"prompt": make_messages(build_prompt(r, cond, None), sp), "kind": r["kind"],
         "gold_hunks": json.dumps(r.get("gold_hunks", [])), "task_id": r["id"]}
        for r in rows
    ])


def load_train(spec: DataSpec, system_prompt=None):
    train, _ = _split(spec, _rows(spec))
    ds = _ds(train, spec, system_prompt).shuffle(seed=spec.seed)
    return ds.select(range(spec.train_limit)) if spec.train_limit else ds


def load_eval(spec: DataSpec, system_prompt=None):
    _, ev = _split(spec, _rows(spec))
    ds = _ds(ev, spec, system_prompt)
    return ds.shuffle(seed=spec.seed).select(range(spec.eval_limit)) if spec.eval_limit else ds


def decision_of(text: str):
    rev, ok = parse_review(text, (0, 0), 0.0)
    return rev, ok


def score(text: str, row: dict) -> float:
    """Eval metric: 1.0 for a correct gate decision (approve clean / block bug). Escalate = 0."""
    rev, _ = decision_of(text)
    return 1.0 if (row["kind"] == "clean" and rev.decision == "approve") or \
                  (row["kind"] == "bug" and rev.decision == "block") else 0.0


def localized(rev, gold_hunks_json: str) -> bool:
    hunks = [Hunk(**h) for h in json.loads(gold_hunks_json or "[]")]
    return any(overlaps(c, h) for c in rev.comments for h in hunks)
