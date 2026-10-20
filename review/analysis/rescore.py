"""Re-score saved harness results against the CURRENT task file.

    python -m review.analysis.rescore review/agent/results/sonnet_pub2_v4b.json [...] [--tasks review/bench/public_eval.jsonl]

Why: the bench changes (docstring mutants dropped Oct 20), and a run's decisions are per task, so a run
made on an older task file is still valid for every task that survived. Writes <label>_rescored.json
next to the original with the same shape (so `run.py --table` reads it) and provenance of both files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from review.bench.schema import Comment, Review
from review.bench.score import by_split, load_tasks, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--tasks", default="review/bench/public_eval.jsonl")
    a = ap.parse_args()
    tasks = load_tasks(a.tasks)
    ids = {t.id for t in tasks}
    for rp in a.results:
        r = json.loads(Path(rp).read_text())
        reviews, strong = {}, {}
        for rev in r["reviews"]:
            if rev["task_id"] not in ids:
                continue
            reviews[rev["task_id"]] = Review(decision=rev["decision"], comments=[Comment(**c) for c in rev.get("comments", [])],
                                             usd=rev.get("usd", 0.0))
            if any(p.get("strong") for p in rev.get("probes", [])):
                strong[rev["task_id"]] = True
        missing = [t.id for t in tasks if t.id not in reviews]
        kept_tasks = [t for t in tasks if t.id in reviews]
        scores = by_split(kept_tasks, reviews, strong)
        out = {**r, "label": r["label"] + "_rescored", "n": len(kept_tasks), "scores": scores,
               "reviews": [rev for rev in r["reviews"] if rev["task_id"] in ids],
               "rescored": {"from_tasks": r.get("provenance", {}).get("tasks_sha256"), "to": provenance(a.tasks),
                            "dropped_reviews": len(r["reviews"]) - len(kept_tasks), "tasks_without_review": missing}}
        p = Path(rp).with_name(Path(rp).stem + "_rescored.json")
        p.write_text(json.dumps(out, indent=1))
        s = scores["all"]
        print(f"{out['label']}: n={len(kept_tasks)} (dropped {out['rescored']['dropped_reviews']} reviews of removed tasks; "
              f"{len(missing)} current tasks unreviewed) bugs_approved={s['bugs_approved']}/{s['n_bug']} auto_ship={s['auto_ship_rate']:.2f} "
              f"escalate={s['escalation_rate']:.2f} blk_prec={s['block_precision']} strong={s['strong_localization']} ${s['usd_per_pr']:.3f}/PR -> {p}")


if __name__ == "__main__":
    main()
