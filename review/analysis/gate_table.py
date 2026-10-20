"""Score a local model's ftlab.eval output as a review gate, in the frontier harness's table.

    python -m review.analysis.gate_table runs/review_grpo_4b/eval_base.json runs/review_grpo_4b/eval_sft.json ... \
        [--tasks review/bench/public_eval.jsonl]

ftlab.eval on the `review` task stores each row's completion (the model's decision JSON). This parses them
with the same parse_review, scores with the same by_split (bugs approved, auto-ship, block precision,
localization, per source), writes review/agent/results/local_<label>.json in the results shape, and prints
`run.py --table` over them so the 4B rows sit next to Sonnet/Haiku. Condition is "gate" (no tools).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from review.agent.run import RESULTS, parse_review, table
from review.bench.score import by_split, load_tasks, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--tasks", default="review/bench/public_eval.jsonl")
    a = ap.parse_args()
    tasks = load_tasks(a.tasks)
    ids = {t.id for t in tasks}
    written = []
    for ep in a.evals:
        e = json.loads(Path(ep).read_text())
        reviews, rows = {}, []
        for r in e["rows"]:
            if r["task_id"] not in ids:
                continue
            rev, ok = parse_review(r["completion"], (0, 0), 0.0)
            reviews[r["task_id"]] = rev
            rows.append({"task_id": r["task_id"], "decision": rev.decision, "usd": 0.0, "probes": [],
                         "comments": [c.__dict__ for c in rev.comments], "parsed": ok, "raw": r["completion"][:4000]})
        kept = [t for t in tasks if t.id in reviews]
        scores = by_split(kept, reviews)
        label = f"local_{Path(ep).stem.replace('eval_', '')}_{Path(ep).parent.name}"
        out = {"label": label, "model": e["model"], "condition": "gate", "n": len(kept),
               "unparsable": sum(1 for r in rows if not r["parsed"]), "wall_s": e.get("wall_s"),
               "provenance": provenance(a.tasks), "scores": scores, "reviews": rows}
        RESULTS.mkdir(exist_ok=True)
        p = RESULTS / f"{label}.json"; p.write_text(json.dumps(out, indent=1)); written.append(str(p))
        s = scores["all"]
        print(f"{label}: n={len(kept)} bugs_approved={s['bugs_approved']}/{s['n_bug']} auto_ship={s['auto_ship_rate']:.2f} "
              f"escalate={s['escalation_rate']:.2f} blk_prec={s['block_precision']} unparsable={out['unparsable']} -> {p}")
    table(written + [str(p) for p in RESULTS.glob("*pub2_v5*.json")] + [str(p) for p in RESULTS.glob("*pub2_v4b_rescored.json")])


if __name__ == "__main__":
    main()
