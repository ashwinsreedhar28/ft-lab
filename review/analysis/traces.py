"""Teacher traces -> SFT dataset for the no-tools triage gate.

    python -m review.analysis.traces review/agent/results/sonnet_train_v5.json [...] \
        [--tasks review/bench/public_train.jsonl] [--out data/review_sft.jsonl] [--keep-escalate]

The teacher (Sonnet under harness v5: suite delta + probes) reviewed the TRAIN tasks. The student
(Qwen3-4B, no tools) will see only diff + context graph (`build_prompt(task, "diff+graph")`) and must emit
the same decision JSON. Keep a trace only when the scorer says the teacher's decision was right:
bug -> block, clean -> approve (escalate dropped unless --keep-escalate; it teaches abstention).
The target is the teacher's final JSON with its comments, re-serialized compactly (no probe field: the
student has no harness to run one). Prompt/completion conversational format for TRL SFTTrainer.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from review.agent.run import SYSTEM, build_prompt, parse_review


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--tasks", default="review/bench/public_train.jsonl")
    ap.add_argument("--out", default="data/review_sft.jsonl")
    ap.add_argument("--keep-escalate", action="store_true")
    ap.add_argument("--balance", action="store_true",
                    help="cap each kind at the smaller class (seeded). 54:25 bug:clean made a block-everything SFT prior "
                         "that GRPO could not move (identical group rewards -> zero advantage).")
    a = ap.parse_args()
    tasks = {t["id"]: t for t in (json.loads(l) for l in Path(a.tasks).read_text().splitlines() if l.strip())}
    out, stats = [], Counter()
    for rp in a.results:
        r = json.loads(Path(rp).read_text())
        for rev in r["reviews"]:
            t = tasks.get(rev["task_id"])
            if not t:
                stats["not_in_tasks"] += 1; continue
            review, ok = parse_review(rev["raw"], (0, 0), 0.0)
            if not ok:
                stats["unparsable"] += 1; continue
            correct = (t["kind"] == "bug" and review.decision == "block") or (t["kind"] == "clean" and review.decision == "approve")
            if review.decision == "escalate" and a.keep_escalate:
                correct = True
            if not correct:
                stats[f"wrong:{t['kind']}->{review.decision}"] += 1; continue
            target = {"decision": review.decision,
                      "comments": [{"file": c.file, "line_start": c.line_start, "line_end": c.line_end, "body": c.body} for c in review.comments][:4]}
            out.append({"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(t, "diff+graph", None)}],
                        "completion": [{"role": "assistant", "content": json.dumps(target)}],
                        "task_id": t["id"], "kind": t["kind"], "source": t.get("source", ""), "teacher": r["label"]})
            stats[f"kept:{t['kind']}"] += 1
    if a.balance:
        import random
        rng = random.Random(0)
        by = {"bug": [x for x in out if x["kind"] == "bug"], "clean": [x for x in out if x["kind"] == "clean"]}
        n = min(len(v) for v in by.values())
        out = [x for k in by for x in rng.sample(by[k], n)]
        rng.shuffle(out)
        stats["balanced_per_kind"] = n
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in out))
    print(f"{len(out)} traces -> {a.out}   " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
