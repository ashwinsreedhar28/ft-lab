"""Teacher rationales for the thinking-mode student (phase 3b v4).

    python -m review.analysis.rationales [--tasks review/bench/public_train.jsonl] [--model claude-sonnet-5-5]
                                         [--out data/review_sft_v4.jsonl] [--balance] [--workers 6]

Same no-tools condition the student sees (diff + context graph), so the teacher's reasoning is reasoning the
student could in principle reproduce. Sonnet writes <= 150 words inside <think>...</think> naming the changed
line(s) it judged and why, then the decision JSON. Kept only when the decision is scorer-correct
(bug -> block, clean -> approve); --balance caps each kind at the smaller class. The SFT completion is
`<think>\\n{rationale}\\n</think>\\n\\n{json}`, which is how Qwen3's template renders a reasoning turn, so
with `enable_thinking: true` train and eval see the same shape.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from review.agent.run import SYSTEM, build_prompt, call_anthropic, parse_review

# Asking for the text inside <think> tags made the API refuse every call (stop_reason=refusal, 128/128):
# it reads as reasoning-tag injection. Ask for <rationale>, convert to Qwen3's <think> locally.
ASK = ("\n\nFirst, inside <rationale> and </rationale>, explain in at most 150 words which changed line(s) you "
       "judged and why they are correct or defective. Then output the JSON object.")
_THINK = re.compile(r"<rationale>(.*?)</rationale>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="review/bench/public_train.jsonl")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--out", default="data/review_sft_v4.jsonl")
    ap.add_argument("--raw", default="review/agent/results/rationales_train.jsonl", help="every teacher reply, for the record")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--thinking", action="store_true", help="leave the teacher's hidden thinking on (default off: the visible rationale is the point)")
    ap.add_argument("--retries", type=int, default=3, help="re-call on refusal / truncated rationale (both are stochastic)")
    a = ap.parse_args()
    import review.agent.run as run_mod
    if not a.thinking:
        run_mod.THINKING_BUDGET = 0
    tasks = [json.loads(l) for l in Path(a.tasks).read_text().splitlines() if l.strip()]
    done = {}
    rp = Path(a.raw)
    if rp.exists():
        for l in rp.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if not d["raw"].startswith("API_"):      # refusals / errors are retried
                    done[d["task_id"]] = d

    def one(t):
        if t["id"] in done:
            return done[t["id"]]
        prompt = build_prompt(t, "diff+graph", None) + ASK
        spent, text, usage = 0.0, "", (0, 0)
        for _ in range(a.retries):
            text, usage, usd = call_anthropic(a.model, prompt, a.max_tokens)
            spent += usd
            if not text.startswith("API_") and _THINK.search(text):
                break
        return {"task_id": t["id"], "kind": t["kind"], "raw": text, "usage": usage, "usd": spent}

    rp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(one, tasks):
            rows.append(r)
            print(f"{r['task_id']:45s} {r['kind']:5s} ${r['usd']:.3f}", flush=True)

    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))    # one row per task, latest attempt
    out, stats = [], Counter()
    by_id = {t["id"]: t for t in tasks}
    for r in rows:
        t = by_id[r["task_id"]]
        m = _THINK.search(r["raw"])
        review, ok = parse_review(_THINK.sub("", r["raw"]), (0, 0), 0.0)   # braces in the prose break the JSON match
        if not ok or not m:
            stats["unparsable_or_no_rationale"] += 1; continue
        correct = (t["kind"] == "bug" and review.decision == "block") or (t["kind"] == "clean" and review.decision == "approve")
        if not correct:
            stats[f"wrong:{t['kind']}->{review.decision}"] += 1; continue
        rationale = " ".join(m.group(1).split())
        target = {"decision": review.decision,
                  "comments": [{"file": c.file, "line_start": c.line_start, "line_end": c.line_end, "body": c.body} for c in review.comments][:4]}
        out.append({"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(t, "diff+graph", None)}],
                    "completion": [{"role": "assistant", "content": f"<think>\n{rationale}\n</think>\n\n{json.dumps(target)}"}],
                    "task_id": t["id"], "kind": t["kind"], "source": t.get("source", ""), "teacher": a.model})
        stats[f"kept:{t['kind']}"] += 1
    if a.balance:
        rng = random.Random(0)
        by = {"bug": [x for x in out if x["kind"] == "bug"], "clean": [x for x in out if x["kind"] == "clean"]}
        n = min(len(v) for v in by.values())
        out = [x for k in by for x in rng.sample(by[k], n)]; rng.shuffle(out)
        stats["balanced_per_kind"] = n
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in out))
    spent = sum(r["usd"] for r in rows)
    print(f"{len(out)} traces -> {a.out}  ${spent:.2f}   " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
