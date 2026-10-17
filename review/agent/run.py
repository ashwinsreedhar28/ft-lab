"""Run a reviewer over review-bench under one context condition and score it.

    python -m review.agent.run --model claude-sonnet-5-5 --condition diff      --label sonnet_diff
    python -m review.agent.run --model claude-haiku-5-5  --condition diff+graph --label haiku_graph
    python -m review.agent.run --table review/agent/results/*.json

Conditions
  diff        the PR diff and file list
  diff+graph  plus the two-hop neighborhood from build.py
  diff+run    plus the repo's full test result at head (precomputed by `--precompute-runs`, docker)
  full        both

Backends: anthropic (ANTHROPIC_API_KEY), openai-compatible (--base-url, for a vLLM endpoint
serving a tuned Qwen; OPENAI_API_KEY may be any string). The reviewer must answer with one
JSON object: {"decision": "approve"|"block"|"escalate", "comments": [{file, line_start,
line_end, body, proposed_test?}]}. Anything unparsable is scored as "escalate" and counted.

This is the *single-shot* reviewer used for baselines. The agentic loop (run suite, write a
probe test, re-decide) lives in review/agent/loop.py and reuses the same prompt pieces.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from review.bench.schema import Comment, Review
from review.bench.score import by_split, load_tasks

RESULTS = Path(__file__).parent / "results"
SYSTEM = (
    "You are the code review gate for a production repository. Your approval merges and deploys "
    "the change with no further human review. Decide:\n"
    "  approve  - you are confident the change is correct and safe to ship.\n"
    "  block    - you found a defect; cite its location.\n"
    "  escalate - you cannot establish safety from what you can see; a human must look.\n"
    "Approving a change that contains a defect is the worst outcome. Blocking a correct change costs "
    "an engineer's time. Escalating is cheap but useless if done for everything.\n"
    "Respond with exactly one JSON object and nothing else:\n"
    '{"decision": "approve|block|escalate", "comments": [{"file": "...", "line_start": N, "line_end": N, '
    '"body": "...", "proposed_test": "optional pytest source"}]}'
)


CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".sh", ".bash", ".yaml", ".yml", ".toml", ".cfg", ".ini",
            ".sql", ".rs", ".go", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".swift", ".dockerfile"}
MAX_DIFF_CHARS = 350_000   # ~90k tokens; Sonnet/Haiku 5.5 take it


def split_diff(diff: str) -> list[tuple[str, str]]:
    """[(path, per-file diff text)] in order."""
    parts, cur, buf = [], None, []
    for ln in diff.splitlines(keepends=True):
        if ln.startswith("diff --git "):
            if cur is not None:
                parts.append((cur, "".join(buf)))
            cur = ln.split(" b/", 1)[1].strip() if " b/" in ln else ln.strip()
            buf = [ln]
        else:
            buf.append(ln)
    if cur is not None:
        parts.append((cur, "".join(buf)))
    return parts


def reviewable_diff(diff: str) -> tuple[str, list[str]]:
    """Keep code files whole; drop data/markup/lockfiles and say which. Never cut a file mid-hunk."""
    kept, omitted, total = [], [], 0
    for path, text in split_diff(diff):
        ext = ("." + path.rsplit(".", 1)[-1].lower()) if "." in path.rsplit("/", 1)[-1] else ""
        is_code = ext in CODE_EXT or path.endswith("Dockerfile") or path.endswith("Makefile")
        if not is_code or total + len(text) > MAX_DIFF_CHARS:
            omitted.append(f"{path} ({len(text)} chars{'' if is_code else ', non-code'})")
            continue
        kept.append(text); total += len(text)
    return "".join(kept), omitted


def build_prompt(task: dict, condition: str, run_result: str | None) -> str:
    diff, omitted = reviewable_diff(task["diff"])
    parts = [f"Repository: {task['repo']}\nFiles changed: {', '.join(task['files'])}"]
    if omitted:
        parts.append("Not shown (non-code or over budget; treat as unreviewed): " + ", ".join(omitted))
    if condition in ("diff+graph", "full") and task.get("neighborhood"):
        parts.append("Context around the change (mechanically extracted):\n" + task["neighborhood"])
    if condition in ("diff+run", "full") and run_result:
        parts.append("Test suite result at the PR head:\n" + run_result)
    parts.append("Diff:\n```diff\n" + diff + "\n```")
    return "\n\n".join(parts)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_review(text: str, usage: tuple[int, int], usd: float) -> tuple[Review, bool]:
    m = _JSON_RE.search(text or "")
    try:
        obj = json.loads(m.group(0)) if m else {}
        dec = obj.get("decision", "").lower()
        if dec not in ("approve", "block", "escalate"):
            raise ValueError(dec)
        comments = [Comment(file=str(c.get("file", "")), line_start=int(c.get("line_start", 0)),
                            line_end=int(c.get("line_end", c.get("line_start", 0))), body=str(c.get("body", "")),
                            proposed_test=c.get("proposed_test"))
                    for c in obj.get("comments", []) if isinstance(c, dict)]
        return Review(dec, comments, usage[0], usage[1], usd), True
    except (ValueError, TypeError, json.JSONDecodeError):
        return Review("escalate", [], usage[0], usage[1], usd), False


THINKING_BUDGET = int(os.environ.get("REVIEW_THINKING_BUDGET", "8000"))   # 0 disables the thinking param

# $/M tokens (input, output); update when prices change
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-5": (3.0, 15.0),
          "claude-haiku-5-5": (1.0, 5.0), "claude-sonnet-5-5": (3.0, 15.0)}  # 5.5 rows assumed = 4.5; fix from the invoice


def call_anthropic(model: str, prompt: str, max_tokens: int) -> tuple[str, tuple[int, int], float]:
    import anthropic

    client = anthropic.Anthropic(max_retries=0)
    last = ""
    for attempt in range(7):
        try:
            kw = dict(model=model, max_tokens=max_tokens, system=SYSTEM,
                      messages=[{"role": "user", "content": prompt}])
            if THINKING_BUDGET:
                kw["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
            r = client.messages.create(**kw)
            text = "".join(getattr(b, "text", "") or "" for b in r.content)
            if not text.strip():
                text = (f"API_EMPTY stop_reason={r.stop_reason} blocks="
                        f"{[getattr(b, 'type', '?') for b in r.content]} out_tokens={r.usage.output_tokens}")
            pin, pout = PRICES.get(model, (0.0, 0.0))
            usd = (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
            return text, (r.usage.input_tokens, r.usage.output_tokens), usd
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 3 * 2 ** attempt))
        except anthropic.APIStatusError as e:
            return f"API_ERROR {e.status_code}: {str(e)[:300]}", (0, 0), 0.0
    return f"API_ERROR gave up: {last}", (0, 0), 0.0


def call_openai(model: str, prompt: str, max_tokens: int, base_url: str) -> tuple[str, tuple[int, int], float]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "x"))
    r = client.chat.completions.create(model=model, temperature=0.0, max_tokens=max_tokens,
                                       messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    u = r.usage
    return r.choices[0].message.content or "", (u.prompt_tokens, u.completion_tokens), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent.parent / "bench/tasks.jsonl"))
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--condition", default="diff", choices=["diff", "diff+graph", "diff+run", "full"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-tokens", type=int, default=14000, help="thinking budget (8000) + room for the JSON")
    ap.add_argument("--workers", type=int, default=2, help="keep low: prompts are 20-90k tokens")
    ap.add_argument("--runs", default=str(Path(__file__).parent / "runs.json"), help="precomputed suite results by head_sha")
    ap.add_argument("--table", nargs="*")
    a = ap.parse_args()
    if a.table:
        return table(a.table)

    tasks = [t.to_json() for t in load_tasks(a.tasks)]
    runs = json.loads(Path(a.runs).read_text()) if Path(a.runs).exists() else {}
    label = a.label or f"{a.model.replace('/', '_')}_{a.condition}"
    call = (lambda p: call_anthropic(a.model, p, a.max_tokens)) if a.backend == "anthropic" \
        else (lambda p: call_openai(a.model, p, a.max_tokens, a.base_url))

    def one(t):
        text, usage, usd = call(build_prompt(t, a.condition, runs.get(t["head_sha"])))
        rev, ok = parse_review(text, usage, usd)
        return t["id"], rev, ok, text

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, tasks))
    reviews = {tid: rev for tid, rev, _, _ in results}
    unparsable = sum(1 for _, _, ok, _ in results if not ok)
    api_errors = sum(1 for _, _, _, raw in results if raw.startswith("API_ERROR") or raw.startswith("API_EMPTY"))
    if api_errors:
        print(f"WARNING: {api_errors} API errors (scored as escalate); see 'raw' in the results file", flush=True)
    scores = by_split(load_tasks(a.tasks), reviews)
    out = {"label": label, "model": a.model, "condition": a.condition, "n": len(tasks), "unparsable": unparsable,
           "wall_s": round(time.time() - t0, 1), "scores": scores,
           "reviews": [{"task_id": tid, "decision": r.decision, "usd": r.usd,
                        "comments": [c.__dict__ for c in r.comments], "raw": raw[:4000]} for tid, r, _, raw in results]}
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{label}.json"
    p.write_text(json.dumps(out, indent=1))
    s = scores["all"]
    print(f"{label}: bugs_approved={s['bugs_approved']}/{s['n_bug']}  auto_ship={s['auto_ship_rate']:.2f}  "
          f"escalate={s['escalation_rate']:.2f}  unparsable={unparsable}  ${s['usd_per_pr']:.4f}/PR  -> {p}")


def table(paths):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    rows.sort(key=lambda r: (r["scores"]["all"]["bugs_approved"], -(r["scores"]["all"]["auto_ship_rate"] or 0)))
    print(f"{'label':<22}{'cond':<11}{'bugs appr':>10}{'auto-ship':>10}{'escalate':>9}{'blk prec':>9}{'local':>7}{'$/PR':>8}")
    for r in rows:
        s = r["scores"]["all"]
        f = lambda v: "   —" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{r['label']:<22}{r['condition']:<11}{s['bugs_approved']:>6}/{s['n_bug']:<3}{f(s['auto_ship_rate']):>10}"
              f"{f(s['escalation_rate']):>9}{f(s['block_precision']):>9}{f(s['localization']):>7}{s['usd_per_pr'] or 0:>8.4f}")


if __name__ == "__main__":
    main()
