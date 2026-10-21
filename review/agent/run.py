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


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_review(text: str, usage: tuple[int, int], usd: float) -> tuple[Review, bool]:
    # thinking-mode students (phase 3b v4) reason inside <think>...</think> first; braces in there must not
    # be swallowed by the greedy JSON match
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text:                      # unterminated think block: the cap cut it off, nothing to parse
        text = text.split("<think>", 1)[0]
    m = _JSON_RE.search(text)
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


THINKING_BUDGET = int(os.environ.get("REVIEW_THINKING_BUDGET", "8000"))   # 0 disables the thinking params
THINKING_EFFORT = os.environ.get("REVIEW_THINKING_EFFORT", "medium")      # 5.5 models: adaptive thinking + effort

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
                kw["thinking"] = {"type": "adaptive"}
                kw["output_config"] = {"effort": THINKING_EFFORT}
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


_REP_RE = re.compile(r"^(.*)_r(\d+)$")


def table(paths):
    """One row per label; `<label>_r<i>` repeats collapse to mean ± half-range. Columns are rates over
    the bench; 'strong' = fraction of blocked bug PRs whose probe reproduced the oracle."""
    rows = [json.loads(Path(p).read_text()) for p in paths]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        m = _REP_RE.match(r["label"])
        groups.setdefault(m.group(1) if m else r["label"], []).append(r)

    def agg(key, rs, denom=None):
        vals = []
        for r in rs:
            s = r["scores"]["all"]
            v = s[key] if denom is None else (s[key] / s[denom] if s[denom] else None)
            if v is not None:
                vals.append(v)
        if not vals:
            return None, None
        return sum(vals) / len(vals), ((max(vals) - min(vals)) / 2 if len(vals) > 1 else None)

    out = []
    for label, rs in groups.items():
        ba, ba_sp = agg("bugs_approved", rs, "n_bug")
        out.append((label, rs[0]["condition"], len(rs), rs[0]["scores"]["all"]["n_bug"],
                    ba, ba_sp, *agg("auto_ship_rate", rs), *agg("escalation_rate", rs),
                    *agg("block_precision", rs), *agg("localization", rs), *agg("strong_localization", rs),
                    *agg("usd_per_pr", rs)))
    out.sort(key=lambda t: ((t[4] or 0), -(t[6] or 0)))
    f = lambda v, sp=None: ("   —" if v is None else f"{v:.2f}") + (f"±{sp:.2f}" if sp else "     ")  # noqa: E731
    print(f"{'label':<20}{'cond':<11}{'runs':>4}{'bugs appr':>12}{'auto-ship':>12}{'escalate':>12}{'blk prec':>12}{'local':>12}{'strong':>12}{'$/PR':>8}")
    for (label, cond, k, nb, ba, bas, au, aus, es, ess, bp, bps, lo, los, st, sts, usd, _) in out:
        print(f"{label:<20}{cond:<11}{k:>4}{f(ba, bas):>12}{f(au, aus):>12}{f(es, ess):>12}{f(bp, bps):>12}{f(lo, los):>12}{f(st, sts):>12}{(usd or 0):>8.3f}")
    # per-source view (real / op / llm): the synthetic-vs-real and operator-vs-LLM gaps, where recorded
    srcs = sorted({k for r in rows for k in r["scores"] if k.startswith("source:")})
    if srcs:
        print(f"\n{'label':<20}" + "".join(f"{k[7:] + ' bugs appr':>18}{k[7:] + ' blocked':>14}" for k in srcs))
        for label, rs in groups.items():
            cells = []
            for k in srcs:
                vals = [r["scores"][k] for r in rs if k in r["scores"]]
                if not vals:
                    cells += [f"{'—':>18}", f"{'—':>14}"]; continue
                ba = sum(v["bugs_approved"] for v in vals) / max(sum(v["n_bug"] for v in vals), 1)
                bl = sum(v.get("bugs_blocked", 0) for v in vals) / max(sum(v["n_bug"] for v in vals), 1)
                cells += [f"{ba:>14.2f} n={vals[0]['n_bug']:<2d}", f"{bl:>14.2f}"]
            print(f"{label:<20}" + "".join(cells))


if __name__ == "__main__":
    main()
