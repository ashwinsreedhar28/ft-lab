"""The review agent: a loop, not a prompt.

    python -m review.agent.loop --model claude-sonnet-5-5 --label sonnet_loop [--max-probes 2]

Per PR:
  1. gather: diff (code files whole), two-hop neighborhood, and the suite result at the PR head
     (run in the task's Docker image; cached in review/agent/runs.json by head_sha)
  2. decide: the model answers with the same JSON as run.py, plus an optional "probe": pytest
     source it wants executed before committing to a decision
  3. probe: if present, the harness writes the test into the container at head_sha and runs it,
     then (for scoring only, never shown to the model) also runs it at the fix/base to know if the
     probe reproduces the oracle. The result goes back to the model, which decides again.
  4. up to --max-probes rounds, then a final decision is forced (no probe field allowed)

Docker is required (same images as the bench). Run from the Mac terminal.
Scoring is review/bench/score.py; "strong localization" uses the probe verdicts gathered here.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from review.agent.run import PRICES, SYSTEM, THINKING_BUDGET, build_prompt, parse_review
from review.bench.score import by_split, load_tasks

RESULTS = Path(__file__).parent / "results"
RUNS = Path(__file__).parent / "runs.json"

LOOP_SYSTEM = SYSTEM + (
    "\n\nYou may run one probe test before deciding. To do so, include a top-level field "
    '"probe": {"path": "tests/test_probe_review.py", "source": "<pytest file>"} in your JSON. '
    "The harness runs it at the PR head and returns the output; then you decide again. "
    "A probe that fails on the PR is evidence for block. If you need no probe, omit the field. "
    "On the final round the probe field is ignored."
)


def docker_run(image: str, script: str, files: dict[str, str] | None = None, timeout: int = 1200) -> tuple[int, str]:
    name = f"rb-{int(time.time() * 1000) % 10**9}"
    subprocess.run(["docker", "create", "--name", name, "--network", "none", "--cpus=2", "--memory=4g",
                    image, "bash", "-lc", script], check=True, capture_output=True)
    try:
        for path, text in (files or {}).items():
            import tempfile
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as f:
                f.write(text); f.flush()
            subprocess.run(["docker", "cp", f.name, f"{name}:{path}"], check=True, capture_output=True)
        p = subprocess.run(["docker", "start", "-a", name], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-6000:]
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return 124, "TIMEOUT"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def suite_at(task: dict, sha: str, subdir: str) -> str:
    """Full suite result (last 40 lines) at a commit, for the diff+run context."""
    rc, out = docker_run(task["image"], f"cd /testbed && git checkout -q {sha} && cd {subdir} && {task['test_cmd']} -q 2>&1 | tail -n 40; echo EXIT=${{PIPESTATUS[0]}}")
    return out


def run_probe(task: dict, sha: str, subdir: str, path: str, source: str) -> tuple[str, str]:
    """Returns (status passed|failed|error, output tail)."""
    full = f"/testbed/{subdir}/{path}".replace("/./", "/")
    rc, out = docker_run(task["image"],
                         f"cd /testbed && git checkout -q {sha} && mkdir -p $(dirname {full}) && cd {subdir} && {task['test_cmd']} {path} 2>&1 | tail -n 40; echo EXIT=${{PIPESTATUS[0]}}",
                         files={full: source})
    m = re.search(r"EXIT=(\d+)", out)
    code = int(m.group(1)) if m else rc
    return ("passed" if code == 0 else "failed" if code == 1 else "error"), out


def call(model: str, system: str, messages: list[dict], max_tokens: int):
    import anthropic

    client = anthropic.Anthropic(max_retries=0)
    for attempt in range(7):
        try:
            kw = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
            if THINKING_BUDGET:
                kw["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
            r = client.messages.create(**kw)
            text = "".join(getattr(b, "text", "") or "" for b in r.content)
            pin, pout = PRICES.get(model, (0.0, 0.0))
            return text, (r.usage.input_tokens, r.usage.output_tokens), (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError):
            time.sleep(min(60, 3 * 2 ** attempt))
    return "API_ERROR gave up", (0, 0), 0.0


def review_one(task: dict, model: str, max_probes: int, runs: dict, max_tokens: int):
    subdir = task.get("subdir") or "."
    image_ok = bool(task.get("image"))
    if image_ok and task["head_sha"] not in runs:
        runs[task["head_sha"]] = suite_at(task, task["head_sha"], subdir)
        RUNS.write_text(json.dumps(runs, indent=1))
    prompt = build_prompt(task, "full" if image_ok else "diff+graph", runs.get(task["head_sha"]))
    messages = [{"role": "user", "content": prompt}]
    usd_total, tok_in, tok_out, probes = 0.0, 0, 0, []
    text = ""
    for round_ in range(max_probes + 1):
        final = round_ == max_probes
        if final:
            messages.append({"role": "user", "content": "Final round: decide now. The probe field is ignored."})
        text, (ti, to), usd = call(model, LOOP_SYSTEM, messages, max_tokens)
        usd_total += usd; tok_in += ti; tok_out += to
        messages.append({"role": "assistant", "content": text})
        m = re.search(r"\{.*\}", text, re.S)
        probe = None
        try:
            probe = json.loads(m.group(0)).get("probe") if m else None
        except json.JSONDecodeError:
            probe = None
        if final or not probe or not image_ok or not isinstance(probe, dict) or "source" not in probe:
            break
        path = probe.get("path") or "tests/test_probe_review.py"
        status, out = run_probe(task, task["head_sha"], subdir, path, probe["source"])
        # scoring-only: does the probe reproduce the oracle? (fails on PR, passes on the fixed code)
        strong = None
        if task["kind"] == "bug" and status == "failed":
            fixed_sha = task.get("fix_sha")
            if fixed_sha:
                s2, _ = run_probe(task, fixed_sha, subdir, path, probe["source"])
                strong = s2 == "passed"
        probes.append({"round": round_, "path": path, "status": status, "strong": strong, "tail": out[-1500:]})
        messages.append({"role": "user", "content": f"Probe result at the PR head: {status}\n```\n{out[-3000:]}\n```\nDecide again (you may probe once more if rounds remain)."})
    rev, ok = parse_review(text, (tok_in, tok_out), usd_total)
    return rev, ok, text, probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent.parent / "bench/tasks.jsonl"))
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-probes", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=14000)
    a = ap.parse_args()
    label = a.label or f"{a.model}_loop"
    tasks = [t.to_json() for t in load_tasks(a.tasks)]
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    reviews, strong, out_rows = {}, {}, []
    t0 = time.time()
    for t in tasks:
        rev, ok, raw, probes = review_one(t, a.model, a.max_probes, runs, a.max_tokens)
        reviews[t["id"]] = rev
        if any(p.get("strong") for p in probes):
            strong[t["id"]] = True
        out_rows.append({"task_id": t["id"], "decision": rev.decision, "usd": rev.usd, "probes": probes,
                         "comments": [c.__dict__ for c in rev.comments], "parsed": ok, "raw": raw[:4000]})
        print(f"{t['id']:45s} {rev.decision:9s} probes={len(probes)} ${rev.usd:.3f}", flush=True)
    scores = by_split(load_tasks(a.tasks), reviews, strong)
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{label}.json"
    p.write_text(json.dumps({"label": label, "model": a.model, "condition": "loop", "n": len(tasks),
                             "unparsable": sum(1 for r in out_rows if not r["parsed"]), "wall_s": round(time.time() - t0, 1),
                             "scores": scores, "reviews": out_rows}, indent=1))
    s = scores["all"]
    print(f"{label}: bugs_approved={s['bugs_approved']}/{s['n_bug']} auto_ship={s['auto_ship_rate']:.2f} "
          f"escalate={s['escalation_rate']:.2f} strong_loc={s['strong_localization']} ${s['usd_per_pr']:.3f}/PR -> {p}")


if __name__ == "__main__":
    main()
