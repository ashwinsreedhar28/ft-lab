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

from review.agent.run import PRICES, SYSTEM, THINKING_BUDGET, THINKING_EFFORT, build_prompt, parse_review
from review.bench.score import by_split, provenance, load_tasks

RESULTS = Path(__file__).parent / "results"
RUNS = Path(__file__).parent / "runs.json"

LOOP_SYSTEM = SYSTEM + (
    "\n\nEvidence rules of this gate:\n"
    "- The existing test suite being green at the PR head is NOT evidence that the change is correct. "
    "Every bug that ever shipped was merged on a green suite. Only a test that exercises the changed "
    "code counts.\n"
    "- You cannot approve without evidence: an approve is accepted only if you ran a probe test that "
    "passed and that calls at least one of the changed functions by name. Otherwise the harness bounces "
    "the approve back to you (or escalates on the final round).\n"
    "- To run a probe, include a top-level field "
    '"probe": {"path": "tests/test_probe_review.py", "source": "<pytest file>", "hypothesis": "<one line>"} '
    "in your JSON. The harness runs it at the PR head and returns the output; then you decide again.\n"
    "- A good probe is a falsifiable hypothesis about the change: state the specific way the change could "
    "be wrong (an off-by-one, a missing guard, a changed default, an unhandled branch) and write the test "
    "so that it FAILS if that hypothesis is true. 'It runs without error' is not a hypothesis. Prefer the "
    "edge the diff makes you uneasy about: empty input, boundary index, the branch the PR added, the "
    "old caller that still expects the old behavior.\n"
    "- A probe that fails on the PR is evidence for block (cite it). A probe that passes rules out one "
    "hypothesis; approve only when your remaining doubt about the changed code is gone.\n"
    "- Blocking or escalating needs no probe. On the final round the probe field is ignored.\n"
    "- You have no tools. Do not emit tool-call or <invoke> syntax; the probe is a field in your JSON reply, "
    "and the harness, not you, runs it. Reply with one JSON object and nothing else."
)


def extract_reply(text: str) -> dict:
    """Find the model's JSON object. Tolerates fences, prose, and the hallucinated
    <invoke><parameter name="probe">{...}</parameter></invoke> form Sonnet 5.5 produced in v4
    (which v4 scored as unparsable -> escalate, dropping 4 of 12 bug-PR probes)."""
    objs = []
    dec = json.JSONDecoder()
    i = 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            o, end = dec.raw_decode(text[j:])
            if isinstance(o, dict):
                objs.append(o)
            i = j + end
        except json.JSONDecodeError:
            i = j + 1
    out = {}
    for o in objs:
        if "decision" in o and "decision" not in out:
            out.update(o)
    if "probe" not in out:
        for o in objs:
            if isinstance(o.get("probe"), dict):
                out["probe"] = o["probe"]; break
            if "source" in o and "path" in o:        # bare probe object (invoke-wrapped form)
                out["probe"] = o; break
    return out


def changed_def_names(task: dict) -> set[str]:
    """Parsed from the neighborhood text ('Changed definitions: file:name, ...')."""
    m = re.search(r"Changed definitions: (.*)", task.get("neighborhood", ""))
    if not m:
        return set()
    return {x.split(":")[-1].strip() for x in m.group(1).split(",") if ":" in x}


def uncovered_defs(task: dict) -> list[str]:
    """Changed defs whose name appears in no test file of the diff's neighborhood listing."""
    names = changed_def_names(task)
    m = re.search(r"Tests importing the changed modules: (.*)", task.get("neighborhood", ""))
    tests = m.group(1) if m else ""
    # we only know test *files*, not their contents, from the neighborhood; treat 'no test file at all'
    # as uncovered and otherwise leave coverage to the probe rule
    return sorted(names) if not tests else []


def _ranges(lines: list[int]) -> str:
    out, start, prev = [], None, None
    for l in sorted(lines):
        if start is None:
            start = prev = l
        elif l == prev + 1:
            prev = l
        else:
            out.append(f"{start}-{prev}" if prev != start else str(start)); start = prev = l
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else str(start))
    return ",".join(out)


def probe_targets_uncovered(task: dict, probe_cov: dict | None) -> bool:
    """probe_cov: {file: set(lines)} executed by the probe (from coverage json), or None if unavailable."""
    if not task.get("uncovered") or probe_cov is None:
        return True   # no coverage data: fall back to the name rule
    for u in task["uncovered"]:
        if set(u["lines"]) & probe_cov.get(u["file"], set()):
            return True
    return False


def probe_is_specific(source: str, task: dict) -> bool:
    names = changed_def_names(task)
    return any(re.search(rf"\b{re.escape(n)}\b", source) for n in names if len(n) > 2)


def docker_run(image: str, script: str, files: dict[str, str] | None = None, timeout: int = 1200, network: str = "none",
               mounts: list[str] | None = None) -> tuple[int, str]:
    name = f"rb-{int(time.time() * 1000) % 10**9}"
    vols = [x for m in (mounts or []) for x in ("-v", m)]
    subprocess.run(["docker", "create", "--name", name, "--network", network, "--cpus=2", "--memory=4g", *vols,
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


def prelude(task: dict, sha: str) -> str:
    """Shell prefix that leaves the shell in /testbed at `sha`. shwin-bench images bake /testbed in;
    mined tasks (assemble.py) carry setup_cmd that clones the mounted /repo and sets PYTHONPATH."""
    if task.get("setup_cmd"):
        return f"{task['setup_cmd']} && git checkout -q {sha}"
    return f"cd /testbed && git checkout -q {sha}"


def container_kw(task: dict, network: str = "none") -> dict:
    """docker_run kwargs a task needs: its clone mounted ro at /repo, and its network if it needs one."""
    kw = {"network": task.get("network") if task.get("network") not in (None, "none") else network}
    if task.get("repo_mount"):
        kw["mounts"] = [f"{task['repo_mount']}:/repo:ro"]
    return kw


def suite_at(task: dict, sha: str, subdir: str) -> str:
    """Full suite result (last 40 lines) at a commit, for the diff+run context."""
    rc, out = docker_run(task["image"], f"{prelude(task, sha)} && cd {subdir} && {task['test_cmd']} -q 2>&1 | tail -n 40; echo EXIT=${{PIPESTATUS[0]}}",
                         **container_kw(task))
    return out


def run_probe(task: dict, sha: str, subdir: str, path: str, source: str, with_cov: bool = False):
    """Returns (status passed|failed|error, output tail, probe_cov|None)."""
    full = f"/testbed/{subdir}/{path}".replace("/./", "/")
    # mined tasks: /testbed does not exist until setup_cmd clones it, so the probe is staged in /tmp and copied
    staged = bool(task.get("setup_cmd"))
    stage = "/tmp/probe_staged.py"
    place = f"mkdir -p $(dirname {full}) && cp {stage} {full}" if staged else f"mkdir -p $(dirname {full})"
    if with_cov:
        cmd = (f"{prelude(task, sha)} && {place} && pip -q install coverage >/dev/null 2>&1; cd {subdir} && "
               f"python -m coverage run -m pytest -o addopts= -p no:cacheprovider -q {path} 2>&1 | tail -n 40; echo EXIT=${{PIPESTATUS[0]}}; "
               f"python -m coverage json -o /tmp/pc.json >/dev/null 2>&1 && echo COVJSON && cat /tmp/pc.json")
    else:
        cmd = f"{prelude(task, sha)} && {place} && cd {subdir} && {task['test_cmd']} {path} 2>&1 | tail -n 40; echo EXIT=${{PIPESTATUS[0]}}"
    rc, out = docker_run(task["image"], cmd, files={stage if staged else full: source},
                         **container_kw(task, network="bridge" if with_cov else "none"))
    m = re.search(r"EXIT=(\d+)", out)
    code = int(m.group(1)) if m else rc
    status = "passed" if code == 0 else "failed" if code == 1 else "error"
    cov = None
    if with_cov and "COVJSON" in out:
        try:
            j = json.loads(out.split("COVJSON", 1)[1].strip())
            cov = {}
            for fpath, fdata in j.get("files", {}).items():
                rel = fpath[len("/testbed/"):] if fpath.startswith("/testbed/") else fpath
                if subdir != "." and not rel.startswith(subdir):
                    rel = f"{subdir}/{rel}"
                cov[rel] = set(fdata.get("executed_lines", []))
        except (ValueError, json.JSONDecodeError):
            cov = None
    return status, out.split("COVJSON")[0], cov


def call(model: str, system: str, messages: list[dict], max_tokens: int):
    import anthropic

    client = anthropic.Anthropic(max_retries=0)
    for attempt in range(7):
        try:
            kw = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
            if THINKING_BUDGET:
                kw["thinking"] = {"type": "adaptive"}
                kw["output_config"] = {"effort": THINKING_EFFORT}
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
    run_txt = runs.get(task["head_sha"])
    if run_txt:
        run_txt = ("(reminder: green here means the pre-existing tests still pass; it does not test the change)\n" + run_txt)
    prompt = build_prompt(task, "full" if image_ok else "diff+graph", run_txt)
    unc = uncovered_defs(task)
    if unc:
        prompt += "\n\nNo existing test file references these changed definitions: " + ", ".join(unc[:20])
    if task.get("uncovered"):
        lines = "; ".join(f"{u['file']}: {_ranges(u['lines'])}" for u in task["uncovered"][:12])
        prompt += ("\n\nChanged lines that NO existing test executes (measured with coverage at the PR head). "
                   "A bug on these lines would ship on a green suite. Probes that count as evidence for approve "
                   "must execute at least one of these lines:\n" + lines)
    messages = [{"role": "user", "content": prompt}]
    evidence = False   # a specific, passing probe exists
    usd_total, tok_in, tok_out, probes = 0.0, 0, 0, []
    text = ""
    rounds, api_s, docker_s = [], 0.0, 0.0   # audit: why no probe, and where the wall-clock went
    format_retried = False
    for round_ in range(max_probes + 1):
        final = round_ == max_probes
        if final:
            messages.append({"role": "user", "content": "Final round: decide now. The probe field is ignored."})
        t_api = time.time()
        text, (ti, to), usd = call(model, LOOP_SYSTEM, messages, max_tokens)
        api_s += time.time() - t_api
        usd_total += usd; tok_in += ti; tok_out += to
        messages.append({"role": "assistant", "content": text})
        obj = extract_reply(text)
        probe, decision = obj.get("probe"), obj.get("decision")
        has_probe = image_ok and isinstance(probe, dict) and "source" in probe
        rounds.append({"round": round_, "decision": decision, "probe_requested": isinstance(probe, dict),
                       "probe_runnable": has_probe, "parsed": bool(obj), "invoke_syntax": "<invoke" in text})
        if decision is None and not has_probe and not final and not format_retried:
            # a format failure is not a decision: one free retry, then it counts
            format_retried = True
            messages.append({"role": "user", "content": "Your reply had no \"decision\" field and no runnable "
                             "\"probe\" (path + source). You have no tools; do not use <invoke> syntax. "
                             "Reply with the single JSON object described in the instructions."})
            continue
        if decision is None and has_probe:
            decision = "pending"   # a probe request without a decision is fine mid-loop
        if decision == "approve" and not evidence and not has_probe:
            if final:
                text = json.dumps({**obj, "decision": "escalate", "harness_note": "approve without a passing specific probe -> escalate"})
                break
            messages.append({"role": "user", "content": "Approve rejected: no passing probe that exercises the changed code. "
                             "Write one now (probe field) or choose block/escalate."})
            continue
        if final or not has_probe:
            break
        path = probe.get("path") or "tests/test_probe_review.py"
        t_dk = time.time()
        status, out, pcov = run_probe(task, task["head_sha"], subdir, path, probe["source"], with_cov=bool(task.get("uncovered")))
        # scoring-only: does the probe reproduce the oracle? (fails on PR, passes on the fixed code)
        strong = None
        if task["kind"] == "bug" and status == "failed":
            fixed_sha = task.get("fix_sha")
            if fixed_sha:
                s2, _, _ = run_probe(task, fixed_sha, subdir, path, probe["source"])
                strong = s2 == "passed"
        docker_s += time.time() - t_dk
        specific = probe_is_specific(probe["source"], task)
        targets = probe_targets_uncovered(task, pcov)
        if status == "passed" and specific and targets:
            evidence = True
        probes.append({"round": round_, "path": path, "status": status, "strong": strong, "specific": specific,
                       "targets_uncovered": targets, "hypothesis": probe.get("hypothesis", ""),
                       "source": probe["source"][:6000], "tail": out[-1500:]})
        note = "" if specific else " (note: this probe does not call any changed function by name, so it does not count as evidence for approve)"
        if specific and not targets:
            note = " (note: this probe executed none of the changed lines the suite leaves uncovered, so it does not count as evidence for approve)"
        messages.append({"role": "user", "content": f"Probe result at the PR head: {status}{note}\n```\n{out[-3000:]}\n```\nDecide again (you may probe once more if rounds remain)."})
    rev, ok = parse_review(text, (tok_in, tok_out), usd_total)
    audit = {"rounds": rounds, "api_s": round(api_s, 1), "docker_s": round(docker_s, 1),
             "no_probe_reason": None if probes else no_probe_reason(rounds, image_ok)}
    return rev, ok, text, probes, audit


def no_probe_reason(rounds: list[dict], image_ok: bool) -> str:
    """Why a review ended with zero probes run: 'the model never asked' vs a harness cause."""
    if not image_ok:
        return "no_image"
    if not rounds or not any(r["parsed"] for r in rounds):
        return "unparsable"
    if any(r["probe_requested"] and not r["probe_runnable"] for r in rounds):
        return "probe_malformed"          # asked for a probe but gave no source
    if any(r.get("invoke_syntax") and r["decision"] is None for r in rounds):
        return "format_failure"           # tool-call syntax, no decision, retry did not recover
    return f"model_decided_round0:{rounds[0]['decision']}"  # chose without asking for evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent.parent / "bench/tasks.jsonl"))
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-probes", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=14000)
    ap.add_argument("--repeat", type=int, default=1, help="run N times, writing <label>_r<i>.json; --table collapses them")
    a = ap.parse_args()
    base_label = a.label or f"{a.model}_loop"
    for rep in range(a.repeat):
        run_once(a, base_label if a.repeat == 1 else f"{base_label}_r{rep}")


def run_once(a, label):
    tasks = [t.to_json() for t in load_tasks(a.tasks)]
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    reviews, strong, out_rows = {}, {}, []
    t0 = time.time()
    for t in tasks:
        rev, ok, raw, probes, audit = review_one(t, a.model, a.max_probes, runs, a.max_tokens)
        reviews[t["id"]] = rev
        if any(p.get("strong") for p in probes):
            strong[t["id"]] = True
        out_rows.append({"task_id": t["id"], "decision": rev.decision, "usd": rev.usd, "probes": probes, **audit,
                         "comments": [c.__dict__ for c in rev.comments], "parsed": ok, "raw": raw[:4000]})
        why = f" ({audit['no_probe_reason']})" if audit["no_probe_reason"] else ""
        print(f"{t['id']:45s} {rev.decision:9s} probes={len(probes)}{why} ${rev.usd:.3f} "
              f"api={audit['api_s']:.0f}s docker={audit['docker_s']:.0f}s", flush=True)
    scores = by_split(load_tasks(a.tasks), reviews, strong)
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{label}.json"
    p.write_text(json.dumps({"label": label, "model": a.model, "condition": "loop", "n": len(tasks),
                             "provenance": provenance(a.tasks, {t.get("image") for t in tasks if t.get("image")}),
                             "unparsable": sum(1 for r in out_rows if not r["parsed"]), "wall_s": round(time.time() - t0, 1),
                             "api_s": round(sum(r["api_s"] for r in out_rows), 1), "docker_s": round(sum(r["docker_s"] for r in out_rows), 1),
                             "no_probe_reasons": {k: sum(1 for r in out_rows if r["no_probe_reason"] == k)
                                                  for k in sorted({r["no_probe_reason"] for r in out_rows if r["no_probe_reason"]})},
                             "scores": scores, "reviews": out_rows}, indent=1))
    s = scores["all"]
    print(f"{label}: bugs_approved={s['bugs_approved']}/{s['n_bug']} auto_ship={s['auto_ship_rate']:.2f} "
          f"escalate={s['escalation_rate']:.2f} strong_loc={s['strong_localization']} ${s['usd_per_pr']:.3f}/PR -> {p}")


if __name__ == "__main__":
    main()
