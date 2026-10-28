"""Review harness v6 (Oct 28): gate model x review tools x evidence policy.

    # tools only: no gate, no API (unless reviewer is in --tools); per-tool fire and gold-hunk hit rates
    python -m review.agent.harness --tools-only --tasks review/bench/tasks.jsonl --tools static,tests --label tools_shwin

    # a gate row of the matrix
    python -m review.agent.harness --tasks review/bench/tasks.jsonl --gate claude-sonnet-5-5 \
        --tools static,tests,reviewer --policy free --label h6_sonnet_all_free

Production repos usually already run review tooling, so the tool is an input with an adapter boundary
(review/tools) and the gate is the model reading diff + context + findings. v5 (review/agent/loop.py) is left untouched so its results stay reproducible.

Policies (comma-separated, measured not assumed):
  free                       no constraint
  block-needs-finding        a block must cite a finding id (F#) or a probe that failed; otherwise bounced once,
                             then escalated
  approve-needs-clean-tools  an approve is rejected while any tool reports a high-severity finding
  approve-needs-probe        the v4b/v5 rule: approve needs a passing probe naming a changed function

Probes (the model writes a pytest file, the harness runs it in Docker) are OFF by default: across ~100 bug-PR runs
on both benches no probe ever reproduced an oracle, and they are the most expensive step.

Gates: claude-* via the Anthropic API. mlx:<repo> (local Qwen3-8B on the Mac) is the next adapter (week of Nov 3).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from review.agent.loop import (RESULTS, RUNS, call, extract_reply, probe_is_specific, probe_targets_uncovered,
                               run_probe)
from review.agent.run import build_prompt, parse_review
from review.bench.score import by_split, load_tasks, provenance
from review.tools import ToolRun, gather, number, render, tool_stats

HARNESS_VERSION = 6
POLICIES = ("free", "block-needs-finding", "approve-needs-clean-tools", "approve-needs-probe")

BASE = (
    "You are the code review gate for a production repository. Your approval merges and deploys the change "
    "with no further human review.\n"
    "  approve  - you are confident the change is correct and safe to ship.\n"
    "  block    - you found a concrete defect in the change; cite where it is.\n"
    "  escalate - you cannot establish safety from the evidence you have; a human must look.\n"
    "Approving a defective change is the worst outcome. Blocking a correct change costs an engineer's time. "
    "Escalating is cheap but useless if done for everything.\n"
)
TOOLS_LINE = ("You get the diff, mechanically extracted context, and the output of the repository's review tools "
              "(findings F1..Fn). Tools miss real bugs and flag correct code: weigh them, do not obey them.\n")
NO_TOOLS_LINE = "You get the diff and mechanically extracted context. No review tools ran on this PR.\n"
POLICY_LINES = {
    "block-needs-finding": "- A block is accepted only if \"evidence\" cites at least one finding id{probe}. "
                           "An uncited block is sent back to you.\n",
    "approve-needs-clean-tools": "- An approve is accepted only if no tool reported a high-severity finding.\n",
    "approve-needs-probe": "- An approve is accepted only if you ran a probe that passed and that calls at least one "
                           "of the changed functions by name.\n",
}
PROBE_LINES = (
    "- You may ask the harness to run a test before deciding: add a top-level field "
    '"probe": {"path": "tests/test_probe_review.py", "source": "<pytest file>", "hypothesis": "<one line>"}. '
    "It runs at the PR head and the output comes back to you. Write it to FAIL if your suspected defect is real.\n"
)
FORMAT = (
    "Respond with exactly one JSON object and nothing else:\n"
    '{"decision": "approve|block|escalate", "evidence": ["F1"], "comments": [{"file": "...", "line_start": N, '
    '"line_end": N, "body": "..."}]}\n'
    '"evidence" lists the finding ids your decision relies on (empty list if none). For block, include one comment '
    "at the defect with a body under 40 words. Do not include proposed tests in comments. You have no tools; do "
    "not emit tool-call or <invoke> syntax."
)


def system_prompt(tools: list[str], policy: list[str], probes: int) -> str:
    rules = "".join(POLICY_LINES[p].format(probe=" or a probe you ran that failed" if probes else "")
                    for p in policy if p in POLICY_LINES)
    if probes:
        rules += PROBE_LINES
    return (BASE + (TOOLS_LINE if tools else NO_TOOLS_LINE)
            + (f"\nRules of this gate:\n{rules}" if rules else "") + "\n" + FORMAT)


def cited(obj: dict, valid_ids: set[str]) -> list[str]:
    """Finding ids the reply relies on: the evidence field plus any F# mentioned in comment bodies."""
    ids = [str(x).strip() for x in (obj.get("evidence") or []) if isinstance(x, (str, int))]
    for c in obj.get("comments") or []:
        if isinstance(c, dict):
            ids += re.findall(r"\bF\d+\b", str(c.get("body", "")))
    return sorted({i for i in ids if i in valid_ids}, key=lambda s: int(s[1:]))


def apply_policy(decision: str | None, cites: list[str], findings, policy: list[str],
                 probe_failed: bool = False, probe_evidence: bool = False, probes_enabled: bool = False) -> str | None:
    """Pure: None if the decision stands, else the reason it is rejected (shown to the model on a bounce)."""
    if decision == "block" and "block-needs-finding" in policy and not cites and not probe_failed:
        return ("Block rejected: this gate requires a block to cite at least one finding id in \"evidence\""
                + (" or a probe that failed" if probes_enabled else "")
                + ". Cite the finding that shows the defect, or choose approve or escalate.")
    if decision == "approve" and "approve-needs-clean-tools" in policy:
        high = [f.id for f in findings if f.severity == "high"]
        if high:
            return (f"Approve rejected: tools reported high-severity finding(s) {', '.join(high)}. "
                    "Block (citing the finding) or escalate.")
    if decision == "approve" and "approve-needs-probe" in policy and not probe_evidence:
        return "Approve rejected: no passing probe that calls a changed function. Write one (probe field) or block/escalate."
    return None


def gate_call(gate: str, system: str, messages: list[dict], max_tokens: int):
    if gate.startswith("claude"):
        return call(gate, system, messages, max_tokens)
    if gate.startswith("mlx:"):
        raise NotImplementedError("MLX gate lands week of Nov 3 (review/agent/mlx_gate.py)")
    raise ValueError(f"unknown gate {gate!r}")


def review_one(task: dict, gate: str, runs: list[ToolRun], tools: list[str], policy: list[str],
               max_probes: int, max_tokens: int):
    findings = number(runs)
    valid = {f.id for f in findings}
    prompt = build_prompt(task, "diff+graph", None)
    block = render(runs)
    if block:
        prompt += "\n\n" + block
    system = system_prompt(tools, policy, max_probes)
    messages = [{"role": "user", "content": prompt}]
    subdir = task.get("subdir") or "."
    image_ok = bool(task.get("image"))
    probes, rounds, bounces = [], [], []
    probe_failed, probe_evidence, format_retried = False, False, False
    usd, tok_in, tok_out, api_s, docker_s = 0.0, 0, 0, 0.0, 0.0
    text, obj = "", {}
    max_rounds = max_probes + 2          # probes + one decision + one policy bounce / format retry
    for r in range(max_rounds):
        final = r == max_rounds - 1
        if final and r > 0:
            nudge = "Final round: decide now." + (" The probe field is ignored." if max_probes else "")
            if messages[-1]["role"] == "user":      # after a bounce or retry: one user turn, not two in a row
                messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n\n" + nudge}
            else:
                messages.append({"role": "user", "content": nudge})
        t0 = time.time()
        text, (ti, to), u = gate_call(gate, system, messages, max_tokens)
        api_s += time.time() - t0
        usd += u; tok_in += ti; tok_out += to
        messages.append({"role": "assistant", "content": text})
        obj = extract_reply(text)
        decision = obj.get("decision")
        probe = obj.get("probe") if max_probes else None
        runnable = image_ok and isinstance(probe, dict) and "source" in probe and len(probes) < max_probes and not final
        rounds.append({"round": r, "decision": decision, "probe": bool(runnable), "parsed": bool(obj)})
        if decision is None and not runnable:
            if not final and not format_retried:
                format_retried = True
                messages.append({"role": "user", "content": "Your reply had no \"decision\" field. Reply with the single "
                                 "JSON object described in the instructions."})
                continue
            break
        if runnable:
            path = probe.get("path") or "tests/test_probe_review.py"
            t1 = time.time()
            status, out, pcov = run_probe(task, task["head_sha"], subdir, path, probe["source"], with_cov=bool(task.get("uncovered")))
            strong = None
            if task["kind"] == "bug" and status == "failed" and task.get("fix_sha"):
                s2, _, _ = run_probe(task, task["fix_sha"], subdir, path, probe["source"])
                strong = s2 == "passed"
            docker_s += time.time() - t1
            specific = probe_is_specific(probe["source"], task)
            probe_failed |= status == "failed"
            probe_evidence |= status == "passed" and specific and probe_targets_uncovered(task, pcov)
            probes.append({"round": r, "path": path, "status": status, "strong": strong, "specific": specific,
                           "hypothesis": probe.get("hypothesis", ""), "source": probe["source"][:6000], "tail": out[-1500:]})
            messages.append({"role": "user", "content": f"Probe result at the PR head: {status}\n```\n{out[-3000:]}\n```\nDecide again."})
            continue
        reason = apply_policy(decision, cited(obj, valid), findings, policy, probe_failed, probe_evidence, max_probes > 0)
        if reason is None:
            break
        bounces.append({"round": r, "decision": decision, "reason": reason[:160]})
        if final:
            text = json.dumps({**obj, "decision": "escalate", "harness_note": f"policy: {reason}"})
            obj = extract_reply(text)
            break
        messages.append({"role": "user", "content": reason})
    rev, ok = parse_review(text, (tok_in, tok_out), usd)
    audit = {"rounds": rounds, "bounces": bounces, "cited": cited(obj, valid), "api_s": round(api_s, 1),
             "docker_s": round(docker_s, 1), "n_findings": {r_.tool: len(r_.findings) for r_ in runs},
             "tools_ok": {r_.tool: r_.ok for r_ in runs}, "tools_usd": round(sum(r_.usd for r_ in runs), 4)}
    return rev, ok, text, probes, audit


def parse_list(s: str, allowed=None) -> list[str]:
    out = [x.strip() for x in (s or "").split(",") if x.strip() and x.strip() != "none"]
    if allowed:
        bad = [x for x in out if x not in allowed]
        if bad:
            raise SystemExit(f"unknown {bad}; choose from {allowed}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent.parent / "bench/tasks.jsonl"))
    ap.add_argument("--gate", default="claude-sonnet-5-5")
    ap.add_argument("--tools", default="static,tests", help="comma list of static,tests,reviewer, or none")
    ap.add_argument("--reviewer-model", default=None)
    ap.add_argument("--policy", default="free", help=f"comma list of {','.join(POLICIES)}")
    ap.add_argument("--probes", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=14000)
    ap.add_argument("--label", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma list of task ids (smoke runs)")
    ap.add_argument("--tools-only", action="store_true", help="run tools and report per-tool stats; no gate calls")
    ap.add_argument("--refresh-tools", action="store_true", help="ignore the findings cache")
    a = ap.parse_args()
    tools = parse_list(a.tools, ("static", "tests", "reviewer"))
    policy = parse_list(a.policy, POLICIES) or ["free"]
    opts = {"reviewer": {"model": a.reviewer_model}} if a.reviewer_model else {}
    tasks = [t.to_json() for t in load_tasks(a.tasks)]
    if a.only:
        keep = set(parse_list(a.only))
        tasks = [t for t in tasks if t["id"] in keep]
    label = a.label or (f"tools_{'+'.join(tools) or 'none'}" if a.tools_only
                        else f"h6_{a.gate.split('-')[1] if a.gate.startswith('claude') else a.gate}_{'+'.join(tools) or 'none'}_{'+'.join(policy)}")

    per_task, t0 = {}, time.time()
    for t in tasks:
        per_task[t["id"]] = gather(t, tools, opts, refresh=a.refresh_tools)
        rs = per_task[t["id"]]
        print(f"{t['id']:45s} {t['kind']:5s} " + "  ".join(f"{r.tool}={'x' if not r.ok else len(r.findings)}" for r in rs), flush=True)
        for r in rs:
            if not r.ok:
                print(f"    {r.tool} did not run: {r.note[:160]}", flush=True)
    stats = tool_stats(per_task, tasks)
    print_tool_stats(stats)
    usd = sum(r.usd for rs in per_task.values() for r in rs)
    if usd:
        print(f"tool API cost this run (cached runs count again): ${usd:.3f}")
    if a.tools_only:
        RESULTS.mkdir(exist_ok=True)
        p = RESULTS / f"{label}.json"
        p.write_text(json.dumps({"label": label, "harness_version": HARNESS_VERSION, "calibration": _calibration(), "tools": tools, "n": len(tasks),
                                 "provenance": provenance(a.tasks), "tool_stats": stats, "wall_s": round(time.time() - t0, 1),
                                 "per_task": {tid: [{"tool": r.tool, "ok": r.ok, "note": r.note,
                                                     "findings": [f.__dict__ for f in r.findings]} for r in rs]
                                              for tid, rs in per_task.items()}}, indent=1))
        print(f"-> {p}")
        return
    for rep in range(a.repeat):
        run_gate(a, tasks, per_task, tools, policy, stats, label if a.repeat == 1 else f"{label}_r{rep}")


def _calibration() -> str:
    from review.tools import static
    return static.SEVERITY_VERSION


def print_tool_stats(stats: dict):
    if not stats:
        return
    print(f"\n{'tool':<10}{'fires on bug':>14}{'fires on clean':>16}{'LR':>7}{'gold-hunk hit':>15}{'chance':>8}"
          f"{'findings':>10}{'not run':>9}{'wall s':>8}")
    for name, s in stats.items():
        fb = f"{s['fired_bug']}/{s['n_bug']}"; fc = f"{s['fired_clean']}/{s['n_clean']}"; gh = f"{s['gold_hit']}/{s['n_bug']}"
        lr = "-" if s.get("lr") is None else "inf" if s["lr"] == float("inf") else f"{s['lr']:.1f}"
        print(f"{name:<10}{fb:>14}{fc:>16}{lr:>7}{gh:>15}{s.get('gold_chance', 0):>8.1f}{s['findings']:>10}{s['did_not_run']:>9}{s['wall_s']:>8.0f}")
    print("(fires = at least one medium-or-higher finding after calibration; LR = P(fire|bug)/P(fire|clean); gold-hunk hit = a\n"
          " finding within 3 lines of the defect; chance = hits expected if the same findings sat on random added lines)")


def run_gate(a, tasks, per_task, tools, policy, stats, label):
    reviews, strong, rows = {}, {}, []
    t0 = time.time()
    for t in tasks:
        rev, ok, raw, probes, audit = review_one(t, a.gate, per_task[t["id"]], tools, policy, a.probes, a.max_tokens)
        reviews[t["id"]] = rev
        if any(p.get("strong") for p in probes):
            strong[t["id"]] = True
        rows.append({"task_id": t["id"], "kind": t["kind"], "decision": rev.decision, "usd": rev.usd, "probes": probes,
                     **audit, "comments": [c.__dict__ for c in rev.comments], "parsed": ok, "raw": raw[:4000]})
        b = f" bounced={len(audit['bounces'])}" if audit["bounces"] else ""
        print(f"{t['id']:45s} {t['kind']:5s} {rev.decision:9s} cites={','.join(audit['cited']) or '-'}{b} ${rev.usd:.3f}", flush=True)
    scores = by_split(load_tasks(a.tasks) if not a.only else [x for x in load_tasks(a.tasks) if x.id in reviews], reviews, strong)
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{label}.json"
    p.write_text(json.dumps({"label": label, "model": a.gate, "harness_version": HARNESS_VERSION, "calibration": _calibration(),
                             "condition": "+".join(tools) or "none", "tools": tools, "policy": policy, "probes": a.probes,
                             "n": len(tasks), "provenance": provenance(a.tasks, {t.get("image") for t in tasks if t.get("image")}),
                             "unparsable": sum(1 for r in rows if not r["parsed"]), "wall_s": round(time.time() - t0, 1),
                             "gate_usd": round(sum(r["usd"] for r in rows), 4), "tools_usd": round(sum(r["tools_usd"] for r in rows), 4),
                             "bounces": sum(len(r["bounces"]) for r in rows), "tool_stats": stats,
                             "scores": scores, "reviews": rows}, indent=1))
    s = scores["all"]
    print(f"{label}: bugs_approved={s['bugs_approved']}/{s['n_bug']} auto_ship={s['auto_ship_rate'] or 0:.2f} "
          f"escalate={s['escalation_rate'] or 0:.2f} blk_prec={s['block_precision'] or 0:.2f} ${s['usd_per_pr'] or 0:.3f}/PR -> {p}")


if __name__ == "__main__":
    main()
