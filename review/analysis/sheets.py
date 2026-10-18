"""Human review sheets (markdown) for the two checks no metric replaces.

    python -m review.analysis.sheets labels   [--n 20]   -> review/analysis/label_sheet.md
    python -m review.analysis.sheets failures <results.json> [...]  -> review/analysis/failure_sheet.md

labels:   a sample of mutants (op and llm, stratified) plus every real positive, each with the PR diff
          and the injected line highlighted. You mark: "a careful reviewer should have caught this:
          yes / no / arguable" and a one-line reason. The yes-rate is the label-noise estimate.
failures: every bug PR a run approved, and every clean PR it blocked, with the decision, the agent's
          comments, and its probes (hypothesis + status). You tag each: wrong-file / right-file-wrong-
          hypothesis / never-probed / vacuous-probe-passed / other. That taxonomy is the error analysis.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).parent


def _diff_block(diff: str, highlight: str | None, max_lines: int = 160) -> str:
    lines = diff.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(diff.splitlines()) - max_lines} more lines)"]
    if highlight:
        lines = [("  >>> " + l if l.startswith("+") and l[1:].strip() == highlight.strip() else l) for l in lines]
    return "```diff\n" + "\n".join(lines) + "\n```"


def labels(tasks_path: str, n: int, seed: int):
    tasks = [json.loads(l) for l in Path(tasks_path).read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    ops = [t for t in tasks if t.get("source") == "op"]; llms = [t for t in tasks if t.get("source") == "llm"]
    real = [t for t in tasks if t.get("source") == "real"]
    rng.shuffle(ops); rng.shuffle(llms)
    sample = ops[: n // 2] + llms[: n - n // 2] + real
    by_id = {t["id"]: t for t in tasks}
    out = ["# Label-noise sheet\n", f"{len(sample)} items: {len(ops[: n // 2])} operator mutants, {len(llms[: n - n // 2])} LLM mutants, {len(real)} real positives.",
           "For each: **should a careful reviewer have caught this?** `yes` / `no` / `arguable`, one line why. "
           "`no` means the labeled bug is not a bug (equivalent mutant, or the 'fix' was a refactor). Fill the `verdict:` lines; "
           "`python -m review.analysis.sheets tally` summarizes.\n"]
    for i, t in enumerate(sample, 1):
        out.append(f"\n---\n## {i}. `{t['id']}`  ({t.get('source')}, {t.get('operator') or ''}; repo {t['repo']})")
        if t.get("source") in ("op", "llm"):
            orig = by_id.get(f"clean__{t['repo']}__{t['fix_sha'][:8]}")
            gh = t["gold_hunks"][0]
            out.append(f"Injected bug: `{gh['file']}` line {gh['line_start']} (marked `>>>` below). The clean original is `{orig['id'] if orig else '?'}`.")
        else:
            out.append(f"Real bug-introducing commit; fixed in `{t['fix_sha'][:8]}`; oracle tests: {', '.join(t['oracle'][:3])}. Gold hunks: " +
                       ", ".join(f"{h['file']}:{h['line_start']}-{h['line_end']}" for h in t["gold_hunks"][:5]))
        hl = None
        if t.get("source") in ("op", "llm"):
            # the mutated line is the added line present in the mutant diff but not in the original's
            orig_added = set(l for l in (orig["diff"].splitlines() if orig else []) if l.startswith("+"))
            cand = [l for l in t["diff"].splitlines() if l.startswith("+") and l not in orig_added and not l.startswith("+++")]
            hl = cand[0][1:] if cand else None
        out.append(_diff_block(t["diff"], hl))
        if t.get("probe_source"):
            out.append("<details><summary>guard probe (fails on the bug, passes on the original)</summary>\n\n```python\n" + t["probe_source"][:2500] + "\n```\n</details>")
        out.append("\nverdict: \nwhy: ")
    p = OUT / "label_sheet.md"; p.write_text("\n".join(out)); print(f"{len(sample)} items -> {p}")


def failures(result_paths: list[str], tasks_path: str):
    tasks = {t["id"]: t for t in (json.loads(l) for l in Path(tasks_path).read_text().splitlines() if l.strip())}
    out = ["# Failure sheet\n", "Every bug PR approved and every clean PR blocked, per run. Tag each `cause:` with one of: "
           "`wrong-file`, `right-file-wrong-hypothesis`, `never-probed`, `vacuous-probe-passed`, `surface-cue-block`, `other`; one line of notes.\n"]
    for rp in result_paths:
        r = json.loads(Path(rp).read_text())
        out.append(f"\n# Run `{r['label']}` ({r['model']})\n")
        for rev in r["reviews"]:
            t = tasks.get(rev["task_id"])
            if not t:
                continue
            bad = (t["kind"] == "bug" and rev["decision"] == "approve") or (t["kind"] == "clean" and rev["decision"] == "block")
            if not bad:
                continue
            kind = "APPROVED A BUG" if t["kind"] == "bug" else "BLOCKED A CLEAN PR"
            out.append(f"\n---\n## {kind}: `{t['id']}`  ({t.get('source')}, {t.get('operator') or ''})")
            if t["kind"] == "bug":
                out.append("Gold: " + ", ".join(f"{h['file']}:{h['line_start']}-{h['line_end']}" for h in t["gold_hunks"][:5]))
            for c in rev.get("comments", [])[:5]:
                out.append(f"- comment `{c['file']}:{c['line_start']}-{c['line_end']}`: {c['body'][:300]}")
            for p in rev.get("probes", []):
                out.append(f"- probe r{p['round']}: status={p['status']} specific={p.get('specific')} strong={p.get('strong')} targets_uncovered={p.get('targets_uncovered')}\n"
                           f"  hypothesis: {p.get('hypothesis', '')[:300]}")
            if not rev.get("probes"):
                out.append(f"- no probe ({rev.get('no_probe_reason')})")
            out.append(_diff_block(t["diff"], None, 80))
            out.append("\ncause: \nnotes: ")
    p = OUT / "failure_sheet.md"; p.write_text("\n".join(out)); print(f"-> {p}")


def tally():
    import re
    p = OUT / "label_sheet.md"
    if p.exists():
        v = re.findall(r"^verdict:\s*(\w+)", p.read_text(), re.M)
        from collections import Counter
        c = Counter(x.lower() for x in v if x)
        print(f"label sheet: {dict(c)}  filled {sum(c.values())}")
        if c:
            print(f"  label-noise estimate (no + arguable/2) / filled = {(c.get('no', 0) + c.get('arguable', 0) / 2) / max(sum(c.values()), 1):.2f}")
    p = OUT / "failure_sheet.md"
    if p.exists():
        v = re.findall(r"^cause:\s*([\w-]+)", p.read_text(), re.M)
        from collections import Counter
        print(f"failure sheet: {dict(Counter(x.lower() for x in v if x))}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("labels"); a.add_argument("--n", type=int, default=20); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--tasks", default="review/bench/public_tasks.jsonl")
    b = sub.add_parser("failures"); b.add_argument("results", nargs="+"); b.add_argument("--tasks", default="review/bench/public_tasks.jsonl")
    sub.add_parser("tally")
    args = ap.parse_args()
    if args.cmd == "labels":
        labels(args.tasks, args.n, args.seed)
    elif args.cmd == "failures":
        failures(args.results, args.tasks)
    else:
        tally()


if __name__ == "__main__":
    main()
