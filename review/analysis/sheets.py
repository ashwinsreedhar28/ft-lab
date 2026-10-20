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


def _pair_lines(t: dict, orig: dict | None) -> tuple[str, str]:
    """(original added line, mutated added line): the one line the mutant and its clean original disagree on."""
    if not orig:
        return "", ""
    oa = [l[1:] for l in orig["diff"].splitlines() if l.startswith("+") and not l.startswith("+++")]
    ma = [l[1:] for l in t["diff"].splitlines() if l.startswith("+") and not l.startswith("+++")]
    before = [l for l in oa if l not in ma]
    after = [l for l in ma if l not in oa]
    return (before[0] if before else ""), (after[0] if after else "")


def _html_escape(x: str) -> str:
    return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ai_note(t: dict, before: str, after: str, ctx: str, probe: str) -> str:
    """Haiku explains the two lines and names the input where they diverge. Description, not a verdict:
    the yes/no stays with the human reviewer; the tally records that notes were AI-assisted."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ""
    try:
        from review.agent.loop import call
        prompt = (f"Repo {t['repo']}, file {t['gold_hunks'][0]['file'] if t['gold_hunks'] else '?'}.\n"
                  f"Original line:\n{before}\nChanged line:\n{after}\n\nSurrounding diff:\n{ctx[:3000]}\n\n"
                  f"A test that fails on the changed line and passes on the original:\n{probe[:2000]}\n\n"
                  "In at most 3 short sentences, plain English, for a code reviewer: (1) what the original line does, "
                  "(2) what the changed line does instead, (3) the concrete input or situation where the two behave differently. "
                  "Do NOT say whether it is a bug or whether to block; describe only.")
        text, _, _ = call("claude-haiku-5-5", "You explain one-line code changes neutrally and concretely.", [{"role": "user", "content": prompt}], 400)
        return text.strip()
    except Exception as e:  # noqa: BLE001
        return f"(note unavailable: {type(e).__name__})"


def labels_html(tasks_path: str, n: int, seed: int):
    """Browser version of the label sheet: before/after line, context, probe, verdict buttons, JSON export."""
    tasks = [json.loads(l) for l in Path(tasks_path).read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    ops = [t for t in tasks if t.get("source") == "op"]; llms = [t for t in tasks if t.get("source") == "llm"]
    real = [t for t in tasks if t.get("source") == "real"]
    rng.shuffle(ops); rng.shuffle(llms)
    sample = ops[: n // 2] + llms[: n - n // 2] + real
    by_id = {t["id"]: t for t in tasks}
    items = []
    for i, t in enumerate(sample, 1):
        orig = by_id.get(f"clean__{t['repo']}__{t['fix_sha'][:8]}") if t.get("source") in ("op", "llm") else None
        before, after = _pair_lines(t, orig)
        gh = t["gold_hunks"][0] if t["gold_hunks"] else {"file": "?", "line_start": 0}
        # context: ±12 lines of the mutant diff around the mutated line
        dl = t["diff"].splitlines()
        idx = next((k for k, l in enumerate(dl) if l.startswith("+") and l[1:] == after), None) if after else None
        ctx = dl[max(0, (idx or 0) - 12): (idx or 0) + 13] if idx is not None else dl[:40]
        ctx_html = "".join(
            f'<div class="{ "mut" if (l.startswith("+") and l[1:] == after) else "add" if l.startswith("+") else "del" if l.startswith("-") else "" }">{_html_escape(l)}</div>'
            for l in ctx)
        probe = t.get("probe_source") or ""
        note = ai_note(t, before, after, "\n".join(ctx), probe) if (before or after) else ""
        doc = ""
        m = __import__("re").search(r'"""(.*?)"""', probe, __import__("re").S)
        if m: doc = m.group(1).strip()
        items.append(f"""
<section id="it{i}" data-id="{_html_escape(t['id'])}">
<h2>{i}. <code>{_html_escape(t['id'])}</code> <span class="tag">{t.get('source')} {t.get('operator') or ''}</span> <span class="tag">{t['repo']}</span></h2>
<p class="where">{_html_escape(gh['file'])} line {gh['line_start']}</p>
<table class="pair"><tr><th>original (clean PR)</th><th>mutated (the labeled bug)</th></tr>
<tr><td><pre>{_html_escape(before) or '<i>real positive: see gold hunks</i>'}</pre></td><td><pre class="mut">{_html_escape(after) or '<i>see diff</i>'}</pre></td></tr></table>
{f'<p class="note"><b>AI note (Haiku, descriptive only):</b> {_html_escape(note)}</p>' if note else ''}
{f'<p class="hint"><b>probe says:</b> {_html_escape(doc)}</p>' if doc else ''}
<details><summary>context (±12 lines of the PR diff; mutated line highlighted)</summary><pre class="ctx">{ctx_html}</pre></details>
<details><summary>guard probe (fails on the mutant, passes on the original)</summary><pre>{_html_escape(probe[:3000])}</pre></details>
<div class="verdict">
<label><input type="radio" name="v{i}" value="yes"> yes, a reviewer should block this</label>
<label><input type="radio" name="v{i}" value="arguable"> arguable (wrong only in an edge case)</label>
<label><input type="radio" name="v{i}" value="no"> no, not a bug (same behavior)</label>
<input type="text" class="why" id="w{i}" placeholder="why (one line)">
</div>
</section>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Label-noise sheet</title>
<style>
body{{font:15px/1.45 -apple-system,system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}}
h2{{font-size:16px;margin:28px 0 4px}} .tag{{font-size:12px;background:#eee;border-radius:4px;padding:1px 6px;margin-left:6px}}
.where{{color:#666;margin:0 0 8px;font-size:13px}}
table.pair{{width:100%;border-collapse:collapse}} table.pair th{{text-align:left;font-size:12px;color:#666;font-weight:600}}
table.pair td{{width:50%;vertical-align:top;padding:4px 8px 4px 0}}
pre{{background:#f6f6f6;padding:8px;border-radius:4px;white-space:pre-wrap;font-size:12.5px;margin:0}}
pre.mut{{background:#fde8e2}} pre.ctx div.mut{{background:#f9c9bd}} pre.ctx div.add{{color:#1a7f37}} pre.ctx div.del{{color:#9a1b1b}}
.hint{{background:#eef5ff;padding:6px 10px;border-radius:4px;font-size:13.5px}}
.note{{background:#f3f0ff;padding:6px 10px;border-radius:4px;font-size:13.5px;white-space:pre-wrap}}
details{{margin:6px 0}} summary{{cursor:pointer;color:#444;font-size:13px}}
.verdict{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px}} .verdict label{{font-size:14px}}
.why{{flex:1;min-width:240px;padding:4px 6px}}
#bar{{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:1px solid #ddd;display:flex;gap:12px;align-items:center}}
button{{padding:6px 12px}} #out{{width:100%;height:80px;font-size:12px}}
</style></head><body>
<div id="bar"><b>Label-noise sheet</b> <span id="count">0/{len(sample)}</span>
<button onclick="exportJson()">Copy verdicts as JSON</button> <span style="color:#666;font-size:13px">then paste into review/analysis/label_verdicts.json and run make sheet-tally</span></div>
<p>{len(sample)} items: {len(ops[: n // 2])} operator mutants, {len(llms[: n - n // 2])} LLM-written mutants, {len(real)} real positives.
For each: <b>would you block this line in review?</b> You are not finding the bug; it is marked. The question is only whether the change is wrong.</p>
{''.join(items)}
<textarea id="out" readonly></textarea>
<script>
function collect(){{const o={{}};document.querySelectorAll('section').forEach(s=>{{const id=s.dataset.id;const r=s.querySelector('input[type=radio]:checked');if(r)o[id]={{verdict:r.value,why:s.querySelector('.why').value}};}});return o;}}
function refresh(){{const o=collect();document.getElementById('count').textContent=Object.keys(o).length+'/{len(sample)}';try{{localStorage.setItem('labelsheet',JSON.stringify(o));}}catch(e){{}}}}
function exportJson(){{const j=JSON.stringify(collect(),null,1);document.getElementById('out').value=j;navigator.clipboard&&navigator.clipboard.writeText(j).catch(()=>{{}});}}
document.addEventListener('change',refresh);document.addEventListener('input',refresh);
try{{const saved=JSON.parse(localStorage.getItem('labelsheet')||'{{}}');document.querySelectorAll('section').forEach(s=>{{const v=saved[s.dataset.id];if(!v)return;const r=s.querySelector(`input[value=${{v.verdict}}]`);if(r)r.checked=true;s.querySelector('.why').value=v.why||'';}});refresh();}}catch(e){{}}
</script></body></html>"""
    vj = OUT / "label_verdicts.json"
    if vj.exists():
        html = html.replace("try{const saved=JSON.parse(localStorage.getItem('labelsheet')||'{}');",
                            "try{const saved=Object.assign(" + vj.read_text() + ",JSON.parse(localStorage.getItem('labelsheet')||'{}'));")
    p = OUT / "label_sheet.html"; p.write_text(html); print(f"{len(sample)} items -> {p}  (open in a browser)")


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
    pj = OUT / "label_verdicts.json"
    p = OUT / "label_sheet.md"
    if pj.exists():
        v = [x.get("verdict", "") for x in json.loads(pj.read_text()).values()]
    elif p.exists():
        v = re.findall(r"^verdict:\s*(\w+)", p.read_text(), re.M)
    else:
        v = []
    if v:
        from collections import Counter
        c = Counter(x.lower() for x in v if x)
        print(f"label sheet: {dict(c)}  filled {sum(c.values())}  (verdicts by Ashwin; per-item AI notes shown, descriptive only)")
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
        labels_html(args.tasks, args.n, args.seed)
    elif args.cmd == "failures":
        failures(args.results, args.tasks)
    else:
        tally()


if __name__ == "__main__":
    main()
