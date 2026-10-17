"""Mutation-injected bug PRs from clean commits (Mac stage, Docker + Haiku for the probe guard).

    python -m review.mine.mutate [--repo R] [--limit N] [--per-pr 3] [--since 2026-01-01]

Why: on mature repos, test-covered fix commits trace back to feature commits (200–10,000 lines) or
to bugs older than the fix's test can transport; real small bug-introducing PRs are rare (bisect.py,
Oct 17: 2 of 24). Training needs hundreds of labeled PRs. So: take a CLEAN small PR, inject one bug
into a line the PR itself changed, keep it only if the suite still passes (a defect review must
catch, because tests won't) AND a probe test can tell the mutant from the original (equivalent-mutant
guard: a mutant with no observable behavior change is not a bug and must not be labeled one).

Per clean PR (non-merge, non-fix subject, ≤ --max-lines, ≥1 code .py with added lines):
  0. baseline   related tests pass at the commit (else skip the PR: flaky/env)
  1. mutate     pick added code lines; apply one operator (regex) or an LLM-written edit; up to --per-pr
  2. stage-1    related test files (import the mutated module) must still pass        (seconds)
  3. stage-2    full suite must still pass, --timeout                                  (minutes)
  4. guard      Haiku writes a pytest that should FAIL on the mutant and PASS on the original;
                run both; keep only if it does (2 attempts). Probe is stored as the strong-loc oracle.

Output work/mutants.jsonl: repo, base (clean sha), parent, file, line, operator, source (op|llm),
original/mutated line, the bug PR's full diff (commit diff + mutation), probe source, stage results.
Each surviving mutant is a positive; its clean original is the size-matched negative, by construction.
Report by `source` downstream: a model that only learned operator shapes shows up as an op-vs-llm gap.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shlex
from collections import Counter
from pathlib import Path

from review.bench.build import _parse_results
from review.mine.scan import FIX_RE, SKIP_RE, git, is_test
from review.mine.validate import DEFAULT_CANDIDATE_INSTALL, WORK, docker, repos

PROBE_MODEL = "claude-haiku-5-5"

# ---- operators: (name, regex, replacement). Applied to ONE occurrence on ONE added line. ----
OPS = [
    ("rel_lt_le", r"(?<![<>=!])<(?!=)", "<="), ("rel_le_lt", r"<=", "<"),
    ("rel_gt_ge", r"(?<![<>=!])>(?!=)", ">="), ("rel_ge_gt", r">=", ">"),
    ("rel_eq_ne", r"==", "!="), ("rel_ne_eq", r"!=", "=="),
    ("bool_and_or", r"\band\b", "or"), ("bool_or_and", r"\bor\b", "and"),
    ("not_drop", r"\bif not\b", "if"), ("not_add", r"\bif (?!not\b)", "if not "),
    ("is_none_flip", r"\bis None\b", "is not None"), ("is_not_none_flip", r"\bis not None\b", "is None"),
    ("plus1_minus1", r"\+ 1\b", "- 1"), ("minus1_plus1", r"- 1\b", "+ 1"),
    ("true_false", r"\bTrue\b", "False"), ("false_true", r"\bFalse\b", "True"),
    ("zero_one", r"(?<![\w.])0(?![\w.])", "1"), ("one_zero", r"(?<![\w.])1(?![\w.])", "0"),
    ("return_none", r"^(\s*)return (?!None\b).+$", r"\1return None"),
    ("raise_pass", r"^(\s*)raise .+$", r"\1pass"),
    ("continue_break", r"\bcontinue\b", "break"), ("break_continue", r"\bbreak\b", "continue"),
    ("slice_drop_last", r"\[:-1\]", "[:]"), ("range_n_minus1", r"range\((\w+)\)", r"range(\1 - 1)"),
]
SKIP_LINE = re.compile(r"^\s*(#|\"\"\"|'''|import |from |@|def |class |pass$|\.\.\.$|$)")


def clean_prs(repo: Path, since: str, max_lines: int) -> list[dict]:
    out = []
    for line in git(repo, "log", "--no-merges", f"--since={since}", "--format=%H%x00%ad%x00%s", "--date=short").splitlines():
        sha, date, subj = line.split("\x00", 2)
        if FIX_RE.search(subj) or SKIP_RE.search(subj):
            continue
        files, n = [], 0
        for ln in git(repo, "show", "--numstat", "--format=", sha).splitlines():
            p = ln.split("\t")
            if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
                n += int(p[0]) + int(p[1]); files.append(p[2])
        code = [f for f in files if f.endswith(".py") and not is_test(f)]
        if not code or n > max_lines or n < 5:
            continue
        out.append({"repo": repo.name, "sha": sha, "parent": git(repo, "rev-parse", f"{sha}^").strip(),
                    "date": date, "subject": subj, "lines": n, "code_files": code,
                    "test_files": [f for f in files if f.endswith(".py") and is_test(f)]})
    return out


def added_lines(repo: Path, sha: str, f: str) -> list[tuple[int, str]]:
    """(new-file line number, text) for lines the commit added in f."""
    out, new_ln = [], 0
    for ln in git(repo, "show", "--unified=0", "--format=", sha, "--", f).splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)", ln)
        if m:
            new_ln = int(m.group(1)); continue
        if ln.startswith("+") and not ln.startswith("+++"):
            out.append((new_ln, ln[1:])); new_ln += 1
        elif not ln.startswith("-") and not ln.startswith("\\"):
            new_ln += 1
    return out


def op_mutants(lines: list[tuple[int, str]], k: int, rng: random.Random) -> list[dict]:
    cands = []
    for ln, text in lines:
        if SKIP_LINE.match(text):
            continue
        for name, pat, rep in OPS:
            if re.search(pat, text):
                new = re.sub(pat, rep, text, count=1)
                if new != text:
                    cands.append({"line": ln, "operator": name, "source": "op", "original": text, "mutated": new})
    rng.shuffle(cands)
    seen, out = set(), []
    for c in cands:                      # at most one mutant per line
        if c["line"] not in seen:
            seen.add(c["line"]); out.append(c)
        if len(out) >= k:
            break
    return out


def llm_mutant(repo: Path, sha: str, f: str, lines: list[tuple[int, str]], model: str) -> dict | None:
    from review.agent.loop import call
    src = git(repo, "show", f"{sha}:{f}").splitlines()
    nums = [ln for ln, _ in lines]
    lo, hi = max(1, min(nums) - 15), min(len(src), max(nums) + 15)
    prompt = (f"File {f} at a commit. Lines marked + were added by the commit:\n" +
              "\n".join(f"{i:5d}|+{src[i-1]}" if i in nums else f"{i:5d}| {src[i-1]}" for i in range(lo, hi + 1)) +
              "\n\nIntroduce ONE realistic bug by editing exactly one of the + lines: the kind a tired author writes and a "
              "reviewer should catch (wrong boundary, wrong variable, missing guard, wrong default, inverted check). "
              "Keep the file syntactically valid and the line's indentation. Reply with JSON only: "
              '{"line": <int>, "mutated": "<the full replacement line>", "why": "<one line>"}')
    try:
        text, _, _ = call(model, "You write minimal, plausible single-line bugs for a code-review benchmark.", [{"role": "user", "content": prompt}], 800)
        m = re.search(r"\{.*\}", text, re.S); o = json.loads(m.group(0))
        ln = int(o["line"])
        if ln not in nums or o["mutated"].strip() == src[ln - 1].strip():
            return None
        return {"line": ln, "operator": "llm", "source": "llm", "original": src[ln - 1], "mutated": o["mutated"], "why": o.get("why", "")}
    except Exception:
        return None


def related_tests(repo: Path, sha: str, f: str, test_files: list[str]) -> list[str]:
    mod = Path(f).stem
    hits = set(test_files)
    for t in git(repo, "ls-tree", "-r", "--name-only", sha).splitlines():
        if t.endswith(".py") and is_test(t) and (mod in Path(t).stem or
                                               re.search(rf"\b{re.escape(mod)}\b", git(repo, "show", f"{sha}:{t}")[:20000])):
            hits.add(t)
    return sorted(hits)


def script(sha: str, inst: str, edits: dict[str, str] | None, probe: tuple[str, str] | None,
           pytest_targets: list[str] | None, want_diff: bool, parent: str = "", timeout_s: int = 1500) -> tuple[str, dict[str, str]]:
    """Container script + the files to mount at /inject. Checkout sha, copy edited files (path -> full
    content) and the probe into place, optionally print the diff vs parent, run pytest with -rA."""
    files, parts = {}, [f"git clone -q /repo /w && cd /w && git checkout -q {sha}", "export PYTHONPATH=/w/src:/w",
                        f"({inst} >/tmp/pip.log 2>&1 || echo INSTALL_WARN)"]
    for i, (path, content) in enumerate((edits or {}).items()):
        files[f"edit_{i}"] = content
        parts.append(f"cp /inject/edit_{i} {shlex.quote(path)}")
    if want_diff:
        parts.append(f"echo DIFF_BEGIN; git diff {parent}; echo DIFF_END")
    if probe:
        files["probe"] = probe[1]
        parts.append(f"mkdir -p $(dirname {shlex.quote(probe[0])}) && cp /inject/probe {shlex.quote(probe[0])}")
    tg = " ".join(shlex.quote(t) for t in (pytest_targets or []))
    parts.append(f"{{ timeout {timeout_s} python -m pytest -rA -q -p no:cacheprovider --no-header {tg} 2>&1 | tail -n 3000; echo EXIT=${{PIPESTATUS[0]}}; }}")
    return " && ".join(parts[:3]) + " && " + " ; ".join(parts[3:]), files


def run(name: str, sha: str, inst: str, net: bool, timeout: int, **kw) -> str:
    sc, files = script(sha, inst, **kw)
    return docker(name, sc, timeout, net, files)[1]


def passed(out: str) -> set[str]:
    return {t for t, st in _parse_results(out).items() if st == "passed"}


def killed_by(baseline_pass: set[str], out: str) -> set[str]:
    """Tests that passed at baseline and no longer pass: the mutation-testing kill criterion."""
    return baseline_pass - passed(out)


def pytest_ok(out: str) -> bool:
    m = re.search(r"EXIT=(\d+)", out)
    return bool(m) and m.group(1) == "0"


def exit_code(out: str) -> str:
    m = re.search(r"EXIT=(\d+)", out)
    return m.group(1) if m else "none"


def mutated_file(repo: Path, sha: str, f: str, line: int, new_text: str) -> str:
    src = git(repo, "show", f"{sha}:{f}").splitlines(keepends=True)
    nl = "\n" if src[line - 1].endswith("\n") else ""
    src[line - 1] = new_text.rstrip("\n") + nl
    return "".join(src)


def gen_probe(repo_name: str, f: str, orig: str, mut: str, context: str, model: str, feedback: str = "") -> str | None:
    from review.agent.loop import call
    prompt = (f"Repo {repo_name}, file {f}. A one-line change was made:\n-{orig}\n+{mut}\n\nSurrounding code:\n{context}\n\n"
              "Write a pytest file that FAILS when the file contains the + line and PASSES with the - line, by calling the "
              "code's public behavior (import from the package, not from tests/). Deterministic, no network, no filesystem "
              "beyond tmp_path. Reply with the file contents only, in a ```python fence." + (f"\n\nPrevious attempt did not distinguish them:\n{feedback}" if feedback else ""))
    try:
        text, _, _ = call(model, "You write small, decisive regression tests.", [{"role": "user", "content": prompt}], 1500)
        m = re.search(r"```python\n(.*?)```", text, re.S)
        return m.group(1) if m else None
    except Exception:
        return None


def process_pr(pr: dict, cfg: dict, per_pr: int, llm: bool, timeout: int, rng: random.Random, log) -> list[dict]:
    name = pr["repo"]; repo = WORK / "repos" / name; r = cfg[name]
    inst, net = r.get("candidate_install", DEFAULT_CANDIDATE_INSTALL), bool(r.get("network"))
    rows = []
    for f in pr["code_files"][:2]:
        lines = added_lines(repo, pr["sha"], f)
        lines = [(n, t) for n, t in lines if not SKIP_LINE.match(t)]
        if not lines:
            continue
        rel = related_tests(repo, pr["sha"], f, pr["test_files"])
        targets = rel or None                       # None -> whole suite
        base_out = run(name, pr["sha"], inst, net, timeout, edits=None, probe=None, pytest_targets=targets, want_diff=False)
        base_pass = passed(base_out)
        if not base_pass:
            n_pass_lines = len(re.findall(r"^PASSED ", base_out, re.M))
            log(f"{name:9s} {pr['sha'][:8]} {f}: baseline collected nothing (exit {exit_code(base_out)}; PASSED lines seen={n_pass_lines}; "
                f"parsed={len(_parse_results(base_out))}; out={len(base_out)} chars), skip\n    " + base_out.strip().replace("\n", "\n    ")[-900:])
            rows.append({**{k: pr[k] for k in ("repo", "sha", "parent", "subject", "lines")}, "file": f, "status": "baseline_empty",
                         "targets": targets, "baseline_tail": base_out[-1500:]}); break
        base_full_pass = None                       # full-suite baseline, computed once if any mutant survives stage 1
        muts = op_mutants(lines, per_pr - (1 if llm else 0), rng)
        if llm:
            m = llm_mutant(repo, pr["sha"], f, lines, PROBE_MODEL)
            if m: muts.append(m)
        for m in muts:
            content = mutated_file(repo, pr["sha"], f, m["line"], m["mutated"])
            row = {**{k: pr[k] for k in ("repo", "sha", "parent", "subject", "lines")}, "file": f, **m,
                   "targets": targets, "baseline_passed": len(base_pass)}
            out1 = run(name, pr["sha"], inst, net, timeout, edits={f: content}, probe=None, pytest_targets=targets, want_diff=False)
            k1 = killed_by(base_pass, out1)
            row["stage1"] = "killed" if k1 else ("survived" if exit_code(out1) in ("0", "1") else "error")
            row["stage1_killers"] = sorted(k1)[:10]; row["stage1_tail"] = out1[-1200:]
            if row["stage1"] != "survived":
                extra = "" if row["stage1"] == "killed" else " (exit " + exit_code(out1) + ")\n    " + out1.strip().replace("\n", "\n    ")[-500:]
                rows.append(row); log(f"{name:9s} {pr['sha'][:8]} {f}:{m['line']} {m['operator']:14s} {row['stage1']} {len(k1)}{extra}"); continue
            if base_full_pass is None:
                base_full_pass = passed(run(name, pr["sha"], inst, net, timeout, edits=None, probe=None, pytest_targets=None, want_diff=False))
            out2 = run(name, pr["sha"], inst, net, timeout, edits={f: content}, probe=None, pytest_targets=None, want_diff=True, parent=pr["parent"])
            k2 = killed_by(base_full_pass, out2)
            row["stage2"] = "killed" if k2 else ("survived" if exit_code(out2) in ("0", "1") else "error")
            row["stage2_killers"] = sorted(k2)[:10]
            row["stage2_tail"] = (out2.split("DIFF_BEGIN")[0] if "DIFF_BEGIN" in out2 else out2)[-1500:]
            if "DIFF_BEGIN" in out2:
                row["bug_diff"] = out2.split("DIFF_BEGIN", 1)[1].split("DIFF_END", 1)[0].strip()
            if row["stage2"] != "survived":
                extra = "" if row["stage2"] == "killed" else " (exit " + exit_code(out2) + ")\n    " + row["stage2_tail"].strip().replace("\n", "\n    ")[-600:]
                rows.append(row); log(f"{name:9s} {pr['sha'][:8]} {f}:{m['line']} {m['operator']:14s} suite-{row['stage2']} {len(k2)}{extra}"); continue
            # guard: a probe that tells mutant from original
            src = git(repo, "show", f"{pr['sha']}:{f}").splitlines()
            ctx = "\n".join(src[max(0, m["line"] - 25): m["line"] + 25])
            row["guard"] = "no_probe"; fb = ""
            for attempt in range(2):
                probe = gen_probe(name, f, m["original"], m["mutated"], ctx, PROBE_MODEL, fb)
                if not probe:
                    continue
                ppath = "tests/test_probe_mutant.py"
                on_mut = run(name, pr["sha"], inst, net, timeout, edits={f: content}, probe=(ppath, probe), pytest_targets=[ppath], want_diff=False)
                on_orig = run(name, pr["sha"], inst, net, timeout, edits=None, probe=(ppath, probe), pytest_targets=[ppath], want_diff=False)
                mut_fail, orig_pass = exit_code(on_mut) == "1", pytest_ok(on_orig)
                row.update(probe=probe, probe_on_mutant="fail" if mut_fail else ("pass" if pytest_ok(on_mut) else "error"),
                           probe_on_original="pass" if orig_pass else ("fail" if exit_code(on_orig) == "1" else "error"))
                if mut_fail and orig_pass:
                    row["guard"] = "distinguished"; break
                fb = f"on mutant: {row['probe_on_mutant']}; on original: {row['probe_on_original']}\n{on_mut[-600:]}"
                row["guard"] = "equivalent_or_untestable"
            row["status"] = "positive" if row["guard"] == "distinguished" else "dropped_" + row["guard"]
            rows.append(row)
            log(f"{name:9s} {pr['sha'][:8]} {f}:{m['line']} {m['operator']:14s} suite-green guard={row['guard']}")
        break                                        # one file per PR is enough
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--limit", type=int, default=0, help="clean PRs to process")
    ap.add_argument("--per-pr", type=int, default=3); ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--max-lines", type=int, default=400); ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-llm", action="store_true", help="operators only (no Haiku mutations; the probe guard still uses Haiku)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = repos(); rng = random.Random(a.seed)
    out_p = WORK / "mutants.jsonl"
    done = set()
    if out_p.exists():
        for l in out_p.read_text().splitlines():
            if l.strip():
                d = json.loads(l); done.add(d["sha"])
    prs = []
    for name in cfg:
        if a.repo and name != a.repo:
            continue
        repo = WORK / "repos" / name
        if (repo / ".git").exists():
            prs += clean_prs(repo, a.since, a.max_lines)
    prs = [p for p in prs if p["sha"] not in done]
    rng.shuffle(prs)
    if a.limit:
        prs = prs[:a.limit]
    print(f"{len(prs)} clean PRs to mutate ({len(done)} already done)", flush=True)
    with out_p.open("a") as fh:
        for pr in prs:
            for row in process_pr(pr, cfg, a.per_pr, not a.no_llm, a.timeout, rng, lambda s: print(s, flush=True)):
                fh.write(json.dumps(row) + "\n"); fh.flush()
    rows = [json.loads(l) for l in out_p.read_text().splitlines() if l.strip()]
    c = Counter(r.get("status") or r.get("stage2", r.get("stage1")) for r in rows)
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(c.items(), key=str)))
    pos = [r for r in rows if r.get("status") == "positive"]
    print(f"positives: {len(pos)}  by source: {dict(Counter(r['source'] for r in pos))}  by operator: {dict(Counter(r['operator'] for r in pos))}")


if __name__ == "__main__":
    main()
