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

from review.mine.scan import FIX_RE, SKIP_RE, git, is_test
from review.mine.validate import DEFAULT_CANDIDATE_INSTALL, WORK, docker, repos

PROBE_MODEL = "claude-haiku-5-5"

# ---- operators: (name, regex, replacement). Applied to ONE occurrence on ONE added line. ----
OPS = [
    ("rel_lt_le", r"(?<![<>=!])<(?!=)", "<="), ("rel_le_lt", r"<=", "<"),
    ("rel_gt_ge", r"(?<![-<>=!])>(?!=)", ">="), ("rel_ge_gt", r">=", ">"),
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
NON_CODE_DIRS = ("scripts/", "docs/", "docs_src/", "examples/", "benchmarks/", "benches/", "tools/", "setup.py")
STATEMENT_OPS = {"return_none", "raise_pass"}   # rewrite a whole statement: only safe on a line that starts one
SKIP_LINE = re.compile(r"^\s*(#|\"\"\"|'''|import |from |@|def |class |pass$|\.\.\.$|$)")


def clean_prs(repo: Path, since: str, max_lines: int, exclude: tuple[str, ...] = ()) -> list[dict]:
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
        code = [f for f in files if f.endswith(".py") and not is_test(f) and not f.startswith(NON_CODE_DIRS) and not f.startswith(exclude)]
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


def docstring_lines(src: list[str]) -> set[int]:
    """1-based line numbers inside triple-quoted strings (both quote styles). Operators must not touch them:
    a `>>>` doctest prompt mutated to `>=>>` passed the guard once (the probe parsed the docstring)."""
    inside, q, out = False, "", set()
    for i, line in enumerate(src, 1):
        j, touched = 0, False
        while True:
            found = [(k, t) for t in ('"""', "'''") for k in [line.find(t, j)] if k >= 0]
            if not found:
                break
            k, t = min(found)
            if not inside:
                inside, q = True, t
            elif t == q:
                inside = False
            touched = True
            j = k + 3
        if inside or touched:               # opening, interior, and closing lines all count
            out.add(i)
    return out


def op_mutants(lines: list[tuple[int, str]], k: int, rng: random.Random, src: list[str] | None = None) -> list[dict]:
    cands = []
    doc = docstring_lines(src) if src else set()
    for ln, text in lines:
        if SKIP_LINE.match(text) or ln in doc or ">>>" in text or text.lstrip().startswith("#"):
            continue
        prev = src[ln - 2].rstrip() if src and ln >= 2 else ""
        continuation = prev.endswith(("(", "[", "{", ",", "\\")) or prev.count("(") > prev.count(")")
        for name, pat, rep in OPS:
            if name in STATEMENT_OPS and continuation:
                continue                       # `return (\n  x,\n  y)`: rewriting line 2 breaks the parse
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


def related_tests(repo: Path, sha: str, f: str, test_files: list[str], exclude: tuple[str, ...] = ()) -> list[str]:
    """Test files that import or name the mutated module. `exclude` = the repo's exclude_paths: an explicit
    pytest argument bypasses --ignore, and pydantic-core/tests/conftest.py loading as `tests.conftest`
    collided with the real one (51 of 92 empty baselines in run 2)."""
    mod = Path(f).stem
    hits = {t for t in test_files if not t.startswith(exclude)}
    for t in git(repo, "ls-tree", "-r", "--name-only", sha).splitlines():
        if t.startswith(exclude):
            continue
        if t.endswith(".py") and is_test(t) and (mod in Path(t).stem or
                                               re.search(rf"\b{re.escape(mod)}\b", git(repo, "show", f"{sha}:{t}")[:20000])):
            hits.add(t)
    return sorted(hits)


def script(sha: str, inst: str, edits: dict[str, str] | None, probe: tuple[str, str] | None,
           pytest_targets: list[str] | None, want_diff: bool, parent: str = "", timeout_s: int = 1500,
           pytest_args: str = "") -> tuple[str, dict[str, str]]:
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
    # junit xml is the record of outcomes (the -rA text is not reliable at scale); warnings that the
    # image's newer pytest raises on older commits must not abort collection under filterwarnings=error
    parts.append(f"{{ timeout {timeout_s} python -m pytest -q -p no:cacheprovider --no-header --continue-on-collection-errors "
                 f"-W ignore::DeprecationWarning -W ignore::ResourceWarning -W ignore::PendingDeprecationWarning "
                 f"--junitxml=/tmp/junit.xml {pytest_args} {tg} 2>&1 | tail -n 200; echo EXIT=${{PIPESTATUS[0]}}; "
                 f"echo JUNIT_BEGIN; cat /tmp/junit.xml 2>/dev/null; echo JUNIT_END; }}")
    return " && ".join(parts[:3]) + " && " + " ; ".join(parts[3:]), files


def run(name: str, sha: str, inst: str, net: bool, timeout: int, tag: str = "", **kw) -> str:
    kw.setdefault("pytest_args", repos()[name].get("pytest_args", ""))
    sc, files = script(sha, inst, **kw)
    out = docker(name, sc, timeout, net, files)[1]
    if tag:                                      # raw output for post-mortems: work/debug/<tag>.txt
        d = WORK / "debug"; d.mkdir(exist_ok=True)
        (d / f"{tag}.txt").write_text(sc + "\n\n=====\n" + out)
    return out


def junit(out: str) -> dict[str, str]:
    """{test id: passed|failed|error|skipped} from the junit xml echoed between JUNIT markers."""
    import xml.etree.ElementTree as ET
    if "JUNIT_BEGIN" not in out:
        return {}
    xml = out.split("JUNIT_BEGIN", 1)[1].split("JUNIT_END", 1)[0].strip()
    if not xml:
        return {}
    res = {}
    try:
        for tc in ET.fromstring(xml).iter("testcase"):
            tid = f"{tc.get('classname', '')}::{tc.get('name', '')}"
            kids = {k.tag for k in tc}
            res[tid] = "error" if "error" in kids else "failed" if "failure" in kids else "skipped" if "skipped" in kids else "passed"
    except ET.ParseError:
        return {}
    return res


def passed(out: str) -> set[str]:
    return {t for t, st in junit(out).items() if st == "passed"}


def stage_status(baseline_pass: set[str], out: str) -> tuple[str, set[str]]:
    """killed: a baseline-passing test no longer passes. survived: nothing killed and the run collected
    at least 90 % of the baseline's passes (so an aborted collection is 'error', not 'survived')."""
    res = junit(out)
    collected = {t for t in baseline_pass if t in res}
    if len(collected) < 0.9 * len(baseline_pass) or not baseline_pass:
        return "error", set()            # most of the baseline never ran: aborted collection, not a verdict
    k = killed_by(baseline_pass, out)
    return ("killed" if k else "survived"), k


def killed_by(baseline_pass: set[str], out: str) -> set[str]:
    """Tests that passed at baseline and RAN and did not pass: the mutation-testing kill criterion.
    A test absent from the run was not collected (aborted/errored collection), which is not a kill."""
    res = junit(out)
    return {t for t in baseline_pass if t in res and res[t] != "passed"}


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
        m = re.search(r"```(?:python|py)?\n(.*?)```", text, re.S)
        return m.group(1) if m else (text if "def test_" in text and "```" not in text else None)
    except Exception:
        return None


def process_pr(pr: dict, cfg: dict, per_pr: int, llm: bool, timeout: int, rng: random.Random, log) -> list[dict]:
    name = pr["repo"]; repo = WORK / "repos" / name; r = cfg[name]
    inst, net = r.get("candidate_install", DEFAULT_CANDIDATE_INSTALL), bool(r.get("network"))
    rows = []
    for f in pr["code_files"][:2]:
        lines = added_lines(repo, pr["sha"], f)
        src_lines = git(repo, "show", f"{pr['sha']}:{f}").splitlines()
        doc = docstring_lines(src_lines)
        # same filter for operator AND LLM candidates: no docstrings, doctest prompts, or comment lines
        lines = [(n, t) for n, t in lines if not SKIP_LINE.match(t) and n not in doc and ">>>" not in t and not t.lstrip().startswith("#")]
        if not lines:
            continue
        rel = related_tests(repo, pr["sha"], f, pr["test_files"], tuple(r.get("exclude_paths", [])))
        targets = rel or None                       # None -> whole suite
        base_out = run(name, pr["sha"], inst, net, timeout, tag=f"{name}_{pr['sha'][:8]}_baseline", edits=None, probe=None, pytest_targets=targets, want_diff=False)
        base_pass = passed(base_out)
        if not base_pass:
            log(f"{name:9s} {pr['sha'][:8]} {f}: baseline collected nothing (exit {exit_code(base_out)}; junit={len(junit(base_out))}), skip\n    "
                + base_out.split("JUNIT_BEGIN")[0].strip().replace("\n", "\n    ")[-900:])
            rows.append({**{k: pr[k] for k in ("repo", "sha", "parent", "subject", "lines")}, "file": f, "status": "baseline_empty",
                         "targets": targets, "baseline_tail": base_out[-1500:]}); break
        log(f"{name:9s} {pr['sha'][:8]} {f}: baseline {len(base_pass)} passing in {len(targets) if targets else 'all'} test files")
        base_full_pass = None                       # full-suite baseline, computed once if any mutant survives stage 1
        muts = op_mutants(lines, per_pr - (1 if llm else 0), rng, git(repo, "show", f"{pr['sha']}:{f}").splitlines())
        if llm:
            m = llm_mutant(repo, pr["sha"], f, lines, PROBE_MODEL)
            if m: muts.append(m)
        for m in muts:
            content = mutated_file(repo, pr["sha"], f, m["line"], m["mutated"])
            row = {**{k: pr[k] for k in ("repo", "sha", "parent", "subject", "lines")}, "file": f, **m,
                   "targets": targets, "baseline_passed": len(base_pass)}
            out1 = run(name, pr["sha"], inst, net, timeout, tag=f"{name}_{pr['sha'][:8]}_{m['line']}_{m['operator']}_s1", edits={f: content}, probe=None, pytest_targets=targets, want_diff=False)
            row["stage1"], k1 = stage_status(base_pass, out1)
            row["stage1_killers"] = sorted(k1)[:10]; row["stage1_tail"] = out1.split("JUNIT_BEGIN")[0][-1200:]; row["stage1_passed"] = len(passed(out1))
            if row["stage1"] != "survived":
                extra = "" if row["stage1"] == "killed" else " (exit " + exit_code(out1) + ")\n    " + out1.strip().replace("\n", "\n    ")[-500:]
                rows.append(row); log(f"{name:9s} {pr['sha'][:8]} {f}:{m['line']} {m['operator']:14s} {row['stage1']} {len(k1)}{extra}"); continue
            if base_full_pass is None:
                base_full_pass = passed(run(name, pr["sha"], inst, net, timeout, edits=None, probe=None, pytest_targets=None, want_diff=False))
            out2 = run(name, pr["sha"], inst, net, timeout, tag=f"{name}_{pr['sha'][:8]}_{m['line']}_{m['operator']}_s2", edits={f: content}, probe=None, pytest_targets=None, want_diff=True, parent=pr["parent"])
            row["stage2"], k2 = stage_status(base_full_pass, out2)
            row["stage2_killers"] = sorted(k2)[:10]
            row["stage2_tail"] = (out2.split("DIFF_BEGIN")[0] if "DIFF_BEGIN" in out2 else out2.split("JUNIT_BEGIN")[0])[-1500:]; row["stage2_passed"] = len(passed(out2)); row["baseline_full_passed"] = len(base_full_pass)
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
                jm, jo = junit(on_mut), junit(on_orig)
                def verdict(j):
                    if not j: return "error"
                    if any(v == "error" for v in j.values()): return "error"
                    return "fail" if any(v == "failed" for v in j.values()) else "pass"
                row.update(probe=probe, probe_on_mutant=verdict(jm), probe_on_original=verdict(jo))
                mut_fail, orig_pass = row["probe_on_mutant"] == "fail", row["probe_on_original"] == "pass"
                if mut_fail and orig_pass:
                    row["guard"] = "distinguished"; break
                fb = f"on mutant: {row['probe_on_mutant']}; on original: {row['probe_on_original']}\n{on_mut.split('JUNIT_BEGIN')[0][-600:]}"
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
                d = json.loads(l)
                if d.get("status") != "baseline_empty":      # env failures are retried after a repos.yaml fix
                    done.add(d["sha"])
    prs = []
    from review.mine.bisect import unshallow
    for name in cfg:
        if a.repo and name != a.repo:
            continue
        repo = WORK / "repos" / name
        if not (repo / ".git").exists():                 # repos added after the scan: clone now
            print(f"{name:10s} cloning …", flush=True)
            import subprocess
            subprocess.run(["git", "clone", "-q", cfg[name]["url"], str(repo)], check=True)
        unshallow(repo, cfg[name]["url"])                # --since reaches back further than the scan's shallow clone
        prs += clean_prs(repo, a.since, a.max_lines, tuple(cfg[name].get("exclude_paths", [])))
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
