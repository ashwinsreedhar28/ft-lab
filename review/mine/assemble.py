"""Assemble the public bench + training set from mining outputs (git only; Mac, no Docker).

    python -m review.mine.assemble [--eval-repos attrs,click] [--out review/bench]

Inputs (review/mine/work/): mutants.jsonl (status=positive rows from mutate.py), positives.jsonl
(status=positive rows from bisect.py), validated.jsonl (for fix shas). Output: review/bench/
public_tasks.jsonl in the agent's Task schema, plus public_train.jsonl / public_eval.jsonl.

Rows produced
  mut__<repo>__<base8>__<line>__<op>   kind=bug    head = a real commit made here on branch
                                       mutant/<id> in the local clone (tree of the clean PR with the
                                       mutated file), parent = the PR's parent, so its diff IS the PR
                                       diff + the injected bug. oracle = the guard probe; gold hunk =
                                       the mutated line.
  clean__<repo>__<base8>               kind=clean  the clean PR itself (one per PR that yielded any
                                       mutant: size-matched negative by construction).
  real__<repo>__<intro8>               kind=bug    bisected introducing commit; oracle = the fix's
                                       tests; fix_sha for strong localization.

Split: by repo, never by task. --eval-repos names the held-out repos; everything from them is eval,
the rest is train. Real positives are always eval. A model trained on mutants is reported on the
mutant eval split AND on real positives separately; the real number is the sanity check, not the claim.

Harness fields for mounted clones (loop.py honors them): image=rb-mine-<repo>, repo_mount=<clone>,
setup_cmd (clone /repo -> /testbed, PYTHONPATH, per-repo candidate install), network, test_cmd with
the repo's pytest_args.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

from review.bench.build import gold_hunks, neighborhood, pr_diff
from review.mine.scan import git
from review.mine.validate import DEFAULT_CANDIDATE_INSTALL, IMG, WORK, repos

PROBE_PATH = "tests/test_probe_mutant.py"


def mutant_commit(repo: Path, base: str, parent: str, path: str, content: str, msg: str) -> str:
    """A commit whose tree = tree(base) with `path` replaced by `content`, parented on `parent`.
    Built with plumbing on a throwaway index so the working tree is untouched. Idempotent per message."""
    env = {"GIT_INDEX_FILE": str(repo / ".git" / "assemble-index")}
    import os
    full_env = {**os.environ, **env, "GIT_AUTHOR_NAME": "review-bench", "GIT_AUTHOR_EMAIL": "bench@local",
                "GIT_COMMITTER_NAME": "review-bench", "GIT_COMMITTER_EMAIL": "bench@local",
                "GIT_AUTHOR_DATE": "2026-10-17T00:00:00Z", "GIT_COMMITTER_DATE": "2026-10-17T00:00:00Z"}
    def g(*a, inp=None):
        return subprocess.run(["git", "-C", str(repo), *a], input=inp, capture_output=True, text=True, check=True, env=full_env).stdout.strip()
    g("read-tree", base)
    blob = g("hash-object", "-w", "--stdin", inp=content)
    g("update-index", "--cacheinfo", f"100644,{blob},{path}")
    tree = g("write-tree")
    sha = g("commit-tree", tree, "-p", parent, inp=msg)
    (repo / ".git" / "assemble-index").unlink(missing_ok=True)
    return sha


def harness_fields(name: str, cfg: dict) -> dict:
    r = cfg[name]
    inst = r.get("candidate_install", DEFAULT_CANDIDATE_INSTALL)
    return {"image": IMG.format(name=name), "repo_mount": str(WORK / "repos" / name),
            "setup_cmd": f"[ -d /testbed/.git ] || git clone -q /repo /testbed; cd /testbed && export PYTHONPATH=/testbed/src:/testbed && ({inst} >/dev/null 2>&1 || true)",
            "network": "bridge" if r.get("network") else "none",
            "test_cmd": f"python -m pytest -p no:cacheprovider -W ignore::DeprecationWarning -W ignore::ResourceWarning {r.get('pytest_args', '')}".strip(),
            "subdir": "."}


def files_of(repo: Path, sha: str) -> list[str]:
    return [f for f in git(repo, "show", "--name-only", "--format=", sha).split() if f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-repos", default="attrs,click")
    ap.add_argument("--out", default="review/bench")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = repos(); eval_repos = set(a.eval_repos.split(","))
    out = Path(a.out); out.mkdir(exist_ok=True)

    def rows(p):
        f = WORK / p
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []

    tasks, seen_clean = [], set()
    for m in rows("mutants.jsonl"):
        if m.get("status") != "positive" or m["repo"] not in cfg:
            continue
        repo = WORK / "repos" / m["repo"]
        mid = f"mut__{m['repo']}__{m['sha'][:8]}__{m['line']}__{m['operator']}"
        content = git(repo, "show", f"{m['sha']}:{m['file']}").splitlines(keepends=True)
        nl = "\n" if content[m["line"] - 1].endswith("\n") else ""
        content[m["line"] - 1] = m["mutated"].rstrip("\n") + nl
        sha = mutant_commit(repo, m["sha"], m["parent"], m["file"], "".join(content), f"{m['subject']}\n\n[review-bench mutant {mid}]")
        subprocess.run(["git", "-C", str(repo), "branch", "-f", f"mutant/{mid}", sha], check=True, capture_output=True)
        hf = harness_fields(m["repo"], cfg)
        tasks.append({"id": mid, "repo": m["repo"], "kind": "bug", "base_sha": m["parent"], "head_sha": sha,
                      "diff": m.get("bug_diff") or pr_diff(repo, sha), "files": files_of(repo, sha),
                      "split": "eval" if m["repo"] in eval_repos else "train",
                      "oracle": [PROBE_PATH], "probe_source": m.get("probe", ""),
                      "gold_hunks": [{"file": m["file"], "line_start": m["line"], "line_end": m["line"]}],
                      "neighborhood": neighborhood(repo, sha), "fix_sha": m["sha"],   # the clean original IS the fix
                      "source": m["source"], "operator": m["operator"], "uncovered": [], "covered_frac": None, **hf})
        if m["sha"] not in seen_clean:
            seen_clean.add(m["sha"])
            tasks.append({"id": f"clean__{m['repo']}__{m['sha'][:8]}", "repo": m["repo"], "kind": "clean",
                          "base_sha": m["parent"], "head_sha": m["sha"], "diff": pr_diff(repo, m["sha"]),
                          "files": files_of(repo, m["sha"]), "split": "eval" if m["repo"] in eval_repos else "train",
                          "oracle": [], "probe_source": "", "gold_hunks": [], "neighborhood": neighborhood(repo, m["sha"]),
                          "fix_sha": "", "source": "clean", "operator": "", "uncovered": [], "covered_frac": None, **hf})

    for p in rows("positives.jsonl"):
        if p.get("status") != "positive" or p["repo"] not in cfg:
            continue
        repo = WORK / "repos" / p["repo"]; hf = harness_fields(p["repo"], cfg)
        intro = p["introducing"]
        tasks.append({"id": f"real__{p['repo']}__{intro[:8]}", "repo": p["repo"], "kind": "bug",
                      "base_sha": p["introducing_parent"], "head_sha": intro, "diff": pr_diff(repo, intro),
                      "files": files_of(repo, intro), "split": "eval", "oracle": p["oracle_tests"], "probe_source": "",
                      "gold_hunks": gold_hunks(repo, intro, p["sha"]), "neighborhood": neighborhood(repo, intro),
                      "fix_sha": p["sha"], "source": "real", "operator": "", "uncovered": [], "covered_frac": None, **hf,
                      "bug_age_commits": p.get("bug_age_commits"), "boundary": p.get("boundary")})

    random.Random(a.seed).shuffle(tasks)
    (out / "public_tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    for split in ("train", "eval"):
        (out / f"public_{split}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks if t["split"] == split))
    c = Counter((t["split"], t["kind"], t["source"]) for t in tasks)
    print(f"{len(tasks)} tasks -> {out}/public_tasks.jsonl")
    for k in sorted(c):
        print(f"  {k[0]:5s} {k[1]:5s} {k[2]:6s} {c[k]:4d}")
    by_repo = Counter((t["repo"], t["split"]) for t in tasks)
    print("  by repo: " + ", ".join(f"{r}/{s}={n}" for (r, s), n in sorted(by_repo.items())))


if __name__ == "__main__":
    main()
