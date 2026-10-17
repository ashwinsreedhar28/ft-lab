# Mining fresh bug-introducing PRs from public repos (phase 3 training data)

Why public repos: shwin-bench gives 42 positives, enough to design the scorer and bound
"bugs approved" at ~7 %. Training a decision model needs hundreds, and a reliability claim
tighter than 1-in-14 needs hundreds in eval too. Public repos supply volume; a date filter
keeps them fresh.

## Pipeline as built (Oct 17, 2026)

Each stage is a `make` target, reads the previous stage's JSONL under `work/`, and is a filter.
Per-repo config (URL, install line, network, excluded paths) is `repos.yaml`.

| stage | target | where | does | output |
|---|---|---|---|---|
| scan | `make mine-scan` | git only | `--shallow-since` clone; non-merge commits whose subject matches fix/bug/regress/crash/… and not revert/bump/docs/…, touching ≥1 test `.py` and ≥1 code `.py`, ≤400 lines | `work/candidates.jsonl` |
| images | `make mine-images` | Docker | one `rb-mine-<repo>` image per repo: python:3.12-slim + build backends + the repo's `install` line at the clone's HEAD | docker images |
| validate | `make mine-validate` | Docker | at the **parent** with the fix's test files overlaid, and at the **fix**: run just those files; classify (grid below) | `work/validated.jsonl` |
| bisect | `make mine-bisect` | Docker | for each valid fix: unshallow the clone, gallop back along first-parent ancestors with the tests overlaid until the oracle stops failing, binary-search the bracket | `work/positives.jsonl` |
| assemble | (to write) | git + Docker | matched clean PRs, neighborhood, gold hunks, coverage, as in `review/bench/build.py` | `review/bench/public_tasks.jsonl` |

### Validate: the 2×2
Rows = oracle tests at the parent (tests overlaid), columns = at the fix.

| | at fix: all pass | at fix: not all pass |
|---|---|---|
| parent: ≥1 **fails** (assertion) | **valid**; oracle = the tests that flip fail→pass | not_green |
| parent: only **errors** (can't import) | interface_leak: the test imports a name the fix adds; no bisect signal | not_green |
| parent: all **pass** | not_reproduced: the "fix" changed no tested behavior | not_green |

`not_green` has always meant the environment, not the commit: a missing pytest plugin
(`pytest-run-parallel` markers), a pinned compiled dependency (pydantic refuses to import unless
`pydantic-core` matches exactly; solved with `network: true` + a deps-on candidate install), a Rust
subproject (`exclude_paths`), or network-only fixtures (requests/httpbin). Fix the install line,
rebuild that image, delete that repo's rows, rerun.

First run, 39 candidates since 2026-06-01 from 8 repos: **24 valid** (click 4/6, fastapi 9/12,
pydantic 10/19, attrs 1/1, requests 0/1; jinja/rich/httpx had no candidates), 1 interface leak,
2 not reproduced, 11 not green, 1 excluded. 62 % yield.

### Bisect: finding the introducing commit
Chain = `git rev-list --first-parent <parent>`; index 0 is the fix's parent, where the oracle
fails by construction. A probe at index *i* = check out `chain[i]`, overlay the fix's test files,
`pip install -e . --no-deps --no-build-isolation`, run the oracle's test files, read the oracle ids:
`fail` (≥1 assertion failure), `pass`, `error` (can't collect/import), or `docker_error` (can't
build there). Search:

1. **Gallop**: probe 64, 128, 256, … back while the result is `fail`. Stops at the first non-fail.
2. **Binary search** the bracket [last fail, first non-fail] to width 1.
3. `introducing = chain[lo]` (fails), `boundary = status(chain[hi])`.

Cost ≈ 2·log₂(age) container runs, ~10 for a bug 200 commits old; never probes much older than
2× the bug's age, so it rarely touches ancient commits where the install breaks (and when it does,
that `docker_error` is just the bracket's far end, `unrunnable_boundary` if it ends up adjacent).
Plain binary search over the whole chain would be 12 probes on a 3,400-commit repo and must build
at the far end, which failed on click's 2014 initial commit. Assumes the oracle is monotone along
the chain; every probed sha and status is stored, so a flaky oracle shows up as a `non_monotone`
row rather than a wrong answer.

Then the shwin-bench rules: `boundary == error` means the test can't import at the older commit,
so the introducing commit created the module (kept, flagged); the introducing commit must change
a non-test `.py` the fix also changes (`no_shared_file` otherwise: an import-boundary artifact,
the rule that dropped aether `f0a823d`); `introducing_lines > 1500` is `too_big`. `bug_age_commits`
is recorded; mature-repo bugs are old (click's four all predate the 4-month scan window, so the
clone is deepened to full history first; `--shallow-since` clones can't be deepened and are
re-cloned).

### Then (assemble, to write)
Matched negative per positive from the same repo (size ±30 %, file-set overlap, suite green at
parent and commit, not itself a fix), two-hop neighborhood, gold hunks = the introducing commit's
hunks in defs the fix touched, coverage at the PR head for the uncovered-lines cue. Split by repo.

## Cost
CPU, Docker, and time. One repo ≈ 30–90 min of machine time for steps 2–6 on a laptop or a
cheap CPU pod ($0.10–0.20/h on Runpod). Budget ~20 repo-hours → ~$4 of pod time or a
weekend of the Mac. No GPU until the GRPO run.

## Leakage checklist (why each step exists)
- reverse(fix) shape → **bisect**: positives are the diffs that actually shipped the bug
- positives smaller than negatives → **size match**
- positives in bug-prone files only → **file-set preference** for negatives
- model has seen the repo's history → **date filter** (fresh commits), and report the
  own-older vs fresh split like shwin-bench does
- same repo in train and eval → **split by repo**
- test file in the PR gives the answer away → positives and negatives both include their
  test changes as they were; the oracle test from the *fix* is never shown

## Selection bias to remember
`scan.py` keeps commits whose subject matches a fix vocabulary and that touch both a test and a
code file, ≤400 lines. That selects for small bugs the author recognized and tested as bugs. The
bench will under-represent silent regressions, bugs fixed without a new test, and bugs fixed inside
feature commits. Fine for a first bench of real positives; say so in any write-up, and do not treat
a rate on it as a rate on "PRs in general".
