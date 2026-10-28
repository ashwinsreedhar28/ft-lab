# review-bench and the review agent (phases 1–2)

Goal: a code review agent whose approval is the gate to prod ("actually reliable"),
built eval-first on Ashwin's own repos, then a small model fine-tuned into the
roles where it holds the number.

## Bench (review/bench)
Source: shwin-bench's 42 validated tasks (base_commit with a real defect, gold patch,
hidden FAIL_TO_PASS tests). Two things turn them into review tasks without leakage:

1. **Bisect to the introducing commit, don't reverse the fix.** `reverse(gold)` yields
   a diff that *removes a guard*; every positive would share that shape and a classifier
   would learn the shape. Instead: apply `test_patch` at each ancestor of `base_commit`
   and find the first commit where a FAIL_TO_PASS test fails (SZZ with an executable
   oracle). That commit's diff is a real PR that shipped a real defect. Tasks whose bug
   predates the test's importability are dropped (expected: some).
2. **Matched negatives.** For each positive, a commit from the same repo whose full
   suite passes before and after, chosen to match the positive's diff size (±30 % lines)
   and overlap its file set when possible. 1:1.

Output: `review/bench/tasks.jsonl`, one row per PR:
`id, repo, kind (bug|clean), base_sha, head_sha, diff, files, oracle (FAIL_TO_PASS tests
for bug rows), gold_hunks (file:line ranges for bug rows), neighborhood (two-hop context
graph serialization, built by review/bench/graph.py), split (fresh|own-older)`.

Scorer (`review/bench/score.py`): a reviewer emits `{"decision": "approve"|"block"|"escalate",
"comments": [{"file","line_start","line_end","body","proposed_test"?}]}`. Metrics:
bugs_approved (primary, target 0), auto_ship_rate (clean approved), escalation_rate,
block_precision, localization (comment overlaps a gold hunk), strong_localization
(proposed_test fails at head_sha and passes at base_sha), cost per PR.
Statistical note: 0/42 approved bounds the true rate at ~7 % (rule of three). Say "under 1 in 14".

Conditions, same PRs, same scorer:
- `diff`      : the diff and file list only
- `diff+run`  : plus the ability to run the suite / a probe test in the task's Docker image
- `diff+graph`: plus the two-hop neighborhood
- `full`      : both

## Agent (review/agent)
A loop, not a prompt: gather neighborhood → run suite → if uncertain, write a probe test
and run it → decide. Frontier model first (Sonnet), iterate the harness until
bugs_approved = 0 with the highest auto_ship_rate; then Haiku; then (phase 3) a tuned
Qwen3-4B/8B in the triage-gate role: approve only what is provably safe, escalate the rest.

Harness v6 (`review/agent/harness.py` + `review/tools/`): the same gate reading cached findings from review tools
(static, test-suite delta, AI reviewer) under evidence policies. See the top-level README.

## Phase 3 reward (GRPO, ftlab/rewards/review.py)
-5.0 approve a bug PR · +1.0 correct approve or block · -0.5 block a clean PR ·
-0.2 escalate anything · +0.5 when a block's comment lands on a gold hunk.
Training data: public repos, commits after 2026-06 that fix a test-covered bug (fresh by
date), same bisect + match recipe, target ~400 PRs. Labels are precomputed, so no tests
run during training.
