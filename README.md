# ft-lab

GRPO fine-tuning experiments on Runpod GPUs: one script, swappable tasks, measured
against a frozen held-out split and frontier-model baselines.

End goal: a code review agent whose approval is the gate to prod, built eval-first on my
own repos (shwin-bench), then a small model fine-tuned into the roles where it holds the number.

| phase | what | signal | status |
|---|---|---|---|
| 0 | GRPO mechanics: GSM8K, Qwen3-1.7B, 1×H100 | answer match | done Oct 17; lessons below |
| 1 | review-bench from shwin-bench: bisected bug PRs + matched clean PRs, 4 context conditions, frontier baselines | bugs approved (target 0), auto-ship rate | v0 built: 4 bug + 4 clean; see below |
| 2 | the review agent: neighborhood → run suite → probe test → decision | same scorer | harness v2 running; see below |
| 3 | mine fresh public repos (~400 PRs); SFT warm-start on frontier review traces (TEMP-selected); GRPO a 4B/8B LoRA into the triage gate with variance-guided rollouts (VIGOR) | asymmetric decision reward | recipe written |
| 4 | topology: 1-colocate / 2-server / 4-colocate | $/1k completions | configs ready |
| backlog | Spider / BIRD / lakehouse text-to-SQL | execution match | adapters kept |

```
pip install -r requirements.txt          # or bash pod/setup.sh on a pod
make test                                # 27 pure-python tests
python -m ftlab.train configs/gsm8k_h100.yaml
python -m ftlab.eval  configs/gsm8k_h100.yaml --label tuned --model runs/gsm8k_h100/final
python -m ftlab.plot  runs/gsm8k_h100
```

Layout: `ftlab/config.py` (YAML → dataclass, `extends:` chains), `ftlab/data/*`
(task adapters: load + `score`), `ftlab/rewards/*` (registry of reward functions),
`ftlab/train.py` (TRL `GRPOTrainer`, vLLM colocate/server, optional LoRA),
`ftlab/eval.py` (greedy vLLM or Anthropic backend, Wilson CI), `ftlab/plot.py`.
Operational steps in `RUNBOOK.md`; design notes and a GRPO primer in `docs/NOTES.md`.

## Phase 0: what the GSM8K run taught (Oct 17, 2026, 1× H100, ~$3)

Not a results table; the accuracy numbers below are not a claim. Qwen3-1.7B, thinking off, full
fine-tune, GRPO 300 steps × 64 completions (4 prompts × 16), lr 1e-6, β 0.04, 2.9 s/step.

- **The harness works end to end** on a real pod: TRL 1.16 + vLLM 0.31 colocated, self-driving pod
  with a results branch, eval with Wilson CIs. Seven script fixes on first contact; now stable.
- **The reward curve was flat** (correct ≈ 0.85 at T=1 throughout, KL < 0.01). The base model is at
  79 % on GSM8K, so a quarter to three quarters of each step's prompts had zero within-group
  variance and contributed no gradient. A task the base already does is the wrong place to look for
  a curve; the hard-pool config and variance-guided rollout allocation (VIGOR) exist for this reason.
- **The policy bought reward with length.** Under a 768-token decode budget the tuned model scored
  75.4 % (base 79.0) because 211 completions hit the cap. At 1,536 tokens it scored 84.7 %
  (CI 82.6–86.5) at 5.4× the output tokens of the base, bimodally: completions under 2k chars are
  96–98 % right, those over 4k chars 32 %. It also broke 79 answers the base had right by looping.
  +5.7 points at 5× decode cost loses to serving a bigger model at 1×. The lesson for every later
  reward: price tokens and abstentions in the reward, or the policy pays for reward with them.
- **LoRA (r=32) matched full fine-tune** within CI (76.2 vs 75.4 at 768 tokens). One task, one seed;
  weak support for the one-base-many-adapters pattern, not proof.
- **Eval protocol is part of the method.** Decode budget and temperature must be fixed before
  training, not chosen after seeing which setting looks best (this run chose 1,536 after the fact;
  the hard-pool run reports at both budgets, decided in advance). Report tokens per answer next to
  accuracy. One seed is an anecdote; `eval.py --table` collapses `<label>_s<seed>` for a reason.
- **What this changes downstream:** model size for the review agent is chosen by $/PR at the target
  reliability (start at 4B/8B), not inherited from here; the review reward gets a token-cost term;
  the 1.7B stays as the smoke-test model.

Raw artifacts: `runs/gsm8k_h100/{eval_*.json, curves.png, log_history.json}` on branch
`results/phase0` and `results/phase0b`.

## Phases 1–2: review-bench v0 and the first harness (Oct 17, 2026, Mac + Docker + API, ~$15)

**Bench.** From shwin-bench's 15 fix commits: 4 were entirely interface-leak tasks (the hidden test
imports a name the fix adds, so it errors at every pre-fix commit; no bisect signal). Per-test bisect
over the full history found the commit where each oracle test first fails; a fail→error boundary
(test can't import any earlier) means the bug was born with the module, and that creating commit is
the PR. Most of emberserve's bugs trace to its 4,427-line initial dump and aether's to a 57k-line
one; dropped. One positive (aether `f0a823d`) was an import-boundary artifact: the test's *imports*
became satisfiable there, but the commit touched none of the files the fix touched; every reviewer
approved it and was right to. Rule added: an introducing commit must share a code file with its fix.
Result: **4 bug PRs** (285–1,155 changed lines; two are themselves fix commits that introduced a
different bug) and **4 size/file-matched clean PRs**, all suite-green at parent and commit. Gold
hunks are narrowed to the functions the fix touched. 0/4 bugs approved bounds the rate at ~1 in 2
(rule of three), so everything below is a design validation, not a reliability claim.

**Diff-only reviewers ship bugs.** Same 5+5 PRs (before the artifact was removed), single call,
adaptive thinking: Sonnet 5.5 approved 3/5 bug PRs reading the diff, 3/5 with the two-hop
neighborhood added; Haiku 5.5 approved 2/5 and 1/5. When they blocked, they blocked the right
hunk (localization 1.0). The frontier model's failure is confidence, not comprehension.

**A green suite made it worse.** Harness v1 (suite result at the PR head + optional probe test):
Sonnet approved 3/5 bugs and auto-shipped 60 % of clean PRs; probes were written for 1 PR in 10
and were trivial (`test_flags_exist`). The suite at a bug-introducing commit is green by
construction (the catching test doesn't exist yet), which is the state of every bug that ever
shipped, and the reviewer read "tests pass" as evidence.

**Harness v2: evidence rules.** The suite is presented as "pre-existing tests still pass, which says
nothing about the change"; changed definitions with no referencing test are listed; an approve is
accepted only with a passing probe that calls a changed function by name, otherwise it is bounced
for one more round and then escalated. Sonnet on the 8-PR bench: **0/4 bugs approved, 2/4 clean
PRs auto-shipped, 5/8 escalated, $0.10/PR**, every approve backed by a specific probe. Same model,
same PRs, diff-only: 3/5 bugs approved. The harness, not the model, moved the number. This is the
"harness engineering" loop: the scoreboard says what the agent needs to see and be forbidden to do.

Raw: `review/bench/tasks.jsonl`, `review/agent/results/` (v1 runs under `results/v1/`).
Next: scale the bench by mining fresh public repos (`review/mine/`), then a small model in the
gate role under the same harness.
