# ft-lab

GRPO fine-tuning experiments on Runpod GPUs: one script, swappable tasks, measured
against a frozen held-out split and frontier-model baselines.

End goal: a code review agent whose approval is the gate to prod, built eval-first on my
own repos (shwin-bench), then a small model fine-tuned into the roles where it holds the number.

| phase | what | signal | status |
|---|---|---|---|
| 0 | GRPO mechanics: GSM8K, Qwen3-1.7B, 1×H100 | answer match | done Oct 17; lessons below |
| 1 | review-bench from shwin-bench: bisected bug PRs + matched clean PRs, 4 context conditions, frontier baselines | bugs approved (target 0), auto-ship rate | designed (`review/`) |
| 2 | the review agent: neighborhood → run suite → probe test → decision | same scorer | planned |
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
