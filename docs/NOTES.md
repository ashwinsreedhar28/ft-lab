# ft-lab notes (written Oct 17, 2026, before the first pod run)

## What this repo is
One GRPO training script that works unchanged from 1 GPU to N, with the task
(data + reward + prompt) swapped by YAML. Phase 0 proves the trainer on GSM8K;
phase 1 does the thing Runpod would actually fine-tune for (text-to-SQL with an
execution reward, the seed of the data-agent fine-tune); phase 2 measures the
topologies you'd run at org scale; phase 3 points it at the lakehouse's own tables.

## Design decisions
- **Full fine-tune by default, LoRA as a flag.** On an H100 a 1.7B full FT with bf16
  weights, fp32 Adam states (~14 GB) and a colocated vLLM at 35 % fits with room.
  `*_lora.yaml` runs the same config with an r=32 adapter so the two can be compared
  on the same frozen split: that comparison is the question you face per use case
  at org scale (one base, an adapter per task, served multi-LoRA).
- **KL on (`beta: 0.04`).** TRL's default is 0 and several papers say it's not needed.
  We keep it so the `kl` curve exists and you can read the term in the loss. Expect
  it to climb slowly; if the reward is flat and KL is high, lower `beta`'s partner
  `learning_rate`, not `beta`.
- **`loss_type: dapo`** (token-level normalization). The original `grpo` loss
  divides each sequence by its own length, which under-penalizes long wrong
  answers. `dr_grpo` divides by a constant; try it if lengths creep up.
- **Rewards are plain functions with `**kwargs`** so one registry serves every task.
  The correctness reward calls the task adapter's `score()`, and so does eval:
  training reward and eval metric are one function.
- **Conversational prompts + `chat_template_kwargs={"enable_thinking": False}`.**
  Qwen3 would otherwise emit a `<think>` block; thinking-off keeps completions
  short and makes format learnable in a few hundred steps. Eval renders the prompt
  with the same tokenizer and flag (`ftlab/prompting.py::render`).
- **Frozen eval before any training.** GSM8K test (1,319) and Spider dev (1,034) are
  disjoint from train by construction. Frontier baselines (Haiku / Sonnet) run the
  identical prompt through `eval.py --backend anthropic`.
- **Wilson 95 % CI on every accuracy.** With n ≈ 1,300 the CI half-width is about
  ±2.5 points; a "lift" inside that is noise.
- **Spider correctness = execution match**, order-sensitive only when the gold query
  has ORDER BY. Same rule as the official Spider execution accuracy. Queries run
  read-only with a 5 s timeout and a 5,000-row cap.
- **Versions are pinned by the pod, not by me.** `pod/setup.sh` installs, writes
  `pod/versions.lock`, and asserts every key in `configs/base.yaml` is a real
  `GRPOConfig` field for that TRL. (Checked against TRL 1.16.0 locally on Oct 17:
  `max_prompt_length` is gone, replaced by `vllm_max_model_length`.)

## How a GRPO step runs (read with `trl/trainer/grpo_trainer.py` open on the pod)
1. **Group sampling** — `_generate_and_score_completions`. Each prompt is repeated
   `num_generations` (G=16) times and vLLM samples G completions at temperature 1.
   A "step" is `per_device_train_batch_size` completions = 64 → 4 prompts × 16.
   Why groups: the baseline for "was this completion good?" is the other 15
   answers to the *same* prompt, so no value network is needed.
2. **Reward** — `_calculate_rewards`. Every reward function gets `completions` plus
   every dataset column (`answer`, `gold_sql`, …) as lists; the trainer sums them
   (weighted by `reward_weights`). Ours: `gsm8k_correct` ∈ {0,1} + `gsm8k_format`
   ∈ {0,0.2}. The format term exists only so the first steps have gradient before
   the model gets anything right.
3. **Advantage** — in the same function: `A_i = (r_i − mean_group) / std_group`
   (`scale_rewards: group`). A prompt where all 16 are right or all 16 are wrong
   has std 0 → advantage 0 → contributes nothing. `frac_reward_zero_std` in the log
   is that fraction; if it is near 1 the task is too easy or too hard for the model.
4. **Loss** — `_compute_loss` / `compute_loss`. Per token:
   `ratio = exp(logp_θ − logp_old)`, clipped to `[1−ε, 1+ε]` (ε = 0.2), times the
   advantage, minus `beta · KL`. With `num_iterations: 1`, `θ_old = θ` so the ratio
   is 1 and the clip never binds; the clipped form only matters when μ > 1.
5. **KL term** — `_get_per_token_logps` on the reference model (a frozen copy of the
   base, or the adapter-disabled base when using LoRA), then the Schulman k3
   estimator `exp(ref − θ) − (ref − θ) − 1`, which is ≥ 0 and unbiased.
6. **Train/inference mismatch** — vLLM's sampling logprobs differ slightly from the
   trainer's; `vllm_importance_sampling_correction: true` reweights tokens by the
   truncated ratio. Watch `sampling/importance_sampling_ratio/mean` ≈ 1.

Things to be able to say from the code: where G enters (`num_generations`), where
the baseline comes from (group mean, not a critic), why std-0 groups vanish, what
`beta` multiplies, and why `num_iterations=1` makes the clip inert.

## What to expect on GSM8K (Qwen3-1.7B, thinking off)
Base accuracy somewhere around 60–75 % greedy. Reward should rise within 50 steps
mostly from the format term, then correctness climbs more slowly. A clean run
shows `completions/mean_length` falling early (dropping chatter) and
`completions/clipped_ratio` near 0. If `kl` runs past ~0.5 the policy is drifting;
if reward stalls with `frac_reward_zero_std` > 0.8, raise `temperature` to 1.1 or
pick harder prompts.

## What to expect on Spider
Base execution accuracy for a 1.7B zero-shot is likely 40–55 % on dev; the schema
prompt is the main lever. `sql/no_sql_frac` should go to 0 within a few steps
(the fence format is easy). `sql_runs` (0.1 shaping) makes "runs at all" learnable
before "runs and is right". Watch for reward hacking on the shaping term: a model
emitting `SELECT 1` runs but never matches, so the 1.0 term still dominates.

## Open items (answer during phase 0)
- Does colocate at 0.35 fit with 64 × 768-token completions on the H100? If OOM:
  `vllm_gpu_memory_utilization: 0.3`, `gradient_accumulation_steps: 2` with
  `per_device_train_batch_size: 32`.
- Step time on the H100 (expect 20–40 s). 300 steps ≈ 2–3 h.
- Whether `report_to: wandb` is worth it; `log_history.json` + `plot.py` is enough
  for the write-up.

## Spending rollouts where the gradient is (added Oct 17 after phase 0's first run)
Phase 0 showed `frac_reward_zero_std` at 0.25–0.75: with a base model at 79 % on GSM8K, a
quarter to three quarters of each step's prompts had all-correct or all-wrong groups and
contributed zero policy gradient. Two papers from Mirzasoleiman's group (UCLA) formalize the fix
at each stage:
- **VIGOR** (Jiang, Liu, Mirzasoleiman, COLM 2026): with binary rewards the GRPO gradient norm
  is bounded by within-group reward std (their Thm 1). Allocate rollouts progressively: m0=2 per
  prompt, keep the top α=0.5 by variance, multiply budget by γ=2, T=4 rounds, feed all rollouts
  to one GRPO update. Same total budget; 1.5–2.8× fewer rollouts to reach a target accuracy;
  accuracy gains themselves are small (0.5–1.2 pts on math). KL off in their setup. Compared
  against GRESO and RL-ZVP (same idea, different selection rules) and PODS.
  Where it matters for us: phase 3 (long diffs, 8B), not GSM8K at 3 s/step. One-round variant
  is enough to start: sample 4, drop zero-variance prompts, resample survivors to 16.
  `configs/gsm8k_hard.yaml` is the static, offline version of the same idea (pick prompts with
  ≥5 calculator steps); online variance selection is strictly better.
- **TEMP** (Jin, Yang, Ghaffari, Morato, Mirzasoleiman, ICML 2026): SFT data selection using
  only the first reasoning tokens. Difficulty = loss on the first ~100 response tokens at a
  randomly perturbed checkpoint; diversity = cluster the loss trajectory over the first ~1k
  tokens across a few noisy checkpoints, sample across clusters. ~1k examples; up to 1.7 pts
  over baselines, 91 % fewer tokens processed than full-trace scoring. SFT only; no RL, no LoRA.
  Where it matters: the SFT warm-start of the review agent (phase 3 step 1) if we mine more
  frontier review traces than we can afford to train on.
SFT vs GRPO in one line: SFT trains on (prompt, gold answer) by cross-entropy, no sampling, cheap,
bounded by the demonstrations; GRPO samples G attempts, scores them with a verifier, pushes toward
the better ones, expensive, can exceed the demonstrations. Pipeline for the review agent: SFT for
the output format and rough judgment, GRPO for the gate decision under the asymmetric reward.
