# RUNBOOK — your side of each phase

Every pod step is: spin up the pod, paste me `host port`, say go. I drive the pod
over SSH from your Mac (`scripts/pod.sh`, key `~/.ssh/runpod`). You review the
numbers I report, decide improve/proceed, and stop the pod.

Pod template for every phase: **Runpod PyTorch** (CUDA 12.8), container disk
**80 GB**, no volume, **SSH over exposed TCP** on. Stop (don't just pause) when done.

## Phase 0 — GSM8K mechanics, 1× H100 80 GB (~3 h, ~$8–10)
1. Pod up → `scripts/pod.sh set <host> <port>` (I do this once you paste it).
2. `pod/setup.sh` → pins land in `pod/versions.lock` (committed afterwards).
3. `configs/gsm8k_smoke.yaml` (3 steps, 2 min). Pass = reward line prints, no OOM.
4. Base eval: `python -m ftlab.eval configs/gsm8k_h100.yaml --label base` (1,319 problems, <2 min).
5. `scripts/pod.sh run configs/gsm8k_h100.yaml` → 300 steps in tmux; I poll `train.log`.
6. Tuned eval: `--label tuned --model runs/gsm8k_h100/final`; `plot.py`; `pod.sh pull`.
7. Optional same pod: `configs/gsm8k_lora.yaml` → `--label lora --model runs/gsm8k_lora/merged`.
8. Stop pod. Commit `runs/gsm8k_h100/{summary.json,eval_*.json,curves.png,log_history.json}` with `git add -f`.
9. Drill (no pod): three questions from `docs/NOTES.md` § "How a GRPO step runs".

**Pass condition:** reward curve rises over 300 steps; tuned acc > base acc by
more than the CI half-width (~2.5 pts); you can point at G, the group baseline,
and `beta` in the trainer source.

## Phase 1 — Spider text-to-SQL, 1× H100 (~4 h, ~$12)
1. Pod up, `setup.sh`, then `scripts/get_spider.sh` (downloads ~100 MB of SQLite files).
2. `configs/spider_smoke.yaml`.
3. Base eval on dev (1,034). Also, from your Mac with `ANTHROPIC_API_KEY` in the
   shell: `--backend anthropic --model claude-haiku-4-5 --label haiku` and the same
   for Sonnet. Same prompt, same scorer.
4. `spider_h100.yaml` (400 steps) → tuned eval → `spider_lora.yaml` → lora eval.
5. `eval.py --table runs/spider_h100/eval_*.json runs/spider_lora/eval_lora.json`
   → the four-to-five-row table for Jeremy (Oct 27).

## Phase 2 — topology, 2× then 4× H100 (~1.5 h, ~$20)
A. 2× H100: GPU 0 runs `VLLM_SERVER_DEV_MODE=1 vllm serve …` (command in
   `configs/gsm8k_server_2gpu.yaml`), GPU 1 trains 50 steps with `vllm_mode: server`.
B. 4× H100: `accelerate launch --num_processes 4 -m ftlab.train configs/gsm8k_colocate_4gpu.yaml`.
Record `step_time` and $/1k completions for 1-colocate, 2-server, 4-colocate.
That's the paragraph for Zack: cost per use case at scale.

## Phase 3 — lakehouse text-to-SQL (after Oct 27; Snowflake trial ends ~Nov 6)
New adapter `ftlab/data/lakehouse.py`: 100–150 questions over `LAKEHOUSE.GOLD`/
`SILVER`, reference SQL, reference result snapshot; reward = runs in Snowflake and
result matches the snapshot. Same registry, same train.py.

## Day-to-day commands (Mac)
```
scripts/pod.sh set <host> <port>     # once per pod
scripts/pod.sh push                  # code -> pod
scripts/pod.sh ssh                   # shell on the pod
scripts/pod.sh run configs/x.yaml    # train in tmux
scripts/pod.sh log                   # tail training
scripts/pod.sh pull                  # results -> ./runs (no checkpoints)
make test                            # 23 pure-python tests, no GPU
```
