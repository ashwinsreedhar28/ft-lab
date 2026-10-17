# RUNBOOK — your side of each phase

Neither Claude's cloud shell nor the Mac-side shell can open SSH to a pod (both are
HTTPS-only), so the pod drives itself. Your loop per phase:

1. Spin up the pod (template below). Copy its SSH host and port.
2. In your Mac terminal, paste ONE command (below). It clones the repo, installs,
   and runs the whole phase in tmux. The pod pushes logs/results to branch
   `results/<phase>` every 5 min; Claude reads that branch and reports.
3. When Claude says the phase is done (or you see `PHASE-DONE` in `tmux attach -t ft`),
   stop the pod. Claude pulls results into `runs/` and writes the numbers up.

One-time: create a fine-grained GitHub PAT scoped to `ashwinsreedhar28/ft-lab`,
Contents: read & write, 30-day expiry. Keep it in your shell as `$FTLAB_PAT`
(e.g. `export FTLAB_PAT=...` in a terminal you don't share). Never paste it in chat.

The command (fill HOST, PORT, phase):
```
ssh -i ~/.ssh/runpod -p PORT root@HOST "export GH_TOKEN=$FTLAB_PAT; curl -sL https://raw.githubusercontent.com/ashwinsreedhar28/ft-lab/main/pod/bootstrap.sh | bash -s phase0"
```
Re-running the same command is safe: it attaches to the running phase instead of
restarting, and each stage is skipped once finished (`runs/<phase>.stages`).

Pod template for every phase: **Runpod PyTorch** (CUDA 12.8), container disk
**80 GB**, no volume, **SSH over exposed TCP** on. Stop (don't just pause) when done.

## Phase 0 — GSM8K mechanics, 1× H100 80 GB (~3–4 h incl. LoRA, ~$10)
`bash -s phase0` runs, in order: `pod/setup.sh` (pins → `pod/versions.lock`) → 3-step
smoke → base eval on 1,319 → 300-step GRPO → tuned eval → `curves.png` → LoRA run +
eval (set `SKIP_LORA=1` in the ssh command's export to skip) → `table.txt`.
Afterwards (no pod): `scripts/results.sh phase0`, Claude writes the numbers into
README, you do the drill in `docs/NOTES.md` / the study artifact.

**Pass condition:** reward curve rises over 300 steps; tuned acc > base acc by
more than the CI half-width (~2.5 pts); you can point at G, the group baseline,
and `beta` in the trainer source.

## Phase 1 — review-bench (Mac + Docker, ~no GPU)
Design in `review/README.md`; builder in `review/bench/build.py`. Git-only stages Claude runs
from its shell; the two Docker stages you run in your Mac terminal (Docker Desktop up,
shwin-bench images built via its `make setup`):
```
cd ~/ft-lab
make rb-prepare            # done Oct 17: 15 fix commits, pools 84/114/37
make rb-bisect             # YOU, docker: ~15 fixes x up to 60 ancestors x seconds each
make rb-match              # Claude
make rb-verify             # YOU, docker: 2 suite runs per negative
make rb-graph              # Claude -> review/bench/tasks.jsonl
export ANTHROPIC_API_KEY=...; make rb-baselines   # YOU: Haiku + Sonnet x {diff, diff+graph}
```
Expect attrition in bisect: emberserve was 5 days old at these commits, so some bugs trace to
the initial dump (>400 lines, dropped). 8–12 positives out of 15 is a fine first bench.

## Phase 2 — the review agent (Mac + Docker; Sonnet then Haiku)
`review/agent`: neighborhood → run suite → probe test if unsure → decision. Iterate the
harness until bugs_approved = 0 on review-bench with the highest auto_ship_rate, then swap
Haiku in and record what breaks. **Phases 1+2 are the Oct 27 deliverable.**

## Phase 3 — fine-tune into the harness (CPU pod for mining, then 1× H100, ~$40)
Mining recipe in `review/mine/README.md`: public repos, commits after 2026-06-01, bisected
positives + matched negatives, ~400 train / 100 eval, split by repo. Then GRPO a Qwen3-4B
or 8B LoRA into the triage-gate role (`ftlab/rewards/review.py`, asymmetric reward).
Report auto-ship at zero bugs approved and $/PR vs the frontier harness, 3 seeds.

## Phase 4 — topology, 2× then 4× H100 (~1.5 h, ~$20), if budget remains
A. 2× H100: GPU 0 runs `VLLM_SERVER_DEV_MODE=1 vllm serve …` (command in
   `configs/gsm8k_server_2gpu.yaml`), GPU 1 trains 50 steps with `vllm_mode: server`.
B. 4× H100: `accelerate launch --num_processes 4 -m ftlab.train configs/gsm8k_colocate_4gpu.yaml`.
Record `step_time` and $/1k completions for 1-colocate, 2-server, 4-colocate.

## Backlog (configs kept, not scheduled)
Spider / BIRD text-to-SQL (`configs/spider_*.yaml`, `configs/bird_eval.yaml`, `pod/phase1.sh`
still runs it as a harness check if wanted), lakehouse text-to-SQL adapter.

## Day-to-day commands (Mac)
```
scripts/results.sh phase0            # pull the pod's pushed results into ./runs
scripts/pod.sh set/ssh/push/log      # manual pod access if you want it yourself
make test                            # 27 pure-python tests, no GPU
```
On the pod: `tmux attach -t ft` (phase), `tmux attach -t hb` (heartbeat), ctrl-b d to detach.
