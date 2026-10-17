# ft-lab

GRPO fine-tuning experiments on Runpod GPUs: one script, swappable tasks, measured
against a frozen held-out split and frontier-model baselines.

End goal: a code review agent whose approval is the gate to prod, built eval-first on my
own repos (shwin-bench), then a small model fine-tuned into the roles where it holds the number.

| phase | what | signal | status |
|---|---|---|---|
| 0 | GRPO mechanics: GSM8K, Qwen3-1.7B, 1×H100 | answer match | scaffolded, not yet run |
| 1 | review-bench from shwin-bench: bisected bug PRs + matched clean PRs, 4 context conditions, frontier baselines | bugs approved (target 0), auto-ship rate | designed (`review/`) |
| 2 | the review agent: neighborhood → run suite → probe test → decision | same scorer | planned |
| 3 | mine fresh public repos (~400 PRs), GRPO a 4B/8B LoRA into the triage gate | asymmetric decision reward | recipe written |
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

Results tables are filled in as runs complete.
