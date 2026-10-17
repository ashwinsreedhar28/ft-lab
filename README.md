# ft-lab

GRPO fine-tuning experiments on Runpod GPUs: one script, swappable tasks, measured
against a frozen held-out split and frontier-model baselines.

| phase | task | reward | status |
|---|---|---|---|
| 0 | GSM8K (Qwen3-1.7B) | answer match | scaffolded, not yet run |
| 1 | Spider text-to-SQL | execution match in SQLite | scaffolded |
| 2 | topology: 1-colocate / 2-server / 4-colocate | step time, $/1k completions | configs ready |
| 3 | serverless-lakehouse text-to-SQL (Snowflake) | execution match | planned |

```
pip install -r requirements.txt          # or bash pod/setup.sh on a pod
make test                                # 23 pure-python tests
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
