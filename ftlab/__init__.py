"""ft-lab: one GRPO training script, many tasks.

Layout
  ftlab/config.py      YAML -> RunConfig (dataclass); every knob lives here
  ftlab/data/          one adapter per task: load -> rows with `prompt` + reward columns
  ftlab/rewards/       one module per task; registry maps name -> callable(completions, **cols)
  ftlab/prompting.py   chat-template helpers (Qwen3 thinking off)
  ftlab/train.py       GRPOTrainer wiring; `python -m ftlab.train configs/x.yaml`
  ftlab/eval.py        greedy vLLM eval of base / tuned / frontier on the frozen split
  ftlab/plot.py        reward / kl / length curves from the trainer log
"""
