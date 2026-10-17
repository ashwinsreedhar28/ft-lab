#!/usr/bin/env bash
# Phase 0b: eval-settings ablation + hard-pool run. Assumes phase0 ran on this pod (runs/gsm8k_h100/final exists).
set -euo pipefail
cd /workspace/ft-lab
export HF_HOME=/workspace/hf
source /workspace/venv/bin/activate
export TOKENIZERS_PARALLELISM=false
STAGES=runs/phase0b.stages; touch $STAGES
done_() { grep -qx "$1" $STAGES; }
mark() { echo "$1" >> $STAGES; echo "=== stage done: $1 ($(date -u +%T))"; }

# 1) same models, eval at T=0.6 and with a 1536-token budget: how much of the regression is decoding?
for cond in "t06:--temperature 0.6 --max-tokens 768" "long:--max-tokens 1536" "t06long:--temperature 0.6 --max-tokens 1536"; do
  name=${cond%%:*}; flags=${cond#*:}
  done_ "eval_$name" || {
    python -m ftlab.eval configs/gsm8k_h100.yaml --label base_$name $flags --max-model-len 2560
    python -m ftlab.eval configs/gsm8k_h100.yaml --label tuned_$name --model runs/gsm8k_h100/final $flags --max-model-len 2560
    mark "eval_$name"
  }
done
python -m ftlab.eval --table runs/gsm8k_h100/eval_*.json | tee runs/gsm8k_h100/table_ablation.txt

# 2) hard pool + length penalty, 300 steps, then eval both ways
done_ hard_train || { mkdir -p runs/gsm8k_hard; python -m ftlab.train configs/gsm8k_hard.yaml 2>&1 | tee -a runs/gsm8k_hard/train.log; mark hard_train; }
done_ hard_eval || {
  python -m ftlab.eval configs/gsm8k_hard.yaml --label tuned --model runs/gsm8k_hard/final --max-tokens 768
  python -m ftlab.eval configs/gsm8k_hard.yaml --label tuned_t06 --model runs/gsm8k_hard/final --temperature 0.6 --max-tokens 768
  python -m ftlab.plot runs/gsm8k_h100 runs/gsm8k_hard
  mark hard_eval
}
python -m ftlab.eval --table runs/gsm8k_h100/eval_base.json runs/gsm8k_h100/eval_base_t06.json runs/gsm8k_hard/eval_*.json | tee runs/gsm8k_hard/table.txt
echo "=== phase0b complete"
