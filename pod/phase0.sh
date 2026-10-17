#!/usr/bin/env bash
# Phase 0 end to end on one H100: setup -> smoke -> base eval -> 300-step GRPO -> tuned eval -> plot.
# Each stage writes a marker in runs/phase0.stages so a re-run skips what finished.
set -euo pipefail
cd /workspace/ft-lab
export HF_HOME=/workspace/hf
[ -x /workspace/venv/bin/python ] && source /workspace/venv/bin/activate
export TOKENIZERS_PARALLELISM=false
STAGES=runs/phase0.stages; touch $STAGES
done_() { grep -qx "$1" $STAGES; }
mark() { echo "$1" >> $STAGES; echo "=== stage done: $1 ($(date -u +%T))"; }

done_ setup || { bash pod/setup.sh; mark setup; }

done_ smoke || {
  echo "=== smoke (3 steps)"; mkdir -p runs/gsm8k_smoke; python -m ftlab.train configs/gsm8k_smoke.yaml 2>&1 | tee runs/gsm8k_smoke/train.log
  mark smoke
}

done_ base_eval || {
  echo "=== base eval (1,319 GSM8K test)"
  python -m ftlab.eval configs/gsm8k_h100.yaml --label base --max-tokens 768
  mark base_eval
}

done_ train || {
  echo "=== GRPO 300 steps"; mkdir -p runs/gsm8k_h100
  if [ -d runs/gsm8k_h100 ] && ls runs/gsm8k_h100/checkpoint-* >/dev/null 2>&1; then
    export RESUME=$(ls -d runs/gsm8k_h100/checkpoint-* | sort -V | tail -1)
  fi
  python -m ftlab.train configs/gsm8k_h100.yaml 2>&1 | tee -a runs/gsm8k_h100/train.log
  mark train
}

done_ tuned_eval || {
  echo "=== tuned eval"
  python -m ftlab.eval configs/gsm8k_h100.yaml --label tuned --model runs/gsm8k_h100/final --max-tokens 768
  mark tuned_eval
}

done_ plot || { python -m ftlab.plot runs/gsm8k_h100; mark plot; }

python -m ftlab.eval --table runs/gsm8k_h100/eval_base.json runs/gsm8k_h100/eval_tuned.json | tee runs/gsm8k_h100/table.txt

# optional LoRA comparison; set SKIP_LORA=1 to stop here
if [ -z "${SKIP_LORA:-}" ]; then
  done_ lora || {
    echo "=== LoRA run"; mkdir -p runs/gsm8k_lora
    python -m ftlab.train configs/gsm8k_lora.yaml 2>&1 | tee -a runs/gsm8k_lora/train.log
    python -m ftlab.eval configs/gsm8k_lora.yaml --label lora --model runs/gsm8k_lora/merged --max-tokens 768
    python -m ftlab.plot runs/gsm8k_h100 runs/gsm8k_lora
    mark lora
  }
  python -m ftlab.eval --table runs/gsm8k_h100/eval_base.json runs/gsm8k_h100/eval_tuned.json runs/gsm8k_lora/eval_lora.json | tee runs/gsm8k_h100/table.txt
fi
echo "=== phase0 complete. Stop the pod after results are pushed."
