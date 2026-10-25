#!/usr/bin/env bash
# Measurement job (Oct 25 audit): Qwen3-4B, three prompt/thinking arms x three decoding seeds on the frozen 61.
# No training. ~30-45 min on an H100 incl. setup. Results: runs/measure_4b_{a,b,c}/eval_s{0,1,2}.json ->
# `python -m review.analysis.measure_table runs/measure_4b_*/eval_s*.json`.
#   A: current prompt, thinking on      B: compact no-test prompt, thinking on      C: compact prompt, thinking off
set -euo pipefail
cd /workspace/ft-lab
export HF_HOME=/workspace/hf
[ -x /workspace/venv/bin/python ] && source /workspace/venv/bin/activate
unset PYTHONPATH
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAGES=runs/measure.stages; touch $STAGES
done_() { grep -qx "$1" $STAGES; }
mark() { echo "$1" >> $STAGES; echo "=== stage done: $1 ($(date -u +%T))"; }
MODEL="${MODEL:-Qwen/Qwen3-4B}"
done_ setup || { bash pod/setup.sh; mark setup; }
source /workspace/venv/bin/activate
for arm in a b c; do for seed in 0 1 2; do
  done_ "$arm$seed" || { python -m ftlab.eval configs/measure_4b_$arm.yaml --label s$seed --model "$MODEL" \
      --max-model-len 16384 --temperature 0.6 --top-p 0.95 --top-k 20 --seed $seed; mark "$arm$seed"; }
done; done
python -m review.analysis.measure_table runs/measure_4b_*/eval_s*.json | tee runs/measure_table.txt
echo PHASE-DONE
