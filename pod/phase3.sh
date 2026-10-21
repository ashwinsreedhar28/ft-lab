#!/usr/bin/env bash
# Phase 3b on 1x H100: setup -> base eval (Qwen3-4B, no tools) -> SFT on teacher traces -> eval -> GRPO -> eval.
# Needs data/review_sft.jsonl and review/bench/public_*.jsonl committed (traces.py + assemble.py on the Mac).
set -euo pipefail
cd /workspace/ft-lab
export HF_HOME=/workspace/hf
[ -x /workspace/venv/bin/python ] && source /workspace/venv/bin/activate
unset PYTHONPATH   # template dist-packages must not shadow the venv
export TOKENIZERS_PARALLELISM=false
STAGES=runs/phase3.stages; touch $STAGES
done_() { grep -qx "$1" $STAGES; }
mark() { echo "$1" >> $STAGES; echo "=== stage done: $1 ($(date -u +%T))"; }
EV="python -m ftlab.eval configs/review_grpo_4b.yaml --max-tokens 500 --max-model-len 16384"

done_ setup || { bash pod/setup.sh; mark setup; }
source /workspace/venv/bin/activate
python -c 'import trl, vllm, peft; print("venv ok:", trl.__version__, vllm.__version__, peft.__version__)'
test -s data/review_sft.jsonl || { echo "no data/review_sft.jsonl: run review/analysis/traces.py on the Mac and commit"; exit 2; }
done_ base_eval || { $EV --label base --model Qwen/Qwen3-4B; mark base_eval; }
done_ sft       || { mkdir -p runs/review_sft_4b; python -m ftlab.sft configs/review_sft_4b.yaml 2>&1 | tee runs/review_sft_4b/train.log; mark sft; }
done_ sft_eval  || { $EV --label sft --model runs/review_sft_4b/final; mark sft_eval; }
done_ grpo      || { mkdir -p runs/review_grpo_4b; python -m ftlab.train configs/review_grpo_4b.yaml 2>&1 | tee -a runs/review_grpo_4b/train.log; mark grpo; }
done_ grpo_eval || { $EV --label grpo --model runs/review_grpo_4b/merged; mark grpo_eval; }   # LoRA run: train.py writes the merged weights to merged/, final/ is the adapter
python -m ftlab.eval --table runs/review_grpo_4b/eval_*.json | tee runs/review_grpo_4b/table.txt
python -m review.analysis.gate_table runs/review_grpo_4b/eval_*.json | tee runs/review_grpo_4b/gate_table.txt
echo PHASE-DONE
