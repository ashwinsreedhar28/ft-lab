#!/usr/bin/env bash
# Phase 3b on 1x H100: setup -> base eval (Qwen3-4B, no tools) -> SFT on teacher traces -> eval -> GRPO -> eval.
# Needs data/review_sft.jsonl and review/bench/public_*.jsonl committed (traces.py + assemble.py on the Mac).
set -euo pipefail
cd /workspace/ft-lab
export HF_HOME=/workspace/hf
[ -x /workspace/venv/bin/python ] && source /workspace/venv/bin/activate
unset PYTHONPATH   # template dist-packages must not shadow the venv
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 7 GiB reserved-but-unallocated at the v3 OOM
V="${V:-}"                      # run suffix: V=_v2 -> configs/review_{sft,grpo}_4b_v2.yaml, runs/phase3_v2.stages
SFT=review_sft_4b$V; GRPO=review_grpo_4b$V
STAGES=runs/phase3$V.stages; touch $STAGES
done_() { grep -qx "$1" $STAGES; }
mark() { echo "$1" >> $STAGES; echo "=== stage done: $1 ($(date -u +%T))"; }
EV="python -m ftlab.eval configs/$GRPO.yaml --max-model-len 16384"   # max tokens = config max_completion_length (500 no-think, 2048 thinking)

done_ setup || { bash pod/setup.sh; mark setup; }
source /workspace/venv/bin/activate
python -c 'import trl, vllm, peft; print("venv ok:", trl.__version__, vllm.__version__, peft.__version__)'
test -s data/review_sft.jsonl || { echo "no data/review_sft.jsonl: run review/analysis/traces.py on the Mac and commit"; exit 2; }
done_ base_eval || { mkdir -p runs/$GRPO; $EV --label base --model Qwen/Qwen3-4B; mark base_eval; }
done_ sft       || { mkdir -p runs/$SFT; python -m ftlab.sft configs/$SFT.yaml 2>&1 | tee runs/$SFT/train.log; mark sft; }
done_ sft_eval  || { $EV --label sft --model runs/$SFT/final; mark sft_eval; }
done_ grpo      || { mkdir -p runs/$GRPO; python -m ftlab.train configs/$GRPO.yaml 2>&1 | tee -a runs/$GRPO/train.log; mark grpo; }
done_ grpo_eval || { $EV --label grpo --model runs/$GRPO/merged; mark grpo_eval; }   # LoRA run: train.py writes the merged weights to merged/, final/ is the adapter
python -m ftlab.eval --table runs/$GRPO/eval_*.json | tee runs/$GRPO/table.txt
python -m review.analysis.gate_table runs/$GRPO/eval_*.json | tee runs/$GRPO/gate_table.txt
echo PHASE-DONE
