#!/usr/bin/env bash
# Phase 1: Spider text-to-SQL. setup -> databases -> smoke -> base eval -> GRPO full FT -> eval -> LoRA -> eval -> table.
# Frontier baselines (Haiku/Sonnet) run from the Mac, not here: see RUNBOOK.
set -euo pipefail
cd /workspace/ft-lab
export HF_HOME=/workspace/hf
[ -x /workspace/venv/bin/python ] && source /workspace/venv/bin/activate
export TOKENIZERS_PARALLELISM=false
STAGES=runs/phase1.stages; touch $STAGES
done_() { grep -qx "$1" $STAGES; }
mark() { echo "$1" >> $STAGES; echo "=== stage done: $1 ($(date -u +%T))"; }

done_ setup || { bash pod/setup.sh; mark setup; }
source /workspace/venv/bin/activate   # created by setup.sh
python -c 'import datasets, trl, vllm; print("venv ok:", datasets.__version__, trl.__version__, vllm.__version__)'
done_ spider_db || { bash scripts/get_spider.sh; mark spider_db; }
done_ smoke || { mkdir -p runs/spider_smoke; python -m ftlab.train configs/spider_smoke.yaml 2>&1 | tee runs/spider_smoke/train.log; mark smoke; }
done_ base_eval || { python -m ftlab.eval configs/spider_h100.yaml --label base --max-tokens 512 --max-model-len 4096; mark base_eval; }
done_ train || {
  mkdir -p runs/spider_h100
  ls runs/spider_h100/checkpoint-* >/dev/null 2>&1 && export RESUME=$(ls -d runs/spider_h100/checkpoint-* | sort -V | tail -1)
  python -m ftlab.train configs/spider_h100.yaml 2>&1 | tee -a runs/spider_h100/train.log; mark train
}
done_ tuned_eval || { python -m ftlab.eval configs/spider_h100.yaml --label tuned --model runs/spider_h100/final --max-tokens 512 --max-model-len 4096; mark tuned_eval; }
done_ plot || { python -m ftlab.plot runs/spider_h100; mark plot; }
if [ -z "${SKIP_LORA:-}" ]; then
  done_ lora || {
    mkdir -p runs/spider_lora
    python -m ftlab.train configs/spider_lora.yaml 2>&1 | tee -a runs/spider_lora/train.log
    python -m ftlab.eval configs/spider_lora.yaml --label lora --model runs/spider_lora/merged --max-tokens 512 --max-model-len 4096
    python -m ftlab.plot runs/spider_h100 runs/spider_lora; mark lora
  }
fi
python -m ftlab.eval --table runs/spider_h100/eval_*.json runs/spider_lora/eval_*.json 2>/dev/null | tee runs/spider_h100/table.txt
echo "=== phase1 complete"
