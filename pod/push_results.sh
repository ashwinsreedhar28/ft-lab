#!/usr/bin/env bash
# Commit runs/** (small files only) + pod/versions.lock + a status file to branch results/<phase>.
# Called by the heartbeat every 5 min and once at the end. Never touches main.
set -uo pipefail
PHASE=${1:-phase0}; TAG=${2:-tick}
cd /workspace/ft-lab
BR="results/${PHASE}"
mkdir -p runs
# status: what's running, GPU, last log lines
{
  echo "time_utc: $(date -u +%FT%TZ)"
  echo "tag: $TAG"
  echo "tmux_ft_alive: $(tmux has-session -t ft 2>/dev/null && echo yes || echo no)"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
  echo "---- last 25 log lines"
  f=$(ls -t runs/${PHASE}*.log 2>/dev/null | head -1); echo "---- $f"; grep -oE "step [0-9]+ +reward=[-0-9.]+ +reward_std=[-0-9.]+ +kl=[-0-9.]+ +mean_length=[-0-9.]+" "$f" 2>/dev/null | tail -12; grep -E "^=== (eval@|stage done)|Error|OutOfMemory" "$f" 2>/dev/null | grep -v "^│" | tail -8; grep -oE "[0-9]+/[0-9]+ \\[[0-9:]+<[0-9:]+" "$f" 2>/dev/null | tail -1
} > runs/STATUS.txt
# keep train.log tails small; full logs stay on the pod
for f in runs/*/train.log; do [ -f "$f" ] && tail -n 400 "$f" > "${f%.log}.tail.log"; done

# work on a detached index so the running checkout is untouched
export GIT_INDEX_FILE=/workspace/.results-index
rm -f "$GIT_INDEX_FILE"
git add -f pod/versions.lock 2>/dev/null
find runs -type f -not -path "*/checkpoint-*" \( -name '*.json' -o -name '*.png' -o -name '*.yaml' -o -name '*.tail.log' -o -name '*.jsonl' -o -name 'STATUS.txt' -o -name "${PHASE}*.log" \) -size -5M -print0 \
  | xargs -0 git add -f 2>/dev/null
TREE=$(git write-tree)
PARENT=$(git rev-parse -q --verify "refs/remotes/origin/${BR}" 2>/dev/null || true)
MAIN=$(git rev-parse origin/main)
if [ -n "$PARENT" ]; then
  COMMIT=$(echo "results ${PHASE} ${TAG} $(date -u +%FT%TZ)" | git commit-tree "$TREE" -p "$PARENT")
else
  COMMIT=$(echo "results ${PHASE} ${TAG} $(date -u +%FT%TZ)" | git commit-tree "$TREE" -p "$MAIN")
fi
unset GIT_INDEX_FILE
git push -q -f origin "${COMMIT}:refs/heads/${BR}" && git fetch -q origin "${BR}" && echo "pushed ${BR} @ ${COMMIT:0:8}"
