#!/usr/bin/env bash
# Mac-side helper. Usage:
#   scripts/pod.sh set <host> <port>      # remember the pod (writes .pod)
#   scripts/pod.sh ssh [cmd...]           # ssh in (interactive if no cmd)
#   scripts/pod.sh push                   # rsync repo -> pod:/workspace/ft-lab (code + configs, not runs/data)
#   scripts/pod.sh pull                   # rsync pod runs/ (json, png, configs; no checkpoints) -> ./runs
#   scripts/pod.sh run <config.yaml>      # push, then start training under tmux "train", log to runs/<name>/train.log
#   scripts/pod.sh log                    # tail the live training log
set -euo pipefail
cd "$(dirname "$0")/.."
KEY=~/.ssh/runpod
POD_FILE=.pod
cmd=${1:-}; shift || true
if [ "$cmd" = set ]; then echo "HOST=$1 PORT=$2" > $POD_FILE; echo saved; exit 0; fi
[ -f $POD_FILE ] || { echo "no .pod; run: scripts/pod.sh set <host> <port>"; exit 1; }
# shellcheck disable=SC1090
source $POD_FILE
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new root@$HOST"
RS="rsync -az -e \"ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new\""
case $cmd in
  ssh)  exec $SSH "$@" ;;
  push) eval $RS --exclude runs --exclude data --exclude .git --exclude '__pycache__' ./ root@$HOST:/workspace/ft-lab/ ;;
  pull) mkdir -p runs; eval $RS --include='*/' --include='*.json' --include='*.png' --include='*.yaml' --include='*.log' --include='*.lock' --exclude='*' root@$HOST:/workspace/ft-lab/runs/ ./runs/ ; eval $RS root@$HOST:/workspace/ft-lab/pod/versions.lock ./pod/versions.lock ;;
  run)  cfg=$1; name=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$cfg'))['name'])")
        "$0" push
        $SSH "cd /workspace/ft-lab && mkdir -p runs/$name && tmux new -d -s train \"python -m ftlab.train $cfg 2>&1 | tee runs/$name/train.log\"" && echo "started tmux session 'train' on the pod" ;;
  log)  $SSH "tail -n 40 -f \$(ls -t /workspace/ft-lab/runs/*/train.log | head -1)" ;;
  *) sed -n '2,10p' "$0" ;;
esac
