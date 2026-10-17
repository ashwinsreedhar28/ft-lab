#!/usr/bin/env bash
# Self-driving pod. From your Mac terminal, ONE command starts a whole phase:
#
#   ssh -i ~/.ssh/runpod -p <PORT> root@<HOST> \
#     'export GH_TOKEN=<fine-grained PAT, contents:write on ft-lab only>; \
#      curl -sL https://raw.githubusercontent.com/ashwinsreedhar28/ft-lab/main/pod/bootstrap.sh | bash -s phase0'
# Re-run after a fix (raw CDN caches ~5 min): cd /workspace/ft-lab && git fetch && git reset --hard origin/main && bash pod/bootstrap.sh phase0
#
# What it does: clones the repo to /workspace/ft-lab, runs pod/setup.sh, starts
# pod/<phase>.sh under tmux (session "ft"), and starts a heartbeat that every
# 5 min commits runs/** (json, png, yaml, log tails) to branch results/<phase>
# so Claude can follow along without SSH. Re-running attaches to the existing tmux.
set -euo pipefail
PHASE=${1:-phase0}
REPO=https://github.com/ashwinsreedhar28/ft-lab
: "${GH_TOKEN:?set GH_TOKEN (fine-grained PAT: contents read/write on ashwinsreedhar28/ft-lab)}"

cd /workspace
if [ ! -d ft-lab/.git ]; then
  git clone -q "https://x-access-token:${GH_TOKEN}@github.com/ashwinsreedhar28/ft-lab" ft-lab
fi
cd ft-lab
git config user.name "ft-lab pod"
git config user.email "pod@ft-lab.local"
git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/ashwinsreedhar28/ft-lab"
git fetch -q origin main && git checkout -q main && git reset -q --hard origin/main

if ! command -v tmux >/dev/null; then apt-get update -qq && apt-get install -y -qq tmux >/dev/null; fi

# heartbeat: push runs/ to results/<phase> every 5 min (runs even if the phase script dies)
cat > /workspace/heartbeat.sh <<EOF
#!/usr/bin/env bash
cd /workspace/ft-lab
while true; do
  bash pod/push_results.sh "$PHASE" >/dev/null 2>&1 || true
  sleep 300
done
EOF
chmod +x /workspace/heartbeat.sh
tmux has-session -t hb 2>/dev/null || tmux new -d -s hb "/workspace/heartbeat.sh"

if tmux has-session -t ft 2>/dev/null && [ -f "runs/${PHASE}.running" ]; then
  echo "phase already running in tmux 'ft'; attach with: tmux attach -t ft"
else
  tmux kill-session -t ft 2>/dev/null || true   # stale session from a failed/finished run
  mkdir -p runs
  rm -f "runs/${PHASE}.running"
  tmux new -d -s ft "touch runs/${PHASE}.running; bash pod/${PHASE}.sh 2>&1 | tee runs/${PHASE}.log; rm -f runs/${PHASE}.running; bash pod/push_results.sh ${PHASE} final; echo PHASE-DONE; sleep 86400"
  echo "started pod/${PHASE}.sh in tmux 'ft'. tail: tmux attach -t ft   (ctrl-b d to detach)"
fi
echo "results branch: ${REPO}/tree/results/${PHASE}"
