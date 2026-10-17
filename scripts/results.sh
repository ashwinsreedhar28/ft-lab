#!/usr/bin/env bash
# Pull the pod's pushed results for a phase into ./runs (works from the Mac or anywhere with the repo).
#   scripts/results.sh phase0
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE=${1:-phase0}
git fetch -q origin "results/${PHASE}"
git checkout -q "origin/results/${PHASE}" -- runs pod/versions.lock 2>/dev/null || git checkout -q "origin/results/${PHASE}" -- runs
git reset -q runs pod/versions.lock 2>/dev/null || true   # leave them unstaged; runs/ is gitignored on main
echo "== runs/STATUS.txt"; cat runs/STATUS.txt 2>/dev/null || echo "(no status yet)"
