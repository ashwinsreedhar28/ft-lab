#!/usr/bin/env bash
# Push log tails + mutant/result summaries to branch live/mac every 10 min so Claude can watch runs from GitHub.
#   nohup scripts/heartbeat_mac.sh > /dev/null 2>&1 &      (stop: pkill -f heartbeat_mac)
# Uses a detached index + commit-tree, so your working tree and main are untouched.
set -u
cd "$(dirname "$0")/.."
while true; do
  mkdir -p live
  for f in review/mine/work/mutate_run5.log review/agent/results/pub2_v5.log review/agent/results/train_traces.log; do
    [ -f "$f" ] && { echo "== $f ($(date -u +%FT%TZ))"; tail -n 25 "$f"; } > "live/$(basename "$f").tail" 2>/dev/null
  done
  [ -f review/mine/work/mutants.jsonl ] && python3 - > live/mutants_summary.txt 2>/dev/null <<'PY'
import json, collections
c = collections.Counter(); by = collections.Counter()
for l in open("review/mine/work/mutants.jsonl"):
    d = json.loads(l); st = d.get("status") or d.get("stage2") or d.get("stage1") or "?"
    c[st] += 1
    if st == "positive": by[d["repo"]] += 1
print("statuses:", dict(c)); print("positives by repo:", dict(by))
PY
  ls -la review/agent/results/*.json 2>/dev/null | awk '{print $6,$7,$8,$9}' > live/results_ls.txt
  export GIT_INDEX_FILE=.git/live-index; rm -f "$GIT_INDEX_FILE"
  git add -f live/ >/dev/null 2>&1
  TREE=$(git write-tree); PARENT=$(git rev-parse -q --verify refs/remotes/origin/live/mac 2>/dev/null || git rev-parse origin/main)
  COMMIT=$(echo "mac heartbeat $(date -u +%FT%TZ)" | git commit-tree "$TREE" -p "$PARENT")
  git push -q -f origin "$COMMIT:refs/heads/live/mac" 2>/dev/null
  unset GIT_INDEX_FILE
  sleep 600
done
