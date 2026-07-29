#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

printf 'git_head=%s\n' "$(git rev-parse HEAD)"
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits |
  awk -F, '$1 + 0 <= 3 {print "gpu=" $0}'
df -h /data02 | tail -n 1
ps -eo pid,etime,cmd |
  grep -E 'ccvr|run_formal' |
  grep -v grep || true

for path in \
  runs/direction_gate/data_audit.json \
  runs/direction_gate/openclip/summary.json \
  runs/direction_gate/eva_clip/summary.json \
  runs/direction_gate/direction_decision.json; do
  if [[ -f "$path" ]]; then
    python - "$path" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
print(sys.argv[1], value.get("status"), value.get("accepted"))
PY
  fi
done

for path in \
  runs/direction_gate/manifest.jsonl \
  runs/direction_gate/openclip/per_query_metrics.jsonl \
  runs/direction_gate/eva_clip/per_query_metrics.jsonl; do
  if [[ -f "$path" ]]; then
    printf '%s rows=%s bytes=%s\n' \
      "$path" "$(wc -l < "$path")" "$(stat -c %s "$path")"
  fi
done

