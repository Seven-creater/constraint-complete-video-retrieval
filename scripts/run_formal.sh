#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="runs/direction_gate"
RAW="data/raw"
FEATURES="data/features"

trap 'ccvr finalize --run "$RUN" || true' EXIT

python -m pip install -e .
ccvr audit-literature \
  --matrix docs/literature_collision_matrix.csv \
  --output "$RUN/literature_audit.json"
ccvr fetch-metadata --output "$RAW"
ccvr audit-dataset --raw "$RAW" --output "$RUN"

python - "$RUN/data_audit.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    assert json.load(handle)["accepted"], "public_dataset_gate_failed"
PY

ccvr fetch-features \
  --backbone OpenCLIP \
  --output "$FEATURES" \
  --gate "$RUN"
CUDA_VISIBLE_DEVICES=0 ccvr score-backbone \
  --backbone OpenCLIP \
  --raw "$RAW" \
  --manifest "$RUN/manifest.jsonl" \
  --features "$FEATURES/OpenCLIP" \
  --output "$RUN/openclip" \
  --device cuda:0

python - "$RUN/openclip/summary.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
assert result["problem_gate"]["accepted"], "openclip_problem_gate_failed"
assert not result["simple_solution"], "solved_by_simple_logic"
PY

ccvr fetch-features \
  --backbone EVA-CLIP \
  --output "$FEATURES" \
  --gate "$RUN"
CUDA_VISIBLE_DEVICES=0 ccvr score-backbone \
  --backbone EVA-CLIP \
  --raw "$RAW" \
  --manifest "$RUN/manifest.jsonl" \
  --features "$FEATURES/EVA-CLIP" \
  --output "$RUN/eva_clip" \
  --device cuda:0
ccvr finalize --run "$RUN"
