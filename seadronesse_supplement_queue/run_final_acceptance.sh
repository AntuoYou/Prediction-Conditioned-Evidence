#!/usr/bin/env bash
set -euo pipefail
cd /app/Salience-DETR

OUT=experiments/supplement_audit/final_acceptance
mkdir -p "$OUT"

# Recompute saved predictions on CPU at the paper's 0.05 score floor and the
# 0.001 sensitivity floor, then run structural and pairwise checks.
python3 -u seadronesse_supplement_queue/run_eval_queue.py \
  --repo-root /app/Salience-DETR \
  --gt /app/data/seadronessee/compressed_version/annotations/instances_val2017.json \
  --out-root experiments/supplement_audit/final_eval \
  --workers 8

python3 -u seadronesse_supplement_queue/accept_supplements.py \
  --repo-root /app/Salience-DETR \
  --out experiments/supplement_audit/final_acceptance
