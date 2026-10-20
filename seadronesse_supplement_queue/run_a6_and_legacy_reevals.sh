#!/usr/bin/env bash
set -euo pipefail

ROOT=/app/Salience-DETR
cd "$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
BASE="experiments/supplement_audit/A6_legacy_batch_${STAMP}"
mkdir -p "$BASE"

# Freeze the interpretation before spending NPU time: A6 and A8 must differ
# only in selector and description; port 8004 must be physical NPU 2.
python3 - <<'PY'
import json, re, subprocess, urllib.request
from protocol_v2.run_a import CONFIGS
a6, a8 = dict(CONFIGS["A6"]), dict(CONFIGS["A8"])
if a6.get("mode") != "packed" or a6.get("select") != "uniform":
    raise SystemExit(f"A6 is not packed legacy uniform: {a6}")
if a8.get("mode") != "packed" or a8.get("select") != "spatial":
    raise SystemExit(f"A8 is not packed spatial uniform: {a8}")
for cfg in (a6, a8):
    cfg.pop("desc", None)
    cfg.pop("select", None)
if a6 != a8:
    raise SystemExit(f"A6/A8 differ beyond selector; refusing comparison: A6={a6}; A8={a8}")
line = subprocess.check_output(["ss", "-ltnpH", "sport = :8004"], text=True).strip()
m = re.search(r"pid=(\d+)", line)
if not m:
    raise SystemExit("Cannot identify service listener on port 8004")
pid = m.group(1)
env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
visible = next((x.decode().split("=", 1)[1] for x in env if x.startswith(b"ASCEND_RT_VISIBLE_DEVICES=")), None)
if visible != "2":
    raise SystemExit(f"port 8004 maps to physical NPU {visible}, expected NPU2")
def health(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
        h = json.load(r)
    if h.get("status") != "healthy":
        raise SystemExit(f"port {port} is unhealthy")
    return h["meta"]["weight_sha256"]
if health(8004) != health(8002):
    raise SystemExit("NPU2 service weight differs from A7/A8 reference")
print("A6/A8 configs match except selector; port 8004 is healthy on NPU2 with matching weights")
PY

GT=/app/data/seadronessee/compressed_version/annotations/instances_val2017.json
EVAL="$BASE/legacy_evals"
mkdir -p "$EVAL"
LEGACY_RUNS=()
for run in A1_v2 A2_full A2_s3; do
  pred="experiments/runs/$run/predictions_coco.json"
  if [ ! -s "$pred" ]; then
    echo "EVAL_MISSING_PREDICTIONS $pred; skip this run"
    continue
  fi
  LEGACY_RUNS+=("$run")
done
if [ "${#LEGACY_RUNS[@]}" -eq 0 ]; then
  echo "No saved legacy predictions found for CPU re-evaluation"
fi

# Re-evaluate saved predictions on CPU while A6 uses NPU2.
EVAL_PIDS=()
for run in "${LEGACY_RUNS[@]}"; do
  for floor in 0.05 0.001; do
    tag=${floor/./}
    out="$EVAL/${run}_score${tag}"
    mkdir -p "$out"
    if [ -e "$out/eval_summary.json" ]; then
      echo "EVAL_EXISTS $run score_floor=$floor; skip"
      continue
    fi
    nohup python3 -u seadronesse_supplement_queue/evaluate_predictions.py \
      --gt "$GT" --predictions "experiments/runs/$run/predictions_coco.json" \
      --out-dir "$out" --score-floor "$floor" \
      > "$out/eval.log" 2>&1 < /dev/null &
    EVAL_PIDS+=("$!")
    echo "EVAL_START $run score_floor=$floor PID=$! log=$out/eval.log"
  done
done

A6_SMOKE="A6_legacy_uniform_smoke_${STAMP}"
A6_RUN="A6_legacy_uniform_v2_${STAMP}"
for id in "$A6_SMOKE" "$A6_RUN"; do
  if [ -e "experiments/runs/$id" ]; then echo "Run exists: $id"; exit 3; fi
done

echo "A6 smoke starts on NPU2; saved-prediction CPU evaluations are running concurrently."
python3 -u -m protocol_v2.run_a --config A6 --limit 20 \
  --url http://127.0.0.1:8004/detect_packet --run-id "$A6_SMOKE" \
  > "$BASE/A6_smoke.log" 2>&1
python3 - "$A6_SMOKE" <<'PY'
import csv, pathlib, sys
p = pathlib.Path("experiments/runs") / sys.argv[1] / "per_image.csv"
if not p.is_file(): raise SystemExit(f"missing smoke rows: {p}")
with p.open(newline="") as f: rows = list(csv.DictReader(f))
ok = sum(r.get("status") == "ok" for r in rows)
if len(rows) != 20 or ok != 20:
    raise SystemExit(f"A6 smoke failed: rows={len(rows)} ok={ok}")
print("A6_SMOKE_PASS 20/20")
PY

echo "A6 smoke passed; starting 1,547-frame full run on NPU2."
nohup python3 -u -m protocol_v2.run_a --config A6 \
  --url http://127.0.0.1:8004/detect_packet --run-id "$A6_RUN" \
  > "$BASE/A6_full.log" 2>&1 < /dev/null &
A6_PID=$!
echo "$A6_PID" > "$BASE/A6_full.pid"
echo "A6_RUN=$A6_RUN PID=$A6_PID NPU2 port=8004"
echo "A6_TAIL=tail -F $BASE/A6_full.log"

wait "$A6_PID"
echo "A6_FULL_PROCESS_DONE $A6_RUN"
python3 - "$A6_RUN" <<'PY'
import csv, pathlib, sys
p = pathlib.Path("experiments/runs") / sys.argv[1] / "per_image.csv"
if not p.is_file(): raise SystemExit(f"missing full-run rows: {p}")
with p.open(newline="") as f: rows = list(csv.DictReader(f))
ok = sum(r.get("status") == "ok" for r in rows)
if len(rows) != 1547 or ok != 1547:
    raise SystemExit(f"A6 full-run acceptance failed: rows={len(rows)} ok={ok}")
print("A6_FULL_STRUCTURE_PASS 1547/1547")
PY
for pid in "${EVAL_PIDS[@]}"; do
  if ! wait "$pid"; then echo "CPU evaluation failed PID=$pid"; exit 4; fi
done
echo "LEGACY_CPU_REEVALS_DONE logs=$EVAL"

# Independent evaluations of the new A6 full-run predictions.
for floor in 0.05 0.001; do
  tag=${floor/./}
  out="$EVAL/A6_legacy_uniform_score${tag}"
  mkdir -p "$out"
  python3 -u seadronesse_supplement_queue/evaluate_predictions.py \
    --gt "$GT" --predictions "experiments/runs/$A6_RUN/predictions_coco.json" \
    --out-dir "$out" --score-floor "$floor" > "$out/eval.log" 2>&1
done
echo "A6_INDEPENDENT_EVALS_DONE run=$A6_RUN"
