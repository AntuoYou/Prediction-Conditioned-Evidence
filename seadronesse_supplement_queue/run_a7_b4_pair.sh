#!/usr/bin/env bash
set -euo pipefail
ROOT=/app/Salience-DETR
cd "$ROOT"
BASE="experiments/supplement_audit/A7_B4_169_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE"
CALCSV="$BASE/byte_candidates.csv"
CALLOG="$BASE/calibrate.log"

echo "PAIR_DIR=$BASE"
echo "Starting byte-only calibration; no NPU is used during this stage."
python3 -u seadronesse_supplement_queue/calibrate_a7_b4_bytes.py \
  --repo-root "$ROOT" \
  --gt /app/data/seadronessee/compressed_version/annotations/instances_val2017.json \
  --img-dir /app/data/seadronessee/compressed_version/val2017 \
  --edge "$ROOT/experiments/edge_cache/edge_best_conf001.jsonl" \
  --a7-target-kib 53.544 --out "$ROOT/$CALCSV" > "$ROOT/$CALLOG" 2>&1
cat "$ROOT/$CALLOG"

CALMANIFEST="${CALCSV%.csv}.manifest.json"
readarray -t PARAMS < <(python3 - "$ROOT/$CALMANIFEST" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d["a7_recommended_ratio_by_byte_gap"])
print(d["b4_recommended_q_roi_highest_not_over_target"])
print(d["a7_recommended_mean_kib"])
print(d["b4_recommended_mean_kib"])
PY
)
A7_RATIO="${PARAMS[0]}"; B4_Q="${PARAMS[1]}"
A7_KIB="${PARAMS[2]}"; B4_KIB="${PARAMS[3]}"
A7_GAP=$(python3 -c "print(abs(float('$A7_KIB')-53.544)/53.544*100)")
B4_GAP=$(python3 -c "print(abs(float('$B4_KIB')-168.99504080478346)/168.99504080478346*100)")
echo "Frozen by bytes: A7 ratio=$A7_RATIO mean=${A7_KIB}KiB gap=${A7_GAP}%; B4 q_roi=$B4_Q mean=${B4_KIB}KiB gap=${B4_GAP}%"
python3 - "$A7_GAP" "$B4_GAP" <<'PY'
import sys
if float(sys.argv[1]) > 5 or float(sys.argv[2]) > 5:
    raise SystemExit("No byte candidate within 5%; stopped before NPU inference. Review byte_candidates.csv")
PY

# Validate two distinct NPU services and identical weights before allocating work.
python3 - <<'PY'
import json, re, subprocess, urllib.request
def service(port, expected):
    text=subprocess.check_output(["ss","-ltnpH",f"sport = :{port}"],text=True).strip()
    m=re.search(r"pid=(\d+)",text)
    if not m: raise SystemExit(f"Cannot identify listener on port {port}")
    pid=m.group(1)
    env=open(f"/proc/{pid}/environ","rb").read().split(b"\0")
    visible=next((x.decode().split("=",1)[1] for x in env if x.startswith(b"ASCEND_RT_VISIBLE_DEVICES=")),None)
    if visible != str(expected): raise SystemExit(f"port {port} maps to NPU {visible}, expected {expected}")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=5) as r: h=json.load(r)
    if h.get("status")!="healthy": raise SystemExit(f"port {port} unhealthy")
    return h["meta"]["weight_sha256"]
if service(8002,0)!=service(8003,1): raise SystemExit("NPU0/NPU1 weight hashes differ")
print("Verified ports: 8002 -> physical NPU0; 8003 -> physical NPU1; weights match")
PY

STAMP=$(date +%Y%m%d_%H%M%S)
A7_SMOKE="A7_support_centers_54kb_smoke_$STAMP"
B4_SMOKE="B4_support_exploration_169kb_smoke_$STAMP"
A7_RUN="A7_support_centers_54kb_$STAMP"
B4_RUN="B4_support_exploration_169kb_$STAMP"
for id in "$A7_SMOKE" "$B4_SMOKE" "$A7_RUN" "$B4_RUN"; do
  if [ -e "experiments/runs/$id" ]; then echo "Run directory exists: $id"; exit 1; fi
done

echo "Starting smoke runs concurrently on NPU0 and NPU1."
python3 -u -m protocol_v2.run_a --config A7 --ratio "$A7_RATIO" --limit 20 \
  --url http://127.0.0.1:8002/detect_packet --run-id "$A7_SMOKE" \
  > "$ROOT/$BASE/A7_smoke.log" 2>&1 & A7PID=$!
REPO_ROOT="$ROOT" B4_BASE_RUN=A5_v3 B4_Q_ROI="$B4_Q" \
  python3 -u seadronesse_supplement_queue/run_b4_from_manifest.py \
  --config B4_169 --limit 20 --url http://127.0.0.1:8003/detect_packet --run-id "$B4_SMOKE" \
  > "$ROOT/$BASE/B4_smoke.log" 2>&1 & B4PID=$!
wait "$A7PID" || { echo "A7 smoke failed; inspect $BASE/A7_smoke.log"; exit 1; }
wait "$B4PID" || { echo "B4 smoke failed; inspect $BASE/B4_smoke.log"; exit 1; }

python3 - "$A7_SMOKE" "$B4_SMOKE" <<'PY'
import csv, pathlib, sys
for run in sys.argv[1:]:
    p=pathlib.Path("experiments/runs")/run/"per_image.csv"
    if not p.is_file(): raise SystemExit(f"Missing smoke output: {p}")
    with p.open(newline="") as f: rows=list(csv.DictReader(f))
    ok=sum(r.get("status")=="ok" for r in rows)
    if len(rows)!=20 or ok!=20: raise SystemExit(f"Smoke failed {run}: rows={len(rows)} ok={ok}")
    print(f"SMOKE_PASS {run}: 20/20")
PY

echo "Smoke checks passed; launching both full runs concurrently."
nohup python3 -u -m protocol_v2.run_a --config A7 --ratio "$A7_RATIO" \
  --url http://127.0.0.1:8002/detect_packet --run-id "$A7_RUN" \
  > "$ROOT/$BASE/A7_full.log" 2>&1 < /dev/null & A7FULL=$!
REPO_ROOT="$ROOT" B4_BASE_RUN=A5_v3 B4_Q_ROI="$B4_Q" \
  nohup python3 -u seadronesse_supplement_queue/run_b4_from_manifest.py \
  --config B4_169 --url http://127.0.0.1:8003/detect_packet --run-id "$B4_RUN" \
  > "$ROOT/$BASE/B4_full.log" 2>&1 < /dev/null & B4FULL=$!
echo "$A7FULL" > "$ROOT/$BASE/A7_full.pid"
echo "$B4FULL" > "$ROOT/$BASE/B4_full.pid"
echo "A7_RUN=$A7_RUN PID=$A7FULL physical_NPU=0 port=8002"
echo "B4_RUN=$B4_RUN PID=$B4FULL physical_NPU=1 port=8003"
echo "TAIL: tail -F $BASE/A7_full.log $BASE/B4_full.log"
