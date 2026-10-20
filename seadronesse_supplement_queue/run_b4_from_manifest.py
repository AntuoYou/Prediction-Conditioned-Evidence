#!/usr/bin/env python3
"""Run B4 using the frozen A5_v3 protocol config and byte-calibrated ROI Q."""
from __future__ import annotations

import json
import os
from pathlib import Path

from protocol_v2 import run_a


root = Path(os.environ.get("REPO_ROOT", "/app/Salience-DETR")).resolve()
base_run = os.environ.get("B4_BASE_RUN", "A5_v3")
quality = os.environ.get("B4_Q_ROI")
if quality is None:
    raise SystemExit("B4_Q_ROI must be set to the frozen byte-calibrated quality")

manifest_path = root / "experiments/runs" / base_run / "run_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"B4 base manifest not found: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
cfg = dict(manifest["cfg"])
if cfg.get("mode") != "regional" or not cfg.get("explore"):
    raise SystemExit(f"B4 base must be support+exploration regional; got {cfg}")
cfg["desc"] = "B4 support+exploration regional, byte-matched to actual B6 mean"
cfg["q_roi"] = int(quality)
run_a.CONFIGS["B4_169"] = cfg
run_a.main()
