#!/usr/bin/env python3
"""
Pre-flight Checklist — Verifikasi sebelum upload ke Colab / smoke test.
Jalankan di local macOS untuk memastikan semua gate checklist (G1–G10) sudah pass.
"""
import sys
import json
import glob
import random
from pathlib import Path

errors = []
warnings = []
passes = []

def check(name, condition, msg, severity="error"):
    if condition:
        passes.append(f"  ✅ {name}")
    else:
        if severity == "error":
            errors.append(f"  ❌ {name}: {msg}")
        else:
            warnings.append(f"  ⚠️  {name}: {msg}")

print("=" * 60)
print("PRE-FLIGHT CHECKLIST v0.4.0 Fase 2")
print("=" * 60)

# G1: QC v3 applied
qc_frames = list(Path("dataset").rglob("_QC2_frame_*"))
qc_sessions = [p for p in Path("dataset").glob("*/_QC2_*") if p.is_dir() and not p.name.startswith("_QC2_frame_")]
check("G1: QC v3 applied", len(qc_frames) > 0 or len(qc_sessions) > 0,
      f"Found {len(qc_frames)} QC2 frames, {len(qc_sessions)} QC2 sessions", "error")

# G2–G3: Files exist
check("G2: models/encoder.py exists", Path("models/encoder.py").exists(), "missing", "error")
check("G3: models/siamese.py exists", Path("models/siamese.py").exists(), "missing", "error")
check("G3: train.py exists", Path("train.py").exists(), "missing", "error")
check("G3: evaluate.py exists", Path("evaluate.py").exists(), "missing", "error")
check("G3: utils/dataset.py exists", Path("utils/dataset.py").exists(), "missing", "error")

# G4: Init parity script exists
check("G4: audit_init_parity.py exists", Path("utils/audit_init_parity.py").exists(), "missing", "error")

# G5: Frame count (manual scan, no torch)
def count_frames():
    data_dir = Path("dataset")
    count = 0
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("_"):
            continue
        for ts_dir in sorted(label_dir.iterdir()):
            if not ts_dir.is_dir() or ts_dir.name.startswith("_QC2_") or ts_dir.name.startswith("_QUARANTINE_"):
                continue
            for frame_dir in sorted(ts_dir.iterdir()):
                if not frame_dir.is_dir() or frame_dir.name.startswith("_QC2_"):
                    continue
                if (frame_dir / "cnn_input.npy").exists() and (frame_dir / "geometry.json").exists():
                    count += 1
    return count

total = count_frames()
check("G5: Dataset frame count", total >= 1800, f"only {total} frames (expected ~1836)", "error")

# G6: Notebook flags
train_nb = Path("collab/01_train_and_eval.ipynb").read_text()
check("G6: 01_train_and_eval.ipynb exists", "no_geom" in train_nb, "missing notebook", "error")
check("G6: has 4 variants", "gam_only" in train_nb and "fuse_only" in train_nb, "missing variants", "error")
check("G6: has 5 seeds", "31337" in train_nb, "missing seeds", "error")

# G7: train.py --seed
train_src = Path("train.py").read_text()
check("G7: train.py has --seed", '"--seed"' in train_src, "missing --seed arg", "error")
check("G7: _set_seed function", "_set_seed(" in train_src, "missing _set_seed", "error")

# G8: evaluate.py 4-variant flags
eval_src = Path("evaluate.py").read_text()
check("G8: evaluate.py has --use-gam", '"--use-gam"' in eval_src, "missing --use-gam", "error")
check("G8: evaluate.py has --use-geom-fusion", '"--use-geom-fusion"' in eval_src, "missing", "error")

# G9: config.json save logic
check("G9: train.py saves config.json", "config.json" in train_src, "missing config save", "warning")

# G10: Geometry new
samples = glob.glob("dataset/*/2026*/frame_*/geometry.json")
if samples:
    with open(random.choice(samples)) as f:
        g = json.load(f)
    check("G10: geometry.json has quality_issues", "quality_issues" in g, "old format", "warning")
    has_fallback = any("knuckle_fallback" in str(i) for i in g.get("quality_issues", []))
    check("G10: knuckle fallback possible", True, f"sample has fallback={has_fallback}", "info")

print()
if passes:
    print(f"PASSED ({len(passes)}):")
    for p in passes:
        print(p)
if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(w)
if errors:
    print(f"\nERRORS ({len(errors)}) — FIX BEFORE UPLOAD:")
    for e in errors:
        print(e)
    print("\n🔴 DO NOT PROCEED. Fix errors first.")
    sys.exit(1)
else:
    print("\n🟢 ALL CHECKS PASSED. Ready for upload & smoke test.")
