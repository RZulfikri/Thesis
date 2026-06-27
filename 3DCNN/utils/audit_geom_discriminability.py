"""
utils/audit_geom_discriminability.py — Audit diskriminabilitas per fitur geometri.

Hitung between-subject std / within-subject std ratio untuk setiap dimensi
fitur geometri (13-dim v5.0.0). Ratio > 1 artinya fitur lebih bervariasi
antar subjek daripada dalam subjek → diskriminatif.

Output:
  - Tabel per fitur: mean, between-std, within-std, B/W ratio
  - Fitur dengan ratio < 1 di-flag sebagai noisy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry_schema import _flatten_geometry
from utils.dataset_lowdata import build_lowdata_splits_with_paths


def audit_discriminability(dataset_root: Path):
    """
    Compute between/within std ratio per feature.

    Returns:
        report: dict dengan keys 'per_feature' dan 'summary'
    """
    dataset_root = Path(dataset_root)
    splits = build_lowdata_splits_with_paths(dataset_root)
    all_items = splits["train"] + splits["val"] + splits["test"] + splits["holdout"]

    # Group by subject
    subject_features: dict[str, list[np.ndarray]] = {}
    for label, frame_dir in all_items:
        geo_path = frame_dir / "geometry.json"
        with open(geo_path) as f:
            geo = json.load(f)
        feat = _flatten_geometry(geo)
        subject_features.setdefault(label, []).append(feat)

    subjects = sorted(subject_features.keys())
    n_subjects = len(subjects)
    n_dims = len(next(iter(subject_features.values()))[0])

    # Compute per-subject mean and pooled within-subject std
    subject_means = []
    within_vars = []
    for subj in subjects:
        feats = np.stack(subject_features[subj])  # (n_sessions, n_dims)
        subject_means.append(feats.mean(axis=0))
        within_vars.append(feats.var(axis=0, ddof=1) if len(feats) > 1 else np.zeros(n_dims))

    subject_means = np.stack(subject_means)  # (n_subjects, n_dims)
    within_std = np.sqrt(np.mean(np.stack(within_vars), axis=0))  # (n_dims,)
    between_std = subject_means.std(axis=0, ddof=1)  # (n_dims,)

    # B/W ratio
    bw_ratio = between_std / np.clip(within_std, 1e-8, None)

    # Feature names
    feature_names = []
    feature_names.extend([f"finger_lengths_mm[{i}]" for i in range(5)])
    feature_names.append("palm_width_mm")
    feature_names.append("palm_height_mm")
    feature_names.append("palm_depth_std_mm")
    feature_names.extend([f"finger_widths_mm[{i+1}]" for i in range(4)])
    feature_names.append("scan_distance_mm")

    per_feature = []
    for i in range(n_dims):
        per_feature.append({
            "feature": feature_names[i],
            "between_std": float(between_std[i]),
            "within_std": float(within_std[i]),
            "bw_ratio": float(bw_ratio[i]),
            "verdict": "✅" if bw_ratio[i] >= 1.5 else ("🟡" if bw_ratio[i] >= 1.0 else "❌"),
        })

    summary = {
        "n_subjects": n_subjects,
        "n_frames": len(all_items),
        "mean_bw_ratio": float(bw_ratio.mean()),
        "min_bw_ratio": float(bw_ratio.min()),
        "max_bw_ratio": float(bw_ratio.max()),
        "n_good": int((bw_ratio >= 1.5).sum()),
        "n_marginal": int(((bw_ratio >= 1.0) & (bw_ratio < 1.5)).sum()),
        "n_poor": int((bw_ratio < 1.0).sum()),
    }

    return {"per_feature": per_feature, "summary": summary}


def print_report(report: dict):
    print("=" * 80)
    print("GEOMETRIC FEATURE DISCRIMINABILITY AUDIT — v5.0.0 (13-dim)")
    print("=" * 80)
    print(f"\n{'Feature':<25} {'Between':>10} {'Within':>10} {'B/W':>8} {'Status'}")
    print("-" * 80)
    for f in report["per_feature"]:
        print(f"{f['feature']:<25} {f['between_std']:>10.3f} {f['within_std']:>10.3f} "
              f"{f['bw_ratio']:>8.2f}  {f['verdict']}")

    s = report["summary"]
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Subjects      : {s['n_subjects']}")
    print(f"  Frames        : {s['n_frames']}")
    print(f"  Mean B/W      : {s['mean_bw_ratio']:.2f}")
    print(f"  Min/Max B/W   : {s['min_bw_ratio']:.2f} / {s['max_bw_ratio']:.2f}")
    print(f"  Good (≥1.5)   : {s['n_good']}")
    print(f"  Marginal (1.0): {s['n_marginal']}")
    print(f"  Poor (<1.0)   : {s['n_poor']}")

    if s["n_poor"] > 0:
        poor = [f["feature"] for f in report["per_feature"] if f["verdict"] == "❌"]
        print(f"\n⚠️  Poor features: {', '.join(poor)}")
        print("    → Pertimbangkan untuk drop atau transformasi tambahan.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit geom feature discriminability")
    p.add_argument("--dataset_root", default="../dataset", help="Dataset root")
    p.add_argument("--output", default=None, help="Simpan report ke JSON")
    args = p.parse_args()

    report = audit_discriminability(args.dataset_root)
    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport disimpan di: {args.output}")
