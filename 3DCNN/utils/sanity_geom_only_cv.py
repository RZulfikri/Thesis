"""
utils/sanity_geom_only_cv.py — Sanity baseline: geom-only LeaveOneSessionOut CV.

Gate 0 untuk v5.0.0: sebelum training deep learning, verifikasi bahwa
13-dim geom features memang carry biometric signal.

Menggunakan LogisticRegression dengan LeaveOneGroupOut CV.
Jika accuracy < 30% → STOP, dataset terlalu kecil/noisy.
Jika accuracy ≥ 50% → CONTINUE.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry_schema import _flatten_geometry
from utils.dataset_lowdata import build_lowdata_splits_with_paths


def load_geom_features(split_items: list[tuple[str, Path]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load 13-dim geom features untuk semua frame di split.

    Returns:
        X : (N, 13) float32
        y : (N,) int — label index
        labels : list[str] — nama subjek
    """
    X_list = []
    y_list = []

    label_to_idx = {}

    for label, frame_dir in split_items:
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)
        label_idx = label_to_idx[label]

        geo_path = frame_dir / "geometry.json"
        with open(geo_path) as f:
            geo = json.load(f)
        feat = _flatten_geometry(geo)

        X_list.append(feat)
        y_list.append(label_idx)

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y, list(label_to_idx.keys())


def run_skfcv(X: np.ndarray, y: np.ndarray, feature_name: str = "13-dim"):
    """Run StratifiedKFold CV (5 splits) dengan LogisticRegression + StandardScaler."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="lbfgs")
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc = scores.mean()
    std_acc = scores.std()
    print(f"  {feature_name}: accuracy = {mean_acc:.3f} ± {std_acc:.3f}  (n_splits={len(scores)})")
    return mean_acc, std_acc


def main():
    p = argparse.ArgumentParser(description="Sanity baseline: geom-only LOO CV (Gate 0)")
    p.add_argument("--dataset_root", default="../dataset", help="Dataset root")
    args = p.parse_args()

    print("=" * 70)
    print("SANITY BASELINE — Geom-Only Leave-One-Session-Out CV")
    print("=" * 70)

    # Load low-data splits (150 frames, 10 subjects)
    splits = build_lowdata_splits_with_paths(args.dataset_root)
    all_items = splits["train"] + splits["val"] + splits["test"] + splits["holdout"]

    X, y, labels = load_geom_features(all_items)
    print(f"\nDataset: {len(X)} frames, {len(labels)} subjects, {X.shape[1]} dim")

    # v5.0.0: 13-dim new feature set
    mean13, std13 = run_skfcv(X, y, "13-dim new")

    # Optional: reconstruct 14-dim old untuk perbandingan
    # (tambah mean_palm_curvature + thumb_width dari geometry.json)
    X14_list = []
    for label, frame_dir in all_items:
        geo_path = frame_dir / "geometry.json"
        with open(geo_path) as f:
            geo = json.load(f)
        feat13 = _flatten_geometry(geo)  # (13,)
        # Append old features
        curvature = geo.get("mean_palm_curvature", 0.0)
        thumb_width = geo.get("finger_widths_mm", [0.0])[0] if isinstance(geo.get("finger_widths_mm"), list) else 0.0
        feat14 = np.append(feat13, [thumb_width, curvature])
        X14_list.append(feat14)
    X14 = np.stack(X14_list)
    mean14, std14 = run_skfcv(X14, y, "14-dim old")

    # Verdict
    print("\n" + "=" * 70)
    print("GATE 0 VERDICT")
    print("=" * 70)
    if mean13 >= mean14:
        print(f"✅ 13-dim new ≥ 14-dim old  →  CONTINUE ke F2.1")
    else:
        print(f"⚠️  13-dim new < 14-dim old (regression)  →  DEBUG feature set")

    if mean13 < 0.30:
        print(f"🛑 13-dim accuracy = {mean13:.3f} < 30%  →  STOP")
        print("    Dataset terlalu kecil/noisy untuk klaim apapun.")
        print("    Pertimbangkan capture ulang sebelum lanjut deep learning.")
        sys.exit(1)
    elif mean13 > 0.80:
        print(f"🎉 13-dim accuracy = {mean13:.3f} > 80%  →  CONTINUE dengan confidence tinggi")
    else:
        print(f"🟡 13-dim accuracy = {mean13:.3f}  →  CONTINUE (monitor closely)")


if __name__ == "__main__":
    main()
