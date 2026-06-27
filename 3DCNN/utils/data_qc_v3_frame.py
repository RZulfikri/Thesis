#!/usr/bin/env python3
"""
QC v3 – Frame-level outlier exclusion.

Strategy (per your request):
- We scan per frame to avoid discarding an entire session because of a few bad frames.
- If a session has only a few bad frames, those frames are excluded individually
  (renamed with _QC2_ prefix).
- If a session has many bad frames (high variation), the entire session is excluded.

Logic:
  1. Per session, compute median and MAD (Median Absolute Deviation) per feature.
  2. A frame is an outlier if ANY feature has |value - median| > k * MAD.
  3. If > session_threshold_ratio of frames in the session are outliers,
     the entire session is bad → rename session folder with _QC2_ prefix.
  4. Otherwise, only rename the bad frame folders with _QC2_ prefix.

Reversible: remove _QC2_ prefix to restore.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def flatten_geometry(geom: dict) -> dict[str, float]:
    """Flatten geometry.json into a flat dict of numeric features."""
    features = {}
    for key, val in geom.items():
        if isinstance(val, (int, float)):
            features[key] = float(val)
        elif isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
            for i, x in enumerate(val):
                features[f"{key}_{i}"] = float(x)
    return features


def compute_mad(values: np.ndarray) -> float:
    """Compute Median Absolute Deviation."""
    med = np.median(values)
    abs_dev = np.abs(values - med)
    mad = np.median(abs_dev)
    return mad


def qc_session(session_dir: Path, k: float = 10.0, session_threshold_ratio: float = 0.5) -> dict:
    """
    Perform frame-level QC on a single session.
    Returns a dict with keys:
      - outlier_frames: list of frame folder names that are outliers
      - total_frames: int
      - outlier_ratio: float
      - session_bad: bool (True if entire session should be excluded)
      - features_mad: dict of per-feature MAD values
    """
    # Collect all frames in this session
    frame_dirs = sorted(
        [p for p in session_dir.iterdir()
         if p.is_dir() and not p.name.startswith("_")]
    )
    if not frame_dirs:
        return {"outlier_frames": [], "total_frames": 0, "outlier_ratio": 0.0,
                "session_bad": False, "features_mad": {}}

    # Load geometry for each frame
    frame_geoms = {}
    for fd in frame_dirs:
        geom_file = fd / "geometry.json"
        if not geom_file.exists():
            continue
        with open(geom_file) as f:
            geom = json.load(f)
        frame_geoms[fd.name] = flatten_geometry(geom)

    if not frame_geoms:
        return {"outlier_frames": [], "total_frames": 0, "outlier_ratio": 0.0,
                "session_bad": False, "features_mad": {}}

    # Build per-feature arrays across frames
    all_features = list(next(iter(frame_geoms.values())).keys())
    feature_values = {feat: [] for feat in all_features}
    frame_names_ordered = []
    for fname, feats in frame_geoms.items():
        frame_names_ordered.append(fname)
        for feat in all_features:
            feature_values[feat].append(feats.get(feat, np.nan))

    feature_arrays = {feat: np.array(vals, dtype=float) for feat, vals in feature_values.items()}

    # Compute per-feature median and MAD
    medians = {}
    mads = {}
    for feat in all_features:
        arr = feature_arrays[feat]
        medians[feat] = np.median(arr)
        mads[feat] = compute_mad(arr)

    # Identify outlier frames
    outlier_frames = []
    for fname in frame_names_ordered:
        feats = frame_geoms[fname]
        is_outlier = False
        for feat in all_features:
            val = feats.get(feat, np.nan)
            med = medians[feat]
            mad = mads[feat]
            # If MAD is 0, use a small epsilon to avoid division by zero issues
            if mad == 0:
                mad = 1e-6
            if abs(val - med) > k * mad:
                is_outlier = True
                break
        if is_outlier:
            outlier_frames.append(fname)

    total = len(frame_names_ordered)
    ratio = len(outlier_frames) / total if total > 0 else 0.0
    session_bad = ratio > session_threshold_ratio

    return {
        "outlier_frames": outlier_frames,
        "total_frames": total,
        "outlier_ratio": ratio,
        "session_bad": session_bad,
        "features_mad": {f: float(v) for f, v in mads.items()},
    }


def main(dataset_root: str, k: float = 10.0, session_threshold_ratio: float = 0.5, dry_run: bool = True, subject_filter: str | None = None):
    root = Path(dataset_root)
    if not root.exists():
        print(f"Dataset root not found: {root}")
        sys.exit(1)

    # Find all subject directories
    subjects = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")])
    if subject_filter:
        subjects = [p for p in subjects if p.name == subject_filter]
        if not subjects:
            print(f"Subject '{subject_filter}' not found in {root}")
            sys.exit(1)

    summary = {
        "total_sessions": 0,
        "bad_sessions": [],      # entire session excluded
        "partial_sessions": [],  # some frames excluded
        "total_frames": 0,
        "excluded_frames": 0,
        "excluded_sessions": 0,
    }

    for subj in subjects:
        sessions = sorted([p for p in subj.iterdir() if p.is_dir() and not p.name.startswith("_")])
        for sess in sessions:
            # Skip already-QC'd sessions
            if sess.name.startswith("_QC2_"):
                continue

            result = qc_session(sess, k=k, session_threshold_ratio=session_threshold_ratio)
            if result["total_frames"] == 0:
                continue

            summary["total_sessions"] += 1
            summary["total_frames"] += result["total_frames"]

            if result["session_bad"]:
                # Entire session is bad
                summary["excluded_sessions"] += 1
                summary["excluded_frames"] += result["total_frames"]
                summary["bad_sessions"].append((subj.name, sess.name, result["outlier_ratio"]))
                new_name = sess.parent / f"_QC2_{sess.name}"
                print(f"[SESSION BAD] {subj.name}/{sess.name}: {len(result['outlier_frames'])}/{result['total_frames']} frames outlier ({result['outlier_ratio']:.1%})")
                if not dry_run:
                    sess.rename(new_name)
            elif result["outlier_frames"]:
                # Only some frames are bad
                summary["partial_sessions"].append((subj.name, sess.name, result["outlier_frames"], result["outlier_ratio"]))
                summary["excluded_frames"] += len(result["outlier_frames"])
                print(f"[PARTIAL] {subj.name}/{sess.name}: {len(result['outlier_frames'])}/{result['total_frames']} frames outlier ({result['outlier_ratio']:.1%}) -> exclude {result['outlier_frames']}")
                if not dry_run:
                    for fname in result["outlier_frames"]:
                        frame_dir = sess / fname
                        new_frame_dir = sess / f"_QC2_{fname}"
                        frame_dir.rename(new_frame_dir)
            else:
                print(f"[OK] {subj.name}/{sess.name}: {result['total_frames']} frames clean")

    print("\n" + "="*60)
    print("QC v3 Summary")
    print("="*60)
    print(f"Total sessions scanned: {summary['total_sessions']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Entire sessions excluded: {summary['excluded_sessions']}")
    print(f"Partial sessions (some frames excluded): {len(summary['partial_sessions'])}")
    print(f"Total frames excluded: {summary['excluded_frames']}")
    print(f"Exclusion rate: {summary['excluded_frames']/summary['total_frames']*100:.2f}%" if summary['total_frames'] > 0 else "N/A")
    print(f"\nBad sessions (ratio > {session_threshold_ratio:.0%}):")
    for subj, sess, ratio in summary["bad_sessions"]:
        print(f"  - {subj}/{sess}: {ratio:.1%}")
    print(f"\nPartial sessions:")
    for subj, sess, frames, ratio in summary["partial_sessions"]:
        print(f"  - {subj}/{sess}: {len(frames)} frames ({ratio:.1%}) -> {frames}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QC v3 – Frame-level outlier exclusion")
    parser.add_argument("dataset_root", default="dataset", nargs="?", help="Path to dataset root")
    parser.add_argument("-k", type=float, default=10.0, help="MAD multiplier threshold")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Session exclusion ratio threshold (>this = exclude entire session)")
    parser.add_argument("--apply", action="store_true", help="Apply renames (default is dry-run)")
    parser.add_argument("--subject", type=str, default=None, help="Process only this subject")
    args = parser.parse_args()

    main(args.dataset_root, k=args.k, session_threshold_ratio=args.threshold, dry_run=not args.apply, subject_filter=args.subject)
