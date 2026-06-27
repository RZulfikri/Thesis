"""
dataset.py — Build dataset dari result_frames untuk training GeoAtt-PointNet++.

Normalisasi fitur geometri dilakukan DI SINI (bukan per-frame saat ekstraksi),
menggunakan StandardScaler yang di-fit hanya pada training set — sehingga tidak
ada data leakage dari validation/test set ke proses normalisasi.

Setiap sample terdiri dari dua input:
  - cnn_input.npy   : (N, 6) float32 — point cloud PCA-aligned + unit-sphere
  - geometry.json   : fitur biometrik mm absolut → di-z-score oleh scaler

Output:
  scaler_geometry.pkl   ← StandardScaler fit dari training set
  dataset_split.json    ← mapping frame ke split (train/val/test) + label

Usage:
  python dataset.py --result_dir result_frames --out_dir dataset_cnn
  python dataset.py --result_dir result_frames --out_dir dataset_cnn --val_ratio 0.15 --test_ratio 0.15
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Fitur yang diambil dari geometry.json (nilai mm absolut)
# scan_distance_mm disertakan sebagai fitur karena berkorelasi dengan
# kualitas/noise scan, tapi bukan sebagai divisor normalisasi.
# ---------------------------------------------------------------------------

GEOMETRY_KEYS = [
    ("finger_lengths_mm",    5),   # [thumb, index, middle, ring, pinky]
    ("palm_width_mm",        1),   # scalar
    ("palm_height_mm",       1),   # scalar
    ("palm_depth_std_mm",    1),   # scalar — kelengkungan permukaan telapak
    ("finger_widths_mm",     5),   # [thumb, index, middle, ring, pinky]
    ("mean_palm_curvature",  1),   # scalar — dimensionless
]
# Total dimensi: 5+1+1+1+5+1 = 14
#
# Tidak dimasukkan sebagai fitur CNN:
#   inter_finger_gaps_mm — pose-dependent (bergantung seberapa jauh jari dibuka
#                          saat scan), bukan sifat anatomis tetap. Tetap disimpan
#                          di geometry.json untuk quality check (fingers_too_close).
#   scan_distance_mm     — hanya metadata kualitas, koordinat 3D sudah dalam mm riil.

FEATURE_DIM = sum(n for _, n in GEOMETRY_KEYS)  # = 14


def geo_to_vector(geo: dict) -> np.ndarray | None:
    """
    Konversi geometry.json → vektor fitur numpy (FEATURE_DIM,) float32.
    Mengembalikan None jika ada field yang hilang atau mengandung None.
    """
    vec = []
    for key, n in GEOMETRY_KEYS:
        val = geo.get(key)
        if val is None:
            return None
        if isinstance(val, list):
            if len(val) != n or any(v is None for v in val):
                return None
            vec.extend(float(v) for v in val)
        else:
            vec.append(float(val))
    return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Load semua sample dari result_frames
# ---------------------------------------------------------------------------

def load_samples(result_dir: Path) -> list[dict]:
    """
    Scan result_dir untuk geometry.json dan cnn_input.npy yang valid.

    Layout: result_dir/[label]/[timestamp]/[frame]/geometry.json

    Returns list of:
        {
          "label":      str,
          "timestamp":  str,
          "frame_id":   str,
          "geo_path":   Path,
          "cnn_path":   Path,
          "geo_vec":    np.ndarray (FEATURE_DIM,),
        }
    """
    samples = []
    geo_files = sorted(result_dir.glob("*/*/frame_*/geometry.json"))

    if not geo_files:
        # Fallback: session layout result/[label]/[timestamp]/geometry.json
        geo_files = sorted(result_dir.glob("*/*/geometry.json"))

    n_skip_invalid = 0
    n_skip_no_cnn  = 0
    n_skip_no_vec  = 0

    for gf in geo_files:
        geo = json.load(gf.open())

        # Skip frame yang tidak valid (fingertip fallback, fingers too close)
        if not geo.get("is_valid", True):
            n_skip_invalid += 1
            continue

        # Cek keberadaan cnn_input.npy
        cnn_path = gf.parent / "cnn_input.npy"
        if not cnn_path.exists():
            n_skip_no_cnn += 1
            continue

        # Konversi ke vektor — skip jika ada field yang hilang
        vec = geo_to_vector(geo)
        if vec is None:
            n_skip_no_vec += 1
            continue

        # Parse label dari struktur folder
        parts = gf.parts
        # frame layout: .../[label]/[timestamp]/frame_NN/geometry.json
        # session layout: .../[label]/[timestamp]/geometry.json
        is_frame = gf.parent.name.startswith("frame_")
        if is_frame:
            label     = gf.parent.parent.parent.name
            timestamp = gf.parent.parent.name
            frame_id  = gf.parent.name
        else:
            label     = gf.parent.parent.name
            timestamp = gf.parent.name
            frame_id  = "frame_00"

        samples.append({
            "label":     label,
            "timestamp": timestamp,
            "frame_id":  frame_id,
            "geo_path":  gf,
            "cnn_path":  cnn_path,
            "geo_vec":   vec,
        })

    total = len(samples) + n_skip_invalid + n_skip_no_cnn + n_skip_no_vec
    print(f"  Ditemukan {total} geometry.json")
    print(f"  Dimuat   : {len(samples)} sample valid")
    if n_skip_invalid: print(f"  Skip (is_valid=False)  : {n_skip_invalid}")
    if n_skip_no_cnn:  print(f"  Skip (no cnn_input)    : {n_skip_no_cnn}")
    if n_skip_no_vec:  print(f"  Skip (missing features): {n_skip_no_vec}")

    return samples


# ---------------------------------------------------------------------------
# Train / Val / Test split — per sesi (timestamp), bukan per frame
# ---------------------------------------------------------------------------

def split_by_session(
    samples: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Split samples menjadi train/val/test.

    Split dilakukan per SESI (timestamp) bukan per frame, agar frame dari
    sesi yang sama tidak tersebar antara train dan test (data leakage).

    Setiap label dijaga proporsinya (stratified by label).
    """
    rng = random.Random(seed)

    # Kelompokkan per label → per sesi
    label_sessions: dict[str, list[str]] = defaultdict(list)
    session_samples: dict[str, list[dict]] = defaultdict(list)

    for s in samples:
        sid = f"{s['label']}/{s['timestamp']}"
        session_samples[sid].append(s)
        if sid not in label_sessions[s["label"]]:
            label_sessions[s["label"]].append(sid)

    train, val, test = [], [], []

    for label, sessions in label_sessions.items():
        rng.shuffle(sessions)
        n = len(sessions)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Terlalu sedikit sesi — masukkan semua ke train
            print(f"  [warn] Label '{label}': hanya {n} sesi, semua masuk train")
            for sid in sessions:
                train.extend(session_samples[sid])
            continue

        for sid in sessions[:n_train]:
            train.extend(session_samples[sid])
        for sid in sessions[n_train:n_train + n_val]:
            val.extend(session_samples[sid])
        for sid in sessions[n_train + n_val:]:
            test.extend(session_samples[sid])

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Build scaler dari training set
# ---------------------------------------------------------------------------

def fit_scaler(train_samples: list[dict]) -> StandardScaler:
    """
    Fit StandardScaler dari training set saja.
    Setiap fitur akan memiliki mean=0, std=1 berdasarkan distribusi training data.
    """
    X = np.stack([s["geo_vec"] for s in train_samples])  # (N_train, FEATURE_DIM)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ---------------------------------------------------------------------------
# Simpan dataset
# ---------------------------------------------------------------------------

def save_dataset(
    splits: dict[str, list[dict]],
    scaler: StandardScaler,
    label_to_id: dict[str, int],
    out_dir: Path,
):
    """
    Simpan scaler, label map, dan split manifest ke out_dir.

    Untuk setiap split, simpan juga pre-computed geometry vectors yang sudah
    di-normalize — sehingga training loop tidak perlu baca ulang json setiap epoch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simpan scaler
    scaler_path = out_dir / "scaler_geometry.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler tersimpan → {scaler_path}")
    print(f"  Feature means : {np.round(scaler.mean_, 2)}")
    print(f"  Feature stds  : {np.round(scaler.scale_, 2)}")

    # Simpan label map
    label_map_path = out_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({"label_to_id": label_to_id,
                   "id_to_label": {v: k for k, v in label_to_id.items()}}, f, indent=2)
    print(f"Label map tersimpan → {label_map_path}  ({len(label_to_id)} kelas)")

    # Simpan per-split: manifest JSON + numpy array geometry yang sudah dinormalisasi
    split_manifest = {}
    for split_name, samples in splits.items():
        if not samples:
            continue

        # Geometry vectors yang sudah dinormalisasi
        X_raw = np.stack([s["geo_vec"] for s in samples])        # (N, 19)
        X_norm = scaler.transform(X_raw).astype(np.float32)       # (N, 19)
        y = np.array([label_to_id[s["label"]] for s in samples], dtype=np.int64)

        geo_npy_path = out_dir / f"{split_name}_geometry.npy"
        y_npy_path   = out_dir / f"{split_name}_labels.npy"
        np.save(geo_npy_path, X_norm)
        np.save(y_npy_path, y)

        # Manifest: mapping index → cnn_path (untuk lazy-load point cloud saat training)
        entries = []
        for i, s in enumerate(samples):
            entries.append({
                "idx":       i,
                "label":     s["label"],
                "label_id":  label_to_id[s["label"]],
                "timestamp": s["timestamp"],
                "frame_id":  s["frame_id"],
                "cnn_path":  str(s["cnn_path"]),
            })
        split_manifest[split_name] = entries

        print(f"  {split_name:5s}: {len(samples):4d} frame  "
              f"→ {geo_npy_path.name}, {y_npy_path.name}")

    manifest_path = out_dir / "dataset_split.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "feature_keys":  [k for k, _ in GEOMETRY_KEYS],
            "feature_dim":   FEATURE_DIM,
            "n_classes":     len(label_to_id),
            "splits":        split_manifest,
        }, f, indent=2)
    print(f"Manifest tersimpan → {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(
    result_dir: Path,
    out_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    print(f"{'='*60}")
    print(f"BUILD DATASET")
    print(f"  result_dir : {result_dir.resolve()}")
    print(f"  out_dir    : {out_dir.resolve()}")
    print(f"  split      : train/{val_ratio}/{test_ratio}  seed={seed}")
    print(f"{'='*60}\n")

    # 1. Load samples
    print("1. Load samples...")
    samples = load_samples(result_dir)
    if not samples:
        print("Tidak ada sample valid ditemukan.")
        return

    # 2. Label encoding
    labels = sorted({s["label"] for s in samples})
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    print(f"\n2. Label: {label_to_id}")

    # 3. Split
    print("\n3. Split per sesi (stratified by label)...")
    splits = split_by_session(samples, val_ratio=val_ratio,
                              test_ratio=test_ratio, seed=seed)
    for split_name, split_samples in splits.items():
        label_counts = defaultdict(int)
        for s in split_samples:
            label_counts[s["label"]] += 1
        print(f"  {split_name:5s}: {len(split_samples):4d} frame  {dict(label_counts)}")

    # 4. Fit scaler HANYA dari training set
    print("\n4. Fit StandardScaler dari training set...")
    if not splits["train"]:
        print("  [error] Training set kosong.")
        return
    scaler = fit_scaler(splits["train"])
    print(f"  Scaler fit dari {len(splits['train'])} frame training")

    # 5. Simpan
    print("\n5. Simpan dataset...")
    save_dataset(splits, scaler, label_to_id, out_dir)

    print(f"\n{'='*60}")
    print(f"Selesai. Dataset tersimpan di: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CNN dataset dari result_frames (scaler fit pada training set)")
    parser.add_argument("--result_dir", default="result_frames",
                        help="direktori hasil process_single_frames (default: result_frames)")
    parser.add_argument("--out_dir", default="dataset_cnn",
                        help="direktori output dataset (default: dataset_cnn)")
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    build_dataset(
        result_dir=Path(args.result_dir),
        out_dir=Path(args.out_dir),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
