"""
utils/dataset.py — Dataset loader dan pair generator untuk palm recognition.

Mendukung dua layout folder:

1. **Session layout** (output process_all_scans.py — multi-frame ICP):
    data_dir/[label]/[timestamp]/cnn_input.npy + geometry.json

2. **Frame layout** (output process_single_frames.py — single-frame tanpa ICP):
    data_dir/[label]/[timestamp]/frame_NN/cnn_input.npy + geometry.json

   Untuk frame layout, split train/val HARUS dilakukan di level sesi (timestamp),
   bukan di level frame, untuk mencegah data leakage.
   Gunakan scan_dataset_frames() + split_sessions_frames() / make_loso_splits_frames().

cnn_input.npy berisi semua titik (N variatif, ~50k-150k) tanpa FPS.
Sampling ke n_points dilakukan on-the-fly di __getitem__.

Normalisasi fitur geometri dilakukan via GeometryNormalizer (z-score, fit dari
training set saja) — bukan dari normalized_geometry.json. Geometry dibaca langsung
dari geometry.json (nilai mm absolut).
"""

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

# ---- Geometry schema (v5.0.0) — import dari geometry_schema.py agar skrip diagnostic ----
# tanpa torch bisa mengakses definisi yang sama.
from utils.geometry_schema import GEOMETRY_KEYS, GEOMETRY_DIM, _flatten_geometry


def _fps_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Farthest Point Sampling (greedy) — backup novelty mode.
    Input: pts (N, D), output: (n, D)
    """
    N = len(pts)
    if N <= n:
        idx = np.concatenate([np.arange(N), np.random.choice(N, n - N, replace=True)])
        return pts[idx]
    selected = np.zeros(n, dtype=np.int64)
    dist = np.full(N, np.inf)
    current = 0
    xyz = pts[:, :3]
    for i in range(n):
        selected[i] = current
        d = np.sum((xyz - xyz[current]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        current = int(np.argmax(dist))
    return pts[selected]


# v8 — skor kualitas scan 3D untuk QA-ArcFace (margin adaptif). Komposit dari sinyal
# geometry.json yang tersedia per-frame: densitas titik (point_count) + jarak scan in-range.
# Output ∈ [0,1]; 1 = kualitas tinggi. Heuristik & tunable (lihat docs/ARCFACE_EXPLAINED.md §7).
QUALITY_PCOUNT_REF = 20000.0   # ~jumlah titik scan telapak yang baik

def compute_quality(point_count: int, scan_distance_mm: float | None = None,
                    palm_depth_std_mm: float | None = None) -> float:
    """Skor kualitas scan ∈ [0,1] dari sinyal geometry.json. Primer = densitas titik;
    sekunder = jarak scan in-range (terbaik ~300mm, 150/450mm → 0). palm_depth_std
    disediakan utk ekstensi (belum dibobot)."""
    density = min(max(float(point_count) / QUALITY_PCOUNT_REF, 0.0), 1.0)
    q = density
    if scan_distance_mm is not None and scan_distance_mm > 0:
        dist = 1.0 - abs(float(scan_distance_mm) - 300.0) / 150.0
        dist = min(max(dist, 0.0), 1.0)
        q = 0.7 * density + 0.3 * dist
    return float(min(max(q, 0.0), 1.0))


def _sample_points(pts: np.ndarray, n: int, method: Literal["random", "fps"] = "random") -> np.ndarray:
    """Sample n titik dari point cloud (N, 6)."""
    N = len(pts)
    if N <= n:
        idx = np.concatenate([np.arange(N), np.random.choice(N, n - N, replace=True)])
        return pts[idx]
    if method == "fps":
        return _fps_sample(pts, n)
    idx = np.random.choice(N, n, replace=False)
    return pts[idx]


# ---------------------------------------------------------------------------
# v7.2.0 — Representation ablation (R1 raw PLY / R2 canonical NPY / R3 pre-FPS)
# ---------------------------------------------------------------------------
# repr_mode menentukan FILE sumber point cloud (geometry.json selalu sama):
#   "canonical_npy" (R2, default) : cnn_input.npy      — PCA-align + unit-sphere
#   "fps_npy"       (R3)          : cnn_input_fps.npy  — R2 + FPS 8192 (no runtime sampling)
#   "raw_ply"       (R1)          : output.ply         — koordinat kamera asli, TANPA kanonikalisasi
REPR_CANONICAL_NPY = "canonical_npy"
REPR_FPS_NPY       = "fps_npy"
REPR_RAW_PLY       = "raw_ply"
# v8 — alignment ablation (full-cloud, TANPA FPS). Diturunkan dari output.ply via
# 3DRegistration/make_align_variants.py (utils/alignment.py = sumber-tunggal).
REPR_ALIGN_CENTER       = "align_center"        # A1: center saja
REPR_ALIGN_CENTERSCALE  = "align_centerscale"   # A2: center + unit-sphere (tanpa rotasi)
REPR_ALIGN_PCA_ROBUST   = "align_pca_robust"    # A4: PCA deterministik (fix 90°)
REPR_ALIGN_ANATOMICAL   = "align_anatomical"    # A5: landmark anatomis (fix 90°)
REPR_MODES = (REPR_CANONICAL_NPY, REPR_FPS_NPY, REPR_RAW_PLY,
              REPR_ALIGN_CENTER, REPR_ALIGN_CENTERSCALE,
              REPR_ALIGN_PCA_ROBUST, REPR_ALIGN_ANATOMICAL)

_REPR_FILE = {
    REPR_CANONICAL_NPY: "cnn_input.npy",
    REPR_FPS_NPY:       "cnn_input_fps.npy",
    REPR_RAW_PLY:       "output.ply",
    REPR_ALIGN_CENTER:       "align_center.npy",
    REPR_ALIGN_CENTERSCALE:  "align_centerscale.npy",
    REPR_ALIGN_PCA_ROBUST:   "align_pca_robust.npy",
    REPR_ALIGN_ANATOMICAL:   "align_anatomical.npy",
}


def _load_ply_xyz_normals(ply_path: Path) -> np.ndarray:
    """
    Load output.ply (R1) → (N, 6) float32 = XYZ + normals, koordinat kamera asli.

    Pakai open3d (sudah dipakai di pipeline 3DRegistration). output.ply hasil regen
    v7.2.0 sudah menyimpan normals, jadi tidak perlu re-estimasi.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        # Fail-fast: open3d mengembalikan cloud KOSONG untuk file hilang (hanya warning),
        # yang baru meledak jauh di hilir sebagai ValueError np.random.choice (run v7.2.0
        # 2026-06-08). output.ply ter-gitignore (3DCNN/dataset/**/*.ply) — pastikan
        # di-push dengan `git add -f` sebelum training repr_mode=raw_ply.
        raise FileNotFoundError(
            f"[raw_ply] {ply_path} tidak ada. Dataset di checkout ini tidak lengkap "
            f"untuk R1 — output.ply kemungkinan ter-gitignore dan belum di-push. "
            f"Regenerasi via process_single_frames.py atau push PLY dengan git add -f."
        )
    import open3d as o3d  # lazy import — hanya dibutuhkan untuk repr_mode=raw_ply
    pcd = o3d.io.read_point_cloud(str(ply_path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if len(xyz) == 0:
        raise ValueError(f"[raw_ply] {ply_path} terbaca tapi kosong (0 titik) — file korup?")
    if pcd.has_normals():
        nrm = np.asarray(pcd.normals, dtype=np.float32)
    else:
        # Fallback (seharusnya tidak terjadi pada regen v7.2.0): estimasi cepat
        pcd.estimate_normals()
        nrm = np.asarray(pcd.normals, dtype=np.float32)
    if len(nrm) != len(xyz):
        nrm = np.zeros_like(xyz)
    return np.concatenate([xyz, nrm], axis=1).astype(np.float32)


def _load_geo_dict(session_dir: Path) -> dict:
    """
    Load geometry dict dari geometry.json (nilai mm absolut).
    Normalisasi dilakukan oleh GeometryNormalizer di training loop — bukan di sini.
    """
    with open(Path(session_dir) / "geometry.json") as f:
        return json.load(f)


def load_geometry(session_dir: Path) -> np.ndarray:
    """
    Load fitur geometri (GEOMETRY_DIM,) dari geometry.json.
    Digunakan untuk fitting GeometryNormalizer pada training set.
    """
    return _flatten_geometry(_load_geo_dict(session_dir))


def load_session(session_dir: Path,
                 repr_mode: str = REPR_CANONICAL_NPY) -> tuple[np.ndarray, np.ndarray]:
    """
    Load satu sesi/frame scan: point cloud (sesuai repr_mode) + geometry.json.

    Args:
        session_dir : frame/session dir
        repr_mode   : "canonical_npy" (R2, default) | "fps_npy" (R3) | "raw_ply" (R1)
                      → lihat REPR_MODES. geometry.json selalu sama lintas repr_mode.

    Returns:
        cloud : (N, 6) float32          — XYZ + normals (R2/R3 PCA-aligned+unit-sphere;
                                          R1 koordinat kamera asli tanpa kanonikalisasi)
        geom  : (GEOMETRY_DIM,) float32 — fitur geometri mm absolut (belum di-normalize)
    """
    session_dir = Path(session_dir)
    if repr_mode == REPR_RAW_PLY:
        cloud = _load_ply_xyz_normals(session_dir / _REPR_FILE[REPR_RAW_PLY])
    else:
        fname = _REPR_FILE.get(repr_mode, _REPR_FILE[REPR_CANONICAL_NPY])
        cloud = np.load(session_dir / fname).astype(np.float32)
    assert cloud.ndim == 2 and cloud.shape[1] == 6, (
        f"point cloud ({repr_mode}) harus (N, 6), dapat {cloud.shape} di {session_dir}. "
        f"Jalankan ulang process_single_frames.py / make_fps.py untuk regenerasi."
    )
    geom = _flatten_geometry(_load_geo_dict(session_dir))
    return cloud, geom


def scan_dataset(data_dir: Path) -> dict[str, list[Path]]:
    """
    Scan folder data_dir untuk semua sesi yang valid.

    Layout yang diharapkan:
        data_dir/[label]/[timestamp]/cnn_input.npy + geometry.json

    Returns:
        dict: {label: [sorted list of session Paths]}
    """
    data_dir = Path(data_dir)
    label_sessions: dict[str, list[Path]] = {}

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        sessions = sorted(
            p for p in label_dir.iterdir()
            if p.is_dir()
            and not p.name.startswith("_QC2_")
            and not p.name.startswith("_QUARANTINE_")
            and (p / "cnn_input.npy").exists()
            and (p / "geometry.json").exists()
        )
        if sessions:
            label_sessions[label_dir.name] = sessions

    return label_sessions


def generate_pairs(
    label_sessions: dict[str, list[Path]],
    seed: int = 42,
) -> list[tuple[Path, Path, float]]:
    """
    Hasilkan pasangan genuine/impostor yang seimbang.

    Genuine  : semua C(n,2) kombinasi dalam label yang sama (label 1.0)
    Impostor : pasangan antar-label secara acak, jumlah sama dengan genuine (label 0.0)

    Returns:
        list of (session_a, session_b, label)
    """
    rng = random.Random(seed)

    genuine_pairs: list[tuple[Path, Path, float]] = []
    for sessions in label_sessions.values():
        for a, b in combinations(sessions, 2):
            genuine_pairs.append((a, b, 1.0))

    n = len(genuine_pairs)
    if n == 0:
        return []

    labels = list(label_sessions.keys())
    impostor_pairs: list[tuple[Path, Path, float]] = []
    for _ in range(n):
        lab_a, lab_b = rng.sample(labels, 2)
        s_a = rng.choice(label_sessions[lab_a])
        s_b = rng.choice(label_sessions[lab_b])
        impostor_pairs.append((s_a, s_b, 0.0))

    all_pairs = genuine_pairs + impostor_pairs
    rng.shuffle(all_pairs)
    return all_pairs


def balance_label_sessions(
    label_sessions: dict[str, list[Path]],
    min_count: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, list[Path]], int]:
    """
    Standarisasi dataset: cap setiap label ke min_count sesi.

    Jika min_count tidak diberikan, otomatis menggunakan jumlah minimum
    di antara semua label (misalnya jika alji=16, rahmat=10, feby=10 → cap ke 10).

    Sesi yang dipilih diacak dengan seed konsisten agar hasil reproducible.
    Ini memastikan setiap subjek berkontribusi sama banyak ke training/evaluasi.

    Args:
        label_sessions : {label: [session_paths]}
        min_count      : jumlah sesi per label (None = min across labels)
        seed           : random seed untuk konsistensi

    Returns:
        balanced   : {label: [session_paths]} dengan len = min_count tiap label
        min_count  : jumlah sesi yang digunakan
    """
    rng = random.Random(seed)
    if min_count is None:
        min_count = min(len(v) for v in label_sessions.values())
    balanced = {}
    for label, sessions in label_sessions.items():
        shuffled = sessions.copy()
        rng.shuffle(shuffled)
        balanced[label] = shuffled[:min_count]
    return balanced, min_count


def balance_label_frames(
    session_groups: dict[str, dict[str, list[Path]]],
    min_sessions: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, dict[str, list[Path]]], int]:
    """
    Standarisasi dataset frame layout: cap setiap label ke min_sessions timestamp.

    Balancing dilakukan di level sesi (timestamp), bukan di level frame,
    untuk menghindari data leakage dan menjaga konsistensi session-level split.

    Args:
        session_groups : {label: {timestamp: [frame_dirs]}}
        min_sessions   : jumlah sesi per label (None = min across labels)
        seed           : random seed

    Returns:
        balanced     : {label: {timestamp: [frame_dirs]}} dengan len(timestamps) = min_sessions
        min_sessions : jumlah sesi yang digunakan
    """
    rng = random.Random(seed)
    if min_sessions is None:
        min_sessions = min(len(ts_dict) for ts_dict in session_groups.values())
    balanced = {}
    for label, ts_dict in session_groups.items():
        timestamps = sorted(ts_dict.keys())
        ts_shuffled = timestamps.copy()
        rng.shuffle(ts_shuffled)
        selected = ts_shuffled[:min_sessions]
        balanced[label] = {ts: ts_dict[ts] for ts in selected}
    return balanced, min_sessions


def split_sessions(
    label_sessions: dict[str, list[Path]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """
    Split sesi per label menjadi train/val (stratified per label).

    Returns:
        train_label_sessions, val_label_sessions
    """
    rng = random.Random(seed)
    train: dict[str, list[Path]] = {}
    val:   dict[str, list[Path]] = {}

    for label, sessions in label_sessions.items():
        shuffled = sessions.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val[label]   = shuffled[:n_val]
        train[label] = shuffled[n_val:]

    return train, val


def make_loso_splits(label_sessions: dict[str, list[Path]]):
    """
    Leave-One-Session-Out cross-validation.

    Setiap fold: satu sesi dari setiap label dijadikan test, sisanya train.
    Jumlah fold = max(len(sessions)) di antara semua label.

    Yields:
        fold_idx       : int
        train_sessions : dict[str, list[Path]]
        test_sessions  : dict[str, list[Path]]
    """
    n_folds = max(len(ss) for ss in label_sessions.values())

    for fold_idx in range(n_folds):
        train: dict[str, list[Path]] = {}
        test:  dict[str, list[Path]] = {}

        for label, sessions in label_sessions.items():
            if not sessions:
                continue
            test_idx = fold_idx % len(sessions)
            test[label]  = [sessions[test_idx]]
            train[label] = [s for i, s in enumerate(sessions) if i != test_idx]

        yield fold_idx, train, test


# ---------------------------------------------------------------------------
# Frame-layout support (process_single_frames.py output)
# ---------------------------------------------------------------------------

def _frame_is_valid(frame_dir: Path) -> bool:
    """
    Kembalikan False jika frame memiliki is_valid=False di geometry.json.
    
    Jika geometry.json valid tetapi ada stale invalid_frame.json dari run lama,
    tetap anggap valid (source of truth = geometry.json).
    """
    geo_path = frame_dir / "geometry.json"
    if not geo_path.exists():
        return False
    with open(geo_path) as f:
        geo = json.load(f)
    # scan_distance_out_of_range sering false positive — frame 160-180mm
    # masih punya geometry lengkap dan point count normal.
    # Hanya reject kalau fingertip detection benar-benar gagal.
    is_valid = geo.get("is_valid", True)
    if not is_valid:
        issues = geo.get("quality_issues", [])
        # Whitelist issue yang tidak membuat frame benar-benar invalid:
        # - scan_distance_out_of_range: false positive, frame 160-180mm masih valid
        # - knuckle_fallback: estimasi anatomis masih menghasilkan geometry yang usable
        real_issues = [
            i for i in issues
            if "scan_distance" not in i and "knuckle_fallback" not in i
        ]
        if not real_issues:
            return True
    return is_valid


def scan_dataset_frames(
    data_dir: Path,
    filter_invalid: bool = True,
) -> tuple[dict[str, list[Path]], dict[str, dict[str, list[Path]]]]:
    """
    Scan frame-level layout: data_dir/[label]/[timestamp]/frame_*/

    Hasil scan di-cache ke `.dataset_scan_cache.pkl` di dalam data_dir
    supaya training run berikutnya (seed/variant lain) tidak scan ulang
    dari Google Drive yang lambat.

    Args:
        filter_invalid : jika True, skip frame individual dengan is_valid=False
                         (bukan seluruh sesi — v0.3.0 fix)

    Returns:
        label_frames   : {label: [list of frame_dirs]}  ← untuk PalmPairDataset
        session_groups : {label: {timestamp: [frame_dirs]}}  ← untuk split level sesi
    """
    import pickle, time
    data_dir = Path(data_dir)
    cache_path = data_dir / ".dataset_scan_cache.pkl"

    # ── Coba load cache ──
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            # Invalidate kalau filter_invalid berubah atau cache terlalu tua (>24 jam)
            if cached.get("filter_invalid") == filter_invalid and \
               (time.time() - cached.get("mtime", 0)) < 86400:
                print(f"[Dataset] Loaded {len(cached['label_frames'])} labels from scan cache")
                return cached["label_frames"], cached["session_groups"]
        except Exception:
            pass  # corrupt cache, scan ulang

    # ── Scan dari disk ──
    label_frames:   dict[str, list[Path]] = {}
    session_groups: dict[str, dict[str, list[Path]]] = defaultdict(dict)

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        all_frames: list[Path] = []
        for ts_dir in sorted(label_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            if ts_dir.name.startswith("_QC2_") or ts_dir.name.startswith("_QUARANTINE_"):
                continue
            frames = sorted(
                p for p in ts_dir.iterdir()
                if p.is_dir()
                and p.name.startswith("frame_")
                and not p.name.startswith("_QC2_")
                and (p / "cnn_input.npy").exists()
                and (p / "geometry.json").exists()
            )
            if not frames:
                continue
            # v0.3.0: filter per frame, bukan per sesi
            if filter_invalid:
                frames = [f for f in frames if _frame_is_valid(f)]
            if not frames:
                continue
            session_groups[label_dir.name][ts_dir.name] = frames
            all_frames.extend(frames)
        if all_frames:
            label_frames[label_dir.name] = all_frames

    # ── Simpan cache ──
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "label_frames": label_frames,
                "session_groups": dict(session_groups),
                "filter_invalid": filter_invalid,
                "mtime": time.time(),
            }, f)
        print(f"[Dataset] Scan cache saved: {cache_path}")
    except Exception as e:
        print(f"[Dataset] Warning: failed to save scan cache: {e}")

    return label_frames, dict(session_groups)


def split_holdout_sessions(
    session_groups: dict[str, dict[str, list[Path]]],
    n_holdout_sessions: int = 1,
    n_probe_frames: int = 3,
    seed: int = 42,
) -> tuple[dict[str, dict[str, list[Path]]], dict[str, list[Path]]]:
    """
    Pisahkan n_holdout_sessions sesi per subjek sebagai real test set.
    Ambil n_probe_frames frame acak dari sesi holdout sebagai probe.

    Dipanggil SETELAH scan + balance, SEBELUM split_sessions_three_way.
    Sesi holdout TIDAK PERNAH masuk training maupun val/test split.

    Args:
        session_groups     : {label: {timestamp: [frame_dirs]}}
        n_holdout_sessions : jumlah sesi per label yang di-hold-out
        n_probe_frames     : jumlah frame acak dari sesi holdout untuk real test
        seed               : random seed (gunakan SPLIT_SEED agar reproducible)

    Returns:
        remaining_groups : session_groups tanpa sesi holdout (untuk training)
        holdout_probes   : {label: [n_probe_frames frame_dirs]} untuk evaluate_holdout
    """
    rng = random.Random(seed)
    remaining: dict[str, dict[str, list[Path]]] = {}
    holdout_probes: dict[str, list[Path]] = {}

    for label, ts_dict in session_groups.items():
        shuffled = sorted(ts_dict.keys())
        rng.shuffle(shuffled)
        holdout_ts   = shuffled[:n_holdout_sessions]
        remaining_ts = shuffled[n_holdout_sessions:]

        all_holdout_frames = [f for ts in holdout_ts for f in ts_dict[ts]]
        holdout_probes[label] = rng.sample(
            all_holdout_frames, min(n_probe_frames, len(all_holdout_frames))
        )
        remaining[label] = {ts: ts_dict[ts] for ts in remaining_ts}

    return remaining, holdout_probes


def split_sessions_three_way(
    session_groups: dict[str, dict[str, list[Path]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]]]:
    """
    Split session_groups → (train_frames, val_frames, test_frames).

    Split dilakukan di level SESI (timestamp) untuk mencegah data leakage,
    lalu di-flatten ke frame dirs untuk PalmPairDataset.
    Untuk 14 sesi per subjek → ~10 train, ~2 val, ~2 test.

    test_ratio = 1 - train_ratio - val_ratio (otomatis).

    Returns:
        train_frames, val_frames, test_frames  — {label: [frame_dirs]}
    """
    rng = random.Random(seed)
    train: dict[str, list[Path]] = {}
    val:   dict[str, list[Path]] = {}
    test:  dict[str, list[Path]] = {}

    for label, ts_dict in session_groups.items():
        shuffled = sorted(ts_dict.keys())
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test  = max(1, round(n * (1 - train_ratio - val_ratio)))
        n_val   = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        test_ts  = shuffled[:n_test]
        val_ts   = shuffled[n_test:n_test + n_val]
        train_ts = shuffled[n_test + n_val:]

        test[label]  = [f for ts in test_ts  for f in ts_dict[ts]]
        val[label]   = [f for ts in val_ts   for f in ts_dict[ts]]
        train[label] = [f for ts in train_ts for f in ts_dict[ts]]

    return train, val, test


def split_sessions_frames(
    session_groups: dict[str, dict[str, list[Path]]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """
    Split di level sesi (timestamp), kemudian expand ke frame.

    Semua frame dari satu timestamp selalu masuk ke split yang sama
    sehingga tidak ada data leakage antar train/val.

    Returns:
        train_label_frames, val_label_frames  — {label: [list of frame_dirs]}
    """
    rng   = random.Random(seed)
    train: dict[str, list[Path]] = {}
    val:   dict[str, list[Path]] = {}

    for label, ts_dict in session_groups.items():
        timestamps = sorted(ts_dict.keys())
        shuffled   = timestamps.copy()
        rng.shuffle(shuffled)
        n_val      = max(1, int(len(shuffled) * val_ratio))

        val_frames   = [f for ts in shuffled[:n_val]  for f in ts_dict[ts]]
        train_frames = [f for ts in shuffled[n_val:]  for f in ts_dict[ts]]

        val[label]   = val_frames
        train[label] = train_frames

    return train, val


def make_loso_splits_frames(
    session_groups: dict[str, dict[str, list[Path]]],
):
    """
    Leave-One-Session-Out cross-validation untuk frame layout.

    Setiap fold: satu timestamp per label dijadikan test, sisanya train.
    Semua frame dari timestamp yang di-hold-out masuk ke test — tidak ada leakage.

    Yields:
        fold_idx            : int
        train_label_frames  : {label: [frame_dirs]}
        test_label_frames   : {label: [frame_dirs]}
    """
    n_folds = max(len(ts_dict) for ts_dict in session_groups.values())

    for fold_idx in range(n_folds):
        train: dict[str, list[Path]] = {}
        test:  dict[str, list[Path]] = {}

        for label, ts_dict in session_groups.items():
            timestamps = sorted(ts_dict.keys())
            if not timestamps:
                continue
            test_ts  = timestamps[fold_idx % len(timestamps)]
            test[label]  = list(ts_dict[test_ts])
            train[label] = [f for ts, frames in ts_dict.items()
                            if ts != test_ts for f in frames]

        yield fold_idx, train, test


class PalmPairDataset(Dataset):
    """
    Dataset pasangan sesi untuk training siamese.

    Semua cnn_input.npy di-preload ke memori saat __init__ untuk menghindari
    pembacaan disk berulang selama training. Sampling ke n_points dilakukan
    on-the-fly di __getitem__.

    Setiap item adalah dict:
        pts_a  : (n_points, 6)    float32 tensor
        geom_a : (GEOMETRY_DIM,)  float32 tensor — fitur dari geometry.json, di-z-score oleh normalizer
        pts_b  : (n_points, 6)    float32 tensor
        geom_b : (GEOMETRY_DIM,)  float32 tensor
        label  : scalar           float32 tensor  (1.0 genuine, 0.0 impostor)
    """

    def __init__(
        self,
        label_sessions: dict[str, list[Path]],
        n_points: int = 4096,
        sampling: Literal["random", "fps"] = "random",
        augment=None,
        geom_augment=None,
        normalizer=None,
        seed: int = 42,
        repr_mode: str = REPR_CANONICAL_NPY,
    ):
        self.pairs        = generate_pairs(label_sessions, seed=seed)
        self.n_points     = n_points
        self.sampling     = sampling
        self.augment      = augment       # point cloud augmentation
        self.geom_augment = geom_augment  # geometry augmentation (setelah normalisasi)
        self.normalizer   = normalizer
        self.repr_mode    = repr_mode

        # Preload semua sesi unik ke cache
        all_dirs = {d for pair in self.pairs for d in (pair[0], pair[1])}
        print(f"Preloading {len(all_dirs)} sesi ke memori (repr={repr_mode})...")
        self._cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {
            d: load_session(d, repr_mode=repr_mode) for d in sorted(all_dirs)
        }
        print(f"Preload selesai. Total pasangan: {len(self.pairs)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        dir_a, dir_b, label = self.pairs[idx]
        cloud_a, geom_a = self._cache[dir_a]
        cloud_b, geom_b = self._cache[dir_b]

        pts_a = _sample_points(cloud_a, self.n_points, self.sampling)
        pts_b = _sample_points(cloud_b, self.n_points, self.sampling)

        if self.augment is not None:
            pts_a = self.augment(pts_a)
            pts_b = self.augment(pts_b)

        if self.normalizer is not None:
            geom_a = self.normalizer.transform(geom_a)
            geom_b = self.normalizer.transform(geom_b)

        if self.geom_augment is not None:
            geom_a = self.geom_augment(geom_a)
            geom_b = self.geom_augment(geom_b)

        return {
            "pts_a":  torch.from_numpy(np.ascontiguousarray(pts_a)),
            "geom_a": torch.from_numpy(np.ascontiguousarray(geom_a)),
            "pts_b":  torch.from_numpy(np.ascontiguousarray(pts_b)),
            "geom_b": torch.from_numpy(np.ascontiguousarray(geom_b)),
            "label":  torch.tensor(label, dtype=torch.float32),
        }


class PalmFrameDataset(Dataset):
    """
    Dataset individual frames untuk training dengan OnlineTripletLoss.

    Mode default (`preload_augment=False`):
        Sampling + augmentasi on-the-fly di `__getitem__`.
        RAM usage rendah (~2-3 GB), tapi CPU sibuk tiap batch.

    Mode `preload_augment=True` (rekomendasi untuk A100 / RAM > 64 GB):
        Semua variasi augmented di-precompute sekali di `__init__`.
        RAM usage lebih tinggi (~10-30 GB tergantung repeat), tapi
        training jauh lebih cepat karena CPU hanya pick dari array.

    Tiap item:
        pts       : (n_points, 6)   float32 tensor
        geom      : (GEOMETRY_DIM,) float32 tensor
        label_idx : long tensor — indeks label (0..n_labels-1)

    Args:
        label_sessions    : {label: [frame_dirs]}
        n_points          : jumlah titik yang di-sample
        sampling          : 'random' atau 'fps'
        augment           : PointCloudAugmentor atau None
        geom_augment      : GeometryAugmentor atau None
        normalizer        : GeometryNormalizer atau None
        repeat            : tiap frame muncul `repeat` kali per epoch
        preload_augment   : jika True, pre-generate semua augmented variant di RAM
    """

    def __init__(
        self,
        label_sessions: dict[str, list[Path]],
        n_points: int = 4096,
        sampling: Literal["random", "fps"] = "random",
        augment=None,
        geom_augment=None,
        normalizer=None,
        repeat: int = 10,
        preload_augment: bool = False,
        repr_mode: str = REPR_CANONICAL_NPY,
    ):
        self.n_points          = n_points
        self.sampling          = sampling
        self.augment           = augment
        self.geom_augment      = geom_augment
        self.normalizer        = normalizer
        self.repeat            = max(1, int(repeat))
        self.preload_augment   = preload_augment
        self.repr_mode         = repr_mode

        # Mapping label string → integer index
        self.labels       = sorted(label_sessions.keys())
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}

        # Flat list of (frame_path, label_idx, session_idx)
        # session_idx: indeks unik per sesi — dipakai untuk cross-session triplet mining (C2)
        self.samples: list[tuple[Path, int, int]] = []
        self._cache:  dict[Path, tuple[np.ndarray, np.ndarray]] = {}

        # Build session → integer index mapping (session = parent dir dari frame_path)
        session_to_idx: dict[Path, int] = {}

        for lbl, frames in label_sessions.items():
            idx = self.label_to_idx[lbl]
            for fp in frames:
                sess = fp.parent
                if sess not in session_to_idx:
                    session_to_idx[sess] = len(session_to_idx)
                self.samples.append((fp, idx, session_to_idx[sess]))
                if fp not in self._cache:
                    self._cache[fp] = load_session(fp, repr_mode=self.repr_mode)

        self.n_sessions = len(session_to_idx)

        # ── Mode preload: pre-generate semua augmented variant ───────────────
        if self.preload_augment:
            self._preloaded: list[dict] = []
            total = len(self.samples) * self.repeat
            print(f"PalmFrameDataset (PRELOAD): precomputing {total} augmented variants ...")
            for fp, label_idx, session_idx in self.samples:
                cloud, geom_raw = self._cache[fp]
                quality = compute_quality(cloud.shape[0],
                                          float(geom_raw[12]) if len(geom_raw) > 12 else None,
                                          float(geom_raw[7]) if len(geom_raw) > 7 else None)
                for _ in range(self.repeat):
                    pts = _sample_points(cloud, self.n_points, self.sampling)
                    if self.augment is not None:
                        pts = self.augment(pts)

                    geom = geom_raw.copy()
                    if self.normalizer is not None:
                        geom = self.normalizer.transform(geom)
                    if self.geom_augment is not None:
                        geom = self.geom_augment(geom)

                    self._preloaded.append({
                        "pts":         torch.from_numpy(np.ascontiguousarray(pts)),
                        "geom":        torch.from_numpy(np.ascontiguousarray(geom)),
                        "label_idx":   torch.tensor(label_idx, dtype=torch.long),
                        "session_idx": torch.tensor(session_idx, dtype=torch.long),
                        "quality":     torch.tensor(quality, dtype=torch.float32),
                    })
            ram_mb = sum(
                v["pts"].numel() * 4 + v["geom"].numel() * 4
                for v in self._preloaded
            ) / (1024 ** 2)
            print(f"  Preload selesai: {len(self._preloaded)} items, "
                  f"~{ram_mb:.1f} MB di RAM")
        else:
            print(f"PalmFrameDataset: {len(self.samples)} unique frames, "
                  f"{len(self.labels)} labels, repeat={self.repeat} → "
                  f"len(dataset)={len(self.samples) * self.repeat}")

    def __len__(self) -> int:
        if self.preload_augment:
            return len(self._preloaded)
        return len(self.samples) * self.repeat

    def __getitem__(self, idx: int) -> dict:
        if self.preload_augment:
            return self._preloaded[idx]

        fp, label_idx, session_idx = self.samples[idx % len(self.samples)]
        cloud, geom = self._cache[fp]

        # kualitas dihitung dari geom MENTAH (sebelum normalize) + jumlah titik cloud
        quality = compute_quality(cloud.shape[0],
                                  float(geom[12]) if len(geom) > 12 else None,
                                  float(geom[7]) if len(geom) > 7 else None)

        pts = _sample_points(cloud, self.n_points, self.sampling)

        if self.augment is not None:
            pts = self.augment(pts)

        if self.normalizer is not None:
            geom = self.normalizer.transform(geom)

        if self.geom_augment is not None:
            geom = self.geom_augment(geom)

        return {
            "pts":         torch.from_numpy(np.ascontiguousarray(pts)),
            "geom":        torch.from_numpy(np.ascontiguousarray(geom)),
            "label_idx":   torch.tensor(label_idx, dtype=torch.long),
            "session_idx": torch.tensor(session_idx, dtype=torch.long),
            "quality":     torch.tensor(quality, dtype=torch.float32),
        }
