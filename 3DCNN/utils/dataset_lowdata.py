"""
utils/dataset_lowdata.py — Low-data regime dataset loader untuk v5.0.0+.

OneFramePerSession + deterministic chronological split.

v7.0.0 changes:
  - Aktifkan subjek gede (11 subjek total, dari 10 di v6)
  - Tambah frame_sampling: "median" (default) | "random" (C1 mix-frame augmentation)
  - Tambah build_lowdata_splits_session_dirs untuk multi-frame fusion (D1+D2)
  - Tambah build_lowdata_splits_all_frames untuk ablation N,M frames

Subjek (11):
  aisah, alji, chrys, fadhil, feby, gede, nola, rahmat, reysa, taufik, yanuar

Split per subjek (15 sesi kronologis pertama):
  Train:   s1 – s8   (8 frames, oldest)
  Val:     s9 – s10  (2 frames)
  Test:    s11 – s12 (2 frames)
  Holdout: s13 – s15 (3 frames newest)

Total v7: 11 subjek × 15 sesi × 1 frame = 165 frames
  Train: 88 | Val: 22 | Test: 22 | Holdout: 33

Median frame picker:
  Untuk setiap frame dalam sesi, ekstrak 13-dim geom vector.
  Hitung median dan MAD per fitur di seluruh frame sesi.
  Pilih frame dengan total z-score (L1) terkecil sebagai representative frame.
  Alasan: hindari edge artifacts dari frame awal/akhir scan.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry_schema import _flatten_geometry

# v7.0.0: gede diaktifkan (dari DROPPED di v5/v6 menjadi aktif)
DROPPED_SUBJECTS: set[str] = set()  # tidak ada subjek yang di-drop
SESSIONS_PER_SUBJECT = 15
SPLIT_INDICES = {
    "train":   slice(0, 8),
    "val":     slice(8, 10),
    "test":    slice(10, 12),
    "holdout": slice(12, 15),
}

# Urutan frame sampling
FRAME_SAMPLING_MEDIAN = "median"
FRAME_SAMPLING_RANDOM = "random"


def _load_geom(frame_dir: Path) -> np.ndarray:
    """Load geometry.json dan flatten ke 13-dim vector."""
    with open(frame_dir / "geometry.json") as f:
        geo = json.load(f)
    return _flatten_geometry(geo)


def _pick_median_frame(frame_dirs: list[Path]) -> Path:
    """
    Pilih representative frame dari list frame_dirs dalam satu sesi.

    Logic:
      1. Flatten geom vector untuk semua frame.
      2. Hitung median dan MAD per fitur.
      3. Hitung z-score L1 distance setiap frame ke median.
      4. Pilih frame dengan distance terkecil.
    """
    if len(frame_dirs) == 1:
        return frame_dirs[0]

    geoms = np.stack([_load_geom(d) for d in frame_dirs])  # (N, 13)
    median = np.median(geoms, axis=0)                       # (13,)
    mad = np.median(np.abs(geoms - median), axis=0)         # (13,)
    mad[mad < 1e-6] = 1e-6  # avoid division by zero

    z_scores = np.abs(geoms - median) / mad                 # (N, 13)
    l1_dist = z_scores.sum(axis=1)                          # (N,)
    best_idx = int(np.argmin(l1_dist))
    return frame_dirs[best_idx]


def _pick_random_frame(frame_dirs: list[Path], rng: random.Random | None = None) -> Path:
    """
    Pilih frame acak dari list frame_dirs (C1: mix-frame augmentation).

    Args:
        frame_dirs : list frame dirs dalam satu sesi
        rng        : random.Random instance (opsional, untuk reproducibility)
    """
    if len(frame_dirs) == 1:
        return frame_dirs[0]
    if rng is not None:
        return rng.choice(frame_dirs)
    return random.choice(frame_dirs)


def _frame_passes_qc(frame_dir: Path) -> bool:
    """
    Frame lolos QC kalau:
    1. Punya cnn_input.npy dan geometry.json (data lengkap)
    2. Tidak punya invalid_frame.json (tidak di-flag QC)
    """
    return (
        (frame_dir / "cnn_input.npy").exists()
        and (frame_dir / "geometry.json").exists()
        and not (frame_dir / "invalid_frame.json").exists()
    )


def _session_is_valid(session_dir: Path) -> bool:
    """
    Session valid kalau punya minimal 1 frame yang lolos QC.
    """
    for frame in session_dir.iterdir():
        if frame.is_dir() and frame.name.startswith("frame_") and _frame_passes_qc(frame):
            return True
    return False


def _get_valid_frames(session_dir: Path) -> list[Path]:
    """
    Kembalikan sorted list frame_dirs yang lolos QC dalam satu sesi.
    Hanya frame tanpa invalid_frame.json yang masuk.
    """
    return sorted(
        p for p in session_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("frame_")
        and _frame_passes_qc(p)
    )


def scan_sessions(dataset_root: Path, n_sessions: int = SESSIONS_PER_SUBJECT) -> dict[str, list[Path]]:
    """
    Scan dataset dan kembalikan {label: [sorted valid session dirs]}.

    Ambil n_sessions sesi VALID kronologis pertama per subjek.
    Session dianggap valid kalau punya minimal 1 frame dengan cnn_input.npy + geometry.json.
    Skip subjek di DROPPED_SUBJECTS dan folder yang diawali '_'.

    v7.0.0: DROPPED_SUBJECTS kosong — gede diaktifkan.
    """
    dataset_root = Path(dataset_root)
    result: dict[str, list[Path]] = {}

    for label_dir in sorted(dataset_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label in DROPPED_SUBJECTS or label.startswith("_"):
            continue

        all_sessions = sorted(
            p for p in label_dir.iterdir()
            if p.is_dir()
            and not p.name.startswith("_")
        )
        if not all_sessions:
            continue

        valid_sessions = [s for s in all_sessions if _session_is_valid(s)]
        n_invalid = len(all_sessions) - len(valid_sessions)
        if n_invalid > 0:
            print(f"[Scan] {label}: skip {n_invalid} session tanpa frame valid")

        first_n = valid_sessions[:n_sessions]
        if len(first_n) < n_sessions:
            print(f"[WARN] {label} hanya punya {len(first_n)} valid session (butuh ≥{n_sessions})")
        result[label] = first_n

    return result


def pick_median_frames(session_dirs: list[Path]) -> list[Path]:
    """
    Untuk setiap session_dir, pilih 1 median frame.

    Returns:
        list[Path] — frame_dir per session yang valid
    """
    representative_frames: list[Path] = []
    for session_dir in session_dirs:
        frames = _get_valid_frames(session_dir)
        if not frames:
            print(f"[WARN] Tidak ada frame valid di {session_dir}, skip session ini.")
            continue
        representative_frames.append(_pick_median_frame(frames))
    return representative_frames


def pick_random_frames(session_dirs: list[Path], seed: int | None = None) -> list[Path]:
    """
    Untuk setiap session_dir, pilih 1 frame secara acak (C1: mix-frame augmentation training).

    Args:
        session_dirs : list session dirs
        seed         : random seed (opsional)

    Returns:
        list[Path] — 1 random frame_dir per session
    """
    rng = random.Random(seed) if seed is not None else None
    representative_frames: list[Path] = []
    for session_dir in session_dirs:
        frames = _get_valid_frames(session_dir)
        if not frames:
            print(f"[WARN] Tidak ada frame valid di {session_dir}, skip session ini.")
            continue
        representative_frames.append(_pick_random_frame(frames, rng))
    return representative_frames


def pick_n_frames(session_dirs: list[Path], n: int, seed: int | None = None) -> dict[Path, list[Path]]:
    """
    Untuk setiap session_dir, kembalikan hingga n frame (untuk multi-frame fusion D1+D2).

    Args:
        session_dirs : list session dirs
        n            : jumlah frame per sesi (0 = semua frame)
        seed         : random seed untuk sampling (None = ambil frame pertama secara deterministik)

    Returns:
        dict {session_dir: [frame_dirs]}
    """
    rng = random.Random(seed) if seed is not None else None
    result: dict[Path, list[Path]] = {}
    for session_dir in session_dirs:
        frames = _get_valid_frames(session_dir)
        if not frames:
            print(f"[WARN] Tidak ada frame valid di {session_dir}, skip.")
            continue
        if n <= 0 or n >= len(frames):
            result[session_dir] = frames
        else:
            if rng is not None:
                result[session_dir] = rng.sample(frames, n)
            else:
                # Deterministik: ambil frames yang terdistribusi merata
                indices = np.linspace(0, len(frames) - 1, n, dtype=int).tolist()
                result[session_dir] = [frames[i] for i in indices]
    return result


def build_lowdata_splits(
    dataset_root: Path,
    frame_sampling: str = FRAME_SAMPLING_MEDIAN,
    sampling_seed: int | None = None,
) -> dict[str, dict[str, list[Path]]]:
    """
    Build low-data splits: train/val/test/holdout.

    Args:
        dataset_root    : root dataset dir
        frame_sampling  : "median" (default, deterministik) | "random" (C1 mix-frame augmentation)
        sampling_seed   : seed untuk random sampling (hanya dipakai jika frame_sampling="random")

    Returns:
        {
            "train":   {label: [frame_dirs]},
            "val":     {label: [frame_dirs]},
            "test":    {label: [frame_dirs]},
            "holdout": {label: [frame_dirs]},
        }
    """
    dataset_root = Path(dataset_root)
    label_sessions = scan_sessions(dataset_root)

    splits: dict[str, dict[str, list[Path]]] = {
        "train": {}, "val": {}, "test": {}, "holdout": {}
    }

    for label, sessions in label_sessions.items():
        n_valid = len(sessions)
        train_end = min(8, n_valid)
        val_end   = min(10, n_valid)
        test_end  = min(12, n_valid)

        if frame_sampling == FRAME_SAMPLING_RANDOM:
            rep_frames = pick_random_frames(sessions, seed=sampling_seed)
        else:
            rep_frames = pick_median_frames(sessions)

        if len(rep_frames) < n_valid:
            print(f"[WARN] {label}: hanya {len(rep_frames)} frame valid dari {n_valid} sesi")

        # Adjust end indices ke panjang rep_frames
        nf = len(rep_frames)
        splits["train"][label]   = rep_frames[0:min(train_end, nf)]
        splits["val"][label]     = rep_frames[min(train_end, nf):min(val_end, nf)]
        splits["test"][label]    = rep_frames[min(val_end, nf):min(test_end, nf)]
        splits["holdout"][label] = rep_frames[min(test_end, nf):]

    for split_name, label_frames in splits.items():
        total = sum(len(v) for v in label_frames.values())
        print(f"[LowData] {split_name:8s}: {total} frames")

    return splits


def build_lowdata_splits_with_paths(
    dataset_root: Path,
    frame_sampling: str = FRAME_SAMPLING_MEDIAN,
    sampling_seed: int | None = None,
) -> dict[str, list[tuple[str, Path]]]:
    """
    Build low-data splits dengan format list[(subject, frame_dir)].

    Berguna untuk pair generation yang tidak memerlukan grouping per label.
    """
    dataset_root = Path(dataset_root)
    label_sessions = scan_sessions(dataset_root)

    splits: dict[str, list[tuple[str, Path]]] = {
        "train": [], "val": [], "test": [], "holdout": []
    }

    for label, sessions in label_sessions.items():
        n_valid = len(sessions)
        train_end = min(8, n_valid)
        val_end   = min(10, n_valid)
        test_end  = min(12, n_valid)

        if frame_sampling == FRAME_SAMPLING_RANDOM:
            rep_frames = pick_random_frames(sessions, seed=sampling_seed)
        else:
            rep_frames = pick_median_frames(sessions)

        nf = len(rep_frames)
        for frame in rep_frames[0:min(train_end, nf)]:
            splits["train"].append((label, frame))
        for frame in rep_frames[min(train_end, nf):min(val_end, nf)]:
            splits["val"].append((label, frame))
        for frame in rep_frames[min(val_end, nf):min(test_end, nf)]:
            splits["test"].append((label, frame))
        for frame in rep_frames[min(test_end, nf):]:
            splits["holdout"].append((label, frame))

    for split_name, items in splits.items():
        print(f"[LowData] {split_name:8s}: {len(items)} frames")

    return splits


def build_lowdata_splits_session_dirs(
    dataset_root: Path,
) -> dict[str, dict[str, list[Path]]]:
    """
    Build low-data splits dalam format session dirs (bukan frame dirs).

    Digunakan untuk multi-frame fusion (D1+D2): caller dapat mengambil
    semua frame dari tiap session_dir untuk enrollment dan probe.

    Returns:
        {split: {label: [session_dirs]}}  — session_dirs, BUKAN frame_dirs
    """
    dataset_root = Path(dataset_root)
    label_sessions = scan_sessions(dataset_root)

    splits: dict[str, dict[str, list[Path]]] = {
        "train": {}, "val": {}, "test": {}, "holdout": {}
    }

    for label, sessions in label_sessions.items():
        n_valid = len(sessions)
        train_end = min(8, n_valid)
        val_end   = min(10, n_valid)
        test_end  = min(12, n_valid)
        splits["train"][label]   = sessions[0:train_end]
        splits["val"][label]     = sessions[train_end:val_end]
        splits["test"][label]    = sessions[val_end:test_end]
        splits["holdout"][label] = sessions[test_end:]

    for split_name, label_sessions_split in splits.items():
        total = sum(len(v) for v in label_sessions_split.values())
        print(f"[LowData-Sessions] {split_name:8s}: {total} sesi")

    return splits


def build_lowdata_splits_all_frames(
    dataset_root: Path,
) -> dict[str, dict[str, list[Path]]]:
    """
    Build low-data splits dengan SEMUA frame dari tiap sesi (bukan 1 frame per sesi).

    Digunakan untuk ablation N,M frame pada multi-frame fusion (D3).

    Returns:
        {split: {label: [frame_dirs]}}
        — semua frame dari semua sesi di split tersebut
    """
    session_splits = build_lowdata_splits_session_dirs(dataset_root)

    frame_splits: dict[str, dict[str, list[Path]]] = {
        "train": {}, "val": {}, "test": {}, "holdout": {}
    }

    for split_name, label_sessions in session_splits.items():
        for label, session_dirs in label_sessions.items():
            all_frames: list[Path] = []
            for session_dir in session_dirs:
                all_frames.extend(_get_valid_frames(session_dir))
            frame_splits[split_name][label] = all_frames

    for split_name, lf in frame_splits.items():
        total = sum(len(v) for v in lf.values())
        print(f"[LowData-AllFrames] {split_name:8s}: {total} frames")

    return frame_splits


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build v7.0.0 low-data deterministic splits")
    p.add_argument("--dataset_root", default="../dataset", help="Dataset root dir")
    p.add_argument("--output", default=None, help="Simpan splits ke JSON (opsional)")
    p.add_argument("--frame-sampling", choices=["median", "random"], default="median",
                   help="Metode pemilihan frame per sesi (default: median)")
    p.add_argument("--sampling-seed", type=int, default=None,
                   help="Seed untuk random frame sampling")
    p.add_argument("--mode", choices=["standard", "sessions", "all-frames"], default="standard",
                   help="Mode split: standard (1 frame/sesi), sessions (session dirs), all-frames (semua frame)")
    args = p.parse_args()

    if args.mode == "sessions":
        splits = build_lowdata_splits_session_dirs(args.dataset_root)
    elif args.mode == "all-frames":
        splits = build_lowdata_splits_all_frames(args.dataset_root)
    else:
        splits = build_lowdata_splits(
            args.dataset_root,
            frame_sampling=args.frame_sampling,
            sampling_seed=args.sampling_seed,
        )

    if args.output:
        out = {
            k: {label: [str(f) for f in frames]
                for label, frames in v.items()}
            for k, v in splits.items()
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSplits disimpan di: {args.output}")
