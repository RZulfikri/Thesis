"""
make_fps.py — Turunkan cnn_input_fps.npy (R3) dari cnn_input.npy (R2).

R3 = FPS(R2): point cloud yang SUDAH PCA-aligned + unit-sphere (cnn_input.npy),
di-downsample ke n_points titik via Farthest Point Sampling. Diturunkan langsung
dari cnn_input.npy — BUKAN dari output.ply — sehingga R3 dijamin identik dengan
R2 kecuali pada strategi sampling (FPS vs random runtime). Ini membuat ablation
R2 vs R3 mengisolasi tepat satu variabel: FPS vs random sampling.

Memakai open3d farthest_point_down_sample (C++, ~0.4s/frame) — jauh lebih cepat
daripada FPS python murni di preprocess_for_cnn.py (~6s/frame).

Output per frame: cnn_input_fps.npy, shape (n_points, 6) float32 [xyz + normals].

Usage:
  python make_fps.py --data_dir ../3DCNN/dataset            # semua frame
  python make_fps.py --data_dir ../3DCNN/dataset --force    # timpa yang sudah ada
  python make_fps.py --data_dir ../3DCNN/dataset --n_points 8192
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def fps_from_cnn_input(cnn: np.ndarray, n_points: int) -> np.ndarray:
    """
    FPS downsample cnn_input (N, 6) → (n_points, 6).

    Jika N < n_points (jarang), pad dengan duplikasi acak agar shape tetap fixed.
    """
    pts = cnn[:, :3].astype(np.float64)
    nrm = cnn[:, 3:6].astype(np.float64)
    N = len(pts)

    if N < n_points:
        extra = np.random.choice(N, n_points - N, replace=True)
        idx = np.concatenate([np.arange(N), extra])
        out = np.concatenate([pts[idx], nrm[idx]], axis=1)
        return out.astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    ds = pcd.farthest_point_down_sample(n_points)
    out = np.concatenate([np.asarray(ds.points), np.asarray(ds.normals)], axis=1)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cnn_input_fps.npy (R3) dari cnn_input.npy (R2)")
    parser.add_argument("--data_dir", default="../3DCNN/dataset",
                        help="root dataset (default: ../3DCNN/dataset)")
    parser.add_argument("--n_points", type=int, default=8192,
                        help="jumlah titik FPS (default: 8192, sesuai input PointNet++)")
    parser.add_argument("--force", action="store_true",
                        help="regenerasi meski cnn_input_fps.npy sudah ada")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Error: '{data_dir}' tidak ditemukan")

    cnn_files = sorted(data_dir.glob("*/*/frame_*/cnn_input.npy"))
    if not cnn_files:
        sys.exit(f"Tidak ada cnn_input.npy di '{data_dir}'")

    print(f"Ditemukan {len(cnn_files)} cnn_input.npy")
    print(f"Target FPS  : {args.n_points} titik\n")

    t_start = time.time()
    n_done = n_skip = n_fail = 0
    for i, cnn_path in enumerate(cnn_files, 1):
        fps_path = cnn_path.parent / "cnn_input_fps.npy"
        if fps_path.exists() and not args.force:
            n_skip += 1
            continue
        try:
            cnn = np.load(cnn_path)
            fps = fps_from_cnn_input(cnn, args.n_points)
            np.save(fps_path, fps)
            n_done += 1
        except Exception as e:
            print(f"  [FAIL] {cnn_path}: {e}")
            n_fail += 1
        if i % 200 == 0:
            print(f"  {i}/{len(cnn_files)}  done={n_done} skip={n_skip} fail={n_fail}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SELESAI ({elapsed:.1f}s)")
    print(f"  Generated : {n_done}")
    print(f"  Skipped   : {n_skip} (sudah ada)")
    print(f"  Failed    : {n_fail}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
