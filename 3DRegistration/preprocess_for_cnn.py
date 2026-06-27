"""
preprocess_for_cnn.py — Normalize a registered palm point cloud for CNN input.

Menghasilkan dua output dari satu file PLY:

  cnn_input.npy      ← full PCA-aligned + unit-sphere cloud, (N, 6) float32 [xyz + normals]
                        N variatif (~50k-150k titik), tidak di-downsample.
                        Ini adalah input utama untuk GeoAtt-PointNet++.

  cnn_input_fps.npy  ← FPS-downsampled ke n_points tetap, (n_points, 6) float32.
                        Backup novelty untuk ablation study.

Usage (kedua output):
  python preprocess_for_cnn.py <ply_file> [--n_points 1024]

Usage (hanya full cloud):
  python preprocess_for_cnn.py <ply_file> --no_fps

Usage (hanya FPS):
  python preprocess_for_cnn.py <ply_file> --no_full
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Canonical alignment helpers
# ---------------------------------------------------------------------------

def pca_align(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Align point cloud sehingga:
      - Arah jari (wrist → fingertip, range terpanjang) → +Y
      - Arah kedalaman kamera (varians terkecil, normal ke permukaan palm) → +Z
      - Centroid dipindah ke origin

    Menggunakan RANGE (max-min) bukan variance untuk memilih sumbu Y,
    karena ketika jari terbuka lebar variance horizontal bisa ≥ variance
    vertikal — padahal range finger direction selalu lebih panjang dari
    lebar palm.
    """
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA via SVD — diurutkan varians menurun
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    # Z = varians terkecil = arah depth kamera (normal ke permukaan palm)
    # Dari dua in-plane axis, Y = yang memiliki RANGE terbesar (finger direction)
    range0 = float(np.ptp(centered @ Vt[0]))  # ptp = max - min
    range1 = float(np.ptp(centered @ Vt[1]))

    if range0 >= range1:
        y_axis, x_axis = Vt[0], Vt[1]   # axis[0] lebih panjang → Y
    else:
        y_axis, x_axis = Vt[1], Vt[0]   # axis[1] lebih panjang → Y

    z_axis = Vt[2]

    # Pastikan right-handed coordinate system
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3)
    aligned_pts = centered @ R.T

    # Pastikan jari mengarah ke +Y: lebih banyak titik di atas median Y
    flip = np.sum(aligned_pts[:, 1] > np.median(aligned_pts[:, 1])) < len(aligned_pts) // 2
    if flip:
        aligned_pts[:, 0] *= -1
        aligned_pts[:, 1] *= -1

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(aligned_pts)

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        aligned_normals = normals @ R.T
        if flip:
            aligned_normals[:, 0] *= -1
            aligned_normals[:, 1] *= -1
        result.normals = o3d.utility.Vector3dVector(aligned_normals)

    return result


def normalize_to_unit_sphere(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Scale point cloud agar semua titik masuk dalam unit sphere (radius = 1)."""
    pts = np.asarray(pcd.points)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-8:
        return pcd
    result = o3d.geometry.PointCloud(pcd)
    result.points = o3d.utility.Vector3dVector(pts / scale)
    return result


def farthest_point_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Greedy farthest point sampling.
    Mengembalikan indeks n titik terpilih dari pts (N, 3).
    """
    N = len(pts)
    if N <= n:
        idx = np.arange(N)
        idx = np.concatenate([idx, np.random.choice(N, n - N)])
        return idx

    selected = np.zeros(n, dtype=np.int64)
    dist = np.full(N, np.inf)
    current = 0
    for i in range(n):
        selected[i] = current
        current_pt = pts[current]
        d = np.sum((pts - current_pt) ** 2, axis=1)
        dist = np.minimum(dist, d)
        current = int(np.argmax(dist))

    return selected


# ---------------------------------------------------------------------------
# Shared: load + align + normalize
# ---------------------------------------------------------------------------

def _load_and_align(ply_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PLY, estimasi normal jika belum ada, PCA align, unit sphere normalize.

    Returns:
        pts : (N, 3) float32 — XYZ dalam koordinat kanonis, unit sphere
        nrm : (N, 3) float32 — normals dalam koordinat kanonis
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        sys.exit(f"Error: {ply_path} tidak ada titik")

    print(f"Loaded {len(pcd.points):,} titik dari {ply_path}")

    if not pcd.has_normals():
        print("Estimasi normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.orient_normals_towards_camera_location()

    pcd = pca_align(pcd)
    pcd = normalize_to_unit_sphere(pcd)

    pts = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else np.zeros_like(pts)
    return pts, nrm


# ---------------------------------------------------------------------------
# preprocess_full — simpan semua titik, tanpa FPS
# ---------------------------------------------------------------------------

def preprocess_full(ply_path: str, output_path: str) -> np.ndarray:
    """
    Simpan full PCA-aligned + unit-sphere cloud sebagai cnn_input.npy.
    Shape: (N, 6) float32, N variatif (tidak di-downsample).
    """
    pts, nrm = _load_and_align(ply_path)
    result = np.concatenate([pts, nrm], axis=1).astype(np.float32)  # (N, 6)

    np.save(output_path, result)
    print(f"Saved full cloud: {output_path}  shape={result.shape}  dtype={result.dtype}")
    print(f"  xyz range: [{pts.min():.3f}, {pts.max():.3f}]")
    return result


# ---------------------------------------------------------------------------
# preprocess_fps — FPS downsample ke n titik tetap (backup novelty)
# ---------------------------------------------------------------------------

def preprocess_fps(ply_path: str, n_points: int, output_path: str) -> np.ndarray:
    """
    FPS downsample ke n_points tetap, simpan sebagai cnn_input_fps.npy.
    Shape: (n_points, 6) float32 — backup novelty untuk ablation study.
    """
    pts, nrm = _load_and_align(ply_path)

    print(f"FPS sampling {n_points} titik dari {len(pts):,}...")
    idx = farthest_point_sample(pts, n_points)
    pts_fps = pts[idx]
    nrm_fps = nrm[idx]

    result = np.concatenate([pts_fps, nrm_fps], axis=1).astype(np.float32)  # (n_points, 6)

    np.save(output_path, result)
    print(f"Saved FPS cloud: {output_path}  shape={result.shape}  dtype={result.dtype}")
    return result


# Alias untuk backward compatibility
def preprocess(ply_path: str, n_points: int, output_path: str) -> np.ndarray:
    """Alias untuk preprocess_fps (backward compatibility)."""
    return preprocess_fps(ply_path, n_points, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess palm PLY untuk CNN input")
    parser.add_argument("ply", help="input PLY file (registered point cloud)")
    parser.add_argument("--n_points", type=int, default=8192,
                        help="jumlah titik untuk FPS output (default: 8192, sesuai input PointNet++ v7.2.0)")
    parser.add_argument("--output_dir", default=None,
                        help="direktori output (default: direktori PLY)")
    parser.add_argument("--no_full", action="store_true",
                        help="skip output full cloud (cnn_input.npy)")
    parser.add_argument("--no_fps",  action="store_true",
                        help="skip output FPS (cnn_input_fps.npy)")
    args = parser.parse_args()

    ply_path  = Path(args.ply)
    out_dir   = Path(args.output_dir) if args.output_dir else ply_path.parent

    if not args.no_full:
        preprocess_full(str(ply_path), str(out_dir / "cnn_input.npy"))

    if not args.no_fps:
        preprocess_fps(str(ply_path), args.n_points, str(out_dir / "cnn_input_fps.npy"))
