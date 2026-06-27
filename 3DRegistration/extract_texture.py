"""
extract_texture.py — Project a registered palm point cloud to a canonical 2D texture image.

The point cloud is PCA-aligned (fingers → +Y), then projected top-down (along -Z).
Each pixel in the output grid stores the closest-point's surface properties.

Output: texture.npy  — float32 numpy array of shape (H, W, 5)
  ch0: depth       — Z coordinate, normalized to [0, 1]
  ch1: normal_x    — nx mapped from [-1, 1] → [0, 1]
  ch2: normal_y    — ny mapped from [-1, 1] → [0, 1]
  ch3: normal_z    — nz mapped from [-1, 1] → [0, 1]
  ch4: curvature   — |1 - |nz||, highlights creases and ridges

Empty grid cells (no nearby point) are filled with 0.

Usage:
  python extract_texture.py <ply_file> [--size 256] [--output texture.npy]

Example:
  python extract_texture.py data_set/scan_20260330_120723/output.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# Reuse canonical alignment from preprocess_for_cnn
from preprocess_for_cnn import pca_align


def extract_texture(ply_path: str, grid_size: int, output_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)

    if len(pcd.points) == 0:
        sys.exit(f"Error: {ply_path} contains no points")

    print(f"Loaded {len(pcd.points):,} points from {ply_path}")

    # Estimate normals if missing
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.orient_normals_towards_camera_location()

    # Canonical alignment: fingers → +Y, depth → +Z
    pcd_aligned = pca_align(pcd)
    pts = np.asarray(pcd_aligned.points, dtype=np.float32)     # (N, 3)
    nrm = np.asarray(pcd_aligned.normals, dtype=np.float32)    # (N, 3)

    # Projection: top-down view along -Z axis → use (X, Y) as 2D coordinates
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min if z_max > z_min else 1.0

    # Build 2D grid: rows = Y (fingers at top), cols = X (left to right)
    # Grid pixel (row i, col j) covers:
    #   x in [x_min + j*(x_max-x_min)/W,  x_min + (j+1)*(x_max-x_min)/W]
    #   y in [y_min + i*(y_max-y_min)/H,  y_min + (i+1)*(y_max-y_min)/H]
    W = H = grid_size

    # Map each point to its grid pixel
    eps = 1e-8
    col = ((x - x_min) / (x_max - x_min + eps) * W).astype(np.int32).clip(0, W - 1)
    row = ((y - y_min) / (y_max - y_min + eps) * H).astype(np.int32).clip(0, H - 1)

    # For each pixel, keep the point with the smallest Z (closest to camera / most frontal)
    # Use a depth buffer approach
    depth_buf = np.full((H, W), np.inf, dtype=np.float32)
    texture = np.zeros((H, W, 5), dtype=np.float32)

    for idx in range(len(pts)):
        r, c = int(row[idx]), int(col[idx])
        if z[idx] < depth_buf[r, c]:
            depth_buf[r, c] = z[idx]
            # ch0: depth normalised to [0, 1]
            texture[r, c, 0] = (z[idx] - z_min) / z_range
            # ch1-ch3: normals mapped from [-1,1] to [0,1]
            texture[r, c, 1] = (nrm[idx, 0] + 1.0) * 0.5
            texture[r, c, 2] = (nrm[idx, 1] + 1.0) * 0.5
            texture[r, c, 3] = (nrm[idx, 2] + 1.0) * 0.5
            # ch4: curvature = |1 - |nz||  (0=flat, 1=perpendicular)
            texture[r, c, 4] = abs(1.0 - abs(nrm[idx, 2]))

    # Fill empty pixels via nearest-neighbour interpolation
    occupied = depth_buf < np.inf
    n_occupied = occupied.sum()
    n_empty = H * W - n_occupied
    print(f"Grid {H}x{W}: {n_occupied} occupied pixels, {n_empty} empty ({100*n_empty/(H*W):.1f}%)")

    if n_empty > 0 and n_occupied > 0:
        rows_occ, cols_occ = np.where(occupied)
        rows_emp, cols_emp = np.where(~occupied)
        tree = cKDTree(np.stack([rows_occ, cols_occ], axis=1))
        _, nn_idx = tree.query(np.stack([rows_emp, cols_emp], axis=1), k=1)
        texture[rows_emp, cols_emp] = texture[rows_occ[nn_idx], cols_occ[nn_idx]]

    # Flip rows so fingers are at top of image (high Y → top row)
    texture = texture[::-1].copy()

    np.save(output_path, texture)
    print(f"Saved texture: {output_path}  shape={texture.shape}  dtype={texture.dtype}")
    print(f"  Depth range (normalised): [{texture[:,:,0].min():.3f}, {texture[:,:,0].max():.3f}]")
    print(f"  Curvature range: [{texture[:,:,4].min():.3f}, {texture[:,:,4].max():.3f}]")

    return texture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract canonical texture image from palm PLY")
    parser.add_argument("ply", help="input PLY file (registered point cloud)")
    parser.add_argument("--size", type=int, default=256, help="output grid size HxW (default: 256)")
    parser.add_argument("--output", default=None, help="output .npy path (default: <ply_dir>/texture.npy)")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    output_path = args.output or str(ply_path.parent / "texture.npy")

    extract_texture(str(ply_path), args.size, output_path)
