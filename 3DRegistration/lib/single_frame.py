"""
lib/single_frame.py — Proses satu depth frame menjadi PLY yang telah diisolasi.

Tidak ada ICP — setiap frame diproses mandiri:
  1. ImageDepth  → raw point cloud (undistort + project + normal)
  2. Voxel downsample + statistical outlier removal
  3. isolate_foreground_point_cloud  → DBSCAN palm isolation + XY clip
  4. Re-estimasi normal setelah isolasi
  5. Simpan ke output PLY

Digunakan oleh process_single_frames.py sebagai building block.
"""

import argparse
from pathlib import Path

import open3d as o3d

from .image_depth import ImageDepth
from .process3d import isolate_foreground_point_cloud

# Parameter default — sama dengan PALM_RUN_ARGS di process_all_scans.py
DEFAULT_ARGS = argparse.Namespace(
    min_depth=0.10,
    max_depth=0.50,
    normal_radius=0.008,
    voxel_size=0.001,
    keep_nearest_cluster=1,
    foreground_depth_range=0.12,
    cluster_connectivity_eps=0.008,
    cluster_connectivity_min_points=20,
    # Outlier removal lebih agresif (std_ratio 1.0, radius filter ketat)
    outlier_nb_neighbors=30,
    outlier_std_ratio=1.0,
    radius_outlier_nb_points=20,
    radius_outlier_radius=0.008,
    xy_clip=3.0,
    # Fallback DBSCAN — dibutuhkan oleh isolate_foreground_point_cloud
    cluster_eps=0.015,
    cluster_min_points=80,
    cluster_z_tolerance=0.06,
)


def process_single_frame(
    calibration_file: str,
    depth_file: str,
    out_ply: str,
    args=DEFAULT_ARGS,
    min_points: int = 1000,
) -> "o3d.geometry.PointCloud | None":
    """
    Muat satu depth frame, isolasi palm foreground, simpan ke out_ply.

    Returns point cloud hasil isolasi, atau None jika gagal/terlalu sparse.
    """
    img = ImageDepth(
        calibration_file=calibration_file,
        image_file=None,
        depth_file=depth_file,
        width=640,
        height=480,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        normal_radius=args.normal_radius,
    )
    pcd = img.pcd
    if pcd is None or len(pcd.points) == 0:
        print(f"  [error] Tidak ada titik valid dari {depth_file}")
        return None

    # Voxel downsample
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)

    # Statistical outlier removal
    if args.outlier_nb_neighbors > 0 and args.outlier_std_ratio > 0:
        _, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )
        pcd = pcd.select_by_index(inlier_idx)

    # DBSCAN foreground isolation + XY clip (reuse dari process3d.py)
    pcd = isolate_foreground_point_cloud(args, pcd)
    if pcd is None or len(pcd.points) < min_points:
        n = len(pcd.points) if pcd is not None else 0
        print(f"  [skip] Hanya {n} titik setelah isolasi (min={min_points})")
        return None

    # Re-estimasi normal setelah isolasi (beberapa titik mungkin sudah hilang)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.normal_radius, max_nn=30
        )
    )
    pcd.orient_normals_towards_camera_location()

    Path(out_ply).parent.mkdir(parents=True, exist_ok=True)
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = pcd.points
    # Simpan normals juga (hasil estimasi di atas, radius normal_radius, oriented ke kamera).
    # R1 (raw PLY) di ablation v7.2.0 butuh 6 channel xyz+normals tanpa re-estimasi,
    # konsisten dengan R2 (cnn_input.npy) yang menurunkan normalnya dari PLY ini.
    if pcd.has_normals():
        pcd_out.normals = pcd.normals
    o3d.io.write_point_cloud(out_ply, pcd_out)
    return pcd
