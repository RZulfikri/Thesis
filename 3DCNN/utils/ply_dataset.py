"""
utils/ply_dataset.py — PLY Direct On-the-Fly Dataset untuk v0.3.0.

Load PLY asli dari disk setiap epoch, lakukan augmentasi di original camera space,
lalu recompute PCA-align + normalize ke unit sphere.

Interface kompatibel dengan PalmFrameDataset untuk drop-in replacement
di collab/train.ipynb:
    DATASET_MODE = 'ply'  # ganti dari 'npy'

Kunci: PCA-align HARUS identik dengan preprocess_for_cnn.py.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import GEOMETRY_DIM, _flatten_geometry, _sample_points

# Optional Open3D — digunakan untuk load PLY dan estimasi normals
try:
    import open3d as o3d
    _HAS_OPEN3D = True
except Exception:
    _HAS_OPEN3D = False


# ---------------------------------------------------------------------------
# PLY loader
# ---------------------------------------------------------------------------

def _load_ply_numpy(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Parse PLY binary little_endian tanpa Open3D.
    Hanya mendukung format vertex dengan property float/double x,y,z [,nx,ny,nz].
    Returns: pts (N,3) float32, normals (N,3) float32 or None
    """
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break

        n_vertices = 0
        has_normals = False
        fmt = "binary_little_endian"
        dtype_parts = []
        properties = []

        for line in header.decode("ascii", errors="ignore").split("\n"):
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element" and parts[1] == "vertex":
                n_vertices = int(parts[2])
            elif parts[0] == "property":
                typ = parts[-2] if parts[-2] in ("float", "double") else parts[-1]
                name = parts[-1]
                properties.append(name)
                if name in ("x", "y", "z"):
                    dtype_parts.append((name, np.float64 if typ == "double" else np.float32))
                elif name in ("nx", "ny", "nz"):
                    has_normals = True
                    dtype_parts.append((name, np.float64 if typ == "double" else np.float32))

        if fmt == "binary_little_endian":
            raw = np.fromfile(f, dtype=np.dtype(dtype_parts), count=n_vertices)
        else:
            # ASCII — parse line by line (slow, fallback)
            lines = f.read().decode("ascii").strip().split("\n")
            raw = np.zeros(n_vertices, dtype=np.dtype(dtype_parts))
            for i, line in enumerate(lines[:n_vertices]):
                vals = list(map(float, line.strip().split()))
                for j, (name, _) in enumerate(dtype_parts):
                    raw[i][name] = vals[j]

        pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
        if has_normals:
            normals = np.stack([raw["nx"], raw["ny"], raw["nz"]], axis=1).astype(np.float32)
        else:
            normals = None
        return pts, normals


def load_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load PLY dengan Open3D jika tersedia, fallback ke parser numpy."""
    if _HAS_OPEN3D:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            raise ValueError(f"PLY kosong atau tidak valid: {ply_path}")
        pts = np.asarray(pcd.points, dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
        return pts, normals
    else:
        return _load_ply_numpy(ply_path)


# ---------------------------------------------------------------------------
# Replicate preprocess_for_cnn.py logic in pure NumPy
# ---------------------------------------------------------------------------

def pca_align_numpy(pts: np.ndarray, normals: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Replicate preprocess_for_cnn.py::pca_align() in pure NumPy.

    Align point cloud sehingga:
      - Arah jari (wrist → fingertip, range terpanjang) → +Y
      - Arah kedalaman kamera (varians terkecil, normal ke permukaan palm) → +Z
      - Centroid dipindah ke origin

    Menggunakan RANGE (max-min) bukan variance untuk memilih sumbu Y.
    """
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA via SVD — diurutkan varians menurun
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    # Z = varians terkecil = arah depth kamera
    # Dari dua in-plane axis, Y = yang memiliki RANGE terbesar (finger direction)
    range0 = float(np.ptp(centered @ Vt[0]))
    range1 = float(np.ptp(centered @ Vt[1]))

    if range0 >= range1:
        y_axis, x_axis = Vt[0], Vt[1]
    else:
        y_axis, x_axis = Vt[1], Vt[0]

    z_axis = Vt[2]

    # Pastikan right-handed coordinate system
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= (np.linalg.norm(z_axis) + 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-12)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3)
    aligned_pts = centered @ R.T

    # Pastikan jari mengarah ke +Y: lebih banyak titik di atas median Y
    flip = np.sum(aligned_pts[:, 1] > np.median(aligned_pts[:, 1])) < len(aligned_pts) // 2
    if flip:
        aligned_pts[:, 0] *= -1
        aligned_pts[:, 1] *= -1

    aligned_normals = None
    if normals is not None:
        aligned_normals = normals @ R.T
        if flip:
            aligned_normals[:, 0] *= -1
            aligned_normals[:, 1] *= -1

    return aligned_pts, aligned_normals


def normalize_to_unit_sphere(pts: np.ndarray) -> np.ndarray:
    """Scale point cloud agar semua titik masuk dalam unit sphere (radius = 1)."""
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-8:
        return pts
    return pts / scale


def preprocess_ply(
    pts: np.ndarray,
    normals: np.ndarray | None = None,
    estimate_normals: bool = True,
) -> np.ndarray:
    """
    Pipeline: estimasi normals (jika perlu) → PCA-align → normalize → concat.

    Returns:
        cloud : (N, 6) float32 — XYZ + normals, PCA-aligned + unit sphere
    """
    if normals is None and estimate_normals and _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location()
        normals = np.asarray(pcd.normals, dtype=np.float32)

    pts_aligned, normals_aligned = pca_align_numpy(pts, normals)
    pts_aligned = normalize_to_unit_sphere(pts_aligned)

    if normals_aligned is not None:
        cloud = np.concatenate([pts_aligned, normals_aligned], axis=1).astype(np.float32)
    else:
        cloud = np.concatenate([pts_aligned, np.zeros_like(pts_aligned)], axis=1).astype(np.float32)
    return cloud


# ---------------------------------------------------------------------------
# Geometry loader (reuse dari dataset.py)
# ---------------------------------------------------------------------------

def _load_geo_dict(session_dir: Path) -> dict:
    with open(session_dir / "geometry.json") as f:
        return json.load(f)


def load_geometry(session_dir: Path) -> np.ndarray:
    """Load geometry.json dan flatten ke (GEOMETRY_DIM,) float32."""
    return _flatten_geometry(_load_geo_dict(session_dir))


# ---------------------------------------------------------------------------
# PLY Direct Dataset
# ---------------------------------------------------------------------------

class PLYDirectDataset(Dataset):
    """
    Dataset yang load PLY asli, augmentasi di original space, lalu PCA-align.

    Interface kompatibel dengan PalmFrameDataset:
        pts       : (n_points, 6)   float32 tensor
        geom      : (GEOMETRY_DIM,) float32 tensor
        label_idx : long tensor

    Args:
        label_sessions    : {label: [frame_dirs]}
        n_points          : jumlah titik yang di-sample
        sampling          : 'random' atau 'fps'
        augment           : OriginalSpaceAugmentor atau None
        geom_augment      : GeometryAugmentor atau None
        normalizer        : GeometryNormalizer atau None
        repeat            : tiap frame muncul `repeat` kali per epoch
        estimate_normals  : jika True dan PLY tanpa normals, estimasi via Open3D
        use_normals       : jika False, channel normals di-zero (untuk ablation)
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
        estimate_normals: bool = True,
        use_normals: bool = True,
    ):
        self.n_points = n_points
        self.sampling = sampling
        self.augment = augment
        self.geom_augment = geom_augment
        self.normalizer = normalizer
        self.repeat = max(1, int(repeat))
        self.estimate_normals = estimate_normals
        self.use_normals = use_normals

        self.labels = sorted(label_sessions.keys())
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}

        self.samples: list[tuple[Path, int]] = []
        for lbl, frames in label_sessions.items():
            idx = self.label_to_idx[lbl]
            for fp in frames:
                self.samples.append((fp, idx))

        print(f"PLYDirectDataset: {len(self.samples)} unique frames, "
              f"{len(self.labels)} labels, repeat={self.repeat} → "
              f"len(dataset)={len(self.samples) * self.repeat}")

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def _load_ply_cached(self, frame_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load PLY + geometry on-the-fly (tanpa cache agar augmentasi fresh tiap epoch)."""
        ply_path = frame_dir / "output.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY tidak ditemukan: {ply_path}")

        pts, normals = load_ply(ply_path)
        geom = load_geometry(frame_dir)
        return pts, normals, geom

    def __getitem__(self, idx: int) -> dict:
        fp, label_idx = self.samples[idx % len(self.samples)]

        # Load PLY fresh tiap kali (penting untuk augmentasi variatif per epoch)
        pts, normals, geom = self._load_ply_cached(fp)

        # 1. Augmentasi di original space (sebelum PCA-align)
        if self.augment is not None:
            pts, normals = self.augment(pts, normals)

        # 2. PCA-align + normalize (replicate preprocess_for_cnn.py)
        cloud = preprocess_ply(pts, normals, estimate_normals=self.estimate_normals)

        # 3. Optional: ablation tanpa normals
        if not self.use_normals:
            cloud[:, 3:6] = 0.0

        # 4. Sample ke n_points
        pts_sampled = _sample_points(cloud, self.n_points, self.sampling)

        # 5. Normalisasi geometry
        if self.normalizer is not None:
            geom = self.normalizer.transform(geom)

        if self.geom_augment is not None:
            geom = self.geom_augment(geom)

        return {
            "pts":       torch.from_numpy(np.ascontiguousarray(pts_sampled)),
            "geom":      torch.from_numpy(np.ascontiguousarray(geom)),
            "label_idx": torch.tensor(label_idx, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Sanity check: verifikasi PLYDirect output == cnn_input.npy (tanpa augmentasi)
# ---------------------------------------------------------------------------

def verify_ply_identity(
    frame_dir: Path,
    n_points: int = 8192,
    sampling: Literal["random", "fps"] = "random",
    tolerance: float = 1e-4,
) -> dict:
    """
    Verifikasi bahwa PLYDirectDataset (tanpa augmentasi) menghasilkan output
    yang identik dengan cnn_input.npy yang sudah ada.

    Returns:
        dict dengan status pass/fail dan metrik perbandingan.
    """
    from .dataset import load_session

    # Load via PLY Direct (tanpa augmentasi)
    ds = PLYDirectDataset(
        {"__test__": [frame_dir]},
        n_points=n_points,
        sampling=sampling,
        augment=None,
        repeat=1,
    )
    item_ply = ds[0]
    pts_ply = item_ply["pts"].numpy()

    # Load via existing pipeline (cnn_input.npy)
    cloud_npy, _ = load_session(frame_dir)
    pts_npy = _sample_points(cloud_npy, n_points, sampling)

    diff = np.abs(pts_ply - pts_npy).max()
    mean_diff = np.abs(pts_ply - pts_npy).mean()

    passed = diff < tolerance
    return {
        "frame": str(frame_dir),
        "max_diff": float(diff),
        "mean_diff": float(mean_diff),
        "tolerance": tolerance,
        "passed": bool(passed),
        "status": "PASS" if passed else "FAIL",
    }
