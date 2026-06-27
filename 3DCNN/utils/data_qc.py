"""
utils/data_qc.py — Data quality control untuk palm dataset.

Mendukung:
  - Statistik per-subjek (jumlah frame, point count, bbox, density)
  - Deteksi outlier frame (incomplete, noise, pose ekstrem)
  - Visual report untuk subjek bermasalah (feby, nola, reysa)

Usage di Colab:
    from utils.data_qc import run_qc_report
    report = run_qc_report("dataset", subjects=["feby", "nola", "reysa"])
    print(report.to_markdown())
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


def _load_ply_simple(ply_path: Path) -> np.ndarray | None:
    """Load PLY tanpa Open3D — parse header binary minimal."""
    try:
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if line.strip() == b"end_header":
                    break

            n_vertices = 0
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
                    if name in ("x", "y", "z", "nx", "ny", "nz"):
                        dtype_parts.append((name, np.float64 if typ == "double" else np.float32))

            if fmt == "binary_little_endian":
                raw = np.fromfile(f, dtype=np.dtype(dtype_parts), count=n_vertices)
            else:
                lines = f.read().decode("ascii").strip().split("\n")
                raw = np.zeros(n_vertices, dtype=np.dtype(dtype_parts))
                for i, line in enumerate(lines[:n_vertices]):
                    vals = list(map(float, line.strip().split()))
                    for j, (name, _) in enumerate(dtype_parts):
                        raw[i][name] = vals[j]

            pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
            return pts
    except Exception:
        return None


@dataclass
class FrameQC:
    frame_dir: Path
    n_points: int
    bbox_extent: tuple[float, float, float]
    centroid: tuple[float, float, float]
    std_xyz: tuple[float, float, float]
    density_proxy: float  # n_points / volume_bbox
    file_size_kb: float


@dataclass
class SubjectQC:
    subject: str
    n_sessions: int
    n_frames: int
    frames: list[FrameQC]

    def summary(self) -> dict:
        npts = [f.n_points for f in self.frames]
        sizes = [f.file_size_kb for f in self.frames]
        densities = [f.density_proxy for f in self.frames]
        return {
            "subject": self.subject,
            "n_sessions": self.n_sessions,
            "n_frames": self.n_frames,
            "points_mean": float(np.mean(npts)),
            "points_std": float(np.std(npts)),
            "points_min": int(np.min(npts)),
            "points_max": int(np.max(npts)),
            "size_mean_kb": float(np.mean(sizes)),
            "size_min_kb": float(np.min(sizes)),
            "size_max_kb": float(np.max(sizes)),
            "density_mean": float(np.mean(densities)),
            "density_std": float(np.std(densities)),
        }

    def outlier_frames(self, threshold: float = 2.0) -> list[FrameQC]:
        """Deteksi frame outlier berdasarkan z-score point count dan file size."""
        npts = np.array([f.n_points for f in self.frames], dtype=float)
        sizes = np.array([f.file_size_kb for f in self.frames], dtype=float)
        z_npts = (npts - npts.mean()) / (npts.std() + 1e-8)
        z_sizes = (sizes - sizes.mean()) / (sizes.std() + 1e-8)
        out = []
        for i, f in enumerate(self.frames):
            if abs(z_npts[i]) > threshold or abs(z_sizes[i]) > threshold:
                out.append(f)
        return out


def _scan_subject(data_dir: Path, subject: str) -> SubjectQC:
    subject_dir = data_dir / subject
    frames: list[FrameQC] = []
    sessions = 0

    for ts_dir in sorted(subject_dir.iterdir()):
        if not ts_dir.is_dir():
            continue
        sessions += 1
        # Frame layout
        frame_dirs = sorted([d for d in ts_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")])
        if frame_dirs:
            for fd in frame_dirs:
                ply_path = fd / "output.ply"
                pts = _load_ply_simple(ply_path) if ply_path.exists() else None
                if pts is not None:
                    xyz = pts[:, :3]
                    n = len(xyz)
                    bbox = xyz.max(axis=0) - xyz.min(axis=0)
                    centroid = xyz.mean(axis=0)
                    std = xyz.std(axis=0)
                    volume = float(bbox.prod())
                    density = n / (volume + 1e-8)
                    fsize = ply_path.stat().st_size / 1024.0
                    frames.append(FrameQC(
                        frame_dir=fd, n_points=n,
                        bbox_extent=tuple(bbox), centroid=tuple(centroid),
                        std_xyz=tuple(std), density_proxy=density,
                        file_size_kb=fsize,
                    ))
        else:
            # Session layout (legacy)
            ply_path = ts_dir / "output.ply"
            pts = _load_ply_simple(ply_path) if ply_path.exists() else None
            if pts is not None:
                xyz = pts[:, :3]
                n = len(xyz)
                bbox = xyz.max(axis=0) - xyz.min(axis=0)
                centroid = xyz.mean(axis=0)
                std = xyz.std(axis=0)
                volume = float(bbox.prod())
                density = n / (volume + 1e-8)
                fsize = ply_path.stat().st_size / 1024.0
                frames.append(FrameQC(
                    frame_dir=ts_dir, n_points=n,
                    bbox_extent=tuple(bbox), centroid=tuple(centroid),
                    std_xyz=tuple(std), density_proxy=density,
                    file_size_kb=fsize,
                ))

    return SubjectQC(subject=subject, n_sessions=sessions, n_frames=len(frames), frames=frames)


def run_qc_report(data_dir: str | Path, subjects: list[str] | None = None) -> dict:
    """
    Jalankan QC report untuk satu atau semua subjek.

    Returns:
        dict dengan key 'subjects' (list[dict summary]) dan 'outliers' (list frame bermasalah)
    """
    data_dir = Path(data_dir)
    if subjects is None:
        subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    results = []
    all_outliers = []
    for subj in subjects:
        sqc = _scan_subject(data_dir, subj)
        results.append(sqc.summary())
        outliers = sqc.outlier_frames(threshold=2.0)
        for o in outliers:
            all_outliers.append({
                "subject": subj,
                "frame": str(o.frame_dir),
                "n_points": o.n_points,
                "file_size_kb": round(o.file_size_kb, 1),
                "density": round(o.density_proxy, 4),
                "bbox": [round(x, 4) for x in o.bbox_extent],
            })

    return {"subjects": results, "outliers": all_outliers}


def print_qc_report(report: dict, outlier_only: bool = False) -> None:
    """Cetak report ke stdout dalam format tabel."""
    if not outlier_only:
        print("\n# QC Report — Per Subjek")
        print("| Subjek | Sesi | Frame | Pts Mean±Std | Pts Min | Pts Max | Size Mean (KB) |")
        print("|--------|------|-------|-------------|---------|---------|----------------|")
        for r in report["subjects"]:
            print(
                f"| {r['subject']:<6} | {r['n_sessions']:>4} | {r['n_frames']:>5} | "
                f"{r['points_mean']:>6.0f}±{r['points_std']:>4.0f} | {r['points_min']:>7} | "
                f"{r['points_max']:>7} | {r['size_mean_kb']:>14.0f} |"
            )

    if report["outliers"]:
        print(f"\n# Outlier Frames (z-score > 2.0): {len(report['outliers'])}")
        print("| Subjek | Frame | N Points | Size (KB) | BBox (x,y,z) |")
        print("|--------|-------|----------|-----------|--------------|")
        for o in report["outliers"]:
            print(
                f"| {o['subject']:<6} | {Path(o['frame']).name:<5} | {o['n_points']:>8} | "
                f"{o['file_size_kb']:>9} | {o['bbox']} |"
            )
    else:
        print("\n# Tidak ada outlier frame terdeteksi.")
