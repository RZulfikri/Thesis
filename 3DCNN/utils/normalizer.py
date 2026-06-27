"""
utils/normalizer.py — Z-score normalization untuk fitur geometri.

Fit HANYA dari training set — hindari data leakage dari val/test.

Workflow:
    normalizer = GeometryNormalizer()
    normalizer.fit([load_geometry(d) for d in train_dirs])  # fit dari train saja
    geom_norm = normalizer.transform(geom_raw)              # apply ke semua split

Dimensi vektor geometri: GEOMETRY_DIM = 13 (lihat utils/geometry_schema.py).
"""

import json
from pathlib import Path

import numpy as np


class GeometryNormalizer:
    """
    Z-score normalizer untuk vektor fitur geometri (GEOMETRY_DIM = 13).

    Setiap fitur dinormalisasi ke mean=0, std=1 berdasarkan distribusi training set.
    Menjaga perbedaan relatif antar individu — berbeda dengan per-sample normalization
    yang menghilangkan informasi ukuran absolut tangan sebagai fitur biometrik.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, geom_list: list[np.ndarray]) -> "GeometryNormalizer":
        """
        Hitung mean dan std dari list vektor geometri training set.

        Args:
            geom_list: list of np.ndarray, each shape (GEOMETRY_DIM,)
        """
        data = np.stack(geom_list, axis=0)  # (N, GEOMETRY_DIM)
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std  = data.std(axis=0).astype(np.float32)
        # Hindari pembagian nol untuk fitur yang konstan
        self.std = np.where(self.std < 1e-8, 1.0, self.std).astype(np.float32)
        return self

    def transform(self, geom: np.ndarray) -> np.ndarray:
        """
        Terapkan z-score normalization menggunakan statistik training set.

        Args:
            geom: (GEOMETRY_DIM,) atau (B, GEOMETRY_DIM) float32

        Returns:
            array ternormalisasi, bentuk sama
        """
        if self.mean is None:
            raise RuntimeError("GeometryNormalizer.fit() harus dipanggil sebelum transform()")
        return ((geom - self.mean) / self.std).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save mean/std to JSON for Colab persistence."""
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted yet")
        data = {
            "mean": self.mean.tolist(),
            "std":  self.std.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GeometryNormalizer":
        """Load normalizer from JSON."""
        with open(path) as f:
            data = json.load(f)
        n = cls()
        n.mean = np.array(data["mean"], dtype=np.float32)
        n.std  = np.array(data["std"],  dtype=np.float32)
        return n
