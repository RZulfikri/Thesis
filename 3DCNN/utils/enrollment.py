"""
utils/enrollment.py — Gallery enrollment strategies untuk identifikasi 1:N.

Mendukung:
  - Simple average (baseline)
  - Median embedding
  - Quality-weighted average
  - Multi-prototype (k-means clustering)

Dengan use_geom=True, quality score dapat dihitung dari statistik geometri.
"""

import numpy as np
from pathlib import Path
from typing import Callable


def _default_quality_score(frame_dir: Path) -> float:
    """
    Hitung quality score sederhana dari point cloud.
    Higher = lebih dense, lebih lengkap.
    """
    try:
        pts = np.load(frame_dir / "points.npy")
        # Simple heuristic: jumlah points / density proxy
        n_pts = len(pts)
        # Normalize ke [0, 1] dengan asumsi max ~20k points
        score = min(n_pts / 20000.0, 1.0)
        return float(score)
    except Exception:
        return 1.0


def enroll_average(embeddings: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Gallery enrollment dengan weighted atau unweighted average.

    Args:
        embeddings : (N, D) — N embeddings untuk 1 subjek
        weights    : (N,) opsional — quality weights

    Returns:
        gallery_emb : (D,) — L2-normalized average embedding
    """
    if weights is not None:
        weights = np.asarray(weights)
        weights = weights / weights.sum()
        avg = np.average(embeddings, axis=0, weights=weights)
    else:
        avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


def enroll_median(embeddings: np.ndarray) -> np.ndarray:
    """
    Gallery enrollment dengan median embedding (robust ke outlier).

    Args:
        embeddings : (N, D)

    Returns:
        gallery_emb : (D,) — L2-normalized median embedding
    """
    median = np.median(embeddings, axis=0)
    norm = np.linalg.norm(median)
    return median / norm if norm > 0 else median


def enroll_kmeans(embeddings: np.ndarray, k: int = 3, max_iter: int = 100) -> np.ndarray:
    """
    Gallery enrollment dengan K-means clustering.
    Return semua prototype (bukan rata-ratanya).

    Args:
        embeddings : (N, D)
        k          : jumlah prototype
        max_iter   : max iterasi k-means

    Returns:
        prototypes : (k, D) — L2-normalized cluster centers
    """
    N, D = embeddings.shape
    k = min(k, N)

    if k == 1:
        return enroll_average(embeddings).reshape(1, -1)

    # K-means++ initialization
    rng = np.random.default_rng(42)
    centers = [embeddings[rng.integers(N)]]
    for _ in range(1, k):
        dists = np.min([np.linalg.norm(embeddings - c, axis=1) for c in centers], axis=0)
        probs = dists / dists.sum()
        idx = rng.choice(N, p=probs)
        centers.append(embeddings[idx])
    centers = np.array(centers)

    # Lloyd iteration
    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update
        new_centers = np.array([
            embeddings[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centers[i]
            for i in range(k)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # Normalize
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = np.where(norms > 0, centers / norms, centers)
    return centers


class GalleryEnroller:
    """
    Gallery enrollment dengan strategy yang bisa dipilih.

    Strategies:
        - "average"       : simple mean (baseline)
        - "weighted"      : quality-weighted mean
        - "median"        : median embedding
        - "multi"         : k-means multi-prototype (default k=3)
    """

    def __init__(self, strategy: str = "multi", k: int = 3,
                 quality_fn: Callable[[Path], float] = _default_quality_score):
        self.strategy = strategy
        self.k = k
        self.quality_fn = quality_fn

    def enroll(self, embeddings: np.ndarray,
               frame_dirs: list[Path] | None = None) -> np.ndarray:
        """
        Args:
            embeddings : (N, D) — embeddings untuk 1 subjek
            frame_dirs : list of Path opsional — untuk quality scoring

        Returns:
            gallery : (D,) atau (k, D) — gallery embedding(s)
        """
        if self.strategy == "average":
            return enroll_average(embeddings)

        elif self.strategy == "weighted":
            if frame_dirs is None:
                weights = None
            else:
                weights = np.array([self.quality_fn(fd) for fd in frame_dirs])
            return enroll_average(embeddings, weights)

        elif self.strategy == "median":
            return enroll_median(embeddings)

        elif self.strategy == "multi":
            return enroll_kmeans(embeddings, k=self.k)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def query_similarity(query_emb: np.ndarray, gallery_emb: np.ndarray) -> float:
    """
    Hitung cosine similarity antara query dan gallery.

    Args:
        query_emb   : (D,)
        gallery_emb : (D,) atau (k, D) untuk multi-prototype

    Returns:
        similarity : float — max similarity untuk multi-prototype
    """
    if gallery_emb.ndim == 1:
        return float(np.dot(query_emb, gallery_emb))
    else:
        # Multi-prototype: ambil similarity tertinggi
        sims = gallery_emb @ query_emb  # (k,)
        return float(sims.max())


def compare_enrollment_strategies(
    embeddings: np.ndarray,
    frame_dirs: list[Path] | None = None,
) -> dict[str, np.ndarray]:
    """
    Bandingkan semua strategi enrollment pada satu set embeddings.
    Berguna untuk ablation study langsung di notebook evaluasi.

    Args:
        embeddings : (N, D) — N embeddings untuk 1 subjek
        frame_dirs : list of Path opsional — untuk quality scoring

    Returns:
        dict: {strategy_name: gallery_embedding}
    """
    results = {}
    for strategy in ("average", "weighted", "median", "multi"):
        enroller = GalleryEnroller(strategy=strategy, k=3)
        results[strategy] = enroller.enroll(embeddings, frame_dirs)
    return results


def batch_enroll_and_compare(
    label_embeddings: dict[str, np.ndarray],
    label_frame_dirs: dict[str, list[Path]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Batch enrollment comparison untuk semua subjek.

    Returns:
        {label: {strategy: gallery_emb}}
    """
    results = {}
    for label, embs in label_embeddings.items():
        dirs = label_frame_dirs.get(label) if label_frame_dirs else None
        results[label] = compare_enrollment_strategies(embs, dirs)
    return results
