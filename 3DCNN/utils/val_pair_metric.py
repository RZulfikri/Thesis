"""
utils/val_pair_metric.py — Val Pair EER/AUC/TAR@FAR Logger untuk model selection.

v5.0.0: Ganti val_loss sebagai metric model selection dengan val pair EER.

Logic:
  1. Di akhir setiap epoch, generate pair dari val set explicit.
  2. Genuine pairs: C(n_val_sessions, 2) per subjek.
  3. Impostor pairs: balanced cross-subject, deterministic sampling.
  4. Compute embedding via encoder eval mode.
  5. Hitung cosine similarity → EER, AUC, TAR@FAR=1%.
  6. Moving average 5-epoch untuk smoothing (karena pair count kecil).

Usage di training loop:
    val_metric = ValPairMetric(device, n_subjects=10)
    val_metric.reset_pairs(val_frames)   # sekali di awal training
    ...
    eer, auc, tar1 = val_metric.compute(model, normalizer)
"""

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import _sample_points, load_session
from utils.metrics import compute_eer, compute_tar_at_far
from sklearn.metrics import roc_auc_score


class ValPairMetric:
    """
    Compute val-set pair EER/AUC/TAR@FAR untuk model selection.

    Args:
        device       : torch.device
        n_points     : jumlah titik sampling (sama dengan training)
        n_impostor   : target jumlah impostor pair total (default 100)
        pair_seed    : seed untuk deterministic impostor sampling
    """

    def __init__(self, device: torch.device, n_points: int = 8192,
                 n_impostor: int = 100, pair_seed: int = 999,
                 repr_mode: str = "canonical_npy"):
        self.device = device
        self.n_points = n_points
        self.n_impostor = n_impostor
        self.pair_seed = pair_seed
        self.repr_mode = repr_mode  # v7.2.0: sumber point cloud val harus sama dgn training
        self.pairs: list[tuple[Path, Path, float]] = []
        self._cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self._history: list[dict] = []

    def reset_pairs(self, label_frames: dict[str, list[Path]]):
        """
        Build deterministic val pairs dari val_frames.

        Args:
            label_frames: {label: [frame_dirs]}
        """
        self.pairs = []
        self._cache = {}
        rng = random.Random(self.pair_seed)

        # Genuine pairs: C(n,2) per label
        genuine_pairs = []
        for label, frames in label_frames.items():
            for a, b in combinations(frames, 2):
                genuine_pairs.append((a, b, 1.0))

        # Balanced impostor pairs
        labels = list(label_frames.keys())
        impostor_pairs = []
        n_target = min(self.n_impostor, len(genuine_pairs) * 10)
        for _ in range(n_target):
            lab_a, lab_b = rng.sample(labels, 2)
            s_a = rng.choice(label_frames[lab_a])
            s_b = rng.choice(label_frames[lab_b])
            impostor_pairs.append((s_a, s_b, 0.0))

        self.pairs = genuine_pairs + impostor_pairs
        rng.shuffle(self.pairs)

        # Preload cache
        all_dirs = {d for pair in self.pairs for d in (pair[0], pair[1])}
        print(f"[ValPairMetric] Preloading {len(all_dirs)} frames for val pair computation...")
        self._cache = {d: load_session(d, repr_mode=self.repr_mode) for d in sorted(all_dirs)}
        print(f"[ValPairMetric] {len(self.pairs)} pairs ready "
              f"({len(genuine_pairs)} genuine, {len(impostor_pairs)} impostor)")

    def compute(self, model, normalizer=None) -> dict[str, float]:
        """
        Compute EER, AUC, TAR@FAR=1% pada val pairs.

        v5.0.1-optimize: Batched encoding — encode semua unique frames sekaligus
        via model.encode() batch, lalu compute pair similarity dari embedding
        cache. Mengurangi waktu validasi dari ~15s → ~1-2s (A100).

        Args:
            model      : SiamesePalmNet atau encoder
            normalizer : GeometryNormalizer atau None

        Returns:
            dict dengan keys: eer, auc, tar_at_far1, n_pairs
        """
        model.eval()

        # --- 1. Build list of unique frames & map ke pair indices ---
        rng = np.random.default_rng(self.pair_seed)
        unique_dirs = sorted(self._cache.keys())
        dir_to_idx = {d: i for i, d in enumerate(unique_dirs)}

        # Pre-allocate numpy arrays untuk semua unique frames
        all_pts = []
        all_geom = []
        with torch.no_grad():
            for d in unique_dirs:
                cloud, geom = self._cache[d]
                pts = _sample_points(cloud, self.n_points, method="random")
                if normalizer is not None:
                    geom = normalizer.transform(geom)
                all_pts.append(pts)
                all_geom.append(geom)

            # --- 2. Batch encode semua unique frames ---
            # Stack jadi tensor: (U, N, 6) dan (U, geom_dim)
            pts_t = torch.from_numpy(np.stack(all_pts)).to(self.device)
            geom_t = torch.from_numpy(np.stack(all_geom)).to(self.device)

            emb_all = model.encode(pts_t, geom_t)  # (U, 128)

        # --- 3. Compute pair similarities dari embedding cache ---
        all_scores = []
        all_labels = []
        for dir_a, dir_b, label in self.pairs:
            idx_a = dir_to_idx[dir_a]
            idx_b = dir_to_idx[dir_b]
            sim = (emb_all[idx_a] * emb_all[idx_b]).sum().item()
            all_scores.append(sim)
            all_labels.append(label)

        labels = np.array(all_labels, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)

        eer, _ = compute_eer(labels, scores)
        auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
        tar1, _ = compute_tar_at_far(labels, scores, far_target=0.01)

        result = {
            "eer": float(eer),
            "auc": float(auc),
            "tar_at_far1": float(tar1),
            "n_pairs": len(self.pairs),
        }
        self._history.append(result)
        return result

    def smoothed_eer(self, window: int = 5) -> float | None:
        """
        Return moving average EER dari history terakhir.

        Returns None kalau history < window.
        """
        if len(self._history) < window:
            return None
        recent = [h["eer"] for h in self._history[-window:]]
        return float(np.mean(recent))

    def best_smoothed_eer(self, window: int = 5) -> tuple[float, int] | None:
        """
        Return (best_smoothed_eer, epoch_index) dari seluruh history.
        epoch_index = 0-based index di history.
        """
        if len(self._history) < window:
            return None
        smoothed = []
        for i in range(len(self._history) - window + 1):
            smoothed.append(float(np.mean([h["eer"] for h in self._history[i:i+window]])))
        best_idx = int(np.argmin(smoothed))
        return smoothed[best_idx], best_idx


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Test ValPairMetric")
    p.add_argument("--dataset_root", default="../dataset")
    args = p.parse_args()

    from utils.dataset_lowdata import build_lowdata_splits
    splits = build_lowdata_splits(args.dataset_root)
    val_frames = splits["val"]

    metric = ValPairMetric(device=torch.device("cpu"), n_points=2048)
    metric.reset_pairs(val_frames)
    print(f"Pairs: {len(metric.pairs)}")
