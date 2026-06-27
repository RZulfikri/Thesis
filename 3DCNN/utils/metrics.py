"""
utils/metrics.py — Biometric evaluation metrics.

Metrics: EER, TAR@FAR=1%, TAR@FAR=0.1%, d-prime, Accuracy@EER, FAR, FRR, AUC
Plots  : ROC curve (multi-model), similarity distribution, t-SNE, DET curve
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def fig_to_tensor(fig) -> np.ndarray:
    """
    Konversi matplotlib Figure → HWC uint8 numpy array untuk TensorBoard.
    """
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) — where FAR == FRR.

    Args:
        labels : (N,) int/float — 1=genuine, 0=impostor
        scores : (N,) float    — similarity score ∈ [-1, 1]

    Returns:
        eer       : float — EER value (lower is better)
        threshold : float — threshold at EER
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr  # FRR = 1 - TPR

    # Cari titik di mana FPR (FAR) ≈ FNR (FRR)
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx])
    return eer, threshold


def compute_tar_at_far(
    labels: np.ndarray,
    scores: np.ndarray,
    far_target: float = 0.01,
) -> tuple[float, float]:
    """
    Compute True Acceptance Rate (TAR) pada target FAR tertentu.

    Metrik penting untuk biometrik: mengukur seberapa banyak genuine pairs
    yang diterima saat FAR dikunci ke nilai tertentu (misalnya 1% atau 0.1%).

    Args:
        labels     : (N,) — 1=genuine, 0=impostor
        scores     : (N,) — similarity scores
        far_target : FAR yang diinginkan (default: 0.01 = 1%)

    Returns:
        tar       : TAR pada FAR ≤ far_target
        threshold : threshold yang digunakan
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    # Cari index dimana fpr ≤ far_target (ambil titik terbesar fpr yang masih ≤ target)
    valid = np.where(fpr <= far_target)[0]
    if len(valid) == 0:
        return 0.0, float(thresholds[0]) if len(thresholds) > 0 else 0.0
    idx = valid[-1]
    return float(tpr[idx]), float(thresholds[idx])


def compute_dprime(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute d-prime (d') — indeks diskriminabilitas.

    d' = (μ_genuine - μ_impostor) / sqrt((σ²_genuine + σ²_impostor) / 2)

    Nilai lebih tinggi = pemisahan distribusi lebih baik.
    d' ≥ 2.0 umumnya dianggap baik untuk biometrik.

    Args:
        labels : (N,) — 1=genuine, 0=impostor
        scores : (N,) — cosine similarity scores

    Returns:
        dprime : float — nilai d' (higher is better)
    """
    genuine  = scores[labels == 1]
    impostor = scores[labels == 0]
    if len(genuine) == 0 or len(impostor) == 0:
        return 0.0
    mu_g   = genuine.mean()
    mu_i   = impostor.mean()
    sig_g  = genuine.std()
    sig_i  = impostor.std()
    denom  = np.sqrt((sig_g ** 2 + sig_i ** 2) / 2.0)
    if denom < 1e-8:
        return 0.0
    return float((mu_g - mu_i) / denom)


def compute_accuracy_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> float:
    """
    Accuracy = (TP + TN) / N pada threshold tertentu.

    Args:
        labels    : (N,) — 1=genuine, 0=impostor
        scores    : (N,) — similarity scores
        threshold : decision threshold

    Returns:
        accuracy : float ∈ [0, 1]
    """
    predicted = (scores >= threshold).astype(int)
    return float((predicted == labels.astype(int)).mean())


def compute_far_frr(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FAR dan FRR pada setiap threshold.

    Args:
        labels     : (N,) — 1=genuine, 0=impostor
        scores     : (N,) — similarity scores
        thresholds : (T,) — threshold values to sweep

    Returns:
        far : (T,) — False Acceptance Rate per threshold
        frr : (T,) — False Rejection Rate per threshold
    """
    genuine_mask  = labels == 1
    impostor_mask = labels == 0
    n_genuine  = genuine_mask.sum()
    n_impostor = impostor_mask.sum()

    far = np.zeros(len(thresholds), dtype=np.float32)
    frr = np.zeros(len(thresholds), dtype=np.float32)

    for i, t in enumerate(thresholds):
        predicted_genuine = scores >= t
        # FAR: impostor diterima
        far[i] = (predicted_genuine & impostor_mask).sum() / max(n_impostor, 1)
        # FRR: genuine ditolak
        frr[i] = (~predicted_genuine & genuine_mask).sum() / max(n_genuine, 1)

    return far, frr


def compute_confusion_matrix(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Hitung confusion matrix 2×2 (Genuine vs Impostor) pada threshold tertentu.
    Default threshold = EER threshold.

    Returns:
        dict dengan keys: threshold, tp, tn, fp, fn,
                          sensitivity, specificity, precision, recall, f1
    """
    if threshold is None:
        threshold, _ = compute_eer(labels, scores)

    preds = (scores >= threshold).astype(int)
    labels = labels.astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = sensitivity
    f1          = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "threshold":    float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
    }


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """
    Hitung semua metrik evaluasi biometrik sekaligus.

    Returns:
        dict dengan keys:
          eer, eer_threshold, auc,
          tar_at_far1, tar_at_far01,    ← TAR@FAR=1% dan TAR@FAR=0.1%
          dprime,
          accuracy_at_eer,
          far_at_eer, frr_at_eer,
          confusion_matrix (dict tp/tn/fp/fn + derived rates)
    """
    eer, eer_thresh = compute_eer(labels, scores)
    auc             = roc_auc_score(labels, scores)
    tar1,  _        = compute_tar_at_far(labels, scores, far_target=0.01)
    tar01, _        = compute_tar_at_far(labels, scores, far_target=0.001)
    dprime          = compute_dprime(labels, scores)
    acc             = compute_accuracy_at_threshold(labels, scores, eer_thresh)
    cm              = compute_confusion_matrix(labels, scores, threshold=eer_thresh)

    thresholds = np.linspace(-1, 1, 500)
    far_arr, frr_arr = compute_far_frr(labels, scores, thresholds)
    eer_idx     = np.argmin(np.abs(far_arr - frr_arr))
    far_at_eer  = float(far_arr[eer_idx])
    frr_at_eer  = float(frr_arr[eer_idx])

    return {
        "eer":              eer,
        "eer_threshold":    eer_thresh,
        "auc":              auc,
        "tar_at_far1":      tar1,    # TAR @ FAR=1%
        "tar_at_far01":     tar01,   # TAR @ FAR=0.1%
        "dprime":           dprime,
        "accuracy_at_eer":  acc,
        "far_at_eer":       far_at_eer,
        "frr_at_eer":       frr_at_eer,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(
    models_results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str | Path | None = None,
) -> None:
    """
    Plot ROC curves untuk beberapa model sekaligus.

    Args:
        models_results : {model_name: (labels, scores)}
        save_path      : jika diberikan, simpan gambar ke path ini
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    for name, (labels, scores) in models_results.items():
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        auc   = roc_auc_score(labels, scores)
        eer, _= compute_eer(labels, scores)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  AUC={auc:.3f}  EER={eer:.3f}")

    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (1 - FRR)")
    ax.set_title("ROC Curves — Palm Recognition")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"ROC plot saved to {save_path}")
    plt.show()


def plot_det(
    models_results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str | Path | None = None,
) -> None:
    """
    Plot DET (Detection Error Tradeoff) curves — FAR vs FRR.
    Kurva yang lebih dekat ke kiri-bawah = lebih baik.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    thresholds = np.linspace(-1, 1, 500)

    for name, (labels, scores) in models_results.items():
        far, frr = compute_far_frr(labels, scores, thresholds)
        eer, _   = compute_eer(labels, scores)
        ax.plot(far * 100, frr * 100, lw=2, label=f"{name}  EER={eer:.3f}")

    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title("DET Curves — Palm Recognition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"DET plot saved to {save_path}")
    plt.show()


def plot_similarity_dist(
    labels: np.ndarray,
    scores: np.ndarray,
    title: str = "Similarity Distribution",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot histogram distribusi similarity genuine vs impostor.

    Args:
        labels    : (N,) — 1=genuine, 0=impostor
        scores    : (N,) — cosine similarity ∈ [-1, 1]
        title     : judul plot
        save_path : jika diberikan, simpan gambar ke path ini
    """
    import matplotlib.pyplot as plt

    genuine_scores  = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    eer, eer_thresh = compute_eer(labels, scores)
    dprime          = compute_dprime(labels, scores)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-1, 1, 40)
    ax.hist(genuine_scores,  bins=bins, alpha=0.6, color="green",
            label=f"Genuine  (n={len(genuine_scores)})")
    ax.hist(impostor_scores, bins=bins, alpha=0.6, color="red",
            label=f"Impostor (n={len(impostor_scores)})")
    ax.axvline(eer_thresh, color="black", linestyle="--", lw=1.5,
               label=f"EER threshold={eer_thresh:.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}  [EER={eer:.4f}  d'={dprime:.2f}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Similarity distribution plot saved to {save_path}")
    plt.show()


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Embedding Space",
    save_path: str | Path | None = None,
) -> None:
    """
    Visualisasi 128-dim embeddings menggunakan t-SNE.

    Args:
        embeddings : (N, 128) float32
        labels     : (N,) int — identity label per embedding
        title      : judul plot
        save_path  : jika diberikan, simpan gambar ke path ini
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne   = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], label=str(lbl), s=20, alpha=0.7)
    ax.set_title(title)
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_table(results: list[dict]) -> None:
    """
    Cetak tabel metrik evaluasi lengkap ke stdout.

    Args:
        results: list of dicts dengan keys:
                 model, eer, auc, tar_at_far1, tar_at_far01,
                 dprime, accuracy_at_eer, far_at_eer, frr_at_eer
    """
    cols = [
        ("Model",       "model",           "<12", lambda v: v),
        ("EER",         "eer",             ">7",  lambda v: f"{v:.4f}"),
        ("AUC",         "auc",             ">7",  lambda v: f"{v:.4f}"),
        ("TAR@FAR1%",   "tar_at_far1",     ">10", lambda v: f"{v:.4f}"),
        ("TAR@FAR0.1%", "tar_at_far01",    ">11", lambda v: f"{v:.4f}"),
        ("d'",          "dprime",          ">6",  lambda v: f"{v:.3f}"),
        ("Acc@EER",     "accuracy_at_eer", ">8",  lambda v: f"{v:.4f}"),
        ("FAR@EER",     "far_at_eer",      ">8",  lambda v: f"{v:.4f}"),
        ("FRR@EER",     "frr_at_eer",      ">8",  lambda v: f"{v:.4f}"),
    ]
    header = "  ".join(f"{label:{width}}" for label, _, width, _ in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        row = "  ".join(f"{fmt(r.get(key, 0)):{width}}"
                        for _, key, width, fmt in cols)
        print(row)


# Alias untuk backward compatibility
def print_ablation_table(results: list[dict]) -> None:
    print_metrics_table(results)


# ---------------------------------------------------------------------------
# Identification metrics (1:N) — Rank-N, CMC, mAP
# ---------------------------------------------------------------------------

def compute_cmc_curve(rank_positions: list[int] | np.ndarray, max_rank: int) -> np.ndarray:
    """
    CMC (Cumulative Matching Characteristic) curve.

    Args:
        rank_positions : posisi rank dari label benar untuk setiap probe (1-indexed)
        max_rank       : panjang kurva CMC (biasanya = jumlah subjek di gallery)

    Returns:
        cmc : (max_rank,) — recognition rate kumulatif per rank ∈ [0, 1]
    """
    rank_positions = np.asarray(rank_positions)
    cmc = np.zeros(max_rank, dtype=np.float64)
    for pos in rank_positions:
        if 1 <= pos <= max_rank:
            cmc[pos - 1:] += 1
    return cmc / max(len(rank_positions), 1)


def compute_rank_n(rank_positions: list[int] | np.ndarray, n: int) -> float:
    """Rank-N accuracy: fraksi probe dengan posisi rank ≤ n."""
    rank_positions = np.asarray(rank_positions)
    if len(rank_positions) == 0:
        return 0.0
    return float((rank_positions <= n).mean())


def compute_map(
    rank_positions: list[int] | np.ndarray,
    gallery_label_counts: dict[str, int] | None = None,
    probe_labels: list[str] | None = None,
) -> float:
    """
    Mean Average Precision untuk identifikasi closed-set.

    Asumsi: setiap subjek punya tepat 1 entri di gallery (mean embedding) →
    AP per probe = 1/rank_pos. mAP = mean(1/rank_pos).

    Args:
        rank_positions : posisi rank label benar (1-indexed) per probe
        gallery_label_counts, probe_labels : tidak digunakan untuk gallery 1-per-subjek;
                                              parameter dipertahankan untuk ekstensi multi-gallery.

    Returns:
        mAP ∈ [0, 1]
    """
    rank_positions = np.asarray(rank_positions, dtype=np.float64)
    if len(rank_positions) == 0:
        return 0.0
    return float((1.0 / rank_positions).mean())


# ---------------------------------------------------------------------------
# Statistical inference — bootstrap CI, paired tests
# ---------------------------------------------------------------------------

def bootstrap_ci(
    sample_a: np.ndarray,
    sample_b: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    paired: bool = True,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval untuk delta = mean(a) - mean(b) (paired) atau
    untuk mean(a) saja (jika b=None).

    Cocok untuk metrik per-seed (a = with_geom seeds, b = no_geom seeds).

    Args:
        sample_a    : (n,) — sampel utama (mis. Rank-1 per seed untuk varian A)
        sample_b    : (n,) atau None — sampel pembanding (paired dengan A)
        n_bootstrap : jumlah resampling (default 1000)
        confidence  : level confidence (default 0.95)
        paired      : True → resample indeks bersama (paired); False → independen
        seed        : RNG seed untuk reproducibility

    Returns:
        dict { mean, ci_low, ci_high, n_bootstrap, confidence }
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(sample_a, dtype=np.float64)

    if sample_b is None:
        n = len(a)
        means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            means[i] = a[idx].mean()
        observed = float(a.mean())
    else:
        b = np.asarray(sample_b, dtype=np.float64)
        if paired:
            assert len(a) == len(b), "Paired bootstrap memerlukan ukuran sampel yang sama"
            n = len(a)
            means = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                means[i] = a[idx].mean() - b[idx].mean()
            observed = float(a.mean() - b.mean())
        else:
            na, nb = len(a), len(b)
            means = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                idx_a = rng.integers(0, na, size=na)
                idx_b = rng.integers(0, nb, size=nb)
                means[i] = a[idx_a].mean() - b[idx_b].mean()
            observed = float(a.mean() - b.mean())

    alpha = 1.0 - confidence
    ci_low  = float(np.quantile(means, alpha / 2))
    ci_high = float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean":        observed,
        "ci_low":      ci_low,
        "ci_high":     ci_high,
        "n_bootstrap": n_bootstrap,
        "confidence":  confidence,
    }


def paired_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    test: str = "auto",
) -> dict:
    """
    Paired test untuk membandingkan dua varian per seed.

    Args:
        sample_a, sample_b : (n,) — sampel berpasangan (mis. Rank-1 per seed)
        test : "ttest", "wilcoxon", atau "auto" (auto pilih wilcoxon jika n<10)

    Returns:
        dict { test, statistic, pvalue, n }
    """
    from scipy import stats
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    assert len(a) == len(b), "Paired test memerlukan ukuran sampel yang sama"
    n = len(a)

    if test == "auto":
        test = "wilcoxon" if n < 10 else "ttest"

    if test == "ttest":
        result = stats.ttest_rel(a, b)
        stat, p = float(result.statistic), float(result.pvalue)
    elif test == "wilcoxon":
        # Wilcoxon butuh perbedaan non-zero
        diff = a - b
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            result = stats.wilcoxon(a, b, zero_method="zsplit")
            stat, p = float(result.statistic), float(result.pvalue)
    else:
        raise ValueError(f"test tidak dikenal: {test}")

    return {"test": test, "statistic": stat, "pvalue": p, "n": n}


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    continuity_correction: bool = True,
) -> dict:
    """
    McNemar test pada keputusan benar/salah per probe untuk dua model.

    Untuk identifikasi: correct_a[i] = (pred_a[i] == true[i]).
    Mengukur apakah pola diskordansi (model A benar & B salah) ≠ (A salah & B benar).

    Args:
        correct_a, correct_b : (n,) bool — keputusan Rank-1 benar per probe
        continuity_correction : True → pakai chi-squared dengan koreksi kontinuitas

    Returns:
        dict { b, c, statistic, pvalue, n }
        di mana b = #(A salah, B benar), c = #(A benar, B salah)
    """
    from scipy import stats
    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    assert len(a) == len(b_arr)
    n = len(a)

    b_count = int((~a & b_arr).sum())   # A salah, B benar
    c_count = int((a & ~b_arr).sum())   # A benar, B salah

    if (b_count + c_count) == 0:
        return {"b": b_count, "c": c_count, "statistic": 0.0, "pvalue": 1.0, "n": n}

    if continuity_correction:
        stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    else:
        stat = (b_count - c_count) ** 2 / (b_count + c_count)
    pvalue = 1.0 - stats.chi2.cdf(stat, df=1)

    return {"b": b_count, "c": c_count, "statistic": float(stat), "pvalue": float(pvalue), "n": n}


def evaluate_identification(
    dir_to_emb: dict,
    rank_n_list: list[int] | None = None,
) -> dict:
    """
    Closed-set identification (1:N) — leave-one-session-out.

    Gallery = mean embedding dari SEMUA sesi suatu identitas KECUALI
    sesi probe (leave-one-out per probe).
    Probe   = setiap sesi individual.

    Returns:
        dict dengan keys:
          rank1, rank5, rank10, map, cmc_curve, rank_positions,
          probe_results (list of dict: true_label, pred_label, rank, probe_path)
    """
    from collections import defaultdict

    if rank_n_list is None:
        rank_n_list = [1, 5, 10]

    # Identity label dari path: .../[label]/[timestamp]/frame_NN/ → label = dir.parent.parent.name
    label_to_dirs: dict[str, list] = defaultdict(list)
    for d in dir_to_emb:
        label = d.parent.parent.name
        label_to_dirs[label].append(d)

    gallery_labels = sorted(label_to_dirs.keys())
    if len(gallery_labels) < 2:
        return {f"rank{n}": 0.0 for n in rank_n_list} | {"map": 0.0, "probe_results": []}

    rank_positions = []
    probe_results = []
    for label, dirs in label_to_dirs.items():
        for d in dirs:
            probe_emb = dir_to_emb[d]

            # Build gallery TANPA probe session ini
            gallery_embs = []
            for lbl in gallery_labels:
                if lbl == label:
                    other = [x for x in label_to_dirs[lbl] if x != d]
                    if not other:          # hanya 1 sesi → fallback pakai dirinya sendiri
                        other = [d]
                else:
                    other = label_to_dirs[lbl]
                embs = np.stack([dir_to_emb[x] for x in other])
                gallery_embs.append(embs.mean(axis=0))

            gallery_matrix = np.stack(gallery_embs)  # (N_id, 128)
            sims = gallery_matrix @ probe_emb         # cosine (sudah L2-norm)

            ranked_idx = np.argsort(sims)[::-1]
            true_idx = gallery_labels.index(label)
            pred_label = gallery_labels[ranked_idx[0]]
            rank = int(np.where(ranked_idx == true_idx)[0][0] + 1)  # 1-indexed
            rank_positions.append(rank)
            probe_results.append({
                "true_label": label,
                "pred_label": pred_label,
                "rank": rank,
                "probe_path": str(d),
            })

    rank_positions = np.array(rank_positions)
    max_rank = len(gallery_labels)
    cmc = compute_cmc_curve(rank_positions, max_rank)

    results: dict = {}
    for n in rank_n_list:
        results[f"rank{n}"] = float(compute_rank_n(rank_positions, n))
    results["map"] = float(compute_map(rank_positions))
    results["cmc_curve"] = [float(v) for v in cmc]
    results["rank_positions"] = [int(v) for v in rank_positions]
    results["probe_results"] = probe_results
    results["gallery_labels"] = gallery_labels
    return results


def test_set_fingerprint(probe_dirs: list, gallery_labels: list | None = None) -> str:
    """
    Hash deterministik dari (sorted) probe paths + gallery labels.

    Dipakai compare.ipynb untuk verifikasi kedua varian dievaluasi pada split test
    yang IDENTIK. Dua run dengan fingerprint berbeda tidak boleh dibandingkan.

    Args:
        probe_dirs    : list of (label, Path) atau list of Path
        gallery_labels : list of str (urutan label di gallery)

    Returns:
        SHA1 hex digest (16 char prefix)
    """
    import hashlib
    import json
    items = []
    for entry in probe_dirs:
        if isinstance(entry, tuple):
            items.append([str(entry[0]), str(entry[1])])
        else:
            items.append(str(entry))
    payload = {
        "probes":  sorted(items, key=lambda x: str(x)),
        "gallery": sorted(map(str, gallery_labels)) if gallery_labels else None,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]
