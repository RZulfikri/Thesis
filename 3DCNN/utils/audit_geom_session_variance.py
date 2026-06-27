"""
utils/audit_geom_session_variance.py — D2: Audit varian fitur geometri
intra-subject (antar sesi) vs antar-subject.

Hipotesis (Plan §D2): fitur 14-dim geometri tidak pose-invariant; varian
antar-sesi besar sehingga signal diskriminatif (between-subject) tertekan.
Bukti yang dicari:
  - per-fitur within-subject std vs between-subject std
  - Fisher's Discriminant Ratio (FDR) = var_between / mean(var_within)
  - histogram subjek 'nola' vs sisanya (subjek yang konsisten gagal di with_geom)

Output:
  eval_results/audits/<ts>/geom_feature_fdr.csv
  eval_results/audits/<ts>/geom_session_variance.json
  eval_results/audits/<ts>/geom_nola_vs_rest.png  (jika matplotlib tersedia)
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset import GEOMETRY_KEYS, GEOMETRY_DIM, _flatten_geometry  # noqa: E402


FEATURE_NAMES = [
    "finger_len_1", "finger_len_2", "finger_len_3", "finger_len_4", "finger_len_5",
    "palm_width", "palm_height", "palm_depth_std",
    "finger_width_1", "finger_width_2", "finger_width_3", "finger_width_4", "finger_width_5",
    "mean_palm_curvature",
]
assert len(FEATURE_NAMES) == GEOMETRY_DIM


def _iter_session_dirs(dataset_root: Path):
    """Yield (subject, session_id, session_path) untuk setiap sesi.

    Layout: dataset/<subject>/<timestamp>/frame_*/geometry.json  (frame-level)
            dataset/<subject>/<timestamp>/geometry.json          (session-level)
    """
    for subject_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        subject = subject_dir.name
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            yield subject, session_dir.name, session_dir


def _load_session_geom(session_dir: Path) -> list[np.ndarray]:
    """Return list vektor (GEOMETRY_DIM,) untuk semua frame dalam sesi."""
    feats: list[np.ndarray] = []
    geo_path = session_dir / "geometry.json"
    if geo_path.exists():
        with open(geo_path) as f:
            feats.append(_flatten_geometry(json.load(f)))
        return feats
    for frame_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        gp = frame_dir / "geometry.json"
        if not gp.exists():
            continue
        try:
            with open(gp) as f:
                feats.append(_flatten_geometry(json.load(f)))
        except (AssertionError, ValueError, KeyError):
            continue
    return feats


def compute_variance(dataset_root: Path) -> dict:
    """Return dict berisi statistik per-fitur.

    Untuk tiap fitur i:
      - within-subject variance (rata-rata var per subjek-sesi vs subject mean)
      - between-subject variance (var dari mean per subjek)
      - FDR = between / mean(within)
      - CV per subjek (std/mean)
    """
    # collect features grouped by subject, session
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    n_sessions = 0
    n_frames = 0
    for subject, session_id, session_dir in _iter_session_dirs(dataset_root):
        feats = _load_session_geom(session_dir)
        if not feats:
            continue
        grouped[subject][session_id] = feats
        n_sessions += 1
        n_frames += len(feats)

    subjects = sorted(grouped.keys())

    # mean per (subject, session) → average geom per session
    subject_session_means: dict[str, list[np.ndarray]] = {}
    for subj in subjects:
        subj_means = []
        for session_id, feats in grouped[subj].items():
            stacked = np.stack(feats, axis=0)
            subj_means.append(stacked.mean(axis=0))
        subject_session_means[subj] = subj_means

    # within-subject var: var antar sesi (mean per sesi) untuk subjek itu
    # between-subject var: var dari mean per subjek
    within_var = np.zeros(GEOMETRY_DIM, dtype=np.float64)
    subject_mean_vectors = []
    valid_subjects = 0
    for subj in subjects:
        sess_means = np.stack(subject_session_means[subj], axis=0)
        if sess_means.shape[0] < 2:
            continue
        within_var += sess_means.var(axis=0, ddof=1)
        subject_mean_vectors.append(sess_means.mean(axis=0))
        valid_subjects += 1
    within_var /= max(valid_subjects, 1)
    subject_means = np.stack(subject_mean_vectors, axis=0)
    between_var = subject_means.var(axis=0, ddof=1)
    fdr = between_var / np.where(within_var < 1e-12, 1e-12, within_var)

    # per-subject CV: std antar-sesi / |mean antar-sesi| (untuk setiap fitur)
    per_subject_cv: dict[str, list[float]] = {}
    for subj in subjects:
        sess_means = np.stack(subject_session_means[subj], axis=0)
        if sess_means.shape[0] < 2:
            continue
        mu = sess_means.mean(axis=0)
        sd = sess_means.std(axis=0, ddof=1)
        cv = sd / np.where(np.abs(mu) < 1e-9, 1e-9, np.abs(mu))
        per_subject_cv[subj] = cv.tolist()

    return {
        "n_subjects": valid_subjects,
        "n_sessions": n_sessions,
        "n_frames": n_frames,
        "feature_names": FEATURE_NAMES,
        "within_subject_std": np.sqrt(within_var).tolist(),
        "between_subject_std": np.sqrt(between_var).tolist(),
        "fdr": fdr.tolist(),
        "per_subject_cv": per_subject_cv,
        "subject_means": {s: m.tolist() for s, m in zip(subjects, subject_mean_vectors)},
    }


def _write_csv(stats: dict, out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "within_subject_std", "between_subject_std", "fdr"])
        for i, name in enumerate(stats["feature_names"]):
            w.writerow([
                name,
                f"{stats['within_subject_std'][i]:.4g}",
                f"{stats['between_subject_std'][i]:.4g}",
                f"{stats['fdr'][i]:.4g}",
            ])


def _maybe_plot_nola(stats: dict, dataset_root: Path, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # collect raw frame features for nola vs rest
    nola_feats: list[np.ndarray] = []
    rest_feats: list[np.ndarray] = []
    for subject, session_id, session_dir in _iter_session_dirs(dataset_root):
        feats = _load_session_geom(session_dir)
        if not feats:
            continue
        target = nola_feats if subject == "nola" else rest_feats
        target.extend(feats)
    if not nola_feats:
        return
    nola_arr = np.stack(nola_feats, axis=0)
    rest_arr = np.stack(rest_feats, axis=0)

    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    axes = axes.ravel()
    for i, name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        ax.hist(rest_arr[:, i], bins=40, alpha=0.5, label="rest", density=True)
        ax.hist(nola_arr[:, i], bins=20, alpha=0.7, label="nola", density=True)
        ax.set_title(f"{name} (FDR={stats['fdr'][i]:.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(len(FEATURE_NAMES), len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Distribusi fitur geometri: subjek 'nola' vs sisanya", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    dataset_root = ROOT / "dataset"
    out_dir = ROOT / "eval_results" / "audits" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_variance(dataset_root)

    json_path = out_dir / "geom_session_variance.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    csv_path = out_dir / "geom_feature_fdr.csv"
    _write_csv(stats, csv_path)

    plot_path = out_dir / "geom_nola_vs_rest.png"
    _maybe_plot_nola(stats, dataset_root, plot_path)

    fdr = np.asarray(stats["fdr"])
    n_low = int((fdr < 1.0).sum())
    median_fdr = float(np.median(fdr))
    verdict_terms = []
    if n_low >= len(fdr) // 2:
        verdict_terms.append(
            f"TERKONFIRMASI sebagian: {n_low}/{GEOMETRY_DIM} fitur memiliki FDR<1 "
            f"(median FDR={median_fdr:.2f}). Signal diskriminatif tipis."
        )
    else:
        verdict_terms.append(
            f"TIDAK SEPENUHNYA: hanya {n_low}/{GEOMETRY_DIM} fitur dengan FDR<1 "
            f"(median FDR={median_fdr:.2f}). Sebagian fitur tetap diskriminatif."
        )

    summary = {
        "audit": "geom_session_variance",
        "stats_path": str(json_path.name),
        "csv_path": str(csv_path.name),
        "plot_path": str(plot_path.name) if plot_path.exists() else None,
        "feature_fdr": dict(zip(FEATURE_NAMES, stats["fdr"])),
        "n_features_fdr_below_1": n_low,
        "median_fdr": median_fdr,
        "verdict": " | ".join(verdict_terms),
    }
    sum_path = out_dir / "summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[D2] Wrote {sum_path}")
    print(f"[D2] verdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
