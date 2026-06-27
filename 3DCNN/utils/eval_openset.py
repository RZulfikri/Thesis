"""
utils/eval_openset.py — Open-set evaluation via Leave-One-Subject-Out (LOSO) untuk v7.0.0.

Protokol LOSO:
  - 11 fold (satu per subjek)
  - Tiap fold: 10 subjek sebagai "known" (gallery), 1 subjek sebagai "unknown" (impostor)
  - Metrik:
      * Closed-set EER: genuine pairs dari known subjects (standard)
      * FAR@unknown   : fraksi unknown probe yang diterima di threshold EER (false alarm rate untuk unknown)
      * FRR@FAR=1%    : FRR pada threshold yang membuat FAR = 1% (operating point realistis)
      * FNMR@FMR=1%   : alias FRR@FAR=1% (False Non-Match Rate at False Match Rate)

Tidak memerlukan subjek baru — memanfaatkan 11 subjek existing.

Usage:
    from utils.eval_openset import run_loso_eval

    loso_results = run_loso_eval(
        model=model,
        session_splits=session_splits,   # dari build_lowdata_splits_session_dirs
        device=device,
        n_enroll=5,
        m_probe=5,
        fusion_strategy="mean",
        normalizer=normalizer,
    )
    print(loso_results["summary"])
"""

import json
from pathlib import Path

import numpy as np

from utils.eval_multiframe import eval_multiframe, fuse_embeddings, _encode_frames, _sample_n_frames
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Core LOSO runner
# ---------------------------------------------------------------------------

def _compute_far_at_threshold(unknown_scores: np.ndarray, threshold: float) -> float:
    """
    Hitung FAR untuk unknown probe: fraksi skor >= threshold (diterima sebagai known).
    """
    if len(unknown_scores) == 0:
        return float("nan")
    return float(np.mean(unknown_scores >= threshold))


def _compute_frr_at_far(
    genuine_labels: np.ndarray, scores: np.ndarray, target_far: float = 0.01
) -> float:
    """
    Hitung FRR pada operating point FAR = target_far.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(genuine_labels, scores, pos_label=1)
    # Cari threshold terdekat yang membuat FAR ≈ target_far
    idx = np.argmin(np.abs(fpr - target_far))
    frr = 1.0 - tpr[idx]
    return float(frr)


def run_loso_fold(
    model,
    all_session_splits: dict[str, dict[str, list[Path]]],
    unknown_subject: str,
    device,
    n_enroll: int = 5,
    m_probe: int = 5,
    fusion_strategy: str = "mean",
    n_points: int = 1024,
    normalizer=None,
    seed: int | None = None,
    split_name: str = "test",
) -> dict:
    """
    Jalankan satu fold LOSO: unknown_subject sebagai impostor, sisanya sebagai known.

    Args:
        all_session_splits  : {split: {label: [session_dirs]}} dari build_lowdata_splits_session_dirs
        unknown_subject     : label subjek yang dianggap unknown di fold ini
        split_name          : split yang digunakan untuk probe ("test" atau "holdout")

    Returns:
        {
            "unknown_subject"   : str,
            "known_subjects"    : list[str],
            "closed_set_eer"    : float — EER hanya dari known subjects,
            "far_at_unknown"    : float — FAR untuk unknown probe di threshold EER,
            "frr_at_far1"       : float — FRR@FAR=1% untuk known subjects,
            "n_unknown_probes"  : int,
            "n_known_pairs"     : int,
        }
    """
    test_splits = all_session_splits[split_name]
    enroll_splits = all_session_splits["train"]   # gallery dari training sesi

    # Known subjects = semua kecuali unknown_subject
    known_subjects = [s for s in sorted(test_splits) if s != unknown_subject]

    # ── Build gallery (enroll) dari known subjects ───────────────
    gallery_embs: dict[str, np.ndarray] = {}
    for label in known_subjects:
        sessions = enroll_splits.get(label, [])
        if not sessions:
            continue
        enroll_sess = sessions[0]
        frames = _sample_n_frames(enroll_sess, n_enroll, seed=seed)
        if not frames:
            continue
        embs = _encode_frames(model, frames, device, n_points, normalizer)
        if len(embs) == 0:
            continue
        gallery_embs[label] = fuse_embeddings(embs, fusion_strategy)

    # ── Known probe pairs ─────────────────────────────────────────
    known_labels: list[int] = []
    known_scores: list[float] = []

    for label in known_subjects:
        sessions = test_splits.get(label, [])
        enroll_sessions = enroll_splits.get(label, [])
        enroll_sess_set = {enroll_sessions[0]} if enroll_sessions else set()

        for probe_sess in sessions:
            if probe_sess in enroll_sess_set:
                continue
            frames = _sample_n_frames(probe_sess, m_probe, seed=seed)
            if not frames:
                continue
            embs = _encode_frames(model, frames, device, n_points, normalizer)
            if len(embs) == 0:
                continue
            probe_emb = fuse_embeddings(embs, fusion_strategy)

            for gallery_label, gallery_emb in gallery_embs.items():
                sim = float(np.dot(probe_emb, gallery_emb))
                is_genuine = int(label == gallery_label)
                known_labels.append(is_genuine)
                known_scores.append(sim)

    # ── Unknown probe scores ──────────────────────────────────────
    unknown_scores: list[float] = []
    unknown_sessions = test_splits.get(unknown_subject, [])

    for probe_sess in unknown_sessions:
        frames = _sample_n_frames(probe_sess, m_probe, seed=seed)
        if not frames:
            continue
        embs = _encode_frames(model, frames, device, n_points, normalizer)
        if len(embs) == 0:
            continue
        probe_emb = fuse_embeddings(embs, fusion_strategy)
        # Max similarity ke semua gallery entries (paling optimis buat impostor)
        if gallery_embs:
            max_sim = max(float(np.dot(probe_emb, ge)) for ge in gallery_embs.values())
            unknown_scores.append(max_sim)

    # ── Compute metrics ───────────────────────────────────────────
    result = {
        "unknown_subject"  : unknown_subject,
        "known_subjects"   : known_subjects,
        "n_known_pairs"    : len(known_labels),
        "n_unknown_probes" : len(unknown_scores),
    }

    if known_labels:
        labels_arr = np.array(known_labels)
        scores_arr = np.array(known_scores)
        metrics    = compute_all_metrics(labels_arr, scores_arr)
        result["closed_set_eer"] = metrics.get("eer", float("nan"))
        result["auc"]            = metrics.get("auc", float("nan"))
        result["dprime"]         = metrics.get("dprime", float("nan"))
        result["tar_at_far1"]    = metrics.get("tar_at_far1", float("nan"))

        eer_threshold = metrics.get("eer_threshold", 0.5)
        result["eer_threshold"]  = eer_threshold

        # FAR@unknown menggunakan threshold dari closed-set EER
        result["far_at_unknown"] = _compute_far_at_threshold(
            np.array(unknown_scores), eer_threshold
        )

        # FRR@FAR=1% (operating point realistis)
        try:
            result["frr_at_far1"] = _compute_frr_at_far(labels_arr, scores_arr, target_far=0.01)
        except Exception:
            result["frr_at_far1"] = float("nan")

    else:
        result.update({
            "closed_set_eer" : float("nan"),
            "auc"            : float("nan"),
            "dprime"         : float("nan"),
            "tar_at_far1"    : float("nan"),
            "eer_threshold"  : float("nan"),
            "far_at_unknown" : float("nan"),
            "frr_at_far1"    : float("nan"),
        })

    return result


def run_loso_eval(
    model,
    all_session_splits: dict[str, dict[str, list[Path]]],
    device,
    n_enroll: int = 5,
    m_probe: int = 5,
    fusion_strategy: str = "mean",
    n_points: int = 1024,
    normalizer=None,
    seed: int | None = None,
    split_name: str = "test",
    verbose: bool = True,
) -> dict:
    """
    Jalankan LOSO evaluation lengkap (11 fold untuk 11 subjek).

    Returns:
        {
            "folds"   : [fold_result, ...],
            "summary" : {
                "closed_set_eer_mean", "closed_set_eer_std",
                "far_at_unknown_mean", "far_at_unknown_std",
                "frr_at_far1_mean",    "frr_at_far1_std",
                "dprime_mean",         "dprime_std",
                "n_folds",
            },
            "n_enroll", "m_probe", "fusion_strategy", "split_name",
        }
    """
    subjects = sorted(all_session_splits[split_name].keys())
    folds    = []

    for i, unknown_subj in enumerate(subjects):
        if verbose:
            print(f"[LOSO {i+1}/{len(subjects)}] unknown={unknown_subj}...")
        fold_res = run_loso_fold(
            model=model,
            all_session_splits=all_session_splits,
            unknown_subject=unknown_subj,
            device=device,
            n_enroll=n_enroll,
            m_probe=m_probe,
            fusion_strategy=fusion_strategy,
            n_points=n_points,
            normalizer=normalizer,
            seed=seed,
            split_name=split_name,
        )
        folds.append(fold_res)
        if verbose:
            ceer = fold_res.get("closed_set_eer", float("nan"))
            far  = fold_res.get("far_at_unknown", float("nan"))
            frr1 = fold_res.get("frr_at_far1", float("nan"))
            dp   = fold_res.get("dprime", float("nan"))
            print(f"  closed_EER={ceer:.4f}  FAR@unk={far:.4f}  FRR@FAR1%={frr1:.4f}  d'={dp:.3f}")

    # ── Summary stats ─────────────────────────────────────────────
    def _safe_mean(key):
        vals = [f[key] for f in folds if not np.isnan(f.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    def _safe_std(key):
        vals = [f[key] for f in folds if not np.isnan(f.get(key, float("nan")))]
        return float(np.std(vals)) if vals else float("nan")

    summary = {
        "closed_set_eer_mean"  : _safe_mean("closed_set_eer"),
        "closed_set_eer_std"   : _safe_std("closed_set_eer"),
        "far_at_unknown_mean"  : _safe_mean("far_at_unknown"),
        "far_at_unknown_std"   : _safe_std("far_at_unknown"),
        "frr_at_far1_mean"     : _safe_mean("frr_at_far1"),
        "frr_at_far1_std"      : _safe_std("frr_at_far1"),
        "dprime_mean"          : _safe_mean("dprime"),
        "dprime_std"           : _safe_std("dprime"),
        "auc_mean"             : _safe_mean("auc"),
        "auc_std"              : _safe_std("auc"),
        "n_folds"              : len(folds),
    }

    if verbose:
        print("\n[LOSO Summary]")
        print(f"  Closed-set EER : {summary['closed_set_eer_mean']:.4f} ± {summary['closed_set_eer_std']:.4f}")
        print(f"  FAR@unknown    : {summary['far_at_unknown_mean']:.4f} ± {summary['far_at_unknown_std']:.4f}")
        print(f"  FRR@FAR=1%     : {summary['frr_at_far1_mean']:.4f} ± {summary['frr_at_far1_std']:.4f}")
        print(f"  d-prime        : {summary['dprime_mean']:.3f} ± {summary['dprime_std']:.3f}")

    return {
        "folds"            : folds,
        "summary"          : summary,
        "n_enroll"         : n_enroll,
        "m_probe"          : m_probe,
        "fusion_strategy"  : fusion_strategy,
        "split_name"       : split_name,
    }


def loso_to_dataframe(loso_results: dict) -> "pd.DataFrame":
    """Konversi hasil LOSO ke pandas DataFrame (per fold)."""
    import pandas as pd
    folds = loso_results.get("folds", [])
    rows  = []
    for fold in folds:
        rows.append({
            "unknown_subject"  : fold.get("unknown_subject"),
            "closed_set_eer"   : fold.get("closed_set_eer"),
            "far_at_unknown"   : fold.get("far_at_unknown"),
            "frr_at_far1"      : fold.get("frr_at_far1"),
            "dprime"           : fold.get("dprime"),
            "auc"              : fold.get("auc"),
            "n_known_pairs"    : fold.get("n_known_pairs"),
            "n_unknown_probes" : fold.get("n_unknown_probes"),
        })
    return pd.DataFrame(rows)
