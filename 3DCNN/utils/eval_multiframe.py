"""
utils/eval_multiframe.py — Multi-frame fusion evaluation untuk v7.0.0.

Skenario deployment: scan 1 detik ≈ 10 frame per sesi (sudah tersedia di dataset).
Modul ini memformalkan evaluasi dengan:
  - N frame per sesi untuk enrollment (gallery)
  - M frame per sesi untuk probe

Fusion strategy yang didukung:
  - "mean"   : rata-rata embedding (baseline, L2-normalized)
  - "median" : median embedding (robust ke outlier)
  - "max"    : max-pool per dimensi (agresif)
  - "first"  : ambil frame pertama saja (ablation: N=1 deterministik)

Ablation matrix:
  Jalankan eval_multiframe_ablation(model, splits, N_list, M_list)
  untuk mendapatkan EER pada setiap kombinasi (N, M).

Usage:
    from utils.eval_multiframe import eval_multiframe, eval_multiframe_ablation
    from utils.dataset_lowdata import build_lowdata_splits_session_dirs

    session_splits = build_lowdata_splits_session_dirs(dataset_root)
    results = eval_multiframe(
        model, session_splits["test"], session_splits["test"],
        n_enroll=5, m_probe=5,
        fusion_strategy="mean",
        device=device,
        normalizer=normalizer,
    )
"""

import time
from pathlib import Path

import numpy as np
import torch

from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Frame loading helpers
# ---------------------------------------------------------------------------

def _load_frame(frame_dir: Path, n_points: int = 1024,
                normalizer=None,
                repr_mode: str = "canonical_npy") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load satu frame → (pts tensor, geom tensor).

    Args:
        frame_dir  : path ke frame_XX dir
        n_points   : jumlah titik point cloud
        normalizer : GeometryNormalizer (opsional)
        repr_mode  : v7.2.0 — "canonical_npy" (R2) | "fps_npy" (R3) | "raw_ply" (R1).
                     Sumber point cloud; geometry.json selalu sama. Lihat utils.dataset.

    Returns:
        pts  : (1, n_points, 6) float32 tensor (XYZ + normals)
        geom : (1, 13) float32 tensor
    """
    from utils.dataset import load_session

    # load_session memilih file sesuai repr_mode (R1 PLY / R2 npy / R3 fps npy)
    pts, geom = load_session(frame_dir, repr_mode=repr_mode)   # (N,6), (13,)
    pts = pts[:, :6]

    # Subsample / pad ke n_points. Untuk R3 (fps_npy, sudah 8192) dengan n_points=8192
    # → len==n_points: tidak ada sampling acak (deterministik).
    if len(pts) >= n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    else:
        idx = np.random.choice(len(pts), n_points, replace=True)
        pts = pts[idx]

    geom = geom.astype(np.float32)
    if normalizer is not None:
        geom = normalizer.transform(geom)

    pts_t  = torch.from_numpy(pts.astype(np.float32)).unsqueeze(0)   # (1, N, 6)
    geom_t = torch.from_numpy(geom).unsqueeze(0)                      # (1, 13)
    return pts_t, geom_t


def _encode_frames(
    model,
    frame_dirs: list[Path],
    device: torch.device,
    n_points: int = 1024,
    normalizer=None,
    repr_mode: str = "canonical_npy",
    frame_cache: dict | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Encode daftar frame_dirs menjadi embedding matrix — BATCHED (GPU efisien).

    frame_cache: bila diberikan (dict), embedding per-frame_dir di-memoize → frame yang
    sama TIDAK di-encode ulang lintas pemanggilan (mis. sweep N×M). Aman untuk paritas
    karena (model, repr_mode, n_points, normalizer) konstan dalam satu sweep; key = frame_dir.

    Frame di-encode dalam batch (default 128) bukan satu-per-satu → utilisasi GPU jauh
    lebih tinggi. Paritas numerik terjaga: model.eval() ⇒ BatchNorm pakai running-stats
    (batch-invariant); forward batch == forward per-sampel.

    Returns:
        embeddings : (len(frame_dirs), D) float32 numpy  (urutan = frame_dirs, gagal-load dibuang)
    """
    model.eval()
    out = [None] * len(frame_dirs)
    pend_i, pend_pts, pend_geom = [], [], []

    def _flush():
        if not pend_i:
            return
        with torch.no_grad():
            pts = torch.cat(pend_pts, 0).to(device)
            geom = torch.cat(pend_geom, 0).to(device)
            emb = model.encoder(pts, geom).cpu().numpy()   # (B, D) L2-normed
        for k, idx in enumerate(pend_i):
            out[idx] = emb[k]
            if frame_cache is not None:
                frame_cache[frame_dirs[idx]] = emb[k]
        pend_i.clear(); pend_pts.clear(); pend_geom.clear()

    for i, fd in enumerate(frame_dirs):
        if frame_cache is not None and fd in frame_cache:
            out[i] = frame_cache[fd]; continue
        try:
            pts_t, geom_t = _load_frame(fd, n_points=n_points, normalizer=normalizer,
                                        repr_mode=repr_mode)
        except Exception as e:
            print(f"[WARN] Gagal encode {fd}: {e}"); continue
        pend_i.append(i); pend_pts.append(pts_t); pend_geom.append(geom_t)
        if len(pend_i) >= batch_size:
            _flush()
    _flush()
    embeddings = [e for e in out if e is not None]
    return np.stack(embeddings) if embeddings else np.zeros((0, 128))


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

def fuse_embeddings(embeddings: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Fusi (N, D) embeddings menjadi 1 (D,) embedding.

    Strategi:
        "mean"   : L2-normalize rata-rata (default)
        "median" : L2-normalize median
        "max"    : L2-normalize max-pool per dimensi
        "first"  : ambil embedding pertama

    Returns:
        fused : (D,) L2-normalized
    """
    if len(embeddings) == 0:
        raise ValueError("embeddings kosong")

    if strategy == "mean":
        fused = embeddings.mean(axis=0)
    elif strategy == "median":
        fused = np.median(embeddings, axis=0)
    elif strategy == "max":
        fused = embeddings.max(axis=0)
    elif strategy == "first":
        fused = embeddings[0]
    else:
        raise ValueError(f"Unknown fusion strategy: '{strategy}'. "
                         "Pilih: 'mean', 'median', 'max', 'first'")

    norm = np.linalg.norm(fused)
    return fused / norm if norm > 0 else fused


# ---------------------------------------------------------------------------
# Session-level frame sampling
# ---------------------------------------------------------------------------

def _get_valid_frames_from_session(session_dir: Path) -> list[Path]:
    return sorted(
        p for p in session_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("frame_")
        and (p / "cnn_input.npy").exists()
        and (p / "geometry.json").exists()
    )


def _sample_n_frames(session_dir: Path, n: int, seed: int | None = None) -> list[Path]:
    """
    Ambil hingga n frame dari session_dir secara deterministik (evenly spaced)
    atau acak jika seed diberikan.
    """
    frames = _get_valid_frames_from_session(session_dir)
    if not frames:
        return []
    if n <= 0 or n >= len(frames):
        return frames
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(frames), n, replace=False)
        idx = sorted(idx.tolist())
        return [frames[i] for i in idx]
    else:
        # Evenly spaced (deterministik)
        indices = np.linspace(0, len(frames) - 1, n, dtype=int).tolist()
        return [frames[i] for i in indices]


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def eval_multiframe(
    model,
    enroll_session_splits: dict[str, list[Path]],
    probe_session_splits: dict[str, list[Path]],
    n_enroll: int = 5,
    m_probe: int = 5,
    fusion_strategy: str = "mean",
    device: torch.device | None = None,
    n_points: int = 1024,
    normalizer=None,
    seed: int | None = None,
    cross_session: bool = True,
    repr_mode: str = "canonical_npy",
    frame_cache: dict | None = None,
    batch_size: int = 128,
) -> dict:
    """
    Evaluasi multi-frame verification (1:1) dengan enrollment dan probe fusion.

    Protocol:
      - Untuk setiap subjek: ambil n_enroll frames dari sesi enroll → fuse → gallery emb
      - Probe: ambil m_probe frames dari sesi probe (sesi berbeda dari enroll jika cross_session=True)
      - Hitung cosine similarity semua genuine & impostor pairs
      - Kembalikan metrik EER, AUC, d-prime, dll

    Args:
        model                  : SiamesePalmNet atau encoder yang memiliki model.encoder
        enroll_session_splits  : {label: [session_dirs]} — sesi untuk enrollment/gallery
        probe_session_splits   : {label: [session_dirs]} — sesi untuk probe
        n_enroll               : jumlah frame per sesi untuk enrollment (0 = semua)
        m_probe                : jumlah frame per sesi untuk probe (0 = semua)
        fusion_strategy        : "mean" | "median" | "max" | "first"
        device                 : torch.device (default: cpu)
        n_points               : jumlah titik point cloud per frame
        normalizer             : GeometryNormalizer (opsional)
        seed                   : seed untuk sampling frame (None = deterministik evenly spaced)
        cross_session          : jika True, pastikan probe sesi berbeda dari enroll sesi

    Returns:
        {
            "eer", "auc", "tar_at_far1", "dprime", "accuracy_at_eer",
            "n_enroll", "m_probe", "fusion_strategy",
            "gallery_embs"  : {label: (D,) gallery embedding},
            "probe_embs"    : [(label, emb)],
            "latency_enroll_s", "latency_probe_s",
        }
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    labels_all   = []
    scores_all   = []
    gallery_embs = {}
    probe_embs   = []

    all_labels = sorted(set(enroll_session_splits) | set(probe_session_splits))

    # ── Enrollment ────────────────────────────────────────────────
    t0_enroll = time.perf_counter()
    for label, sessions in enroll_session_splits.items():
        if not sessions:
            continue
        # Gunakan sesi pertama untuk enrollment
        enroll_sess = sessions[0]
        frames = _sample_n_frames(enroll_sess, n_enroll, seed=seed)
        if not frames:
            continue
        embs = _encode_frames(model, frames, device, n_points, normalizer, repr_mode=repr_mode,
                              frame_cache=frame_cache, batch_size=batch_size)
        if len(embs) == 0:
            continue
        gallery_embs[label] = fuse_embeddings(embs, fusion_strategy)
    t_enroll = time.perf_counter() - t0_enroll

    # ── Probe ─────────────────────────────────────────────────────
    t0_probe = time.perf_counter()
    for label, sessions in probe_session_splits.items():
        # Pilih sesi probe: hindari sesi yang dipakai untuk enroll jika cross_session
        enroll_sessions = enroll_session_splits.get(label, [])
        enroll_sess_set = {enroll_sessions[0]} if enroll_sessions else set()

        probe_sessions = [s for s in sessions if s not in enroll_sess_set] if cross_session else sessions
        if not probe_sessions:
            probe_sessions = sessions  # fallback

        for probe_sess in probe_sessions:
            frames = _sample_n_frames(probe_sess, m_probe, seed=seed)
            if not frames:
                continue
            embs = _encode_frames(model, frames, device, n_points, normalizer, repr_mode=repr_mode,
                                  frame_cache=frame_cache, batch_size=batch_size)
            if len(embs) == 0:
                continue
            probe_emb = fuse_embeddings(embs, fusion_strategy)
            probe_embs.append((label, probe_emb))

            # Hitung similarity ke semua gallery entries
            for gallery_label, gallery_emb in gallery_embs.items():
                sim = float(np.dot(probe_emb, gallery_emb))
                is_genuine = int(label == gallery_label)
                labels_all.append(is_genuine)
                scores_all.append(sim)

    t_probe = time.perf_counter() - t0_probe

    # ── Metrics ───────────────────────────────────────────────────
    if not labels_all:
        return {"error": "Tidak ada pair yang dapat dievaluasi"}

    labels_arr = np.array(labels_all)
    scores_arr = np.array(scores_all)
    metrics    = compute_all_metrics(labels_arr, scores_arr)

    # Hapus confusion_matrix nested
    cm = metrics.pop("confusion_matrix", {})
    metrics.update({
        "n_enroll"           : n_enroll,
        "m_probe"            : m_probe,
        "fusion_strategy"    : fusion_strategy,
        "n_gallery_subjects" : len(gallery_embs),
        "n_probe_sessions"   : len(probe_embs),
        "n_pairs"            : len(labels_all),
        "latency_enroll_s"   : round(t_enroll, 4),
        "latency_probe_s"    : round(t_probe, 4),
        "gallery_embs"       : gallery_embs,
        "probe_embs"         : probe_embs,
    })
    return metrics


# ---------------------------------------------------------------------------
# Ablation: sweep (N, M) matrix
# ---------------------------------------------------------------------------

def eval_multiframe_ablation(
    model,
    enroll_session_splits: dict[str, list[Path]],
    probe_session_splits: dict[str, list[Path]],
    n_list: list[int] = (1, 3, 5, 10),
    m_list: list[int] = (1, 3, 5, 10),
    fusion_strategy: str = "mean",
    device: torch.device | None = None,
    n_points: int = 1024,
    normalizer=None,
    seed: int | None = None,
    verbose: bool = True,
    repr_mode: str = "canonical_npy",
    enc_batch_size: int = 128,
) -> dict:
    """
    Jalankan eval_multiframe untuk semua kombinasi (N, M) dalam n_list × m_list.
    enc_batch_size: batch encoding (forward) — naikkan utk GPU besar (G100/A100).

    Returns:
        {
            (n, m): metrics_dict,
            ...
        }
    """
    results = {}
    total = len(n_list) * len(m_list)
    count = 0

    # cache embedding per-frame: frame yang sama tak di-encode ulang lintas (N,M) → ~Nx lebih cepat.
    # Aman utk paritas: model/repr_mode/n_points/normalizer konstan di seluruh sweep ini.
    frame_cache: dict = {}

    for n in n_list:
        for m in m_list:
            count += 1
            if verbose:
                print(f"[Ablation {count}/{total}] N={n}, M={m}, fusion={fusion_strategy}...")

            res = eval_multiframe(
                model=model,
                enroll_session_splits=enroll_session_splits,
                probe_session_splits=probe_session_splits,
                n_enroll=n,
                m_probe=m,
                fusion_strategy=fusion_strategy,
                device=device,
                n_points=n_points,
                normalizer=normalizer,
                seed=seed,
                repr_mode=repr_mode,
                frame_cache=frame_cache,
                batch_size=enc_batch_size,
            )
            results[(n, m)] = res

            if verbose:
                eer = res.get("eer", float("nan"))
                auc = res.get("auc", float("nan"))
                dp  = res.get("dprime", float("nan"))
                lat = res.get("latency_probe_s", float("nan"))
                print(f"  EER={eer:.4f}  AUC={auc:.4f}  d'={dp:.3f}  lat={lat:.3f}s")

    return results


def ablation_to_dataframe(ablation_results: dict) -> "pd.DataFrame":
    """
    Konversi hasil ablation dict ke pandas DataFrame.
    Berguna untuk heatmap visualisasi.
    """
    import pandas as pd
    rows = []
    for (n, m), res in ablation_results.items():
        rows.append({
            "n_enroll"         : n,
            "m_probe"          : m,
            "eer"              : res.get("eer"),
            "auc"              : res.get("auc"),
            "dprime"           : res.get("dprime"),
            "tar_at_far1"      : res.get("tar_at_far1"),
            "accuracy_at_eer"  : res.get("accuracy_at_eer"),
            "latency_probe_s"  : res.get("latency_probe_s"),
            "fusion_strategy"  : res.get("fusion_strategy"),
        })
    return pd.DataFrame(rows)


def fusion_strategy_ablation(
    model,
    enroll_session_splits: dict[str, list[Path]],
    probe_session_splits: dict[str, list[Path]],
    n_enroll: int = 5,
    m_probe: int = 5,
    strategies: list[str] = ("mean", "median", "max", "first"),
    device: torch.device | None = None,
    n_points: int = 1024,
    normalizer=None,
    seed: int | None = None,
    verbose: bool = True,
    repr_mode: str = "canonical_npy",
) -> dict:
    """
    Bandingkan semua fusion strategies pada konfigurasi (n_enroll, m_probe) tetap.

    Returns:
        {strategy: metrics_dict}
    """
    results = {}
    for strategy in strategies:
        if verbose:
            print(f"[FusionAblation] strategy={strategy}, N={n_enroll}, M={m_probe}...")
        res = eval_multiframe(
            model=model,
            enroll_session_splits=enroll_session_splits,
            probe_session_splits=probe_session_splits,
            n_enroll=n_enroll,
            m_probe=m_probe,
            fusion_strategy=strategy,
            device=device,
            n_points=n_points,
            normalizer=normalizer,
            seed=seed,
            repr_mode=repr_mode,
        )
        results[strategy] = res
        if verbose:
            eer = res.get("eer", float("nan"))
            dp  = res.get("dprime", float("nan"))
            print(f"  EER={eer:.4f}  d'={dp:.3f}")
    return results
