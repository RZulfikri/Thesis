"""
evaluate.py — Evaluasi lengkap model ablation M1–M4.

Memuat checkpoint untuk setiap varian model, menghitung metrik biometrik lengkap
(EER, AUC, TAR@FAR=1%, TAR@FAR=0.1%, d-prime, Accuracy@EER, FAR@EER, FRR@EER),
mencetak tabel hasil, dan menghasilkan plot ROC, DET, similarity distribution, t-SNE.

Mendukung dataset balancing (--balance) untuk evaluasi yang adil antar subjek.

Usage:
    python evaluate.py \\
        --data_dir dataset \\
        --checkpoints M1=runs/m1/fold_0/best.pth \\
                      M4=runs/m4/fold_0/best.pth \\
        --output_dir eval_results \\
        --balance
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from models.siamese import SiamesePalmNet
from utils.dataset import (
    GEOMETRY_DIM,
    PalmPairDataset,
    scan_dataset,
    scan_dataset_frames,
    balance_label_sessions,
    balance_label_frames,
    split_holdout_sessions,
)
from utils.dataset_lowdata import build_lowdata_splits
from utils.metrics import (
    compute_all_metrics,
    fig_to_tensor,
    plot_det,
    plot_roc,
    plot_similarity_dist,
    plot_tsne,
    print_metrics_table,
)
from utils.normalizer import GeometryNormalizer


def _auto_config():
    """
    Deteksi GPU memory, CPU cores, dan system RAM → return config optimal.
    Sama persis dengan train.ipynb / evaluate.ipynb.
    """
    import os, subprocess
    try:
        n_cpu = os.cpu_count() or 2
    except Exception:
        n_cpu = 2
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    sys_ram_gb = int(line.split()[1]) / (1024 ** 2)
                    break
            else:
                sys_ram_gb = 16
    except Exception:
        sys_ram_gb = 16
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            text=True
        )
        vram_mb = int(out.strip().split('\n')[0])
    except Exception:
        vram_mb = 0
    vram_gb = vram_mb / 1024

    # ── GPU name & compute capability detection ──
    try:
        out_name = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
            text=True
        )
        gpu_name = out_name.strip().split('\n')[0].strip()
    except Exception:
        gpu_name = "Unknown"

    try:
        if torch.cuda.is_available():
            cc_major, cc_minor = torch.cuda.get_device_capability()
            compute_capability = f"{cc_major}.{cc_minor}"
        else:
            compute_capability = "N/A"
    except Exception:
        compute_capability = "N/A"

    is_h100 = "H100" in gpu_name

    if vram_gb >= 90 or (is_h100 and vram_gb >= 85):
        # G4 96GB / A100 96GB / H100 NVL 94GB / Blackwell RTX PRO 6000 96GB
        # bs=512/n_pts=12288 risiko OOM saat embedding cache eval dataset; 256/8192 aman.
        bs, nw = 256, min(8, n_cpu)
        n_pts = 8192
        label = f'96GB class ({gpu_name}, CC={compute_capability})'
    elif is_h100 and vram_gb >= 75:
        # H100 80GB — compute capability 9.0, lebih kencang dari A100
        bs, nw = 512, min(8, n_cpu)
        n_pts = 12288
        label = f'H100 80GB class ({gpu_name}, CC={compute_capability})'
    elif vram_gb >= 75:        # A100 80GB
        bs, nw = 512, min(8, n_cpu)
        n_pts = 8192
        label = f'A100 80GB class ({gpu_name}, CC={compute_capability})'
    elif vram_gb >= 35:        # A100 40GB
        bs, nw = 384, min(8, n_cpu)
        n_pts = 8192
        label = 'A100 40GB'
    elif vram_mb > 0:          # T4 / L4 / V100 16-24GB
        bs, nw = 128, min(2, n_cpu)
        n_pts = 4096
        label = 'T4/L4/V100 class'
    else:                      # CPU
        bs, nw = 32, 0
        n_pts = 2048
        label = 'CPU'

    print(f'[Auto-config] {label} | VRAM={vram_gb:.1f}GB | CPU={n_cpu} core | RAM={sys_ram_gb:.1f}GB')
    print(f'              batch_size={bs}, num_workers={nw}, n_points={n_pts}')
    return bs, nw, n_pts


def parse_args():
    # Defaults auto-detected dari hardware; bisa di-override via CLI
    _bs, _nw, _np = _auto_config()

    p = argparse.ArgumentParser(description="Evaluasi model palm recognition — metrik lengkap")
    p.add_argument("--data_dir", default="dataset",
                   help="Dataset root (default: dataset)")
    p.add_argument(
        "--checkpoints", nargs="+", metavar="NAME=PATH",
        help='Path checkpoint, misal: M4=runs/m4/fold_0/best.pth',
        default=[],
    )
    p.add_argument(
        "--normalizer", default=None,
        help="Path ke normalizer.json (default: cari otomatis di sebelah checkpoint)",
    )
    p.add_argument("--output_dir",  default="eval_results")
    p.add_argument("--geom_dim",    type=int, default=GEOMETRY_DIM,
                   help=f"Dimensi fitur geometri (default: {GEOMETRY_DIM}, otomatis dari dataset.py)")
    p.add_argument("--n_points",    type=int, default=_np,
                   help=f"Jumlah titik yang di-sample dari full cloud (default: {_np}, auto-detect)")
    p.add_argument("--sampling",    default="random", choices=["random", "fps"])
    p.add_argument("--batch_size",  type=int, default=_bs,
                   help=f"Batch size inference (default: {_bs}, auto-detect)")
    p.add_argument("--num_workers", type=int, default=_nw,
                   help=f"Jumlah DataLoader workers (default: {_nw}, auto-detect)")
    p.add_argument("--balance",     action="store_true",
                   help="Cap setiap label ke jumlah sesi minimum sebelum evaluasi")
    p.add_argument("--save_scores", action="store_true",
                   help="Simpan labels dan scores ke scores.npz per model")
    p.add_argument("--holdout", action="store_true",
                   help="Evaluasi pada holdout probes (session yang tidak pernah dilihat training)")
    p.add_argument("--n_holdout_sessions", type=int, default=1,
                   help="Jumlah sesi per subjek yang di-hold-out (default: 1)")
    p.add_argument("--n_probe_frames", type=int, default=3,
                   help="Jumlah frame probe per subjek dari sesi holdout (default: 3)")
    p.add_argument("--holdout_seed", type=int, default=42,
                   help="Seed untuk split holdout (HARUS sama dengan training, default: 42)")
    # ---- Ablasi GeoAtt (sama dengan train.py) ----
    p.add_argument("--use-gam", dest="use_gam", action="store_true",
                   help="Aktifkan Geometric Attention Module (GAM1+GAM2)")
    p.add_argument("--use-geom-fusion", dest="use_geom_fusion", action="store_true",
                   help="Aktifkan concat geom_emb ke fusion head")
    p.add_argument("--use-geom", dest="use_geom", action="store_true",
                   help="Shortcut: aktifkan --use-gam DAN --use-geom-fusion (baseline with_geom)")
    p.add_argument("--no-geom", action="store_true",
                   help="Nonaktifkan geometric attention branch (ablation no-geom)")
    p.add_argument("--log_tensorboard", action="store_true",
                   help="Log evaluasi ke TensorBoard (output_dir/tensorboard)")
    p.add_argument("--frames-per-session", choices=["1", "all"], default="all",
                   help="v5.0.0: '1' = low-data regime (eval pada test/holdout dari splits.json); "
                        "'all' = semua frame (default)")
    p.add_argument("--eval-split", choices=["test", "holdout", "all"], default="all",
                   help="v5.0.0: Split yang dievaluasi jika frames-per-session=1")
    p.add_argument("--repr-mode", choices=["canonical_npy", "fps_npy", "raw_ply"],
                   default="canonical_npy",
                   help="v7.2.0: sumber point cloud (R2 cnn_input.npy / R3 cnn_input_fps.npy / "
                        "R1 output.ply). Harus sama dengan repr_mode saat training.")
    return p.parse_args()


def _is_frame_layout(data_dir: Path) -> bool:
    for label_dir in data_dir.iterdir():
        if not label_dir.is_dir():
            continue
        for ts_dir in label_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            if any(ts_dir.glob("frame_*")):
                return True
    return False


def load_model(ckpt_path: str, geom_dim: int, device: torch.device,
               use_geom: bool = True, use_gam: bool | None = None,
               use_geom_fusion: bool | None = None) -> SiamesePalmNet:
    """Load checkpoint .pth — kompatibel dengan torch.compile dan checkpoint pre-v0.3.0
    yang menggunakan satu `proj` head (di-rename ke proj_with_geom/proj_no_geom)."""
    model = SiamesePalmNet(
        geom_dim=geom_dim,
        use_geom=use_geom,
        use_gam=use_gam,
        use_geom_fusion=use_geom_fusion,
    ).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    # Migrate legacy `encoder.proj.*` → `encoder.proj_with_geom.*` (jika use_geom_fusion aktif)
    # atau `encoder.proj_no_geom.*` (jika tidak). Hanya rename, bobot tetap sama.
    if any(k.startswith("encoder.proj.") for k in state.keys()):
        target = "proj_with_geom" if model.encoder.use_geom_fusion else "proj_no_geom"
        migrated = {}
        for k, v in state.items():
            if k.startswith("encoder.proj."):
                migrated[k.replace("encoder.proj.", f"encoder.{target}.", 1)] = v
            else:
                migrated[k] = v
        state = migrated
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: SiamesePalmNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Jalankan model pada semua pasangan di loader.
    Gunakan run_inference_cached untuk dataset besar — jauh lebih cepat.
    """
    all_labels, all_scores, all_embs_a, all_embs_b = [], [], [], []

    pbar = tqdm(loader, desc="Inference", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        pts_a  = batch["pts_a"].to(device, non_blocking=True)
        geom_a = batch["geom_a"].to(device, non_blocking=True)
        pts_b  = batch["pts_b"].to(device, non_blocking=True)
        geom_b = batch["geom_b"].to(device, non_blocking=True)
        label  = batch["label"].numpy()
        emb_a, emb_b, sim = model(pts_a, geom_a, pts_b, geom_b)
        all_labels.append(label)
        all_scores.append(sim.cpu().numpy())
        all_embs_a.append(emb_a.cpu().numpy())
        all_embs_b.append(emb_b.cpu().numpy())

    return (np.concatenate(all_labels), np.concatenate(all_scores),
            np.concatenate(all_embs_a), np.concatenate(all_embs_b))


@torch.no_grad()
def run_inference_cached(
    model: SiamesePalmNet,
    dataset,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hitung embedding sekali per sesi unik, lalu rekonstruksi skor semua pasangan
    secara vektorisasi. ~100–200x lebih cepat dari run_inference.

    Misal: 450 sesi unik → 450 forward pass encoder (bukan 116.760).
    """
    unique_dirs = list({p[0] for p in dataset.pairs} | {p[1] for p in dataset.pairs})
    n_unique    = len(unique_dirs)
    rng         = np.random.default_rng(42)  # seed tetap → eval deterministik
    n_pts       = dataset.n_points

    dir_to_emb: dict = {}
    pbar = tqdm(range(0, n_unique, batch_size),
                desc=f"Embedding {n_unique} sesi", unit="batch", dynamic_ncols=True)

    for start in pbar:
        batch_dirs          = unique_dirs[start:start + batch_size]
        pts_list, geom_list = [], []

        for d in batch_dirs:
            cloud, geom = dataset._cache[d]
            idx  = rng.choice(len(cloud), size=n_pts, replace=len(cloud) < n_pts)
            pts  = cloud[idx].astype(np.float32)
            geom = geom.copy().astype(np.float32)
            if dataset.normalizer is not None:
                geom = dataset.normalizer.transform(geom)
            pts_list.append(pts)
            geom_list.append(geom)

        pts_t  = torch.from_numpy(np.stack(pts_list)).to(device, non_blocking=True)
        geom_t = torch.from_numpy(np.stack(geom_list)).to(device, non_blocking=True)
        embs   = model.encoder(pts_t, geom_t)  # (B, 128) L2-normed

        for d, emb in zip(batch_dirs, embs.cpu().numpy()):
            dir_to_emb[d] = emb

    # Rekonstruksi skor semua pasangan secara vektorisasi — tanpa loop Python
    labels = np.array([p[2] for p in dataset.pairs], dtype=np.float32)
    ea     = np.stack([dir_to_emb[p[0]] for p in dataset.pairs])  # (N_pairs, 128)
    eb     = np.stack([dir_to_emb[p[1]] for p in dataset.pairs])  # (N_pairs, 128)
    scores = (ea * eb).sum(axis=1)                                  # cosine similarity

    return labels, scores, ea, eb, dir_to_emb


def _tb_log_confusion_matrix(writer, cm_dict, global_step, tag="confusion_matrix"):
    """Log 2×2 confusion matrix sebagai image ke TensorBoard."""
    import matplotlib.pyplot as plt
    tp, tn, fp, fn = cm_dict["tp"], cm_dict["tn"], cm_dict["fp"], cm_dict["fn"]
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred: Impostor", "Pred: Genuine"],
                yticklabels=["True: Impostor", "True: Genuine"],
                cbar=False, square=True)
    ax.set_title(f"{tag} — TP={tp} TN={tn} FP={fp} FN={fn}")
    fig.tight_layout()
    writer.add_image(tag, fig_to_tensor(fig), global_step, dataformats="HWC")
    plt.close(fig)


def _tb_log_identification_cm(writer, probe_results, gallery_labels, global_step, tag="identification_cm"):
    """Log N×N identification confusion matrix sebagai image ke TensorBoard."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    y_true = [r["true_label"] for r in probe_results]
    y_pred = [r["pred_label"] for r in probe_results]
    cm = confusion_matrix(y_true, y_pred, labels=gallery_labels)
    fig, ax = plt.subplots(figsize=(max(6, len(gallery_labels)), max(6, len(gallery_labels))))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=gallery_labels, yticklabels=gallery_labels,
                cbar=False, square=True)
    ax.set_xlabel("Predicted Identity")
    ax.set_ylabel("True Identity")
    ax.set_title(tag)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    writer.add_image(tag, fig_to_tensor(fig), global_step, dataformats="HWC")
    plt.close(fig)


def _tb_log_eval(writer, model, model_name, result, dir_to_emb, device, geom_dim, global_step=0):
    """
    Log evaluasi lengkap ke TensorBoard:
      - Scalar metrics
      - Similarity distribution image
      - Confusion matrix images
      - Embeddings (add_embedding dengan identity labels)
      - Model graph (add_graph)
    """
    if writer is None:
        return

    metrics = result

    # ── Scalars ──────────────────────────────────────────────────
    scalar_tags = [
        "eer", "auc", "tar_at_far1", "tar_at_far01", "dprime",
        "accuracy_at_eer", "far_at_eer", "frr_at_eer",
        "rank1", "rank5", "rank10", "map",
        "tp", "tn", "fp", "fn", "precision", "recall", "f1",
        "cm_threshold",
    ]
    for tag in scalar_tags:
        val = metrics.get(tag)
        if val is not None and isinstance(val, (int, float)):
            writer.add_scalar(f"eval/{tag}", val, global_step)

    # ── Similarity Distribution Image ────────────────────────────
    labels = metrics.get("labels")
    scores = metrics.get("scores")
    if labels is not None and scores is not None:
        import matplotlib.pyplot as plt
        from utils.metrics import compute_eer, compute_dprime
        eer, eer_thresh = compute_eer(labels, scores)
        dprime = compute_dprime(labels, scores)
        genuine = scores[labels == 1]
        impostor = scores[labels == 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(-1, 1, 40)
        ax.hist(genuine, bins=bins, alpha=0.6, color="green",
                label=f"Genuine  (n={len(genuine)})")
        ax.hist(impostor, bins=bins, alpha=0.6, color="red",
                label=f"Impostor (n={len(impostor)})")
        ax.axvline(eer_thresh, color="black", linestyle="--", lw=1.5,
                   label=f"EER thr={eer_thresh:.3f}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"{model_name} — Similarity  [EER={eer:.4f}  d'={dprime:.2f}]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        writer.add_image("eval/similarity_distribution", fig_to_tensor(fig), global_step, dataformats="HWC")
        plt.close(fig)

    # ── Binary Confusion Matrix Image ────────────────────────────
    cm_flat = {k: metrics.get(k) for k in ["tp", "tn", "fp", "fn"]}
    if all(v is not None for v in cm_flat.values()):
        _tb_log_confusion_matrix(writer, cm_flat, global_step, tag="eval/confusion_matrix")

    # ── Identification Confusion Matrix ──────────────────────────
    if "probe_results" in metrics and "gallery_labels" in metrics:
        probe_results = metrics["probe_results"]
        gallery_labels = metrics["gallery_labels"]
        if probe_results and gallery_labels:
            _tb_log_identification_cm(
                writer, probe_results, gallery_labels, global_step,
                tag="eval/identification_confusion_matrix",
            )

    # ── Embeddings (add_embedding) ───────────────────────────────
    if dir_to_emb is not None and len(dir_to_emb) > 0:
        unique_dirs = sorted(dir_to_emb.keys())
        emb_matrix = np.stack([dir_to_emb[d] for d in unique_dirs])
        emb_labels = [d.parent.parent.name for d in unique_dirs]
        writer.add_embedding(
            emb_matrix,
            metadata=emb_labels,
            tag=f"{model_name}_embeddings",
            global_step=global_step,
        )

    # ── Model Graph ──────────────────────────────────────────────
    try:
        # Trace encoder dengan dummy input
        dummy_pts = torch.randn(1, 2048, 6, device=device)
        dummy_geom = torch.randn(1, geom_dim, device=device)
        writer.add_graph(model.encoder, (dummy_pts, dummy_geom))
    except Exception as e:
        # Graph logging opsional — jangan crash evaluasi
        print(f"  [TensorBoard] Graph logging skipped: {e}")

    writer.flush()


def evaluate_model(
    model: SiamesePalmNet,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    output_dir: Path,
    save_scores: bool = False,
    dataset=None,
) -> dict:
    """
    Evaluasi satu model — hitung semua metrik dan buat plot similarity distribution.
    Jika dataset disertakan, gunakan run_inference_cached (~100x lebih cepat).
    """
    if dataset is not None:
        labels, scores, embs_a, embs_b, dir_to_emb = run_inference_cached(model, dataset, device)
    else:
        labels, scores, embs_a, embs_b = run_inference(model, loader, device)
        dir_to_emb = None

    metrics = compute_all_metrics(labels, scores)

    # Flatten confusion_matrix dict ke top-level untuk aggregation
    cm = metrics.pop("confusion_matrix", {})
    metrics["cm_threshold"] = cm.get("threshold")
    metrics["tp"] = cm.get("tp")
    metrics["tn"] = cm.get("tn")
    metrics["fp"] = cm.get("fp")
    metrics["fn"] = cm.get("fn")
    metrics["precision"] = cm.get("precision")
    metrics["recall"] = cm.get("recall")
    metrics["f1"] = cm.get("f1")

    # Identification (1:N) metrics — hanya kalau ada dir_to_emb
    probe_results = []
    gallery_labels = []
    if dir_to_emb is not None and len(dir_to_emb) > 0:
        from utils.metrics import evaluate_identification
        id_metrics = evaluate_identification(dir_to_emb, rank_n_list=[1, 5, 10])
        metrics.update({k: v for k, v in id_metrics.items() if k.startswith("rank") or k == "map"})
        probe_results = id_metrics.get("probe_results", [])
        gallery_labels = id_metrics.get("gallery_labels", [])
        # Simpan identification predictions ke JSON
        id_out = output_dir / f"{model_name}_identification.json"
        with open(id_out, "w") as f:
            json.dump({
                "gallery_labels": gallery_labels,
                "probe_results": probe_results,
            }, f, indent=2)

    plot_similarity_dist(
        labels, scores,
        title=f"{model_name} — Distribusi Similarity",
        save_path=output_dir / f"{model_name}_sim_dist.png",
    )

    if save_scores:
        np.savez(output_dir / f"{model_name}_scores.npz",
                 labels=labels, scores=scores)

    result = {"model": model_name, **metrics}
    result["labels"]     = labels
    result["scores"]     = scores
    # Gabung embs untuk t-SNE (hanya gunakan embs_a agar tidak double-count)
    result["embeddings"] = embs_a
    result["dir_to_emb"] = dir_to_emb
    result["probe_results"] = probe_results
    result["gallery_labels"] = gallery_labels
    return result


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir     = Path(args.data_dir)
    frame_layout = _is_frame_layout(data_dir)
    print(f"Layout data: {'frame' if frame_layout else 'session'}")

    # Load + balance dataset
    if getattr(args, "frames_per_session", "all") == "1":
        # v5.0.0: Low-data regime — load dari build_lowdata_splits
        print("[v5.0.0] Low-data regime: loading deterministic splits...")
        splits = build_lowdata_splits(data_dir)
        eval_split = getattr(args, "eval_split", "all")
        if eval_split == "test":
            label_sessions = splits["test"]
            print(f"[Eval] Test split: {sum(len(v) for v in label_sessions.values())} frames")
        elif eval_split == "holdout":
            label_sessions = splits["holdout"]
            print(f"[Eval] Holdout split: {sum(len(v) for v in label_sessions.values())} frames")
        else:
            # all: gabungkan test + holdout untuk evaluasi lengkap
            label_sessions = {}
            for label in splits["test"]:
                label_sessions[label] = splits["test"][label] + splits["holdout"].get(label, [])
            print(f"[Eval] Test+Holdout: {sum(len(v) for v in label_sessions.values())} frames")
    elif frame_layout:
        label_frames, session_groups = scan_dataset_frames(data_dir)
        if args.balance:
            session_groups, min_s = balance_label_frames(session_groups)
            label_frames = {
                label: [f for ts_frames in ts_dict.values() for f in ts_frames]
                for label, ts_dict in session_groups.items()
            }
            print(f"[Balance] Setiap label = {min_s} sesi")
        if args.holdout:
            _, holdout_probes = split_holdout_sessions(
                session_groups,
                n_holdout_sessions=args.n_holdout_sessions,
                n_probe_frames=args.n_probe_frames,
                seed=args.holdout_seed,
            )
            label_sessions = holdout_probes
            n_probes = sum(len(v) for v in holdout_probes.values())
            print(f"[Holdout] {n_probes} probe frames dari {len(holdout_probes)} subjek "
                  f"(n_holdout_sessions={args.n_holdout_sessions}, n_probe_frames={args.n_probe_frames}, seed={args.holdout_seed})")
        else:
            label_sessions = label_frames
    else:
        if args.holdout:
            sys.exit("--holdout hanya didukung untuk frame-layout dataset (folder frame_*).")
        label_sessions = scan_dataset(data_dir)
        if args.balance:
            label_sessions, min_s = balance_label_sessions(label_sessions)
            print(f"[Balance] Setiap label = {min_s} sesi")

    n_sessions_total = sum(len(v) for v in label_sessions.values())
    print(f"Label: {len(label_sessions)}  Total: {n_sessions_total}")

    # Parse checkpoints: "NAME=path/to/best.pth"
    checkpoints = {}
    for item in args.checkpoints:
        if "=" not in item:
            sys.exit(f"Format checkpoint tidak valid '{item}'. Gunakan NAME=path.")
        name, path = item.split("=", 1)
        checkpoints[name] = path

    if not checkpoints:
        print("Tidak ada checkpoint. Evaluasi dengan model bobot acak.")
        checkpoints["untrained"] = None

    # Load normalizer (cari otomatis di sebelah checkpoint pertama)
    normalizer_path = args.normalizer
    if normalizer_path is None and checkpoints:
        first_ckpt = next(iter(checkpoints.values()))
        if first_ckpt:
            candidate = Path(first_ckpt).parent / "normalizer.json"
            if candidate.exists():
                normalizer_path = str(candidate)

    normalizer = None
    if normalizer_path and Path(normalizer_path).exists():
        normalizer = GeometryNormalizer.load(normalizer_path)
        print(f"Loaded normalizer dari {normalizer_path}")

    dataset = PalmPairDataset(
        label_sessions=label_sessions,
        n_points=args.n_points,
        sampling=args.sampling,
        augment=None,
        normalizer=normalizer,
        repr_mode=getattr(args, "repr_mode", "canonical_npy"),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    all_results = []
    roc_data    = {}
    det_data    = {}

    # TensorBoard writer
    writer = None
    if getattr(args, "log_tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[TensorBoard] Logging ke {tb_dir}")

    # Resolve GeoAtt flags (sama dengan train.py)
    use_gam = bool(getattr(args, "use_gam", False) or getattr(args, "use_geom", False))
    use_geom_fusion = bool(getattr(args, "use_geom_fusion", False) or getattr(args, "use_geom", False))
    # --no-geom overrides everything
    if getattr(args, "no_geom", False):
        use_gam = False
        use_geom_fusion = False

    for name, ckpt_path in checkpoints.items():
        print(f"\nEvaluasi {name}...")
        
        # Auto-detect variant from config.json (prevents human error)
        ckpt_use_gam = use_gam
        ckpt_use_geom_fusion = use_geom_fusion
        ckpt_variant = None
        
        if ckpt_path and Path(ckpt_path).exists():
            config_path = Path(ckpt_path).parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    ckpt_config = json.load(f)
                ckpt_use_gam = ckpt_config.get("use_gam", use_gam)
                ckpt_use_geom_fusion = ckpt_config.get("use_geom_fusion", use_geom_fusion)
                ckpt_variant = ckpt_config.get("variant", None)
                
                # Warn if CLI flags mismatch config
                if (use_gam != ckpt_use_gam or use_geom_fusion != ckpt_use_geom_fusion):
                    print(f"  ⚠️  WARNING: CLI flags mismatch config.json!")
                    print(f"     CLI: use_gam={use_gam}, use_geom_fusion={use_geom_fusion}")
                    print(f"     Config: use_gam={ckpt_use_gam}, use_geom_fusion={ckpt_use_geom_fusion}")
                    print(f"     → Using config.json values (override CLI)")
        
        if ckpt_path and Path(ckpt_path).exists():
            model = load_model(ckpt_path, args.geom_dim, device,
                              use_geom=(ckpt_use_gam or ckpt_use_geom_fusion),
                              use_gam=ckpt_use_gam, use_geom_fusion=ckpt_use_geom_fusion)
            if ckpt_variant:
                print(f"  [Config] Variant: {ckpt_variant}")
        else:
            print(f"  Checkpoint tidak ditemukan, gunakan bobot acak untuk {name}")
            model = SiamesePalmNet(geom_dim=args.geom_dim,
                                  use_geom=(ckpt_use_gam or ckpt_use_geom_fusion),
                                  use_gam=ckpt_use_gam, use_geom_fusion=ckpt_use_geom_fusion).to(device)
            model.eval()

        result = evaluate_model(
            model, loader, device, name, output_dir,
            save_scores=args.save_scores,
            dataset=dataset,
        )
        all_results.append(result)
        roc_data[name] = (result["labels"], result["scores"])
        det_data[name] = (result["labels"], result["scores"])

        print(
            f"  EER={result['eer']:.4f}  AUC={result['auc']:.4f}  "
            f"TAR@FAR1%={result['tar_at_far1']:.4f}  "
            f"TAR@FAR0.1%={result['tar_at_far01']:.4f}  "
            f"d'={result['dprime']:.3f}"
        )

        # Log ke TensorBoard
        if writer is not None:
            _tb_log_eval(
                writer, model, name, result,
                result.get("dir_to_emb"), device,
                geom_dim=args.geom_dim,
                global_step=0,
            )

    # Tabel lengkap
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print_metrics_table(all_results)

    # Simpan hasil ke JSON
    summary = []
    for r in all_results:
        s = {k: v for k, v in r.items()
             if k not in ("labels", "scores", "embeddings", "dir_to_emb", "probe_results", "gallery_labels")}
        # Enrich with variant & seed from config.json for aggregation
        ckpt_path = checkpoints.get(r.get("model", ""), None)
        if ckpt_path and Path(ckpt_path).exists():
            config_path = Path(ckpt_path).parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                s["variant"] = cfg.get("variant", r.get("model", "unknown"))
                s["seed"] = cfg.get("seed", None)
        summary.append(s)
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nHasil disimpan di {output_dir}/results.json")

    # Plot ROC
    if roc_data:
        plot_roc(roc_data, save_path=output_dir / "roc_curves.png")

    # Plot DET
    if det_data:
        plot_det(det_data, save_path=output_dir / "det_curves.png")

    # t-SNE untuk model terakhir — gunakan identity labels dari dir_to_emb
    if all_results:
        last = all_results[-1]
        dir_to_emb_last = last.get("dir_to_emb")
        if dir_to_emb_last and len(dir_to_emb_last) > 0:
            # Ekstrak identity labels dari path: .../[label]/[timestamp]/frame_NN/
            sorted_dirs = sorted(dir_to_emb_last.keys())
            tsne_embs = np.stack([dir_to_emb_last[d] for d in sorted_dirs])
            tsne_labels = np.array([d.parent.parent.name for d in sorted_dirs])
            unique_labels = sorted(set(tsne_labels))
            label_to_int = {l: i for i, l in enumerate(unique_labels)}
            tsne_int_labels = np.array([label_to_int[l] for l in tsne_labels])
            plot_tsne(
                tsne_embs,
                tsne_int_labels,
                title=f"{last['model']} — Embedding Space (t-SNE)",
                save_path=output_dir / f"{last['model']}_tsne.png",
            )
        else:
            # Fallback: pair embeddings tanpa identity info
            plot_tsne(
                last["embeddings"],
                np.zeros(len(last["embeddings"]), dtype=int),
                title=f"{last['model']} — Embedding Space (t-SNE)",
                save_path=output_dir / f"{last['model']}_tsne.png",
            )

        # Confusion matrix (N×N identification) dari probe_results
        probe_results_last = last.get("probe_results", [])
        gallery_labels_last = last.get("gallery_labels", [])
        if probe_results_last and gallery_labels_last:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            y_true = [r["true_label"] for r in probe_results_last]
            y_pred = [r["pred_label"] for r in probe_results_last]
            cm = confusion_matrix(y_true, y_pred, labels=gallery_labels_last)
            acc = np.trace(cm) / cm.sum() * 100 if cm.sum() > 0 else 0.0

            fig_cm, ax_cm = plt.subplots(
                figsize=(max(7, len(gallery_labels_last) * 0.8),
                         max(6, len(gallery_labels_last) * 0.7))
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=gallery_labels_last
            )
            disp.plot(ax=ax_cm, cmap="Blues", values_format="d",
                      colorbar=True, xticks_rotation=45)
            ax_cm.set_title(
                f"{last['model']} — Identification Confusion Matrix\n"
                f"(Rank-1 acc={acc:.1f}%, {len(probe_results_last)} probes)"
            )
            fig_cm.tight_layout()
            cm_path = output_dir / f"{last['model']}_confusion_matrix.png"
            fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig_cm)
            print(f"Confusion matrix saved to {cm_path}")

    if writer is not None:
        writer.close()
        print(f"[TensorBoard] Writer closed.")

    print(f"\nSemua output di: {output_dir}/")


if __name__ == "__main__":
    main()
