"""
train.py — Training script untuk GeoAtt-PointNet++ palm recognition.

Mendukung dua layout data:
  Session layout (ICP, lama):
    python train.py --data_dir dataset
  Frame layout (single-frame, baru):
    python train.py --data_dir dataset

Layout terdeteksi otomatis berdasarkan ada tidaknya subdirektori frame_*.

Fitur:
  - Early stopping: hentikan training jika val_loss tidak membaik selama N epoch
  - Two-phase training: Phase 1 (main) + Phase 2 (fine-tune dengan LR lebih kecil)
  - Dataset balancing: cap setiap label ke jumlah sesi minimum untuk hasil yang adil

Usage (lokal):
    python train.py --data_dir dataset --output_dir runs/exp1

Usage dengan semua fitur:
    python train.py --data_dir dataset --output_dir runs/exp1 \\
                    --patience 15 --finetune_epochs 20 --balance

Usage (Colab — lihat notebook 01_train.ipynb untuk setup lengkap):
    python train.py --data_dir /content/drive/MyDrive/palm_poc/dataset \\
                    --output_dir /content/drive/MyDrive/palm_poc/runs/exp1
"""

import argparse
import json

from tqdm import tqdm
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

sys.path.insert(0, str(Path(__file__).parent))

from losses.contrastive import ContrastiveLoss
from losses.triplet import OnlineTripletLoss
from models.siamese import SiamesePalmNet
from utils.augmentation import PointCloudAugmentor, GeometryAugmentor
from utils.dataset import (
    GEOMETRY_DIM,
    PalmPairDataset, PalmFrameDataset, load_geometry,
    make_loso_splits, scan_dataset,
    make_loso_splits_frames, scan_dataset_frames,
    balance_label_sessions, balance_label_frames,
    split_holdout_sessions, split_sessions_three_way,
)
from utils.dataset_lowdata import build_lowdata_splits
from utils.normalizer import GeometryNormalizer
from utils.val_pair_metric import ValPairMetric


def _seed_worker(worker_id):
    """Independent RNG per DataLoader worker."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    import random
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class _MarginCECriterion:
    """v8: criterion cross-entropy untuk semua margin head (margin sudah diterapkan di model head).
    Antarmuka `.compute_loss(logits, labels)` agar kompatibel dgn call site lama."""
    def compute_loss(self, logits, labels):
        import torch.nn.functional as F
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Hentikan training jika val_loss tidak membaik selama `patience` epoch.

    Menyimpan bobot model terbaik dan memulihkannya saat stopped.

    Args:
        patience  : jumlah epoch tanpa improvement sebelum berhenti
        min_delta : perubahan minimum yang dianggap sebagai improvement
        verbose   : cetak pesan saat improvement atau stop
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, verbose: bool = True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: dict | None = None
        self.stopped_epoch: int = 0

    def step(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """
        Periksa apakah training harus dihentikan.

        Returns:
            True jika harus stop, False jika lanjut.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Simpan salinan bobot terbaik di CPU agar tidak memakan GPU memory
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"    [EarlyStopping] Improved → val_loss={val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    [EarlyStopping] No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

    def restore_best(self, model: torch.nn.Module, device: torch.device) -> None:
        """Muat kembali bobot terbaik ke model."""
        if self.best_state is not None:
            state = {k: v.to(device) for k, v in self.best_state.items()}
            model.load_state_dict(state)
            if self.verbose:
                print(f"    [EarlyStopping] Bobot terbaik dipulihkan "
                      f"(val_loss={self.best_loss:.4f})")


def _is_frame_layout(data_dir: Path) -> bool:
    """Deteksi otomatis: True jika ditemukan subdirektori frame_* di bawah timestamp."""
    for label_dir in data_dir.iterdir():
        if not label_dir.is_dir():
            continue
        for ts_dir in label_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            if any(ts_dir.glob("frame_*")):
                return True
    return False


# Cache agar _auto_config tidak double-print saat dipanggil dari parse_args + clamp
_AUTO_CONFIG_CACHE = None


def _auto_config():
    """
    Deteksi GPU memory, CPU cores, dan system RAM → return config optimal.
    Sama persis dengan train.ipynb / evaluate.ipynb.
    """
    global _AUTO_CONFIG_CACHE
    if _AUTO_CONFIG_CACHE is not None:
        return _AUTO_CONFIG_CACHE

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
    is_a100 = "A100" in gpu_name

    if vram_gb >= 90 or (is_h100 and vram_gb >= 85):
        # G4 96GB / A100 96GB / H100 NVL 94GB / Blackwell RTX PRO 6000 96GB
        # Empirical (May 18): bs=384/n_pts=12288 OOM di ball_query argsort (20 GiB
        # scratch tunggal + Siamese 2× branch saturasi 85 GB). 256/8192 ≈ 50 GB
        # peak — aman untuk no_geom dan with_geom.
        bs, nw, lr, flr = 1536, min(8, n_cpu), 2e-3, 2e-4   # v8 AGRESIF: BS 1536 (GPU 95GB ~30GB); lr di-override notebook
        n_pts = 8192
        label = f'96GB class ({gpu_name}, CC={compute_capability})'
    elif is_h100 and vram_gb >= 75:
        # H100 80GB — compute capability 9.0, lebih kencang dari A100
        # Bisa lebih agresif: naikkan bs 50% vs A100 80GB
        bs, nw, lr, flr = 1536, min(8, n_cpu), 2e-3, 2e-4   # v8 AGRESIF: BS 1536
        n_pts = 8192
        label = f'H100 80GB class ({gpu_name}, CC={compute_capability})'
    elif vram_gb >= 75:        # A100 80GB
        # bs=512/n_pts=8192 OOM di ball_query untuk with_geom.
        # 256/8192 ≈ 70-80 GB peak, aman untuk semua variant.
        bs, nw, lr, flr = 1536, min(8, n_cpu), 2e-3, 2e-4   # v8 AGRESIF: BS 1536 (GPU 95GB ~30GB); lr di-override notebook
        n_pts = 8192
        label = f'A100 80GB class ({gpu_name}, CC={compute_capability})'
    elif vram_gb >= 35:        # A100 40GB
        bs, nw, lr, flr = 768, min(8, n_cpu), 2e-3, 2e-4   # v8: BS fixed 768
        n_pts = 8192
        label = 'A100 40GB'
    elif vram_gb >= 20:        # L4 24GB — n_points=8192 MUAT; v8 BS fixed 768 (~15GB, no_geom)
        bs, nw, lr, flr = 768, min(8, n_cpu), 1e-3, 1e-4   # jangan clamp ke 192/4096 (rusak protokol)
        n_pts = 8192
        label = 'L4 24GB class'
    elif vram_mb > 0:          # T4 / V100 16GB
        bs, nw, lr, flr = 128, min(2, n_cpu), 1e-3, 1e-4
        n_pts = 4096
        label = 'T4/V100 16GB class'
    else:                      # CPU
        bs, nw, lr, flr = 32, 0, 1e-3, 1e-4
        n_pts = 2048
        label = 'CPU'

    print(f'[Auto-config] {label} | VRAM={vram_gb:.1f}GB | CPU={n_cpu} core | RAM={sys_ram_gb:.1f}GB')
    print(f'              batch_size={bs}, num_workers={nw}, n_points={n_pts}, lr={lr}, finetune_lr={flr}')
    _AUTO_CONFIG_CACHE = (bs, nw, lr, flr, n_pts)
    return _AUTO_CONFIG_CACHE


def parse_args():
    # Defaults auto-detected dari hardware; bisa di-override via CLI
    _bs, _nw, _lr, _flr, _np = _auto_config()

    p = argparse.ArgumentParser(description="Train GeoAtt-PointNet++ palm recognition")
    p.add_argument("--data_dir",    default="dataset",
                   help="Dataset root — session layout: [label]/[timestamp]/ "
                        "atau frame layout: [label]/[timestamp]/frame_*/  (default: dataset)")
    p.add_argument("--output_dir",  default="runs/exp1", help="Direktori untuk checkpoint dan log")
    p.add_argument("--epochs",      type=int,   default=100,
                   help="Jumlah epoch untuk Phase 1 (main training, default: 100)")
    p.add_argument("--batch_size",  type=int,   default=_bs,
                   help=f"Batch size training (default: {_bs}, auto-detect)")
    p.add_argument("--lr",          type=float, default=_lr,
                   help=f"Learning rate Phase 1 (default: {_lr}, auto-detect)")
    p.add_argument("--margin",      type=float, default=0.5)
    p.add_argument("--geom_dim",    type=int,   default=GEOMETRY_DIM,
                   help=f"Dimensi fitur geometri (default: {GEOMETRY_DIM}, otomatis dari dataset.py)")
    p.add_argument("--n_points",    type=int,   default=_np,
                   help=f"Jumlah titik yang di-sample dari full cloud per item (default: {_np}, auto-detect)")
    p.add_argument("--sampling",    default="random", choices=["random", "fps"],
                   help="Metode sampling: random (default) atau fps (backup novelty)")
    p.add_argument("--num_workers", type=int,   default=_nw,
                   help=f"Jumlah DataLoader workers (default: {_nw}, auto-detect)")
    p.add_argument("--checkpoint",  default=None, help="Path ke .pth checkpoint untuk resume")
    p.add_argument("--fold",        type=int,   default=0, help="LOSO fold index (0-indexed)")
    p.add_argument("--all_folds",   action="store_true", help="Train semua LOSO fold secara berurutan")
    # Fixed split (new pipeline)
    p.add_argument("--fixed_split", action="store_true",
                   help="Gunakan fixed train/val/test split (bukan LOSO)")
    p.add_argument("--split_seed",  type=int,   default=42,
                   help="Seed untuk pembagian data fixed split (default: 42)")
    p.add_argument("--seed",        type=int,   default=None,
                   help="Seed untuk PyTorch, NumPy, dan Python random. Kalau None, gunakan split_seed (default: None)")
    p.add_argument("--holdout_sessions", type=int, default=1,
                   help="Jumlah sesi per subjek yang di-hold-out untuk real test (default: 1)")
    p.add_argument("--holdout_frames",   type=int, default=3,
                   help="Jumlah frame probe dari sesi holdout (default: 3)")
    p.add_argument("--train_ratio", type=float, default=0.70,
                   help="Ratio sesi untuk train setelah holdout (default: 0.70)")
    p.add_argument("--val_ratio",   type=float, default=0.15,
                   help="Ratio sesi untuk val setelah holdout (default: 0.15)")
    # Early stopping
    p.add_argument("--patience",    type=int,   default=15,
                   help="Jumlah epoch tanpa improvement sebelum early stopping (default: 15)")
    p.add_argument("--min_delta",   type=float, default=1e-4,
                   help="Minimum improvement yang dianggap valid (default: 1e-4)")
    # Fine-tuning phase
    p.add_argument("--finetune_epochs", type=int,   default=20,
                   help="Jumlah epoch fine-tuning Phase 2 (0 = skip fine-tuning, default: 20)")
    p.add_argument("--finetune_lr",     type=float, default=_flr,
                   help=f"Learning rate Phase 2 fine-tuning (default: {_flr}, auto-detect)")
    # Dataset balancing
    p.add_argument("--balance",     action="store_true",
                   help="Cap setiap label ke jumlah sesi minimum untuk hasil yang adil")
    # ---- v5.0.0 Low-Data Regime ----
    p.add_argument("--frames-per-session", choices=["1", "all"], default="all",
                   help="v5.0.0: '1' = one median frame per session (low-data); "
                        "'all' = semua frame (all-frame regime, default)")
    p.add_argument("--loss", choices=["triplet", "contrastive", "arcface", "hybrid",
                                     "cosface", "subcenter_arcface",
                                     # v8: head margin terpadu (losses/margin_heads.py)
                                     "arcface_true", "adacos", "curricularface", "qa_arcface"],
                   default="contrastive",
                   help="v5.0.0+: Loss function. 'arcface' (v6, linear cosθ−m), "
                        "v8: 'arcface_true' (cos(θ+m) sejati), 'cosface', 'subcenter_arcface', "
                        "'adacos', 'curricularface', 'qa_arcface' (quality-adaptive); "
                        "'triplet'/'hybrid'/'contrastive'")
    p.add_argument("--triplet-margin", type=float, default=0.3,
                   help="v5.0.0: Margin untuk OnlineTripletLoss (default: 0.3)")
    p.add_argument("--arcface-margin", type=float, default=0.5,
                   help="v6.0.0: Additive angular margin (m) untuk ArcFace/CosFace/SubCenter (default: 0.5)")
    p.add_argument("--arcface-scale", type=float, default=30.0,
                   help="v6.0.0: Scale factor (s) untuk ArcFace/CosFace/SubCenter (default: 30)")
    p.add_argument("--arcface-variant", choices=["linear", "true"], default="linear",
                   help="v8: untuk --loss arcface — 'linear' (cosθ−m, == v7.x, default) atau "
                        "'true' (cos(θ+m) sejati). (--loss arcface_true setara 'true'.)")
    p.add_argument("--qa-floor", type=float, default=0.3,
                   help="v8: lantai margin QA-ArcFace (m_eff = m*(floor+(1-floor)*quality)). Default 0.3")
    p.add_argument("--subcenter-k", type=int, default=3,
                   help="v7.0.0: Jumlah sub-centers per kelas untuk subcenter_arcface (default: 3)")
    # ---- v7.0.0 mix-frame augmentation (C1) ----
    p.add_argument("--frame-sampling", choices=["median", "random"], default="median",
                   help="v7.0.0: 'median' = 1 median frame per sesi (default, deterministik); "
                        "'random' = 1 random frame per sesi per epoch (C1 mix-frame augmentation)")
    p.add_argument("--cross-session-mining", action="store_true",
                   help="v7.0.0 C2: aktifkan cross-session triplet mining — "
                        "positive pair hanya dari sesi berbeda. Hanya berlaku untuk loss=triplet.")
    # ---- v7.2.0 Representation ablation (R1/R2/R3) ----
    p.add_argument("--repr-mode",
                   choices=["canonical_npy", "fps_npy", "raw_ply",
                            # v8 alignment ablation (full-cloud, tanpa FPS)
                            "align_center", "align_centerscale", "align_pca_robust", "align_anatomical"],
                   default="canonical_npy",
                   help="v7.2.0: sumber point cloud. 'canonical_npy' (R2) = cnn_input.npy; "
                        "'fps_npy' (R3); 'raw_ply' (R1) = output.ply. "
                        "v8: 'align_center'(A1), 'align_centerscale'(A2), 'align_pca_robust'(A4), "
                        "'align_anatomical'(A5) = file align_*.npy dari make_align_variants.py.")
    p.add_argument("--frame-mode", choices=["median", "all"], default="median",
                   help="v7.2.0: dalam rezim low-data (--frames-per-session 1), 'median' = 1 frame "
                        "median per sesi (v7.1.x); 'all' = semua frame per sesi (C1/C2/C3 ablation, "
                        "tetap dalam budget 15 sesi/subjek)")
    p.add_argument("--val-metric", choices=["loss", "pair_eer"], default="loss",
                   help="v5.0.0: Metric untuk model selection. 'pair_eer' = val pair EER (v5 default); "
                        "'loss' = val loss (backward compat)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="v5.0.0: Nonaktifkan early stopping (fixed budget mode)")
    p.add_argument("--use-aux-loss", action="store_true",
                   help="v5.0.0: Aktifkan auxiliary classification loss pada geom branch")
    p.add_argument("--val_freq", type=int, default=1,
                   help="v5.0.1: Jalankan validasi + val pair EER setiap N epoch. "
                        "Default 1 (setiap epoch). Naikkan ke 3-5 untuk low-data regime "
                        "dengan sedikit batch per epoch agar training lebih cepat.")
    # ---- v5.0.0 GPU/RAM Auto-tuning ----
    p.add_argument("--preload-augment", action="store_true",
                   help="v5.0.0: Pre-generate semua augmented variants di RAM (rekomendasi A100/H100). "
                        "Menghilangkan CPU bottleneck augmentation per batch. "
                        "RAM usage: ~repeat × n_frames × n_points × 24 bytes.")
    p.add_argument("--repeat", type=int, default=None,
                   help="v5.0.0: Berapa kali tiap frame muncul per epoch (default: 1 untuk low-data, 10 untuk all-frame). "
                        "Naikkan untuk lebih banyak augmentation diversity per epoch.")
    # ---- Ablasi GeoAtt (Plan v0.3.0 §D4) ----
    # Default: tanpa fitur geometri (no_geom). Aktifkan komponen secara independen
    # untuk ablasi terpisah. Kombinasi `--use-gam --use-geom-fusion` setara dengan
    # baseline with_geom.
    p.add_argument("--use-gam", dest="use_gam", action="store_true",
                   help="Aktifkan Geometric Attention Module (GAM1+GAM2) di forward path")
    p.add_argument("--use-geom-fusion", dest="use_geom_fusion", action="store_true",
                   help="Aktifkan concat geom_emb ke fusion head")
    p.add_argument("--use-geom", dest="use_geom", action="store_true",
                   help="Shortcut: aktifkan --use-gam DAN --use-geom-fusion (baseline with_geom)")
    # ---- Performance ----
    p.add_argument("--siamese-mode", choices=["concat", "split"], default="concat",
                   help="'concat' (default, v0.4.0-optimize): 1 forward call, BN over 2B. "
                        "'split' (v0.3.0/v0.4.0-baseline): 2 panggilan terpisah, BN per-branch.")
    # ---- Mixed precision training ----
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none",
                   help="Mixed precision mode. bf16 recommended for H100/A100 (no GradScaler "
                        "needed, same exponent range as fp32). fp16 untuk GPU lebih lama. "
                        "Default: none (fp32).")
    return p.parse_args()


def _build_amp(args, device):
    """Return (scaler, amp_dtype) from args.amp. scaler=None for bf16/none."""
    mode = getattr(args, "amp", "none")
    if device.type != "cuda" or mode == "none":
        return None, None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("[AMP] bf16 tidak didukung GPU ini → fallback ke fp32")
            return None, None
        print("[AMP] Mixed precision: bf16 (no GradScaler, optimal H100/A100)")
        return None, torch.bfloat16
    if mode == "fp16":
        print("[AMP] Mixed precision: fp16 (with GradScaler)")
        return torch.cuda.amp.GradScaler(), torch.float16
    return None, None


def _resolve_geom_flags(args) -> tuple[bool, bool]:
    """Resolusi flag use_gam/use_geom_fusion dari CLI.

    `--use-geom` menjadi shortcut untuk keduanya. Default semua False = no_geom.
    """
    use_gam = bool(getattr(args, "use_gam", False) or getattr(args, "use_geom", False))
    use_fuse = bool(getattr(args, "use_geom_fusion", False) or getattr(args, "use_geom", False))
    return use_gam, use_fuse


def _run_epoch(model, loader, criterion, optimizer, device,
               train=True, scaler=None, amp_dtype=None, desc=""):
    """Jalankan satu epoch training atau validasi. Kembalikan rata-rata loss.

    AMP: kalau amp_dtype != None → autocast. scaler hanya dipakai untuk fp16.
    """
    model.train() if train else model.eval()
    total_loss = 0.0
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    use_scaler = use_amp and (scaler is not None)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            pts_a  = batch["pts_a"].to(device, non_blocking=True)
            geom_a = batch["geom_a"].to(device, non_blocking=True)
            pts_b  = batch["pts_b"].to(device, non_blocking=True)
            geom_b = batch["geom_b"].to(device, non_blocking=True)
            label  = batch["label"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, _, sim = model(pts_a, geom_a, pts_b, geom_b)
                    loss = criterion(sim, label)
                if train:
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            else:
                _, _, sim = model(pts_a, geom_a, pts_b, geom_b)
                loss = criterion(sim, label)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.detach()
            if pbar.n % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss.item() / max(len(loader), 1)


def _run_epoch_triplet(model, loader, criterion, optimizer, device,
                       train=True, scaler=None, amp_dtype=None, desc=""):
    """Jalankan satu epoch dengan OnlineTripletLoss."""
    model.train() if train else model.eval()
    total_loss = 0.0
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    use_scaler = use_amp and (scaler is not None)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            pts    = batch["pts"].to(device, non_blocking=True)
            geom   = batch["geom"].to(device, non_blocking=True)
            labels = batch["label_idx"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    emb  = model.encode(pts, geom)
                    loss = criterion(emb, labels)
                if train:
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            else:
                emb  = model.encode(pts, geom)
                loss = criterion(emb, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(loader), 1)


def _run_epoch_arcface(model, loader, criterion, optimizer, device,
                        train=True, scaler=None, amp_dtype=None, desc=""):
    """Jalankan satu epoch dengan ArcFace loss."""
    model.train() if train else model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    use_scaler = use_amp and (scaler is not None)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            pts    = batch["pts"].to(device, non_blocking=True)
            geom   = batch["geom"].to(device, non_blocking=True)
            labels = batch["label_idx"].to(device, non_blocking=True)
            quality = batch["quality"].to(device, non_blocking=True) if "quality" in batch else None
            if train:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model.forward_arcface(pts, geom, labels, quality)
                    loss = criterion.compute_loss(logits, labels)
                if train:
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            else:
                logits = model.forward_arcface(pts, geom, labels, quality)
                loss = criterion.compute_loss(logits, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.detach()
            pred = logits.argmax(dim=1)
            total_acc += (pred == labels).float().mean().detach()
            if pbar.n % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{total_acc.item()/(pbar.n+1):.3f}")
    return total_loss.item() / max(len(loader), 1), total_acc.item() / max(len(loader), 1)


def _run_epoch_hybrid(model, loader, criterion, optimizer, device,
                      train=True, scaler=None, amp_dtype=None, desc=""):
    """Jalankan satu epoch dengan Hybrid ArcFace + Triplet loss."""
    model.train() if train else model.eval()
    total_loss = 0.0
    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    use_scaler = use_amp and (scaler is not None)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            pts    = batch["pts"].to(device, non_blocking=True)
            geom   = batch["geom"].to(device, non_blocking=True)
            labels = batch["label_idx"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    emb = model.encode(pts, geom)
                    loss = criterion(emb, labels)
                if train:
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            else:
                emb = model.encode(pts, geom)
                loss = criterion(emb, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.detach()
            if pbar.n % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss.item() / max(len(loader), 1)


def _run_epoch_triplet_v5(model, loader, criterion, optimizer, device,
                          train=True, scaler=None, amp_dtype=None, desc="",
                          use_aux_loss=False, aux_weight=0.3):
    """
    v5.0.0: Jalankan satu epoch dengan OnlineTripletLoss + optional auxiliary loss.

    Returns:
        dict dengan keys: total_loss, triplet_loss, aux_loss, aux_acc (jika applicable)
    """
    model.train() if train else model.eval()
    total_loss_sum = 0.0
    triplet_loss_sum = 0.0
    aux_loss_sum = 0.0
    aux_correct = 0
    aux_total = 0
    n_batches = 0

    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    use_scaler = use_amp and (scaler is not None)
    ctx = torch.enable_grad() if train else torch.no_grad()

    # Deteksi CrossSessionTripletLoss untuk pass session_ids
    from losses.triplet import CrossSessionTripletLoss
    is_cross_session = isinstance(criterion, CrossSessionTripletLoss)

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            pts         = batch["pts"].to(device, non_blocking=True)
            geom        = batch["geom"].to(device, non_blocking=True)
            labels      = batch["label_idx"].to(device, non_blocking=True)
            session_ids = batch.get("session_idx")
            if session_ids is not None:
                session_ids = session_ids.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            def _compute_loss(emb, labels, session_ids):
                if is_cross_session and session_ids is not None:
                    return criterion(emb, labels, session_ids)
                return criterion(emb, labels)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    if use_aux_loss:
                        emb, aux_logits = model.encode(pts, geom, return_aux=True)
                        triplet_loss = _compute_loss(emb, labels, session_ids)
                        aux_loss = torch.nn.functional.cross_entropy(aux_logits, labels)
                        loss = triplet_loss + aux_weight * aux_loss
                    else:
                        emb = model.encode(pts, geom)
                        loss = _compute_loss(emb, labels, session_ids)
                        triplet_loss = loss
                        aux_loss = torch.tensor(0.0, device=device)
                if train:
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            else:
                if use_aux_loss:
                    emb, aux_logits = model.encode(pts, geom, return_aux=True)
                    triplet_loss = _compute_loss(emb, labels, session_ids)
                    aux_loss = torch.nn.functional.cross_entropy(aux_logits, labels)
                    loss = triplet_loss + aux_weight * aux_loss
                else:
                    emb = model.encode(pts, geom)
                    loss = _compute_loss(emb, labels, session_ids)
                    triplet_loss = loss
                    aux_loss = torch.tensor(0.0, device=device)
                if train:
                    loss.backward()
                    optimizer.step()

            total_loss_sum += loss.detach()
            triplet_loss_sum += triplet_loss.detach()
            aux_loss_sum += aux_loss.detach()
            n_batches += 1

            if use_aux_loss and aux_logits is not None:
                pred = aux_logits.argmax(dim=1)
                aux_correct += (pred == labels).sum().detach()
                aux_total += labels.size(0)

            if pbar.n % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    triplet=f"{triplet_loss.item():.4f}",
                    aux=f"{aux_loss.item():.4f}" if use_aux_loss else None,
                )

    result = {
        "total_loss": total_loss_sum / max(n_batches, 1),
        "triplet_loss": triplet_loss_sum / max(n_batches, 1),
        "aux_loss": aux_loss_sum / max(n_batches, 1),
    }
    if use_aux_loss and aux_total > 0:
        result["aux_acc"] = aux_correct / aux_total
    return result


def _run_epoch_arcface_v5(model, loader, criterion, optimizer, device,
                          train=True, scaler=None, amp_dtype=None, desc=""):
    """
    v6.0.0: Jalankan satu epoch ArcFace pada PalmFrameDataset (low-data regime).

    Returns dict bentuk sama dengan triplet v5 (total_loss, triplet_loss, aux_loss,
    aux_acc) supaya logging path identik dan val_pair_metric tetap kompatibel.
    """
    model.train() if train else model.eval()
    total_loss_sum = 0.0
    correct = 0
    seen = 0
    n_batches = 0

    use_amp = (amp_dtype is not None) and (device.type == "cuda")
    use_scaler = use_amp and (scaler is not None)
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            pts    = batch["pts"].to(device, non_blocking=True)
            geom   = batch["geom"].to(device, non_blocking=True)
            labels = batch["label_idx"].to(device, non_blocking=True)
            quality = batch["quality"].to(device, non_blocking=True) if "quality" in batch else None
            if train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model.forward_arcface(pts, geom, labels, quality)
                    loss = criterion.compute_loss(logits, labels)
                if train:
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            else:
                logits = model.forward_arcface(pts, geom, labels, quality)
                loss = criterion.compute_loss(logits, labels)
                if train:
                    loss.backward()
                    optimizer.step()

            total_loss_sum += loss.detach()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().detach()
            seen += labels.size(0)
            n_batches += 1

            if pbar.n % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(correct.item()/max(seen,1)):.3f}",
                )

    return {
        "total_loss":   total_loss_sum / max(n_batches, 1),
        "triplet_loss": total_loss_sum / max(n_batches, 1),  # alias for logging compat
        "aux_loss":     torch.tensor(0.0),
        "aux_acc":      (correct / max(seen, 1)) if seen > 0 else 0.0,
    }


def _save_checkpoint(output_dir, model, optimizer, epoch, best_val_loss, fold_idx, name):
    ckpt = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss":        best_val_loss,
        "fold":                 fold_idx,
    }
    torch.save(ckpt, output_dir / name)


def train_one_fold(
    fold_idx: int,
    train_sessions: dict,
    test_sessions: dict,
    args,
    device: torch.device,
) -> dict:
    """
    Train satu LOSO fold dengan two-phase training dan early stopping.

    Phase 1 (Main):   lr=args.lr, StepLR, early stopping patience=args.patience
    Phase 2 (Finetune): lr=args.finetune_lr, CosineAnnealingLR, patience=patience//2
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}")
    print(f"{'='*60}")

    output_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard logger
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Simpan config untuk reproducibility
    train_seed = args.seed if args.seed is not None else args.split_seed
    config = {
        "seed": train_seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "margin": args.margin,
        "geom_dim": args.geom_dim,
        "n_points": args.n_points,
        "sampling": args.sampling,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "balance": args.balance,
        "use_gam": getattr(args, "use_gam", False) or getattr(args, "use_geom", False),
        "use_geom_fusion": getattr(args, "use_geom_fusion", False) or getattr(args, "use_geom", False),
        "variant": "with_geom" if (getattr(args, "use_geom", False)) else ("gam_only" if getattr(args, "use_gam", False) else ("fuse_only" if getattr(args, "use_geom_fusion", False) else "no_geom")),
        "siamese_mode": getattr(args, "siamese_mode", "concat"),
        "amp": getattr(args, "amp", "none"),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Fit normalizer pada geometry training saja (hindari data leakage)
    all_train_dirs = [s for ss in train_sessions.values() for s in ss]
    train_geoms    = [load_geometry(d) for d in all_train_dirs]
    normalizer     = GeometryNormalizer()
    normalizer.fit(train_geoms)
    normalizer.save(output_dir / "normalizer.json")

    augmentor      = PointCloudAugmentor()
    geom_augmentor = GeometryAugmentor(noise_sigma=0.02)  # ±2% std per fitur

    train_dataset = PalmPairDataset(
        label_sessions=train_sessions,
        n_points=args.n_points,
        sampling=args.sampling,
        augment=augmentor,
        geom_augment=geom_augmentor,   # aktif hanya untuk training
        normalizer=normalizer,
    )
    val_dataset = PalmPairDataset(
        label_sessions=test_sessions,
        n_points=args.n_points,
        sampling=args.sampling,
        augment=None,
        geom_augment=None,             # tidak augment saat validasi
        normalizer=normalizer,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=8 if args.num_workers > 0 else None,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=8 if args.num_workers > 0 else None,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
    )

    _use_gam, _use_fuse = _resolve_geom_flags(args)
    model     = SiamesePalmNet(
        geom_dim=args.geom_dim,
        use_geom=(_use_gam or _use_fuse),
        use_gam=_use_gam,
        use_geom_fusion=_use_fuse,
        siamese_mode=getattr(args, "siamese_mode", "concat"),
    ).to(device)
    print(f"[Siamese] mode = {getattr(args, 'siamese_mode', 'concat')}")
    criterion = ContrastiveLoss(margin=args.margin)

    # ----------------------------------------------------------------
    # Phase 1 — Main Training
    # ----------------------------------------------------------------
    print(f"\n--- Phase 1: Main Training (lr={args.lr:.2e}, epochs≤{args.epochs}) ---")
    optimizer = Adam(model.parameters(), lr=args.lr, fused=(device.type == "cuda"))
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    early_stop = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    scaler, amp_dtype = _build_amp(args, device)

    start_epoch   = 0
    best_val_loss = float("inf")

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resume dari epoch {start_epoch}")

    log_path = output_dir / "train_log.csv"
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("phase,epoch,train_loss,val_loss,lr\n")

    for epoch in range(start_epoch, args.epochs):
        t0         = time.time()
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True,
                                 scaler=scaler, amp_dtype=amp_dtype)
        val_loss   = _run_epoch(model, val_loader,   criterion, None,      device, train=False,
                                 scaler=None,   amp_dtype=amp_dtype)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(
            f"P1 Epoch {epoch+1:03d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={current_lr:.2e}  t={elapsed:.1f}s"
        )

        with open(log_path, "a") as f:
            f.write(f"1,{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.6e}\n")

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/val", val_loss, epoch + 1)
            writer.add_scalar("learning_rate", current_lr, epoch + 1)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            _save_checkpoint(output_dir, model, optimizer, epoch, best_val_loss, fold_idx, "best.pth")
            print(f"  Terbaik: val_loss={best_val_loss:.4f} disimpan")
        elif (epoch + 1) % 10 == 0:
            _save_checkpoint(output_dir, model, optimizer, epoch, best_val_loss, fold_idx,
                             f"epoch_{epoch+1:03d}.pth")

        if early_stop.step(val_loss, model, epoch + 1):
            print(f"  [EarlyStopping] Berhenti di epoch {epoch+1}")
            break

    # Pulihkan bobot terbaik Phase 1 sebelum fine-tuning
    early_stop.restore_best(model, device)

    # ----------------------------------------------------------------
    # Phase 2 — Fine-Tuning (opsional)
    # ----------------------------------------------------------------
    if args.finetune_epochs > 0:
        print(f"\n--- Phase 2: Fine-Tuning "
              f"(lr={args.finetune_lr:.2e}, epochs≤{args.finetune_epochs}) ---")
        ft_optimizer = Adam(model.parameters(), lr=args.finetune_lr, fused=(device.type == "cuda"))
        ft_scheduler = CosineAnnealingLR(ft_optimizer, T_max=args.finetune_epochs, eta_min=1e-6)
        ft_patience  = max(5, args.patience // 2)
        ft_early     = EarlyStopping(patience=ft_patience, min_delta=args.min_delta)
        ft_best_loss = best_val_loss
        # Fresh scaler untuk Phase 2.
        ft_scaler, _ = _build_amp(args, device)

        for ft_epoch in range(args.finetune_epochs):
            t0         = time.time()
            train_loss = _run_epoch(model, train_loader, criterion, ft_optimizer, device, train=True,
                                     scaler=ft_scaler, amp_dtype=amp_dtype)
            val_loss   = _run_epoch(model, val_loader,   criterion, None,          device, train=False,
                                     scaler=None,      amp_dtype=amp_dtype)

            ft_scheduler.step()
            current_lr = ft_scheduler.get_last_lr()[0]
            elapsed    = time.time() - t0

            print(
                f"P2 Epoch {ft_epoch+1:03d}/{args.finetune_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={current_lr:.2e}  t={elapsed:.1f}s"
            )

            with open(log_path, "a") as f:
                f.write(f"2,{ft_epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.6e}\n")

            is_best = val_loss < ft_best_loss
            if is_best:
                ft_best_loss = val_loss
                best_val_loss = ft_best_loss
                _save_checkpoint(output_dir, model, ft_optimizer, ft_epoch,
                                 best_val_loss, fold_idx, "best.pth")
                print(f"  Terbaik (FT): val_loss={best_val_loss:.4f} disimpan")

            if ft_early.step(val_loss, model, ft_epoch + 1):
                print(f"  [EarlyStopping] Fine-tune berhenti di epoch {ft_epoch+1}")
                break

        ft_early.restore_best(model, device)
        # Simpan final fine-tuned model
        _save_checkpoint(output_dir, model, ft_optimizer, args.finetune_epochs - 1,
                         best_val_loss, fold_idx, "best_finetuned.pth")

    if writer:
        writer.close()
    return {"fold": fold_idx, "best_val_loss": best_val_loss}


def train_fixed_split(
    train_frames: dict,
    val_frames: dict,
    args,
    device: torch.device,
) -> dict:
    """
    Training dengan fixed split (non-LOSO) + two-phase.

    v5.0.0 enhancements:
      - Support low-data regime (1 frame/sesi) via PalmFrameDataset + OnlineTripletLoss
      - Val pair EER metric untuk model selection (alternatif val_loss)
      - Fixed budget mode (tanpa early stopping)
      - Auxiliary classification loss untuk geom branch

    Hasil checkpoint disimpan langsung di args.output_dir.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard logger
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Resolve flags
    _use_gam, _use_fuse = _resolve_geom_flags(args)
    use_aux = getattr(args, "use_aux_loss", False)
    loss_type = getattr(args, "loss", "contrastive")
    val_metric = getattr(args, "val_metric", "loss")
    frames_per_session = getattr(args, "frames_per_session", "all")
    repr_mode = getattr(args, "repr_mode", "canonical_npy")  # v7.2.0 R1/R2/R3
    no_early_stop = getattr(args, "no_early_stop", False)
    triplet_margin = getattr(args, "triplet_margin", 0.3)
    preload_augment = getattr(args, "preload_augment", False)
    # repeat default: 1 untuk low-data, 10 untuk all-frame; bisa override via CLI
    repeat_default = 1 if frames_per_session == "1" else 10
    repeat_override = getattr(args, "repeat", None)
    repeat = int(repeat_override) if repeat_override is not None else repeat_default

    # Fit normalizer pada geometry training saja (hindari data leakage)
    all_train_dirs = [s for ss in train_frames.values() for s in ss]
    train_geoms    = [load_geometry(d) for d in all_train_dirs]
    normalizer     = GeometryNormalizer()
    normalizer.fit(train_geoms)
    normalizer.save(output_dir / "normalizer.json")

    # ----------------------------------------------------------------
    # Dataset & Loss selection
    # ----------------------------------------------------------------
    is_triplet        = loss_type in ("triplet", "hybrid")
    is_arcface        = loss_type == "arcface"
    is_cosface        = loss_type == "cosface"
    is_subcenter      = loss_type == "subcenter_arcface"
    is_contrastive    = loss_type == "contrastive"
    # v8: head margin terpadu (losses/margin_heads.py) — semua loss classification-based.
    from losses.margin_heads import MARGIN_LOSS_TYPES
    is_margin_loss    = loss_type in MARGIN_LOSS_TYPES   # arcface(+true/cosface/subcenter/adacos/curricular/qa)
    # v6.0.0+: margin & triplet pakai PalmFrameDataset (per-frame, classification)
    is_perframe       = is_triplet or is_margin_loss
    arc_margin          = getattr(args, "arcface_margin", 0.5)
    arc_scale           = getattr(args, "arcface_scale", 30.0)
    arcface_variant     = getattr(args, "arcface_variant", "linear")
    qa_floor            = getattr(args, "qa_floor", 0.3)
    subcenter_k         = getattr(args, "subcenter_k", 3)
    frame_sampling      = getattr(args, "frame_sampling", "median")
    cross_session_mining = getattr(args, "cross_session_mining", False)

    augmentor      = PointCloudAugmentor()
    geom_augmentor = GeometryAugmentor(noise_sigma=0.05)  # v5.0.0: ±5% std (lebih agresif)

    if is_perframe:
        # Triplet/ArcFace mode: PalmFrameDataset (individual frames)
        print(f"[Dataset] mode={loss_type}  repeat={repeat}  preload_augment={preload_augment}")
        train_dataset = PalmFrameDataset(
            label_sessions=train_frames,
            n_points=args.n_points,
            sampling=args.sampling,
            augment=augmentor,
            geom_augment=geom_augmentor,
            normalizer=normalizer,
            repeat=repeat,
            preload_augment=preload_augment,
            repr_mode=repr_mode,
        )
        val_dataset = PalmFrameDataset(
            label_sessions=val_frames,
            n_points=args.n_points,
            sampling=args.sampling,
            augment=None,
            geom_augment=None,
            normalizer=normalizer,
            repeat=1,
            preload_augment=False,
            repr_mode=repr_mode,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            drop_last=True, persistent_workers=args.num_workers > 0,
            prefetch_factor=8 if args.num_workers > 0 else None,
            worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=8 if args.num_workers > 0 else None,
            worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        )
        if is_margin_loss:
            # v8: head margin ada di MODEL (build_margin_head). Criterion = cross-entropy saja
            # (head sudah menerapkan margin/scale yang benar per loss_type → perbandingan jujur).
            criterion = _MarginCECriterion()
            print(f"[Loss] margin head '{loss_type}' (variant={arcface_variant}, m={arc_margin}, "
                  f"s={arc_scale}, K={subcenter_k}); criterion=cross-entropy")
        elif cross_session_mining:
            from losses.triplet import CrossSessionTripletLoss
            criterion = CrossSessionTripletLoss(margin=triplet_margin)
            print(f"[Loss] CrossSessionTripletLoss: margin={triplet_margin}")
        else:
            criterion = OnlineTripletLoss(margin=triplet_margin, mining="batch_hard")
    else:
        # Contrastive / ArcFace mode: PalmPairDataset (pairs)
        train_dataset = PalmPairDataset(
            label_sessions=train_frames,
            n_points=args.n_points,
            sampling=args.sampling,
            augment=augmentor,
            geom_augment=geom_augmentor,
            normalizer=normalizer,
        )
        val_dataset = PalmPairDataset(
            label_sessions=val_frames,
            n_points=args.n_points,
            sampling=args.sampling,
            augment=None,
            geom_augment=None,
            normalizer=normalizer,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            drop_last=True, persistent_workers=args.num_workers > 0,
            prefetch_factor=8 if args.num_workers > 0 else None,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=8 if args.num_workers > 0 else None,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        )
        criterion = ContrastiveLoss(margin=args.margin)

    # Model
    n_subjects = len(train_frames)
    model = SiamesePalmNet(
        geom_dim=args.geom_dim,
        use_geom=(_use_gam or _use_fuse),
        use_gam=_use_gam,
        use_geom_fusion=_use_fuse,
        siamese_mode=getattr(args, "siamese_mode", "concat"),
        use_aux_loss=use_aux,
        n_subjects=n_subjects,
        # v6.0.0: margin head hanya dibuat ketika loss margin aktif
        num_classes=(n_subjects if is_margin_loss else 0),
        arc_margin=arc_margin,
        arc_scale=arc_scale,
        # v8: head margin dipilih sesuai loss_type (factory build_margin_head)
        loss_type=loss_type,
        arcface_variant=arcface_variant,
        subcenter_k=subcenter_k,
        qa_floor=qa_floor,
    ).to(device)
    print(f"[Siamese] mode = {getattr(args, 'siamese_mode', 'concat')}, "
          f"loss = {loss_type}, aux_loss = {use_aux}")

    # Val pair metric (for pair_eer model selection)
    val_pair_metric = None
    if val_metric == "pair_eer":
        val_pair_metric = ValPairMetric(
            device=device, n_points=args.n_points,
            n_impostor=100, pair_seed=999,
            repr_mode=repr_mode,
        )
        val_pair_metric.reset_pairs(val_frames)

    # ----------------------------------------------------------------
    # Phase 1 — Main Training
    # ----------------------------------------------------------------
    print(f"\n--- Phase 1: Main Training (lr={args.lr:.2e}, epochs={args.epochs}) ---")
    optimizer  = Adam(model.parameters(), lr=args.lr, fused=(device.type == "cuda"))
    scheduler  = StepLR(optimizer, step_size=30, gamma=0.5)
    early_stop = None if no_early_stop else EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    scaler, amp_dtype = _build_amp(args, device)

    start_epoch   = 0
    best_val_loss = float("inf")
    best_val_eer  = float("inf")
    best_smoothed_eer = float("inf")  # v5.0.1: model selection pakai smoothed EER window=5

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resume dari epoch {start_epoch}")

    log_path = output_dir / "train_log.csv"
    if start_epoch == 0:
        if is_perframe:
            with open(log_path, "w") as f:
                f.write("phase,epoch,train_loss,triplet_loss,aux_loss,val_loss,val_eer,aux_acc,lr\n")
        else:
            with open(log_path, "w") as f:
                f.write("phase,epoch,train_loss,val_loss,lr\n")

    # v6.0.0+: dispatcher per-frame epoch (triplet / arcface / cosface / subcenter)
    def _run_perframe_epoch(loader, optim_, train_, scaler_, amp_dtype_):
        if is_margin_loss:
            # CosFace & SubCenter-ArcFace menggunakan runner yang sama dengan ArcFace
            # (semua adalah classification-based margin loss dengan interface identik)
            return _run_epoch_arcface_v5(
                model, loader, criterion, optim_, device,
                train=train_, scaler=scaler_, amp_dtype=amp_dtype_,
            )
        return _run_epoch_triplet_v5(
            model, loader, criterion, optim_, device,
            train=train_, scaler=scaler_, amp_dtype=amp_dtype_,
            use_aux_loss=use_aux, aux_weight=0.3,
        )

    # v5.0.0: Reset VRAM peak tracker untuk audit utilization
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Training
        if is_perframe:
            train_results = _run_perframe_epoch(train_loader, optimizer, True, scaler, amp_dtype)
            train_loss = train_results["total_loss"]
            triplet_loss_val = train_results["triplet_loss"]
            aux_loss_val = train_results["aux_loss"]
            aux_acc = train_results.get("aux_acc", 0.0)
        else:
            train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True,
                                     scaler=scaler, amp_dtype=amp_dtype)
            triplet_loss_val = 0.0
            aux_loss_val = 0.0
            aux_acc = 0.0

        # Validation (skip kalau bukan val_freq epoch)
        val_loss = float("nan")
        val_eer = None
        do_val = ((epoch + 1) % args.val_freq == 0)
        if do_val:
            if is_perframe:
                val_results = _run_perframe_epoch(val_loader, None, False, None, amp_dtype)
                val_loss = val_results["total_loss"]
            else:
                val_loss = _run_epoch(model, val_loader, criterion, None, device, train=False,
                                       scaler=None, amp_dtype=amp_dtype)

            # Val pair EER (jika aktif)
            if val_pair_metric is not None:
                vm = val_pair_metric.compute(model, normalizer=normalizer)
                val_eer = vm["eer"]

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        # Print log
        if is_perframe:
            log_msg = (
                f"P1 Epoch {epoch+1:03d}/{args.epochs}  "
                f"train={train_loss:.4f}  triplet={triplet_loss_val:.4f}  "
            )
            if use_aux:
                log_msg += f"aux={aux_loss_val:.4f}  aux_acc={aux_acc:.3f}  "
            if do_val:
                log_msg += f"val_loss={val_loss:.4f}  "
                if val_eer is not None:
                    log_msg += f"val_eer={val_eer:.4f}  "
            else:
                log_msg += f"[skip val]  "
            log_msg += f"lr={current_lr:.2e}  t={elapsed:.1f}s"
            print(log_msg)
        else:
            if do_val:
                print(
                    f"P1 Epoch {epoch+1:03d}/{args.epochs}  "
                    f"train={train_loss:.4f}  val={val_loss:.4f}  "
                    f"lr={current_lr:.2e}  t={elapsed:.1f}s"
                )
            else:
                print(
                    f"P1 Epoch {epoch+1:03d}/{args.epochs}  "
                    f"train={train_loss:.4f}  [skip val]  "
                    f"lr={current_lr:.2e}  t={elapsed:.1f}s"
                )

        # v5.0.0: report VRAM peak setelah epoch pertama (audit utilization)
        if device.type == "cuda" and epoch == start_epoch:
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  [VRAM Peak Ep1] allocated={peak_gb:.1f}GB  reserved={reserved_gb:.1f}GB  "
                  f"({100*reserved_gb/total_gb:.0f}% of {total_gb:.0f}GB)")

        with open(log_path, "a") as f:
            if is_perframe:
                val_loss_log = val_loss if do_val else -1
                val_eer_log = val_eer if val_eer is not None else -1
                f.write(f"1,{epoch+1},{train_loss:.6f},{triplet_loss_val:.6f},"
                        f"{aux_loss_val:.6f},{val_loss_log:.6f},"
                        f"{val_eer_log:.6f},"
                        f"{aux_acc:.6f},{current_lr:.6e}\n")
            else:
                val_loss_log = val_loss if do_val else -1
                f.write(f"1,{epoch+1},{train_loss:.6f},{val_loss_log:.6f},{current_lr:.6e}\n")

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("learning_rate", current_lr, epoch + 1)
            if is_perframe:
                writer.add_scalar("loss/triplet", triplet_loss_val, epoch + 1)
                if use_aux:
                    writer.add_scalar("loss/aux", aux_loss_val, epoch + 1)
                    writer.add_scalar("aux/accuracy", aux_acc, epoch + 1)
            if do_val:
                writer.add_scalar("loss/val", val_loss, epoch + 1)
                if val_eer is not None:
                    writer.add_scalar("val/pair_eer", val_eer, epoch + 1)
                    writer.add_scalar("val/pair_auc", vm.get("auc", 0), epoch + 1)
                    writer.add_scalar("val/pair_tar_at_far1", vm.get("tar_at_far1", 0), epoch + 1)

        # Model selection (hanya pada epoch validasi)
        is_best = False
        if do_val:
            if val_metric == "pair_eer" and val_eer is not None:
                # v5.0.1: Gunakan smoothed EER (window=5) untuk model selection
                # kalau history cukup, fallback ke raw EER untuk epoch awal
                smoothed_eer_val = val_pair_metric.smoothed_eer(window=5)
                if smoothed_eer_val is not None:
                    is_best = smoothed_eer_val < best_smoothed_eer
                    if is_best:
                        best_smoothed_eer = smoothed_eer_val
                        best_val_eer = val_eer  # tetap track raw untuk compat
                        best_val_loss = val_loss
                else:
                    # History < 5 epoch: pakai raw EER
                    is_best = val_eer < best_val_eer
                    if is_best:
                        best_val_eer = val_eer
                        best_val_loss = val_loss
            else:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

            if is_best:
                _save_checkpoint(output_dir, model, optimizer, epoch, best_val_loss, 0, "best.pth")
                if val_metric == "pair_eer" and val_eer is not None:
                    se = val_pair_metric.smoothed_eer(window=5)
                    se_str = f"smoothed={se:.4f} " if se is not None else ""
                    print(f"  Terbaik: val_eer={val_eer:.4f} {se_str}disimpan")
                else:
                    print(f"  Terbaik: val_loss={best_val_loss:.4f} disimpan")
            elif (epoch + 1) % 10 == 0:
                _save_checkpoint(output_dir, model, optimizer, epoch, best_val_loss, 0,
                                 f"epoch_{epoch+1:03d}.pth")

            if early_stop is not None and early_stop.step(val_loss, model, epoch + 1):
                print(f"  [EarlyStopping] Berhenti di epoch {epoch+1}")
                break
        else:
            # Skip epoch: tetap save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                _save_checkpoint(output_dir, model, optimizer, epoch, best_val_loss, 0,
                                 f"epoch_{epoch+1:03d}.pth")

    if early_stop is not None:
        early_stop.restore_best(model, device)

    # ----------------------------------------------------------------
    # Phase 2 — Fine-Tuning (opsional)
    # ----------------------------------------------------------------
    if args.finetune_epochs > 0:
        print(f"\n--- Phase 2: Fine-Tuning "
              f"(lr={args.finetune_lr:.2e}, epochs={args.finetune_epochs}) ---")
        ft_optimizer = Adam(model.parameters(), lr=args.finetune_lr, fused=(device.type == "cuda"))
        ft_scheduler = CosineAnnealingLR(ft_optimizer, T_max=args.finetune_epochs, eta_min=1e-6)
        ft_patience  = max(5, args.patience // 2)
        ft_early     = None if no_early_stop else EarlyStopping(patience=ft_patience, min_delta=args.min_delta)
        ft_best_loss = best_val_loss
        ft_best_eer  = best_val_eer
        ft_best_smoothed_eer = best_smoothed_eer  # v5.0.1: carry over dari Phase 1
        ft_scaler, _ = _build_amp(args, device)

        for ft_epoch in range(args.finetune_epochs):
            t0 = time.time()

            if is_perframe:
                train_results = _run_perframe_epoch(train_loader, ft_optimizer, True, ft_scaler, amp_dtype)
                train_loss = train_results["total_loss"]
                triplet_loss_val = train_results["triplet_loss"]
                aux_loss_val = train_results["aux_loss"]
                aux_acc = train_results.get("aux_acc", 0.0)
            else:
                train_loss = _run_epoch(model, train_loader, criterion, ft_optimizer, device, train=True,
                                         scaler=ft_scaler, amp_dtype=amp_dtype)
                triplet_loss_val = 0.0
                aux_loss_val = 0.0
                aux_acc = 0.0

            # Validation (skip kalau bukan val_freq epoch)
            val_loss = float("nan")
            val_eer = None
            do_val = ((ft_epoch + 1) % args.val_freq == 0)
            if do_val:
                if is_perframe:
                    val_results = _run_perframe_epoch(val_loader, None, False, None, amp_dtype)
                    val_loss = val_results["total_loss"]
                else:
                    val_loss = _run_epoch(model, val_loader, criterion, None, device, train=False,
                                           scaler=None, amp_dtype=amp_dtype)

                if val_pair_metric is not None:
                    vm = val_pair_metric.compute(model, normalizer=normalizer)
                    val_eer = vm["eer"]

            ft_scheduler.step()
            current_lr = ft_scheduler.get_last_lr()[0]
            elapsed    = time.time() - t0

            if is_perframe:
                log_msg = (
                    f"P2 Epoch {ft_epoch+1:03d}/{args.finetune_epochs}  "
                    f"train={train_loss:.4f}  triplet={triplet_loss_val:.4f}  "
                )
                if use_aux:
                    log_msg += f"aux={aux_loss_val:.4f}  aux_acc={aux_acc:.3f}  "
                if do_val:
                    log_msg += f"val_loss={val_loss:.4f}  "
                    if val_eer is not None:
                        log_msg += f"val_eer={val_eer:.4f}  "
                else:
                    log_msg += f"[skip val]  "
                log_msg += f"lr={current_lr:.2e}  t={elapsed:.1f}s"
                print(log_msg)
            else:
                if do_val:
                    print(
                        f"P2 Epoch {ft_epoch+1:03d}/{args.finetune_epochs}  "
                        f"train={train_loss:.4f}  val={val_loss:.4f}  "
                        f"lr={current_lr:.2e}  t={elapsed:.1f}s"
                    )
                else:
                    print(
                        f"P2 Epoch {ft_epoch+1:03d}/{args.finetune_epochs}  "
                        f"train={train_loss:.4f}  [skip val]  "
                        f"lr={current_lr:.2e}  t={elapsed:.1f}s"
                    )

            with open(log_path, "a") as f:
                if is_perframe:
                    val_loss_log = val_loss if do_val else -1
                    val_eer_log = val_eer if val_eer is not None else -1
                    f.write(f"2,{ft_epoch+1},{train_loss:.6f},{triplet_loss_val:.6f},"
                            f"{aux_loss_val:.6f},{val_loss_log:.6f},"
                            f"{val_eer_log:.6f},"
                            f"{aux_acc:.6f},{current_lr:.6e}\n")
                else:
                    val_loss_log = val_loss if do_val else -1
                    f.write(f"2,{ft_epoch+1},{train_loss:.6f},{val_loss_log:.6f},{current_lr:.6e}\n")

            if writer:
                writer.add_scalar("loss/train", train_loss, args.epochs + ft_epoch + 1)
                writer.add_scalar("learning_rate", current_lr, args.epochs + ft_epoch + 1)
                if is_perframe:
                    writer.add_scalar("loss/triplet", triplet_loss_val, args.epochs + ft_epoch + 1)
                    if use_aux:
                        writer.add_scalar("loss/aux", aux_loss_val, args.epochs + ft_epoch + 1)
                        writer.add_scalar("aux/accuracy", aux_acc, args.epochs + ft_epoch + 1)
                if do_val:
                    writer.add_scalar("loss/val", val_loss, args.epochs + ft_epoch + 1)
                    if val_eer is not None:
                        writer.add_scalar("val/pair_eer", val_eer, args.epochs + ft_epoch + 1)

            if do_val:
                is_best = False
                if val_metric == "pair_eer" and val_eer is not None:
                    # v5.0.1: smoothed EER untuk model selection
                    smoothed_eer_val = val_pair_metric.smoothed_eer(window=5)
                    if smoothed_eer_val is not None:
                        is_best = smoothed_eer_val < ft_best_smoothed_eer
                        if is_best:
                            ft_best_smoothed_eer = smoothed_eer_val
                            ft_best_eer = val_eer
                            ft_best_loss = val_loss
                    else:
                        is_best = val_eer < ft_best_eer
                        if is_best:
                            ft_best_eer = val_eer
                            ft_best_loss = val_loss
                else:
                    is_best = val_loss < ft_best_loss
                    if is_best:
                        ft_best_loss = val_loss

                if is_best:
                    best_val_loss = ft_best_loss
                    best_val_eer = ft_best_eer
                    _save_checkpoint(output_dir, model, ft_optimizer, ft_epoch,
                                     best_val_loss, 0, "best.pth")
                    if val_metric == "pair_eer" and val_eer is not None:
                        se = val_pair_metric.smoothed_eer(window=5)
                        se_str = f"smoothed={se:.4f} " if se is not None else ""
                        print(f"  Terbaik (FT): val_eer={val_eer:.4f} {se_str}disimpan")
                    else:
                        print(f"  Terbaik (FT): val_loss={best_val_loss:.4f} disimpan")

                if ft_early is not None and ft_early.step(val_loss, model, ft_epoch + 1):
                    print(f"  [EarlyStopping] Fine-tune berhenti di epoch {ft_epoch+1}")
                    break

        if ft_early is not None:
            ft_early.restore_best(model, device)
        _save_checkpoint(output_dir, model, ft_optimizer, args.finetune_epochs - 1,
                         best_val_loss, 0, "best_finetuned.pth")

    if writer:
        writer.close()
    return {"best_val_loss": best_val_loss, "best_val_eer": best_val_eer}


def _set_seed(seed: int):
    """Set seed untuk PyTorch, NumPy, dan Python random."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Seed] RNG seeds set to {seed}")


def _set_cuda_perf_mode(benchmark: bool = True):
    """Enable CUDA performance mode: cuDNN benchmark + TF32."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = benchmark
        # TF32 on Ampere+ (A100, H100) — free ~1.5-2× speedup for matmuls
        torch.set_float32_matmul_precision('high')
        print(f"[CUDA] cudnn.benchmark={benchmark}, TF32=enabled")


def _clamp_args_to_safe_limits(args):
    """
    v5.0.0: Clamp batch_size dan n_points ke safe hardware limits.
    Dynamic probe / user override bisa merekomendasikan BS/N yang terlalu
    agresif (e.g. BS=1079 N=16384 di A100 40GB → OOM di ball_query).
    Fungsi ini memastikan args tidak melebihi batas yang telah diverifikasi
    oleh auto-config untuk tiap kelas GPU.
    """
    if not torch.cuda.is_available():
        return args

    _bs_safe, _, _, _, _np_safe = _auto_config()
    # _auto_config sudah print info; kita hanya clamp tanpa print ulang
    original_bs = args.batch_size
    original_np = args.n_points

    if args.batch_size > _bs_safe:
        print(f"[SafetyClamp] batch_size {args.batch_size} → {_bs_safe} "
              f"(melebihi safe limit GPU)")
        args.batch_size = _bs_safe
    if args.n_points > _np_safe:
        print(f"[SafetyClamp] n_points {args.n_points} → {_np_safe} "
              f"(melebihi safe limit GPU)")
        args.n_points = _np_safe

    # Heuristic: untuk N_POINTS > 8192, bs harus lebih konservatif lagi
    # karena ball_query distance matrix scale O(B * S * N).
    # Kalau N=16384, bs harus ≤ ~192 bahkan di A100 80GB.
    if args.n_points > 8192 and args.batch_size > 192:
        print(f"[SafetyClamp] batch_size {args.batch_size} → 192 "
              f"(n_points={args.n_points} > 8192, limit distance matrix memory)")
        args.batch_size = 192

    if original_bs != args.batch_size or original_np != args.n_points:
        print(f"[SafetyClamp] Final config: batch_size={args.batch_size}, "
              f"n_points={args.n_points}")
    return args


def main():
    args = parse_args()
    args = _clamp_args_to_safe_limits(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set RNG seed + CUDA perf mode
    train_seed = args.seed if args.seed is not None else args.split_seed
    _set_seed(train_seed)
    _set_cuda_perf_mode(benchmark=True)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Error: data_dir '{data_dir}' tidak ditemukan")

    frame_layout = _is_frame_layout(data_dir)
    print(f"Layout data: {'frame (single-frame)' if frame_layout else 'session (ICP multi-frame)'}")

    # ========================================================================
    # Fixed Split (new pipeline — default untuk thesis)
    # ========================================================================
    if args.fixed_split:
        if not frame_layout:
            sys.exit("Fixed split saat ini hanya mendukung frame layout. "
                     "Gunakan --fixed_split dengan dataset hasil process_single_frames.py.")

        # v5.0.0+: Low-data regime (1 frame/sesi)
        if getattr(args, "frames_per_session", "all") == "1":
            _frame_sampling = getattr(args, "frame_sampling", "median")
            _sampling_seed  = getattr(args, "seed", None)
            _frame_mode     = getattr(args, "frame_mode", "median")
            _repr_mode      = getattr(args, "repr_mode", "canonical_npy")
            if _frame_mode == "all":
                # v7.2.0 C1/C2/C3: semua frame per sesi, tetap dalam budget 15 sesi/subjek
                from utils.dataset_lowdata import build_lowdata_splits_all_frames
                print(f"\n[v7.2.0] Low-data ALL-FRAME regime: repr_mode={_repr_mode}")
                splits = build_lowdata_splits_all_frames(data_dir)
            else:
                print(f"\n[v7.0.0] Low-data regime: frame_sampling={_frame_sampling}, repr_mode={_repr_mode}")
                splits = build_lowdata_splits(
                    data_dir,
                    frame_sampling=_frame_sampling,
                    sampling_seed=_sampling_seed if _frame_sampling == "random" else None,
                )
            train_frames = splits["train"]
            val_frames   = splits["val"]
            test_frames  = splits["test"]
            holdout_frames = splits["holdout"]

            print(f"\nSplit (low-data, deterministic chronological):")
            for label in sorted(train_frames):
                print(f"  {label:<18} train={len(train_frames[label]):3d}  "
                      f"val={len(val_frames[label]):3d}  "
                      f"test={len(test_frames[label]):3d}  "
                      f"holdout={len(holdout_frames[label])}")

            # Simpan splits
            from utils.dataset_lowdata import DROPPED_SUBJECTS as _DROPPED
            splits_data = {
                "mode": "low_data_allframe" if _frame_mode == "all" else "low_data_1fps",
                "frame_mode": _frame_mode,
                "repr_mode": _repr_mode,
                "dropped_subjects": sorted(_DROPPED),
                "train":   {label: [str(f) for f in frames]
                            for label, frames in train_frames.items()},
                "val":     {label: [str(f) for f in frames]
                            for label, frames in val_frames.items()},
                "test":    {label: [str(f) for f in frames]
                            for label, frames in test_frames.items()},
                "holdout": {label: [str(f) for f in frames]
                            for label, frames in holdout_frames.items()},
            }
            splits_path = Path(args.output_dir) / "splits.json"
            splits_path.parent.mkdir(parents=True, exist_ok=True)
            with open(splits_path, "w") as f:
                json.dump(splits_data, f, indent=2)
            print(f"\nsplits.json disimpan di: {splits_path}")

            # v7.2.0 E9: instrumentasi kecepatan & resource training
            n_train_frames = sum(len(v) for v in train_frames.values())
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _t_train0 = time.perf_counter()
            result = train_fixed_split(train_frames, val_frames, args, device)
            _train_wall_s = time.perf_counter() - _t_train0
            perf = {
                "repr_mode": _repr_mode,
                "frame_mode": _frame_mode,
                "n_train_frames": n_train_frames,
                "n_points": args.n_points,
                "epochs": args.epochs,
                "finetune_epochs": args.finetune_epochs,
                "batch_size": args.batch_size,
                "total_train_wall_s": round(_train_wall_s, 2),
                "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                                    if device.type == "cuda" else None),
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            }
            with open(Path(args.output_dir) / "perf.json", "w") as f:
                json.dump(perf, f, indent=2)
            print(f"[v7.2.0 perf] train_wall={_train_wall_s:.1f}s  "
                  f"peak_gpu={perf['peak_gpu_mem_mb']}MB  n_train_frames={n_train_frames}")
            best_metric = result.get("best_val_eer" if getattr(args, "val_metric", "loss") == "pair_eer" else "best_val_loss")
            print(f"\nTraining selesai. Best {getattr(args, 'val_metric', 'loss')} = {best_metric:.4f}")
            return

        # All-frame regime (existing logic)
        label_frames, session_groups = scan_dataset_frames(data_dir, filter_invalid=True)

        if args.balance:
            session_groups, min_sessions = balance_label_frames(session_groups)
            label_frames = {
                label: [f for ts_frames in ts_dict.values() for f in ts_frames]
                for label, ts_dict in session_groups.items()
            }
            print(f"[Balance] Setiap label = {min_sessions} sesi "
                  f"(minimum dari semua label)")

        n_labels   = len(label_frames)
        n_sessions = sum(len(ts) for ts in session_groups.values())
        print(f"Label      : {n_labels}")
        print(f"Total sesi : {n_sessions}")
        for label in sorted(label_frames):
            ns = len(session_groups.get(label, {}))
            nf = len(label_frames[label])
            print(f"  {label}: {ns} sesi, {nf} frame")

        if n_labels < 2:
            sys.exit("Butuh minimal 2 label untuk membentuk pasangan impostor")

        # Holdout real test
        session_groups, holdout_probes = split_holdout_sessions(
            session_groups,
            n_holdout_sessions=args.holdout_sessions,
            n_probe_frames=args.holdout_frames,
            seed=args.split_seed,
        )
        n_holdout = sum(len(v) for v in holdout_probes.values())
        print(f"[Holdout] {args.holdout_sessions} sesi/subjek → {n_holdout} frame probe")

        # Three-way split pada sesi tersisa
        train_frames, val_frames, test_frames = split_sessions_three_way(
            session_groups,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )

        print(f"\nSplit:")
        for label in sorted(train_frames):
            print(f"  {label:<18} train={len(train_frames[label]):3d}  "
                  f"val={len(val_frames[label]):3d}  "
                  f"test={len(test_frames[label]):3d}  "
                  f"holdout={len(holdout_probes[label])}")

        # Simpan splits
        splits_data = {
            "split_seed":     args.split_seed,
            "train_ratio":    args.train_ratio,
            "val_ratio":      args.val_ratio,
            "holdout_probes": {label: [str(f) for f in frames]
                               for label, frames in holdout_probes.items()},
            "test":           {label: [str(f) for f in frames]
                               for label, frames in test_frames.items()},
        }
        splits_path = Path(args.output_dir) / "splits.json"
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump(splits_data, f, indent=2)
        print(f"\nsplits.json disimpan di: {splits_path}")

        result = train_fixed_split(train_frames, val_frames, args, device)
        print(f"\nTraining selesai. Best val_loss = {result['best_val_loss']:.4f}")
        return

    # ========================================================================
    # LOSO (legacy pipeline)
    # ========================================================================
    if frame_layout:
        label_frames, session_groups = scan_dataset_frames(data_dir)

        if args.balance:
            session_groups, min_sessions = balance_label_frames(session_groups)
            from collections import defaultdict as _dd
            label_frames = {
                label: [f for ts_frames in ts_dict.values() for f in ts_frames]
                for label, ts_dict in session_groups.items()
            }
            print(f"[Balance] Setiap label = {min_sessions} sesi "
                  f"(minimum dari semua label)")

        n_labels       = len(label_frames)
        n_frames_total = sum(len(v) for v in label_frames.values())
        n_sessions     = sum(len(ts) for ts in session_groups.values())
        print(f"Label      : {n_labels}")
        print(f"Total sesi : {n_sessions}")
        print(f"Total frame: {n_frames_total}")
        for label in sorted(label_frames):
            ns = len(session_groups.get(label, {}))
            nf = len(label_frames[label])
            print(f"  {label}: {ns} sesi, {nf} frame")

        if n_labels < 2:
            sys.exit("Butuh minimal 2 label untuk membentuk pasangan impostor")

        loso_splits = list(make_loso_splits_frames(session_groups))
    else:
        label_sessions = scan_dataset(data_dir)

        if args.balance:
            label_sessions, min_sessions = balance_label_sessions(label_sessions)
            print(f"[Balance] Setiap label = {min_sessions} sesi "
                  f"(minimum dari semua label)")

        n_labels = len(label_sessions)
        n_sessions_total = sum(len(v) for v in label_sessions.values())
        print(f"Label      : {n_labels}")
        print(f"Total sesi : {n_sessions_total}")
        for label, sessions in label_sessions.items():
            print(f"  {label}: {len(sessions)} sesi")

        if n_labels < 2:
            sys.exit("Butuh minimal 2 label untuk membentuk pasangan impostor")

        loso_splits = list(make_loso_splits(label_sessions))

    if args.all_folds:
        results = []
        for fold_idx, train_sessions, test_sessions in loso_splits:
            result = train_one_fold(fold_idx, train_sessions, test_sessions, args, device)
            results.append(result)
        print("\n=== LOSO Summary ===")
        for r in results:
            print(f"  Fold {r['fold']}: best_val_loss={r['best_val_loss']:.4f}")
    else:
        fold_idx, train_sessions, test_sessions = loso_splits[args.fold]
        train_one_fold(fold_idx, train_sessions, test_sessions, args, device)


if __name__ == "__main__":
    main()
