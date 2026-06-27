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

# v0.4.0-baseline snapshot: ensure parent project root is on path for dataset/losses/utils
sys.path.insert(0, str(Path(__file__).parent.parent))
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
from utils.normalizer import GeometryNormalizer


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
    is_a100 = "A100" in gpu_name

    if vram_gb >= 90 or (is_h100 and vram_gb >= 85):
        # G4 96GB / A100 96GB / H100 NVL 94GB / Blackwell RTX PRO 6000 96GB
        # Empirical (May 18): bs=384/n_pts=12288 OOM di ball_query argsort (20 GiB
        # scratch tunggal + Siamese 2× branch saturasi 85 GB). 256/8192 ≈ 50 GB
        # peak — aman untuk no_geom dan with_geom.
        bs, nw, lr, flr = 256, min(8, n_cpu), 2e-3, 2e-4
        n_pts = 8192
        label = f'96GB class ({gpu_name}, CC={compute_capability})'
    elif is_h100 and vram_gb >= 75:
        # H100 80GB — compute capability 9.0, lebih kencang dari A100
        # Bisa lebih agresif: naikkan bs 50% vs A100 80GB
        bs, nw, lr, flr = 384, min(8, n_cpu), 2e-3, 2e-4
        n_pts = 8192
        label = f'H100 80GB class ({gpu_name}, CC={compute_capability})'
    elif vram_gb >= 75:        # A100 80GB
        # bs=512/n_pts=8192 OOM di ball_query untuk with_geom.
        # 256/8192 ≈ 70-80 GB peak, aman untuk semua variant.
        bs, nw, lr, flr = 256, min(8, n_cpu), 2e-3, 2e-4
        n_pts = 8192
        label = f'A100 80GB class ({gpu_name}, CC={compute_capability})'
    elif vram_gb >= 35:        # A100 40GB
        bs, nw, lr, flr = 192, min(8, n_cpu), 2e-3, 2e-4
        n_pts = 8192
        label = 'A100 40GB'
    elif vram_mb > 0:          # T4 / L4 / V100 16-24GB
        bs, nw, lr, flr = 128, min(2, n_cpu), 1e-3, 1e-4
        n_pts = 4096
        label = 'T4/L4/V100 class'
    else:                      # CPU
        bs, nw, lr, flr = 32, 0, 1e-3, 1e-4
        n_pts = 2048
        label = 'CPU'

    print(f'[Auto-config] {label} | VRAM={vram_gb:.1f}GB | CPU={n_cpu} core | RAM={sys_ram_gb:.1f}GB')
    print(f'              batch_size={bs}, num_workers={nw}, n_points={n_pts}, lr={lr}, finetune_lr={flr}')
    return bs, nw, lr, flr, n_pts


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
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(loader), 1)


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
            if train:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model.forward_arcface(pts, geom, labels)
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
                logits = model.forward_arcface(pts, geom, labels)
                loss = criterion.compute_loss(logits, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_acc += (pred == labels).float().mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{total_acc/(pbar.n+1):.3f}")
    return total_loss / max(len(loader), 1), total_acc / max(len(loader), 1)


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
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(loader), 1)


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
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    _use_gam, _use_fuse = _resolve_geom_flags(args)
    model     = SiamesePalmNet(
        geom_dim=args.geom_dim,
        use_geom=(_use_gam or _use_fuse),
        use_gam=_use_gam,
        use_geom_fusion=_use_fuse,
    ).to(device)
    criterion = ContrastiveLoss(margin=args.margin)

    # ----------------------------------------------------------------
    # Phase 1 — Main Training
    # ----------------------------------------------------------------
    print(f"\n--- Phase 1: Main Training (lr={args.lr:.2e}, epochs≤{args.epochs}) ---")
    optimizer = Adam(model.parameters(), lr=args.lr)
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
        ft_optimizer = Adam(model.parameters(), lr=args.finetune_lr)
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
    Training dengan fixed split (non-LOSO) + two-phase + early stopping.

    Hasil checkpoint disimpan langsung di args.output_dir (tanpa subdir fold_*
    karena tidak ada fold).
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard logger
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Fit normalizer pada geometry training saja (hindari data leakage)
    all_train_dirs = [s for ss in train_frames.values() for s in ss]
    train_geoms    = [load_geometry(d) for d in all_train_dirs]
    normalizer     = GeometryNormalizer()
    normalizer.fit(train_geoms)
    normalizer.save(output_dir / "normalizer.json")

    augmentor      = PointCloudAugmentor()
    geom_augmentor = GeometryAugmentor(noise_sigma=0.02)

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
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    _use_gam, _use_fuse = _resolve_geom_flags(args)
    model     = SiamesePalmNet(
        geom_dim=args.geom_dim,
        use_geom=(_use_gam or _use_fuse),
        use_gam=_use_gam,
        use_geom_fusion=_use_fuse,
    ).to(device)
    criterion = ContrastiveLoss(margin=args.margin)

    # ----------------------------------------------------------------
    # Phase 1 — Main Training
    # ----------------------------------------------------------------
    print(f"\n--- Phase 1: Main Training (lr={args.lr:.2e}, epochs≤{args.epochs}) ---")
    optimizer  = Adam(model.parameters(), lr=args.lr)
    scheduler  = StepLR(optimizer, step_size=30, gamma=0.5)
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
            _save_checkpoint(output_dir, model, optimizer, epoch, best_val_loss, 0, "best.pth")
            print(f"  Terbaik: val_loss={best_val_loss:.4f} disimpan")
        elif (epoch + 1) % 10 == 0:
            _save_checkpoint(output_dir, model, optimizer, epoch, best_val_loss, 0,
                             f"epoch_{epoch+1:03d}.pth")

        if early_stop.step(val_loss, model, epoch + 1):
            print(f"  [EarlyStopping] Berhenti di epoch {epoch+1}")
            break

    early_stop.restore_best(model, device)

    # ----------------------------------------------------------------
    # Phase 2 — Fine-Tuning (opsional)
    # ----------------------------------------------------------------
    if args.finetune_epochs > 0:
        print(f"\n--- Phase 2: Fine-Tuning "
              f"(lr={args.finetune_lr:.2e}, epochs≤{args.finetune_epochs}) ---")
        ft_optimizer = Adam(model.parameters(), lr=args.finetune_lr)
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
                ft_best_loss  = val_loss
                best_val_loss = ft_best_loss
                _save_checkpoint(output_dir, model, ft_optimizer, ft_epoch,
                                 best_val_loss, 0, "best.pth")
                print(f"  Terbaik (FT): val_loss={best_val_loss:.4f} disimpan")

            if ft_early.step(val_loss, model, ft_epoch + 1):
                print(f"  [EarlyStopping] Fine-tune berhenti di epoch {ft_epoch+1}")
                break

        ft_early.restore_best(model, device)
        _save_checkpoint(output_dir, model, ft_optimizer, args.finetune_epochs - 1,
                         best_val_loss, 0, "best_finetuned.pth")

    if writer:
        writer.close()
    return {"best_val_loss": best_val_loss}


def _set_seed(seed: int):
    """Set seed untuk PyTorch, NumPy, dan Python random."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Seed] RNG seeds set to {seed}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set RNG seed
    train_seed = args.seed if args.seed is not None else args.split_seed
    _set_seed(train_seed)

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
