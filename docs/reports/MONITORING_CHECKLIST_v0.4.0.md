# Monitoring Checklist — v0.4.0 Fase 2 Fair Ablation

> **Tanggal Mulai**: _isi tanggal_  
> **Target Selesai**: _isi tanggal + 5–14 hari_ (tergantung branch B1/B2)  
> **Environment**: Google Colab (A100)  
> **Dataset**: 11 subjek, 1,836 valid frame (post-QC v3, geometry.json baru)  
> **Seeds**: `[7, 42, 123, 2026, 31337]`  
> **Split Seed**: `42` (tetap, tidak diubah antar varian)  

---

## 0. Pre-Execution Checklist (WAJIB — jangan skip)

### 0.1 Local Pre-Flight (sebelum upload)
Jalankan di local macOS:
```bash
cd ~/Projects/Thesis/3DCNN
python3 preflight_check.py
```
**Harus keluar: 🟢 ALL CHECKS PASSED.**

### 0.2 Upload ke Google Drive
- [ ] Upload `3DCNN/` ke `MyDrive/3DCNN/` (include dataset, models, utils, collab, train.py, evaluate.py)
- [ ] Verifikasi `dataset/` punya `_QC2_frame_*` folders (bukti QC v3 applied)
- [ ] Verifikasi `collab/train.ipynb` dan `collab/evaluate.ipynb` ter-upload

### 0.3 Colab Environment
- [ ] Mount Drive: `drive.mount('/content/drive')`
- [ ] `!nvidia-smi` → A100 40GB (bukan T4)
- [ ] `torch.cuda.get_device_name(0)` → NVIDIA A100-SXM4-40GB
- [ ] `!python -c "import torch; print(torch.__version__)"` → 2.6.0+cu124
- [ ] `!ls /content/drive/MyDrive/3DCNN/dataset` → 11 folder subjek
- [ ] `!ls /content/drive/MyDrive/3DCNN/dataset/aisah` → ada folder `_QC2_*`

---

## 1. Phase A — Pilot (1 seed, 4 variants)

**Goal:** Dapatkan directional signal B1/B2/B3/B4 sebelum komitmen penuh.  
**Durasi:** ~2 jam A100  
**Seed:** `42`

### 1.1 Smoke Test (WAJIB — 3 epoch, no_geom)
```python
# Di train.ipynb cell Konfigurasi:
SEEDS = [42]
EPOCHS = 3
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 0
PHASE3_EPOCHS = 0
USE_GAM = False
USE_GEOM_FUSION = False
```
- [ ] Training berjalan tanpa error sampai epoch 3
- [ ] Loss ArcFace turun (epoch 1 > epoch 3)
- [ ] Checkpoint `best_rank1.pth` tersimpan di `runs/no_geom/<ts>/seed42/`
- [ ] `config.json` tersimpan & valid JSON (buka dan cek ada field `seed`, `variant`)

**Jika GAGAL → STOP. Fix bug → re-run smoke test.**

### 1.2 Pilot no_geom (seed 42, full epoch)
```python
SEEDS = [42]
EPOCHS = 100
PHASE1_EPOCHS = 100
PHASE2_EPOCHS = 30
PHASE3_EPOCHS = 20
USE_GAM = False
USE_GEOM_FUSION = False
```
- [ ] Training selesai
- [ ] Best val_loss tercatat
- [ ] `best_rank1.pth` + `best_loss.pth` tersimpan

### 1.3 Pilot with_geom (seed 42)
```python
USE_GAM = True
USE_GEOM_FUSION = True  # atau USE_GEOM = True
```
- [ ] Training selesai
- [ ] Best val_loss tercatat

### 1.4 Pilot gam_only (seed 42)
```python
USE_GAM = True
USE_GEOM_FUSION = False
```
- [ ] Training selesai

### 1.5 Pilot fuse_only (seed 42)
```python
USE_GAM = False
USE_GEOM_FUSION = True
```
- [ ] Training selesai

### 1.6 Evaluasi Phase A
- [ ] Buka `evaluate.ipynb`
- [ ] Evaluasi ke-4 varian dengan checkpoint Phase A
- [ ] Catat Rank-1 per varian:

| Varian | Rank-1 seed 42 |
|--------|---------------|
| no_geom | _% |
| with_geom | _% |
| gam_only | _% |
| fuse_only | _% |

---

## 2. Decision Gate B1

**Bandingkan no_geom vs with_geom (seed 42):**

| Kondisi | Gap | Branch | Action |
|---------|-----|--------|--------|
| with_geom ≈ no_geom (≤ 1 ppt) | B1 | **B1** | Lanjut Phase B1 (full 5 seed, 2 varian). Skip gam_only/fuse_only. |
| with_geom ≪ no_geom (> 2 ppt) | B2/B3/B4 | **B2** | Lanjut Phase B2 (full 5 seed, 4 varian). |

**Catat di sini:**
- Gap no_geom vs with_geom = __ ppt
- Branch terpilih = __

---

## 3. Phase B1 — Full Baseline (B1 terpilih)

**Varian:** no_geom + with_geom  
**Seeds:** [7, 42, 123, 2026, 31337]  
**Durasi:** ~8 jam A100

### 3.1 Batch no_geom × 5 seed
- [ ] Seed 7
- [ ] Seed 42
- [ ] Seed 123
- [ ] Seed 2026
- [ ] Seed 31337

### 3.2 Batch with_geom × 5 seed
- [ ] Seed 7
- [ ] Seed 42
- [ ] Seed 123
- [ ] Seed 2026
- [ ] Seed 31337

### 3.3 Evaluasi & Statistik
- [ ] Evaluasi 10 checkpoint (2 varian × 5 seed)
- [ ] Wilcoxon paired (n=5)
- [ ] Bootstrap CI 95%
- [ ] McNemar pooled
- [ ] **Verdict:** Gap hilang? → Tag `v0.4.0-baseline`

---

## 4. Phase B2 — Full Decomposition (B2/B3/B4 terpilih)

**Varian:** no_geom + with_geom + gam_only + fuse_only  
**Seeds:** [7, 42, 123, 2026, 31337]  
**Durasi:** ~38 jam A100

### 4.1 Batch gam_only × 5 seed
- [ ] Seed 7
- [ ] Seed 42
- [ ] Seed 123
- [ ] Seed 2026
- [ ] Seed 31337

### 4.2 Batch fuse_only × 5 seed
- [ ] Seed 7
- [ ] Seed 42
- [ ] Seed 123
- [ ] Seed 2026
- [ ] Seed 31337

### 4.3 Evaluasi & Decompose
- [ ] Evaluasi 20 checkpoint
- [ ] Wilcoxon + Bootstrap + McNemar
- [ ] Decision B2/B3/B4:
  - B2 (gam_only ≪ no_geom, fuse_only ≈ no_geom) → F2.3 Cross-Attention GAM
  - B3 (fuse_only ≪ no_geom, gam_only ≈ no_geom) → F2.4 Gated/FiLM Fusion
  - B4 (keduanya ≪ no_geom) → F2.5 Feature Engineering + Aux Loss

---

## 5. Post-Training Checklist

- [ ] Semua checkpoint tersimpan di Drive
- [ ] Semua `config.json` valid dan lengkap
- [ ] Semua `training.log` tersimpan
- [ ] `eval_results/` punya hasil per varian
- [ ] `REPORT.MD` diupdate dengan hasil Fase 2
- [ ] `SWARM_LOG.md` diupdate
- [ ] Tag `v0.4.0-baseline` (atau `v0.4.0-decompose`) dibuat

---

## Quick Reference Commands

### Local pre-flight
```bash
cd ~/Projects/Thesis/3DCNN
python3 preflight_check.py
```

### Colab GPU check
```python
!nvidia-smi
import torch
print(torch.cuda.get_device_name(0))
```

### Colab variant switch
```python
# no_geom
USE_GAM = False
USE_GEOM_FUSION = False

# with_geom
USE_GAM = True
USE_GEOM_FUSION = True

# gam_only
USE_GAM = True
USE_GEOM_FUSION = False

# fuse_only
USE_GAM = False
USE_GEOM_FUSION = True
```

---

*Last updated: 2026-05-17 — Fase 2 Fair Ablation*
