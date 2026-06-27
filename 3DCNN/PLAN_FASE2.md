# ARTIFACT: PLAN_FASE2
# Created by: Planning Agent
# Date: 2026-05-17T08:10:00+07:00
# Related to: v0.4.0 Fase 2 — Fair Ablation Training (4 varian × 5 seed)
# Status: DRAFT → APPROVED (setelah review Lead Agent)

# Rencana Eksekusi Fase 2 — Fair Ablation Training v0.4.0

> **Versi:** 1.0
> **Tanggal:** 2026-05-17
> **Agent:** Planning Agent
> **Scope:** Training 4 varian GeoAtt-PointNet++ (no_geom, with_geom, gam_only, fuse_only) masing-masing 5 seed di Google Colab A100
> **Dataset:** 1,869 valid frame post-QC v3
> **Referensi:** `IMPROVEMENT_PLAN_v0.4.0.md` §F2.1–F2.2, `REPORT.MD` Bagian 4

---

## 1. Pre-Flight Checklist

> **Jalankan semua item di bawah sebelum training pertama dimulai.** Tandai [x] di Colab atau catat di log.

### 1.1 Kode & Repo
| # | Item | Cara Verifikasi | Must Pass |
|---|------|-----------------|-----------|
| 1 | `models/encoder.py` F1.1 sudah di-commit & di-upload ke Drive | `git log --oneline -3` di local; cek timestamp file di Drive | ✅ |
| 2 | `models/siamese.py` F1.2 flag ablasi sudah di-upload | Import tanpa error: `python -c "from models.siamese import SiamesePalmNet; print('OK')"` | ✅ |
| 3 | `train.py` F1.2 CLI flag (`--use-gam`, `--use-geom-fusion`) sudah di-upload | `--help` menampilkan ketiga flag | ✅ |
| 4 | `evaluate.py` F1.3 backward-compat sudah di-upload | Load checkpoint dummy v0.3.0 tanpa error | ✅ |
| 5 | `utils/dataset.py` QC v3 skip `_QC2_frame_*` sudah di-upload | `scan_dataset_frames()` return **1,869** frame | ✅ |

### 1.2 Reproducibility & RNG
| # | Item | Cara Verifikasi | Must Pass |
|---|------|-----------------|-----------|
| 6 | Init parity audit pass (58/58 layer identik) | Jalankan `utils/audit_init_parity.py` → output `identical: 58/58, max|Δ|=0.0` | ✅ |
| 7 | `SPLIT_SEED = 42` tetap (tidak diubah antar varian) | Cek cell Konfigurasi di notebook | ✅ |
| 8 | `SEEDS = [7, 42, 123, 2026, 31337]` tercatat | Cek cell Konfigurasi + `seeds.json` | ✅ |
| 9 | `torch.backends.cudnn.deterministic = True` (opsional, untuk reproducibilitas maksimal) | Cek cell Setup — **catatan:** A100 TF32 bisa menyebabkan nondeterministik kecil. Ini diterima. | ⚠️ |

### 1.3 Dataset & Environment
| # | Item | Cara Verifikasi | Must Pass |
|---|------|-----------------|-----------|
| 10 | Dataset di Drive lengkap & ter-mount | `!ls /content/drive/MyDrive/3DCNN/dataset` menampilkan 11 folder subjek | ✅ |
| 11 | Folder `_QC2_frame_*` dan `_QC2_*` ada di dataset (QC v3 applied) | `find dataset -name "_QC2*" | wc -l` > 0 | ✅ |
| 12 | GPU A100 tersedia & VRAM ≥ 40GB | `nvidia-smi` output menampilkan A100 | ✅ |
| 13 | RAM Colab ≥ 128GB (untuk preload augment) | `!cat /proc/meminfo | head -1` | ✅ |
| 14 | Drive space cukup untuk 20 run (~500MB per run = ~10GB) | `!df -h /content/drive` | ✅ |

### 1.4 Notebook & Config
| # | Item | Cara Verifikasi | Must Pass |
|---|------|-----------------|-----------|
| 15 | `collab/train.ipynb` sudah di-update support 4 varian | Cell Konfigurasi punya `USE_GAM`, `USE_GEOM_FUSION`, `_VARIANT` | ✅ |
| 16 | `collab/evaluate.ipynb` sudah di-update support 4 varian | Cell load model bisa set `use_gam` / `use_geom_fusion` | ✅ |
| 17 | `OUTPUT_DIR` auto-generate berdasar varian | `runs/{variant}/{timestamp}` | ✅ |
| 18 | `config.json` tersimpan otomatis di setiap run | Cek save logic di `train.py` | ✅ |

### 1.5 Quick Smoke Test (WAJIB — 1 seed, 1 varian, 3 epoch)
```python
# Di cell awal notebook, override sementara:
SEEDS = [42]
EPOCHS = 3
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 0
PHASE3_EPOCHS = 0
USE_GAM = False
USE_GEOM_FUSION = False
# Jalankan Cell 1 → Cell 2 → Cell 3 → Cell 4 (training)
```
**Kriteria pass smoke test:**
- Training berjalan tanpa error sampai epoch 3
- Loss ArcFace turun (epoch 1 > epoch 3)
- Checkpoint `best_rank1.pth` tersimpan di `runs/no_geom/<ts>/seed42/`
- `config.json` tersimpan & valid JSON

**Jika smoke test GAGAL:** Jangan lanjutkan. Stop → fix → re-run smoke test.

---

## 2. Execution Order

### 2.1 Mengapa Urutan Ini?

| Urutan | Varian | Justifikasi |
|--------|--------|-------------|
| **1** | `no_geom` | Baseline. Harus stabil dulu. Jika no_geom sendiri tidak mencapai ~99% Rank-1, ada masalah lebih fundamental (data, bug, atau hyperparameter) sebelum kita bisa menyalahkan GeoAtt. |
| **2** | `with_geom` | Full GeoAtt. Langsung bandingkan dengan no_geom untuk melihat apakah gap masih ada setelah init fair. Ini pertanyaan ilmiah utama. |
| **3** | `gam_only` | Isolasi GAM. Dijalankan setelah no_geom & with_geom karena hasilnya bergantung pada apakah gap dengan with_geom masih ada. Kalau with_geom ≈ no_geom, gam_only tidak perlu dieksekusi (branch B1). |
| **4** | `fuse_only` | Isolasi fusion concat. Alasan sama: hanya bermakna kalau gap ada. |

### 2.2 Urutan Seed dalam Satu Varian

Di dalam satu varian, jalankan seed dalam urutan: `[42, 7, 123, 2026, 31337]`

- **Seed 42 pertama** → untuk validasi cepat: jika seed 42 hasilnya anomali, bisa stop & diagnose tanpa membuang 4 seed lainnya.
- **Seed 7 kedua** → cross-check: seed berbeda tapi dekat.
- **Seed 123, 2026, 31337** → fill-in untuk n=5 statistik.

### 2.3 Visualisasi Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  SMOKE TEST: no_geom, seed=42, 3 epoch                      │
│  └── Jika GAGAL → STOP, fix, retry                          │
└─────────────────────────────────────────────────────────────┘
                              ↓ PASS
┌─────────────────────────────────────────────────────────────┐
│  BATCH A: no_geom × 5 seed                                  │
│  └── Evaluasi: Rank-1 ~99%?                                 │
│      └── Jika TIDAK → STOP, diagnose data/hyperparameter    │
└─────────────────────────────────────────────────────────────┘
                              ↓ PASS
┌─────────────────────────────────────────────────────────────┐
│  BATCH B: with_geom × 5 seed                                │
│  └── Evaluasi: gap vs no_geom masih ada?                    │
│      └── Jika TIDAK → Branch B1 (gap = init parity problem) │
└─────────────────────────────────────────────────────────────┘
                              ↓ GAP MASIH ADA
┌─────────────────────────────────────────────────────────────┐
│  BATCH C: gam_only × 5 seed                                 │
│  BATCH D: fuse_only × 5 seed                                │
│  └── Evaluasi: decompose GAM vs fusion                      │
│      └── Branch B2/B3/B4                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Per-Variant Training Steps (Colab Notebook)

> **Platform:** Google Colab A100 (atau A100 40GB fallback)
> **Notebook:** `collab/train.ipynb`
> **Mode:** Runtime → Change runtime type → GPU → A100

### 3.1 Persiapan Umum per Batch

**Langkah 0: Factory Reset Runtime**
- Runtime → Factory reset runtime (untuk bersihkan memory)
- Tunggu sampai runtime baru siap

**Langkah 1: Jalankan Cell 1 — Setup & Mount Drive**
```
[Cell 1] Setup & Cek Environment
```
Verifikasi output:
```
GPU detect : A100/H100 class | VRAM=95.6GB | ...
Auto-config: BATCH_SIZE=512, NUM_WORKERS=8, N_POINTS=8192
AMP (mixed prec.) : aktif
```

**Langkah 2: Jalankan Cell 2 — Konfigurasi**

### 3.2 no_geom (Batch A)

**Cell 2 config:**
```python
EXPERIMENT_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
USE_GAM         = False   # <— matikan GAM
USE_GEOM_FUSION = False   # <— matikan fusion
USE_GEOM        = False   # <— backward-compat flag
SEEDS           = [7, 42, 123, 2026, 31337]
SPLIT_SEED      = 42
LOSS_FN         = 'arcface'
PHASE1_EPOCHS   = 100
PHASE2_EPOCHS   = 30
PHASE3_EPOCHS   = 20
EPOCHS          = 100
BATCH_SIZE      = _AUTO_BS   # 512 (A100)
N_POINTS        = _AUTO_N_PTS # 8192
FRAME_REPEAT    = _AUTO_REPEAT # 30
```

**Cell 3:** Dataset — Scan, Filter, Balance, Split

**Cell 4:** Training (Fixed Split + Multi-Seed)

**Setelah selesai:**
- Cek `runs/no_geom/<timestamp>/` di Drive
- Pastikan ada 5 subfolder: `seed7/`, `seed42/`, `seed123/`, `seed2026/`, `seed31337/`
- Setiap subfolder punya: `best_rank1.pth`, `best_loss.pth`, `config.json`, `training.log`

**Catat timestamp** di log: `RUN_NO_GEOM_TS = '<timestamp>'`

### 3.3 with_geom (Batch B)

**Factory reset runtime** (penting untuk bersihkan CUDA cache)

**Cell 2 config:**
```python
EXPERIMENT_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
USE_GAM         = True    # <— aktifkan GAM
USE_GEOM_FUSION = True    # <— aktifkan fusion
USE_GEOM        = True    # <— backward-compat flag
SEEDS           = [7, 42, 123, 2026, 31337]
SPLIT_SEED      = 42      # SAMA — split identik dengan no_geom
LOSS_FN         = 'arcface'
PHASE1_EPOCHS   = 100
PHASE2_EPOCHS   = 30
PHASE3_EPOCHS   = 20
# Hyperparameter lain identik dengan no_geom
```

**Cell 3:** Dataset — Split identik karena `SPLIT_SEED=42` tetap

**Cell 4:** Training

**Setelah selesai:**
- Cek `runs/with_geom/<timestamp>/`
- Pastikan 5 subfolder seed lengkap

**Catat timestamp:** `RUN_WITH_GEOM_TS = '<timestamp>'`

### 3.4 gam_only (Batch C)

**Hanya jalankan jika** hasil Batch B menunjukkan gap with_geom vs no_geom masih ada (Rank-1 with_geom < no_geom dengan margin > 1% atau signifikan secara visual).

**Cell 2 config:**
```python
EXPERIMENT_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
USE_GAM         = True    # <— GAM aktif
USE_GEOM_FUSION = False   # <— fusion MATI
USE_GEOM        = False   # <— jangan pakai shortcut
SEEDS           = [7, 42, 123, 2026, 31337]
SPLIT_SEED      = 42
# Hyperparameter lain identik
```

**Catat timestamp:** `RUN_GAM_ONLY_TS = '<timestamp>'`

### 3.5 fuse_only (Batch D)

**Hanya jalankan jika** hasil Batch B menunjukkan gap masih ada.

**Cell 2 config:**
```python
EXPERIMENT_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
USE_GAM         = False   # <— GAM mati
USE_GEOM_FUSION = True    # <— fusion aktif
USE_GEOM        = False
SEEDS           = [7, 42, 123, 2026, 31337]
SPLIT_SEED      = 42
# Hyperparameter lain identik
```

**Catat timestamp:** `RUN_FUSE_ONLY_TS = '<timestamp>'`

### 3.6 Ringkasan Perubahan Cell 2 per Varian

| Varian | `USE_GAM` | `USE_GEOM_FUSION` | `USE_GEOM` | `_VARIANT` (auto) |
|--------|-----------|-------------------|------------|-------------------|
| no_geom | False | False | False | `no_geom` |
| with_geom | True | True | True | `with_geom` |
| gam_only | True | False | False | `gam_only` |
| fuse_only | False | True | False | `fuse_only` |

---

## 4. Checkpoint Naming Convention

### 4.1 Struktur Folder Output

```
runs/
├── no_geom/
│   └── 20260517_HHMMSS/          # ← EXPERIMENT_TIMESTAMP
│       ├── config.json             # hyperparameter & metadata
│       ├── seeds.json              # [7, 42, 123, 2026, 31337]
│       ├── seed7/
│       │   ├── best_rank1.pth      # checkpoint terbaik (val rank-1)
│       │   ├── best_loss.pth       # checkpoint terbaik (val loss)
│       │   ├── training.log        # log per epoch
│       │   └── metrics.json        # (opsional) summary metrik
│       ├── seed42/
│       │   └── ...
│       ├── seed123/
│       ├── seed2026/
│       └── seed31337/
├── with_geom/
│   └── 20260517_HHMMSS/
│       └── ...
├── gam_only/
│   └── 20260517_HHMMSS/
│       └── ...
└── fuse_only/
    └── 20260517_HHMMSS/
        └── ...
```

### 4.2 File Naming Rules

| File | Pattern | Kapan Tersimpan |
|------|---------|-----------------|
| Best checkpoint (Rank-1) | `seed{SEED}/best_rank1.pth` | Setiap kali val rank-1 meningkat |
| Best checkpoint (Loss) | `seed{SEED}/best_loss.pth` | Setiap kali val loss menurun |
| Config | `config.json` | Sekali per varian (sebelum training) |
| Seeds | `seeds.json` | Sekali per varian |
| Training log | `seed{SEED}/training.log` | Append setiap epoch |
| Normalizer | `normalizer.json` | Sekali per varian (fit dari train split) |

### 4.3 Metadata Wajib di `config.json`

```json
{
  "variant": "no_geom",
  "use_gam": false,
  "use_geom_fusion": false,
  "seeds": [7, 42, 123, 2026, 31337],
  "split_seed": 42,
  "loss_fn": "arcface",
  "phase1_epochs": 100,
  "phase2_epochs": 30,
  "phase3_epochs": 20,
  "batch_size": 512,
  "n_points": 8192,
  "frame_repeat": 30,
  "lr": 0.002,
  "finetune_lr": 0.0002,
  "arcface_margin": 0.5,
  "arcface_scale": 30.0,
  "dataset_frames": 1869,
  "qc_version": "v3",
  "timestamp": "20260517_HHMMSS",
  "pytorch_version": "2.6.0+cu124",
  "git_commit": "<hash>"  // kalau tersedia
}
```

### 4.4 Evaluasi Output

```
eval_results/
├── no_geom/
│   └── 20260517_HHMMSS_eval/
│       ├── seed7_results.json      # Rank-1, Rank-5, EER, mAP per seed
│       ├── seed42_results.json
│       ├── ...
│       └── summary.json            # mean ± std across 5 seeds
└── with_geom/
    └── ...
```

---

## 5. Post-Training Verification Steps

### 5.1 Verifikasi per Varian (langsung setelah training selesai)

| # | Verifikasi | Cara | Kriteria Pass |
|---|------------|------|---------------|
| 1 | Jumlah checkpoint | `ls runs/{variant}/{ts}/seed*/best_rank1.pth` | 5 file |
| 2 | Ukuran checkpoint | `du -sh runs/{variant}/{ts}/seed*/best_rank1.pth` | ~3–5MB per file |
| 3 | `config.json` valid | `python -c "import json; json.load(open('config.json'))"` | No error |
| 4 | `seeds.json` cocok | Bandingkan dengan daftar seed di plan | [7, 42, 123, 2026, 31337] |
| 5 | Loss curve turun | Plot Cell 5 atau cek `training.log` | Phase 1 loss turun; phase 2/3 stabil |
| 6 | Val rank-1 tidak 0% | `grep "val_rank1" seed*/training.log` | > 80% untuk semua seed |
| 7 | Training time normal | `grep "epoch 100" seed*/training.log` | ~1.5–2 jam per seed (A100) |

### 5.2 Verifikasi Cross-Variant (setelah 2+ varian selesai)

| # | Verifikasi | Cara | Kriteria Pass |
|---|------------|------|---------------|
| 8 | Split identik | Bandingkan `test_fingerprint` atau holdout frames di log no_geom vs with_geom | Sama |
| 9 | Hyperparameter identik | Diff `config.json` antar varian — hanya `use_gam`, `use_geom_fusion`, `variant` yang boleh beda | Pass |
| 10 | Init parity masih valid (opsional, quick) | Load `best_rank1.pth` seed=42 dari 2 varian → bandingkan shared layer | max|Δ| < 1e-6 |

### 5.3 Evaluasi Statistik (menggunakan `collab/evaluate.ipynb`)

**Untuk setiap varian:**
1. Jalankan `collab/evaluate.ipynb` dengan `VARIANT` yang sesuai
2. Gunakan multi-prototype enrollment k=3 (sama dengan v0.3.0)
3. Simpan hasil per seed sebagai JSON

**Ringkasan metrik yang diharapkan:**

| Varian | Rank-1 Mean | Rank-1 Std | EER | mAP |
|--------|-------------|------------|-----|-----|
| no_geom | ~99% | < 1% | < 0.1% | ~99% |
| with_geom | ??? | ??? | ??? | ??? |
| gam_only | ??? | ??? | ??? | ??? |
| fuse_only | ??? | ??? | ??? | ??? |

### 5.4 Komparasi Statistik (setelah semua varian selesai)

Jalankan `collab/compare.ipynb` (atau skrip `utils/compare_variants.py` jika ada):
- Wilcoxon paired (n=5 seed) untuk no_geom vs with_geom
- Bootstrap CI 95% untuk Δ Rank-1
- McNemar pooled (n=5×holdout probes) untuk no_geom vs with_geom
- Plot: CMC curve, training curve, confusion matrix

---

## 6. Risk Assessment & Fallback Plan

### 6.1 Risk Register

| ID | Risk | Likelihood | Impact | Mitigasi |
|----|------|------------|--------|----------|
| R1 | **Colab quota habis** (12 jam/session atau daily limit) | Tinggi (batch ke-3+) | Tinggi | Gunakan fallback 2 varian; simpan checkpoint di Drive setiap epoch |
| R2 | **Runtime disconnect** karena idle | Sedang | Tinggi | Install Colab keep-alive extension; cek setiap 30 menit |
| R3 | **A100 tidak tersedia** → downgrade ke T4/V100 | Sedang | Sedang | Auto-config notebook sudah handle: BS turun, N_POINTS=4096, compile off |
| R4 | **no_geom tidak mencapai ~99%** (bug/masalah data) | Rendah | Kritis | Stop → audit dataset & kode → fix → re-run smoke test |
| R5 | **Init parity pecah kembali** (edit kode tidak sengaja) | Rendah | Kritis | Re-run `audit_init_parity.py` sebelum training; hardcode urutan modul |
| R6 | **Drive sync lag** → checkpoint belum tersimpan saat disconnect | Sedang | Sedang | Force sync: `!sync` atau `drive.flush_and_unmount()` sebelum disconnect |
| R7 | **QC v3 terlalu agresif** → dataset terlalu kecil | Rendah | Sedang | Bandingkan hasil eval di `dataset_qc2` vs `dataset_full` (per keputusan user) |
| R8 | **Training time > 2.5 jam/seed** (A100 lambat) | Rendah | Rendah | Cek GPU throttling; restart runtime; atau kurangi PHASE2/PHASE3 epoch |

### 6.2 Fallback Plans

#### Fallback A: Colab Quota Habis setelah Batch A (no_geom)
- **Prioritas:** Simpan hasil no_geom. Evaluasi no_geom sendiri sudah bisa memberi baseline.
- **Action:** Download hasil no_geom ke local. Tunggu 24 jam untuk reset quota. Lanjutkan with_geom besok.

#### Fallback B: Colab Quota Habis setelah Batch A+B (no_geom + with_geom)
- **Prioritas:** Evaluasi no_geom vs with_geom dulu. Jika gap tidak ada → Branch B1, selesai.
- **Jika gap masih ada:** Skip gam_only & fuse_only untuk sementara. Dokumentasikan: "2 varian selesai, 2 varian isolasi ditunda karena quota."
- **Action:** Lanjutkan gam_only + fuse_only di session Colab berikutnya.

#### Fallback C: no_geom Rank-1 < 95% (anomali)
- **Stop semua training.**
- **Diagnose:**
  1. Cek `training.log` — apakah loss konvergen?
  2. Cek `config.json` — apakah hyperparameter sama dengan v0.3.0?
  3. Re-run `scan_dataset_frames()` — masih 1,869?
  4. Cek init parity — `audit_init_parity.py` masih pass?
  5. Cek split — apakah holdout terlalu banyak?
- **Fix & retry smoke test sebelum lanjut.**

#### Fallback D: T4/V100 downgrade (VRAM < 40GB)
- **Action:** Auto-config notebook akan:
  - BATCH_SIZE=128, N_POINTS=4096, FRAME_REPEAT=10
  - Training time naik ~2× (4 jam/seed)
  - Kemungkinan hasil sedikit berbeda (lebih sedikit points, lebih kecil batch)
  - **Catat di config** bahwa run ini pakai T4. Bandingkan dengan A100 secara eksplisit di laporan.

---

## 7. Decision Branch B1–B4 Trigger Conditions

> Branching terjadi **setelah Batch B selesai** (no_geom + with_geom dievaluasi). Batch C & D hanya dijalankan jika kondisi memerlukan.

### 7.1 Decision Tree

```
F2.1: no_geom × 5 seed  +  with_geom × 5 seed  → evaluasi
                        │
                        ▼
    ┌──────────────────────────────────────────────┐
    │ B1: with_geom ≈ no_geom                      │
    │     (|Δ Rank-1| < 0.5% AND McNemar p > 0.05) │
    └──────────────────────────────────────────────┘
         │
         │ YES → Stop. Masalah dominan adalah init parity (D1).
         │       Tulis v0.4.0-baseline report.
         │       v0.5.0 fokus ke improvement lain (PLY Direct, dsb.)
         │
         │ NO  → Gap masih ada. Lanjutkan Batch C & D.
         ▼
    ┌──────────────────────────────────────────────┐
    │ B2: gam_only ≪ no_geom, fuse_only ≈ no_geom  │
    │     (gam_only gap > 1%, fuse_only gap < 0.5%)│
    └──────────────────────────────────────────────┘
         │
         │ YES → GAM penyebab utama.
         │       → F2.3: Implementasi Cross-Attention GAM
         │       → v0.5.0 = GAM v2
         │
         │ NO  → Lanjut ke B3
         ▼
    ┌──────────────────────────────────────────────┐
    │ B3: fuse_only ≪ no_geom, gam_only ≈ no_geom  │
    │     (fuse_only gap > 1%, gam_only gap < 0.5%)│
    └──────────────────────────────────────────────┘
         │
         │ YES → Fusion concat penyebab utama.
         │       → F2.4: Implementasi Gated/FiLM Fusion
         │       → v0.5.0 = Fusion v2
         │
         │ NO  → Lanjut ke B4
         ▼
    ┌──────────────────────────────────────────────┐
    │ B4: keduanya ≪ no_geom                       │
    │     (gam_only gap > 1% AND fuse_only gap > 1%)│
    └──────────────────────────────────────────────┘
         │
         │ YES → Cara mengonsumsi geom secara umum salah.
         │       → F2.5: Feature Engineering + Auxiliary Loss
         │       → v0.5.0 = GeoAtt v2 (redesign total)
         │
         │ NO  → Edge case (salah satu > 0.5% tapi < 1%, dll.)
         │       → Analisis lebih dalam dengan Analysis Agent.
         │       → Mungkin kombinasi B2+B3 ringan.
```

### 7.2 Decision Matrix (Numerik)

| Branch | Δ Rank-1 (with−no) | gam_only gap | fuse_only gap | McNemar p | Aksi |
|--------|-------------------|--------------|---------------|-----------|------|
| **B1** | < 0.5% | N/A | N/A | > 0.05 | Stop. Init parity adalah masalah utama. |
| **B2** | > 1% | > 1% | < 0.5% | < 0.05 | GAM redesign (cross-attention). |
| **B3** | > 1% | < 0.5% | > 1% | < 0.05 | Fusion redesign (FiLM/gated). |
| **B4** | > 1% | > 1% | > 1% | < 0.05 | GeoAtt total redesign (feature eng + aux loss). |
| **Edge** | 0.5–1% | 0.5–1% | 0.5–1% | 0.03–0.10 | Analisis lanjut; mungkin dropout tuning (F2.6) cukup. |

> **Catatan:** "gap" dihitung sebagai |Rank-1(varian) − Rank-1(no_geom)|. Bootstrap CI Δ harus melingkupi atau tidak melingkupi 0 sebagai konfirmasi.

### 7.3 Contoh Ilustratif

| Skenario | no_geom | with_geom | gam_only | fuse_only | Branch |
|----------|---------|-----------|----------|-----------|--------|
| A (ideal) | 99.8% | 99.7% | — | — | **B1** |
| B (GAM jahat) | 99.8% | 95.0% | 95.2% | 99.6% | **B2** |
| C (fusion jahat) | 99.8% | 95.0% | 99.5% | 95.5% | **B3** |
| D (geom jahat) | 99.8% | 95.0% | 95.5% | 95.3% | **B4** |
| E (edge) | 99.8% | 98.5% | 98.8% | 98.2% | Edge → F2.6 dropout tuning |

---

## 8. Estimated Timeline

### 8.1 Asumsi
- GPU: A100 (80GB), ~2 jam per seed per varian
- Overhead: setup, evaluasi, upload/download, verifikasi
- Operasional: 1 varian per hari (considering Colab session limit & verifikasi)

### 8.2 Timeline Detail

| Hari | Aktivitas | Durasi | Output |
|------|-----------|--------|--------|
| **Hari 1 (Prep)** | Pre-flight checklist + smoke test | 1–2 jam | Smoke test pass, log verifikasi |
| **Hari 2** | Batch A: no_geom × 5 seed | ~10 jam | 5 checkpoint, training logs |
| **Hari 3 (Pagi)** | Verifikasi no_geom + evaluasi | 2 jam | Rank-1 no_geom confirmed ~99% |
| **Hari 3 (Sore)** | Batch B: with_geom × 5 seed | ~10 jam | 5 checkpoint, training logs |
| **Hari 4 (Pagi)** | Verifikasi with_geom + evaluasi | 2 jam | Gap assessment |
| **Hari 4 (Sore)** | **Decision Gate B1** — Bandingkan no_geom vs with_geom | 1 jam | Branch decision |

#### Jika Branch B1 (gap hilang):
| **Hari 5** | Laporan v0.4.0-baseline + dokumentasi | 4 jam | `REPORT.MD` update, tag v0.4.0 |
| | **Total: ~5 hari kerja** | | |

#### Jika Branch B2/B3/B4 (gap masih ada):
| **Hari 5** | Batch C: gam_only × 5 seed | ~10 jam | 5 checkpoint |
| **Hari 6 (Pagi)** | Batch D: fuse_only × 5 seed | ~10 jam | 5 checkpoint |
| **Hari 6 (Sore)** | Evaluasi 4 varian + komparasi statistik | 3 jam | Wilcoxon, Bootstrap, McNemar |
| **Hari 7** | **Decision Gate B2/B3/B4** — Decompose | 2 jam | Branch terpilih |
| **Hari 8+** | Implementasi F2.3/F2.4/F2.5 sesuai branch | 2–3 hari | Kode + training ulang varian terkait |
| | **Total: ~10–12 hari kerja** | | |

### 8.3 Timeline Pessimistic (quota Colab terbatas)

| Skenario | Durasi Total |
|----------|-------------|
| Optimistic (A100 tersedia terus, tidak disconnect) | 5–7 hari |
| Realistic (1–2x disconnect/quota delay) | 7–10 hari |
| Pessimistic (sering downgrade ke T4, quota habis) | 10–14 hari |

### 8.4 Critical Path

```
[Smoke Test] → [no_geom × 5] → [Eval no_geom] → [with_geom × 5] → [Eval with_geom]
                                                                           ↓
                                                              [Decision Gate B1]
                                                                   /        \
                                                                B1(YES)   B1(NO)
                                                                 /            \
                                                   [Report v0.4.0]         [gam_only × 5]
                                                                                ↓
                                                                           [fuse_only × 5]
                                                                                ↓
                                                                           [Decision B2/B3/B4]
                                                                                ↓
                                                                           [Implement F2.x]
```

**Critical path = no_geom + with_geom + eval + decision.** Ini adalah minimum viable deliverable Fase 2. Batch C & D adalah investigasi lanjut yang bergantung pada hasil.

---

## 9. Communication & Handoff

### 9.1 Setelah Setiap Batch Selesai
1. **Screenshot** terminal output terakhir (epoch terakhir + best val metrics)
2. **Upload** screenshot ke `docs/images/fase2_<variant>_training.png`
3. **Update** `SWARM_LOG.md` dengan entry baru
4. **Verifikasi** checkpoint di Drive (pastikan tidak corrupt)

### 9.2 Setelah Decision Gate B1
- Planning Agent menulis `DECISION_MEMO_B1.md` dengan:
  - Ringkasan metrik no_geom vs with_geom
  - Verdict B1/B2/B3/B4
  - Rekomendasi arah v0.5.0
- Lead Agent review → `REVIEW_GATE.md` (PROCEED / REVISE)

### 9.3 Artifact Checklist Akhir Fase 2

| Artifact | Path | Deadline |
|----------|------|----------|
| Plan ini | `PLAN_FASE2.md` | Sebelum training |
| Training logs (4×5) | `runs/*/<ts>/seed*/training.log` | Per batch |
| Checkpoints (4×5) | `runs/*/<ts>/seed*/best_rank1.pth` | Per batch |
| Config JSONs | `runs/*/<ts>/config.json` | Per batch |
| Evaluasi per varian | `eval_results/*/<ts>_eval/` | Setelah semua training |
| Komparasi statistik | `ANALYSIS_3DCNN_<timestamp>.md` | Setelah semua eval |
| Decision memo | `DECISION_MEMO_B1.md` | Setelah B1 gate |
| Updated REPORT.MD | `../REPORT.MD` | Setelah B1 gate |
| Updated SWARM_LOG | `SWARM_LOG.md` | Terus-menerus |

---

## 10. Quick Reference — Commands

### Local (pre-flight)
```bash
# Verifikasi init parity
python utils/audit_init_parity.py

# Verifikasi dataset
python -c "from utils.dataset import scan_dataset_frames; print(len(scan_dataset_frames('dataset')))"

# Verifikasi import
cd ~/Projects/Thesis/3DCNN && python -c "import models.encoder; print('OK')"
```

### Colab (during training)
```python
# Cek GPU
torch.cuda.get_device_name(0)

# Cek memori GPU
!nvidia-smi

# Force sync Drive
from google.colab import drive
drive.flush_and_unmount()

# Cek folder output
!ls -R /content/drive/MyDrive/3DCNN/runs/no_geom/
```

### Local (post-training)
```bash
# Download dari Drive (opsional)
rsync -av "drive:MyDrive/3DCNN/runs/" ./runs/

# Verifikasi checkpoint integrity
python -c "import torch; ckpt = torch.load('best_rank1.pth'); print(ckpt.keys())"
```

---

*Plan ini disusun oleh Planning Agent berdasarkan `IMPROVEMENT_PLAN_v0.4.0.md`, `REPORT.MD`, dan `AGENTS.md`. Perubahan harus melalui Lead Agent review.*

*Last updated: 2026-05-17*
