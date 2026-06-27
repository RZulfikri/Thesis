# Dokumentasi Proyek 3DCNN — Identifikasi Telapak Tangan

> **Penulis:** Rahmat Zulfikri
> **Program:** S2 Teknik Elektro, Universitas Gadjah Mada
> **Topik tesis:** GeoAtt-PointNet++ untuk identifikasi telapak tangan menggunakan kamera TrueDepth iPhone
> **Dokumen ini:** rangkuman teknis seluruh komponen proyek (arsitektur, dataset, training, evaluasi)

---

## 1. Tujuan Riset

Membangun sistem identifikasi biometrik telapak tangan **1:N (closed-set identification)** berbasis point cloud 3D yang dihasilkan dari sensor TrueDepth iPhone. Kontribusi utama: **Geometric Attention Module (GAM)** yang menggunakan fitur geometris eksplisit (panjang jari, lebar telapak, kelengkungan) untuk memperkaya feature map PointNet++.

Validasi kontribusi GAM dilakukan via **ablation study**:
- **Variant A (with_geom)**: GeoAtt-PointNet++ — full model dengan GAM
- **Variant B (no_geom)**: PointNet++ murni — tanpa GeometryEncoder, tanpa GAM

Klaim utama (akan dibuktikan secara statistik): *GeoAtt-PointNet++ secara signifikan lebih akurat dibanding PointNet++ baseline pada dataset palm 3D yang sama*.

---

## 2. Dataset

### 2.1 Akuisisi
- Sensor: **iPhone TrueDepth front camera** (depth + RGB)
- Subjek: 11 orang (alji, aisah, fadhil, nola, rahmat, reysa, dll.)
- Per subjek: 15–25 sesi scan (timestamp), tiap sesi 5 frame
- Format mentah: depth map + RGB + intrinsics

### 2.2 Preprocessing
Pipeline preprocessing (di luar repo ini, di proyek iOS): depth+RGB → 3D point cloud → segmentasi palm → PCA alignment → unit-sphere normalization → simpan sebagai `cnn_input.npy` (N × 6: XYZ + normals).

Setiap frame menghasilkan:
- `cnn_input.npy` — point cloud (N variatif ~50k-150k titik)
- `geometry.json` — fitur geometri terukur (mm absolut) + flag `is_valid`

### 2.3 Struktur folder
```
dataset/
  [label]/
    [timestamp]/
      frame_00/
        cnn_input.npy
        geometry.json
      frame_01/
        cnn_input.npy
        geometry.json
      ...
```

### 2.4 Fitur Geometri (14 dimensi)
File `geometry.json` berisi pengukuran anatomis dalam **mm absolut**:

| Fitur | Dimensi | Deskripsi |
|---|---|---|
| `finger_lengths_mm` | 5 | Panjang tiap jari dari wrist center |
| `palm_width_mm` | 1 | Lebar telapak di knuckle row |
| `palm_height_mm` | 1 | Tinggi telapak wrist→knuckle |
| `palm_depth_std_mm` | 1 | Std dev Z area palm (kelengkungan) |
| `finger_widths_mm` | 5 | Lebar tiap jari di zona atas knuckle |
| `mean_palm_curvature` | 1 | Mean `\|1 − \|nz\|\|` (0=rata, 1=melengkung) |
| **Total** | **14** | |

Normalisasi: **z-score per fitur**, fit dari training set saja (`GeometryNormalizer.fit()`), simpan ke `normalizer.json`.

### 2.5 Filter Sesi Invalid
`scan_dataset_frames(filter_invalid=True)` membaca `geometry.json` dan **skip seluruh sesi** jika ada satu frame dengan `is_valid=False`. Filter ini menghilangkan ~95 frame yang scan-nya terlalu dekat (di luar rentang kalibrasi sensor).

### 2.6 Dataset Splitting

Pipeline split data **fixed split + multi-seed**, di-level **sesi (timestamp)** untuk mencegah data leakage:

```
Semua sesi per subjek
  │
  ├─ scan_dataset_frames(filter_invalid=True)
  ├─ balance_label_frames(seed=SPLIT_SEED)         → 15 sesi/subjek
  ├─ split_holdout_sessions(n_holdout=1, n_probe=3) → sesi 1 holdout
  │       ↳ holdout_probes: 3 frame acak per subjek (real test)
  └─ split_sessions_three_way(0.70, 0.15)          → dari 14 sesi sisa
          ├─ 10 sesi → train  (~100 frame/subjek)
          ├─  2 sesi → val    (~20 frame/subjek)
          └─  2 sesi → test   (~20 frame/subjek)
```

**Tersimpan di `splits.json`** untuk reproducibility dan untuk `evaluate.ipynb`.

> **Key design**: `SPLIT_SEED=42` tetap (tidak ikut multi-seed) → kedua varian (with_geom dan no_geom) mendapat split data **identik** → memungkinkan paired comparison yang valid.

---

## 3. Arsitektur Model

### 3.1 GeoAtt-PointNet++ Encoder

```
Input:
  pts   (B, N, 6)   — XYZ + normals (N=4096 hasil sampling)
  geom  (B, 14)     — fitur geometri ter-normalisasi

         │
  ┌──────┴──────────────────────────────────────┐
  │                                             │
  │       Point Cloud Branch                    │  Geometry Branch
  │                                             │
  │  SA1: 1024 → 512 pts, r=0.05, k=32          │  GeometryEncoder
  │       MLP[32, 32, 64] → feat1 (B,512,64)    │    Linear(14→64)
  │       └─ GAM1(feat1, geom_emb) → (B,512,64) │    BatchNorm, ReLU
  │                                             │    Linear(64→64)
  │  SA2: 512 → 128 pts, r=0.15, k=64           │    → geom_emb (B,64)
  │       MLP[64, 64, 128] → feat2 (B,128,128)  │
  │       └─ GAM2(feat2, geom_emb) → (B,128,128)│
  │                                             │
  │  SA3: 128 → 1 pt (global pool)              │
  │       MLP[128, 256, 256] → (B,256)          │
  │                                             │
  └──────┬──────────────────────────────────────┘
         │
       concat → (B, 320)            ← 256 + 64
         │
  Fusion Head:
    Linear(320→256, bias=False)
    BatchNorm1d, ReLU
    Dropout(p=0.3)                  ← regularisasi
    Linear(256→128)
         │
       L2-normalize
         │
  Embedding (B, 128)
```

### 3.2 Geometric Attention Module (GAM)

```python
class GeometricAttentionModule:
    """
    Per-channel gating: sigmoid(W·geom_emb) ⊙ sa_feat
    """
    def forward(sa_feat, geom_emb):
        gate = sigmoid(Linear(geom_emb))           # (B, sa_ch)
        return sa_feat * gate.unsqueeze(1)          # broadcast over points
```

Tujuan: geometry yang stabil di tiap subjek (panjang jari, lebar palm) berfungsi sebagai *prior* — channel feature mana yang harus diperkuat/dilemahkan untuk subjek dengan profil geometris tertentu.

### 3.3 Siamese Network

```python
class SiamesePalmNet:
    encoder = GeoAttPointNetEncoder(...)   # SHARED weights

    def encode(pts, geom):
        return encoder(pts, geom)          # (B, 128) untuk training/eval
    
    def forward(pts_a, geom_a, pts_b, geom_b):
        emb_a = encoder(pts_a, geom_a)
        emb_b = encoder(pts_b, geom_b)
        sim   = (emb_a * emb_b).sum(dim=1)  # cosine similarity
        return emb_a, emb_b, sim
```

Satu encoder **shared** (bukan dua encoder paralel) — efisien dan menjamin kedua embedding berada di space yang sama.

---

## 4. Loss Function

### 4.1 Online Triplet Loss (default)
Sejak optimasi terbaru, training menggunakan **OnlineTripletLoss + batch-hard mining**:

```
Untuk setiap anchor a di batch:
  - hardest positive p:  same-label dengan d(a,p) terbesar
  - hardest negative n:  different-label dengan d(a,n) terkecil

Loss = mean( ReLU(d(a,p) - d(a,n) + margin) )    margin=0.3
```

**Kenapa lebih baik dari ContrastiveLoss**:
- Random pairs (lama): mayoritas impostor "mudah" → loss ≈ 0 → tidak ada gradien
- Batch-hard mining (baru): fokus pada kasus paling sulit di tiap batch → gradient signal lebih kuat
- Sangat efektif untuk dataset kecil (11 subjek)

### 4.2 Contrastive Loss (legacy, masih tersedia)
```
d    = sqrt(2*(1 - cos_sim))
loss = y · d²  +  (1-y) · max(0, margin - d)²        margin=0.5
```

Tetap dipertahankan via `LOSS_FN='contrastive'` untuk perbandingan / fallback.

---

## 5. Pipeline Training

### 5.1 Two-Phase Training

```
Phase 1 — Main Training
  Optimizer  : Adam, lr=1e-3
  Scheduler  : StepLR (step=30, γ=0.5)
  Early stop : patience=5, min_delta=1e-4
  Max epoch  : 100

Phase 2 — Fine-Tuning  (lanjut dari best Phase 1)
  Optimizer  : Adam, lr=1e-4
  Scheduler  : CosineAnnealingLR (T_max=20, η_min=1e-6)
  Early stop : patience=3
  Max epoch  : 20
```

Checkpoint `best.pth` di-overwrite setiap kali val_loss baru lebih rendah.

### 5.2 Multi-Seed Training

```python
SEEDS = [42, 123, 2026, 7, 31337]   # 5 run independen
```

#### Kenapa Multi-Seed?

Training deep learning itu **stokastik** (acak). Walaupun arsitektur, data, dan hyperparameter identik, hasil metrik bisa berbeda antar run karena:

| Sumber Randomness | Dikontrol oleh Seed? |
|---|---|
| Bobot awal model (weight init) | ✅ Ya |
| Urutan shuffle DataLoader | ✅ Ya |
| Augmentasi (rotasi, jitter, dropout) | ✅ Ya |
| BatchNorm running statistics | ✅ Ya |

**Satu seed = satu "jalan hidup" model.** Untuk dataset kecil (11 subjek), varians antar run sangat tinggi. Satu seed bisa kebetulan bagus (keberuntungan), seed lain bisa kebetulan buruk (sial). Multi-seed memberikan estimasi **mean ± std** yang lebih valid daripada mengandalkan satu angka tunggal.

#### Kenapa Angka-Angka Itu?

Secara matematis: **seed adalah sembarang.** Tidak ada rumus ajaib. Yang penting:
1. Berbeda satu sama lain (tidak duplikat)
2. Reproducible (run ulang dengan seed sama = hasil identik)
3. Cukup banyak untuk estimasi varians

Filosofi pemilihan:

| Seed | Alasan Pemilihan |
|---|---|
| **42** | Convention populer ML (referensi *Hitchhiker's Guide to the Galaxy*) |
| **123** | Angka sederhana yang naik berurutan — mudah diingat |
| **2026** | Tahun saat ini (relevan dengan waktu penelitian) |
| **7** | Angka keberuntungan / prima kecil — sering dipakai sebagai seed "baseline" |
| **31337** | *Leetspeak* dari "ELITE" — culture reference di programming community |

Kalau diganti jadi `[1, 2, 3, 4, 5]` pun **sah-sah saja**. Hasil mean ± std-nya akan mirip. Yang penting konsisten.

#### Berapa Seed Ideal?

| Jumlah | Kegunaan | Keterangan |
|---|---|---|
| 1 | Smoke test, debug | Tidak bisa estimasi varians |
| 3 | Bare minimum paper | Std masih tidak stabil |
| **5** | **Standar tesis S2** | Balance cost vs confidence ✅ |
| 10 | Robust, high confidence | 2× lebih mahal computasi |
| 30+ | Very confident interval | Hampir never worth it untuk tesis |

#### Seed Harus SAMA untuk with_geom vs no_geom

Ini **krusial** untuk perbandingan valid (apple-to-apple):

```
            Seed 42    Seed 123   Seed 2026   Seed 7     Seed 31337
with_geom   0.93       0.94       0.92        0.93       0.91
no_geom     0.87       0.88       0.86        0.87       0.85
Delta       +0.06      +0.06      +0.06       +0.06      +0.06
```

Kalau seed beda-beda antar varian, delta bisa bias karena keberuntungan seed, bukan karena arsitektur. Oleh karena itu `SPLIT_SEED` (data split) dan `SEEDS` (training stochasticity) **harus identik** untuk kedua varian.

#### Output

Hasil tiap seed disimpan di:
```
runs/with_geom/{timestamp}/seed_{42,123,2026,7,31337}/
├── best.pth
├── train_log.csv
└── normalizer.json
```

Agregasi mean ± std dilakukan di `evaluate.ipynb` dan `compare.ipynb`.

### 5.3 Optimasi GPU (A100-ready)

| Optimasi | Setting | Dampak |
|---|---|---|
| Mixed Precision | `torch.cuda.amp.GradScaler()` | ~2× speedup (FP16) |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = True` | Auto-tune conv kernel |
| TF32 matmul | `torch.backends.cuda.matmul.allow_tf32 = True` | A100 TensorCore aktif |
| TF32 cuDNN | `torch.backends.cudnn.allow_tf32 = True` | A100 TensorCore aktif |
| Persistent workers | `DataLoader(persistent_workers=True)` | Hilangkan overhead worker re-spawn |
| Batch size | `256` (dari 128) | Manfaatkan 80GB VRAM A100 |
| `torch.compile` | `USE_COMPILE=False` (opsional) | 5–15% speedup, off default karena custom ops |

### 5.4 Augmentasi

**Point cloud (`PointCloudAugmentor`)**:
1. **Z-rotation**: ±15° (default) **atau** ±90° (probabilitas 30%) — simulasi tangan diputar
2. **X-tilt**: ±15° (probabilitas 50%) — ujung jari mendekat/menjauh kamera
3. **Y-tilt**: ±15° (probabilitas 50%) — tepi jempol/kelingking mendekati kamera
4. **Jitter**: Gaussian σ=0.01 pada XYZ
5. **Scale**: uniform [0.9, 1.1]
6. **Point dropout**: 5% titik dihilangkan + re-sample
7. **XY translation**: ±2 cm (probabilitas 50%) — simulasi palm tidak tepat di tengah frame *(baru)*

**Geometry (`GeometryAugmentor`)**:
- Gaussian noise σ=0.02 (dalam unit z-score = ±2% std) pada vektor 14-dim
- Hanya pada training, tidak pada val/test

---

## 6. Pipeline Evaluasi

### 6.1 Standar Test (`evaluate.ipynb`)
Load `splits.json['test']` (2 sesi/subjek):
- **Gallery**: 1 sesi → embed semua frame → average → L2-normalize → 1 vektor per subjek
- **Probe**: 1 sesi lainnya → embed setiap frame → 1 vektor per frame
- Match: cosine similarity probe vs gallery

**Metrik verifikasi (pair-based)**:
- EER (Equal Error Rate)
- AUC (Area Under ROC)
- TAR @ FAR=1%, TAR @ FAR=0.1%
- d-prime
- Accuracy @ EER threshold

**Metrik identifikasi (1:N)**:
- Rank-1 accuracy
- Rank-5 accuracy
- mAP (mean Average Precision)
- Per-subject breakdown
- CMC curve

### 6.2 Holdout Real Test (terintegrasi di `evaluate.ipynb`)
Load `splits.json['holdout_probes']` (3 frame × 11 subjek = 33 probe):
- Sesi holdout **tidak pernah** dilihat model di seluruh pipeline training
- Probe ini paling dekat dengan kondisi *real-world deployment*
- Hasil dilaporkan terpisah dari standar test untuk transparansi

### 6.3 Statistical Inference (`compare.ipynb`)
Membandingkan `with_geom` vs `no_geom` (5 seed paired):
- **Paired t-test / Wilcoxon signed-rank**: pada per-seed Rank-1, EER, dst.
- **McNemar test**: pada per-probe correctness (binary)
- **Bootstrap CI**: confidence interval untuk delta metric
- **Test set fingerprint**: hash deterministik dari probe paths untuk verifikasi kedua varian dievaluasi di test set identik

---

### 6.4 Ablation Study: `with_geom` vs `no_geom`

Ini adalah **controlled ablation** — satu-satunya variabel yang diubah adalah **kontribusi fitur geometri**. Semua yang lain dikontrol (controlled) agar perbandingan valid.

#### Hipotesis

> *Penambahan fitur geometri anatomis (14 fitur dari `geometry.json`) ke PointNet++ meningkatkan akurasi identifikasi 3D palm secara signifikan.*

#### Perbedaan Eksak: `train.ipynb` vs `train_no_geom.ipynb`

| Aspek | `train.ipynb` (with_geom) | `train_no_geom.ipynb` (no_geom) | Alasan Perbedaan |
|---|---|---|---|
| **`USE_GEOM`** | `True` | `False` | **Variabel kontrol utama** |
| **Model** | `GeoAtt-PointNet++` (SA + GAM + GeometryEncoder) | `PointNet++ murni` (SA saja) | `use_geom=False` mematikan `GeometryEncoder` dan `GAM` |
| **`GEOM_DIM`** | `14` | `0` | Dimensi fitur geometri dari `geometry.json` |
| **Geometry Normalizer** | ✅ `GeometryNormalizer` (fit di training set) | ❌ `None` | Tanpa geometri = tidak perlu normalisasi |
| **Geometry Augmentor** | ✅ `GeometryAugmentor(σ=0.02)` | ❌ `None` | Tanpa geometri = tidak perlu augmentasi fitur |
| **Output path** | `runs/with_geom/` | `runs/no_geom/` | Agar hasil tidak overwrite |
| **Point cloud augmentasi** | ✅ Sama (rotasi, tilt, jitter, scale, dropout, XY translation) | ✅ Sama | **Identik** — hanya branch geometri yang beda |
| **Loss function** | ✅ `OnlineTripletLoss` (batch-hard) | ✅ Sama | **Identik** |
| **Hyperparameter** | ✅ `_auto_config()` (BATCH_SIZE, LR, FRAME_REPEAT, dll.) | ✅ Sama | **Identik** — auto-detect dari hardware |
| **`SPLIT_SEED`** | `42` | `42` | **HARUS identik** — split data persis sama |
| **`SEEDS`** | `[42, 123, 2026, 7, 31337]` | `[42, 123, 2026, 7, 31337]` | **HARUS identik** — stochasticity sama |

#### Perbedaan Eksak: `evaluate.ipynb` vs `evaluate_no_geom.ipynb`

| Aspek | `evaluate.ipynb` (with_geom) | `evaluate_no_geom.ipynb` (no_geom) | Alasan Perbedaan |
|---|---|---|---|
| **`USE_GEOM`** | `True` | `False` | Sesuai model yang dievaluasi |
| **Checkpoint source** | `runs/with_geom/` | `runs/no_geom/` | Load model hasil training varian masing-masing |
| **Geometry Normalizer** | ✅ Load dari `seed_{S}/normalizer.json` | ❌ `None` | Model no_geom tidak punya normalizer |
| **Output path** | `eval_results/with_geom/` | `eval_results/no_geom/` | Agar hasil tidak overwrite |
| **Metrik** | ✅ Rank-1, Rank-5, mAP, EER, AUC, d-prime | ✅ Sama | **Identik** |
| **Test fingerprint** | ✅ Hash dari probe paths | ✅ Sama | **Identik** — dievaluasi di data yang sama |

#### Ilustrasi Arsitektur

```
with_geom (GeoAtt-PointNet++):
  Point Cloud (N, 6) ──→ SA1 ──→ GAM1 ──→ SA2 ──→ GAM2 ──→ SA3 ──→ [256]
  Geometry (14) ───────→ GeometryEncoder ─────────────────────→ [64] ──┐
                                                                        ├──→ Fusion ──→ 128-dim
                                                                        
no_geom (PointNet++ baseline):
  Point Cloud (N, 6) ──→ SA1 ──→ SA2 ──→ SA3 ──→ [256] ──→ Fusion ──→ 128-dim
  (tidak ada geometry branch sama sekali)
```

#### Controlled Variables (APA yang HARUS Identik)

Kalau ada perbedaan tidak disengaja antar varian, hasil comparison menjadi **tidak valid** (confounding variable):

| Variabel | Konsekuensi Kalau Beda |
|---|---|
| `SPLIT_SEED` | Data test tidak sama → perbandingan tidak adil |
| `SEEDS` | Keberuntungan seed beda → delta bias |
| `BATCH_SIZE` | LR effective beda → konvergensi berbeda |
| `FRAME_REPEAT` | Volume data beda → generalisasi berbeda |
| `N_POINTS` | Resolusi input beda → kapasitas model beda |
| `TRIPLET_MARGIN` | Loss landscape beda → optimal point berbeda |
| `Dropout` | Regularisasi beda → overfit level berbeda |
| **Augmentasi** | Robustness training beda → generalisasi berbeda |

#### Independent Variable (APA yang HARUS Beda)

Hanya satu — **`USE_GEOM`**:

| `USE_GEOM=True` | `USE_GEOM=False` |
|---|---|
| `encoder.geom_encoder` aktif | `encoder.geom_encoder` tidak ada |
| `encoder.gam1`, `gam2` aktif | `encoder.gam1`, `gam2` tidak ada |
| `encoder.proj` input = 320 (256+64) | `encoder.proj` input = 256 |
| Fusion head melihat geometri + point cloud | Fusion head hanya melihat point cloud |

#### Contoh Hasil yang Diharapkan

```
            Seed 42    Seed 123   Seed 2026   Seed 7     Seed 31337   Mean±Std
with_geom   0.93       0.94       0.92        0.93       0.91         0.926±0.012
no_geom     0.87       0.88       0.86        0.87       0.85         0.866±0.011
Delta       +0.06      +0.06      +0.06       +0.06      +0.06        +0.060±0.003
```

Kalau `with_geom` secara konsisten lebih tinggi di semua 5 seed → **hipotesis terbukti**. Signifikansi statistik diuji via paired t-test / McNemar di `compare.ipynb`.

---

## 7. Struktur Repository

```
3DCNN/
├── models/
│   ├── encoder.py         # GeoAttPointNetEncoder (+ Dropout di fusion head)
│   ├── gam.py             # GeometricAttentionModule
│   ├── geometry_encoder.py
│   ├── pointnet_utils.py  # SetAbstraction, FPS, BallQuery
│   └── siamese.py         # SiamesePalmNet wrapper (+ encode() method)
│
├── losses/
│   ├── contrastive.py     # ContrastiveLoss (legacy)
│   └── triplet.py         # OnlineTripletLoss + batch-hard mining (BARU)
│
├── utils/
│   ├── dataset.py         # PalmPairDataset, PalmFrameDataset (BARU),
│   │                      # scan_dataset_frames(filter_invalid), 
│   │                      # split_holdout_sessions, split_sessions_three_way
│   ├── augmentation.py    # PointCloudAugmentor (+ XY translation), GeometryAugmentor
│   ├── normalizer.py      # GeometryNormalizer (z-score)
│   └── metrics.py         # EER, AUC, Rank-N, mAP, paired_test, mcnemar, bootstrap_ci
│
├── collab/
│   ├── train.ipynb            # GeoAtt-PointNet++ (with_geom)
│   ├── train_no_geom.ipynb    # PointNet++ baseline (no_geom)
│   ├── evaluate.ipynb         # Evaluasi with_geom + holdout
│   ├── evaluate_no_geom.ipynb # Evaluasi no_geom + holdout
│   └── compare.ipynb          # Paired statistical comparison
│
├── train.py               # CLI training (+ _run_epoch_triplet)
├── evaluate.py            # CLI evaluation
└── DOCUMENTATION.md       # File ini
```

---

## 8. Hyperparameter Default

| Kategori | Parameter | Nilai |
|---|---|---|
| **Data split** | `BALANCE_DATASET` | True |
| | `HOLDOUT_SESSIONS` | 1 sesi/subjek |
| | `HOLDOUT_FRAMES` | 3 probe/subjek |
| | `TRAIN_RATIO` | 0.70 |
| | `VAL_RATIO` | 0.15 |
| | `SPLIT_SEED` | 42 (fixed) |
| **Model** | `N_POINTS` | **auto-detect** — T4: 4096, A100: 8192 |
| | `GEOM_DIM` | 14 |
| | `Embedding dim` | 128 (L2-normalized) |
| | `Dropout` | 0.3 (fusion head) |
| **Loss** | `LOSS_FN` | `'triplet'` |
| | `TRIPLET_MARGIN` | 0.3 |
| | `FRAME_REPEAT` | **auto-detect** — T4: 10, RAM≥64G: 20, RAM≥128G: 30 |
| **Training** | `SEEDS` | [42, 123, 2026, 7, 31337] |
| | `EPOCHS` | 100 (Phase 1) |
| | `FINETUNE_EPOCHS` | 20 (Phase 2) |
| | `LR` / `FINETUNE_LR` | **auto-detect** — T4: 1e-3/1e-4, A100: 2e-3/2e-4 |
| | `BATCH_SIZE` | **auto-detect** — T4: 128/256, A100 40GB: 384/768, A100 80GB: 512/1024 |
| | `NUM_WORKERS` | **auto-detect** — T4: 2, A100: 8 |
| | `PATIENCE` | 5 / 3 |
| **Augmentasi** | `LARGE_ROTATION_PROB` | 0.3 (rotasi Z ±90°) |
| | `TILT_PROB` | 0.5 (X dan Y, independen) |
| | `TRANSLATE_PROB` | 0.5 (XY shift) |
| | `TRANSLATE_RANGE` | 0.02 (±2 cm) |
| **GPU** | `USE_AMP` | True (FP16) |
| | TF32 matmul/cuDNN | True (A100) |
| | `persistent_workers` | True |
| | `USE_COMPILE` | **auto-detect** — A100: True, T4: False |
| **RAM** | `preload_augment` | **auto-detect** — True kalau RAM ≥32 GB + VRAM ≥16 GB |

---

## 9. Riwayat Optimasi (Apa yang Sudah Dikerjakan)

### 9.1 Iterasi 1 — Fixed Split + Holdout + Augmentasi Robust
- Migrasi dari LOSO cross-validation ke **fixed train/val/test split** dengan multi-seed
- Tambah **session-level holdout**: 1 sesi/subjek dikecualikan sepenuhnya, 3 frame jadi probe real test
- Filter sesi invalid otomatis berdasarkan `is_valid` di `geometry.json`
- Tambah augmentasi rotasi Z ±90° dan tilt X/Y ±15° untuk handle tangan miring

### 9.2 Iterasi 2 — Optimasi Loss, Regularisasi, A100, Augmentasi
**Loss function**: ganti dari ContrastiveLoss (random pairs) ke **OnlineTripletLoss + batch-hard mining**. Random pairs di dataset kecil terdominasi oleh impostor "mudah" → gradient lemah. Batch-hard mining memilih kasus tersulit di tiap batch.

**Regularisasi**: tambah `Dropout(p=0.3)` di fusion head encoder. Krusial untuk dataset kecil (11 subjek) yang rentan overfitting.

**GPU optimization (auto-detect)**:
- `nvidia-smi` + `/proc/meminfo` auto-detect → `_AUTO_BS`, `_AUTO_NW`, `_AUTO_LR`, `_AUTO_FLR`, `_AUTO_N_PTS`, `_AUTO_COMPILE`, `_AUTO_PRELOAD`, `_AUTO_REPEAT`
- Heuristic:
  - ≥75 GB VRAM → `BATCH_SIZE=512/1024`, `N_POINTS=8192`, `NUM_WORKERS=8`, LR 2e-3/2e-4, `torch.compile=True`
  - ≥35 GB VRAM → `BATCH_SIZE=384/768`, `N_POINTS=8192`, `NUM_WORKERS=8`, LR 2e-3/2e-4, `torch.compile=True`
  - <35 GB VRAM  → `BATCH_SIZE=128/256`, `N_POINTS=4096`, `NUM_WORKERS=2`, LR 1e-3/1e-4, `torch.compile=False`
  - CPU           → `BATCH_SIZE=32`, `N_POINTS=2048`, `NUM_WORKERS=0`
- `FRAME_REPEAT` auto-scale: RAM≥128G → 30, RAM≥64G → 20, else → 10
- TF32 enabled (`matmul.allow_tf32`, `cudnn.allow_tf32`) — efektif hanya di A100
- `persistent_workers=True` (hilangkan overhead worker spawn tiap epoch)

**Augmentasi baru**: XY translation ±2 cm untuk simulasi palm tidak tepat di tengah frame.

**Preload augment (RAM optimization)**:
- `PalmFrameDataset` mendukung `preload_augment=True` — precompute semua augmented variant di RAM
- Worth it kalau system RAM besar (≥32 GB): training lebih cepat 20-40% karena CPU tidak perlu augmentasi on-the-fly
- Estimasi RAM: 1.100 frame × repeat(30) × 8.192 pts × 6 dim × 4 byte ≈ **~6.5 GB** (masih sangat aman untuk 176 GB)
- Auto-enable via `_auto_config()` berdasarkan deteksi `/proc/meminfo`

**Konsistensi**: `train_no_geom.ipynb` mendapat **perubahan identik** dengan `train.ipynb` (kecuali `USE_GEOM=False`) agar perbandingan dengan `with_geom` tetap apple-to-apple.

### 9.3 Iterasi 3 — Integrasi Holdout Evaluation
- Holdout eval tidak lagi notebook terpisah (`evaluate_holdout.ipynb` dihapus)
- Diintegrasi ke `evaluate.ipynb` dan `evaluate_no_geom.ipynb`
- Output: standar test metrics + holdout metrics + delta antar keduanya

---

## 10. Verifikasi & Smoke Test

### 10.1 Smoke Test (1 seed, 3 epoch)
Ubah di cell konfigurasi:
```python
SEEDS = [42]
EPOCHS = 3
FINETUNE_EPOCHS = 1
```
Jalankan notebook. Cek:
1. Tidak ada error
2. `train_loss` dan `val_loss` turun monoton (atau hampir)
3. File output: `best.pth`, `normalizer.json`, `train_log.csv`, `training_curves.png`, `splits.json`

### 10.2 Verifikasi Komponen
**Dropout aktif saat training**:
```python
model.train()
e1 = model.encode(pts, geom)
e2 = model.encode(pts, geom)
# e1 != e2 → Dropout aktif

model.eval()
e1 = model.encode(pts, geom)
e2 = model.encode(pts, geom)
# e1 == e2 → Dropout off
```

**TripletLoss bekerja**:
```python
# Dummy batch: 4 subjek × 4 frame
emb = torch.randn(16, 128); emb = F.normalize(emb, dim=1)
labels = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3])
loss = OnlineTripletLoss(margin=0.3)(emb, labels)
# loss > 0 jika margin belum terpenuhi
```

**A100 TF32 aktif**:
```python
print(torch.backends.cuda.matmul.allow_tf32)  # True
print(torch.backends.cudnn.allow_tf32)         # True
```

### 10.3 Verifikasi Pipeline End-to-End
1. **Train** → `best.pth` + `splits.json` tersimpan
2. **Evaluate** → load `best.pth`, hitung Rank-1/EER pada test + holdout
3. **Compare** → load runs from both variants, paired t-test → p-value

---

## 11. Keterbatasan & Catatan untuk Tesis

### 11.1 Dataset
- **11 subjek** sangat kecil untuk standar biometrik (literatur: 100–500 subjek)
- Implikasi: hasil sebaiknya dilaporkan sebagai **proof-of-concept**, bukan production-ready
- Holdout real test (33 probe) terlalu kecil untuk klaim generalisasi kuat — perlu disebut eksplisit
- **Rekomendasi tesis**: dokumentasi pengakuan keterbatasan + perbandingan dengan dataset publik yang lebih besar (TouchDIP, MPD, dll.)

### 11.2 Checkpoint Compatibility
Penambahan `Dropout` di fusion head menambah parameter `nn.Dropout`. **Checkpoint dari pipeline lama TIDAK kompatibel** dengan kode baru — perlu training ulang.

### 11.3 Val Loss Scale
Triplet loss biasanya jauh lebih kecil dari contrastive loss (0.01–0.1 vs 0.1–0.5). Ini normal — yang penting val_loss turun konsisten. Early stopping bekerja pada perbedaan relatif.

### 11.4 Sinkronisasi ke Google Drive
File lokal yang diupdate **harus di-upload ulang ke Drive** sebelum dijalankan di Colab:
- `utils/dataset.py`, `utils/augmentation.py`, `utils/metrics.py`
- `models/encoder.py`, `models/siamese.py`
- `losses/triplet.py` (file baru)
- `train.py`
- `collab/train.ipynb`, `collab/train_no_geom.ipynb`
- `collab/evaluate.ipynb`, `collab/evaluate_no_geom.ipynb`

---

## 12. Referensi Cepat

### Quick Start (Colab)
```bash
1. Upload semua file lokal ke /content/drive/MyDrive/3DCNN/
2. Buka collab/train.ipynb → Run all cells
3. Setelah selesai, buka collab/evaluate.ipynb
4. Edit TRAIN_RUN_DIR ke output training terbaru → Run all cells
5. Bandingkan kedua varian via collab/compare.ipynb
```

### Quick Smoke Test
```python
# di cell konfigurasi train.ipynb
SEEDS = [42]
EPOCHS = 3
FINETUNE_EPOCHS = 1
LOSS_FN = 'triplet'
```

### Reset ke Contrastive Loss
```python
LOSS_FN = 'contrastive'  # akan pakai PalmPairDataset + ContrastiveLoss
```

### Disable XY Translation
```python
TRANSLATE_PROB = 0.0
```

---

**Dokumen terakhir diperbarui:** 16 Mei 2026
