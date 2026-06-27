# Progress Report — Identifikasi Telapak Tangan 3D Berbasis iPhone TrueDepth Camera

**Rahmat Zulfikri** · Magister Teknik Elektro UGM
Tanggal: Mei 2026

> Dokumen ini disusun dengan pola **Input → Proses → Output** per tahap pipeline,
> dirancang agar mudah dipecah menjadi slide presentasi.

---

## Slide 1 — Topik & Tujuan

**Judul:** Identifikasi Telapak Tangan 3D Berbasis iPhone TrueDepth Camera

**Tujuan:**
- Membangun sistem biometrik telapak tangan berbasis 3D, bukan 2D (citra)
- Memanfaatkan sensor TrueDepth iPhone (tersedia luas) sebagai sensor kedalaman
- Menggabungkan fitur geometri eksplisit (anatomi tangan) dengan point cloud deep learning
- Menghasilkan *embedding* 128-dim yang dapat digunakan untuk identifikasi 1-to-N

**Kontribusi (novelty):**
1. Pipeline end-to-end dari sensor konsumen (iPhone) → identitas
2. Arsitektur **GeoAtt-PointNet++** — PointNet++ dengan Geometric Attention Module (GAM)
3. Strategi **full-cloud + on-the-fly sampling** sebagai augmentasi implisit

---

## Slide 2 — Arsitektur Sistem (Big Picture)

```
┌─────────────────────────────────────────────────────────┐
│  TrueDepthScan  (iOS App)                               │
│  Rekam depth frames → export .bin + calibration.json    │
└───────────────────────────┬─────────────────────────────┘
                            │ depth*.bin + calibration.json
                            ▼
┌─────────────────────────────────────────────────────────┐
│  3DRegistration  (Python)                               │
│  ICP multi-frame → output.ply                           │
│  Ekstraksi fitur → geometry.json (33 nilai)             │
│  Normalisasi PCA + unit sphere → cnn_input.npy          │
└───────────────────────────┬─────────────────────────────┘
                            │ cnn_input.npy + geometry.json
                            ▼
┌─────────────────────────────────────────────────────────┐
│  3DCNN  (Python / Google Colab)                         │
│  GeoAtt-PointNet++ Siamese Network                      │
│  Training → 128-dim identity embedding                  │
│  Identifikasi → nearest neighbor di enrollment DB       │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 3 — Tahap 1: TrueDepthScan (iOS)

### Input
- Telapak tangan user, jarak 10–50 cm dari kamera TrueDepth iPhone
- Label subjek (misal: `rahmat`) yang diketik user sebelum scan

### Proses
1. **Sesi scan 2 detik** (countdown 3 detik sebelum dimulai)
2. **Sinkronisasi frame depth + video** via `AVCaptureDataOutputSynchronizer`
3. **Deteksi kesiapan** berbasis depth (7×7 grid, threshold 10–60 cm)
4. **Vision Hand Pose** — deteksi ROI telapak + chirality (kiri/kanan)
5. **Frame decimation** — ambil 1 frame dari tiap 5 (efektif ~6 fps)
6. **Dua filter berurutan** di `Open3DExporter.swift`:
   - Depth range [0.10–0.50 m] + ROI masking
   - Neighborhood density filter (≥3 tetangga valid, toleransi ±50 mm)
7. **Majority vote handedness** dari semua frame

### Output (per sesi scan)
Folder `Documents/[label]_YYYYMMDD_HHMMSS/`:

| File | Isi |
|---|---|
| `depth00.bin … depthNN.bin` | Raw depth Float32, 640×480, typical 10–12 frame |
| `calibration.json` | fx, fy, cx, cy, lens distortion (dari `AVCameraCalibrationData`) |
| `metadata.json` | label, handedness, frameCount, timestamp |

---

## Slide 4 — Tahap 2: 3DRegistration (Python)

### Input
Folder hasil export iOS:
```
dataset/rahmat_20260411_202327/
├── depth00.bin … depth10.bin
├── calibration.json
└── metadata.json
```

### Proses
1. **Konversi depth → point cloud** per frame menggunakan intrinsik kamera:
   `X = (u - cx) · d / fx`, `Y = (v - cy) · d / fy`, `Z = d`
2. **Registrasi ICP sekuensial** — gabungkan semua frame menjadi satu cloud
3. **Filter noise:**
   - Statistical outlier removal (`outlier_std_ratio = 1.5`)
   - Cluster connectivity DBSCAN (`eps = 0.008 m`)
   - Depth range filter
4. **Estimasi normal** (`normal_radius = 0.008 m`)
5. **Ekstraksi fitur geometri** (`extract_geometry.py`):
   - PCA alignment (jari → sumbu +Y)
   - Deteksi wrist ROI (18% bawah Y)
   - Deteksi baris knuckle (MCP)
   - Deteksi 5 ujung jari (dengan handedness sebagai prior)
   - Hitung panjang/lebar/rasio dalam mm
6. **Preprocessing untuk CNN** (`preprocess_for_cnn.py`):
   - PCA canonical alignment
   - Normalisasi ke unit sphere (`pts / max(‖pts‖)`)
   - Simpan full cloud (+ FPS 1024 sebagai backup)

### Output (per sesi)
Folder `result/[label]/[timestamp]/`:

| File | Shape | Fungsi |
|---|---|---|
| `output.ply` | ~50k–150k titik | Sumber (bisa di-regenerate yang lain) |
| `geometry.json` | 33 nilai | **Input GeometryEncoder** |
| `cnn_input.npy` | (N, 6) | **Input utama PointNet++ backbone** |
| `cnn_input_fps.npy` | (1024, 6) | Backup untuk ablation |
| `texture.npy` | (256, 256, 5) | Cadangan (belum dipakai) |

---

## Slide 5 — Fitur Geometri (33 nilai biometrik)

### Komposisi
| Field | Dim | Keterangan |
|---|---|---|
| `finger_lengths_mm` | 5 | Panjang tiap jari [ibu, telunjuk, tengah, manis, kelingking] |
| `finger_ratios` | 5 | Panjang relatif terhadap jari tengah (scale-invariant) |
| `palm_width_mm` | 1 | Lebar telapak di baris knuckle (MCP) |
| `palm_height_mm` | 1 | Tinggi telapak dari wrist ke knuckle |
| `palm_aspect_ratio` | 1 | palm_width / palm_height |
| `finger_to_palm_ratios` | 5 | Panjang jari / palm_height |
| `inter_finger_gaps_mm` | 4 | Celah kosong horizontal antar jari |
| `finger_widths_mm` | 5 | Lebar tiap jari (p5–p95 rentang X) |
| `finger_width_to_length_ratios` | 5 | Rasio bentuk tiap jari |
| `mean_palm_curvature` | 1 | Kelengkungan rata-rata telapak |

**Total:** 5+5+1+1+1+5+4+5+5+1 = **33**

### Perubahan dari versi lama (23 → 33)
- `inter_finger_depths_mm` → **`inter_finger_gaps_mm`** (celah horizontal, lebih stabil)
- **[baru]** `finger_widths_mm` — lebar anatomis jari
- **[baru]** `finger_width_to_length_ratios` — rasio bentuk jari

---

## Slide 6 — Tahap 3: GeoAtt-PointNet++ (Arsitektur Model)

### Input per sesi
- `cnn_input.npy` → sample **4096 titik** on-the-fly (N, 6) = [x, y, z, nx, ny, nz]
- `geometry.json` → vektor **33-dim**

### Encoder (shared di Siamese)
```
Point Cloud (4096, 6)              Geometry (33)
        │                                │
        ▼                                ▼
┌────────────────┐              ┌─────────────────┐
│ SA1: 512 pts   │              │ GeometryEncoder │
│ (r=0.05,ns=32) │              │ MLP 33→64→64    │
│ → (512, 64)    │◄─GAM₁────────┤                 │
└────────┬───────┘              │  g (64-dim)     │
         ▼                      │                 │
┌────────────────┐              │                 │
│ SA2: 128 pts   │              │                 │
│ (r=0.15,ns=64) │◄─GAM₂────────┤                 │
│ → (128, 128)   │              └─────────────────┘
└────────┬───────┘                       │
         ▼                               │
┌────────────────┐                       │
│ SA3: 1 pt      │                       │
│ (r=5.0,ns=128) │                       │
│ → (1, 256)     │                       │
└────────┬───────┘                       │
         │              ┌────────────────┘
         ▼              ▼
    Concat (256 + 64 = 320)
         │
         ▼
  MLP 320 → 256 → 128
         │
         ▼
   L2-normalize → embedding 128-dim
```

### Geometric Attention Module (GAM) — inti novelty
```
GAM(sa_feat, g):
    g_proj  = Linear(g → sa_ch)                       # project geom
    concat  = [sa_feat, g_proj_expand]                # B × N × 2C
    gate    = Sigmoid(MLP(concat))                    # B × N × C
    return  gate ⊙ sa_feat                            # modulasi per-point
```
Ide: **fitur geometri eksplisit memandu perhatian model** ke bagian tangan yang penting.

### Siamese + Loss
- Siamese: shared encoder, cosine similarity antara dua embedding
- **Contrastive loss** (margin = 0.5):
  - Pair genuine (label=1): minimalkan jarak
  - Pair impostor (label=0): pertahankan margin

---

## Slide 7 — Strategi Training

### Konfigurasi
| Parameter | Nilai |
|---|---|
| Epochs | 100 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Scheduler | StepLR, step=30, γ=0.5 |
| Optimizer | Adam |
| Margin | 0.5 |
| Titik per sample | 4096 (random dari full cloud) |
| Geom dim | 33 |
| Seed | 42 |

### Split data (stratified per label)
- **Train 70%** — untuk pembentukan encoder + enrollment database
- **Val 15%** — Rank-1 accuracy dicek tiap 5 epoch
- **Test 15%** — evaluasi final di notebook `02_evaluate.ipynb`

### Augmentasi (train only)
- Rotasi Z random ±15°
- Jitter Gaussian σ=0.01
- Scale random 0.9–1.1
- Dropout titik 5%

### Pair generation
- Genuine pair: kombinasi 2 sesi label sama
- Impostor pair: pasangan label berbeda (jumlah 1:1 dengan genuine)

### Pencegahan data leakage
Normalizer (mean/std) fitur geometri **di-fit hanya dari training sessions**, disimpan ke `normalizer.json`, dan dipakai sama persis saat val/test.

---

## Slide 8 — Enrollment & Identifikasi

### Alur identifikasi (saat inference)
```
1. Enrollment (sekali saat deployment):
   untuk tiap label di train_s:
       emb_list = [encoder(session) for session in train_sessions]
       database[label] = L2_normalize(mean(emb_list))

2. Query (setiap scan baru):
   emb_query = encoder(new_session)
   scores    = {label: cosine(emb_query, db_emb) for label, db_emb in database}
   predicted = argmax(scores)
```

### Metrik
- **Rank-1 Accuracy** — % query yang prediksi top-1-nya benar
- **Rank-k / CMC Curve** — % query yang label aslinya masuk top-k
- **Confusion Matrix** — baris = label asli, kolom = prediksi
- **Similarity Score per Orang** — bar chart cosine similarity ke setiap kandidat

---

## Slide 9 — Desain Ablation Study

Empat varian model untuk memvalidasi kontribusi tiap komponen:

| Model | PointNet++ | Geometry | Curvature | GAM |
|---|:---:|:---:|:---:|:---:|
| **M1 Baseline** | ✓ | — | — | — |
| **M2 +Curvature** | ✓ | 22 fitur | ✓ | — |
| **M3 +GAM** | ✓ | 22 fitur | ✓ | ✓ |
| **M4 GeoAtt** (full) | ✓ | **33 fitur** | ✓ | ✓ |

**Hipotesis:**
- M1 → M2: menambahkan fitur biometrik anatomis menaikkan Rank-1
- M2 → M3: GAM memberi kontribusi di atas concat sederhana
- M3 → M4: fitur anatomi yang lebih kaya (inter-finger gaps, finger width/ratio) menaikkan lagi

---

## Slide 10 — Status Progres Saat Ini

### Yang sudah selesai
- [x] iOS app `TrueDepthScan` — scanning, export, share-sheet
- [x] Deteksi handedness via Vision + majority vote
- [x] Neighborhood density filter (menghilangkan titik terisolasi)
- [x] Pipeline `3DRegistration` lengkap (ICP, filter, ekstraksi 33 fitur)
- [x] Preprocessing dual (`cnn_input.npy` full + `cnn_input_fps.npy` backup)
- [x] Arsitektur `GeoAtt-PointNet++` + `SiamesePalmNet`
- [x] Notebook training `01_train.ipynb` dengan cek Rank-1 per 5 epoch
- [x] Notebook evaluasi `02_evaluate.ipynb` + ablation M1–M4
- [x] Data awal: **20 sesi, 2 identitas** (rahmat & feby, @ 10 sesi)

### Yang masih dikerjakan
- [ ] Penambahan subjek (target: minimal ≥ 10 identitas)
- [ ] Eksekusi training aktual di Colab untuk M1–M4
- [ ] Plot training curve + CMC curve + confusion matrix (hasil nyata)
- [ ] Analisis error: kapan sistem salah, dan mengapa
- [ ] Uji robustness: scan dengan posisi/pencahayaan berbeda
- [ ] Penulisan bab tesis

---

## Slide 11 — Hasil Eksperimen (Placeholder)

> Akan diisi setelah training di Colab dieksekusi.

### Rank-1 Accuracy per model (ablation)
| Model | Rank-1 | Rank-3 | Rank-5 |
|---|:---:|:---:|:---:|
| M1 Baseline | TBD | TBD | TBD |
| M2 +Curvature | TBD | TBD | TBD |
| M3 +GAM | TBD | TBD | TBD |
| **M4 GeoAtt** | **TBD** | TBD | TBD |

### Figur yang akan disertakan
- `training_curves.png` — train/val loss + Rank-1 over epoch
- `cmc_curve.png` — CMC untuk M4
- `cmc_ablation.png` — CMC semua model
- `confusion_matrix.png` — error pattern
- `similarity_per_person.png` — distribusi cosine similarity per query

---

## Update Mei 2026 — Optimasi Training v0.4.0

Pada Mei 2026 dilakukan serangkaian optimasi kecepatan training untuk mengurangi waktu per-seed dari **~3 jam → ~40–60 menit** (speedup ~3–5×). Optimasi ini tidak mengubah struktur eksperimen maupun hipotesis, melainkan murni engineering improvement:

| # | Optimasi | Dampak Kecepatan | Keterangan |
|---|----------|------------------|------------|
| 1 | `ball_query`: `argsort` → `torch.topk` | 2–3× | Math identik, hemat ~24 GB VRAM sort buffer |
| 2 | `torch.compile(mode="default")` | 30–50% | Graph compilation untuk H100/Blackwell |
| 3 | bf16 mixed precision (`--amp bf16`) | 1.5–2× | Negligible noise (<0.1%) |
| 4 | Siamese concat-then-split (1 forward) | 15–25% | BN over 2B, estimator lebih stabil |
| 5 | Adam `fused=True` kernel | 5–10% | Math identik |
| 6 | DataLoader `persistent_workers` + prefetch | 5–10% | Mengurangi bottleneck data loading |

**Fair ablation tetap valid** — keempat variant (`no_geom`, `gam_only`, `fuse_only`, `with_geom`) menggunakan code path yang identik. Hipotesis utama (dampak GeoAtt terhadap akurasi) tidak terpengaruh.

File dokumentasi:
- `3DCNN/OPTIMIZATION_REPORT.md` — laporan lengkap optimasi
- `3DCNN/collab/VERSIONS.md` — perbandingan versi baseline vs optimize

---

## Slide 12 — Next Steps & Timeline

| Milestone | Deskripsi | Status |
|---|---|---|
| M1 | iOS scanning + pipeline 3DRegistration | ✓ |
| M2 | Ekstraksi 33 fitur geometri | ✓ |
| M3 | Arsitektur GeoAtt-PointNet++ | ✓ |
| M4 | Dataset awal (2 subjek × 10 sesi) | ✓ |
| **M5** | **Training M1–M4 di Colab + evaluasi** | **in-progress** |
| M6 | Penambahan subjek (≥ 10 identitas) | pending |
| M7 | Uji robustness & analisis error | pending |
| M8 | Draft bab hasil tesis | pending |

---

## Slide 13 — Kesimpulan Progres

- Pipeline **end-to-end dari iPhone ke embedding 128-dim** sudah berjalan
- Arsitektur **GeoAtt-PointNet++** siap, ablation study sudah di-setup
- Dataset proof-of-concept (2 subjek × 10 sesi) sudah tersedia
- Komponen novelty (GAM, 33 fitur biometrik, full-cloud sampling) sudah terimplementasi
- Prioritas berikutnya: **eksekusi training + pengumpulan subjek tambahan** untuk validasi statistik
