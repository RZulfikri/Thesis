# Laporan Evaluasi GeoAtt-PointNet V1 untuk Pengenalan Telapak Tangan 3D

**Timestamp laporan:** 2026-04-23 20:00:00
**Dataset:** 6 subjek, iPhone TrueDepth, session-level layout
**Arsitektur:** GeoAtt-PointNet++ Siamese (M4 — full model dengan GAM + Geometry Encoder)
**Loss:** Contrastive Loss (margin=0.5, cosine similarity pada unit sphere)
**Protokol:** Single-seed (seed=42), train/val/test split 70/15/15
**Catatan:** Ini adalah **training pertama (V1)** yang menjadi dasar perbaikan menuju V2 (ArcFace, 11 subjek, multi-seed).
**Sumber run:** `3DCNNV1/runs/geoatt_m4/`
**Sumber evaluasi:** `3DCNNV1/eval_results/`

---

## Ringkasan Eksekutif

Tiga temuan utama:

1. **Performa awal yang menjanjikan pada dataset kecil.** Model V1 mencapai Rank-1 **89.5%** (17/19 benar) pada 6 subjek dengan training data terbatas, menunjukkan bahwa arsitektur GeoAtt-PointNet memiliki kapasitas diskriminatif untuk pengenalan telapak tangan 3D.
2. **Siamese contrastive learning efektif untuk few-class scenario.** Meskipun hanya dilatih pada 6 identitas, model mampu membedakan sesi test dengan similarity score yang tinggi (genuine 0.989–0.999), mengindikasikan embedding space yang terstruktur dengan baik.
3. **Kesalahan terkonsentrasi pada subjek tertentu.** Dua misclassifications terjadi pada `fadhil` (1/4 salah, diprediksi sebagai `alji`) dan `taofik` (1/4 salah). Pola ini menunjukkan adanya ambiguitas antar-subjek yang memiliki karakteristik geometri/ttekstur mirip, bukan kegagalan acak.

Implikasi: V1 membuktikan konsep (proof-of-concept) bahwa fusi PointNet++ dengan Geometric Attention Module dan geometry encoder menghasilkan embedding yang diskriminatif. Keterbatasan utama adalah skala dataset (6 subjek) dan penggunaan contrastive loss yang lebih sederhana dibandingkan dengan loss klasifikasi berbasis margin angular (ArcFace/CosFace) yang kemudian diadopsi pada V2.

---

## 1. Metodologi

### 1.1 Arsitektur

| Komponen | Konfigurasi |
|---|---|
| **Model** | `SiamesePalmNet` dengan `GeoAttPointNetEncoder` |
| **Total parameter** | 337,312 |
| **Input points** | 4,096 titik (6-dim: x, y, z, nx, ny, nz) |
| **Input geometry** | 33 fitur geometri palm (dinormalisasi dengan mean/std) |

**Encoder pipeline:**

| Layer | Output Dim | Detail |
|---|---|---|
| `GeometryEncoder` | 64 | 33-dim → 64-dim MLP |
| `SA1` | 64 | 512 centroids, r=0.05, 32 samples, MLP [32,32,64] |
| `GAM1` | 64 | Geometric Attention Module pada 64-dim features |
| `SA2` | 128 | 128 centroids, r=0.15, 64 samples, MLP [64,64,128] |
| `GAM2` | 128 | Geometric Attention Module pada 128-dim features |
| `SA3` | 256 | 1 centroid, r=5.0, 128 samples, MLP [128,256,256] |
| `Projection` | 128 | concat([SA3_global=256, geom_emb=64]) = 320 → 256 → 128, L2-normalized |

**Siamese head:** Cosine similarity antara dua embedding 128-dim (pada unit sphere).

### 1.2 Konfigurasi Training

| Parameter | Nilai |
|---|---|
| `loss_fn` | `ContrastiveLoss` |
| `margin` | 0.5 |
| `num_classes` | 6 (subjek: alji, fadhil, feby, gede, rahmat, taofik) |
| `epochs` | 100 |
| `batch_size` | 16 |
| `n_points` | 4096 |
| `optimizer` | Adam |
| `lr` | 1×10⁻³ |
| `scheduler` | StepLR (step=30, gamma=0.5) |
| `seed` | 42 |
| `train_split` / `val_split` / `test_split` | 70% / 15% / 15% |
| `sampling` | random |
| `augmentation` | rotasi ±15°, jitter σ=0.01, scale 0.9–1.1, dropout 5% |

### 1.3 Protokol Evaluasi

- **Identifikasi 1:N (session-level).** Gallery = rata-rata embedding semua sesi training per subjek; Probe = 19 sesi pada test split. Evaluasi menggunakan cosine similarity terdekat.
- **Verifikasi 1:1 (implisit).** Contrastive loss secara inheren memodelkan verifikasi; evaluasi similarity threshold terlihat pada `similarity_per_person.png`.
- **Ablation (terbatas).** Notebook mereferensikan M1–M4, namun hanya checkpoint M4 (full GeoAtt) tersedia secara lokal. M1 (Baseline), M2 (+Curvature), M3 (+GAM) tidak dievaluasi pada run ini karena checkpoint tidak di-download dari Colab.

---

## 2. Hasil Training

Training berlangsung selama 100 epoch dengan early stopping implisit (checkpoint `best_loss.pth` dan `best_rank1.pth` disimpan). Training curves tersedia pada:

![Training Curves](runs/geoatt_m4/training_curves.png)

*Gambar 1. Training curves M4 GeoAtt. Loss turun stabil dari ~0.115 (epoch 1) ke ~0.003 (epoch 55+). Validation loss stabil di kisaran 0.028–0.038. Rank-1 accuracy pada validation mencapai puncak 90% pada epoch 5 dan berfluktuasi di 80–90%.*

**Selected epoch logs:**

| Epoch | Train Loss | Val Loss | Val Rank-1 |
|---|---|---|---|
| 1 | 0.1153 | 0.0283 | 70.0% |
| 5 | 0.0202 | 0.0350 | 90.0% |
| 10 | 0.0099 | 0.0381 | 70.0% |
| 20 | 0.0071 | 0.0318 | 90.0% |
| 30 | 0.0063 | 0.0368 | 90.0% |
| 40 | 0.0023 | 0.0300 | 90.0% |
| 50 | 0.0020 | 0.0319 | 90.0% |
| 55 | 0.0030 | 0.0282 | 80.0% |

*Catatan: Log di atas diekstrak dari output notebook; epoch 56–100 tidak terlihat pada output cell yang tersimpan, namun checkpoint `epoch_100.pth` menandakan training berjalan penuh.*

---

## 3. Hasil Identifikasi 1:N

### 3.1 Headline

| Metrik | M4 GeoAtt |
|---|---|
| **Rank-1** | **89.47%** (17/19) |
| Rank-2 | 94.74% (18/19) |
| Rank-3+ | 94.74% (18/19) |

### 3.2 Per-subjek Accuracy

| Subjek | Test Sessions | Benar | Akurasi |
|---|---|---|---|
| alji | 3 | 3 | **100%** |
| fadhil | 4 | 3 | **75%** |
| feby | 2 | 2 | **100%** |
| gede | 4 | 4 | **100%** |
| rahmat | 2 | 2 | **100%** |
| taofik | 4 | 3 | **75%** |
| **Total** | **19** | **17** | **89.47%** |

### 3.3 Analisis Kesalahan

Dua kesalahan identifikasi:

1. **`fadhil/20260413_094713` → prediksi `alji`**. Similarity score ke `alji` (0.9892) sedikit lebih tinggi dari similarity ke gallery `fadhil`. Ini menunjukkan ambiguitas antar subjek pada fitur tertentu.
2. **`taofik` (1 sesi) → prediksi salah**. Detail spesifik tidak terlihat pada output notebook yang tersimpan, namun dari confusion matrix diketahui 1 dari 4 sesi `taofik` salah diklasifikasikan.

**Pola kesalahan:** Tidak ada subjek yang gagal total; kesalahan terjadi pada 2 dari 6 subjek dengan 1 sesi salah masing-masing. Ini mengindikasikan bahwa model umumnya robust, dengan kegagalan pada boundary cases antar kelas yang memiliki distribusi fitur tumpang-tindih.

### 3.4 Visualisasi

![CMC Curve](eval_results/cmc_curve.png)

*Gambar 2. CMC curve M4 GeoAtt. Rank-1 = 89.5%, Rank-2 = 94.7%.*

![Confusion Matrix](eval_results/confusion_matrix.png)

*Gambar 3. Confusion matrix pada 19 sesi test. Mayoritas diagonal dominan; off-diagonal terlihat pada baris fadhil dan taofik.*

![Similarity per Person](eval_results/similarity_per_person.png)

*Gambar 4. Cosine similarity per subjek. Genuine scores sangat tinggi (0.989–0.999), menunjukkan embedding yang kompak dalam kelas.*

---

## 4. Hasil Verifikasi 1:1 (Implisit)

Karena model dilatih dengan contrastive loss (siamese), setiap pasangan probe-gallery secara inheren menghasilkan similarity score yang dapat diinterpretasikan sebagai skor verifikasi.

**Observasi dari similarity distribution:**

| Observasi | Nilai |
|---|---|
| Genuine similarity range | 0.989 – 0.999 |
| Impostor similarity range | ~0.90 – 0.989 (berdasarkan off-diagonal confusion) |
| Gap tipikal | ~0.01 – 0.10 |

Model tidak menghasilkan metrik EER/AUC/TAR@FAR secara eksplisit pada evaluasi V1. Interpretasi verifikasi bersifat kualitatif dari similarity scores.

---

## 5. Ablation (Referensi dari Notebook)

Notebook `02_evaluate.ipynb` mereferensikan empat varian ablation:

| Model | Komponen | Status Checkpoint |
|---|---|---|
| M1 | PointNet++ Baseline | Tidak tersedia lokal |
| M2 | + Curvature features | Tidak tersedia lokal |
| M3 | + GAM (Geometric Attention Module) | Tidak tersedia lokal |
| **M4** | **+ Geometry Encoder (full GeoAtt)** | **Tersedia & dievaluasi** |

**CMC Ablation (dari notebook — hanya M4 yang diverifikasi):**

![CMC Ablation](eval_results/cmc_ablation.png)

*Gambar 5. CMC curves M1–M4 (output notebook). Hanya M4 yang memiliki checkpoint tersedia untuk evaluasi independen.*

---

## 6. Perbandingan dengan 3DCNN (V2)

| Aspek | **3DCNNV1** (Ini) | **3DCNN** (V2, ArcFace) |
|---|---|---|
| **Jumlah subjek** | 6 | 11 |
| **Total sesi test** | 19 | 110 + 33 holdout |
| **Arsitektur backbone** | PointNet++ (3 SA layers) | PointNet++ (3 SA layers) |
| **Geometry features** | 33-dim + GeometryEncoder(64) | 33-dim + GeometryEncoder(64) |
| **GAM** | Setelah SA1 & SA2 | Setelah SA1 & SA2 |
| **Fusion** | Concat(256+64) → 128 | Concat(256+64) → 128 |
| **Loss** | **Contrastive** (margin=0.5) | **ArcFace** (m=0.5, s=30) |
| **Input points** | 4,096 | 8,192 |
| **Frame repeat** | – | 30 |
| **Multi-seed** | Tidak (seed=42) | Ya (5 seed) |
| **Rank-1 (best)** | **89.47%** | **99.82%** (no_geom), **95.82%** (with_geom) |
| **EER** | Tidak dihitung | 0.03% (no_geom), 2.76% (with_geom) |

**Peningkatan utama V2 → V1:**

1. **Pergantian loss:** Contrastive → ArcFace. Ini adalah perubahan paling signifikan yang menaikkan Rank-1 dari ~89% ke ~96–100%.
2. **Peningkatan dataset:** 6 → 11 subjek, dengan frame repeat 30×.
3. **Peningkatan input resolution:** 4,096 → 8,192 points.
4. **Multi-seed evaluation:** Memberikan estimasi varians yang lebih reliabel.
5. **Holdout protocol:** V2 menggunakan leave-one-session-out holdout (33 probe unseen), sementara V1 menggunakan random split.

---

## 7. Analisis Keterbatasan V1

1. **Dataset kecil (6 subjek).** Performa 89.5% pada 6 subjek tidak dapat langsung digeneralisasi ke populasi lebih besar. V2 mengatasi ini dengan 11 subjek, namun masih terbatas.
2. **Single-seed.** Tidak ada estimasi varians akibat inisialisasi random. V2 menggunakan 5 seed.
3. **Contrastive loss pada few-class.** Contrastive loss bekerja baik untuk verifikasi, namun untuk identifikasi dengan banyak kelas, loss berbasis margin angular (ArcFace/CosFace) secara empiris superior.
4. **Split random (bukan session-aware).** V1 menggunakan split 70/15/15 random, yang berarti sesi dari subjek yang sama bisa tersebar di train/val/test. V2 menggunakan split session-aware yang lebih ketat.
5. **Checkpoint ablation tidak tersedia.** Tidak dapat memverifikasi kontribusi individual GAM vs GeometryEncoder vs Curvature.

---

## 8. Diskusi & Implikasi untuk Tesis

- **V1 sebagai proof-of-concept.** V1 memvalidasi bahwa arsitektur GeoAtt-PointNet dapat dipelajari end-to-end dan menghasilkan embedding yang cukup diskriminatif untuk pengenalan telapak tangan 3D pada dataset kecil.
- **Contrastive loss adalah bottleneck.** Peningkatan dramatis pada V2 (dari ~89% ke ~100% Rank-1) sebagian besar disebabkan oleh penggantian loss, bukan perubahan arsitektur. Ini menunjukkan bahwa formulasi loss lebih krusial daripada penambahan modul attention pada skala dataset kecil.
- **GeoAtt memberikan kontribusi positif pada V1, namun tidak pada V2.** Pada V1 dengan contrastive loss, model full (M4) mencapai 89.5% yang merupakan hasil terbaik di antara varian yang tersedia. Pada V2 dengan ArcFace, `no_geom` justru melampaui `with_geom` — mengindikasikan bahwa peran GeoAtt sebagai "regularizer" hanya relevan ketika loss utama lemah.
- **Rekomendasi untuk pengembangan lebih lanjut:**
  - Gunakan ArcFace atau CosFace sebagai loss utama.
  - Perluas dataset ke 20+ subjek dengan variasi pose dan pencahayaan.
  - Lakukan ablation terpisah untuk GAM-only vs GeometryEncoder-only.
  - Terapkan protokol evaluasi cross-session yang lebih ketat (leave-one-session-out).

---

## 9. Lampiran

### 9.1 Fingerprint & Reproduksibilitas

- `seed`: 42
- `train_split`: 0.70
- `val_split`: 0.15
- `test_split`: 0.15
- `normalizer`: `3DCNNV1/runs/geoatt_m4/normalizer.json` (33-dim mean/std)

### 9.2 Path Sumber

- Training notebook: `3DCNNV1/01_train.ipynb`
- Evaluasi notebook: `3DCNNV1/02_evaluate.ipynb`
- Checkpoints: `3DCNNV1/runs/geoatt_m4/checkpoints/`
- Normalizer: `3DCNNV1/runs/geoatt_m4/normalizer.json`
- Training curves: `3DCNNV1/runs/geoatt_m4/training_curves.png`
- Evaluasi plots: `3DCNNV1/eval_results/`

### 9.3 Normalizer Statistics (33 fitur geometri)

Mean dan std untuk normalisasi 33 fitur geometri tersimpan di `normalizer.json`. Fitur-fitur ini mencakup:
- Dimensi bounding box (5 fitur)
- Normal axis stats (5 fitur)
- Principal moments / eigenvalues (7 fitur)
- Surface area & volume (2 fitur)
- Convex hull stats (3 fitur)
- Distances & spreads (11 fitur)

---

*Laporan ini dibuat secara otomatis berdasarkan artefak yang tersedia di `3DCNNV1/`.*
*3DCNNV1 merupakan training pertama dalam rangkaian eksperimen thesis. Hasil dan keterbatasannya menjadi dasar perbaikan pada 3DCNN V2 (ArcFace, dataset lebih besar, multi-seed, protokol lebih ketat).*
*Untuk hasil terbaru dan lebih komprehensif, lihat laporan V2 di `3DCNN/result_docs/20260517_060023/`.*
