# Laporan Evaluasi — Fair Ablation GeoAtt-PointNet++

**Tanggal:** 2026-05-21  
**Dataset:** 11 subjek, frame-level layout  
**Evaluasi:** 4 variant × 5 seeds (7, 42, 123, 2026, 31337)  
**Split:** Train/Val/Test + Holdout (1 session per subjek)

---

## 1. Ringkasan Hasil

### Test Set (Mean ± Std, 5 Seeds)

| Variant   | EER (↓)      | AUC (↑)      | TAR@FAR=1% (↑) | d' (↑)       | Acc@EER (↑)  |
|-----------|-------------|-------------|---------------|-------------|-------------|
| no_geom   | 0.17 ± 0.26%| 0.9995±0.001| 99.85 ± 0.27% | 2.06 ± 0.14 | 99.83 ± 0.26%|
| gam_only  | 26.98±3.62% | 0.799±0.041 | 10.11 ± 4.43% | 0.85 ± 0.21 | 73.02 ± 3.62%|
| fuse_only | 13.92±1.83% | 0.907±0.012 | 49.65 ± 8.55% | 0.25 ± 0.08 | 86.08 ± 1.83%|
| with_geom | 20.16±2.43% | 0.857±0.023 | 25.78 ± 6.90% | 0.64 ± 0.21 | 79.84 ± 2.43%|

### Holdout Set (Mean ± Std, 5 Seeds)

| Variant   | EER (↓)      | AUC (↑)      | TAR@FAR=1% (↑) | d' (↑)       | Acc@EER (↑)  |
|-----------|-------------|-------------|---------------|-------------|-------------|
| no_geom   | 0.00 ± 0.00%| 1.000±0.000 | 100.0 ± 0.00% | 2.25 ± 0.11 | 100.0 ± 0.00%|
| gam_only  | 21.82±3.95% | 0.850±0.046 | 24.85 ±13.45% | 1.06 ± 0.20 | 78.18 ± 3.95%|
| fuse_only | 5.45 ± 2.75%| 0.982±0.013 | 73.33 ±15.66% | 1.99 ± 0.41 | 94.55 ± 2.75%|
| with_geom | 11.21±0.83% | 0.951±0.018 | 61.21 ±17.33% | 1.86 ± 0.12 | 88.79 ± 0.83%|

---

## 2. Uji Statistik

### with_geom vs no_geom (Paired t-test)

| Metrik     | Test (t, p)          | Holdout (t, p)      |
|-----------|---------------------|---------------------|
| EER       | t=20.46, p<0.001*** | t=30.21, p<0.001*** |
| AUC       | t=-14.70, p<0.001***| t=-6.30, p=0.003**  |
| TAR@FAR1% | t=-24.88, p<0.001***| t=-5.01, p=0.008**  |

### fuse_only vs gam_only (Paired t-test)

| Metrik     | Test (t, p)          | Holdout (t, p)      |
|-----------|---------------------|---------------------|
| EER       | t=-5.71, p=0.005**  | t=-7.75, p=0.002**  |
| AUC       | t=4.86, p=0.008**   | t=6.25, p=0.003**   |
| TAR@FAR1% | t=7.60, p=0.002**   | t=11.04, p<0.001*** |

---

## 3. Temuan Kritis

### 3.1 no_geom Terlalu Sempurna — Mencurigakan

- **Test EER = 0.17%**, **Holdout EER = 0.00%** (tidak ada error sama sekali di holdout)
- AUC = 0.9995 (test), 1.0000 (holdout)
- Untuk sistem biometrik 11 subjek, angka ini **tidak realistis**
- Kemungkinan penyebab:
  1. **Data leakage** antara train/test/holdout
  2. Pair generation terlalu mudah (genuine pairs sangat mirip, impostor sangat beda)
  3. Model memiliki kapasitas terlalu besar untuk dataset kecil → overfit

### 3.2 GeoAtt MEMPERBURUK Performa (Bukan Memperbaiki)

Ranking performa (best → worst):

**Test Set:**  no_geom >> fuse_only > with_geom >> gam_only  
**Holdout Set:** no_geom >> fuse_only > with_geom >> gam_only

| Peringkat | Test EER | Holdout EER |
|-----------|----------|-------------|
| 1 (best)  | no_geom  | no_geom     |
| 2         | fuse_only| fuse_only   |
| 3         | with_geom| with_geom   |
| 4 (worst) | gam_only | gam_only    |

**Implikasi:**
- **GAM (Geometric Attention Module)** adalah komponen yang paling merusak performa
- **Fusion** (concat geometry) sedikit membantu, tapi tidak cukup untuk mengimbangi kerusakan dari GAM
- Kombinasi **GAM + Fusion** (with_geom) lebih buruk dari Fusion saja

### 3.3 Konsistensi Lintas Seed

Pola no_geom >> fuse_only > with_geom >> gam_only **berlaku untuk SEMUA 5 seed** (7, 42, 123, 2026, 31337). Ini bukan kebetulan — pola ini konsisten dan signifikan secara statistik.

---

## 4. Hipotesis Penyebab

### Hipotesis A: Bug di Implementasi GeoAtt

GAM mungkin memiliki bug yang merusak embedding:
- Channel attention atau spatial attention menghasilkan nilai NaN/Inf
- Weight normalization tidak benar
- Gradient flow ke GAM mengganggu training point cloud utama

### Hipotesis B: Skala Feature Mismatch

Geometry features (14-dim) mungkin mendominasi 128-dim point cloud embedding:
- Tanpa normalisasi skala, 14-dim geometry bisa overwhelm 128-dim cloud
- Model menjadi terlalu bergantung pada geometry → tidak generalize ke variasi pose/capture

### Hipotesis C: Geometry Features Terlalu Noisy

14 fitur biometrik yang diekstrak dari pipeline 3DRegistration mungkin:
- Tidak stabil antar-frame (variansi tinggi)
- Tidak diskriminatif (tidak membantu membedakan subjek)
- Mengandung noise dari proses segmentasi DBSCAN

### Hipotesis D: Overfitting pada Dataset Kecil

11 subjek × ~14 sesi × ~10 frame = ~1540 frames:
- no_geom (PointNet++ murni) overfit ke pattern point cloud → perfect di test/holdout
- GeoAtt menambah kompleksitas → tidak cukup data untuk train geometry branch

### Hipotesis E: Data Leakage

Holdout EER = 0.00% untuk no_geom menunjukkan kemungkinan leakage:
- Holdout probes mungkin berasal dari sesi yang overlap dengan training
- Split holdout menggunakan `split_holdout_sessions(seed=42)` — mungkin ada bug di fungsi ini

---

## 5. Rekomendasi Aksi

### Prioritas 1: Verifikasi Data Leakage (URGENT)

1. Periksa apakah holdout probes benar-benar dari sesi yang TIDAK pernah masuk training
2. Bandingkan `session_groups` sebelum dan sesudah `split_holdout_sessions`
3. Verifikasi path frame holdout tidak ada di `train_frames` / `val_frames`

```python
# Quick check
for label in session_groups:
    holdout_dirs = set(holdout_probes[label])
    train_dirs = set(train_frames[label])
    overlap = holdout_dirs & train_dirs
    if overlap:
        print(f"LEAKAGE: {label} has {len(overlap)} overlapping dirs!")
```

### Prioritas 2: Debug GeoAtt Implementation

1. Periksa output GAM — pastikan tidak ada NaN/Inf
2. Visualisasikan attention weights — apakah mereka meaningful?
3. Bandingkan embedding norm antara no_geom dan with_geom — apakah with_geom menghasilkan embedding yang degenerate?

```python
# Cek embedding stats
embs_no_geom = ...  # dari no_geom eval
embs_with_geom = ...  # dari with_geom eval
print(np.isnan(embs_with_geom).any())  # Harus False
print(np.linalg.norm(embs_with_geom, axis=1).std())  # Harus ~1 (L2 norm)
```

### Prioritas 3: Scale Normalization untuk Geometry

Tambahkan layer normalisasi sebelum fusion:
```python
# Contoh: LayerNorm atau BatchNorm untuk geometry features
self.geom_norm = nn.LayerNorm(geom_dim)
geom_emb = self.geom_norm(geom_emb)
# Lalu concat ke point cloud embedding
```

### Prioritas 4: Ablation yang Lebih Ketat

Jika bug tidak ditemukan, pertimbangkan:
1. **Hapus GAM** — gunakan fuse_only saja (ini yang paling mendekati no_geom)
2. **Ganti loss function** — coba OnlineTripletLoss atau ArcFace dengan margin yang lebih besar
3. **Tambah data** — kumpulkan lebih banyak subjek (target: 30-50 subjek)

### Prioritas 5: Re-train dengan Config Tracking

1. Simpan config.json di setiap run untuk verifikasi flags
2. Log TensorBoard selama training untuk monitor validation metrics per epoch
3. Gunakan early stopping yang lebih sensitif

---

## 6. Kesimpulan

**Hipotesis utama thesis: "GeoAtt memberikan dampak signifikan terhadap identifikasi telapak tangan"**

**Status: DITOLAK oleh data.**

Data menunjukkan bahwa:
1. **Baseline (no_geom) mencapai performa near-perfect** — mencurigakan, perlu verifikasi leakage
2. **GeoAtt secara konsisten memperburuk performa** — ranking: no_geom >> fuse_only > with_geom >> gam_only
3. **GAM adalah komponen paling merusak** — menghapus GAM meningkatkan performa signifikan
4. **Perbedaan signifikan secara statistik** — paired t-test p < 0.001 untuk with_geom vs no_geom

**Langkah segera yang harus diambil:**
1. Verifikasi data leakage
2. Debug implementasi GeoAtt (cek NaN, attention weights, embedding quality)
3. Jika tidak ada bug → kemungkinan geometry features tidak cocok untuk dataset ini → pertimbangkan menghapus GAM

---

*Report generated: 2026-05-21*
