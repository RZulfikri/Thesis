# Analisis: Kenapa Loss Bagus tapi Evaluasi Buruk?

**Tanggal:** 2026-05-21
**Fokus:** Seed 42 (representatif, pola sama di semua seed)

---

## 1. Fakta: Loss dengan_geom BUKAN yang Paling Bagus

| Variant     | Phase 2 Train Loss (end) | Phase 2 Val Loss (end) |
|-------------|--------------------------|------------------------|
| no_geom     | 0.000115                 | 0.002019               |
| gam_only    | 0.000051                 | 0.003678               |
| fuse_only   | 0.000019                 | 0.000309               |
| with_geom   | 0.000042                 | 0.004294               |

**Kesimpulan dari Tabel Loss:**

1. Training loss with_geom (0.000042) memang sangat kecil - model **overfit** ke training data
2. Validation loss with_geom (0.004294) adalah yang **PALING BURUK** di antara semua variant
3. fuse_only punya val_loss terbaik (0.000309), tapi evaluasi juga masih buruk (EER=14.65%)
4. no_geom val_loss bukan yang terbaik (0.002019), tapi evaluasi PERFECT (EER=0.01%)

**Ini menunjukkan: Contrastive Loss != Biometric Metrics.**
Model bisa punya training loss rendah (overfit) tapi evaluation metrics buruk.

---

## 2. Penyebab Utama: Similarity Distribution

Perhatikan distribusi similarity scores dari test set:

| Variant     | Genuine (mean+/-std) | Impostor (mean+/-std) | Separation |
|-------------|----------------------|-----------------------|------------|
| no_geom     | 0.9998 +/- 0.0019    | 0.6901 +/- 0.1951     | **0.3098** |
| gam_only    | 0.9727 +/- 0.0557    | 0.8890 +/- 0.1234     | 0.0837     |
| fuse_only   | 0.9759 +/- 0.0752    | 0.9491 +/- 0.0779     | 0.0269     |
| with_geom   | 0.9703 +/- 0.0499    | 0.9345 +/- 0.0474     | 0.0358     |

### Gambaran Visual:

no_geom:   Genuine [0.998] <<< GAP besar >>> Impostor [0.690]
with_geom: Genuine [0.970] <<< GAP kecil >>> Impostor [0.934]

---

## 3. Analisis Mendalam

### 3.1 no_geom: Embeddings Perfectly Separated

- Genuine similarity = 0.9998 (hampir semua = 1.0)
- Impostor similarity = 0.6901 (jauh di bawah)
- Threshold EER = 0.9628 bisa memisahkan dengan sempurna
- **TP=153638, TN=153628, FP=20, FN=10** (hanya 30 error dari 307k pairs)

### 3.2 with_geom: Embeddings Collapsed / Overlapping

- Genuine similarity = 0.9703
- Impostor similarity = 0.9345
- Gap hanya 0.036 (10x lebih kecil dari no_geom)
- **TP=125916, TN=125916, FP=27732, FN=27732** (55k error)

### 3.3 Apa yang Terjadi?

**Contrastive Loss** hanya memaksa:
- Genuine pairs: similarity > margin (misal 0.5)
- Impostor pairs: similarity < margin (misal 0.5)

Tapi dengan_geom model belajar:
- Training pairs di-push ke margin (similarity ~0.97 vs ~0.93)
- Tapi di test set, semua embeddings terkompresi ke region yang sama
- Akibatnya: tidak bisa membedakan genuine vs impostor

**no_geom** belajar:
- Point cloud features yang sangat diskriminatif
- Embeddings tersebar di sphere (L2 norm = 1)
- Genuine pairs = sangat mirip (sama identitas = cluster erat)
- Impostor pairs = sangat beda (beda identitas = jauh di sphere)

---

## 4. Hipotesis Penyebab

### A. Geometry Features Corrupt Embedding Space

Point cloud features (xyz+normal = 6 dim) sudah cukup diskriminatif.
Tambahan geometry (14 dim) malah:
1. Menambah noise ke embedding space
2. Menggeser embeddings ke arah yang salah
3. Mengurangi separation antar identitas

### B. GAM Menghancurkan Point Cloud Features

GAM (Geometric Attention Module) mungkin:
1. Mengalihkan attention dari point cloud ke geometry
2. Point cloud features jadi "dilupakan" karena geometry mendominasi
3. Akibatnya: embeddings kehilangan kemampuan membedakan identitas

### C. Scale Mismatch

Geometry features (mm) vs point cloud features (normalized):
- Tanpa proper scaling, geometry bisa overwhelm point cloud
- Model jadi bergantung pada geometry yang noisy

### D. Overfitting pada Dataset Kecil

11 subjek x ~14 sesi = ~154 sesi:
- no_geom: 58 layer (parameter lebih sedikit) -> generalize baik
- with_geom: tambahan GAM + fusion (parameter lebih banyak) -> overfit

---

## 5. Kesimpulan

| Pertanyaan | Jawaban |
|------------|---------|
| "Kenapa loss with_geom bagus?" | Training loss rendah = overfit ke training pairs |
| "Kenapa eval with_geom buruk?" | Embeddings tidak terpisah (collapsed) di test set |
| "Kenapa no_geom sempurna?" | Point cloud features sudah sangat diskriminatif; geometry malah merusak |
| "Apakah GeoAtt bekerja?" | **Tidak.** Data menunjukkan GeoAtt memperburuk performa |

### Langkah Segera:
1. Verifikasi data leakage (holdout EER=0% sangat mencurigakan)
2. Debug GAM: cek attention weights, NaN/Inf, embedding norms
3. Jika tidak ada bug: geometry features tidak cocok untuk dataset ini
4. Pertimbangkan menghapus GAM dan hanya gunakan fusion (atau bahkan no_geom saja)
