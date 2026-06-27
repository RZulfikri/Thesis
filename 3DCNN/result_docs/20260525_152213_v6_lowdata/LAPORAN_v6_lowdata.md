# Laporan Eksperimen v6.0.0 — Low-Data Regime

**Tanggal analisis**: 2026-05-25 (run id `v6_lowdata_20260525_152213`)
**Tujuan**: Membandingkan PointNet++ dengan loss **Triplet (standard)** vs **ArcFace (m=0.5, s=30)** pada rezim data terbatas.
**Setup**: 10 subjek × 15 sesi × 1 median frame = **150 frame**.
**Seeds (10)**: 42, 123, 2024, 7, 31337, 0, 1, 2, 3, 4.
**Artefak utama**:
- Training: [runs/v6_lowdata/standard](../../runs/v6_lowdata/standard), [runs/v6_lowdata/arcface](../../runs/v6_lowdata/arcface)
- Evaluasi: [eval_results/v6_lowdata](../../eval_results/v6_lowdata)
- Analisis statistik: [analysis/v6_lowdata_20260525_152213](../../analysis/v6_lowdata_20260525_152213)

---

## 1. Ringkasan Eksekutif

- **Kedua varian tampil sangat baik** pada generalisasi holdout (EER ≈ 1.5–2.3%).
- ArcFace **sedikit lebih baik** pada kedua split, tetapi **selisihnya tidak signifikan secara statistik** (Wilcoxon p > 0.6 pada keduanya).
- Keuntungan ArcFace muncul terutama pada **stabilitas (std lebih kecil)** dan **separability embedding (d-prime ≈ 2× lipat)**, bukan pada penurunan EER absolut.
- **Verdict**: Pada N=10 subjek dan rezim 150 frame, ArcFace tidak memberikan peningkatan yang dapat dideteksi secara statistik dibanding Triplet. Pilihan loss menjadi pertimbangan sekunder; faktor pembatas saat ini adalah ukuran dataset.

---

## 2. Hasil Evaluasi (10 seeds)

### 2.1 Test EER (primary metric)

| Varian   | EER mean | EER std | min   | max  | AUC mean | TAR@FAR1% | d-prime | Rank-1 |
|----------|---------:|--------:|------:|-----:|---------:|----------:|--------:|-------:|
| arcface  | **0.060** | 0.0316 | 0.050 | 0.150 | 0.947 | 0.890 | **3.40** | 0.890 |
| standard | 0.065 | 0.0474 | 0.050 | 0.200 | 0.899 | 0.890 | 1.51 | 0.885 |

### 2.2 Holdout EER (generalisasi)

| Varian   | EER mean | EER std | min | max   | AUC mean | TAR@FAR1% | d-prime | Rank-1 |
|----------|---------:|--------:|----:|------:|---------:|----------:|--------:|-------:|
| arcface  | **0.0150** | 0.0123 | 0.000 | 0.0333 | 0.997 | 0.927 | **4.96** | 0.980 |
| standard | 0.0233 | 0.0274 | 0.000 | 0.0667 | 0.995 | 0.947 | 2.96 | 0.960 |

Sumber: [aggregate_test.csv](../../analysis/v6_lowdata_20260525_152213/aggregate_test.csv), [aggregate_holdout.csv](../../analysis/v6_lowdata_20260525_152213/aggregate_holdout.csv).

### 2.3 Uji Statistik (Wilcoxon paired, arcface − standard)

| Split   | Δ EER (arc−std) | Wilcoxon stat | p-value | Kesimpulan |
|---------|----------------:|--------------:|--------:|------------|
| Test    | −0.0050 | 1.0  | 1.0000 | Tidak signifikan |
| Holdout | −0.0083 | 14.0 | 0.6328 | Tidak signifikan |

Detail: [wilcoxon_tests.json](../../analysis/v6_lowdata_20260525_152213/wilcoxon_tests.json).

---

## 3. Training

- **Skrip**: [train.py](../../train.py) (ArcFace runner ditambahkan v6.0.0, ref. memori 1311–1313).
- **Auto-tuning GPU/RAM** aktif (warisan v5.0.0): preload dataset + batch sizing adaptif.
- **Konvergensi**: keduanya konvergen ke val_eer ≈ 0 sebelum epoch 30 (fase 2). Contoh seed 42:
  - Standard: train_loss turun dari 0.656 → ~0 pada epoch 20-an; val_eer 0.0 stabil.
  - ArcFace: train_loss turun cepat; aux_acc ≈ 0.997–1.000 pada akhir training (classification head ArcFace menghafal sempurna training set).
- **Catatan**: ArcFace mempelajari klasifikasi identitas hingga akurasi ~100% pada training set, tapi val_loss tetap berfluktuasi (mis. 0.006–0.072 di akhir) — konsisten dengan kapasitas margin penalty yang memberi tekanan separability lebih kuat tanpa overfit metric verifikasi.
- Trayektori: [train_loss_trajectory.png](../../analysis/v6_lowdata_20260525_152213/train_loss_trajectory.png), [val_eer_trajectory.png](../../analysis/v6_lowdata_20260525_152213/val_eer_trajectory.png).

---

## 4. Evaluasi

- **Protokol**: pair verification (test & holdout) + identifikasi (rank-1/5/10, mAP) per seed; gallery 10 subjek.
- **Per seed**: tersedia ROC/DET, t-SNE, distribusi similarity, dan confusion identifikasi. Contoh `arcface/seed_42/test`: EER=0.05, AUC=0.93, Rank-1=0.9, Rank-10=1.0.
- **Holdout audit temporal**: gap maksimum train→holdout per subjek tetap rendah (lanjutan pola "capture burst", ref. memori 1275) — interpretasi: holdout EER yang rendah sebagian dijelaskan oleh kedekatan temporal sesi, bukan hanya generalisasi murni.

---

## 5. Analisis Komparatif

- **Boxplot**: [boxplots_test_holdout.png](../../analysis/v6_lowdata_20260525_152213/boxplots_test_holdout.png) menunjukkan median yang setara, dengan ArcFace memiliki ekor atas lebih pendek (lebih konsisten antar seed).
- **Paired diff per seed**: [per_seed_paired_diff.png](../../analysis/v6_lowdata_20260525_152213/per_seed_paired_diff.png) — perbedaan kecil dan tidak berarah konsisten ke salah satu varian.
- **d-prime gap besar (3.40 vs 1.51 di test, 4.96 vs 2.96 di holdout)** menjadi sinyal paling kuat: ArcFace menghasilkan distribusi skor genuine/impostor yang **lebih terpisah**, meski EER pada threshold optimal tidak banyak berubah karena efek lantai (banyak seed mencapai EER 0).

---

## 6. Keterbatasan & Konteks dengan v5

- N=10 subjek dan 150 frame → uji statistik kekurangan power; banyak seed menyentuh EER=0 (efek floor).
- Konsisten dengan temuan v5.0.0 (memori 1295–1298): pada rezim sangat-kecil, kompleksitas tambahan (geom-attention atau ArcFace) **tidak superior** secara signifikan terhadap baseline PointNet++.
- ArcFace **tidak mengulangi pola memorisasi val seperti GeoAtt v5**; trayektori val_eer terlihat sehat.

---

## 7. Rekomendasi

1. **Pilih ArcFace** sebagai default untuk eksperimen berikutnya **bukan karena EER**, melainkan karena: (a) d-prime ~2× lebih baik, (b) std antar-seed lebih kecil, (c) embedding lebih siap untuk threshold tuning ketat (high-FAR regime: TAR@FAR1% comparable, tapi dengan margin separability lebih besar).
2. **Tingkatkan N subjek** atau jumlah sesi per subjek sebelum melakukan uji statistik lanjutan; saat ini Wilcoxon tidak punya power.
3. **Audit ulang temporal gap** untuk holdout (lanjutkan investigasi memori 1275) — pertimbangkan menambahkan split holdout dengan gap minimal X hari untuk uji generalisasi temporal yang lebih ketat.
4. **Eksperimen lanjutan**: ArcFace dengan margin lebih kecil (m=0.3) untuk melihat apakah margin = 0.5 terlalu agresif pada subset 10 subjek, dan bandingkan terhadap CosFace / SubCenter-ArcFace.

---

*Dihasilkan dari `runs/v6_lowdata`, `eval_results/v6_lowdata`, dan `analysis/v6_lowdata_20260525_152213`. Notebook sumber: [v6_standard_train_eval.ipynb](../../collab/v6_standard_train_eval.ipynb), [v6_arcface_train_eval.ipynb](../../collab/v6_arcface_train_eval.ipynb), [v6_standard_arcface_compare.ipynb](../../collab/v6_standard_arcface_compare.ipynb).*
