# Kesimpulan Eksperimen Ablasi GeoAtt-PointNet++

**Tanggal:** 2026-05-22
**Dataset:** 11 subjek, frame-level (~14 sesi/subjek)
**Konfigurasi:** 4 variant × 5 seed (7, 42, 123, 2026, 31337) = 20 run
**Sumber data:** `runs/`, `eval_results/`, `analysis/aggregate_*.csv`, TensorBoard `loss/val`

---

## 1. Ringkasan Eksekutif

Hasil eksperimen menunjukkan **GeoAtt-PointNet++ (with_geom) konsisten kalah dari baseline PointNet++ (no_geom)** di semua metrik dan semua seed. Namun **kesimpulan ini belum valid** karena ditemukan **tiga bias eksperimental serius** yang membuat hasil saat ini tidak dapat dipakai sebagai bukti penolakan hipotesis thesis.

Ringkasan singkat:

| Variant   | Test EER     | Holdout EER  | AUC test     | Rank |
|-----------|--------------|--------------|--------------|------|
| no_geom   | **0.17 %**   | **0.00 %**   | **0.9995**   | 1    |
| fuse_only | 13.92 %      | 5.45 %       | 0.907        | 2    |
| with_geom | 20.16 %      | 11.21 %      | 0.857        | 3    |
| gam_only  | 26.98 %      | 21.82 %      | 0.799        | 4    |

Ranking ini **identik di semua 5 seed**. Paired t-test menunjukkan perbedaan signifikan statistik (p<0.001 untuk with_geom vs no_geom).

**Status hipotesis thesis** ("GeoAtt meningkatkan kualitas identifikasi telapak"): **belum dapat disimpulkan** — bukan ditolak, bukan diterima. Tiga bias di bawah harus diperbaiki dulu sebelum klaim apa pun.

---

## 2. Tiga Bias Eksperimental yang Ditemukan

### 2.1 Bias #1 — Kebocoran Split Temporal

**Bukti**: Inspeksi `runs/no_geom/seed_42/splits.json` menunjukkan sesi train, test, dan holdout untuk satu subjek **berasal dari rentang capture yang sama (selisih < 2 menit)**.

Contoh subjek `aisah`:

| Split    | Timestamp sesi                                                                                              |
|----------|-------------------------------------------------------------------------------------------------------------|
| Train    | 140321, 140322, 140324, 140325, 140339, 140341, 140342, 140346, 140347, 140349, 140350, 140356, 140405–140409 |
| Test     | 140334, 140357, 140358                                                                                       |
| Holdout  | 140400                                                                                                       |

Sesi test dan holdout **terselip di antara** sesi train. Semuanya dari satu rentang capture ±90 detik dengan kondisi pencahayaan, pose, dan jarak ke kamera yang nyaris identik.

**Konsekuensi**: Test dan holdout **bukan held-out sebenarnya**. Mereka mengukur kemampuan model "mengenali tangan dalam satu sesi rekaman", bukan generalisasi ke kondisi capture baru. Ini menjelaskan kenapa `no_geom` mencapai EER 0.00 % di holdout — angka yang tidak realistis untuk biometrik 11 subjek.

### 2.2 Bias #2 — Val_loss Anti-Korelasi dengan Generalisasi

**Bukti** (TensorBoard `loss/val`, smoothed final):

| Variant   | Val_loss range    | Test EER   |
|-----------|-------------------|------------|
| gam_only  | **0.0002 – 0.003** (terendah)  | **27 %** (terburuk)  |
| with_geom | 0.0021 – 0.0042   | 20 %       |
| fuse_only | 0.0012 – 0.0043   | 14 %       |
| no_geom   | **0.002 – 0.009** (tertinggi)  | **0.17 %** (terbaik) |

**Pola: semakin rendah val_loss, semakin buruk test EER.**

Karena val set berasal dari distribusi yang sama dengan train (akibat bias #1), val_loss mengukur kemampuan **memorisasi artefak per-capture-session**, bukan generalisasi. Variant dengan kapasitas memorisasi besar (gam_only, with_geom — punya 14 geom features tambahan + GAM) menang di val_loss tapi kalah di test pair yang lebih luas distribusinya.

**Konsekuensi**: Checkpoint `best.pth` (yang dipilih berdasarkan val_loss terendah) **adalah checkpoint paling overfit** untuk variant geometry. Evaluasi memakai checkpoint yang salah.

### 2.3 Bias #3 — Training Budget Tidak Seragam

**Bukti** (TensorBoard `Step` final phase-2):

| Run                       | Step terakhir |
|---------------------------|---------------|
| with_geom/seed_2026       | 20            |
| with_geom/seed_42         | 24            |
| fuse_only/seed_31337      | 25            |
| ...                       | ...           |
| fuse_only/seed_42         | **55**        |

Rentang 20 → 55 epoch artinya **early stopping memicu di waktu yang sangat berbeda** antar run. Karena early stopping memakai val_loss (yang sudah bias #2), variant dengan val_loss yang bisa diturunkan terus-menerus (gam_only, with_geom) mendapat budget overfit yang lebih lama.

**Konsekuensi**: Perbandingan antar variant tidak adil dalam hal jumlah optimisasi yang dialokasikan.

---

## 3. Bukti Pendukung dari Training Log

Phase-2 training loss vs val loss (seed_42):

| Variant   | Train loss | Val loss | Gap     |
|-----------|------------|----------|---------|
| no_geom   | 0.0001     | 0.002    | ~20×    |
| with_geom | 0.00004    | 0.004    | **~100×** |
| gam_only  | 0.00005    | 0.003    | ~60×    |
| fuse_only | 0.00002    | 0.0004   | ~20×    |

`with_geom` dan `gam_only` menunjukkan **gap train-val 60–100×** — overfitting yang sangat parah. Tapi val_loss tetap turun (bias #2), jadi training tidak berhenti dan checkpoint yang disimpan sudah overfit.

---

## 4. Apa yang Bisa Disimpulkan dari Hasil Saat Ini

### Yang dapat disimpulkan:

1. **Setup eksperimen saat ini tidak valid** untuk menguji hipotesis thesis. Tiga bias di atas (split bocor, val_loss menyesatkan, training budget tidak seragam) menyebabkan hasil bias secara sistematis.
2. **Cabang geometry (14-dim fitur + GAM) menunjukkan kapasitas memorisasi tinggi** — terbukti dari train loss yang turun lebih dalam dari no_geom dan val_loss yang lebih rendah.
3. **Tanpa fix split, no_geom akan selalu "menang"** karena memerlukan kapasitas minimum untuk menghafal pattern point cloud yang sangat mirip antar sesi.
4. **Pola ranking konsisten lintas seed** menunjukkan ini bukan variansi statistik — ini bias sistematis dari setup eksperimen.

### Yang **belum** dapat disimpulkan:

1. ❌ Apakah GeoAtt benar-benar memperburuk performa pada split yang adil.
2. ❌ Apakah GAM komponen yang merusak atau sebenarnya bisa membantu.
3. ❌ Apakah 14 fitur geometric biometrik diskriminatif atau tidak.
4. ❌ Apakah hipotesis thesis ditolak atau diterima.

---

## 5. Rekomendasi Tindakan (Prioritas Berurut)

### Prioritas 1 — Perbaiki Split (WAJIB sebelum analisis ulang)

Ganti `split_holdout_sessions` dan random train/val/test split dengan **time-gap aware split**:

- Urutkan semua sesi per subjek berdasarkan timestamp.
- Test set: sesi yang **selisih waktu ≥ N menit** dari sesi train terdekat (target: minimal 1 jam, idealnya hari berbeda).
- Holdout set: sesi yang lebih jauh lagi atau dari sesi capture yang benar-benar terpisah.
- Verifikasi explicit: untuk setiap pasangan (train_session, test_session), `|timestamp_diff| ≥ threshold`.

### Prioritas 2 — Ganti Metrik Model Selection

Tambahkan **val pair EER atau val pair AUC** sebagai metrik per epoch di training loop. Simpan `best.pth` berdasarkan **val EER terendah** atau **val AUC tertinggi**, bukan val_loss. Log juga ke TensorBoard untuk monitoring.

### Prioritas 3 — Seragamkan Training Budget

Pilih salah satu:

- **Fixed epochs**: hapus early stopping, latih semua variant dengan jumlah epoch yang sama (mis. 50 phase-1 + 30 phase-2).
- **Early stopping pada metrik baru**: pakai val EER (dari Prioritas 2), bukan val_loss.

### Prioritas 4 — Re-run Eksperimen

Setelah 3 fix di atas:

- Re-train 4 variant × 5 seed dengan setup baru.
- Hitung paired t-test pada hasil baru.
- Bandingkan kurva training dan kualitas attention map GAM.

### Prioritas 5 — Audit Implementasi GAM (Opsional)

Sebagai sanity check selama menunggu split fix:

- Visualisasi attention weight GAM — periksa apakah meaningful atau collapse ke channel tertentu.
- Cek apakah skala 14 geom features (std 0.01 – 20 di `normalizer.json`) menyebabkan masalah gradien — tambahkan LayerNorm sebelum fusion.
- Verifikasi tidak ada NaN/Inf di output GAM.

---

## 6. Kesimpulan Akhir

**Hasil eksperimen saat ini menunjukkan pola yang konsisten secara statistik, namun pola ini bukan refleksi performa sebenarnya — melainkan artefak dari tiga bias eksperimental:**

1. Split temporal yang bocor (test/holdout dari capture session yang sama dengan train).
2. Val_loss yang anti-korelasi dengan generalisasi (memilih checkpoint paling overfit).
3. Training budget yang tidak seragam akibat early stopping pada metrik yang bias.

**Sebelum bias-bias ini diperbaiki, hasil tidak dapat dipakai untuk klaim thesis** — baik untuk menerima maupun menolak hipotesis bahwa GeoAtt meningkatkan kualitas identifikasi telapak.

**Langkah selanjutnya yang harus diambil:** implementasi 3 perbaikan (split, metric, budget) lalu re-run eksperimen lengkap. Estimasi waktu: 1–2 minggu development + 1 minggu re-training × 20 run.

**Kabar baiknya**: temuan ini **bukan bug arsitektur GeoAtt**. Setelah eksperimen dijalankan dengan benar, ada kemungkinan GeoAtt menunjukkan keunggulan yang sebenarnya — atau menunjukkan bahwa pada dataset sekecil ini cabang geometry memang tidak menambah nilai. Kedua kesimpulan akan **valid secara metodologis** setelah fix diterapkan.

---

## Lampiran A — Sumber Data

- Agregat metrik: `analysis/aggregate_test.csv`, `analysis/aggregate_holdout.csv`
- Per-seed eval: `eval_results/{variant}/seed_{N}/results.json`
- Training log: `runs/{variant}/seed_{N}/train_log.csv`
- TensorBoard: `runs/{variant}/seed_{N}/tensorboard/`
- Splits: `runs/{variant}/seed_{N}/splits.json`
- Normalizer stats: `runs/{variant}/seed_{N}/normalizer.json`

## Lampiran B — Dokumen Terkait

- `EVALUATION_REPORT.md` — laporan awal dengan hipotesis penyebab.
- `DOCUMENTATION.md` — dokumentasi teknis pipeline.
- `DECISION_MEMO.md`, `DECISION_MEMO_DATASET.md` — keputusan arsitektur dan dataset sebelumnya.
