# Monitoring Checklist — v0.3.0 Improvement Execution

> **Tanggal Mulai**: _isi tanggal_  
> **Target Selesai**: _isi tanggal + 7 hari_  
> **Environment**: Google Colab (A100)  
> **Dataset**: 11 subjek, frame layout  
> **Seeds**: `[7, 42, 123, 2026, 31337]`  
> **Split Seed**: `42`

---

## 0. Pre-Execution Checklist

### 0.1 Environment & Data
- [ ] Google Drive ter-mount di Colab
- [ ] Folder `3DCNN/` di Drive sudah di-sync dengan repo terbaru (commit `35495a4` atau lebih baru)
- [ ] `models/siamese.py` punya `forward_arcface` dan `num_classes` support
- [ ] `losses/arcface.py` tersedia
- [ ] `utils/enrollment.py` tersedia (dengan `GalleryEnroller`)
- [ ] `utils/data_qc.py` tersedia (baru diupload)
- [ ] `utils/ply_dataset.py` tersedia (baru diupload — untuk E3)
- [ ] `utils/augmentation.py` sudah diupdate dengan `OriginalSpaceAugmentor`
- [ ] Checkpoint v0.2.0 masih tersedia di `runs/with_geom/20260516_073445/` dan `runs/no_geom/20260516_073519/`

### 0.2 Konfigurasi Notebook
- [ ] `collab/train.ipynb` dan `collab/train_no_geom.ipynb` di-sync
- [ ] `collab/evaluate.ipynb` dan `collab/evaluate_no_geom.ipynb` di-sync
- [ ] `EXPERIMENT_TIMESTAMP` diisi dengan timestamp baru (jangan overwrite v0.2.0)

---

## 1. E1 — Multi-Prototype Enrollment (S1)

**Tujuan**: Ukur dampak enrollment strategy terhadap Rank-1, tanpa retrain model.

### 1.1 Evaluasi Baseline (Mean Enrollment)
- [ ] Buka `collab/evaluate.ipynb`
- [ ] Set `TRAIN_TIMESTAMP = "20260516_073445"` (with_geom baseline)
- [ ] Set `ENROLL_STRATEGY = "average"`
- [ ] Jalankan evaluasi 5 seed
- [ ] Catat Rank-1, Rank-5, mAP, EER ke tabel E1

### 1.2 Evaluasi Multi-Prototype (k=3)
- [ ] Di notebook yang sama, set `ENROLL_STRATEGY = "multi"`
- [ ] Set `ENROLL_K = 3`
- [ ] Jalankan evaluasi 5 seed (split identik)
- [ ] Catat Rank-1, Rank-5, mAP, EER ke tabel E1

### 1.3 Evaluasi no_geom (Opsional, untuk fairness)
- [ ] Buka `collab/evaluate_no_geom.ipynb`
- [ ] Ulangi 1.1 dan 1.2 untuk checkpoint `runs/no_geom/20260516_073519/`
- [ ] Catat hasil ke tabel E1

### 1.4 Tabel Hasil E1

| Config | Enrollment | Rank-1 | Rank-5 | mAP | EER | Catatan |
|--------|-----------|--------|--------|-----|-----|---------|
| with_geom | average (baseline) | | | | | v0.2.0 |
| with_geom | multi (k=3) | | | | | **E1** |
| no_geom | average (baseline) | | | | | v0.2.0 |
| no_geom | multi (k=3) | | | | | **E1** |

### 1.5 Decision E1
- [ ] **Jika Rank-1 with_geom naik >5%**: Multi-prototype efektif, gunakan untuk E2 dan E3
- [ ] **Jika Rank-1 naik <5%**: Enrollment bukan bottleneck utama → fokus ke loss (E2)

---

## 2. Data QC — Outlier Subjects (S3)

**Temuan Awal** (dari `utils/data_qc.py`):

| Subjek | Pts Mean | Pts Min | Pts Max | Z-BBox Normal | Catatan |
|--------|----------|---------|---------|---------------|---------|
| feby | 17,091 | 15,352 | 18,307 | ~0.045 | Normal |
| nola | 17,755 | 16,588 | 19,138 | ~0.032 | Normal |
| **reysa** | **16,473** | **6,889** | **17,396** | **~0.010** | **Outlier kuat** |

### 2.1 Inspeksi Visual reysa
- [ ] Jalankan cell QC di Colab:
  ```python
  from utils.data_qc import run_qc_report, print_qc_report
  report = run_qc_report("dataset", subjects=["reysa"])
  print_qc_report(report)
  ```
- [ ] Buka 3–5 frame outlier reysa (pts < 10,000) di Open3D visualizer
- [ ] **Periksa**: apakah tangan incomplete / terpotong / pose ekstrem?

### 2.2 Keputusan QC
- [ ] **Jika scan rusak** → Re-scan sesi reysa (dan feby/nola jika perlu)
- [ ] **Jika scan normal** → Masalahnya registration/alignment, bukan data mentah
- [ ] Catat keputusan di bawah:

```
Keputusan QC: _________________________________
Tanggal: _________________________________
```

---

## 3. E2 — Hybrid ArcFace + Triplet Loss (S2)

**Tujuan**: Ukur dampak loss function terhadap intra-class variance dan Rank-1.

### 3.1 Training with_geom
- [ ] Buka `collab/train.ipynb`
- [ ] Set `LOSS_FN = 'hybrid'`
- [ ] Set `PHASE1_EPOCHS = 100`, `PHASE2_EPOCHS = 30`, `PHASE3_EPOCHS = 20`
- [ ] Set `ARCFACE_MARGIN = 0.50`, `ARCFACE_SCALE = 30.0`
- [ ] Set `USE_GEOM = True`
- [ ] Jalankan multi-seed (5 seeds) dengan `split_seed=42`
- [ ] Simpan checkpoint ke `runs/with_geom/<timestamp>/`

### 3.2 Training no_geom
- [ ] Buka `collab/train_no_geom.ipynb`
- [ ] Konfigurasi identik dengan 3.1
- [ ] Jalankan multi-seed
- [ ] Simpan checkpoint ke `runs/no_geom/<timestamp>/`

### 3.3 Evaluasi E2
- [ ] Evaluasi kedua run dengan `ENROLL_STRATEGY = "multi"` (dari hasil E1)
- [ ] Catat metrik ke tabel E2

### 3.4 Tabel Hasil E2

| Config | Loss | Enrollment | Rank-1 | Rank-5 | mAP | EER | Holdout Rank-1 |
|--------|------|-----------|--------|--------|-----|-----|----------------|
| with_geom | Triplet (v0.2.0) | average | 59.8% | 92.4% | 73.1% | 29.0% | 72.7% |
| with_geom | **Hybrid** | multi | | | | | |
| no_geom | Triplet (v0.2.0) | average | 55.5% | 88.2% | 69.6% | 28.5% | 66.7% |
| no_geom | **Hybrid** | multi | | | | | |

### 3.5 Decision Gate — Lanjut ke PLY Direct?

Berdasarkan hasil E1 + E2:

| Kondisi | Threshold | Keputusan |
|---------|-----------|-----------|
| with_geom Hybrid Rank-1 holdout | **> 80%** | ✅ Skip PLY Direct. Fokus thesis writing. |
| with_geom Hybrid Rank-1 holdout | **70–80%** | ⚠️ Lanjut E3 (PLY Direct Minimal) untuk gain marginal. |
| with_geom Hybrid Rank-1 holdout | **< 70%** | ⚠️ Lanjut E3 + investigasi data QC lebih dalam. |

**Keputusan Gate**: _________________________________  
**Tanggal**: _________________________________

---

## 4. E3 — PLY Direct On-the-Fly (Jika Gate Terpenuhi)

**Tujuan**: Ukur dampak augmentasi original-space terhadap generalisasi.

### 4.1 Implementasi PLY Direct
- [ ] Di `collab/train.ipynb`, ganti dataset creation:
  ```python
  from utils.ply_dataset import PLYDirectDataset
  from utils.augmentation import OriginalSpaceAugmentor

  pc_augmentor = OriginalSpaceAugmentor(seed=42)  # atau None untuk deterministic
  train_ds = PLYDirectDataset(
      label_sessions=train_frames,
      n_points=N_POINTS,
      sampling=SAMPLING,
      augment=pc_augmentor,
      geom_augment=geom_augmentor,
      normalizer=normalizer,
      repeat=_AUTO_REPEAT,
  )
  ```
- [ ] Jalankan `verify_ply_identity` pada 3 frame acak untuk sanity check:
  ```python
  from utils.ply_dataset import verify_ply_identity
  for fd in [train_frames["rahmat"][0], train_frames["aisah"][0], train_frames["feby"][0]]:
      print(verify_ply_identity(fd, n_points=N_POINTS, sampling=SAMPLING))
  ```
- [ ] **Wajib PASS** sebelum training dimulai

### 4.2 Training E3
- [ ] Gunakan config identik E2 (Hybrid loss, multi-prototype eval)
- [ ] Jalankan 5 seed untuk with_geom dan no_geom
- [ ] Catat training time per epoch (bandingkan dengan baseline: ~15–30s)

### 4.3 Evaluasi E3
- [ ] Evaluasi dengan `ENROLL_STRATEGY = "multi"`
- [ ] Catat metrik ke tabel E3

### 4.4 Tabel Hasil E3

| Config | Dataset | Loss | Enrollment | Rank-1 | Rank-5 | mAP | EER | Time/Epoch |
|--------|---------|------|-----------|--------|--------|-----|-----|------------|
| with_geom | `npy` (E2) | Hybrid | multi | | | | | ~__s |
| with_geom | `ply` (E3) | Hybrid | multi | | | | | ~__s |
| no_geom | `npy` (E2) | Hybrid | multi | | | | | ~__s |
| no_geom | `ply` (E3) | Hybrid | multi | | | | | ~__s |

---

## 5. Normals Quick Ablation (Opsional)

**Tujuan**: Validasi apakah normals memberi gain sebelum invest di pipeline normals penuh.

### 5.1 Toggle in_channels=6
- [ ] Di `collab/train.ipynb`, cari cell inisialisasi model:
  ```python
  model = SiamesePalmNet(
      geom_dim=GEOM_DIM, use_geom=USE_GEOM,
      num_classes=num_classes,
      arc_margin=ARCFACE_MARGIN,
      arc_scale=ARCFACE_SCALE,
      in_channels=6,  # default 3 — toggle ini
  )
  ```
- [ ] Pastikan `cnn_input.npy` yang dimuat adalah (N, 6) — sudah terbukti valid
- [ ] Train 2 seed (seed 42 dan 123) dengan Hybrid loss
- [ ] Evaluasi dan catat Rank-1

### 5.2 Decision Normals
- [ ] **Jika Rank-1 naik >3%** → pertimbangkan include normals di PLY Direct future work
- [ ] **Jika Rank-1 naik <3%** → skip normals, fokus ke yang lain

---

## 6. Ringkasan Perbandingan Semua Eksperimen

| Eksperimen | with_geom Rank-1 | with_geom EER | no_geom Rank-1 | no_geom EER | Signifikan? |
|-----------|-----------------|---------------|----------------|-------------|-------------|
| v0.2.0 Baseline (Triplet + avg) | 59.8% ± 2.6% | 29.0% | 55.5% ± 13.6% | 28.5% | p=1.0 |
| E1 (Triplet + multi) | | | | | |
| E2 (Hybrid + multi) | | | | | |
| E3 (PLY + Hybrid + multi) | | | | | |

---

## 7. Final Checklist — Sebelum Thesis Writing

- [ ] Semua hasil tersimpan di `eval_results/` dengan timestamp terstruktur
- [ ] `results.json` per eksperimen lengkap dengan `test_fingerprint`
- [ ] Comparison report di-generate (`comparison_report.md`)
- [ ] Plot CMC, ROC, t-SNE tersimpan per eksperimen
- [ ] Confusion matrix per seed tersimpan
- [ ] Holdout per-subject accuracy tercatat
- [ ] Semua checkpoint di-backup ke Drive (jangan hanya di local runtime)
- [ ] Notebook evaluasi di-commit ke git (`.ipynb` dengan output cleared atau tidak)

---

## 8. Catatan & Issue Log

| Tanggal | Issue | Solusi | Status |
|---------|-------|--------|--------|
| | | | |

---

## 9. Risiko & Mitigasi Real-Time

| Risiko | Tanda Awal | Mitigasi |
|--------|-----------|----------|
| PLY Direct I/O bottleneck di Colab | Time/epoch > 60s | Switch ke Pre-Augmentasi Offline (§9 Backup Plan di IMPROVEMENT_PLAN) |
| ArcFace loss NaN/instabil | train_acc tidak naik atau loss = nan | Turunkan `ARCFACE_MARGIN` ke 0.30 atau `ARCFACE_SCALE` ke 20.0 |
| Multi-prototype k-means gagal (N < k) | Error di evaluate.ipynb | Set `ENROLL_K = min(3, N)` per subjek |
| Outlier subjek merusak mean metrics | Rank-1 std > 15% | Report dengan dan tanpa outlier subjects |

---

*Checklist ini harus di-update setiap hari selama eksekusi v0.3.0.*
