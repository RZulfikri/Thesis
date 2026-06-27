# Improvement Plan v0.3.0: PLY Direct On-the-Fly Augmentation

> **Status**: Revisi 1 — disetujui untuk eksekusi
> **Tanggal**: 2026-05-17
> **Versi**: 0.3.0-rev-1
> **Environment**: Google Colab (notebook-based execution)

---

## 1. Ringkasan Eksekutif

### Masalah yang Diidentifikasi (v0.2.0)
- `with_geom` Rank-1 = 59.8%±2.6%, `no_geom` = 55.5%±13.6% — **tidak signifikan** (Wilcoxon p = 1.0)
- **Loss triplet stagnan ~0.73 sejak epoch 5–10** → bottleneck utama adalah loss/sampling, bukan augmentasi
- **Gallery enrollment naif (mean 1 sesi)** → Rank-1 60% vs Rank-5 92% mengindikasikan template underrepresent intra-class variance
- **EER ~29%** → separasi genuine/impostor sangat lemah, embedding partial collapse
- **Data quality outlier**: feby (0% holdout), nola (33%), reysa (variabel) → kemungkinan scan/registration issue

### Diagnosa Prioritas
| # | Akar Masalah | Dampak | Solusi |
|---|-------------|--------|--------|
| 1 | Loss triplet saturasi di dataset kecil | **Tinggi** | ArcFace / Hybrid loss (sudah di notebook) |
| 2 | Gallery enrollment mean 1 sesi | **Tinggi** | Multi-prototype k-means (sudah di notebook) |
| 3 | Data quality outlier (feby/nola/reysa) | **Tinggi** | Visual QC + re-scan jika perlu |
| 4 | Augmentasi canonical artificial | Sedang | PLY Direct original-space (opsional, gate-gated) |
| 5 | Normals belum dieksplorasi | Rendah | Toggle `in_channels=6` di notebook lama (1 hari) |

### Strategi Utama
**Jalankan Quick Wins (S1–S3) dulu, gate-check hasilnya, baru putuskan apakah PLY Direct masih diperlukan.**

---

## 2. Quick Wins (Kerjakan Sebelum / Paralel dengan PLY Direct)

### S1. Multi-Prototype Enrollment (estimasi: 0.5 hari)
**Status kode**: ✅ Sudah diimplementasi di `utils/enrollment.py` dan di-integrasikan ke `collab/evaluate.ipynb` (commit `3289d5a`).

**Yang perlu dilakukan**:
1. Buka `collab/evaluate.ipynb` dan `collab/evaluate_no_geom.ipynb`
2. Pastikan `ENROLL_STRATEGY = "multi"` dan `ENROLL_K = 3`
3. Jalankan evaluasi ulang pada checkpoint v0.2.0 yang sama (isolasi dampak enrollment)
4. Bandingkan Rank-1 vs baseline v0.2.0 (mean enrollment)

**Hipotesis**: Rank-1 holdout akan naik signifikan karena gallery sekarang punya 3 prototype per subjek yang menangkap variasi intra-class.

### S2. Hybrid ArcFace + Triplet (estimasi: 1 hari)
**Status kode**: ✅ Sudah diimplementasi di `losses/arcface.py` dan di-integrasikan ke `collab/train.ipynb` (commit `35495a4` + `3289d5a`).

**Yang perlu dilakukan**:
1. Buka `collab/train.ipynb` / `collab/train_no_geom.ipynb`
2. Set `LOSS_FN = 'hybrid'` (atau `'arcface'` untuk ablasion)
3. Training 3-phase otomatis aktif:
   - Phase 1: ArcFace pretrain (supervised, 11-class)
   - Phase 2: Hybrid ArcFace+Triplet (70/30)
   - Phase 3: Pure Triplet refinement
4. Jalankan multi-seed (5 seeds) dengan split identik v0.2.0 (`split_seed=42`)

**Catatan implementasi**: `ArcMarginProduct` saat ini menggunakan approximasi linear (`phi = cosine - m`). Ini lebih mirip CosFace/Additive Margin. Untuk v0.3.0 ini cukup, tapi jika hasil kurang optimal, pertimbangkan upgrade ke full `cos(θ + m)` di iterasi berikutnya.

### S3. Data QC Outlier Subjects (estimasi: 0.5 hari investigasi)
**Temuan**: reysa memiliki PLY dengan ukuran sangat bervariasi (min 165KB vs max 417KB → 40% dari max). feby dan nola ukuran normal tapi holdout 0–33%.

**Yang perlu dilakukan**:
1. Visual inspect beberapa frame PLY feby/nola/reysa di `3DCNN/dataset/<subjek>/<session>/frame_*/output.ply`
2. Periksa apakah ada: incomplete cloud, pose ekstrem, atau noise berlebihan
3. Jika scan rusak → re-scan 3 subjek tersebut (biaya rendah, dampak besar)
4. Jika scan normal → masalahnya adalah model/registrasi, bukan data

---

## 3. Decision Gate: Lanjut ke PLY Direct?

Setelah S1 + S2 selesai, ukur ulang pada checkpoint baru:

| Kondisi Hasil | Keputusan |
|--------------|-----------|
| **Rank-1 holdout > 80% & EER < 15%** | PLY Direct di-skip / di-postpone. Fokus ke thesis writing. |
| **Rank-1 holdout 70–80%** | Lanjut **PLY Direct Minimal** (§5.1 Option A) untuk gain marginal |
| **Rank-1 holdout < 70%** | Lanjut **PLY Direct Minimal** + investigasi data QC lebih dalam |

> **Prinsip**: Jangan invest 5–7 hari di pipeline PLY Direct jika bottleneck sebenarnya sudah terselesaikan oleh Quick Wins.

---

## 4. Justifikasi PLY Direct (Jika Gate Terpenuhi)

### 4.1 Literatur Review
*(Tidak berubah dari draft — tetap valid)*
- **Hand PointNet (Ge et al., CVPR 2018)**: Normals meningkatkan representasi surface
- **Svoboda et al. (IJCB 2020)**: PointNet++ baseline lemah (30–53%), butuh semantic regularization
- **Micucci & Iula (2023)**: Multimodal fusion palmprint + geometry → EER 1.18% → 0.06%
- **Zhang et al. (MDPI 2023)**: Multi-view projection lebih efektif dari extractor 3D kompleks untuk dataset kecil

### 4.2 Analisis Teknis: Canonical vs Original Space
*(Tidak berubah dari draft — argumen matematis tetap benar)*

| Aspek | Canonical Space | Original Space |
|-------|----------------|----------------|
| Rotasi Z ±30° | Artificial | Realistis |
| Tilt ±15° | Artificial | Realistis |
| Jitter/Dropout | Tidak ubah PCA | **Mengubah principal components** → canonical frame berbeda |

### 4.3 Overhead
| Operasi | Waktu per Frame | Per Epoch (2131 frame) |
|---------|----------------|------------------------|
| Load PLY | ~0.5 ms | ~1.1 s |
| Augmentasi original | ~0.2 ms | ~0.4 s |
| PCA-align (SVD 3×3) | ~0.3 ms | ~0.6 s |
| Normalize + Sample | ~0.1 ms | ~0.2 s |
| **Total** | **~1.1 ms** | **~2.3 s** |

Baseline training A100: ~15–30s/epoch → overhead **+8–15%**, acceptable.

---

## 5. Rencana Implementasi PLY Direct (Gate-Dependent)

### 5.1 Scope PLY Direct (Pilih Satu)

#### **A. Minimal** (Rekomendasi)
- `PLYDirectDataset` load `output.ply` → augment original space → PCA-align → normalize
- `OriginalSpaceAugmentor` (rotasi Z, tilt, translate, scale, jitter, dropout)
- Integrasi ke `collab/train.ipynb` via toggle `DATASET_MODE = 'ply'`
- **Tidak termasuk**: normals pipeline, pre-aug offline
- **Estimasi**: 2–3 hari

#### B. Full
- Minimal + normals extraction + pre-aug offline backup
- **Estimasi**: 5–7 hari (tidak direkomendasikan untuk timeline saat ini)

### 5.2 Komponen yang Perlu Dibangun (Minimal)

#### `utils/ply_dataset.py` — `PLYDirectDataset`
```python
class PLYDirectDataset(Dataset):
    """
    Load PLY asli, augmentasi di original space, PCA-align, normalize.
    """
    def __init__(self, root_dir, n_points=8192, augment=True,
                 use_normals=False, geom_dim=33, use_geom=True):
        
    def _load_ply(self, ply_path):
        """Parse PLY binary, return (N, 3) float32 points."""
        
    def _pca_align(self, pts):
        """
        Replicate preprocess_for_cnn.py logic:
        1. Center points
        2. SVD on centered points
        3. Y-axis = component with largest range (finger direction)
        4. Z-axis = component with smallest variance (depth)
        5. Ensure right-handed coordinate system
        6. Flip check: fingers point to +Y
        """
        
    def _augment_original_space(self, pts):
        """
        - Rotation Z: ±30°
        - Tilt X/Y: ±15°
        - Translate: ±2cm
        - Scale: 0.9–1.1
        - Jitter: σ = 2mm
        - Dropout: 10–15%
        """
```

**Kritis**: PCA-align harus **identik** dengan `preprocess_for_cnn.py`. Unit test: output PLYDirect (tanpa augmentasi) harus identik dengan `cnn_input.npy`.

#### `utils/augmentation.py` — `OriginalSpaceAugmentor`
```python
class OriginalSpaceAugmentor:
    def __init__(self,
                 rot_z=30.0, tilt=15.0, translate=0.02,
                 scale=(0.9, 1.1), jitter=0.002, dropout=0.15):
```

### 5.3 Integrasi ke Notebook Colab
- `collab/train.ipynb`: tambah cell konfigurasi `DATASET_MODE = 'ply'` / `'npy'`
- `collab/train_no_geom.ipynb`: sama
- **Tidak perlu modifikasi `train.py` CLI** — eksekusi utama via notebook

---

## 6. Ablation Plan (Dipotong dari 6 → 3 Eksperimen)

| Exp | Setup | Loss | Enrollment | Pertanyaan |
|-----|-------|------|------------|------------|
| **E1** | `npy` (baseline v0.2.0) | Triplet | **Multi-prototype** (S1) | Apakah enrollment fix Rank-1 gap? |
| **E2** | `npy` | **ArcFace / Hybrid** (S2) | Multi-prototype | Apakah loss yang bottleneck, bukan augmentasi? |
| **E3** | `ply` (original-space aug) | Hybrid | Multi-prototype | Apakah augmentasi realistic memberi gain marginal? |

**Fairness Constraint**:
- Semua seed identik: `[7, 42, 123, 2026, 31337]`
- Semua split identik (`split_seed=42`)
- Hanya `USE_GEOM` flag yang berbeda untuk ablation with_geom vs no_geom

**Total run**: 3 eksperimen × 2 configs (with/no_geom) × 5 seeds = **30 run** — feasible di A100 Colab.

---

## 7. Normals Ablation (Opsional, 0.5–1 hari)

**Pendekatan paling murah** (sebelum bangun pipeline PLY+normals):
1. `cnn_input.npy` sudah menyimpan `(N, 6)` (xyz + nxnynz)
2. Di `collab/train.ipynb`, modifikasi satu baris:
   ```python
   model = SiamesePalmNet(..., in_channels=6)  # default 3
   ```
3. Train ulang 1–2 seed dengan loss hybrid
4. Jika Rank-1 tidak naik → skip normals di PLY Direct
5. Jika Rank-1 naik >3% → pertimbangkan include normals di PLY Direct future work

---

## 8. Risk Analysis

| Risk | Probability | Impact | Mitigasi |
|------|-------------|--------|----------|
| PCA-align training ≠ preprocessing | Medium | **High** | Unit test: PLYDirect output == cnn_input.npy (tanpa aug) |
| PCA-drift karena augmentasi | Medium | Medium | Gallery harus di-recompute via PLYDirect juga; jangan pakai `cnn_input.npy` lama |
| Overhead PLY I/O di Colab Drive | Medium | Medium | Benchmark 1 epoch dulu; fallback ke pre-aug offline |
| Flip-check unstable di pose ekstrem | Low | Medium | Monitor flip rate distribution; clamp jika >10% |
| ArcFace approximasi linear kurang efektif | Low | Medium | Jika E2 gagal, upgrade ke full `cos(θ + m)` |

---

## 9. Backup Plan: Pre-Augmentasi Offline

Jika PLY Direct on-the-fly terlalu lambat di Colab Drive:

```python
# One-time preprocess (2 menit di local / Colab)
for each frame:
    for i in range(20):
        aug = OriginalSpaceAugmentor()
        pts_aug = aug(ply_points)
        pts_canonical = pca_align(pts_aug)
        np.save(f'cnn_input_aug{i}.npy', pts_canonical)

# Training: load random cnn_input_aug{i}.npy per epoch (cepat)
```

**Trade-off**: 20× disk space (~16 GB) tapi training tetap cepat.

---

## 10. Timeline (7 Hari)

| Hari | Task | Output | Catatan |
|------|------|--------|---------|
| 1 | **E1**: Eval multi-prototype pada checkpoint v0.2.0 | Rank-1 vs baseline | Buka evaluate.ipynb, set `ENROLL_STRATEGY='multi'`, rerun |
| 1 | **S3**: Visual QC feby/nola/reysa | Go/No-go re-scan | Inspeksi PLY di viewer |
| 2 | **E2**: Training Hybrid ArcFace+Triplet (5 seed) | Checkpoint baru | train.ipynb, `LOSS_FN='hybrid'` |
| 3 | Evaluasi E2 + perbandingan E1 vs E2 | Laporan perbandingan | evaluate.ipynb |
| 3 | **Decision Gate**: Rank-1 > 80%? | Keputusan lanjut PLY Direct | Lihat §3 |
| 4–5 | **E3** (jika gate terpenuhi): Implementasi PLY Direct Minimal + training | Checkpoints + eval | `DATASET_MODE='ply'` |
| 6 | Normals quick ablation (jika masih ada waktu) | 2-seed comparison | Toggle `in_channels=6` |
| 7 | Analisis hasil + final report | Report + plots | Bandingkan E1→E2→E3 |

---

## 11. Keputusan yang Diperlukan

### 11.1 [DECISION] Scope v0.3.0
- **A. Minimal + Quick Wins** (Rekomendasi): E1–E3 + S1–S3. PLY Direct hanya jika gate terpenuhi.
- **B. Full**: Minimal + Normals + Pre-aug + semua ablation. **Tidak direkomendasikan** untuk timeline 7 hari.

### 11.2 [DECISION] Dataset Mode (jika gate terpenuhi)
- **A. PLY Direct On-the-Fly**: Load PLY setiap epoch. Overhead ~1ms/frame.
- **B. Pre-Augmentasi Offline**: Generate 20 variants/frame sekali. 16GB disk.
- **C. Hybrid** (Rekomendasi): PLY Direct utama, Pre-aug sebagai fallback jika I/O bottleneck.

### 11.3 [DECISION] Normals
- **A. Skip dulu, quick ablation via cnn_input.npy** (Rekomendasi): Toggle `in_channels=6` di notebook lama untuk validasi cepat.
- **B. Include di PLY Direct**: Hanya jika quick ablation menunjukkan gain >3%.

---

## 12. Checklist Pre-Eksekusi

- [ ] `collab/evaluate.ipynb` dan `evaluate_no_geom.ipynb` sudah di-sync ke Drive
- [ ] `utils/enrollment.py` tersedia di Drive (commit `3289d5a`)
- [ ] `losses/arcface.py` tersedia di Drive (commit `35495a4`)
- [ ] Checkpoint v0.2.0 (with_geom & no_geom) masih tersedia di `runs/`
- [ ] `split_seed=42` dan seeds `[7, 42, 123, 2026, 31337]` tercatat untuk reproducibility
- [ ] PLY files feby/nola/reysa sudah diinspeksi visual

---

## 13. Referensi

1. Ge et al., "Hand PointNet: 3D Hand Pose Estimation Using Point Sets", CVPR 2018
2. Svoboda et al., "Clustered Dynamic Graph CNN for Biometric 3D Hand Shape Recognition", IJCB 2020
3. Zhang et al., "Lightweight CNN for Palmprint Recognition with 3D Point Cloud", MDPI 2023
4. Micucci & Iula, "Multimodal Fusion of Palmprint and Hand Geometry from 3D Ultrasound", 2023
5. Liu et al., "Deep Learning in Palmprint Recognition: A Comprehensive Survey", 2025
6. Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets", NeurIPS 2017

---

*Dokumen ini adalah revisi dari draft v0.3.0-draft-1 setelah review evaluasi. Perubahan utama: prioritas Quick Wins (S1–S3) sebelum PLY Direct, pemotongan ablation matrix 6→3 eksperimen, penambahan decision gate criteria, dan penyesuaian untuk eksekusi Colab-only.*
