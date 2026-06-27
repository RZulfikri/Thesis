# ARTIFACT: DECISION_MEMO_DATASET
# Created by: Lead Agent
# Date: 2026-05-17T09:15:00+07:00
# Related to: v0.4.0 Fase 2 — Dataset Replacement vs. Proceed
# Status: FINAL

# DECISION MEMO: Replace 3DCNN Dataset with Re-Registered + QC v3?

**Pertanyaan:** Apakah Fase 2 fair ablation harus menunggu dataset baru (re-run 3DRegistration + QC v3), atau boleh lanjut dengan dataset saat ini?

**Verdict: PARALLEL** — Jalankan Fase 2 pilot dengan dataset **saat ini**, sambil menyiapkan dataset baru di background. Dataset baru hanya di-switch jika verifikasi menunjukkan improvement signifikan pada geometri stabilitas.

---

## 1. State of the Current Dataset

| Metric | Value | Source |
|--------|-------|--------|
| Total geometry.json | 2,131 | `find dataset -name geometry.json \| wc -l` |
| Non-quarantine paths | 1,916 | Frame di luar `_QC2_*` dan `_QUARANTINE_*` |
| `is_valid=True` (strict) | 962 | Tanpa whitelist |
| `is_valid=True` (dataset.py whitelist) | **1,916** | `_frame_is_valid` whitelist `scan_distance` + `knuckle_fallback` |
| QC v3 dry-run exclusion | 67 frame (3.5%) | `data_qc_v3_frame.py dataset` |
| QC v3 expected valid | ~1,849 | 1,916 − 67 |
| Subjects | 11 | aisah, alji, chrys, fadhil, feby, gede, nola, rahmat, reysa, taufik, yanuar |
| Sessions | 210 | Non-quarantine |

**Catatan penting:**
- Geometry.json sudah di-*re-extract* pagi ini (2026-05-17 07:56) dengan `extract_geometry.py` terbaru (knuckle fallback + scan_distance threshold 150 mm).
- Artinya: **fitur geometri sudah fres**h. Yang "lama" hanyalah file `output.ply` (hasil registrasi/isolasi frame) dan `cnn_input.npy` (hasil PCA + unit sphere) yang belum di-regenerate.

---

## 2. What Would Change in a "New" Dataset?

Re-running `process_single_frames.py` pada raw `depth*.bin` dengan kode **saat ini** akan menghasilkan perubahan pada `output.ply`:

| Parameter | Old (saat dataset dibuat) | Current (kode sekarang) | Impact |
|-----------|---------------------------|------------------------|--------|
| `outlier_std_ratio` | 1.5 | **1.0** | Lebih agresif — buang lebih banyak noise outlier |
| `radius_outlier_nb_points` | — (tidak ada) | **20** | Tambahan filter radius |
| `radius_outlier_radius` | — (tidak ada) | **0.008 m** | Titik tanpa cukup neighbor dalam 8 mm dibuang |

**Implikasi:**
- Point count per frame akan **turun** (lebih banyak titik dibuang).
- Noise berkurang, tapi detail permukaan halus bisa ikut hilang.
- `cnn_input.npy` (PCA-aligned + unit sphere) akan berubah karena input PLY berubah.
- `geometry.json` akan **sama** dengan yang sudah di-re-extract pagi ini (asumsi `extract_geometry.py` tidak berubah lagi).

---

## 3. Risk Assessment

### 3.1 Risks of Replacing Dataset BEFORE Fase 2

| # | Risk | Severity | Likelihood | Evidence |
|---|------|----------|------------|----------|
| R1 | **Training delay 1–2 hari** | Tinggi | Pasti | Pipeline ~214 sesi × ~10 frame = ~2,140 frame. Proses + copy + QC + upload Drive membutuhkan waktu. |
| R2 | **QC v3 threshold tidak valid lagi** | Sedang | Sedang | QC v3 menggunakan MAD per-session dari distribusi fitur saat ini. Jika re-registration menggeser distribusi (misal: `palm_width` turun karena radius outlier memotong tepi tangan), k=10 bisa terlalu agresif/lemah. |
| R3 | **Point cloud jadi terlalu sparse** | Sedang | Sedang | Radius outlier 0.008 m + std_ratio 1.0 bisa memotong frame yang sebelumnya ~16k titik menjadi <10k. `validate_dataset.py` flag WARN/FAIL. |
| R4 | **Split train/val/test berubah** | Rendah | Pasti | `SPLIT_SEED=42` pada dataset yang berbeda (jumlah frame/session beda) akan menghasilkan split berbeda. Ini **tidak fatal** karena Fase 2 sudah tidak dibandingkan langsung dengan v0.3.0 (beda init parity + beda arsitektur). |
| R5 | **Init parity invalid** | Rendah | Tidak mungkin | Init parity (`audit_init_parity.py`) adalah verifikasi **bobot model** saat inisialisasi, tidak bergantung pada dataset. Perubahan dataset tidak mempengaruhi hasil init parity. Klaim "geometry.json values change → init parity invalid" adalah **misconception**. |
| R6 | **Normalizer Geometry z-score berubah** | Rendah | Pasti | `GeometryNormalizer` di-fit dari training split. Jika dataset baru punya distribusi fitur geometri berbeda, mean/std normalizer berbeda. Ini **fair** selama semua varian Fase 2 pakai dataset yang sama (masing-masing fit normalizer sendiri). |
| R7 | **Fase 2 baseline tidak reproducible** | Rendah | Pasti | Hasil no_geom pada dataset baru tidak bisa dibandingkan dengan no_geom v0.3.0. Namun, Fase 2 adalah **fair ablation internal** (no_geom vs with_geom vs gam_only vs fuse_only) pada kondisi identik, bukan perbandingan historis. |

### 3.2 Risks of Proceeding with Current Dataset

| # | Risk | Severity | Likelihood | Evidence |
|---|------|----------|------------|----------|
| R8 | **output.ply masih pakai parameter cleaning lama** | Rendah | Pasti | Std_ratio 1.5 vs 1.0 berarti masih ada noise outlier yang seharusnya dibuang. Tapi v0.3.0 baseline (Rank-1 99.82% no_geom, 95.82% with_geom) **sudah terbukti bekerja** pada dataset ini. |
| R9 | **Geometry extraction mismatch** | Rendah | Rendah | `geometry.json` sudah di-re-extract dengan kode terbaru, tapi `output.ply`-nya adalah hasil kode lama. Ada kemungkinan kecil inconsistency (misal: knuckle fallback membutuhkan PLY yang lebih "bersih" untuk akurat). Tapi tidak ada bukti empiris bahwa ini menyebabkan error sistematis. |
| R10 | **Missed opportunity dari cleaning lebih agresif** | Rendah | Sedang | Jika noise outlier di PLY lama adalah penyebab dengan_geom performa buruk (bukan init parity / dropout), maka dataset baru bisa memperbaikinya. Tapi diagnostic Fase 1 sudah mengidentifikasi init parity dan dropout sebagai penyebab dominan. |

---

## 4. Verification Steps for the New Dataset

Jika dataset baru disiapkan, verifikasi **wajib** sebelum di-switch:

| # | Step | Pass Criteria | Est. Time |
|---|------|---------------|-----------|
| V1 | Run `process_single_frames.py --force` pada seluruh raw scans | 0 error, semua session diproses | ~1–2 jam |
| V2 | Point count audit: bandingkan old vs new PLY | Tidak ada frame yang turun di bawah 5,000 titik (critical) atau 10,000 titik (warn) | ~15 min |
| V3 | Geometry feature distribution comparison | Mean/std tiap fitur (14 dim) tidak menyimpang >20% dari dataset lama | ~15 min |
| V4 | Apply QC v3 (`data_qc_v3_frame.py --apply`) | Total valid frame ≥ 1,500 (lihat §5) | ~5 min |
| V5 | `scan_dataset_frames()` check | Jumlah frame yang direturn konsisten (±2%) dengan harapan | ~1 min |
| V6 | `validate_dataset.py` pada new dataset | FAIL rate < 5%, tidak ada subject yang kehilangan >50% frame | ~5 min |
| V7 | Smoke test 1 seed, 3 epoch | Loss turun, training tidak crash | ~30 min (Colab) |

---

## 5. Minimum Viable Dataset Requirement for Fair Ablation

Fase 2 membutuhkan dataset yang cukup untuk:
- 11 subjek
- Hold-out 1 sesi per subjek untuk real test
- Split sisa ke train/val/test (70/15/15) di level sesi
- Multi-prototype enrollment k=3 membutuhkan cukup frame per subjek

**Threshold minimal:**
- **Total frame ≥ 1,500** (untuk memastikan train set cukup besar dan hold-out probes representative)
- **Minimum 8 sesi per subjek** (setelah hold-out 1 sesi, sisa 7 sesi untuk train/val/test split)
- **Tidak ada subjek dengan < 100 frame valid** (untuk menghindari class imbalance ekstrem)
- **MAD-based QC v3 exclusion rate < 10%** (jika >10%, threshold k perlu di-tune ulang)

Dataset saat ini (1,916 frame pre-QC, ~1,849 post-QC) **melebihi** threshold ini dengan margin aman.

---

## 6. Verdict & Recommendation

### Verdict: **PARALLEL**

**Alasan:**
1. **Tidak ada bukti kuat** bahwa dataset saat ini defective. v0.3.0 baseline sudah mencapai Rank-1 99.82% (no_geom) pada dataset ini. Masalah utama teridentifikasi adalah **init parity** dan **dropout asimetri**, bukan kualitas registrasi.
2. **Delay risk > benefit risk.** Menunggu dataset baru menunda Fase 2 1–2 hari tanpa jaminan improvement. Colab quota dan timeline thesis sensitif terhadap delay.
3. **Init parity tidak terpengaruh dataset.** Klaim di prompt bahwa re-registration bisa invalidate init parity adalah **salah**. Init parity adalah invariant terhadap data.
4. **Dataset baru bisa disiapkan di background.** Pipeline registrasi lokal (macOS) bisa berjalan paralel dengan training pilot di Colab.

### Rekomendasi Eksekusi

```
┌─────────────────────────────────────────────────────────────┐
│  LANGSUNG (Hari Ini)                                        │
│  1. Jalankan smoke test Fase 2: no_geom, seed=42, 3 epoch   │
│     menggunakan dataset SAAT INI.                           │
│  2. Jika smoke test PASS → lanjutkan Batch A: no_geom × 5   │
│     seed.                                                   │
│                                                             │
│  PARALLEL (Background, Local Mac)                           │
│  3. Re-run process_single_frames.py --force pada seluruh    │
│     raw scans di 3DRegistration/dataset/.                   │
│  4. Copy hasil ke 3DCNN/dataset_new/.                       │
│  5. Apply QC v3, jalankan V1–V6 verification.               │
│                                                             │
│  DECISION GATE (Setelah Batch A selesai & dataset baru      │
│  terverifikasi)                                             │
│  6. Bandingkan smoke test dataset baru vs dataset lama.     │
│  7. SWITCH ke dataset baru HANYA JIKA:                      │
│     - Valid frame count ≥ 1,500                             │
│     - Smoke test loss curve lebih baik atau setara          │
│     - Tidak ada subjek yang kehilangan >30% frame           │
│     Jika tidak memenuhi → LANJUTKAN dengan dataset lama.    │
└─────────────────────────────────────────────────────────────┘
```

### Fallback
- Jika smoke test dataset saat ini GAGAL (loss tidak turun / crash) → **STOP**. Audit dataset & kode dulu sebelum lanjut, terlepas dari dataset baru.
- Jika dataset baru gagal verifikasi V1–V6 → **BUANG**. Dataset lama adalah source of truth untuk Fase 2.

---

## 7. Key Considerations Recap

| Consideration | Current Dataset | New Dataset | Verdict |
|---------------|-----------------|-------------|---------|
| Frame count post-QC | ~1,849–1,916 | Unknown (est. similar) | Comparable |
| Geometry extraction | Fresh (re-extract 07:56) | Fresh | Same |
| PLY cleaning params | Old (std_ratio 1.5) | New (std_ratio 1.0 + radius) | New slightly better |
| Point cloud density | Higher (less aggressive cleaning) | Lower (more aggressive) | Old safer for now |
| Training readiness | **Ready now** | Needs 1–2 days | **Proceed with old** |
| Init parity risk | **None** | **None** | N/A |
| QC v3 threshold validity | Valid for current distributions | May need re-tuning | Old known, new unknown |

---

*Ditulis oleh Lead Agent berdasarkan audit file system, git diff, dan analisis risiko metodologis.*
*Referensi: `3DCNN/AGENTS.md`, `3DRegistration/process_all_scans.py`, `3DRegistration/lib/single_frame.py`, `3DCNN/utils/data_qc_v3_frame.py`, `3DCNN/PLAN_FASE2.md`.*
