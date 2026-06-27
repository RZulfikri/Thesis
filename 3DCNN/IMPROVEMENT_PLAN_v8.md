# IMPROVEMENT PLAN v8 — Point Cloud Alignment (Normalisasi) × ArcFace (target IEEE)

**Status:** ▶️ implementasi (kode inti selesai; notebook + run Colab menyusul)
**Fokus (keputusan pembimbing):** dua pilar setara — (1) **alignment/normalisasi** point cloud,
(2) **ArcFace** — untuk submission konferensi IEEE. Dataset: in-house 11 subjek (publik = future work).
**Protokol evaluasi:** multi-frame fusion **N×M (primer N5M5)**, mean (terbukti v7.x, SF→MF −71%).
**Tanpa FPS** (semua varian full-cloud, runtime random-sample 8192).

---

## 1. Latar & temuan pemicu
- v7.2.0: kanonikalisasi menentukan akurasi (R1 raw EER ~9,9% vs R2/R3 ~0%), tapi PCA punya
  **singularitas rotasi 90°** (swap sumbu saat range seri). → v8 memperdalam & **memperbaiki**.
- Temuan kode: `losses/arcface.py` memakai `cosθ − m` (rumus **CosFace**) berlabel "ArcFace";
  head margin di `siamese.py` selalu sama → "cosface/subcenter" v7.x tak benar-benar beda head.
  → v8 menyatukan head (`losses/margin_heads.py`) + **true-ArcFace** `cos(θ+m)`.

## 2. Study A — Alignment / Normalisasi (full-cloud)
Isolasi tiap komponen kanonikalisasi (loss dikunci arcface-linear m0.4 s30 agar reuse C1/C2 valid):
| ID | Mode (`utils/alignment.py`) | Isi | Reuse |
|----|------|------|------|
| A0 | raw | koordinat kamera (output.ply) | v7.2.0 C1 |
| A1 | center | translasi saja | — |
| A2 | centerscale | center + unit-sphere (tanpa rotasi) | — |
| A3 | pca | PCA v7.2.0 (range-Y, median-flip) + unit-sphere | v7.2.0 C2 |
| A4 | pca_robust | **FIX 90°** tanpa landmark (tie-break variance + skewness-sign) | baru |
| A5 | anatomical | **FIX 90° berbasis landmark** (Y=wrist→jari-tengah, X-sign handedness) | baru |

**Validasi numerik (sudah lulus, `utils/alignment.py` self-test):** rotation-invariance —
`pca` TIDAK invarian (maxΔ=1,72 → reproduksi bug 90°); `pca_robust` & `anatomical` INVARIAN (maxΔ=0,0000).
**Eval:** kurva rotasi 0–180° (A3 spike 90°; A4/A5 datar), N×M EER (MF N5M5), DET/ROC, determinism.
**Hipotesis:** H-A1 translasi+scale perlu tak cukup; H-A2 rotasi-kanonik menentukan; H-A3 A4&A5 hapus
90° dgn akurasi ≥ A3; H-A4 A5 (anatomis) ≥ A4 (PCA-deterministik).

## 3. Study B — ArcFace (head pengenalan), pada alignment terbaik
- B1 loss compare: **true-ArcFace** vs CosFace vs SubCenter vs Softmax(CE) vs (arcface-linear ref).
- B2 margin sweep m∈{0.2,0.3,0.4,0.5,0.6}; B3 scale sweep s∈{16,30,64}.
**Hipotesis:** angular/cosine-margin ≫ softmax (d′/open-set); margin moderat optimal; s≈30 stabil.

## 4. Track C — ArcFace Improvement (Colab terpisah, backup gap)
Tangga: Softmax, true-ArcFace, CosFace, SubCenter, **AdaCos**, **CurricularFace**, **QA-ArcFace (usulan)**.
QA-ArcFace: `m_eff = m·(q_floor+(1−q_floor)·q)`, `q∈[0,1]` komposit dari `geometry.json`
(point_count/densitas, scan_distance, palm_depth_std, penalti warnings).
**Bukti pembeda:** EER **terstratifikasi kualitas** (QA-ArcFace unggul di subset kualitas-rendah).

## 5. Komponen kode
- `3DCNN/utils/alignment.py` ✅ — align_points(mode∈{raw,center,centerscale,pca,pca_robust,anatomical}); self-test lulus.
- `3DCNN/losses/margin_heads.py` ✅ — ArcFace(linear/true), CosFace, SubCenter, AdaCos, CurricularFace, QA-ArcFace + factory.
- `3DRegistration/make_align_variants.py` ⏳ — generate align_*.npy dari output.ply (mirror make_fps.py).
- `3DCNN/utils/dataset.py` ⏳ — repr mode align_*, `compute_quality(geometry.json)`, `"quality"` di __getitem__.
- `3DCNN/models/siamese.py` + `train.py` ⏳ — pakai `build_margin_head(loss_type,…)`, `--arcface-variant`, plumb quality.
- `3DCNN/collab/v8_alignment_arcface.ipynb` ⏳ — Study A+B (clone harness v7.2.0; rotation headline; tanpa FPS).
- `3DCNN/collab/v8b_arcface_lab.ipynb` ⏳ — Track C (loss ladder + eval terstratifikasi kualitas).

## 6. Gates
- **Gate-A:** A4/A5 menghapus spike 90° (ΔEER≈0) dgn EER N5M5 ≥ A3 (tidak menurun signifikan).
- **Gate-B:** loss compare benar (head sesuai tipe); true-ArcFace ter-verifikasi (m=0 ≡ softmax-cosine).
- **Gate-C:** QA-ArcFace ≥ ArcFace pada subset kualitas-rendah (interaksi kualitas×margin).

## 7. Compute (A100-80GB, ~23 mnt/seed; cache+resume)
Study A baru A1,A2,A4,A5 × 5 = 20 run (~8j); Study B core ~20 run (~8j); Track C ~20 run (~8j).
Reuse v7.2.0 C1(A0)/C2(A3). Checkpoint cache aman lintas sesi.

## 8. Catatan reproduksibilitas
- Loss Study A = arcface-linear m0.4 s30 (identik v7.2.0) → reuse checkpoint valid.
- `utils/alignment.py` = sumber-tunggal (offline make_align_variants & runtime rotation test → parity).
- Branch dev `claude/gracious-pasteur-9mp0hp`; PR ke `colab`.
