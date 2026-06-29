# PAPER_DESIGN — v8 Faktorial (Alignment × Angular-Margin) untuk submission IEEE

Dokumen acuan tetap saat menulis paper. Mengunci judul, klaim, kontribusi, baseline, protokol,
metrik, dan pemetaan eksperimen → figur. Implementasi: `3DCNN/collab/v8_lib.py` (source of truth) +
shell `v8_train_seed{S}.ipynb` (×5) + `v8_eval.ipynb`.

## 0. Judul & framing

**Judul (final):**
> **Canonical Alignment and Angular-Margin Learning for Robust 3D Palm Recognition with PointNet++**

- **Kenapa "Angular-Margin Learning", bukan "ArcFace" di judul:** yang diuji adalah **margin sudut sebagai
  variabel** (softmax = margin 0 vs ArcFace = margin 0.5; head identik). "Angular-margin" = konsep/keluarga
  (akurat menamai variabel + tidak terbaca sebagai "paper aplikasi satu tool"). **ArcFace tetap disebut
  eksplisit di Abstrak + Keywords** untuk searchability: *"...we adopt **ArcFace (additive angular margin)**
  as the angular-margin head..."*.
- **"with PointNet++"**: backbone point-based (lihat §11). **Bukan 3D CNN** — jangan menyebut 3D CNN di paper
  (nama folder repo `3DCNN` hanya warisan penamaan, bukan klasifikasi metode).

## 0a. Hierarki kontribusi (urutan klaim di paper)

1. **(Primer) Kanonikalisasi PCA-robust menutup singularitas 90°** → robustness rotasi sejati (A4, §7a).
2. **(Sekunder) Angular-margin (ArcFace) menaikkan akurasi** pada representasi ter-normalisasi (H2).
3. **(Bingkai pembuka) PointNet++ bisa mencapai EER ~0%** untuk telapak 3D — **membantah** temuan terdahulu
   "PointNet++ lemah untuk biometrik tangan" (Svoboda IJCB'20 ~30–53% acc; Zhang'23 EER 34–47%).
4. **(Bonus) Interaksi**: ArcFace menolong **lebih banyak** justru saat ter-align (paling jelas A4:
   2.18% → 0.24%, ~9×) — hanya terlihat di desain faktorial.

## 1. Klaim (hipotesis)
- **H1 — Normalisasi → robustness.** Representasi point cloud yang ter-kanonikalisasi membuat pengenalan
  **tahan terhadap variasi pose (rotasi sekitar sumbu pandang kamera)**; tanpa normalisasi (raw) akurasi
  runtuh saat dirotasi. **Nuansa (terbukti):** bukan *sembarang* normalisasi — center/scale saja (A1/A2)
  TIDAK robust (runtuh ~33%@90°); PCA polos (A3) **hampir** robust tapi **patah di 90°** (singularitas);
  **hanya A4 (PCA-robust) yang benar-benar datar** termasuk 90°. A4 = bukti H1.
- **H2 — ArcFace → accuracy.** Angular-margin meningkatkan akurasi dibanding softmax. **Nuansa (terbukti):**
  berlaku **pada representasi ter-normalisasi** (A1/A3/A4); pada raw (A0) margin justru memperburuk.
  Framing jujur: *"ArcFace meningkatkan akurasi pada representasi ter-normalisasi."*

## 2. Prinsip anti-sirkular (kenapa faktorial)
**Softmax = basis netral (tak perlu dijustifikasi).** **ArcFace + nilai margin = yang DIUJI**, jadi
tidak boleh dikunci sebagai asumsi. Kita ukur **semua kombinasi sekaligus** (desain faktorial); setiap
klaim terbukti *lintas* faktor lain → bukan kebetulan.

## 3. Desain faktorial
Grid **{softmax, arcface} × {A0..A5}** = 12 config × 5 seed = **60 run**.

| Sumbu | Nilai |
|---|---|
| Representasi (alignment) | A0 raw_ply (**baseline**), A1 center, A2 center+scale, A3 PCA canonical, A4 PCA-robust, A5 anatomical |
| Loss | **softmax** (=`arcface_true` m=0, cosine-softmax; basis terkontrol), **arcface** (=`arcface_true` m=0.5, standar Deng et al. 2019) |

- **H1** dibaca **antar-baris** (alignment), harus berlaku di **kedua kolom** loss.
- **H2** dibaca **antar-kolom** (softmax vs arcface), harus berlaku di **semua** alignment.
- **Interaksi** (bonus): apakah ArcFace menolong lebih banyak saat ter-align.

`softmax` = `arcface_true` m=0 menjaga **arsitektur head identik** → perbandingan H2 terkontrol penuh.

## 3a. Hasil utama (60 run, holdout, N5M5, mean±std)
EER% (↓ lebih baik) — baris alignment, kolom loss:

| alignment | softmax | arcface |
|---|---|---|
| A0 raw_ply | 12.94 ± 2.78 | 16.91 ± 1.42 |
| A1 align_center | 2.52 ± 1.41 | **1.15 ± 1.05** |
| A2 align_centerscale | 0.06 ± 0.12 | 0.06 ± 0.12 |
| A3 canonical_npy | 0.03 ± 0.06 | **0.00 ± 0.00** |
| A4 align_pca_robust | 2.18 ± 2.61 | **0.24 ± 0.21** |
| A5 align_anatomical | 0.18 ± 0.29 | 0.52 ± 0.53 |

**Robustness (EER% vs rotasi-Z):** A0/A1/A2 runtuh (~33–40%@90°); **A3 datar KECUALI paku tajam @90°
(~40%)** = singularitas PCA; **A4 datar penuh** (0.85% → 0.76%@90° → 1.1%@180°). A5 datar tapi lantai ~14%.

**Pemenang (untuk paper):** **A4 (pca_robust) + ArcFace** = Pareto-winner (akurasi ~A3, EER 0.24%, **dan**
robust penuh). A3 lebih akurat di pose-kanonik tetapi **tidak robust** (patah 90°).
> A* otomatis di `analyze()` = EER terendah kolom arcface **pada pose-kanonik** → saat ini **A3**. Untuk
> narasi robustness, **A4** adalah winner; trade-off ini WAJIB dibahas eksplisit.

## 4. Baseline
- Lower-bound = **`(A0 raw, softmax)`** = 12.94% EER.
- Metode penuh = **`(A4, arcface)`** (akurat + robust). A3 sebagai pembanding "PCA polos tak cukup".
- Eksternal (framing): Svoboda IJCB'20 ~30–53% acc; Zhang'23 EER 34–47% → kontras dengan hasil kita.

## 5. Protokol evaluasi
- **Closed-set**: gallery = sesi **train** (template enrol), probe = sesi **HOLDOUT** (33 sesi, pristine).
  **TANPA LOSO.** Semua 11 identitas ada di gallery & probe.
- **Multi-frame fusion N×M** (grid N,M ∈ {1,3,5,10}), headline **N=5, M=5**, strategi mean.
- **5 seed** → mean ± std. Split per-subjek kronologis: train 88 / val 22 / test 22 / **holdout 33** sesi.

## 6. Metrik
| Tujuan | Metrik | Figur/tabel |
|---|---|---|
| Verifikasi (1:1) | **EER** (primer), AUC, d′ | Tabel 6×2, `det_<loss>.png`, `roc_<loss>.png` |
| Identifikasi (1:N) | **rank-1 acc, CMC, confusion** | `cmc_<loss>.png`, `confusion/<cfg>.png` |
| **Robustness (H1)** | EER vs rotasi 0–180° (Δ EER) | `rotation_sensitivity.png` (overlay softmax vs arcface) |
| Kualitatif | t-SNE separabilitas | `tsne/<cfg>.png` |
| Efisiensi | latency, VRAM, disk | `speed_resource.csv` |

## 7. Asumsi scope (robustness = rotasi saja)
QC-gate (`extract_geometry`/`check_scan`/`min_points`/knuckle) menjamin hanya scan **lengkap & bersih**
masuk training/eval → nuisance operasional tersisa = **pose tangan = rotasi**. Jitter/noise/dropout
**di luar scope by design** (sudah tersaring QC), bukan kelalaian.

## 7a. Geometri rotasi & singularitas 90° (untuk Method/Discussion)
- **Sumbu:** Z = depth/garis-pandang kamera (= normal telapak, varians terkecil); Y = arah jari
  (wrist→fingertip, range terpanjang); X = lebar telapak.
- **Rotasi yang diuji = mengelilingi Z** (`_rotz`, in-plane). Setup: HP datar, kamera ke atas, telapak
  menghadap kamera → satu-satunya nuisance = **user memutar tangan di bidang telapak** = rotasi-Z.
  Out-of-plane (pitch/yaw) tersaring QC (scan jadi tak lengkap).
- **Kenapa A3 gagal @90°:** PCA mengembalikan sumbu **tanpa label/tanda** unik. A3 pakai 2 heuristik
  rapuh — pilih Y by *range* (biner) + tanda by *median-Y* (biner). Di ~90° range jari ≈ range lebar →
  keputusan **terbalik/tercermin** → frame kanonik beda → EER ~40%. @180° aman (flip tanda konsisten).
- **Kenapa A4 lolos:** ganti dengan **tie-break varians** (saat nyaris-seri) + **tanda by skewness**
  (momen-3) untuk 3 sumbu → orientasi unik & rotation-invarian. (self-test `utils/alignment._selftest`
  buktikan pca_robust INVARIAN 30/60/90/180°.)
- **A5 (anatomical):** alternatif berbasis **landmark** (wrist→jari + handedness) — robust tapi lantai
  EER lebih tinggi dari A4.

## 8. Pemetaan eksperimen → figur paper
1. **`factorial_eer_heatmap.png` + `factorial_eer_6x2.csv`** — centerpiece (H1 baris + H2 kolom + interaksi).
2. **`rotation_sensitivity.png`** — H1: alignment ter-normalisasi datar; A0 naik; **A3 paku 90°, A4 datar**.
3. **DET/ROC/CMC** = figur **perbandingan** → 1 file per loss, overlay 6 alignment:
   `det_softmax.png`, `det_arcface.png`, `roc_softmax.png`, `roc_arcface.png`, `cmc_softmax.png`, `cmc_arcface.png`.
4. **confusion & t-SNE** = **1 file PER config** (mudah disisipkan satu-satu ke paper):
   `confusion/<cfg>.png` (12 file, berlabel identitas), `tsne/<cfg>.png` (12 file).
5. **`SUMMARY.md`** — ringkasan otomatis.

> **Aturan figur (permintaan user):** JANGAN gabung banyak panel jadi satu gambar besar untuk t-SNE &
> confusion — **satu hasil = satu file** agar bisa disisipkan individual ke paper. DET/ROC/CMC boleh
> overlay (memang figur perbandingan).

## 9. Reproducibility & eksekusi
- Dataset diregenerasi dari `Raw Depth Data/` (raw committed) via `generate_dataset.py`; cache di Drive.
- Checkpoint → Google Drive (Opsi B); resume granular per **unit (cfg_id, seed)** (skip bila `perf.json` ada).
- **Paralel**: 5 file `v8_train_seed{S}.ipynb`, 1 per runtime; 1 unit ≈ 150 epoch (~30 mnt @ GPU 95GB);
  1 seed = 12 unit (~6 jam); 5 runtime paralel → ~6 jam wall-clock 60 unit.
- Evidence (CSV/figur) → `3DCNN/analysis/v8_factorial_<TS>/` (ter-commit). **Re-run `analyze()` instan**
  (baca cache `ablation_results.pkl`) → regen figur tanpa eval ulang.

## 10. Estimasi compute
- Training: 60 run (~6 jam bila 5 runtime paralel @95GB; ~20–30 jam L4 single).
- Eval+analisa: ringan (~menit s/d ~1 jam); re-run analisa instan dari cache.

## 11. Rationale backbone (kenapa PointNet++, bukan CNN/voxel) — untuk Method
Data = **awan titik 3D** (TrueDepth depth → unproject pakai intrinsics → `output.ply`, N×6 xyz+normal).

| Kandidat | Kenapa TIDAK |
|---|---|
| **2D CNN** (depth image) | tergantung sudut pandang; **kanonikalisasi 3D mustahil di 2D** (buang pilar riset); palmprint 2D sudah jenuh |
| **3D CNN (voxel)** | boros O(n³) untuk permukaan tipis (voxel kosong); kuantisasi hapus detail telapak |
| **Heavy point-net** (DGCNN/KPConv/Point Transformer) | lebih haus data → overfit @11 subjek; future work |

**PointNet++ dipilih:** (1) padanan natural point cloud (tanpa voxelisasi/kuantisasi); (2) hirarkis (set
abstraction + ball query) → struktur lokal multi-skala telapak; (3) permutation-invariant + dukung normal;
(4) ringan → aman dataset kecil; (5) memungkinkan kontribusi kanonikalisasi 3D; (6) menjawab gap
"PointNet++ lemah untuk tangan".
> Taksonomi: voxel (3D CNN) | **point-based (PointNet++) ← kita** | multi-view | mesh/graph.

## 12. Tinjauan pustaka — klaster (perluas `docs/literature/`)
**a. Backbone point cloud** — PointNet (Qi 2017), **PointNet++ (Qi 2017)**; konteks DGCNN, KPConv, Point Transformer.
**b. Metric learning / margin loss** — Softmax/CE, **ArcFace (Deng 2019)**, CosFace (Wang 2018), SphereFace,
   SubCenter-ArcFace, AdaCos, CurricularFace, AdaFace, MagFace, Triplet/Contrastive. (kode `losses/margin_heads.py`.)
**c. Kanonikalisasi pose point cloud** ← *pilar* — PCA/principal-axis, OBB; **T-Net/STN** (alignment
   dipelajari, pembanding "kenapa PCA deterministik"); **rotation-invariant / SO(3)-equivariant**: Vector
   Neurons, RIConv/RIConv++, SRINet, ClusterNet (alternatif untuk masalah rotasi — wajib diposisikan);
   analogi face-landmark alignment (justifikasi A5).
**d. Biometrik telapak/tangan 3D** ← *related work inti* — **Svoboda IJCB'20**, **Zhang 2023**; 3D palmprint,
   palm vein, contactless palmprint (banyak ArcFace/CosFace di **2D** → gap = 3D), hand geometry.
**e. Evaluasi biometrik** — EER, ROC/DET, CMC, closed/open-set, score-/multi-frame fusion.
**f. Sensing** — iPhone TrueDepth / structured-light; depth→pointcloud unprojection.
**g. (pembanding) Augmentasi rotasi** sebagai alternatif kanonikalisasi (argumen efisiensi).
> Hedge klaim novelty: *"sepanjang pengetahuan kami / berdasarkan tinjauan pustaka"*. Basis:
> `docs/literature/KEYWORDS_v8_alignment_arcface.md`, `literatur_review_geoatt_pointnet_palmprint.md`
> (perluas klaster c & e yang masih tipis).

## 13. Checklist kelengkapan evaluasi
- [x] Tabel + heatmap EER 6×2 (mean±std)
- [x] Rotation sensitivity (overlay softmax vs arcface, semua alignment)
- [x] DET & ROC — **semua 6 alignment × 2 loss** (1 file per loss; sebelumnya hanya A0 & A3)
- [x] CMC — semua alignment, 1 file per loss
- [x] Confusion matrix — **1 file PER config**, berlabel identitas + Prediksi/Sebenarnya + colorbar + rank-1
- [x] t-SNE — **1 file PER config** (★ gallery / ● probe, warna per identitas)
- [ ] (opsional) speed/resource & determinism table
- [ ] (opsional) uji signifikansi berpasangan softmax vs arcface (per alignment)
- [ ] (opsional) analisa kegagalan: identitas mana tertukar di A0/A3@90°

## Catatan (di luar scope sekarang)
- **Track C / QA-ArcFace** **di-drop** — H1/H2 tuntas oleh grid faktorial. Kode loss tetap di
  `3DCNN/losses/margin_heads.py` bila kelak butuh novelty kedua.
- Margin ArcFace = m=0.5 standar (Deng et al.) + sitasi; margin sweep opsional bila reviewer minta.
