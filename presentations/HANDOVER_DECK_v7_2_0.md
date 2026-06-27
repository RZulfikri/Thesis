---
marp: true
title: "3D Palm Recognition — Progress & Research Status (v1 → v7.2.0)"
theme: palm
paginate: true
---

<!--
================================================================================
HANDOVER DECK — untuk Agen Desain Claude
================================================================================
TUJUAN: dokumen serah-terima (handover) berisi SELURUH isi deck progress/status
riset "3D Palm Recognition". Agen desain membaca file ini BERSAMA `design.md`
(sistem visual) dan folder `assets/` (gambar bukti) untuk merender deck final.

CARA BACA:
- Tiap slide dipisah `---`.
- Tag `[TYPE]` di tiap slide → memetakan ke komponen di design.md §4.
- "Aset:" = file gambar di presentations/assets/.
- "Takeaway:" = 1 kalimat kesimpulan WAJIB ditampilkan (slide bukti).
- "Catatan render" = instruksi khusus warna/penekanan (ikuti semantik design.md §2).
- Angka ditulis verbatim dari sumber; jangan diubah. Sumber dicantумkan di caption.
- Bahasa Indonesia; istilah teknis tetap asli.

SUMBER ANGKA (traceable):
- VERSION.md, 3DCNN/IMPROVEMENT_PLAN_v7.0.0.md, 3DCNN/collab/VERSIONS.md
- 3DCNN/result_docs/.../LAPORAN_*.md (v7_lowdata, v7_1_1, v7_2_0)
- 3DCNN/analysis/v7_1_1_20260605_083050/*.csv, v7_2_0_20260614_033631/*.csv
- docs/literature/literatur_review_geoatt_pointnet_palmprint.md
================================================================================
-->

# BAGIAN 0 — PEMBUKA

---
<!-- [COVER] -->
## (Slide 1) Cover
**Judul:** 3D Palm Recognition dengan PointNet++
**Sub-judul:** Progress & Status Riset — dari GeoAtt-PointNet++ menuju Pure PointNet++ + ArcFace + Canonical
**Baris kecil:** Rahmat Zulfikri · 15 Juni 2026 · Ringkasan eksperimen v1 → v7.2.0
**Catatan render:** latar `midnight`, judul putih besar; opsional latar samar `assets/data_ply_rahmat.png`.

---
<!-- [CONTENT] -->
## (Slide 2) Agenda
- **Masalah & motivasi** — biometrik telapak tangan 3D berbasis point cloud
- **Data & pipeline** — iPhone TrueDepth → 3DRegistration → 3DCNN
- **Fondasi metode** — PointNet++, ArcFace, 3 representasi (R1/R2/R3)
- **Riwayat eksperimen v1 → v7.2.0** — apa yang dicoba & ditemukan
- **Blocker GeoAtt & reframing** ke pure PointNet++
- **Ablation testing & hasil** — loss sweep, multi-frame fusion, representasi
- **Gap riset & kontribusi** + keterbatasan & future work

Takeaway: Deck ini menelusuri **alur progres** eksperimen secara berurutan hingga temuan final.

---

# BAGIAN 1 — MASALAH & MOTIVASI

---
<!-- [SECTION] -->
## (Slide 3) Bagian 1 — Masalah & Motivasi
Mengapa telapak tangan 3D, mengapa point cloud, dan apa pertanyaan risetnya.

---
<!-- [CONTENT] -->
## (Slide 4) Mengapa biometrik telapak tangan 3D?
- Telapak tangan: area luas, fitur kaya (garis utama, kerutan, **geometri jari/telapak**).
- **Point cloud 3D** mempertahankan kedalaman & struktur permukaan yang hilang pada citra 2D → lebih tahan variasi pencahayaan, pose, dan **spoofing**.
- Tren riset: deep learning untuk palmprint (closed-set, open-set, multimodal) berkembang pesat (Liu et al. 2025).

Aset: `assets/lit_timeline.png` (timeline riset biometrik tangan 3D 2009–2025)
Takeaway: Point cloud 3D adalah modalitas menjanjikan, namun pemrosesannya menantang.

---
<!-- [CONTENT] -->
## (Slide 5) Pipeline end-to-end
```
iPhone TrueDepth  →  3DRegistration  →  3DCNN (PointNet++)
 (Swift/iOS)         (Python/Open3D)     (PyTorch)
 depth.bin +         per-frame:          training & evaluasi
 calibration.json    isolasi palm,       (EER, AUC, CMC, ...)
                     normalisasi
```
- **Akuisisi:** depth frame + kalibrasi kamera dari sensor Face ID.
- **Preprocessing:** proyeksi 3D, pembersihan, isolasi telapak, (opsional) kanonikalisasi.
- **Recognition:** embedding 128-d via PointNet++, dicocokkan dengan cosine similarity.

Takeaway: Sistem tiga tahap; fokus tesis pada tahap recognition + representasi input.

---
<!-- [CONTENT] -->
## (Slide 6) Pertanyaan riset
1. Loss & arsitektur apa yang membuat PointNet++ bekerja untuk identifikasi telapak?
2. Seberapa penting **preprocessing kanonikalisasi** (PCA-align + unit-sphere)?
3. Apakah **multi-frame fusion** (gabung beberapa frame) menurunkan error?
4. Representasi mana yang **Pareto-optimal** (akurasi × kecepatan × disk) untuk deployment?

Takeaway: Empat pertanyaan ini memandu seluruh rangkaian eksperimen v1–v7.2.0.

---

# BAGIAN 2 — DATA & DATASET

---
<!-- [SECTION] -->
## (Slide 7) Bagian 2 — Akuisisi Data & Dataset

---
<!-- [FIGURE] -->
## (Slide 8) Scanner iOS TrueDepth
Aset: `assets/data_ios_ready.png` (kiri) + `assets/data_ios_history.png` (kanan)
- Aplikasi Swift menangkap depth frame (640×480 Float32, meter) + `calibration.json` (fx, fy, cx, cy, distorsi).
- Hand ROI detection (Vision), ekspor per sesi: `depthNN.bin`, `calibration.json`, `metadata.json`.
Takeaway: Data biometrik nyata diakuisisi langsung dari sensor TrueDepth iPhone.

---
<!-- [FIGURE] -->
## (Slide 9) Contoh hasil point cloud (output.ply)
Aset: `assets/data_ply_alji_1.png` + `assets/data_ply_rahmat.png`
- Tiap frame → point cloud telapak ter-isolasi (~15–20 rb titik, xyz + normals).
Takeaway: Telapak ter-segmentasi bersih, siap jadi input model.

---
<!-- [METRICS] -->
## (Slide 10) Dataset (regen v7.2.0)
- **11 subjek** · **214 sesi** · **2.131 frame** (0 gagal, 0 QC-fail)
- Per subjek: aisah 200, alji 150, chrys 200, fadhil 150, feby 210, gede 200, nola 221, rahmat 150, reysa 250, taufik 200, yanuar 200
- QC gate berbasis **point-cloud** (PLY ≥ min-points DAN scan_distance ∈ [150,450] mm)
Sumber: VERSION.md §v7.2.0
Takeaway: Dataset kecil tapi terkontrol — **11 subjek** menjadi batas daya statistik yang konsisten dibahas.

---

# BAGIAN 3 — FONDASI METODE

---
<!-- [SECTION] -->
## (Slide 11) Bagian 3 — Fondasi Metode

---
<!-- [CONTENT] -->
## (Slide 12) PointNet++ singkat
- Mengonsumsi point cloud langsung (tanpa voxel/citra).
- **Hierarchical Set Abstraction**: tiap level = **FPS** (pilih centroid) → **ball query** (tetangga lokal) → mini-PointNet.
- Menangkap struktur lokal multi-skala; fondasi mayoritas riset point cloud biometrik.
Aset: `assets/lit_accuracy_pointnetpp.png` (perbandingan akurasi metode point cloud)
Takeaway: PointNet++ kuat untuk shape, tetapi literatur menilainya **lemah** untuk biometrik tangan (lihat Bagian 7).

---
<!-- [CONTENT] -->
## (Slide 13) ArcFace & protokol matching
- **ArcFace** (additive angular margin) memaksa margin sudut antar kelas → embedding lebih terpisah.
- **Siamese / metric learning**: embedding 128-d, cocokkan via **cosine similarity**.
- **Multi-frame fusion** (deployment-realistic): enroll **N** frame × probe **M** frame, fuse = rata-rata embedding.
Takeaway: Kombinasi ArcFace + multi-frame fusion adalah tulang punggung protokol evaluasi v7.x.

---
<!-- [COMPARE] -->
## (Slide 14) Tiga representasi input (R1/R2/R3)
| **R1 — raw_ply** (`rust`) | **R2 — canonical_npy** (`azure`) | **R3 — fps_npy** (`jade`) |
|---|---|---|
| `output.ply` apa adanya | PCA-align + unit-sphere | R2 + FPS 8192 (fixed) |
| koordinat kamera (~0,25) | kanonik, radius ≤ 1 | kanonik, jumlah titik tetap |
| tanpa kanonikalisasi | baseline kanonik | hemat disk, tanpa runtime sampling |
| **kontrol negatif** | **acuan** | **kandidat deployment** |
Takeaway: Ketiganya dari sumber sama → ablation mengisolasi **kanonikalisasi** (R1→R2) dan **FPS vs random** (R2→R3).

---
<!-- [CONTENT] -->
## (Slide 15) Bagaimana representasi dibuat (algoritma)
- **Pipeline hulu (sama untuk semua):** depth.bin → undistort (lookup) → **proyeksi pinhole** `X=(u−cx)/fx·z` → estimasi normal (KDTree) → voxel 1 mm → outlier removal → **isolasi palm DBSCAN** → `output.ply`.
- **R2 canonical:** PCA via **SVD**; sumbu **Y = rentang (ptp) terbesar** (arah jari, bukan variance), Z = varians terkecil; right-handed; disambiguasi median-Y; **unit-sphere** (bagi norm maks).
- **R3 fps:** **Farthest Point Sampling greedy** (Open3D) dari R2 → 8192 titik bercakupan merata.
Sumber: `LAPORAN_v7_2_0.md` §3
Takeaway: Setiap langkah eksplisit & deterministik — penting untuk reproduksibilitas.

---
<!-- [CONTENT] -->
## (Slide 16) Justifikasi pemilihan algoritma
- **PCA/SVD** dipilih (vs ICP-to-template / learned alignment): unsupervised, murah, deterministik, dan sumbu telapak (jari/lebar/depth) selaras alami dengan principal axes. *Trade-off:* ambiguitas sumbu (lihat anomali 90°).
- **RANGE (ptp) bukan variance** untuk sumbu Y: saat jari terbuka, variance horizontal bisa ≥ vertikal, tapi rentang arah jari selalu paling panjang.
- **unit-sphere**: skala-invarian (jarak scan bervariasi) + cocok dengan asumsi radius `ball_query` PointNet++.
- **FPS (vs random/voxel)**: cakupan permukaan merata + jumlah titik tetap (ramah batch) + lebih deterministik.
Takeaway: Pilihan algoritma berbasis sifat data telapak + kebutuhan PointNet++, dengan trade-off yang didokumentasikan.

---

# BAGIAN 4 — RIWAYAT EKSPERIMEN (v1 → v7.2.0)

---
<!-- [SECTION] -->
## (Slide 17) Bagian 4 — Riwayat Eksperimen, Versi demi Versi
Hipotesis awal: **GeoAtt-PointNet++** (fusi fitur geometri). Lalu menemui blocker → reframing.

---
<!-- [TIMELINE] -->
## (Slide 18) Garis waktu versi
| Versi | Tgl | Inti | Metrik kunci |
|---|---|---|---|
| v1 | 23 Apr | PoC (contrastive, 6 subjek) | Rank-1 **89,47%** |
| v2 | 16 Mei | Triplet, 11 subjek, LOSO | EER **~28,95%** (bottleneck) |
| v3 | 17 Mei | **ArcFace** | EER **0,03%** no_geom ⚠️ leakage |
| v4 | 17–21 Mei | Diagnostik fairness | init parity 13/33 → **58/58** |
| v5 | 21–24 Mei | Pivot low-data | GeoAtt **merugikan** (p=0,002) |
| v6 | 25 Mei | Drop GeoAtt, fair split | ArcFace d′ **3,40** vs Triplet 1,51 |
| v7.1.0 | 30 Mei | Multi-frame + 8-loss | arcface_m04 **1,32%** |
| v7.1.1 | 5 Jun | Re-run @dataset regen | arcface_m04 **1,14%** |
| v7.2.0 | 14 Jun | Ablation R1/R2/R3 | R1 **9,9%** vs R2/R3 **0%** |
**Catatan render:** node pivot (v5, v6) warna `amber`; hasil positif (v7.x) `jade`; v3 beri badge ⚠️.
Takeaway: Sembilan versi: dari PoC → krisis metodologi → reframing → hasil final.

---
<!-- [CONTENT] -->
## (Slide 19) v1 — Proof of Concept (23 Apr)
- Hipotesis: **GeoAtt-PointNet++** bisa belajar embedding diskriminatif end-to-end.
- Data: **6 subjek**, 4096 titik, loss **Contrastive**; split acak (belum session-aware).
- Hasil: **Rank-1 89,47%** (17/19 sesi tes benar). EER/AUC belum dihitung.
Takeaway: Konsep terbukti layak; perlu skala & loss yang lebih kuat.

---
<!-- [TABLE] -->
## (Slide 20) v2 — Skala-up + Triplet (16 Mei): bottleneck
| Metrik | with_geom | no_geom |
|---|---|---|
| Rank-1 | **59,82% ± 2,64%** | 55,45% ± 13,55% |
| EER | **28,95% ± 2,13%** | 28,45% ± 4,66% |
| AUC | 78,38% | 78,65% |
- 11 subjek, split LOSO, 8192 titik, **Online Triplet** (margin 0,3), 5 seed.
- Wilcoxon p=1,000 (tak signifikan); GeoAtt hanya menurunkan variansi (efek regularisasi).
Takeaway: **Loss adalah bottleneck** — Triplet mentok ~60% Rank-1 → memicu pindah ke ArcFace.

---
<!-- [TABLE] -->
## (Slide 21) v3 — Revolusi ArcFace (17 Mei)
| Metrik | with_geom | no_geom |
|---|---|---|
| Rank-1 | 95,82% ± 1,59% | **99,82% ± 0,36%** |
| EER | 2,76% ± 1,41% | **0,03% ± 0,04%** |
| TAR@FAR=1% | 92,87% | **100,00%** |
- Hanya loss yang berubah (Triplet → ArcFace m=0,5 s=30): **Rank-1 +40 ppt**.
- **Temuan tak terduga:** GeoAtt justru **merugikan** (McNemar p=1,8×10⁻⁵, no_geom menang 23 vs 1).
**Catatan render:** badge `amber` "⚠ angka ini kemudian dikoreksi: ada temporal data-leakage".
Takeaway: ArcFace melompatkan performa; tapi GeoAtt mulai tampak kontra-produktif.

---
<!-- [FINDING] -->
## (Slide 22) Krisis metodologi (22 Mei) — "blocker" GeoAtt
Tiga bias membuat hasil GeoAtt **tidak bisa dipakai** sebelum diperbaiki:
1. **Temporal split leakage** — train/test/holdout dari burst tangkapan yang sama (< 2 menit).
2. **val_loss anti-korelasi** dengan generalisasi — varian geometri punya val_loss terendah tapi test EER **tertinggi** (gam_only test EER **27%**).
3. **Training budget bias** — early-stopping pada metrik bias → jumlah epoch beda antar varian.
Sumber: `result_docs/20260522_092309/KESIMPULAN_REPORT.md`
**Catatan render:** callout border kiri `rust`.
Takeaway: Perbandingan GeoAtt awal **invalid** secara metodologis → wajib desain ulang.

---
<!-- [CONTENT] -->
## (Slide 23) v4 — Diagnostik fairness (17–21 Mei)
- **RNG init parity rusak:** 13/33 layer ter-inisialisasi beda meski seed sama → diperbaiki menjadi **58/58 layer identik** (semua sub-modul selalu dibangun, flag hanya mengubah forward).
- **Fitur geometri tidak noise** (median FDR 3,77), TAPI `nola.finger_width_5` **CV 0,497** = 8,85× outlier (tidak stabil antar-sesi).
- QC v3 frame-level: buang 160 frame (7,5%).
Aset: `assets/v4_variant_heatmap.png` (heatmap metrik 4 varian: no_geom/with_geom/gam_only/fuse_only)
Takeaway: Setelah fairness diperbaiki, GeoAtt **tetap tidak unggul** → cost-benefit gagal.

---
<!-- [COMPARE] -->
## (Slide 24) v4 — Bukti visual: GeoAtt vs pure PointNet++
| **no_geom (pure)** (`jade`) | **with_geom (GeoAtt)** (`rust`) |
|---|---|
| t-SNE: klaster rapat & terpisah | t-SNE: klaster bercampur |
| DET lebih rendah | DET lebih tinggi |
Aset kiri: `assets/v4_no_geom_tsne.png` · Aset kanan: `assets/v4_with_geom_tsne.png`
(opsional baris kedua: `assets/v4_no_geom_det.png` vs `assets/v4_with_geom_det.png`)
Takeaway: Secara visual pun, **pure PointNet++ memisahkan subjek lebih baik** daripada GeoAtt.

---
<!-- [TABLE] -->
## (Slide 25) v5 — Pivot low-data (24 Mei): GeoAtt merugikan
| Metrik | no_geom | with_geom | p |
|---|---|---|---|
| Test EER | **0,05 ± 0,00** | 0,425 ± 0,226 | — |
| Holdout EER | **0,035 ± 0,027** | 0,375 ± 0,104 | **0,002** |
- Rezim 1 frame/sesi (10 subjek × 150 frame); fokus geser ke **effect size & multi-frame**, bukan p-value.
Aset: `assets/v5_boxplots.png`
**Catatan render:** kolom with_geom `rust`.
Takeaway: Di rezim low-data, GeoAtt **jelas merugikan** (p=0,002) → diputuskan **drop GeoAtt**.

---
<!-- [TABLE] -->
## (Slide 26) v6 — Drop GeoAtt; ArcFace vs Triplet (25 Mei)
| Metrik | ArcFace | Triplet | catatan |
|---|---|---|---|
| Test EER | **6,0% ± 3,16%** | 6,5% ± 4,74% | p=1,00 (NS) |
| d-prime (test) | **3,40** | 1,51 | **2,25× lebih baik** |
| d-prime (holdout) | **4,96** | 2,96 | 1,68× |
- Pure PointNet++ (use_geom=False) jadi default; RNG init-parity fix membuat perbandingan adil.
Aset: `assets/v6_boxplots.png`
Takeaway: ArcFace dipilih karena **separabilitas (d′) jauh lebih baik** — meski EER belum signifikan pada N kecil.

---
<!-- [FINDING] -->
## (Slide 27) Reframing: dari GeoAtt → Pure PointNet++
- Hipotesis awal (fusi geometri) **gugur**: GeoAtt tidak membantu (bahkan merugikan di ArcFace) + perbandingan awal tercemar leakage.
- **Arah baru:** pure PointNet++ + ArcFace + (nanti) kanonikalisasi & multi-frame fusion.
- **Mengapa relevan:** literatur menilai PointNet++ *lemah* untuk biometrik tangan (Bagian 7) → ruang kontribusi terbuka.
**Catatan render:** border kiri `jade`; sertakan badge "negative result = contribution".
Takeaway: **Negative result GeoAtt** mengarahkan riset ke pendekatan yang lebih sederhana & ternyata lebih kuat.

---
<!-- [SECTION] -->
## (Slide 28) Bagian 4b — Ablation Testing (v7.x)
Multi-frame fusion, loss sweep, dan representasi — semuanya pure PointNet++.

---
<!-- [TABLE] -->
## (Slide 29) v7.1.0 — Loss sweep 8 varian (30 Mei, MF N5M5)
| Varian | SF EER | **MF EER (N5M5)** | Δ (improve) |
|---|---|---|---|
| **arcface_m04** 🥇 | 4,55% | **1,32% ± 1,42%** | −71,0% |
| arcface_m03 | 6,82% | 1,77% ± 2,45% | −74,0% |
| hybrid | 4,55% | 1,75% ± 1,61% | −61,5% |
| arcface_s64 | 5,00% | 1,75% ± 3,19% | −65,0% |
| cosface | 5,45% | 1,93% ± 2,83% | −64,6% |
| subcenter | 6,36% | 2,02% ± 2,17% | −68,2% |
| arcface_m05 | 4,55% | 2,30% ± 1,94% | −49,5% |
| standard (Triplet) | 4,55% | 3,86% ± 3,98% | −15,1% |
Sumber: `analysis/v7_1_1...` & LAPORAN; 10 seed, 11 subjek.
Takeaway: Semua margin-loss rapat (<2,1%); **arcface_m04 dipilih sebagai anchor**.

---
<!-- [FIGURE] -->
## (Slide 30) v7.1.0 — Multi-frame fusion = tuas terbesar
Aset: `assets/v710_ablation_heatmap.png` (grid EER N×M)
- SF → MF (N5M5): EER **4,55% → 1,32%** (turun **71%**).
- Efek fusi (15–74% penurunan) **jauh melampaui** beda antar-loss (maks ~2,3 pp).
Takeaway: **Multi-frame fusion** adalah pengungkit performa terbesar dalam pipeline (H1 terkonfirmasi).

---
<!-- [TABLE] -->
## (Slide 31) v7.1.0 — Open-set LOSO (generalisasi ke subjek baru)
| Varian | Closed EER | FAR@unknown | d-prime |
|---|---|---|---|
| **subcenter** 🥇 | **1,48% ± 0,93%** | **13,64%** | **4,14** |
| hybrid | 1,57% | 13,18% | 2,59 |
| cosface | 1,69% | 15,00% | 3,64 |
| arcface_m04 | 2,08% | 18,18% | 3,91 |
Takeaway: Untuk **penolakan orang asing** (open-set), subcenter/hybrid unggul — relevan untuk future work.

---
<!-- [TABLE] -->
## (Slide 32) v7.1.1 — Re-run di dataset regen (5 Jun)
| Varian | v7.1.0 (lama) | v7.1.1 (regen) | Δ |
|---|---|---|---|
| **arcface_m04** (anchor) | 1,32% ± 1,42% | **1,14% ± 1,18%** | −0,18 pp ✅ |
| cosface | 1,93% | 0,36% ± 0,37% | (numerik #1) |
- Re-run loss sweep di dataset baru agar **anchor sebanding** dengan ablation v7.2.0.
- Winner numerik bergeser ke cosface, **tapi dalam noise** (selisih < 1σ arcface_m04).
Aset: `assets/v711_confusion_sfmf.png` (confusion SF vs MF) + `assets/v711_tsne_sfmf.png`
Takeaway: Anchor **arcface_m04 tereproduksi** (1,14% ≈ 1,32%); Gate v7.1.1 → v7.2.0 **LOLOS**.

---
<!-- [SECTION] -->
## (Slide 33) v7.2.0 — Representation Ablation (headline)
arcface_m04 dikunci; hanya representasi (R1/R2/R3) yang berubah. A100-80GB, 8192 titik, 5 seed.

---
<!-- [TABLE] -->
## (Slide 34) v7.2.0 — Akurasi: grid EER N×M
| N×M | **R1 raw_ply** | **R2 canonical** | **R3 fps** |
|---|---|---|---|
| 1×1 | 0,162 ± 0,022 | 0,000 | 0,000 |
| 3×3 | 0,114 ± 0,015 | 0,000 | 0,000 |
| **5×5** (primer) | **0,099 ± 0,026** | **0,000** | **0,000** |
| 5×10 | 0,078 ± 0,007 | 0,000 | 0,000 |
| 10×10 | 0,076 ± 0,020 | 0,000 | 0,000 |
| **AUC** | **0,894** | **1,000** | **1,000** |
**Catatan render:** kolom R1 `rust`, R2 `azure`, R3 `jade`; highlight baris 5×5.
Takeaway: R2/R3 di lantai 0 di **seluruh** grid; R1 raw runtuh (≈9,9% di N5M5).

---
<!-- [FIGURE] -->
## (Slide 35) v7.2.0 — DET / ROC
Aset: `assets/v720_det_roc.png`
- R2/R3 AUC = **1,000**; R1 raw AUC = **0,894** — terpisah jelas.
Takeaway: Tanpa kanonikalisasi, kurva DET/ROC R1 jauh lebih buruk.

---
<!-- [FINDING] -->
## (Slide 36) v7.2.0 — Temuan #1: Kanonikalisasi adalah penentu
- R1 raw: **EER ~9,9%** (gagal). R2/R3 canonical: **EER ≈ 0%**.
- Jurang **~10×** — efek single-variable terbesar di seluruh seri v7.x.
**Catatan render:** callout border kiri `jade`, angka besar `9,9%` (`rust`) vs `≈0%` (`jade`).
Takeaway: **Kanonikalisasi (PCA-align + unit-sphere) WAJIB** untuk PointNet++ di tugas ini.

---
<!-- [FIGURE] -->
## (Slide 37) v7.2.0 — Robustness rotasi (+ anomali 90°)
Aset: `assets/v720_rotation.png`
| θ° | R1 | R2 | R3 |
|---|---|---|---|
| 0 | 0,068 | 0,000 | 0,000 |
| 60 | 0,201 | 0,000 | 0,000 |
| **90** | 0,426 | **0,393** | **0,428** |
| 180 | 0,428 | 0,000 | 0,000 |
**Catatan render:** sorot kolom 90° dengan `amber`.
Takeaway: R1 runtuh seiring rotasi; R2/R3 invarian **kecuali singularitas 90°** (ambiguitas tukar sumbu PCA) — keterbatasan terdokumentasi.

---
<!-- [TABLE] -->
## (Slide 38) v7.2.0 — Kecepatan & disk
| Metrik | R1 | R2 | R3 |
|---|---|---|---|
| Train wall (s) | 1400,9 | 1409,4 | **1357,5** |
| Latency probe N5M5 (s) | 12,509 | 12,292 | **12,157** |
| Load/frame (ms) | **2,9** | 0,3 | 0,3 |
| **Disk (MB)** | 1829,7 | 914,9 | **399,8** |
Takeaway: Kecepatan end-to-end setara; **pembeda tegas = disk** — R3 **4,6× lebih kecil** dari R1.

---
<!-- [FIGURE] -->
## (Slide 39) v7.2.0 — Sintesis Pareto
Aset: `assets/v720_pareto.png`
- Akurasi R2 = R3; disk & kecepatan: **R3 menang**.
Takeaway: **R3 (fps_npy) = pilihan deployment Pareto-optimal**; R2 cadangan setara-akurasi; R1 ditolak.

---

# BAGIAN 5 — TEMUAN & BUKTI KONSOLIDASI

---
<!-- [SECTION] -->
## (Slide 40) Bagian 5 — Temuan Konsolidasi

---
<!-- [CONTENT] -->
## (Slide 41) Ringkasan temuan kunci
1. **Loss menentukan** (v2→v3): Triplet→ArcFace, EER 28,95% → 0,03% (Rank-1 +40 ppt).
2. **Multi-frame fusion** = tuas terbesar (v7.1.0): SF→MF turun **71%**.
3. **Kanonikalisasi WAJIB** (v7.2.0): R1 9,9% vs R2/R3 ≈0%.
4. **R3 (canonical+FPS)** Pareto-optimal: akurasi = R2, disk 4,6× lebih kecil.
Takeaway: Tiga pengungkit performa: **loss (ArcFace) → fusion → kanonikalisasi**.

---
<!-- [CONTENT] -->
## (Slide 42) Catatan kejujuran (caveat) — wajib di tesis
- **EER = 0% ≠ sempurna** → "di lantai resolusi pengukuran" (test ≈ 22 sesi, kuantisasi ~1/220). R2 vs R3 **tak terbedakan** secara akurasi.
- **Kegagalan R1 mengonflasikan 2 sebab**: tanpa kanonikalisasi **dan** skala mentah ~0,25 vs radius `ball_query` (terkalibrasi untuk unit-sphere).
- **Determinism R2≈R3 inkonklusif** (0,0086 vs 0,0080; didominasi noise cudnn, bukan sampling).
- **Daya statistik rendah** (11 subjek, 5 seed) → laporkan mean±std + arah, hindari klaim signifikansi antar loss top.
**Catatan render:** seluruh slide nuansa `amber`.
Takeaway: Klaim disampaikan **hati-hati & jujur** sesuai batas data.

---

# BAGIAN 6 — REFRAMING (RINGKAS)

---
<!-- [QUOTE] -->
## (Slide 43) Narasi reframing dalam satu layar
> "GeoAtt justru merugikan pada setup ArcFace." — Evaluation Report v2 (17 Mei)
- Hipotesis fusi-geometri gugur + perbandingan tercemar leakage →
- **Pivot ke pure PointNet++** (sederhana, fair, reproducible) →
- + ArcFace + kanonikalisasi + multi-frame fusion → hasil kuat (EER ~1,1% / ≈0%).
Takeaway: Hasil negatif yang dikelola dengan rigor metodologi **mengarahkan** ke kontribusi sebenarnya.

---

# BAGIAN 7 — GAP RISET & KONTRIBUSI

---
<!-- [SECTION] -->
## (Slide 44) Bagian 7 — Gap Riset & Kontribusi

---
<!-- [TABLE] -->
## (Slide 45) Konteks literatur: PointNet++ "lemah" untuk tangan
| Metode | NNHand (Top-1/EER) | HKPolyU v1 (Top-1/EER) |
|---|---|---|
| **PointNet++** (baseline) | 53,42% / 47,19% | 30,40% / 34,28% |
| DGCNN | 76,20% / 21,70% | 84,63% / 19,03% |
| Clustered DGCNN | 98,23% / 14,45% | 99,27% / 7,92% |
Sumber: Svoboda et al. IJCB 2020 (`docs/literature/...`)
Aset: `assets/lit_accuracy_pointnetpp.png`
Takeaway: Literatur menilai **PointNet++ baseline lemah** (30–53%) untuk biometrik tangan — ini latar gap kita.

---
<!-- [GAP] -->
## (Slide 46) Gap yang diisi penelitian ini
*(Berdasarkan literatur review; sepanjang pengetahuan kami)*
1. **PointNet++ + ArcFace (metric learning) untuk identifikasi telapak 3D** — karya PointNet++ terdahulu memakai klasifikasi/softmax & menilai PointNet++ lemah; **belum ada** yang memakai ArcFace. → Kita capai EER **~1,1%** (v7.1.1) hingga **≈0%** (v7.2.0).
2. **PointNet++ + canonical (PCA-align) sebagai ablation untuk palm-ID** — normalisasi sebelumnya ad hoc (OBB untuk pose; centering untuk face-ear); **belum ada** yang mengisolasi efek kanonikalisasi pada akurasi palm-ID. → Kita kuantifikasi (raw gagal, canonical ≈0%).
3. **Canonical + FPS sebagai representasi deployment** — FPS biasanya internal; memakai **canonical+FPS pra-komputasi** sebagai representasi tersimpan & meng-ablasinya (akurasi = canonical, disk 4,6× lebih kecil) adalah **baru**.
**Catatan render:** tiap nomor border kiri `jade`; sub-teks bukti `slate`.
Takeaway: Tiga gap konkret + bukti pendukung dari eksperimen kita.

---
<!-- [FINDING] -->
## (Slide 47) Kontribusi utama
- **Membantah** anggapan "PointNet++ lemah untuk biometrik tangan": dengan **ArcFace + kanonikalisasi + multi-frame fusion**, pure PointNet++ mencapai EER ~1,1% / ≈0% pada 11 subjek.
- **Bukti kuantitatif** bahwa kanonikalisasi adalah faktor dominan (jurang ~10×).
- **Representasi efisien (R3)**: akurasi setara, disk 4,6× lebih kecil → praktis untuk deployment.
- **Hasil negatif GeoAtt** yang terdokumentasi (kapan fusi geometri tidak membantu).
Takeaway: Kontribusi = positif (resep yang bekerja) **dan** negatif (yang tidak), keduanya berbukti.

---

# BAGIAN 8 — KETERBATASAN & FUTURE WORK

---
<!-- [CONTENT] -->
## (Slide 48) Keterbatasan
- **Dataset kecil**: 11 subjek, 5 seed, test ≈ 22 sesi → EER terkuantisasi kasar; EER=0 di lantai pengukuran.
- **Singularitas PCA 90°**: kanonikalisasi tidak invarian-rotasi sempurna (tukar sumbu utama).
- **R1 bukan isolasi murni** "kanonikalisasi saja" (terkonflasi skala vs ball_query).
- **Determinism R2 vs R3** belum terpisah bersih (noise GPU dominan).
- **Closed-set** fokus utama; open-set (LOSO) hanya di v7.1.0.
Takeaway: Temuan kuat secara arah, namun perlu hati-hati pada klaim absolut.

---
<!-- [CONTENT] -->
## (Slide 49) Future work
- **Disambiguasi sumbu** berbasis anatomi (bukan hanya rentang PCA) → hilangkan anomali 90°.
- **Perbesar dataset** (subjek & sesi) untuk daya statistik → uji-p yang valid.
- **Isolasi bersih R1** (skala-normalize raw tanpa PCA) untuk memisahkan efek skala vs kanonikalisasi.
- **Open-set** mendalam (penolakan orang asing) — subcenter/hybrid menjanjikan.
- **Revisit GeoAtt** dengan protokol bebas-bias (bila ingin menutup pertanyaan fusi geometri).
Takeaway: Jalur lanjutan jelas, dibangun di atas temuan yang sudah kokoh.

---

# BAGIAN 9 — PENUTUP & LAMPIRAN

---
<!-- [FINDING] -->
## (Slide 50) Kesimpulan
- Perjalanan: **PoC → krisis metodologi → reframing → resep final**.
- **Resep yang bekerja:** pure PointNet++ + **ArcFace** + **kanonikalisasi** + **multi-frame fusion**.
- **Pilihan representasi:** **R3 (canonical + FPS)** — akurasi ≈ R2, disk 4,6× lebih kecil.
- Kontribusi mengisi 3 gap literatur (ArcFace, canonical, canonical+FPS untuk palm-ID).
Takeaway: Pendekatan **sederhana namun tepat** mengalahkan kompleksitas arsitektur (GeoAtt).

---
<!-- [CONTENT] -->
## (Slide 51) Provenance & reproducibility
- **v7.2.0:** commit `54e74f7e`, GPU **A100-SXM4-80GB**, n_points=8192, batch=192, bf16, seeds [0,42,123,2024,31337].
- 15 checkpoint (`runs/v7_2_0/{C1,C2,C3}/seed_*/best.pth`) + cache eval (anti-ulang).
- Tiap angka traceable ke CSV/LAPORAN (lihat caption per slide).
Takeaway: Hasil dapat direproduksi penuh dari artefak yang tersimpan.

---
<!-- [QUOTE] -->
## (Slide 52) Referensi kunci (literatur)
- Qi et al. (2017) PointNet; PointNet++ — fondasi point cloud DL.
- Svoboda et al. (IJCB 2020) — Clustered DGCNN; PointNet++ baseline lemah untuk hand biometrics.
- Zhang et al. (MDPI 2023) — TMBNet/MVP vs PointNet++ untuk palmprint.
- Micucci & Iula (2023) — fusi palmprint 3D + hand geometry (EER 1,18% → 0,06%).
- Liu et al. (2025) — survey deep learning palmprint recognition.
Sumber: `docs/literature/literatur_review_geoatt_pointnet_palmprint.md`
Takeaway: Posisi kontribusi jelas relatif terhadap state-of-the-art.

---
<!-- [COVER] -->
## (Slide 53) Terima kasih
**3D Palm Recognition — Pure PointNet++ + ArcFace + Canonical (FPS)**
Diskusi & pertanyaan dipersilakan.
**Catatan render:** latar `midnight`, logo/identitas opsional.

<!--
================================================================================
ASET CADANGAN (tersedia di assets/, opsional dipakai agen desain bila perlu
memperkaya/mengganti slide). Bukan file yatim — sengaja disertakan:
- lit_eer_comparison.png      → alternatif Slide 45 (konteks literatur, EER)
- lit_flops_accuracy.png      → trade-off FLOPs vs akurasi (PointNet++ berat)
- lit_multimodal_fusion.png   → Slide 52 (Micucci & Iula: fusi 1,18% → 0,06%)
- v4_boxplots_test.png        → alternatif Slide 23/24 (sebaran 4 varian)
- v5_train_loss.png, v5_val_eer.png, v5_paired_diff.png → detail v5 low-data
- v6_val_eer.png              → kurva val EER v6
- v710_confusion_sfmf.png, v710_tsne_sfmf.png → bukti SF vs MF v7.1.0
- v711_ablation_heatmap.png   → heatmap N×M v7.1.1 (alternatif Slide 32)
================================================================================
-->
