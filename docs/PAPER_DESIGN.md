# PAPER_DESIGN — v8 Faktorial (Alignment × ArcFace) untuk submission IEEE

Dokumen acuan tetap saat menulis paper. Mengunci klaim, baseline, protokol, metrik, dan
pemetaan eksperimen → figur. Implementasi: `3DCNN/collab/v8_lib.py` (source of truth) +
shell `v8_train_seed{S}.ipynb` (×5) + `v8_eval.ipynb`.

## 1. Klaim (hipotesis)
- **H1 — Normalisasi → robustness.** Representasi point cloud yang ter-normalisasi/kanonik membuat
  pengenalan **tahan terhadap variasi pose (rotasi)**; tanpa normalisasi (raw) akurasi runtuh saat
  scan dirotasi.
- **H2 — ArcFace → accuracy.** Angular-margin (ArcFace) meningkatkan akurasi pengenalan dibanding
  softmax (tanpa margin).

## 2. Prinsip anti-sirkular (kenapa faktorial)
**Softmax = basis netral (tak perlu dijustifikasi).** **ArcFace + nilai margin = yang DIUJI**, jadi
tidak boleh dikunci sebagai asumsi. Karena itu kita **tidak** mengunci ArcFace lalu "membuktikan"-nya;
kita ukur **semua kombinasi sekaligus** (desain faktorial). Setiap klaim lalu terbukti *lintas* faktor
lain → bukan kebetulan.

## 3. Desain faktorial
Grid **{softmax, arcface} × {A0..A5}** = 12 config × 5 seed = **60 run**.

| Sumbu | Nilai |
|---|---|
| Representasi (alignment) | A0 raw_ply (**baseline**), A1 center, A2 center+scale, A3 PCA canonical, A4 PCA-robust, A5 anatomical |
| Loss | **softmax** (=`arcface_true` m=0, cosine-softmax; basis terkontrol), **arcface** (=`arcface_true` m=0.5, nilai standar Deng et al. 2019) |

- **H1** dibaca **antar-baris** (alignment) — dan harus berlaku di **kedua kolom** loss → robustness murni efek representasi.
- **H2** dibaca **antar-kolom** (softmax vs arcface) — dan harus berlaku di **semua** alignment.
- **Interaksi** (bonus): apakah ArcFace menolong lebih banyak saat ter-align.

`softmax` = `arcface_true` m=0 menjaga **arsitektur head identik** dengan arcface (hanya margin yang
beda) → perbandingan H2 terkontrol penuh.

## 4. Baseline
- Lower-bound = sudut grid **`(A0 raw, softmax)`**.
- Metode penuh = **`(A*, arcface)`** dengan A* = alignment terbaik (kolom arcface, EER terendah; dipilih
  & dicetak otomatis oleh `v8_eval`).
- Eksternal (framing): PointNet++ palm terdahulu (Svoboda IJCB'20 ~30–53% acc; Zhang'23 EER 34–47%) →
  kontras dengan hasil kita.

## 5. Protokol evaluasi
- **Closed-set**: gallery = sesi **train** (template enrol), probe = sesi **HOLDOUT** (33 sesi, pristine,
  tak pernah dipakai training/early-stop). **TANPA LOSO.**
- Semua 11 identitas ada di gallery & probe.
- **Multi-frame fusion N×M** (grid N,M ∈ {1,3,5,10}), titik headline **N=5, M=5**, strategi mean.
- **5 seed** → laporkan **mean ± std**.
- Split per-subjek (deterministik kronologis): train 88 / val 22 (early-stop) / test 22 / **holdout 33** sesi.

## 6. Metrik
| Tujuan | Metrik | Figur/tabel |
|---|---|---|
| Verifikasi (1:1) | **EER** (primer), AUC, d′ | Tabel 6×2, `det_roc.png` |
| Identifikasi (1:N) | **rank-1 acc, CMC, confusion matrix** | `cmc.png`, `confusion_matrix.png` |
| **Robustness (H1)** | EER vs rotasi 0–180° (Δ EER) | `rotation_sensitivity.png` (overlay softmax vs arcface) |
| Kualitatif | t-SNE separabilitas | `tsne.png` |
| Efisiensi | latency, VRAM, disk | `speed_resource.csv` |

## 7. Asumsi scope (robustness = rotasi saja)
Pipeline punya **QC-gate** (`extract_geometry`/`check_scan`/`min_points`/deteksi knuckle) yang menjamin
hanya scan **lengkap & bersih** (jari + knuckle utuh) yang masuk training/eval. Maka nuisance
operasional yang tersisa di deployment = **pose tangan = rotasi**. Jitter/noise/point-dropout
**di luar scope by design** (sudah tersaring QC), bukan kelalaian.

## 8. Pemetaan eksperimen → figur paper
1. **Tabel/heatmap EER 6×2** (`factorial_eer_6x2.csv`, `factorial_eer_heatmap.png`) — centerpiece: H1 (baris) + H2 (kolom) + interaksi + baseline `(A0,softmax)` + A*.
2. **`rotation_sensitivity.png`** — H1: alignment ter-normalisasi datar (kedua loss), A0 raw naik.
3. **`det_roc.png`, `cmc.png`, `confusion_matrix.png`, `tsne.png`** — H2 (4 config kunci: `(A0,sm)`,`(A0,arc)`,`(A*,sm)`,`(A*,arc)`).
4. **`SUMMARY.md`** — ringkasan otomatis (tabel + klaim).

## 9. Reproducibility & eksekusi
- Dataset diregenerasi dari `Raw Depth Data/` (raw committed) via `generate_dataset.py`; cache di Drive.
- Checkpoint → Google Drive (Opsi B); resume granular per **unit (cfg_id, seed)** (skip bila `perf.json` ada).
- **Paralel**: 5 file `v8_train_seed{S}.ipynb`, 1 per runtime/GPU → ~2 jam; atau 1 runtime semua seed (~10 jam GPU cepat).
- Evidence (CSV/figur) → `3DCNN/analysis/v8_factorial_<TS>/` (ter-commit ke git).

## 10. Estimasi compute
- Training: 60 run (~10 jam GPU cepat / ~20–30 jam L4; ~2 jam bila 5 runtime paralel).
- Eval+analisa: ringan (forward-pass + cache), ~menit s/d ~1 jam; bisa GPU murah.

## Catatan (di luar scope sekarang)
- **Track C / QA-ArcFace** (margin adaptif kualitas) **di-drop** — H1/H2 sudah tuntas oleh grid faktorial.
  Kode loss (QA-ArcFace/AdaCos/CurricularFace/CosFace/SubCenter) tetap di `3DCNN/losses/margin_heads.py`
  bila kelak dibutuhkan novelty kedua.
- Margin ArcFace = m=0.5 standar (Deng et al.) + sitasi; margin sweep opsional bila reviewer minta.
