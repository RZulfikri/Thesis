# Notebook & Code Versions

Catatan versi training/evaluation untuk Phase 2 thesis 3DCNN.

---

## v7.0.0-lowdata (CURRENT — Mei 2026) — Multi-Frame Fusion + Loss Sweep + Open-Set Evaluation

**Konsep:** Ekstensi deployment-oriented dari v6.0.0. Fokus pada tiga hipotesis: *(H1) multi-frame fusion meningkatkan EER secara berarti (Cohen's d ≥ 0.5) vs single-frame baseline; (H2) margin/loss-function terbaik dapat ditentukan via grid sweep dengan effect size; (H3) effect size + bootstrap CI lebih informatif daripada p-value pada N=11.*

**Perubahan kunci vs v6.0.0:**

| Aspek | v6.0.0 | v7.0.0 |
|---|---|---|
| Subjek | 10 (gede dropped) | **11** (gede aktif — `DROPPED_SUBJECTS = set()`) |
| Loss variants | Triplet + ArcFace(m=0.5) | **+ CosFace, SubCenter-ArcFace (K=3), ArcFace grid sweep** |
| Protocol | Single-frame (1 median/sesi) | **+ Multi-frame fusion (N enroll × M probe)** |
| Mining | Batch-hard intra-session | **+ Cross-session triplet mining (A&P dari sesi berbeda)** |
| Evaluation | Closed-set EER | **+ Open-set LOSO (unknown subject rejection per fold)** |
| Metrik primer | Wilcoxon p-value | **Cohen's d + bootstrap 95% CI (Wilcoxon informational)** |
| Frame sampling | Median saja | **+ Random sampling (C1), all-frames ablation (D3-D5)** |

**Files (active):**
- `collab/v7_train_eval.ipynb` — train+eval 8 varian × N seed (standard/arcface_m03/m04/m05/s64/cosface/subcenter/hybrid)
- `collab/v7_multiframe_compare.ipynb` — multi-frame ablation (N×M heatmap), fusion strategy, LOSO open-set, Gate-2 effect size

**Backend changes:**

*New files:*
- `losses/cosface.py` — `CosFaceLoss` (margin pada cosine value, bukan angle: cos(θ)−m)
- `losses/subcenter_arcface.py` — `SubCenterArcFaceLoss` (K sub-centers per class, robust intra-class variation)
- `utils/eval_multiframe.py` — `eval_multiframe()`, `eval_multiframe_ablation()`, `fusion_strategy_ablation()` (D1-D5)
- `utils/eval_openset.py` — `run_loso_eval()`, `run_loso_fold()` untuk open-set LOSO A4

*Modified files:*
- `utils/dataset_lowdata.py` — `DROPPED_SUBJECTS = set()` (A1), `build_lowdata_splits_session_dirs()`, `build_lowdata_splits_all_frames()`, random frame sampling support (C1)
- `utils/dataset.py` — `PalmFrameDataset` tracks `session_idx` per sample → batch dict includes `"session_idx"` key (C2)
- `losses/triplet.py` — `CrossSessionTripletLoss` added (C2): positive pair hanya valid jika anchor+positive dari sesi berbeda; fallback ke intra-session jika tidak ada cross-session positive di batch
- `train.py` — CLI flags baru: `--loss cosface|subcenter_arcface`, `--subcenter-k`, `--frame-sampling median|random`, `--cross-session-mining`; `_run_epoch_triplet_v5` mendeteksi `CrossSessionTripletLoss` dan pass `session_ids`

**Decision Gates (v7.0.0):**
- **Gate-1** (protocol readiness): 11 subjek confirmed, splits valid, smoke test 1 seed × 2 varian × 5 epoch tidak crash
- **Gate-2** (deployment verdict): Cohen's d ≥ 0.5 untuk H1 MF fusion, bootstrap 95% CI tidak mencakup 0, latency enroll+probe ≤ 1s
- **Gate-3** (reproducibilitas): 3 seed, std EER < 0.03

**Output dirs:**
- `runs/v7_lowdata/{variant}/seed_{N}/`
- `eval_results/v7_lowdata/{variant}/seed_{N}/`
- `analysis/v7_lowdata_<TS>/`

**Rencana lengkap:** `IMPROVEMENT_PLAN_v7.0.0.md`

**Tag git:** `v6.0.0` (endpoint v6, digunakan sebagai baseline perbandingan)

### Changelog — 28 Mei 2026

**Confusion matrix & t-SNE visualization improvements**

| File | Perubahan |
|------|-----------|
| `collab/v7_multiframe_compare.ipynb` | **Section 8c code cell (NEW)**: confusion matrix nearest-neighbor single-frame vs multi-frame (N=5) untuk 4 varian (`standard`, `arcface_m05`, `cosface`, `subcenter`). Menggunakan `extract_fused_embeddings()` dari section 8b, cosine similarity NN matching, dan `ConfusionMatrixDisplay`. Output: `confusion_matrix_sf_vs_mf.png` |
| `collab/v7_train_eval.ipynb` | **Bug fix section 10 (t-SNE)**: import `build_lowdata_splits_with_paths` → `build_lowdata_splits`. Fungsi `_with_paths` mengembalikan `{split: [(label, frame_dir), ...]}` (list of tuples), tetapi kode mengiterasi `.items()` yang membutuhkan `{split: {label: [frame_dirs]}}` (dict). Tanpa fix ini, cell akan crash saat dieksekusi |
| `evaluate.py` | **Fix t-SNE labels**: placeholder `np.zeros(...)` → identity labels asli dari `dir_to_emb` path (`d.parent.parent.name`), sehingga plot t-SNE menampilkan warna per subjek. **Tambah confusion matrix PNG**: N×N identification confusion matrix dari `probe_results` disimpan otomatis sebagai `{model}_confusion_matrix.png` dengan Rank-1 accuracy di title |

**Status visualisasi per notebook:**

| Notebook | t-SNE | Confusion Matrix |
|----------|-------|-----------------|
| `v7_train_eval.ipynb` | §10 — semua 8 varian (seed=42) ✅ | §11 — semua 8 varian, NN test→train (seed=42) ✅ |
| `v7_multiframe_compare.ipynb` | §8b — 4 varian, SF vs MF side-by-side ✅ | §8c — 4 varian, SF vs MF side-by-side ✅ (NEW) |
| `evaluate.py` (CLI) | Auto-generate per model (identity-colored) ✅ | Auto-generate per model (N×N identification) ✅ (NEW) |

---

## v6.0.0-lowdata (ARCHIVED — Mei 2026) — Loss-Function Comparison (Triplet vs ArcFace)

**Konsep:** Pivot framing dari "GeoAtt-centric" ke **Loss-Function-centric**. Drop GeoAtt sepenuhnya, fokus pada hipotesis: *"ArcFace loss memberikan performa superior dibandingkan Triplet batch-hard loss pada 3D palm identification berbasis PointNet++ dalam regime enrollment terbatas (1 sampel per sesi)."* Backbone, preprocessing, augmentasi, split, training budget, model selection — semuanya identik antar varian. **Hanya loss yang berbeda.** Rencana lengkap: [`IMPROVEMENT_PLAN_v0.6.0.md`](../IMPROVEMENT_PLAN_v0.6.0.md).

**Files (active):**
- `collab/v6_standard_train_eval.ipynb` — train+eval standard (PointNet++ + Triplet, margin=0.3)
- `collab/v6_arcface_train_eval.ipynb` — train+eval proposed (PointNet++ + ArcFace, m=0.5 s=30)
- `collab/v6_standard_arcface_compare.ipynb` — agregasi 2 varian × 10 seed, paired Wilcoxon + bootstrap CI, Gate 2 verdict
- `train.py` (root) — **v6.0.0 patch**: ArcFace di-wire ke low-data pipeline (lihat backend changes)
- `models/siamese.py` — `num_classes` di-pass dari train.py saat `--loss arcface` (head `ArcMarginProduct` aktif)

**Backend changes (`train.py`):**
- CLI flag baru `--arcface-margin` (default 0.5) dan `--arcface-scale` (default 30.0) untuk F2.6 hyperparameter search
- `is_arcface = loss_type == "arcface"`; `is_perframe = is_triplet or is_arcface` — ArcFace di low-data sekarang share dataset path (`PalmFrameDataset` + `preload_augment` + `repeat`) yang sama dengan Triplet
- `criterion = ArcFaceLoss(num_classes=len(train_frames), margin, scale, embedding_dim=128)` saat `--loss arcface`
- `SiamesePalmNet(..., num_classes=n_subjects if is_arcface else 0, arc_margin=..., arc_scale=...)` — head ArcFace hanya dibuat ketika diperlukan, sehingga checkpoint Triplet tetap kompatibel (load `strict=False`)
- Function baru `_run_epoch_arcface_v5()` — bentuk return identik dengan `_run_epoch_triplet_v5()` (keys `total_loss`, `triplet_loss` alias, `aux_loss=0`, `aux_acc`=classification accuracy) → CSV/TB schema seragam, `val_pair_metric` tetap pakai `model.encode()` (apple-to-apple model selection)
- Closure dispatcher `_run_perframe_epoch()` dipakai di semua 4 call site (P1 train/val, P2 train/val)

**Carry-over fixes dari v5.0.0 (semua dipertahankan):**
- `--frames-per-session 1` — 1 median frame per sesi
- `--val-metric pair_eer` — smoothed EER window=5 untuk model selection
- `--no-early-stop --epochs 120 --finetune_epochs 30` — fixed budget identik antar varian
- Chronological split 8/2/2/3, 10 subjek (gede dropped), deterministic
- Depth-focused augmentation
- 10 seeds: 42, 123, 2024, 7, 31337, 0, 1, 2, 3, 4

**Yang di-DROP dari v5.0.0:**
- `--use-geom`, `--use-gam`, `--use-geom-fusion` — tidak di-pass; default `no_geom` (PointNet++ murni)
- `--use-aux-loss` — tidak dipakai; v6 ingin fair comparison loss-only tanpa kontaminasi aux signal
- Modul `models/gam.py`, `models/geometry_encoder.py` tetap di repo (dormant, tidak di-import oleh path v6)
- F2.0 plan menyarankan strip total, tapi pendekatan dormant lebih aman (tidak break backward-compat dengan checkpoint v4/v5)

**Comparison matrix (apple-to-apple):**

| Komponen | Standard (baseline) | ArcFace (proposed) |
|---|---|---|
| Backbone | PointNet++ (no_geom) | PointNet++ (no_geom) — identik |
| Preprocessing / FPS / aug | Identik | Identik |
| Dataset / Split / Seed | Identik | Identik |
| Training budget | 120 + 30 epoch fixed | Identik |
| Optimizer / LR / Batch | Identik | Identik |
| Embedding dim | 128 (L2-normalized) | 128 (L2-normalized) |
| Model selection | Val pair EER smoothed | Identik |
| **Loss function** | **Triplet batch-hard, m=0.3** | **ArcFace, m=0.5 s=30** |
| **Head tambahan** | — | `ArcMarginProduct(128 → 10)` |
| Matching (eval) | Cosine similarity | Cosine similarity |

**Decision Gates (v6.0.0):**
- **Gate 0**: smoke test 1 seed × 2 varian × 5 epoch — pipeline tidak crash, val EER trajectory turun
- **Gate 1**: trajectory 1 seed × 2 varian full (≈30 min/run di A100) — kedua varian plateau <25% di epoch 50
- **Gate 2 PRIMARY**: paired Wilcoxon Test EER (n=10) — verdict hipotesis (🎉 terkonfirmasi / 🟡 neutral / 🔴 ditolak)
- **Gate 2 SECONDARY**: Holdout EER, Rank-1, bootstrap CI Δ EER

**Estimasi waktu:**
- Per run: ~25-30 menit di A100 (sama dengan v5.0.0)
- Full run 10 seed × 2 varian = ~10 jam
- + Eval: ~30 menit
- + Compare notebook: ~5 menit

**Tag git:** `v0.6.0` (planned), `v0.6.0-final` setelah Gate 2 pass

**Catatan masukan untuk eksekusi (review v6 implementation):**
1. **Hyperparameter search F2.6 belum di-script.** Plan v0.6.0 §F2.6 menyarankan grid 4 kombinasi (m,s) × 1 seed sebelum full run. Sekarang notebook ArcFace langsung pakai `m=0.5, s=30`. Kalau ingin grid search, duplikat notebook arcface dengan loop `for (m,s) in [(0.3,30),(0.5,30),(0.5,64),(0.7,30)]` pada 1 seed; bandingkan val pair EER terbaik sebelum lanjut full 10 seed.
2. **Sanity bisa lebih ketat.** `evaluate.py` menggunakan `load_state_dict(strict=False)` — bobot `arcface.*` di checkpoint akan di-skip silently saat eval (eval pakai `model.encode()`, tidak butuh head ArcFace). Aman, tapi disarankan tambahkan assertion di evaluate.py kalau checkpoint mengandung `arcface.*` keys: print "ArcFace head di-skip (eval pakai encoder embedding)" supaya transparan.
3. **Init parity bonus.** Karena backbone identik dan kedua varian dijalankan dengan `--seed` sama, init weights backbone identik per-seed — keuntungan fair comparison yang tidak perlu patch khusus (otomatis tercapai). Plan §"Mengapa 2 varian" baris pertama: ✅ confirmed by design.
4. **Risiko utama yang masih hidup**: ArcFace dengan batch besar (BS ≥ 256) dan hanya 10 kelas bisa "kelas habis" dalam 1 batch → gradient skewed. Plan tidak menyentuh ini. Mitigasi: kalau probe VRAM menghasilkan BS > 256 untuk ArcFace, pertimbangkan kunci BS = 128 (komentar di notebook ArcFace di cell ke-3). Belum di-implement; flag sebagai *known unknown*.
5. **`models/siamese.py` punya 2 `ArcMarginProduct`** (di model `self.arcface` + di criterion `ArcFaceLoss.arc_margin`) saat run. `_run_epoch_arcface_v5` pakai `model.forward_arcface()` → hanya `model.arcface` yang aktif; head di criterion adalah dead weight (~5 KB). Tidak salah, hanya redundant. Cleanup minor: ganti criterion ke `nn.CrossEntropyLoss()` murni di train.py — TODO opsional.

---

## v5.0.0-lowdata (Mei 2026)

**Konsep:** Pivot framing dari "ablation 4-arah" ke **low-data regime study**. Fokus pengujian hipotesis: *"GeoAtt menyediakan inductive bias yang menguntungkan dalam regime enrollment terbatas (1 sampel per sesi)."* Berbasis temuan v4.0.0 yang menunjukkan 3 bias eksperimental (split bocor temporal, val_loss anti-korelasi dengan generalization, training budget tidak seragam).

**Files (active):**
- `collab/v5_lowdata_train_eval.ipynb` — training + evaluation low-data regime
- `collab/v5_lowdata_compare.ipynb` — analisis statistik + Wilcoxon paired test (Gate 2 verdict)
- `train.py` (root) — v5.0.0 backend
- `evaluate.py` (root) — v5.0.0 backend
- `models/encoder.py` — `geom_dim=13` default
- `models/gam.py` — **bug fix v5.0.0**: residual + bidirectional tanh gating (α ∈ [-0.5, 0.5], identity-safe)
- `models/geometry_encoder.py` — LayerNorm di akhir (stabilisasi geom_emb)
- `models/siamese.py` — auxiliary classifier head (`use_aux_loss=True`, `n_subjects=10`)
- `utils/dataset.py` — import dari `geometry_schema`
- `utils/normalizer.py` — `GEOMETRY_DIM = 13`
- `utils/augmentation.py` — depth-focused (rotation ±45°, tilt ±20°, XY+Z translation ±3cm, scale 0.95–1.05)

**Files (new utilities):**
- `utils/geometry_schema.py` — pure-numpy schema 13-dim (bisa di-import tanpa torch)
- `utils/dataset_lowdata.py` — `OneFramePerSession` + chronological split (8/2/2/3)
- `utils/val_pair_metric.py` — val pair EER metric (untuk model selection)
- `utils/sanity_geom_only_cv.py` — Gate 0: geom-only LogReg CV sebelum deep learning
- `utils/audit_temporal_gap.py` — audit time-gap antar split (dokumentasi limitation)
- `utils/audit_geom_discriminability.py` — B/W ratio per feature

**Backend dependencies:**
- `--frames-per-session 1` — sampling 1 median frame per sesi via MAD-based picker
- `--loss triplet --triplet-margin 0.3` — Triplet batch-hard mining (cocok untuk ~14 sampel/class)
- `--val-metric pair_eer` — val EER per epoch (smoothed window=5), bukan val_loss
- `--no-early-stop` — fixed budget 120 + 30 epoch
- `--use-aux-loss` — auxiliary CE classifier head di geom_emb (weight 0.3)
- `--use-geom` — shortcut untuk `--use-gam --use-geom-fusion` (varian with_geom)

**Perubahan kunci dari v4.0.0:**

| Aspek | v4.0.0 | v5.0.0-lowdata |
|---|---|---|
| **Framing hipotesis** | "GeoAtt vs PointNet++ secara umum" | "GeoAtt menang di low-data regime?" |
| **Varian** | 4 (no_geom, with_geom, gam_only, fuse_only) | **2** (no_geom, with_geom) |
| **Dataset** | All-frame ~1869 sampel | **1 median frame/sesi** = 150 sampel |
| **Subjek** | 11 | **10** (gede dropped, <15 sesi valid) |
| **Split** | Random 70/15/15 + holdout random | **Deterministic chronological 8/2/2/3** |
| **Geom features** | 14 (incl. mean_palm_curvature, thumb_width) | **13** (drop curvature B/W=0.76 + thumb_width B/W=1.38, add scan_distance) |
| **Geom ratios** | — | **Tidak ditambah** (redundant dengan absolute di low-data) |
| **Loss** | ArcFace | **Triplet batch-hard** |
| **Model selection** | val_loss | **val pair EER (smoothed)** |
| **Training budget** | Early stop on val_loss (bervariasi 20–55 epoch) | **Fixed 120+30 epoch, no early stop** |
| **GAM architecture** | Sigmoid gating only (suppress saja) | **Residual + tanh gating (identity-safe + bidirectional)** |
| **Auxiliary loss** | — | **CE classifier di geom_emb (forcing function)** |
| **Augmentation** | Generic | **Depth-focused** (rotation/tilt naik, +Z translation, no lighting) |
| **Seed count** | 5 | **10** (compensate low-data variance) |

**Decision Gates eksplisit:**
- **Gate 0** (sanity baseline): geom-only LogReg accuracy ≥30% untuk lanjut. **Hasil pre-flight: 98.7% (PASSED)**.
- **Gate 1** (smoke test): val EER trajectory turun monoton, plateau <30% epoch 50.
- **Gate 2** (full run): paired Wilcoxon Test EER → verdict hipotesis (confirmed/neutral/problematic).
- **Gate 3** (kalau F2.10 trigger): GAM fix mengecilkan gap?
- **Gate 4** (replikasi all-frame): plot gap-vs-size sebagai central finding.

**Conditional fallbacks:**
- **F2.10 GAM architecture fix** (sudah included: residual + tanh) — kalau Gate 2 problematic, eskalasi ke FiLM modulation.
- **F2.11 Auxiliary loss** — sudah default ON.

**Estimasi waktu:**
- Per run: ~25 menit di A100
- Full run 10 seed × 2 varian = ~10 jam
- + Eval (40 calls): ~30 menit
- + Compare notebook: ~5 menit

**Rencana lengkap:** `IMPROVEMENT_PLAN_v5.0.0.md`

**Laporan diagnostik v4.0.0 → v5.0.0:** `result_docs/20260522_092309/KESIMPULAN_REPORT.md`

### Changelog — 24 Mei 2026

**Notebook workflow: GitHub clone + auto-push results**

Kedua notebook diubah dari workflow "upload manual ke Drive" ke **GitHub-based workflow**:
- **Code**: `git clone` dari GitHub saat setup, selalu dapat versi terbaru
- **Dataset**: symlink dari Google Drive (tetap di Drive, terlalu besar untuk Git)
- **Output** (runs, eval_results, analysis): di dalam repo, di-commit + push ke branch `colab`
- **Di lokal**: `git fetch origin && git checkout colab` untuk akses semua hasil

Requires: `GITHUB_TOKEN` di Colab Secrets (sidebar 🔑 → Add Secret).

**Perubahan pada `v5_lowdata_train_eval.ipynb` (26 → 36 cells):**

| Perubahan | Detail |
|-----------|--------|
| Setup cell | Clone dari GitHub, checkout branch `colab`, symlink dataset dari Drive |
| BATCH_SIZE comment | Tambah penjelasan kenapa BS=32 (80 train frames, auto-config A100 terlalu besar) |
| Runtime Shutdown Guard | `atexit` + `shutdown_colab()` — auto-shutdown jika crash |
| Git Save Helper | `git_save(message, push)` — commit + push, auto-skip file >95MB |
| Incremental git push | Push ke GitHub setelah tiap varian selesai (Pro+ safety) |
| Git Push Results | Final push sebelum shutdown |
| Auto-Shutdown | Countdown 60 detik + `runtime.unassign()`, bisa cancel |

**Perubahan pada `v5_lowdata_compare.ipynb` (23 → 33 cells):**

| Perubahan | Detail |
|-----------|--------|
| Setup cell | Clone dari GitHub, checkout branch `colab` |
| Runtime Shutdown Guard | `atexit` + `shutdown_colab()` |
| Git Save Helper | Same as train_eval |
| None-safe formatting | `_fmt()` helper untuk `wilcoxon_p` yang bisa `None` |
| Training Loss Trajectory | Plot baru (section 9b): triplet+aux loss per epoch |
| Git Push Results | Final push sebelum shutdown |
| Auto-Shutdown | Countdown 60 detik |

**Bug fixes:**
- Fix `git_save()`: `find` command crash kalau `runs/` belum ada → cek `isdir()` + `try/except`
- Fix cell [31] train_eval: source ter-split per-karakter (1145 entries × 1 char) → rejoin ke 33 baris
- Fix `:.4f` format spec pada `wilcoxon_p = None` → `TypeError` → gunakan `_fmt()` helper

### Changelog — 24 Mei 2026 (lanjutan)

**GPU/RAM Auto-tuning + dataset preload**

Notebook `v5_lowdata_train_eval.ipynb` cell-7 (Configuration) ditambah **auto-tune VRAM tier** (target ~90% GPU utilization). Strategi untuk low-data 80 train frames: maksimalkan compute per-iteration daripada per-epoch.

**Tier config:**

| Tier | VRAM | BS | N_points | AMP | Preload | Repeat | Aug variants/epoch |
|------|------|----|----|----|----|----|----|
| T3 | ≥90 GB | 80 | 16384 | bf16 | ✅ | 20 | 1600 |
| T2 | ≥80 GB | 80 | 16384 | bf16 | ✅ | 16 | 1280 |
| T1 | ≥40 GB | 64 | 12288 | bf16 | ✅ | 12 | 960 |
| T0 | <40 GB | 32 | 8192 | fp16 | ❌ | 5 | 400 |

**Perubahan train.py:**
- CLI flag baru `--preload-augment`: aktifkan `PalmFrameDataset.preload_augment=True` (pre-generate semua augmented variants ke RAM saat init, hapus CPU bottleneck per batch)
- CLI flag baru `--repeat N`: berapa kali tiap frame muncul per epoch (override default 1 untuk low-data, 10 untuk all-frame)
- `train_fixed_split()` membaca kedua flag, pass ke `PalmFrameDataset`

**Perubahan notebook:**
- Cell-7 (Configuration): auto-detect VRAM via `nvidia-smi` → set BS, N_POINTS, AMP_MODE, PRELOAD_AUGMENT, REPEAT, NUM_WORKERS per tier
- Cell-15 (run_training): pass auto-tune params via `--amp`, `--preload-augment`, `--repeat`, `--num_workers`, `--siamese-mode concat`; print GPU memory pre/post tiap run
- Cell-5 (TensorBoard logdir): `/content/drive/.../runs/v5_lowdata` → `/content/Thesis/3DCNN/runs/v5_lowdata` (in-repo path, sesuai GitHub workflow)
- Cell-0 (header): tambah tabel tier + strategi optimasi

**Expected speedup:**
- GPU utilization: ~30-40% → **~85-95%**
- Wall-clock per run di A100 80GB: ~25 min → **~10-15 min** (lebih banyak compute per iteration tapi efisien)
- RAM cache: ~500-1000 MB (trivial, Colab Pro 50+ GB)

---

## v5.0.1-lowdata (Mei 2026) — OOM Fix + Efficiency Optimizations

**Trigger:** Training v5.0.0 mengalami CUDA OOM saat dynamic VRAM probe merekomendasikan BS=1079, N=16384 di A100 40GB. OOM terjadi di `ball_query` karena alokasi distance matrix contiguous ~17 GB.

### Changelog — 24 Mei 2026

**1. Memory-safe `ball_query` (`models/pointnet_utils.py`)**
- Implementasi chunked: proses centroid per chunk=256, bukan seluruh S sekaligus
- Peak memory: O(B × S × N) → O(B × chunk × N)
- Untuk B=1024, S=512, N=16384: 17 GB → ~4 GB

**2. Safety clamp di `train.py`**
- `_clamp_args_to_safe_limits()`: enforce hardware-safe caps pada batch_size dan n_points
- Hard cap: N_POINTS > 8192 → BS ≤ 192 (regardless of GPU class)
- Auto-config cache untuk hindari double-print

**3. Batched `ValPairMetric.compute()` (`utils/val_pair_metric.py`)**
- Sebelumnya: 110 pasangan di-encode satu per satu → ~15s/epoch
- Sekarang: 20 unique frames di-encode sekaligus dalam 1 batch, pair similarity dari embedding cache → **~1-2s/epoch**

**4. `--val_freq` CLI flag (`train.py`)**
- Default 1 (setiap epoch). Naikkan ke 3-5 untuk skip validasi di epoch non-target
- Mengurangi overhead validation saat epoch sangat pendek (low-data regime)

**5. Smoothed EER model selection (`train.py`)**
- Model selection sekarang pakai `val_pair_metric.smoothed_eer(window=5)` kalau history cukup
- Fallback ke raw EER untuk epoch 1-4 (belum cukup history)
- Mengurangi "lucky draw" dari noise sampling point cloud random

**6. Notebook probe lebih konservatif (`v5_lowdata_train_eval.ipynb`)**
- `TARGET_VRAM_FRACTION`: 0.90 → 0.75 (account untuk fragmentation + real-loop overhead)
- Safety margin: 0.95× → 0.90×
- `MAX_BS_FOR_LARGE_N = 192` hard cap untuk N_POINTS > 8192
- `compute_repeat` min_steps: 2 → 4 (lebih banyak batch per epoch)

---

## v4.0.0 (Mei 2026) — 4-Variant Ablation

**Konsep:** Ablation 4 varian dengan fair init parity (v0.4.0 baseline backend). Hasil teridentifikasi punya **3 bias eksperimental** yang menggugurkan verdict (lihat `KESIMPULAN_REPORT.md`).

**Files (renamed, archived):**
- `collab/v4_train_and_eval.ipynb` — (sebelumnya `01_train_and_eval.ipynb`)
- `collab/v4_compare_analyze.ipynb` — (sebelumnya `02_compare_analyze.ipynb`)
- `runs_v4/` — (sebelumnya `runs/`)
- `eval_results_v4/` — (sebelumnya `eval_results/`)

**Hasil (untuk perbandingan vs v5.0.0):**
| Variant | Test EER | Holdout EER | AUC test |
|---|---|---|---|
| no_geom | 0.17% | 0.00% (leak-perfect) | 0.9995 |
| fuse_only | 13.92% | 5.45% | 0.907 |
| with_geom | 20.16% | 11.21% | 0.857 |
| gam_only | 26.98% | 21.82% | 0.799 |

**Tag git:** `v4.0.0`

**Status:** Archived. Hasil ditangguhkan karena bias setup. Replaced oleh v5.0.0-lowdata.

---

## v0.4.0-baseline (~3 jam/seed)

**Konsep:** Versi **fair ablation v0.4.0 sebelum speed optimasi**. Sudah termasuk perbaikan metodologi (fair init, proper holdout, QC v3, 4-variant CLI) tetapi **tanpa** speed optimasi (topk, concat, fused, compile).

**Files:**
- `collab/legacy/01_train_and_eval_v0.4.0_baseline.ipynb` — training + evaluation baseline (moved to legacy)
- `history/v0.4.0_baseline/train.py` — snapshot baseline
- `history/v0.4.0_baseline/models/siamese.py` — Siamese split (2 call terpisah)
- `history/v0.4.0_baseline/models/pointnet_utils.py` — ball_query argsort (original)

**Cara pakai:**
1. Buka `collab/legacy/01_train_and_eval_v0.4.0_baseline.ipynb` di Colab.
2. Cell Setup otomatis mengarahkan `sys.path` ke `history/v0.4.0_baseline` agar import model mengambil snapshot baseline.
3. `run_training()` memanggil `history/v0.4.0_baseline/train.py`.
4. Auto-tune cell import `models.siamese` juga dari snapshot baseline.

**Backend state:**
- `train.py` — Adam vanilla (no fused), no `--compile`, no `--siamese-mode`
- `models/siamese.py` — 2 panggilan encoder terpisah, BN per-branch (B sampel)
- `models/pointnet_utils.py` — `dist.argsort()[:, :k]` (O(N log N) sort)
- `evaluate.py` — shared dengan optimize (backward-compatible)

---

## v0.4.0-optimize (~40-60 menit/seed)

**Konsep:** Versi **v0.4.0 + speed optimasi penuh**. Estimasi speedup gabungan ~3-5× per epoch. Backend ini menjadi fondasi untuk v4.0.0 dan v5.0.0.

**Files:**
- `collab/v4_train_and_eval.ipynb` — Phase 2 v4.0.0 (renamed dari `01_train_and_eval.ipynb`)
- `collab/v4_compare_analyze.ipynb` — analisis v4.0.0 (renamed dari `02_compare_analyze.ipynb`)
- `collab/legacy/01_train_and_eval_v0.4.0_optimize.ipynb` — snapshot terkunci (legacy)
- `collab/legacy/02_compare_analyze_v0.4.0_optimize.ipynb` — snapshot analisis (legacy)
- `train.py` (root) — optimize backend (v5.0.0 enhanced)
- `models/siamese.py` (root) — optimize backend (v5.0.0 enhanced)
- `models/pointnet_utils.py` (root) — optimize backend (unchanged di v5.0.0)

**Backend dependencies:**
- `train.py` — Adam `fused=True`, `--amp bf16`, `--siamese-mode concat`
- `models/siamese.py` — Siamese concat-then-split (BN over 2B sampel)
- `models/pointnet_utils.py` — `ball_query` pakai `torch.topk` (O(N+k log k))
- `evaluate.py` — `--holdout`, `--save_scores`, backward-compat checkpoint loading

**Optimasi stack:**

| # | Opsi | Speedup | Risiko | Status |
|---|------|---------|--------|--------|
| 1 | `ball_query`: `argsort` → `topk` | 2-3× | None — math identik | ✅ |
| 2 | Siamese concat (1 forward call, bukan 2) | 15-25% | Low — BN over 2B (estimator lebih stabil) | ✅ |
| 3 | Adam `fused=True` kernel | 5-10% | None — drop-in | ✅ |
| 4 | bf16 mixed precision (`--amp bf16`) | 1.5-2× | Low — <0.1% fp noise | ✅ |
| 5 | DataLoader `persistent_workers=True` + `prefetch_factor=4` | 5-10% | Low | ✅ (sudah di baseline) |

> **Catatan:** `torch.compile` dievaluasi dan **dihapus** dari stack akhir karena PointNet++ sampling layers menghasilkan dynamic shapes yang memicu recompilation loop, justru memperlambat training dan menurunkan reproducibilitas.

**Total speedup**: ~3-5× per epoch vs v0.4.0-baseline.

**Implikasi metodologi:**
- Keempat variant menggunakan code path yang **sama**, jadi **fair ablation tetap valid**.
- Hipotesis `with_geom` vs `no_geom` tidak terpengaruh — perbedaan hanya di jalur numerik training, bukan struktur eksperimen.
- `topk`, `fused Adam`, dan `bf16` adalah math-identik atau negligible noise.
- `Siamese concat` mengubah BN running statistics (estimator lebih stabil), tapi ini **standar** untuk Siamese contrastive (SimCLR, MoCo).

---

## Perbandingan Baseline vs Optimize

| Aspek | v0.4.0-baseline | v0.4.0-optimize |
|---|---|---|
| ball_query indexing | `dist.argsort()[:, :k]` | `torch.topk(largest=False)` |
| Siamese forward | 2 panggilan terpisah | concat → 1 panggilan |
| Optimizer | `Adam(params, lr)` | `Adam(params, lr, fused=True)` |
| Precision | fp32 | bf16 mixed (`--amp bf16`) |
| Graph compile | None | None (dihapus — dynamic shapes di PointNet++ tidak kompatibel) |
| Per-seed (estimasi) | ~3 jam | ~40-60 menit |
| Phase B full (20 run) | ~60 jam | ~15-20 jam |

---

## v0.3.0-baseline (Mei 2026)

**Files** (`collab/legacy/`):
- `train_v030.ipynb`
- `evaluate_v030.ipynb`
- `compare_v030.ipynb`

**Catatan:** v0.3.0 adalah baseline historis sebelum perbaikan metodologi v0.4.0 (belum ada fair init, holdout eval, QC v3). Hasil training v0.3.0 dilaporkan di `IMPROVEMENT_PLAN_v0.4.0.md`. Phase 1 diagnostik menemukan RNG initialization bias.

---

## Reproduce Checklist

| Versi yang ingin di-run | Notebook | Backend | Output dir |
|---|---|---|---|
| **v7.0.0-lowdata (current)** | `collab/v7_train_eval.ipynb` + `v7_multiframe_compare.ipynb` | root `train.py` (v7 patch), `losses/{cosface,subcenter_arcface,triplet}.py`, `utils/{eval_multiframe,eval_openset,dataset_lowdata,dataset}.py` | `runs/v7_lowdata/`, `eval_results/v7_lowdata/` |
| v6.0.0-lowdata (archived) | `collab/v6_standard_train_eval.ipynb` + `v6_arcface_train_eval.ipynb` + `v6_standard_arcface_compare.ipynb` | root `train.py` (v6 patch) | `runs/v6_lowdata/`, `eval_results/v6_lowdata/` |
| v5.0.1-lowdata (archived) | `collab/v5_lowdata_train_eval.ipynb` + `v5_lowdata_compare.ipynb` | root `train.py`, `models/`, `utils/` (v5.0.1) | `runs/v5_lowdata/`, `eval_results/v5_lowdata/` |
| v4.0.0 (archived) | `collab/v4_train_and_eval.ipynb` + `v4_compare_analyze.ipynb` | root (sebelum v5 enhancements) | `runs_v4/`, `eval_results_v4/` |
| v0.4.0-baseline | `collab/legacy/01_train_and_eval_v0.4.0_baseline.ipynb` | `history/v0.4.0_baseline/` | — |
| v0.4.0-optimize | `collab/v4_train_and_eval.ipynb` (alias) | root `train.py`, `models/` | — |
| v0.3.0-historical | `collab/legacy/train_v030.ipynb` | git checkout v0.3.0-era commit | — |

---

## Folder Convention (post-v5.0.0)

```
3DCNN/
├── collab/
│   ├── v7_train_eval.ipynb                  ← v7.0.0 CURRENT (train+eval 8 varian)
│   ├── v7_multiframe_compare.ipynb          ← v7.0.0 CURRENT (MF ablation + LOSO + Gate-2)
│   ├── v6_standard_train_eval.ipynb         ← v6.0.0 archived (Triplet baseline)
│   ├── v6_arcface_train_eval.ipynb          ← v6.0.0 archived (ArcFace proposed)
│   ├── v6_standard_arcface_compare.ipynb    ← v6.0.0 archived (compare)
│   ├── v5_lowdata_train_eval.ipynb          ← v5.0.1 archived
│   ├── v5_lowdata_compare.ipynb             ← v5.0.1 archived
│   ├── v4_train_and_eval.ipynb              ← v4.0.0 archived
│   ├── v4_compare_analyze.ipynb             ← v4.0.0 archived
│   ├── legacy/                              ← v0.3.0 & v0.4.0 snapshots
│   └── VERSIONS.md                          ← this file
├── losses/
│   ├── cosface.py                           ← v7.0.0 new
│   ├── subcenter_arcface.py                 ← v7.0.0 new
│   └── triplet.py                           ← v7.0.0 modified (CrossSessionTripletLoss)
├── utils/
│   ├── eval_multiframe.py                   ← v7.0.0 new
│   ├── eval_openset.py                      ← v7.0.0 new
│   ├── dataset_lowdata.py                   ← v7.0.0 modified (11 subj, session_dirs, all_frames)
│   └── dataset.py                           ← v7.0.0 modified (session_idx tracking)
├── runs_v4/                                  ← v4.0.0 checkpoints (archived)
├── runs/v5_lowdata/                          ← v5.0.0 checkpoints (archived)
├── runs/v6_lowdata/{standard,arcface}/       ← v6.0.0 checkpoints (archived)
├── runs/v7_lowdata/{variant}/seed_{N}/       ← v7.0.0 checkpoints (current)
├── eval_results_v4/                          ← v4.0.0 eval (archived)
├── eval_results/v5_lowdata/                  ← v5.0.0 eval (archived)
├── eval_results/v6_lowdata/{standard,arcface}/ ← v6.0.0 eval (archived)
├── eval_results/v7_lowdata/{variant}/seed_{N}/ ← v7.0.0 eval (current)
└── analysis/v7_lowdata_<TS>/                 ← v7.0.0 analysis outputs (per run)
```

---

## Roadmap

- **v7.1.0** (current): multi-frame fusion (N×M ablation), loss sweep (Triplet/ArcFace grid/CosFace/SubCenter), cross-session triplet mining, open-set LOSO; Gate-2 verdict via Cohen's d + bootstrap CI
- **v7.2.0** (pending, setelah v7.1.0 Gate-2): representation ablation — Raw PLY vs Canonical NPY vs Pre-FPS NPY × loss function; dekomposisi kontribusi preprocessing vs loss
- **v7.x-final**: tag setelah semua gate lulus — klaim novelty thesis: (1) deployment-realistic multi-frame protocol, (2) preprocessing pipeline ablation, (3) ArcFace pada consumer depth sensor
- **v6.0.0** (archived, tag `v6.0.0`): baseline endpoint — 2 varian × 10 seed, Wilcoxon Test EER; ArcFace sedikit lebih baik dari Triplet (effect size kecil)
- **v5.0.1-lowdata** (archived): GeoAtt hypothesis low-data regime
- **v4.0.0** (archived): 4-variant ablation dengan 3 bias eksperimental teridentifikasi
