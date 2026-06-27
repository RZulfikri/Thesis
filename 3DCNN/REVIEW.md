# Peer-Review Report — v0.4.0 Fase 1

> **Reviewer:** Peer-Review Agent  
> **Date:** 2026-05-17  
> **Scope:** F1.1 (RNG parity), F1.2 (4-variant CLI), F1.3 (backward-compat loader), QC v3 scanner + script  
> **Referensi:** `AGENTS.md` (3DCNN/), `IMPROVEMENT_PLAN_v0.4.0.md`

---

## Executive Summary

Fase 1 menyelesaikan pemecahan flag geometri dan backward-compat loader dengan benar, **namun terdapat dua gap kritis yang harus diperbaiki sebelum Fase 2 training dimulai**:

1. **`train.py` tidak memiliki kontrol seed** — fair ablation 4 varian × 5 seed tidak dapat direproduksi.
2. **`evaluate.py` tidak mendukung flag `--use-gam` / `--use-geom-fusion`** — evaluasi varian `gam_only` dan `fuse_only` tidak bisa dilakukan dari CLI.

---

## 1. `models/encoder.py` — RNG Parity Fix (F1.1)

### Findings

| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 1.1 | **WARNING** | 58 | `geom_dim` default `33` tidak cocok dengan `GEOMETRY_DIM = 14` di `dataset.py` | Jika encoder di-instantiate langsung (misal di notebook Colab) tanpa passing `geom_dim`, akan terjadi shape mismatch saat `geom` masuk ke `GeometryEncoder`. |
| 1.2 | **WARNING** | 103–106 | Komentar tidak sinkron dengan implementasi | Komentar menyebutkan "adapter linear (256→320) yang berperan sebagai pass-through deterministik nol-padding", namun kode membangun dua `nn.Sequential` terpisah (`proj_with_geom` dan `proj_no_geom`). Fungsionalitas benar, tetapi komentar menyesatkan. |
| 1.3 | **INFO** | 78–123 | RNG parity design sound | Semua sub-modul dibangun dengan urutan tetap; hanya forward path yang bercabang. Ini memenuhi requirement Plan §D1. |
| 1.4 | **INFO** | 126–128 | Property `proj` backward-compat | Alias untuk introspection/hooks. Baik. |

### Verdict: **REQUEST_CHANGES**

**Action required:**
- Sinkronkan default `geom_dim` dengan `GEOMETRY_DIM` (14) atau hapus default agar wajib explicit.
- Perbarui komentar di lines 103–106 agar mencerminkan implementasi aktual (dua head terpisah).

---

## 2. `train.py` — 4-Variant CLI Flags (F1.2)

### Findings

| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 2.1 | **CRITICAL** | 182–245, seluruh file | **Tidak ada argumen `--seed` dan tidak ada pemanggilan `torch.manual_seed` / `np.random.seed` / `random.seed`** | Fase 2 mensyaratkan training dengan 5 seed berbeda (`[7, 42, 123, 2026, 31337]`). Tanpa seed control, inisialisasi bobot dan sampling point cloud tidak reproducible. Ini membatalkan validitas ablation. |
| 2.2 | **CRITICAL** | 335–415 | `_run_epoch_arcface`, `_run_epoch_hybrid`, `_run_epoch_triplet` adalah *dead code* | Fungsi-fungsi ini tidak pernah dipanggil di `train_one_fold` maupun `train_fixed_split`. Training hanya menggunakan `ContrastiveLoss`. Jika Fase 2 direncanakan menggunakan ArcFace/Triplet/Hybrid (lihat `AGENTS.md`: "Loss: ArcFace + Triplet hybrid"), wiring belum ada. |
| 2.3 | **WARNING** | 130–149 | `_auto_config()` gagal di macOS local dev | Akses `/proc/meminfo` dan `nvidia-smi` gagal di macOS. Walaupun di-catch, fallback tidak optimal untuk local dev. Bukan blocker untuk Colab. |
| 2.4 | **WARNING** | 606–772 | `train_fixed_split` tidak menyimpan metadata konfigurasi varian | `args` (termasuk `use_gam`, `use_geom_fusion`, `split_seed`, dll) tidak diserialisasi ke file JSON/YAML. Tracking eksperimen Fase 2 akan sulit. |
| 2.5 | **INFO** | 486–492, 658–664 | Flag resolution `(_use_gam or _use_fuse)` untuk `use_geom` di `SiamesePalmNet` | Konsisten dengan desain encoder. OK. |

### Verdict: **REQUEST_CHANGES**

**Action required:**
1. Tambahkan `--seed` ke `parse_args()` dan panggil `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)` di awal `train_one_fold` / `train_fixed_split`. Jika CUDA: `torch.cuda.manual_seed_all(seed)` dan `torch.backends.cudnn.deterministic = True` (dengan catatan performance trade-off).
2. Jika Fase 2 memang menggunakan ContrastiveLoss saja, hapus dead code atau tambahkan `--loss_type` CLI flag untuk mengaktifkannya. Jika Fase 2 planned hybrid, wiring harus diselesaikan sekarang.
3. (Recommended) Simpan `config.json` di `output_dir` berisi semua `args` untuk reproducibility.

---

## 3. `evaluate.py` — Backward-Compat Loader (F1.3)

### Findings

| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 3.1 | **CRITICAL** | 104–138 | **Tidak ada flag `--use-gam` dan `--use-geom-fusion` di CLI** | `parse_args()` hanya memiliki `--use_geom` / `--no_geom`. `main()` memanggil `load_model(..., use_geom=args.use_geom)` tanpa passing `use_gam` / `use_geom_fusion`. Akibatnya, **tidak mungkin mengevaluasi varian `gam_only` (M2) dan `fuse_only` (M3) dari command line**. |
| 3.2 | **WARNING** | 179 | `model.load_state_dict(state, strict=False)` | Diperlukan untuk backward-compat rename `proj` → `proj_with_geom`/`proj_no_geom`, tetapi `strict=False` bisa menyembunyikan mismatch bobot yang tidak diinginkan. Risk acceptable untuk migration, tetapi sebaiknya ditambahkan warning log jika ada missing/unexpected keys. |
| 3.3 | **INFO** | 226–262 | `run_inference_cached` mengakses `dataset._cache` | Akses atribut privat (`_cache`) dari luar kelas. Bukan bug fungsional, tetapi melangkap enkapsulasi. |
| 3.4 | **INFO** | 165–178 | Logic rename legacy `encoder.proj.*` → `proj_with_geom` / `proj_no_geom` | Benar dan cukup untuk checkpoint pre-v0.3.0. |

### Verdict: **REQUEST_CHANGES**

**Action required:**
1. Tambahkan `--use-gam` dan `--use-geom-fusion` ke `parse_args()`, lalu pass ke `load_model()` di `main()`.
2. (Recommended) Log warning jika `load_state_dict` menghasilkan missing/unexpected keys.

---

## 4. `utils/dataset.py` — QC v3 Scanner Update

### Findings

| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 4.1 | **WARNING** | 90–99, 607–608 | `_sample_points` menggunakan `np.random.choice` tanpa seeded RNG | Sampling titik dari point cloud (`random` method) bergantung pada state global NumPy. Tanpa seeding eksplisit di `train.py`, sampling antar-run tidak deterministic. Ini menambah noise variance di Fase 2. |
| 4.2 | **WARNING** | 476–480 | `split_sessions_three_way` bisa menghasilkan `n_train = 0` untuk dataset kecil | `n_test = max(1, round(...))` dan `n_val = max(1, round(...))`. Jika `n = 2` (ekstrem), `n_train = 0`. Untuk dataset thesis (~14 sesi/subjek setelah holdout), risiko rendah, tetapi edge case tetap ada. |
| 4.3 | **INFO** | 328–355 | `_frame_is_valid` whitelist approach | Menyaring `scan_distance_out_of_range` dan `knuckle_fallback` sebagai false positive. Konsisten dengan kualitas data aktual. |
| 4.4 | **INFO** | 358–406 | `scan_dataset_frames` dengan `filter_invalid=True` | Implementasi v0.3.0 fix (per-frame, bukan per-sesi). Benar. |

### Verdict: **NEEDS_DISCUSSION**

**Action required:**
1. Diskusikan apakah `_sample_points` perlu menerima `rng: np.random.Generator` parameter untuk sampling deterministik. Jika ya, refactor `PalmPairDataset` dan `PalmFrameDataset` untuk menerima seed.
2. Diskusikan apakah perlu guard `assert n_train > 0` di `split_sessions_three_way`.

---

## 5. `utils/data_qc_v3_frame.py` — QC v3 Script

### Findings

| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 5.1 | **WARNING** | 28–37 | `flatten_geometry` menggunakan SEMUA key numerik dari `geometry.json` | Sementara `dataset.py` hanya menggunakan `GEOMETRY_KEYS` (6 key, 14 dim). QC membuat keputusan berdasarkan fitur seperti `scan_distance_mm` dan `inter_finger_gaps_mm` yang sengaja **dikecualikan** dari training. Ini inkonsisten: frame bisa di-reject karena outlier pada fitur yang model tidak pernah lihat. |
| 5.2 | **INFO** | 96–98, 108–112 | `np.median` pada array dengan `NaN` mengembalikan `NaN` | Fitur yang hilang di `geometry.json` diisi `np.nan`, lalu `mad == 0` check tidak menangkap `NaN` (karena `NaN != 0`). Missing features di-silent-ignore. Bukan bug, tetapi perilaku tersembunyi. |
| 5.3 | **INFO** | 120 | `session_bad` menggunakan `ratio > threshold` (strict) | Konsisten dengan docstring, tetapi jika `ratio == 0.5` tepat, sesi tidak di-exclude. Mungkin tidak masalah untuk threshold 0.5. |
| 5.4 | **INFO** | 131–217 | Dry-run default aman (`--apply` diperlukan) | Good practice. Renaming reversible. |

### Verdict: **NEEDS_DISCUSSION**

**Action required:**
1. Diskusikan apakah `flatten_geometry` di QC harus menggunakan `GEOMETRY_KEYS` yang sama dengan training, ataukah menggunakan superset memang disengaja (karena QC ingin memanfaatkan semua informasi kualitas). Jika disengaja, dokumentasikan di komentar.

---

## Risks for Fase 2 Training

| Risk ID | Severity | Description | Mitigasi |
|---------|----------|-------------|----------|
| R1 | **CRITICAL** | **Non-reproducible initialization & sampling**: `train.py` tidak set seed. 5-seed ablation tidak valid secara statistik karena variance antar-run mengandung noise dari RNG global. | Tambahkan `--seed` dan seeding routine sebelum training loop. |
| R2 | **CRITICAL** | **Evaluation pipeline incomplete**: `evaluate.py` tidak bisa load `gam_only` / `fuse_only` dari CLI. Fase 2 akan menghasilkan 20 checkpoint (4 varian × 5 seed) yang tidak bisa dieval otomatis untuk 2 varian. | Patch `evaluate.py` CLI flags sebelum training. |
| R3 | **WARNING** | **Dead loss implementations**: ArcFace/Triplet/Hybrid runners ada tetapi tidak terhubung. Jika Fase 2 tiba-tiba butuh hybrid loss, wiring baru perlu dikerjakan di tengah training run. | Tentukan loss type untuk Fase 2 sekarang dan wire ke `train.py`. |
| R4 | **WARNING** | **Point sampling noise**: `np.random.choice` tanpa seed di `_sample_points` menambah variance antar-epoch dan antar-run. | Refactor `Dataset` untuk menerima seeded RNG. |
| R5 | **WARNING** | **QC-training feature mismatch**: QC mengecualikan frame berdasarkan fitur yang tidak masuk ke model. Potensi data loss tanpa justification dari sudut pandang model. | Sinkronkan `flatten_geometry` dengan `GEOMETRY_KEYS`, atau dokumentasikan rationale. |
| R6 | **INFO** | **No config serialization**: Metadata eksperimen tidak tersimpan. Reproducibility dan debugging sulit untuk 20 run. | Simpan `config.json` per run. |

---

## Overall Verdict

| File | Verdict | Blocker Fase 2? |
|------|---------|-----------------|
| `models/encoder.py` | REQUEST_CHANGES | No (default mismatch hanya saat instantiate manual) |
| `train.py` | **REQUEST_CHANGES** | **YES** — seed control & loss wiring |
| `evaluate.py` | **REQUEST_CHANGES** | **YES** — 4-variant CLI flags |
| `utils/dataset.py` | NEEDS_DISCUSSION | No |
| `utils/data_qc_v3_frame.py` | NEEDS_DISCUSSION | No |

**Rekomendasi gatekeep Fase 2:**
- [ ] Fix `--seed` + seeding routine di `train.py`
- [ ] Fix `--use-gam` / `--use-geom-fusion` di `evaluate.py`
- [ ] (Optional tapi strongly recommended) Simpan `config.json` per run
- [ ] (Optional) Tentukan dan wire loss type untuk Fase 2

Setelah dua item CRITICAL di atas diperbaiki, Fase 2 dapat di-*approve* untuk dimulai.

---

*Review completed by Peer-Review Agent — 2026-05-17*
