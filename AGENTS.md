# Multi-Agent Swarm — Thesis Project (Global)

> **Kontrak Agent Global** — Dokumen ini berlaku untuk **seluruh repository** (`3DCNN/`, `3DRegistration/`, `TrueDepthScan/`). Setiap sub-project bisa punya override spesifik di `SUBPROJECT/AGENTS.md` (lebih spesifik menggantikan yang general).
>
> **Proyek dalam repo ini:**
> - `3DCNN/` — GeoAtt-PointNet++ Palm Recognition (deep learning, PyTorch)
> - `3DRegistration/` — ICP Point Cloud Registration (C++/Python)
> - `TrueDepthScan/` — iOS TrueDepth Scanner (Swift, data acquisition)

> **Kontrak Agent** — Dokumen ini mendefinisikan 5 role agent untuk project ini. Setiap agent membaca file ini sebelum bertugas untuk memahami scope, communication pattern, dan deliverables.

---

## Project Overview

Project ini adalah sistem **3D Palm Recognition** end-to-end untuk thesis Master. Sistem terdiri dari 3 komponen utama yang bekerja secara berurutan:

```
iPhone TrueDepthScan (Swift/iOS)
    ↓ export: depth[NN].bin + calibration.json + metadata.json
3DRegistration (Python/C++)
    ↓ per-frame processing (tanpa ICP)
3DCNN (PyTorch)
    ↓ training & evaluation
GeoAtt-PointNet++ palm recognition model
```

### 1. `TrueDepthScan/` — Data Acquisition
Aplikasi iOS yang menangkap frame depth dari kamera TrueDepth (Face ID), melakukan hand ROI detection menggunakan Vision framework, dan mengekspor raw depth frames beserta kalibrasi kamera. Setiap sesi scan menghasilkan folder dengan format:
```
[label]_YYYYMMDD_HHMMSS/
├── depth00.bin       # Float32 array (width × height), 0.0 = invalid
├── depth01.bin
├── calibration.json  # fx, fy, cx, cy, lens distortion lookup tables
└── metadata.json     # frame count, label, handedness, timestamp
```

### 2. `3DRegistration/` — Point Cloud Preprocessing
Pipeline Python yang memproses raw depth frames menjadi point cloud yang siap untuk deep learning. Pipeline utama saat ini adalah **single-frame** (tanpa ICP registration):
- `lib/single_frame.py` — memuat satu frame depth, voxel downsampling, outlier removal, DBSCAN isolation
- `extract_geometry.py` — mengekstrak 14 fitur biometrik (mm) + QC
- `preprocess_for_cnn.py` — PCA-align + normalize to unit sphere → `cnn_input.npy`
- `dataset.py` — stratified train/val/test split + StandardScaler fit on train only

Output disimpan di `result_frames/[label]/[timestamp]/frame_NN/`.

### 3. `3DCNN/` — Deep Learning Recognition
Model **GeoAtt-PointNet++** (PyTorch) untuk palm recognition dengan geometric attention. Fitur utama:
- Pure PyTorch implementation (no custom CUDA extensions)
- Two-phase training: Phase 1 (main, lr=1e-3) + Phase 2 (fine-tune, lr=1e-4)
- Loss options: OnlineTripletLoss (default), ArcFace, Hybrid
- Fair ablation dengan RNG init parity (58/58 layer identical)
- Multi-seed statistical validation (5 seeds)
- Evaluation metrics: EER, AUC, TAR@FAR=1%, TAR@FAR=0.1%, d-prime, CMC

---

## Technology Stack

| Subproject | Language | Framework/Library | Platform |
|-----------|----------|-------------------|----------|
| `3DCNN` | Python 3.12+ | PyTorch 2.6.0, NumPy, scikit-learn, SciPy, matplotlib, seaborn, pandas, tensorboard, tqdm | Google Colab (A100/H100) for training; macOS for dev |
| `3DRegistration` | Python 3.12+ / C++14 | Open3D, NumPy, SciPy, OpenCV, PyTorch, scikit-learn, pandas; Eigen3, Ceres Solver, pybind11 | macOS / Linux |
| `TrueDepthScan` | Swift 5.0, Obj-C, Obj-C++, Metal | SwiftUI, AVFoundation, Vision, MetalKit, Accelerate, CoreVideo | iOS 17.6+ (iPhone dengan TrueDepth) |

---

## Project Structure & Code Organization

### Root Directory
```
Thesis/
├── AGENTS.md                  # File ini
├── README.md                  # Project overview (human-readable)
├── REPORT.MD                  # Living document — hasil eksperimen
├── SWARM_LOG.md               # Activity log swarm (append-only)
├── PROGRESS_REPORT.md         # Laporan progress thesis
├── LAPORAN_PROGRESS_TESIS.md  # Laporan progress (Indonesian)
├── IMPROVEMENT_PLAN_v0.3.0.md
├── IMPROVEMENT_PLAN_v0.4.0.md # Plan aktif
├── MONITORING_CHECKLIST_v*.md # Checklist monitoring per versi
├── docs/images/               # Figures untuk laporan
├── 3DCNN/                     # Subproject: deep learning
├── 3DRegistration/            # Subproject: registration & preprocessing
├── TrueDepthScan/             # Subproject: iOS scanner
└── Raw Depth Data/            # Raw zip files dari iPhone
```

### `3DCNN/` Structure
```
3DCNN/
├── train.py                   # CLI training script (~1,098 baris)
├── evaluate.py                # CLI evaluation script (~547 baris)
├── requirements.txt           # Dependencies
├── preflight_check.py         # Pre-flight checklist sebelum upload Colab
├── clean_dataset_qc.py        # Quarantine bad sessions
├── models/                    # Neural architectures
│   ├── encoder.py             # GeoAtt-PointNet++ encoder
│   ├── gam.py                 # Geometric Attention Module
│   ├── geometry_encoder.py    # MLP untuk 14-dim geometry features
│   ├── pointnet_utils.py      # FPS, BallQuery, SetAbstraction (pure PyTorch)
│   └── siamese.py             # Siamese wrapper + ArcFace head
├── losses/                    # Loss functions
│   ├── triplet.py             # OnlineTripletLoss + batch-hard mining
│   ├── arcface.py             # ArcFace margin + hybrid loss
│   └── contrastive.py         # Legacy contrastive loss
├── utils/                     # Data & evaluation infrastructure
│   ├── dataset.py             # Dataset scanner, splitters, loaders
│   ├── augmentation.py        # Point cloud & geometry augmentation
│   ├── metrics.py             # EER, AUC, TAR@FAR, CMC, t-SNE, statistical tests
│   ├── normalizer.py          # Z-score normalizer (fit-on-train-only)
│   ├── enrollment.py          # Gallery enrollment strategies
│   ├── data_qc_v3_frame.py    # Frame-level MAD outlier exclusion
│   ├── audit_init_parity.py   # RNG init parity audit
│   ├── audit_embedding_stats.py
│   ├── audit_geom_session_variance.py
│   └── compare_utils.py       # Statistical aggregation across variants/seeds
├── collab/                    # Google Colab notebooks
│   ├── 01_train_and_eval.ipynb
│   ├── 02_compare_analyze.ipynb
│   └── VERSIONS.md            # Notebook/code version tracking
├── local/                     # Local dev notebooks
├── dataset/                   # Raw data (11 subjects, frame-level layout)
├── eval_results/              # Evaluation outputs (plots, reports)
├── runs/                      # Training checkpoints & logs
├── result_docs/               # Formal evaluation reports (markdown)
└── history/                   # Baseline code snapshots
```

### `3DRegistration/` Structure
```
3DRegistration/
├── run.py                     # CLI entry point untuk ICP registration
├── process_single_frames.py   # CURRENT MAIN PIPELINE — per-frame processing
├── process_all_scans.py       # Legacy ICP multi-frame batch pipeline
├── extract_geometry.py        # Ekstrak 14 fitur biometrik + QC
├── preprocess_for_cnn.py      # PCA-align + unit-sphere → cnn_input.npy
├── dataset.py                 # Stratified train/val/test split untuk 3DCNN
├── validate_dataset.py        # QC: PASS/WARN/FAIL per frame/session
├── reextract_all_geometry.py  # Batch re-extractor dengan parallel workers
├── requirements.txt           # Python dependencies
├── run_palm.sh                # Shell wrapper untuk palm-optimized params
├── run_with_venv.sh           # Venv activation wrapper
├── lib/                       # Core Python library modules
│   ├── image_depth.py         # ImageDepth class: load depth, undistort, project 3D
│   ├── process3d.py           # ICP & vision-based registration, DBSCAN, meshing
│   ├── single_frame.py        # Single-frame PLY builder (no registration)
│   └── pose_graph_shim.py     # Pure-Python pose-graph optimizer (fallback)
├── cpp/                       # C++ acceleration module
│   ├── CMakeLists.txt         # CMake config (C++14, Eigen3, Ceres, pybind11)
│   └── pose_graph.cpp         # Ceres-based pose graph optimizer
├── result/                    # Legacy output dir untuk ICP-merged clouds
└── result_frames/             # CURRENT OUTPUT DIR untuk single-frame pipeline
```

### `TrueDepthScan/` Structure
```
TrueDepthScan/
├── TrueDepthStreamer.xcodeproj/   # Xcode project (no SPM/CocoaPods)
├── TrueDepthStreamer/             # Main source directory
│   ├── TrueDepthStreamerApp.swift # SwiftUI @main entry point
│   ├── ContentView.swift          # Main SwiftUI view
│   ├── CameraManager.swift        # Core: AVCaptureSession, scanning, export (~1,169 baris)
│   ├── Open3DExporter.swift       # Export depth frames ke .bin + JSON
│   ├── 3DProcessingService.swift  # Bridge: PLY export, outlier removal
│   ├── ProcessingQueue.swift      # Background OperationQueue untuk post-scan export
│   ├── ScanHistoryManager.swift   # Manajemen saved scans di Documents
│   ├── ScanHistoryView.swift      # SwiftUI sheet untuk scan history
│   ├── ApiService.swift           # HTTP client untuk cloud upload
│   ├── VideoMixer.swift           # Metal-based video + JET depth mixer
│   ├── DepthToJETConverter.swift  # Metal compute: depth → JET colormap
│   ├── PreviewMetalView.swift     # MTKView subclass untuk 2D preview
│   ├── PointCloud/                # Objective-C++ point cloud renderer
│   ├── Shaders/                   # Metal shader sources
│   ├── Info.plist                 # iOS app configuration
│   └── TrueDepthStreamer-Bridging-Header.h
├── Configuration/
│   └── SampleCode.xcconfig        # Bundle ID disambiguator
├── STREAMTRUEDEPTH.md             # Dokumentasi detail (Indonesian)
├── MY_APP_OUTPUT.ply              # Sample PLY output
└── .trae/documents/               # Planning docs dari Trae IDE
```

---

## Build and Test Commands

### `3DCNN/`

**Install dependencies:**
```bash
cd 3DCNN
pip install -r requirements.txt
```

**Training (local — smoke test only):**
```bash
python train.py --data_dir dataset --output_dir runs/exp1 --use-geom --fixed_split
```

**Training (Colab — full training):**
- Buka `collab/01_train_and_eval.ipynb` di Google Colab
- Mount Google Drive
- Upload kode dari local → Drive → Colab runtime
- Output tersimpan di `runs/<variant>/<timestamp>/`

**Evaluation:**
```bash
python evaluate.py --data_dir dataset --checkpoints M4=runs/m4/fold_0/best.pth
```

**Pre-flight check (wajib sebelum upload ke Colab):**
```bash
python preflight_check.py
```

**Audit RNG init parity (wajib setelah ubah encoder):**
```bash
python utils/audit_init_parity.py
```

**Audit embedding stats:**
```bash
python utils/audit_embedding_stats.py
```

**Smoke test (1 seed, 3 epochs):**
```bash
python train.py --data_dir dataset --output_dir runs/smoke_test --epochs 3 --seed 7
```

### `3DRegistration/`

**Python environment setup:**
```bash
cd 3DRegistration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**C++ module build (optional — ada fallback Python):**
```bash
cd cpp
# Pastikan Eigen3 dan Ceres sudah terinstall (via Homebrew: brew install eigen ceres-solver)
git submodule update --init --recursive  # untuk pybind11
mkdir build && cd build
cmake ..
make
# Compiled .so akan di-install ke parent dir (3DRegistration/)
```

**Run main pipeline (single-frame, current):**
```bash
python process_single_frames.py --dataset_dir dataset --output_dir result_frames
```

**Validate dataset quality:**
```bash
python validate_dataset.py --input_dir result_frames
```

**Build dataset splits untuk 3DCNN:**
```bash
python dataset.py --input_dir result_frames --output_dir ../3DCNN/dataset
```

**Shell wrappers:**
```bash
./run_palm.sh <session_folder>        # Palm-optimized ICP params
./run_with_venv.sh <args>             # Run dengan venv auto-activated
```

### `TrueDepthScan/`

**Build:**
- Buka `TrueDepthStreamer.xcodeproj` di Xcode
- Build target `TrueDepthStreamer` (requires physical iPhone dengan TrueDepth camera)
- Tidak ada dependency manager (SPM/CocoaPods) — pure Xcode build

**Run:**
- Deploy ke iPhone dengan TrueDepth camera
- Aplikasi tidak bisa di-run di Simulator (memerlukan hardware depth camera)

---

## Code Style Guidelines

### Python (`3DCNN/` dan `3DRegistration/`)
- **Indentasi:** 4 spaces (no tabs)
- **Docstrings:** Gunakan triple-quote `"""` di awal module/function dengan penjelasan dalam Bahasa Indonesia + English technical terms
- **Imports:** Group dalam urutan: stdlib → third-party → local. Gunakan `sys.path.insert(0, str(Path(__file__).parent))` untuk local imports
- **Naming:**
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Type hints:** Tidak digunakan secara konsisten; jangan ubah style existing untuk menambah type hints massal
- **Comments:** Bahasa Indonesia untuk penjelasan logika, English untuk terminology teknis
- **Baris panjang:** ~100-120 chars (tidak strict)
- **No `black`/`flake8` config** — tidak ada formatter otomatis

### Swift (`TrueDepthScan/`)
- **Indentasi:** 4 spaces
- **Naming:** Swift standard — `PascalCase` types, `camelCase` functions/variables
- **Access control:** Explicit (`private`, `fileprivate`, `internal`) untuk property dan method
- **Concurrency:** `@MainActor` untuk UI-bound observable objects; `OperationQueue` untuk background tasks
- **Metal shaders:** `.metal` files menggunakan Metal Shading Language conventions
- **Bridging:** Objective-C++ files (`.mm`) untuk interop dengan Swift

### C++ (`3DRegistration/cpp/`)
- **Standard:** C++14
- **Indentasi:** 4 spaces
- **Naming:** `snake_case` untuk functions/variables, `PascalCase` untuk classes
- **Memory:** Gunakan `pybind11::array_t<double>` untuk NumPy interop; tidak ada raw pointers

---

## Testing Instructions

### `3DCNN/`
**Tidak ada formal unit test suite** (no `pytest`, `unittest`, `tox`). Testing dilakukan via:

1. **Pre-flight Checklist** (`preflight_check.py`):
   - Verifikasi QC v3 sudah diterapkan
   - Verifikasi key files exist
   - Verifikasi frame count ≥ 1,800
   - Verifikasi notebooks mengandung flags/seeds yang benar

2. **Audit / Diagnostic Scripts:**
   - `utils/audit_init_parity.py` — Verifikasi `use_geom=True` vs `False` consume RNG identically (58/58 layers must match)
   - `utils/audit_embedding_stats.py` — Cek train-mode vs eval-mode embedding consistency
   - `utils/audit_geom_session_variance.py` — Analisis geometry feature variance per session

3. **Smoke Tests:**
   - 1 seed, 3 epochs quick run untuk verifikasi loss decreases dan files tercreate
   - Command: `python train.py --data_dir dataset --output_dir runs/smoke --epochs 3 --seed 7`

4. **Statistical Verification** (di notebook `02_compare_analyze.ipynb`):
   - Paired t-test / Wilcoxon signed-rank pada 5-seed results
   - McNemar test pada per-probe correctness
   - Bootstrap CI untuk metric deltas
   - Test set fingerprint hash untuk verifikasi identical evaluation data

### `3DRegistration/`
**Tidak ada formal unit test suite.** Testing dilakukan via data-driven QC:

1. **`validate_dataset.py`** — Post-hoc QC pada `geometry.json`, assign `PASS` / `WARN` / `FAIL`
2. **`extract_geometry.py`** — Embeds sanity checks (knuckle Y validation, fingertip fallback counting, scan distance range, inter-finger gap thresholds)
3. **`process_single_frames.py`** — Filters frames dengan `< min_points` (default 1000) setelah DBSCAN isolation
4. **Visual verification** — `view_results.py` untuk inspect PLY dengan Open3D viewer

### `TrueDepthScan/`
**Tidak ada formal tests** (no XCTest targets). Testing manual:
1. Build dan run di physical iPhone dengan TrueDepth camera
2. Visual confirmation JET colormap overlay
3. Export verification — inspect `.bin`, `.json`, `.ply` files
4. Referensi fungsional: `STREAMTRUEDEPTH.md`

---

## Security Considerations

- **`TrueDepthScan/Info.plist`** mengandung hardcoded `APP_API_KEY` — jangan expose ke public repo
- **`ApiService.swift`** menggunakan HTTP client untuk upload ke cloud function (`POST /run_registration`) — verify base URL dan API key tidak leaked
- **No secrets management** — tidak ada `.env` files atau secret vault
- **Dataset (`3DCNN/dataset/` dan `3DRegistration/dataset/`)** mengandung biometric data (palm scans) — tidak di-track di Git (gitignored), jangan upload ke repository publik
- **`.gitignore`** sudah mengexclude dataset dirs, build artifacts, dan compiled binaries

---

## Agent Communication Model

**Agent tidak bisa chat langsung antar-agent.** Semua komunikasi terjadi via **shared artifacts** (file di repo).

```
┌─────────────┐    write     ┌─────────────┐
│   Agent A   │ ────────────→│  ARTIFACT   │
└─────────────┘              └──────┬──────┘
                                    │ read
                                    ↓
┌─────────────┐    write     ┌─────────────┐
│   Agent B   │ ←────────────│  ARTIFACT   │
└─────────────┘              └─────────────┘
```

**Shared Artifacts:**
| File | Purpose | Updated by |
|------|---------|------------|
| `AGENTS.md` | This file — role definitions | Root agent (me) |
| `AGENTS.md` (root) | Role definitions global | Root agent |
| `AGENTS.md` (subproject) | Override spesifik subproject | Root agent |
| `REPORT.MD` / `README.md` | Source of truth — results, decisions | Documentation Agent |
| `SWARM_LOG.md` | Activity log — who did what when | All agents (append-only) |
| `DECISION_QUEUE.md` | Pending decisions needing user input | Planning Agent |
| `REVIEW.md` | Code review findings | Peer-Review Agent |
| `PLAN_*.md` | Execution plans per milestone | Planning Agent |
| `ANALYSIS_*.md` | Deep-dive analysis reports | Analysis Agent |
| `CHANGES.md` | Summary of code changes per PR | Code Agent |

---

## Role Definitions

### 0. 🎯 Lead Agent (Orchestrator / Decision Gate)

**Scope:** Review dan pertimbangan setiap keputusan sebelum dieksekusi. Memastikan konsistensi, validitas metodologis, dan alignment dengan tujuan thesis.

**When to spawn:**
- **Before execution:** Review plan/code changes/analysis sebelum dieksekusi (gatekeeping)
- **After execution:** Review hasil untuk validasi akurasi klaim
- **On ambiguity:** Kalau ada trade-off yang tidak jelas, Lead Agent framing pro/kontra

**Deliverables:**
- `DECISION_MEMO.md` — pertimbangan pro/kontra untuk setiap keputusan besar
- `REVIEW_GATE.md` — verdict: **PROCEED** / **REVISE** / **ESCALATE_TO_USER**
- Risk assessment untuk setiap rencana eksekusi

**Handoff to:**
- **PROCEED** → eksekusi oleh agent yang relevan
- **REVISE** → kembali ke agent sebelumnya dengan feedback spesifik
- **ESCALATE_TO_USER** → saya (root agent) presentasikan ke user dengan framing Lead Agent

**Constraints:**
- Lead Agent tidak boleh langsung edit kode/data — hanya review dan rekomendasi
- Setiap pertimbangan harus punya basis: metodologis, statistik, atau engineering
- Kalau ada konflik antar agent, Lead Agent mediasi dan rekomendasikan resolusi

**Communication Model:**
```
Planning Agent → writes PLAN.md
     ↓
Lead Agent → reads PLAN.md → writes DECISION_MEMO.md
     ↓ (verdict: PROCEED / REVISE / ESCALATE)
Code Agent / Training / Analysis → eksekusi
     ↓
Lead Agent → review hasil → writes REVIEW_GATE.md
```

---

### 1. 🔧 Code Agent

**Scope:** Implementasi, refactoring, bug fix, arsitektur baru.

**When to spawn:**
- Perubahan >5 baris di codebase apapun (Python, Swift, C++, notebook)
- Refactor arsitektur (misal: GAM, FiLM fusion, ICP pipeline, scanning flow)
- Bug fix non-trivial
- Integrasi antar subproject (misal: output TrueDepthScan → input 3DCNN)

**Deliverables:**
- Code yang di-test (minimal smoke test / build test)
- `CHANGES.md` — summary perubahan, file yang diubah, breaking changes
- Update `tests/` kalau ada
- Untuk Swift: verifikasi build di Xcode
- Untuk C++: verifikasi compile

**Handoff to:** Peer-Review Agent (setelah CHANGES.md ditulis)

**Constraints:**
- Jangan ubah test tanpa dokumentasi
- Selalu pertahankan backward compat kalau bisa
- Ikuti coding style existing

---

### 2. 📊 Analysis Agent

**Scope:** Statistik inferensial, diagnostic, visualisasi, data audit.

**When to spawn:**
- Evaluasi hasil training selesai (download dari Colab)
- Diagnostic deep-dive (init parity, embedding stats, QC, registration accuracy)
- Plot/generate figure untuk laporan
- Analisis data scanning (quality metrics, coverage, noise)

**Deliverables:**
- `ANALYSIS_<timestamp>.md` — report dengan tabel, plot, interpretasi
- Figures di `docs/images/` atau subproject `result/` / `eval_results/`
- JSON/CSV raw data untuk reproducibility

**Handoff to:** Documentation Agent (untuk update REPORT.MD)

**Constraints:**
- Sertakan p-value, CI, dan effect size
- Jangan over-interpret — bedakan observasi vs kausalitas
- Plot harus ada caption yang menjelaskan apa yang dilihat

---

### 3. 🗺️ Planning Agent

**Scope:** Rencana milestone, estimasi waktu, risk assessment, decision framing.

**When to spawn:**
- Milestone baru (v0.5.0+, registration v2.0+, scan app v2.0+)
- Fase baru dalam milestone
- User stuck — butuh framing keputusan
- Integrasi antar subproject (misal: scan → registration → recognition pipeline)

**Deliverables:**
- `PLAN_<version>.md` — plan lengkap dengan checklist, estimasi, risk
- `DECISION_QUEUE.md` — daftar keputusan yang perlu user ambil
- Update `IMPROVEMENT_PLAN_vX.X.X.md`

**Handoff to:** User (via ExitPlanMode untuk approval)

**Constraints:**
- Plan harus punya fallback plan
- Estimasi waktu realistis (bukan best-case)
- Setiap keputusan harus di-frame dengan trade-off

---

### 4. 📝 Documentation Agent

**Scope:** Update REPORT.MD, README, plan files, changelog, glossary.

**When to spawn:**
- Setelah milestone selesai
- Setelah analysis report selesai
- Setelah keputusan penting diambil

**Deliverables:**
- Updated `REPORT.MD` / `README.md` / `LAPORAN_PROGRESS_TESIS.md`
- Updated `MONITORING_CHECKLIST_vX.X.X.md`
- Updated `PROGRESS_REPORT.md`
- Changelog per subproject

**Handoff to:** Peer-Review Agent (verifikasi akurasi data di dokumen)

**Constraints:**
- Jangan hapus entry lama — append atau update status
- Setiap angka harus traceable ke source
- Glossary harus di-update kalau ada istilah baru

---

### 5. 🔍 Peer-Review Agent

**Scope:** Verifikasi kode, verifikasi klaim metodologis, verifikasi akurasi dokumen.

**When to spawn:**
- Setelah Code Agent selesai (review kode)
- Setelah Documentation Agent selesai (review angka & klaim)
- Setelah Analysis Agent selesai (review metodologi statistik)

**Deliverables:**
- `REVIEW.md` — findings, severity (critical/warning/info), recommendations
- Verdict: **APPROVE** / **REQUEST_CHANGES** / **NEEDS_DISCUSSION**

**Handoff to:** Code Agent (kalau REQUEST_CHANGES) atau User (kalau NEEDS_DISCUSSION)

**Constraints:**
- Review harus spesifik (file, baris, masalah apa)
- Bedakan "style preference" vs "actual bug"
- Kalau klaim statistik tidak valid, jelaskan kenapa

---

## Communication Protocol

### 1. Spawn Protocol

```
Root Agent (me) evaluates task
    ↓
Decide which agent(s) needed
    ↓
Spawn agent with FULL context in prompt
    ↓
Agent reads relevant AGENTS.md section
    ↓
Agent executes task → writes artifact
    ↓
Agent reports completion to root
    ↓
Root decides next step (handoff, review, or done)
```

### 2. Context Transfer

Setiap agent prompt HARUS include:
- **Goal spesifik** — apa yang harus dicapai
- **File paths** — file yang relevan (absolute path)
- **Current state** — apa yang sudah selesai, apa yang belum
- **Constraints** — batasan (jangan ubah X, backward compat, dsb.)
- **Expected deliverable** — format output

### 3. Artifact Format

Setiap artifact harus punya header:
```markdown
# ARTIFACT: <name>
# Created by: <Agent Role>
# Date: <ISO timestamp>
# Related to: <milestone / task>
# Status: [DRAFT / REVIEWED / FINAL]
```

---

## Swarm Log

Log aktivitas swarm di `SWARM_LOG.md`. Format:
```markdown
| Timestamp | Agent | Task | Status | Artifact |
|-----------|-------|------|--------|----------|
| 2026-05-17T09:00:00 | Code Agent | Patch F1.1 RNG parity | DONE | CHANGES_20260517_090000.md |
```

---

## Current Swarm State (Global)

| Project | Milestone | Status | Active Agents | Pending Tasks |
|---------|-----------|--------|---------------|---------------|
| **3DCNN** | v0.4.0 Fase 1 | ✅ DONE | — | — |
| **3DCNN** | v0.4.0 Fase 2 | 🔄 READY | — | Training 4 varian × 5 seed di Colab |
| **3DCNN** | v0.5.0 | ⏳ NOT STARTED | — | TBD based on Fase 2 results |
| **3DRegistration** | v1.0 | ⏳ NOT STARTED | — | Initial ICP pipeline review |
| **TrueDepthScan** | v1.0 | ✅ STABLE | — | Data acquisition ongoing |

## Subproject Override

Setiap subproject bisa punya `AGENTS.md` sendiri yang **override** section tertentu dari file ini. Aturan precedence:
1. `SUBPROJECT/AGENTS.md` — paling spesifik
2. `~/Projects/Thesis/AGENTS.md` (this file) — global default
3. Root agent instructions — highest priority (user direct command)

---

*Last updated: 2026-05-21*
