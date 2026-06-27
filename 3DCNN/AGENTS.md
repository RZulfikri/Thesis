# Multi-Agent Swarm — 3DCNN / GeoAtt-PointNet++ Palm Recognition

> **Kontrak Agent Lokal** — Dokumen ini adalah **override** spesifik untuk project `3DCNN/`.
> 
> **Referensi global:** `~/Projects/Thesis/AGENTS.md` (definisi role, communication protocol, artifact format)
> **Project:** GeoAtt-PointNet++ Palm Recognition (PyTorch, Google Colab)
>
> Kalau ada konflik antara file ini dan global AGENTS.md, **file ini menang** untuk semua file di bawah `3DCNN/`.

---

## Project-Specific Context

### Tech Stack
- **Framework:** PyTorch 2.6.0+cu124
- **Training:** Google Colab (A100), notebook: `collab/train.ipynb`, `collab/evaluate.ipynb`
- **Local dev:** macOS, Python 3.12.8 (CPU-only for code dev & diagnostic)
- **Dataset:** `dataset/` — frame-level layout (11 subjek, ~1,869 valid frame post-QC v3)
- **Loss:** ArcFace (m=0.5, s=30) + Triplet hybrid
- **Enrollment:** Multi-prototype k-means (k=3)

### Key Files
| File | Purpose | Diubah oleh Agent |
|------|---------|-------------------|
| `models/encoder.py` | GeoAtt-PointNet++ encoder | Code Agent |
| `models/siamese.py` | Siamese wrapper + ArcFace head | Code Agent |
| `train.py` | CLI training script | Code Agent |
| `evaluate.py` | Evaluation + backward-compat loader | Code Agent |
| `utils/dataset.py` | Dataset scanner (frame-level) | Code Agent |
| `utils/data_qc_v3_frame.py` | QC v3 frame-level exclusion | Code Agent |
| `collab/train.ipynb` | Colab training notebook | Code Agent |
| `collab/evaluate.ipynb` | Colab evaluation notebook | Code Agent |
| `REPORT.MD` (root) | Living document | Documentation Agent |
| `IMPROVEMENT_PLAN_v0.4.0.md` | Plan lama (v0.4.0) | Planning Agent |
| `IMPROVEMENT_PLAN_v5.0.0.md` | Plan lama (v5.0.0 — di-pause) | Planning Agent |
| `IMPROVEMENT_PLAN_v0.6.0.md` | **Plan aktif** — ArcFace vs Triplet | Planning Agent |

### Active Milestone
- **v0.4.0 Fase 1:** ✅ DONE — diagnostik + patch kode
- **v0.4.0 Fase 2:** ✅ DONE — fair ablation training (4 varian × 5 seed)
- **v5.0.0:** ⏳ NOT STARTED — rencana GeoAtt low-data (di-pause)
- **v0.6.0:** 🔄 READY — pivot ke ArcFace vs Triplet loss comparison (plan aktif)

### 4 Varian Ablasi
| Varian | Flag | use_gam | use_geom_fusion |
|--------|------|---------|-----------------|
| `no_geom` | (none) | False | False |
| `with_geom` | `--use-geom` | True | True |
| `gam_only` | `--use-gam` | True | False |
| `fuse_only` | `--use-geom-fusion` | False | True |

### Seeds
Training seeds: `[7, 42, 123, 2026, 31337]`  
Split seed: `42`

---

## Project-Specific Agent Constraints

### Code Agent
- **Never** edit `collab/*.ipynb` tanpa verifikasi cell order (Colab sensitive)
- Selalu test import: `python3 -c "import models.encoder; print('OK')"`
- Kalau ubah `models/encoder.py`, jalankan `utils/audit_init_parity.py` untuk verifikasi RNG parity
- Checkpoint backward-compat: pastikan `evaluate.py` masih bisa load v0.3.0 checkpoints

### Analysis Agent
- Statistik utama: Wilcoxon paired, Bootstrap CI, McNemar pooled
- Plot wajib: CMC curve, confusion matrix, training curve (loss/accuracy/val)
- Data source: download dari Colab `runs/<variant>/<timestamp>/`
- Format output: `ANALYSIS_3DCNN_<timestamp>.md`

### Planning Agent
- Estimasi training: ~2 jam per varian per seed di A100 → 4 varian × 5 seed = ~40 jam total
- Fallback: kalau anggaran Colab habis, prioritaskan no_geom + with_geom dulu
- Decision branch B1–B4 menentukan arah v0.5.0

### Documentation Agent
- Update `REPORT.MD` di root (bukan di 3DCNN/)
- Update `IMPROVEMENT_PLAN_v0.4.0.md` untuk plan spesifik
- Glossary istilah: GeoAtt, GAM, ArcFace, FiLM, CMC, EER, mAP (lihat root REPORT.MD)

### Peer-Review Agent
- Verifikasi RNG init parity: `utils/audit_init_parity.py` harus pass (58/58 identical)
- Verifikasi dataset QC: `scan_dataset_frames()` harus return ~1,869 frame
- Verifikasi backward-compat: load v0.3.0 checkpoint tanpa error
- Statistik: pastikan p-value dan CI dihitung dengan benar (n=5 seed)

### Lead Agent
- Gatekeep sebelum training: pastikan init parity verified, dataset QC applied, notebook updated
- Gatekeep sebelum v0.5.0: pastikan decision branch B1–B4 sudah jelas dari data
- Risk assessment: Colab quota, training time, reproducibility

---

## Communication Shortcuts

Kalau agent butuh context cepat:
- **Hasil v0.3.0:** lihat `REPORT.MD` Bagian 2
- **Diagnostik Fase 1:** lihat `result_docs/20260517_064046/diagnostic_phase1.md`
- **Plan Fase 2:** lihat `IMPROVEMENT_PLAN_v0.4.0.md`
- **QC v3:** lihat `utils/data_qc_v3_frame.py` + hasil di terminal

---

*Referensi global: ~/Projects/Thesis/AGENTS.md*  
*Last updated: 2026-05-17*
