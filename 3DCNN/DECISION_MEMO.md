# ARTIFACT: DECISION_MEMO_v0.4.0_Fase2
# Created by: Lead Agent
# Date: 2026-05-17T08:10:00+07:00
# Related to: v0.4.0 Fase 2 — Fair Ablation Training (4 varian × 5 seed)
# Status: FINAL

# DECISION MEMO — Gate Review v0.4.0 Fase 2

**Lead Agent Review** untuk transisi dari Fase 1 (diagnostik + patch) ke Fase 2 (training ablasi fair).  
**Referensi:** `REPORT.MD` §3–4, `IMPROVEMENT_PLAN_v0.4.0.md`, `diagnostic_phase1.md`.

---

## 1. Executive Summary — QC v3 Outcome

**Verdict QC v3: ADOPTED — siap digunakan sebagai training set Fase 2.**

| Parameter | Nilai | Kriteria | Status |
|-----------|-------|----------|--------|
| Total frame pre-QC | 2,120 | — | Baseline |
| Frame excluded (individual) | 160 (7.5%) | Reversibel (`_QC2_frame_*` prefix) | ✅ |
| Entire session excluded | 1 sesi (`yanuar/20260513_092145`, 60% outlier) | Reversibel (`_QC2_*` prefix) | ✅ |
| **Total exclusion rate** | **8.02%** | < 10% (acceptable loss) | ✅ |
| **Valid frames post-QC** | **1,869** | Diverifikasi scanner | ✅ |
| Metode | Frame-level MAD (k=10, threshold=0.5) | Sesuai user request | ✅ |
| Scanner skip `_QC2_` | `dataset.py` L384–391 | Verified in code | ✅ |

**Distribusi exclusion per subjek** konsisten dengan temuan D2 Fase 1:
- `reysa` 11.3%, `feby` 11.4%, `chrys` 11.0% → subjek dengan high variance / knuckle fallback / bimodal.
- `rahmat` 2.0%, `alji` 4.6%, `fadhil` 4.6% → subjek paling bersih.
- `nola` 6.7% → finger_width_5 instability terserap secara frame-level (bukan cherry-pick manual).

**Kelebihan QC v3 vs v2:**
- v2 session-level membuang 16.5% sesi (terlalu agresif, ditolak user).
- v3 frame-level hanya membuang 8.02% data, dengan granularitas lebih tinggi.
- Reversibel: hapus prefix `_QC2_` untuk restore.

---

## 2. Risk Assessment for Fase 2 Training

### 2.1 Risk Register

| ID | Risk | Likelihood | Impact | Severity | Mitigasi |
|----|------|------------|--------|----------|----------|
| R1 | **Colab A100 quota habis** sebelum 20 run selesai | Medium | High | 🔴 | Pilot seed-42 dulu (4 run = ~2 jam). Jika quota habis, prioritaskan no_geom + with_geom 5 seed (B1 verdict). |
| R2 | **Init parity regresi** akibat edit kode terakhir | Low | High | 🟡 | Re-run `audit_init_parity.py` sebagai gate checklist item G4. |
| R3 | **Dataset scanner tidak skip `_QC2_`** di Colab (path case-sensitivity) | Low | Medium | 🟡 | Verifikasi frame count = 1,869 di gate checklist G5. |
| R4 | **Notebook Colab tidak support 4 varian** (`USE_GAM`/`USE_GEOM_FUSION`) | Low | High | 🟡 | Gate checklist G6 — inspect cell flag di `collab/train.ipynb`. |
| R5 | **Normaliser leakage** (fit dari seluruh dataset, bukan train split) | Low | High | 🟡 | Gate checklist G7 — verifikasi `fit()` dipanggil setelah split, sebelum training. |
| R6 | **Checkpoint v0.3.0 accidentally di-load** (fine-tune vs from-scratch) | Low | Medium | 🟢 | Gate checklist G8 — explicit `from_scratch=True` atau hapus folder run sebelum training. |
| R7 | **Hard drive Colab penuh** (checkpoint 4 varian × 5 seed) | Medium | Medium | 🟡 | Auto-delete intermediate epoch, hanya simpan `best_rank1.pth` + `best_loss.pth`. |

### 2.2 Residual Risk (post-mitigasi)

- **Primary:** Colab quota. Risiko tidak sepenuhnya eliminasi, hanya dikurangi via strategi pilot-first.
- **Secondary:** Waktu. Estimasi 40 jam A100 untuk full run. Jika kuota terbatas, fallback ke 2 varian (no_geom + with_geom) menghambat decision branch B2–B4.

---

## 3. Gate Checklist — MUST Verify Before Training Starts

Setiap item **HARUS** ✅ sebelum runtime pertama di-submit ke Colab.

| # | Item | Verification Method | Owner | Priority |
|---|------|---------------------|-------|----------|
| G1 | QC v3 applied di folder `dataset/` (ada prefix `_QC2_`) | `ls dataset/*/2026*/_QC2_*` returns files | Code Agent | 🔴 Critical |
| G2 | `utils/dataset.py` committed & pushed ke repo | `git diff HEAD -- utils/dataset.py` clean | Code Agent | 🔴 Critical |
| G3 | `models/encoder.py`, `siamese.py`, `train.py`, `evaluate.py` committed | `git status` clean di branch aktif | Code Agent | 🔴 Critical |
| G4 | **Init parity pass** (58/58 identical, max\|Δ\|=0) | `python utils/audit_init_parity.py` | Peer-Review Agent | 🔴 Critical |
| G5 | **Dataset scanner returns 1,869 frames** | `python -c "from utils.dataset import scan_dataset_frames; print(len(...))"` | Peer-Review Agent | 🔴 Critical |
| G6 | **Colab notebook supports 4 variant flags** | Buka `collab/train.ipynb`, verifikasi ada cell `USE_GAM` dan `USE_GEOM_FUSION` | Code Agent | 🔴 Critical |
| G7 | **GeometryNormalizer fit-on-train-only** | Cek `train.py` — `normalizer.fit()` dipanggil setelah `split_sessions_three_way()`, sebelum `PalmFrameDataset` | Peer-Review Agent | 🟡 High |
| G8 | **Explicit from-scratch (no warm-start)** | Verifikasi `train.py` tidak auto-load checkpoint existing; atau set `resume=False` | Code Agent | 🟡 High |
| G9 | **Reproducibility config tercatat** | `config.json` template siap: seeds, split_seed, hyperparameters, variant flags | Documentation Agent | 🟡 High |
| G10 | **Backup strategy** | Google Drive `runs/` folder punya quota; atau set checkpoint upload otomatis | Code Agent | 🟢 Medium |

**Lead Agent Note:** G4 dan G5 adalah *kill criteria* — kalau salah satu gagal, training tidak fair atau data tidak bersih, dan seluruh Fase 2 menjadi invalid. Jangan skip.

---

## 4. Verdict

# ✅ PROCEED — with pilot-first execution strategy

**Basis keputusan:**
1. **QC v3 valid:** Frame-level exclusion 8.02% adalah metode yang konservatif, reversibel, dan didukung oleh temuan diagnostik (D2: nola outlier, reysa/feby high variance).
2. **Init parity verified:** 58/58 layer identik untuk semua seed — fair-ablation strict tercapai.
3. **Dataset scanner updated:** `scan_dataset_frames()` secara eksplisit skip `_QC2_*` frame folders (L390) dan `_QC2_*` session folders (L384).
4. **Code patches committed:** F1.1–F1.4 semua diimplementasikan dan terdokumentasi.
5. **Risiko manageable:** Risiko utama adalah compute budget, bukan validitas metodologis. Strategi pilot-first mengurangi exposure.

**Bukan REVISE** karena tidak ada bug atau inconsistency yang terdeteksi di gate checklist.
**Bukan ESCALATE** karena tidak ada trade-off ambigu yang memerlukan input user — arah eksekusi sudah jelas dari plan.

---

## 5. Prioritized Execution Order (PROCEED)

### 5.1 Strategi: Pilot → Full

Berdasarkan risk assessment R1 (Colab quota), eksekusi **TIDAK** dilakukan 20 run sekaligus. Gunakan pendekatan **pilot → full** untuk memaksimalkan informasi per jam compute:

#### Phase A — Pilot (1 seed, 4 variants) — ~2 jam A100
**Goal:** Dapatkan directional signal untuk decision branch B1–B4 sebelum komitmen penuh.

| Urutan | Varian | Seed | Flag | Tujuan | Durasi Est. |
|--------|--------|------|------|--------|-------------|
| A1 | `no_geom` | 42 | (none) | Baseline + verifikasi init fair tidak merusak no_geom | ~30 min |
| A2 | `with_geom` | 42 | `--use-geom` | Direct comparison: apakah gap masih ada? | ~30 min |
| A3 | `gam_only` | 42 | `--use-gam` | Isolasi GAM | ~30 min |
| A4 | `fuse_only` | 42 | `--use-geom-fusion` | Isolasi fusion concat | ~30 min |

**Decision gate setelah Phase A:**
- **B1:** `with_geom` ≈ `no_geom` (gap ≤ 1 ppt) → **Phase B1** (full 5 seed hanya no_geom + with_geom). Skip gam_only/fuse_only. Tutup investigasi.
- **B2/B3/B4:** `with_geom` ≪ `no_geom` (gap > 2 ppt) → **Phase B2** (full 5 seed untuk SEMUA 4 varian). Butuh gam_only/fuse_only untuk decompose.

#### Phase B1 — Full Baseline (5 seeds, 2 variants) — ~8 jam A100
*Hanya jika B1 terpenuhi (gap hilang).*

| Varian | Seeds | Durasi Est. |
|--------|-------|-------------|
| `no_geom` | 7, 42, 123, 2026, 31337 | ~4 jam |
| `with_geom` | 7, 42, 123, 2026, 31337 | ~4 jam |

**Deliverable:** Tag `v0.4.0-baseline`, laporan: init parity adalah penyebab dominan gap v0.3.0.

#### Phase B2 — Full Decomposition (5 seeds, 4 variants) — ~38 jam A100
*Hanya jika B2/B3/B4 terpenuhi (gap persisten).*

| Varian | Seeds | Durasi Est. |
|--------|-------|-------------|
| `no_geom` | 7, 42, 123, 2026, 31337 | ~4 jam |
| `with_geom` | 7, 42, 123, 2026, 31337 | ~4 jam |
| `gam_only` | 7, 42, 123, 2026, 31337 | ~4 jam |
| `fuse_only` | 7, 42, 123, 2026, 31337 | ~4 jam |

**Deliverable:** Decision branch B2/B3/B4 jelas → arah F2.3/F2.4/F2.5.

### 5.2 Parallelization di Colab

- A1 dan A2 bisa sequential dalam satu session (total ~1 jam).
- A3 dan A4 bisa sequential dalam session kedua (total ~1 jam) → bisa di-run paralel dengan session pertama kalau punya 2 akun Colab.
- Phase B: gunakan **loop notebook** — set seed, train, eval, simpan hasil, ulang. Jangan buka session baru per seed (overhead mount Drive ~5 menit).

### 5.3 Fallback Compute

Jika quota A100 habis di tengah jalan:
1. **Prioritas 1:** `no_geom` + `with_geom` 5 seed (B1 verdict minimum).
2. **Prioritas 2:** `gam_only` + `fuse_only` seed 42 saja (directional signal B2–B4).
3. **Prioritas 3:** Full 5 seed untuk gam_only + fuse_only (statistical power penuh).

---

## 6. Key Recommendations

### 6.1 Untuk Code Agent (pre-flight)
- [ ] **Verifikasi G4 (init parity)** sekali lagi setelah commit terakhir: `python utils/audit_init_parity.py`.
- [ ] **Verifikasi G5 (frame count)** di local sebelum upload: `python -c "from utils.dataset import scan_dataset_frames; lf, sg = scan_dataset_frames('dataset'); print(sum(len(v) for v in lf.values()))"` → must return 1,869.
- [ ] **Update MONITORING_CHECKLIST** dari v0.3.0 ke v0.4.0 — dokumen saat ini masih berisi plan lama (Triplet, PLY Direct, Normals Ablation). Ini bukan blocker training, tapi akan membingungkan saat eksekusi.

### 6.2 Untuk Analysis Agent (post-pilot)
- Setelah Phase A selesai, generate **quick-look comparison** (bukan statistik penuh):
  - Rank-1 per variant (seed 42)
  - Training curve (loss, train_acc, val_acc) — visual inspection overfit/underfit
  - Verdict directional: B1 / B2 / B3 / B4
- Jangan tunggu 5 seed untuk decision branching — gunakan pilot.

### 6.3 Untuk Documentation Agent
- Update `MONITORING_CHECKLIST_v0.4.0.md` dengan checklist Fase 2 spesifik (bukan v0.3.0 lama).
- Update `REPORT.MD` §4.5 checklist: ubah status training items dari `[ ]` ke `[~]` (in-progress) saat Phase A dimulai.

### 6.4 Untuk User (Rahmat)
- **Approve compute budget:** ~2 jam untuk Phase A pilot. Setelah pilot, review hasil directional sebelum komitmen ~38 jam Phase B.
- **Monitor Colab quota:** Jika quota A100 menipis, switch ke T4 untuk Phase B2 varian non-kritis (gam_only/fuse_only) — no_geom/with_geom tetap A100 untuk konsistensi dengan baseline.

---

## 7. Sign-off

| Role | Name | Verdict | Date |
|------|------|---------|------|
| Lead Agent | (Auto) | ✅ PROCEED — Pilot-First | 2026-05-17 |
| Next Action | | Spawn Code Agent untuk pre-flight G1–G10 | |
| Handoff to | | Code Agent → verifikasi gate checklist → eksekusi Phase A | |

---

*This memo is append-only. Any revision must be noted in a new section with timestamp.*
