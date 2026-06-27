# Executive Report — GeoAtt-PointNet++ Palm Recognition

> **Tanggal Update:** 2026-05-17  
> **Penulis:** Rahmat Zulfikri  
> **Tujuan:** Ringkasan eksekutif seluruh eksperimen (v0.2.0 → v0.3.0 → v0.4.0) dengan checklist pelacakan progress dan rencana tindak lanjut.  
> **Cara pakai:** Update dokumen ini setiap kali milestone tercapai. Setiap bagian punya checklist yang harus dicek/di-uncheck sesuai status aktual.

---

## Daftar Isi

1. [Ringkasan Eksekutif Global](#ringkasan-eksekutif-global)
2. [v0.2.0 — Triplet Baseline](#bagian-1--v020-triplet-baseline)
3. [v0.3.0 — ArcFace Leap](#bagian-2--v030-arcface-leap)
4. [v0.4.0 Fase 1 — Diagnostik (Selesai)](#bagian-3--v040-diagnostik--fair-ablation)
5. [v0.4.0 Fase 2 — Fair Ablation Training (Rencana)](#bagian-4--rencana-v040-fase-2-akan-dilakukan)
6. [Artefak & Path Referensi](#bagian-5--artefak--path-referensi)
7. [Keputusan yang Perlu Diambil](#bagian-6--ringkasan-keputusan-yang-perlu-diambil)
8. [Hypothesis Log](#bagian-7--hypothesis-log)
9. [Decision Log](#bagian-8--decision-log)
10. [Known Issues & Workarounds](#bagian-9--known-issues--workarounds)
11. [Aborted / Paused Experiments](#bagian-10--aborted--paused-experiments)
12. [Reproducibility Checklist (Pre-Run)](#bagian-11--reproducibility-checklist-pre-run)
13. [Data Manifest](#bagian-12--data-manifest)
14. [Hardware / Environment Log](#bagian-13--hardware--environment-log)
15. [Future Work / Backlog](#bagian-14--future-work--backlog)
16. [Template v0.5.0+](#bagian-15--template-v050)
17. [Glossary](#bagian-16--glossary)

---

## Ringkasan Eksekutif Global

| Versi | Loss | Dataset | no_geom Rank-1 | with_geom Rank-1 | Verdict GeoAtt |
|-------|------|---------|----------------|------------------|----------------|
| **v0.2.0** | Triplet (batch-hard) | 11 subjek | 55.5% ± 13.6% | 59.8% ± 2.6% | **Netral** (p=1.0) |
| **v0.3.0** | ArcFace (m=0.5, s=30) | 11 subjek | **99.82% ± 0.36%** | 95.82% ± 1.59% | **Merugikan** (p=1.8×10⁻⁵) |
| **v0.4.0** | ArcFace + fair init *(planned)* | 11 subjek (post-QC) | TBD | TBD | **Investigasi aktif** |

**Narasi:** GeoAtt berperan sebagai "regularizer" ketika loss lemah (Triplet), tetapi justru merugikan begitu backbone + ArcFace sudah sangat kuat. v0.4.0 sedang menginvestigasi apakah gap ini intrinsic atau artefak metodologis (init unfair + data noisy).

---

# Bagian 1 — v0.2.0: Triplet Baseline

## 1.1 Konteks & Tujuan

- **Loss:** Online Triplet (batch-hard), margin 0.3
- **Enrollment:** Mean 1 sesi per subjek (naif)
- **Dataset:** 11 subjek, frame-level layout, ~2,100 frame
- **Training:** 100 epoch, batch 512, n_points 4096 → 8192
- **Hypothesis:** GeoAtt meningkatkan akurasi dan mengurangi varians

## 1.2 Hasil Utama

| Metrik | with_geom | no_geom | Δ | Signifikansi |
|--------|-----------|---------|---|--------------|
| Rank-1 | 59.8% ± 2.6% | 55.5% ± 13.6% | +4.4 ppt | **Tidak signifikan** (Wilcoxon p=1.0) |
| Rank-5 | 92.4% ± 1.8% | 88.2% ± 7.8% | +4.2 ppt | — |
| mAP | 73.1% ± 2.1% | 69.6% ± 11.2% | +3.5 ppt | — |
| EER | ~29.0% | ~28.5% | +0.5 ppt | — |

## 1.3 Temuan Kritis

1. **Loss triplet stagnan ~0.73 sejak epoch 5–10** → bottleneck utama adalah loss/sampling, bukan augmentasi.
2. **Gallery enrollment naif** → Rank-1 60% vs Rank-5 92% mengindikasikan template underrepresent intra-class variance.
3. **EER ~29%** → separasi genuine/impostor sangat lemah, embedding partial collapse.
4. **Data quality outlier:** `feby` (0% holdout), `nola` (33%), `reysa` (variabel) → kemungkinan scan/registration issue.

## 1.4 Checklist v0.2.0 — Status

- [x] Training Triplet loss (with_geom & no_geom)
- [x] Multi-seed evaluation (5 seeds)
- [x] Identifikasi bottleneck loss stagnan
- [x] Identifikasi gallery enrollment naif sebagai masalah
- [x] Data QC awal (outlier subjek: feby, nola, reysa)
- [x] Laporan baseline v0.2.0

---

# Bagian 2 — v0.3.0: ArcFace Leap

## 2.1 Konteks & Tujuan

- **Loss:** ArcFace (m=0.5, s=30) — 3-phase training
  - Phase 1: ArcFace pretrain (supervised, 11-class, 100 ep)
  - Phase 2: Hybrid ArcFace+Triplet (70/30, 30 ep)
  - Phase 3: Pure Triplet refinement (20 ep)
- **Enrollment:** Multi-prototype k-means (k=3) per subjek
- **Augmentasi:** Original-space (rotasi Z ±30°, tilt ±15°, jitter, dropout)
- **Hypothesis:** ArcFace mengatasi stagnasi triplet; GeoAtt akan memperkuat margin

## 2.2 Hasil Utama

| Metrik | with_geom | no_geom | Δ | Signifikansi |
|--------|-----------|---------|---|--------------|
| **Rank-1** | 95.82% ± 1.59% | **99.82% ± 0.36%** | **−4.0 ppt** | **Signifikan** (McNemar p=1.8×10⁻⁵) |
| Rank-5 | 99.64% ± 0.73% | 100.00% ± 0.00% | −0.4 ppt | — |
| mAP | 97.29% ± 1.08% | 99.88% ± 0.24% | −2.6 ppt | — |
| **EER** | **2.76%** | **0.03%** | **+2.7 ppt** | — |
| AUC | 0.9962 | ~1.0000 | −0.004 | — |

### Statistik Inferensial

| Uji | Hasil | Interpretasi |
|-----|-------|--------------|
| Wilcoxon paired (n=5) | W=0, p=0.0625 | Borderline (power rendah karena n kecil) |
| Bootstrap CI 95% Δ | [−0.0527, −0.0309] | **Tidak melingkupi 0** → konsisten merugikan |
| McNemar pooled (n=550) | χ²=18.375, p=1.8×10⁻⁵ | **Signifikan** — 23 probe benar hanya di no_geom vs 1 hanya di with_geom |

## 2.3 Temuan Kritis

1. **Lompatan absolut +36–44 ppt** hanya dari ganti loss → bottleneck v0.2.0 memang loss, bukan arsitektur.
2. **Pembalikan arah GeoAtt.** Pada Triplet: GeoAtt sedikit membantu. Pada ArcFace: GeoAtt merugikan.
3. **Kesalahan with_geom terkonsentrasi pada `nola`.** 4/5 seed nola gagal di holdout with_geom; no_geom tidak pernah gagal.
4. **no_geom mendekati sempurna** (99.82% Rank-1, 0.03% EER) → headroom evaluasi sangat tipis.

## 2.4 Checklist v0.3.0 — Status

### Quick Wins (S1–S3)
- [x] **S1:** Implementasi multi-prototype enrollment (`utils/enrollment.py`)
- [x] **S2:** Implementasi Hybrid ArcFace + Triplet (`losses/arcface.py`)
- [x] **S3:** Data QC outlier subjects (audit feby, nola, reysa)

### Training & Evaluasi
- [x] Training ArcFace 5-seed (with_geom)
- [x] Training ArcFace 5-seed (no_geom)
- [x] Evaluasi multi-prototype (k=3)
- [x] Komparasi statistik (Wilcoxon, Bootstrap, McNemar)
- [x] Laporan evaluasi v0.3.0-baseline

### PLY Direct (E3) — **DI-TUNDA**
- [ ] `PLYDirectDataset` + `OriginalSpaceAugmentor`
- [ ] Integrasi ke `collab/train.ipynb`
- [ ] Verifikasi identitas output PLYDirect vs `cnn_input.npy`
- [ ] Training E3 (PLY + Hybrid + multi)

> **Catatan:** PLY Direct ditunda karena hasil v0.3.0 menunjukkan bottleneck bukan augmentasi. Plan dialihkan ke v0.4.0 diagnostik.
>
> **Update 2026-05-17:** `geometry.json` di-re-extract menggunakan `3DRegistration/extract_geometry.py` terbaru (knuckle detection fallback fix). QC v3 di-run ulang → 185 frame excluded (vs 160 sebelumnya). Dataset siap untuk Fase 2.

---

# Bagian 3 — v0.4.0: Diagnostik & Fair Ablation

## 3.1 Konteks & Motivasi

Setelah v0.3.0 menunjukkan GeoAtt merugikan pada ArcFace, muncul pertanyaan ilmiah:

> *"Apakah GeoAtt memang merugikan secara intrinsik, atau ada bias eksperimen yang menutupi efeknya?"*

## 3.2 Fase 1: Diagnostik — Status **SELESAI**

### 7 Hipotesis Diuji

| # | Hipotesis | Verdict | Bukti |
|---|-----------|---------|-------|
| 1 | Saturasi ceiling pada 11 subjek | **DITOLAK** | Gap with−no **MEMBESAR** di hard probes (Δ=−0.056 → −0.107) |
| 2 | Fitur 14-dim geom mayoritas noise (FDR<1) | **DITOLAK** | 0/14 fitur FDR<1; median FDR=3.77 |
| 3 | Kesalahan with_geom sistematis pada subjek `nola` | **TERKONFIRMASI** | `nola.finger_width_5` CV=0.497 vs rata-rata 0.056 (**8.85× outlier**) |
| 4 | RNG init parity rusak antara use_geom=True/False | **TERKONFIRMASI** | 13/33 shared layer beda init; hanya 1.5% elemen identik |
| 5 | Geom-emb shared GAM1+GAM2 → gradient bottleneck | TIDAK DIUJI | Memerlukan ablasi arsitektur (Fase 2) |
| 6 | Dropout(0.3) hanya di fusion head → asimetri train/eval | **TERKONFIRMASI ringan** | cos(eval,train_dropout): with_geom 0.908 vs no_geom 0.950 |
| 7 | Z-score menghapus skala absolut tangan | TIDAK TERBUKTI RELEVAN | Separabilitas dataset-wide tetap baik |

### Sintesis Akar Penyebab

GeoAtt **memang** memberi kontribusi merugikan, **tetapi** dengan dua kontaminasi eksperimen:

1. **Ablasi tidak fair (D1):** with_geom vs no_geom mulai dari titik awal berbeda karena RNG init parity rusak.
2. **Subjek/fitur instabil (D2+D3):** instabilitas `finger_width_5` nola antar-sesi dilewatkan ke ArcFace yang scale-sensitive.

### Perbaikan Kode Sudah Diimplementasikan

| Patch | File | Deskripsi | Status |
|-------|------|-----------|--------|
| **F1.1** | `models/encoder.py` | RNG parity — semua sub-modul selalu dibangun, flag hanya kontrol forward | ✅ Verified (58/58 layer identik) |
| **F1.2** | `models/siamese.py`, `train.py` | Flag ablasi 4 varian: `no_geom`, `with_geom`, `gam_only`, `fuse_only` | ✅ CLI ready |
| **F1.3** | `evaluate.py` | Backward-compat checkpoint loader (pre-v0.4.0 → new proj heads) | ✅ Smoke test pass |
| **F1.4** | `utils/` × 4 skrip | Diagnostik: init parity, geom variance, embedding stats, hard probes | ✅ Output tersedia |

### QC v2 → v3: Frame-Level Exclusion — Status **SELESAI**

**QC v2 (session-level, deprecated):**
- **Skrip:** `utils/data_qc_v2.py`
- **Kriteria:** within-session std fitur mana pun > k × median_global
- **Hasil:** 35 sesi flagged (16.5%) → **ditolak user** karena terlalu agresif

**QC v3 (frame-level, adopted):**
- **Skrip:** `utils/data_qc_v3_frame.py`
- **Threshold:** k=10, session_threshold=0.5
- **Kriteria per frame:** |value − median_session| > k × MAD_session untuk fitur mana pun
- **Kriteria per sesi:** >50% frame outlier → seluruh sesi di-exclude
- **Folder flagged:** frame di-rename `_QC2_frame_XX`, sesi di-rename `_QC2_YYYY...` (reversibel)

**Hasil QC v3:**

| Parameter | Nilai |
|-----------|-------|
| Total frame | 2,120 |
| Frame excluded | 160 (7.5%) |
| Entire session excluded | 1 (yanuar/20260513_092145, 60% outlier) |
| Partial sessions | 89 sesi |
| Exclusion rate | **8.02%** |

**Distribusi frame excluded per subjek:**

| Subjek | Frame Excluded | Rate | Notes |
|--------|---------------|------|-------|
| aisah | 14/200 | 7.0% | 9 partial sessions |
| alji | 7/150 | 4.6% | 6 partial sessions |
| chrys | 22/200 | 11.0% | 12 partial sessions (knuckle fallback) |
| fadhil | 7/150 | 4.6% | 5 partial sessions |
| feby | 24/210 | 11.4% | 11 partial sessions (bimodal) |
| gede | 13/200 | 6.5% | 8 partial sessions |
| nola | 15/220 | 6.7% | 7 partial sessions |
| rahmat | 3/150 | 2.0% | 2 partial sessions (cleanest) |
| reysa | 27/240 | 11.3% | 13 partial sessions (high variance) |
| taufik | 12/200 | 6.0% | 7 partial sessions |
| yanuar | 16/200 | 8.0% | 9 partial + 1 entire session |

**Rationale:** User memilih frame-level karena hanya 1–2 frame buruk dalam sesi kemungkinan akibat user bergerak saat start/stop scan. Sesi dengan bimodal penuh (feby, chrys) tetap kehilangan banyak frame karena instabilitas deteksi. — **Decision 2026-05-17**

## 3.3 Checklist v0.4.0 Fase 1 — Status

- [x] Audit D1: RNG init parity (pre-patch)
- [x] Audit D2: Geometric session variance (FDR + outlier per subjek)
- [x] Audit D3: Embedding stats (dropout asymmetry)
- [x] Audit D5: Hard-probe saturation check
- [x] Patch F1.1: RNG parity di `encoder.py`
- [x] Patch F1.2: Flag 4 varian di `train.py`
- [x] Patch F1.3: Backward-compat loader di `evaluate.py`
- [x] Patch F1.4: 4 skrip diagnostik baru
- [x] Re-audit D1 post-patch (58/58 identical)
- [x] QC v2: Filter 35 sesi dengan kriteria objektif (k=10)
- [x] Laporan diagnostik Fase 1 (`diagnostic_phase1.md`)
- [x] Verifikasi dataset QC v2 (179 sesi bersih, 35 flagged) → deprecated
- [x] Verifikasi init parity pasca-patch (58/58 identical, max|Δ|=0)
- [x] Update `collab/train.ipynb` untuk support 4 varian (USE_GAM + USE_GEOM_FUSION)
- [x] Update `collab/evaluate.ipynb` untuk support 4 varian
- [x] Fix `utils/dataset.py` — skip folder `_QC2_` dan `_QUARANTINE_` di session + frame level
- [x] QC v3 frame-level (geometry lama): 160 frame excluded, 1 entire session excluded (8.02%)
- [x] Re-extract geometry.json dengan knuckle fallback fix (`3DRegistration/extract_geometry.py`)
- [x] QC v3 frame-level (geometry BARU): 185 frame excluded, 2 entire sessions excluded (8.54%)
- [x] Verifikasi scanner pasca-QC v3: **1,836 valid frame** dari 2,095 total
- [x] Fix CRITICAL Peer-Review: `train.py` --seed + RNG seeding + config.json
- [x] Fix CRITICAL Peer-Review: `evaluate.py` --use-gam / --use-geom-fusion flags

---

# Bagian 4 — Rencana v0.4.0 Fase 2 (Akan Dilakukan)

## 4.1 Tujuan

Verifikasi apakah gap with_geom vs no_geom **tetap ada** setelah:
1. Init parity diperbaiki (F1.1)
2. Data noisy di-filter via QC v2

## 4.2 Strategi Training: From Scratch, 4 Varian

**Harus from scratch** (bukan fine-tune dari v0.3.0) karena:
- Patch F1.1 mengubah urutan init → bobot awal berbeda dari v0.3.0
- 4 varian baru (`gam_only`, `fuse_only`) tidak punya checkpoint sumber

| Varian | Flag | Tujuan Diagnostik |
|--------|------|-------------------|
| **no_geom** | — | Baseline murni |
| **with_geom** | `--use-geom` | Full GeoAtt (re-baseline) |
| **gam_only** | `--use-gam` | Isolasi efek GAM saja |
| **fuse_only** | `--use-geom-fusion` | Isolasi efek fusion concat saja |

**Hyperparameter identik v0.3.0:**
- ArcFace m=0.5, s=30
- Batch 512, n_points 8192
- Phase1 100ep / Phase2 30ep / Phase3 20ep
- LR 2e-3 → 2e-4
- Seeds: [7, 42, 123, 2026, 31337]

## 4.3 Decision Branch (Berdasar Hasil Fase 2)

```
F2.1 Training 4-arah × 5 seed
        │
        ▼
┌─────────────────────────────────────┐
│ B1: with_geom ≈ no_geom             │
│     → Masalah dominan D1 (init)     │
│     → Tutup investigasi             │
└─────────────────────────────────────┘
        │
        ▼ gap masih ada
┌─────────────────────────────────────┐
│ B2: gam_only ≪ no_geom,             │
│     fuse_only ≈ no_geom             │
│     → GAM penyebab utama            │
│     → F2.3: Cross-attention GAM     │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ B3: fuse_only ≪ no_geom,            │
│     gam_only ≈ no_geom              │
│     → Fusion concat penyebab utama  │
│     → F2.4: Gated/FiLM fusion       │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ B4: keduanya ≪ no_geom              │
│     → Konsumsi geom umum salah      │
│     → F2.5: Aux loss + feature eng  │
└─────────────────────────────────────┘
```

## 4.4 Target Metrik v0.4.0

| Metrik | v0.3.0 no_geom | v0.3.0 with_geom | Target v0.4.0 with_geom |
|--------|----------------|------------------|-------------------------|
| Rank-1 mean | 99.82% | 95.82% | **≥ no_geom** atau **> no_geom + 0.5%** |
| Rank-1 std | 0.36% | 1.59% | ≤ no_geom std |
| EER | 0.03% | 2.76% | ≤ 0.10% |
| McNemar pooled | — | 23 vs 1 (no_geom menang) | b ≈ c (no significant winner) |
| Wilcoxon p | — | 0.0625 | p > 0.10 (atau favoring with_geom) |

**Kriteria sukses minimum:** GeoAtt tidak terbukti merugikan secara statistik (McNemar p>0.05, bootstrap CI Δrank-1 melingkupi 0).

## 4.5 Checklist v0.4.0 Fase 2 — Status

- [ ] Training `no_geom` × 5 seed (from scratch, init fair)
- [ ] Training `with_geom` × 5 seed (from scratch, init fair)
- [ ] Training `gam_only` × 5 seed
- [ ] Training `fuse_only` × 5 seed
- [ ] Evaluasi 4 varian dengan multi-prototype (k=3)
- [ ] Komparasi statistik 4-arah (Wilcoxon + Bootstrap + McNemar)
- [ ] Decision branch B1/B2/B3/B4
- [ ] Jika B2 → Implementasi F2.3 Cross-Attention GAM
- [ ] Jika B3 → Implementasi F2.4 Gated/FiLM Fusion
- [ ] Jika B4 → Implementasi F2.5 Feature Engineering + Auxiliary Loss
- [ ] Dropout tuning (F2.6) jika arsitektur stabil
- [ ] Tag `v0.4.0-baseline` setelah stabil

---

# Bagian 5 — Artefak & Path Referensi

## Laporan & Dokumen

| Dokumen | Path | Isi |
|---------|------|-----|
| Laporan v0.3.0-baseline | `3DCNN/result_docs/20260517_060023/GeoAtt_PointNet_Palm_Recognition_Evaluation_Report_v2.md` | Hasil ArcFace + komparasi dengan Triplet |
| Laporan diagnostik Fase 1 | `3DCNN/result_docs/20260517_064046/diagnostic_phase1.md` | 7 hipotesis + audit D1–D5 |
| Plan v0.4.0 | `3DCNN/IMPROVEMENT_PLAN_v0.4.0.md` | Rencana lengkap fair ablation |
| QC v2 Report | `3DCNN/eval_results/qc_v2/20260517_080129/qc_report.md` | 35 sesi flagged dengan justifikasi numerik |
| Komparasi v0.3.0 | `3DCNN/eval_results/compare/20260516_225103/comparison_report.md` | Statistik inferensial no_geom vs with_geom |

## Checkpoints & Runs

| Versi | with_geom | no_geom |
|-------|-----------|---------|
| v0.3.0-baseline | `runs/with_geom/20260516_210959/` | `runs/no_geom/20260516_211407/` |
| Evaluasi | `eval_results/with_geom/20260516_223830/` | `eval_results/no_geom/20260516_223800/` |

## Output Audit

| Audit | Path |
|-------|------|
| D1 init parity (pre-patch) | `eval_results/audits/20260517_062539/init_parity.json` |
| D1 init parity (post-patch) | `eval_results/audits/20260517_063518/init_parity.json` |
| D2 geom variance | `eval_results/audits/20260517_062722/` |
| D3 embedding stats | `eval_results/audits/20260517_063240/embedding_stats.json` |
| D5 hard probes | `eval_results/audits/20260517_063331/hard_probes.json` |

---

# Bagian 6 — Ringkasan Keputusan yang Perlu Diambil

1. **Scope Fase 2:** 4 varian (recommended, ~8 jam A100) atau 2 varian saja (fallback, ~4 jam)?
2. **Dataset training:** `dataset_full` (semua sesi) atau `dataset_qc2` (post-QC, 16.5% sesi di-drop)?
   - Rekomendasi plan: training di `dataset_qc2`, eval paralel (`dataset_qc2` vs `dataset_full`)
3. **Evaluasi paralel:**
   - **Eval-clean:** training & test di `dataset_qc2` → "GeoAtt bekerja di data ideal?"
   - **Eval-realistic:** training di `dataset_qc2`, test di `dataset_full` → "GeoAtt bekerja saat menghadapi data realistis?"

---

# Bagian 7 — Hypothesis Log

> Catatan: jangan hapus hipotesis yang sudah diuji — tambahkan baris baru. Tujuannya supaya kita tidak menguji hal yang sama dua kali.

| ID | Hipotesis | Versi Diuji | Metode | Verdict | Catatan |
|----|-----------|-------------|--------|---------|---------|
| H-001 | GeoAtt meningkatkan akurasi dan mengurangi varians (Triplet) | v0.2.0 | Ablasi with_geom vs no_geom | **DITOLAK** (p=1.0) | Efek +4.4 ppt tidak signifikan |
| H-002 | ArcFace mengatasi stagnasi Triplet | v0.3.0 | Ganti loss, keep arsitektur | **TERKONFIRMASI** | +36–44 ppt Rank-1 |
| H-003 | GeoAtt memperkuat margin ArcFace | v0.3.0 | Ablasi with_geom vs no_geom (ArcFace) | **DITOLAK** | with_geom malah turun 4 ppt |
| H-004 | Saturasi ceiling — GeoAtt bantu di probe sulit | v0.4.0 F1 | Hard-probe analysis | **DITOLAK** | Gap membesar di hard subset |
| H-005 | Fitur geom mayoritas noise | v0.4.0 F1 | FDR analysis per fitur | **DITOLAK** | Median FDR=3.77, 0/14 FDR<1 |
| H-006 | Kesalahan with_geom sistematis pada nola | v0.4.0 F1 | CV per-subjek per fitur | **TERKONFIRMASI** | finger_width_5 CV=0.497 |
| H-007 | RNG init parity rusak | v0.4.0 F1 | audit_init_parity.py | **TERKONFIRMASI** | 13/33 layer beda, max|Δ|=0.65 |
| H-008 | Dropout asimetri memperburuk with_geom | v0.4.0 F1 | audit_embedding_stats.py | **TERKONFIRMASI ringan** | cos gap 1.84× |
| H-009 | Z-score menghapus skala absolut → merugikan | v0.4.0 F1 | FDR analysis + inspeksi | **TIDAK RELEVAN** | Separabilitas tetap baik |
| H-010 | Init fair memperkecil gap with_geom vs no_geom | v0.4.0 F2 *(planned)* | Training 4 varian from scratch | **PENDING** | Menunggu eksekusi |
| H-011 | QC v2 (filter noisy session) memperbaiki with_geom | v0.4.0 F2 *(planned)* | Training di dataset_qc2 | **PENDING** | Menunggu eksekusi |

---

# Bagian 8 — Decision Log

> Setiap keputusan penting yang mempengaruhi arah riset dicatat di sini. Kenapa ambil keputusan X? Supaya 2 minggu lagi kita masih ingat.

| Tanggal | Keputusan | Alternatif | Alasan | Dampak | Dibuat oleh |
|---------|-----------|------------|--------|--------|-------------|
| 2026-05-16 | Ganti Triplet → ArcFace | Pertahankan Triplet + tuning | Triplet stagnan ~0.73 sejak epoch 5 | +36–44 ppt Rank-1 | User |
| 2026-05-16 | Multi-prototype enrollment (k=3) | Mean 1 sesi (naif) | Rank-5 92% vs Rank-1 60% → underrepresent | Rank-1 naik signifikan | User |
| 2026-05-17 | PLY Direct ditunda | Implementasi PLY Direct Minimal | Bottleneck terbukti bukan augmentasi | Plan dialihkan ke v0.4.0 | User |
| 2026-05-17 | Patch RNG init parity (F1.1) | Biarkan, anggap gap intrinsic | Ablasi tidak fair → tidak bisa simpulkan | Semua varian sekarang init identik | User + AI |
| 2026-05-17 | 4 varian ablasi (no_geom/with_geom/gam_only/fuse_only) | 2 varian saja | Perlu decompose GAM vs fusion | Bisa eksekusi decision branch B1–B4 | User + AI |
| 2026-05-17 | QC v2 universal objektif (k=10) | Cherry-pick nola saja | Cherry-pick tidak kredibel di tesis | 35 sesi flagged (16.5%), reversibel | User |
| 2026-05-17 | Training v0.4.0 from scratch | Fine-tune dari v0.3.0 | Init berubah + 2 varian baru tidak punya sumber | Lebih lama tapi klaim fair valid | Plan v0.4.0 |
| 2026-05-17 | QC v3 frame-level exclusion | QC v2 session-level (35 sesi = 16.5%) | User: "kalau hanya sebagian frame bermasalah, mungkin user tidak sengaja bergerak" | Hanya 8% frame excluded vs 16.5% sesi | User |
| — | *(tambahkan keputusan baru di sini)* | — | — | — | — |

---

# Bagian 9 — Known Issues & Workarounds

> Masalah yang sudah dikenali tapi belum di-fix. Kalau ada yang mau di-fix, pindahkan ke checklist versi aktif.

| ID | Masalah | Lokasi | Severity | Workaround Saat Ini | Fix Ideal | Status |
|----|---------|--------|----------|---------------------|-----------|--------|
| KI-001 | `finger_width_5` nola sangat tidak stabil antar-sesi (CV=0.497) | `extract_geometry.py` | **High** | QC v2 flag sesi outlier | Fix landmark pinky di pose ekstrem | Open |
| KI-002 | Skala absolut 2× lebih besar dari realitas (finger_len ~208mm vs ~85mm) | `extract_geometry.py` / ICP | Low | Tidak mempengaruhi identifikasi | Cek formula konversi mm | Open |
| KI-003 | `palm_depth_std` FDR terendah (1.71) → fitur paling lemah | `geometry.json` | Low | Masih dipakai (FDR>1) | Drop atau ganti fitur | Open |
| KI-004 | Dropout(0.3) hanya di fusion head → asimetri train/eval | `models/encoder.py` | Medium | — | Turunkan ke 0.1 atau simetris ke geom_encoder | Planned (F2.6) |
| KI-005 | Knuckle detection rapuh → 5 sesi FAIL QC di v0.2.0 | `extract_geometry.py` | Medium | `dataset_manifest.json` skip FAIL | Fallback histogram X-width | Open |
| KI-006 | Checkpoint v0.3.0 tidak bisa di-load langsung ke v0.4.0 | `evaluate.py` | Low | Backward-compat loader otomatis rename | — | **Fixed** (F1.3) |
| KI-007 | reysa punya folder `_QC2_*` dan `_QUARANTINE_*` yang belum dibersihkan | `dataset/reysa/` | Low | — | Bersihkan folder quarantine lama | **Fixed** (dataset.py skip auto) |
| — | *(tambahkan issue baru di sini)* | — | — | — | — | — |

---

# Bagian 10 — Aborted / Paused Experiments

> Eksperimen yang pernah direncanakan tapi batal atau ditunda. Kalau mau dilanjutkan, pindahkan ke checklist versi aktif.

| ID | Eksperimen | Versio | Alasan Batal / Ditunda | Kapan Bisa Dilanjutkan |
|----|------------|--------|------------------------|------------------------|
| AE-001 | PLY Direct On-the-Fly Augmentation | v0.3.0 E3 | Bottleneck terbukti bukan augmentasi | Kalau v0.4.0 menunjukkan butuh lebih banyak variasi data |
| AE-002 | Normals Ablation (in_channels=6) | v0.3.0 §7 | Prioritas lebih rendah dari ArcFace + enrollment | Kalau v0.4.0 stabil dan masih punya waktu |
| AE-003 | Pre-Augmentasi Offline (20× disk space) | v0.3.0 §9 | PLY Direct ditunda → pre-aug ikut ditunda | Kalau PLY Direct dihidupkan kembali |
| AE-004 | CosFace / SubCenter ArcFace | v0.3.0 | Belum dieksplorasi karena ArcFace sudah cukup baik | Kalah v0.4.0 ArcFace masih kurang optimal |
| AE-005 | 3DRegistration fix skala absolut (2× bug) | v0.2.0 | Tidak mempengaruhi identifikasi (semua sesi sama-sama 2×) | Sebelum publikasi angka absolut |
| — | *(tambahkan eksperimen batal/ditunda di sini)* | — | — | — |

---

# Bagian 11 — Reproducibility Checklist (Pre-Run)

> Copy-paste checklist ini sebelum setiap training run. Ini untuk memastikan tidak ada yang terlewat.

- [ ] Seed PyTorch + NumPy + Python `random` sudah di-set (dan tercatat)
- [ ] `split_seed` tercatat (default: 42)
- [ ] `torch.backends.cudnn.deterministic=True` dan `benchmark=False`
- [ ] GPU yang dipakai tercatat (CUDA_VISIBLE_DEVICES)
- [ ] Versi PyTorch + CUDA tercatat (`torch.__version__`)
- [ ] `train.py` flag sesuai varian (`--use-geom`, `--use-gam`, `--use-geom-fusion`, atau tidak ada flag)
- [ ] `DATASET_MODE` sesuai (`'npy'` atau `'ply'` — kalau PLY Direct dihidupkan)
- [ ] `dataset_qc2` vs `dataset_full` sudah diputuskan dan tercatat
- [ ] `EXPERIMENT_TIMESTAMP` diisi dengan format `YYYYMMDD_HHMMSS` (jangan overwrite run sebelumnya)
- [ ] `normalizer.json` hasil fit dari training split saja (hindari leakage)
- [ ] `config.json` tersimpan di folder run
- [ ] `seeds.json` tersimpan (list seed yang dijalankan)
- [ ] Checkpoint `best_rank1.pth` dan `best_loss.pth` disimpan
- [ ] TensorBoard / wandb log aktif (kalau dipakai)
- [ ] Folder `eval_results/` siap dengan struktur `<varian>/<timestamp>/`
- [ ] `test_fingerprint` tercatat untuk verifikasi identitas test split antar run

---

# Bagian 12 — Data Manifest

> Snapshot dataset terkini. Update kalau ada perubahan (scan baru, QC, dsb).

| Parameter | Nilai | Catatan |
|-----------|-------|---------|
| Jumlah subjek | 11 | aisah, alji, chrys, fadhil, feby, gede, nola, rahmat, reysa, taufik, yanuar |
| Total sesi | 212 | Dari semua subjek |
| Total frame (pre-QC) | 2,120 | 10 frame/sesi |
| Total frame PLY | 2,131 | `output.ply` per frame |
| Total `cnn_input.npy` (pre-QC) | 2,036 | 95 frame gagal preprocessing |
| **Total frame valid (post-QC v3)** | **1,869** | Scanner `scan_dataset_frames` verified |
| Frame excluded QC v3 | 160 | `_QC2_frame_XX` prefix |
| Entire session excluded QC v3 | 1 | `yanuar/20260513_092145` (60% outlier) |
| Frame per sesi | 10 | Layout frame-level |
| Sesi FAIL (knuckle detection) | 5 | Dari laporan v0.2.0 |
| Sesi WARN | 1 | `gede/20260415_101438` |
| Subjek dengan frame excluded terbanyak | reysa (27), feby (24), chrys (22) | reysa=high variance, feby=bimodal, chrys=knuckle fallback |
| Subjek paling bersih | rahmat (3), alji (7), fadhil (7) | — |

---

# Bagian 13 — Hardware / Environment Log

> Catat setiap kali training dijalankan. Ini untuk debug kalau hasil tidak konsisten.

| Tanggal | Environment | GPU | PyTorch | CUDA | Batch | n_points | Catatan |
|---------|-------------|-----|---------|------|-------|----------|---------|
| 2026-05-16 | Google Colab | A100 | 2.6.0+cu124 | 12.4 | 512 | 8192 | v0.3.0-baseline ArcFace |
| 2026-05-17 | Local macOS | M3 (CPU) | 2.6.0 | — | — | — | Diagnostik + patch kode |
| — | *(tambahkan row baru per run)* | — | — | — | — | — | — |

---

# Bagian 14 — Future Work / Backlog

> Ide-ide yang belum sempat dieksekusi. Prioritas bisa berubah. Kalau mulai dikerjakan, pindahkan ke checklist versi aktif.

| ID | Ide | Motivasi | Estimasi Waktu | Prioritas | Tergantung Pada |
|----|-----|----------|----------------|-----------|-----------------|
| FW-001 | Cross-Attention GAM (F2.3) | GAM saat ini sigmoid scale/shift sederhana; cross-attention sejati bisa menolak noise per-titik | 2–3 hari | **High** | Hasil B2 (gam_only ≪ no_geom) |
| FW-002 | Gated / FiLM Fusion (F2.4) | Alternatif concat: modulasi γ(geom)×feat + β(geom), ukuran fitur tetap 256 | 2–3 hari | **High** | Hasil B3 (fuse_only ≪ no_geom) |
| FW-003 | Auxiliary Loss dari geom_emb | Predict identity langsung dari geom_emb tanpa concat ke main embedding | 1–2 hari | Medium | Hasil B4 (keduanya ≪ no_geom) |
| FW-004 | Dropout tuning (F2.6) | Turunkan Dropout(0.3) → 0.1 atau simetris ke geom_encoder | 0.5 hari | Low | Setelah arsitektur stabil |
| FW-005 | PLY Direct On-the-Fly | Load PLY asli + augment original-space + PCA-align per epoch | 2–3 hari | Low | Kalau butuh variasi data lebih banyak |
| FW-006 | Normals Ablation (in_channels=6) | Toggle `in_channels=6` di model, pakai `cnn_input.npy` (N,6) | 0.5–1 hari | Low | Quick win kalau masih ada waktu |
| FW-007 | Dataset expansion (≥20 subjek) | Headroom evaluasi tipis pada 11 subjek; klaim GeoAtt butuh dataset lebih besar | 2–4 minggu | **High** | Tidak tergantung versi |
| FW-008 | Cross-device generalization | Uji model di scan dari device lain (iPhone berbeda / Android TOF) | 1–2 minggu | Medium | Dataset expansion |
| FW-009 | Fix skala absolut 2× di `extract_geometry.py` | Angka finger_length ~208mm tidak realistis | 0.5 hari | Low | Sebelum publikasi |
| FW-010 | Fix knuckle detection (fallback histogram) | 5 sesi FAIL karena knuckle tidak terdeteksi | 1 hari | Medium | Sebelum dataset expansion |
| FW-011 | Test-Time Augmentation (TTA) | Average embedding dari 5 augmentasi saat inference | 0.5 hari | Low | Kalah masih butuh marginal gain |
| — | *(tambahkan ide baru di sini)* | — | — | — | — |

---

# Bagian 15 — Template v0.5.0+

> Copy template ini untuk versi berikutnya. Isi saat versi tersebut mulai direncanakan.

```markdown
## v0.X.0: [Judul Versi]

### Konteks & Tujuan
- **Pemicu:** [Kenapa versi ini dibuat?]
- **Hypothesis:** [Apa yang mau dibuktikan?]

### Perubahan dari v0.(X-1).0
- [ ] 
- [ ] 

### Checklist v0.X.0
- [ ] 
- [ ] 

### Hasil Utama
| Metrik | with_geom | no_geom | Δ | Signifikansi |
|--------|-----------|---------|---|--------------|
| Rank-1 | | | | |
| EER | | | | |

### Temuan Kritis
1. 
2. 

### Keputusan yang Diambil
| Tanggal | Keputusan | Alasan |
|---------|-----------|--------|
| | | |
```

---

# Bagian 16 — Glossary

| Istilah | Definisi |
|---------|----------|
| **ArcFace** | Loss function berbasis angular margin (m=0.5, s=30) untuk face/palm recognition |
| **Bootstrap CI** | Confidence interval dari resampling data untuk estimasi variabilitas metrik |
| **CMC** | Cumulative Match Characteristic — kurva Rank-k vs identification rate |
| **EER** | Equal Error Rate — titik di mana FAR = FRR |
| **FDR** | Fisher Discriminant Ratio — between-variance / within-variance per fitur |
| **FiLM** | Feature-wise Linear Modulation — γ(geom)×feat + β(geom) |
| **FRR/FAR** | False Rejection Rate / False Acceptance Rate |
| **GAM** | Geometric Attention Module — modulasi per-point berbasis fitur geometri |
| **Hard probe** | Probe dengan similarity terendah ke ground-truth (bottom 25%) |
| **Holdout** | Sesi yang sama sekali tidak dilihat saat training (unseen session) |
| **mAP** | mean Average Precision — metrik ranking untuk identifikasi |
| **McNemar** | Uji signifikansi untuk perbandingan pasangan classifier pada data yang sama |
| **Multi-prototype enrollment** | Gallery punya >1 template per subjek (k-means k=3) |
| **no_geom** | Varian model tanpa fitur geometri (PointNet++ vanilla) |
| **PLY Direct** | Load PLY asli + augmentasi original-space + PCA-align on-the-fly |
| **QC v2** | Quality Control versi 2 — filter sesi berdasar within-session std objektif (deprecated) |
| **QC v3** | Quality Control versi 3 — filter frame individual berdasar MAD per-session; >50% outlier → exclude sesi |
| **RNG init parity** | Konsistensi bobot awal antar varian model dengan seed identik |
| **TAR@FAR** | True Acceptance Rate pada False Acceptance Rate tertentu |
| **Triplet (batch-hard)** | Loss function berbasis triplet dengan hardest negative mining |
| **Wilcoxon** | Uji non-parametrik untuk perbandingan paired (n seed kecil) |
| **with_geom** | Varian model dengan GeoAtt penuh (GAM + fusion) |

---

*Dokumen ini adalah "living document" — update setiap kali milestone tercapai, keputusan diambil, atau hipotesis diuji.*
