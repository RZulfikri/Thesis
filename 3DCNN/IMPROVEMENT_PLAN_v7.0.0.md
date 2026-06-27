# IMPROVEMENT PLAN v7.0.0

**Status**: v7.1.0 ✅ SELESAI (Gate-2 PASS) | v7.1.1 ✅ SELESAI (Gate LOLOS, anchor `arcface_m04` 1.14%) | v7.2.0 ▶️ SIAP-IMPLEMENTASI (C1/C2/C3)
**Tanggal**: 2026-05-25 (update 2026-06-05)
**Baseline**: v6.0.0 (low-data, 10 subjek × 15 sesi, PointNet++ Triplet vs ArcFace)
**Konteks**: [LAPORAN_v6_lowdata.md](result_docs/20260525_152213_v6_lowdata/LAPORAN_v6_lowdata.md)

**Struktur rilis:**
- **v7.1.0**: Multi-frame fusion + loss sweep ✅ SELESAI (dataset lama, pra-regenerasi) — [LAPORAN_v7_lowdata.md](result_docs/20260530_080633_v7_lowdata/LAPORAN_v7_lowdata.md)
- **v7.1.1**: Re-run v7.1.0 di **dataset regen** (Section 9b) ✅ SELESAI — Gate LOLOS, anchor `arcface_m04` MF N5M5 = 1.14% ± 1.18% — [LAPORAN_v7_1_1.md](result_docs/20260605_083050_v7_1_1/LAPORAN_v7_1_1.md)
- **v7.2.0**: Representation ablation + training frame strategy (Section 10) ▶️ siap implementasi (C0 = anchor v7.1.1; C1/C2/C3 = 15 run baru)

**Keputusan arsitektur evaluasi (update 2026-05-31):**
- LOSO open-set evaluation **dihapus** dari scope v7.2.0 dan seterusnya
- Fokus: **closed-set identification** — pertanyaan thesis adalah "siapa orang ini?" bukan "apakah orang ini terdaftar?"
- LOSO dengan N=11 subjek menghasilkan FAR@unknown 13–22% yang tidak dapat dikalibrasi dengan baik; tidak representatif untuk klaim deployment
- Metrik primer: Rank-1 accuracy, EER, confusion matrix, CMC curve

---

## 1. Motivasi

v6.0.0 menunjukkan bahwa ArcFace marjinal lebih baik dari Triplet (Δ EER ≈ −0.005 test, −0.008 holdout, d-prime ~2× lebih baik), **tetapi Wilcoxon paired p=1.00 (test) dan p=0.63 (holdout)** — tidak signifikan. Tiga akar masalah:

1. **Efek lantai (floor effect)**: banyak seed mencapai EER=0 pada holdout sehingga uji statistik kehilangan power.
2. **Capture burst**: gap temporal train→holdout maksimum hanya 25 detik (ref. memori 1275). Holdout EER rendah sebagian dijelaskan oleh kedekatan temporal, bukan generalisasi.
3. **N kecil**: 10 subjek tidak cukup untuk mendeteksi Δ EER ≈ 0.01 dengan power yang layak.

v7.0.0 **tidak mengganti arsitektur**. Fokusnya: protokol evaluasi yang lebih ketat + skala dataset + refinement loss yang sudah terbukti unggul (ArcFace).

---

## 2. Tujuan & Hipotesis

**Konteks**: Tidak ada rencana akuisisi dataset tambahan. Dataset tetap **11 subjek × 15–25 sesi × 10 frame/sesi**. Karena itu klaim statistik signifikansi (p < 0.05) **tidak dijadikan tujuan primer** — power tidak akan memadai. Plan ini fokus pada **deployment-oriented improvements** yang valuable terlepas dari signifikansi.

**Tujuan utama**: Memaksimalkan utilisasi dataset existing melalui (a) multi-frame fusion di inference (mirror real deployment), (b) loss refinement yang sudah terbukti unggul pada v6 (ArcFace), (c) protokol evaluasi yang lebih realistis (cross-session, open-set via leave-one-subject-out).

**Hipotesis**:
- **H1**: Multi-frame fusion (enroll & probe = mean embedding dari N frame) akan menurunkan EER **lebih besar** daripada perubahan loss function — bagian "1 detik scan" sudah ada di dataset (10 frame per sesi = 1 capture burst).
- **H2**: ArcFace dengan margin lebih kecil (m=0.3 atau 0.4) akan mengungguli m=0.5 default pada N=11 subjek — margin agresif lebih cocok untuk dataset besar.
- **H3**: Effect size (Cohen's d, rank-biserial) + interval kepercayaan bootstrap akan memberikan klaim yang lebih informatif daripada Wilcoxon p-value pada N kecil.

**Catatan unit statistik**: Frame ekstra **bukan substitusi** untuk skala subjek/sesi. Unit independen tetap sesi; frame berperan sebagai (a) augmentation source saat training, dan (b) **ensemble source saat inference** (ini yang dieksploitasi v7).

---

## 3. Scope

### In-scope
- Aktifkan subjek `gede` (11 subjek total, dari 10 baseline v6).
- Cross-session pair mining (training & evaluasi).
- Open-set evaluation via **leave-one-subject-out** (LOSO) — tidak perlu subjek baru.
- ArcFace margin & scale sweep + perbandingan CosFace / SubCenter-ArcFace.
- Combined loss (ArcFace + Triplet weighted) — infra sudah ada.
- Mix-frame augmentation (saat ini hanya median frame).
- **Multi-frame fusion di inference (enroll & probe), ablation N frame, latency budget** — fokus utama v7.
- Bootstrap CI + effect size pelaporan (pengganti uji signifikansi).

### Out-of-scope
- **Akuisisi dataset tambahan** (subjek atau sesi baru).
- Holdout dengan gap temporal ≥ 7 hari (mustahil dengan capture burst).
- Klaim p < 0.05 sebagai primary metric.
- Ganti backbone (PointNet++ sudah cukup pada skala ini — memori 1295).
- GAM / attention modules (memori 1296: pola memorisasi val v5).
- Quantization, distillation, mobile deployment.

---

## 4. Jalur Kerja

### Jalur A — Protokol Evaluasi (tanpa akuisisi data)

| # | Task | Acceptance Criteria |
|---|------|---------------------|
| A1 | Aktifkan subjek `gede` (sudah ada di dataset, di-skip v6 karena threshold) | Pipeline lowdata menerima 11 subjek; semua memiliki ≥ 15 sesi |
| A2 | ~~Holdout gap ≥ 7 hari~~ — **dropped** (mustahil tanpa akuisisi). Gantikan dengan: laporkan eksplisit distribusi gap train→holdout di setiap eksperimen | `temporal_gap_audit.json` dilampirkan ke setiap result_docs |
| A3 | Cross-session pair generation di evaluasi (probe & gallery dari sesi berbeda) | Tidak ada pair dengan `session_id_probe == session_id_gallery` |
| A4 | Open-set evaluation via **leave-one-subject-out** (LOSO): tiap fold pakai 10 subjek sebagai known, 1 sebagai unknown | 11 fold; lapor FAR@unknown, FRR@FAR=1% (rata-rata + std antar fold) |

### Jalur B — Loss Refinement

**8 varian eksplisit yang diuji di v7.1.0:**

| # | Nama varian | Loss function | Parameter kunci |
|---|-------------|---------------|-----------------|
| 1 | `standard` | Triplet batch-hard | margin=0.3 |
| 2 | `arcface_m03` | ArcFace | margin=0.3, scale=30 |
| 3 | `arcface_m04` | ArcFace | margin=0.4, scale=30 |
| 4 | `arcface_m05` | ArcFace | margin=0.5, scale=30 (default v6) |
| 5 | `arcface_s64` | ArcFace | margin=0.5, scale=64 |
| 6 | `cosface` | CosFace | margin=0.35, scale=30 |
| 7 | `subcenter` | SubCenter-ArcFace | K=3, margin=0.5, scale=30 |
| 8 | `hybrid` | ArcFace + Triplet | α=0.5 |

**Task breakdown:**

| # | Task | Acceptance Criteria |
|---|------|---------------------|
| B1 | ArcFace margin sweep: m ∈ {0.3, 0.4, 0.5}, s ∈ {30, 64} → varian 2–5 | 4 konfigurasi × 10 seed; aggregate report |
| B2 | CosFace (m=0.35, s=30) → varian 6 | Eval lengkap dengan protokol Jalur A |
| B3 | SubCenter-ArcFace (K=3) → varian 7 | Eval lengkap; uji ketahanan label noise |
| B4 | Combined loss: `L = α·ArcFace + (1−α)·Triplet`, α=0.5 → varian 8 | Eval lengkap |

### Jalur C — Augmentasi & Robustness

| # | Task | Acceptance Criteria |
|---|------|---------------------|
| C1 | Mix-frame augmentation: random frame per sesi setiap epoch (bukan median statis) | `PalmFrameDataset` mendukung `frame_sampling="random"`; ablation vs median |
| C2 | Cross-session triplet mining: enforce anchor & positive dari sesi berbeda | Implementasi di `losses/`; ablation report |
| C3 | *Deprioritized* — TTA sintetis (rotasi/translation kecil). Digantikan oleh Jalur D (multi-frame fusion asli, lebih realistis) | — |

### Jalur D — Multi-frame Fusion (Real-world Deployment Protocol)

Skenario deployment: scan 1 detik ≈ 10–30 frame per attempt. v6 hanya memakai 1 median frame, **under-utilizes capture**. Jalur D menjadikan multi-frame fusion sebagai protokol evaluasi primer.

| # | Task | Acceptance Criteria |
|---|------|---------------------|
| D1 | Multi-frame enrollment: gallery embedding = mean/median dari N frame di sesi enroll | `evaluate.py` mendukung `--enroll-frames N`; default N=5 |
| D2 | Multi-frame probe: probe embedding = mean dari M frame di scan attempt | `evaluate.py` mendukung `--probe-frames M`; default M=5 |
| D3 | Ablation matrix: (N, M) ∈ {1, 3, 5, 10} × {1, 3, 5, 10} → 16 konfigurasi | Heatmap EER vs (N, M); identifikasi titik diminishing returns |
| D4 | Fusion strategy ablation: mean vs median vs max-pool vs attention-weighted | Lapor EER & d-prime per strategi |
| D5 | Latency budget: ukur end-to-end inference time per probe (per N, M) | Klaim "1 detik scan" tervalidasi; tabel latency vs akurasi |
| D6 | Protokol primer v7 = (N=5, M=5) untuk perbandingan loss di Jalur B | Semua eksperimen Jalur B dilaporkan dengan **dua** protokol: single-frame (kompat. v6) dan multi-frame (5,5) |

---

## 5. Quality Gates

Eksperimen v7.0.0 **tidak boleh** masuk ke Jalur B/C sebelum Gate-1 lulus.

- **Gate-1 — Protocol readiness**:
  - 11 subjek aktif (termasuk `gede`).
  - Cross-session pair generation aktif.
  - Open-set evaluation LOSO terimplementasi & terverifikasi.
  - Multi-frame fusion (N=5, M=5) wired ke pipeline lowdata.

- **Gate-2 — Deployment claim (bukan signifikansi)**:
  - Multi-frame fusion (5,5) menurunkan EER **dengan effect size Cohen's d ≥ 0.5** dibanding single-frame baseline pada minimal satu varian loss.
  - Bootstrap 95% CI dilaporkan untuk EER, FRR@FAR=1%, d-prime di semua varian final.
  - Delta EER single-frame → multi-frame dilaporkan untuk validasi H1.
  - Wilcoxon p-value tetap dilaporkan **sebagai informasi sekunder**, tanpa klaim "signifikan".
  - Latency end-to-end ≤ 1 detik untuk konfigurasi rekomendasi (validasi klaim deployment).

- **Gate-3 — Reproducibility**:
  - 10 seeds lengkap untuk semua varian final.
  - Notebook self-contained (memori `feedback_style`); Indonesia untuk output/komentar.
  - Tag git `v7.0.0` dan laporan di `result_docs/<timestamp>_v7_*/`.

---

## 5b. Kontribusi v7.1.0 ke Thesis

v7.1.0 menjawab tiga pertanyaan penelitian yang berkontribusi langsung ke argumen novelty thesis:

**Kontribusi 1 — Deployment-realistic evaluation protocol**
> "Kami mengusulkan protokol evaluasi multi-frame fusion (N enroll × M probe) untuk 3D palm recognition berbasis consumer depth sensor, yang mencerminkan skenario deployment nyata di mana pengguna melakukan scan 1 detik (≈10 frame). Protokol ini menunjukkan peningkatan EER dengan effect size Cohen's d ≥ [X] dibandingkan single-frame baseline yang umum digunakan di literatur."

**Kontribusi 2 — Loss function analysis untuk small-scale 3D biometrics**
> "Kami melakukan perbandingan komprehensif 8 varian loss function (Triplet, ArcFace dengan 4 konfigurasi margin/scale, CosFace, SubCenter-ArcFace, dan hybrid) pada dataset 3D palm dengan 11 subjek, menggunakan effect size dan bootstrap CI sebagai metrik primer — pendekatan yang lebih tepat daripada p-value pada skala dataset kecil."

**Kontribusi 3 — Open-set evaluation via LOSO**
> "Kami mengevaluasi kemampuan penolakan subjek unknown menggunakan leave-one-subject-out (11 fold), menunjukkan bahwa sistem tidak hanya bekerja pada closed-set tetapi juga mampu menolak orang asing dengan FAR@unknown = [Y]%."

**Bagaimana ketiga kontribusi terhubung ke v7.2.0:**
- Loss function terbaik dari kontribusi 2 → dipakai di v7.2.0 representation ablation
- Protokol multi-frame dari kontribusi 1 → dipakai sebagai metrik evaluasi v7.2.0
- Gabungan v7.1.0 + v7.2.0 → argumen end-to-end: preprocessing + loss + evaluation protocol

---

## 6. Deliverables

1. **Dataset**: snapshot terverifikasi (Gate-0).
2. **Kode**:
   - `train.py`: dukungan CosFace, SubCenter-ArcFace, combined loss, mix-frame sampling, cross-session triplet mining.
   - `evaluate.py`: open-set protocol, cross-session pair enforcement, **multi-frame enrollment & probe (`--enroll-frames`, `--probe-frames`, `--fusion-strategy`), latency profiling**.
   - `utils/`: power analysis script & temporal gap audit yang sudah dipakai di v6.
3. **Notebook**: `collab/v7_*_train_eval.ipynb` dan `collab/v7_compare_analyze.ipynb`.
4. **Laporan**: `result_docs/<timestamp>_v7/LAPORAN_v7.md` dengan tabel agregat, uji Wilcoxon, dan keputusan Gate-2.
5. **Tag git**: `v7.0.0` setelah Gate-3 lulus.

---

## 7. Timeline Indikatif (tanpa akuisisi data)

| Hari | Aktivitas |
|------|-----------|
| 1   | A1 (aktifkan gede) + A3 (cross-session pair) + Gate-1 awal |
| 2   | D1–D2 (multi-frame wiring; infra ada di `utils/enrollment.py`) |
| 3–4 | A4 (open-set LOSO) + verifikasi Gate-1 |
| 5–6 | B1 (ArcFace sweep 6 config) pada protokol (5,5) |
| 7   | B2 (CosFace) implementasi + eval |
| 8   | B3 (SubCenter-ArcFace) implementasi + eval |
| 9   | B4 (combined loss, infra siap) + C1 (mix-frame) |
| 10  | C2 (cross-session triplet mining) |
| 11  | D3 (ablation N×M = 16 config) — inference only |
| 12  | D4 (fusion strategy) + D5 (latency profiling) |
| 13–14 | Bootstrap CI, effect size, analisis, laporan, Gate-2/Gate-3, tagging |

**Total: ~14 hari kerja** (terkalibrasi dari estimasi Section 9.6).

---

## 8. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Open-set LOSO antar fold sangat bervariasi | FRR/FAR std besar | Lapor per-fold; gunakan median antar fold sebagai point estimate |
| ArcFace margin sweep tidak menunjukkan winner | Klaim H2 gagal | Tetap valuable sebagai null-result; rekomendasi ArcFace m=0.5 default tetap valid |
| Multi-frame fusion tidak menurunkan EER (H1 gagal) | Klaim deployment lemah | Tetap valuable: validasi bahwa median frame v6 sudah cukup; lapor sebagai temuan deployment |
| Latency multi-frame > 1 detik | Klaim "1 detik scan" gagal | Cari (N,M) dengan trade-off terbaik di kurva Pareto akurasi vs latency; lapor bisa < 1 detik di config mana |
| EER terus floor pada banyak seed | Effect size kecil meskipun ada perbaikan | Gunakan FRR@FAR=0.1% dan d-prime sebagai metrik primer (lebih sensitif daripada EER) |

---

## 9. Status Aktual Project & Kelayakan (audit 2026-05-25)

### 9.1 Dataset aktual

| Aspek | Aktual | Status v7 |
|-------|--------|-----------|
| Subjek | **11** (aisah, alji, chrys, fadhil, feby, **gede**, nola, rahmat, reysa, taufik, yanuar) | Final — aktifkan semua 11 (vs 10 di v6) |
| Sesi/subjek | min 15 (alji, fadhil, rahmat), max 25 (reysa), median ~19 | Final — dataset existing |
| Frame/sesi | **10 uniform** (213 sesi 10-frame, 1 anomali) | OK; mendukung ablation N,M ∈ {1,3,5,10} |
| Gap temporal train→holdout | maks ~25 detik (capture burst, memori 1275) | Dilaporkan eksplisit, tidak di-enforce |

**Catatan**: `gede` (20 sesi) tidak dipakai di v6 karena pipeline lowdata mem-filter ke 10 subjek. Bisa diaktifkan di v7 menjadi 11 subjek tanpa akuisisi tambahan.

### 9.2 Kode yang sudah ada (no work needed)

| Komponen | Lokasi | Status |
|----------|--------|--------|
| Multi-frame enrollment (mean/median/kmeans) | [utils/enrollment.py](utils/enrollment.py) | **Siap pakai**; tinggal wiring ke pipeline lowdata |
| Multi-probe untuk holdout | [evaluate.py:173](evaluate.py:173) `--n_probe_frames` (default 3) | **Aktif**; perlu diperluas ke test split |
| ArcFace loss + margin/scale CLI | [losses/arcface.py](losses/arcface.py), [train.py:311](train.py:311) | **Siap** (B1 tinggal sweep) |
| Hybrid ArcFace + Triplet | [losses/arcface.py](losses/arcface.py) `HybridArcFaceTriplet` | **Siap** (B4 selesai infra) |
| Median frame selection | [utils/dataset_lowdata.py:53](utils/dataset_lowdata.py:53) `_pick_median_frame` | Ada; baseline v6 |
| Temporal gap audit | [result_docs/.../temporal_gap_audit.json](result_docs/20260525_053041_v6_arcface/temporal_gap_audit.json) | Ada; tinggal di-enforce di split |

### 9.3 Kode — status implementasi (update 2026-05-28)

| Komponen | Status | File |
|----------|--------|------|
| CosFace loss (B2) | ✅ Selesai | [losses/cosface.py](losses/cosface.py) |
| SubCenter-ArcFace (B3) | ✅ Selesai | [losses/subcenter_arcface.py](losses/subcenter_arcface.py) |
| Cross-session triplet mining (C2) | ✅ Selesai | [losses/triplet.py](losses/triplet.py) `CrossSessionTripletLoss` |
| Random frame sampling (C1) | ✅ Selesai | [utils/dataset_lowdata.py](utils/dataset_lowdata.py) `frame_sampling="random"` |
| Multi-frame eval pipeline (D1-D5) | ✅ Selesai | [utils/eval_multiframe.py](utils/eval_multiframe.py) |
| Open-set LOSO evaluation (A4) | ✅ Selesai | [utils/eval_openset.py](utils/eval_openset.py) |
| Session tracking di dataset (C2) | ✅ Selesai | [utils/dataset.py](utils/dataset.py) `session_idx` di batch |
| 11 subjek aktif (A1) | ✅ Selesai | [utils/dataset_lowdata.py](utils/dataset_lowdata.py) `DROPPED_SUBJECTS = set()` |
| CLI flags train.py (B2,B3,C1,C2) | ✅ Selesai | [train.py](train.py) `--loss cosface/subcenter_arcface`, `--cross-session-mining`, `--frame-sampling` |
| Training notebook | ✅ Selesai | [collab/v7_train_eval.ipynb](collab/v7_train_eval.ipynb) |
| Analysis notebook | ✅ Selesai | [collab/v7_multiframe_compare.ipynb](collab/v7_multiframe_compare.ipynb) |

**Next step**: run notebook di Colab (training 8 varian × N seed → analisis → Gate-2).

### 9.4 Kelayakan & Strategi Realistis

**Yang feasible TANPA akuisisi data baru:**
- Aktifkan 11 subjek (termasuk `gede`) — sudah +1 dari baseline v6.
- Multi-frame fusion N,M ∈ {1,3,5,10} — 10 frame per sesi mencukupi.
- ArcFace sweep, CosFace, SubCenter, combined loss (Jalur B lengkap).
- Mix-frame augmentation (Jalur C1) — utilisasi 10 frame yang sudah ada.
- Cross-session triplet mining (Jalur C2).

**Yang TIDAK feasible tanpa akuisisi tambahan:**
- Subjek ≥ 25 (target Gate-0 awal).
- Gap temporal train→holdout ≥ 7 hari.
- Power analysis untuk deteksi Δ EER ≈ 0.01 dengan p < 0.05 — dengan 11 subjek tetap underpowered.

### 9.5 Keputusan: tanpa akuisisi data

**Konfirmasi user (2026-05-25)**: tidak ada rencana akuisisi dataset tambahan untuk v7. Konsekuensinya:

- Klaim p < 0.05 **drop** dari tujuan primer.
- Fokus geser sepenuhnya ke **deployment-oriented metrics**: multi-frame fusion efficacy, latency, effect size, bootstrap CI.
- 11 subjek (termasuk `gede` yang sebelumnya di-skip v6) menjadi konfigurasi final.
- Open-set evaluation menggunakan **LOSO** (leave-one-subject-out) — eksploitasi 11 subjek tanpa butuh subjek baru.
- Cross-session protocol tetap dilaksanakan (probe & gallery sesi berbeda).
- Gap temporal ≥ 7 hari **dihapus** dari requirement; gantinya gap distribution dilaporkan eksplisit di tiap result.

### 9.6 Estimasi effort per jalur (dataset aktual)

| Jalur | Implementasi | Eksekusi (10 seeds) | Total |
|-------|--------------|---------------------|-------|
| A1 (aktifkan gede) | ~1 jam | — | 1 jam |
| A3 (cross-session pair) | ~1 hari | — | 1 hari |
| A4 (open-set, butuh subjek unknown — gunakan leave-one-subject-out) | ~2 hari | +20% runtime | 3 hari |
| B1 (ArcFace sweep 6 config) | <1 jam | ~6× runtime v6 | 1–2 hari |
| B2 (CosFace) | ~1 hari | 1× runtime | 1–2 hari |
| B3 (SubCenter) | ~1 hari | 1× runtime | 1–2 hari |
| B4 (combined, infra siap) | <1 jam | 1× runtime | 1 hari |
| C1 (mix-frame) | ~0.5 hari | 1× runtime | 1 hari |
| C2 (cross-session triplet mining) | ~1 hari | 1× runtime | 2 hari |
| D1–D2 (multi-frame wiring, infra siap) | ~0.5 hari | — | 0.5 hari |
| D3 (ablation N×M = 16 config) | <1 jam | inference only | 0.5 hari |
| D4 (fusion strategy) | <1 jam | inference only | 0.5 hari |
| D5 (latency) | ~0.5 hari | — | 0.5 hari |

**Total tanpa akuisisi**: ~15 hari kerja (Gate-0a path).

---

## 9b. v7.1.1 — Re-run v7.1.0 di Dataset Regen (anchor untuk v7.2.0)

**Status**: 🔜 SIAP-RUN (2026-06-03)
**Tujuan**: mengulang loss sweep + multi-frame fusion v7.1.0 di **dataset regenerasi** agar
seluruh rangkaian eksperimen (loss → representasi) berada di **satu dataset yang sama**,
menutup celah metodologis "v7.1.0 di data lama, v7.2.0 di data regen". Hasil v7.1.1 menjadi
**acuan resmi** (winner + angka baseline) untuk v7.2.0.

### 9b.1 Kenapa perlu re-run (bukan reuse v7.1.0)
Validasi pasca-regen menemukan ~16% frame berbeda dari dataset lama (ambiguitas kanonikalisasi
PCA — lihat §10.10), plus churn komposisi frame (295 frame baru, 536 terhapus). Angka absolut
v7.1.0 (EER 1.32% dst.) adalah **pra-regenerasi** dan tidak boleh dicampur langsung dengan
v7.2.0. Ranking relatif antar-loss diharapkan **tetap** (transformasi per-frame konsisten ke
semua varian), tapi re-run memberi anchor apple-to-apple + konfirmasi.

### 9b.2 Spec (delta dari v7.1.0 — hanya 3 perubahan)
| Aspek | v7.1.0 | v7.1.1 |
|---|---|---|
| Dataset | lama (pra-regen) | **regen** (`3DCNN/dataset/`, post-mirror) |
| Seeds | 10 (`0,1,2,3,4,7,42,123,2024,31337`) | **5** (`0, 42, 123, 2024, 31337`) |
| Evaluasi | Test + Holdout + **LOSO open-set** | **Test + Holdout saja** (LOSO dibuang) |
| Loss varian | 8 | **8** (sama — faithful re-run, scope A) |
| Training frames | median per sesi | median per sesi (sama) |
| Protokol primer | MF N=5, M=5 | MF N=5, M=5 (sama) |

- **Seed dikurangi → 5**: standar umum (3–5 seed). Karena tujuan = konfirmasi ranking + anchor
  (bukan klaim signifikansi baru), 5 seed cukup. Lapor **mean±std + effect size**, bukan p-value.
- **LOSO dibuang**: LOSO = evaluasi open-set/stranger-rejection (pertanyaan riset berbeda).
  Scope thesis = **closed-set identification** → Test + Holdout adalah protokol yang benar.
  Konsisten dengan keputusan arsitektur evaluasi di header.
- **Seed sama dipakai juga di v7.2.0** agar C0 (anchor) sebanding lurus.

### 9b.3 Ringan untuk Colab
v7.1.1 hanya butuh **R2** (`cnn_input.npy` + `geometry.json`) — TIDAK butuh `output.ply` (R1)
atau `cnn_input_fps.npy` (R3). Maka keputusan repo berat (fps ~418 MB + PLY ter-gitignore)
**ditunda ke v7.2.0**; untuk v7.1.1 cukup commit perubahan `cnn_input.npy`+`geometry.json`
(sudah git-tracked) ke branch `colab`, push, jalankan notebook.

### 9b.4 Langkah eksekusi
| # | Task | Status |
|---|---|---|
| F0 | Commit `cnn_input.npy`+`geometry.json` regen ke branch `colab`, push | pending |
| F1 | `collab/v7_train_eval.ipynb`: set seed list = 5, **matikan blok LOSO** | pending |
| F2 | Jalankan training 8 varian × 5 seed di Colab (~40 run) | pending (user) |
| F3 | Analisa: ranking loss, EER MF N5M5, SF→MF delta, Test+Holdout | pending |
| F4 | **Gate v7.1.1**: konfirmasi arcface_m04 tetap juara + EER anchor; sanity-check vs v7.1.0 lama (~1.3%) | pending |
| F5 | `LAPORAN_v7.1.1.md` di `result_docs/<timestamp>_v7.1.1/` | pending |

### 9b.5 Gate v7.1.1 → v7.2.0
- **Lolos** bila ranking loss stabil (arcface_m04 di puncak) dan EER MF N5M5 sebanding v7.1.0
  (ballpark ~1–2%). → kunci arcface_m04 + N5M5 sebagai acuan v7.2.0; **C0 v7.2.0 = arcface_m04
  v7.1.1** (tidak perlu run ulang).
- **Gagal** bila winner berubah atau EER meleset jauh (mis. ke ~5%). → dataset regen berdampak
  material; tinjau ulang sebelum lanjut v7.2.0.

---

## 10. v7.2.0 — Representation Ablation

**Status**: ▶️ SIAP-IMPLEMENTASI (E2–E8) — Gate v7.1.1 LOLOS (2026-06-05)
**Prasyarat**: ✅ **Gate v7.1.1 lolos** (§9b.5, [LAPORAN_v7_1_1.md](result_docs/20260605_083050_v7_1_1/LAPORAN_v7_1_1.md)) — anchor `arcface_m04` tereproduksi di dataset regen (MF N5M5 = 1.14% ± 1.18% vs 1.32% v7.1.0); winner numerik bergeser ke `cosface` tetapi **tidak signifikan** (dalam noise) → C0 = arcface_m04 v7.1.1 dipertahankan, `cosface` dibawa sebagai pembanding sekunder

### 10.1 Satu Pertanyaan Inti

> *"Seberapa penting preprocessing kanonikalisasi (PCA alignment + unit-sphere normalization) dalam pipeline ini? Apakah raw PLY bisa langsung dipakai? Apakah Pre-FPS yang lebih ringkas mempertahankan performa?"*

Semua variabel lain **dikunci** berdasarkan keputusan dari v7.1.0 dan diskusi desain:

| Variabel | Nilai yang dikunci | Dasar keputusan |
|---|---|---|
| Loss function | **arcface_m04** (m=0.4, s=30) | Terbaik di v7.1.0, MF EER 1.32% |
| Training frames | **Semua 10 frame per sesi** | Kondisi deployment nyata; semua frame tersedia |
| Evaluasi protokol | **Multi-frame N=5, M=5** | Tervalidasi di v7.1.0 sebagai sweet spot |
| Evaluasi jenis | **Closed-set identification** | Fokus thesis: "siapa orang ini?" |
| LOSO | **Dihapus** | 11 subjek tidak cukup untuk kalibrasi threshold |

Satu-satunya yang divariasikan: **representasi input**.

### 10.2 Tiga Representasi yang Diuji

```
iPhone TrueDepth scan
        |
        v  process_single_frames.py
        |
        +-- output.ply          (R1) koordinat kamera asli, ~15K-18K titik
        |
        +-- cnn_input.npy       (R2) PCA-aligned + unit-sphere + normals
        |                            ~15K-18K titik, pose-invariant
        |
        +-- cnn_input_fps.npy   (R3) sama dengan R2 + FPS 8192 titik
                                     sudah fixed 8192, tidak perlu runtime sampling
```

| ID | Nama | File | Preprocessing | Perbedaan vs R2 |
|---|---|---|---|---|
| **R1** | Raw PLY | `output.ply` | Load (xyz+normals) + random sample 8192 titik | Tidak ada kanonikalisasi pose |
| **R2** | Canonical NPY | `cnn_input.npy` | PCA align + unit sphere + random sample 8192 | **Baseline saat ini** |
| **R3** | Pre-FPS NPY | `cnn_input_fps.npy` | PCA align + unit sphere + FPS 8192 (pre-computed) | Sampling spatially uniform, tidak ada runtime sampling |

> **Catatan generasi (update 2026-06-03):** `output.ply` kini menyimpan **normals** (fix `lib/single_frame.py`) sehingga R1 langsung punya 6 channel xyz+normals tanpa re-estimasi — membuat R1 vs R2 mengisolasi tepat kanonikalisasi pose. `cnn_input_fps.npy` (R3) diturunkan **langsung dari `cnn_input.npy` (R2)** via `make_fps.py` (open3d `farthest_point_down_sample`, ~0.4 dtk/frame), BUKAN dari PLY — menjamin R3 = R2 + FPS persis.
>
> **Koreksi (validasi pasca-regen):** klaim awal "regenerasi bit-identik dengan v7.1.0" **TIDAK berlaku**. Validasi menyeluruh menemukan **~291 dari 1836 frame (~16%) BERBEDA** dari dataset lama — bukan karena cleaning (point set IDENTIK, sorted-xyz diff 0.000) melainkan **ambiguitas kanonikalisasi PCA** di `pca_align()`: (a) mayoritas flip 180° terhadap sumbu Y karena sign sumbu X tidak di-kanonikalisasi (langsung dari SVD `Vt`, hanya sumbu Y yang punya disambiguasi median-Y); (b) sebagian perbedaan resolusi axis lebih besar (borderline `range0≈range1` / median-Y flip terbalik). Konsekuensi: **basis/normalizer v7.1.0 dan C0 TIDAK boleh di-reuse** — semua direkomputasi fresh di dataset regen, dan old+new tidak dicampur. `pca_align()` **tidak diubah** (R2 tetap sesuai definisi v7.1.0 yang dikunci); instabilitas X-sign/axis ini dicatat sebagai **sifat/keterbatasan representasi R2 canonical** yang justru relevan dibahas di thesis (lihat §10.10).

**Apa yang diukur dari setiap perbandingan:**
- **R1 vs R2**: apakah kanonikalisasi pose (PCA + unit-sphere) kritis?
- **R2 vs R3**: apakah FPS lebih baik dari random sampling? Dan seberapa besar efisiensi gain-nya?
- **R1 vs R3**: kontribusi full preprocessing pipeline dari raw ke canonical + FPS

### 10.3 Konfigurasi Eksperimen

| ID | Representasi | Loss | Training frames | Keterangan |
|---|---|---|---|---|
| **C0** | R2 Canonical NPY | arcface_m04 | 1 median frame | **= arcface_m04 dari v7.1.1** (anchor) — tidak perlu run terpisah |
| **C1** | R1 Raw PLY | arcface_m04 | semua 10 frame | Uji tanpa kanonikalisasi |
| **C2** | R2 Canonical NPY | arcface_m04 | semua 10 frame | Baseline baru dengan semua frame |
| **C3** | R3 Pre-FPS NPY | arcface_m04 | semua 10 frame | Uji FPS vs random sampling |

- **C0** = arcface_m04 dari v7.1.1 (sudah dijalankan di tahap v7.1.1) → **0 run baru** di v7.2.0
- C1, C2, C3: masing-masing **5 seed** (seed sama dgn v7.1.1) = **15 run baru**
- **Total Colab runtime v7.2.0**: ~15 run x ~10 menit ≈ ~2,5 jam (di luar ~40 run v7.1.1)

**Bonus perbandingan dari C0 vs C2**: langsung mengukur manfaat semua 10 frame (F2) vs 1 median frame (F1) pada representasi dan loss yang sama — C0 (median, dari v7.1.1) vs C2 (semua 10 frame), keduanya R2 + arcface_m04 di dataset regen.

### 10.4 Protokol Evaluasi

- **Split**: Test (s10-s11) dan Holdout (s12-s14), semua 11 subjek enrolled
- **Protokol**: Multi-frame fusion N=5 enrollment frames, M=5 probe frames, strategy=mean
> **Insight pengukuran (penting):** R1/R2/R3 sama-sama memberi **8192 titik ke jaringan yang sama** → **FLOPs forward-pass model IDENTIK**. Maka perbedaan kecepatan **bukan** di model melainkan **murni di data pipeline** (baca disk → parse → sampling → normalisasi). Karena itu kita ukur **load/preprocess time per frame** dan **bottleneck dataloader (GPU starve)**, bukan sekadar forward time. Narasi: R1 vs R2 = pertanyaan **akurasi** (kanonikalisasi); R2 vs R3 = pertanyaan **efisiensi pipeline**.

**Daftar metrik final v7.2.0** (semua tier 🟢 Core + 🟡 Recommended; semua dilaporkan **mean ± std lintas 5 seed**, GPU tipe sama, R1/R2/R3 sebanding). Stretch (CMC Rank-k, kurva degradasi #titik, noise, UMAP, energi) = opsional bila ada waktu.

**A. Akurasi / identifikasi** *(🟢)*
- Rank-1 accuracy (primary) · EER · DET & ROC curve + AUC · d-prime (separabilitas genuine↔impostor) · confusion matrix per representasi · **TAR @ FAR=1% & 0.1%** *(🟡)* · std EER lintas seed (robustness) · EER per-subjek (fairness)

**B. Kecepatan training** *(🟢)*
- Waktu / epoch (dtk) · total waktu sampai konvergen (dtk) · epochs-to-converge · **split I/O vs compute** *(🟡)* (data-load vs GPU compute — bukti bottleneck pipeline) · throughput (frame/dtk)

**C. Kecepatan evaluasi / inference** *(🟢)*
- Latency enrollment / subjek (N=5) · latency probe / identifikasi (M=5) · **preprocess latency / frame** (PLY parse vs np.load vs no-sampling, dipisah dari forward GPU) · end-to-end identification latency · inference throughput (frame/dtk)

**D. Resource / memori** *(🟢)*
- Peak GPU memory (training & inference, MB) · peak host RAM (dataloader) · disk footprint / frame & total dataset · model size + FLOPs (= **kontrol**, identik — laporkan sebagai konstanta)

**E. Probe isolasi (robustness)** *(🟡 — "money plots")*
- **Sensitivitas rotasi/pose**: input diputar acak → ekspektasi R2/R3 invarian, R1 jeblok. **Uji kuantitatif langsung hipotesis kanonikalisasi (H4)** → ΔEER vs sudut rotasi.
- **Determinisme sampling**: forward frame sama berulang → R3 (FPS fixed) varian embedding ≈ 0; R1/R2 (random sample) ada varian. Ukur std embedding antar-ulangan (R2 vs R3).

**F. Sintesis** *(🟢)*
- **Pareto frontier** akurasi × kecepatan × disk — repr mana dominan / didominasi · scatter Rank-1 (atau EER) vs latency probe · effect size (Cliff's δ / Cohen's d) antar repr, **bukan p-value** (daya statistik rendah).

### 10.5 Hipotesis

- **H4**: Kanonikalisasi pose (R2 vs R1) memberikan peningkatan Rank-1 yang lebih besar dari sekadar menambah jumlah training frame — preprocessing adalah komponen paling kritis dalam pipeline.
- **H5**: Pre-FPS (R3) memberikan Rank-1 setara R2 (delta < 1 pp) dengan **training maupun inference lebih cepat** (training throughput lebih tinggi + latency probe lebih rendah) karena tidak ada runtime sampling — dikuatkan dengan pengukuran kuantitatif E9 (§10.6).
- **H6**: Semua 10 frame (C2 vs C0) memberikan peningkatan moderat — ada gain tapi tidak dramatis karena 10 frame dalam satu sesi burst berkorelasi tinggi.
- **H7**: Raw PLY (R1) **paling lambat** di training & evaluasi (PLY parse + Open3D + runtime sampling); bila akurasinya juga tidak unggul, R1 kalah di kedua dimensi (akurasi & kecepatan).
- **H8**: R1 **runtuh saat input dirotasi** (tidak ada kanonikalisasi), sedangkan R2/R3 invarian — bukti kuantitatif paling tajam untuk kritikalitas PCA alignment (probe E10).

### 10.6 Efisiensi Representasi

**Karakteristik statis (a priori):**

| Aspek | R1 (PLY) | R2 (Canonical) | R3 (Pre-FPS) |
|---|---|---|---|
| Disk per frame | ~360-930 KB | ~428 KB | ~196 KB |
| Load time | PLY parse + Open3D (lambat) | np.load (cepat) | np.load (cepat) |
| Runtime sampling | Ya — random 8192 dari 15K | Ya — random 8192 dari 15K | Tidak — sudah 8192 |
| Butuh Open3D di Colab | Ya | Tidak | Tidak |

**Pengukuran kuantitatif (diisi dari E9 — rata-rata 5 seed, GPU: _catat_):**

| Metrik | R1 (PLY) | R2 (Canonical) | R3 (Pre-FPS) | Sumber |
|---|---|---|---|---|
| **Training** | | | | E9 |
| Waktu / epoch (dtk) | _ukur_ | _ukur_ | _ukur_ | |
| Total waktu sampai konvergen (dtk) | _ukur_ | _ukur_ | _ukur_ | |
| Epochs-to-converge | _ukur_ | _ukur_ | _ukur_ | |
| Split I/O vs compute (% data-load) | _ukur_ | _ukur_ | _ukur_ | bukti bottleneck |
| Training throughput (frame/dtk) | _ukur_ | _ukur_ | _ukur_ | |
| **Evaluasi** | | | | E9 |
| Preprocess latency / frame (ms) | _ukur_ | _ukur_ | _ukur_ | parse+sampling |
| Latency enrollment / subjek (dtk, N=5) | _ukur_ | _ukur_ | _ukur_ | |
| Latency probe / identifikasi (dtk, M=5) | _ukur_ | _ukur_ | _ukur_ | |
| End-to-end identification latency (dtk) | _ukur_ | _ukur_ | _ukur_ | |
| Inference throughput (frame/dtk) | _ukur_ | _ukur_ | _ukur_ | |
| **Resource** | | | | E9 |
| Peak GPU memory — training (MB) | _ukur_ | _ukur_ | _ukur_ | |
| Peak GPU memory — inference (MB) | _ukur_ | _ukur_ | _ukur_ | |
| Peak host RAM — dataloader (MB) | _ukur_ | _ukur_ | _ukur_ | |
| Disk / frame | ~360-930 KB | ~428 KB | ~196 KB | statis |
| Disk total dataset | _hitung_ | _hitung_ | _hitung_ | |
| Model size + FLOPs / forward | _idem (kontrol)_ | _idem_ | _idem_ | identik |
| **Probe isolasi** | | | | E10 |
| ΔEER vs rotasi (sensitivitas pose) | _ukur_ | ~0 (ekspektasi) | ~0 (ekspektasi) | uji H4 |
| Std embedding antar-ulangan (determinisme) | _ukur_ | _ukur_ | ~0 (FPS fixed) | R2 vs R3 |

> Hipotesis efisiensi: R3 tercepat di training & inference (tanpa runtime sampling), R1 terlambat (PLY parse + Open3D + sampling). Inti pertanyaan: **apakah penghematan kecepatan/disk R3 sepadan dengan kemungkinan kehilangan akurasi vs R2** (uji bersama H5), dan **apakah R1 runtuh saat dirotasi** sementara R2/R3 stabil (uji H4). Sintesis akhir = **Pareto frontier** akurasi × kecepatan × disk.

### 10.7 Implementasi yang Dibutuhkan

| # | Task | Effort | Status |
|---|---|---|---|
| E0 | **Regenerasi dataset dari raw depth** (`3DRegistration`): perbaiki QC knuckle (lihat 10.10), simpan normals di `output.ply`, hasilkan ketiga representasi konsisten untuk 11 subjek | ~1 jam | ✅ selesai (214 sesi, 2131 frame, 0 fail; di-mirror ke `3DCNN/dataset`) |
| E1 | Generate `cnn_input_fps.npy` @8192 untuk semua frame via `make_fps.py` (dari `cnn_input.npy`, open3d FPS) | ~15 mnt | ✅ selesai (2131/2131, shape (8192,6)) |
| E2 | `repr_mode`/`frame_mode` di data layer — diimplementasikan di `utils/dataset.py` (`load_session(repr_mode)`), bukan `dataset_lowdata.py`; `--frame-mode all` pakai `build_lowdata_splits_all_frames` | ~0.5 hari | ✅ selesai (commit `dd532007`) |
| E3 | `repr_mode="raw_ply"` — `load_session` load `output.ply` via open3d (`_load_ply_xyz_normals`, xyz+normals) | ~1 hari | ✅ selesai |
| E4 | `repr_mode="fps_npy"` — `load_session` load `cnn_input_fps.npy` (8192, tanpa runtime sampling) | ~0.5 hari | ✅ selesai |
| E5 | `train.py` + `evaluate.py`: flag `--repr-mode {canonical_npy,fps_npy,raw_ply}` & `--frame-mode {median,all}`; diteruskan ke `PalmFrameDataset`/`PalmPairDataset`/`ValPairMetric`/`eval_multiframe` | ~0.5 hari | ✅ selesai |
| E6 | Notebook `collab/v7_2_0_repr_ablation.ipynb` (self-contained, 36 cells) — C1/C2/C3 training+eval (C0 reuse v7.1.1) + seluruh analisis E7–E11 | ~1 hari | ✅ selesai (commit `7b207a49`) — **siap run di Colab** |
| E7 | Analisis akurasi: EER/Rank-1, confusion, TAR@FAR, d-prime, std lintas seed | ~0.5 hari | ✅ di notebook — terisi **setelah run** |
| E8 | **Grafik**: DET/ROC+AUC per representasi; chart → `analysis/v7_2_0_<ts>/` (disalin ke `result_docs/<ts>_v7_2_0/figs/` saat tulis laporan) | ~0.5 hari | ✅ di notebook — terisi setelah run |
| E9 | **Instrumentasi kecepatan & resource**: `train.py` tulis `perf.json` (total_train_wall_s, peak_gpu_mem_mb, gpu_name, n_train_frames); notebook ukur latency enroll/probe (dari `eval_multiframe`), microbench `load_session`/frame, disk footprint per repr | ~0.5 hari | ✅ kode selesai — angka terisi setelah run |
| E10 | **Probe isolasi**: (a) rotasi **pipeline-level** (rotasi RAW → re-kanonikalisasi PCA utk R2/R3, R1 tidak) → uji H4/H8; (b) determinisme R2-vs-R3. Sel notebook | ~0.5 hari | ✅ di notebook — terisi setelah run |
| E11 | **Sintesis trade-off**: scatter EER vs latency + **Pareto frontier** + effect size | ~0.5 hari | ✅ di notebook — terisi setelah run |

**Total estimasi**: implementasi **SELESAI** ✅ — tinggal ~2,5–3 jam Colab runtime (15 run + pass rotasi/determinisme; C0 reuse v7.1.1) lalu tulis `LAPORAN_v7_2_0.md`.

### 10.8 Quality Gates v7.2.0

- **Gate-4**: R2 (C2) mengungguli R1 (C1) dengan delta Rank-1 >= 5 pp — kanonikalisasi pose terbukti penting.
- **Gate-5**: R3 (C3) vs R2 (C2) — delta Rank-1 < 2 pp (setara) dengan throughput lebih tinggi.
- **Gate-6**: C2 (all frames) vs C0 (1 frame) — ada perbaikan Rank-1, memvalidasi manfaat semua frame.

### 10.9 Kontribusi Novelty

```
Kontribusi 1 (v7.1.0 — SELESAI):
  "Multi-frame fusion protocol: N=5 enrollment + M=5 probe menurunkan EER
   71% vs single-frame (4.55% -> 1.32%) tanpa mengubah arsitektur CNN."

Kontribusi 2 (v7.2.0):
  "Ablation representasi input menunjukkan bahwa kanonikalisasi pose
   (PCA + unit-sphere) adalah komponen paling kritis dalam pipeline:
   R2 vs R1 memberikan delta Rank-1 [X] pp. Representasi Pre-FPS (R3)
   mempertahankan performa setara dengan efisiensi lebih tinggi.
   Penggunaan semua frame enrollment memberikan peningkatan tambahan [Y] pp."
```

---

## 10.10 Temuan & Keputusan QC saat Regenerasi (2026-06-03)

Saat audit untuk regenerasi ditemukan masalah QC fundamental:

- **955 dari 1.946 frame (49%)** ber-`is_valid=False` di `geometry.json`, **semuanya** karena `knuckle_fallback` — dan ~930 di antaranya lolos diam-diam ke training v7.1.0 (filter QC 3DCNN hanya cek `invalid_frame.json`, bukan `is_valid`).
- `extract_geometry.py:372-375` sudah menyatakan knuckle "TIDAK lagi menjadi quality gate", tapi kodenya tetap memasukkan `knuckle_fallback` ke `quality_issues` → inkonsistensi.

**Keputusan: QC point-cloud untuk dataset CNN.** `knuckle_fallback` adalah isu ekstraksi **fitur geometri hand-crafted** (landmark buku jari), sedangkan CNN memakai **point cloud**. Spot-check membuktikan frame knuckle_fallback justru lebih padat (19.6K vs 17.9K titik), scan distance dalam rentang, dan secara visual telapak+5 jari lengkap — tak terbedakan dari frame clean. Anomali `palm_width/height` pada frame ini adalah artefak ekstraksi geometri yang gagal, bukan cacat point cloud.

**Definisi frame valid untuk CNN:**
- PLY ter-isolasi dengan ≥ `min_points` titik, **DAN**
- `scan_distance_mm` ∈ [150, 450] (ambang `extract_geometry.py` saat ini).
- `knuckle_fallback` → pindah ke field `warnings` (non-gating). `fingertip_fallback`/`fingers_too_close` tetap gate (kelengkapan jari = isu point cloud).

**Fix kode yang diterapkan:**
| File | Perubahan |
|---|---|
| `3DRegistration/extract_geometry.py` | `knuckle_fallback` → `warnings`, bukan `quality_issues`; `is_valid` kini hanya gate point-cloud |
| `3DRegistration/lib/single_frame.py` | `output.ply` menyimpan normals (R1 = 6 channel) |
| `3DRegistration/preprocess_for_cnn.py` | default `--n_points` 1024 → 8192 |
| `3DRegistration/make_fps.py` (baru) | generate `cnn_input_fps.npy` @8192 dari `cnn_input.npy` (open3d FPS) |

**Catatan:** 114 `invalid_frame.json` lama adalah legacy ambang scan_distance 200mm — dibersihkan saat regenerasi fresh (ambang baru [150,450] tidak menandai frame mana pun di data saat ini).

### Hasil regenerasi (final, 2026-06-03)

| Metrik | Nilai |
|---|---|
| Sesi diproses | 214 |
| Frame total | **2.131** (0 gagal, 0 QC-fail) |
| Frame valid (gate point-cloud) | **2.131 / 2.131** (100%) |
| `quality_issues` (gate) | 0 |
| `warnings: knuckle_fallback` (non-gate) | 1.021 |
| `output.ply` ber-normals | 2.131 / 2.131 |
| `cnn_input_fps.npy` (8192, 6) | 2.131 / 2.131 |
| `invalid_frame.json` | 0 |
| Per subjek | aisah 200, alji 150, chrys 200, fadhil 150, feby 210, gede 200, nola 221, rahmat 150, reysa 250, taufik 200, yanuar 200 |

Dataset di-mirror penuh ke `3DCNN/dataset/` (3.1 GB; `cnn_input.npy`+`geometry.json`+`output.ply`+`cnn_input_fps.npy` per frame).

### Temuan kanonikalisasi PCA (validasi vs dataset lama)

Validasi bit-identity vs `3DCNN/dataset` lama menemukan **~291 dari 1.836 frame (~16%) BERBEDA** — **bukan** karena perbedaan cleaning (point set **identik**: sorted-xyz diff 0.000000) melainkan **ambiguitas kanonikalisasi PCA** di `pca_align()`:

1. **Mayoritas — flip 180° terhadap sumbu Y** (x,z dinegasikan, y persis sama; residual setelah koreksi 180°-Y = 0.000000). Penyebab: sign sumbu **X tidak di-kanonikalisasi** (langsung dari `np.linalg.svd` `Vt`); hanya sumbu **Y** yang punya disambiguasi (median-Y, `preprocess_for_cnn.py:75-79`).
2. **Sebagian — perbedaan resolusi axis lebih besar** (mis. `range0≈range1` borderline atau flip median-Y terbalik) → pose kanonik benar-benar berbeda, bukan sekadar 180°.

**Keputusan (user, opsi A):** pakai dataset regen **utuh**, **recompute basis/normalizer fresh**, **jangan reuse C0/basis v7.1.0**, jangan campur old+new. `pca_align()` **TIDAK diubah** — R2 tetap sesuai definisi v7.1.0 yang dikunci agar ablation R1/R2/R3 tidak terkontaminasi perubahan metodologi.

**Implikasi thesis:** instabilitas X-sign/axis ini adalah **keterbatasan nyata representasi R2 canonical** (kanonikalisasi PCA tidak sepenuhnya invariant terhadap rotasi 180° in-plane). Relevan dibahas sebagai analisis: ablation R1 vs R2 dapat memperlihatkan apakah model belajar invariansi yang seharusnya dijamin kanonikalisasi. Perbaikan (disambiguasi sign X + stabilisasi axis) ditunda sebagai future work agar tidak mengubah baseline.

---

## 11. Referensi Internal

- Baseline & temuan v6: [LAPORAN_v6_lowdata.md](result_docs/20260525_152213_v6_lowdata/LAPORAN_v6_lowdata.md)
- Plan v5 (template format): [IMPROVEMENT_PLAN_v5.0.0.md](IMPROVEMENT_PLAN_v5.0.0.md)
- Temporal gap audit script & contoh: [result_docs/20260525_053041_v6_arcface/temporal_gap_audit.json](result_docs/20260525_053041_v6_arcface/temporal_gap_audit.json)
- Preprocessing pipeline: [3DRegistration/preprocess_for_cnn.py](../3DRegistration/preprocess_for_cnn.py) — `preprocess_full()` (R2) + `preprocess_fps()` (R3)
- Memori relevan: 1275 (capture burst), 1295–1298 (low-data findings v5), 1311–1313 (ArcFace integration v6).
