# Rancangan Improvement v0.6.0 — Reframing: ArcFace Loss untuk PointNet++ pada 3D Palm Identification

**Tanggal:** 2026-05-24
**Baseline yang dianalisis:** `v5.0.0` (GeoAtt low-data study, 2 varian × Triplet)
**Pemicu:** Diskusi strategis menegaskan bahwa **ArcFace loss function pada PointNet++ untuk 3D palm point cloud** merupakan kontribusi novelty yang belum dieksplorasi di literatur — lebih bersih secara metodologis dibandingkan perdebatan GeoAtt yang kompleks dan terkontaminasi bias arsitektur.
**Status implementasi:** Fase 1 (reframing & perencanaan) — **SELESAI**. Fase 2 (implementasi + training 2 varian) — **siap dijalankan**.

Laporan baseline v5.0.0: [`IMPROVEMENT_PLAN_v5.0.0.md`](IMPROVEMENT_PLAN_v5.0.0.md)
Technical handover asli: [`technical_handover_kimi_code.md`](../../technical_handover_kimi_code.md)

---

## Konteks & Motivasi

### Mengapa pivot dari GeoAtt ke ArcFace

Setelah eksplorasi panjang pada v0.3.0 → v0.4.0 → v5.0.0, temuan konsisten menunjukkan bahwa **GeoAtt memberikan kontribusi yang tidak stabil** — terkadang merugikan, terkadang netral, dan sangat sensitif terhadap bias eksperimental (init parity, feature noise, split leakage). Sementara itu, **ArcFace loss pada PointNet++ untuk 3D palm recognition** belum pernah dipublikasikan secara eksplisit:

1. **BézierPalm 2022** = ArcFace pada **2D palm** (bukan 3D point cloud).
2. **MDPI 2025** = ArcFace pada **3D face** (bukan palm).
3. **Zhang et al. 2019** = Mobile **2D** palm (bukan 3D, bukan ArcFace).
4. **Qi et al. 2017** (PointNet++) = classification/segmentation (bukan recognition dengan metric learning).

**Gap literatur ini bersih dan tidak terkontaminasi** — tidak ada kompleksitas arsitektur GAM, tidak ada pertanyaan "apakah fitur geometri membantu", tidak ada 14-dim feature engineering. Hanya satu pertanyaan:

> **"Apakah ArcFace loss, yang telah terbukti superior untuk face recognition, dapat meningkatkan performa 3D palm identification berbasis PointNet++ dibandingkan Triplet loss baseline?"**

### Mengapa ini lebih superior secara metodologis

| Aspek | GeoAtt-Centric (v5.0.0) | ArcFace-Centric (v6.0.0) |
|---|---|---|
| **Novelty claim** | Ambigu — GeoAtt pada 3D palm pernah dieksplorasi di literatur lain (walaupun tidak identik) | **Bersih** — ArcFace + PointNet++ + 3D palm = **belum ada publikasi** |
| **Fair comparison** | Sulit — GeoAtt menambah parameter & kompleksitas; baseline vs proposed tidak apple-to-apple | **Mudah** — backbone identik, preprocessing identik, **hanya loss yang berbeda** |
| **Reproducibility** | Bergantung pada kualitas 14-dim feature extraction (pose-sensitive) | **Tinggi** — loss function pure software; tidak bergantung pada ekstraksi fitur geometri |
| **Deployment story** | GeoAtt butuh pipeline ekstraksi geometri di runtime | ArcFace = end-to-end deep learning; lebih mudah di-deploy |
| **Statistical clarity** | Variansi tinggi akibat feature noise & arsitektur instability | Variansi terkontrol; perbedaan murni dari sifat loss function |

### Hubungan dengan technical handover asli

Rancangan awal thesis (`technical_handover_kimi_code.md`) memang sudah merancang eksperimen **PointNet++ + Triplet (baseline) vs PointNet++ + ArcFace (proposed)** pada ~150 frame (10 subjek, ~15 sesi, 1 frame/sesi). Versi v6.0.0 ini adalah **return to original framing** dengan pengalaman & infrastruktur yang sudah matang dari iterasi sebelumnya:
- ✅ Fair ablation & init parity (dipelajari dari v0.4.0).
- ✅ Low-data regime & chronological split (dipelajari dari v5.0.0).
- ✅ Val pair EER sebagai model selection metric (dipelajari dari v5.0.0).
- ✅ Fixed training budget (dipelajari dari v5.0.0).
- ✅ Augmentation depth-only (dipelajari dari v5.0.0).

**Yang di-drop:** GeoAtt, geometric attention module, 14-dim/13-dim feature set, GAM, FiLM fusion, auxiliary loss. Semua tidak lagi relevan untuk pertanyaan ilmiah yang baru.

---

## Hipotesis & Temuan Carry-Over dari v5.0.0

Berikut temuan diagnostik dari v4.0.0/v5.0.0 yang **wajib di-mitigate** di v6.0.0 agar verdict loss-function comparison bersih:

| # | Temuan | Status di v6.0.0 | Tindakan |
|---|---|---|---|
| 1 | Split bocor temporal (train/test/holdout dalam ±90 detik) | **Fixed** — chronological deterministic split (s1–s8 / s9–s10 / s11–s12 / s13–s15) | Dipertahankan dari v5.0.0 F2.1 |
| 2 | Val_loss anti-korelasi dengan test EER | **Fixed** — model selection via val pair EER | Dipertahankan dari v5.0.0 F2.2 |
| 3 | Training budget tidak seragam (early stopping berbeda antar run) | **Fixed** — fixed budget 120 + 30 epoch | Dipertahankan dari v5.0.0 F2.4 |
| 4 | Intra-session redundancy (10 frame/sesi hampir identik) | **Fixed** — 1 median frame per sesi | Dipertahankan dari v5.0.0 F2.1 |
| 5 | Augmentation tidak cukup untuk depth-only | **Fixed** — rotation Z ±45°, tilt ±20°, translation XYZ ±3cm, scale 0.95–1.05, jitter σ=1mm | Dipertahankan dari v5.0.0 F2.6 |
| 6 | ArcFace scale s=30 cocok untuk small dataset | **Diuji ulang** — jadi variable utama, bukan diganti ke Triplet | F2.5 — hyperparameter search m & s |

---

## Strategi v6.0.0: Loss-Function Comparison, 2-Variant, Depth-Only

### Pertanyaan ilmiah yang baru

> **"Apakah ArcFace loss memberikan keuntungan signifikan dibandingkan Triplet batch-hard loss pada 3D palm identification berbasis PointNet++ dalam regime enrollment terbatas (1 sampel per sesi)?"**

### Mengapa 2 varian (Triplet vs ArcFace)

Pertanyaan ilmiah memerlukan perbandingan yang **sefair mungkin**:

| Komponen | Baseline (Triplet) | Proposed (ArcFace) | Catatan |
|---|---|---|---|
| **Backbone** | PointNet++ (SA + FP + global pooling) | PointNet++ (identik) | 100% identik |
| **Preprocessing** | Normalisasi + FPS 8192 + dropout | Normalisasi + FPS 8192 + dropout | Identik |
| **Augmentasi** | Pose + distance depth-only | Pose + distance depth-only | Identik |
| **Dataset** | 1 median frame/sesi, 10 subjek, split 8/2/2/3 | Identik | Split deterministic, IDENTIK antar varian & seed |
| **Training Budget** | Phase 1: 120 ep, Phase 2: 30 ep | Identik | Fixed, no early stopping |
| **Model Selection** | Val pair EER (smoothed, window=5) | Identik | Apple-to-apple |
| **Optimizer** | Adam, lr=2e-3 → 2e-4 | Identik | Identik |
| **Batch Size** | 64 | 64 | Identik |
| **Embedding Dim** | 512 | 512 | Identik |
| **L2 Normalization** | Ya (sebelum loss & matching) | Ya | Identik |
| **Loss Function** | **Triplet (batch-hard, margin=0.3)** | **ArcFace (m=0.5, s=30)** | **Hanya ini yang berbeda** |
| **Matching** | Cosine similarity | Cosine similarity | Identik |

**Keuntungan dari kesederhanaan ini:**
1. Tidak ada argumen "unfair comparison" dari reviewer — setiap komponen kecuali loss adalah identik.
2. RNG init parity otomatis tercapai karena backbone identik; tidak perlu patch init parity khusus seperti di v0.4.0.
3. Parameter count identik (tidak ada branch geometri yang menambah/mengurangi parameter).

### Mengapa tetap low-data (1 frame/sesi)

Meskipun ArcFace secara teori "butuh banyak sampel per kelas untuk stabilitas margin", hasil v0.3.0 menunjukkan ArcFace pada PointNet++ **mencapai 99.82% Rank-1** di 11 subjek dengan ~170 sampel (all-frame). Pertanyaan menariknya bukan "apakah ArcFace bisa konvergen?" tapi **"apakah ArcFace lebih baik dari Triplet di low-data?"** — yang persis adalah scenario deployment biometric realistis.

| Aspek | All-frame (~1.869 frame) | Low-data (150 frame, 1/sesi) |
|---|---|---|
| Ceiling effect | **Tinggi** — kedua loss bisa mencapai >99%; sulit bedakan | **Moderat** — cukup challenging untuk bedakan performa |
| Deployment relevance | Rendah | **Tinggi** — 1 sampel per enrollment = realistis |
| Novelty framing | "ArcFace juga bisa" | **"ArcFace unggul di low-data"** — klaim lebih kuat |
| Compute cost | ~1 jam/varian | **~25 menit/varian** — lebih banyak seed bisa diuji |

**Keputusan:** fokus utama = **low-data regime** (150 frame). Eksperimen all-frame opsional sebagai sanity check (3 seed saja) kalau compute tersisa.

### Dataset Protocol (Carry-Over dari v5.0.0 F2.1)

- **10 subjek**: aisah, alji, chrys, fadhil, feby, nola, rahmat, reysa, taufik, yanuar.
- **Dropped**: gede (9 sesi, di bawah minimum 15).
- **Per subjek**: 15 sesi kronologis → 1 median frame per sesi.
- **Total**: 150 frame = 80 train + 20 val + 20 test + 30 holdout.
- **Split**: deterministic chronological 8/2/2/3, **IDENTIK** untuk Triplet & ArcFace & semua seed.
- **No randomness di split** — semua variansi berasal dari `model_seed` saja.

---

## Fase 2 — Rencana Tindak Lanjut

### F2.0 Code Cleanup: Strip GeoAtt (Prioritas 0 — Preflight)

**Tujuan:** menghapus seluruh dependency GeoAtt agar pipeline bersih untuk loss-function-only comparison.

**File yang dimodifikasi:**
- `models/encoder.py` — hapus `geom_encoder`, `gam1`, `gam2`, `proj_with_geom`, flag `use_gam`/`use_geom_fusion`. Sederhanakan ke PointNet++ murni.
- `models/siamese.py` — hapus parameter `geom` di forward; hapus `aux_classifier` (jika ada dari v5.0.0).
- `train.py` — hapus flag `--use-geom`, `--use-gam`, `--use-geom-fusion`, `--use-aux-loss`; hapus loading `geometry.json`.
- `evaluate.py` — hapus dependency geometry; hanya butuh `cnn_input.npy`.
- `utils/dataset.py` — mode tanpa geometry menjadi default; geometry loader tetap ada untuk backward-compat tapi tidak dipakai.

**File yang tidak perlu dihapus (biarkan dormant):**
- `models/gam.py`, `models/geometry_encoder.py` — tetap di repo, tidak di-import.
- `utils/audit_geom_*.py` — tetap di repo.

**Smoke test:**
```bash
python -c "import models.encoder; print('PointNet++ murni OK')"
python train.py --help | grep -E '(geom|aux)'  # harus kosong
```

**Effort:** ~60 menit.

---

### F2.1 Dataset Loader `OneFramePerSession` + Chronological Split (Prioritas 1, WAJIB — Reuse v5.0.0)

**Reuse langsung** implementasi dari v5.0.0 F2.1:
- `utils/dataset_lowdata.py` — jika sudah dibuat di v5.0.0, gunakan as-is.
- `utils/val_pair_metric.py` — jika sudah dibuat, gunakan as-is.
- Split protokol 8/2/2/3, drop gede, deterministic.

**Jika file belum ada:** implementasi persis sesuai v5.0.0 F2.1 dan F2.2.

---

### F2.2 Val Pair EER Metric Logger (Prioritas 1, WAJIB — Reuse v5.0.0)

Model selection berdasarkan **val pair EER** (bukan val_loss), dengan smoothing window=5 epoch. Log ke TensorBoard: `val/pair_eer`, `val/pair_auc`.

---

### F2.3 Fixed Training Budget (Prioritas 1, WAJIB — Reuse v5.0.0)

- Phase 1: **120 epoch**
- Phase 2 (fine-tune, lr lebih rendah): **30 epoch**
- Total: **150 epoch fixed**, tanpa early stopping default.
- `best.pth` dipilih berdasarkan val pair EER terbaik sepanjang trajectory.

---

### F2.4 Loss Function Configuration (Prioritas 1, WAJIB)

**Baseline — Triplet (sudah ada di `losses/triplet.py`):**
```bash
python train.py \
  --output_dir runs/v6_lowdata/triplet/seed_${seed} \
  --frames-per-session 1 \
  --loss triplet \
  --triplet-margin 0.3 \
  --val-metric pair_eer \
  --epochs-phase1 120 --epochs-phase2 30 \
  --batch-size 64 \
  --seed ${seed}
```

**Proposed — ArcFace (sudah ada di `losses/arcface.py` atau `models/siamese.py`):**
```bash
python train.py \
  --output_dir runs/v6_lowdata/arcface/seed_${seed} \
  --frames-per-session 1 \
  --loss arcface \
  --arcface-margin 0.5 \
  --arcface-scale 30 \
  --val-metric pair_eer \
  --epochs-phase1 120 --epochs-phase2 30 \
  --batch-size 64 \
  --seed ${seed}
```

**Perhatian:** ArcFace memerlukan `num_classes` (jumlah subjek = 10). Pastikan di-set otomatis dari dataset scanner.

---

### F2.5 Augmentation Strategy: Pose + Distance Only (Prioritas 1 — Reuse v5.0.0)

Identik dengan v5.0.0 F2.6:
- Rotation Z: ±45°
- Tilt X/Y: ±20°
- Translation XYZ: ±3cm (Z = jarak ke sensor, baru)
- Random scale: 0.95–1.05
- Point subsampling: 8192 dari N total
- Random jitter Gaussian: σ=1mm

**Tidak diperlukan (depth-only):** color jitter, brightness, shadow.

---

### F2.6 Hyperparameter Search ArcFace (Prioritas 2 — Opsional tapi Direkomendasikan)

Technical handover menyarankan `m=0.5, s=30` untuk small dataset. Tapi untuk kelengkapan, lakukan **grid search singkat** pada 1 seed sebelum full run:

| Margin (m) | Scale (s) | Justifikasi |
|---|---|---|
| 0.3 | 30 | Lebih conservative (smaller angular penalty) |
| 0.5 | 30 | **Default technical handover** |
| 0.5 | 64 | Scale lebih besar ( literature default untuk face) |
| 0.7 | 30 | Lebih aggressive margin |

**Prosedur:**
1. Jalankan 4 kombinasi × 1 seed = 4 run (~1.5 jam).
2. Pilih kombinasi dengan **val pair EER terendah**.
3. Gunakan kombinasi terbaik untuk full run 10 seed.

**Fallback:** jika semua hasil mirip (within noise), gunakan default `m=0.5, s=30` untuk konsistensi dengan technical handover.

---

### F2.7 Eksperimen Utama (Prioritas 1)

**Setup:** 2 varian × **10 seed** × 1 median frame/sesi × split chronological 8/2/2/3.

**Seed values:** 42, 123, 2024, 7, 31337, 0, 1, 2, 3, 4 (10 seed total).

**Metrics yang dilaporkan:**
- `val/pair_eer` → untuk `best.pth` selection
- **`test/eer`** → metric utama untuk Wilcoxon paired test
- **`test/rank1`** → secondary
- **`holdout/eer`** → klaim generalization
- **`test/auc`** → completeness
- **`test/tar_at_far1`** → completeness

**Estimasi wall-time:** ~25 menit/run × 20 run = **~8–10 jam pada A100**.

---

### F2.8 Eksperimen Pendamping: All-Frame Sanity Check (Prioritas 3, Opsional)

**Tujuan:** verifikasi bahwa ArcFace tidak "hancur" di all-frame regime (sanity check saja, bukan klaim utama).

**Setup:** 2 varian × 3 seed × all-frame regime.
- Phase 1: 80 epoch, Phase 2: 20 epoch.
- Batch size: 256.

**Estimasi wall-time:** ~1 jam/run × 6 run = **~6 jam**.

---

### F2.9 Analisis & Plotting (Prioritas 1)

**Skrip baru:** `analysis/v6_loss_comparison.py`

Output utama:
1. **Tabel ringkasan**: mean ± std EER, Rank-1, AUC, TAR@FAR=1% untuk 2 varian × 10 seed.
2. **Box plot**: EER distribution Triplet vs ArcFace across seeds.
3. **ROC Curve overlay**: Triplet vs ArcFace (mean ± 95% CI across seeds).
4. **CMC Curve overlay**: Triplet vs ArcFace.
5. **Paired Wilcoxon test**: `test/eer` Triplet vs ArcFace (n=10 seed).
6. **Bootstrap CI** untuk Δ EER (ArcFace − Triplet), n_resample=1000.
7. **Training curve comparison**: val pair EER trajectory (mean across seeds, shaded std).
8. **Per-subjek analysis**: subjek mana yang paling diuntungkan/dirugikan oleh ArcFace.

---

## Target Metrik v0.6.0

### Low-Data Regime (split 8/2/2/3, n=10 seed, 10 subjek)

| Metrik | Triplet (baseline) target | ArcFace (proposed) target | Status sukses |
|---|---|---|---|
| **Val EER** (selection) | Trajectory turun monoton | Trajectory turun monoton | Keduanya converge |
| **Test EER** (primary, paired Wilcoxon) | Baseline | **< Triplet, p < 0.05** | 🎉 **ArcFace superior** |
| **Test Rank-1** | Baseline | **> Triplet** | Secondary confirmation |
| **Holdout EER** | Baseline | **≤ Triplet** | Generalization consistent |
| Test EER std antar seed | < 5% | < 5% | Stabil |
| Bootstrap CI Δ Test EER | n/a | n/a | **CI tidak melingkupi 0, sisi negatif (ArcFace menang)** |

### Kriteria Sukses Minimum

ArcFace **tidak signifikan merugikan** dibanding Triplet (Wilcoxon p > 0.05 untuk arah merugikan). Ini sudah cukup untuk klaim **"ArcFace adalah alternatif yang valid dan competitive"**.

### Kriteria Sukses Maksimum

ArcFace **signifikan lebih baik** dari Triplet di low-data (p < 0.05, bootstrap CI Δ EER tidak melingkupi 0). Klaim thesis menjadi:

> *"ArcFace loss, yang telah terbukti superior untuk face recognition, secara signifikan meningkatkan performa 3D palm identification berbasis PointNet++ dalam regime enrollment terbatas, memberikan kemiripan intra-kelas yang lebih kompak dan separasi antar-kelas yang lebih tegas dibandingkan Triplet batch-hard loss."*

---

## Decision Gates — Kriteria Stop/Continue

### Gate 0 — Setelah F2.0 (Code Cleanup & Smoke Test)

**Kriteria:**

| Kondisi | Aksi |
|---|---|
| PointNet++ murni bisa train & eval tanpa error (1 seed × 2 varian, 5 epoch) | ✅ **CONTINUE** ke Gate 1 |
| Import error / checkpoint load fail / geometry dependency masih tersisa | ⚠️ **DEBUG**: selesaikan cleanup GeoAtt |
| Training crash / NaN / OOM | 🛑 **STOP**: fix infrastructure |

### Gate 1 — Setelah Smoke Test 1 Seed × 2 Varian Low-Data (~1 jam)

**Kriteria:**

| Kondisi | Aksi |
|---|---|
| Val pair EER trajectory turun monoton untuk **kedua** varian dan plateau < 25% di epoch 50 | ✅ **CONTINUE** ke full run F2.7 |
| ArcFace val EER stuck > 40% atau tidak konvergen | ⚠️ **DEBUG**: cek scale s (mungkin 30 terlalu kecil? coba 64), cek num_classes, cek gradient clipping. Triplet sebagai fallback stabil. |
| Triplet collapse (embedding seragam) | ⚠️ **DEBUG**: cek batch-hard mining, cek margin, cek learning rate. |
| Kedua varian tidak turun sama sekali | 🛑 **STOP**: audit pair sampling & loss computation. |

### Gate 2 — Setelah F2.7 (Full Run 10 Seed × 2 Varian Low-Data)

**Primary metric:** `test/eer`

| Kondisi (paired Wilcoxon n=10) | Verdict | Aksi |
|---|---|---|
| p < 0.05, ArcFace test EER < Triplet test EER | 🎉 **HIPOTESIS TERKONFIRMASI** | Lanjut F2.9 analysis & laporan. Tag `v0.6.0-final`. |
| p > 0.10 (tidak ada significant difference) | 🟡 **HIPOTESIS NETRAL** | Klaim: "ArcFace competitive dengan Triplet; keunggulan muncul di stabilitas konvergensi / deployment simplicity". Laporan tetap valid. |
| p < 0.05, ArcFace test EER > Triplet test EER (ArcFace kalah) | 🔴 **HIPOTESIS DITOLAK** | Dokumentasikan dengan integritas. Klaim fallback: "Di low-data regime dengan PointNet++, Triplet masih superior; ArcFace memerlukan lebih banyak data per kelas untuk unggul". Pertimbangkan F2.8 all-frame untuk lihat apakah ArcFace menang di data lebih banyak. |
| Variansi sangat tinggi → CI sangat lebar | ⚠️ **POWER KURANG** | Tambah ke 15 seed. |

**Sub-check (Holdout EER consistency):**
- Jika Test & Holdout arah sama → verdict diperkuat.
- Jika berbeda → dokumentasikan distribution shift; klaim hanya untuk test set.

---

## Risiko & Mitigasi

| Risiko | Probabilitas | Mitigasi |
|---|---|---|
| ArcFace tidak lebih baik dari Triplet di low-data | Sedang | Triplet adalah baseline yang kuat; klaim "competitive" masih valid. Laporan tetap ilmiah. |
| ArcFace scale/margin tidak optimal | Sedang | F2.6 grid search singkat sebelum full run. |
| Triplet collapse di low-data | Rendah | Monitor embedding norm; fallback semi-hard mining. |
| Variansi seed tinggi → tidak ada significance | Sedang | 10 seed; bootstrap CI n=1000; tambah ke 15 seed jika perlu. |
| Split masih bocor walau sudah chronological | Rendah | Dokumentasikan eksplisit limitation (capture window ~2 menit). Klaim "low-data robustness" tetap valid. |
| GPU time blow-up | Rendah | Smoke test dulu; budget 10 jam total. |

---

## Catatan Sejarah & Hubungan dengan Plan Sebelumnya

- **v0.2.0-baseline:** Triplet loss, GeoAtt, ~60% Rank-1. Hipotesis: GeoAtt sebagai regularizer.
- **v0.3.0-baseline:** ArcFace, no_geom 99.82%, with_geom 95.82%. Menemukan init parity issue.
- **v0.4.0-baseline:** Init parity fixed, fair ablation 4 varian, QC v3. Menemukan split leakage & val_loss anti-korelasi.
- **v4.0.0:** Re-eval all-frame. Verdict no_geom > with_geom — kemudian terungkap 3 bias eksperimental.
- **v5.0.0:** Pivot ke low-data regime study (GeoAtt vs no_geom, Triplet loss). Memperbaiki bias & framing. **Tidak pernah dieksekusi fully** (waiting for implementation).
- **v0.6.0 (rencana ini):** **Pivot kedua** — drop GeoAtt sepenuhnya; fokus pada **ArcFace vs Triplet** sebagai kontribusi novelty utama. Return to original technical handover framing dengan infrastruktur matang dari v5.0.0.

### Status kerja

- [x] Reframing: ArcFace sebagai novelty utama (vs GeoAtt)
- [x] Identifikasi carry-over fix dari v5.0.0 (split, val metric, fixed budget, augmentation)
- [x] Perencanaan fair comparison matrix (identik kecuali loss)
- [ ] F2.0: Code cleanup — strip GeoAtt dari pipeline
- [ ] F2.0: Smoke test import & 5-epoch run
- [ ] F2.1–F2.3: Reuse/low-touch dataset loader, val pair EER, fixed budget
- [ ] F2.5: Reuse augmentation depth-only
- [ ] F2.6: Grid search ArcFace m & s (1 seed, 4 kombinasi)
- [ ] **Gate 0**: smoke test clean pass
- [ ] **Gate 1**: 1 seed × 2 varian low-data trajectory check
- [ ] F2.7: Full run 10 seed × 2 varian low-data (~8–10 jam)
- [ ] **Gate 2**: Wilcoxon test → verdict hipotesis
- [ ] F2.8 (opsional): All-frame sanity check 3 seed × 2 varian
- [ ] F2.9: Analysis & plotting
- [ ] Tag baru `v0.6.0-final` setelah Gate 2 pass
- [ ] Update laporan thesis & technical documentation

---

## Lampiran — File yang Akan Dimodifikasi/Dibuat

### Source code (modifikasi — strip GeoAtt):
- `models/encoder.py` — simplify ke PointNet++ murni; hapus geom branch & GAM
- `models/siamese.py` — hapus geom input & aux_classifier
- `train.py` — hapus flag GeoAtt & aux; default loss = triplet; tambah flag `--loss {triplet|arcface}`
- `evaluate.py` — hapus dependency geometry.json
- `utils/dataset.py` — default tanpa geometry; geometry loader dormant

### Source code (reuse/low-touch dari v5.0.0):
- `utils/dataset_lowdata.py` — `OneFramePerSession` loader (jika belum ada, buat baru)
- `utils/val_pair_metric.py` — val EER/AUC logger per epoch (jika belum ada, buat baru)
- `utils/augmentation.py` — depth-only augmentation (reuse)

### Skrip baru:
- `analysis/v6_loss_comparison.py` — plot & statistical analysis untuk central finding
- `utils/audit_loss_config.py` — verifikasi bahwa konfigurasi Triplet & ArcFace identik kecuali loss

### Output rencana:
- `runs/v6_lowdata/triplet/seed_{N}/...` — checkpoint, train_log, TensorBoard
- `runs/v6_lowdata/arcface/seed_{N}/...` — checkpoint, train_log, TensorBoard
- `eval_results/v6_lowdata/{variant}/seed_{N}/...`
- `analysis/v6/aggregate.csv`, `eer_boxplot.png`, `roc_overlay.png`, `cmc_overlay.png`
- `result_docs/{ts}/v6_arcface_finding.md` — laporan central finding

### Dokumen referensi:
- [`technical_handover_kimi_code.md`](../../technical_handover_kimi_code.md) — rancangan awal thesis
- [`IMPROVEMENT_PLAN_v5.0.0.md`](IMPROVEMENT_PLAN_v5.0.0.md) — plan sebelumnya (GeoAtt low-data)
- [`IMPROVEMENT_PLAN_v0.4.0.md`](IMPROVEMENT_PLAN_v0.4.0.md) — plan fair ablation & init parity

---

## Klaim Thesis yang Diharapkan

Setelah eksekusi v0.6.0 selesai, klaim thesis yang **paling kuat** (jika hasil sesuai prediksi):

> **"Pada palm identification berbasis 3D point cloud yang diakuisisi dari TrueDepth Camera iPhone, penggunaan ArcFace loss function pada backbone PointNet++ memberikan performa superior dibandingkan Triplet batch-hard loss dalam regime enrollment terbatas (1 sampel per sesi). Eksperimen pada 10 subjek dengan 10 seed independen menunjukkan bahwa ArcFace secara konsisten mencapai EER lebih rendah dan Rank-1 lebih tinggi (Wilcoxon paired p < 0.05, bootstrap CI Δ EER tidak melingkupi nol), menunjukkan bahwa margin angular yang tegas dan kompak intra-class yang dihasilkan ArcFace lebih efektif untuk membedakan identitas telapak tangan dalam kondisi data minim."**

**Jika hasil netral:**
> "ArcFace loss menawarkan alternatif yang competitive dan lebih mudah di-deploy (end-to-end, tanpa feature engineering) dibandingkan Triplet loss untuk 3D palm identification berbasis PointNet++, dengan performa setara dalam regime enrollment terbatas."

**Jika hasil negatif (ArcFace kalah):**
> "Dalam regime low-data (1 sampel per sesi) dengan PointNet++ murni, Triplet batch-hard loss tetap lebih efektif dibandingkan ArcFace. ArcFace memerlukan lebih banyak sampel per kelas untuk mengekspresikan keunggulan margin angularnya — sebuah temuan yang menginformasikan desain loss function untuk biometric 3D point cloud di masa depan."

---

*Dokumen ini disusun sebagai Improvement Plan v0.6.0 untuk pivot framing thesis dari GeoAtt-centric ke Loss-Function-centric (ArcFace vs Triplet). Semua keputusan didasarkan pada technical handover asli dan pengalaman infrastruktur dari v0.4.0/v5.0.0.*
