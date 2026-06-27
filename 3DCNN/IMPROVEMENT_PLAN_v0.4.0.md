# Rancangan Improvement v0.4.0 — Diagnostik & Perbaikan Fair Ablation GeoAtt

**Tanggal:** 2026-05-17
**Baseline yang dianalisis:** `v0.3.0-baseline` (ArcFace, no_geom Rank-1 99.82%, with_geom 95.82%)
**Pemicu:** Pembalikan verdict GeoAtt vs laporan v0.2.0 (Triplet). GeoAtt tampak merugikan, padahal secara teori harus membantu.
**Status implementasi:** Fase 1 (diagnostik + perbaikan kode untuk fair ablation) — **SELESAI**. Fase 2 (training 4-arah + perbaikan struktural) — siap dijalankan.

Plan asli: `/Users/rahmatzulfikri/.claude/plans/evaluasi-hasil-cnn-3dcnn-serene-pixel.md`
Laporan diagnostik: `result_docs/20260517_064046/diagnostic_phase1.md`

---

## Konteks & Motivasi

Setelah loss diganti dari Triplet → ArcFace di v0.3.0:
- `no_geom`: Rank-1 99.82% ± 0.36%, EER 0.03% (hampir sempurna pada 11 subjek).
- `with_geom`: Rank-1 95.82% ± 1.59%, EER 2.76% — turun.
- McNemar p=1.8×10⁻⁵; Bootstrap CI Δrank-1 [−0.053, −0.031] → bukan kebetulan.

Pertanyaan ilmiah: apakah GeoAtt *memang* merugikan secara intrinsik, atau ada bias eksperimen yang menutupi efeknya?

---

## Hipotesis & Hasil Uji (Fase 1)

| # | Hipotesis | Verdict | Bukti |
|---|---|---|---|
| 1 | Saturasi ceiling pada 11 subjek | **DITOLAK** | Hard-probe analysis: gap with−no **MEMBESAR** di bottom-25% probe sulit (overall Δ=−0.056, hard Δ=−0.107) |
| 2 | Fitur 14-dim geom mayoritas noise (FDR<1) | **DITOLAK** | 0/14 fitur FDR<1; median FDR=3.77 |
| 3 | Kesalahan with_geom sistematis pada subjek `nola` | **TERKONFIRMASI** | `nola.finger_width_5` CV antar-sesi 0.497 vs rata-rata subjek lain 0.056 (**8.85× outlier**) |
| 4 | RNG init parity rusak antara use_geom=True/False | **TERKONFIRMASI** | 13/33 shared layer berbeda init; hanya 1.5% elemen identik; max\|Δ\|≈0.65 |
| 5 | Geom-emb shared GAM1+GAM2 → gradient bottleneck | TIDAK DIUJI | Memerlukan ablasi arsitektur (Fase 2) |
| 6 | Dropout(0.3) hanya di fusion head → asimetri train/eval | **TERKONFIRMASI ringan** | cos(eval, train_dropout): with_geom 0.908 vs no_geom 0.950 (gap 1.84× lebih besar) |
| 7 | Z-score menghapus skala absolut tangan | TIDAK TERBUKTI RELEVAN | Konsisten dengan D2: separabilitas dataset-wide tetap baik (finger_len FDR 5–21) |

### Sintesis akar penyebab

GeoAtt **memang** memberi kontribusi merugikan pada setup ArcFace, **tetapi** dengan dua kontaminasi eksperimen:

1. **Ablasi tidak fair (D1):** with_geom vs no_geom mulai dari titik awal berbeda; tidak bisa menyimpulkan kontribusi modul.
2. **Subjek/fitur instabil (D2+D3):** instabilitas `finger_width_5` nola antar-sesi dilewatkan ke ArcFace yang scale-sensitive lewat fusion head → meracuni keputusan pada probe terkait nola.

Saturasi (H1) dan noise feature (H2) yang dulu jadi hipotesis dominan — keduanya **gagal**.

---

## Perbaikan Fase 1 — Sudah Diimplementasikan

### F1.1 RNG Init Parity (perbaikan D1)

**File:** `models/encoder.py`

Sebelum: `geom_encoder`, `gam1`, `gam2` hanya dibangun jika `use_geom=True`. Skip ini menggeser konsumsi RNG global → SA1/SA2/SA3/proj/ArcFace head punya bobot awal berbeda antar varian.

Sesudah: **selalu** bangun seluruh sub-modul (`geom_encoder`, `sa1`, `gam1`, `sa2`, `gam2`, `sa3`, `proj_with_geom`, `proj_no_geom`) di `__init__` dengan urutan tetap. Flag baru `use_gam` / `use_geom_fusion` hanya mengontrol forward path; modul nonaktif tidak dipakai tetapi tetap ada di model.

**Bukti perbaikan:** re-audit `utils/audit_init_parity.py` → 58/58 layer identik, max|Δ|=0 untuk semua seed.

**Catatan ukuran:** parameter count antar 4 varian sekarang identik (436,448 params). Trade-off: model no_geom membawa ~10rb parameter "mati" dari geom branch. Ini sengaja, agar fair-ablation strict.

### F1.2 Flag Ablasi Modul

**File:** `models/encoder.py`, `models/siamese.py`, `train.py`

Tiga flag CLI baru di `train.py`:
- `--use-gam` — aktifkan GAM1/GAM2 di forward
- `--use-geom-fusion` — aktifkan concat geom_emb ke proj head
- `--use-geom` — shortcut: keduanya aktif (= varian with_geom v0.3.0)

Default (tanpa flag) = no_geom. Empat varian eksperimen:

| Nama | use_gam | use_geom_fusion | Tujuan diagnostik |
|---|---|---|---|
| `no_geom` | False | False | Baseline murni |
| `with_geom` | True | True | Baseline penuh GeoAtt |
| `gam_only` | True | False | Isolasi efek GAM saja |
| `fuse_only` | False | True | Isolasi efek fusion concat saja |

### F1.3 Backward-Compat Checkpoint Loader

**File:** `evaluate.py`

Checkpoint pre-v0.4.0 menyimpan `encoder.proj.*`. Setelah patch, encoder punya `proj_with_geom.*` dan `proj_no_geom.*`. `load_model()` otomatis rename berdasarkan flag aktif. Smoke test: kedua checkpoint v0.3.0-baseline load tanpa error.

### F1.4 Skrip Diagnostik

Empat skrip baru di `utils/` (tetap relevan untuk re-audit di Fase 2):
- `audit_init_parity.py` — verifikasi parity pasca-patch.
- `audit_geom_session_variance.py` — FDR per fitur + outlier per-subjek.
- `audit_embedding_stats.py` — perilaku dropout pada checkpoint.
- `eval_hard_probes.py` — komparasi pada subset probe sulit.

---

## Strategi Data QC: v2 → v3 (Frame-Level Exclusion)

**Konteks:** Audit D2 menemukan `nola.finger_width_5` (kelingking) CV antar-sesi 0.497 — 8.85× outlier dibanding rata-rata subjek lain (0.056). Pertanyaan: apakah kita filter data nola yang bermasalah?

### Evolusi Keputusan QC

| Versi QC | Level | Kriteria | Hasil | Status |
|---|---|---|---|---|
| QC v2 | **Session-level** | within-session std > k × median_global | 35 sesi flagged (16.5%) | **Deprecated** — user merasa terlalu agresif |
| **QC v3** | **Frame-level** | \|value − median_session\| > k × MAD_session per frame; >50% outlier → exclude sesi | 160 frame + 1 sesi excluded (8.02%) | **Adopted** |

### Logika QC v3 (User Request)

> *"Aku prefer exclude per frame saja. Dengan adanya banyak frame dalam satu sesi kita bisa memastikan apakah sesi tersebut bermasalah atau tidak. Kalau dalam satu sesi memiliki banyak variasi / std cukup besar artinya sesi itu bermasalah. Namun apabila hanya sebagian/sedikit saja yang bermasalah mungkin ada kesalahan ketika start/stop scanning sehingga user tidak sengaja bergerak/berpindah posisi."*

**Implementasi `utils/data_qc_v3_frame.py`:**
1. Per sesi, hitung **median dan MAD** (Median Absolute Deviation) per fitur (21 fitur flatten dari geometry.json).
2. Frame outlier jika **fitur mana pun** melebihi k × MAD dari median sesi.
3. **≤50% outlier** → hanya rename frame (`_QC2_frame_XX`), sisanya tetap training.
4. **>50% outlier** → seluruh sesi invalid (`_QC2_YYYY...`).

**Hasil QC v3 (k=10, threshold=0.5):**

| Subjek | Frame Excluded | Rate | Pattern Dominan |
|---|---|---|---|
| reysa | 27/240 | 11.3% | High variance (single extreme outliers) |
| feby | 24/210 | 11.4% | Bimodal distribution |
| chrys | 22/200 | 11.0% | Knuckle detection fallback |
| yanuar | 16/200 | 8.0% | Mixed (1 entire session + 9 partial) |
| aisah | 14/200 | 7.0% | Single outliers |
| nola | 15/220 | 6.7% | finger_width_5 instability |
| gede | 13/200 | 6.5% | Mixed |
| taufik | 12/200 | 6.0% | Knuckle fallback |
| alji | 7/150 | 4.6% | Clean |
| fadhil | 7/150 | 4.6% | Clean |
| **rahmat** | **3/150** | **2.0%** | **Cleanest subject** |
| **Total** | **160/2,120** | **7.5%** | + 1 entire session (yanuar/20260513_092145) |

**Valid frames setelah QC v3: 1,869** (diverifikasi scanner `scan_dataset_frames`).

### Perubahan Dataset Scanner

`utils/dataset.py` diupdate:
- `scan_dataset()`: skip session folders `_QC2_*` dan `_QUARANTINE_*` (existing)
- `scan_dataset_frames()`: **tambahan** skip frame folders `_QC2_frame_*` inside each session

---

## Strategi Training v0.4.0: From Scratch vs Fine-tune

**Konteks:** v0.3.0 menggunakan strategi fine-tune dari hasil v0.2.0 (Triplet → ArcFace) untuk menghemat waktu. Pertanyaan natural untuk v0.4.0: bisakah kita melanjutkan dari checkpoint v0.3.0?

**Jawaban: tidak. v0.4.0 harus from scratch.** Alasan teknis:

1. **Patch F1.1 mengubah urutan inisialisasi modul di `encoder.py`.** Semua sub-modul (`geom_encoder`, `gam1`, `gam2`, `proj_with_geom`, `proj_no_geom`) sekarang selalu dibangun di `__init__` dengan urutan tetap. Konsekuensi: seed yang sama menghasilkan bobot awal yang **berbeda** dari v0.3.0. Yang penting, bobot awal antar 4 varian v0.4.0 sekarang **identik** (verified: 58/58 layer, max|Δ|=0).
2. **Skema parameter berubah.** v0.3.0 menyimpan `encoder.proj.*` (satu head). v0.4.0 punya `proj_with_geom.*` **dan** `proj_no_geom.*` (keduanya selalu ada di state_dict). Backward-compat loader hanya bisa migrate satu head — head sebelah random init.
3. **Dua varian baru tidak punya checkpoint sumber.** `gam_only` dan `fuse_only` adalah konfigurasi baru. Tidak ada titik fine-tune yang valid.

### Trade-off Opsi

| Opsi | Biaya | Klaim metodologis | Menjawab pertanyaan ilmiah inti |
|---|---|---|---|
| Fine-tune dari v0.3.0 (no_geom + with_geom) | ~30 ep × 2 | ✗ Mewarisi bias init unfair v0.3.0 — fix F1.1 tidak berdampak | **Tidak** — hanya mempercepat konvergensi titik akhir baseline lama |
| From scratch, 2 varian (no_geom + with_geom dengan init fair) | ~120 ep × 2 | ✓ Fair-ablation strict | ✓ Tapi tidak bisa decompose GAM vs fusion |
| **From scratch, 4 varian (Recommended)** | ~120 ep × 4 | ✓ Fair-ablation strict + ablasi modul | ✓ + bisa eksekusi decision branch (B1–B4) |

### Mengapa Fine-tune Menggugurkan Tujuan v0.4.0

Pertanyaan ilmiah utama: *"setelah init fair, apakah GeoAtt masih merugikan?"*

Jika kita fine-tune dari v0.3.0:
- Bobot awal pasangan with_geom/no_geom tetap warisan v0.3.0 yang init-unfair. Patch F1.1 jadi tidak terpakai.
- v0.3.0 sudah konvergen di basin lokal masing-masing. Fine-tune 20–30 epoch tidak cukup keluar dari basin tersebut.
- Hasil akhir akan sangat mirip v0.3.0 (with_geom kalah ~4%), dan kita **tidak akan tahu** apakah itu karena GeoAtt memang merugikan atau hanya gravitasi basin v0.3.0.

### Rekomendasi Eksekusi

**Default: from scratch, 4 varian, 5 seed.** Biayanya 4× v0.3.0 (~8 jam pada A100 jika 1 varian ~2 jam). Justifikasi:
- Satu-satunya jalan dengan klaim "fair ablation" yang sah.
- Decision branch (B1–B4) di F2.2 baru bermakna jika init benar-benar identik antar 4 varian.
- Hasil layak di-tag `v0.4.0-baseline` dengan reproduksibilitas penuh.

**Fallback anggaran ketat: from scratch, 2 varian** (no_geom + with_geom). Skip `gam_only`/`fuse_only` sampai hasil 2-arah menunjukkan gap masih ada. Risiko: kalau gap persisten, kita belum tahu sumber spesifiknya (GAM, fusion, atau keduanya) — investigasi terhenti di tingkat permukaan.

**Catatan untuk transparansi laporan v0.4.0:** sebutkan eksplisit bahwa v0.4.0 *tidak* fine-tune dari v0.3.0 — beda strategi dengan v0.3.0 yang fine-tune dari v0.2.0. Alasannya: di v0.3.0 perubahan hanya pada loss function (struktur model tetap), sehingga warm-start dari v0.2.0 valid. Di v0.4.0 struktur model berubah (F1.1) dan ablasi memerlukan init identik, sehingga warm-start gugur.

---

## Fase 2 — Rencana Tindak Lanjut (Belum Dijalankan)

### F2.1 Re-baseline 4-Arah dengan Init Fair (Prioritas 1)

**Tujuan:** verifikasi apakah gap with_geom vs no_geom tetap ada setelah init parity diperbaiki. Jika gap mengecil signifikan → masalah dominan adalah F1.1 (sudah diperbaiki). Jika tidak → lanjut ke F2.2.

**Eksperimen:** training penuh 5 seed (42, 123, 2026, 7, 31337) untuk 4 varian. Hyperparameter identik dengan v0.3.0-baseline (ArcFace m=0.5, s=30; batch 512; n_points 8192; phase1 100ep / phase2 30ep / phase3 20ep; lr 2e-3 → 2e-4).

```bash
python train.py --output_dir runs/no_geom/<ts>
python train.py --output_dir runs/with_geom/<ts>  --use-geom
python train.py --output_dir runs/gam_only/<ts>   --use-gam
python train.py --output_dir runs/fuse_only/<ts>  --use-geom-fusion
```

Lalu `collab/compare.ipynb` komparasi 4-arah dengan Wilcoxon paired + bootstrap CI + McNemar pooled.

### F2.2 Decision Branch Berdasar Hasil F2.1

- **B1 (gap hilang):** masalah dominan adalah init parity. Tutup investigasi; GeoAtt tidak terbukti merugikan setelah ablasi fair. Tulis v0.4.0-baseline report.
- **B2 (gam_only ≪ no_geom, fuse_only ≈ no_geom):** GAM penyebab utama. → F2.3 cross-attention GAM.
- **B3 (fuse_only ≪ no_geom, gam_only ≈ no_geom):** fusion concat penyebab utama. → F2.4 gated/FiLM fusion.
- **B4 (keduanya ≪ no_geom):** cara mengonsumsi geom secara umum bermasalah. → F2.5 feature engineering + auxiliary loss.

### F2.3 Cross-Attention GAM (Prioritas 2, branch B2)

Ganti GAM sederhana (sigmoid scale/shift) dengan **cross-attention** sejati: queries dari SA features, keys/values dari geom_emb yang di-broadcast. Lebih dalam, mampu menolak fitur geom yang noisy per-titik.

### F2.4 Gated Fusion / FiLM (Prioritas 2, branch B3)

Alternatif concat: FiLM modulation (`feat * γ(geom) + β(geom)`) atau gated fusion (`σ(geom_proj) * feat + (1-σ) * geom`). Mempertahankan ukuran fitur tetap 256 sehingga dropout & ArcFace tidak terkontaminasi.

### F2.5 Feature Engineering + Auxiliary Loss (Prioritas 3, branch B4)

- Tambah QC: tandai sesi dengan `finger_width_5` outlier per-subjek (z>3 dalam subject) sebagai unusable untuk geom branch.
- Tambah fitur turunan pose-invariant: rasio `finger_len_i / finger_len_3`, `palm_width / palm_height`, dst.
- Eksperimen: gunakan geom hanya sebagai **auxiliary loss** (predict identity langsung dari geom_emb dengan softmax CE), tidak di-concat ke main embedding.

### F2.6 Dropout Tuning (Pelengkap, prioritas 4)

Setelah arsitektur stabil:
- Turunkan `Dropout(p=0.3)` di proj head → 0.1.
- Atau tambahkan `Dropout(p=0.1)` simetris di tail `geom_encoder` agar regularisasi seimbang antar cabang.

---

## Target Metrik v0.4.0

| Metrik | v0.3.0-baseline no_geom | v0.3.0-baseline with_geom | Target v0.4.0 (with_geom, fair ablation) |
|---|---|---|---|
| Rank-1 mean | 99.82% | 95.82% | ≥ no_geom (no harm) atau >no_geom + 0.5% (small win) |
| Rank-1 std | 0.36% | 1.59% | ≤ no_geom std |
| EER | 0.03% | 2.76% | ≤ 0.10% |
| McNemar pooled (b vs c) | n/a | 23 vs 1 (no_geom menang) | b ≈ c (no significant winner) |
| Wilcoxon p (paired n=5) | n/a | 0.0625 | p > 0.10 (atau favoring with_geom) |

**Kriteria sukses minimum:** GeoAtt tidak terbukti merugikan secara statistik (McNemar p>0.05, bootstrap CI Δrank-1 melingkupi 0).
**Kriteria sukses maksimum:** with_geom > no_geom dengan p<0.05 dan CI tidak melingkupi 0.

---

## Catatan Sejarah & Hubungan dengan Plan Sebelumnya

- **v0.2.0-baseline:** Triplet loss, kedua varian buruk (~60% Rank-1). Hipotesis dominan: GeoAtt sebagai regularizer.
- **v0.3.0:** Ganti loss ke ArcFace. no_geom melompat ke 99.82%; with_geom 95.82%. Hipotesis lama runtuh.
- **v0.4.0 (rencana ini):** Diagnostik mengidentifikasi 4 dari 7 hipotesis. Tindakan langsung:
  - Perbaikan struktural fair-ablation (F1.x) — **sudah jalan**.
  - Re-baseline 4-arah (F2.1) — **menunggu eksekusi di GPU**.
  - Branching arsitektur (F2.3/F2.4/F2.5) — bergantung hasil F2.1.

Status kerja:
- [x] Skrip diagnostik & audit (D1–D5)
- [x] Patch encoder/siamese/train/evaluate untuk fair ablation
- [x] Re-verify init parity pasca-patch
- [x] Laporan `result_docs/20260517_064046/diagnostic_phase1.md`
- [ ] Jalankan training 4-arah multi-seed (F2.1)
- [ ] Komparasi 4-arah & decision branch (F2.2)
- [ ] Iterasi arsitektur (F2.3/F2.4/F2.5) — jika diperlukan
- [ ] Tag baru `v0.4.0-baseline` setelah F2.1 stabil

---

## Lampiran — File yang Dimodifikasi/Dibuat di Fase 1

**Source code (modifikasi):**
- `models/encoder.py` — RNG parity + flag use_gam/use_geom_fusion
- `models/siamese.py` — pass-through flag baru
- `train.py` — CLI ablasi baru
- `evaluate.py` — backward-compat checkpoint loader

**Skrip diagnostik (baru):**
- `utils/audit_init_parity.py`
- `utils/audit_geom_session_variance.py`
- `utils/audit_embedding_stats.py`
- `utils/eval_hard_probes.py`

**Output audit:**
- `eval_results/audits/20260517_062539/init_parity.json` (pre-patch)
- `eval_results/audits/20260517_063518/init_parity.json` (post-patch, 58/58 identical)
- `eval_results/audits/20260517_062722/` (D2 geom variance)
- `eval_results/audits/20260517_063240/embedding_stats.json` (D3)
- `eval_results/audits/20260517_063331/hard_probes.json` (D5)

**Laporan:**
- `result_docs/20260517_064046/diagnostic_phase1.md` — laporan diagnostik lengkap
- `result_docs/20260517_060023/GeoAtt_PointNet_Palm_Recognition_Evaluation_Report_v2.md` — laporan baseline v0.3.0 (referensi)
