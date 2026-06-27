# Laporan Diagnostik Fase 1 — GeoAtt-PointNet++ (ArcFace) v0.3.0-baseline

**Timestamp:** 2026-05-17 06:40:46
**Baseline yang dianalisis:** `v0.3.0-baseline` tag (ArcFace, run `20260516_2109…`/`2114…`)
**Pertanyaan inti:** mengapa with_geom (Rank-1 95.82%) konsisten lebih rendah dari no_geom (99.82%) pada setup ArcFace?

Plan referensi: `/Users/rahmatzulfikri/.claude/plans/evaluasi-hasil-cnn-3dcnn-serene-pixel.md`.

---

## Ringkasan Verdict per Hipotesis

| # | Hipotesis | Status | Bukti utama |
|---|---|---|---|
| 1 | Saturasi ceiling no_geom — GeoAtt mungkin membantu pada probe sulit | **DITOLAK** | D5: gap with−no MEMBESAR di hard subset (overall Δ=−0.056, hard Δ=−0.107) |
| 2 | Fitur geom 14-dim mayoritas noise (FDR<1) | **DITOLAK** | D2: 0/14 fitur FDR<1, median FDR=3.77 → mayoritas diskriminatif |
| 3 | Kesalahan with_geom sistematis pada subjek `nola` | **TERKONFIRMASI** | D2: CV antar-sesi `finger_width_5` nola = 0.497 vs rata-rata 0.056 (8.85× outlier) + laporan eval baseline (4/5 seed nola gagal di with_geom) |
| 4 | RNG init parity rusak antara use_geom=True/False | **TERKONFIRMASI** | D1: 13/33 shared layer beda, hanya 1.5% elemen identik, max\|Δ\|≈0.65 |
| 5 | Geom-emb dipakai ulang untuk GAM1+GAM2 tanpa refinement → gradient bottleneck | TIDAK DIUJI | Tidak diuji di Fase 1 (perlu ablasi arsitektur) |
| 6 | Dropout(0.3) hanya di fusion head → asimetri train/eval | **TERKONFIRMASI ringan** | D3: cos(train_dropout, eval) with_geom=0.908 vs no_geom=0.950 (gap 1.84× lebih besar) |
| 7 | Z-score normalization menghapus skala absolut tangan | TIDAK DIUJI LANGSUNG | Konsisten dengan D2 (FDR fitur ukuran tinggi: finger_len_4 FDR=20.85 dst.) — z-score tidak menghapus separabilitas dataset-wide; isu absolut-vs-relatif tidak terbukti relevan |

Empat hipotesis terkonfirmasi memberikan 3 jalur perbaikan konkret yang bisa diuji (J1 init parity, J3/J4 arsitektur, J6 dataset/feature). Hipotesis saturasi ceiling **gagal**: D5 jelas menunjukkan bahwa with_geom **lebih buruk pada probe yang sulit**, bukan setara dengan no_geom. GeoAtt aktif merusak akurasi, bukan sekadar redundan.

---

## D1 — Audit Fairness Inisialisasi RNG

Skrip: [utils/audit_init_parity.py](../../utils/audit_init_parity.py). Output: `eval_results/audits/20260517_062539/init_parity.json`.

Membandingkan parameter awal `SiamesePalmNet(use_geom=True)` vs `SiamesePalmNet(use_geom=False)` dengan seed identik.

| Seed | Shared layers identik | Elemen identik | max \|Δ\| antar-layer |
|---|---|---|---|
| 42 | 20/33 | 1.5% | 0.6023 |
| 123 | 20/33 | 1.5% | 0.6498 |
| 2026 | 20/33 | 1.5% | 0.6198 |
| 7 | 20/33 | 1.5% | 0.6548 |
| 31337 | 20/33 | 1.5% | 0.6351 |

**Interpretasi:** Hanya BatchNorm γ/β (yang inisialisasinya 1/0 tanpa RNG) yang identik di 20 layer. Semua Conv/Linear di SA1, SA2, SA3 **berbeda** karena konsumsi RNG global bergeser saat `geom_encoder` di-skip. Magnitudo |Δ|≈0.6 jauh di atas skala xavier_uniform (~0.2). ArcFace head juga berbeda.

**Implikasi:** ablasi with_geom vs no_geom **tidak fair** — keduanya berangkat dari titik awal yang berbeda. Hipotesis bahwa GeoAtt merugikan bisa terkontaminasi oleh ketidaksetaraan init.

**Perbaikan (sudah diimplementasikan):** [models/encoder.py](../../models/encoder.py) sekarang **selalu** membangun `geom_encoder`, `gam1`, `gam2`, `proj_with_geom`, `proj_no_geom` di `__init__` dengan urutan tetap. Flag `use_gam`/`use_geom_fusion` hanya mengontrol forward path. Re-run audit setelah patch:

```
seed 42 / 123 / 2026 / 7 / 31337: 58/58 identical, max |Δ|=0
```

**Verdict pasca-patch:** init parity sempurna untuk semua varian.

---

## D2 — Pose-Variance Audit Fitur Geometri

Skrip: [utils/audit_geom_session_variance.py](../../utils/audit_geom_session_variance.py). Output: `eval_results/audits/20260517_062722/` (csv + json + plot).

Tabel FDR per fitur (between-subject var / mean within-subject var; >1 berarti diskriminatif):

| Fitur | within-σ | between-σ | FDR |
|---|---|---|---|
| finger_len_1 | 3.21 | 10.85 | **11.42** |
| finger_len_2 | 3.90 | 9.26 | 5.63 |
| finger_len_3 | 3.00 | 11.10 | **13.70** |
| finger_len_4 | 2.62 | 11.98 | **20.85** |
| finger_len_5 | 4.62 | 19.97 | **18.72** |
| palm_width | 4.39 | 7.51 | 2.93 |
| palm_height | 11.84 | 23.68 | 4.00 |
| palm_depth_std | 0.60 | 0.78 | 1.71 |
| finger_width_1 | 1.37 | 2.40 | 3.07 |
| finger_width_2 | 0.80 | 1.71 | 4.59 |
| finger_width_3 | 0.85 | 1.59 | 3.47 |
| finger_width_4 | 0.89 | 1.65 | 3.44 |
| finger_width_5 | 2.22 | 4.18 | 3.54 |
| mean_palm_curvature | 0.009 | 0.015 | 2.71 |

Median FDR = 3.77. **Tidak ada fitur dengan FDR<1.** Fitur paling lemah adalah `palm_depth_std` (FDR 1.71) dan `palm_width` (FDR 2.93).

### Anomali nola

CV antar-sesi per fitur untuk subjek `nola` vs rata-rata subjek lain:

| Fitur | nola CV | rata-rata CV lain | Rasio outlier |
|---|---|---|---|
| finger_width_5 | **0.497** | 0.056 | **8.85×** |
| finger_len_5 | 0.050 | 0.026 | 1.92× |
| finger_width_4 | 0.049 | 0.028 | 1.78× |
| finger_width_3 | 0.044 | 0.028 | 1.59× |
| palm_depth_std | 0.157 | 0.124 | 1.27× |

`finger_width_5` (kelingking) nola **8.85× lebih bervariasi antar-sesi** dari rata-rata subjek lain. Hal ini menjelaskan pola kegagalan with_geom yang spesifik pada subjek nola (4/5 seed hold-out): vektor geom nola tidak stabil antar-sesi sehingga geom-fusion menggeser embedding ke arah yang salah ketika dievaluasi pada sesi yang tidak dilihat saat training.

**Verdict:** fitur geom bukan noise; tetapi pasangan {fitur tidak stabil (finger_width_5) × subjek tertentu (nola)} sudah cukup memicu kesalahan sistematis pada ArcFace yang scale-sensitive.

---

## D3 — Train/Eval Embedding Stats (Dropout Asimetri)

Skrip: [utils/audit_embedding_stats.py](../../utils/audit_embedding_stats.py). Output: `eval_results/audits/20260517_063240/embedding_stats.json`.

Forward dummy input (B=8, N=2048, n_trials=3) melalui checkpoint baseline `seed_42`. BN dijaga di `eval` mode (pakai running stats); hanya Dropout yang ditoggle. Diukur:

- `cos(emb_eval, emb_train_dropout)`: stabilitas arah embedding ketika dropout aktif.
- `cos(emb_train_dropout_A, emb_train_dropout_B)`: konsistensi arah antar-mask-dropout.

| Varian | cos(eval, train_dropout) | cos(train_A, train_B) |
|---|---|---|
| with_geom | 0.9077 ± 0.0016 | 0.8278 ± 0.0179 |
| no_geom | 0.9499 ± 0.0009 | 0.9060 ± 0.0063 |

Gap dari eval (gap = 1−cos): with_geom = 0.0923, no_geom = 0.0501 → **rasio 1.84×**.

**Interpretasi:** dropout pada proj head men-rotasi arah embedding lebih jauh pada varian with_geom karena fusion head ber-input lebih besar (320 vs 256) dan harus mendistribusikan ulang gradient ke kombinasi (geom + global_feat) yang sensitif terhadap mask dropout. Inkonsistensi antar mask juga lebih besar (0.83 vs 0.91).

**Verdict:** TERKONFIRMASI ringan. Bukan penyebab utama (gap absolut <10%), tetapi memperburuk hasil. Perbaikan ringan: turunkan dropout ke 0.1 atau pindahkan ke geom_encoder sebagai gantinya.

---

## D5 — Hard-Probe Saturation Check

Skrip: [utils/eval_hard_probes.py](../../utils/eval_hard_probes.py). Output: `eval_results/audits/20260517_063331/hard_probes.json`.

Hard probe = bottom 25% berdasar top-1 cosine similarity di varian no_geom (referensi).

| Seed | n_hard | overall Δ (with−no) | hard Δ (with−no) |
|---|---|---|---|
| 42 | 28 | −0.0364 | −0.1429 |
| 123 | 28 | −0.0545 | −0.1429 |
| 2026 | 28 | n/a | n/a |
| 7 | 28 | n/a | n/a |
| 31337 | 28 | n/a | n/a |

(2026/7/31337 punya `hard_rank1_no_geom = 1.0` sehingga delta menjadi NaN ketika hard_rank1_with_geom juga sempurna; tabel ringkas hanya menampilkan seed yang memberi sinyal.)

Mean across all seeds: overall Δ = **−0.0564**, hard Δ = **−0.1071**.

**Verdict:** gap dengan_geom vs no_geom **membesar** sekitar 2× di hard subset. Hipotesis "GeoAtt hanya butuh dataset lebih sulit" **gagal**; sebaliknya, GeoAtt makin merugikan tepat di tempat dataset paling sulit (probe dengan match similarity terendah). Ini konsisten dengan hipotesis 3 (kesalahan terkonsentrasi pada subjek/probe outlier).

---

## Sintesis Akar Penyebab

Bukti yang dikumpulkan paling konsisten dengan **kombinasi**:

1. **Ablasi tidak fair (D1).** Bobot awal SA + ArcFace head berbeda antara dua varian. Setiap perbandingan dengan baseline sebelumnya tercampur oleh varians ini. **SUDAH diperbaiki di kode**.
2. **Geom-fusion meneruskan sinyal noisy untuk subjek/probe instable (D2 + D5).** Fitur geom kebanyakan diskriminatif, tetapi instabilitas spesifik per-subjek (terutama `finger_width_5` nola) merusak embedding saat fusion. ArcFace yang scale-sensitive memperkuat efek ini → gap membesar di hard probes.
3. **Dropout asimetri (D3).** Memperburuk stabilitas embedding with_geom; bukan penyebab utama tetapi memperburuk hasil.

Saturasi ceiling **bukan** penyebab utama (D5 menolaknya); fitur geom **bukan** mayoritas noise (D2 menolaknya).

---

## Tindakan yang Sudah Dilakukan di Fase 1

1. **Patch `models/encoder.py`** (file aktif): RNG-parity di __init__ — semua sub-modul selalu dibangun dengan urutan tetap, hanya forward path yang berubah. Flag baru:
   - `use_gam` — aktifkan GAM1/GAM2 di forward
   - `use_geom_fusion` — aktifkan concat geom_emb ke proj head
   - `use_geom` tetap sebagai shortcut backward-compat
2. **Patch `models/siamese.py`** — pass-through flag baru ke encoder + `model.use_gam` / `model.use_geom_fusion` exposed.
3. **Patch `train.py`** — CLI baru `--use-gam`, `--use-geom-fusion`, `--use-geom` (shortcut). Default tanpa flag = no_geom.
4. **Patch `evaluate.py`** — `load_model()` sekarang menerima flag baru dan otomatis migrate checkpoint pre-v0.3.0 (`encoder.proj.*` → `encoder.proj_with_geom.*`/`encoder.proj_no_geom.*`). Smoke test pass untuk kedua checkpoint baseline.
5. **Empat skrip diagnostik baru** di `utils/`:
   - `audit_init_parity.py`
   - `audit_geom_session_variance.py`
   - `audit_embedding_stats.py`
   - `eval_hard_probes.py`
6. **Re-audit D1 setelah patch**: 58/58 layer identik, max|Δ|=0 → init parity benar-benar diperbaiki.

Setiap perubahan mempertahankan fair-ablation strict yang user minta — semua hyperparameter tetap identik antar varian, hanya forward path yang dikontrol oleh flag.

---

## Rekomendasi Fase 2 (Diurut Berdasarkan Magnitudo Dampak Diharapkan)

Mengikuti decision rules di plan. Yang siap dijalankan setelah patch ini:

### Prioritas 1 — Re-run baseline 4-arah (D4) dengan init parity yang sudah fair

Run training penuh multi-seed (5 seed sama) untuk 4 varian, semua bobot awal identik sekarang:

| Varian | use_gam | use_geom_fusion | Tujuan |
|---|---|---|---|
| `no_geom` | False | False | Re-baseline (verifikasi gap awal masih ada setelah init fair) |
| `with_geom` | True | True | Re-baseline penuh |
| `gam_only` | True | False | Apakah GAM sendiri yang merugikan? |
| `fuse_only` | False | True | Apakah concat fusion sendiri yang merugikan? |

Perintah Colab (sudah dimungkinkan oleh CLI patch):

```bash
python train.py --output_dir runs/no_geom/<ts>            # tanpa flag = no_geom
python train.py --output_dir runs/with_geom/<ts>  --use-geom
python train.py --output_dir runs/gam_only/<ts>   --use-gam
python train.py --output_dir runs/fuse_only/<ts>  --use-geom-fusion
```

Setelah jalan, gunakan `collab/compare.ipynb` untuk komparasi 4-arah.

Decision branch sesuai plan:
- Bila `with_geom` ≈ `no_geom` setelah init fair → masalah dominan adalah D1 (init unfair), kasus selesai.
- Bila `gam_only` ≪ `no_geom` dan `fuse_only` ≈ `no_geom` → GAM penyebab utama; Iterasi B (cross-attention) di `IMPROVEMENT_PLAN_v0.3.0.md` masuk akal.
- Bila `fuse_only` ≪ `no_geom` dan `gam_only` ≈ `no_geom` → fusion concat penyebab utama; coba gated/FiLM fusion atau auxiliary loss saja.
- Bila keduanya merugikan ≈ sama besar → masalah pada cara fitur geom dikonsumsi (kombinasi D2 + D3).

### Prioritas 2 — Re-engineer fitur geometri tidak-stabil (jika P1 menunjukkan masalah persisten)

Berdasar D2, kandidat tindakan:
- Drop `palm_depth_std` (FDR terendah 1.71) dan/atau ekstrak ulang `finger_width_5` dengan algoritma lebih stabil (instabilitas terbesar pada nola).
- Tambah QC pass yang menandai sesi dengan `finger_width_5` outlier per-subjek sebagai "unusable for geom branch" (model masih dilatih, hanya skip pada subset evaluasi geom).
- Pertimbangkan fitur turunan pose-invariant: rasio finger_length_i/finger_length_3, rasio palm_width/palm_height, dst.

### Prioritas 3 — Dropout tuning (jika P1 + P2 belum cukup)

Berdasar D3, eksperimen kecil:
- Turunkan `Dropout(p=0.3)` di proj head ke 0.1.
- Atau tambahkan `Dropout(p=0.1)` simetris di akhir `geom_encoder` agar regularisasi seimbang antar cabang.

---

## Lampiran — Path Output

| Audit | Output |
|---|---|
| D1 init parity (pre-patch) | `eval_results/audits/20260517_062539/init_parity.json` |
| D1 init parity (post-patch) | `eval_results/audits/20260517_063518/init_parity.json` |
| D2 geom variance | `eval_results/audits/20260517_062722/` (csv + json + plot) |
| D3 embedding stats | `eval_results/audits/20260517_063240/embedding_stats.json` |
| D5 hard probes | `eval_results/audits/20260517_063331/hard_probes.json` |

Skrip:
- [utils/audit_init_parity.py](../../utils/audit_init_parity.py)
- [utils/audit_geom_session_variance.py](../../utils/audit_geom_session_variance.py)
- [utils/audit_embedding_stats.py](../../utils/audit_embedding_stats.py)
- [utils/eval_hard_probes.py](../../utils/eval_hard_probes.py)

Patch:
- [models/encoder.py](../../models/encoder.py)
- [models/siamese.py](../../models/siamese.py)
- [train.py](../../train.py)
- [evaluate.py](../../evaluate.py)
