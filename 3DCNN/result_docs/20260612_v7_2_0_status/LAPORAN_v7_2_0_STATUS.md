# LAPORAN v7.2.0 — Status Run Colab: GAGAL (belum ada hasil eksperimen)

# ARTIFACT: LAPORAN_v7_2_0_STATUS
# Created by: Analysis Agent
# Date: 2026-06-12
# Related to: v7.2.0 representation ablation (R1/R2/R3)
# Status: FINAL

**Tanggal pemeriksaan:** 2026-06-12
**Run yang diperiksa:** Colab `v7_2_0_repr_ablation.ipynb`, run #1 (2026-06-06) dan run #2 (2026-06-08)
**Status v7.2.0:** ❌ **TRAINING GAGAL — 0 dari 15 run menghasilkan model.** Tidak ada angka EER/latency yang bisa dilaporkan. Laporan ini adalah **laporan status & akar masalah**, bukan laporan hasil.

---

## 1. Ringkasan Eksekutif

1. **Pesan commit Colab menyesatkan.** Dua commit berbunyi *"v7.2.0: training C1 (raw_ply) complete (5 seeds)"*, tetapi **kelima seed C1 crash sebelum epoch pertama**. Pesan "complete" dihasilkan otomatis setelah loop seed selesai, terlepas dari sukses/gagal (sudah diperbaiki, §5).
2. **Tidak ada satupun artefak hasil.** `runs/v7_2_0/C1/seed_*/` hanya berisi `splits.json`, `normalizer.json`, `train_stdout.log`, dan event tensorboard 88 byte (kosong). **Tidak ada `best.pth`, `train_log.json`, `perf.json`.** Direktori C2/C3, `eval_results/v7_2_0/`, dan `analysis/v7_2_0_*` **tidak ada sama sekali**.
3. **Akar masalah C1: `output.ply` tidak ada di checkout Colab.** `.gitignore` mengecualikan `3DCNN/dataset/**/*.ply`; hanya **424 dari 2.131** PLY yang ter-push (sisa tak sengaja dari era v7.1.0). Loader Open3D mengembalikan cloud kosong untuk file hilang (hanya warning) → crash membingungkan di `_sample_points` (`ValueError: a must be greater than 0`).
4. **C2/C3 sebenarnya siap jalan** — `cnn_input.npy` dan `cnn_input_fps.npy` lengkap 2.131/2.131 di repo. Loop berhenti setelah C1 sehingga keduanya belum pernah dimulai.
5. **Blocker satu-satunya:** push **1.707 file `output.ply` yang hilang** (≈1,5 GB) dari mesin lokal — hanya bisa dilakukan user (§6, langkah L1).

---

## 2. Kronologi Dua Run yang Gagal

| Run | Tanggal | Commit | Gejala | Akar masalah |
|---|---|---|---|---|
| #1 | 2026-06-06 | `30746d04` | 5 seed C1 crash diam-diam; tanpa log (output training tidak disimpan) | `import open3d` gagal — setup cell (disalin dari v7.1.1 yang berbasis `.npy`) tidak meng-install open3d. Didiagnosa & diperbaiki di `3776c8fb` (install open3d + tee `train_stdout.log` + pesan `[GAGAL]`). |
| #2 | 2026-06-08 | `1169526a` | 5 seed C1 crash; kali ini **ter-log** berkat fix run #1 | `output.ply` tidak ada di checkout untuk mayoritas frame (§3). Open3D hanya mengeluarkan warning `Read PLY failed: unable to open file`, cloud kosong diteruskan, lalu meledak saat preload dataset. |

Traceback identik di kelima seed run #2 (`runs/v7_2_0/C1/seed_*/train_stdout.log`):

```
[Open3D WARNING] Read PLY failed: unable to open file: .../dataset/yanuar/20260513_092117/frame_09/output.ply
PalmFrameDataset (PRELOAD): precomputing 7839 augmented variants ...
  File ".../utils/dataset.py", line 65, in _sample_points
    idx = np.concatenate([np.arange(N), np.random.choice(N, n - N, replace=True)])
ValueError: a must be greater than 0 unless no samples are taken
```

Training berhenti **sebelum epoch pertama** (event tensorboard 88 byte = header saja). Setelah C1, tidak ada artefak maupun commit C2/C3 — eksekusi tidak berlanjut (runtime berakhir / dihentikan setelah C1).

---

## 3. Akar Masalah: PLY Ter-gitignore, Push Tidak Pernah Dilengkapi

`.gitignore` berisi aturan (ditulis saat PLY masih dianggap "visualisasi, bisa regenerate"):

```
3DCNN/dataset/**/*.ply
```

v7.2.0 menjadikan `output.ply` **input training** (R1 raw), dan keputusan "repo berat" memang **ditunda dari v7.1.1 ke v7.2.0** (lihat `VERSION.md` §v7.1.1) — tetapi langkah push PLY **tidak pernah dieksekusi**. Yang ada di repo hanyalah sisa historis: 424 PLY ikut ter-commit lewat `git add -A` Colab pada era v7.1.0 (commit `f47b1716`, 2026-05-29); karena file yang **sudah ter-track tidak terkena gitignore**, file-file itu ikut ter-update saat dataset regen di-push per subjek (2026-06-04) — sementara PLY subjek lain tertolak oleh gitignore.

Kelengkapan `output.ply` di repo per subjek (dari `git ls-files`, total **424/2.131**):

| Subjek | PLY ada | PLY hilang | Subjek | PLY ada | PLY hilang |
|---|---|---|---|---|---|
| aisah | 0/200 | 200 | nola | 0/221 | 221 |
| alji | 143/150 | 7 | rahmat | 147/150 | 3 |
| chrys | 0/200 | 200 | reysa | 0/250 | 250 |
| fadhil | 134/150 | 16 | taufik | 0/200 | 200 |
| feby | 0/210 | 210 | yanuar | 0/200 | 200 |
| gede | 0/200 | 200 | **Total** | **424** | **1.707** |

Catatan: 424 PLY yang ada **valid untuk R1** (binary little-endian, ada normals `nx,ny,nz`, ~20K titik — diverifikasi sampel), berasal dari dataset regen yang sama. Masalahnya murni **kelengkapan**, bukan kualitas.

Dataset lengkap (2.131/2.131 PLY ber-normals) **ada di mesin lokal user** (`3DCNN/dataset/`, mirror dari `3DRegistration/result_frames_v720/` — lihat VERSION.md §v7.2.0 hasil regenerasi).

---

## 4. Dampak ke Rencana v7.2.0

- **C1 (R1 raw_ply): GAGAL** — terblokir data; retrain otomatis begitu PLY lengkap (`best.pth` tidak ada → tidak ter-skip).
- **C2 (R2 canonical_npy) & C3 (R3 fps_npy): BELUM JALAN** — data sudah lengkap di repo; tidak terblokir apa pun.
- **C0 (anchor v7.1.1): aman** — reuse, tanpa run ulang.
- Hipotesis H4–H8 dan seluruh metrik §10.4/§10.6 IMPROVEMENT_PLAN belum punya satu pun titik data.
- Kerugian: ±2 sesi GPU Colab terbuang untuk crash preload (< 1 menit/seed); tidak ada kontaminasi hasil.

---

## 5. Perbaikan yang Sudah Dilakukan (commit ini)

| # | File | Perbaikan | Tujuan |
|---|---|---|---|
| P1 | `3DCNN/utils/dataset.py` | `_load_ply_xyz_normals`: **fail-fast** — `FileNotFoundError` dengan pesan jelas bila `output.ply` tidak ada; `ValueError` bila file terbaca tapi 0 titik | Error langsung menunjuk akar masalah, bukan `ValueError np.random.choice` di hilir. Diuji: frame tanpa PLY → raise; frame ber-PLY → load `(20012, 6) float32`; R2 tidak terpengaruh. |
| P2 | `collab/v7_2_0_repr_ablation.ipynb` | Cell baru **§5b Preflight**: verifikasi kelengkapan file representasi per config sebelum loop training → hasilkan `RUNNABLE_CONFIGS`. Config tidak lengkap **di-SKIP dengan peringatan** (rincian missing per subjek); abort hanya bila tidak ada config lengkap *(update 2026-06-12: semula abort total, diubah jadi filter agar C2/C3 tidak ikut terblokir C1)* | Tidak membakar GPU bila dataset tidak lengkap, dan config yang siap tetap jalan. Disimulasikan pada checkout saat ini: `C1 KURANG 1707/2131 → SKIP`, hasil `RUNNABLE_CONFIGS = [C2, C3]`. |
| P3 | `collab/v7_2_0_repr_ablation.ipynb` | Pesan `git_save` jujur: `complete` hanya bila 5/5 seed menghasilkan `best.pth`, selain itu `FAILED (k/5 seeds OK)` | Mencegah salah baca "collab sudah selesai" seperti pada run #1/#2. |

Fix run #1 (`3776c8fb`: install open3d + tee log + pesan `[GAGAL]`) sudah masuk sebelumnya dan terbukti bekerja — run #2 meninggalkan log yang bisa didiagnosa.

---

## 6. Langkah Selanjutnya (urutan disarankan)

- **L1 — (USER, mesin lokal) Push PLY lengkap** — satu-satunya blocker C1:
  ```bash
  cd ~/Projects/Thesis
  git add -f '3DCNN/dataset/'*'/'*'/frame_'*'/output.ply'   # force — bypass gitignore
  git commit -m 'dataset v7.2.0: lengkapi output.ply (R1) 2131/2131'
  git push origin main   # atau branch yang di-merge notebook Colab
  ```
  Estimasi bobot: 1.707 file × ~0,87 MB ≈ **1,5 GB** tambahan (total PLY ≈ 1,9 GB; jauh di bawah limit 100 MB/file GitHub, tetapi clone Colab makin berat). Pertimbangkan sekalian menghapus/mempersempit aturan `3DCNN/dataset/**/*.ply` di `.gitignore` agar tidak ada partial-tracking lagi.
- **L2 — Jalankan C2/C3 lebih dulu (otomatis, tidak perlu L1).** Preflight §5b kini men-skip C1 dan menjalankan C2/C3 yang datanya lengkap (10 run langsung mulai) — tidak perlu mengubah urutan `REPR_CONFIGS` manual. Bisa dikerjakan paralel dengan L1.
- **L3 — Run ulang notebook** setelah L1: preflight harus **3/3 config siap**, C1 retrain otomatis (seed C2/C3 yang sudah punya `best.pth` di-skip). Pastikan commit hasil berbunyi `complete`, bukan `FAILED`.
- **L4 — Lanjut eval & analysis** (§7–§13 notebook) → baru setelah itu `LAPORAN_v7_2_0.md` hasil eksperimen bisa ditulis menggantikan laporan status ini.

---

## 7. Lampiran — Artefak yang Diperiksa

```
runs/v7_2_0/C1/seed_{0,42,123,2024,31337}/
├── splits.json            # ada (ditulis sebelum preload)
├── normalizer.json        # ada
├── train_stdout.log       # ada (run #2) — traceback identik 5 seed
└── tensorboard/events.*   # 88 byte = kosong (2 file: run #1 & #2)
# TIDAK ADA: best.pth, train_log.json, perf.json
# TIDAK ADA: runs/v7_2_0/{C2,C3}/, eval_results/v7_2_0/, analysis/v7_2_0_*
```

Sumber bukti: log `runs/v7_2_0/C1/seed_*/train_stdout.log`; `git ls-files '3DCNN/dataset/**/output.ply'`; commit `30746d04`, `3776c8fb`, `1169526a`, `f47b1716`; VERSION.md §v7.1.1/§v7.2.0; IMPROVEMENT_PLAN_v7.0.0.md §10.
