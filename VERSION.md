# VERSION — Riwayat Dataset & Pipeline

Catatan versi lintas-komponen (3DRegistration pipeline + 3DCNN dataset/eksperimen).
Entri terbaru di atas. Detail metodologi penuh: `3DCNN/IMPROVEMENT_PLAN_v7.0.0.md`.

---

## v7.1.1 — Re-run v7.1.0 di dataset regen (SELESAI, 2026-06-05)

**Status:** ✅ SELESAI | **Gate v7.1.1 → v7.2.0: LOLOS** (dengan catatan)

### Hasil (analysis `v7_1_1_20260605_083050`)
- **Anchor tereproduksi:** `arcface_m04` MF N5M5 EER **1.14% ± 1.18%** (regen) vs **1.32% ± 1.42%** (v7.1.0) — di dalam pita error.
- **Semua 8 varian membaik** (Δ EER negatif) di dataset regen → regenerasi tidak meregresi baseline.
- **Winner numerik bergeser** `arcface_m04 → cosface` (0.36% ± 0.37%) — **tidak signifikan** (selisih < std; 5 loss teratas tak terbedakan pada 11 subjek / 5 seed).
- **Keputusan:** C0 v7.2.0 = `arcface_m04` (anchor) dipertahankan demi konsistensi; `cosface` dibawa sebagai pembanding sekunder.
- Laporan lengkap: `3DCNN/result_docs/20260605_083050_v7_1_1/LAPORAN_v7_1_1.md`.

### Kenapa
Dataset regen v7.2.0 ~16% beda dari dataset lama (ambiguitas kanonikalisasi PCA, bukan
cleaning) + churn komposisi frame. Angka v7.1.0 adalah **pra-regenerasi** dan tidak boleh
dicampur dengan v7.2.0. v7.1.1 mengulang loss sweep + multi-frame fusion **di dataset regen**
supaya seluruh rangkaian (loss → representasi) berada di **satu dataset** → menutup celah
metodologis. Hasil v7.1.1 jadi **acuan resmi** untuk v7.2.0.

### Spec (delta dari v7.1.0 — 3 perubahan)
| Aspek | v7.1.0 | v7.1.1 |
|---|---|---|
| Dataset | lama (pra-regen) | **regen** (`3DCNN/dataset/`) |
| Seeds | 10 | **5** (`0, 42, 123, 2024, 31337`) |
| Evaluasi | Test + Holdout + **LOSO** | **Test + Holdout** (LOSO dibuang) |
| Loss varian | 8 | 8 (faithful re-run) |
| Training frames / protokol | median / MF N5M5 | sama |

- **LOSO dibuang**: itu evaluasi open-set/stranger-rejection — di luar scope thesis yang
  **closed-set identification**. Test + Holdout adalah protokol yang benar.
- **5 seed**: standar umum; tujuan = konfirmasi ranking + anchor, bukan klaim signifikansi.
  Lapor mean±std + effect size, bukan p-value.
- **Seed sama dipakai di v7.2.0** agar C0 (anchor) sebanding lurus.

### Ringan untuk Colab
Hanya butuh **R2** (`cnn_input.npy` + `geometry.json`) — TIDAK butuh `output.ply`/`cnn_input_fps.npy`.
Maka keputusan repo berat (fps ~418 MB + PLY ter-gitignore) **ditunda ke v7.2.0**; v7.1.1 cukup
commit `cnn_input.npy`+`geometry.json` ke branch `colab` lalu push.

### Langkah
F0 commit+push R2 → F1 notebook (seed=5, matikan LOSO) → F2 run 8 varian × 5 seed (~40 run, di
Colab) → F3 analisa → **F4 Gate**: konfirmasi arcface_m04 tetap juara + EER anchor (sanity-check
vs v7.1.0 ~1.3%) → F5 `LAPORAN_v7.1.1.md`. Detail: `IMPROVEMENT_PLAN_v7.0.0.md` §9b.

**Gate v7.1.1 → v7.2.0:** lolos bila ranking loss stabil & EER sebanding → C0 v7.2.0 =
arcface_m04 v7.1.1 (tanpa run ulang). Gagal bila winner berubah / EER meleset jauh → tinjau ulang.

---

## v7.2.0 — Regenerasi dataset & representation ablation (2026-06-03)

**Status:** ✅ **SELESAI** — dataset regen + implementasi (E2–E11) + 15 run training (A100-80GB, on-spec) + evaluasi penuh + laporan.

### Update 2026-06-14 — hasil final eval (analysis `v7_2_0_20260614_033631`)
Kanonikalisasi terbukti penentu utama akurasi:
- **R1 raw_ply: EER N5M5 ~9,9%** (AUC 0,894) — runtuh; sensitif rotasi (ΔEER@180° +0,36).
- **R2 canonical & R3 fps: EER ≈ 0%** (AUC 1,0, di lantai pengukuran) — tak terbedakan akurasi.
- **Keputusan: kanonikalisasi WAJIB; R3 (fps_npy) = pilihan deployment Pareto-optimal**
  (akurasi = R2, disk 399,8 MB = 4,6× < R1 / 2,3× < R2, sedikit tercepat).
- **Temuan:** R2/R3 invarian rotasi KECUALI singularitas di 90° (ambiguitas tukar sumbu PCA);
  determinism R2≈R3 inkonklusif (didominasi noise cudnn).
- Laporan final: `3DCNN/result_docs/20260614_v7_2_0/LAPORAN_v7_2_0.md`.

### Update 2026-06-12 — hasil pemeriksaan run Colab
Kedua run C1 (raw_ply) crash sebelum epoch pertama di 5 seed; C2/C3 belum pernah jalan.
- Run #1: `import open3d` gagal (setup cell tidak install) — diperbaiki `3776c8fb`.
- Run #2: **`output.ply` tidak ada di checkout Colab** — `.gitignore` mengecualikan
  `3DCNN/dataset/**/*.ply`; hanya 424/2.131 PLY ter-push (sisa era v7.1.0: alji 143,
  fadhil 134, rahmat 147; 8 subjek lain 0). Open3D mengembalikan cloud kosong →
  `ValueError` di `_sample_points` saat preload.
- Perbaikan: fail-fast PLY hilang di `utils/dataset.py`; cell preflight kelengkapan
  representasi + pesan commit jujur (complete vs FAILED) di notebook.
- **Blocker:** push 1.707 `output.ply` yang hilang (≈1,5 GB) dari mesin lokal (`git add -f`).
- Detail: `3DCNN/result_docs/20260612_v7_2_0_status/LAPORAN_v7_2_0_STATUS.md`.

### Ringkasan
Re-ekstraksi raw depth dan re-proses seluruh pipeline 3DRegistration agar menghasilkan
**tiga representasi konsisten** untuk ablation v7.2.0 (R1 raw PLY vs R2 canonical NPY vs
R3 pre-FPS NPY), plus redefinisi QC dari strict `is_valid` ke **QC point-cloud**.

### Hasil regenerasi (final)
| Metrik | Nilai |
|---|---|
| Sesi diproses | 214 |
| Frame total | **2.131** (0 gagal, 0 QC-fail) |
| Frame valid (gate point-cloud) | **2.131 / 2.131** (100%) |
| `quality_issues` (gate) | 0 |
| `warnings: knuckle_fallback` (non-gate) | 1.021 |
| `output.ply` ber-normals (R1) | 2.131 / 2.131 |
| `cnn_input.npy` (R2) | 2.131 / 2.131 |
| `cnn_input_fps.npy` (8192,6 — R3) | 2.131 / 2.131 |
| `invalid_frame.json` | 0 |

Per subjek: aisah 200, alji 150, chrys 200, fadhil 150, feby 210, gede 200, nola 221,
rahmat 150, reysa 250, taufik 200, yanuar 200.

Dataset di-mirror penuh dari `3DRegistration/result_frames_v720/` ke `3DCNN/dataset/` (3.1 GB).

### Keputusan QC: point-cloud, bukan strict `is_valid`
`knuckle_fallback` adalah kegagalan ekstraksi **fitur geometri hand-crafted** (landmark buku
jari), sedangkan CNN memakai **point cloud**. Spot-check: frame knuckle_fallback justru lebih
padat (19.6K vs 17.9K titik), scan distance dalam rentang, telapak+5 jari lengkap secara
visual. Maka knuckle_fallback dipindah ke `warnings` (non-gating); `fingertip_fallback` /
`fingers_too_close` tetap gate. Definisi valid: PLY ter-isolasi ≥ `min_points` titik **DAN**
`scan_distance_mm` ∈ [150, 450].

### Temuan kanonikalisasi PCA (validasi vs dataset lama)
~291 dari 1.836 frame (~16%) **berbeda** dari dataset lama — **bukan** karena cleaning
(point set identik, sorted-xyz diff 0.000000) melainkan **ambiguitas kanonikalisasi PCA** di
`pca_align()`:
1. Mayoritas: flip 180° terhadap sumbu Y (x,z dinegasikan, y sama) — sign sumbu **X tidak
   di-kanonikalisasi** (langsung dari SVD `Vt`); hanya sumbu Y punya disambiguasi (median-Y).
2. Sebagian: perbedaan resolusi axis lebih besar (`range0≈range1` borderline / median-Y flip
   terbalik) — pose kanonik benar-benar beda.

→ Klaim awal "regen bit-identik dengan v7.1.0" **DIBATALKAN**.

**Keputusan (user, opsi A):** pakai dataset regen **utuh**, **recompute basis/normalizer
fresh**, **JANGAN reuse C0/basis v7.1.0**, jangan campur old+new. `pca_align()` **TIDAK
diubah** (R2 tetap definisi v7.1.0 yang dikunci agar ablation tidak terkontaminasi).
Instabilitas X-sign/axis dicatat sebagai **keterbatasan representasi R2 canonical** untuk
dibahas di thesis; perbaikan = future work.

### Perubahan kode
| File | Perubahan |
|---|---|
| `3DRegistration/extract_geometry.py` | `knuckle_fallback` → field `warnings` (non-gating), bukan `quality_issues`; `is_valid` kini hanya gate point-cloud |
| `3DRegistration/lib/single_frame.py` | `output.ply` menyimpan **normals** (R1 = 6 channel xyz+normals) |
| `3DRegistration/preprocess_for_cnn.py` | default `--n_points` 1024 → 8192 |
| `3DRegistration/make_fps.py` (baru) | generate `cnn_input_fps.npy` @8192 dari `cnn_input.npy` via open3d `farthest_point_down_sample` (~0.4 dtk/frame) |
| `3DRegistration/validate_v720.py` (baru) | validasi dataset regen: per-subjek, kelengkapan artefak, PLY normals, bit-identity vs lama, distribusi warnings, shape FPS |

### Dokumentasi diperbarui
- `3DCNN/IMPROVEMENT_PLAN_v7.0.0.md` — §9b spec v7.1.1 (re-run, anchor), §10.2 koreksi bit-identity, §10.3 C0=anchor v7.1.1, §10.7 E0/E1 ✅, §10.10 hasil final + temuan PCA
- `3DCNN/result_docs/dataset_qc/LAPORAN_QC_DATASET.md` — §0 angka final + catatan PCA
- `README.md` — tabel artefak (output.ply xyz+normals R1, cnn_input_fps.npy 8192 R3)

### Konfigurasi ablation (untuk training, belum dijalankan)
| ID | Representasi | Training frames | Catatan |
|---|---|---|---|
| C0 | R2 canonical | 1 median frame | = arcface_m04 dari v7.1.1 (anchor) — tanpa run terpisah |
| C1 | R1 raw PLY | semua 10 frame | uji tanpa kanonikalisasi |
| C2 | R2 canonical | semua 10 frame | baseline baru |
| C3 | R3 pre-FPS | semua 10 frame | uji FPS vs random sampling |

Loss dikunci `arcface_m04`; multi-frame N=5/M=5; closed-set; **5 seed** (sama dgn v7.1.1).
C0 = anchor dari v7.1.1 → hanya C1/C2/C3 = **15 run baru** (≈ 2,5 jam Colab), di luar ~40 run v7.1.1.

**Dua dimensi hasil:** (1) **akurasi** (Rank-1, EER, d-prime, DET/ROC) dan (2) **kecepatan & resource**
(waktu training/epoch, total training, peak GPU mem, latency enroll/probe, throughput, disk). PLY vs
NPY vs FPS dibandingkan pada keduanya, disajikan sebagai trade-off akurasi↔biaya (lihat IMPROVEMENT_PLAN §10.6).

### Pending (belum dikerjakan)
- **Keputusan commit/repo:** swap menambah ~1.916 `cnn_input_fps.npy` (~418 MB) + 2.642 modifikasi + 536 hapus di `3DCNN/dataset/`. Belum di-commit.
- **R1 PLY tidak ter-track:** `.gitignore` mengabaikan `3DCNN/dataset/**/*.ply`, padahal R1 butuh `output.ply` di Colab — perlu di-un-ignore atau jalur lain (Drive).
- **E2–E11:** loader (`frame_mode="all"`, `repr_mode` raw_ply/fps_npy) + `train.py` flags + notebook `collab/v7_repr_ablation.ipynb` + analisis akurasi (E7); grafik DET/ROC + loss-curve (E8); instrumentasi kecepatan/resource (E9); probe isolasi rotasi + determinisme (E10); sintesis Pareto trade-off (E11). Daftar metrik final (🟢+🟡): IMPROVEMENT_PLAN §10.4/§10.6.

---

## v7.1.0 — Baseline canonical + ArcFace (selesai sebelumnya)

Baseline GeoAtt-PointNet++ dengan R2 (canonical NPY), loss `arcface_m04`, single median
frame. Gate-2 PASS. Detail: `3DCNN/IMPROVEMENT_PLAN_v7.0.0.md`. Dataset v7.1.0 kini
**digantikan** oleh dataset regen v7.2.0 (lihat temuan PCA di atas — tidak bit-identik).

> Riwayat versi lebih lama (V1–V6) terdokumentasi di `docs/reports/REPORT_THESIS_V1_V4/`
> dan seri `3DCNN/IMPROVEMENT_PLAN_v*.md`.
