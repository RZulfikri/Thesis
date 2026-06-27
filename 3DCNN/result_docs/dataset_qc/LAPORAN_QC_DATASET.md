# Laporan Quality Control Dataset

**Tanggal audit**: 2026-05-31 (revisi besar 2026-06-03)  
**Dataset**: 11 subjek × 15–25 sesi × ~10 frame/sesi (214 sesi raw total)  
**Tujuan**: Memastikan hanya frame yang lolos QC point-cloud yang masuk ke proses training dan evaluasi

---

## 0. UPDATE 2026-06-03 — Temuan `knuckle_fallback` & Redefinisi QC

Audit awal (di bawah) hanya menghitung `invalid_frame.json` dan menyimpulkan satu-satunya
issue adalah `scan_distance_out_of_range`. **Itu tidak lengkap.** Saat regenerasi dataset
untuk v7.2.0 ditemukan:

- **955 dari 1.946 frame (49%)** ber-`is_valid=False` di `geometry.json`, **semuanya** karena
  `knuckle_fallback` — issue yang tidak tercatat di audit awal karena tidak menulis
  `invalid_frame.json`.
- **~930 frame** ber-`is_valid=False` ini **lolos diam-diam ke training v7.1.0**: filter QC
  3DCNN (`_frame_passes_qc`) hanya memeriksa keberadaan `invalid_frame.json`, bukan field
  `is_valid` di `geometry.json`.
- 114 `invalid_frame.json` yang dihitung audit awal adalah **legacy** flag scan_distance pada
  ambang lama 200mm; `extract_geometry.py` sekarang memakai ambang [150, 450]mm sehingga
  **0 frame** di data saat ini sebenarnya kena scan_distance.

**Keputusan (user, 2026-06-03): pakai QC POINT-CLOUD, bukan strict `is_valid`.**
`knuckle_fallback` adalah kegagalan ekstraksi **fitur geometri hand-crafted** (landmark buku
jari), sedangkan CNN memakai **point cloud**. Spot-check membuktikan frame knuckle_fallback
justru lebih padat (19.6K vs 17.9K titik), scan distance dalam rentang, dan secara visual
telapak + 5 jari lengkap — tak terbedakan dari frame clean.

**Definisi frame valid untuk CNN (berlaku mulai v7.2.0):**
1. PLY ter-isolasi dengan ≥ `min_points` titik, **DAN**
2. `scan_distance_mm` ∈ [150, 450]mm.
3. `knuckle_fallback` → field `warnings` (non-gating). `fingertip_fallback` /
   `fingers_too_close` tetap gate.

**Konsekuensi (hasil regenerasi final, 2026-06-03):** dataset CNN regen = **214 sesi, 2.131
frame, 100% valid** (0 gate issue, 1.021 `warnings: knuckle_fallback` non-gate, 0
`invalid_frame.json`). Per subjek: aisah 200, alji 150, chrys 200, fadhil 150, feby 210,
gede 200, nola 221, rahmat 150, reysa 250, taufik 200, yanuar 200. Setiap frame kini punya
`output.ply` (xyz+normals, R1), `cnn_input.npy` (R2), `cnn_input_fps.npy` (8192,6 — R3).
Dataset di-mirror penuh ke `3DCNN/dataset/`.

**Catatan validasi PCA:** ~291/1.836 frame (~16%) berbeda dari dataset lama murni karena
ambiguitas kanonikalisasi PCA (flip 180° sumbu Y / resolusi axis), **bukan** perbedaan
point cloud. Keputusan: pakai dataset regen utuh + recompute basis fresh, `pca_align`
tidak diubah. Detail: `IMPROVEMENT_PLAN_v7.0.0.md` §10.10.

Tabel per-subjek di bawah (Bagian 2–5) mencerminkan **interpretasi lama** (legacy
`invalid_frame.json` + ambang scan_distance 200mm) dan **sudah usang** — angka final ada di
paragraf di atas. Dipertahankan sebagai jejak audit historis.

---

## 1. Faktor Penentu QC

Sebuah frame dinyatakan **lolos QC** jika memenuhi **semua** syarat berikut:

| # | Syarat | File yang dicek | Keterangan |
|---|---|---|---|
| 1 | Data lengkap — punya `cnn_input.npy` | `frame_XX/cnn_input.npy` | Point cloud sudah diproses (PCA + unit-sphere) |
| 2 | Metadata lengkap — punya `geometry.json` | `frame_XX/geometry.json` | 23 fitur geometri telapak tangan |
| 3 | Tidak di-flag QC — **tidak** punya `invalid_frame.json` | `frame_XX/invalid_frame.json` | Flag otomatis dari pipeline akuisisi |

**Satu-satunya jenis issue yang ditemukan**: `scan_distance_out_of_range`  
→ Scanner terlalu dekat ke telapak tangan saat capture (< 200 mm; rentang optimal: 200–450 mm)  
→ Depth data tidak akurat pada jarak terlalu dekat karena karakteristik sensor TrueDepth

**Sebuah sesi dinyatakan valid** jika punya minimal 1 frame yang lolos QC.

---

## 2. Ringkasan Per Subjek

| Subjek | Total Frame | Frame Invalid | Frame Valid | Valid% | Sesi Valid | Keterangan |
|---|---|---|---|---|---|---|
| aisah | 126 | 21 | 105 | 83.3% | 15 | s05 seluruhnya invalid (0/10) |
| alji | 143 | 7 | 136 | 95.1% | 15 | s10 parsial (4/9 valid) |
| chrys | 141 | 0 | 141 | 100.0% | 15 | Bersih |
| fadhil | 143 | 12 | 131 | 91.6% | **14** | s14 seluruhnya invalid (0/10) |
| feby | 125 | 0 | 125 | 100.0% | 15 | Bersih |
| gede | 146 | 0 | 146 | 100.0% | 15 | Bersih |
| nola | 137 | 0 | 137 | 100.0% | 15 | Bersih |
| rahmat | 147 | 5 | 142 | 96.6% | 15 | Parsial di beberapa sesi train |
| reysa | 177 | 19 | 158 | 89.3% | 15 | s12 dan s13 holdout seluruhnya invalid |
| taufik | 149 | 0 | 149 | 100.0% | 15 | Bersih |
| yanuar | 153 | 0 | 153 | 100.0% | 15 | Bersih |
| **TOTAL** | **1.587** | **64** | **1.523** | **95.9%** | **164/165** | fadhil kehilangan 1 sesi |

---

## 3. Detail Sesi Bermasalah

Sesi yang memiliki frame invalid (terurut per dampak):

### Kritis — Seluruh Sesi Invalid (0 frame valid)

| Subjek | Split | Sesi | Valid/Total | Dampak |
|---|---|---|---|---|
| **aisah** | train | s05 | 0/10 | Sesi training dilewati; aisah efektif 7 sesi train |
| **fadhil** | holdout | s14 | 0/10 | Sesi holdout dilewati; fadhil hanya 2 holdout session |
| **reysa** | holdout | s12 | 0/5 | Sesi holdout dilewati |
| **reysa** | holdout | s13 | 0/10 | Sesi holdout dilewati; reysa hanya 1 holdout session (s14) |

### Parsial — Sebagian Frame Invalid

| Subjek | Split | Sesi | Valid/Total | Dampak |
|---|---|---|---|---|
| aisah | train | s06 | 6/9 | 3 frame dilewati, sesi tetap valid |
| aisah | val | s08 | 2/10 | Hanya 2 frame valid — **di bawah threshold N=5** |
| alji | test | s10 | 4/9 | 5 frame dilewati — **di bawah threshold M=5** |
| alji | holdout | s14 | 6/8 | 2 frame dilewati, sesi tetap valid |
| fadhil | train | s00 | 9/10 | 1 frame dilewati, sesi tetap valid |
| fadhil | test | s11 | 7/8 | 1 frame dilewati, sesi tetap valid |
| rahmat | train | s00 | 7/10 | 3 frame dilewati, sesi tetap valid |
| rahmat | train | s04 | 8/10 | 2 frame dilewati, sesi tetap valid |
| reysa | train | s00 | 9/10 | 1 frame dilewati, sesi tetap valid |
| reysa | holdout | s14 | 6/9 | 3 frame dilewati, sesi tetap valid |

---

## 4. Dampak ke Split Training/Evaluasi

### Mode Single-Frame (1 median frame per sesi)

| Split | Sebelum fix QC | Setelah fix QC | Perubahan |
|---|---|---|---|
| train | 88 frame | 88 frame | Tidak berubah (median frame dipilih dari frame valid) |
| val | 22 frame | 22 frame | Tidak berubah |
| test | 22 frame | 22 frame | Tidak berubah |
| holdout | 33 frame | **32 frame** | −1 (fadhil kehilangan s14) |

### Mode All-Frame (semua frame valid per sesi — untuk v7.2.0)

| Split | Frame valid tersedia | Catatan |
|---|---|---|
| train | ~740–760 frame | Variasi karena jumlah frame valid per sesi berbeda-beda |
| val | ~190–200 frame | |
| test | ~185–195 frame | alji s10 hanya 4 frame (di bawah M=5) |
| holdout | ~275–285 frame | reysa hanya 1 sesi holdout valid (s14 dengan 6 frame) |

### Sesi di Bawah Threshold untuk Multi-Frame N=5, M=5

| Subjek | Split | Sesi | Frame Valid | Masalah |
|---|---|---|---|---|
| aisah | val | s08 | 2 | Tidak cukup untuk N=5 di validasi |
| alji | test | s10 | 4 | Tidak cukup untuk M=5 di evaluasi |

**Penanganan**: pakai semua frame yang tersedia (sample with replacement jika dibutuhkan N/M tepat, atau gunakan min(available, N/M)).

---

## 5. Kondisi Holdout Per Subjek

Holdout adalah evaluasi temporal paling ketat (sesi paling baru). Kondisi setelah fix QC:

| Subjek | Sesi Holdout Valid | Frame Valid | Status |
|---|---|---|---|
| aisah | 3 (s12–s14) | 25 | ✅ Normal |
| alji | 3 (s12–s14) | 26 | ✅ Normal |
| chrys | 3 (s12–s14) | 26 | ✅ Normal |
| **fadhil** | **2 (s12–s13)** | **20** | ⚠️ s14 dilewati |
| feby | 3 (s12–s14) | 27 | ✅ Normal |
| gede | 3 (s12–s14) | 30 | ✅ Normal |
| nola | 3 (s12–s14) | 28 | ✅ Normal |
| rahmat | 3 (s12–s14) | 30 | ✅ Normal |
| **reysa** | **1 (s14 saja)** | **6** | ❌ Hanya 1 sesi, 6 frame |
| taufik | 3 (s12–s14) | 24 | ✅ Normal |
| yanuar | 3 (s12–s14) | 20 | ✅ Normal |

**Catatan reysa**: dengan hanya 6 frame valid di 1 sesi holdout, multi-frame probe M=5 masih bisa dijalankan (6 ≥ 5). Tapi representasi holdout reysa sangat terbatas — hasil evaluasi holdout untuk reysa harus diinterpretasi dengan hati-hati.

---

## 6. Perubahan Kode

File yang diubah: `utils/dataset_lowdata.py`

**Tambahan fungsi `_frame_passes_qc()`**:
```python
def _frame_passes_qc(frame_dir: Path) -> bool:
    return (
        (frame_dir / "cnn_input.npy").exists()
        and (frame_dir / "geometry.json").exists()
        and not (frame_dir / "invalid_frame.json").exists()
    )
```

`_session_is_valid()` dan `_get_valid_frames()` diperbarui untuk menggunakan fungsi ini.

**Sebelum fix**: frame dengan `invalid_frame.json` yang juga punya `cnn_input.npy` masuk ke training (24 dari 114 frame invalid lolos filter lama).

**Setelah fix**: semua frame dengan `invalid_frame.json` dikecualikan tanpa terkecuali.

---

## 7. Rekomendasi

1. **Reysa holdout**: dokumentasikan sebagai keterbatasan dataset di thesis — hanya 1 sesi holdout valid dari 3 yang direncanakan.
2. **Fadhil holdout**: kehilangan 1 sesi (s14), masih punya 2 sesi holdout yang valid.
3. **Threshold MF**: untuk sesi dengan frame valid < N atau M, gunakan semua frame yang tersedia (tidak perlu di-skip seluruh sesi).
4. **Tidak ada re-akuisisi**: keterbatasan ini bersifat fixed — cukup dicatat sebagai konteks di laporan eksperimen.

---

_Audit dilakukan pada 2026-05-31. Sumber: `dataset/*/frame_*/invalid_frame.json`._
