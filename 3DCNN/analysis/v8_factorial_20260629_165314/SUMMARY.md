# v8 Factorial (alignment × loss) — Summary

**Tanggal**: 20260629_165314
**Desain**: faktorial {softmax, arcface} × {A0..A5}; closed-set gallery=train, probe=HOLDOUT (tanpa LOSO); multi-frame fusion N5M5; 5 seed.

## Tabel EER% N5M5 (mean±std) — baris alignment, kolom loss

| alignment | softmax | arcface |
|---|---|---|
| A0 raw_ply | 12.06±3.34 | 15.55±3.79 |
| A1 align_center | 3.76±2.24 | 0.64±0.92 |
| A2 align_centerscale | 0.00±0.00 | 0.00±0.00 |
| A3 canonical_npy | 0.03±0.06 | 0.00±0.00 |
| A4 align_pca_robust | 1.30±1.42 | 0.55±0.87 |
| A5 align_anatomical | 0.12±0.24 | 0.45±0.54 |

**Baseline** (A0, softmax) = 12.06% EER.
**A*** (alignment terbaik, kolom arcface) = **A2**.

## Klaim
- **H1 (normalisasi → robustness)**: bandingkan antar-baris (lihat rotation_sensitivity.png; alignment ter-normalisasi datar di kedua loss, A0 raw naik).
- **H2 (ArcFace → accuracy)**: bandingkan antar-kolom (softmax vs arcface) di tabel di atas.

_Dihasilkan otomatis oleh v8_lib.analyze()_