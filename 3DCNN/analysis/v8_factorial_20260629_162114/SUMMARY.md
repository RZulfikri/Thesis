# v8 Factorial (alignment × loss) — Summary

**Tanggal**: 20260629_162114
**Desain**: faktorial {softmax, arcface} × {A0..A5}; closed-set gallery=train, probe=HOLDOUT (tanpa LOSO); multi-frame fusion N5M5; 5 seed.

## Tabel EER% N5M5 (mean±std) — baris alignment, kolom loss

| alignment | softmax | arcface |
|---|---|---|
| A0 raw_ply | 12.06±2.31 | 15.03±2.17 |
| A1 align_center | 4.42±2.37 | 1.39±1.38 |
| A2 align_centerscale | 0.03±0.06 | 0.00±0.00 |
| A3 canonical_npy | 0.03±0.06 | 0.00±0.00 |
| A4 align_pca_robust | 1.67±1.86 | 0.58±0.93 |
| A5 align_anatomical | 0.48±0.97 | 0.45±0.54 |

**Baseline** (A0, softmax) = 12.06% EER.
**A*** (alignment terbaik, kolom arcface) = **A2**.

## Klaim
- **H1 (normalisasi → robustness)**: bandingkan antar-baris (lihat rotation_sensitivity.png; alignment ter-normalisasi datar di kedua loss, A0 raw naik).
- **H2 (ArcFace → accuracy)**: bandingkan antar-kolom (softmax vs arcface) di tabel di atas.

_Dihasilkan otomatis oleh v8_lib.analyze()_