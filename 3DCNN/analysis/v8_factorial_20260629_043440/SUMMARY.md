# v8 Factorial (alignment × loss) — Summary

**Tanggal**: 20260629_043440
**Desain**: faktorial {softmax, arcface} × {A0..A5}; closed-set gallery=train, probe=HOLDOUT (tanpa LOSO); multi-frame fusion N5M5; 5 seed.

## Tabel EER% N5M5 (mean±std) — baris alignment, kolom loss

| alignment | softmax | arcface |
|---|---|---|
| A0 raw_ply | 12.94±2.78 | 16.91±1.42 |
| A1 align_center | 2.52±1.41 | 1.15±1.05 |
| A2 align_centerscale | 0.06±0.12 | 0.06±0.12 |
| A3 canonical_npy | 0.03±0.06 | 0.00±0.00 |
| A4 align_pca_robust | 2.18±2.61 | 0.24±0.21 |
| A5 align_anatomical | 0.18±0.29 | 0.52±0.53 |

**Baseline** (A0, softmax) = 12.94% EER.
**A*** (alignment terbaik, kolom arcface) = **A3**.

## Klaim
- **H1 (normalisasi → robustness)**: bandingkan antar-baris (lihat rotation_sensitivity.png; alignment ter-normalisasi datar di kedua loss, A0 raw naik).
- **H2 (ArcFace → accuracy)**: bandingkan antar-kolom (softmax vs arcface) di tabel di atas.

_Dihasilkan otomatis oleh v8_lib.analyze()_