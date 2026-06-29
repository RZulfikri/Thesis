# v8 Factorial (alignment × loss) — Summary

**Tanggal**: 20260629_172431
**Desain**: faktorial {softmax, arcface} × {A0..A5}; closed-set gallery=train, probe=HOLDOUT (tanpa LOSO); multi-frame fusion N5M5; 5 seed.

## Tabel EER% N5M5 (mean±std) — baris alignment, kolom loss

| alignment | softmax | arcface |
|---|---|---|
| A0 raw_ply | 10.82±4.04 | 16.73±3.84 |
| A1 align_center | 2.48±0.48 | 1.79±1.47 |
| A2 align_centerscale | 0.00±0.00 | 0.00±0.00 |
| A3 canonical_npy | 0.03±0.06 | 0.00±0.00 |
| A4 align_pca_robust | 1.12±1.32 | 0.58±0.93 |
| A5 align_anatomical | 0.15±0.30 | 0.52±0.54 |

**Baseline** (A0, softmax) = 10.82% EER.
**A_accuracy** (EER pose-kanonik terendah, kolom arcface) = **A2**.
**A_robust** (worst-case EER terendah pada rotasi θ>0, kolom arcface) = **A4**.
> ⚠️ A_accuracy bisa ≠ A_robust: alignment yang sempurna di pose-kanonik (mis. center/scale atau PCA polos) dapat **runtuh saat dirotasi** (lihat rotation_sensitivity.png). Untuk **deployment & klaim H1, pakai A_robust**.

## Signifikansi softmax vs arcface (paired t-test, 5 seed, N5M5)

| alignment | EER sm | EER arc | Δ(sm−arc) | arcface lebih baik | p-value | sig<0.05 |
|---|---|---|---|---|---|---|
| A0 | 10.82 | 16.73 | -5.91 | tidak | 0.003 | ✔ |
| A1 | 2.48 | 1.79 | +0.70 | ya | 0.487 | — |
| A2 | 0.00 | 0.00 | +0.00 | tidak | 1.000 | — |
| A3 | 0.03 | 0.00 | +0.03 | ya | 0.374 | — |
| A4 | 1.12 | 0.58 | +0.55 | ya | 0.508 | — |
| A5 | 0.15 | 0.52 | -0.36 | tidak | 0.289 | — |

## Klaim
- **H1 (normalisasi → robustness)**: lihat rotation_sensitivity.png + A_robust; alignment rotation-robust (A4) datar di semua θ, sedangkan A0/A1/A2 runtuh & A3 paku di 90°.
- **H2 (ArcFace → accuracy)**: bandingkan antar-kolom (softmax vs arcface) + tabel signifikansi.

_Dihasilkan otomatis oleh v8_lib.analyze()_