# v8 Factorial (alignment × loss) — Summary

**Tanggal**: 20260629_182718
**Desain**: faktorial {softmax, arcface} × {A0..A5}; closed-set gallery=train, probe=HOLDOUT (tanpa LOSO); multi-frame fusion N5M5; 5 seed.

## Tabel EER% N5M5 (mean±std) — baris alignment, kolom loss

| alignment | softmax | arcface |
|---|---|---|
| A0 raw_ply | 11.85±3.17 | 17.64±2.44 |
| A1 align_center | 4.58±2.09 | 1.45±1.19 |
| A2 align_centerscale | 0.06±0.12 | 0.00±0.00 |
| A3 canonical_npy | 0.03±0.06 | 0.00±0.00 |
| A4 align_pca_robust | 1.33±1.33 | 0.52±0.88 |
| A5 align_anatomical | 0.48±0.90 | 0.58±0.53 |

**Baseline** (A0, softmax) = 11.85% EER.
**A_accuracy** (EER pose-kanonik terendah, kolom arcface) = **A2**.
**A_robust** (worst-case EER terendah pada rotasi θ>0, kolom arcface) = **A4**.
> ⚠️ A_accuracy bisa ≠ A_robust: alignment yang sempurna di pose-kanonik (mis. center/scale atau PCA polos) dapat **runtuh saat dirotasi** (lihat rotation_sensitivity.png). Untuk **deployment & klaim H1, pakai A_robust**.

## Signifikansi softmax vs arcface (paired t-test, 5 seed)
> EER@N5M5 mentok lantai (≈0) → uji-t underpowered. **Bukti H2 utama = d′** (punya headroom); EER@N1M1 (single-frame) sbg channel kedua. `improve_arcface`>0 ⇒ arcface lebih baik.

### d' N5M5

| alignment | softmax | arcface | improve(arcface) | arcface lebih baik | p-value | sig<0.05 |
|---|---|---|---|---|---|---|
| A0 | 1.89 | 1.80 | -0.09 | tidak | 0.574 | — |
| A1 | 3.28 | 6.63 | +3.34 | ya | 0.000 | ✔ |
| A2 | 4.20 | 9.49 | +5.28 | ya | 0.001 | ✔ |
| A3 | 4.09 | 8.87 | +4.78 | ya | 0.000 | ✔ |
| A4 | 4.02 | 8.02 | +4.00 | ya | 0.000 | ✔ |
| A5 | 3.93 | 8.11 | +4.19 | ya | 0.003 | ✔ |

### EER N1M1

| alignment | softmax | arcface | improve(arcface)% | arcface lebih baik | p-value | sig<0.05 |
|---|---|---|---|---|---|---|
| A0 | 18.73 | 20.48 | -1.76 | tidak | 0.187 | — |
| A1 | 5.91 | 2.85 | +3.06 | ya | 0.145 | — |
| A2 | 0.67 | 0.00 | +0.67 | ya | 0.180 | — |
| A3 | 0.79 | 0.76 | +0.03 | ya | 0.749 | — |
| A4 | 3.00 | 1.70 | +1.30 | ya | 0.175 | — |
| A5 | 0.97 | 1.58 | -0.61 | tidak | 0.275 | — |

### EER N5M5

| alignment | softmax | arcface | improve(arcface)% | arcface lebih baik | p-value | sig<0.05 |
|---|---|---|---|---|---|---|
| A0 | 11.85 | 17.64 | -5.79 | tidak | 0.038 | ✔ |
| A1 | 4.58 | 1.45 | +3.12 | ya | 0.025 | ✔ |
| A2 | 0.06 | 0.00 | +0.06 | ya | 0.374 | — |
| A3 | 0.03 | 0.00 | +0.03 | ya | 0.374 | — |
| A4 | 1.33 | 0.52 | +0.82 | ya | 0.290 | — |
| A5 | 0.48 | 0.58 | -0.09 | tidak | 0.870 | — |

## Klaim
- **H1 (normalisasi → robustness)**: lihat rotation_sensitivity.png + A_robust; alignment rotation-robust (A4) datar di semua θ, sedangkan A0/A1/A2 runtuh & A3 paku di 90°.
- **H2 (ArcFace → accuracy)**: bukti utama via **d′** (ArcFace ~2× separabilitas di semua representasi ternormalisasi) + EER@N1M1; EER@N5M5 mentok lantai (tak informatif).

_Dihasilkan otomatis oleh v8_lib.analyze()_