# v6.0.0 Low-Data Regime — Hasil Eksperimen

**Tanggal**: 20260525_152213
**Setup**: 10 subjek × 15 sesi × 1 median frame = 150 frames
**Varian**: standard (PointNet++ + Triplet) vs arcface (PointNet++ + ArcFace m=0.5, s=30)
**Seeds**: [42, 123, 2024, 7, 31337, 0, 1, 2, 3, 4]

## Test EER (Primary Metric)

| variant   |   ('eer', 'mean') |   ('eer', 'std') |   ('eer', 'min') |   ('eer', 'max') |
|:----------|------------------:|-----------------:|-----------------:|-----------------:|
| arcface   |             0.06  |           0.0316 |             0.05 |             0.15 |
| standard  |             0.065 |           0.0474 |             0.05 |             0.2  |

## Holdout EER (Generalization)

| variant   |   ('eer', 'mean') |   ('eer', 'std') |   ('eer', 'min') |   ('eer', 'max') |
|:----------|------------------:|-----------------:|-----------------:|-----------------:|
| arcface   |            0.015  |           0.0123 |                0 |           0.0333 |
| standard  |            0.0233 |           0.0274 |                0 |           0.0667 |

## Statistical Test (Wilcoxon paired, arcface vs standard)

**Test EER**:
- arcface  mean: 0.0600 ± 0.0316
- standard mean: 0.0650 ± 0.0474
- Δ (arc - std): -0.0050
- Wilcoxon stat: 1.0, p: 1.0000

**Holdout EER**:
- arcface  mean: 0.0150 ± 0.0123
- standard mean: 0.0233 ± 0.0274
- Δ (arc - std): -0.0083
- Wilcoxon stat: 14.0, p: 0.6328
