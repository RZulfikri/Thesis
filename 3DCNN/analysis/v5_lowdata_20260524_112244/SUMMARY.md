# v5.0.0 Low-Data Regime — Hasil Eksperimen

**Tanggal**: 20260524_112244
**Setup**: 10 subjek × 15 sesi × 1 median frame = 150 frames
**Varian**: no_geom (PointNet++ murni) vs with_geom (GeoAtt full)
**Seeds**: [42, 123, 2024, 7, 31337, 0, 1, 2, 3, 4]

## Test EER (Primary Metric)

| variant   |   ('eer', 'mean') |   ('eer', 'std') |   ('eer', 'min') |   ('eer', 'max') |
|:----------|------------------:|-----------------:|-----------------:|-----------------:|
| no_geom   |             0.05  |           0      |             0.05 |             0.05 |
| with_geom |             0.425 |           0.2264 |             0.1  |             0.75 |

## Holdout EER (Generalization)

| variant   |   ('eer', 'mean') |   ('eer', 'std') |   ('eer', 'min') |   ('eer', 'max') |
|:----------|------------------:|-----------------:|-----------------:|-----------------:|
| no_geom   |             0.035 |           0.0266 |              0   |           0.0833 |
| with_geom |             0.375 |           0.1043 |              0.2 |           0.5333 |

## Statistical Test (Wilcoxon paired)

**Test EER**:
- with_geom mean: 0.4250 ± 0.2264
- no_geom   mean: 0.0500 ± 0.0000
- Δ (with-no): +0.3750
- Wilcoxon stat: 0.0, p: 0.0020

**Holdout EER**:
- with_geom mean: 0.3750 ± 0.1043
- no_geom   mean: 0.0350 ± 0.0266
- Δ (with-no): +0.3400
- Wilcoxon stat: 0.0, p: 0.0020
