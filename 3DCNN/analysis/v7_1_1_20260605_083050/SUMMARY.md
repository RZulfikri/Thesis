# v7.0.0 Low-Data Regime — Summary

**Tanggal**: 20260605_083050
**Setup**: 11 subjek × 15 sesi × 10 frame/sesi = 165 frame
**Protokol primer**: multi-frame fusion N=5, M=5, strategy=mean
**Seeds**: [0, 42, 123, 2024, 31337]

## Single-Frame vs Multi-Frame EER

| Variant | SF EER | MF EER (5,5) | Δ |
|---------|--------|--------------|---|
| standard | 0.0273±0.0545 | 0.0205±0.0253 | -0.0068 |
| arcface_m03 | 0.0545±0.0530 | 0.0095±0.0125 | -0.0450 |
| arcface_m04 | 0.0545±0.0668 | 0.0114±0.0118 | -0.0432 |
| arcface_m05 | 0.0091±0.0182 | 0.0168±0.0206 | +0.0077 |
| arcface_s64 | 0.0000±0.0000 | 0.0164±0.0180 | +0.0164 |
| cosface | 0.0545±0.0445 | 0.0036±0.0037 | -0.0509 |
| subcenter | 0.0000±0.0000 | 0.0123±0.0203 | +0.0123 |
| hybrid | 0.0000±0.0000 | 0.0091±0.0141 | +0.0091 |

## Gate-2: PASS

_Dihasilkan otomatis oleh v7_1_1_multiframe_compare.ipynb_