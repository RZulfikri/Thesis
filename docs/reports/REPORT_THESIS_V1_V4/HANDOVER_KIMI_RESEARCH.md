# Handover: GeoAtt-PointNet++ Palm Recognition Thesis — Complete Timeline & Results

**Prepared for:** Kimi Research Review  
**Date:** 2026-05-24  
**Tag:** `v5.0.1`  
**Repository:** `RZulfikri/Thesis` (branch `colab`)  

---

## 1. Research Question (Evolution)

**Original (V1–V3):**
> Does geometric attention (GeoAtt) improve palm identification accuracy?

**Current (V5.0.0–V5.0.1):**
> Does GeoAtt provide beneficial inductive bias in a **low-data enrollment regime** (1 sample per session)?

**Why the pivot:** V4 discovered 3 fatal experimental biases that invalidated all previous results. The existing dataset lacks temporal diversity to fix the split leakage. Rather than fix the split, we reframed the thesis to a question that (a) does not require temporal split, (b) is more relevant for practical biometric enrollment, and (c) focuses on inductive bias rather than memorization capacity.

---

## 2. Executive Summary Table

| Version | Date | Focus | Best Rank-1 | Loss | Seeds | Status |
|---|---|---|---|---|---|---|
| V1 | 23 Apr 2026 | Proof-of-concept (6 subjects) | 89.47% | Contrastive | 1 | ✅ Done |
| V2 | 16 May 2026 | Scale-up + Triplet (11 subjects) | 59.8% | Triplet | 5 | ✅ Done |
| V3 | 17 May 2026 | ArcFace switch (11 subjects) | 99.82% (no_geom) | ArcFace | 5 | ✅ Done |
| V4 | 17–21 May 2026 | Fair ablation + diagnostic | 99.1% (no_geom)* | ArcFace | 5 | ✅ Done |
| V5.0.0 | 21–23 May 2026 | Pivot to low-data regime | — | Triplet | — | ✅ Pivot complete |
| V5.0.1 | 24 May 2026 | OOM fix + efficiency | 89.5% (no_geom)** | Triplet | 10 | 🔄 Training in progress |

\* Identical ranking across all 5 seeds. \*\* Preliminary from first 5 seeds (0–4).

---

## 3. Complete Timeline

### V1 — Proof-of-Concept (23 Apr 2026)
- **Dataset:** 6 subjects, 11–16 sessions each
- **Input:** 4,096 points (xyz + normals)
- **Geometry:** 33-dim
- **Loss:** Contrastive (margin=0.5)
- **Split:** Random 70/15/15 (not session-aware)
- **Result:** Rank-1 = 89.47% (17/19 test sessions correct)
- **Limitations:** Too small, single-seed, contrastive not scalable, no ablation

### V2 — Scale-Up + Triplet (16 May 2026)
- **Dataset:** 11 subjects, frame-level layout (~1,869 frames)
- **Input:** 8,192 points
- **Loss:** Online Triplet (batch-hard, margin=0.3)
- **Split:** LOSO + holdout (1 session × 3 frames per subject)
- **Augmentation:** Expanded spatial (large rot ±90°, tilt ±15°, translate ±2cm)
- **Result:**

| Variant | Rank-1 | EER | AUC |
|---|---|---|---|
| with_geom | 59.82 ± 2.64% | 28.95 ± 2.13% | 78.38 ± 2.27% |
| no_geom | 55.45 ± 13.55% | 28.45 ± 4.66% | 78.65 ± 5.78% |

- **Wilcoxon:** p = 1.000 (not significant)
- **Finding:** Performance very poor. GeoAtt acted as a "stabilizer" (lower std) but improvement not significant.
- **Bottleneck identified:** Loss function, not architecture.

### V3 — ArcFace Revolution (17 May 2026)
- **Loss:** ArcFace (m=0.5, s=30) + hybrid multi-phase (ArcFace → Hybrid Arc+Triplet → Triplet)
- **Early stopping:** Patience 5 (Phase 1), 3 (fine-tune)
- **Enrollment:** Multi-prototype k=3
- **Result:**

| Variant | Rank-1 | EER | AUC | TAR@1% | TAR@0.1% |
|---|---|---|---|---|---|
| no_geom | **99.82 ± 0.36%** | **0.03 ± 0.04%** | ~1.000 | 100.0% | 99.72% |
| with_geom | 95.82 ± 1.59% | 2.76 ± 1.41% | 0.996 | 92.87% | 87.97% |

- **Wilcoxon:** p = 0.0625 (borderline, low power n=5)
- **Bootstrap CI 95% Δ:** [-5.27%, -3.09%] (does not include 0)
- **McNemar pooled (n=550):** p = 1.8×10⁻⁵ (highly significant, no_geom wins 23 vs 1)
- **Surprise:** GeoAtt went from "slightly helpful" (V2) to "significantly harmful" (V3).
- **Systematic failure on subject `nola`:** with_geom failed on `nola` in 4/5 seeds; no_geom never failed.

### V4 — Fair Ablation & Methodological Reset (17–21 May 2026)

#### V4 Phase 1 — Diagnostics (✅ Done)
Seven hypotheses tested:

| # | Hypothesis | Verdict | Key Evidence |
|---|---|---|---|
| H1 | Ceiling saturation (11 subjects too easy) | **REJECTED** | Hard-probe gap larger (Δ=-0.107 vs Δ=-0.056) |
| H2 | Geometry features mostly noise (FDR<1) | **REJECTED** | 0/14 features FDR<1; median FDR=3.77 |
| H3 | Systematic failure on subject `nola` | **ACCEPTED** | `nola.finger_width_5` CV=0.497 vs avg 0.056 (8.85× outlier) |
| H4 | RNG init parity broken | **ACCEPTED** | 13/33 shared layers different; only 1.5% elements identical |
| H5 | geom_emb shared by GAM1+GAM2 → gradient bottleneck | Untested | Needs architecture ablation (Phase 2) |
| H6 | Dropout only in fusion head → train/eval asymmetry | Weakly accepted | cos(eval, train_dropout): 0.908 vs 0.950 |
| H7 | Z-score normalization destroys absolute hand scale | Untested | Consistent with H2, relevance unproven |

**Patches applied:**
- F1.1 RNG Init Parity: 58/58 layers identical across 4 variants
- F1.2 4-way ablation flags: `--use-gam`, `--use-geom-fusion`
- F1.3 Backward-compat loader
- QC v3 frame-level: 1,869 valid frames (was 2,120; 8.02% excluded)
- Speed optimizations: ~3–5× speedup (3h/seed → 40–60 min/seed)

#### V4 Phase 2 — Full Run Results (21 May 2026)
4 variants × 5 seeds = 20 runs.

| Variant | Test EER | Holdout EER | AUC (test) | Rank-1 (test) |
|---|---|---|---|---|
| **no_geom** | **0.17%** | **0.00%** | **0.9995** | **99.1%** |
| fuse_only | 13.92% | 5.45% | 0.907 | 86.4% |
| with_geom | 20.16% | 11.21% | 0.857 | 80.0% |
| gam_only | 26.98% | 21.82% | 0.799 | 72.7% |

**Paired t-test no_geom vs with_geom: p < 0.001.** Ranking identical across all 5 seeds.

#### Three Fatal Biases Discovered
1. **Split temporal leakage:** Train/test/holdout for one subject came from the same capture range (< 2 min apart). Test sessions interleaved with train sessions. This means test measured "recognizing a hand in one recording session," not generalization to new conditions.
2. **Val_loss anti-correlated with generalization:** `gam_only` had lowest val_loss (0.0002) but worst test EER (27%). `no_geom` had highest val_loss (0.009) but best test EER (0.17%). Model selection was picking the most overfit checkpoint for geometry variants.
3. **Training budget non-uniform:** Early stopping triggered at epoch 20–55 (35-epoch range). Geometry variants got longer budgets because val_loss kept decreasing (bias #2).

**Conclusion:** Results statistically strong but **methodologically invalid**. Cannot be used to accept or reject the thesis hypothesis.

**Strategic decision:** Pivot to low-data regime instead of fixing split on existing dataset.

---

## 4. V5.0.0 — Low-Data Regime Pivot

### Why Low-Data?
- Does not require temporal split (1 frame/session = independent data point)
- More relevant for practical biometric enrollment
- Enables fixed budget (no early stopping)
- Focuses on inductive bias, not memorization capacity

### Key Changes from V4 → V5.0.0

| Aspect | V4 | V5.0.0 |
|---|---|---|
| **Dataset** | ~1,869 frames (all frames) | **150 frames (1 median frame/session)** |
| **Subjects** | 11 | **10** (gede dropped, <15 valid sessions) |
| **Split** | Fixed 70/15/15 | **Deterministic chronological 8/2/2/3** |
| **Loss** | ArcFace | **Triplet batch-hard (margin=0.3)** |
| **Training budget** | Early stop (patience=15/7) | **Fixed 120 + 30 epoch** |
| **Model selection** | val_loss | **val pair EER (110 pairs)** |
| **Auxiliary loss** | — | **CE classifier on geom_emb (weight=0.3)** |
| **GAM** | Sigmoid only | **Residual + tanh gating (identity-safe)** |
| **Augmentation** | Canonical reality | **Depth-focused** (±45° rot, ±20° tilt, ±3cm XY+Z, scale 0.95–1.05) |
| **Variants** | 4 (no/gam/fuse/with) | **2 (no_geom, with_geom)** |
| **Seeds** | 5 | **10** (compensate low-data variance) |

---

## 5. V5.0.1 — OOM Fix & Efficiency Optimizations (24 May 2026)

### Problems Found During First V5.0.0 Run
- Dynamic VRAM probe recommended BS=1079, N=16384 on A100 40GB → CUDA OOM
- Probe too aggressive (target 90% VRAM); did not account for fragmentation + real-loop overhead
- Ball_query distance matrix scales O(B × S × N): ~17 GB contiguous allocation for BS=1079

### Fixes Applied

| # | Fix | File | Impact |
|---|---|---|---|
| 1 | **Chunked ball_query** (chunk=256) | `models/pointnet_utils.py` | Memory: 17 GB → ~4 GB |
| 2 | **Safety clamp** on batch_size/n_points | `train.py` | Auto-config enforces hardware limits |
| 3 | **Batched ValPairMetric** (encode all unique frames in 1 batch) | `utils/val_pair_metric.py` | Validation: ~15s → ~1–2s/epoch |
| 4 | **`--val_freq N` flag** | `train.py` | Skip validation every N epochs |
| 5 | **Smoothed EER model selection** (window=5) | `train.py` | Reduces "lucky draw" from random sampling noise |
| 6 | **Conservative probe** (TARGET_VRAM=0.75, margin=0.90×) | Notebook | Prevents OOM recommendations |

### Training Configuration (Post-Fix)
- **BS=192, N=8192, repeat=5** (A100 40GB)
- **2 batches/epoch** (80 frames × 5 repeat = 400 / 192)
- **VRAM peak:** ~6.3 GB / 40 GB (15%) — normal for ~400K param model
- **Epoch time:** ~24s → **~8–12s** (with val_freq=5 + batched validation)
- **No OOM**

---

## 6. Preliminary Results (V5.0.1, First 5 Seeds: 0–4)

### Aggregate — Test Set

| Variant | EER (mean±std) | AUC (mean±std) | TAR@1% (mean±std) | d' (mean±std) | Rank-1 (mean±std) |
|---|---|---|---|---|---|
| **no_geom** | **0.05 ± 0.00** | **0.905 ± 0.008** | **0.90 ± 0.00** | **1.65 ± 0.70** | **0.895 ± 0.016** |
| with_geom | 0.425 ± 0.226 | 0.564 ± 0.207 | 0.32 ± 0.253 | -0.37 ± 0.27 | 0.440 ± 0.223 |

### Aggregate — Holdout Set

| Variant | EER (mean±std) | AUC (mean±std) | TAR@1% (mean±std) | d' (mean±std) | Rank-1 (mean±std) |
|---|---|---|---|---|---|
| **no_geom** | **0.035 ± 0.027** | **0.993 ± 0.007** | **0.917 ± 0.086** | **2.99 ± 0.74** | **0.960 ± 0.038** |
| with_geom | 0.375 ± 0.104 | 0.658 ± 0.136 | 0.233 ± 0.157 | 0.53 ± 0.52 | 0.403 ± 0.206 |

### Wilcoxon Paired Test (5 Seeds)

| Set | with_geom mean | no_geom mean | Δ | Wilcoxon stat | p-value |
|---|---|---|---|---|---|
| **Test EER** | 0.425 ± 0.226 | 0.050 ± 0.000 | +0.375 | 0.0 | **0.0020** |
| **Holdout EER** | 0.375 ± 0.104 | 0.035 ± 0.027 | +0.340 | 0.0 | **0.0020** |

### Training Trajectory (no_geom seed=42, First 8 Epochs)

| Epoch | Train Loss | Val Loss | Aux Acc | Val EER | Note |
|---|---|---|---|---|---|
| 1 | 1.444 | 0.967 | 0.198 | 0.510 | — |
| 2 | 1.148 | 0.833 | 0.513 | 0.500 | — |
| 3 | 0.941 | 0.715 | 0.758 | 0.500 | — |
| 4 | 0.809 | 0.618 | 0.930 | 0.415 | Best raw EER |
| 5 | 0.715 | 0.549 | 0.953 | 0.505 | — |
| 6 | 0.637 | 0.498 | 0.969 | 0.405 | — |
| 7 | 0.586 | 0.457 | 0.982 | 0.400 | Best raw EER |
| 8 | 0.543 | 0.426 | 0.982 | 0.600 | Spike (noise) |

- Train loss: 1.44 → 0.54 ✅
- Val loss: 0.97 → 0.43 ✅
- Aux acc: 0.20 → 0.98 ✅
- EER volatile (std=0.069 across 8 epochs) — why smoothed EER (window=5) was implemented

> **Important:** These are preliminary results from the first 5 seeds. **Do not conclude yet.** 5 seeds are insufficient for generalization. Wait for all 10 seeds + Gate 2 verdict.

---

## 7. Key Artifacts & File Locations

### Code (tag `v5.0.1`)
| File | Purpose |
|---|---|
| `3DCNN/train.py` | Main training script (v5.0.1) |
| `3DCNN/models/pointnet_utils.py` | Chunked ball_query |
| `3DCNN/utils/val_pair_metric.py` | Batched validation EER computation |
| `3DCNN/models/encoder.py` | GeoAtt-PointNet++ encoder (residual + tanh GAM) |
| `3DCNN/collab/v5_lowdata_train_eval.ipynb` | Colab training notebook |

### Data & Results
| File | Path |
|---|---|
| V5 aggregate test CSV | `analysis/v5_lowdata_20260524_112244/aggregate_test.csv` |
| V5 aggregate holdout CSV | `analysis/v5_lowdata_20260524_112244/aggregate_holdout.csv` |
| V5 Wilcoxon tests JSON | `analysis/v5_lowdata_20260524_112244/wilcoxon_tests.json` |
| V4 aggregate test CSV | `analysis/aggregate_test.csv` |
| V4 aggregate holdout CSV | `analysis/aggregate_holdout.csv` |
| V4 conclusion report (3 biases) | `result_docs/20260522_092309/KESIMPULAN_REPORT.md` |
| V5 analysis summary | `analysis/v5_lowdata_20260524_112244/SUMMARY.md` |

### Images (for reference, not embedded)
| Image | Description |
|---|---|
| `analysis/v5_lowdata_20260524_112244/train_loss_trajectory.png` | Loss per epoch (10 seeds, both variants) |
| `analysis/v5_lowdata_20260524_112244/val_eer_trajectory.png` | Val EER per epoch + smoothed window=5 |
| `analysis/v5_lowdata_20260524_112244/aux_loss_trajectory.png` | Aux classifier accuracy per epoch |
| `analysis/v5_lowdata_20260524_112244/boxplots_test_holdout.png` | Metric distribution test vs holdout |
| `analysis/v5_lowdata_20260524_112244/per_seed_paired_diff.png` | Δ no_geom vs with_geom per seed |
| `analysis/boxplots_test.png` | V4 metric distribution per variant (test) |
| `analysis/confusion_matrices_test.png` | V4 confusion matrices per variant |
| `analysis/variant_metric_heatmap.png` | V4 heatmap of all metrics per variant |

---

## 8. Current Status & Next Steps

### ✅ Completed
- [x] V1–V3 training & evaluation
- [x] V4 Phase 1 — diagnostics & patches
- [x] V4 Phase 2 — full 4-variant × 5-seed run, 3 biases identified
- [x] V5.0.0 — pivot to low-data regime, complete redesign
- [x] V5.0.1 — OOM fix, batched validation, smoothed EER, conservative probe

### 🔄 In Progress
- [ ] Training 10 seeds × 2 variants (no_geom, with_geom)
- [ ] Evaluation test + holdout per seed
- [ ] Aggregate analysis + Wilcoxon paired test (full 10 seeds)

### ⏳ Pending (Depends on Results)
- [ ] **Gate 2 verdict:** Does with_geom significantly outperform no_geom?
  - **Confirmed** → v5.1.0: replicate all-frame for gap-vs-size plot
  - **Neutral** → Accept null; thesis conclusion: GeoAtt not harmful but not helpful in low-data
  - **Problematic** → Escalate F2.10 (FiLM modulation) or F2.11 (aux loss tuning)
- [ ] Chapter writing

---

## 9. Critical Decisions Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 17 May | Switch V2→V3: Triplet→ArcFace | Triplet bottleneck at ~60% Rank-1 | +36–44 ppt jump |
| 21 May | V4 Phase 1: diagnostic audit | V3 result suspicious (GeoAtt "harmful") | Found 3 fatal biases |
| 22 May | Pivot V4→V5: low-data regime | Dataset lacks temporal diversity for split fix | New research question |
| 24 May | V5.0.1: OOM + efficiency fixes | Dynamic probe OOM at BS=1079, N=16384 | Stable training at BS=192, N=8192 |
| 24 May | Smoothed EER model selection | EER volatile (std=0.069) due to random point sampling | More reliable checkpoint selection |
| 24 May | Fixed budget (no early stopping) | V4 bias #3: non-uniform training budget | Fair comparison across variants |

---

## 10. Contact & Repository

- **Author:** Rahmat Zulfikri
- **Repo:** `https://github.com/RZulfikri/Thesis` (branch `colab` for Colab outputs)
- **Tag:** `v5.0.1`
- **Active notebook:** `3DCNN/collab/v5_lowdata_train_eval.ipynb`
- **Active analysis notebook:** `3DCNN/collab/v5_lowdata_compare.ipynb`

---

*This handover document is self-contained. All metrics, tables, and file paths are included. For image references, see Section 7 (Images). For raw CSV/JSON data, see Section 7 (Data & Results).*
