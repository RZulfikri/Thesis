# Rancangan Improvement v0.3.0 — GeoAtt-PointNet++ Palm Recognition

**Tanggal:** 2026-05-16  
**Baseline:** v0.2.0-baseline (Triplet Loss, Rank-1 ~60%, EER ~29%)  
**Tujuan:**  
1. Menunjukkan `with_geom` memberikan **nilai tambah signifikan** vs `no_geom`  
2. Meningkatkan **akurasi identifikasi** (Rank-1 > 80%, EER < 15%)  
3. Memperbaiki **confusion matrix** (lebih sedikit false positive antar subjek)

---

## Prinsip Kunci: Fair Ablation Study

**Aturan emas:** Kedua konfigurasi (`with_geom` dan `no_geom`) harus:
- ✅ Menggunakan **dataset split yang identik** (SPLIT_SEED=42, splits.json sama)
- ✅ Menggunakan **augmentasi yang identik**
- ✅ Menggunakan **loss function yang identik**
- ✅ Menggunakan **gallery enrollment yang identik**
- ✅ Hanya berbeda pada **satu variabel: USE_GEOM (True vs False)**

**Implikasi:** Karena augmentasi dan loss function berubah, **kedua konfigurasi harus dilatih ulang** secara penuh.

---

## Analisis Root Cause dari Baseline

### Kenapa `with_geom` Tidak Signifikan?

| Root Cause | Eviden | Dampak |
|------------|--------|--------|
| Geometry encoder terlalu dangkal | `33 → 64 → 64` single layer | Signal geometri lemah, tidak memberikan informasi diskriminatif yang cukup |
| Fusion hanya concatenation | `[SA3_output, geom_emb] → Linear` | Geometri tidak "memandu" ekstraksi fitur point cloud |
| GAM sederhana | Scale/shift saja, bukan cross-attention | Point features tidak "bertanya" ke geometri untuk attention |
| Z-score normalization geometri | Menghilangkan absolute scale | Signal ukuran tangan absolut hilang |
| Triplet loss stuck | Loss stagnan ~0.73 sejak epoch awal | Model tidak belajar margin yang cukup besar antar subjek |

### Kenapa Identifikasi Tidak Sempurna?

| Root Cause | Eviden | Dampak |
|------------|--------|--------|
| Rank-5 ~92% tapi Rank-1 ~60% | CMC curve | Embedding space dense, top-1 tidak cukup confident |
| Gallery enrollment = simple average | 1 sesi → rata-rata embedding | Tidak robust ke outlier frame dalam sesi |
| Intra-class variance tinggi | EER ~29% | Frame dari subjek yang sama terlalu tersebar |
| Dataset kecil (11 subjek) | — | Triplet mining tidak cukup "lihat" variasi antar identitas |

---

## Strategi Improvement (Iteratif)

### ITERASI A: Loss Function + Augmentasi + Enrollment (Tidak Ubah Encoder)

**Rationale:** Kalau perubahan encoder structure dilakukan, `no_geom` juga harus retrain untuk comparison yang adil. Iterasi A fokus pada komponen yang tidak mengubah architecture: loss function, augmentasi, dan gallery enrollment.

#### A1. Ganti Loss: ArcFace (Additive Angular Margin)

**Kenapa ArcFace?**
- Biasanya jauh lebih baik untuk dataset kecil (11 subjek) karena memaksa antar-kelas terpisah pada hypersphere dengan margin angular yang eksplisit.
- Triplet loss hanya memaksimalkan jarak anchor-negative, tapi tidak menekan intra-class variance secara eksplisit.
- ArcFace memiliki classifier head (num_classes = 11) yang memberikan signal supervision lebih kuat.

**Implementasi:**
```python
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=11, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
    
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m  # additive angular margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
```

**Training Pipeline:**
- **Phase 1 (Pretrain):** ArcFace loss — 100 epoch max, early stopping patience=15
- **Phase 2 (Fine-tune):** Hybrid ArcFace + Triplet — 30 epoch, LR lebih kecil
- **Phase 3 (Metric Refinement):** Pure Triplet dengan hard negative mining — 20 epoch

#### A2. Augmentasi Lebih Agresif

Saat ini augmentasi sudah cukup baik (rotasi, tilt, jitter, scale, dropout, translate). Tingkatkan parameter:

| Parameter | Baseline | Improvement | Rationale |
|-----------|----------|-------------|-----------|
| Point dropout | 5% | **15%** | Memaksa model belajar dari subset points, lebih robust |
| Jitter σ | 0.01m | **0.02m** | Mensimulasikan noise sensor TrueDepth yang lebih realistis |
| Scale range | (0.9, 1.1) | **(0.85, 1.15)** | Variasi jarak tangan ke kamera lebih besar |
| Large rotation | ±90° @ 30% | **±90° @ 50%** | Palm scan sering diputar |
| Tilt range | ±15° | **±25°** | Pose tangan lebih variatif |
| Translate range | 0.02m | **0.05m** | Tangan tidak selalu di tengah frame |

**Kenapa ini membantu `with_geom` lebih dari `no_geom`?**
- Ketika point cloud di-rotasi, di-scale, dan di-dropout secara agresif, informasi geometri absolute (posisi, ukuran) menjadi lebih penting untuk mengidentifikasi subjek.
- `no_geom` kehilangan anchor geometri dan menjadi lebih rentan terhadap augmentasi agresif.
- `with_geom` tetap punya geometri sebagai "compass" untuk identifikasi.

#### A3. Gallery Enrollment: Multi-Prototype + Quality Weighting

**Masalah saat ini:** Gallery = rata-rata embedding dari semua frame 1 sesi. Kalau ada frame outlier (cloud buruk), gallery menjadi terdistorsi.

**Solusi:**
1. **Clustering per subjek:** K-means (k=3) pada embeddings gallery frame → 3 prototype per subjek.
2. **Query matching:** Similarity = max(sim(query, prototype_1), sim(query, prototype_2), sim(query, prototype_3))
3. **Quality score:** Gunakan statistik geometri (jumlah points, mean confidence, density) untuk weighting saat clustering.

**Kenapa ini meningkatkan Rank-1?**
- Menangani variabilitas intra-sesi: 1 sesi bisa punya frame bagus dan frame buruk.
- Top-1 confidence meningkat karena query dibandingkan dengan prototype terbaik, bukan average yang terdilusi.

#### A4. Drop Normalisasi Geometri (Eksperimental)

**Hipotesis:** Z-score normalization menghilangkan informasi ukuran tangan absolut yang diskriminatif.

**Eksperimen:**
- Varian A: Tanpa normalisasi geometri (raw features, hanya clipping outliers)
- Varian B: Min-max normalization (mempertahankan range)
- Varian C: LayerNorm pada geometry encoder (adaptif, tidak menghilangkan signal)

---

### ITERASI B: Perbaikan Encoder Geometry (Kalau Iterasi A Belum Cukup)

**Catatan:** Kalau Iterasi A sudah memberikan improvement signifikan (Rank-1 >75% dengan with_geom), Iterasi B bisa diskip. Tapi kalau tidak, baru ubah encoder.

**Kalau encoder diubah, `no_geom` juga harus retrain** dengan structure yang sama (tanpa geom branch) untuk ablation yang valid.

#### B1. Geometry Encoder Lebih Dalam

```python
self.geom_encoder = nn.Sequential(
    nn.Linear(geom_dim, 128),
    nn.BatchNorm1d(128), nn.ReLU(inplace=True),
    nn.Linear(128, 128),
    nn.BatchNorm1d(128), nn.ReLU(inplace=True),
    nn.Linear(128, 64)
)
```

#### B2. True Cross-Attention GAM

```python
class GeometricCrossAttention(nn.Module):
    def __init__(self, sa_ch, geom_ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(sa_ch, sa_ch)
        self.kv_proj = nn.Linear(geom_ch, sa_ch * 2)
        self.out_proj = nn.Linear(sa_ch, sa_ch)
    
    def forward(self, feat, geom_emb):
        # feat: (B, N, sa_ch), geom_emb: (B, geom_ch)
        # Query dari point features, Key/Value dari geometry
        B, N, C = feat.shape
        q = self.q_proj(feat).view(B, N, self.num_heads, C//self.num_heads)
        kv = self.kv_proj(geom_emb).view(B, 1, 2, self.num_heads, C//self.num_heads)
        # ... multi-head attention computation ...
        return self.out_proj(attn_output)
```

#### B3. Geometry-Guided Spatial Attention

Geometry encoder → MLP → attention weights (B, N) → reweight point features sebelum SA layers.

---

### ITERASI C: Data-Centric (Kalau masih kurang)

- Re-scan subjek yang sulit (feby, nola, reysa) dengan pose standar
- Synthetic data augmentation dengan GAN/VAE (mungkin terlalu kompleks untuk thesis)
- Transfer learning dari ModelNet40 / ShapeNet pretrain (waktu training lebih lama)

---

## Timeline Implementasi

| Hari | Task | File yang Diubah |
|------|------|------------------|
| 1 | Implementasi ArcFace loss module | `losses/arcface.py` |
| 1 | Implementasi multi-prototype gallery enrollment | `utils/enrollment.py` |
| 2 | Update `train.ipynb`: pipeline ArcFace + augmentasi baru | `collab/train.ipynb` |
| 2 | Update `train_no_geom.ipynb`: pipeline ArcFace + augmentasi baru | `collab/train_no_geom.ipynb` |
| 2 | Update `evaluate.ipynb`: gallery enrollment baru + evaluasi | `collab/evaluate.ipynb` |
| 2 | Update `evaluate_no_geom.ipynb`: gallery enrollment baru | `collab/evaluate_no_geom.ipynb` |
| 3 | Run training with_geom (5 seeds) | `runs/with_geom/NEW_TS/` |
| 3 | Run training no_geom (5 seeds) | `runs/no_geom/NEW_TS/` |
| 4 | Run evaluation + compare | `eval_results/with_geom/NEW_TS/`, `eval_results/no_geom/NEW_TS/`, `compare/` |
| 5 | Analisis hasil, bandingkan dengan baseline v0.2.0 | Laporan |

---

## Metrik Keberhasilan

### Target Minimum (Iterasi A berhasil)

| Metrik | Baseline v0.2.0 | Target v0.3.0 | Delta |
|--------|----------------|---------------|-------|
| with_geom Rank-1 | 59.8 ± 2.6% | **> 75%** | **+15%** |
| with_geom Rank-5 | 92.4 ± 1.8% | **> 95%** | **+3%** |
| with_geom EER | 29.0 ± 2.1% | **< 15%** | **-14%** |
| with_geom AUC | 78.4 ± 2.3% | **> 90%** | **+12%** |
| with_geom vs no_geom p-value | 1.0 | **< 0.05** | Signifikan |

### Target Ideal (Iterasi A + B berhasil)

| Metrik | Target |
|--------|--------|
| Rank-1 | > 85% |
| Rank-5 | > 98% |
| EER | < 10% |
| Holdout Rank-1 | > 80% |

---

## Catatan Penting: Dataset Identik

**SPLIT_SEED = 42** harus tetap sama untuk kedua konfigurasi. `splits.json` harus identik. Kalau perlu regenerate, jalankan `train.ipynb` terlebih dahulu dengan `SPLIT_SEED=42`, lalu copy `splits.json` ke `train_no_geom.ipynb`.

---

## Checklist Implementasi

- [ ] Buat `losses/arcface.py` — ArcMarginProduct + ArcFace loss wrapper
- [ ] Update `models/siamese.py` — tambah ArcFace head (opsional, hanya saat training)
- [ ] Update `utils/augmentation.py` — tingkatkan parameter augmentasi
- [ ] Buat `utils/enrollment.py` — multi-prototype gallery enrollment
- [ ] Update `collab/train.ipynb` — pipeline ArcFace + triplet hybrid
- [ ] Update `collab/train_no_geom.ipynb` — pipeline ArcFace + triplet hybrid
- [ ] Update `collab/evaluate.ipynb` — gallery enrollment baru
- [ ] Update `collab/evaluate_no_geom.ipynb` — gallery enrollment baru
- [ ] Training with_geom 5 seeds
- [ ] Training no_geom 5 seeds
- [ ] Evaluation + compare
- [ ] Analisis statistik (Wilcoxon, bootstrap)
- [ ] Update laporan
