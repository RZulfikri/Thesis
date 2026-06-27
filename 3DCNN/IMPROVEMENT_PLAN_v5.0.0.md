# Rancangan Improvement v5.0.0 — Reframing: Low-Data Regime Study (Depth-Only Palm Identification)

**Tanggal:** 2026-05-22
**Baseline yang dianalisis:** `v4.0.0` (4 varian × 5 seed, all-frame regime)
**Pemicu:** Hasil v4.0.0 menunjukkan no_geom ≫ with_geom, namun investigasi diagnostik menemukan 3 bias eksperimental yang menggugurkan verdict. Setelah diskusi strategis, framing thesis dipivot ke **low-data regime study** — pertanyaan yang lebih spesifik, lebih relevan untuk deployment biometric, dan lebih sesuai dengan kekuatan inductive bias geom features pada data depth-only.
**Status implementasi:** Fase 1 (diagnostik bias) — **SELESAI**. Fase 2 (pivot framing + re-train 2 varian low-data) — **siap dijalankan**.

Laporan diagnostik: [`result_docs/20260522_092309/KESIMPULAN_REPORT.md`](result_docs/20260522_092309/KESIMPULAN_REPORT.md)
Laporan baseline v4.0.0: [`result_docs/20260521_152852/EVALUATION_REPORT.md`](result_docs/20260521_152852/EVALUATION_REPORT.md)

---

## Konteks & Motivasi

### Apa yang terjadi di v4.0.0

Setelah Fase 2 v0.4.0 dieksekusi (4 varian × 5 seed pada A100):

- `no_geom`: Test EER 0.17 % ± 0.26 %, Holdout EER **0.00 %**, AUC 0.9995
- `with_geom`: Test EER 20.16 %, Holdout EER 11.21 %, AUC 0.857
- Ranking konsisten 5 seed, paired t-test p<0.001

Sekilas tampak menolak hipotesis thesis, **tetapi tiga temuan diagnostik menggugurkan verdict ini**:

1. **Split bocor temporal** — sesi train/test/holdout semua dari rentang capture <2 menit. Bukan ukuran generalisasi.
2. **Val_loss anti-korelasi dengan test EER** — model selection memilih checkpoint paling overfit.
3. **Training budget tidak seragam** (20–55 epoch antar run akibat early stopping pada val_loss yang bias).

### Mengapa pivot, bukan hanya fix

Aslinya rencana V5 fokus pada perbaikan ketiga bias dan re-train 4 varian dengan setup adil. Tapi setelah analisis lebih dalam, ditemukan bahwa **pertanyaan ilmiah aslinya juga bisa diperbaiki**, bukan cuma eksperimennya.

Pertanyaan awal: *"Apakah GeoAtt > PointNet++ secara umum?"*
Pertanyaan baru: *"Apakah GeoAtt membantu PointNet++ dalam regime enrollment terbatas?"*

Framing baru lebih superior secara metodologis:

1. **Hipotesis lebih spesifik** dan punya landasan literature yang kuat (inductive bias menang di low-data — SIFT/HOG vs CNN pre-2012, XGBoost vs DL di tabular, minutiae vs DL di fingerprint).
2. **Lebih relevan untuk deployment** — enrollment biometric realistis = 1–3 capture, bukan 10+.
3. **Plays to GeoAtt's strength** — hand-crafted features (finger_lengths, palm_width, palm_height, finger_widths, palm_curvature) **persis** mengompensasi keterbatasan data.
4. **Depth-only memperkuat argumen** — tanpa RGB, satu-satunya variasi yang relevan adalah pose + jarak, yang **persis** dicakup hand-crafted features pose-invariant.

---

## Hipotesis Diagnostik v4.0.0 (Sudah Teridentifikasi)

| # | Hipotesis | Verdict | Bukti |
|---|---|---|---|
| 1 | Split test/holdout bocor secara temporal | **TERKONFIRMASI** | Inspeksi `splits.json`: untuk semua 11 subjek, sesi train/test/holdout dalam rentang capture ±90 detik. |
| 2 | Val_loss anti-korelasi dengan test EER | **TERKONFIRMASI** | gam_only val_loss terendah (0.0002) → test EER terburuk (27 %). no_geom val_loss tertinggi (0.009) → test EER terbaik (0.17 %). |
| 3 | Training budget tidak seragam | **TERKONFIRMASI** | TensorBoard `Step` final phase-2: rentang 20–55 epoch antar run. |
| 4 | `best.pth` (selected via val_loss) = checkpoint paling overfit untuk variant geom | **TERKONFIRMASI** | Konsekuensi langsung dari H2. |
| 5 | Gap train-val loss ekstrem mengindikasikan overfit parah | **TERKONFIRMASI** | with_geom seed_42: train 4e-5 vs val 4e-3 (gap 100×). |
| 6 | Skala 14-dim geom features tidak seragam menyebabkan dominasi gradien | TIDAK DIUJI | Sanity fix opsional (LayerNorm). |
| 7 | Intra-session frames terlalu redundan → pseudo-augmentation tersembunyi yang inflate dataset | **TERKONFIRMASI** | 10 frame/sesi nyaris identik (selisih milidetik); pair intra-session trivial mendominasi training. |

### Sintesis akar penyebab

Tiga masalah saling memperkuat dan menjadi alasan pivot:

1. **Setup eksperimen v4.0.0 punya bias sistematis** (H1, H2, H3) — verdict tidak valid.
2. **Dataset all-frame regime menyembunyikan overfitting** (H7) — model menghafal artefak per-capture lewat pair intra-session yang trivial.
3. **Pertanyaan ilmiah aslinya terlalu generik** — tidak memberi guidance kapan GeoAtt berguna.

**Solusi:** pivot ke low-data regime + perbaikan bias eksperimental simultan.

---

## Strategi v5.0.0: Low-Data Regime, 2-Variant, Depth-Only Augmentation

### Pertanyaan ilmiah yang baru

> **"Apakah GeoAtt-PointNet++ memberi keuntungan signifikan dibandingkan PointNet++ murni dalam regime enrollment terbatas (1 sampel per sesi) pada palm identification berbasis depth?"**

### Mengapa 2 varian, bukan 4

Diskusi sebelumnya merencanakan ablation 4-arah (no_geom, with_geom, gam_only, fuse_only). Pivot ke 2 varian (no_geom, with_geom) karena:

1. **Klaim thesis lebih bersih** — "GeoAtt sebagai whole vs PointNet++ murni" lebih mudah dipahami dan dipertahankan.
2. **Hemat 50 % compute** — 2 varian × 10 seed ~ 10 jam vs 4 varian × 10 seed ~ 20 jam.
3. **Ablation bisa ditunda** — kalau hasil 2-varian sudah signifikan, ablation menjadi pekerjaan publication berikutnya. Kalau tidak signifikan, ablation tidak akan menyelamatkan klaim utama.
4. **Variansi lebih terkontrol** — 10 seed × 2 varian memberi statistical power lebih baik daripada 5 seed × 4 varian.

### Mengapa low-data (1 frame/sesi), bukan all-frame

| Aspek | All-frame (current) | 1 frame/sesi (v5.0.0) |
|---|---|---|
| Total sampel | 1.869 frame | ~150 sesi |
| Pair genuine intra-session "gampang" | Banyak (dominan) | **Nol** |
| Match deployment scenario | Mismatch | **Match** (enrollment realistis 1–3 capture) |
| Overfitting capacity | Tinggi (mudah hafal) | Terbatas (data terlalu kecil untuk dihafal sempurna) |
| Inductive bias dari geom features | Termasked (PointNet++ bisa pelajari setara) | **Dominan** (model tidak punya data cukup untuk pelajari setara) |

**Catatan penting**: 1 frame/sesi **tidak menyelesaikan** bias kebocoran temporal #1 jika sesi-sesi masih dalam window 2 menit. Tapi mengurangi efeknya secara substansial karena pair intra-session yang trivial dihilangkan. Time-gap split (Prioritas 3 di bawah) tetap diupayakan kalau dataset mendukung.

### Mengapa depth-only memperkuat hipotesis

Dataset kamu **tidak punya RGB** — hanya depth dari iPhone TrueDepth. Variasi yang relevan:

- ✅ **Pose** (rotation, tilt tangan)
- ✅ **Jarak** (distance ke sensor)
- ❌ Pencahayaan (tidak relevan untuk depth)
- ❌ Skin tone (tidak ada di depth)
- ❌ Background (sudah disegmentasi via DBSCAN)

Hand-crafted geom features yang ada **sangat cocok** untuk variasi yang tersisa:

| Geom Feature | Invariant terhadap | Diskriminatif untuk |
|---|---|---|
| `finger_lengths × 5` | Translation, rotation Z (kalau diukur dari knuckle ke tip) | Identitas (palm geometry unik per orang) |
| `palm_width`, `palm_height` | Translation, rotation Z | Identitas |
| `finger_widths × 5` | Translation, rotation Z | Identitas |
| `palm_depth_std` | Translation, scale | Identitas (palm curvature unik) |
| `mean_palm_curvature` | Translation, rotation, scale | Identitas |

PointNet++ murni harus belajar invariance ini dari raw points — butuh data banyak. GeoAtt punya invariance ini sudah pre-computed.

**Prediksi**: pada depth-only + low-data, GeoAtt seharusnya menang. Kalau tidak menang, ini sinyal kuat ada bug arsitektur GAM yang aktif merugikan (bukan netral) — bukan kegagalan konseptual inductive bias.

---

## Fase 2 — Rencana Tindak Lanjut

### F2.0 Revisi Feature Set Geometri (Prioritas 0 — Preflight)

**Tujuan:** mengubah komposisi 14-dim geom features berdasarkan **diskriminabilitas aktual (between/within std ratio)** dari inspeksi dataset langsung pada 2026-05-22.

**Temuan dari audit dataset:**

| Feature | B/W ratio | Verdict |
|---|---|---|
| finger_lengths × 5 | 2.19 – 5.88 (avg 3.87) | ★★★ workhorse — pertahankan |
| palm_width, palm_height, palm_depth_std | 1.90 – 2.36 | ★★ pertahankan |
| finger_widths × 4 (index, middle, ring, pinky) | 2.19 – 2.77 | ★★ pertahankan |
| `finger_widths[0]` (thumb) | 1.38 | ✗ buang — paling noisy di group |
| `mean_palm_curvature` | **0.76** | ✗ buang — within > between, lebih noisy daripada sinyal |
| `inter_finger_gaps × 4` | 1.45 avg, 0.98 min | ✗ tidak ditambah — pose-dependent, lemah |
| `scan_distance_mm` (BARU) | 1.45 | △ tambah sebagai context (scaling hint, bukan biometric) |

**Feature set final (13-dim):**

| # | Feature | Sumber | Peran |
|---|---|---|---|
| 1–5 | `finger_lengths_mm` × 5 | thumb, index, middle, ring, pinky | Biometric utama |
| 6 | `palm_width_mm` | — | Biometric |
| 7 | `palm_height_mm` | — | Biometric |
| 8 | `palm_depth_std_mm` | — | Biometric (curvature) |
| 9–12 | `finger_widths_mm[1:5]` | index, middle, ring, pinky (skip thumb) | Biometric |
| 13 | `scan_distance_mm` | — | Context (scaling hint) |
| **Total** | | | **13 dim** |

**Keputusan TIDAK menambah ratio features.** Setelah analisis, rasio diputuskan **tidak diperlukan** karena:
1. Absolute features kamu sudah dalam **real mm fisik** dari TrueDepth metric depth — tidak ada conflation dengan jarak yang perlu di-fix.
2. Variasi pose di data kecil dan sudah di-handle oleh augmentation eksplisit (rotation/tilt).
3. MLP 3-layer geom_encoder cukup mudah belajar rasio secara implicit dari raw absolute features.
4. Information-theoretic redundancy: kalau model punya A dan B, dia sudah punya semua info A/B.
5. Di low-data regime, menambah feature dim = lebih banyak parameter = lebih mudah overfit.
6. Concern empiris: rasio bisa membuat dua subjek dengan tangan beda ukuran tapi proporsi sama terlihat identik (contoh nyata: reysa rasio 0.7653 vs nola 0.7820 padahal palm_width fisik beda 10mm).

**Konsekuensi**: dimensi geom turun dari 14 → 13, tapi **setiap dim B/W ≥ 1.45** dan sebagian besar ≥ 2.0. Lebih ramping, lebih efficient.

**Implementasi:**

`utils/dataset.py`:
```python
GEOMETRY_KEYS = [
    "finger_lengths_mm",   # list[5]
    "palm_width_mm",       # float
    "palm_height_mm",      # float
    "palm_depth_std_mm",   # float
    "finger_widths_mm",    # list[5] → flatten[1:5] saja di _flatten_geometry
    "scan_distance_mm",    # float — BARU, context
]
GEOMETRY_DIM = 13  # 5 + 1 + 1 + 1 + 4 + 1
```

Modifikasi `_flatten_geometry()`:
- Untuk `finger_widths_mm`: ambil index `[1:5]` saja (skip thumb)
- Tambah `scan_distance_mm` di urutan terakhir
- Hapus `mean_palm_curvature`

Re-extract normalizer stats wajib karena feature set berubah.

**Sanity baseline (sebelum full training):**

Jalankan **geom-only LeaveOneSessionOut CV** sebelum lanjut ke F2.1:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

# X14_old: 14-dim feature set lama (current v4.0.0)
# X13_new: 13-dim feature set baru
for X, name in [(X14_old, '14-dim old'), (X13_new, '13-dim new')]:
    clf = LogisticRegression(C=1.0, max_iter=1000)
    cv = LeaveOneGroupOut()
    scores = cross_val_score(clf, X, y, groups=session_ids, cv=cv)
    print(f"{name}: accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
```

**Interpretasi expected**:
- Kalau `13-dim new` ≥ `14-dim old` → swap fitur lemah (curv, thumb_width) ke context (scan_distance) berhasil → lanjut F2.1.
- Kalau `13-dim new` jauh lebih rendah → revisit (mungkin scan_distance malah mengganggu, atau curvature ternyata penting di non-linear classifier).
- Kalau **kedua < 50%** → indikasi dataset terlalu kecil untuk klaim apapun → pertimbangkan capture ulang sebelum lanjut.

**Effort:** ~45 menit total (30 menit code change + 15 menit run sanity test).

---

### F2.1 Dataset Loader `OneFramePerSession` + Chronological Split (Prioritas 1, WAJIB)

**Tujuan:** sample 1 median frame per sesi + split sesi secara **chronological deterministic** (no randomness).

**File baru:** `utils/dataset_lowdata.py`

**Subjek yang dipakai (10 subjek)**: aisah, alji, chrys, fadhil, feby, nola, rahmat, reysa, taufik, yanuar.

**Subjek di-drop**: **gede** (cuma 9 sesi, di bawah minimum 15).

**Split per subjek (deterministic chronological, IDENTIK untuk no_geom & with_geom & semua 10 seed):**

```
15 sesi kronologis: [s1, s2, ..., s15]
                                 │
  Train:   s1 – s8      (8 frames, oldest)
  Val:     s9 – s10     (2 frames, untuk model selection via val EER)
  Test:    s11 – s12    (2 frames, final reportable metric)
  Holdout: s13 – s15    (3 frames newest, generalization claim)
```

**Total dataset**: 10 subjek × 15 sesi × 1 median frame = **150 frames**
- Train: 80 | Val: 20 | Test: 20 | Holdout: 30

**Pair counts (genuine):**
- Train: C(8,2) × 10 = 280
- Val: C(2,2) × 10 = 10 (kecil, monitoring only)
- Test: C(2,2) × 10 = 10 (mitigation: impostor balanced)
- Holdout: C(3,2) × 10 = 30

**Implementasi:**

1. Per sesi, hitung **median frame** dengan logika:
   - Untuk setiap frame, ekstrak geom feature vector (13-d post-F2.0).
   - Hitung jarak ke median session-level (MAD-based).
   - Pilih frame dengan jarak terkecil sebagai "representative frame".
   - Alasan: hindari edge artifacts dari frame awal/akhir scan.
2. Split sesi secara chronological:
   ```python
   DROPPED_SUBJECTS = {'gede'}

   def build_split(dataset_root):
       splits = {'train': [], 'val': [], 'test': [], 'holdout': []}
       for subject in sorted(os.listdir(dataset_root)):
           if subject in DROPPED_SUBJECTS or subject.startswith('_'):
               continue
           sessions = sorted([s for s in os.listdir(f'{dataset_root}/{subject}')
                            if not s.startswith('_')])
           assert len(sessions) >= 15, f"{subject} has {len(sessions)} sessions, need ≥15"

           first15 = sessions[:15]  # take first 15 chronologically
           splits['train']   += [(subject, s) for s in first15[0:8]]
           splits['val']     += [(subject, s) for s in first15[8:10]]
           splits['test']    += [(subject, s) for s in first15[10:12]]
           splits['holdout'] += [(subject, s) for s in first15[12:15]]
       return splits
   ```
3. **Tidak ada `SPLIT_SEED`** — tidak ada randomness sama sekali di split. Semua randomness eksperimen = murni dari training stochasticity (model init, augmentation random, dataloader shuffle) yang dikendalikan oleh `model_seed`.
4. **Identik antar varian (no_geom & with_geom)**: paired comparison strict.
5. **Identik antar 10 seed**: variansi murni dari training stochasticity.

**Mengapa chronological deterministic**:
- Tidak ada bias "lucky split"
- Reproducibility maksimal (siapa pun bisa generate split sama dari folder dataset)
- Semantic temporal progression: train = oldest → holdout = newest → maksimalkan time-gap walau terbatas (capture window ~2 menit per subjek)
- Tidak butuh dokumentasi magic number

**Modifikasi:** `train.py` dan `evaluate.py` — tambah flag `--frames-per-session 1` (default: all-frame untuk backward compat).

### F2.2 Val Pair EER Metric Logger (Prioritas 1, WAJIB — carry over dari plan sebelumnya)

**Tujuan:** ganti val_loss sebagai metric model selection dengan val pair EER.

**File baru:** `utils/val_pair_metric.py`

**Implementasi:**

1. Di akhir setiap epoch, generate pair val dari **val set explicit** (2 sesi × 10 subjek = 20 frames):
   - Genuine pairs: C(2,2) × 10 = 10 pair (kecil).
   - Impostor pairs: balanced (cross-subject, deterministic sampling) — target 100 impostor.
   - Total: ~110 pair untuk val EER computation.
   - **Karena pair count kecil, expect variansi val EER per-epoch agak tinggi** — pakai moving average 5-epoch untuk smoothing.
2. Compute embedding via encoder eval mode (kedua frame di val set).
3. Hitung cosine similarity → derive **val EER**, **val AUC**, **val TAR@FAR=1%**.
4. Log ke TensorBoard: `val/pair_eer`, `val/pair_auc`, `val/pair_tar_at_far1`.
5. **Model selection:** simpan `best.pth` berdasarkan **val EER terendah (smoothed)** sepanjang training.

### F2.3 Temporal Split — Simplified (Prioritas 1, sudah ter-integrasi di F2.1)

**Catatan**: opsi A/B/C di plan sebelumnya **digugurkan**. Setelah keputusan chronological deterministic split di F2.1, tidak ada lagi tuning threshold time-gap.

**Aturan final:**
- Holdout = **3 sesi terakhir kronologis** per subjek (s13, s14, s15).
- Test = 2 sesi sebelum holdout (s11, s12).
- Val = 2 sesi sebelum test (s9, s10).
- Train = 8 sesi terlama (s1–s8).

**Limitasi yang didokumentasikan eksplisit di laporan thesis**:
- Capture window semua sesi per subjek ~2 menit. Time-gap train→holdout maksimal ~90 detik.
- Klaim "generalization to future time" **belum tervalidasi sepenuhnya** dengan data current.
- Future work: capture ulang 6+ subjek dengan sesi terpisah hari/minggu untuk validasi generalisasi temporal sejati.

**File baru**: `utils/audit_temporal_gap.py` — script untuk melaporkan time-gap per subjek di setiap split (untuk dokumentasi limitation di laporan).

### F2.4 Fixed Training Budget (Prioritas 1, WAJIB)

**Tujuan:** menjamin semua varian × seed mendapat alokasi training identik.

**Setup:**
- **Hapus early stopping default.** Semua run berhenti di epoch yang sama.
- Phase 1: **120 epoch** (lebih banyak dari v4.0.0 karena data lebih kecil, perlu banyak iterasi)
- Phase 2 (fine-tune lr lebih rendah): **30 epoch**
- Total: 150 epoch fixed
- `best.pth` dipilih berdasarkan val pair EER terbaik di seluruh trajectory.

### F2.5 Loss Function: Triplet Batch-Hard (Prioritas 1)

**Tujuan:** ganti ArcFace ke loss yang lebih cocok untuk ~14 sampel/class.

**Rasional:**
- ArcFace dirancang untuk classification dengan margin pada softmax. Butuh **banyak sampel per class** untuk stabilitas margin. Di low-data (~14 sampel/subjek), ArcFace gradient menjadi noisy.
- Triplet dengan batch-hard mining lebih cocok karena bekerja per-pair, tidak butuh banyak sampel per class. Sudah tersedia di [`losses/triplet.py`](losses/triplet.py).

**Setup:**
- Margin: 0.3 (default Triplet)
- Mining: batch-hard (`semi-hard` fallback kalau collapse)
- Batch size: 32–64 (lebih kecil dari ArcFace karena pair sampling lebih intensif memory)

### F2.6 Augmentation Strategy: Pose + Distance Only (Prioritas 1)

**Tujuan:** kompensasi data kecil dengan augmentation yang **fokus ke variasi depth-only**.

**Tidak diperlukan (depth-only):**
- ❌ Color jitter (tidak ada RGB)
- ❌ Brightness/contrast variation (depth tidak terpengaruh pencahayaan)
- ❌ Shadow simulation (depth tidak punya shadow)

**Diperlukan (variasi pose & jarak):**
- ✅ Rotation Z: ±45° (naik dari ±30° di v4.0.0)
- ✅ Tilt X/Y: ±20° (naik dari ±15° di v4.0.0)
- ✅ Translation XY: ±3cm
- ✅ Translation Z (jarak ke sensor): ±3cm (BARU, sebelumnya tidak ada)
- ✅ Random scale: 0.95–1.05 (BARU, kompensasi variasi jarak)
- ✅ Point subsampling: 8192 dari N total (natural augmentation)
- ✅ Random jitter Gaussian: σ=1mm (lebih kecil dari σ=2mm karena depth iPhone TrueDepth relatif clean)

**Implementasi:** modifikasi [`utils/augmentation.py`](utils/augmentation.py) — tambah `random_z_translation` dan `random_scale`.

### F2.7 Eksperimen Utama (Prioritas 1)

**Setup:** 2 varian × 10 seed × 1 median frame/sesi × split chronological 8/2/2/3 (10 subjek, gede dropped).

**Dataset:** 150 frames total (80 train + 20 val + 20 test + 30 holdout). Split identik untuk semua run.

**Seed values:** 42, 123, 2024, 7, 31337, 0, 1, 2, 3, 4 (10 seed total). `model_seed` saja yang bervariasi; dataset split tidak berubah.

**Metrics yang dilaporkan**:
- `val/pair_eer` per-epoch → untuk `best.pth` selection
- **`test/eer`** → metric utama untuk Wilcoxon paired test (Gate 2 verdict)
- **`holdout/eer`** → klaim generalization tambahan

```bash
# no_geom (PointNet++ murni)
for seed in 42 123 2024 7 31337 0 1 2 3 4; do
  python train.py \
    --output_dir runs/v5_lowdata/no_geom/seed_${seed} \
    --frames-per-session 1 \
    --loss triplet \
    --triplet-margin 0.3 \
    --val-metric pair_eer \
    --epochs-phase1 120 --epochs-phase2 30 \
    --batch-size 64 \
    --seed ${seed}
done

# with_geom (GeoAtt full)
for seed in 42 123 2024 7 31337 0 1 2 3 4; do
  python train.py \
    --output_dir runs/v5_lowdata/with_geom/seed_${seed} \
    --frames-per-session 1 \
    --loss triplet \
    --triplet-margin 0.3 \
    --val-metric pair_eer \
    --epochs-phase1 120 --epochs-phase2 30 \
    --batch-size 64 \
    --seed ${seed} \
    --use-geom
done
```

Estimasi wall-time: ~25 menit/run × 20 run = **~10 jam pada A100**.

### F2.8 Eksperimen Pendamping: All-Frame Replikasi (Prioritas 2)

**Tujuan:** plot **gap-vs-dataset-size** sebagai *central finding* thesis.

**Setup:** 2 varian × 3 seed × all-frame regime × setup v5.0.0 (val EER metric + fixed budget + augmentation baru).

```bash
for seed in 42 123 2024; do
  python train.py \
    --output_dir runs/v5_allframe/no_geom/seed_${seed} \
    --frames-per-session all \
    --loss triplet --val-metric pair_eer \
    --epochs-phase1 80 --epochs-phase2 20 \
    --batch-size 256 --seed ${seed}
  python train.py \
    --output_dir runs/v5_allframe/with_geom/seed_${seed} \
    --frames-per-session all \
    --loss triplet --val-metric pair_eer \
    --epochs-phase1 80 --epochs-phase2 20 \
    --batch-size 256 --seed ${seed} --use-geom
done
```

Estimasi wall-time: ~1 jam/run × 6 run = **~6 jam pada A100**.

### F2.10 Conditional Fallback: GAM Architecture Fix (Prioritas 3, aktif kalau F2.7 gagal)

**Trigger**: kalau hasil F2.7 menunjukkan `with_geom` < `no_geom` signifikan (Wilcoxon p < 0.05 dengan arah merugikan).

**Tujuan:** memperbaiki kelemahan struktural GAM yang teridentifikasi dari analisis arsitektur:

1. **GAM saat ini hanya sigmoid gating** — α ∈ [0,1] cuma bisa meredam, tidak bisa amplify. Kalau α kollaps ke ~0, informasi hilang permanen.
2. **Tidak ada residual connection** — `output = α * feat`, tidak ada skip safety net.
3. **Geom_emb di-broadcast identik ke semua N titik** — bukan true per-point attention, hanya per-channel modulation global.
4. **Geom_proj di GAM tidak ada BN/Dropout** — mudah overfit di low-data.

**Implementasi (urutan dari paling cheap):**

**Step 1 — Residual GAM** (effort: 10 menit, paling penting)

`models/gam.py`:
```python
def forward(self, sa_feat, geom_emb):
    # ... existing geom_proj + concat + attn_gate ...
    delta = alpha * sa_feat
    return sa_feat + delta  # residual — α=0 = identity safe
```

**Step 2 — Tanh+offset gating** (effort: 1 baris, allows amplification)

Ganti final sigmoid di `attn_gate` dengan `α = 1 + 0.5 * tanh(...)` → range [0.5, 1.5]. Sekarang GAM bisa amplify channel yang relevan, bukan cuma meredam.

**Step 3 — LayerNorm di geom branch** (effort: 5 menit)

Tambah `nn.LayerNorm(64)` di akhir [`models/geometry_encoder.py`](models/geometry_encoder.py) sebelum return. Menstabilkan skala geom_emb sebelum masuk GAM/fusion.

**Decision logic:**
- Implementasi Step 1+2+3 secara bersamaan.
- Re-run training 5 seed × 2 varian (low-data) — ~5 jam.
- Bandingkan dengan F2.7 baseline.
- Jika gap menutup → identifikasi arsitektur GAM sebagai akar masalah.
- Jika gap masih ada → lanjut F2.10b (FiLM modulation) atau F2.11 (auxiliary loss).

**F2.10b — FiLM Modulation (escalation)**

Kalau Step 1-3 tidak cukup, ganti GAM seluruhnya dengan FiLM:
```python
gamma = MLP_gamma(geom_emb)   # (B, C) — multiplicative
beta  = MLP_beta(geom_emb)    # (B, C) — additive
output = gamma.unsqueeze(1) * sa_feat + beta.unsqueeze(1)
```
Lebih ekspresif daripada gating; bisa amplify, shift, atau silence.

---

### F2.11 Optional Enhancement: Auxiliary Classification Loss (Prioritas 2)

**Tujuan:** memaksa geom_encoder belajar representasi diskriminatif dengan **direct supervision**, bukan mengandalkan gradient via Triplet loss saja.

**Rasional:**
- Di low-data, geom branch beresiko jadi "passenger" — PointNet++ dominate gradient, geom_encoder underfit.
- Auxiliary loss memberi sinyal training **eksplisit dan langsung** ke geom_encoder: "kamu harus bisa mengklasifikasi subjek dari fitur geom saja".
- Sebagai diagnostic bonus: trajectory aux_loss menunjukkan apakah geom branch beneran belajar (turun) atau idle (plateau di atas chance level).

**Implementasi:**

`models/siamese.py`:
```python
class SiameseEncoder(nn.Module):
    def __init__(self, ..., n_subjects=10, use_aux_loss=False):  # 10 subjek (gede dropped)
        super().__init__()
        # ... existing ...
        if use_aux_loss:
            self.aux_classifier = nn.Linear(64, n_subjects)  # geom_emb → logits

    def forward(self, pts, geom, return_aux=False):
        # ... existing forward ...
        emb = F.normalize(self.encoder(pts, geom), p=2, dim=1)
        if return_aux and hasattr(self, 'aux_classifier'):
            geom_emb = self.encoder.geom_encoder(geom)
            aux_logits = self.aux_classifier(geom_emb)
            return emb, aux_logits
        return emb
```

`train.py`:
```python
emb, aux_logits = model(pts, geom, return_aux=True)
triplet_loss = triplet_loss_fn(emb, labels)
aux_loss = F.cross_entropy(aux_logits, labels)
total_loss = triplet_loss + 0.3 * aux_loss  # bobot 0.3 = secondary
```

**Trigger conditional:**
- Aktif **default** di v5.0.0 (low cost, high upside di low-data).
- Bisa disable via flag `--use-aux-loss=false` untuk ablation kalau diperlukan.

**Hasil yang diharapkan:**
- aux_loss turun monoton selama training → geom branch belajar diskriminatif
- Final aux accuracy > 50% → geom features memang carry biometric signal
- Final aux accuracy ≈ chance (1/10 = 10%) → indikasi geom branch broken atau features tidak diskriminatif

**Effort:** ~30 baris kode. Compute overhead < 5 %.

---

### F2.9 Analisis & Plotting

**Skrip baru:** `analysis/v5_gap_vs_size.py`

Output utama:
1. **Tabel ringkasan**: rerata ± std EER, AUC, TAR@FAR=1% untuk 2 varian × 2 regime (low-data, all-frame).
2. **Plot gap-vs-size**: x-axis = sampel/subjek, y-axis = gap EER (with_geom − no_geom). Diharapkan: gap negatif (with_geom menang) di low-data; gap positif/zero di all-frame.
3. **Paired Wilcoxon test** for low-data regime (n=10 seed).
4. **Bootstrap CI** untuk gap dengan n_resample=1000.
5. **Per-subjek confusion matrix** untuk identifikasi subjek mana yang paling diuntungkan/dirugikan GeoAtt.

---

## Target Metrik v5.0.0

### Low-Data Regime (split 8/2/2/3, n=10 seed, 10 subjek)

| Metrik | no_geom target | with_geom target | Status sukses |
|---|---|---|---|
| **Val EER** (selection signal) | Trajectory turun monoton | Trajectory turun monoton | Keduanya converge < 25 % |
| **Test EER** (primary, paired Wilcoxon) | Naik signifikan dari v4.0.0 (no leak shortcut) | Naik moderat | **with_geom test EER < no_geom test EER, p < 0.05** |
| **Holdout EER** (secondary, generalization) | Naik dari 0 % (tidak lagi leak-perfect) | ≤ no_geom holdout EER | Konsisten arah dengan Test EER |
| Test EER std antar seed | < 5 % | < 3 % | with_geom lebih stabil (lower variance dari inductive bias) |
| Bootstrap CI Δ Test EER | n/a | n/a | **CI tidak melingkupi 0, sisi negatif (with_geom menang)** |

### All-Frame Regime (replikasi, n=3 seed)

| Metrik | no_geom | with_geom |
|---|---|---|
| Test EER | Tetap rendah (mungkin masih dominan karena kapasitas memorisasi) | Rendah tapi mungkin lebih tinggi dari no_geom |
| Gap with_geom − no_geom | Diharapkan **positif kecil atau nol** | — |

### Plot Gap-vs-Size (Central Finding)

```
Gap EER (with_geom − no_geom)
    │
    │       ●  ← all-frame (1869 sampel)
    │      /
    │     /
────●─────────────────────  zero gap (no harm)
    │    \
    │     \
    │      ●  ← low-data (150 sampel)
    │
    └─────────────────────► Dataset size
```

**Kriteria sukses utama**: low-data gap **negatif signifikan** (with_geom menang), all-frame gap kecil atau positif kecil. Ini akan menjadi *finding utama thesis*.

### Kriteria sukses minimum

with_geom **tidak signifikan merugikan** di low-data regime (Wilcoxon p > 0.05 untuk arah merugikan). Ini sudah cukup untuk klaim "GeoAtt tidak merugikan; sebaliknya, di low-data ada indikasi membantu walau belum konklusif statistik".

### Kriteria sukses maksimum

with_geom > no_geom di low-data dengan p < 0.05, **dan** gap mengecil/membalik di all-frame. Klaim thesis menjadi: *"GeoAtt menyediakan inductive bias yang berguna di regime enrollment terbatas, yang relevan untuk skenario deployment realistis."*

---

## Catatan Sejarah & Hubungan dengan Plan Sebelumnya

- **v0.2.0-baseline:** Triplet loss, kedua varian ~60% Rank-1. Hipotesis: GeoAtt sebagai regularizer.
- **v0.3.0-baseline:** ArcFace. no_geom 99.82%; with_geom 95.82%. Investigasi mengarah ke init parity.
- **v0.4.0-baseline:** Init parity fixed, QC v3, 4 varian × 5 seed. Tag `v0.4.0`.
- **v4.0.0:** Re-eval lengkap dengan holdout session. Verdict no_geom > with_geom tampak konsisten — kemudian terungkap bias setup. Tag `v4.0.0`.
- **v5.0.0 (rencana ini):** Pivot framing ke **low-data regime study**, dengan 2 varian (no_geom, with_geom), 10 seed, Triplet loss, augmentation depth-focused, fixed budget, val pair EER metric. Eksperimen pendamping all-frame untuk plot gap-vs-size.

### Status kerja

- [x] Identifikasi 3 bias eksperimental dari v4.0.0
- [x] Inspeksi `splits.json`, TensorBoard `loss/val`, `train_log.csv`
- [x] Laporan [`KESIMPULAN_REPORT.md`](result_docs/20260522_092309/KESIMPULAN_REPORT.md)
- [x] Pivot framing dari "ablation 4-arah" → "low-data 2-arah" + plot gap-vs-size
- [x] Audit diskriminabilitas feature geometri langsung dari dataset (B/W ratio per feature)
- [x] Keputusan feature set: 13-dim (drop curvature, drop thumb_width, add scan_distance, no ratios)
- [x] Keputusan split protokol: chronological deterministic 8/2/2/3, drop gede, IDENTIK untuk no_geom + with_geom + semua 10 seed
- [x] Keputusan metrics: Test EER = primary (Wilcoxon), Holdout EER = secondary (generalization claim)
- [ ] Update `utils/dataset.py` ke 13-dim baru (F2.0)
- [ ] Sanity baseline geom-only LeaveOneSessionOut CV (F2.0)
- [ ] Implementasi `utils/dataset_lowdata.py` (F2.1)
- [ ] Implementasi `utils/val_pair_metric.py` (F2.2)
- [ ] Decision A/B/C untuk time-gap split (F2.3) — default A untuk smoke test
- [ ] Patch `train.py` untuk fixed budget + val EER selection (F2.4)
- [ ] Switch loss back to Triplet di config default (F2.5)
- [ ] Update `utils/augmentation.py` dengan random_z_translation + random_scale (F2.6)
- [ ] **Gate 0**: sanity baseline geom-only CV — stop kalau accuracy < 30%
- [ ] Smoke test 1 seed × 2 varian × low-data (~1 jam)
- [ ] **Gate 1**: smoke test trajectory check — debug kalau val EER stuck > 30%
- [ ] Full run 10 seed × 2 varian × low-data (~10 jam) — F2.7
- [ ] **Gate 2**: Wilcoxon test → verdict hipotesis (confirmed/neutral/problematic)
- [ ] Conditional F2.10: GAM architecture fix (kalau Gate 2 trigger)
- [ ] Conditional F2.11: Auxiliary loss (kalau perlu forcing function)
- [ ] **Gate 3**: kalau F2.10 dijalankan — verdict setelah fix
- [ ] Replikasi all-frame 3 seed × 2 varian (~6 jam) — F2.8
- [ ] **Gate 4**: plot gap-vs-size → verdict central finding
- [ ] Analisis & plot gap-vs-size — F2.9
- [ ] Tag baru `v5.0.0-lowdata` (atau `v5.0.0-lowdata-gamfix`) setelah hasil stabil
- [ ] Optional: capture ulang 6 subjek untuk validasi time-gap split (Opsi B di F2.3)

---

## Decision Gates — Kriteria Stop/Continue Eksplisit

Setiap milestone punya **decision gate** yang menentukan: lanjut, debug, atau pivot. Tujuannya menghindari "sunk cost fallacy" — buang waktu di setup yang sudah jelas tidak akan menghasilkan hasil meaningful.

### Gate 0 — Setelah F2.0 (Sanity Baseline Geom-only CV)

**Kriteria:**

| Kondisi | Aksi |
|---|---|
| 13-dim new accuracy ≥ 14-dim old accuracy, **dan** keduanya ≥ 50% | ✅ **CONTINUE** ke F2.1 |
| 13-dim new < 14-dim old (regression > 5%) | ⚠️ **DEBUG**: cek apakah scan_distance mengganggu, atau curvature ternyata penting di nonlinear. Iterasi feature set. |
| Kedua < 30% accuracy | 🛑 **STOP**: dataset terlalu kecil/noisy untuk klaim apapun. Pertimbangkan capture ulang sebelum lanjut deep learning. |
| 13-dim new accuracy > 80% | 🎉 **CONTINUE dengan confidence tinggi** — geom features memang sangat diskriminatif. Hipotesis low-data sangat well-founded. |

**Mengapa kritikal**: kalau geom-only sudah underperform di sanity baseline, deep learning tidak akan menyelamatkan. Hemat 16 jam compute sebelum sia-sia.

### Gate 1 — Setelah Smoke Test 1 Seed × 2 Varian Low-Data (~1 jam)

**Kriteria:**

| Kondisi | Aksi |
|---|---|
| Val pair EER trajectory turun monoton untuk **kedua** varian dan plateau < 20% di epoch 50 | ✅ **CONTINUE** ke full run F2.7 |
| Val EER stuck > 30% di epoch 50 untuk salah satu varian | ⚠️ **DEBUG**: cek augmentation strength, learning rate, batch size. Triplet collapse? |
| Val EER tidak turun sama sekali (semi-monoton naik) | 🛑 **STOP**: ada bug di pipeline. Audit pair sampling, loss computation, embedding normalization. |
| Training crash / OOM / NaN | 🛑 **STOP**: fix infrastructure dulu. |

**Mengapa kritikal**: smoke test 1 jam jauh lebih murah daripada full run 10 jam yang gagal.

### Gate 2 — Setelah F2.7 (Full Run 10 Seed × 2 Varian Low-Data)

**Primary metric**: `test/eer` (dari 2 sesi test per subjek, total 20 test frames).
**Secondary metric**: `holdout/eer` (untuk klaim generalization tambahan).

**Kriteria (berdasarkan Test EER):**

| Kondisi (paired Wilcoxon n=10 seed) | Verdict | Aksi |
|---|---|---|
| p < 0.05, with_geom test EER < no_geom test EER (with_geom menang) | 🎉 **HIPOTESIS TERKONFIRMASI** | Lanjut F2.8 replikasi all-frame untuk plot gap-vs-size. Tag `v5.0.0-lowdata-final`. |
| p > 0.10 (tidak ada significant difference) | 🟡 **HIPOTESIS NETRAL** | Klaim: "GeoAtt tidak terbukti merugikan; ada indikasi netral di low-data". Lanjut F2.11 (aux loss) untuk lihat apakah forcing geom branch membantu. |
| p < 0.05, with_geom test EER > no_geom test EER (with_geom kalah signifikan) | 🔴 **HIPOTESIS BERMASALAH** | Trigger F2.10 (GAM architecture fix). Jangan langsung tolak hipotesis — mungkin arsitektur yang salah, bukan ide. |
| Variansi sangat tinggi → CI sangat lebar | ⚠️ **STATISTICAL POWER KURANG** | Tambah ke 15 seed; atau pertimbangkan capture ulang dataset. |

**Sub-check (Holdout EER consistency):**
- Kalau Test EER dan Holdout EER **arah sama** (kedua-duanya favor varian yang sama) → verdict diperkuat.
- Kalau arah **berbeda** (Test favor with_geom, Holdout favor no_geom misalnya) → ada distribution shift antara test ↔ holdout walau temporal gap kecil → tambah analisis di F2.9.

### Gate 3 — Setelah F2.10 (Fallback GAM Fix, kalau Gate 2 trigger)

**Kriteria:**

| Kondisi | Aksi |
|---|---|
| Gap menutup signifikan setelah fix (with_geom ≥ no_geom) | ✅ **HIPOTESIS TERKONFIRMASI** dengan catatan "membutuhkan modifikasi arsitektur GAM". Tag `v5.0.0-lowdata-gamfix`. |
| Gap mengecil tapi tetap with_geom < no_geom signifikan | ⚠️ Eskalasi ke F2.10b (FiLM modulation) + F2.11 (aux loss). |
| Tidak ada perubahan setelah fix | 🛑 **HIPOTESIS DITOLAK** dengan integritas. Klaim final: "geom features dalam setup arsitektur kami tidak memberi keuntungan; future work: eksplorasi cross-attention atau dataset expansion". |

### Gate 4 — Setelah F2.8 (All-Frame Replikasi)

**Kriteria untuk klaim "gap-vs-size":**

| Kondisi | Klaim valid |
|---|---|
| Gap low-data **negatif** (with_geom menang) AND gap all-frame **positif/nol** | 🎉 **CENTRAL FINDING**: "GeoAtt berguna di low-data; redundant di high-data" |
| Gap low-data **negatif** AND gap all-frame **juga negatif** tapi lebih kecil | 🟡 **PARTIAL FINDING**: "GeoAtt berguna terutama di low-data" |
| Kedua gap **konsisten arah yang sama** | ⚠️ **NO MEANINGFUL DIFFERENCE BETWEEN REGIMES** — klaim direvisi |

**Catatan**: kalau gate 4 menunjukkan tidak ada diferensiasi antara regime, klaim thesis tetap valid tapi diubah dari "low-data advantage" ke "GeoAtt berguna secara umum" atau sesuai arah hasil.

---

## Risiko & Mitigasi

| Risiko | Probabilitas | Mitigasi |
|---|---|---|
| Variansi seed tinggi di low-data → tidak ada significance | **Tinggi** | 10 seed (vs 5 di v4.0.0); bootstrap CI dengan n_resample=1000; tambah ke 15 seed kalau perlu |
| Semua varian underfit → ranking tidak meaningful | Sedang | Cek val EER trajectory; kalau plateau di atas 30 %, naikkan augmentation atau pertimbangkan capture ulang |
| Triplet collapse (embedding seragam) | Sedang | Monitor embedding norm + spread; fallback ke semi-hard mining; pakai distance-weighted sampling |
| with_geom tetap kalah signifikan di low-data | Sedang | Indikasi bug GAM yang merugikan, bukan netral. Pivot ke F2.8 (cross-attention GAM) atau drop GAM, pakai fusion saja. |
| no_geom tetap perfect di test (leakage tidak teratasi) | **Tinggi** (Opsi A) | Dokumentasikan eksplisit limitation; klaim "low-data robustness" tetap valid; lanjut Opsi B (capture ulang) untuk validasi final |
| GPU time blow-up di iterasi berikutnya | Sedang | Smoke test dulu 1 seed × 2 varian sebelum full run; budget 16 jam total Phase 2 |

---

## Lampiran — File yang Akan Dimodifikasi/Dibuat di Fase 2

**Source code (akan modifikasi):**
- `utils/dataset.py` — `GEOMETRY_KEYS` (drop curvature, drop thumb_width, add scan_distance), `GEOMETRY_DIM = 13`, `_flatten_geometry()` (slice finger_widths[1:5])
- `models/geometry_encoder.py` — sesuaikan `in_dim=13` (atau pakai default `GEOMETRY_DIM`); opsional tambah LayerNorm di akhir (F2.10)
- `models/gam.py` — opsional residual connection + tanh+offset gating (F2.10), atau full FiLM modulation (F2.10b)
- `models/siamese.py` — opsional aux_classifier head untuk auxiliary loss (F2.11)
- `train.py` — argumen `--frames-per-session`, `--loss`, `--val-metric`, `--epochs-phase1/2`, `--use-aux-loss`; hapus early stopping default; loss default → triplet
- `evaluate.py` — adaptasi ke setup low-data
- `utils/augmentation.py` — tambah `random_z_translation`, `random_scale`

**Skrip baru:**
- `utils/audit_geom_discriminability.py` — hitung B/W ratio per feature (untuk re-audit setelah F2.0)
- `utils/sanity_geom_only_cv.py` — geom-only LeaveOneSessionOut CV (sanity baseline F2.0)
- `utils/dataset_lowdata.py` — `OneFramePerSession` loader dengan median frame picker
- `utils/val_pair_metric.py` — val EER/AUC logger per epoch
- `utils/split_temporal.py` — time-gap aware split (untuk Opsi B; opsional)
- `utils/audit_split_temporal.py` — verifikasi time-gap di split keluaran
- `analysis/v5_gap_vs_size.py` — plot dan statistical analysis untuk central finding

**Output rencana:**
- `runs/v5_lowdata/{variant}/seed_{N}/...` — checkpoint, train_log, TensorBoard low-data
- `runs/v5_allframe/{variant}/seed_{N}/...` — replikasi all-frame
- `eval_results/v5_lowdata/{variant}/seed_{N}/...`
- `eval_results/v5_allframe/{variant}/seed_{N}/...`
- `analysis/v5/aggregate_lowdata.csv`, `aggregate_allframe.csv`, `gap_vs_size.png`
- `result_docs/{ts}/v5_lowdata_finding.md` — laporan central finding

**Dokumen referensi:**
- [`KESIMPULAN_REPORT.md`](result_docs/20260522_092309/KESIMPULAN_REPORT.md) — temuan 3 bias
- [`EVALUATION_REPORT.md`](result_docs/20260521_152852/EVALUATION_REPORT.md) — baseline v4.0.0
- [`IMPROVEMENT_PLAN_v0.4.0.md`](IMPROVEMENT_PLAN_v0.4.0.md) — plan sebelumnya untuk konteks evolusi

---

## Klaim Thesis yang Diharapkan

Setelah eksekusi v5.0.0 selesai, klaim thesis yang **paling kuat** (kalau hasil sesuai prediksi):

> "Pada palm identification berbasis depth-only dengan TrueDepth iPhone, hand-crafted geometric features (finger lengths, palm dimensions, palm curvature) yang diintegrasikan via Geometric Attention Module memberikan inductive bias yang signifikan dalam regime enrollment terbatas (1 sampel per sesi capture). Eksperimen pada 11 subjek dengan 10 seed independen menunjukkan that GeoAtt-PointNet++ secara konsisten mengungguli PointNet++ murni di low-data regime (Wilcoxon paired p<0.05, bootstrap CI Δ EER tidak melingkupi nol). Pada regime data berlimpah, perbedaan ini mengecil — menunjukkan bahwa PointNet++ dapat belajar representasi setara secara empiris ketika data cukup, tetapi pada skenario deployment realistis dimana enrollment cepat dibutuhkan, integrasi prior biometric memberikan keunggulan yang menentukan."

Klaim ini **lebih kuat secara metodologis** daripada klaim absolut "GeoAtt > PointNet++" karena:
- Spesifik tentang kapan GeoAtt berguna (low-data, depth-only).
- Tidak konflik dengan finding v4.0.0 (yang masih reproducible di all-frame regime).
- Punya implikasi praktis langsung untuk biometric deployment.
- Konsisten dengan literature ML klasik (inductive bias menang di low-data).
