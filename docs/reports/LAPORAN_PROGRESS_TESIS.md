# Laporan Progress Penelitian Tesis

**Judul:** Identifikasi Telapak Tangan Berbasis 3D CNN Menggunakan TrueDepth Camera
**Nama:** Rahmat Zulfikri
**Program:** Magister Teknik Elektro — Universitas Gadjah Mada
**Tanggal laporan:** 23 April 2026
**Status:** Eksperimen awal — training & evaluasi pertama selesai

---

## 1. Ringkasan Eksekutif

Penelitian ini sedang **membangun sistem identifikasi biometrik telapak tangan berbasis data 3D** menggunakan kamera **TrueDepth** pada iPhone. Pendekatan yang diambil adalah membangun tiga aplikasi terpisah yang dirangkai menjadi satu pipeline end-to-end:

1. **TrueDepthScan** — aplikasi iOS (Swift/SwiftUI) untuk akuisisi *depth frames* dan ekspor data siap proses. **Sudah dibuat dan stabil**.
2. **3DRegistration** — pipeline Python untuk registrasi ICP multi-frame dan ekstraksi 33 fitur geometri biometrik. **Sudah dibuat dan sudah di-*batch run* pada seluruh dataset**.
3. **3DCNN (GeoAtt-PointNet++)** — model *Siamese* PointNet++ yang diperkuat modul perhatian geometri (*Geometric Attention Module*, GAM). **Sudah dibuat dan sudah di-training satu kali**.

**Semua komponen sudah berjalan dan sudah menghasilkan output nyata, tetapi performanya belum sempurna — ini yang akan menjadi fokus iterasi berikutnya.**

Pengambilan data awal dilakukan terhadap **6 subjek** (87 sesi scan), dengan tingkat kelulusan *quality control* **93,1 %** (81 PASS, 1 WARN, 5 FAIL). Training pertama M4 berjalan 100 *epoch* di Google Colab.

**Hasil pertama (yang belum sempurna):**

| Metrik | Nilai | Penilaian |
|---|---:|---|
| Rank-1 accuracy (closed-set, 6 subjek, 19 probe) | **89,5 %** | 17/19 benar — 2 *misclassification* |
| Rank-2 accuracy | 94,7 % | Hampir semua benar dalam 2 kandidat teratas |
| Rank-6 accuracy | 100 % | Seluruh ground-truth selalu masuk top-6 |
| *Train loss* akhir | ~0,001 | Sangat kecil — model menghafal training set |
| *Val loss* akhir | ~0,03 | *Plateau* sejak epoch 40 — **overfitting jelas** |
| Rank-1 (val) sepanjang epoch | berosilasi 70–90 % | Tidak ada tren naik setelah epoch 5 |

**Interpretasi singkat:** pipeline sudah terbukti dapat membedakan identitas (bukti: Rank-1 89,5 %), tetapi ada **dua kelompok masalah besar** yang belum ditangani:

1. **Masalah di level data/akuisisi** — pose scan tidak konsisten, jarak antar jari bervariasi, kalibrasi satuan masih meleset (lihat Bab 7.4), 5 sesi FAIL karena *knuckle detection* rapuh.
2. **Masalah di level training & strategi** — belum ada *fine-tuning* / *early stopping* / regularisasi kuat, belum mencoba *per-frame training* sebagai alternatif *session-level training*, belum eksplorasi efek normalisasi geometri.

Bab 8 & 9 khusus membahas kedua kelompok ini beserta hipotesis perbaikan.

---

## 2. Arsitektur Sistem

```
┌────────────────────────────────────────────────────────────┐
│  TrueDepthScan  (iOS App)                                  │
│  Rekam 10–15 frame Float32 640×480 → depth*.bin           │
│                              + calibration.json            │
│                              + metadata.json (label, hand) │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│  3DRegistration  (Python / Open3D)                         │
│  • Depth → point cloud (per frame)                         │
│  • ICP multi-frame → output.ply (~50k–150k titik)         │
│  • Ekstraksi 33 fitur biometrik → geometry.json            │
│  • PCA + unit-sphere normalization → cnn_input.npy (N×6)   │
│  • FPS 1024 backup → cnn_input_fps.npy (ablation)         │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│  3DCNN  (Python / PyTorch / Google Colab)                  │
│  GeoAtt-PointNet++ Siamese Network                         │
│  3× SetAbstraction + 2× GAM + GeometryEncoder              │
│  → 128-dim L2-normalized embedding                         │
│  → cosine similarity + contrastive loss                    │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Progress Komponen 1 — TrueDepthScan (iOS)

### 3.1 Status: **Selesai dan stabil**

Aplikasi iOS berbasis **Swift + SwiftUI + AVFoundation + Vision** sudah dapat:

| Fungsi | Status |
|---|---|
| Konfigurasi sesi kamera TrueDepth (DepthFloat16, 640×480) | Selesai |
| Sinkronisasi RGB + depth via `AVCaptureDataOutputSynchronizer` | Selesai |
| Live preview 2D (JET depth colormap) dan 3D (Metal point cloud) | Selesai |
| ROI detection telapak tangan menggunakan Vision framework | Selesai |
| Refactor ke ARKit untuk *multi-frame scanning* | Selesai |
| Export `depth*.bin`, `calibration.json`, `metadata.json` | Selesai |
| Scan history, label subjek, *handedness* (kanan/kiri) | Selesai |

### 3.2 Tampilan Aplikasi

**Mode live scan** — *overlay* telapak tangan dengan JET depth colormap (merah = dekat, biru = jauh) dan status `Ready to scan` saat ROI sudah valid:

![Tampilan live scan TrueDepthScan — subjek gede, mode JET colormap dengan ROI telapak aktif](docs/images/ios_scan_ready.png)

**Mode history** — daftar sesi scan per subjek, lengkap dengan *timestamp*, tombol *share* (.ply export) dan hapus:

![Riwayat sesi scan subjek gede — 4 sesi terlihat pada 15 Apr 2026](docs/images/ios_scan_history.png)

### 3.3 Iterasi Arsitektur

Pengembangan aplikasi iOS melewati beberapa refactor besar (tercatat pada folder `.trae/documents/`):

1. **Convert TrueDepthStreamer to SwiftUI** — migrasi dari UIKit sample code Apple ke SwiftUI untuk kemudahan iterasi UI.
2. **Implement CPU-based Point Cloud Generation** — POC konversi depth → point cloud di CPU.
3. **Implement GPU-Based Point Cloud Rendering (Apple Sample Style)** — optimasi dengan Metal shader untuk preview real-time.
4. **Implement Object Scan Feature** & **Implement Object Scan to PLY** — pipeline export awal.
5. **Refactor to ARKit for Multi-Frame Scanning** — iterasi final, menggunakan ARKit agar multi-frame tersinkron.

### 3.4 Format Output per Sesi

```
dataset/
└── <label>_YYYYMMDD_HHMMSS/
    ├── calibration.json   ← fx, fy, cx, cy, distortion lookup
    ├── metadata.json      ← label, handedness, frameCount, depth range
    ├── depth00.bin        ← 640×480 float32
    ├── depth01.bin
    └── …  (10–15 frame)
```

Contoh `metadata.json` nyata dari dataset:

```json
{
  "frameCount":       11,
  "handedness":       "left",
  "height":           480,
  "width":            640,
  "depthMinMeters":   0.1,
  "depthMaxMeters":   0.5,
  "purpose":          "3d-cnn-palm-recognition",
  "label":            "gede",
  "frameDecimation":  5,
  "exportTimestamp":  "2026-04-15T10:12:29+0700"
}
```

### 3.5 Contoh Output Aplikasi iOS (Raw Scan Session)

Setiap sesi scan di aplikasi menghasilkan satu folder dengan nama `<label>_YYYYMMDD_HHMMSS`. Contoh dari sesi **`alji_20260413_091549`** (ukuran total ≈ 11 MB):

```
alji_20260413_091549/
├── calibration.json      ←   764 B   parameter intrinsic + lens distortion
├── metadata.json         ←   293 B   label, handedness, frame count, range
├── depth00.bin           ← 1,17 MB   frame depth float32 640×480
├── depth01.bin           ← 1,17 MB
├── depth02.bin           ← 1,17 MB
├── depth03.bin           ← 1,17 MB
├── depth04.bin           ← 1,17 MB
├── depth05.bin           ← 1,17 MB
├── depth06.bin           ← 1,17 MB
├── depth07.bin           ← 1,17 MB
└── depth08.bin           ← 1,17 MB   total 9 frame untuk sesi ini
```

Ukuran tiap `depth*.bin` = `640 × 480 × 4 byte` = **1 228 800 byte** (Float32 satu kanal).

**1. `metadata.json`** — sidecar informatif untuk pipeline Python:

```json
{
  "frameCount":        9,
  "handedness":        "left",
  "width":             640,
  "height":            480,
  "depthMinMeters":    0.1,
  "depthMaxMeters":    0.5,
  "videoChannels":     0,
  "frameDecimation":   5,
  "purpose":           "3d-cnn-palm-recognition",
  "label":             "alji",
  "exportTimestamp":   "2026-04-13T09:15:53+0700"
}
```

**2. `calibration.json`** — parameter intrinsic TrueDepth + lookup distortion (dua string base64 panjang, dipotong untuk keterbacaan):

```json
{
  "width":           4032,
  "height":          3024,
  "fx":              2715.1184,
  "fy":              2715.1184,
  "cx":              2015.9999,
  "cy":              1512.0,
  "pixelSize":       0.001,
  "lensDistortionCenter":      [2015.9999, 1512.0],
  "lensDistortionLookup":        "AAAAACzcyzhlIMs5...",  
  "inverseLensDistortionLookup": "AAAAAG2Nyrh5scm5..."   
}
```

> Catatan: `fx/fy/cx/cy` di atas dikalibrasi untuk resolusi penuh kamera (4032×3024). Pipeline Python men-*scale* ke resolusi depth 640×480 sebelum melakukan *back-projection* depth → point cloud.

**3. `depth*.bin`** — raw depth frame. Cara membaca di Python:

```python
import numpy as np
depth = np.fromfile("depth00.bin", dtype=np.float32).reshape(480, 640)
# Nilai dalam meter. 0.0 artinya invalid (tidak terdeteksi).

# Statistik frame contoh (alji/depth00.bin):
#   depth range     : 0.0000 .. 0.4663 meter
#   non-zero pixels : 49 526 / 307 200 (16,1 % — sisanya background terpotong)
#   mean (non-zero) : 0.2739 m (~27 cm dari kamera)
```

*Patch* 4×4 piksel dari pusat frame:

```
[[0.2670831  0.2670831  0.26652783 0.2670831 ]
 [0.2670831  0.26652783 0.26652783 0.26652783]
 [0.26764244 0.2670831  0.26652783 0.26652783]
 [0.26764244 0.2670831  0.2670831  0.2670831 ]]
```

Nilai piksel yang hampir seragam menunjukkan permukaan telapak yang datar pada jarak ~26,7 cm dari kamera.

### 3.6 Kendala & Catatan
- **Thermal throttling** — sempat menjadi perhatian saat streaming panjang, tetapi untuk *capture* 10–15 frame per sesi tidak menjadi masalah signifikan.
- **Kalibrasi** — lookup table distortion dari iPhone disimpan apa adanya; saat ini **belum diaplikasikan** pada pipeline Python karena rekonstruksi tanpa koreksi sudah cukup presisi untuk ICP.

---

## 4. Progress Komponen 2 — 3DRegistration (Pipeline Python)

### 4.1 Status: **Selesai, sudah batch-run pada seluruh dataset**

Pipeline Python mengubah 10–15 *depth frame* per sesi menjadi 5 file output yang siap pakai. Seluruh tahap dijalankan via `process_all_scans.py` dalam mode *batch*.

```
depth*.bin + calibration.json + metadata.json
        │
        ▼  run.py  (ICP multi-frame)
output.ply  (~50k–150k titik berisi XYZ + normal)
        │
        ├──▶ extract_geometry.py   ─▶  geometry.json         (33 fitur biometrik)
        ├──▶ extract_texture.py    ─▶  texture.npy           (256×256×5, cadangan)
        ├──▶ preprocess_full()     ─▶  cnn_input.npy         (N×6 full cloud — input utama)
        └──▶ preprocess_fps()      ─▶  cnn_input_fps.npy     (1024×6, backup novelty)
```

### 4.2 Parameter Registrasi ICP

| Parameter | Nilai | Alasan |
|---|---:|---|
| `min_depth` | 0,10 m | Memotong noise dekat sensor |
| `max_depth` | 0,50 m | Membatasi skena hanya pada telapak (≤ 50 cm) |
| `voxel_size` | 0,001 m | Resolusi 1 mm — cukup untuk fitur jari |
| `normal_radius` | 0,008 m | Estimasi normal stabil untuk permukaan halus |
| `outlier_std_ratio` | 1,5 | Pembersihan *statistical outlier* relatif agresif |
| `cluster_connectivity_eps` | 0,008 | DBSCAN memotong *floater* kecil |

### 4.3 Ekstraksi 33 Fitur Geometri

#### 4.3.1 Daftar Fitur

Fitur yang masuk ke **GeometryEncoder** model:

| Kategori | Dim | Keterangan |
|---|---:|---|
| `finger_lengths_mm` | 5 | Panjang jari (ibu, telunjuk, tengah, manis, kelingking) |
| `finger_ratios` | 5 | Panjang tiap jari / jari tengah (*scale-invariant*) |
| `palm_width_mm` | 1 | Lebar di baris *knuckle* (MCP) |
| `palm_height_mm` | 1 | Tinggi telapak dari *wrist* ke MCP |
| `palm_aspect_ratio` | 1 | *width / height* |
| `finger_to_palm_ratios` | 5 | Panjang jari / *palm_height* |
| `inter_finger_gaps_mm` | 4 | Celah horizontal antar jari |
| `finger_widths_mm` | 5 | Lebar anatomis tiap jari |
| `finger_width_to_length_ratios` | 5 | Rasio bentuk tiap jari |
| `mean_palm_curvature` | 1 | Kelengkungan rata-rata permukaan telapak |
| **Total** | **33** | |

Perubahan dari versi sebelumnya (23 → 33 fitur):

- `inter_finger_gaps_mm` menggantikan `inter_finger_depths_mm` — mengukur celah horizontal, lebih stabil terhadap variasi pose.
- Ditambahkan `finger_widths_mm` dan `finger_width_to_length_ratios` sebagai fitur *shape-intrinsic* yang robust terhadap skala.

#### 4.3.2 Pipeline Ekstraksi 8 Tahap (`extract_geometry.py`)

Dari `output.ply` ke `geometry.json`, ekstraksi berjalan melalui 8 tahap berurutan:

```
output.ply (raw)
     │
     ▼
[Tahap 0] PCA canonical alignment → jari → sumbu +Y, depth → +Z
     │
     ▼
[Tahap 1] Deteksi Wrist ROI  (18% bawah Y-range)
     │    └─ output: wrist_center, y_top
     ▼
[Tahap 2] Deteksi Knuckle Row (scan 40 iris Y, cari X-width max)
     │    └─ output: knuckle_y
     ▼
[Tahap 3] Deteksi 5 Fingertip (X-band + dual-signal handedness)
     │    └─ output: tips_mm[5,3] urutan [thumb, index, middle, ring, pinky]
     ▼
[Tahap 4] Panjang jari = tip.y − wrist_center.y
     │    └─ output: finger_lengths_mm[5], finger_ratios[5]
     ▼
[Tahap 5] Dimensi telapak
     │    └─ palm_width_mm  = X-extent di ±10 mm sekitar knuckle_y
     │    └─ palm_height_mm = knuckle_y − wrist_y_top
     │    └─ palm_aspect_ratio = width / height
     ▼
[Tahap 6] Rasio jari terhadap telapak
     │    └─ output: finger_to_palm_ratios[5] = finger_length / palm_height
     ▼
[Tahap 7] Lebar jari & celah antar jari (di zona jari, p5–p95 X-extent)
     │    └─ output: finger_widths_mm[5], inter_finger_gaps_mm[4],
     │               finger_width_to_length_ratios[5]
     ▼
[Tahap 8] Kelengkungan telapak
          └─ output: mean_palm_curvature = mean|1 − |nz|| di area telapak
     │
     ▼
geometry.json (33 nilai)
```

**Penjelasan per tahap:**

**Tahap 0 — PCA Alignment.** Point cloud mentah hasil registrasi ICP tidak punya orientasi yang konsisten (bergantung ke arah mana tangan menghadap kamera). PCA rotasi dipakai supaya:
- Sumbu Y panjang utama (jari → atas, +Y)
- Sumbu X lebar telapak (thumb–pinky)
- Sumbu Z normal telapak (menjauhi kamera)

Semua pengukuran berikutnya memakai koordinat yang sudah "di-luruskan" ini.

**Tahap 1 — Wrist ROI (Region of Interest pergelangan).** Ambil titik-titik di 18 % bagian bawah Y-range. Titik-titik ini diasumsikan bagian pergelangan — menjadi *anchor* yang stabil. Jika jumlah titik < 30 (wrist terpotong), threshold dilebarkan ke 28 %. Dari ROI ini diambil:
- `wrist_center` (titik tengah) — baseline untuk panjang jari
- `y_top` (Y tertinggi ROI) — batas bawah telapak

**Tahap 2 — Knuckle Row (baris MCP).** Scan 40 level Y dari `wrist_y_top` ke 55 % total ketinggian. Untuk setiap level, ukur lebar X (range min–max). **Y dengan lebar X maksimum** = baris MCP (*knuckle*), karena di sinilah kelima tulang metacarpal berjejer paling lebar. Inilah langkah yang rapuh dan menyebabkan 5 sesi FAIL di Bab 7.3.

**Tahap 3 — Fingertip Detection dengan *dual-signal handedness*.** Bagi rentang X menjadi 5 band sama lebar. Ambil titik Y-tertinggi di tiap band → 5 *tip* kandidat berurutan kiri-ke-kanan. Untuk menentukan sisi mana yang thumb (urutkan [thumb, index, middle, ring, pinky]), digunakan **dua sinyal**:
- **Signal A (tinggi):** thumb biasanya sedikit lebih pendek dari pinky → tip Y thumb < Y pinky (~95 % akurat, tetapi gagal kalau thumb bengkok).
- **Signal B (celah):** web space thumb–index lebih lebar dari ring–pinky (~90 % akurat, komplementer).

Kalau `handedness` diketahui dari metadata dan dua sinyal **tidak setuju**, Signal B yang menang (lebih tahan pose). Kalau tidak diketahui, hanya Signal A dipakai.

**Tahap 4 — Panjang jari.** `finger_length[i] = tips_mm[i].y − wrist_center.y`. Hasilnya 5 nilai dalam mm. Kemudian dibagi `finger_length[middle]` untuk dapat `finger_ratios[5]` yang *scale-invariant*.

**Tahap 5 — Dimensi telapak.**
- `palm_height_mm = knuckle_y − wrist_y_top`
- `palm_width_mm` = X-extent titik-titik di irisan ±10 mm sekitar `knuckle_y`
- `palm_aspect_ratio = width / height`

**Tahap 6 — Rasio jari terhadap telapak.** `finger_to_palm_ratio[i] = finger_length[i] / palm_height`. Ini fitur biometrik utama — proporsi jari vs telapak cenderung khas per individu dan tahan variasi skala.

**Tahap 7 — Lebar jari & celah antar jari.** Di zona jari (Y > `knuckle_y`, di bawah 90 % total tinggi), untuk tiap X-band:
- Ambil persentil 5 % dan 95 % dari koordinat X titik di band tersebut.
- `finger_width[i] = x_p95 − x_p5` → robust terhadap *outlier* tepi.
- `inter_finger_gap[i] = band[i+1].x_p5 − band[i].x_p95` → *ruang kosong* nyata antar jari (bukan jarak antar sumbu).

Kalau thumb terdeteksi di sisi kanan (X besar), urutan dibalik supaya tetap `[thumb...pinky]`.

**Tahap 8 — Kelengkungan telapak.** Ambil semua titik di area telapak (antara wrist dan knuckle), lalu hitung `mean(|1 − |n_z||)`. Interpretasi:
- `n_z = 1` → normal vektor sejajar sumbu Z → permukaan rata → nilai 0.
- `n_z = 0` → normal vektor horizontal → tepi/lekukan → nilai 1.

Sehingga `mean_palm_curvature ∈ [0, 1]` dengan 0 = telapak sangat datar, 1 = sangat melengkung.

#### 4.3.3 Normalisasi Z-score Sebelum Training

33 fitur mentah punya satuan dan rentang yang sangat berbeda — `finger_length` bisa 200 mm sementara `finger_ratio` hanya 0,8 sampai 1,0. Kalau langsung masuk ke *neural network*, fitur bersatuan mm akan mendominasi *gradient*.

Solusi: `GeometryNormalizer` (di `3DCNN/utils/normalizer.py`) melakukan **Z-score normalization** per-dimensi:

```python
# Fit hanya pada data training (hindari data leakage)
normalizer = GeometryNormalizer()
normalizer.fit(train_geom_list)   # hitung mean, std dari 33 dim
normalizer.save("normalizer.json")

# Transform saat training dan inference
geom_normalized = (geom - mean) / std   # (33,) → (33,) dengan mean 0, std 1
```

File `normalizer.json` disimpan per-fold di `runs/geoatt_m4/normalizer.json` — berisi `mean[33]` dan `std[33]` hasil fit pada training split, supaya saat inference/evaluasi bisa *reproduce* tepat sama.

#### 4.3.4 Alur Pemanfaatan di Model

Setelah dinormalisasi, vektor 33-dim ini dikonsumsi oleh dua blok model secara berurutan (detail arsitektur di Bab 5.2):

```
geom (33) ─► GeometryEncoder ─► geom_emb (64)
                                     │
                                     ├─► GAM1 di atas SA1 feat (B, 512, 64)
                                     │
                                     ├─► GAM2 di atas SA2 feat (B, 128, 128)
                                     │
                                     └─► Fusion head: concat [global_feat (256) + geom_emb (64)]
                                                     → Linear → 128-dim L2-normed embedding
```

**GeometryEncoder** adalah MLP 3-lapis `Linear(33→64) → BN → ReLU → Linear(64→64) → BN → ReLU → Linear(64→64) → ReLU`. Output 64-dim `geom_emb` berfungsi sebagai **ringkasan biometrik eksplisit** yang menemani jalur PointNet++.

**Geometric Attention Module (GAM)** di setiap SA layer bekerja sebagai berikut:
1. Proyeksikan `geom_emb` (64) → dimensi channel SA (64 atau 128).
2. Gabungkan dengan fitur SA per-titik: `concat = [sa_feat, geom_projected]` (2C channel).
3. Lewatkan ke *attention gate* (2-layer MLP + Sigmoid) → hasilkan `alpha ∈ [0, 1]^C` per titik.
4. Reweight: `output = alpha * sa_feat`.

Efeknya: **fitur point cloud di tiap titik "ditimbang" oleh pemahaman geometri telapak tangan secara global**. Kalau geometri menunjukkan tangan dengan jari sangat panjang, GAM memberi bobot lebih besar pada titik-titik di ujung jari. Ini pembeda utama versus PointNet++ vanilla yang tidak punya konteks global eksplisit.

**Fusion Head** di akhir menggabungkan `global_feat` (256-dim dari SA3) dan `geom_emb` (64-dim) → MLP → 128-dim *embedding* yang di-L2-normalize untuk similarity kosinus.

### 4.4 Contoh Output Fitur

Cuplikan `geometry.json` dari subjek `alji` (sesi pertama):

```json
{
  "scan_id":           "20260413_091549",
  "point_count":       104916,
  "handedness":        "left",
  "finger_lengths_mm": [199.79, 216.11, 257.46, 260.26, 242.73],
  "finger_ratios":     [0.776, 0.8394, 1.0, 1.0109, 0.9428],
  "palm_width_mm":     109.77,
  "palm_height_mm":    109.67,
  "palm_aspect_ratio": 1.0009,
  "mean_palm_curvature": 0.3566
}
```

### 4.5 Representasi untuk Model

- **`cnn_input.npy`** berbentuk `(N, 6)` dengan `N ≈ 50k–150k` (point cloud penuh, ter-PCA-align, ter-normalisasi ke unit sphere). *Sampling* ke 4.096 titik dilakukan **on-the-fly** saat training sehingga setiap epoch menjadi bentuk *implicit augmentation*.
- **`cnn_input_fps.npy`** berbentuk `(1024, 6)`, disediakan sebagai *backup* untuk *ablation study* (membandingkan pengaruh sampling strategi).

### 4.6 Contoh Output Pipeline Registrasi (Processed Session)

Setelah `process_all_scans.py` selesai, struktur hasil untuk sesi yang sama (`alji/20260413_091549`) adalah:

```
result/alji/20260413_091549/
├── output.ply                 ← 5,10 MB   registered point cloud (104 916 titik)
├── geometry.json              ←   720 B   33 fitur biometrik mentah
├── normalized_geometry.json   ←   743 B   turunan scale-invariant dari geometry
├── texture.npy                ← 1,25 MB   proyeksi 2D (256×256×5)
├── cnn_input.npy              ← 2,40 MB   full cloud float32 (104 916 × 6)
└── cnn_input_fps.npy          ←  24,1 KB  FPS backup (1024 × 6)
```

**1. `output.ply`** — point cloud biner Open3D. Header:

```
ply
format binary_little_endian 1.0
comment Created by Open3D
element vertex 104916
property double x
property double y
property double z
property double nx
property double ny
property double nz
property uchar red
property uchar green
property uchar blue
end_header
<binary payload…>
```

Setiap titik menyimpan XYZ (double), normal vektor (double), dan warna RGB (uchar). Untuk sesi `alji/20260413_091549` jumlah titik = **104 916**.

**Visualisasi point cloud** (dibuka di Quick Look macOS) — contoh sesi sukses `rahmat/20260411_202501` dengan pose tangan ideal:

![Sesi sukses rahmat/20260411_202501 — point cloud telapak tangan dengan thumb di kiri, empat jari spread rapi ke kanan, telapak datar menghadap kamera](docs/images/output_ply_rahmat_202501.png)

Statistik `geometry.json` untuk sesi ini: 68 896 titik · `palm_width_mm` 149,17 · `palm_height_mm` 48,48 · `palm_aspect_ratio` 3,08 · `finger_to_palm_max` 3,88 · `mean_palm_curvature` **0,3213** (rendah — telapak datar).

Observasi dari visualisasi:

- Kelima jari jelas terpisah dan terlihat penuh — thumb di kiri, telunjuk–kelingking spread merata ke kanan.
- Distribusi titik padat dan merata — registrasi ICP multi-frame berhasil menggabungkan semua frame tanpa *drift* besar.
- Terlihat sedikit *salt-and-pepper noise* khas output depth sensor TrueDepth — wajar, dan sudah difilter via *statistical outlier removal* (Bab 4.2).
- Pergelangan tangan terpotong bersih di batas ROI — konsisten dengan `min_depth=0.10` dan `max_depth=0.50`.
- `mean_palm_curvature = 0,32` (di bawah median PASS 0,34) — menandakan telapak benar-benar datar menghadap kamera, bukan menekuk atau terputar. Ini adalah **acuan "success"** yang ideal.
- Pose seperti ini memastikan ekstraksi 33 fitur geometri stabil: deteksi wrist–knuckle–fingertip semuanya konsisten, dan semua kriteria QC terpenuhi.

Ini menjadi referensi visual untuk bagaimana **scan sukses** seharusnya terlihat. Pembandingnya ada di Bab 6.3 ketika menelaah sesi FAIL.

**2. `geometry.json`** — 33 fitur biometrik mentah dalam satuan mm / tanpa-unit:

```json
{
  "scan_id":                      "20260413_091549",
  "point_count":                  104916,
  "handedness":                   "left",
  "finger_lengths_mm":            [199.79, 216.11, 257.46, 260.26, 242.73],
  "finger_ratios":                [0.776, 0.8394, 1.0, 1.0109, 0.9428],
  "palm_width_mm":                109.77,
  "palm_height_mm":               109.67,
  "palm_aspect_ratio":            1.0009,
  "finger_to_palm_ratios":        [1.8218, 1.9706, 2.3476, 2.3731, 2.2133],
  "inter_finger_gaps_mm":         [4.14, 3.16, 3.37, 3.84],
  "finger_widths_mm":             [26.62, 31.6, 31.31, 30.58, 29.63],
  "finger_width_to_length_ratios":[0.1332, 0.1462, 0.1216, 0.1175, 0.1221],
  "mean_palm_curvature":          0.3566
}
```

**3. `normalized_geometry.json`** — fitur yang bergantung satuan (mm) dibagi dengan **`palm_width_mm`** sebagai referensi skala, supaya kompatibel dengan scan dari kamera lain atau jarak berbeda:

```json
{
  "scan_id":                "20260413_091549",
  "scale_ref_mm":           109.77,      // palm_width_mm sebagai referensi
  "scale_valid":            true,
  "finger_lengths_norm":    [1.820078, 1.968753, 2.34545, 2.370957, 2.21126],
  "palm_height_norm":       0.999089,
  "inter_finger_gaps_norm": [0.037715, 0.028787, 0.030701, 0.034982],
  "finger_widths_norm":     [0.242507, 0.287875, 0.285233, 0.278582, 0.269928],
  "finger_ratios":          [0.776, 0.8394, 1.0, 1.0109, 0.9428],
  "palm_aspect_ratio":      1.0009,
  "finger_to_palm_ratios":  [1.8218, 1.9706, 2.3476, 2.3731, 2.2133],
  "finger_width_to_length_ratios": [0.1332, 0.1462, 0.1216, 0.1175, 0.1221],
  "mean_palm_curvature":    0.3566
}
```

Fitur `*_ratio` yang sudah dimensionless tidak perlu dinormalisasi lagi — tersimpan apa adanya.

**4. `cnn_input.npy`** — full point cloud siap training. Cara membaca & stats aktual:

```python
import numpy as np
cloud = np.load("cnn_input.npy")
# shape       : (104916, 6)   dtype: float32
# xyz   range : -0.9843 .. 0.8565    (sudah dalam unit sphere)
# normal range: -1.0000 .. 1.0000    (vektor satuan)
```

Tiga baris pertama (masing-masing baris = `[x, y, z, nx, ny, nz]`):

```
[[ 0.10075027 -0.84476840 -0.06793717 -0.28656715  0.07116120 -0.95541370]
 [ 0.19743884 -0.70998275 -0.01574478  0.78673930  0.01628059 -0.61707070]
 [ 0.20315541 -0.70392960 -0.01087067  0.53741735 -0.10625726 -0.83659550]]
```

Koordinat `y` negatif pada titik-titik pertama menandakan ujung jari (yang oleh PCA dipindah ke arah −y sebelum *flip canonical* ke +y).

**5. `cnn_input_fps.npy`** — subset FPS (Farthest Point Sampling) dengan 1024 titik:

```python
np.load("cnn_input_fps.npy").shape   # (1024, 6)
```

Digunakan hanya untuk *ablation study* (membandingkan random sampling vs FPS pada jumlah titik yang sama).

**6. `texture.npy`** — tekstur 2D proyeksi top-down (cadangan untuk eksperimen 2D CNN):

```python
tex = np.load("texture.npy")
# shape : (256, 256, 5)  dtype: float32
# Nilai range per kanal (dari sesi contoh):
#   ch0  depth     : 0.000 .. 0.960
#   ch1  normal_x  : 0.000 .. 1.000
#   ch2  normal_y  : 0.002 .. 1.000
#   ch3  normal_z  : 0.000 .. 0.830
#   ch4  curvature : 0.000 .. 1.000
```

Belum dikonsumsi pipeline saat ini; disiapkan untuk eksperimen hybrid 2D+3D di masa depan.

### 4.7 Dari 11 MB Raw ke 9 MB Processed

| Stage | Ukuran per sesi | Konten |
|---|---:|---|
| Raw (iPhone export) | ~11 MB | 9–11 depth frame + kalibrasi |
| Processed (registrasi) | ~9 MB | PLY + 5 file turunan |
| **Input aktual ke model** | `cnn_input.npy` (2,4 MB) + `geometry.json` (720 B) | full cloud + 33 fitur |

---

## 5. Progress Komponen 3 — 3DCNN: GeoAtt-PointNet++

### 5.1 Status: **Implementasi selesai, training pertama berhasil**

### 5.2 Arsitektur Model (Siamese)

```
                    ┌────────────────────────────────────┐
pts_A (B,N,6) ─────▶│                                    │
                    │        GeoAtt-PointNet++           │──▶ emb_A (B,128) L2-normed
geom_A (B,33) ─────▶│       (shared weights)            │
                    └────────────────────────────────────┘
                                                                │
                                                                ▼ cosine similarity
                    ┌────────────────────────────────────┐     ┌──────────┐
pts_B (B,N,6) ─────▶│        GeoAtt-PointNet++           │──▶ │  sim ∈   │
                    │       (shared weights)            │     │ [-1, 1]  │
geom_B (B,33) ─────▶│                                    │     └──────────┘
                    └────────────────────────────────────┘
```

**Rincian encoder:**

| Blok | Konfigurasi | Output |
|---|---|---|
| SA1 | `n_point=512, radius=0.05, k=32, mlp=[32,32,64]` | `(B, 512, 64)` |
| GAM1 | *attention* antara SA1 dan `geom_emb` | `(B, 512, 64)` |
| SA2 | `n_point=128, radius=0.15, k=64, mlp=[64,64,128]` | `(B, 128, 128)` |
| GAM2 | *attention* antara SA2 dan `geom_emb` | `(B, 128, 128)` |
| SA3 (global) | `n_point=1, radius=5.0, k=128, mlp=[128,256,256]` | `(B, 1, 256)` |
| GeometryEncoder | `MLP 33 → 64 → 64` (pararel) | `(B, 64)` |
| Fusion head | `Linear(256+64 → 256) → BN → ReLU → Linear(256 → 128)` | `(B, 128)` |
| Output | `F.normalize` (L2) | embedding 128-D |

### 5.3 Setup Training

| Parameter | Nilai |
|---|---:|
| Loss | Contrastive loss (margin=0,5) |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | StepLR |
| Batch size | 16 |
| Epoch | 100 |
| `n_points` per item | 4.096 (random sampling) |
| `geom_dim` | 33 |
| Validasi | LOSO (Leave-One-Session-Out) per *fold* |
| Normalisasi geometri | *Z-score* berdasarkan data training saja (hindari *leakage*) |
| Augmentasi | `PointCloudAugmentor` (rotasi kecil, jitter, dropout titik) |

### 5.4 Checkpoint yang Tersimpan

Pada `3DCNN/runs/geoatt_m4/checkpoints/`:

```
best_loss.pth    best_rank1.pth
epoch_010.pth    epoch_020.pth    epoch_030.pth
epoch_040.pth    epoch_050.pth    epoch_060.pth
epoch_070.pth    epoch_080.pth    epoch_090.pth    epoch_100.pth
```

`normalizer.json` berisi *mean* dan *std* dari 33 fitur geometri untuk reproduksi inferensi.

---

## 6. Hasil Training & Evaluasi Awal (Google Colab)

Training dijalankan di Google Colab (GPU T4). Berikut tangkapan layar dari notebook `01_train.ipynb` dan `02_evaluate.ipynb`:

### 6.1 Proses Training

![Kurva training GeoAtt-PointNet++ M4 — contrastive loss (kiri) dan Rank-1 accuracy di val set (kanan), 100 epoch](docs/images/training_curves.png)

**Cuplikan log training (per-epoch)** — dipotong untuk keterbacaan, tanda `★` = *best Rank-1* tersimpan ke `best_rank1.pth`:

```
Epoch   1/100  train=0.1153  val=0.0283  Rank-1=70.0% (7/10)  ★   t=28.5s
Epoch   2/100  train=0.0384  val=0.0304                          t=14.0s
Epoch   3/100  train=0.0332  val=0.0283                          t=13.9s
Epoch   4/100  train=0.0231  val=0.0277                          t=14.2s
Epoch   5/100  train=0.0202  val=0.0350  Rank-1=90.0% (9/10)  ★   t=23.4s
Epoch  10/100  train=0.0099  val=0.0381  Rank-1=70.0% (7/10)      t=24.6s
Epoch  20/100  train=0.0071  val=0.0318  Rank-1=90.0% (9/10)      t=24.9s
Epoch  30/100  train=0.0063  val=0.0368  Rank-1=90.0% (9/10)      t=24.6s
Epoch  40/100  train=0.0023  val=0.0300  Rank-1=90.0% (9/10)      t=24.9s
Epoch  44/100  train=0.0022  val=0.0112                          t=13.8s   ← val loss minimum
Epoch  50/100  train=0.0020  val=0.0319  Rank-1=90.0% (9/10)      t=25.0s
Epoch  64/100  train=0.0015  val=0.0117                          t=14.1s   ← val loss minimum ke-2
Epoch  80/100  train=0.0014  val=0.0313  Rank-1=90.0% (9/10)      t=23.5s
Epoch 100/100  train=0.0010  val=0.0320  Rank-1=80.0% (8/10)      t=22.4s

Selesai. Best Rank-1: 90.0 %   Best val loss: 0.0112
```

**Analisis kurva + log:**

| Aspek | Pengamatan |
|---|---|
| **Train loss** | Turun dramatis dari 0,115 → 0,02 pada 5 *epoch* pertama (6×), lalu perlahan ke 0,001 di *epoch* 100 (total 100×). |
| **Val loss** | Mulai di 0,028, *plateau* di 0,025–0,040 sepanjang training. Ada *spike* di *epoch* 8 (0,052) dan 23 (0,053), serta dua *dip* tajam di *epoch* 44 (**0,0112**) dan 64 (**0,0117**). |
| **Gap train–val** | Signifikan — train → 0,001, val *stagnan* ≈ 0,03. **Indikasi *overfitting* yang kuat** karena dataset hanya 6 subjek. |
| **Rank-1 (val, 10 sesi)** | Berosilasi 70–90 %; pertama mencapai 90 % pada *epoch* 5, lalu sempat turun ke 70 % pada *epoch* 10 & 15. Tidak pernah stabil di atas 90 %. |
| **Konvergensi efektif** | Sudah tercapai **pada *epoch* ~5**. Epoch 6–100 tidak memberi perbaikan Rank-1 yang berarti (paling tinggi tetap 90 % = 9/10). |
| **Durasi** | ~14 s per *epoch* tanpa val, ~24 s per *epoch* dengan val (setiap 5 epoch) → total **~23 menit** untuk 100 *epoch*. |

**Tiga temuan dari log:**

1. **Checkpoint `best_rank1.pth` kemungkinan berasal dari *epoch* 5.** Tanda `★` hanya muncul di *epoch* 1 (70 %) dan *epoch* 5 (90 %). Setelah itu Rank-1 90 % muncul berkali-kali tetapi *tie-break* memakai *epoch* pertama → tidak pernah ter-update. Artinya **95 % sisa training (epoch 6–100) tidak menghasilkan checkpoint yang lebih baik**. Ini bukti kuat bahwa *early stopping* akan sangat membantu (9.3.1).

2. **Dua *dip* val loss dramatis di *epoch* 44 (0,0112) dan 64 (0,0117).** Ini turun **~3× dari plateau**. Bisa dua hal: (a) *lucky batch* — pair yang kebetulan mudah; atau (b) model sesekali menemukan *minima* sempit. Karena Rank-1 di *epoch* 45 justru **turun ke 80 %**, kemungkinan besar ini batch-specific, bukan perbaikan nyata. Validasi ini memperkuat keputusan untuk memakai Rank-1 / val-loss dengan *smoothing* kalau mau menentukan *best checkpoint*.

3. **Perbedaan ukuran validation set antara training vs evaluasi akhir.** Log training memakai **10 sesi val**, sementara notebook evaluasi akhir memakai **19 sesi test** (Bab 6.2). Perbedaan ini perlu diperjelas ke pembimbing — apakah 19 sesi itu subset yang sama (fold ≠ 0) atau *gallery/probe split* yang berbeda.

**Interpretasi keseluruhan:** model sudah belajar *embedding* yang baik sejak *epoch* 5. Sisa 95 *epoch* hanya mempertajam *overfitting* pada training. Solusi utama untuk memperbaiki Rank-1 bukan mengganti arsitektur, melainkan **(a) menambah data** (lihat Bab 9) dan **(b) menghentikan training lebih awal dengan regularisasi lebih kuat** (Bab 8.3.1–8.3.3).

### 6.2 Hasil Identifikasi — Rank-1 per Sesi

Evaluasi *closed-set identification* dilakukan pada 19 sesi test dengan *enrollment database* berisi seluruh 6 subjek (`alji, fadhil, feby, gede, rahmat, taofik`). Setiap sesi probe dibandingkan ke seluruh *enrollment embedding*, dan label prediksi diambil dari *nearest neighbor* (cosine similarity terbesar).

**Rekapitulasi — Rank-1 Accuracy: 17 / 19 = 89,5 %**

| Subjek | Sesi test | Benar | Salah | Rank-1 |
|---|---:|---:|---:|---:|
| alji | 3 | 3 | 0 | 100 % |
| fadhil | 4 | 3 | 1 | 75 % |
| feby | 2 | 2 | 0 | 100 % |
| gede | 4 | 4 | 0 | 100 % |
| rahmat | 2 | 2 | 0 | 100 % |
| taofik | 4 | 3 | 1 | 75 % |
| **Total** | **19** | **17** | **2** | **89,5 %** |

**Kasus salah klasifikasi (perlu ditelaah lebih lanjut):**

| Sesi probe | Prediksi | Skor Rank-1 | Skor ground truth | Margin |
|---|---|---:|---:|---:|
| `fadhil/20260413_094713` | `alji` | 0,9892 | tidak di top-3 | > 0,04 |
| `taofik/20260413_092743` | `rahmat` | 0,9841 | `taofik` @ 0,9781 (Rank-2) | **0,006** |

Catatan:
- Kasus `taofik/…743` adalah *near-miss* — margin hanya 0,006; calibrasi *threshold* atau *TTA* (test-time augmentation) berpotensi memperbaikinya.
- Kasus `fadhil/…713` lebih berat — `fadhil` bahkan tidak muncul di top-3. Menariknya sesi ini adalah **sesi pertama** subjek `fadhil` (awal perekaman, kemungkinan pose belum stabil). Sesi `fadhil` lainnya semuanya BENAR.

**Contoh output identifikasi yang benar (3 sampel acak):**

```
Sesi   : gede/20260415_101216
Label  : gede            → Prediksi: gede      [✓]
  Rank-1: gede    0.9998 ← prediksi
  Rank-2: alji    0.9302
  Rank-3: feby    0.8638

Sesi   : alji/20260413_091549
Label  : alji            → Prediksi: alji      [✓]
  Rank-1: alji    0.9982 ← prediksi
  Rank-2: feby    0.9682
  Rank-3: rahmat  0.9376

Sesi   : rahmat/20260411_202341
Label  : rahmat          → Prediksi: rahmat    [✓]
  Rank-1: rahmat  0.9898 ← prediksi
  Rank-2: fadhil  0.9614
  Rank-3: feby    0.9609
```

**Observasi tambahan:**
- Rata-rata skor Rank-1 pada kasus BENAR > 0,99 — pemisahan *intra*-subjek vs *inter*-subjek tajam.
- Subjek `gede` memiliki skor Rank-1 paling diskriminatif (0,9993–0,9998) — kemungkinan karena pose konsisten dan kualitas registrasi terbaik.
- Subjek `feby` dan `rahmat` sering menjadi Rank-2/Rank-3 untuk probe lain — menandakan *embedding*-nya berada relatif di tengah *space* identitas. Perlu evaluasi lanjutan apakah ini properti dataset atau bias model.

### 6.3 CMC Curve

![CMC Curve — Rank-1 89,5 %, Rank-2–5 ≈ 94,7 %, Rank-6 100 %](docs/images/cmc_curve.png)

| Rank | Identification Rate |
|---:|---:|
| 1 | 89,5 % |
| 2 | 94,7 % |
| 3 | 94,7 % |
| 4 | 94,7 % |
| 5 | 94,7 % |
| 6 | 100 % |

Interpretasi: hanya satu kasus yang identitasnya sangat jauh (perlu sampai Rank-6 untuk tertangkap) — kemungkinan besar `fadhil/…713`, sesi awal yang posenya menyimpang. Kasus lain yang salah di Rank-1 langsung tertangkap di Rank-2, menegaskan bahwa *embedding*-nya sudah dekat dengan ground truth.

### 6.4 Confusion Matrix

![Confusion matrix — diagonal dominan, dua sel off-diagonal: fadhil→alji (1) dan taofik→rahmat (1)](docs/images/confusion_matrix.png)

| True \\ Pred | alji | fadhil | feby | gede | rahmat | taofik |
|---|---:|---:|---:|---:|---:|---:|
| alji | **3** | 0 | 0 | 0 | 0 | 0 |
| fadhil | 1 | **3** | 0 | 0 | 0 | 0 |
| feby | 0 | 0 | **2** | 0 | 0 | 0 |
| gede | 0 | 0 | 0 | **4** | 0 | 0 |
| rahmat | 0 | 0 | 0 | 0 | **2** | 0 |
| taofik | 0 | 0 | 0 | 0 | 1 | **3** |

Diagonal dominan (17 dari 19 sel on-diagonal). Dua *off-diagonal* mengkonfirmasi kasus yang sama: `fadhil → alji` dan `taofik → rahmat`.

### 6.5 Distribusi Similarity per Sesi

![Similarity score per orang untuk tiap sesi test — bar hijau = prediksi benar, bar merah = prediksi salah, bar biru = subjek enrollment lain](docs/images/similarity_per_person.png)

Tiap panel merepresentasikan satu *probe session*. Sumbu-x adalah 6 subjek enrollment, sumbu-y adalah *cosine similarity*. Bar **hijau** = prediksi sesuai ground truth (kasus BENAR), bar **merah** = prediksi tidak sesuai ground truth (kasus SALAH).

**Pola yang terlihat:**
- Mayoritas panel menunjukkan *peak* hijau yang jauh lebih tinggi dari bar biru lainnya — *margin* pemisahan jelas.
- Dua panel dengan bar merah tinggi mengkonfirmasi dua kasus *misclassification* (`fadhil/…713` dan `taofik/…743`) — pada panel tersebut, bar untuk ground-truth subject justru lebih pendek dari bar subject lain.
- Beberapa panel menunjukkan beberapa subjek memiliki skor relatif tinggi (>0,9) — menandakan *embedding space* masih cukup padat di rentang atas (kesulitan karena dataset kecil).

### 6.6 Ablation Study — Status Saat Ini

Output notebook ablation terbaru:

```
Skip M1 Baseline   — checkpoint tidak ditemukan
Skip M2 +Curvature — checkpoint tidak ditemukan
Skip M3 +GAM       — checkpoint tidak ditemukan

=======================================================
ABLATION STUDY — Identifikasi Telapak Tangan
=======================================================
Model           Rank-1   Rank-3   Rank-5
-------------------------------------------------------
M4 GeoAtt        94.7%    94.7%    94.7%
=======================================================
```

**Catatan:** hanya M4 yang sudah memiliki *checkpoint* terlatih. Tiga varian pembanding (M1 Baseline, M2 +Curvature, M3 +GAM) **belum dijalankan trainingnya** — ini menjadi item prioritas di rencana tindak lanjut (Bab 9) agar ablation bisa dibaca secara komparatif.

![CMC Curve Ablation Study — hanya M4 GeoAtt yang ter-plot (Rank-1 94,7 %, Rank-6 100 %) karena M1/M2/M3 belum ter-train](docs/images/cmc_ablation.png)

Kurva di atas secara visual menyerupai CMC utama (Bab 6.3), tetapi protokolnya berbeda. Setelah M1/M2/M3 dilatih, plot ini akan menampilkan 4 garis berdampingan untuk memperlihatkan kontribusi tiap komponen (GAM, GeometryEncoder, curvature).

Perhatikan juga bahwa angka Rank-1 ablation (**94,7 %**) berbeda dari evaluasi identifikasi utama (**89,5 %**). Perbedaan ini berasal dari **protokol evaluasi yang berbeda** (mis. *gallery/probe split* tidak sama atau menggunakan pair-wise verification di notebook ablation vs nearest-neighbor closed-set di notebook utama). Perlu disamakan protokolnya sebelum dilaporkan ke pembimbing secara final.

### 6.7 Ringkasan Metrik

| Protokol | Model | Rank-1 | Rank-2/3 | Rank-6 | Catatan |
|---|---|---:|---:|---:|---|
| Closed-set identification (19 probe, 6 enrollment) | **M4 GeoAtt** | **89,5 %** | 94,7 % | 100 % | Hasil utama, 17/19 benar |
| Ablation notebook | **M4 GeoAtt** | **94,7 %** | 94,7 % | — | Protokol perlu diverifikasi |
| Ablation notebook | M1 Baseline | — | — | — | Belum di-train |
| Ablation notebook | M2 +Curvature | — | — | — | Belum di-train |
| Ablation notebook | M3 +GAM | — | — | — | Belum di-train |

*Catatan:* notebook evaluasi saat ini **belum menghitung EER / AUC / ROC**. Untuk identifikasi *closed-set*, Rank-k CMC adalah metrik utama. Kalau nanti beralih ke *verification* atau *open-set* (Sprint 4 di Bab 9), EER/AUC perlu ditambahkan ke `02_evaluate.ipynb` dengan cell baru yang menghitung pasangan similarity *genuine vs impostor*.

---

## 7. Status Dataset

### 7.1 Statistik Akuisisi

Dataset yang sudah direkam dan di-proses:

| Subjek | Jumlah sesi | PASS | WARN | FAIL |
|---|---:|---:|---:|---:|
| alji | 16 | 14 | 0 | 2 |
| fadhil | 17 | 16 | 0 | 1 |
| feby | 10 | 10 | 0 | 0 |
| gede | 17 | 14 | 1 | 2 |
| rahmat | 10 | 10 | 0 | 0 |
| taofik | 17 | 17 | 0 | 0 |
| **Total** | **87** | **81 (93,1 %)** | **1 (1,1 %)** | **5 (5,7 %)** |

### 7.2 Analisis Kegagalan QC

Laporan `qc_summary.txt` mencatat kegagalan pada 5 sesi; semuanya bersumber dari **kegagalan deteksi knuckle** di `extract_geometry.py`, yang menyebabkan `palm_height_mm = 0` dan `finger_to_palm_ratios` meledak:

| Sesi | Root cause |
|---|---|
| `alji/20260413_091603` | Knuckle tidak terdeteksi → `palm_height=0` |
| `alji/20260413_091619` | Sama |
| `fadhil/20260413_095031` | `palm_aspect_ratio=47.2`, *ordering* jari salah (pinky > middle) |
| `gede/20260415_101308` | `max(finger_to_palm)=10.23` |
| `gede/20260415_101327` | `palm_height=9.82 mm` (terlalu kecil) |

Satu *warning* (`gede/20260415_101438`, `palm_height=38,9 mm`) masih dianggap *passable* tetapi menandakan pose terlalu dekat atau tangan belum membuka sempurna.

### 7.3 Telaah Mendalam 5 Sesi FAIL + 1 WARN

Seluruh `geometry.json` dari sesi bermasalah sudah dibuka dan dibandingkan dengan baseline PASS. Berikut adalah **nilai referensi median dari 81 sesi PASS** untuk acuan:

| Fitur | Median PASS | Keterangan |
|---|---:|---|
| `palm_width_mm` | 131,6 | Lebar telapak normal ~11–14 cm |
| `palm_height_mm` | 87,3 | Tinggi telapak normal 6–11 cm |
| `palm_aspect_ratio` | 1,50 | Proporsi wajar 1,0–2,0 |
| `finger_length[middle]` | 208,2 mm (!) | Angka ini **jauh lebih besar** dari jari manusia sesungguhnya (≈ 80–90 mm) → indikasi **bug skala konsisten di seluruh dataset** (lihat 8.4) |
| `max(finger_to_palm_ratio)` | ≤ 3,5 | Rasio masuk akal |
| `mean_palm_curvature` | 0,34 | Telapak relatif datar |

**7.3.1 FAIL 1 & 2 — `alji/20260413_091603` dan `alji/20260413_091619`**

Kedua sesi memiliki pola yang identik:

| Fitur | Nilai aktual | Median PASS | Diagnosa |
|---|---:|---:|---|
| `palm_height_mm` | **0,00** | 87,31 | **Knuckle tidak terdeteksi** — algoritme fallback ke 0 |
| `palm_width_mm` | 94,9 / 98,1 | 131,6 | Wajar-wajar saja |
| `finger_to_palm_ratios` (max) | **291,8 / 299,9** | ≤ 3,5 | Meledak karena pembagian dengan nol (dibulatkan) |
| `finger_lengths` (middle) | 291,6 / 299,2 mm | 208,2 | Sedikit lebih panjang — wajar, point cloud lebih besar |
| `point_count` | 126 214 / 121 218 | ~100k | Scan normal |

**Lokasi bug di kode:** `extract_geometry.py` → pencarian *knuckle row* (baris MCP) — algoritme mencari `y` dengan lebar X-range maksimum **di atas wrist zone (18 % bawah)**. Pada sesi ini, telapak kemungkinan terlalu tegak atau *wrist zone* terlalu lebar, sehingga tidak ada `y` yang memenuhi kriteria → `palm_height = max_y_knuckle − max_y_wrist = 0`.

**Rekomendasi:** tambahkan fallback berbasis histogram lebar X (mencari local-max) bila kriteria utama gagal.

**Bukti visual — perbandingan `output.ply` PASS vs FAIL pada subjek yang sama:**

| Sesi PASS (`alji/091549`) | Sesi FAIL (`alji/091603`) |
|:---:|:---:|
| ![PASS — palm_height=109,67 mm, knuckle terdeteksi normal](docs/images/output_ply_alji_091549.png) | ![FAIL — palm_height=0, tapi point cloud VISUAL terlihat normal](docs/images/output_ply_alji_091603.png) |
| 104 916 titik · `palm_height = 109,67 mm` · `finger_to_palm_max = 2,37` | 126 214 titik · **`palm_height = 0,0 mm`** · **`finger_to_palm_max = 291,82`** |

**Kunci temuan:** kedua point cloud terlihat **hampir identik secara visual** — jari terpisah baik, telapak lengkap, *wrist* terpotong bersih. Tidak ada indikasi bahwa scan FAIL ini punya kualitas data yang lebih buruk (bahkan `point_count`-nya **lebih tinggi**).

Artinya **kegagalan sepenuhnya ada di lapisan ekstraksi fitur (`extract_geometry.py`), bukan di aplikasi iOS atau registrasi ICP.** Ini memperkuat argumen Bab 8.2.3: heuristic deteksi *wrist zone* + *knuckle row* rapuh terhadap variasi pose kecil yang tidak terlihat mata telanjang, meskipun variasi itu tidak cukup besar untuk mengubah bentuk point cloud keseluruhan. Fix MediaPipe Hand akan menyelesaikan kelima kasus FAIL tanpa perlu scan ulang.

**7.3.2 FAIL 3 — `fadhil/20260413_095031`**

Kasus paling menarik, dengan **dua bug berlapis**:

| Fitur | Nilai aktual | Median PASS | Diagnosa |
|---|---:|---:|---|
| `point_count` | **60 126** | ~100k | Scan kurang komplit — tangan mungkin keluar ROI |
| `palm_height_mm` | **2,04** | 87,31 | Knuckle terdeteksi, tapi hampir di posisi wrist (terlalu rendah) |
| `palm_aspect_ratio` | **47,22** | 1,50 | Konsekuensi langsung dari palm_height yang kecil |
| `finger_lengths_mm` | [118, 129, 136, 161, **191**] | middle > pinky | **Urutan terbalik** — kelingking terdeteksi terpanjang |
| `finger_ratios[4]` (kelingking) | **1,4073** | < 1,0 | Anomali — kelingking > tengah (seharusnya pinky < middle) |

**Root cause:** kombinasi dua masalah.
1. *PCA orientation flip* — ketika `handedness = left` tapi point cloud mengarah sebaliknya, urutan jari jadi terbalik dari ibu → kelingking menjadi kelingking → ibu.
2. Wrist zone & knuckle hampir tumpang-tindih → `palm_height` hampir nol.

Sinyal yang bisa di-trust di sini: `finger_ratios[4] > 1,3` adalah *red flag* yang sudah ditangkap QC. **Sinyal ganda** yang sebelumnya ditambahkan (tinggi jari + celah ibu-telunjuk vs manis-kelingking) **tidak cukup untuk menahan kasus ekstrem ini**.

**Rekomendasi:** tambahkan *sanity check* berbasis `point_count < 0,6 × median subjek` sebagai *early reject* sebelum ekstraksi geometri.

**7.3.3 FAIL 4 & 5 — `gede/20260415_101308` dan `gede/20260415_101327`**

Keduanya gagal karena `palm_height` terlalu kecil, tapi tingkatannya berbeda:

| Fitur | 101308 | 101327 | Median PASS |
|---|---:|---:|---:|
| `palm_height_mm` | 29,37 | **9,82** | 87,31 |
| `palm_aspect_ratio` | 2,67 | **8,63** | 1,50 |
| `max(finger_to_palm)` | 10,23 | 30,66 | ≤ 3,5 |
| `finger_widths_mm` | 26–31 | 26–31 | konsisten |
| `mean_palm_curvature` | **0,45** | **0,48** | 0,34 |

Pola spesifik pada `gede`:
- `mean_palm_curvature` di atas 0,45 pada keempat sesi FAIL/WARN (baseline median 0,34) → menandakan *point cloud* mengandung terlalu banyak area tepi / jari yang melengkung daripada area telapak yang datar. Kemungkinan besar **pose tangan menutup sebagian** saat di-scan sehingga bagian tengah telapak tidak tertangkap kamera, menyebabkan wrist dan knuckle berdekatan.
- `palm_width_mm` di kisaran 78–84 (jauh di bawah median 131,6) → lebar telapak tidak terukur penuh.

**Rekomendasi:** instruksikan subjek `gede` untuk re-scan dengan tangan lebih terbuka dan menjauh dari kamera sekitar 5 cm.

**7.3.4 WARN — `gede/20260415_101438`**

| Fitur | Nilai | Diagnosa |
|---|---:|---|
| `palm_height_mm` | 38,93 | Di bawah threshold WARN (40 mm) tetapi masih > 0 |
| `palm_aspect_ratio` | 2,01 | Masih dalam rentang wajar |
| `max(finger_to_palm)` | 7,64 | Tinggi, tapi belum sampai 10 |
| `mean_palm_curvature` | 0,45 | Masih menunjukkan pose belum ideal |

Sesi ini **tidak di-exclude** dari training, tapi menandakan kualitas *borderline*. Bila hasil model sensitif terhadap satu-dua sesi ini, perlu dipertimbangkan untuk membuang.

**7.3.5 Ringkasan Root Cause per Bug**

| Root cause | Sesi terdampak | Fix yang direkomendasikan |
|---|---|---|
| *Knuckle detection* gagal → `palm_height = 0` | `alji/091603`, `alji/091619` | Fallback histogram X-width |
| Pose tangan tertutup → wrist ≈ knuckle | `gede/101308`, `gede/101327`, `gede/101438` (WARN) | Re-scan dengan instruksi pose; validasi `mean_palm_curvature < 0,42` |
| Orientasi PCA terbalik + pose menyimpang | `fadhil/095031` | *Early reject* berdasarkan `point_count`; sanity-check `finger_ratios[4]` ≤ 1,1 |

### 7.4 Temuan Tambahan — Kesalahan Skala Global

Saat menelaah median PASS, ditemukan **median `finger_length[middle] = 208,2 mm`** — ini **dua kali lipat** panjang jari tengah manusia nyata (~80–90 mm). Artinya seluruh pipeline sebenarnya menghasilkan nilai dalam **satuan ×2** dari yang diharapkan.

Hipotesis akar:
1. *Voxel grid* registrasi mungkin mengalami *rescaling* yang tidak ter-inverted.
2. Atau formula konversi mm ke koordinat Open3D meleset (2× faktor).

**Tidak** mempengaruhi akurasi identifikasi (fitur tetap diskriminatif antar subjek), karena semua sesi terkena bias skala yang sama. Tetapi perlu diperbaiki sebelum **publikasi angka absolut** atau **cross-device comparison**.

**Rekomendasi jangka dekat:** tambahkan *unit test* `finger_length < 120 mm` di akhir `extract_geometry.py` dan telaah formula konversi.

### 7.5 Tindak Lanjut Dataset

- *Excludes list* sesi FAIL dari *training split* (sudah terfasilitasi via `dataset_manifest.json`).
- Melakukan *re-scan* pada subjek yang sesi FAIL terlalu banyak (alji & gede) untuk menyamakan jumlah sesi per subjek.
- Perlu rekrut **minimal 14 subjek tambahan** untuk mencapai target minimal 20 subjek → ini akan dimulai pada minggu depan.

---

## 8. Analisis Keterbatasan & Hipotesis Perbaikan

Bab ini adalah **fokus utama laporan**. Setelah pipeline end-to-end berjalan dan memberikan *baseline* Rank-1 89,5 %, tampak beberapa sumber ketidaksempurnaan yang **bukan satu bug tunggal**, melainkan keputusan desain awal yang belum dioptimalkan. Aku kelompokkan menjadi tiga tingkat: **data**, **pipeline/representasi**, dan **strategi training**.

Kalimat kunci: *pipeline sudah jalan dan menghasilkan angka, tapi setiap bagian masih punya "utang" iterasi yang belum dibayar.*

### 8.1 Keterbatasan di Level Data (Akuisisi)

**8.1.1 Pose scan tangan tidak konsisten antar-sesi**

Saat ini subjek hanya diminta "letakkan telapak di depan kamera". Tidak ada *jig* fisik atau *visual guide* yang memastikan:

- Jarak tangan ke kamera (terbukti di Bab 6.3 kasus `gede` — *curvature* 0,45–0,48 menandakan tangan tidak rata / terlalu dekat).
- Sudut kemiringan tangan terhadap bidang kamera.
- Apakah kelima jari sudah terbuka penuh atau sebagian tertutup.

Konsekuensi: **variabilitas *intra*-subjek yang berlebihan** — 10 sesi subjek yang sama bisa menghasilkan 10 *embedding* yang cukup berbeda. Inilah alasan *val loss* sulit turun meski training loss hampir nol: model melihat *noise pose* sebagai sinyal identitas.

*Hipotesis perbaikan:*
- Tambahkan *overlay guide* di aplikasi iOS (siluet tangan referensi) sebagai panduan posisi.
- Tampilkan umpan balik real-time *flatness* — status `Ready to scan` hanya muncul bila telapak cukup rata.
- Standarisasi jarak: hanya terima frame bila `depthMean ∈ [25 cm, 35 cm]`.
- **Notifikasi error eksplisit saat scan gagal** — misal banner "Pose tidak valid: rentangkan jari", "Tangan terlalu dekat/jauh", atau "Tangan tidak rata". Saat ini aplikasi menerima scan apapun dan baru ketahuan gagal di tahap pipeline Python, padahal subjek sudah pulang. Feedback real-time akan mempercepat iterasi akuisisi dan memastikan *konsistensi pose* yang memang jadi akar banyak masalah di Bab 6.3.

**8.1.2 Celah antar jari (*inter-finger gap*) tidak terkontrol**

Fitur `inter_finger_gaps_mm` masuk ke 33-dim geometri langsung ke model. Tetapi celah ini sangat bergantung pada *seberapa lebar subjek merenggangkan jari saat scan* — bukan murni anatomi tangannya.

Bukti nyata dari dataset:

| Sesi | `inter_finger_gaps_mm` |
|---|---|
| `alji/091549` (PASS) | [4.14, 3.16, 3.37, 3.84] |
| `alji/091603` (FAIL) | [5.76, 2.81, 2.84, 5.29] |

Subjek yang sama, dua sesi berbeda — pola celah **sangat berbeda**. Artinya fitur ini **bukan murni biometrik**, tetapi **biometrik + pose**.

*Hipotesis perbaikan:*
- Instruksikan subjek: "rapatkan jari" agar `inter_finger_gaps` dekat nol dan stabil (tidak jadi noise).
- Atau sebaliknya: *drop* fitur ini dari GeometryEncoder input dan lihat dampaknya via ablation.

**8.1.3 Normalisasi data geometri belum dikonsumsi model**

Pipeline menghasilkan **dua versi** fitur geometri per sesi:

| File | Konten | Dipakai training saat ini? |
|---|---|:---:|
| `geometry.json` | 33 fitur mentah dalam mm | **Ya** — masuk ke `GeometryEncoder` |
| `normalized_geometry.json` | Versi *scale-invariant* (dibagi `palm_width_mm`) | **Tidak** — disimpan tetapi tidak dipakai |

Saat ini Z-score *normalizer* (lihat `normalizer.json`) dipakai **sebelum** masuk model, tetapi itu hanya memusatkan distribusi — **tidak menghilangkan ketergantungan pada skala absolut**. Jika nanti dataset ditambah dari kamera lain atau jarak scan berbeda, model akan kebingungan.

*Hipotesis perbaikan:*
- *Swap* input GeometryEncoder dari fitur mentah → `normalized_geometry.json` (field `*_norm` + field `*_ratio`).
- Uji apakah ini menaikkan ketahanan *cross-session* atau tidak.

**8.1.4 Kesalahan skala global (lihat Bab 7.4)**

Median `finger_length[middle] = 208 mm` padahal jari manusia nyata ~80–90 mm → ada **faktor ×2** yang meleset di konversi satuan. Tidak memengaruhi identifikasi relatif karena semua sesi terkena bias yang sama, tetapi menunjukkan **validasi satuan belum ada**.

*Hipotesis perbaikan:* *assertion* sederhana `50 < finger_length_middle < 120` di akhir `extract_geometry.py`.

### 8.2 Keterbatasan di Level Pipeline / Representasi

**8.2.1 Per-frame vs Registered-Object — keputusan representasi yang belum dieksplorasi**

Pipeline saat ini memperlakukan **1 sesi = 1 training sample**: 10–15 *depth frames* digabung via ICP → `output.ply` tunggal → `cnn_input.npy` tunggal → 1 *pair* training.

Alternatif yang belum dicoba: **per-frame training**, di mana **tiap frame → 1 sample**.

| Aspek | Session-level (saat ini) | Per-frame (alternatif) |
|---|---|---|
| Sample per subjek | 10–17 | **100–200** (~10× lebih banyak) |
| Kualitas per sample | Tinggi (hasil registrasi ICP) | Lebih rendah (single frame, lebih noisy) |
| Variabilitas sample | Rendah (sudah di-*average*) | Tinggi (augmentasi alami) |
| Cocok untuk dataset kecil? | *Overfit* cepat | Lebih ideal — seperti *implicit augmentation* |
| Cocok untuk identifikasi pada inference? | Perlu tetap register di client | Bisa langsung 1 frame — lebih cepat |

Dukungan di kode **sudah ada** — `train.py` punya auto-detection via keberadaan subdirektori `frame_*/` (lihat fungsi `_is_frame_layout`). Jadi ini tinggal menjalankan, bukan coding.

*Hipotesis perbaikan:* jalankan per-frame training sebagai *ablation* utama. Hipotesis: Rank-1 akan naik karena *effective dataset size* naik 10×.

**8.2.2 Ketergantungan terhadap ICP yang rapuh**

Jika ICP gagal *converge* untuk satu frame dalam sesi, seluruh `output.ply` bisa rusak. Karena kita hanya menyimpan hasil registrasi, sulit memverifikasi *per-frame quality* tanpa regenerasi.

*Hipotesis perbaikan:* simpan *per-frame point cloud* sebelum ICP (`frame_00.ply ... frame_NN.ply`) sebagai cadangan dan sekaligus input per-frame training.

**8.2.3 *Feature extractor* geometri berbasis heuristik**

Seluruh 5 sesi FAIL di Bab 6.3 disebabkan oleh dua heuristik yang sama — *wrist zone* 18 % bawah Y-range + *knuckle row* = Y dengan lebar X maksimum. Tidak ada fallback berbasis *landmark* yang lebih kuat.

*Hipotesis perbaikan:* integrasi MediaPipe Hand di atas *top-down depth projection* untuk mendapatkan 21 landmark jari yang jauh lebih stabil dan deterministik.

### 8.3 Keterbatasan di Level Strategi Training

**8.3.1 Belum ada *early stopping***

Dari kurva training (Bab 6.1), *val loss* sudah *plateau* sejak *epoch* ~5. Training sampai *epoch* 100 **hanya mempertajam overfitting**. Saat ini checkpoint terbaik diambil manual dari `best_rank1.pth`, tapi karena Rank-1 berosilasi 70–90 %, pemilihan ini *noisy*.

*Hipotesis perbaikan:* implementasi *early stopping* dengan `patience = 10 epoch` berdasarkan **val loss** (lebih halus dari Rank-1).

**8.3.2 Belum ada *fine-tuning*, *warm-up*, atau LR scheduling canggih**

Model dilatih *from scratch* dengan `StepLR` default. Tidak ada:

- *Warm-up* LR di awal.
- *Cosine annealing* atau *OneCycle*.
- *Pre-training* pada dataset point cloud besar (ShapeNet, ModelNet) lalu *fine-tune*.

Untuk dataset sekecil ini, *pre-training + fine-tuning* bisa sangat membantu konvergensi awal.

*Hipotesis perbaikan:*
- Uji *cosine annealing* dengan T_max = 50 epoch.
- *Pre-train* SA layers di ModelNet40 (klasifikasi objek) lalu *fine-tune* di telapak tangan.

**8.3.3 Regularisasi minimal**

Model saat ini hanya mengandalkan *L2 normalization* pada *embedding* output dan augmentasi ringan (rotasi kecil, jitter). **Tidak ada:**

- Dropout di *fusion head* atau GeometryEncoder.
- *Weight decay* di optimizer (Adam, default = 0).
- Augmentasi agresif (*point cloud cutmix*, *random scaling*, *erasure*).

Dengan hanya 6 subjek, regularisasi yang ada **tidak cukup kuat** untuk mencegah menghafal.

*Hipotesis perbaikan:* tambahkan `weight_decay=1e-4`, dropout `p=0.3` di *fusion head*, augmentasi rotasi sumbu-Z ±15°.

**8.3.4 Evaluasi baru 1 fold (bukan proper LOSO)**

`train.py` sebenarnya mendukung *Leave-One-Session-Out*, tetapi angka 89,5 % yang dilaporkan **hanya dari 1 fold** (`fold_0`). Variansi antar-fold belum diketahui — angka sebenarnya bisa 75 % atau 95 % di fold lain.

*Hipotesis perbaikan:* jalankan `--all_folds`, laporkan `mean ± std` Rank-1 di semua fold.

**8.3.5 Baseline pembanding (M1/M2/M3) belum ada**

*Ablation* yang sudah dijalankan hanya memuat M4 (lihat output Bab 6.6). Kontribusi GAM dan GeometryEncoder **tidak bisa dibuktikan** tanpa membandingkan varian tanpa komponen tersebut.

*Hipotesis perbaikan:* train berurutan:
- **M1** — PointNet murni.
- **M2** — PointNet++ base (tanpa GAM, tanpa GeometryEncoder).
- **M3** — PointNet++ + GeometryEncoder (tanpa GAM).
- **M4** — saat ini (PointNet++ + GeometryEncoder + GAM).

Kemudian plot CMC M1–M4 dalam satu gambar.

### 8.4 Ringkasan Prioritas Perbaikan

| Prioritas | Item | Effort | Dampak potensial |
|---|---|:---:|---|
| 🔥 Tinggi | *Early stopping* + *weight decay* + dropout | Kecil | Stabilkan Rank-1 val, kurangi *overfit* |
| 🔥 Tinggi | Tambah subjek ≥ 20 | Besar | Solusi fundamental overfit |
| 🔥 Tinggi | LOSO *all folds* | Kecil | Angka yang bisa dilaporkan secara jujur |
| ⚡ Menengah | Per-frame training (9.2.1) | Menengah | 10× sample → potensi naik signifikan |
| ⚡ Menengah | Fix skala satuan (8.4) | Kecil | Kredibilitas angka absolut |
| ⚡ Menengah | Baseline M1/M2/M3 (9.3.5) | Menengah | Membuktikan kontribusi GAM |
| 💡 Rendah | MediaPipe Hand integrasi (9.2.3) | Menengah | Selesaikan 5 sesi FAIL |
| 💡 Rendah | Swap ke `normalized_geometry.json` (9.1.3) | Kecil | Cross-device robustness |
| 💡 Rendah | Pose guide di iOS app (9.1.1) | Menengah | Konsistensi data akuisisi |

---

## 9. Rencana Tindak Lanjut

Mapping langsung dari Bab 8 ke timeline kerja:

### 9.1 Sprint 1 — Perbaikan murah & berdampak tinggi (1–2 minggu)

*Item ini semuanya adalah kode satu-hari, tidak perlu scan ulang.*

- [ ] **Training hygiene** (9.3.1–9.3.3): tambahkan *early stopping* (patience=10, kriteria val loss), `weight_decay=1e-4`, dropout `p=0.3` di *fusion head*, dan augmentasi rotasi sumbu-Z ±15°.
- [ ] **LOSO all folds** (9.3.4): jalankan `python train.py --all_folds` lalu laporkan `mean ± std` Rank-1.
- [ ] **Baseline M1/M2/M3** (9.3.5): train tiga varian pembanding supaya CMC ablation bermakna.
- [ ] **Fix skala satuan** (8.4 / 9.1.4): *assert* `50 < finger_length_middle < 120` di `extract_geometry.py`.

### 9.2 Sprint 2 — Eksperimen representasi & pipeline (2–3 minggu)

- [ ] **Per-frame vs session-level training** (9.2.1): regenerasi dataset dengan layout `frame_*/`, jalankan training ulang, bandingkan Rank-1. Ini eksperimen utama untuk menjawab pertanyaan "apakah kita perlu registrasi sama sekali?"
- [ ] **Swap ke `normalized_geometry.json`** (9.1.3): ablation normalisasi geometri.
- [ ] **Drop `inter_finger_gaps` sebagai ablation** (9.1.2): cek apakah fitur ini *noise* atau sinyal nyata.
- [ ] **Fix *knuckle detection*** (9.2.3) dengan integrasi MediaPipe Hand sebagai fallback. Regenerasi `process_all_scans.py --force` pada 5 sesi FAIL.

### 9.3 Sprint 3 — Perluasan dataset (3–4 minggu, bisa paralel dengan Sprint 1–2)

- [ ] **Tambah minimum 14 subjek** → total ≥ 20 subjek, ≥ 10 sesi per subjek.
- [ ] **Pose guide di iOS app** (9.1.1): siluet overlay + validasi `depthMean` + gating `Ready to scan`.
- [ ] **Re-scan subjek FAIL berulang** (alji, gede) setelah fix *knuckle detection*.

### 9.4 Sprint 4 — Eksperimen lanjutan & dokumentasi (2 minggu, setelah Sprint 1–3)

- [ ] **Pre-training di ModelNet40** (9.3.2) lalu fine-tuning di telapak tangan.
- [ ] **Cosine annealing LR** (9.3.2).
- [ ] **Cross-session verification** (EER, FAR/FRR) + open-set identification.
- [ ] **t-SNE visualization** untuk melihat *embedding space*.
- [ ] **Write-up Bab 4 (Hasil) & Bab 5 (Kesimpulan) tesis.**

### 9.5 Kriteria "siap sidang"

| Metrik | Target minimum |
|---|---|
| Jumlah subjek | ≥ 20 |
| Rank-1 LOSO *mean* | ≥ 90 % |
| Ablation 4 model (M1–M4) dengan bukti GAM membantu | Terbukti via plot CMC |
| Variansi antar-fold | *std* < 5 % |
| Dokumentasi lengkap semua komponen | Tersedia |

---

## 10. Lampiran — Referensi File Penting

| Berkas | Isi |
|---|---|
| `README.md` | Ikhtisar proyek & alur data |
| `3DRegistration/README.md` | Dokumentasi pipeline registrasi |
| `3DRegistration/RESULT_FILES.md` | Spesifikasi 5 file output per sesi |
| `3DRegistration/result/qc_summary.txt` | Laporan QC dataset |
| `3DRegistration/result/dataset_manifest.json` | Manifest PASS/WARN/FAIL tiap sesi |
| `3DCNN/runs/geoatt_m4/training_curves.png` | Kurva training |
| `3DCNN/runs/geoatt_m4/normalizer.json` | Mean/std 33 fitur (training split) |
| `3DCNN/eval_results/*.png` | Output evaluasi (CMC, confusion, similarity) |
| `TrueDepthScan/STREAMTRUEDEPTH.md` | Dokumentasi sumber streaming TrueDepth |
| `TrueDepthScan/.trae/documents/` | Catatan iterasi refactor iOS |

---

## 11. Status Lampiran Gambar

| # | Item | Status | Path yang direferensikan |
|---|---|:---:|---|
| 1 | iOS app — live scan (JET colormap, Ready to scan) | ✅ diterima | `docs/images/ios_scan_ready.png` |
| 2 | iOS app — history list | ✅ diterima | `docs/images/ios_scan_history.png` |
| 3 | Training curves (loss + Rank-1) | ✅ diterima | `docs/images/training_curves.png` |
| 4 | Output identifikasi Rank-1 per sesi | ✅ diterima (teks) | — (dimasukkan sebagai tabel & kode) |
| 5 | CMC curve (evaluasi utama) | ✅ diterima | `docs/images/cmc_curve.png` |
| 6 | Confusion matrix | ✅ diterima | `docs/images/confusion_matrix.png` |
| 7 | Similarity score per orang | ✅ diterima | `docs/images/similarity_per_person.png` |
| 8 | Ablation summary (teks) | ✅ diterima | — |
| 9 | CMC ablation | ✅ diterima | `docs/images/cmc_ablation.png` |
| 10 | Visualisasi `output.ply` success example (rahmat/202501) | ✅ diterima | `docs/images/output_ply_rahmat_202501.png` |
| 11 | Visualisasi `output.ply` PASS subjek sama (alji/091549) | ✅ diterima | `docs/images/output_ply_alji_091549.png` |
| 12 | Visualisasi `output.ply` FAIL subjek sama (alji/091603) | ✅ diterima | `docs/images/output_ply_alji_091603.png` |
| 12 | ROC / EER / AUC | ❌ tidak dipakai | Notebook evaluasi saat ini hanya Rank-k CMC |
| 13 | t-SNE embedding | ⏳ opsional | — |
| 14 | Visualisasi PLY `fadhil/095031` (bug PCA flip) | ⏳ opsional | Akan memperkuat Bab 7.3.2 |
| 15 | Visualisasi PLY `gede/101327` (pose tertutup) | ⏳ opsional | Akan memperkuat Bab 7.3.3 |
| 16 | Visualisasi PLY `fadhil/094713` (misclassification fadhil→alji) | ⏳ opsional | Akan memperkuat Bab 6.2 |
| 17 | Visualisasi PLY `taofik/092743` (misclassification taofik→rahmat) | ⏳ opsional | Akan memperkuat Bab 6.2 |

### 11.1 Langkah Menyimpan Screenshot ke Folder

Agar gambar ter-render saat laporan dibuka (di VS Code, GitHub, atau viewer MD lain), simpan ketujuh screenshot ke folder `docs/images/` dengan nama di kolom "Path" di atas. Contoh:

```
/Users/rahmatzulfikri/Projects/Thesis/docs/images/
├── ios_scan_ready.png
├── ios_scan_history.png
├── training_curves.png
├── cmc_curve.png
├── confusion_matrix.png
├── similarity_per_person.png
├── cmc_ablation.png
├── output_ply_rahmat_202501.png         ← contoh sukses (Bab 4.6)
├── output_ply_alji_091549.png           ← PASS subjek sama (Bab 7.3.1)
└── output_ply_alji_091603.png           ← FAIL subjek sama (Bab 7.3.1)
```

---

*Laporan ini akan diperbarui setelah M1/M2/M3 selesai dilatih dan angka EER/AUC dihasilkan.*
