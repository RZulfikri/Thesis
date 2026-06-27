# Penjelasan File Output Hasil Pemrosesan Scan

Setiap sesi scan menghasilkan lima file yang disimpan di:
```
result/[label]/[timestamp]/
├── output.ply           ← registered point cloud (file sumber)
├── geometry.json        ← 33 fitur geometri biometrik
├── texture.npy          ← tekstur kanonis 2D (cadangan)
├── cnn_input.npy        ← full cloud PCA-aligned + unit sphere (input utama)
└── cnn_input_fps.npy    ← FPS 1024 titik (backup novelty, ablation study)
```

---

## 1. `output.ply` — Registered Point Cloud

### Apa itu?
File point cloud 3D dalam format PLY (Polygon File Format). Berisi kumpulan titik-titik 3D yang merepresentasikan permukaan telapak tangan setelah proses registrasi ICP (Iterative Closest Point).

### Isi
- Setiap titik memiliki koordinat XYZ (dalam meter) dan normal vektor (nx, ny, nz)
- Jumlah titik: ~50.000–150.000 titik tergantung kualitas scan
- Koordinat dalam sistem kamera TrueDepth (meter), belum di-align secara kanonik

### Bagaimana dibuat?
Dihasilkan oleh `run.py` melalui pipeline berikut:
1. Baca frame-frame depth dari `depth00.bin ... depth10.bin` (hasil export iOS)
2. Konversi setiap frame depth → point cloud menggunakan parameter kalibrasi kamera (`calibration.json`)
3. Registrasi semua frame menjadi satu cloud menggunakan ICP sekuensial
4. Filter noise: statistical outlier removal, cluster connectivity (DBSCAN), depth range filter
5. Simpan sebagai PLY dengan normals

### Parameter registrasi yang digunakan
| Parameter | Nilai | Keterangan |
|---|---|---|
| `min_depth` | 0.10 m | Jarak minimum dari kamera |
| `max_depth` | 0.50 m | Jarak maksimum (50 cm) |
| `voxel_size` | 0.001 m | Resolusi 1 mm |
| `normal_radius` | 0.008 m | Radius estimasi normal |
| `outlier_std_ratio` | 1.5 | Agresivitas hapus outlier |

### Kegunaannya
File sumber utama. Semua file output lainnya diturunkan dari file ini.

### Cara inspect
```bash
python3 view_results.py result/rahmat/20260401_200613/output.ply
```

---

## 2. `geometry.json` — Fitur Geometri Biometrik

### Apa itu?
File JSON berisi **33 nilai numerik** yang merepresentasikan ciri-ciri biometrik geometris telapak tangan.

### Isi lengkap (contoh nyata dari scan)
```json
{
  "scan_id":                     "rahmat_20260401_200613",
  "point_count":                 77761,
  "finger_lengths_mm":           [190.25, 205.51, 205.80, 195.98, 133.69],
  "finger_ratios":               [0.9244, 0.9986, 1.0000, 0.9523, 0.6496],
  "palm_width_mm":               110.37,
  "palm_height_mm":              60.01,
  "palm_aspect_ratio":           1.8393,
  "finger_to_palm_ratios":       [3.1704, 3.4247, 3.4296, 3.2659, 2.2279],
  "inter_finger_gaps_mm":        [14.20, 8.55, 7.30, 9.80],
  "finger_widths_mm":            [22.1, 17.8, 18.5, 17.0, 13.4],
  "finger_width_to_length_ratios":[0.116, 0.087, 0.090, 0.087, 0.100],
  "mean_palm_curvature":         0.3018
}
```

### Penjelasan setiap fitur

| Field | Dimensi | Satuan | Keterangan |
|---|---|---|---|
| `finger_lengths_mm` | 5 | mm | Panjang tiap jari: [ibu, telunjuk, tengah, manis, kelingking] |
| `finger_ratios` | 5 | — | Panjang tiap jari dibagi panjang jari tengah (scale-invariant) |
| `palm_width_mm` | 1 | mm | Lebar telapak di baris knuckle (MCP) |
| `palm_height_mm` | 1 | mm | Tinggi telapak dari wrist top ke baris knuckle |
| `palm_aspect_ratio` | 1 | — | palm_width / palm_height |
| `finger_to_palm_ratios` | 5 | — | Panjang tiap jari dibagi palm_height |
| `inter_finger_gaps_mm` | 4 | mm | Jarak kosong antar jari (celah horizontal): [ibu-telunjuk, telunjuk-tengah, tengah-manis, manis-kelingking] |
| `finger_widths_mm` | 5 | mm | Lebar tiap jari diukur dari p5–p95 rentang X di zona jari |
| `finger_width_to_length_ratios` | 5 | — | Lebar jari / panjang jari per jari (rasio bentuk) |
| `mean_palm_curvature` | 1 | — | Rata-rata kelengkungan permukaan telapak (0=rata, 1=sangat melengkung) |

**Total: 5 + 5 + 1 + 1 + 1 + 5 + 4 + 5 + 5 + 1 = 33 nilai**

### Bagaimana dibuat?
Dihasilkan oleh `extract_geometry.py` dari `output.ply`:
1. PCA alignment — jari diarahkan ke sumbu +Y
2. Deteksi Wrist ROI (18% bawah Y range) sebagai titik acuan
3. Deteksi baris knuckle (MCP) — Y dengan lebar X maksimum di atas wrist
4. Deteksi 5 ujung jari; jika `handedness` tersedia, gunakan sinyal ganda (tinggi jari + celah ibu-telunjuk vs manis-kelingking)
5. Hitung panjang, lebar, celah jari, dan rasio dalam satuan mm

### Mengapa 33 fitur?
Dibandingkan versi 23 fitur sebelumnya, tiga fitur baru ditambahkan:
- `inter_finger_gaps_mm` — menggantikan `inter_finger_depths_mm` (kini mengukur celah kosong horizontal, bukan kedalaman lembah)
- `finger_widths_mm` — lebar anatomis tiap jari
- `finger_width_to_length_ratios` — rasio bentuk tiap jari

### Kegunaannya
Dimasukkan ke **GeometryEncoder** dalam model GeoAtt-PointNet++ → 64-dim embedding → mengarahkan Geometric Attention Module (GAM).

---

## 3. `texture.npy` — Tekstur Kanonis 2D

### Apa itu?
Representasi 2D telapak tangan hasil proyeksi top-down dari point cloud yang sudah di-align.

### Isi
```
shape : (256, 256, 5)  float32
```

| Channel | Nama | Range | Keterangan |
|---|---|---|---|
| ch0 | `depth` | [0, 1] | Koordinat Z ternormalisasi |
| ch1 | `normal_x` | [0, 1] | Komponen X normal vektor, dipetakan dari [-1,1] ke [0,1] |
| ch2 | `normal_y` | [0, 1] | Komponen Y normal vektor |
| ch3 | `normal_z` | [0, 1] | Komponen Z normal vektor |
| ch4 | `curvature` | [0, 1] | `\|1 - \|nz\|\|` — 0=rata, tinggi=tepi/lekukan |

### Kegunaannya
**Belum digunakan** dalam pipeline GeoAtt-PointNet++ saat ini. Dipersiapkan sebagai cadangan untuk eksperimen 2D CNN atau hybrid 2D+3D di masa depan.

---

## 4. `cnn_input.npy` — Full Point Cloud Kanonis (Input Utama)

### Apa itu?
Point cloud **full resolution** yang sudah di-PCA-align dan dinormalisasi ke unit sphere. Ini adalah **input utama** model GeoAtt-PointNet++ saat training dan inference.

### Isi
```
shape : (N, 6)  float32    ← N variatif, ~50k–150k titik
kolom : [x, y, z, nx, ny, nz]
```

| Kolom | Keterangan |
|---|---|
| x, y, z | Koordinat 3D dalam unit sphere (range ≈ [-1, 1]) |
| nx, ny, nz | Normal vektor di setiap titik (range ≈ [-1, 1]) |

> **Catatan:** N bervariasi per scan (tidak tetap). Sampling ke jumlah titik tetap (default 4096) dilakukan **on-the-fly** di `PalmPairDataset.__getitem__()` saat training.

### Bagaimana dibuat?
Dihasilkan oleh `preprocess_for_cnn.preprocess_full()` dari `output.ply`:

**Tahap 1 — PCA Canonical Alignment**
- Point cloud dirotasi sehingga jari-jari selalu mengarah ke sumbu +Y
- Arah kedalaman kamera → sumbu +Z
- Memastikan orientasi konsisten antar scan berbeda

**Tahap 2 — Normalisasi ke Unit Sphere**
- Seluruh point cloud discale agar masuk dalam bola dengan radius 1
- `pts = pts / max(||pts||)`
- Membuat model scale-invariant

**Simpan sebagai float32 (tanpa FPS)**
- Semua titik dipertahankan — tidak ada downsampling

### Mengapa tidak di-FPS dulu?
Menyimpan full cloud memberi fleksibilitas:
- Sampling random setiap epoch = augmentasi implisit (berbeda subset titik tiap kali)
- Jumlah titik bisa disesuaikan (`N_POINTS` di konfigurasi) tanpa regenerasi file
- FPS tersedia sebagai mode terpisah (`cnn_input_fps.npy`) untuk ablation

### Cara pakai di 3DCNN
```python
cloud = np.load(session_dir / 'cnn_input.npy')  # (N, 6) — full resolution
pts   = cloud[np.random.choice(len(cloud), 4096, replace=False)]  # sample 4096
```

---

## 5. `cnn_input_fps.npy` — FPS Point Cloud (Backup Novelty)

### Apa itu?
Point cloud yang sama (`cnn_input.npy`) tetapi sudah di-downsample ke **jumlah titik tetap** menggunakan Farthest Point Sampling. Digunakan sebagai **backup novelty** dalam ablation study.

### Isi
```
shape : (n_points, 6)  float32    ← n_points tetap (default: 1024)
kolom : [x, y, z, nx, ny, nz]
```

Identik dengan format lama `cnn_input.npy` (sebelum pipeline diperbarui).

### Bagaimana dibuat?
Dihasilkan oleh `preprocess_for_cnn.preprocess_fps()`:
1. PCA alignment + unit sphere (sama seperti full cloud)
2. **Farthest Point Sampling** → pilih n_points titik paling tersebar merata
3. Simpan sebagai `(n_points, 6)` float32

### Kegunaannya
Ablation study: bandingkan M4 dengan full-cloud + random sampling vs M4 dengan FPS 1024 untuk menunjukkan keunggulan menggunakan full cloud.

---

## Hubungan Antar File

```
depth00.bin ... depthNN.bin   ← raw frames dari iPhone TrueDepth
calibration.json              ← parameter kamera (fx, fy, cx, cy, distortion)
        │
        ▼ run.py (registrasi ICP)
output.ply                    ← point cloud gabungan (~50k–150k titik, raw koordinat)
        │
        ├──▶ extract_geometry.py    ──▶  geometry.json        (33 fitur biometrik)
        ├──▶ extract_texture.py     ──▶  texture.npy          (256×256×5 tekstur 2D)
        ├──▶ preprocess_full()      ──▶  cnn_input.npy        (N×6 full cloud, input utama)
        └──▶ preprocess_fps()       ──▶  cnn_input_fps.npy   (1024×6 FPS, backup novelty)
```

## Mana yang Diperlukan untuk Training?

| File | Digunakan di 3DCNN? | Keterangan |
|---|---|---|
| `output.ply` | Tidak langsung | Sumber untuk regenerasi; diperlukan jika perlu regenerasi file lain |
| `geometry.json` | **Ya** — GeometryEncoder | Input 33-dim fitur geometri |
| `texture.npy` | Belum | Cadangan untuk eksperimen 2D masa depan |
| `cnn_input.npy` | **Ya** — PointNet++ encoder | Input point cloud utama (full resolution, on-the-fly sampling) |
| `cnn_input_fps.npy` | Opsional — ablation | Backup novelty; gunakan dengan `--sampling fps` di train.py |

## Cara Regenerasi

Jika format berubah (misal geometry.json dari 23 ke 33 fitur), regenerasi dengan:
```bash
# Regenerasi semua output (paksa)
python process_all_scans.py --skip_registration --force

# Regenerasi hanya geometry
python extract_geometry.py result/rahmat/20260401_200613/output.ply \
    result/rahmat/20260401_200613/geometry.json

# Regenerasi hanya cnn_input (full cloud)
python preprocess_for_cnn.py result/rahmat/20260401_200613/output.ply --no_fps

# Regenerasi hanya cnn_input_fps
python preprocess_for_cnn.py result/rahmat/20260401_200613/output.ply --no_full
```
