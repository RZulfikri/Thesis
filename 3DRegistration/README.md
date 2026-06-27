# 3DRegistration — Palm Scan Processing Pipeline

Pipeline Python untuk memproses hasil scan telapak tangan dari iPhone TrueDepth
(via app **TrueDepthScan**) menjadi dataset siap training model GeoAtt-PointNet++.

---

## Gambaran Pipeline

```
iPhone TrueDepthScan app
        │  export: depth[NN].bin + calibration.json + metadata.json
        ▼
dataset/[label]_YYYYMMDD_HHMMSS/
        │
        ▼
process_single_frames.py   ← proses tiap frame secara independen
        │
        ├── output.ply          ← PLY single-frame (DBSCAN isolated)
        ├── geometry.json       ← 14 fitur biometrik (mm absolut)
        └── cnn_input.npy       ← (N, 6) float32, PCA-aligned + unit-sphere
        │
        ▼
result_frames/[label]/[timestamp]/frame_[NN]/
        │
        ▼
validate_dataset.py        ← QC: cek kelengkapan dan kualitas tiap frame
        │
        ▼
3DCNN/dataset/             ← copy ke sini untuk training
```

---

## Prasyarat

### 1. Buat virtual environment

```bash
cd /path/to/3DRegistration
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Catatan macOS (Apple Silicon):** torch dan open3d mungkin perlu waktu lebih lama saat install pertama kali.

---

## Struktur Folder

```
3DRegistration/
├── dataset/                        ← input: hasil export dari TrueDepthScan app
│   ├── alji_20260505_210153/       ← nama: [label]_YYYYMMDD_HHMMSS
│   │   ├── calibration.json        ← parameter kamera (fx, fy, cx, cy)
│   │   ├── metadata.json           ← label, handedness, frameCount
│   │   ├── depth00.bin             ← frame depth float32
│   │   ├── depth01.bin
│   │   └── ...
│   └── ...
│
├── result_frames/                  ← output: hasil processing
│   └── alji/
│       └── 20260505_210153/
│           ├── frame_00/
│           │   ├── output.ply
│           │   ├── geometry.json
│           │   └── cnn_input.npy
│           └── frame_01/ ...
│
├── process_single_frames.py        ← script utama
├── extract_geometry.py             ← ekstraksi 14 fitur biometrik
├── validate_dataset.py             ← QC dataset
├── dataset.py                      ← build dataset untuk training (opsional)
└── requirements.txt
```

---

## Cara Menjalankan

### Aktifkan venv dulu (setiap kali buka terminal baru)

```bash
source venv/bin/activate
```

---

### Step 1 — Proses semua scan

```bash
python process_single_frames.py
```

Akan memproses semua sesi di `dataset/` dan menyimpan output ke `result_frames/`.

**Opsi berguna:**

```bash
# Proses satu sesi saja
python process_single_frames.py --data_dir dataset/alji_20260505_210153

# Paksa proses ulang (timpa output yang sudah ada)
python process_single_frames.py --force

# Proses ulang hanya satu sesi
python process_single_frames.py --data_dir dataset/alji_20260505_210153 --force

# Lewati pembuatan cnn_input.npy (lebih cepat jika hanya butuh geometry)
python process_single_frames.py --skip_cnn

# Lewati ekstraksi geometry.json
python process_single_frames.py --skip_geometry

# Filter frame yang terlalu sparse (default: min 1000 titik)
python process_single_frames.py --min_points 2000
```

---

### Step 2 — Validasi dataset

```bash
python validate_dataset.py --result_dir result_frames
```

Output:
- `result_frames/dataset_manifest.json` — status PASS/WARN/FAIL per frame
- `result_frames/qc_summary.txt` — ringkasan teks

**Status QC:**

| Status | Arti |
|--------|------|
| `PASS` | Frame valid, siap digunakan untuk training |
| `WARN` | Frame valid tapi ada catatan (misal: scan sparse) |
| `FAIL` | Frame tidak valid — tidak akan digunakan training |

---

### Step 3 — Salin ke 3DCNN

Setelah semua frame valid, salin ke folder dataset 3DCNN:

```bash
cp -r result_frames/. ../3DCNN/dataset/
```

Untuk Google Colab: upload folder `result_frames/` ke Google Drive sebagai
`MyDrive/3DCNN/dataset/`.

---

## Format Output geometry.json

Setiap frame menghasilkan **14 fitur biometrik** (nilai absolut dalam mm):

| Field | Dim | Keterangan |
|-------|-----|------------|
| `finger_lengths_mm` | 5 | Panjang tiap jari [ibu jari → kelingking] |
| `palm_width_mm` | 1 | Lebar telapak di baris buku jari |
| `palm_height_mm` | 1 | Tinggi telapak (wrist → buku jari) |
| `palm_depth_std_mm` | 1 | Kelengkungan permukaan telapak (std Z) |
| `finger_widths_mm` | 5 | Lebar tiap jari |
| `mean_palm_curvature` | 1 | Kelengkungan rata-rata telapak |

Field tambahan (metadata, tidak masuk training):
- `scan_distance_mm` — jarak kamera ke telapak saat scan
- `handedness` — `"right"` atau `"left"`
- `point_count` — jumlah titik di point cloud
- `quality_issues` — daftar masalah QC (kosong = bersih)
- `is_valid` — `true` jika frame lolos semua QC gate

---

## Parameter QC

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `SCAN_DIST_MIN_MM` | 180 mm | Jarak minimum kamera → telapak |
| `SCAN_DIST_MAX_MM` | 450 mm | Jarak maksimum TrueDepth |

> **Jarak ideal saat scan: 200–350 mm** (20–35 cm dari kamera).

---

## Troubleshooting

**`Tidak ada folder sesi di 'dataset'`**
→ Pastikan folder `dataset/` berisi subfolder dengan format `[label]_YYYYMMDD_HHMMSS`.

**Frame banyak yang `is_valid: false`**
→ Jalankan `validate_dataset.py` untuk detail. Penyebab umum:
- Tangan terlalu dekat (< 180 mm) atau terlalu jauh (> 450 mm)
- Point cloud terlalu sparse — jari tidak terdeteksi

**`Warning: C++ pose graph module not found`**
→ Normal — pipeline menggunakan Python fallback, tidak mempengaruhi hasil.

**`ModuleNotFoundError: No module named 'open3d'`**
→ Venv belum diaktifkan atau belum install:
```bash
source venv/bin/activate
pip install -r requirements.txt
```
