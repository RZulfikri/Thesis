# Identifikasi Telapak Tangan 3D Berbasis iPhone TrueDepth Camera

**Rahmat Zulfikri — Tesis S2, Magister Teknik Elektro, Universitas Gadjah Mada**

Sistem identifikasi biometrik telapak tangan menggunakan kamera TrueDepth iPhone sebagai sensor kedalaman, dengan model **GeoAtt-PointNet++** untuk mengenali identitas seseorang dari geometri 3D telapak tangan.

> ⚠️ **Dataset tidak disertakan dalam repository ini.** Raw data tersimpan terpisah di `Raw Depth Data/` (file ZIP hasil scan iPhone). Lihat [Alur Data](#alur-data-lengkap) di bawah untuk instruksi pemrosesan.

---

## Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TrueDepthScan  (iOS App)                                                │
│  Rekam depth frames → export .bin + calibration.json                    │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │ depth*.bin + calibration.json
                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3DRegistration  (Python)                                                │
│  1. ICP multi-frame → registered point cloud (.ply)                     │
│  2. Ekstraksi fitur → geometry.json (33 nilai biometrik)                │
│  3. Preprocessing → cnn_input.npy (point cloud siap training)           │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │ cnn_input.npy + geometry.json
                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3DCNN  (Python / Google Colab)                                         │
│  GeoAtt-PointNet++ Siamese Network                                      │
│  Training → 128-dim identity embedding                                  │
│  Identifikasi → nearest neighbor di enrollment DB                       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Struktur Repository

```
Thesis/
├── TrueDepthScan/          ← iOS app untuk scan telapak tangan
├── 3DRegistration/         ← Pipeline registrasi ICP + ekstraksi fitur
├── 3DCNN/                  ← Model GeoAtt-PointNet++ + training + evaluasi
├── Raw Depth Data/         ← Raw dataset (ZIP, tidak di-track Git)
├── presentations/          ← File presentasi progress thesis
├── docs/                   ← Laporan, plan, literatur, project logs
│   ├── reports/
│   ├── plans/
│   ├── literature/
│   └── project_logs/
├── AGENTS.md               ← Global agent swarm contract
└── README.md               ← File ini
```

---

## Komponen

### `TrueDepthScan/` — iOS Scanning App

Aplikasi iPhone untuk merekam scan 3D telapak tangan menggunakan kamera TrueDepth (Face ID).

- Merekam 10–15 frame depth Float32 (640×480) dengan AVFoundation
- ROI detection menggunakan Vision framework untuk membatasi area telapak
- Export ke format siap pakai: `depth*.bin`, `calibration.json`, `metadata.json`
- Field `handedness` (right/left) disimpan di metadata untuk akurasi deteksi jari

**Stack:** Swift, SwiftUI, AVFoundation, Vision, Accelerate

### `3DRegistration/` — Processing Pipeline

Pipeline Python untuk mengubah raw depth frames menjadi representasi 3D siap training.

**Input:** Folder hasil export iOS (`[label]_YYYYMMDD_HHMMSS/`) yang berisi:
- `depth00.bin ... depthNN.bin` — raw depth frames
- `calibration.json` — intrinsik kamera
- `metadata.json` — label, handedness, timestamp

**Output per sesi** (`result/[label]/[timestamp]/frame_XX/`):

| File | Keterangan |
|---|---|
| `output.ply` | Single-frame point cloud ~15k–20k titik (xyz + normals, setelah DBSCAN isolation) — representasi R1 |
| `geometry.json` | Fitur geometri biometrik + `is_valid`/`warnings` (QC point-cloud) |
| `cnn_input.npy` | Full cloud PCA-aligned + unit sphere `(N, 6)` — **input utama 3DCNN (R2)** |
| `cnn_input_fps.npy` | FPS 8192 titik `(8192, 6)` — representasi R3 untuk ablation v7.2.0 |

**Fitur geometri (33 nilai):** panjang jari (5), rasio jari (5), lebar & tinggi telapak (2), celah antar jari (4), lebar jari (5), rasio lebar/panjang jari (5), kelengkungan telapak (1).

**Cara pakai:**
```bash
cd 3DRegistration
pip install -r requirements.txt

# 1. Tempatkan raw data hasil scan iPhone di folder dataset/
#    (misal: extract dari Raw Depth Data/)

# 2. Proses semua scan (ICP registrasi + ekstraksi fitur + preprocessing)
python process_all_scans.py --data_dir dataset

# Atau untuk frame-level layout (dipakai di 3DCNN saat ini):
python process_single_frames.py --data_dir dataset
```

### `3DCNN/` — Model & Training

Implementasi **GeoAtt-PointNet++** untuk identifikasi telapak tangan.

**Arsitektur:**
- **PointNet++ backbone** — 3 layer SetAbstraction (512→128→1 titik)
- **GeometryEncoder** — MLP 33→64 dim untuk fitur geometri
- **Geometric Attention Module (GAM)** — gabungkan geometry embedding dengan SA features
- **Siamese network** — shared encoder, cosine similarity, contrastive/arcface loss
- **Output** — 128-dim L2-normalized embedding per sesi

**Cara pakai:**
```bash
# Lokal
cd 3DCNN
python train.py --data_dir dataset --output_dir runs/exp1

# Google Colab → gunakan notebook collab/01_train_and_eval.ipynb
```

---

## Alur Data Lengkap

```
1. Raw Data (tidak di-repo)
   └─ Raw Depth Data/
        scans_15-05-26,_12.27.zip
        scans_15-05-26,_12.31.zip
   
   ↓ Extract ke 3DRegistration/dataset/

2. Registrasi & Ekstraksi Fitur
   └─ cd 3DRegistration
   └─ python process_single_frames.py --data_dir dataset
      → result/[label]/[timestamp]/frame_XX/
            output.ply
            geometry.json
            cnn_input.npy
            cnn_input_fps.npy
   
   ↓ Copy/link ke 3DCNN/dataset/

3. Training (Local / Colab)
   └─ cd 3DCNN
   └─ python train.py --data_dir dataset --output_dir runs/exp1
      → checkpoint best_rank1.pth
      → normalizer.json
   
   ↓ Evaluasi

4. Evaluasi
   └─ python evaluate.py --checkpoints runs/exp1/best_rank1.pth
      → Rank-1 Accuracy, CMC Curve, Confusion Matrix, ROC/DET
```

**Catatan tentang dataset:**
- **Raw data** (`.bin` hasil scan iPhone) tidak disertakan dalam Git — tersimpan di `Raw Depth Data/` sebagai ZIP.
- **Processed data** (`cnn_input.npy`, `geometry.json`) berada di `3DCNN/dataset/` dan di-track Git karena ukurannya manageable setelah diproses.
- Jika clone repo baru, pastikan raw data sudah di-extract ke `3DRegistration/dataset/` sebelum menjalankan pipeline.

---

## Evolusi Eksperimen (V1 → V4)

| Versi | Tanggal | Loss | Rank-1 (terbaik) | Catatan |
|---|---|---|---|---|
| **V1** | Apr 2026 | Contrastive | 89.5% (6 subjek) | Proof-of-concept |
| **V2** | Mei 2026 | Triplet | ~60% (11 subjek) | Bottleneck loss |
| **V3** | Mei 2026 | ArcFace | ~99.8% (no_geom) | Lompatan masif, GeoAtt terlihat merugikan |
| **V4** | Mei 2026 | ArcFace | TBD | Fair ablation — investigasi bias metodologi |

Laporan detail timeline tersedia di: `docs/reports/REPORT_THESIS_V1_V4/`

---

## Dependensi

### iOS (`TrueDepthScan`)
- iOS 15+, iPhone dengan Face ID (TrueDepth camera)
- Xcode 15+

### Python (`3DRegistration`)
```bash
pip install -r 3DRegistration/requirements.txt
# open3d, numpy, scipy, opencv-python
```

### Python (`3DCNN`)
```bash
pip install -r 3DCNN/requirements.txt
# torch, numpy, scikit-learn, matplotlib
```

---

## Dokumentasi Detail

| Dokumen | Lokasi |
|---|---|
| Arsitektur iOS & pipeline streaming | `TrueDepthScan/STREAMTRUEDEPTH.md` |
| Format file output registrasi | `3DRegistration/RESULT_FILES.md` |
| Pipeline registrasi & parameter | `3DRegistration/README.md` |
| Timeline progress V1→V4 + bukti grafis | `docs/reports/REPORT_THESIS_V1_V4/` |
| Monitoring checklist & plan | `docs/reports/` & `docs/plans/` |
| Project logs (swarm activity) | `docs/project_logs/SWARM_LOG.md` |

---

*Repository ini merupakan bagian dari thesis S2 Magister Teknik Elektro UGM.*
