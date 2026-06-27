# StreamTrueDepth — Dokumentasi Aplikasi iOS

Aplikasi iOS untuk mengambil scan telapak tangan 3D menggunakan kamera TrueDepth (kamera depan Face ID) pada iPhone.
Merupakan tahap pertama dalam pipeline identifikasi telapak tangan sebelum data dikirim ke Open3DRegistration untuk diproses.

---

## Gambaran Umum

```
Pengguna mengarahkan telapak tangan ke kamera
        ↓
Aplikasi mendeteksi kedalaman secara real-time (TrueDepth sensor)
        ↓
Scan otomatis selama ~2 detik (countdown 3 detik sebelum mulai)
        ↓
Export ke folder [label]_YYYYMMDD_HHMMSS/ di Documents
        ↓
Kirim ke Mac via AirDrop / Files app
        ↓
Proses lanjut di Open3DRegistration
```

---

## Komponen Utama

### `CameraManager.swift`
Otak dari aplikasi. Mengatur seluruh siklus hidup kamera, proses scan, dan export.

**Tanggung jawab:**
- Menginisialisasi sesi `AVCaptureSession` dengan TrueDepth camera
- Sinkronisasi frame depth + video secara bersamaan via `AVCaptureDataOutputSynchronizer`
- Deteksi kesiapan scan berdasarkan data kedalaman real-time
- Akumulasi titik 3D ke voxel grid selama scanning
- Memicu export setelah scan selesai

**State utama yang dipantau UI:**
| State | Keterangan |
|---|---|
| `depthState` | `.noObject` / `.tooClose` / `.tooFar` / `.inRange` |
| `isCountingDown` | Countdown 3 detik sebelum scan dimulai |
| `isScanning` | Scan sedang berlangsung (durasi ~2 detik) |
| `isProcessing` | Export sedang diproses setelah scan selesai |

---

### `ContentView.swift`
Tampilan utama aplikasi — kamera live + kontrol scan.

**Elemen UI:**
- **Preview kamera** — live feed TrueDepth dengan overlay kedalaman warna JET (colormap biru→merah)
- **Label di tengah atas** — nama subjek (contoh: `rahmat`). Tap untuk edit
- **Tombol history** (kiri atas) — buka daftar scan tersimpan
- **Toggle cloud/local** (kanan atas) — saat ini hanya mode local yang didukung
- **Badge status** — menampilkan instruksi posisi: *"Too close"*, *"Too far"*, *"Ready to scan"*
- **Tombol scan** (lingkaran merah) — aktif hanya ketika `depthState == .inRange`
- **Countdown overlay** — angka besar 3→2→1 ditampilkan saat countdown berlangsung

---

### `Open3DExporter.swift`
Mengekspor raw depth frames dari kamera ke format yang bisa dibaca Open3DRegistration.

**Yang dilakukan:**
1. Menerima array `CVPixelBuffer` (depth frames) + data kalibrasi kamera
2. Untuk setiap frame:
   - Konversi depth buffer ke array Float32 (mendukung format Float16 dan Float32 via Accelerate)
   - Terapkan filter secara berurutan: (1) depth range [0.10–0.50 m] + ROI masking, (2) neighborhood density filter
   - Simpan sebagai file biner `depth00.bin`, `depth01.bin`, dst.
3. Export `calibration.json` (parameter kamera: fx, fy, cx, cy + lens distortion)
4. Export `metadata.json` (info jumlah frame, resolusi, label, handedness, timestamp)

**Urutan pipeline di dalam `exportDepthFrame`:**
```
Baca buffer Float32/Float16
    ↓
Filter (1): depth range [0.10–0.50 m] + ROI masking
    → pixel di luar range atau di luar ROI → set ke 0.0
    ↓
Filter (2): neighborhood density filter (8-connected, ≥3 tetangga valid, toleransi ±50 mm)
    → pixel terisolasi → set ke 0.0
    ↓
Tulis ke .bin sebagai array Float32 linier
```

**Neighborhood density filter** — pixel depth dibuang jika kurang dari 3 tetangga valid dalam radius 8-connected dengan toleransi ±50 mm. Ini menghilangkan titik-titik terisolasi di latar belakang yang lolos dari filter ROI.

*Contoh:* pixel di tepi dengan depth=0.30m tapi semua tetangganya bernilai 0 (terlalu jauh/dekat) akan dibuang. Pixel depth=0.30m yang memiliki ≥3 tetangga dengan depth dalam rentang 0.25–0.35m akan dipertahankan.

---

### `3DProcessingService.swift`
Layer perantara antara `CameraManager` dan `Open3DExporter`.

**Fungsi utama:**
- `exportForOpen3D(...)` — meneruskan panggilan ke `Open3DExporter.exportFrames()`
- `copyPixelBuffer(...)` — membuat salinan aman dari `CVPixelBuffer` untuk disimpan ke buffer

---

### `ScanHistoryManager.swift`
Mengelola daftar scan tersimpan di Documents directory.

- Memindai semua folder dengan format `[label]_YYYYMMDD_HHMMSS` di Documents (dideteksi dengan regex)
- Membaca `metadata.json` dari setiap folder untuk mendapatkan label dan tanggal
- Menyediakan fungsi hapus (satu atau batch)

---

### `ScanHistoryView.swift`
Tampilan riwayat scan — sheet yang muncul saat tombol history ditekan.

- Daftar semua scan dengan label dan tanggal
- Setiap baris: tombol **share** (kirim sebagai .zip) dan **delete**
- Mode **Select** untuk operasi batch (share/delete banyak scan sekaligus)

---

## Alur Kerja Detail

### 1. Persiapan Scan
```
User ketuk label di tengah atas
  → ketik nama (contoh: "rahmat")
  → label disimpan di @AppStorage (persisten antar sesi)

User arahkan telapak tangan 10–50 cm dari kamera
  → CameraManager samples 7×7 grid di area tengah depth buffer
  → Jika ≥5 sample valid di range → depthState = .inRange
  → Badge berubah hijau: "Ready to scan"
```

**Catatan:** trigger scan (`depthState`) ditentukan **hanya** dari data depth — threshold depth yang diukur langsung dari sensor. Vision (`VNDetectHumanHandPoseRequest`) dijalankan terpisah pada buffer video RGB dan hanya menghasilkan ROI masking, bukan trigger.

### 2. Proses Scan — Dua Alur Paralel

Saat scan berlangsung, dua proses berjalan secara paralel di setiap frame:

```
Setiap frame dari AVCaptureDataOutputSynchronizer:
  ├── [Alur A] Frame Capture (untuk export .bin)
  │       → Setiap frame ke-5 (frameDecimation=5):
  │           - Salin depth buffer → capturedDepthFrames[]
  │           - Salin ROI telapak → capturedDepthROIs[]
  │           - Catat vote handedness → capturedHandednessVotes[]
  │
  └── [Alur B] Voxel Accumulation (real-time preview, dibuang setelah scan)
          → Setiap 0.1 detik:
              - Proyeksikan depth pixels → XYZ menggunakan intrinsics kamera
              - Akumulasi ke voxel grid (resolusi 2mm) untuk deduplikasi temporal
```

**Mengapa frame ke-5?** (bukan setiap frame)
1. **Efisiensi memori** — satu frame 640×480×4 byte = 1.2 MB. Scan 2 detik @ 30fps = 60 frame = 72 MB. Dengan decimation 5, hanya ~12 frame = ~15 MB
2. **Kualitas ICP** — ICP membutuhkan frame yang *berbeda* secara geometris. Frame berurutan hampir identik. Frame ke-5 (~6fps) sudah cukup bervariasi untuk ICP yang bermakna

**Mengapa dua alur?** Alur A menghasilkan output yang dikirim ke Mac (Open3DRegistration). Alur B hanya untuk visualisasi point cloud real-time di aplikasi (tidak disimpan).

### 3. Vision dan Deteksi Tangan

```
Setiap frame video RGB masuk ke detectHandROI():
  → VNDetectHumanHandPoseRequest dijalankan
  → Jika ada tangan terdeteksi:
      - Ambil bounding box dari 21 keypoint (wrist + jari-jari)
      - Expand ROI 25% ke setiap sisi sebagai safety margin
      - (iOS 15+) Baca chirality → "right" atau "left"
      - Return HandDetectionResult(roi: CGRect, handedness: String)
```

**Orientasi Vision:** Vision menggunakan sistem koordinat portrait (y-up, origin kiri bawah). Depth buffer menggunakan sistem koordinat landscape (kamera fisik). Fungsi `visionOrientation(for:)` mengonversi `UIInterfaceOrientation` ke `CGImagePropertyOrientation` yang diberikan ke Vision agar bounding box yang dikembalikan sudah dalam koordinat yang konsisten.

**Mapping koordinat Vision → depth buffer:**
```
bufferX (kolom) = (1 - vision_y) * bufferWidth
bufferY (baris)  = (1 - vision_x) * bufferHeight
```
(berlaku untuk orientasi portrait dengan kamera depan yang di-mirror)

### 4. Deteksi Handedness (Chirality)

```
Setiap frame yang di-capture (Alur A) → catat vote:
  - "right" jika VNHumanHandPoseObservation.chirality == .right (iOS 15+)
  - "left"  jika .left
  - "unknown" jika iOS < 15 atau tidak terdeteksi

Saat stopScanning():
  → Hitung vote: berapa "right" vs "left"
  → Majority wins → simpan ke metadata.json sebagai handedness
  → Jika tidak ada vote valid → "unknown"
```

Majority vote digunakan karena deteksi per-frame bisa fluktuatif. Hasil akhir adalah handedness yang paling sering muncul selama sesi scan.

### 5. Export
```
stopScanning() dipanggil
  → isProcessing = true (tampilkan loading overlay)
  → Hitung majority vote handedness dari capturedHandednessVotes[]
  → Buat nama folder: "[label]_YYYYMMDD_HHMMSS"
      contoh: "rahmat_20260401_200613"

D3ProcessingService.exportForOpen3D(..., handedness: handedness) dipanggil:
  → Open3DExporter memproses setiap depth frame
  → Tulis depth00.bin ... depthNN.bin
  → Tulis calibration.json
  → Tulis metadata.json (termasuk field handedness)

shouldShareExport = true
  → iOS Share Sheet muncul otomatis
  → Folder di-zip → user bisa AirDrop / simpan ke Files
```

**Kalibrasi kamera** (`calibration.json`) berasal dari `AVCameraCalibrationData` — data hardware yang disediakan langsung oleh driver Apple. Tidak ada proses kalibrasi manual; intrinsics (fx, fy, cx, cy) dan lens distortion lookup table sudah tersedia dari sistem operasi.

---

## Output yang Dihasilkan

Setelah scan, sebuah folder tersimpan di:
```
Documents/[label]_YYYYMMDD_HHMMSS/
```

### File-file di dalam folder:

#### `depth00.bin` … `depthNN.bin`
Raw depth frames dalam format biner Float32.

- **Format:** array linier Float32, panjang = `width × height` (biasanya 640×480 = 307.200 nilai)
- **Nilai:** kedalaman dalam meter. `0.0` = pixel tidak valid (terlalu dekat, terlalu jauh, di luar ROI, atau dibuang filter)
- **Urutan:** row-major, baris pertama = baris paling atas depth buffer (landscape)
- **Jumlah file:** bergantung pada `frameDecimation` (default 5) dan durasi scan (default 2 detik). Biasanya 10–12 frame
- **Dibaca oleh:** `run.py` di Open3DRegistration untuk rekonstruksi 3D via ICP

#### `calibration.json`
Parameter optik kamera TrueDepth saat scan dilakukan.

```json
{
  "width": 640,
  "height": 480,
  "fx": 585.3,
  "fy": 585.3,
  "cx": 320.1,
  "cy": 240.8,
  "lensDistortionLookup": "...(base64)...",
  "inverseLensDistortionLookup": "...(base64)...",
  "lensDistortionCenter": [320.1, 240.8],
  "pixelSize": 0.005
}
```

| Field | Keterangan |
|---|---|
| `fx`, `fy` | Focal length dalam pixel |
| `cx`, `cy` | Principal point (pusat optik) |
| `lensDistortionLookup` | Tabel koreksi distorsi lensa (Base64 encoded Float32 array) |
| `pixelSize` | Ukuran pixel fisik dalam mm |

Digunakan oleh `run.py` untuk mengonversi depth pixel → koordinat 3D XYZ:
```
X = (u - cx) * depth / fx
Y = (v - cy) * depth / fy
Z = depth
```

Sumber: `AVCameraCalibrationData` dari Apple — **tidak perlu kalibrasi manual**.

#### `metadata.json`
Informasi umum tentang sesi scan.

```json
{
  "frameCount": 11,
  "width": 640,
  "height": 480,
  "depthMinMeters": 0.1,
  "depthMaxMeters": 0.5,
  "frameDecimation": 5,
  "videoChannels": 0,
  "exportTimestamp": "2026-04-01T20:06:13+0700",
  "purpose": "3d-cnn-palm-recognition",
  "label": "rahmat",
  "handedness": "right"
}
```

| Field | Keterangan |
|---|---|
| `frameCount` | Jumlah depth frame yang tersimpan |
| `label` | Nama subjek — digunakan untuk nama folder di Open3DRegistration |
| `purpose` | Identifier tujuan penggunaan data |
| `frameDecimation` | Tiap frame ke-N yang diambil (5 = ambil 1 dari setiap 5 frame) |
| `handedness` | Tangan yang di-scan: `"right"`, `"left"`, atau `"unknown"` |

---

## Parameter Scanning

| Parameter | Nilai | Keterangan |
|---|---|---|
| Depth range (export) | 10–50 cm | Di luar range → depth = 0 (invalid) |
| Depth range (UI state) | 10–60 cm | Threshold lebih toleran untuk `depthState == .inRange` |
| Durasi scan | 2 detik | `maxRecordingDuration` |
| Countdown | 3 detik | `countdownSeconds` |
| Frame decimation | 5 | Ambil 1 frame dari setiap 5 (efektif ~6fps dari kamera 30fps) |
| Max frames | 60 | `maxFramesToCapture` |
| Voxel size | 2 mm | Resolusi deduplication temporal (Alur B, tidak di-export) |
| Palm missing threshold | 10 frame | ~0.3 detik sebelum scan berhenti otomatis |
| Neighborhood filter | ≥3 tetangga, ±50 mm | Toleransi tetangga valid di neighborhood filter |

**Catatan:** Ada dua threshold depth yang berbeda — UI menggunakan 60 cm (lebih toleran agar badge "Too far" tidak terlalu sensitif), sedangkan filter export menggunakan 50 cm (lebih ketat untuk kualitas data ICP). Keduanya disengaja.

---

## Cara Transfer Data ke Mac

Setelah scan selesai, Share Sheet muncul otomatis. Pilih salah satu:

1. **AirDrop** — langsung ke Mac (paling cepat)
2. **Files app** → iCloud Drive → ambil dari Mac
3. **iTunes File Sharing** → Finder → iPhone → Files → TrueDepthStreamer

Folder yang diterima (format `.zip`) harus di-extract lalu dipindah ke:
```
Open3DRegistration/dataset/[nama_folder]/
```

Kemudian jalankan:
```bash
python process_all_scans.py
```

---

## Hubungan dengan Project Lain

```
StreamTrueDepth (iOS)
    export → Documents/rahmat_20260401_200613/
                depth00.bin ... depthNN.bin
                calibration.json
                metadata.json  (termasuk handedness)
        ↓  (transfer manual ke Mac)
Open3DRegistration/dataset/rahmat_20260401_200613/
    python process_all_scans.py
        ↓
Open3DRegistration/result/rahmat/20260401_200613/
    output.ply        ← point cloud terregistrasi
    geometry.json     ← 23 fitur biometrik
    texture.npy       ← tekstur 2D kanonis
    cnn_input.npy     ← input CNN (1024×6)
        ↓  (copy manual)
3DCNN/dataset/rahmat/20260401_200613/
    geometry.json
    cnn_input.npy
        ↓
01_train.ipynb  →  training GeoAtt-PointNet++
02_evaluate.ipynb  →  evaluasi identifikasi
```

---

## Catatan Implementasi

### Mengapa Vision tidak bisa jadi trigger scan?
Vision (`VNDetectHumanHandPoseRequest`) dijalankan pada buffer **video RGB**, sementara trigger scan didasarkan pada **data depth**. Dua hal berbeda:
- Mendeteksi ada tidaknya tangan di gambar ≠ mendeteksi tangan dalam jarak yang tepat dari sensor
- Depth check langsung dan cepat; Vision lebih lambat (inference neural network) dan dijalankan asinkron

### Mengapa `AVCameraCalibrationData`?
iPhone menyimpan parameter optik kamera di hardware. Apple mengeksposnya melalui `AVCameraCalibrationData` sehingga tidak perlu proses kalibrasi eksternal (seperti checkerboard). Focal length, principal point, dan lens distortion table langsung tersedia per-frame dari driver.

### Reliabilitas deteksi handedness
- iOS 15+ menggunakan `VNHumanHandPoseObservation.chirality` — ini adalah hasil Vision ML, bukan sensor hardware
- Hasil per-frame bisa fluktuatif → majority vote memberikan hasil yang lebih stabil
- Disarankan untuk selalu scan **satu tangan yang sama** (misalnya selalu tangan kanan) untuk konsistensi dataset
- Jika iOS < 15, field `handedness` akan selalu `"unknown"`
