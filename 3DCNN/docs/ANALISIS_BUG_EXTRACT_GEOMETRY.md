# Analisis Bug: `extract_geometry.py` — Instabilitas `finger_width_5` pada Subjek Nola

**Tanggal:** 2026-05-17  
**File yang dianalisis:** `3DRegistration/extract_geometry.py`  
**Dataset:** `3DCNN/dataset/nola/` (22 sesi, 10 frame/sesi, 220 frame total)  
**Tools:** `utils/data_qc_v2.py`, `utils/audit_geom_session_variance.py`

---

## 1. Ringkasan Temuan

| Aspek | Kesimpulan |
|-------|-----------|
| Data PLY mentah nola | ✅ **Normal** — point count, bbox, density konsisten dengan subjek lain |
| Fitur geometri ekstraksi | ❌ **Bermasalah** — `detect_knuckle_y` gagal pada pose tertentu |
| Fitur paling terdampak | `finger_width_5` (kelingking) — CV antar-sesi **8.85×** rata-rata subjek lain |
| Root cause | Heuristik `max(X-width)` di `detect_knuckle_y` tidak robust untuk variasi pose |
| Dampak ke model | `with_geom` gagal pada nola di 4/5 seed; `no_geom` tidak pernah gagal |

---

## 2. Eviden Numerik

### 2.1 Distribusi Bimodal `finger_width_5` Nola

Dari 220 frame nola, `finger_width_5` menunjukkan **distribusi bimodal** yang ekstrem:

| Cluster | Sesi | fw5 Mean | fw5 Within-Std | `palm_height` | Keterangan |
|---------|------|----------|---------------|---------------|------------|
| A | 112503, 112505, 112508 | ~17.9 mm | **4.9 mm** | ~30–40 mm | Instabil dalam sesi; knuckle gagal parcial |
| B | 112515, 112519, 112520 | ~25.5 mm | 0.7–1.1 mm | ~8 mm | Knuckle gagal konsisten; semua fw ≈ band_w |
| C/D | 112528–112558 (15+4 sesi) | ~8.6 mm | 0.4–0.8 mm | ~64 mm | **Normal**; knuckle terdeteksi dengan benar |

**Pinky biologis tidak mungkin 8.5 mm → 26 mm pada orang yang sama.** Variasi ini adalah artefak algoritma, bukan anatomi.

### 2.2 Korelasi Palm Height vs Finger Width

Korelasi `palm_height` vs `fw5` untuk nola: **r = −0.971** (negatif hampir sempurna).

Semakin kecil `palm_height`, semakin tinggi `fw5`. Ini menunjukkan bahwa ketika `detect_knuckle_y` gagal (menghasilkan `knuckle_y` terlalu rendah), zona jari menjadi terlalu besar, dan `finger_widths` mendekati lebar band (`palm_width / 5`).

### 2.3 Min/Max Finger Width Ratio

| Kondisi | Min/Max fw Ratio | Interpretasi |
|---------|-----------------|--------------|
| Normal (nola C/D, feby) | 0.30–0.77 | Pinky jauh lebih tipis dari jari lain |
| Knuckle gagal parcial (nola A) | 0.62–0.70 | Semua jari mendekati lebar serupa |
| Knuckle gagal total (nola B, rahmat, taufik) | 0.87–0.92 | Semua jari hampir identik ≈ `band_w` |

---

## 3. Root Cause Analysis

### 3.1 Fungsi `detect_knuckle_y` (Baris 79–108)

```python
def detect_knuckle_y(pts_mm, wrist_y_top):
    y_scan_max = y_min + (y_max - y_min) * 0.55
    # ... scan slice Y dari wrist_y_top ke y_scan_max
    # knuckle_y = Y dengan X-width maksimum
```

**Masalah:** Fungsi mengasumsikan bahwa **lebar X maksimum** selalu terjadi di knuckle row. Pada pose tertentu:
- Jari-jari tidak terbuka lebar → knuckle row tidak memiliki X-width maksimum
- Area wrist lebih lebar dari knuckle → `max_width` tertangkap di wrist
- Pose ekstrem menyebabkan knuckle "tertutup" → lebar maksimum bergeser

### 3.2 Dampak Rantai (Chain Effect)

Ketika `detect_knuckle_y` gagal:

1. `knuckle_y` terdeteksi terlalu rendah (dekat wrist)
2. `palm_height = knuckle_y - wrist_y_top` → sangat kecil (8–40 mm vs normal 60+ mm)
3. `compute_finger_widths_and_gaps` menggunakan `zone_mask = Y > knuckle_y`
4. Zona jari mencakup hampir seluruh tangan → `band_pts` di setiap band sangat besar
5. `finger_widths = p95 - p5` di setiap band mendekati `band_w = (x_max - x_min) / 5`
6. Karena fw5 (pinky) biologisnya paling kecil, **deviasi relatifnya paling besar** → terdeteksi sebagai outlier di audit antar-sesi

### 3.3 Kenapa Hanya Nola yang Outlier?

Subjek lain (rahmat, taufik, fadhil, chrys, gede, alji) juga mengalami **knuckle detection failure** di HAMPIR SEMUA sesi mereka (`palm_height` ≈ 0–10 mm). Namun, karena kegagalan tersebut **konsisten antar-sesi**, `finger_width_5` mereka tetap stabil (~23 mm). CV antar-sesi rendah → tidak terdeteksi sebagai outlier.

**Nola unik karena:** beberapa sesi knuckle gagal, beberapa sesi normal → **bimodal** → CV tinggi.

---

## 4. Reproduksi Bug

```bash
cd 3DRegistration
python extract_geometry.py \
    ../3DCNN/dataset/nola/20260513_112503/frame_00/output.ply \
    --handedness right
```

Output:
```
  Wrist ROI: y_center=-76.3mm, y_top=-65.2mm
  Knuckle Y: -56.9mm          ← SALAH (seharusnya ~0 mm)
  Palm: 105.1 x 8.4 mm (W×H)  ← palm_height anomali (normal ~64 mm)
  Finger widths (mm): [24.23, 24.53, 25.59, 25.38, 21.35]
                        ↑ semua mendekati band_w = 21.0 mm
```

Bandingkan dengan sesi normal:
```bash
python extract_geometry.py \
    ../3DCNN/dataset/nola/20260513_112541/frame_00/output.ply \
    --handedness right
```

Output:
```
  Wrist ROI: y_center=-75.0mm, y_top=-63.4mm
  Knuckle Y: 0.7mm            ← BENAR
  Palm: 125.8 x 64.1 mm (W×H) ← normal
  Finger widths (mm): [22.94, 25.3, 25.43, 25.06, 9.68]
                                          ↑ pinky normal ~9.7 mm
```

---

## 5. Rekomendasi Perbaikan

### 5.1 Perbaikan Segera (Hotfix)

Tambahkan **sanity check** setelah `detect_knuckle_y`:

```python
# Setelah detect_knuckle_y
def _validate_knuckle_y(knuckle_y, wrist_y_top, y_max):
    """
    Validasi knuckle_y terdeteksi masuk akal.
    Knuckle row seharusnya minimal 30mm di atas wrist top,
    dan tidak lebih dari 80% total tinggi tangan.
    """
    palm_height = knuckle_y - wrist_y_top
    total_height = y_max - wrist_y_top
    if palm_height < 20.0 or palm_height > total_height * 0.75:
        return False
    return True

if not _validate_knuckle_y(knuckle_y, wrist_y_top, y_max):
    # Fallback: gunakan estimasi anatomis
    # Knuckle row biasanya ~45-55% total tinggi dari wrist
    knuckle_y = wrist_y_top + total_height * 0.50
    quality_issues.append("knuckle_fallback:heuristic_used")
```

### 5.2 Perbaikan Struktural (v0.4.1+)

Ganti heuristik `max(X-width)` dengan pendekatan yang lebih robust:

| Pendekatan | Deskripsi | Kompleksitas |
|-----------|-----------|--------------|
| **A. Gradient-based** | Cari Y di mana laju perubahan X-width maksimum (turunan width terhadap Y) | Sedang |
| **B. Multi-slice voting** | Gunakan 3 slice heights berbeda, ambil modus knuckle_y | Rendah |
| **C. Template matching** | Cocokkan profil X-width dengan template knuckle row ideal | Sedang |
| **D. Landmark-based** | Deteksi fingertip dulu, lalu estimasi knuckle sebagai % panjang jari | Rendah |

**Rekomendasi:** Implementasikan **B (Multi-slice voting)** sebagai hotfix — murah dan robust:

```python
def detect_knuckle_y_v2(pts_mm, wrist_y_top, n_slices_list=[20, 40, 80]):
    """Voting dari multiple slice resolutions untuk robustness."""
    candidates = []
    for n_slices in n_slices_list:
        # ... hitung knuckle_y untuk slice ini
        candidates.append(knuckle_y)
    # Ambil median (robust terhadap outlier)
    return float(np.median(candidates))
```

### 5.3 Fitur Geometri Alternatif (Future Work)

Jika perbaikan knuckle detection tidak cukup, pertimbangkan:

1. **Drop `finger_width_5`** dari vektor geometri — fitur paling tidak stabil
2. **Gunakan rasio** `finger_width_i / finger_width_3` — scale-invariant, lebih robust
3. **Ekstrak fitur dari cnn_input.npy** (point cloud) secara end-to-end — tidak bergantung pada heuristic landmark

---

## 6. Decision Matrix

| Opsi | Biaya | Validitas | Timeline |
|------|-------|-----------|----------|
| A. Hotfix sanity check + fallback | 2 jam | ✓ Menangkap gagal deteksi | v0.4.0 |
| B. Re-extract geometry full dataset | 4–6 jam | ✓ Data bersih | v0.4.0 |
| C. Perbaikan struktural detect_knuckle_y | 1–2 hari | ✓ Paling ideal jangka panjang | v0.4.1 / Future Work |
| D. Drop finger_width_5 dari fitur | 30 menit | ⚠️ Mengurangi dimensi fitur | v0.4.0 (fallback) |

**Rekomendasi untuk v0.4.0:**
1. Jalankan **hotfix sanity check** (Opsi A)
2. Re-extract geometry untuk nola (3 sesi cluster A + 3 sesi cluster B) + subjek lain yang ter-flag QC v2
3. Jika masih ada masalah, gunakan **Opsi D** sebagai fallback cepat

---

## 7. Lampiran: Kode Verifikasi

```python
# Verifikasi knuckle detection failure per sesi
import json
import numpy as np
from pathlib import Path

for sess in ['20260513_112503', '20260513_112541', '20260513_112555']:
    ph_vals, fw5_vals = [], []
    for frame_dir in Path(f'dataset/nola/{sess}').glob('frame_*'):
        with open(frame_dir / 'geometry.json') as f:
            geo = json.load(f)
        ph_vals.append(geo['palm_height_mm'])
        fw5_vals.append(geo['finger_widths_mm'][4])
    print(f"{sess}: palm_height={np.mean(ph_vals):.1f}±{np.std(ph_vals):.1f}, "
          f"fw5={np.mean(fw5_vals):.1f}±{np.std(fw5_vals):.1f}")
```

Output:
```
20260513_112503: palm_height=30.2±28.5, fw5=17.9±4.9   ← knuckle GAGAL
20260513_112541: palm_height=63.8±1.7,  fw5=9.4±0.5    ← knuckle BENAR
20260513_112555: palm_height=63.8±1.7,  fw5=8.4±0.4    ← knuckle BENAR
```
