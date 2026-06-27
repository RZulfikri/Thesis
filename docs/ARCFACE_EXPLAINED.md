# ArcFace untuk Identifikasi Telapak Tangan 3D — Penjelasan Komprehensif

> Dokumen referensi (Bahasa Indonesia) untuk riset v8 & penulisan paper IEEE.
> Menjelaskan: apa itu ArcFace, matematikanya, relasinya ke identifikasi telapak,
> status implementasi di repo ini, apakah ArcFace bisa dipakai di luar wajah,
> dan celah perbaikan ArcFace khusus telapak (backup novelty).

---

## 1. Apa itu ArcFace?

**ArcFace** (Deng et al., CVPR 2019 — *Additive Angular Margin Loss*) adalah fungsi loss
untuk **deep metric learning**: melatih jaringan agar menghasilkan **embedding** (vektor fitur)
yang **diskriminatif antar identitas**. Intuisinya:

- Embedding di-proyeksikan ke **permukaan hypersphere** (semua vektor dinormalisasi L2 → panjang 1).
- Identitas yang sama dikumpulkan rapat; identitas berbeda didorong **terpisah secara sudut**.
- Caranya: menambahkan **margin sudut (angular margin) `m`** pada sudut antara embedding dan
  pusat kelasnya saat training → memaksa jaringan belajar batas antar-kelas yang lebih lebar.

Berbeda dari klasifikasi softmax biasa (yang hanya memisahkan kelas seadanya), ArcFace
**memaksa jarak antar-kelas (inter-class) lebih besar** dan **intra-class lebih rapat** —
persis yang dibutuhkan biometrik (banyak kelas, sedikit sampel, harus general ke data baru).

---

## 2. Matematika

### 2.1 Dari softmax ke cosine
Softmax klasik memakai logit `W_jᵀx + b`. ArcFace menghapus bias, **menormalkan** bobot kelas
`W_j` dan embedding `x` (L2), sehingga logit menjadi **cosine similarity**:

```
cos(θ_j) = (W_j / ‖W_j‖) · (x / ‖x‖)
```

`θ_j` = sudut antara embedding dan pusat kelas `j`.

### 2.2 Margin sudut
ArcFace menambah margin `m` (radian) pada sudut kelas **target** `y`:

```
logit_target  = s · cos(θ_y + m)
logit_nontarget = s · cos(θ_j),  j ≠ y
```

dengan `s` = **scale** (suhu) yang memperbesar rentang logit sebelum softmax/cross-entropy.
Penjabaran (yang dipakai di implementasi):

```
cos(θ + m) = cosθ · cos m − sinθ · sin m,   sinθ = √(1 − cos²θ)
```

biasanya dgn **easy-margin**: bila `cosθ` sudah sangat besar, margin dikurangi agar stabil numerik.
Loss akhir = **cross-entropy** atas logit ber-skala tersebut.

### 2.3 Peran `s` (scale) dan `m` (margin)
- `m` besar → batas antar-kelas makin tegas, tapi training makin sulit (bisa tak konvergen).
- `s` mengontrol "ketajaman" distribusi softmax; terlalu kecil → gradien lemah, terlalu besar → overfit.
- Nilai umum: `m≈0.5`, `s≈30–64`. Pada dataset kecil, `m` moderat (0.3–0.5) lebih aman.

### 2.4 Keluarga margin loss (beda formula)
| Loss | Logit target | Jenis margin |
|---|---|---|
| **Softmax (cosine)** | `s·cosθ` | tanpa margin |
| **CosFace** (Wang 2018) | `s·(cosθ − m)` | additive **cosine** margin |
| **ArcFace** (Deng 2019) | `s·cos(θ + m)` | additive **angular** margin |
| **SubCenter-ArcFace** (Deng 2020) | `max_k cos` lalu margin | K sub-pusat/kelas (tahan noise intra-kelas) |
| **AdaCos** (Zhang 2019) | `s_dyn·cosθ` | scale `s` **adaptif otomatis** (tanpa tuning) |
| **CurricularFace** (Huang 2020) | margin + modulasi negatif sulit | kurikulum easy→hard |

> **Catatan penting:** `cos(θ+m)` (ArcFace) **≠** `cosθ − m` (CosFace). Keduanya beda mekanisme;
> hanya kebetulan mirip untuk `θ` kecil. Ini relevan untuk §3.

---

## 3. Status implementasi di repo ini (PENTING untuk klaim paper)

Di `3DCNN/losses/arcface.py` (v7.x), `ArcMarginProduct.forward` memakai:

```python
phi = cosine - self.m          # baris ~61 — "approksimasi linear"
```

Ini **rumus CosFace** (`cosθ − m`), **bukan** ArcFace sejati (`cos(θ+m)`). Selain itu, di
`models/siamese.py` head margin **selalu** `ArcMarginProduct`, sedangkan `criterion`
(CosFace/SubCenter) hanya dipakai cross-entropy-nya — sehingga di v7.x **semua "loss margin"
sebenarnya memakai head yang sama** (`cosθ − m`), berbeda hanya pada nilai margin.

**Implikasi untuk paper:** klaim "ArcFace" harus akurat. v8 menambahkan **true-ArcFace**
(`cos(θ+m)`, easy-margin) dan menyatukan semua head di `losses/margin_heads.py` agar
perbandingan ArcFace vs CosFace vs SubCenter vs AdaCos/CurricularFace **jujur dan benar**.
(Head `arcface` varian `linear` dipertahankan identik dgn v7.x untuk reproduksibilitas &
reuse checkpoint v7.2.0.)

---

## 4. Relasi ArcFace ↔ identifikasi telapak tangan

Pipeline pengenalan telapak (v7.x/v8):
1. **Backbone** (PointNet++) memetakan point cloud telapak → **embedding 128-D** (L2-normalized).
2. Saat **training**: head ArcFace menempatkan tiap subjek (identitas = kelas) pada pusat di
   hypersphere dgn margin sudut → embedding antar-subjek terpisah.
3. Saat **enrollment**: beberapa scan subjek → rata-rata embedding (multi-frame fusion N).
4. Saat **probe/verifikasi**: scan uji → embedding → **cosine similarity** vs galeri;
   ambang menentukan match/non-match. Metrik: **EER, AUC, TAR@FAR**.
5. **Open-set**: subjek baru (tak ada di training) tetap bisa dibandingkan karena yang dipelajari
   adalah **ruang embedding**, bukan classifier tetap — inilah kekuatan metric learning untuk biometrik.

Jadi ArcFace **hanya dipakai saat training** (membentuk ruang embedding yang baik); saat
inferensi yang dipakai cuma encoder + cosine. Ini sama persis dgn face recognition modern.

---

## 5. Apakah ArcFace bisa untuk telapak / non-wajah? Kenapa?

**Bisa — ArcFace agnostik-modalitas.** ArcFace bekerja pada **embedding + label identitas**,
bukan pada piksel wajah. Syaratnya hanya dua:
1. Ada **backbone** yang memetakan input (citra/point cloud/audio) → embedding.
2. Ada **label identitas** untuk training.

Karena itu ArcFace/CosFace sudah terbukti di banyak modalitas selain wajah:
- **Speaker verification** (suara) — mis. varian AAM-softmax di ECAPA-TDNN.
- **Fingerprint / finger-vein**, **person re-ID**, **palmprint 2D** (citra telapak).
- **Point cloud** (objek/3D) — angular margin pada embedding 3D.

**Kenapa cocok untuk telapak khususnya:** telapak punya **variasi intra-kelas tinggi**
(pose tangan, jarak scan, oklusi, kualitas depth). Margin sudut memaksa model menstabilkan
embedding identitas yang sama meski penampakannya bervariasi → lebih tahan daripada softmax biasa.
Bukti di proyek ini: ArcFace melompatkan Rank-1 dari ~60% (Triplet) ke ~99% dan d′ jauh lebih
tinggi (v3/v6), serta EER turun ke ~1% (v7.x) — pada PointNet++ yang literatur sebelumnya
nilai "lemah" untuk biometrik tangan.

---

## 6. Apakah ada "yang seperti ArcFace tapi untuk telapak"?

- **Palmprint 2D**: beberapa karya memakai ArcFace/CosFace pada citra telapak (CNN 2D).
- **Hand/3D**: karya point-cloud tangan terdahulu (Svoboda IJCB'20; Zhang MDPI'23) memakai
  **softmax/klasifikasi biasa** sebagai baseline dan menilai PointNet++ lemah — **belum ada
  yang memakai ArcFace (angular margin) pada PointNet++ untuk identifikasi telapak 3D**.
- **Multimodal**: Micucci & Iula (2023) memfusikan palmprint 3D + hand-geometry (score-level)
  → EER 1,18% → 0,06% (bukan ArcFace, tapi menunjukkan nilai geometri tangan).

→ **Celah riset (gap):** ArcFace (dan true-ArcFace) pada PointNet++ untuk **identifikasi
telapak 3D** adalah wilayah yang belum dieksplor — inilah salah satu kontribusi v8.

---

## 7. Celah perbaikan ArcFace untuk telapak (backup novelty — Track C)

Margin tetap (`m` konstan) memperlakukan semua sampel sama, padahal scan telapak **sangat
bervariasi kualitasnya**. Arah perbaikan (mengikuti tren quality-aware di face recognition):

- **AdaCos** (auto-scale `s`) — hilangkan tuning `s`.
- **CurricularFace** — kurikulum easy→hard (fokus bertahap ke sampel sulit).
- **AdaFace / MagFace** — margin/skala diadaptasi **kualitas** (AdaFace pakai ‖fitur‖ sbg proxy).
- **USULAN proyek ini — QA-ArcFace (Quality-Adaptive ArcFace):** margin sudut diadaptasi
  **skor kualitas scan 3D dari `geometry.json`** (densitas titik `point_count`, `scan_distance_mm`,
  `palm_depth_std_mm`, penalti `warnings` spt knuckle_fallback):

  ```
  m_eff = m · (q_floor + (1 − q_floor) · q),   q ∈ [0,1]
  ```

  Sampel kualitas-rendah → margin longgar (tidak over-penalize scan buruk); kualitas-tinggi →
  margin penuh. Karena `geometry.json` sudah mengandung sinyal kualitas yang **interpretable &
  palm-specific**, pendekatan ini lebih cocok untuk telapak daripada proxy ‖fitur‖ (yang di repo
  ini hilang karena encoder me-normalize embedding sebelum loss, `encoder.py:168`).

  **Hipotesis pembeda (H-C2):** QA-ArcFace paling menolong pada **subset scan kualitas-rendah**
  → dibuktikan lewat **EER terstratifikasi kualitas** (bukan hanya EER global).

Implementasi semua head ini ada di `3DCNN/losses/margin_heads.py`; eksperimen di
`collab/v8b_arcface_lab.ipynb` (Track C, Colab terpisah untuk paper-writing).

---

## 8. Ringkasan untuk paper

- ArcFace = angular-margin metric learning; agnostik-modalitas; ideal untuk biometrik open-set.
- Untuk telapak 3D + PointNet++: **belum dieksplor** → kontribusi.
- Repo v7.x sebenarnya memakai rumus CosFace berlabel "ArcFace" → v8 memperbaiki dgn true-ArcFace
  + perbandingan loss yang benar.
- Backup novelty: **QA-ArcFace** — margin adaptif berbasis kualitas scan 3D (geometry.json),
  diuji via EER terstratifikasi kualitas.

**Referensi kunci:** Deng et al. 2019 (ArcFace); Wang et al. 2018 (CosFace); Deng et al. 2020
(Sub-center ArcFace); Zhang et al. 2019 (AdaCos); Huang et al. 2020 (CurricularFace); Kim et al.
2022 (AdaFace); Meng et al. 2021 (MagFace); Micucci & Iula 2023 (palmprint+geometry fusion);
Qi et al. 2017 (PointNet/PointNet++). Detail di `docs/literature/`.
