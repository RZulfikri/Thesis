# Keyword Pencarian Literatur — v8: Point Cloud Alignment × ArcFace (target IEEE)

Daftar keyword untuk mencari referensi (Google Scholar, IEEE Xplore, arXiv, Semantic Scholar).
Dikelompokkan per tema. Gunakan kombinasi `("A" AND "B")` untuk mempersempit.

---

## 1. Alignment / Normalisasi Point Cloud (pilar 1)
- "point cloud canonicalization"
- "pose normalization point cloud"
- "PCA alignment point cloud recognition"
- "principal axis normalization 3D"
- "rotation-invariant point cloud learning"
- "SO(3) invariance / equivariance point cloud"
- "canonical pose estimation point cloud"
- "T-Net spatial transformer PointNet" / "learned input transform point cloud"
- "oriented bounding box (OBB) normalization hand"
- "PCA axis ambiguity sign disambiguation"
- "anatomical alignment hand 3D" / "landmark-based alignment"
- "rotation robustness 3D recognition ablation"

## 2. ArcFace / Angular-Margin Metric Learning (pilar 2)
- "ArcFace additive angular margin loss"
- "CosFace large margin cosine loss"
- "SphereFace angular softmax"
- "Sub-center ArcFace noisy"
- "angular margin loss open-set recognition"
- "deep metric learning biometrics"
- "cosine similarity verification embedding"
- "softmax vs margin loss face recognition"

## 3. Margin Adaptif / Quality-Aware (Track C — backup gap)
- "AdaCos adaptive scale cosine"
- "AdaFace quality adaptive margin"
- "MagFace magnitude quality recognition"
- "CurricularFace curriculum margin"
- "quality-aware face recognition margin"
- "sample quality weighting recognition"
- "adaptive margin biometric low quality"
- "image quality assessment biometrics"

## 4. Palm / Hand 3D Recognition (domain)
- "3D palmprint recognition point cloud"
- "contactless palmprint deep learning"
- "hand biometrics PointNet / PointNet++"
- "3D hand shape recognition biometric"
- "palm vein 3D recognition"
- "palmprint hand geometry fusion"
- "HKPolyU 3D hand database" / "contact-free 3D/2D hand images"
- "depth camera palmprint recognition" / "TrueDepth hand biometrics"
- "structured light 3D palmprint"

## 5. Gabungan / Posisi Kontribusi (untuk klaim novelty)
- "PointNet++ ArcFace" / "point cloud angular margin recognition"
- "canonicalization ablation recognition accuracy"
- "rotation-robust palmprint recognition"
- "normalization effect 3D biometric recognition"
- "multi-frame fusion biometric verification"

---

## Query siap-pakai (copy-paste)
- `("point cloud" OR "3D") AND ("ArcFace" OR "angular margin") AND ("palm" OR "hand" OR "biometric")`
- `("PointNet" OR "PointNet++") AND ("palmprint" OR "hand recognition")`
- `("canonicalization" OR "PCA alignment" OR "pose normalization") AND "point cloud" AND ("ablation" OR "rotation")`
- `("AdaFace" OR "MagFace" OR "AdaCos" OR "CurricularFace") AND ("quality" OR "adaptive margin")`
- `("hand geometry" OR "finger length" OR "palm width") AND ("alignment" OR "normalization") AND 3D`

## Referensi yang sudah dipegang (dari lit review repo)
- Qi et al. 2017 — PointNet / PointNet++.
- Svoboda et al. IJCB 2020 — Clustered DGCNN; PointNet++ baseline lemah (30–53% acc) untuk hand biometrics.
- Zhang et al. MDPI 2023 — TMBNet/MVP vs PointNet/PointNet++ (PolyU-CFHD).
- Ge et al. CVPR 2018 — Hand PointNet (OBB-PCA normalization untuk pose).
- Micucci & Iula 2023 — palmprint 3D + hand geometry score-fusion (EER 1,18%→0,06%).
- Deng et al. 2019 (ArcFace), Wang 2018 (CosFace), Deng 2020 (Sub-center), Zhang 2019 (AdaCos),
  Huang 2020 (CurricularFace), Kim 2022 (AdaFace), Meng 2021 (MagFace).
- Liu et al. 2025 — survey deep learning palmprint recognition.

## Saran venue IEEE
- **IEEE IJCB** (International Joint Conference on Biometrics) — paling relevan.
- **IEEE FG** (Face & Gesture) / **IEEE WACV** / **IEEE ICIP** — biometrik/vision.
- **IEEE Access** (jurnal, akses cepat) bila butuh outlet lebih longgar.
- Workshop biometrik di **CVPR/ICCV** (IEEE-sponsored).

> Tips: untuk klaim "belum ada PointNet++ + ArcFace untuk telapak 3D", lakukan pencarian
> kombinasi grup 5 + grup 4 + grup 2; simpan tanggal & hasil agar bisa ditulis "to the best
> of our knowledge" dengan dasar yang kuat.
