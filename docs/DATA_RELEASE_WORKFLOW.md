# Data Release Workflow — Dataset sebagai GitHub Release (repo ramping)

> Mulai **v8**, dataset **tidak lagi disimpan di git**. Tiap versi data dikemas jadi
> tarball + manifest dan di-upload sebagai **aset GitHub Release**. Colab mengunduh +
> extract dataset dari Release (bukan meng-clone-nya dari git).

## Kenapa

`.git` sempat membengkak ke **~12.8 GB** karena dataset (`output.ply` / `cnn_input*.npy` /
`geometry.json` / `align_*.npy`, 2131 frame) di-commit & **diregenerasi berulang** lintas
v7.x. Tiap `git clone` (termasuk tiap runtime Colab baru) jadi mengunduh 12.8 GB.

**Yang ingin disimpan di git hanya code + report + evaluasi** (`3DCNN/result_docs/`,
`3DCNN/analysis/`, `docs/` — total ~18 MB). Dataset → **GitHub Releases** (aset di luar git,
tak dihitung di ukuran clone, bisa dihapus per versi).

> Catatan: **git tag saja tidak menolong** — tag hanya pointer ke commit; blob tetap di
> pack & tetap ter-clone. Yang benar adalah **release asset**.

## Komponen

| File | Peran |
|------|-------|
| `3DRegistration/pack_dataset_release.py` | Kemas `3DCNN/dataset/**` → `dataset_<ver>.tar.zst` (+ split bila >1.9 GB) + `MANIFEST_<ver>.json` (sha256 tiap part). |
| `3DRegistration/release_assets.py` | Helper REST GitHub Releases (pure-`urllib`): `create_or_get_release`, `upload_asset(s)`, `download_asset`, `delete_release`, `pull_dataset`. |
| `3DRegistration/make_align_variants.py` | Generate `align_*.npy` dari `output.ply` (dipakai sebelum pack bila align belum ada). |
| Notebook training (`v8_*.ipynb`, `v8b_*.ipynb`) sel **§2c Data Bootstrap** | `release_assets.pull_dataset(...)` → unduh+extract dataset (idempotent). |
| Notebook training sel **§5a** | Verifikasi `align_*.npy` lengkap; fallback generate lokal **tanpa commit**. |
| `3DCNN/collab/maintenance_strip_dataset.ipynb` | (Colab, **destruktif**) buang `3DCNN/dataset` dari seluruh history → force-push. |
| `.gitignore` | `3DCNN/dataset/**` di-ignore (kecuali `README.md`); `_release_dl/` di-ignore. |

Batas aset Release GitHub = **2 GB/file** → tarball >1.9 GB otomatis dipecah multi-part dan
disatukan ulang saat `pull_dataset`.

## Siklus pakai

### A. Membuat versi data baru (maintainer, di Colab atau mesin ber-disk cukup)

```bash
cd 3DRegistration
export GITHUB_TOKEN=ghp_xxx            # PAT scope repo

# 1) (bila align belum ada di dataset) generate dulu
python make_align_variants.py --data_dir ../3DCNN/dataset

# 2) kemas → /tmp/rel/dataset_v8.tar.zst(.partNN) + MANIFEST_v8.json
python pack_dataset_release.py --data_dir ../3DCNN/dataset --version v8 --out_dir /tmp/rel

# 3) upload SEMUA part + MANIFEST sebagai aset Release (tag data-v8)
python release_assets.py upload --repo RZulfikri/Thesis --tag data-v8 --files /tmp/rel/*

# 4) (opsional) hapus versi lama agar storage Release tetap ramping
python release_assets.py delete --repo RZulfikri/Thesis --tag data-v7
```

> `zstd` disarankan (`apt-get install -y zstd` / sudah ada di Colab). Tanpa zstd → fallback gzip.

### B. Memakai dataset saat training (Colab — otomatis)

Notebook training menjalankan **§2c Data Bootstrap** yang memanggil:

```python
import release_assets as ra
ra.pull_dataset('RZulfikri/Thesis', 'data-v8', DATA_DIR, GITHUB_TOKEN, workdir='/content/_release_dl')
```

→ unduh `MANIFEST` + semua part → verifikasi sha256 → satukan → extract ke `3DCNN/dataset/`.
**Idempotent**: bila folder sudah lengkap (file_count ≥ manifest), langsung skip. Path dataset
tidak berubah, sehingga kode training/eval tidak perlu disesuaikan.

Manual (di luar notebook):
```bash
python 3DRegistration/release_assets.py pull \
    --repo RZulfikri/Thesis --tag data-v8 --data_dir 3DCNN/dataset
```

## Verifikasi integritas

- `MANIFEST_<ver>.json` menyimpan `file_count`, `frame_count`, dan **sha256 tiap part**.
- `pull_dataset` menolak part yang sha256-nya tidak cocok (unduh ulang lalu re-cek).
- Setelah extract, `pull_dataset` mencetak jumlah file aktual vs `file_count` manifest.

## Runbook: history rewrite (sekali, untuk reclaim 12.8 GB lama)

Mengubah go-forward (releases) **tidak** mengecilkan `.git` yang sudah ada — blob dataset
masih ada di history. Untuk benar-benar reclaim, jalankan **`maintenance_strip_dataset.ipynb`
di Colab** (disk cukup, langsung ke github.com). Ringkasnya:

1. **Precondition**: Release `data-v8` ada & konsisten (semua part cocok manifest). _Jangan
   hapus history sebelum data aman di Release._
2. **Backup**: `git bundle --all` dari mirror → upload sbg Release `backup-prestrip-v8`
   (pemulihan total bila perlu).
3. **Rewrite**: `git filter-repo --path 3DCNN/dataset --invert-paths` pada mirror clone, lalu gc.
4. **Force-push**: `git push --force --all` + `--tags` (bukan `--mirror`, agar tag Release tak terhapus).
5. **Verifikasi**: clone segar `--single-branch --depth 1 --branch colab` → kecil & cepat;
   `result_docs/` & `analysis/` masih ada; `3DCNN/dataset/` hanya berisi `README.md`.

> ⚠️ **Destruktif & irreversible-ish**: semua commit SHA berubah; **PR lama invalid**; semua
> clone lama harus di-clone ulang. Notebook butuh `CONFIRM_DESTRUCTIVE=True` untuk lanjut.
>
> **Restore dari backup** (bila perlu):
> ```bash
> cat backup_prestrip_v8.bundle.part* > b.bundle   # bila displit
> git clone b.bundle restored_repo
> ```

## Aturan emas

- **Jangan** `git add` apa pun di bawah `3DCNN/dataset/` (sudah di-gitignore).
- Workdir unduhan (`_release_dl/`) selalu **di luar** working tree git (mis. `/content`).
- Satu versi data = satu Release tag (`data-vN`); hapus tag lama saat sudah tak dipakai.
- Report & evaluasi tetap di git (itulah yang ingin kita simpan permanen).
