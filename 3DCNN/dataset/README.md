# dataset/ — diregenerasi dari raw (BUKAN disimpan di git)

Repo ini **tidak menyimpan dataset turunan**. Isi folder ini
(`output.ply` / `geometry.json` / `cnn_input.npy` / `cnn_input_fps.npy` / `align_*.npy`)
di-gitignore dan **dibangun ulang dari raw scan** (`Raw Depth Data/`) agar:
- repo tetap ramping (raw cuma ~91 MB vs dataset ~7 GB), dan
- dataset **reproducible** (= fungsi dari raw + kode).

## Cara mengisi folder ini

```bash
cd 3DRegistration
pip install -r requirements.txt          # open3d, opencv, scipy, sklearn, ...
python generate_dataset.py               # unzip raw → ply → geometry → cnn → fps → align
```

Di Colab otomatis: sel **§2c Generate Dataset** pada notebook v8 memanggil
`generate_dataset.py` (idempotent — skip bila sudah ada di sesi itu).

Estimasi ~35–90 menit (didominasi `process_single_frames`). Lihat README utama repo.
