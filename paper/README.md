# Paper — Canonical Alignment & Angular-Margin Learning for 3D Palm Recognition

Draf submission. **Target utama: ICAST-ES (ISAS) 2026** (internasional, IEEE Xplore,
**double-blind**, template IEEE, Bahasa Inggris). **Cadangan: IEIT Polinema** (jika
tak terpilih / lewat deadline). **Deadline ICAST-ES: 8 Juli 2026.**

## Build
Cara termudah: unggah folder `paper/` ke **Overleaf** (compiler pdfLaTeX), set `main.tex` sbg dokumen utama.

Lokal:
```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Isi
- `main.tex` — paper IEEEtran (conference), **anonim** untuk double-blind.
- `refs.bib` — referensi (beberapa entri ditandai `[CEK]` → verifikasi sebelum submit).
- `figs/` — figur dari `3DCNN/analysis/v8_factorial_20260629_182718/` + ilustrasi alignment.

## Sumber angka (reproducible)
Semua tabel/figur berasal dari `3DCNN/analysis/v8_factorial_20260629_182718/`
(`factorial_eer_6x2.csv`, `metrics_full.csv`, `significance_softmax_vs_arcface.csv`,
`rotation_sensitivity.csv`). Lihat `docs/PAPER_DESIGN.md`.

## TODO sebelum submit
- [ ] **Dataset**: isi jumlah sesi/frame & jumlah sesi holdout (cari `TODO[CONFIRM]` di `main.tex`).
- [ ] **Akuisisi**: model perangkat depth, jarak scan, kondisi, etik/konsen subjek.
- [ ] **Referensi**: verifikasi entri `[CEK]` di `refs.bib` (Svoboda IJCB'20, Zhang'23) + tambah sitasi pembimbing bila ada.
- [ ] **Double-blind (ICAST-ES)**: pastikan PDF tak memuat nama/afiliasi; hindari sitasi-diri yang membongkar identitas.
- [ ] **IEIT (jika geser)**: lepas anonim — isi `\author{}` dgn nama+afiliasi; cek apakah IEIT double-blind atau tidak.
- [ ] **Batas halaman**: sesuaikan dgn aturan venue (IEEE conf umumnya 4–6 hal); rapikan figur bila lebih.
- [ ] **Submission system**: ICAST-ES via EDAS.
