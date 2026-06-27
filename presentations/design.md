# DESIGN.md — Sistem Desain Deck "3D Palm Recognition: Progress & Research Status"

> **Apa ini.** Dokumen sistem desain berbasis teks (mengikuti konvensi *Stitch DESIGN.md* — lih. https://github.com/VoltAgent/awesome-design-md). Dibaca oleh **agen desain (Claude)** untuk merender **deck presentasi** yang konsisten dari konten di `HANDOVER_DECK_v7_2_0.md`. Semua slide WAJIB patuh ke aturan visual di sini.
>
> **Pasangan file.** `HANDOVER_DECK_v7_2_0.md` = isi (apa yang ditulis per slide). `design.md` = tampilan (bagaimana merendernya). `assets/` = gambar bukti.

---

## 1. Visual Theme & Atmosphere

- **Karakter:** akademik–riset, bersih, *data-forward*. Setiap slide = satu ide + satu bukti. Rasio sinyal-terhadap-noise tinggi; tidak ada dekorasi yang tidak membawa informasi.
- **Mood:** tenang, kredibel, presisi. Cocok untuk sidang tesis / progress review.
- **Densitas:** sedang. Maksimum ~6 bullet atau 1 gambar besar + 1 kalimat *takeaway* per slide.
- **Aspek:** 16:9 (1280×720 px master).
- **Bahasa:** Bahasa Indonesia. Istilah teknis (EER, AUC, ArcFace, PointNet++, FPS, canonical, point cloud, multi-frame fusion) ditulis apa adanya, tidak diterjemahkan.
- **Prinsip emas:** angka selalu **verbatim** dari sumber; klaim novelty selalu **di-hedge** ("berdasarkan literatur review / sepanjang pengetahuan kami").

---

## 2. Color Palette & Roles

| Peran | Nama | Hex | Penggunaan |
|---|---|---|---|
| Primary (brand) | `ink` | `#0F172A` | judul, teks utama di latar terang |
| Surface | `paper` | `#F8FAFC` | latar slide default |
| Surface alt | `cloud` | `#EEF2F7` | kartu/callout, baris tabel selang-seling |
| **Accent (winner)** | `jade` | `#0E9F6E` | menandai **R3 / canonical / hasil positif / keputusan** |
| Accent-2 (fokus) | `azure` | `#2563EB` | highlight metrik kunci, garis utama chart |
| **Danger (rejected)** | `rust` | `#DC2626` | menandai **R1 raw / GeoAtt gagal / regresi / blocker** |
| Warning (caveat) | `amber` | `#D97706` | catatan keterbatasan, anomali 90°, "data leakage" |
| Muted | `slate` | `#64748B` | caption, sumber, label sekunder |
| Divider/section bg | `midnight` | `#0B1220` | latar slide pembatas bagian (teks putih) |

**Aturan warna semantik (WAJIB konsisten di semua chart & tabel):**
- **R1 / raw_ply / kegagalan** → `rust`.
- **R2 / canonical** → `azure`.
- **R3 / fps / pemenang** → `jade`.
- Naik-baik (EER turun) → `jade`; memburuk → `rust`; netral → `slate`.
- Maksimum 2 warna aksen menonjol per slide.

---

## 3. Typography Rules

- **Font:** Inter / Helvetica Neue / system-sans (heading & body). Angka & monospace: `JetBrains Mono` atau `SF Mono`.
- **Hierarki:**
  | Elemen | Ukuran (pt @16:9) | Berat | Warna |
  |---|---|---|---|
  | Judul slide | 32–40 | 700 | `ink` |
  | Sub-judul | 22–26 | 600 | `ink` |
  | Body / bullet | 18–22 | 400 | `ink` |
  | Angka metrik besar (KPI) | 54–72 | 800 | `azure`/`jade`/`rust` |
  | Caption / sumber | 12–14 | 400 | `slate` |
  | Mono (angka tabel, path) | 16–18 | 500 | `ink` |
- **Aturan angka:** EER & metrik proporsi tampil **persen 1 desimal** untuk headline (mis. `9,9%`) tetapi **boleh 4 desimal** di tabel detail (mis. `0,0991 ± 0,0263`); konsisten gunakan **koma desimal** (gaya Indonesia). Selalu sertakan `± std` bila ada.

---

## 4. Component Stylings (tipe slide)

Setiap entri di handover diberi tag `[TYPE]`. Render sesuai pola berikut.

1. **`[COVER]`** — latar `midnight`, judul putih besar tengah, sub-judul + tanggal/versi + penulis kecil. Tanpa gambar atau 1 PLY samar sebagai latar.
2. **`[SECTION]`** — pembatas bagian: latar `midnight`, nomor bagian besar (`jade`) + judul bagian putih, 1 kalimat ringkas.
3. **`[CONTENT]`** — judul atas; 3–6 bullet; opsional ikon. Sisakan ruang kosong; jangan penuh.
4. **`[FIGURE]`** — 1 gambar dominan (≥60% area) dari `assets/`; caption `slate` di bawah; **takeaway** 1 kalimat tebal di pita bawah (warna sesuai semantik). Judul kecil di atas.
5. **`[METRICS]`** — kartu KPI besar (1–3) memakai angka raksasa + label; latar kartu `cloud`. Untuk menonjolkan 1 angka penentu (mis. `1,14%`).
6. **`[TABLE]`** — tabel rapi; header `ink` di atas `cloud`; baris selang-seling; **sel pemenang di-highlight** (`jade` lembut), sel terburuk (`rust` lembut). Maks ~6 kolom × ~9 baris; jika lebih, pecah 2 slide.
7. **`[FINDING]`** / callout — kartu besar `cloud` dengan border kiri tebal (warna semantik) berisi 1 temuan kunci + angka pendukung. Untuk "kanonikalisasi WAJIB", "R3 Pareto winner", dll.
8. **`[COMPARE]`** — 3 kolom sejajar **R1 | R2 | R3** (atau with_geom | no_geom). Tiap kolom: nama, 1 angka utama, ikon ✓/✗, warna semantik kolom.
9. **`[GAP]`** — slide kontribusi/gap: daftar bernomor; tiap gap = klaim (di-hedge) + bukti pendukung kecil di bawahnya.
10. **`[QUOTE]`** — kutipan/temuan dari laporan atau sitasi literatur; teks miring besar + sumber `slate`.
11. **`[TIMELINE]`** — riwayat versi v1→v7.2.0: pita horizontal bernode; tiap node = versi + 1 metrik kunci; warnai milestone (pivot = `amber`, hasil positif = `jade`).

---

## 5. Layout Principles

- **Grid:** 12 kolom, margin aman 64 px tiap sisi; *gutter* 24 px.
- **Spacing scale:** 8 / 16 / 24 / 40 / 64.
- **Pola wajib `[FIGURE]`:** gambar + **takeaway** (jangan pernah gambar tanpa kalimat kesimpulan).
- **Satu ide per slide.** Bila butuh 2 ide → pecah jadi 2 slide (tak ada batas jumlah slide).
- **Konsistensi posisi:** judul selalu kiri-atas; nomor slide + label bagian kanan-bawah (`slate`).
- **Tabel besar/heatmap:** beri ruang penuh; takeaway di bawah.

---

## 6. Depth & Elevation

- Latar slide datar (`paper`). Kedalaman hanya untuk **menonjolkan**: kartu KPI & callout `[FINDING]` memakai shadow halus (`0 1px 3px rgba(15,23,42,.12)`) + radius 12 px.
- Border kiri 4–6 px berwarna semantik untuk callout. Hindari shadow berlebihan/gradien ramai.

---

## 7. Do's & Don'ts

**Do**
- Kutip angka **persis** dari sumber; cantumkan `± std` dan sumber file di caption (`slate`, kecil).
- Pakai warna semantik konsisten (R1=`rust`, R2=`azure`, R3=`jade`).
- Selalu beri **takeaway** 1 kalimat pada slide bukti.
- Hedge klaim kebaruan; tandai caveat dengan `amber`.
- Bedakan visual: hasil **valid** vs hasil **tercemar leakage** (beri badge `amber` "⚠ data leakage").

**Don't**
- Jangan klaim "EER 0% = sistem sempurna" — selalu tulis "di lantai resolusi pengukuran".
- Jangan campur angka A100 vs T4 di satu tabel kecepatan tanpa keterangan GPU.
- Jangan chartjunk (3D bar, gradien ramai, ikon dekoratif tak bermakna).
- Jangan padatkan >6 bullet; jangan tabel >6 kolom.
- Jangan terjemahkan istilah teknis.

---

## 8. Responsive / Aspect

- **Master 16:9** (1280×720). Ekspor PDF & PPTX 16:9.
- **Safe margin** 64 px; font minimum **18 pt** (proyeksi ruang sidang).
- Gambar: jaga rasio asli, jangan distorsi; heatmap/tabel lebar → boleh *full-bleed* dengan margin 32 px.
- Kontras minimal AA (teks `ink` di `paper`/`cloud`; teks putih di `midnight`).

---

## 9. Agent Prompt Guide (untuk agen desain Claude)

Gunakan prompt siap-pakai berikut; isi `{…}` dari entri handover.

- **Cover:** *"Buat slide COVER 16:9 latar `midnight`. Judul putih 700 '{judul}'. Sub-judul '{subjudul}', baris kecil '{penulis} · {tanggal} · {versi}'. Minimalis."*
- **Section:** *"Slide SECTION latar `midnight`: nomor bagian '{n}' besar warna `jade`, judul putih '{judul bagian}', satu kalimat '{ringkas}'."*
- **Figure:** *"Slide FIGURE: judul kecil '{judul}'. Tempatkan `assets/{file}` dominan (≥60%). Caption `slate` '{sumber}'. Pita bawah takeaway tebal '{takeaway}' warna {semantik}."*
- **Metrics/KPI:** *"Slide METRICS: kartu '{angka}' raksasa warna {semantik} + label '{label}'. Opsional 2 kartu pendukung. Latar kartu `cloud`."*
- **Table:** *"Slide TABLE judul '{judul}'. Render tabel markdown berikut; highlight sel pemenang `jade`, terburuk `rust`. Takeaway bawah '{takeaway}'."*
- **Finding:** *"Slide FINDING: callout `cloud` border kiri {semantik} berisi '{temuan}' + angka '{angka}'."*
- **Compare:** *"Slide COMPARE 3 kolom R1|R2|R3 warna rust|azure|jade; tiap kolom angka utama + ✓/✗."*
- **Gap:** *"Slide GAP: daftar bernomor {gap}, tiap item klaim (hedged) + bukti kecil."*
- **Timeline:** *"Slide TIMELINE pita versi v1..v7.2.0; node {versi:metrik}; pivot `amber`, positif `jade`."*

---

## 10. Build / Render (opsional)

Handover bersifat Marp-compatible (`---` antar slide). Untuk render cepat:

```bash
# PDF
npx @marp-team/marp-cli presentations/HANDOVER_DECK_v7_2_0.md -o deck.pdf
# PPTX (editable)
npx @marp-team/marp-cli presentations/HANDOVER_DECK_v7_2_0.md --pptx -o deck.pptx
# HTML preview
npx @marp-team/marp-cli -s presentations/
```

Tema Marp dapat dibuat dari palet & tipografi di §2–§3 (CSS `/* @theme palm */`). Untuk hasil desain penuh, serahkan `HANDOVER_DECK_v7_2_0.md` + `design.md` + `assets/` ke agen desain Claude dan ikuti §9.
