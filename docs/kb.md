# Basis pengetahuan (KB)

Sumber: `kb/knowledge.yaml`. Dari sini dibangkitkan:

- `kb/chunks.jsonl`: potongan kecil untuk retrieval (BM25 dan embedding).
- `kb/faq.jsonl`: pertanyaan dan variasinya untuk FAQ semantik.
- `kb/entities.json`: alias fakultas dan peran untuk pengaman FAQ.

`kb/build_kb.py` juga menulis `knowledge.txt` (format lama) dan `faq_review.md` (daftar untuk
ditinjau manusia). Keduanya tidak dibaca mesin v4, jadi tidak dibawa ke repo.

```bash
cd kb
python build_kb.py --check     # validasi saja
python build_kb.py             # bangun ulang chunks, faq, entities
```

Data tabel (`kb/data/*.csv`: UKT dan IPI 2026) juga dibaca `build_kb.py`.

Setelah mengubah KB, jalankan `make test`. Tes engine golden akan memberi tahu bila rute FAQ berubah
(tes itu butuh fixture, lihat `docs/testing.md`). Embedding dihitung ulang otomatis saat service
start, lalu disimpan di `runtime/embcache.json`.

Fakta yang berubah tahunan (pimpinan, biaya, tanggal) harus diperbarui di `knowledge.yaml`. Field
`berlaku_hingga` menghentikan FAQ atau chunk yang sudah basi tanpa menghapusnya.

## Perawatan data bertanggal

Entri yang lewat `berlaku_hingga` tidak dijawab lagi, jadi pertanyaannya ditolak. Supaya tidak terjadi
diam-diam:

- Saat service start, log mencatat `KB: ...` untuk entri yang sudah kedaluwarsa dan yang akan
  kedaluwarsa dalam 14 hari (`journalctl --user -u brone-talk | grep KB:`).
- `make kb-check` menampilkan daftar yang sama tanpa menulis apa pun. Jalankan tiap minggu.
- Saat memperbarui: ganti isi dan `berlaku_hingga`, tambahkan sumber resminya, lalu
  `python kb/build_kb.py` dan `make test`, kemudian restart service.

Data yang paling cepat basi: jadwal wisuda (dua periode sekitar sebulan sekali), acara fakultas,
kalender akademik, serta biaya dan jalur mahasiswa baru (tahunan).
