# BRONE Talk

Pipeline percakapan suara robot BRONE (asisten Universitas Brawijaya) yang berjalan di Intel NUC.
Robot bangun saat disapa **"halo brone"**, menjawab pertanyaan seputar UB dari basis pengetahuan,
dan bisa diajak ngobrol santai. Telemetri tiap tahap dikirim lewat MQTT ke Jetson.

> **Status (26 Okt 2026):** produksi di NUC sebagai `brone-talk.service` (unit `--user`).
> Wake word "halo brone" dan pemantau internet aktif sejak 26 Okt. Suara Piper fine-tune D21 aktif
> lewat drop-in (`deploy/tts_voice.sh`). Analisis lengkap: [docs/analisis-sistem.md](docs/analisis-sistem.md).

## Cara kerja

```
"halo brone" (openWakeWord, selalu mendengar; STT mati saat tidur)
  -> "Ya?" -> jendela percakapan 60 s: semua pertanyaan dijawab tanpa "halo brone"
     ("halo brone" lagi di dalam jendela = "Ya?" lagi, jendela dimulai ulang)
  -> STT: MAI-Transcribe-2 (cloud)  | cadangan faster-whisper small (lokal)
  -> mesin jawaban: jam | FAQ semantik (bge-m3) | obrolan tetap | RAG BM25+embedding
                    -> reranker -> LLM DeepSeek (cadangan LFM2.5 lokal) -> cek grounding
  -> TTS Piper -> speaker -> jendela habis -> kembali tidur
"halo brone" saat Bron bicara = interupsi (belum bekerja di robot, lihat Masalah)
```

| Bagian | Isi |
|---|---|
| Perangkat | NUC i7-1260P/30 GB, mic reSpeaker XVF3800 (+ mic wireless), speaker USB CS202, broker MQTT di Jetson |
| Layanan | `brone-talk` + llama.cpp `brone-v4-llm` (8081), `-embed` (8082), `-rerank` (8083); total RAM ~5,5 GB |
| Basis pengetahuan | 147 FAQ (672 variasi), 232 chunk RAG, 43 obrolan tetap, data UKT/IPI 2026 |
| Kode | ~3.900 baris Python, 90 tes unit (OK), konfigurasi env `BRONE_*` di `config.py` |

## Rangkuman analisis (26 Okt 2026)

**Kinerja terukur** (log era wake word, ~2,3 jam):

- 18 wake terdeteksi (skor 0,62–0,94), 21 jawaban.
- FAQ dan obrolan tetap ~60 ms. LLM cloud median 3,1 s.
- Jeda dari akhir pertanyaan sampai Bron mulai menjawab: FAQ ~2–2,5 s, LLM ~5–7 s.
- Sejak wake word, tidak ada lagi jawaban yang dipicu obrolan tim. Di mode lama, 18 dari 67 jawaban
  dipicu nama yang disebut di tengah obrolan.

**Kekuatan**

- Mesin jawaban berlapis: FAQ tanpa LLM, reranker sebagai gerbang, grounding anti-karangan.
- Ada cadangan di setiap tahap: MAI→whisper, DeepSeek→LFM2.5, mic wireless→XVF3800.
- Kode modular, teruji, dan terdokumentasi.

**Masalah utama**

Status per 26 Okt malam: T2, T4, T5 **sudah diperbaiki dan live**. T1 dicoba, tetapi belum berhasil.

| # | Masalah | Dampak | Perbaikan |
|---|---|---|---|
| T1 | Interupsi "halo brone" gagal saat Bron bicara: suara Bron sendiri menutupi mic (RMS 0,05–0,17, clipping), echo-cancel chip XVF3800 tidak dapat referensi | tinggi | **belum teratasi.** Referensi AEC ke XVF3800 lewat sink gabungan sudah dicoba: gema berkurang, tetapi chip ikut membungkam mic dan "halo brone" tetap 0/7, jadi dicabut lagi. Sisa opsi: jauhkan/kecilkan speaker dari mic, latih ulang KWS dengan gema Bron, atau interupsi non-suara (tombol/layar) |
| T2 | Batas waktu MAI 3,5 s tidak membatasi total; STT sampai 7–26 s saat internet goyah | tinggi | **diperbaiki:** satu tenggat total 3,5 s (cloud di thread, lewat tenggat langsung whisper) |
| T3 | Bron bisa mendengar audio dari speakernya sendiri (bangun/menjawab dari suara speaker) | sedang | ikut teratasi oleh echo-cancel T1 |
| T4 | Jendela masih 1 pertanyaan/8 s; diminta 1 menit percakapan + "halo brone" ulang = "Ya?" lagi; "Udah, BRONE." keliru dianggap panggilan | sedang | **diperbaiki:** jendela 60 s (`BRONE_KWS_WINDOW_SECONDS=60`), KWS ikut mendengar saat merekam di jendela, teks "…BRONE" tanpa "halo" tidak lagi memicu "Ya?" |
| T5 | Ucapan bisa ~16 s walau batasnya 8 s (sisa jendela ikut ditambahkan) | rendah | **diperbaiki:** batas dihitung sejak ucapan mulai |
| T6 | Suara D21, `lexnorm.py`, dan 2 commit lokal sudah masuk PR #1; tersisa `models` symlink ke folder lama | rendah | pindahkan model |
| T7 | ~~Dokumen lama belum memuat mode wake word~~ architecture/decisions diperbarui 26 Okt | - | selesai |
| T8 | ~~FAQ wisuda kedaluwarsa~~ diperbarui ke periode VII-VIII; log kini memperingatkan data KB basi | rendah | `make kb-check` tiap minggu |

**Urutan kerja berikutnya:** T1 (opsi perangkat keras/latih ulang) → T6 → T7/T8. Setelah itu, ukur alarm palsu
KWS di lab dan recall penutur lain. Detail bukti, angka, dan penyebab ada di
[docs/analisis-sistem.md](docs/analisis-sistem.md).

## Mulai cepat

```bash
make install                         # venv, paket, unit systemd --user (tidak dijalankan)
make test                            # tes unit dan golden
make lint                            # ruff
make ask Q="siapa dekan filkom"      # uji mesin jawaban lewat teks
make self-test                       # validasi model dan mesin tanpa mic/speaker
systemctl --user restart brone-talk  # jalankan ulang layanan di NUC
journalctl --user -u brone-talk -f   # log langsung
bash deploy/tts_voice.sh baru|lama   # suara TTS: D21 fine-tune (default) / news_tts lama
```

Rahasia (`BRONE_DEEPSEEK_API_KEY`, `BRONE_MAI_API_KEY`) hanya ada di `~/.config/brone-talk/brone.env`.
Repo ini tidak menyimpan kunci. Kosongkan `BRONE_KWS_MODEL` untuk kembali ke mode lama (nama disebut di
teks STT).

## Struktur

```
src/brone_talk/     paket: config, app, session, cli, audio/ (VAD, KWS, rekam), stt/, engine/, tts/, nlu/
kb/                 basis pengetahuan (knowledge.yaml, build_kb.py, faq/chunks/smalltalk, data/)
tests/              unit, alur, dan golden
scripts/            make_golden.py, gen_config_docs.py
deploy/             install.sh, mic_check.sh, tts_voice.sh, unit systemd, konfigurasi PipeWire
docs/               analisis, arsitektur, konfigurasi, kontrak MQTT, keputusan, deploy, pengujian
```

## Dokumentasi

- [Analisis sistem 26 Okt 2026](docs/analisis-sistem.md): kinerja, kekuatan, temuan T1–T9, prioritas.
- [Arsitektur](docs/architecture.md): alur giliran, mesin jawaban, state percakapan.
- [Konfigurasi](docs/configuration.md): semua variabel env, dibangkitkan dari `config.py`.
- [Kontrak MQTT](docs/mqtt-contract.md): topik dan payload untuk state machine dan monitor.
- [Keputusan](docs/decisions.md): alasan setiap ambang dan pilihan.
- [Deploy](docs/deployment.md): pasang, smoke test, rollback.
- [Pengujian](docs/testing.md): yang sudah dan belum dibuktikan.
- [Basis pengetahuan](docs/kb.md), [baseline produksi](docs/baseline-config.md), [fixture golden](docs/golden.md).

## Lisensi

Belum ditentukan. Sampai ada lisensi, kode ini tidak boleh dipakai ulang tanpa izin dari pemilik repo.
