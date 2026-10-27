# Analisis sistem BRONE Talk (26 Oktober 2026)

Analisis keseluruhan BRONE Talk yang berjalan di Intel NUC robot BRONE, berdasarkan kode di
`~/brone_talk` (HEAD `5156c62` + perubahan belum di-commit), layanan yang hidup, log `journalctl`
19–26 Okt 2026, uji unit, dan pengukuran akustik langsung di robot pada 26 Okt malam.

Ringkasannya ada di [README](../README.md). Dokumen ini berisi bukti dan detailnya.

---

## 1. Gambaran sistem

BRONE Talk adalah pipeline suara robot asisten Universitas Brawijaya: robot mendengar sapaan
"halo brone", menjawab pertanyaan seputar UB dari basis pengetahuan, dan bisa diajak ngobrol santai.

### Perangkat keras

| Komponen | Perangkat | Peran |
|---|---|---|
| Komputer suara | Intel NUC12WSHi7 (i7-1260P, 16 thread, 30 GB RAM), Ubuntu 24.04 | seluruh pipeline BRONE Talk |
| Mic utama | reSpeaker XVF3800 4-mic (USB), kanal ch0 olahan chip → node `brone-mic-xvf` | KWS, VAD, STT |
| Mic cadangan | mic wireless + RNNoise → node `brone-mic-wireless` (didahulukan bila tersambung) | VAD, STT |
| Speaker | kartu USB CS202 (volume PipeWire 0,72) | suara Bron |
| Jetson Orin Nano | broker MQTT (mosquitto) + `brone_jetson_manager` | menerima telemetri / ekspresi |

### Layanan (systemd `--user`, linger aktif)

| Unit | Isi | RAM |
|---|---|---|
| `brone-talk.service` | `brone-talk run` (Python 3.12, paket `src/brone_talk`) | ~1,0 GB |
| `brone-v4-llm.service` | llama.cpp LFM2.5-2.6B Q4_K_M, port 8081 (mode local eksplisit) | ~3,0 GB |
| `brone-v4-embed.service` | llama.cpp bge-m3 F16, port 8082 (embedding) | ~1,5 GB |
| `brone-v4-rerank.service` | llama.cpp bge-reranker-v2-m3 Q8_0, port 8083 | ~0,9 GB |

Total pemakaian RAM sistem 5,5 GB dari 30 GB. Disk 39%. CPU idle tinggi. Sumber daya NUC
**bukan** hambatan.

### Alur satu interaksi (mode wake word, produksi sejak 26 Okt 2026)

```
TIDUR: openWakeWord "halo brone" (halo_brone.onnx, thr 0,5) mendengar terus, STT mati
  -> terdeteksi -> tunggu jeda 0,3 s -> TTS "Ya?"
  -> rekam 1 ucapan (Silero VAD, mulai dalam 8 s) -> STT MAI-Transcribe-2 (cloud, Voice Live)
       gagal/offline -> faster-whisper small (lokal)
  -> filter sampah STT -> keputusan (stop / "Ya?" lagi / jawab)
  -> mesin jawaban:
       jam sistem | FAQ semantik (bge-m3) + pengaman | obrolan tetap (smalltalk.yaml)
       | perintah gerak | RAG hybrid BM25+embedding -> reranker -> LLM DeepSeek (5 s)
       -> cek grounding (tolak jika LLM gagal) | persona obrolan LLM
  -> TTS Piper (suara D21 fine-tune BRONE sejak 26 Okt 19:35) -> speaker
  -> bila Bron bertanya balik: 1 tanggapan tanpa wake word dalam 8 s, lainnya kembali TIDUR
Selama berpikir/berbicara: thread pemantau KWS kedua = interupsi "halo brone" (lihat temuan T1)
Pemantau internet: rute default + ping Voice Live tiap 5 s -> putus = STT langsung whisper
Telemetri tiap tahap -> MQTT Jetson (192.168.100.2:1883)
```

### Basis pengetahuan (`kb/`)

| Berkas | Isi |
|---|---|
| `faq.jsonl` | 147 FAQ, 672 variasi pertanyaan (60 FAQ dibangkitkan otomatis), dengan `berlaku_hingga`, entitas, dan guard |
| `chunks.jsonl` | 232 potongan dokumen untuk RAG |
| `knowledge.yaml` | sumber KB (95 KB teks), dibangun dengan `build_kb.py` |
| `smalltalk.yaml` | 43 entri obrolan tetap (jawaban acak, ekspresi, tanya-balik) |
| `data/` | `ipi_2026.csv`, `ukt_mandiri_2026.csv` |

FAQ dan chunk yang lewat `berlaku_hingga` otomatis tidak dipakai. Saat ini satu FAQ kedaluwarsa:
`wisuda-berikutnya` (berlaku hingga 11 Okt 2026), sehingga pertanyaan wisuda selalu ditolak.

### Kode

3.900 baris Python di 11 sub-modul. Modul terbesar: `app.py` (403), `stt/transcriber.py` (476),
`tts/normalize.py` (361), `engine/answer.py` (349), `audio/capture.py` (308), `config.py` (281).
Semua konfigurasi dibaca sekali dari env (`config.py`, ~60 variabel `BRONE_*`), rahasia hanya di
`~/.config/brone-talk/brone.env`.

---

## 2. Kinerja terukur (log 26 Okt 17:29–19:50, era wake word)

Entri 19:22–19:25 berasal dari uji akustik saya (klip "halo brone" dan TTS diputar lewat speaker
robot), jadi tidak dihitung sebagai interaksi nyata.

| Metrik | Nilai |
|---|---|
| Wake "halo brone" terdeteksi | 18 kali (skor 0,62–0,94), 1 di antaranya tanpa pertanyaan sesudahnya (19:03) |
| Jawaban | 21: FAQ 7, obrolan tetap 6, obrolan LLM 6, LLM RAG 1, tolak 1 |
| Latensi mesin jawaban | FAQ/obrolan tetap ~60 ms; LLM cloud median 3,1 s (p90 4,6 s) |
| STT MAI (normal) | ~0,7–2,0 s per ucapan |
| STT saat MAI gagal | **7–26 s** (lihat T2) |
| Sintesis TTS | median 0,64 s, p90 1,1 s; jawaban terpanjang 31 s audio |
| Wake → "Ya?" terdengar | ~1,1 s |
| Akhir pertanyaan → mulai jawaban | FAQ ~2–2,5 s; LLM ~5–7 s |
| Interupsi "halo brone" saat Bron bicara | **0 berhasil** (lihat T1) |
| Internet/MAI putus | 8 kali dalam ~2,3 jam |

Perbandingan dengan mode lama (nama di teks STT, 19–20 Okt): saat itu 18 dari 67 jawaban dipicu
nama "Brone" di tengah obrolan tim. Sejak wake word, **tidak ada** jawaban yang dipicu obrolan sekitar,
karena STT hanya berjalan setelah "halo brone".

Uji otomatis: `python -m unittest discover -s tests` → **83 tes OK** (5 tes golden dilewati karena
fixture tidak ada di NUC). `ruff check` → 1 pelanggaran (E501, `tts/lexnorm.py:43`, berkas baru).

---

## 3. Kekuatan

1. **Mesin jawaban berlapis dan hemat.** Pertanyaan yang ada di FAQ dijawab ~60 ms tanpa LLM.
   LLM hanya dipanggil bila reranker yakin ada konteks UB. Pengaman FAQ (peran, fakultas, tahun, kata
   tanya) dan cek grounding (angka/nama harus ada di konteks) mencegah jawaban karangan.
2. **Cadangan STT dan mikrofon.** MAI → whisper lokal, mic wireless → XVF3800.
   Pemantau internet memindahkan STT ke whisper sebelum ucapan berikutnya.
3. **Wake word menutup masalah lama.** Jawaban salah karena nama disebut di tengah obrolan hilang,
   dan biaya MAI turun karena audio hanya dikirim setelah dibangunkan. Uji offline model: 39/39 klip
   asli (38/39 saat diuji di NUC, held-out 9/9), 0 alarm palsu pada 285 klip pertanyaan.
4. **Kode rapi dan teruji.** Paket modular, satu sumber konfigurasi, 83 tes unit dan alur dengan
   komponen palsu, dokumentasi keputusan (`docs/decisions.md`), skrip deploy dan rollback.
5. **Operasional mandiri.** Unit `--user` dengan linger, `mic_check.sh` sebelum start, restart otomatis,
   telemetri MQTT untuk ekspresi dan monitor.

---

## 4. Temuan dan risiko

Urut dari dampak terbesar.

### T1. Interupsi "halo brone" tidak bekerja saat Bron berbicara (tinggi)

**Gejala.** Uji suara 26 Okt: 12 wake saat robot diam terdeteksi, tetapi tidak satu pun saat Bron
sedang bicara. Pemantau interupsi berjalan tanpa error; skor KWS-nya tetap ≈ 0,00.

**Penyebab (diukur).** Speaker CS202 keras dan dekat mic. Suara Bron sendiri masuk ke mic dengan RMS
0,05–0,17 dan **clipping** (puncak 1,0), yaitu 4–13× derau ruangan. Jack output XVF3800 tidak
tersambung ke speaker, jadi echo-cancel bawaan chip tidak punya referensi.

| Uji (klip "halo brone" asli) | Hasil |
|---|---|
| Klip lewat speaker, tanpa TTS | 7/7 terdeteksi |
| Klip di atas TTS, mic ch0 | 0/7 pada semua level (0, −6, −12 dB) |
| Menurunkan threshold | tidak menolong: skor di bawah gema ≈ 0,00–0,01 |
| AEC WebRTC PipeWire di ch0 | gema turun hanya 1–2 dB (ch0 sudah diolah chip, nonlinier) |
| AEC WebRTC di mic mentah (ch2) | gema turun 15–25 dB, tetapi double-talk tetap 0–3/7 |
| **TTS juga dikirim ke sink USB XVF3800 sebagai referensi AEC chip** | gema ch0 turun ke ≈ 0,00–0,02 setelah ~12 s konvergensi |

**Percobaan 26 Okt malam (setelah analisis).** Sink gabungan CS202 + output USB XVF3800 dipasang
(`combine-stream` PipeWire, lalu `module-combine-sink` pulse) dan diuji double-talk 2× masing-masing:
gema berkurang di sebagian waktu, tetapi chip juga membungkam mic total (RMS 0,001) di jeda-jeda,
dan "halo brone" dari speaker tetap **0/7** terdeteksi. Sink itu dicabut lagi, dan audio kembali ke
CS202 langsung, supaya awal pertanyaan setelah Bron bicara tidak ikut terbungkam. Opsi yang tersisa:
jauhkan atau kecilkan speaker dari mic (hilangkan clipping), latih ulang KWS dengan augmentasi gema
Bron, atau interupsi non-suara (tombol/layar sentuh).

**Rekomendasi awal (sebelum percobaan).** Jadikan XVF3800 penerima referensi suara Bron. Caranya: PipeWire `combine-sink`
yang mengirim TTS ke CS202 dan ke output USB XVF3800, atau sambungkan amplifier speaker ke jack
XVF3800. Referensi perlu tetap mengalir agar AEC chip tidak harus konvergen ulang setiap jawaban.
Uji double-talk berikutnya belum dijalankan. Bila masih kurang, latih ulang KWS dengan augmentasi
gema Bron dan turunkan volume/gain agar mic tidak clipping.

### T2. Batas waktu MAI 3,5 s tidak benar-benar membatasi (tinggi)

**Gejala.** Log menulis "timed out in 3.5s", tetapi STT memakan 7,3 / 7,5 / 11,0 / 13,9 / 22,8 / 26,0 s
pada malam 26 Okt. Penanya menunggu 10–30 s tanpa respons.

**Penyebab (dari kode `stt/transcriber.py`).** Batas 3,5 s hanya berlaku per fase, bukan total:
1. `_transcribe_voicelive` menunggu `_vl_lock` tanpa batas. Bila probe pemantau internet sedang membuka
   koneksi baru (DNS + TLS + `session.update`), ucapan menunggu probe selesai dulu.
2. Bila koneksi belum siap, `_voicelive_open(3,5 s)` dijalankan (ditambah DNS 1,5 s), lalu mengirim audio
   tanpa batas waktu, baru kemudian menunggu transkrip 3,5 s lagi.

**Rekomendasi.** Satu tenggat total untuk upaya MAI, misalnya 3,5 s sejak ucapan selesai. Kunci yang
sedang dipegang probe = langsung whisper. Pengiriman dijalankan di thread yang ditinggalkan bila tenggat
lewat. Target latensi STT terburuk ≈ 3,5 s + whisper (~1,5 s).

### T3. Bron bisa mendengar dirinya sendiri (sedang)

- Saat uji akustik 19:22, klip "halo brone" yang diputar dari speaker robot membangunkan Bron. Penjaga
  suara-sendiri hanya memeriksa teks kalimat yang sedang diucapkan Bron, bukan audio lain dari speaker.
- Pada jendela "Ya?", STT mentranskripsi audio TTS yang diputar ("Fakultas di Universitas Brawijaya
  adalah …") lalu menjawabnya lewat LLM. Dalam pemakaian normal ini terjadi bila ada pengeras suara
  lain di dekat robot, atau gema jawaban Bron tertangkap di jendela tanya-balik.
- Echo-cancel dari T1 juga menyelesaikan sebagian besar masalah ini.

### T4. Perilaku jendela pertanyaan belum sesuai keinginan pengguna (sedang)

- Sekarang: satu pertanyaan per "halo brone", jendela 8 s (`BRONE_KWS_WINDOW_SECONDS=8`).
- Diminta (26 Okt): setelah "Ya?", **semua** pertanyaan dalam 1 menit dijawab tanpa "halo brone";
  "halo brone" di dalam menit itu dijawab "Ya?" lagi dan jendela dimulai ulang. **Belum dibangun.**
- "Halo brone" di dalam jendela kini dikenali dari teks STT (`decide` → CONFIRM), bukan dari model KWS.
  Akibatnya, ucapan yang berakhir dengan nama juga dianggap panggilan: "Udah, BRONE." (jawaban atas
  tanya-balik, 19:06) dibalas "Ya?". Seharusnya KWS juga dipasang pada aliran rekaman di jendela, dan
  CONFIRM dari teks hanya untuk "halo … bron".

### T5. Panjang ucapan maksimum tidak konsisten (rendah)

`BRONE_MAX_UTTERANCE_SECONDS=8`, tetapi `Recorder._endpointer` menambahkan sisa jendela tunggu
(`window_limit = max_duration + start_s`). Ucapan yang langsung dimulai setelah "Ya?" bisa sampai
~16 s (tercatat klip 15,97 s). Ucapan sepanjang itu biasanya obrolan atau gema, bukan pertanyaan, dan
memperlama STT.

### T6. Kode yang berjalan tidak sama dengan commit mana pun (sedang, proses)

- Layanan di-restart 26 Okt 19:35 dengan suara Piper baru **D21** lewat drop-in
  `~/.config/systemd/user/brone-talk.service.d/suara.conf` (`deploy/tts_voice.sh baru|lama`).
- `tts/lexnorm.py`, `tests/test_tts_lexnorm.py`, `deploy/tts_voice.sh`, serta perubahan pada
  `config.py`, `tts/normalize.py`, dan `tts/speaker.py` **belum di-commit**. Rollback lewat git tidak
  akan mengembalikan kondisi ini.
- Cabang `main` di NUC **unggul 2 commit** dari GitHub (`1bc34d0` KWS, `5156c62` pemantau internet),
  belum di-push.
- `models` adalah symlink ke folder lama `~/brone-talk/models`. Menghapus folder lama mematikan layanan.
- Unit llama.cpp masih bernama `brone-v4-*` (rencana ganti nama ditunda sejak refactor).

### T7. Dokumentasi tertinggal (rendah)

- README lama menyebut panggilan lewat kata "Brone" dan status "cutover belum dilakukan". Keduanya sudah
  tidak berlaku: wake word aktif dan repo ini sudah produksi.
- `docs/architecture.md`: diagram state belum memuat mode wake word dan interupsi.
- `docs/decisions.md` saling bertentangan: "panggilan di mana pun dalam kalimat" dan
  "`BRONE_AWAIT_SECONDS` 15 detik" berbeda dengan keputusan 20 Okt (nama di awal/akhir, AWAIT 0).
  Juga "kode `speaker_lock/` masih ada", padahal sudah dihapus di `82b0efe`.

### T8. Basis pengetahuan perlu perawatan berkala (rendah)

FAQ wisuda kedaluwarsa sejak 11 Okt, sehingga semua pertanyaan wisuda ditolak. Data dengan tanggal
(wisuda, pendaftaran, UKT) perlu jadwal pembaruan dan pemilik yang jelas.

### T9. Hal yang belum terukur

- Alarm palsu KWS di keramaian lab per jam (baru 2,3 jam log; 1 wake tanpa pertanyaan).
- Recall KWS lintas penutur. Data uji held-out hanya dari satu orang.
- Kualitas suara D21 dibanding news_tts (belum ada uji dengar/ASR-back).
- 3 kali "MQTT terputus" ke broker Jetson dalam seminggu. Dampaknya ke ekspresi belum dicek.

---

## 5. Prioritas perbaikan

| # | Pekerjaan | Mengatasi | Usaha |
|---|---|---|---|
| 1 | Interupsi: speaker dijauhkan/dikecilkan, latih ulang KWS dengan gema, atau tombol (referensi AEC XVF3800 sudah dicoba, gagal) | T1, T3 | sedang–besar |
| 2 | ~~Tenggat total MAI~~ **selesai 26 Okt malam** (`_mai_within_deadline`) | T2 | kecil |
| 3 | ~~Jendela percakapan 60 s + KWS di jendela + CONFIRM hanya "halo bron"~~ **selesai 26 Okt malam** | T4 | sedang |
| 4 | ~~Commit/push perubahan suara D21 + 2 commit lokal; perbaiki E501~~ **PR #1, 26 Okt** | T6 | kecil |
| 5 | ~~Batasi ucapan sejak ucapan mulai~~ **selesai 26 Okt malam** (`Endpointer.speech_limit`) | T5 | kecil |
| 6 | ~~Perbarui README/architecture/decisions; jadwal pembaruan KB~~ **selesai 26 Okt** (peringatan KB basi, `make kb-check`, FAQ wisuda VII-VIII) | T7, T8 | kecil |
| 7 | Ukur alarm palsu KWS di lab + rekam penutur lain; latih ulang bila perlu | T9, T1 | besar |
