# Keputusan dan alasannya

Setiap keputusan di sini diambil dari perilaku produksi dan pengukuran yang sudah dilakukan di
NUC. Kalau mau mengubahnya, ukur ulang di hardware yang sama dulu.

## Audio

- **Endpointing satu implementasi.** Jalur ALSA (dengan arah suara) dan jalur sounddevice memakai
  `audio/endpointing.py` yang sama. Dua loop lama identik secara logika; hanya sumber sampel dan
  DOA yang berbeda. Jalur sounddevice kini juga mengirim `doa=None` di telemetri.
- **Gate energi.** Klip dengan RMS di bawah `BRONE_MIN_UTTERANCE_RMS` (0,005; sebelum 20 Nov 2026: 0,030) dibuang. Gate sengaja dibuat sangat rendah karena STT sekarang kuat untuk suara pelan; penyaringan obrolan latar dipindah ke LLM (jenis "abaikan"). Derau ruangan
  NUC biasanya 0,01 sampai 0,05.
- **Mic dipilih tiap rekaman.** Urutan `BRONE_INPUT_PIPEWIRE_NODES` adalah prioritas. Mic wireless
  dengan RNNoise didahulukan. XVF3800 tidak diberi RNNoise karena chip-nya sudah meredam derau dan
  double-NS merusak STT.
- **Derau tidak diredam di kode.** `BRONE_NOISE_REDUCE` tetap default `0`.

## Kunci penanya (dinonaktifkan 19 Okt 2026)

Kunci penanya (sidik suara CAM++ dan arah SRP-PHAT, 12 Okt 2026) **dinonaktifkan atas permintaan
pengguna**. Pertanyaan lanjutan tanpa nama membuat robot menjawab suara orang lain yang kebetulan
cocok dengan penanya sebelumnya, dan tidak sesuai harapan. Kode `speaker_lock/` serta env `BRONE_LOCK`
dan `BRONE_DOA_*` sudah dihapus (commit `82b0efe`, 20 Okt 2026).

## Panggilan lewat teks (mode cadangan, `BRONE_KWS_MODEL` kosong)

- Nama dihitung sebagai panggilan hanya di **awal kalimat** atau **akhir ucapan** (20 Okt 2026).
  Nama di tengah obrolan ("Kalau udah bilang ya, Kak BRONE, ...") tidak membangunkan Bron.
- `BRONE_AWAIT_SECONDS=0`: nama saja dijawab dengan ajakan menyebut pertanyaan, tanpa "Ya?" dan
  tanpa pertanyaan lanjutan tanpa nama.
- Produksi memakai wake word (bagian berikut); mode ini hanya cadangan.

## Wake word "halo brone" (KWS, disepakati 23 Okt 2026)

**Default sejak 26 Okt 2026**: `BRONE_KWS_MODEL` bawaan `models/kws/halo_brone.onnx` (openWakeWord,
dilatih di AI-Center 25 Okt 2026), jadi wake word, jendela 60 s, dan interupsi aktif tanpa env.
Isi kosong (`BRONE_KWS_MODEL=`) = mode lama (nama disebut di teks STT). Model tidak ada di git
(`models` di-gitignore); service menolak start bila file model tidak ada.

- **Bangun hanya lewat model KWS**, tidak lewat teks. STT (MAI/whisper) baru jalan setelah bangun,
  jadi biaya MAI turun dan obrolan sekitar tidak pernah ditranskripsi.
- **Jendela percakapan 60 s (diminta pengguna 26 Okt 2026, menggantikan "satu pertanyaan per bangun").**
  Alur: tidur -> "halo brone" -> tunggu jeda bicara -> "Ya?" -> semua pertanyaan dalam
  `BRONE_KWS_WINDOW_SECONDS` (60 s sejak "Ya?") dijawab tanpa menyebut nama -> jendela habis -> tidur.
  Riwayat (pertanyaan sebelumnya) tetap dipakai di dalam jendela dan dikosongkan setiap "Ya?".
  Tanya-balik Bron memberi minimal `BRONE_SOCIAL_AWAIT_SECONDS` walau sisa jendela lebih pendek.
- **"halo brone" di dalam jendela = "Ya?" lagi**, dan jendela dimulai ulang. Model KWS ikut mendengar
  aliran rekaman di jendela (`audio.kws.watch`). Cadangan dari teks hanya untuk "halo/alo/hello ... bron".
  Nama di ujung tanpa sapaan ("Udah, BRONE.") ditanggapi sebagai ucapan biasa, bukan panggilan ulang.
- **Panjang ucapan dihitung sejak ucapan mulai** (`BRONE_MAX_UTTERANCE_SECONDS`), bukan sejak rekaman
  dibuka. Sebelumnya sisa jendela ikut ditambahkan, sehingga ucapan bisa ~16 s.
  Ucapan sebelum "Ya?" dibuang (tidak ada "halo brone, <pertanyaan>" sekali napas). Tidak bicara
  dalam jendela = tidur lagi tanpa suara.
- **"halo brone" juga interupsi** (`BRONE_KWS_INTERRUPT`, menggantikan rencana "Brone stop"):
  selama berpikir/berbicara, pemantau KWS di thread lain menghentikan TTS dan berhenti menunggu
  LLM, lalu langsung "Ya?". Thread mesin jawaban tidak dibunuh; hasilnya dibuang dan panggilan
  mesin berikutnya menunggu thread itu selesai.
- **Penjaga suara sendiri:** deteksi diabaikan bila kalimat yang sedang diucapkan Bron memuat
  "halo ... bron". Persona tidak dilarang mengucapkannya (pengguna menilai kecil kemungkinannya).
- **Tanya-balik tetap tanpa wake word:** tanggapan atas pertanyaan balik Bron diterima dalam
  `BRONE_SOCIAL_AWAIT_SECONDS`.
- **Pemantau membuka mic sendiri** (arecord kedua lewat PipeWire), bukan berbagi aliran dengan
  perekam. Lebih sederhana, dan PipeWire mengizinkan beberapa klien rekam pada node yang sama.
- **openwakeword dipasang `--no-deps`:** paketnya menuntut tflite-runtime di Linux yang tidak
  punya wheel py3.12. BRONE hanya memakai ONNX.

## STT

- **Tenggat MAI total (26 Okt 2026).** `BRONE_MAI_TIMEOUT_SECONDS` (3,5 s) kini berlaku untuk
  seluruh upaya cloud: menunggu kunci probe, membuka koneksi, kirim, dan menunggu transkrip. Panggilan
  berjalan di thread; lewat tenggat, klip langsung ke whisper. Sebelumnya batasnya per fase, dan STT
  tercatat 7-26 s saat internet goyah.
- **MAI-Transcribe-2 utama, whisper cadangan.** Whisper dimuat sejak awal dan dipakai saat internet
  lambat atau Azure gagal. Batas tunggu MAI 3,5 detik; lewat itu, cloud dihentikan 30 detik.
- **Pemantau internet (pilihan A, disepakati 22 Okt 2026, aktif 26 Okt 2026).** Penjaga latar
  mengecek rute default OS (`/proc/net/route`) dan ping koneksi Voice Live tiap 5 detik
  (`BRONE_MAI_KEEPALIVE_SECONDS`, balasan maks 2 detik). Gagal -> cloud ditandai offline, ucapan
  berikutnya langsung ke whisper tanpa menunggu 3,5 detik. Probe tetap jalan; begitu ping/koneksi
  baru berhasil, STT kembali ke MAI dan sisa istirahat 30 detik dibatalkan. Istirahat 401/402/403
  (kunci/saldo) tidak di-probe. Biaya ~2,6 MB/hari; Azure menagih per audio, ping gratis. Bukan
  dipilih: whisper paralel (B), C, D.
- **Model whisper: `faster-whisper-small`** (produksi). README lama menyebut large-v3-turbo, tetapi
  drop-in aktif memakai small.
- **Threads = 4.** Jumlah P-core fisik i7-1260P. Diukur: 12 threads ~1,8 detik per klip, 16 threads
  ~2,0 detik, 4 threads ~1,5 detik. Jangan naikkan tanpa mengukur ulang.
- **Beam = 5 (produksi).** Kode lama default 3 (identik teks, 8% lebih cepat). Produksi memakai 5;
  1 dan 2 terbukti membuang kata ("apa kabar" menjadi "apakah").
- **Prompt baku tanpa kosakata KB.** Kode lama bisa menambahkan judul KB ke prompt, tetapi hanya
  bila `BRONE_WHISPER_TOPICS=1`. Produksi = 0, jadi prompt sama persis dengan prompt baku. Repo
  baru tidak membaca `knowledge.txt` sama sekali. Ini menyederhanakan dan tidak mengubah perilaku.
- **Filter sampah** (`stt/filters.py`) dibaca bersama kosakata prompt. Fixture golden dibuat
  tanpa kosakata, jadi tes golden untuk `garbage_reason` belum membuktikan filter ini (lihat
  `docs/golden.md`).

## Mesin jawaban

- **FAQ dulu, RAG kemudian, LLM terakhir.** Pertanyaan yang sudah ada di FAQ tidak pernah sampai ke
  LLM. Bila kemiripan chunk teratas di bawah `BRONE_CHUNK_MIN_SIM`, robot menolak tanpa LLM.
- **Pengaman FAQ** (`entities.py`, `FaqIndex._guard`). Menghindari jawaban yang salah tetapi mirip,
  misalnya "wakil rektor" dijawab FAQ "rektor", atau "rektor tahun 2010" dijawab rektor sekarang.
- **Grounding.** Angka dan nama di jawaban LLM harus ada di konteks atau pertanyaan. Bila tidak, jawaban
  diganti penolakan. Ini mencegah LLM mengarang fakta yang tidak ada di KB.
- **Data bertanggal tidak boleh mati diam-diam.** FAQ/chunk yang lewat `berlaku_hingga` tidak dipakai
  (pertanyaannya ditolak). FAQ wisuda sempat basi 11-26 Okt 2026 tanpa ada yang tahu, jadi kini
  service mencatat peringatan `KB: ...` saat start (basi dan basi dalam 14 hari), dan `make kb-check`
  menampilkan daftar yang sama.
- **Tidak ada semantic cache.** Jawaban LLM tidak disimpan untuk dipakai ulang, supaya tidak ada
  jawaban basi.
- **Cloud 5 detik** (produksi). Kode lama default 8 detik. Lewat batas, pindah ke LLM lokal.
- **Riwayat tidak dipakai.** `BRONE_HISTORY_MESSAGES` ada di unit lama tetapi mesin v4 tidak memakai
  riwayat percakapan; hanya pertanyaan sebelumnya yang dipakai untuk pertanyaan pendek.

## Percakapan (session.py)

- **Panggilan di awal kalimat atau akhir ucapan (mode teks).** "Brone, UB di mana?" dan "UB di mana,
  Brone?" menjawab; nama di tengah kalimat tidak. "BRONE FILKOM" adalah nama proyek, bukan panggilan.
- **"Ya?" hanya sekali** per penantian. Ucapan tidak jelas saat menunggu juga hanya diminta diulang
  sekali per giliran, supaya derau tidak membuat robot terus bicara.
- **Riwayat dibersihkan** setiap kali nama disebut di luar penantian.

## Yang dihapus dari versi lama

- RAG v1, v2, v3 (RagLite*), `knowledge.txt`, dan semua jalur Ollama/DeepSeek di kelas `Conversation`.
  Mesin v4 menggantikannya seluruhnya.
- Soniox STT dan TTS (Soniox TTS dihapus 14 Okt 2026). Google STT juga tidak dipakai produksi.
- `BRONE_ENGINE=legacy`. Tidak ada lagi jalur lama di repo ini.
- Skrip pengembangan lama (`_bench_*.py`, berkas `.bak*`). Riwayat sekarang di git.

## Nama dan deploy

- **Unit tetap `brone-talk.service`.** `state_machine_node.py` memanggil nama itu lewat systemctl,
  jadi nama tidak diubah.
- **Nama unit llama.cpp (`brone-v4-*`) belum diganti.** Layanan itu sedang dipakai. Penggantian
  ke `brone-llm|embed|rerank` ditunda sampai cutover, supaya tidak memutus layanan yang hidup.
- **Serial perangkat di konfigurasi PipeWire disensor** di salinan repo. Nilai aslinya tetap di
  `~/.config/pipewire/pipewire.conf.d/` di NUC.
- **Rahasia tidak masuk repo.** Kunci DeepSeek dan MAI dibaca dari
  `~/.config/brone-talk/brone.env`. `EnvironmentFile` menimpa `Environment=` di unit, jadi file itu
  hanya berisi rahasia.

## Pembaruan 12 November 2026

- **Reranker dihapus, diganti gerbang kemiripan chunk.** `BRONE_CHUNK_MIN_SIM` = 0,60 (kosinus bge-m3 chunk
  teratas). Pada Eval-60 (42 pertanyaan berjawaban) dan 63 pertanyaan di luar cakupan: reranker meloloskan
  95,2% / 22,2%, gerbang dense 0,60 meloloskan 92,9% / 15,9%. Menghemat sekitar 1,5 detik per giliran RAG,
  satu server llama.cpp (~0,9 GB RAM) dan ~350 CPU-detik per 145 query. Harga yang dibayar: Hit@1 chunk
  0,857 vs 0,905 (konteks 4 chunk, jadi chunk benar tetap masuk: Hit@4 sama 0,952). Laporan: Google Doc tab
  "Perubahan 12 Nov".
- **`OPENBLAS_NUM_THREADS=1` di unit** (thread default membuang ~1,3 detik CPU per query).
- **Kalimat sapaan tidak boleh ada di dua sumber.** FAQ `sapaan` hanya menyisakan "Halo Bron, apa kabar?" dan
  "hai apa kabar kamu"; sapaan umum ditangani `smalltalk.yaml`. FAQ: 141 FAQ, 794 kalimat.

## Konteks percakapan satu giliran (12 November 2026)

- **Pertanyaan lanjutan dilengkapi sebelum masuk ke FAQ/RAG** (`engine/context.py`, `FollowUpContext`). Yang disimpan hanya
  dua slot dari satu giliran terakhir: entitas (fakultas) dan atribut (lokasi, dekan, prodi, sejarah, ...), dalam
  bentuk yang sudah dilengkapi, sehingga rantai ("kalau FT?" lalu "dekannya?") tetap jalan.
- **Aturan:** hanya entitas ("kalau FT?") -> warisi atribut lama; hanya atribut fakultas dengan penanda lanjutan
  ("dekannya?", "di sana", "dan dekan?") -> warisi entitas lama; selainnya tidak mewarisi. Sisa kata harus kata pengisi.
  Pertanyaan tingkat universitas (UB, rektor, moto) menghapus slot. Slot kedaluwarsa 90 detik dan dibersihkan saat
  `AnswerEngine.reset()`. Konfigurasi: `BRONE_FOLLOWUP`, `BRONE_FOLLOWUP_TTL_SECONDS`, `BRONE_FOLLOWUP_MAX_WORDS`.
- **Hasil lengkap memakai pertanyaan utama FAQ yang bersangkutan** (kisi atribut x fakultas dari `faq.jsonl`), supaya tidak
  jatuh ke FAQ tetangga. Prinsip: bila ragu, tidak mewarisi (sama dengan perilaku lama).
- Laporan dan angka evaluasi: Google Doc tab "Konteks Percakapan 12 Nov".


## Akhir sesi "selesai brone" toleran salah dengar STT (20 Nov 2026)

Regex lama (`nlu/wake.py`) hanya menerima ejaan "selesai"/"selasai"/"sosai" dan nama BRONE persis, sehingga
107/200 klip `runtime/uji-end-wake` (derau, jarak jauh) tidak mengakhiri sesi. Sekarang `is_session_end()`
mencocokkan token: kata-akhir (selesai, usai, atau kata berakhiran -ai yang bukan kata umum seperti
sampai/pakai/mulai, atau jarak edit <=2 dari "selesai") bersebelahan dengan nama (termasuk Broning/Bronya/Kebrone),
didahului hanya kata pengisi, dengan gema istilah prompt (UB, Universitas Brawijaya, ...) diabaikan dan kata tanya
atau kata ganti (kapan, saya, belum, ...) menolak. Hasil: 143/200 terdeteksi, false positive tetap 1/188
(klip "stop" yang ditranskrip "Selasai, BRONE!"), 0 dari 382 ucapan nyata `turns.jsonl`. Sisa yang gagal sebagian
besar STT kehilangan kata "selesai" sama sekali ("Terima kasih.", "UB, UB, UB"). Frasa akhir juga dilewatkan dari
filter "frasa berulang" STT.

## Satu pemantau internet untuk STT dan LLM (20 Nov 2026)

Saat offline, probe MAI hanya tiap 5 s (`keepalive_s`) dan LLM cloud tidak memakai pemantau sama sekali: tiap
pertanyaan mencoba DeepSeek sampai timeout/DNS menggantung (11-41 s di journal 20 Nov) baru pindah ke lokal.
Sekarang penjaga mem-probe tiap `BRONE_MAI_OFFLINE_PROBE_SECONDS` (1 s) saat offline, dan mesin jawaban (mode auto)
membaca status yang sama: offline -> langsung LLM lokal; gagal cloud -> lokal sampai probe berikutnya sukses;
probe sukses -> STT dan LLM kembali ke cloud bersamaan. Penyebab putus di lapangan: DNS router 192.168.1.1
sering gagal ~10 s berturut-turut dan sinyal WiFi ROBOTIIK lemah (-68 dBm).

## LLM lokal LFM2.5 dihapus (24 Nov 2026)

Cadangan LFM2.5-2.6B (llama.cpp, port 8081) dibuang: jawaban hanya dari FAQ/obrolan tetap dan DeepSeek cloud.
Alasannya menyamakan dengan `origin/main` (commit "remove local model support") dan membebaskan ~3 GB RAM serta
1,6 GB disk di NUC. Akibatnya `BRONE_MODE=local` ditolak (`ValueError`; mode sah: auto, cloud, faq), env
`BRONE_LOCAL_*` tidak berlaku, dan unit `brone-v4-llm` dihapus (unit `brone-v4-rerank` ikut dibersihkan). Saat
cloud gagal atau offline, pertanyaan yang butuh LLM ditolak sopan dan tidak lagi dijawab model lokal; pertanyaan
FAQ dan obrolan tetap tidak terpengaruh. Catatan 20 Nov di atas ("offline -> langsung LLM lokal") tidak berlaku lagi.
