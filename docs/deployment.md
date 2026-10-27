# Deploy di Intel NUC

Repo ini belum menggantikan layanan produksi. Dokumen ini menjelaskan cara memasang dan
menguji di samping versi yang sedang jalan, lalu langkah cutover.

## Prasyarat

- `~/brone-talk-v4` dan `~/brone-talk` tetap ada (rujukan dan jalan mundur).
- Layanan llama.cpp hidup: `brone-v4-embed` (8082), `brone-v4-rerank` (8083). `brone-v4-llm` (8081) hanya diperlukan untuk mode local eksplisit.
- Model: `~/brone-talk/models` (whisper, Piper, Silero VAD, CAM++). Model tidak ada di git.
- Konfigurasi PipeWire `10-brone-xvf3800.conf` dan `20-brone-rnnoise-wireless.conf` sudah terpasang.
  Salinan repo ada di `deploy/pipewire/`, dengan serial yang disensor. Jangan timpa yang di NUC.

## Pasang (tidak menjalankan apa pun)

```bash
cd ~/brone_talk
make install                       # venv, paket, symlink models, unit systemd --user
```

`deploy/install.sh` tidak mengaktifkan atau memulai layanan. Setelah pasang, isi rahasia:

```bash
$EDITOR ~/.config/brone-talk/brone.env   # BRONE_DEEPSEEK_API_KEY, BRONE_MAI_API_KEY
```

Berkas itu dibuat dengan `chmod 600` dan hanya berisi rahasia.

Konfigurasi PipeWire di `deploy/pipewire/` memakai placeholder yang harus diganti di NUC sebelum
disalin ke `~/.config/pipewire/pipewire.conf.d/`:

- `<SERIAL-XVF3800>`, `<SERIAL-RECEIVER-WIRELESS>`: serial perangkat. Ambil dari konfigurasi yang
  sudah berjalan di NUC (`target.object`), jangan dari repo.
- `<HOME>`: home user NUC (mis. `/home/<user>`), untuk plugin RNNoise.

## Uji tanpa mengganggu robot

Versi produksi dan repo baru tidak boleh jalan bersamaan (mic dan speaker satu). Urutannya:

1. Hentikan versi yang aktif: `~/bin/brone-pakai stop`.
2. `make self-test`. Harus mencetak `SELF_TEST=ok`.
3. `make ask Q="siapa dekan filkom"`. Harus menjawab dari FAQ.
4. `brone-talk mic-test 5` (lewat `.venv/bin/brone-talk`). Bicara 5 kali, periksa skor STT.

## Smoke test perangkat keras (sebelum cutover)

Jalankan repo baru sebagai layanan terpisah:

```bash
systemctl --user stop brone-talk brone-talk-v4     # pastikan mic dan speaker bebas
systemctl --user start brone-talk-next
journalctl --user -u brone-talk-next -f
```

Unit uji bernama `brone-talk-next` supaya `brone-talk` (produksi) tidak berubah sebelum cutover.

Periksa berurutan:

- [ ] Warmup selesai, `voice_status` = `ready`.
- [ ] Ucapkan "Brone" saja: Bron mengingatkan agar nama disebut bersama pertanyaan (speaker benar).
- [ ] "Brone, UB ada di mana?" dan "UB ada di mana, Brone?": dijawab. Ekspresi di `robot/expression` berubah.
- [ ] "kode BRONE talk itu sudah di-push belum": diam (nama di tengah kalimat).
- [ ] Pertanyaan lanjutan tanpa nama: diam.
- [ ] "Brone, apa kabar?" -> Bron bertanya balik; satu tanggapan tanpa nama dalam 8 detik dijawab.
- [ ] `mosquitto_sub -h 192.168.100.2 -t 'robot/#'` menerima tahap yang sama.
- [ ] `brone_talk_monitor.py` di `brone_main_ws` menampilkan status yang benar.
- [ ] `state_machine_node.py` bisa start dan stop layanan lewat perintah `talk`.

Bandingkan latensi dengan `~/brone-talk-v4/runtime/turns.jsonl` (rute, `total_ms`).

## Cutover (belum dilakukan)

Dilakukan hanya setelah smoke test lulus semua dan Anda setuju:

1. `systemctl --user stop brone-talk-next` dan `systemctl --user disable brone-talk-v4`.
2. Arsipkan unit lama: `mkdir -p ~/.config/systemd/user/archive && mv ~/.config/systemd/user/brone-talk.service ~/.config/systemd/user/archive/`,
   lalu `mv ~/.config/systemd/user/brone-talk-next.service ~/.config/systemd/user/brone-talk.service` dan `systemctl --user daemon-reload`.
3. Perbarui `~/dion/brone_talk.sh` dan `~/bin/brone-pakai` agar menunjuk ke repo baru, atau pensiunkan.
4. `systemctl --user enable brone-talk` bila ingin otomatis saat boot.
5. `state_machine_node.py` tidak berubah: nama unit tetap.

**Rollback:** `systemctl --user stop brone-talk`, pulihkan unit arsip ke `brone-talk.service`,
`daemon-reload`, lalu `systemctl --user enable --now brone-talk-v4` (atau `~/bin/brone-pakai v4`).

## Setelah cutover

- Ganti nama unit llama.cpp (`brone-v4-*` menjadi `brone-llm|embed|rerank`), hanya bila perlu.
- Hapus `~/brone-talk-v4` dan `~/brone-talk` setelah beberapa hari stabil. Simpan `models/`.
