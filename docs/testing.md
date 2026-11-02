# Pengujian

```bash
make test                       # semua tes (golden dilewati bila fixture belum ada)
make lint                       # ruff
make ask Q="siapa dekan filkom" # uji mesin jawaban lewat teks
```

Jalankan dengan venv yang punya dependensi (`make install` atau `PY=~/brone-talk/.venv/bin/python make test`).

## Yang diuji

| Tes | Yang dibuktikan | Butuh |
|---|---|---|
| `test_endpointing.py` | state machine endpointing: mulai, selesai karena senyap, preroll 8 jendela, batas jendela, penutupan sumber, propagasi stop/lost | - |
| `test_session.py` | aturan keputusan per ucapan (panggil, konfirmasi, stop, diam, terkunci) | - |
| `test_app_flow.py` | alur giliran lengkap dengan komponen palsu (rekam, STT, speaker, kunci, mesin) | - |
| `test_tts_normalize.py` | normalisasi TTS sama dengan produksi (117 kasus jawaban nyata) | fixture golden |
| `test_nlu_wake.py` | wake, collapse, jawaban jam, stop sama dengan produksi (587 kasus) | fixture golden |
| `test_stt_golden.py` | koreksi dan strip prompt sama dengan produksi (587 kasus) | fixture golden |
| `test_engine_golden.py` | rute FAQ dan gerbang reranker 60 pertanyaan eval sama dengan produksi | fixture golden + server 8082/8083 |

`self-test` dan `ask` tidak termasuk unit test; keduanya dijalankan manual (lihat `docs/deployment.md`).

## Fixture golden (tidak di-commit)

Fixture berisi transkrip STT asli dari mikrofon robot, termasuk percakapan orang lain. Karena
itu tidak masuk git. Bangkitkan lokal:

```bash
make golden   # butuh ~/brone-talk-v4 (kode lama) dan server 8082/8083 hidup
```

Tanpa fixture, tes golden dilewati (`skipUnless`), dan itu terlihat di output sebagai "skipped".

## Yang belum dibuktikan

- **Perangkat keras.** Mic, speaker, dan jalur PipeWire belum diuji dari repo baru. Itu butuh
  `systemctl --user start brone-talk` di NUC (lihat `docs/deployment.md`, langkah smoke test).
- **Filter sampah STT.** Fixture golden direkam tanpa kosakata, jadi `garbage_reason` hampir selalu
  kosong dan belum terbukti.
- **MQTT langsung.** Kontrak topik dicek lewat kode dan `docs/mqtt-contract.md`, bukan subscriber
  nyata ke broker Jetson.
- **Jalur LLM (DeepSeek).** Tes golden memakai `BRONE_MODE=faq` sehingga LLM tidak
  dipanggil. Jawaban LLM belum dibandingkan.
- **MAI cloud.** Hanya jalur whisper yang bisa diuji tanpa kunci Azure.
