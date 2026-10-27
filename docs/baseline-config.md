# Konfigurasi aktif (disimpan 19 Okt 2026, default repo)

Ini konfigurasi yang dipakai `brone-talk.service` (repo `~/brone_talk`) dan sudah disetujui
pengguna. Nilai ini juga menjadi **default di `config.py`** dan `deploy/systemd/brone-talk.service`.

| Variabel | Nilai | Catatan |
|---|---|---|
| BRONE_MODE | auto | cloud DeepSeek; jika gagal, pertanyaan ditolak |
| BRONE_CLOUD_TIMEOUT_SECONDS | 5 | |
| BRONE_STT_BACKEND | mai | MAI-Transcribe-2, cadangan whisper |
| BRONE_WHISPER_MODEL | models/faster-whisper-small | |
| BRONE_WHISPER_BEAM | 5 | |
| BRONE_STT_NO_SPEECH / BRONE_STT_LOGPROB | 0.4 / -1.0 | |
| BRONE_MAI_PROVIDER / ENDPOINT | voicelive / japanwest | |
| BRONE_MAI_TIMEOUT_SECONDS / COOLDOWN | 3.5 / 30 | |
| BRONE_AUDIO_INPUT_BACKEND | alsa | |
| BRONE_INPUT_ALSA_DEVICE | pipewire | |
| BRONE_INPUT_PIPEWIRE_NODE | brone-mic-xvf | channel 0 olahan chip (dipilih 19 Okt 2026) |
| BRONE_INPUT_PIPEWIRE_NODES | Wireless microphone=brone-mic-wireless,XVF3800=brone-mic-xvf | prioritas mic |
| BRONE_MAX_UTTERANCE_SECONDS | 8 | |
| BRONE_AWAIT_SECONDS | 0 | jendela satu pertanyaan setelah "Ya?"; 0 = mati (20 Okt 2026, dulu 15) |
| BRONE_SOCIAL_AWAIT_SECONDS | 8 | jendela satu tanggapan setelah Bron bertanya balik (obrolan santai) |
| BRONE_CHAT_LLM | 1 | obrolan di luar kb/smalltalk.yaml dijawab LLM persona; 0 = ditolak |
| BRONE_OUTPUT_DEVICE / BRONE_PLAYBACK | default / 1 | |
| BRONE_MQTT_BROKER / PORT | 192.168.100.2 / 1883 | |

Rahasia (BRONE_DEEPSEEK_API_KEY, BRONE_MAI_API_KEY) tetap di `~/.config/brone-talk/brone.env`.

---

# Konfigurasi efektif produksi (baseline Fase 0)

Diambil dari `systemctl --user show brone-talk-v4.service` pada 19 Okt 2026, lalu
drop-in dan EnvironmentFile digabung. Nilai rahasia disensor. Ini titik acuan historis
refactor; mode auto sekarang menolak saat cloud gagal, tanpa cadangan LLM lokal.

## Variabel yang berpengaruh pada v4

| Variabel | Nilai produksi | Default di kode v4 | Catatan |
|---|---|---|---|
| BRONE_ENGINE | v4 | v4 | |
| BRONE_MODE | auto | auto | cloud lalu lokal |
| BRONE_STT_BACKEND | mai | whisper | override dari drop-in 20-mai-stt.conf |
| BRONE_MAI_PROVIDER | voicelive | voicelive | |
| BRONE_MAI_ENDPOINT | https://japanwest.api.cognitive.microsoft.com | sama | |
| BRONE_MAI_TIMEOUT_SECONDS | 3.5 | | drop-in aktif; ada juga 2.5 di 20-mai-stt.conf.bak_20261014_t25 (tidak aktif) |
| BRONE_MAI_COOLDOWN_SECONDS | 30 | | |
| BRONE_WHISPER_MODEL | .../faster-whisper-small | large-v3-turbo | README lama salah; yang dipakai small |
| BRONE_WHISPER_BEAM | 5 | | drop-in 10-whisper-model.conf; unit utama juga set 1 (dikalahkan drop-in) |
| BRONE_STT_NO_SPEECH | 0.4 | | |
| BRONE_STT_LOGPROB | -1.0 | | |
| BRONE_MAX_UTTERANCE_SECONDS | 8 | 12 | beda dengan default kode |
| BRONE_HISTORY_MESSAGES | 4 | 10 | beda dengan default kode |
| BRONE_LOCK | 1 | 1 | kunci penanya aktif |
| BRONE_LOCK_TIMEOUT_SECONDS | 15 | | |
| BRONE_DOA_PIPEWIRE_NODE | brone-mic-raw4 | | |
| BRONE_INPUT_DEVICE | reSpeaker XVF3800 | | |
| BRONE_INPUT_ALSA_DEVICE | pipewire | | |
| BRONE_INPUT_PIPEWIRE_NODE | brone-mic-xvf | | |
| BRONE_INPUT_PIPEWIRE_NODES | XVF3800=brone-mic-xvf,Wireless microphone=brone-mic-wireless | | |
| BRONE_OUTPUT_DEVICE | default | | sink default saat ini reSpeaker, bukan CS202 |
| BRONE_PLAYBACK | 1 | | |
| BRONE_MQTT_BROKER | 192.168.100.2 | | |
| BRONE_MQTT_PORT | 1883 | | |
| BRONE_DEEPSEEK_MODEL | deepseek-flash | | dari deepseek.env |
| BRONE_DEEPSEEK_API_KEY | <redacted> | | dari deepseek.env (tidak masuk git) |
| BRONE_MAI_API_KEY | <redacted> | | dari mai.env (tidak masuk git) |

## Variabel tidak berpengaruh pada v4 (warisan)

BRONE_RAG_MODE, BRONE_EMBED_MODEL, BRONE_OLLAMA_URL, BRONE_LLM_MODEL,
BRONE_OLLAMA_NUM_THREAD, BRONE_LLM_TIMEOUT_SECONDS, BRONE_KB_HEAD_CHARS, BRONE_LLM_OFFLINE_FALLBACK,
BRONE_CLOUD_RETRY_SECONDS.

Catatan: BRONE_CLOUD_TIMEOUT_SECONDS=5 di unit berlaku untuk v4 (dibaca `EngineConfig`; default kode 8).

## Endpoint llama.cpp (dipakai v4)

| Layanan | Port | Model |
|---|---|---|
| brone-v4-llm | 8081 | LFM2.5-2.6B-Q4_K_M.gguf (cadangan lokal) |
| brone-v4-embed | 8082 | bge-m3-F16 |
| brone-v4-rerank | 8083 | bge-reranker-v2-m3-Q8_0 |

## Versi terkunci

- Python 3.12.3, interpreter venv: ~/brone-talk/.venv
- requirements: ~/brone-talk/requirements-nuc.txt (faster-whisper 1.2.1, numpy 1.26.4, onnxruntime 1.19.2, paho-mqtt 2.1.0, piper-tts <2, scipy 1.15.3, sherpa-onnx 1.13.8, sounddevice 0.5.3, websockets 17.1)
- llama.cpp b11440
