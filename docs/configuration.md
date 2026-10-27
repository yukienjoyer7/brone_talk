# Konfigurasi BRONE Talk

Dibangkitkan otomatis oleh `scripts/gen_config_docs.py` dari `src/brone_talk/config.py`.
Jangan edit berkas ini; ubah `config.py` lalu jalankan ulang skripnya.

Default di tabel = **nilai produksi efektif** (lihat `docs/baseline-config.md`). Beberapa berbeda
dengan default kode versi lama; selisihnya ditandai pada kolom catatan.

Variabel warisan yang tidak lagi berpengaruh (RAG v1-v3, Ollama, Soniox, KB format lama)
tidak dibawa ke sini; lihat `docs/decisions.md`.

## MQTT (telemetri ke Jetson)

`MqttConfig`

| Field | Default | Variabel env | Keterangan |
|---|---|---|---|
| `broker` | `"192.168.100.2"` | `BRONE_MQTT_BROKER` |  |
| `port` | `1883` | `BRONE_MQTT_PORT` |  |

## Audio: mic dan speaker

`AudioConfig`

| Field | Default | Variabel env | Keterangan |
|---|---|---|---|
| `backend` | `"alsa"` | `BRONE_AUDIO_INPUT_BACKEND` | : alsa | sounddevice |
| `input_device` | `"reSpeaker XVF3800"` | `BRONE_INPUT_DEVICE` | (jalur sounddevice) |
| `alsa_device` | `"pipewire"` | `BRONE_INPUT_ALSA_DEVICE` |  |
| `pw_node` | `"brone-mic-xvf"` | `BRONE_INPUT_PIPEWIRE_NODE` | (cadangan) |
| `pw_nodes` | `"Wireless microphone=brone-mic-wireless,XVF3800=brone-mic-xvf"` | `(tetap)` |  |
| `output_device` | `"default"` | `BRONE_OUTPUT_DEVICE` |  |
| `playback` | `True` | `BRONE_PLAYBACK` |  |
| `vad_threshold` | `0.7` | `BRONE_VAD_THRESHOLD` |  |
| `end_silence_s` | `0.8` | `BRONE_END_SILENCE_SECONDS` |  |
| `min_speech_s` | `0.35` | `BRONE_MIN_SPEECH_SECONDS` |  |
| `max_utterance_s` | `8.0` | `BRONE_MAX_UTTERANCE_SECONDS` | (kode lama: 12) |
| `min_utterance_rms` | `0.030` | `BRONE_MIN_UTTERANCE_RMS` |  |
| `noise_reduce` | `False` | `BRONE_NOISE_REDUCE` |  |
| `noise_prop_decrease` | `0.9` | `BRONE_NOISE_PROP_DECREASE` |  |
| `noise_stationary` | `False` | `BRONE_NOISE_STATIONARY` |  |
| `kws_model` | `str(ROOT / "models" / "kws" / "halo_brone.onnx")` | `BRONE_KWS_MODEL` | : kosong = mode teks |
| `kws_threshold` | `0.5` | `BRONE_KWS_THRESHOLD` |  |
| `kws_window_s` | `60.0` | `BRONE_KWS_WINDOW_SECONDS` | : jendela percakapan tanpa wake word sejak "Ya?" |
| `kws_interrupt` | `True` | `BRONE_KWS_INTERRUPT` | : "halo brone" menghentikan jawaban |

## STT (whisper lokal)

`SttConfig`

| Field | Default | Variabel env | Keterangan |
|---|---|---|---|
| `backend` | `"mai"` | `BRONE_STT_BACKEND` | : mai | whisper |
| `whisper_model` | `str(ROOT / "models" / "faster-whisper-small")` | `BRONE_WHISPER_MODEL` |  |
| `whisper_threads` | `4` | `BRONE_WHISPER_THREADS` |  |
| `whisper_beam` | `5` | `BRONE_WHISPER_BEAM` | (kode lama: 3) |
| `no_speech_threshold` | `0.4` | `BRONE_STT_NO_SPEECH` |  |
| `log_prob_threshold` | `-1.0` | `BRONE_STT_LOGPROB` |  |
| `temperature` | `"0.0"` | `BRONE_WHISPER_TEMPERATURE` |  |
| `max_new_tokens` | `64` | `BRONE_WHISPER_MAX_NEW_TOKENS` |  |
| `prompt_maxchars` | `700` | `BRONE_WHISPER_PROMPT_MAXCHARS` |  |
| `prompt` | `""` | `BRONE_WHISPER_PROMPT` | (kosong = prompt baku) |
| `hotwords` | `""` | `BRONE_WHISPER_HOTWORDS` | (kosong = daftar baku) |
| `fix` | `True` | `BRONE_STT_FIX` |  |
| `filter` | `True` | `BRONE_STT_FILTER` |  |

## STT cloud (MAI-Transcribe-2 via Azure Voice Live)

`MaiConfig`

| Field | Default | Variabel env | Keterangan |
|---|---|---|---|
| `provider` | `"voicelive"` | `BRONE_MAI_PROVIDER` |  |
| `api_key` | `""` | `BRONE_MAI_API_KEY` | (dari mai.env, jangan di repo) |
| `endpoint` | `"https://japanwest.api.cognitive.microsoft.com"` | `BRONE_MAI_ENDPOINT` |  |
| `timeout_s` | `3.5` | `BRONE_MAI_TIMEOUT_SECONDS` |  |
| `connect_timeout_s` | `6.0` | `BRONE_MAI_CONNECT_TIMEOUT_SECONDS` |  |
| `dns_timeout_s` | `1.5` | `BRONE_MAI_DNS_TIMEOUT_SECONDS` |  |
| `keepalive_s` | `5.0` | `BRONE_MAI_KEEPALIVE_SECONDS` |  |
| `probe_timeout_s` | `2.0` | `BRONE_MAI_PROBE_TIMEOUT_SECONDS` | : batas balasan ping |
| `cooldown_s` | `30.0` | `BRONE_MAI_COOLDOWN_SECONDS` |  |
| `auth_cooldown_s` | `600.0` | `BRONE_MAI_AUTH_COOLDOWN_SECONDS` |  |
| `ws_max_age_s` | `600.0` | `BRONE_MAI_WS_MAX_AGE_SECONDS` |  |
| `chat_model` | `"gpt-4.1-mini"` | `BRONE_MAI_CHAT_MODEL` |  |
| `language` | `"id"` | `BRONE_MAI_LANGUAGE` |  |
| `phrases` | `""` | `BRONE_MAI_PHRASES` | : dipisah koma; "-" = tanpa daftar |
| `default_prob` | `0.9` | `BRONE_MAI_DEFAULT_PROB` |  |

## TTS (Piper)

`TtsConfig`

| Field | Default | Variabel env | Keterangan |
|---|---|---|---|
| `piper_bin` | `str(Path(sys.executable).parent / "piper")` | `BRONE_PIPER_BIN` |  |
| `voice_model` | `ROOT / "models" / "voices" / "id_ID-brone_D21-medium.onnx"` | `BRONE_PIPER_VOICE` |  |
| `voice_config` | `ROOT / "models" / "voices" / "id_ID-brone_D21-medium.onnx.json"` | `BRONE_PIPER_CONFIG` |  |
| `textnorm_dir` | `str(ROOT / "models" / "voices" / "brone_D21_teks")` | `BRONE_TTS_TEXTNORM` | : kosong = news_tts |

## Mesin jawaban (FAQ, RAG, reranker, LLM)

`EngineSettings`

| Field | Default | Variabel env | Keterangan |
|---|---|---|---|
| `mode` | `"auto"` | `BRONE_MODE` | : auto | cloud | local | faq |
| `cloud_key` | `""` | `BRONE_DEEPSEEK_API_KEY` | (dari deepseek.env) |
| `cloud_url` | `"https://api.deepseek.com/chat/completions"` | `BRONE_DEEPSEEK_URL` |  |
| `cloud_model` | `"deepseek-flash"` | `BRONE_DEEPSEEK_MODEL` |  |
| `cloud_timeout_s` | `5.0` | `BRONE_CLOUD_TIMEOUT_SECONDS` | (kode lama: 8) |
| `local_url` | `"http://127.0.0.1:8081"` | `BRONE_LOCAL_URL` |  |
| `local_model` | `"lfm2.5-2.6b"` | `BRONE_LOCAL_MODEL` |  |
| `local_timeout_s` | `25.0` | `BRONE_LOCAL_TIMEOUT_SECONDS` |  |
| `embed_url` | `"http://127.0.0.1:8082"` | `BRONE_EMBED_URL` |  |
| `rerank_url` | `"http://127.0.0.1:8083"` | `BRONE_RERANK_URL` |  |
| `rerank_min` | `0.30` | `BRONE_RERANK_MIN` |  |
| `rerank_top` | `8` | `BRONE_RERANK_TOP` |  |
| `rerank_doc_chars` | `700` | `BRONE_RERANK_DOC_CHARS` |  |
| `rerank_timeout_s` | `10.0` | `BRONE_RERANK_TIMEOUT_SECONDS` |  |
| `faq_min_sim` | `0.80` | `BRONE_FAQ_MIN_SIM` |  |
| `faq_min_margin` | `0.0` | `BRONE_FAQ_MIN_MARGIN` |  |
| `faq_stt_min_mean` | `0.50` | `BRONE_FAQ_STT_MIN_MEAN` |  |
| `faq_stt_min_seg` | `0.40` | `BRONE_FAQ_STT_MIN_SEG` |  |
| `faq_low_stt_min_sim` | `0.90` | `BRONE_FAQ_LOW_STT_MIN_SIM` |  |
| `context_chars` | `1400` | `BRONE_CONTEXT_CHARS` |  |
| `context_chunks` | `4` | `BRONE_CONTEXT_CHUNKS` |  |
| `max_tokens` | `160` | `BRONE_LLM_MAX_TOKENS` |  |
| `grounding` | `True` | `BRONE_GROUNDING` |  |
| `chat_llm` | `True` | `BRONE_CHAT_LLM` | : obrolan di luar kb/smalltalk.yaml dijawab LLM persona |
| `server_wait_s` | `120.0` | `BRONE_SERVER_WAIT_SECONDS` |  |

## Pengaturan lain

| Variabel | Default | Keterangan |
|---|---|---|
| `BRONE_WARMUP_LLM` | `1` | validasi mesin jawaban saat start ("Halo Bron") |
| `BRONE_LOG_LEVEL` | `INFO` | tingkat log |
| `BRONE_AWAIT_SECONDS` | `0` | jendela satu pertanyaan tanpa nama setelah "Ya?"; 0 = mati |
| `BRONE_SOCIAL_AWAIT_SECONDS` | `8` | jendela satu tanggapan setelah Bron bertanya balik |

Rahasia (`BRONE_DEEPSEEK_API_KEY`, `BRONE_MAI_API_KEY`) hanya di `~/.config/brone-talk/brone.env`.
