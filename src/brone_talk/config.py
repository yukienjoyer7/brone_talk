"""Semua pengaturan BRONE Talk dibaca dari sini, sekali, saat start.

Nilai default = konfigurasi produksi efektif (lihat docs/baseline-config.md), BUKAN default
di kode versi lama. Beberapa berbeda: BRONE_MAX_UTTERANCE_SECONDS (12 di kode lama, 8 di
produksi), BRONE_HISTORY_MESSAGES (10 vs 4), BRONE_WHISPER_BEAM (3 vs 5). Yang di produksi
dipakai, jadi itu yang default di sini.

Variabel warisan yang tidak berpengaruh pada mesin v4 (RAG v1-v3, Ollama, Soniox, TTS
selain Piper, KB format lama) tidak dibawa ke sini.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except ValueError:
        return default


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except ValueError:
        return default


def _str(env: Mapping[str, str], name: str, default: str) -> str:
    return env.get(name, default).strip()


@dataclass(frozen=True)
class MqttConfig:
    broker: str = "192.168.100.2"  # BRONE_MQTT_BROKER
    port: int = 1883  # BRONE_MQTT_PORT


@dataclass(frozen=True)
class AudioConfig:
    backend: str = "alsa"  # BRONE_AUDIO_INPUT_BACKEND: alsa | sounddevice
    input_device: str = "reSpeaker XVF3800"  # BRONE_INPUT_DEVICE (jalur sounddevice)
    alsa_device: str = "pipewire"  # BRONE_INPUT_ALSA_DEVICE
    pw_node: str = "brone-mic-xvf"  # BRONE_INPUT_PIPEWIRE_NODE (cadangan)
    # BRONE_INPUT_PIPEWIRE_NODES: kartu=node, urutan = prioritas
    pw_nodes: str = "Wireless microphone=brone-mic-wireless,XVF3800=brone-mic-xvf"
    output_device: str = "default"  # BRONE_OUTPUT_DEVICE
    playback: bool = True  # BRONE_PLAYBACK
    vad_threshold: float = 0.7  # BRONE_VAD_THRESHOLD
    end_silence_s: float = 0.8  # BRONE_END_SILENCE_SECONDS
    min_speech_s: float = 0.35  # BRONE_MIN_SPEECH_SECONDS
    max_utterance_s: float = 8.0  # BRONE_MAX_UTTERANCE_SECONDS (kode lama: 12)
    min_utterance_rms: float = 0.030  # BRONE_MIN_UTTERANCE_RMS
    noise_reduce: bool = False  # BRONE_NOISE_REDUCE
    noise_prop_decrease: float = 0.9  # BRONE_NOISE_PROP_DECREASE
    noise_stationary: bool = False  # BRONE_NOISE_STATIONARY
    # Wake word "halo brone" (openWakeWord) = default sejak 26 Okt 2026: bangun, jendela 60 s, interupsi.
    # BRONE_KWS_MODEL kosong = mode lama (nama disebut di teks STT).
    kws_model: str = str(ROOT / "models" / "kws" / "halo_brone.onnx")  # BRONE_KWS_MODEL: kosong = mode teks
    kws_threshold: float = 0.5  # BRONE_KWS_THRESHOLD
    kws_window_s: float = 60.0  # BRONE_KWS_WINDOW_SECONDS: jendela percakapan tanpa wake word sejak "Ya?"
    kws_interrupt: bool = True  # BRONE_KWS_INTERRUPT: "halo brone" menghentikan jawaban




@dataclass(frozen=True)
class SttConfig:
    backend: str = "mai"  # BRONE_STT_BACKEND: mai | whisper
    whisper_model: str = str(ROOT / "models" / "faster-whisper-small")  # BRONE_WHISPER_MODEL
    whisper_threads: int = 4  # BRONE_WHISPER_THREADS
    whisper_beam: int = 5  # BRONE_WHISPER_BEAM (kode lama: 3)
    no_speech_threshold: float = 0.4  # BRONE_STT_NO_SPEECH
    log_prob_threshold: float = -1.0  # BRONE_STT_LOGPROB
    temperature: str = "0.0"  # BRONE_WHISPER_TEMPERATURE
    max_new_tokens: int = 64  # BRONE_WHISPER_MAX_NEW_TOKENS
    prompt_maxchars: int = 700  # BRONE_WHISPER_PROMPT_MAXCHARS
    prompt: str = ""  # BRONE_WHISPER_PROMPT (kosong = prompt baku)
    hotwords: str = ""  # BRONE_WHISPER_HOTWORDS (kosong = daftar baku)
    fix: bool = True  # BRONE_STT_FIX
    filter: bool = True  # BRONE_STT_FILTER


@dataclass(frozen=True)
class MaiConfig:
    provider: str = "voicelive"  # BRONE_MAI_PROVIDER
    api_key: str = ""  # BRONE_MAI_API_KEY (dari mai.env, jangan di repo)
    endpoint: str = "https://japanwest.api.cognitive.microsoft.com"  # BRONE_MAI_ENDPOINT
    timeout_s: float = 3.5  # BRONE_MAI_TIMEOUT_SECONDS
    connect_timeout_s: float = 6.0  # BRONE_MAI_CONNECT_TIMEOUT_SECONDS
    dns_timeout_s: float = 1.5  # BRONE_MAI_DNS_TIMEOUT_SECONDS
    # Pemantau internet (26 Okt 2026): rute default + ping MAI tiap keepalive_s; gagal -> STT langsung whisper.
    keepalive_s: float = 5.0  # BRONE_MAI_KEEPALIVE_SECONDS
    probe_timeout_s: float = 2.0  # BRONE_MAI_PROBE_TIMEOUT_SECONDS: batas balasan ping
    cooldown_s: float = 30.0  # BRONE_MAI_COOLDOWN_SECONDS
    auth_cooldown_s: float = 600.0  # BRONE_MAI_AUTH_COOLDOWN_SECONDS
    ws_max_age_s: float = 600.0  # BRONE_MAI_WS_MAX_AGE_SECONDS
    chat_model: str = "gpt-4.1-mini"  # BRONE_MAI_CHAT_MODEL
    language: str = "id"  # BRONE_MAI_LANGUAGE
    phrases: str = ""  # BRONE_MAI_PHRASES: dipisah koma; "-" = tanpa daftar
    default_prob: float = 0.9  # BRONE_MAI_DEFAULT_PROB


@dataclass(frozen=True)
class TtsConfig:
    piper_bin: str = str(Path(sys.executable).parent / "piper")  # BRONE_PIPER_BIN
    # Default sejak 26 Okt 2026: suara Bron fine-tune D21. Suara lama: deploy/tts_voice.sh lama
    voice_model: Path = ROOT / "models" / "voices" / "id_ID-brone_D21-medium.onnx"  # BRONE_PIPER_VOICE
    voice_config: Path = ROOT / "models" / "voices" / "id_ID-brone_D21-medium.onnx.json"  # BRONE_PIPER_CONFIG
    # Kosong = pakai clean_for_speech (untuk suara lama news_tts)
    textnorm_dir: str = str(ROOT / "models" / "voices" / "brone_D21_teks")  # BRONE_TTS_TEXTNORM: kosong = news_tts


@dataclass(frozen=True)
class EngineSettings:
    """Pengaturan mesin jawaban (brone_talk.engine.answer.EngineConfig), diisi di sini."""

    mode: str = "auto"  # BRONE_MODE: auto | cloud | local | faq
    cloud_key: str = ""  # BRONE_DEEPSEEK_API_KEY (dari deepseek.env)
    cloud_url: str = "https://api.deepseek.com/chat/completions"  # BRONE_DEEPSEEK_URL
    cloud_model: str = "deepseek-flash"  # BRONE_DEEPSEEK_MODEL
    cloud_timeout_s: float = 5.0  # BRONE_CLOUD_TIMEOUT_SECONDS (kode lama: 8)
    local_url: str = "http://127.0.0.1:8081"  # BRONE_LOCAL_URL
    local_model: str = "lfm2.5-2.6b"  # BRONE_LOCAL_MODEL
    local_timeout_s: float = 25.0  # BRONE_LOCAL_TIMEOUT_SECONDS
    embed_url: str = "http://127.0.0.1:8082"  # BRONE_EMBED_URL
    rerank_url: str = "http://127.0.0.1:8083"  # BRONE_RERANK_URL
    rerank_min: float = 0.30  # BRONE_RERANK_MIN
    rerank_top: int = 8  # BRONE_RERANK_TOP
    rerank_doc_chars: int = 700  # BRONE_RERANK_DOC_CHARS
    rerank_timeout_s: float = 10.0  # BRONE_RERANK_TIMEOUT_SECONDS
    faq_min_sim: float = 0.80  # BRONE_FAQ_MIN_SIM
    faq_min_margin: float = 0.0  # BRONE_FAQ_MIN_MARGIN
    faq_stt_min_mean: float = 0.50  # BRONE_FAQ_STT_MIN_MEAN
    faq_stt_min_seg: float = 0.40  # BRONE_FAQ_STT_MIN_SEG
    faq_low_stt_min_sim: float = 0.90  # BRONE_FAQ_LOW_STT_MIN_SIM
    context_chars: int = 1400  # BRONE_CONTEXT_CHARS
    context_chunks: int = 4  # BRONE_CONTEXT_CHUNKS
    max_tokens: int = 160  # BRONE_LLM_MAX_TOKENS
    grounding: bool = True  # BRONE_GROUNDING
    chat_llm: bool = True  # BRONE_CHAT_LLM: obrolan di luar kb/smalltalk.yaml dijawab LLM persona
    server_wait_s: float = 120.0  # BRONE_SERVER_WAIT_SECONDS


@dataclass(frozen=True)
class Settings:
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    mai: MaiConfig = field(default_factory=MaiConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    engine: EngineSettings = field(default_factory=EngineSettings)
    await_s: float = 0.0  # BRONE_AWAIT_SECONDS: jendela pertanyaan lanjutan setelah "Ya?"; 0 = mati
    social_await_s: float = 8.0  # BRONE_SOCIAL_AWAIT_SECONDS: jendela satu tanggapan setelah Bron bertanya balik
    warmup_llm: bool = True  # BRONE_WARMUP_LLM
    log_level: str = "INFO"  # BRONE_LOG_LEVEL
    runtime_dir: Path = ROOT / "runtime"
    kb_dir: Path = ROOT / "kb"
    models_dir: Path = ROOT / "models"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        d = MqttConfig()
        a = AudioConfig()
        s = SttConfig()
        m = MaiConfig()
        t = TtsConfig()
        e = EngineSettings()
        mqtt = MqttConfig(
            broker=_str(env, "BRONE_MQTT_BROKER", d.broker),
            port=_int(env, "BRONE_MQTT_PORT", d.port),
        )
        audio = AudioConfig(
            backend=_str(env, "BRONE_AUDIO_INPUT_BACKEND", a.backend).lower(),
            input_device=_str(env, "BRONE_INPUT_DEVICE", a.input_device),
            alsa_device=_str(env, "BRONE_INPUT_ALSA_DEVICE", a.alsa_device),
            pw_node=_str(env, "BRONE_INPUT_PIPEWIRE_NODE", a.pw_node),
            pw_nodes=_str(env, "BRONE_INPUT_PIPEWIRE_NODES", a.pw_nodes),
            output_device=_str(env, "BRONE_OUTPUT_DEVICE", a.output_device),
            playback=_flag(env, "BRONE_PLAYBACK", a.playback),
            vad_threshold=_float(env, "BRONE_VAD_THRESHOLD", a.vad_threshold),
            end_silence_s=_float(env, "BRONE_END_SILENCE_SECONDS", a.end_silence_s),
            min_speech_s=_float(env, "BRONE_MIN_SPEECH_SECONDS", a.min_speech_s),
            max_utterance_s=_float(env, "BRONE_MAX_UTTERANCE_SECONDS", a.max_utterance_s),
            min_utterance_rms=_float(env, "BRONE_MIN_UTTERANCE_RMS", a.min_utterance_rms),
            noise_reduce=_flag(env, "BRONE_NOISE_REDUCE", a.noise_reduce),
            noise_prop_decrease=_float(env, "BRONE_NOISE_PROP_DECREASE", a.noise_prop_decrease),
            noise_stationary=_flag(env, "BRONE_NOISE_STATIONARY", a.noise_stationary),
            kws_model=_str(env, "BRONE_KWS_MODEL", a.kws_model),
            kws_threshold=_float(env, "BRONE_KWS_THRESHOLD", a.kws_threshold),
            kws_window_s=_float(env, "BRONE_KWS_WINDOW_SECONDS", a.kws_window_s),
            kws_interrupt=_flag(env, "BRONE_KWS_INTERRUPT", a.kws_interrupt),
        )
        stt = SttConfig(
            backend=_str(env, "BRONE_STT_BACKEND", s.backend).lower(),
            whisper_model=_str(env, "BRONE_WHISPER_MODEL", s.whisper_model),
            whisper_threads=_int(env, "BRONE_WHISPER_THREADS", s.whisper_threads),
            whisper_beam=_int(env, "BRONE_WHISPER_BEAM", s.whisper_beam),
            no_speech_threshold=_float(env, "BRONE_STT_NO_SPEECH", s.no_speech_threshold),
            log_prob_threshold=_float(env, "BRONE_STT_LOGPROB", s.log_prob_threshold),
            temperature=_str(env, "BRONE_WHISPER_TEMPERATURE", s.temperature),
            max_new_tokens=_int(env, "BRONE_WHISPER_MAX_NEW_TOKENS", s.max_new_tokens),
            prompt_maxchars=_int(env, "BRONE_WHISPER_PROMPT_MAXCHARS", s.prompt_maxchars),
            prompt=_str(env, "BRONE_WHISPER_PROMPT", s.prompt),
            hotwords=_str(env, "BRONE_WHISPER_HOTWORDS", s.hotwords),
            fix=_flag(env, "BRONE_STT_FIX", s.fix),
            filter=_flag(env, "BRONE_STT_FILTER", s.filter),
        )
        mai = MaiConfig(
            provider=_str(env, "BRONE_MAI_PROVIDER", m.provider).lower(),
            api_key=_str(env, "BRONE_MAI_API_KEY", m.api_key),
            endpoint=_str(env, "BRONE_MAI_ENDPOINT", m.endpoint).rstrip("/"),
            timeout_s=_float(env, "BRONE_MAI_TIMEOUT_SECONDS", m.timeout_s),
            connect_timeout_s=_float(env, "BRONE_MAI_CONNECT_TIMEOUT_SECONDS", m.connect_timeout_s),
            dns_timeout_s=_float(env, "BRONE_MAI_DNS_TIMEOUT_SECONDS", m.dns_timeout_s),
            keepalive_s=_float(env, "BRONE_MAI_KEEPALIVE_SECONDS", m.keepalive_s),
            probe_timeout_s=_float(env, "BRONE_MAI_PROBE_TIMEOUT_SECONDS", m.probe_timeout_s),
            cooldown_s=_float(env, "BRONE_MAI_COOLDOWN_SECONDS", m.cooldown_s),
            auth_cooldown_s=_float(env, "BRONE_MAI_AUTH_COOLDOWN_SECONDS", m.auth_cooldown_s),
            ws_max_age_s=_float(env, "BRONE_MAI_WS_MAX_AGE_SECONDS", m.ws_max_age_s),
            chat_model=_str(env, "BRONE_MAI_CHAT_MODEL", m.chat_model),
            language=_str(env, "BRONE_MAI_LANGUAGE", m.language),
            phrases=_str(env, "BRONE_MAI_PHRASES", m.phrases),
            default_prob=_float(env, "BRONE_MAI_DEFAULT_PROB", m.default_prob),
        )
        tts = TtsConfig(
            piper_bin=_str(env, "BRONE_PIPER_BIN", t.piper_bin),
            voice_model=Path(_str(env, "BRONE_PIPER_VOICE", str(t.voice_model))),
            voice_config=Path(_str(env, "BRONE_PIPER_CONFIG", str(t.voice_config))),
            textnorm_dir=_str(env, "BRONE_TTS_TEXTNORM", t.textnorm_dir),
        )
        engine = EngineSettings(
            mode=_str(env, "BRONE_MODE", e.mode).lower(),
            cloud_key=_str(env, "BRONE_DEEPSEEK_API_KEY", e.cloud_key),
            cloud_url=_str(env, "BRONE_DEEPSEEK_URL", e.cloud_url),
            cloud_model=_str(env, "BRONE_DEEPSEEK_MODEL", e.cloud_model),
            cloud_timeout_s=_float(env, "BRONE_CLOUD_TIMEOUT_SECONDS", e.cloud_timeout_s),
            local_url=_str(env, "BRONE_LOCAL_URL", e.local_url),
            local_model=_str(env, "BRONE_LOCAL_MODEL", e.local_model),
            local_timeout_s=_float(env, "BRONE_LOCAL_TIMEOUT_SECONDS", e.local_timeout_s),
            embed_url=_str(env, "BRONE_EMBED_URL", e.embed_url),
            rerank_url=_str(env, "BRONE_RERANK_URL", e.rerank_url),
            rerank_min=_float(env, "BRONE_RERANK_MIN", e.rerank_min),
            rerank_top=_int(env, "BRONE_RERANK_TOP", e.rerank_top),
            rerank_doc_chars=_int(env, "BRONE_RERANK_DOC_CHARS", e.rerank_doc_chars),
            rerank_timeout_s=_float(env, "BRONE_RERANK_TIMEOUT_SECONDS", e.rerank_timeout_s),
            faq_min_sim=_float(env, "BRONE_FAQ_MIN_SIM", e.faq_min_sim),
            faq_min_margin=_float(env, "BRONE_FAQ_MIN_MARGIN", e.faq_min_margin),
            faq_stt_min_mean=_float(env, "BRONE_FAQ_STT_MIN_MEAN", e.faq_stt_min_mean),
            chat_llm=_flag(env, "BRONE_CHAT_LLM", e.chat_llm),
            faq_stt_min_seg=_float(env, "BRONE_FAQ_STT_MIN_SEG", e.faq_stt_min_seg),
            faq_low_stt_min_sim=_float(env, "BRONE_FAQ_LOW_STT_MIN_SIM", e.faq_low_stt_min_sim),
            context_chars=_int(env, "BRONE_CONTEXT_CHARS", e.context_chars),
            context_chunks=_int(env, "BRONE_CONTEXT_CHUNKS", e.context_chunks),
            max_tokens=_int(env, "BRONE_LLM_MAX_TOKENS", e.max_tokens),
            grounding=_str(env, "BRONE_GROUNDING", "1") not in ("0", "false", "no"),
            server_wait_s=_float(env, "BRONE_SERVER_WAIT_SECONDS", e.server_wait_s),
        )
        return cls(
            mqtt=mqtt,
            audio=audio,
            stt=stt,
            mai=mai,
            tts=tts,
            engine=engine,
            warmup_llm=_flag(env, "BRONE_WARMUP_LLM", True),
            await_s=_float(env, "BRONE_AWAIT_SECONDS", 0.0),
            social_await_s=_float(env, "BRONE_SOCIAL_AWAIT_SECONDS", 8.0),
            log_level=_str(env, "BRONE_LOG_LEVEL", "INFO").upper(),
        )
