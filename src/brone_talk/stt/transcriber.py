"""Transkripsi ucapan: MAI-Transcribe-2 (cloud, Voice Live) dengan cadangan faster-whisper lokal.

Dipindah dari brone_talk.py (v4). Hanya __init__ yang ditulis ulang untuk membaca config;
logika transkripsi, penjaga koneksi, dan cadangan tidak berubah.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from brone_talk.audio.constants import SAMPLE_RATE
from brone_talk.config import MaiConfig, SttConfig
from brone_talk.stt.confidence import stt_confidence_from_segments
from brone_talk.stt.corrections import _mai_fix, _stt_fix
from brone_talk.stt.filters import stt_prompt_terms, stt_vocab
from brone_talk.telemetry import Telemetry

LOG = logging.getLogger("brone-talk")


_MAI_PHRASES: tuple[str, ...] = (
    "BRONE", "Bron", "Universitas Brawijaya", "UB", "Malang",
    "rektor", "rektorat", "dekan", "dekanat", "pascasarjana", "vokasi",
    "UKM", "unit kegiatan mahasiswa", "letak", "UKT", "IPI",
    "SNBP", "SNBT", "PIMNAS", "QS", "BISS", "RSUB",
    # 18 fakultas/sekolah UB (nama & kode resmi 2026)
    "FH", "FEB", "FIA", "FBiPK", "FAST", "FT", "FK", "FPIK", "FSTeM",
    "FTAB", "FISIP", "FIB", "FKH", "FILKOM", "FKG", "FIKES", "SPUB",
    # nama lama yang masih umum diucapkan (sebelum ganti nama 2 Mei 2026)
    "FAPET", "FP", "FTP", "FMIPA",
    "Veteran", "Lowokwaru", "UTBK", "FILKOM UB", "BRONE FILKOM",
)


def _parse_temperature(value: str):
    parts = [float(p) for p in value.replace(" ", "").split(",") if p]
    return parts[0] if len(parts) == 1 else tuple(parts or [0.0])


def resolve_host(host: str, port: int, timeout_s: float) -> None:
    """Resolusi DNS dengan batas waktu; gagal/lewat batas -> exception.

    14 Okt 2026: timeout socket/WebSocket tidak mencakup getaddrinfo. Saat WiFi tersambung tapi DNS
    bermasalah, getaddrinfo menggantung ~20 s sebelum "Temporary failure in name resolution", jadi STT
    sempat 21-28 s walau BRONE_MAI_TIMEOUT_SECONDS=3.5 (journal 20:18-20:21). Polanya sama dengan
    _post_cloud: kerjakan di thread, tunggu dengan batas TOTAL.
    """
    box: dict[str, Any] = {}
    done = Event()

    def work() -> None:
        try:
            socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except Exception as exc:  # diteruskan ke pemanggil di bawah
            box["error"] = exc
        finally:
            done.set()

    Thread(target=work, name="dns", daemon=True).start()
    if not done.wait(timeout_s):
        raise TimeoutError(f"DNS {host} tidak menjawab dalam {timeout_s:.1f} s")
    if "error" in box:
        raise box["error"]


def has_default_route(route_file: str = "/proc/net/route") -> bool:
    """True bila OS punya rute default (WiFi/LAN tersambung). Tanpa /proc (bukan Linux) dianggap ada."""
    try:
        with open(route_file) as handle:
            next(handle, None)  # baris judul kolom
            return any(line.split()[1:2] == ["00000000"] for line in handle)
    except OSError:
        return True


class Transcriber:
    def __init__(self, stt: SttConfig, mai: MaiConfig, telemetry: Telemetry) -> None:
        self.telemetry = telemetry
        self.backend = stt.backend
        self.mai_default_prob = mai.default_prob
        if self.backend == "mai":
            # MAI-Transcribe-2 (Microsoft, cloud). Whisper tetap dimuat sebagai cadangan saat
            # internet/API gagal. Provider: voicelive (Voice Live WebSocket, transkripsi saja;
            # satu-satunya yang jalan di langganan Azure for Students), azure (REST), openrouter.
            self.mai_provider = mai.provider
            self.mai_api_key = mai.api_key
            # model utama DIKUNCI ke MAI-Transcribe-2 (keputusan 14 Okt 2026)
            self.mai_model = "MAI-Transcribe-2"
            self.mai_vl_models = ("mai-transcribe-2", "mai-transcribe")
            self.mai_active_model = self.mai_vl_models[0]
            self.mai_endpoint = mai.endpoint
            # Voice Live mewajibkan model chat di URL; respons LLM tidak pernah diminta.
            self.mai_chat_model = mai.chat_model
            # koneksi Voice Live dipakai ulang antar ucapan; dibuka ulang setelah ini
            self.mai_ws_max_age_s = mai.ws_max_age_s
            self._vl_ws: Any = None
            self._vl_opened = 0.0
            self.mai_language = mai.language
            # batas tunggu cloud per ucapan; lewat dari ini -> whisper
            self.mai_timeout_s = mai.timeout_s
            # membuka koneksi (DNS+TLS+WebSocket+sesi) ~2 s dari NUC; batasnya dipisah
            self.mai_connect_timeout_s = mai.connect_timeout_s
            # batas resolusi DNS sebelum membuka koneksi cloud
            self.mai_dns_timeout_s = mai.dns_timeout_s
            # penjaga latar: buka lebih dulu, ping tiap N detik, buka ulang saat tua/putus
            self.mai_keepalive_s = mai.keepalive_s
            self._vl_lock = Lock()
            self._vl_keeper: Any = None
            # setelah gagal, cloud dilewati selama ini; kunci salah/saldo habis lebih lama
            self.mai_cooldown_s = mai.cooldown_s
            self.mai_auth_cooldown_s = mai.auth_cooldown_s
            # Pemantau internet (pilihan A, disepakati 22 Okt 2026): penjaga mengecek rute default OS dan
            # ping MAI tiap keepalive_s. Gagal -> cloud_online False, ucapan berikutnya langsung ke whisper
            # tanpa menunggu timeout 3.5 s. Probe tetap jalan; berhasil -> kembali ke MAI.
            self.mai_probe_timeout_s = mai.probe_timeout_s
            self.cloud_online = True
            self._mai_skip_auth = False
            env_phrases = [t.strip() for t in mai.phrases.split(",") if t.strip()]
            # "-" = tanpa phrase list (untuk uji banding)
            self.mai_phrases = [] if env_phrases == ["-"] else env_phrases or list(_MAI_PHRASES)
            self.mai_skip_until = 0.0
            if self.mai_provider not in ("voicelive", "azure", "openrouter"):
                raise RuntimeError("BRONE_MAI_PROVIDER harus 'voicelive', 'azure', atau 'openrouter'")
            if not self.mai_api_key:
                LOG.warning("BRONE_MAI_API_KEY kosong: STT memakai whisper lokal saja")
                self.mai_skip_until = float("inf")
        elif self.backend != "whisper":
            raise RuntimeError("BRONE_STT_BACKEND harus 'mai' atau 'whisper'")
        self.last_engine = "whisper"
        self.last_raw = ""
        self.model_name = stt.whisper_model
        # thread 4 = jumlah P-core fisik i7-1260P; lebih banyak terbukti LEBIH LAMBAT
        # (diukur 17 Sep 2026). Jangan naikkan tanpa mengukur ulang di hardware yang sama.
        self.threads = stt.whisper_threads
        # beam 3 identik dengan 5 untuk teks, 8% lebih cepat (diukur). 1-2 membuang kata.
        self.beam_size = stt.whisper_beam
        self.model: Any | None = None
        # hotwords menstabilkan istilah domain (FILKOM/BRONE/UB) di beam search
        baseline_hotwords = (
            "BRONE, Bron, Universitas Brawijaya, UB, FILKOM, Malang, rektorat, "
            "UKM, unit kegiatan mahasiswa, letak, UKT, SNBP, SNBT, RSUB, BISS, QS, PIMNAS, FMIPA"
        )
        # prompt baku. Daftar kosakata dari knowledge base TIDAK dipakai (BRONE_WHISPER_TOPICS
        # produksi = 0); lihat docs/decisions.md.
        baseline_prompt = (
            "Percakapan dengan Brone, robot maskot Universitas Brawijaya UB di Kota Malang. "
            "Istilah: UB, Universitas Brawijaya, Brone, Malang, Veteran, Lowokwaru, "
            "fakultas, program studi, mahasiswa, kampus, rektorat, perpustakaan, akreditasi, "
            "UTBK, SNBP, SNBT, UKM, unit kegiatan mahasiswa, letak, lokasi."
        )
        self.hotwords = stt.hotwords or baseline_hotwords
        self.no_speech_threshold = stt.no_speech_threshold
        self.log_prob_threshold = stt.log_prob_threshold
        self.fix_enabled = stt.fix
        self.prompt = stt.prompt or baseline_prompt[: stt.prompt_maxchars]
        # default faster-whisper mengulang decode s/d 6 temperatur saat gagal; 0.0 = satu decode
        self.temperature = _parse_temperature(stt.temperature)
        # decode derau berjalan sampai batas token Whisper; 0 = tanpa batas
        self.max_new_tokens = stt.max_new_tokens
        self.filter_enabled = stt.filter
        self.vocab = stt_vocab(self.prompt, self.hotwords)
        self.prompt_terms = stt_prompt_terms(self.prompt, self.hotwords)
        self.last_stats: dict[str, float] = {}

    def load(self) -> None:
        if self.backend == "mai" and self.mai_provider == "voicelive" and self.mai_api_key \
                and self._vl_keeper is None:
            self._vl_keeper = Thread(target=self._voicelive_keeper, name="mai-keeper", daemon=True)
            self._vl_keeper.start()
        if self.model is not None:
            return
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        self.model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.threads,
        )
        LOG.info("faster-whisper %s siap dalam %.1fs", self.model_name, time.perf_counter() - started)

    def transcribe(self, audio_path: Path) -> str:
        model_label = self.mai_active_model if self.backend == "mai" else self.model_name
        self.telemetry.stage("stt", "start", model=model_label, backend=self.backend)
        started = time.perf_counter()
        if self.backend == "mai":
            text = self._transcribe_mai_or_whisper(audio_path)
        else:
            text = self._transcribe_whisper(audio_path)
            self.last_engine, self.last_raw = "whisper", text
        model_label = self.mai_active_model if self.last_engine == "mai" else self.model_name
        if self.fix_enabled:
            text = _stt_fix(text)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        self.telemetry.stage("stt", "done", ms=elapsed_ms, text=text[:200], model=model_label)
        LOG.info("STT %.2fs: %r", elapsed_ms / 1000, text)
        return text

    def _transcribe_whisper(self, audio_path: Path) -> str:
        self.load()
        assert self.model is not None
        segments, _info = self.model.transcribe(
            str(audio_path),
            language="id",
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=self.prompt or None,
            hotwords=self.hotwords or None,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            compression_ratio_threshold=1.8,
            hallucination_silence_threshold=2.0,
            temperature=self.temperature,
            **({"max_new_tokens": self.max_new_tokens} if self.max_new_tokens > 0 else {}),
        )
        # generator: dulu dikonsumsi dua kali, sehingga filter `kept` di bawah tak pernah jalan
        segments = list(segments)
        self.last_stats = {
            "max_cr": max((s.compression_ratio for s in segments), default=0.0),
            "mean_lp": (sum(s.avg_logprob for s in segments) / len(segments)) if segments else 0.0,
        }
        # v4: skor confidence untuk gerbang FAQ, dihitung dari segmen yang sudah ada.
        self.last_stats.update(stt_confidence_from_segments(segments))
        text = "".join(segment.text for segment in segments).strip()
        # Segmen halusinasi dibuang (no_speech tinggi DAN logprob buruk).
        kept = [
            segment.text
            for segment in segments
            if segment.no_speech_prob <= self.no_speech_threshold
            and segment.avg_logprob >= self.log_prob_threshold
        ]
        if kept:
            text = "".join(kept).strip()
        return text

    def _transcribe_mai_or_whisper(self, audio_path: Path) -> str:
        """Cloud dulu; gagal apa pun (tanpa internet, timeout, HTTP error) -> whisper pada klip yang sama."""
        if self.cloud_online and time.monotonic() >= self.mai_skip_until:
            try:
                self.last_raw = self._mai_within_deadline(audio_path)
                self.last_engine = "mai"
                return _mai_fix(self.last_raw)
            except Exception as exc:
                code = getattr(exc, "code", None) or getattr(getattr(exc, "__cause__", None), "code", None)
                self._mai_skip_auth = code in (401, 402, 403)
                rest = self.mai_auth_cooldown_s if self._mai_skip_auth else self.mai_cooldown_s
                self.mai_skip_until = time.monotonic() + rest
                LOG.warning("MAI-Transcribe gagal (%s); whisper lokal dipakai, cloud dicoba lagi saat ping "
                            "berhasil (maks %.0f s)", str(exc)[:160], rest)
                self.telemetry.stage("stt", "fallback", detail=str(exc)[:200])
        self.last_engine = "whisper"
        self.last_raw = self._transcribe_whisper(audio_path)
        return self.last_raw

    def _mai_within_deadline(self, audio_path: Path) -> str:
        """_transcribe_mai dengan SATU tenggat total BRONE_MAI_TIMEOUT_SECONDS sejak ucapan selesai.

        26 Okt 2026: batas lama hanya per fase (menunggu kunci probe, buka koneksi, kirim, tunggu transkrip),
        sehingga STT tercatat 7-26 s saat internet goyah. Panggilan cloud kini jalan di thread; lewat tenggat,
        thread itu ditinggalkan (selesai/gagal sendiri di latar) dan klip langsung ke whisper.
        """
        box: dict[str, Any] = {}
        done = Event()

        def run() -> None:
            try:
                box["text"] = self._transcribe_mai(audio_path)
            except BaseException as exc:  # noqa: BLE001 - diteruskan ke thread utama
                box["error"] = exc
            finally:
                done.set()

        Thread(target=run, name="mai-stt", daemon=True).start()
        if not done.wait(self.mai_timeout_s):
            raise TimeoutError(f"tidak ada transkrip dalam {self.mai_timeout_s:.1f} s (total)")
        if "error" in box:
            raise box["error"]
        return box["text"]

    def _transcribe_mai(self, audio_path: Path) -> str:
        if self.mai_provider == "voicelive":
            text = self._transcribe_voicelive(audio_path)
            self._set_mai_stats([])
            return text
        audio = audio_path.read_bytes()
        if self.mai_provider == "azure":
            definition: dict[str, Any] = {"enhancedMode": {"enabled": True, "model": self.mai_model}}
            if self.mai_language:
                definition["locales"] = [self.mai_language]
            if self.mai_phrases:
                definition["phraseList"] = {"phrases": self.mai_phrases}
            boundary = uuid.uuid4().hex
            body = (
                f'--{boundary}\r\nContent-Disposition: form-data; name="definition"\r\n\r\n'
                f"{json.dumps(definition)}\r\n"
                f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{audio_path.name}"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode("utf-8") + audio + f"\r\n--{boundary}--\r\n".encode("utf-8")
            req = urllib.request.Request(
                f"{self.mai_endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15",
                data=body, method="POST",
                headers={"Ocp-Apim-Subscription-Key": self.mai_api_key,
                         "Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
        else:
            payload: dict[str, Any] = {
                "model": f"microsoft/{self.mai_model.lower()}",
                "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": "wav"},
            }
            if self.mai_language:
                payload["language"] = self.mai_language
            if self.mai_phrases:  # diteruskan apa adanya ke Azure (provider.options.azure)
                payload["provider"] = {"options": {"azure": {"phraseList": {"phrases": self.mai_phrases}}}}
            req = urllib.request.Request(
                f"{self.mai_endpoint}/audio/transcriptions", data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"Authorization": f"Bearer {self.mai_api_key}", "Content-Type": "application/json"},
            )
        resolve_host(urllib.parse.urlsplit(req.full_url).hostname or "", 443, self.mai_dns_timeout_s)
        try:
            with urllib.request.urlopen(req, timeout=self.mai_timeout_s) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"MAI HTTP {exc.code}: {detail}") from exc
        if self.mai_provider == "azure":
            text = " ".join(p.get("text", "") for p in res.get("combinedPhrases", [])).strip()
            confs = [p["confidence"] for p in res.get("phrases", []) if isinstance(p.get("confidence"), (int, float))]
        else:
            text = (res.get("text") or "").strip()
            confs = [res["confidence"]] if isinstance(res.get("confidence"), (int, float)) else []
        self._set_mai_stats(confs)
        return text

    def _set_mai_stats(self, confs: list[float]) -> None:
        # Statistik logprob whisper tidak ada untuk cloud. Skor yakin dipakai v4 (gerbang FAQ) bila dikirim
        # API; kalau tidak, BRONE_MAI_DEFAULT_PROB (cloud dianggap yakin).
        prob = self.mai_default_prob
        self.last_stats = {"stt_mean_prob": round(sum(confs) / len(confs), 4) if confs else prob,
                           "stt_min_prob": round(min(confs), 4) if confs else prob}

    def _voicelive_close(self) -> None:
        if self._vl_ws is not None:
            try:
                self._vl_ws.close()
            except Exception:
                pass
        self._vl_ws = None

    def _voicelive_open(self, timeout: float) -> Any:
        from websockets.sync.client import connect

        self._voicelive_close()
        host = self.mai_endpoint.split("://", 1)[-1]
        resolve_host(urllib.parse.urlsplit(self.mai_endpoint).hostname or host, 443, self.mai_dns_timeout_s)
        ws = connect(
            f"wss://{host}/voice-live/realtime?api-version=2026-04-10&model={self.mai_chat_model}",
            additional_headers={"api-key": self.mai_api_key}, open_timeout=timeout, max_size=None,
        )
        deadline = time.monotonic() + timeout
        candidates = self.mai_vl_models
        for n, model in enumerate(candidates):
            transcription: dict[str, Any] = {"model": model}
            if self.mai_language:
                transcription["language"] = self.mai_language
            if self.mai_phrases:
                transcription["phrase_list"] = self.mai_phrases
            # turn_detection None: buffer di-commit manual per ucapan VAD kita, jadi tidak ada respons LLM.
            ws.send(json.dumps({"type": "session.update", "session": {
                "modalities": ["text"], "input_audio_format": "pcm16", "input_audio_sampling_rate": SAMPLE_RATE,
                "turn_detection": None, "input_audio_transcription": transcription,
            }}))
            while True:
                event = json.loads(ws.recv(timeout=max(0.1, deadline - time.monotonic())))
                if event.get("type") in ("session.updated", "error"):
                    break
            if event.get("type") == "session.updated":
                got = ((event.get("session") or {}).get("input_audio_transcription") or {}).get("model") or model
                if got != self.mai_active_model:
                    LOG.warning("Model STT Azure sekarang %r (sebelumnya %r)", got, self.mai_active_model)
                    self.telemetry.stage("stt", "model_changed", model=got)
                self.mai_active_model = got
                break
            if n == len(candidates) - 1:
                ws.close()
                raise RuntimeError(f"Voice Live: {json.dumps(event.get('error'))[:200]}")
            LOG.warning("Voice Live menolak %s (%s); coba %s",
                        model, json.dumps(event.get("error"))[:160], candidates[n + 1])
        self._vl_ws, self._vl_opened = ws, time.monotonic()
        return ws

    def _set_cloud_online(self, online: bool, detail: str) -> None:
        if online and not self._mai_skip_auth:
            self.mai_skip_until = 0.0  # istirahat setelah ucapan gagal diakhiri probe yang berhasil
        if online == self.cloud_online:
            return
        self.cloud_online = online
        if online:
            LOG.info("Internet pulih (%s): STT kembali ke MAI", detail)
        else:
            LOG.warning("Internet/MAI putus (%s): STT langsung whisper sampai ping berhasil", detail)
        self.telemetry.stage("stt", "cloud_online" if online else "cloud_offline", detail=detail[:200])

    def _ping(self, ws: Any) -> bool:
        try:
            return bool(ws.ping().wait(self.mai_probe_timeout_s))
        except Exception:
            return False

    def _probe_cloud(self) -> None:
        """Satu probe pemantau internet. Dipanggil penjaga tiap keepalive_s."""
        if self._mai_skip_auth and time.monotonic() < self.mai_skip_until:
            return  # kunci salah/saldo habis: jangan membanjiri Azure
        if not has_default_route():
            self._set_cloud_online(False, "tidak ada rute default")
            return
        if not self._vl_lock.acquire(blocking=False):
            return  # koneksi sedang dipakai ucapan
        try:
            ws = self._vl_ws
            fresh = ws is not None and time.monotonic() - self._vl_opened < self.mai_ws_max_age_s
            if fresh and self._ping(ws):
                self._set_cloud_online(True, "ping MAI")
            else:
                # koneksi tua/putus/ping tak dibalas: buka baru sekaligus menguji DNS+TLS+sesi
                self._voicelive_open(self.mai_connect_timeout_s)
                self._set_cloud_online(True, "koneksi MAI baru")
        except Exception as exc:
            self._voicelive_close()
            self._set_cloud_online(False, str(exc)[:120] or type(exc).__name__)
        finally:
            self._vl_lock.release()

    def _voicelive_keeper(self) -> None:
        """Latar: pemantau internet + jaga satu koneksi Voice Live siap pakai."""
        while True:
            self._probe_cloud()
            time.sleep(self.mai_keepalive_s)

    def _transcribe_voicelive(self, audio_path: Path) -> str:
        with self._vl_lock:
            return self._transcribe_voicelive_locked(audio_path)

    def _transcribe_voicelive_locked(self, audio_path: Path, retry: bool = True) -> str:
        from websockets.exceptions import ConnectionClosed

        with wave.open(str(audio_path)) as handle:
            if (handle.getframerate(), handle.getsampwidth(), handle.getnchannels()) != (SAMPLE_RATE, 2, 1):
                raise RuntimeError("Voice Live: klip harus PCM16 mono 16 kHz")
            pcm = handle.readframes(handle.getnframes())
        reused = self._vl_ws is not None and time.monotonic() - self._vl_opened < self.mai_ws_max_age_s
        # Buka di jalur ucapan hanya kalau penjaga belum menyiapkan koneksi; batasnya batas transkrip.
        ws = self._vl_ws if reused else self._voicelive_open(self.mai_timeout_s)
        try:
            for i in range(0, len(pcm), 2 * SAMPLE_RATE):  # potongan 1 detik
                ws.send(json.dumps({"type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(pcm[i:i + 2 * SAMPLE_RATE]).decode("ascii")}))
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except ConnectionClosed:
            self._voicelive_close()
            if not (reused and retry):
                raise
            # Koneksi lama ditutup server saat menganggur: buka baru sekali, kirim ulang.
            return self._transcribe_voicelive_locked(audio_path, retry=False)
        deadline = time.monotonic() + self.mai_timeout_s
        item_id = None
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"Voice Live: tidak ada transkrip dalam {self.mai_timeout_s:.0f} s")
                event = json.loads(ws.recv(timeout=left))
                kind = event.get("type")
                if kind == "input_audio_buffer.committed":
                    item_id = event.get("item_id")
                elif kind == "conversation.item.input_audio_transcription.completed" and (
                        item_id is None or event.get("item_id") == item_id):
                    return (event.get("transcript") or "").strip()
                elif kind in ("error", "conversation.item.input_audio_transcription.failed"):
                    raise RuntimeError(f"Voice Live {kind}: {json.dumps(event)[:200]}")
        except ConnectionClosed:
            self._voicelive_close()
            if not (reused and retry and item_id is None):
                raise
            return self._transcribe_voicelive_locked(audio_path, retry=False)
        except Exception:
            self._voicelive_close()
            raise
