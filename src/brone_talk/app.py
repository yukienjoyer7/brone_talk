"""Layanan BRONE Talk: wiring komponen, warmup, dan loop percakapan.

Urutan per ucapan:
  rekam -> STT -> panggilan (nama di awal/akhir) -> filter sampah -> keputusan (session.decide)
  -> jawaban jam (tanpa LLM) atau mesin jawaban -> TTS.

Mode wake word (default; BRONE_KWS_MODEL kosong = mode teks; desain 23 Okt, jendela 26 Okt 2026):
  tidur -> model KWS mendengar "halo brone" -> tunggu jeda -> "Ya?" -> jendela percakapan
  BRONE_KWS_WINDOW_SECONDS (60 s) sejak "Ya?": SEMUA pertanyaan di dalamnya dijawab tanpa menyebut
  nama -> jendela habis -> tidur. "halo brone" di dalam jendela = "Ya?" lagi dan jendela dimulai ulang.
  STT hanya jalan setelah bangun. Selama Bron berpikir/berbicara, "halo brone" = interupsi: jawaban
  dibatalkan, "Ya?". Tanya-balik Bron memberi minimal BRONE_SOCIAL_AWAIT_SECONDS untuk menanggapi.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from threading import Event
from typing import Any

from brone_talk import messages
from brone_talk.audio.capture import Recorder
from brone_talk.audio.kws import InterruptMonitor, WakeHeard, WakeWord, self_wake
from brone_talk.audio.vad import SileroVad
from brone_talk.config import Settings
from brone_talk.engine.answer import AnswerEngine, EngineConfig
from brone_talk.nlu.wake import NAME_RE, call_text, clock_answer, collapse_repeats
from brone_talk.session import Decision, Outcome, Session, decide
from brone_talk.stt.filters import stt_garbage_reason, stt_strip_prompt_sentence
from brone_talk.stt.transcriber import Transcriber
from brone_talk.telemetry import Telemetry
from brone_talk.text import stt_tokens
from brone_talk.tts.speaker import Speaker

LOG = logging.getLogger("brone-talk")
ENGINE_LABEL = "faq+rag"
# rute yang tidak diucapkan untuk tanggapan di jendela tanya-balik (biasanya obrolan tim)
_SILENT_ON_REPLY = ("tolak", "tidak_dimengerti", "gagal", "gerakan")


class Interrupted(Exception):
    """Jawaban dibatalkan oleh "halo brone" di tengah berpikir/berbicara."""


def engine_config(settings: Settings, runtime_dir: Path) -> EngineConfig:
    e = settings.engine
    return EngineConfig(
        kb_dir=settings.kb_dir,
        runtime_dir=runtime_dir,
        embed_url=e.embed_url,
        rerank_url=e.rerank_url,
        local_url=e.local_url,
        local_model=e.local_model,
        cloud_url=e.cloud_url,
        cloud_model=e.cloud_model,
        cloud_key=e.cloud_key,
        mode=e.mode,
        cloud_timeout_s=e.cloud_timeout_s,
        local_timeout_s=e.local_timeout_s,
        max_tokens=e.max_tokens,
        faq_min_sim=e.faq_min_sim,
        faq_min_margin=e.faq_min_margin,
        stt_min_mean=e.faq_stt_min_mean,
        stt_min_seg=e.faq_stt_min_seg,
        faq_low_stt_min_sim=e.faq_low_stt_min_sim,
        rerank_min=e.rerank_min,
        rerank_top=e.rerank_top,
        rerank_doc_chars=e.rerank_doc_chars,
        rerank_timeout_s=e.rerank_timeout_s,
        context_chars=e.context_chars,
        context_chunks=e.context_chunks,
        grounding=e.grounding,
        chat_llm=e.chat_llm,
        startup_wait_s=e.server_wait_s,
    )


class TalkService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._check_assets()
        self.runtime_dir = settings.runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = Event()
        self.telemetry = Telemetry(settings.mqtt)
        vad = SileroVad(settings.models_dir / "silero_vad.onnx")
        self.recorder = Recorder(settings.audio, self.telemetry, self.stop_event, vad, self.runtime_dir)
        self.transcriber = Transcriber(settings.stt, settings.mai, self.telemetry)
        self.speaker = Speaker(settings.tts, settings.audio, self.telemetry, self.runtime_dir)
        self.engine = AnswerEngine(engine_config(settings, self.runtime_dir))
        self.session = Session()
        a = settings.audio
        self.kws = WakeWord(Path(a.kws_model), a.kws_threshold) if a.kws_model else None
        # pemantau interupsi berjalan di thread lain, jadi memakai instance model sendiri
        self.kws_interrupt = WakeWord(Path(a.kws_model), a.kws_threshold) if a.kws_model and a.kws_interrupt else None
        self._woken = False  # interupsi barusan: langsung "Ya?" tanpa menunggu wake word lagi
        self._convo_until = 0.0  # time.monotonic() akhir jendela percakapan (mode KWS)
        self._interrupt: threading.Event | None = None  # diset pemantau saat jawaban sedang berjalan
        self._answer_thread: threading.Thread | None = None

    def _check_assets(self) -> None:
        required = [
            self.settings.models_dir / "silero_vad.onnx",
            self.settings.tts.voice_model,
            self.settings.tts.voice_config,
        ]
        if self.settings.audio.kws_model:
            required.append(Path(self.settings.audio.kws_model))
        missing = [str(p) for p in required if not Path(p).is_file()]
        if missing:
            raise RuntimeError(f"Aset model wajib belum ada: {', '.join(missing)}")

    # -- jawaban (pembungkus mesin jawaban yang juga mengirim telemetri LLM) --------
    def _answer(self, text: str, replying_to: str = "") -> dict[str, Any]:
        self.telemetry.stage("llm", "start", model=ENGINE_LABEL, backend="v4")
        result = self._cancellable(lambda: self.engine.answer(text, self.transcriber.last_stats,
                                                              replying_to=replying_to))
        meta = result.get("meta", {})
        self.telemetry.stage("llm", "done", ms=meta.get("total_ms"), route=result.get("route"),
                             backend=(meta.get("llm") or {}).get("backend", ""), model=ENGINE_LABEL)
        return result

    def _cancellable(self, call):
        """Jalankan call; dengan pemantau interupsi aktif, berhenti menunggu (Interrupted) saat "halo brone".

        Thread mesin jawaban tidak bisa dibunuh: ia selesai di latar dan hasilnya dibuang.
        Panggilan mesin berikutnya menunggu thread itu selesai dulu (_join_answer).
        """
        self._join_answer()
        if self._interrupt is None:
            return call()
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["result"] = call()
            except BaseException as exc:  # noqa: BLE001 - diteruskan ke thread utama
                box["error"] = exc

        self._answer_thread = threading.Thread(target=run, name="answer", daemon=True)
        self._answer_thread.start()
        while self._answer_thread.is_alive():
            if self._interrupt.wait(0.05):
                raise Interrupted()
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _join_answer(self) -> None:
        if self._answer_thread is not None and self._answer_thread.is_alive():
            LOG.info("Menunggu jawaban yang diinterupsi selesai di latar")
            self._answer_thread.join(timeout=self.settings.engine.local_timeout_s + 5)

    def _clear_conversation(self) -> None:
        self._join_answer()
        self.engine.reset()

    def _expire_await(self, now: float | None = None) -> None:
        """Jendela "Ya?" habis: pertanyaan berikutnya harus menyebut nama lagi."""
        now = time.monotonic() if now is None else now
        if self.session.expired(now, self.settings.await_s):
            self.session.end_wait()
            self.telemetry.stage("idle", "standby", reason="await_timeout")

    # -- warmup -------------------------------------------------------------------
    def warmup(self) -> None:
        started = time.perf_counter()
        self.telemetry.status("warming", "memuat VAD")
        self.recorder.vad.reset()
        self.telemetry.status("warming", "memuat faster-whisper")
        self.transcriber.load()
        self.telemetry.status("warming", f"memvalidasi TTS ({self.speaker.backend})")
        path, _ms, _duration = self.speaker.synthesize("Halo.")
        path.unlink(missing_ok=True)
        self.telemetry.status("warming", "memuat FAQ dan RAG (llama.cpp)")
        self.engine.start()
        if self.settings.warmup_llm:
            self.telemetry.status("warming", "memvalidasi mesin jawaban")
            # "Halo Bron" menyentuh FAQ sapaan (tanpa LLM). prev_question tidak di-reset, sama seperti produksi.
            self.engine.answer("Halo Bron", None)
        elapsed = time.perf_counter() - started
        self.telemetry.status("ready", f"{elapsed:.1f}s")
        self.telemetry.stage("idle", "standby")
        self.telemetry.expression("idle")
        LOG.info("Warmup selesai dalam %.1fs", elapsed)

    def stop(self, _signal: int | None = None, _frame: Any = None) -> None:
        self.stop_event.set()

    # -- loop utama ----------------------------------------------------------------
    def run(self) -> None:
        self.warmup()
        LOG.info("BRONE talk siap, menunggu %s", "'halo brone' (KWS)" if self.kws else "kata 'Brone'")
        while not self.stop_event.is_set():
            self.turn()
        self.telemetry.stage("idle", "stopped")
        self.telemetry.expression("idle")
        self.telemetry.close()
        LOG.info("BRONE talk berhenti")

    def turn(self) -> None:
        """Satu giliran: satu ucapan dari mic sampai jawaban diucapkan (atau diabaikan)."""
        if self.kws is not None:
            self._turn_kws()
            return
        heard = self._hear()
        if heard is None:
            return
        text, called = heard
        outcome = decide(text, called=called, awaiting=self.session.awaiting_question)

        if outcome.decision is Decision.STOP:
            self.speaker.say(messages.SAY_STOP, "neutral")
            self.session.end_wait()
            self.telemetry.stage("idle", "standby", reason="voice_stop")
            return
        if outcome.decision is Decision.STANDBY:
            self.telemetry.stage("idle", "standby", heard=text[:100])
            return
        if called:
            # setiap ucapan yang memanggil Bron (nama di awal/akhir) dijawab; giliran baru mengosongkan riwayat
            self._clear_conversation()
        if outcome.decision is Decision.CONFIRM:
            self._confirm(text)
        else:
            self._reply(outcome, text)

    def _turn_kws(self) -> None:
        """Mode wake word: tidur -> "halo brone" -> "Ya?" -> pertanyaan-pertanyaan dalam jendela -> tidur."""
        session = self.session
        if not session.awaiting_question:
            if not self._woken and not self.recorder.wait_for_wake(self.kws):
                return
            self._woken = False
            self._wake_confirm(reason="wake_word")
        try:
            heard = self._hear(start_s=self._remaining_window(), wake=self.kws)
        except WakeHeard:  # "halo brone" lagi di dalam jendela
            self._wake_confirm(reason="wake_word")
            return
        if heard is None:
            return
        text, called = heard
        outcome = decide(text, called=called, awaiting=True)
        if outcome.decision is Decision.STOP:
            self.speaker.say(messages.SAY_STOP, "neutral")
            session.end_wait()
            self._convo_until = 0.0
            self.telemetry.stage("idle", "standby", reason="voice_stop")
            return
        if outcome.decision is Decision.CONFIRM:
            if self_wake(text):  # cadangan bila model KWS melewatkan "halo brone"
                self._wake_confirm(reason="wake_text")
                return
            # nama di ujung tanpa "halo" ("Udah, BRONE.") bukan panggilan ulang: tanggapi utuh
            outcome = Outcome(Decision.ANSWER, question=text)
        # riwayat (pertanyaan sebelumnya) tetap dipakai sepanjang jendela; dikosongkan saat "Ya?"
        self._reply_interruptible(outcome, text)
        if not self._woken:
            self._continue_conversation()

    def _wake_confirm(self, reason: str) -> None:
        self._clear_conversation()
        self.speaker.say(messages.SAY_CONFIRM, "happy")
        window = self.settings.audio.kws_window_s
        self.session.start_wait(window)
        self._convo_until = time.monotonic() + window
        self.telemetry.stage("idle", "awaiting_question", reason=reason)

    def _continue_conversation(self) -> None:
        """Setelah menjawab: sisa jendela percakapan tetap terbuka tanpa wake word; habis -> tidur."""
        s = self.session
        left = self._convo_until - time.monotonic()
        if s.awaiting_question:  # Bron bertanya balik: jendela = yang lebih panjang
            s.await_window = max(s.await_window, left)
            return
        if left > 1.0:
            s.start_wait(left)
            self.telemetry.stage("idle", "awaiting_question", reason="percakapan")
            return
        self.telemetry.stage("idle", "standby", reason="jendela_habis")

    def _remaining_window(self) -> float:
        s = self.session
        window = s.await_window or self.settings.audio.kws_window_s
        return max(0.5, window - (time.monotonic() - s.awaiting_since))

    def _reply_interruptible(self, outcome: Outcome, text: str) -> None:
        """_reply dengan pemantau "halo brone": interupsi menghentikan LLM/TTS lalu langsung "Ya?"."""
        if self.kws_interrupt is None:
            self._reply(outcome, text)
            return
        self.speaker.cancelled.clear()
        monitor = InterruptMonitor(self.kws_interrupt, self.recorder.stream, self.speaker.stop,
                                   lambda: self.speaker.current_text)
        self._interrupt = monitor.fired
        try:
            with monitor:
                try:
                    self._reply(outcome, text)
                except Interrupted:
                    pass
        finally:
            self._interrupt = None
            self.speaker.cancelled.clear()
        if monitor.fired.is_set():
            self.session.end_wait()
            self.telemetry.stage("idle", "interrupted")
            self.recorder.wait_silence()
            self._woken = True

    # -- dengar: rekam -> STT -> panggilan -> filter sampah ---------------------------
    def _hear(self, start_s: float = 0.0, wake: WakeWord | None = None) -> tuple[str, bool] | None:
        """(teks, dipanggil) dari satu ucapan, atau None bila tidak ada yang perlu diputuskan.

        wake: model KWS ikut mendengar selama merekam; "halo brone" -> WakeHeard (diteruskan).
        """
        text = self._record_and_transcribe(start_s, wake)
        if not text:
            return None
        if NAME_RE.search(text):
            text = collapse_repeats(text)
        text, called = self._addressing(text)
        stt = self.transcriber
        if not stt.filter_enabled:
            return text, called
        reason = stt_garbage_reason(text, stt.last_stats, stt.vocab, stt.prompt_terms)
        if reason:
            LOG.info("STT diabaikan (%s): %r", reason, text[:120])
            self.telemetry.stage("stt", "ignored", reason=reason, text=text[:100])
            # sedang menunggu pertanyaan dan ucapannya tidak jelas: minta diulang, sekali per penantian
            if len(stt_tokens(text)) >= 3 and self.session.should_say_unclear():
                self.speaker.say(messages.MSG_UNCLEAR_AUDIO, "confused")
                self.session.awaiting_since = time.monotonic()  # jendela dihitung ulang setelah minta diulang
            return None
        cleaned = stt_strip_prompt_sentence(text)
        if cleaned == text:
            return text, called
        if not cleaned:
            LOG.info("STT diabaikan (gema kalimat prompt): %r", text[:120])
            self.telemetry.stage("stt", "ignored", reason="gema kalimat prompt", text=text[:100])
            return None
        LOG.info("Gema kalimat prompt dibuang: %r -> %r", text[:120], cleaned[:120])
        return self._addressing(cleaned)

    def _record_and_transcribe(self, start_s: float = 0.0, wake: WakeWord | None = None) -> str:
        try:
            capture = self.recorder.record(start_s, wake) if wake else self.recorder.record(start_s)
        except WakeHeard:
            raise
        except Exception as exc:
            self.telemetry.stage("error", "audio", detail=str(exc)[:200])
            LOG.exception("Perekaman gagal")
            time.sleep(1)
            return ""
        if capture is None:
            self._expire_await()
            return ""
        self._expire_await(time.monotonic() - capture.duration_s)
        try:
            return self.transcriber.transcribe(capture.path)
        except Exception as exc:
            self.telemetry.stage("error", "stt", detail=str(exc)[:200])
            LOG.exception("STT gagal")
            return ""
        finally:
            capture.path.unlink(missing_ok=True)

    @staticmethod
    def _addressing(text: str) -> tuple[str, bool]:
        """Nama di awal kalimat atau akhir ucapan = panggilan; di tengah kalimat bukan (nlu.wake.call_text)."""
        called_part = call_text(text)
        return (called_part, True) if called_part else (text, False)

    # -- tanggapi ---------------------------------------------------------------------
    def _confirm(self, text: str) -> None:
        """Hanya nama ("Halo Brone"): ingatkan agar nama disebut bersama pertanyaan, atau "Ya?"."""
        if self.settings.await_s <= 0:
            # tanpa jendela susulan: ucapan tanpa nama tidak pernah dijawab (20 Okt 2026)
            self.speaker.say(messages.SAY_CALL_HINT, "happy")
            self.telemetry.stage("idle", "standby", wake_text=text[:100])
            return
        self.speaker.say(messages.SAY_CONFIRM, "happy")
        self.session.start_wait(self.settings.await_s)
        self.telemetry.stage("idle", "awaiting_question", wake_text=text[:100])

    def _reply(self, outcome: Outcome, text: str) -> None:
        if outcome.wake:
            self.telemetry.stage("idle", "answering", wake_text=outcome.question[:100])
        else:
            self.telemetry.stage("idle", "answering", follow_up_text=text[:100])
        question = outcome.question
        social_reply = not outcome.wake and self.session.social
        replying_to = self.session.bot_text if social_reply else ""
        self.session.end_wait()

        clock = clock_answer(question)
        if clock:
            LOG.info("Jawaban jam sistem: %r -> %r", question[:80], clock)
            self.speaker.say(clock, "happy")
            self.telemetry.stage("idle", "standby")
            return

        try:
            answer = self._answer(question, replying_to)
        except Interrupted:
            raise
        except Exception:
            self.telemetry.status("error", "mesin jawaban tidak tersedia")
            self.speaker.say(messages.SAY_SERVICE_ERROR, "confused")
            LOG.exception("Mesin jawaban gagal")
            self.telemetry.stage("idle", "standby")
            return

        route = str(answer.get("route", ""))
        if route == "abaikan" or (social_reply and route.startswith(_SILENT_ON_REPLY)):
            # ucapan bukan untuk Bron, atau obrolan tim di jendela tanya-balik: diam, jangan "Maaf, ..."
            LOG.info("Tidak dijawab (%s): %r", route, question[:80])
            self.telemetry.stage("idle", "standby", reason=route)
            return
        response = str(answer.get("response", "")).strip()
        expression = answer.get("expression", "neutral")
        if not response:
            response = messages.SAY_FALLBACK
            expression = "confused"
        self.speaker.say(response, expression)
        if answer.get("ask_back") and self.settings.social_await_s > 0:
            self.session.start_wait(self.settings.social_await_s, social=True, bot_text=response)
            self.telemetry.stage("idle", "awaiting_question", reason="tanya_balik")
            return
        self.telemetry.stage("idle", "standby")
