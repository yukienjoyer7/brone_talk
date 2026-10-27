"""Wake word "halo brone": ChunkBuffer, WakeWord (skor palsu), Endpointer start/speech_limit, dan alur
TalkService mode KWS (tidur -> "Ya?" -> jendela percakapan -> tidur, interupsi) dengan komponen palsu.
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from brone_talk import messages
from brone_talk.app import TalkService
from brone_talk.audio.constants import VAD_WINDOW
from brone_talk.audio.endpointing import Endpointer
from brone_talk.audio.kws import KWS_CHUNK, ChunkBuffer, InterruptMonitor, WakeHeard, WakeWord, self_wake, watch
from brone_talk.config import Settings
from brone_talk.session import Session
from test_app_flow import FakeCapture, FakeEngine, FakeTelemetry, FakeTranscriber


def block(value=0.01):
    return np.full(VAD_WINDOW, value, dtype=np.float32)


class ChunkBufferTest(unittest.TestCase):
    def test_512_blocks_become_1280_chunks(self):
        buf = ChunkBuffer()
        sizes = [len(c) for _ in range(5) for c in buf.push(block())]
        # 5 x 512 = 2560 sampel = tepat 2 potongan 1280
        self.assertEqual(sizes, [KWS_CHUNK, KWS_CHUNK])
        self.assertEqual(buf.push(np.zeros(0, np.float32)), [])

    def test_float_is_scaled_to_int16(self):
        buf = ChunkBuffer(size=4)
        (chunk,) = buf.push(np.array([0.0, 0.5, -1.0, 2.0], np.float32))
        self.assertEqual(chunk.tolist(), [0, 16383, -32767, 32767])


class WakeWordTest(unittest.TestCase):
    def test_fires_when_any_chunk_crosses_threshold(self):
        scores = iter([0.1, 0.2, 0.9])
        ww = WakeWord(Path("x.onnx"), 0.5, predictor=lambda chunk: next(scores))
        blocks = [block() for _ in range(10)]  # 10 x 512 = 4 potongan; skor ke-3 = 0.9
        self.assertTrue(ww.listen(iter(blocks)))

    def test_source_end_without_wake_returns_false(self):
        ww = WakeWord(Path("x.onnx"), 0.5, predictor=lambda chunk: 0.1)
        self.assertFalse(ww.listen(iter([block() for _ in range(10)])))

    def test_self_wake_guard(self):
        self.assertTrue(self_wake("Halo, saya Brone!"))
        self.assertTrue(self_wake("hallo bron"))
        self.assertTrue(self_wake("Alo, BRONE."))
        self.assertFalse(self_wake("Udah, BRONE."))
        self.assertFalse(self_wake("Brone siap membantu."))
        self.assertFalse(self_wake(""))


class EndpointerStartLimitTest(unittest.TestCase):
    class Vad:
        def __init__(self, probs):
            self.probs = list(probs)

        def reset(self):
            pass

        def probability(self, samples):
            return self.probs.pop(0) if self.probs else 0.0

    def test_gives_up_when_speech_does_not_start(self):
        ep = Endpointer(self.Vad([0.0] * 50), FakeTelemetry(), 0.5, silence_limit=3, window_limit=100, start_limit=10)
        utt = ep.run(block() for _ in range(100))
        self.assertEqual(utt.frames, [])
        self.assertEqual(utt.speech_end, 10 * VAD_WINDOW)

    def test_speech_limit_counts_from_speech_start(self):
        probs = [0.0] * 30 + [0.9] * 200
        ep = Endpointer(self.Vad(probs), FakeTelemetry(), 0.5, silence_limit=3, window_limit=500, start_limit=40,
                        speech_limit=50)
        utt = ep.run(block() for _ in range(500))
        self.assertEqual(utt.speech_end, (30 + 50 + 1) * VAD_WINDOW)

    def test_watch_raises_when_wake_word_heard(self):
        scores = iter([0.1, 0.9])
        ww = WakeWord(Path("x.onnx"), 0.5, predictor=lambda chunk: next(scores))
        passed = []
        with self.assertRaises(WakeHeard):
            for b in watch(iter([block() for _ in range(10)]), ww):
                passed.append(b)
        self.assertEqual(len(passed), 4)  # potongan ke-2 lengkap di blok ke-5

    def test_speech_started_in_time_runs_past_start_limit(self):
        probs = [0.0] * 5 + [0.9] * 20 + [0.0] * 3
        ep = Endpointer(self.Vad(probs), FakeTelemetry(), 0.5, silence_limit=3, window_limit=100, start_limit=10)
        utt = ep.run(block() for _ in range(100))
        self.assertGreater(len(utt.frames), 20)


class InterruptMonitorTest(unittest.TestCase):
    def open_blocks(self, stop):
        while not stop.is_set():
            yield block()
            time.sleep(0.001)

    def test_fires_and_calls_on_wake(self):
        hits = []
        ww = WakeWord(Path("x.onnx"), 0.5, predictor=lambda chunk: 0.9)
        with InterruptMonitor(ww, self.open_blocks, lambda: hits.append(1), lambda: "") as mon:
            self.assertTrue(mon.fired.wait(2))
        self.assertEqual(hits, [1])

    def test_ignores_own_voice_saying_halo_brone(self):
        ww = WakeWord(Path("x.onnx"), 0.5, predictor=lambda chunk: 0.9)
        with InterruptMonitor(ww, self.open_blocks, lambda: None, lambda: "Halo, saya Brone.") as mon:
            self.assertFalse(mon.fired.wait(0.2))


class KwsRecorder:
    """Rekaman palsu: setiap wait_for_wake memakai satu entri `wakes` (True = "halo brone")."""

    def __init__(self):
        self.queue, self.wakes, self.starts, self.silences = [], [], [], 0
        self.stream_blocks = None

    def wait_for_wake(self, detector):
        return self.wakes.pop(0) if self.wakes else False

    def record(self, start_s=0.0, wake=None):
        self.starts.append(start_s)
        item = self.queue.pop(0) if self.queue else None
        if item == "WAKE":  # "halo brone" terdengar selama merekam di jendela percakapan
            raise WakeHeard()
        return item

    def wait_silence(self):
        self.silences += 1

    def stream(self, stop):
        while not stop.is_set():
            yield block()
            time.sleep(0.001)


class KwsSpeaker:
    def __init__(self):
        self.said, self.current_text = [], ""
        self.cancelled = threading.Event()

    def say(self, text, expression):
        if self.cancelled.is_set():
            return False
        self.said.append(text)
        return True

    def stop(self):
        self.cancelled.set()


class KwsFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        svc = self.svc = object.__new__(TalkService)
        svc.settings = Settings.from_env({"BRONE_KWS_MODEL": "halo_brone.onnx"})
        svc.telemetry = FakeTelemetry()
        svc.recorder = KwsRecorder()
        svc.transcriber = FakeTranscriber()
        svc.speaker = KwsSpeaker()
        svc.engine = FakeEngine()
        svc.session = Session()
        svc.runtime_dir = self.dir
        svc.kws = WakeWord(Path("x.onnx"), 0.5, predictor=lambda chunk: 0.0)
        svc.kws_interrupt = None
        svc._woken, svc._interrupt, svc._answer_thread, svc._convo_until = False, None, None, 0.0

    def tearDown(self):
        self._tmp.cleanup()

    def queue(self, *texts):
        for i, text in enumerate(texts):
            wav = self.dir / f"in-{i}-{len(self.svc.recorder.queue)}.wav"
            wav.write_bytes(b"")
            self.svc.recorder.queue.append(FakeCapture(wav))
            self.svc.transcriber.queue.append(text)

    def test_sleeps_without_wake_word_and_never_runs_stt(self):
        self.queue("Brone, UB ada di mana?")  # nama di teks tidak membangunkan Bron di mode KWS
        self.svc.turn()
        self.assertEqual(self.svc.speaker.said, [])
        self.assertEqual(self.svc.transcriber.queue, ["Brone, UB ada di mana?"])

    def test_window_answers_many_questions_without_wake_word(self):
        self.svc.recorder.wakes = [True]
        self.queue("siapa rektor UB sekarang", "dan dekan FILKOM siapa")
        self.svc.turn()
        self.assertEqual(self.svc.speaker.said, [messages.SAY_CONFIRM, "JAWAB siapa rektor UB sekarang"])
        self.assertAlmostEqual(self.svc.recorder.starts[0], 60.0, delta=0.5)
        self.assertTrue(self.svc.session.awaiting_question)  # jendela 60 s masih terbuka
        self.svc.turn()  # tanpa wake word
        self.assertEqual(self.svc.engine.calls, ["siapa rektor UB sekarang", "dan dekan FILKOM siapa"])
        self.assertLess(self.svc.recorder.starts[1], 60.0)  # sisa jendela, bukan 60 s baru

    def test_window_over_goes_back_to_sleep(self):
        self.svc.recorder.wakes = [True]
        self.queue("siapa rektor UB sekarang", "obrolan setelah jendela")
        self.svc.turn()
        self.svc._convo_until = time.monotonic() - 1  # jendela habis selagi menjawab
        self.svc.session.end_wait()
        self.svc._continue_conversation()
        self.assertFalse(self.svc.session.awaiting_question)
        self.svc.turn()  # tidur: wake word tidak terdengar -> ucapan tidak diproses
        self.assertEqual(self.svc.engine.calls, ["siapa rektor UB sekarang"])

    def test_halo_brone_inside_window_says_ya_again_and_restarts(self):
        svc = self.svc
        svc.recorder.wakes = [True]
        self.queue("siapa rektor UB sekarang")
        svc.recorder.queue.append("WAKE")
        svc.turn()
        first_until = svc._convo_until
        time.sleep(0.01)
        svc.turn()  # "halo brone" terdengar oleh KWS saat merekam
        self.assertEqual(svc.speaker.said, [messages.SAY_CONFIRM, "JAWAB siapa rektor UB sekarang",
                                            messages.SAY_CONFIRM])
        self.assertGreater(svc._convo_until, first_until)
        self.assertTrue(svc.session.awaiting_question)

    def test_name_at_end_without_halo_is_not_a_new_wake(self):
        self.svc.recorder.wakes = [True]
        self.queue("Udah, BRONE.")
        self.svc.turn()
        self.assertEqual(self.svc.speaker.said[0], messages.SAY_CONFIRM)
        self.assertEqual(len(self.svc.engine.calls), 1)  # dijawab, bukan "Ya?" kedua
        self.assertNotEqual(self.svc.speaker.said[-1], messages.SAY_CONFIRM)

    def test_halo_brone_text_fallback_says_ya(self):
        self.svc.recorder.wakes = [True]
        self.queue("Alo, BRONE.")
        self.svc.turn()
        self.assertEqual(self.svc.speaker.said, [messages.SAY_CONFIRM, messages.SAY_CONFIRM])
        self.assertEqual(self.svc.engine.calls, [])

    def test_wake_then_silence_goes_back_to_sleep_quietly(self):
        self.svc.recorder.wakes = [True]
        self.svc.session.awaiting_since = 0
        self.svc.turn()  # record() -> None (tidak ada ucapan)
        self.svc.session.awaiting_since -= 100  # jendela 8 s sudah lewat
        self.svc._expire_await()
        self.assertEqual(self.svc.speaker.said, [messages.SAY_CONFIRM])
        self.assertFalse(self.svc.session.awaiting_question)

    def test_name_prefix_in_question_is_stripped(self):
        self.svc.recorder.wakes = [True]
        self.queue("Halo Brone, di mana gedung FILKOM?")
        self.svc.turn()
        self.assertEqual(self.svc.engine.calls, ["di mana gedung FILKOM?"])

    def test_interrupt_cancels_answer_and_reopens_question_window(self):
        svc = self.svc
        svc.kws_interrupt = WakeWord(Path("x.onnx"), 0.5, predictor=lambda chunk: 0.9)
        release = threading.Event()

        def slow_answer(text, stt, **kw):
            release.wait(2)
            return {"route": "faq", "response": "JAWAB LAMBAT", "expression": "happy", "meta": {}}

        svc.engine.answer = slow_answer
        svc.recorder.wakes = [True]
        self.queue("berapa biaya kuliah di UB")
        svc.turn()
        release.set()
        self.assertEqual(svc.speaker.said, [messages.SAY_CONFIRM])  # jawaban lambat tidak diucapkan
        self.assertTrue(svc._woken)
        self.assertEqual(svc.recorder.silences, 1)
        self.queue("kapan wisuda UB berikutnya")
        svc.kws_interrupt = None
        svc.engine = FakeEngine()
        svc.turn()  # langsung "Ya?" tanpa wake word
        self.assertEqual(svc.speaker.said[1:], [messages.SAY_CONFIRM, "JAWAB kapan wisuda UB berikutnya"])

    def test_ask_back_reply_is_accepted_without_wake_word(self):
        svc = self.svc
        replies = iter([
            {"route": "obrolan", "response": "Baik! Kamu sendiri gimana?", "expression": "happy",
             "meta": {}, "ask_back": True},
            {"route": "obrolan", "response": "Syukurlah.", "expression": "happy", "meta": {}},
        ])
        svc.engine.answer = lambda text, stt, **kw: next(replies)
        svc.recorder.wakes = [True]
        self.queue("apa kabar", "aku juga baik")
        svc.turn()
        self.assertTrue(svc.session.social)
        svc.turn()
        self.assertEqual(svc.speaker.said[-1], "Syukurlah.")
        self.assertFalse(svc.session.social)  # kembali ke jendela percakapan biasa
        self.assertTrue(svc.session.awaiting_question)


if __name__ == "__main__":
    unittest.main()
