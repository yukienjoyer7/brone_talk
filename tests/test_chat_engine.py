"""Obrolan santai: isi kb/smalltalk.yaml dan pemilahan jawaban LLM obrolan (tanpa server)."""

import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from brone_talk import messages
from brone_talk.engine.answer import AnswerEngine, EngineConfig, is_motion_command
from brone_talk.engine.faq import load_smalltalk

ROOT = Path(__file__).resolve().parents[1]


class SmalltalkFile(unittest.TestCase):
    def setUp(self):
        self.rows = load_smalltalk(ROOT / "kb" / "smalltalk.yaml")

    def test_entries_are_complete_and_unique(self):
        self.assertGreater(len(self.rows), 30)
        ids = [r["id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        for r in self.rows:
            self.assertTrue(r["q"] and r["jawaban"] and all(r["jawaban"]), r["id"])
            self.assertIn(r["ekspresi"], {"happy", "neutral", "sad", "excited", "confused"}, r["id"])

    def test_answers_hold_no_numbers(self):
        # fakta (angka, tanggal) hanya boleh ada di knowledge.yaml
        for r in self.rows:
            for a in r["jawaban"]:
                self.assertIsNone(re.search(r"\d", a), f"{r['id']}: {a}")

    def test_ask_back_answers_end_with_question(self):
        for r in self.rows:
            if r["tanya_balik"]:
                self.assertTrue(all(a.endswith("?") for a in r["jawaban"]), r["id"])


class FakeBackend:
    name = "cloud"

    def __init__(self, raw):
        self.raw, self.messages = raw, None

    def chat(self, messages, temperature=0.1):
        self.messages = messages
        return self.raw, {"ms": 1}


class ChatOrRefuse(unittest.TestCase):
    def engine(self, raw, mode="auto"):
        tmp = Path(tempfile.mkdtemp())
        eng = object.__new__(AnswerEngine)
        eng.cfg = EngineConfig(kb_dir=ROOT / "kb", runtime_dir=tmp, mode=mode)
        eng.cloud, eng.local = FakeBackend(raw), None
        eng.turn_log = tmp / "turns.jsonl"
        return eng

    def run_chat(self, raw, text="kamu lagi apa", replying_to="", mode="auto"):
        eng = self.engine(raw, mode)
        return eng, eng._chat_or_refuse(text, replying_to, "tolak_tidak_ada_info", {}, 0.0)

    def test_small_talk_is_answered(self):
        _, res = self.run_chat('{"jenis": "obrolan", "response": "Bron lagi siaga nih! Kamu sendiri?", '
                               '"expression": "happy", "tanya_balik": true}')
        self.assertEqual(res["route"], "obrolan_cloud")
        self.assertTrue(res["ask_back"])

    def test_info_request_is_still_refused(self):
        _, res = self.run_chat('{"jenis": "info", "response": ""}', text="kapan FILKOM berdiri")
        self.assertEqual((res["route"], res["response"]), ("tolak_tidak_ada_info", messages.MSG_NO_INFO))

    def test_chatter_is_ignored(self):
        _, res = self.run_chat('{"jenis": "abaikan", "response": ""}', text="jalanin lagi dari awal")
        self.assertEqual((res["route"], res["response"]), ("abaikan", ""))

    def test_made_up_names_or_numbers_are_blocked(self):
        for answer in ("Bron dibuat oleh Pak Budi Santoso.", "Bron sudah berumur 3 tahun!"):
            _, res = self.run_chat(f'{{"jenis": "obrolan", "response": "{answer}"}}')
            self.assertEqual(res["route"], "tolak_grounding_obrolan", answer)
            self.assertEqual(res["response"], messages.MSG_CHAT_FALLBACK)

    def test_reply_turn_never_asks_back_and_passes_context(self):
        eng, res = self.run_chat('{"jenis": "obrolan", "response": "Syukurlah! Kamu senang?", "tanya_balik": true}',
                                 text="aku baik", replying_to="Kamu sendiri gimana kabarnya?")
        self.assertFalse(res["ask_back"])
        self.assertIn("BRON SEBELUMNYA BERKATA: Kamu sendiri gimana kabarnya?", eng.cloud.messages[1]["content"])

    def test_self_reference_is_bron(self):
        _, res = self.run_chat('{"jenis": "obrolan", "response": "Aku robot, jadi saya tidak main bola."}')
        self.assertEqual(res["response"], "Bron robot, jadi Bron tidak main bola.")

    def test_cloud_failure_never_uses_local_for_chat_or_rag(self):
        for failure in (OSError("offline"), "invalid JSON"):
            with self.subTest(failure=failure):
                eng = self.engine("")
                eng.cloud.chat = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(
                    return_value=(failure, {}))
                eng.local = Mock()
                chat = eng._chat_or_refuse("apa kabar", "", "tolak_tidak_ada_info", {}, 0.0)
                self.assertEqual(chat["response"], messages.MSG_NO_INFO)
                eng.retriever = SimpleNamespace(search=lambda *a, **kw: [0],
                                                chunks=[{"_doc": "UB", "id": "1", "judul": "UB", "teks": "UB"}])
                eng.reranker = SimpleNamespace(score=lambda *a: [1.0])
                eng.prev_question = ""
                rag = eng._rag_answer("tentang UB", None, "", {}, 0.0)
                self.assertIn(rag["response"], (messages.MSG_SERVICE_DOWN, messages.MSG_NOT_UNDERSTOOD))
                eng.local.chat.assert_not_called()

    def test_missing_cloud_refuses_rag(self):
        eng = self.engine("")
        eng.cloud = None
        eng.local = Mock()
        eng.retriever = SimpleNamespace(search=lambda *a, **kw: [0],
                                        chunks=[{"_doc": "UB", "id": "1", "judul": "UB", "teks": "UB"}])
        eng.reranker = SimpleNamespace(score=lambda *a: [1.0])
        eng.prev_question = ""
        result = eng._rag_answer("tentang UB", None, "", {}, 0.0)
        self.assertEqual(result["response"], messages.MSG_SERVICE_DOWN)
        eng.local.chat.assert_not_called()

    def test_explicit_local_mode_still_uses_local(self):
        eng = self.engine("", mode="local")
        eng.local = FakeBackend('{"jenis": "obrolan", "response": "Halo!"}')
        result = eng._chat_or_refuse("apa kabar", "", "tolak", {}, 0.0)
        self.assertEqual(result["response"], "Halo!")
        self.assertIsNotNone(eng.local.messages)
        self.assertIsNone(eng.cloud.messages)

    def test_faq_mode_skips_llm(self):
        eng, res = self.run_chat('{"jenis": "obrolan", "response": "Halo!"}', mode="faq")
        self.assertEqual(res["route"], "tolak_tidak_ada_info")
        self.assertIsNone(eng.cloud.messages)



class MotionCommand(unittest.TestCase):
    def test_short_commands_are_motion(self):
        for text in ("lambaikan tangan", "coba berdiri dong", "ikut aku", "duduk", "tolong melambai", "wave"):
            self.assertTrue(is_motion_command(text), text)

    def test_questions_and_chatter_are_not_motion(self):
        # 20 Okt 2026: dulu semua ini dijawab "kontrol gerakan Bron belum diaktifkan"
        for text in ("jalanin lagi", "jalan ke FILKOM lewat mana", "sejarah berdiri FILKOM", "kapan FILKOM berdiri",
                     "kamu bisa melambai?", "jalan-jalan yuk", "ikut UKM yang seru", "tempat duduk perpustakaan",
                     "bisakah kamu duduk"):
            self.assertFalse(is_motion_command(text), text)


if __name__ == "__main__":
    unittest.main()
