"""Modul nlu.wake harus sama dengan produksi untuk seluruh masukan fixture nlu_wake.jsonl.

Fixture dibangkitkan lokal (scripts/make_golden.py) dan tidak di-commit. Tanpa fixture, tes dilewati.
"""

import json
import unittest
from pathlib import Path

from brone_talk.nlu import wake

GOLDEN = Path(__file__).resolve().parent / "golden" / "nlu_wake.jsonl"


@unittest.skipUnless(GOLDEN.is_file(), "fixture golden belum dibangkitkan (lihat scripts/make_golden.py)")
class WakeGolden(unittest.TestCase):
    def test_matches_production_output(self):
        rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
        mismatches = []
        for r in rows:
            got = {
                "called": bool(wake.NAME_RE.search(r["input"])),
                "collapsed": wake.collapse_repeats(r["input"]),
                "wake_question": wake.wake_question(wake.collapse_repeats(r["input"])),
                "clock": wake.clock_answer(r["input"], now=1_760_000_000.0),
                "stop": wake.matches(wake.STOP_PATTERNS, r["input"]),
            }
            for key, value in got.items():
                if value != r[key]:
                    mismatches.append((r["input"], key, r[key], value))
        self.assertEqual([], mismatches[:5], f"{len(mismatches)} selisih dari {len(rows)} masukan")


class CallText(unittest.TestCase):
    def test_call_starts_at_last_calling_sentence(self):
        self.assertEqual(wake.call_text("Satu, bentar. Oke. BRONE, siapa kamu?"), "BRONE, siapa kamu?")
        self.assertEqual(wake.call_text("Ya, Bey.\nBRONE, di mana gedung UB?"), "BRONE, di mana gedung UB?")
        self.assertEqual(wake.call_text("Brone, UB ada di mana?"), "Brone, UB ada di mana?")

    def test_name_inside_chatter_is_not_a_call(self):
        for text in ("Enggak, BRONE. Saya belum datang ke FIB.\nOh, iya.",
                     "Paling based on BRONE kali. Oke.", "Loh, tuh di BRONE tuh. Oke, sentar aja ke sana."):
            self.assertEqual(wake.call_text(text), "", text)

    def test_name_at_head_or_tail_is_a_call(self):
        for text in ("Brone, UB ada di mana?", "Halo Brone", "halo, BRONE. Jam berapa sekarang?",
                     "UB ada di mana, Brone?", "UB ada di mana Brone", "BRONE FILKOM itu apa"):
            self.assertTrue(wake.addressed(text), text)

    def test_name_mid_sentence_is_not_a_call(self):
        for text in ("kode BRONE talk itu", "Ada 2 BRONE ini sih", "Aku UB, BRONE, Kalian Masih Disini",
                     "Siapa yang membuat Brone tadi?", "saya mau makan siang", ""):
            self.assertFalse(wake.addressed(text), text)


if __name__ == "__main__":
    unittest.main()


class SessionEnd(unittest.TestCase):
    """10 Nov 2026: kasus dari uji sintesis runtime/uji-end-wake (STT-E-04, E-06, E-10)."""

    def ends(self, text):
        return bool(wake.SESSION_END_RE.search(text))

    def test_end_phrase_with_short_fillers(self):
        for text in ("selesai brone", "Selesai, BRONE.", "selesai bron", "Oke selesai, BRONE.",
                     "Sudah selesai BRONE!", "Terima kasih, selesai BRONE.", "Selesai, BRONE. Terima kasih.",
                     "Baik BRONE, selesai BRONE.", "Selesai BRONE, sampai jumpa.", "Ya, selesai BRONE ya."):
            self.assertTrue(self.ends(text), text)

    def test_whisper_spelling_variants(self):
        for text in ("Selasai, BRONE.", "Solasai BRONE.", "Silasai BRONE.", "Selosai, BRONE.", "Ia selesai, BRONE.",
                     "Terima kasih, Sosai BRONE.", "Oke selasai, BRONE."):
            self.assertTrue(self.ends(text), text)

    def test_question_with_selesai_and_name_does_not_end(self):
        for text in ("Kapan skripsi saya selesai, BRONE?", "Berapa lama kuliah di UB sampai selesai, BRONE?",
                     "BRONE, berapa lama kuliah UB sampai selesai, BRONE ke Saibuan?",
                     "Sudah saya sudah selesai, BRONE."):
            self.assertFalse(self.ends(text), text)

    def test_own_greeting_echo_does_not_end(self):
        for text in ("Hai, BRONE di sini, kalau selesai tinggal, bilang selesai BRONE.",
                     "Kalau selesai tinggal bilang selesai BRONE.", "Sawa selesai kundang dan selesai BRONE."):
            self.assertFalse(self.ends(text), text)

    def test_without_name_does_not_end(self):
        for text in ("selesai", "Selesai.", "diam", "stop", "sampai jumpa", "BRONE", "halo brone"):
            self.assertFalse(self.ends(text), text)


class SessionEndFuzzy(unittest.TestCase):
    """20 Nov 2026: salah dengar Whisper dari runtime/uji-end-wake yang dulu tidak mengakhiri sesi."""

    def test_misheard_end_word_and_name(self):
        for text in ("Kelasai BRONE.", "Selakai BRONE, terima kasih.", "Baik, BRONE, lakai, BRONE.", "Selatai, BRONE.",
                     "Ya, Suasai BRONE, ya.", "Kelastai BRONE sampai jumpa.", "Filasai BRONE.", "Selasai Broning.",
                     "Selasai Bronie, terima kasih.", "Yasilasai, Broneya.", "Ya, selesai, Bronya.",
                     "Selasai, BRONE, Universitas Brawijaya.", "Selasai BRONE, Santai Jinta.",
                     "Bike BRONE, Selasai BRONE.", "Sangat selesai, BRONE.", "Terima kasih, Lesai BRONE.",
                     "Selesai, BRONE, ya selesai, BRONE, ya selesai, BRONE, ya selesai, BRONE.", "brone selesai"):
            self.assertTrue(wake.is_session_end(text), text)

    def test_common_ai_words_and_questions_do_not_end(self):
        for text in ("Sampai jumpa BRONE", "Mulai BRONE", "Pakai BRONE", "Nilai BRONE berapa?", "Kapan selesai, BRONE?",
                     "Sama saya, BRONE.", "Terima kasih, BRONE.", "saya selesai brone", "BRONE sudah selesai belum?",
                     "Siapa yang membuat BRONE selesai?", "Jam berapa kuliah selesai, BRONE?"):
            self.assertFalse(wake.is_session_end(text), text)


class ThanksOnlyIgnored(unittest.TestCase):
    """20 Nov 2026: STT "terima kasih" saja = derau/salah dengar -> dibuang."""

    def test_thanks_only_is_garbage(self):
        from brone_talk.stt.filters import stt_garbage_reason
        for text in ("Terima kasih.", "terima kasih", "Terima kasih, BRONE.", "Terima kasih banyak!", "Makasih ya.",
                     "Thank you.", "Terima kasih. Terima kasih.", "BRONE terima kasih"):
            self.assertEqual(stt_garbage_reason(text), "hanya terima kasih", text)

    def test_thanks_with_content_passes(self):
        from brone_talk.stt.filters import stt_garbage_reason
        for text in ("Terima kasih, selesai BRONE.", "Terima kasih infonya, BRONE, gedung FILKOM di mana?"):
            self.assertNotEqual(stt_garbage_reason(text), "hanya terima kasih", text)
