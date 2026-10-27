"""Normalisasi teks untuk suara fine-tune D21 (models/voices/brone_D21_teks). Tanpa folder itu, tes dilewati."""

import unittest
from pathlib import Path

from brone_talk.tts.lexnorm import load_spoken

TEKS = Path(__file__).resolve().parents[1] / "models" / "voices" / "brone_D21_teks"


@unittest.skipUnless((TEKS / "tools" / "textnorm.py").is_file(), "textnorm D21 belum dipasang")
class SpokenD21(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spoken = staticmethod(load_spoken(TEKS))

    def test_lexicon_acronyms(self):
        self.assertEqual("U be terakreditasi Unggul dari be-a-en-pe-te.", self.spoken("UB terakreditasi Unggul dari BAN-PT."))
        self.assertIn("es en be pe", self.spoken("Jalur SNBP dibuka."))

    def test_gigi_and_website(self):
        self.assertIn("Kedokteran gi gi", self.spoken("Fakultas Kedokteran Gigi"))
        self.assertIn("u be dot a ce dot i de", self.spoken("Buka ub.ac.id ya."))

    def test_iya_alone(self):
        self.assertEqual("I yaa!", self.spoken("Iya."))
        self.assertEqual("Betul. I yaa! Ada di gedung ef.", self.spoken("Betul. Iya. Ada di gedung F."))
        self.assertEqual("I yaa!", self.spoken("Ya?"))
        self.assertEqual("Iya, betul.", self.spoken("Iya, betul."))
        self.assertIn("ya", self.spoken("Sebentar ya. Bron carikan."))

    def test_markdown_and_old_map(self):
        out = self.spoken("**Bank BRI** ada di dekat UGM.")
        self.assertNotIn("*", out)
        self.assertIn("be er i", out)
        self.assertIn("u ge em", out)


if __name__ == "__main__":
    unittest.main()
