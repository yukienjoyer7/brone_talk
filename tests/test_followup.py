"""Konteks satu giliran (engine/context.py): pewarisan atribut/entitas hanya untuk pertanyaan lanjutan yang belum lengkap."""

import json
import unittest
from pathlib import Path

from brone_talk.engine.context import FollowUpContext
from brone_talk.engine.entities import EntityMatcher

KB = Path(__file__).resolve().parents[1] / "kb"
ENT = json.loads((KB / "entities.json").read_text(encoding="utf-8"))


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


FAQS = [json.loads(line) for line in (KB / "faq.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def make(clock=None, faqs=None):
    return FollowUpContext(ENT, EntityMatcher(ENT), ttl_s=90, max_words=7, clock=clock or Clock(), faqs=faqs)


def after(ctx, question, route="faq"):
    """Ucapan lengkap yang dijawab -> slot tersimpan."""
    r = ctx.resolve(question)
    ctx.update(r.text, route)
    return r


class FollowUp(unittest.TestCase):
    def test_contoh_pengguna(self):
        c = make()
        self.assertFalse(after(c, "di mana lokasi Filkom?").inherited)
        r = after(c, "kalau FT?")
        self.assertEqual(("atribut", "Di mana FT?"), (r.mode, r.text))

    def test_rantai_entitas_lalu_atribut(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        after(c, "kalau FT?")
        r = after(c, "dekannya siapa?")
        self.assertEqual(("entitas", "Siapa dekan FT?"), (r.mode, r.text))
        r = after(c, "kalau FK?")
        self.assertEqual("Siapa dekan FK?", r.text)

    def test_atribut_saja_mewarisi_entitas(self):
        c = make()
        after(c, "siapa dekan FILKOM?")
        for q, want in [("alamatnya di mana?", "Di mana FILKOM?"),
                        ("ada prodi apa saja di sana?", "Apa saja program studi di FILKOM?")]:
            c.slots = {"entity": "F14", "attr": "dekan", "text": "x", "t": 0.0}
            self.assertEqual(want, c.resolve(q).text, q)

    def test_entitas_dengan_bagaimana_dan_perintah_sopan(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        for q, want in [("FT gimana?", "Di mana FT?"), ("kalau FK bagaimana?", "Di mana FK?")]:
            self.assertEqual(want, c.resolve(q).text, q)
        c.slots = {"entity": "F14", "attr": "lokasi", "text": "x", "t": 0.0}
        self.assertEqual("Bagaimana sejarah FILKOM?", c.resolve("ceritakan sejarahnya").text)
        self.assertEqual("Bagaimana sejarah FILKOM?", c.resolve("kapan berdirinya?").text)
        self.assertFalse(c.resolve("siapa FT?").inherited)

    def test_grid_memilih_pertanyaan_utama_faq(self):
        c = make(faqs=FAQS)
        self.assertEqual("Apa saja program studi di Fakultas Ilmu Komputer?", c.grid[("prodi", "F14")])
        self.assertEqual("Bagaimana sejarah FILKOM?", c.grid[("sejarah", "F14")])
        self.assertEqual("Di mana Fakultas Teknik?", c.grid[("lokasi", "F08")])
        after(c, "di mana lokasi Filkom?")
        c.slots = {"entity": "F08", "attr": "prodi", "text": "x", "t": 0.0}
        self.assertEqual("Apa saja program studi di Fakultas Ilmu Komputer?", c.resolve("kalau FILKOM?").text)

    def test_fakultas_dan_sedangkan_sebagai_pengisi(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        self.assertEqual("Di mana FT?", c.resolve("kalau fakultas FT?").text)
        self.assertEqual("Di mana FT?", c.resolve("sedangkan Fakultas Teknik?").text)
        self.assertFalse(c.resolve("fakultas apa saja?").inherited)

    def test_nya_terpisah_dan_tempat(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        self.assertEqual("Di mana FT?", c.resolve("FT nya?").text)
        c.slots = {"entity": "F08", "attr": "dekan", "text": "x", "t": 0.0}
        self.assertEqual("Di mana FT?", c.resolve("tempatnya di mana?").text)

    def test_penanda_lanjutan(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        for q, want in [("dan dekan?", "Siapa dekan FILKOM?"),
                        ("kalau prodinya apa saja?", "Apa saja program studi di FILKOM?"),
                        ("sejarahnya bagaimana?", "Bagaimana sejarah FILKOM?")]:
            c.slots = {"entity": "F14", "attr": "lokasi", "text": "x", "t": 0.0}
            self.assertEqual(want, c.resolve(q).text, q)

    def test_lengkap_tidak_mewarisi(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        for q in ["siapa dekan FT?", "di mana Fakultas Hukum?", "apa moto UB?", "siapa rektor UB sekarang?",
                  "mahasiswa FT berapa?", "mahasiswa FT", "terima kasih", "apa itu UKT?", "siapa dekan?",
                  "apa saja prodi?", "berapa biaya kuliah?",
                  "kalau FT dan FK?", "siapa dekan dan alamat FT?"]:
            self.assertFalse(c.resolve(q).inherited, q)

    def test_pertanyaan_universitas_menghapus_konteks(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        after(c, "apa moto UB?")
        self.assertFalse(c.resolve("kalau FT?").inherited)

    def test_basa_basi_tidak_mengubah_slot(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        c.update("halo", "obrolan")
        self.assertTrue(c.resolve("kalau FT?").inherited)

    def test_tanpa_konteks_atau_kedaluwarsa(self):
        clock = Clock()
        c = make(clock)
        self.assertFalse(c.resolve("kalau FT?").inherited)
        after(c, "di mana lokasi Filkom?")
        clock.t = 91.0
        self.assertFalse(c.resolve("kalau FT?").inherited)

    def test_reset(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        c.reset()
        self.assertFalse(c.resolve("kalau FT?").inherited)

    def test_entitas_sama_tidak_diulang(self):
        c = make()
        after(c, "di mana lokasi Filkom?")
        self.assertFalse(c.resolve("kalau FILKOM?").inherited)

    def test_atribut_tanpa_template_mengganti_sebutan_entitas(self):
        c = make()
        after(c, "apa akreditasi FILKOM?")
        r = c.resolve("kalau FT?")
        self.assertEqual("apa akreditasi ft?", r.text.lower())


if __name__ == "__main__":
    unittest.main()
