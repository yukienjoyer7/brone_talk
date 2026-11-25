"""Perbaikan dari uji cepat TTS 9 Nov 2026 (SYM-002 ekstensi, SYM-003 tanggal, SYM-008 simbol, SYM-015 kenyaringan)."""

import unittest

import numpy as np

from brone_talk.tts.loudness import PEAK_DBFS, TARGET_LUFS, integrated_loudness, level_for_playback
from brone_talk.tts.normalize import clean_for_speech, normalize_for_speech, strip_markdown


class SymbolStripping(unittest.TestCase):  # SYM-008
    def test_emoji_and_symbols_removed(self):
        cases = {
            "Terima kasih sudah bertanya 😊": "Terima kasih sudah bertanya",
            "Halo! 👋🤖 Saya Brone.": "Halo! Saya Brone.",
            "✅ Sudah dibuka. ❌ Belum tersedia.": "Sudah dibuka. Belum tersedia.",
            "Silakan <b>datang</b> besok.": "Silakan datang besok.",
            "~~Gratis~~ Biaya sepuluh ribu.": "Gratis Biaya sepuluh ribu.",
            "_Mohon_ antre dengan tertib.": "Mohon antre dengan tertib.",
            "Langkah → isi formulir → serahkan.": "Langkah, isi formulir, serahkan.",
            "Ok ^_^": "Ok",
        }
        for src, want in cases.items():
            self.assertEqual(strip_markdown(src), want, src)

    def test_table_becomes_pauses(self):
        self.assertEqual(strip_markdown("Lihat tabel:\n| Hari | Jam |\n|---|---|\n| Senin | 08.00 |"),
                         "Lihat tabel: Hari, Jam, Senin, 08.00")

    def test_identifiers_kept(self):
        self.assertEqual(strip_markdown("Gunakan `kode_akses` yang diberikan."), "Gunakan kode_akses yang diberikan.")

    def test_tilde_before_amount_is_sekitar(self):
        self.assertEqual(clean_for_speech("Harga ~Rp20.000"), "Harga sekitar 20.000 rupiah")


class DatesAndExtensions(unittest.TestCase):  # SYM-003, SYM-002
    def test_slash_date(self):
        self.assertEqual(normalize_for_speech("Acara tanggal 6/1/2030 di aula."), "Acara tanggal 6 Januari 2030 di aula.")

    def test_not_a_date(self):
        self.assertEqual(normalize_for_speech("Tahun akademik 2026/2027."), "Tahun akademik 2026/2027.")
        self.assertEqual(normalize_for_speech("tanggal 45/13/2030"), "tanggal 45/13/2030")

    def test_extension_lists_spelled_consistently(self):
        self.assertEqual(normalize_for_speech("ekstensi 117 dan 118"), "ekstensi satu satu tujuh dan satu satu delapan")
        self.assertEqual(normalize_for_speech("ekstensi 117/118"), "ekstensi satu satu tujuh atau satu satu delapan")

    def test_phone_after_extension_untouched(self):
        out = normalize_for_speech("ekstensi 117 dan 118, atau langsung 0341-575754.")
        self.assertIn("satu satu delapan, atau langsung kosong tiga empat satu", out)


class Loudness(unittest.TestCase):  # SYM-015
    def test_level_and_peak(self):
        fs = 22050
        t = np.arange(int(fs * 3)) / fs
        x = (0.99 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        y = level_for_playback(x, fs)
        self.assertLessEqual(20 * np.log10(np.max(np.abs(y))), PEAK_DBFS + 1e-6)
        self.assertLessEqual(integrated_loudness(y, fs), TARGET_LUFS + 0.05)

    def test_quiet_speech_with_spike_reaches_target(self):  # 17 Nov: limiter, kriteria +-1 LU
        fs = 22050
        t = np.arange(int(fs * 3)) / fs
        x = 0.1 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        need = 10 ** ((TARGET_LUFS - integrated_loudness(x, fs)) / 20)
        x[fs] = 1.4 * 10 ** (PEAK_DBFS / 20) / need  # puncak sesaat +3 dB di atas batas setelah disetarakan
        y = level_for_playback(x.astype(np.float32), fs)
        self.assertLessEqual(20 * np.log10(np.max(np.abs(y))), PEAK_DBFS + 1e-6)
        self.assertLess(abs(integrated_loudness(y, fs) - TARGET_LUFS), 1.0)  # versi 9 Nov: ~3 LU di bawah target

    def test_silence_unchanged(self):
        z = np.zeros(1000, dtype=np.float32)
        self.assertTrue(np.array_equal(level_for_playback(z, 22050), z))


if __name__ == "__main__":
    unittest.main()
