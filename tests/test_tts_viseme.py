import unittest

from brone_talk.tts.viseme import schedule, speech_payload

SR = 22050
VALID = {"A", "I", "U", "E", "O", "REST", "DIAM"}  # VALID_VISEME BroneGUI-Tauri (protocol.rs)


class VisemeScheduleTest(unittest.TestCase):
    def test_times_follow_alignment_samples(self):
        # "^ b ˈ a $": bibir rapat, lalu A tepat setelah b (tekanan ˈ digabung ke vokal), REST di akhir kalimat
        al = [("^", 2205), ("b", 441), ("ˈ", 441), ("a", 882), ("$", 441)]
        self.assertEqual(schedule([al], SR), [
            {"t": 0.0, "v": "REST"}, {"t": 0.12, "v": "A"}, {"t": 0.18, "v": "REST"}, {"t": 0.2, "v": "DIAM"}])

    def test_onset_takes_next_vowel_coda_keeps_previous(self):
        # "kan tu": k ikut a (pembuka suku kata), n ikut a (penutup sebelum spasi), t ikut u
        al = [("^", 441), ("k", 441), ("a", 441), ("n", 441), (" ", 441), ("t", 441), ("u", 441), ("$", 441)]
        self.assertEqual([x["v"] for x in schedule([al], SR)], ["REST", "A", "U", "REST", "DIAM"])
        self.assertEqual(schedule([al], SR)[1]["t"], 0.02)  # A mulai di k, bukan di a

    def test_bilabial_between_vowels_closes_lips(self):
        al = [("a", 441), ("m", 441), ("a", 441)]
        self.assertEqual([x["v"] for x in schedule([al], SR)], ["A", "REST", "A", "DIAM"])

    def test_chunks_accumulate(self):
        c1, c2 = [("^", 441), ("a", 441), ("$", 441)], [("^", 441), ("o", 441), ("$", 441)]
        j = schedule([c1, c2], SR)
        self.assertIn({"t": 0.08, "v": "O"}, j)  # 3 * 441 + 441 sampel
        self.assertEqual(j[-1], {"t": 0.12, "v": "DIAM"})

    def test_offset_shifts_and_clamps(self):
        al = [("^", 441), ("a", 441), ("$", 441)]
        self.assertEqual([x["t"] for x in schedule([al], SR, offset_s=0.1)], [0.1, 0.12, 0.14, 0.16])
        self.assertEqual(schedule([al], SR, offset_s=-0.05)[0]["t"], 0.0)

    def test_only_gui_codes_and_sorted(self):
        al = [("^", 300), ("ɡ", 200), ("ˈ", 100), ("ɛ", 300), ("d", 200), ("u", 100), ("ŋ", 300), (" ", 100),
              ("r", 100), ("ə", 100), ("ʃ", 100), ("ɔ", 100), ("p", 100), ("i", 100), ("?", 100), ("$", 100)]
        j = schedule([al], SR)
        self.assertTrue({x["v"] for x in j} <= VALID)
        self.assertEqual([x["t"] for x in j], sorted(x["t"] for x in j))
        self.assertTrue(all(a["v"] != b["v"] for a, b in zip(j, j[1:])))

    def test_empty(self):
        self.assertEqual(schedule([], SR), [])

    def test_payload_expression(self):
        self.assertEqual(speech_payload("happy", 1.234, [])["expression"], "happy")
        self.assertEqual(speech_payload("Neutral", 1.0, [])["expression"], "Neutral")
        # nama yang tidak dikenal GUI membuat seluruh pesan ditolak -> "speaking" agar jadwal tetap jalan
        self.assertEqual(speech_payload("bangga", 1.0, [])["expression"], "speaking")
        self.assertEqual(speech_payload("", 1.0, [])["duration"], 1.0)


if __name__ == "__main__":
    unittest.main()
