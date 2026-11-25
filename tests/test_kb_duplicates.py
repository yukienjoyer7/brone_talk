"""Satu kalimat tidak boleh ada di dua sumber (FAQ vs obrolan, atau dua FAQ/obrolan berbeda).

Kalimat kembar membuat pemenang ditentukan selisih pembulatan float (bergantung jumlah thread BLAS), bukan desain
(uji OpenBLAS 12 Nov 2026: 'selamat pagi' ada di FAQ sapaan dan obrolan halo).
"""

import json
import re
import unittest
from collections import defaultdict
from pathlib import Path

import yaml

KB = Path(__file__).resolve().parents[1] / "kb"


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


class KbDuplicates(unittest.TestCase):
    def test_no_duplicate_sentences_across_sources(self):
        seen = defaultdict(set)
        for line in (KB / "faq.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                f = json.loads(line)
                for t in [f["q"], *f.get("variasi", [])]:
                    seen[_norm(t)].add(("faq", f["id"]))
        for o in yaml.safe_load((KB / "smalltalk.yaml").read_text(encoding="utf-8"))["obrolan"]:
            for t in [o["q"], *o.get("variasi", [])]:
                seen[_norm(t)].add(("obrolan", o["id"]))
        dup = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
        self.assertEqual({}, dup)


if __name__ == "__main__":
    unittest.main()
