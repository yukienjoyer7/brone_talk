"""Rute engine (FAQ dan obrolan) harus sama dengan produksi untuk 60 pertanyaan eval.

Butuh server embedding (127.0.0.1:8082) yang sedang hidup.
Fixture tests/golden/engine_routes.jsonl tidak di-commit. Tanpa fixture atau server, tes dilewati.
"""

import json
import os
import socket
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "engine_routes.jsonl"
KB_DIR = Path(os.getenv("BRONE_KB_DIR", ROOT / "kb"))


def _server_up(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


@unittest.skipUnless(GOLDEN.is_file(), "fixture golden belum dibangkitkan (lihat scripts/make_golden.py)")
@unittest.skipUnless(_server_up(8082), "server embedding (8082) tidak hidup")
class EngineRoutesGolden(unittest.TestCase):
    def test_routes_match_production(self):
        from brone_talk.engine.answer import AnswerEngine, EngineConfig

        runtime = ROOT / "runtime" / "test-engine"
        runtime.mkdir(parents=True, exist_ok=True)
        engine = AnswerEngine(EngineConfig(kb_dir=KB_DIR, runtime_dir=runtime, mode="faq"))
        engine.start()
        rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
        mismatches = []
        for r in rows:
            res = engine.answer(r["input"], None)
            engine.reset()
            faq_id = (res.get("meta", {}).get("faq") or {}).get("id")
            if res["route"] != r["route"] or faq_id != r["faq_id"]:
                mismatches.append((r["input"], r["route"], res["route"], r["faq_id"], faq_id))
        self.assertEqual([], mismatches[:5], f"{len(mismatches)} dari {len(rows)} rute berbeda")


if __name__ == "__main__":
    unittest.main()
