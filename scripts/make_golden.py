#!/usr/bin/env python3
"""Bekukan perilaku pipeline lama sebagai fixture golden (Fase 0).

Dijalankan sekali dengan venv versi lama, hasilnya di-commit ke tests/golden/.
Refactor harus menghasilkan keluaran yang sama dengan fixture ini.

  ~/brone-talk/.venv/bin/python scripts/make_golden.py

Sumber masukan:
  - runtime/turns.jsonl versi v4 (teks STT dan jawaban nyata)
  - ~/brone-talk/dataset_week1.jsonl (prompt bersih)
  - ~/brone-talk-v4/eval/questions.jsonl (60 pertanyaan dengan rute yang diharapkan)

Rute engine direkam dengan BRONE_MODE=faq, sehingga tidak ada panggilan LLM. Skor gerbang
reranker dihitung terpisah untuk setiap pertanyaan agar bisa dibandingkan tanpa LLM.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LEGACY = Path(os.getenv("BRONE_LEGACY_DIR", Path.home() / "brone-talk-v4"))
OUT = Path(__file__).resolve().parents[1] / "tests" / "golden"
RUNTIME = Path(__file__).resolve().parents[1] / "runtime" / "golden-engine"

sys.path.insert(0, str(LEGACY))
os.environ.setdefault("BRONE_MODE", "faq")  # tanpa LLM saat merekam rute

import brone_talk as bt  # noqa: E402  (modul lama, sumber kebenaran Fase 0)
import brone_v4 as v4  # noqa: E402


def write_jsonl(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{name}: {len(rows)} baris")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_texts() -> tuple[list[str], list[str], list[str]]:
    turns = load_jsonl(LEGACY / "runtime" / "turns.jsonl")
    stt_texts = {t["text"] for t in turns if t.get("text")}
    replies = {t["response"] for t in turns if t.get("response")}
    dataset = load_jsonl(Path.home() / "brone-talk" / "dataset_week1.jsonl")
    prompts = {d["prompt"] for d in dataset} | {d["stt_ground_truth"] for d in dataset}
    evals = [q["q"] for q in load_jsonl(LEGACY / "eval" / "questions.jsonl")]
    return sorted(stt_texts | prompts | set(evals)), sorted(replies), evals


def golden_tts(replies: list[str]) -> None:
    rows = [{"input": r, "expected": bt.clean_for_speech(r)} for r in replies]
    write_jsonl("tts_normalize.jsonl", rows)


def golden_nlu(texts: list[str]) -> None:
    rows = []
    for t in texts:
        collapsed = bt.collapse_repeats(t)
        rows.append({
            "input": t,
            "called": bool(bt._NAME_RE.search(t)),
            "collapsed": collapsed,
            "wake_question": bt.wake_question(collapsed),
            "clock": bt.clock_answer(t, now=1_760_000_000.0),  # waktu tetap agar deterministik
            "stop": bt.matches(bt.STOP_PATTERNS, t),
        })
    write_jsonl("nlu_wake.jsonl", rows)


def golden_stt_filters(texts: list[str]) -> None:
    rows = []
    for t in texts:
        rows.append({
            "input": t,
            "stt_fix": bt._stt_fix(t),
            "mai_fix": bt._mai_fix(t),
            "garbage_reason": bt.stt_garbage_reason(t),
            "strip_prompt": bt.stt_strip_prompt_sentence(t),
        })
    write_jsonl("stt_filters.jsonl", rows)


def golden_engine_routes(evals: list[str]) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    cfg = v4.EngineConfig(kb_dir=LEGACY / "kb", runtime_dir=RUNTIME, mode="faq")
    engine = v4.AnswerEngine(cfg)
    engine.start()
    expect = {q["q"]: q for q in load_jsonl(LEGACY / "eval" / "questions.jsonl")}
    rows = []
    for text in evals:
        res = engine.answer(text, None)
        meta = res.get("meta", {})
        engine.reset()
        qvec = engine.embedder.embed([text])[0]
        ranked = engine.retriever.search_scored(qvec, top=1)
        top_sim = round(ranked[0][1], 4) if ranked else None
        rows.append({
            "input": text,
            "expect": expect[text]["expect"],
            "route": res["route"],
            "faq_id": (meta.get("faq") or {}).get("id"),
            "faq_score": (meta.get("faq") or {}).get("score"),
            "top_chunk_sim": top_sim,
            "chunk_gate_passes": top_sim is not None and top_sim >= cfg.chunk_min_sim,
        })
    write_jsonl("engine_routes.jsonl", rows)


def main() -> int:
    texts, replies, evals = collect_texts()
    print(f"masukan: {len(texts)} teks STT/prompt, {len(replies)} jawaban, {len(evals)} eval")
    golden_tts(replies)
    golden_nlu(texts)
    golden_stt_filters(texts)
    golden_engine_routes(evals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
