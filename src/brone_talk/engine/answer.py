"""Mesin jawaban: FAQ, obrolan santai, permintaan gerakan, lalu RAG (retrieval + reranker) + LLM + grounding.

Urutan per ucapan ada di AnswerEngine.answer; setiap jalur mengembalikan dict lewat _result dengan
`route` yang dicatat di runtime/turns.jsonl.
"""

from __future__ import annotations

import datetime as _dt
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from brone_talk.engine.util import norm_text, wait_for_server
from brone_talk.engine.servers import Embedder, Reranker
from brone_talk.engine.entities import EntityMatcher
from brone_talk.engine.faq import FaqIndex, load_smalltalk
from brone_talk.engine.retrieval import HybridRetriever
from brone_talk.engine.llm import CHAT_PROMPT, LLMBackend, SYSTEM_PROMPT, parse_llm_json
from brone_talk.engine.grounding import grounding_problems
from brone_talk.messages import MSG_CHAT_FALLBACK, MSG_MOTION, MSG_NOT_UNDERSTOOD, MSG_NO_INFO, MSG_SERVICE_DOWN
from brone_talk.engine.util import LOG

# Permintaan gerakan = perintah pendek yang DIAWALI kata gerak ("lambaikan tangan", "coba berdiri dong").
# 20 Okt 2026: dulu kata gerak di mana saja, termasuk "jalan\w*" dan "ikut\w*", jadi "jalanin lagi",
# "jalan ke FILKOM lewat mana" dan "sejarah berdiri FILKOM" ditolak sebagai perintah gerakan.
_MOTION_RE = re.compile(
    r"(?:me)?(?:lambai\w*|duduk|berdiri|jalan|berjalan|wave|sit|stand|follow)\b(?! jalan)"
    r"|ikut(?:i|lah)? (?:aku|saya|kami|kita)\b")
_MOTION_FILLER = {"tolong", "coba", "ayo", "yuk", "ayuk", "dong", "deh", "sih", "please", "kamu", "sekarang",
                  "boleh", "bron", "brone"}
_MOTION_QUESTION_RE = re.compile(
    r"\b(bisa|dapat|mampu|apakah|\w+kah|apa|siapa|mana|dimana|kemana|berapa|kapan|gimana|bagaimana"
    r"|kenapa|mengapa|lewat)\b", re.I)

_EXPRESSIONS = {"happy", "neutral", "sad", "excited", "confused"}

# permintaan hiburan tidak pernah dikarang LLM: kata kunci -> id entri di kb/smalltalk.yaml
_FUN_REQUESTS = [
    ("pantun", re.compile(r"\bpantun\w*", re.I)),
    ("tebak-tebakan", re.compile(r"\b(tebak[- ]?tebakan|teka[- ]?teki)\b", re.I)),
    ("lelucon", re.compile(r"\b(lelucon|jokes?|lawak\w*|melucu|lucu-lucuan|bikin (aku )?ketawa)\b", re.I)),
]

# LLM obrolan kadang tetap memakai "aku"/"saya" untuk dirinya; tim memilih "Bron"
_SELF_RE = re.compile(r"\b(aku|saya)\b", re.I)

# galat backend LLM yang menghasilkan penolakan
_LLM_ERRORS = (OSError, urllib.error.URLError, ValueError, KeyError)


def is_motion_command(text: str) -> bool:
    """Perintah gerak fisik pendek, bukan pertanyaan: "lambaikan tangan", "ikut aku", "coba duduk dong"."""
    if "?" in text or _MOTION_QUESTION_RE.search(text):
        return False
    words = [w for w in norm_text(text).split() if w not in _MOTION_FILLER]
    return 0 < len(words) <= 4 and bool(_MOTION_RE.match(" ".join(words)))


def _expression(value: Any, default: str) -> str:
    expression = str(value or default).lower()
    return expression if expression in _EXPRESSIONS else default


@dataclass
class EngineConfig:
    kb_dir: Path
    runtime_dir: Path
    embed_url: str = "http://127.0.0.1:8082"
    rerank_url: str = "http://127.0.0.1:8083"
    local_url: str = "http://127.0.0.1:8081"
    local_model: str = "lfm2.5-2.6b"
    cloud_url: str = "https://api.deepseek.com/chat/completions"
    cloud_model: str = "deepseek-flash"
    cloud_key: str = ""
    mode: str = "auto"  # auto | cloud | local | faq (diisi dari config.py)
    cloud_timeout_s: float = 5.0  # nilai produksi (lihat docs/baseline-config.md)
    local_timeout_s: float = 25.0
    max_tokens: int = 160
    faq_min_sim: float = 0.80
    faq_min_margin: float = 0.0
    stt_min_mean: float = 0.50
    stt_min_seg: float = 0.40
    # STT kurang yakin -> FAQ tetap boleh, asal kemiripannya setinggi ini (lihat AnswerEngine.answer).
    faq_low_stt_min_sim: float = 0.90
    rerank_min: float = 0.30
    rerank_top: int = 8
    rerank_doc_chars: int = 700
    rerank_timeout_s: float = 10.0
    context_chars: int = 1400
    context_chunks: int = 4
    grounding: bool = True
    smalltalk: bool = True  # kb/smalltalk.yaml: obrolan santai berjawaban tetap
    chat_llm: bool = True  # obrolan di luar daftar dijawab LLM persona (bukan mode faq)
    startup_wait_s: float = 120.0


class AnswerEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        cfg.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder(cfg.embed_url, cfg.runtime_dir / "embcache.json")
        self.reranker = Reranker(cfg.rerank_url, timeout_s=cfg.rerank_timeout_s)
        entities = json.loads((cfg.kb_dir / "entities.json").read_text(encoding="utf-8"))
        self.entities = EntityMatcher(entities)
        self.faq = FaqIndex(cfg.kb_dir / "faq.jsonl", self.embedder, self.entities, cfg.faq_min_sim, cfg.faq_min_margin)
        chat = load_smalltalk(cfg.kb_dir / "smalltalk.yaml") if cfg.smalltalk else []
        self.smalltalk = FaqIndex(None, self.embedder, self.entities, cfg.faq_min_sim, cfg.faq_min_margin,
                                  records=chat) if chat else None
        self._chat_by_id = {c["id"]: c for c in chat}
        self.retriever = HybridRetriever(cfg.kb_dir / "chunks.jsonl", self.embedder)
        self.cloud = LLMBackend("cloud", cfg.cloud_url, cfg.cloud_model, cfg.cloud_key, cfg.cloud_timeout_s,
                                cfg.max_tokens, {"thinking": {"type": "disabled"}}) if cfg.cloud_key else None
        # LFM2.5 adalah model "berpikir": tanpa --reasoning-budget 0 di server (dan kwarg ini) seluruh
        # max_tokens habis untuk penalaran berbahasa Inggris dan jawabannya kosong.
        self.local = LLMBackend("local", cfg.local_url.rstrip("/") + "/v1/chat/completions", cfg.local_model, "",
                                cfg.local_timeout_s, cfg.max_tokens,
                                {"chat_template_kwargs": {"enable_thinking": False}})
        self.turn_log = cfg.runtime_dir / "turns.jsonl"
        self.prev_question = ""

    def start(self) -> None:
        t0 = time.perf_counter()
        self.embedder.start(self.cfg.startup_wait_s)
        self.faq.build()
        if self.smalltalk:
            self.smalltalk.build()
        self.retriever.build()
        self.embedder.save()
        self.reranker.start(self.cfg.startup_wait_s)
        if self.cfg.mode == "local" and not wait_for_server(self.cfg.local_url, self.cfg.startup_wait_s):
            LOG.warning("LLM lokal %s belum siap sampai server hidup",
                        self.cfg.local_url)
        LOG.info("Mesin jawaban siap dalam %.1fs: %d FAQ (%d kalimat), %d obrolan, %d chunk, cloud=%s, mode=%s",
                 time.perf_counter() - t0, len(self.faq.faqs), self.faq.sentence_count,
                 len(self.smalltalk.faqs) if self.smalltalk else 0, len(self.retriever.chunks),
                 "ya" if self.cloud else "tidak", self.cfg.mode)

    def reset(self) -> None:
        self.prev_question = ""

    # -- util
    def _stt_ok(self, stt: dict | None) -> tuple[bool, str]:
        if not stt or "stt_mean_prob" not in stt:
            return True, "tanpa skor (input teks)"
        ok = stt["stt_mean_prob"] >= self.cfg.stt_min_mean and stt["stt_min_prob"] >= self.cfg.stt_min_seg
        return ok, f"stt_mean={stt['stt_mean_prob']:.2f} stt_min={stt['stt_min_prob']:.2f}"

    def _log(self, rec: dict) -> None:
        try:
            with self.turn_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _result(self, route: str, response: str, expression: str, rec: dict, t0: float,
                ask_back: bool = False) -> dict:
        rec.update(route=route, response=response, total_ms=round((time.perf_counter() - t0) * 1000))
        if ask_back:
            rec["tanya_balik"] = True
        self._log(rec)
        LOG.info("jawab %s %dms: %r", route, rec["total_ms"], response[:120])
        return {"type": "question", "feature": "", "response": response, "expression": expression,
                "route": route, "ask_back": ask_back, "meta": rec}

    # -- inti
    def _backends(self) -> list[LLMBackend]:
        chosen = {"auto": [self.cloud], "cloud": [self.cloud], "local": [self.local]}[self.cfg.mode]
        return [b for b in chosen if b is not None]

    def _ask_llm(self, messages: list[dict], rec: dict, key: str,
                 temperature: float = 0.1) -> tuple[dict | None, list[str]]:
        """Panggil backend sesuai mode dan terima hanya JSON yang valid.

        rec[key] = backend terakhir yang menjawab; kembalian (json atau None, daftar galat).
        """
        backends = self._backends()
        errors: list[str] = [] if backends else ["LLM tidak tersedia"]
        for be in backends:
            try:
                raw, stats = be.chat(messages, temperature=temperature)
            except _LLM_ERRORS as exc:
                errors.append(f"{be.name}: {str(exc)[:120]}")
                LOG.warning("LLM %s (%s) gagal: %s", be.name, key, exc)
                continue
            rec[key] = {"backend": be.name, **stats}
            parsed = parse_llm_json(raw)
            if parsed is not None:
                return parsed, errors
            errors.append(f"{be.name}: JSON tidak valid {raw[:80]!r}")
        return None, errors

    def _chat_or_refuse(self, text: str, replying_to: str, refusal_route: str, rec: dict, t0: float) -> dict:
        """Tidak ada data UB untuk ucapan ini: obrolan santai dijawab LLM persona, permintaan info tetap ditolak.

        20 Okt 2026: tim ingin Bron lebih ramah ("how are you", "sudah makan belum"), bukan hanya QnA.
        LLM memilah jenis ucapan; jawaban obrolan tidak boleh memuat nama/angka yang tidak ada di ucapan.
        """
        if not self.cfg.chat_llm or self.cfg.mode == "faq":
            return self._result(refusal_route, MSG_NO_INFO, "neutral", rec, t0)
        user = f"UCAPAN: {text}"
        if replying_to:
            user = f"BRON SEBELUMNYA BERKATA: {replying_to}\n{user}"
        messages = [{"role": "system", "content": CHAT_PROMPT}, {"role": "user", "content": user}]
        parsed, errors = self._ask_llm(messages, rec, "chat_llm", temperature=0.7)
        if errors:
            rec["chat_errors"] = errors
        if parsed is None:
            return self._result(refusal_route, MSG_NO_INFO, "neutral", rec, t0)
        kind = str(parsed.get("jenis", "")).lower()
        response = str(parsed.get("response", "")).strip()
        rec["chat_kind"] = kind
        if kind == "abaikan":
            return self._result("abaikan", "", "neutral", rec, t0)
        if kind != "obrolan" or not response:
            return self._result(refusal_route, MSG_NO_INFO, "neutral", rec, t0)
        problems = grounding_problems(response, "", text)
        if problems:
            rec["grounding"] = problems
            rec["llm_response"] = response
            return self._result("tolak_grounding_obrolan", MSG_CHAT_FALLBACK, "happy", rec, t0)
        response = _SELF_RE.sub("Bron", response)
        ask_back = bool(parsed.get("tanya_balik")) and response.endswith("?") and not replying_to
        return self._result(f"obrolan_{rec['chat_llm']['backend']}", response,
                            _expression(parsed.get("expression"), "happy"), rec, t0, ask_back)

    def answer(self, text: str, stt: dict | None = None, replying_to: str = "") -> dict:
        """Jawab satu ucapan. replying_to: kata-kata Bron yang sedang ditanggapi (jendela tanya-balik), atau "".

        Urutan: hiburan (kata kunci) -> FAQ / obrolan berjawaban tetap -> permintaan gerakan
        -> RAG + LLM (cloud dalam mode auto) -> grounding. Tanpa data UB, obrolan dijawab LLM persona.
        """
        t0 = time.perf_counter()
        text = text.strip()
        rec: dict[str, Any] = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "text": text, "stt": stt}
        if replying_to:
            rec["replying_to"] = replying_to
        if len(norm_text(text)) < 2:
            return self._result("tidak_dimengerti", MSG_NOT_UNDERSTOOD, "confused", rec, t0)

        qvec = self.embedder.embed([text])[0]
        rec["embed_ms"] = round((time.perf_counter() - t0) * 1000)
        fixed = self._fixed_answer(text, qvec, stt, replying_to, rec, t0)
        if fixed is not None:
            return fixed
        # permintaan gerakan dicek setelah FAQ, supaya "kamu bisa melambai?" dijawab FAQ
        if is_motion_command(text):
            return self._result("gerakan", MSG_MOTION, "neutral", rec, t0)
        if self.cfg.mode == "faq":
            return self._result("tolak_mode_faq", MSG_NO_INFO, "neutral", rec, t0)
        return self._rag_answer(text, qvec, replying_to, rec, t0)

    def _fixed_answer(self, text: str, qvec: Any, stt: dict | None, replying_to: str,
                      rec: dict, t0: float) -> dict | None:
        """Jawaban tetap tanpa LLM: hiburan (kata kunci), FAQ, atau obrolan dari kb/smalltalk.yaml.

        STT kurang yakin -> FAQ hanya bila kemiripannya sangat tinggi.
        14 Okt 2026: dulu FAQ dilewati sama sekali. Tetapi whisper-small memberi "Terima kasih." yang bersih
        skor 0,38-0,62 (13 dari 24 di bawah 0,50; skor FAQ-nya 0,91-0,98), sedangkan gema prompt whisper
        justru 0,40-0,79 -- skor STT tidak memisahkan ucapan asli dari sampah, dan teks yang dilewatkan
        tetap berakhir di RAG/LLM. MAI tidak mengirim skor (selalu BRONE_MAI_DEFAULT_PROB).
        """
        stt_ok, stt_note = self._stt_ok(stt)
        min_sim = None if stt_ok else self.cfg.faq_low_stt_min_sim
        rec["stt_gate"] = stt_note + (" -> FAQ boleh" if stt_ok else f" -> FAQ hanya bila skor >= {min_sim:.2f}")
        fun = next((self._chat_by_id.get(cid) for cid, rx in _FUN_REQUESTS
                    if self.smalltalk and rx.search(text)), None)
        if fun:
            rec["obrolan"] = {"id": fun["id"], "score": 1.0, "matched": "kata kunci"}
            return self._result("obrolan", random.choice(fun["jawaban"]), fun["ekspresi"], rec, t0)
        hit = self.faq.match(text, qvec, min_sim)
        # obrolan santai berjawaban tetap; menang bila skornya setara atau lebih tinggi dari FAQ
        chat = self.smalltalk.match(text, qvec, min_sim) if self.smalltalk else None
        if chat and (not hit or chat.score >= hit.score):
            rec["obrolan"] = {"id": chat.faq["id"], "score": round(chat.score, 4), "matched": chat.matched}
            return self._result("obrolan", random.choice(chat.faq["jawaban"]), chat.faq["ekspresi"], rec, t0,
                                ask_back=chat.faq["tanya_balik"] and not replying_to)
        if hit:
            rec["faq"] = {"id": hit.faq["id"], "score": round(hit.score, 4), "matched": hit.matched,
                          "margin": round(hit.margin, 4), "rejected": hit.rejected}
            self.prev_question = text
            return self._result("faq", hit.faq["a"], "happy", rec, t0)
        return None

    def _rag_answer(self, text: str, qvec: Any, replying_to: str, rec: dict, t0: float) -> dict:
        """Retrieval + gerbang reranker -> LLM dengan konteks -> grounding."""
        query = text
        if self.prev_question and len(norm_text(text).split()) <= 5:
            # pertanyaan susulan pendek ("kalau FILKOM?") dicari bersama pertanyaan sebelumnya
            query = f"{self.prev_question} {text}"
            qvec = self.embedder.embed([query])[0]
        t1 = time.perf_counter()
        cand = self.retriever.search(query, qvec, top=self.cfg.rerank_top)
        try:
            scores = self.reranker.score(
                query, [self.retriever.chunks[i]["_doc"][: self.cfg.rerank_doc_chars] for i in cand])
        except Exception as exc:  # reranker mati: jangan tebak, tolak saja
            LOG.warning("Reranker gagal: %s", exc)
            rec["error"] = f"reranker: {exc}"
            return self._result("tolak_reranker_error", MSG_NO_INFO, "neutral", rec, t0)
        ranked = sorted(zip(cand, scores, strict=False), key=lambda p: -p[1])
        rec["retrieval_ms"] = round((time.perf_counter() - t1) * 1000)
        rec["top"] = [(self.retriever.chunks[i]["id"], round(s, 3)) for i, s in ranked[:5]]
        if not ranked or ranked[0][1] < self.cfg.rerank_min:
            return self._chat_or_refuse(text, replying_to, "tolak_tidak_ada_info", rec, t0)

        context = self._context(ranked)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"KONTEKS:\n{context}\n\nPERTANYAAN: {text}"}]
        parsed, errors = self._ask_llm(messages, rec, "llm")
        if errors:
            rec["llm_errors"] = errors
        if parsed is None:
            if errors and all("JSON" not in e for e in errors):
                return self._result("gagal_llm", MSG_SERVICE_DOWN, "confused", rec, t0)
            return self._result("tidak_dimengerti", MSG_NOT_UNDERSTOOD, "confused", rec, t0)

        response = str(parsed.get("response", "")).strip()
        if parsed.get("answerable") is False or not response:
            return self._chat_or_refuse(text, replying_to, "tolak_llm", rec, t0)
        if self.cfg.grounding:
            problems = grounding_problems(response, context, text)
            if problems:
                rec["grounding"] = problems
                rec["llm_response"] = response
                return self._result("tolak_grounding", MSG_NO_INFO, "neutral", rec, t0)
        self.prev_question = text
        return self._result(f"llm_{rec['llm']['backend']}", response,
                            _expression(parsed.get("expression"), "neutral"), rec, t0)

    def _context(self, ranked: list[tuple[int, float]]) -> str:
        """Gabungan chunk teratas, dibatasi context_chunks dan context_chars (chunk pertama selalu masuk)."""
        parts, used = [], 0
        for i, score in ranked[: self.cfg.context_chunks]:
            if score < self.cfg.rerank_min * 0.5 and parts:
                break
            chunk = self.retriever.chunks[i]
            doc = f"[{chunk['judul']}] {chunk['teks']}"
            if used + len(doc) > self.cfg.context_chars and parts:
                break
            parts.append(doc)
            used += len(doc)
        return "\n".join(parts)
