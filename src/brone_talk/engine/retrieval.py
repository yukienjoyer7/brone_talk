"""retrieval: bagian mesin jawaban v4 (dipindah dari brone_v4.py tanpa perubahan logika)."""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from brone_talk.engine.util import norm_text, warn_stale
from brone_talk.engine.servers import Embedder

_BM25_STOP = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "adalah", "itu", "ini", "apa", "apakah", "siapa", "berapa",
    "bagaimana", "dimana", "mana", "kapan", "mengapa", "kenapa", "ada", "akan", "bisa", "boleh", "saya", "kamu",
    "kita", "kami", "tolong", "dong", "ya", "kah", "sebutkan", "jelaskan", "saja", "aja", "dengan", "atau", "juga",
    "sudah", "belum", "tidak", "bron", "brone", "robot", "tahu", "punya", "sekarang", "tentang", "nya", "gimana",
}

def bm25_tokens(text: str) -> list[str]:
    out = []
    for w in norm_text(text).split():
        for suf in ("nya", "kah", "lah"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        if w not in _BM25_STOP and (len(w) > 1 or w.isdigit()):
            out.append(w)
    return out

class HybridRetriever:
    def __init__(self, chunks_path: Path, embedder: Embedder, k1: float = 1.4, b: float = 0.6) -> None:
        today = _dt.date.today().isoformat()
        raw = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        warn_stale("chunk", raw, today)
        self.chunks = []
        for c in raw:
            if c.get("berlaku_hingga") and str(c["berlaku_hingga"]) < today:
                continue
            c["_doc"] = f"{c['judul']}. {c['teks']}"
            self.chunks.append(c)
        self.embedder = embedder
        self.k1, self.b = k1, b
        self._toks = [bm25_tokens(c["_doc"]) for c in self.chunks]
        self._tf = [Counter(t) for t in self._toks]
        self._len = np.asarray([len(t) for t in self._toks], dtype=np.float32)
        self._avg = float(self._len.mean()) if len(self._len) else 1.0
        df = Counter(w for t in self._toks for w in set(t))
        n = len(self.chunks)
        self._idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self._mat: np.ndarray | None = None

    def build(self) -> None:
        self._mat = self.embedder.embed([c["_doc"] for c in self.chunks], timeout_s=180)

    def bm25(self, query: str) -> np.ndarray:
        q = bm25_tokens(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for w in set(q):
            idf = self._idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self._tf):
                f = tf.get(w)
                if f:
                    norm = 1 - self.b + self.b * self._len[i] / self._avg
                    scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return scores

    def search(self, query: str, qvec: np.ndarray, top_each: int = 20, top: int = 10, rrf_k: int = 60) -> list[int]:
        assert self._mat is not None
        dense = self._mat @ qvec
        sparse = self.bm25(query)
        fused: dict[int, float] = {}
        for ranking in (np.argsort(-dense)[:top_each], [i for i in np.argsort(-sparse)[:top_each] if sparse[i] > 0]):
            for r, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + r + 1)
        return [i for i, _s in sorted(fused.items(), key=lambda kv: -kv[1])[:top]]
