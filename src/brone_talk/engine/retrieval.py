"""retrieval: bagian mesin jawaban v4 (dipindah dari brone_v4.py tanpa perubahan logika)."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from brone_talk.engine.util import warn_stale
from brone_talk.engine.servers import Embedder

class DenseRetriever:
    """Retrieval dense murni (cosine pada embedding chunk terstruktur)."""

    def __init__(self, chunks_path: Path, embedder: Embedder) -> None:
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
        self._mat: np.ndarray | None = None

    def build(self) -> None:
        self._mat = self.embedder.embed([c["_doc"] for c in self.chunks], timeout_s=180)

    def search(self, query: str, qvec: np.ndarray, top: int = 10) -> list[int]:
        assert self._mat is not None
        dense = self._mat @ qvec
        return [int(i) for i in np.argsort(-dense)[:top]]

    def search_scored(self, qvec: np.ndarray, top: int = 5) -> list[tuple[int, float]]:
        """Chunk teratas berikut kemiripan kosinusnya (vektor sudah dinormalisasi)."""
        assert self._mat is not None
        dense = self._mat @ qvec
        return [(int(i), float(dense[i])) for i in np.argsort(-dense)[:top]]
