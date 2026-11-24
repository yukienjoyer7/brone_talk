"""servers: bagian mesin jawaban v4 (dipindah dari brone_v4.py tanpa perubahan logika)."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from brone_talk.engine.util import _http_json, wait_for_server

class Embedder:
    """bge-m3 via llama-server /v1/embeddings, dengan cache disk per teks."""

    def __init__(self, url: str, cache_path: Path, timeout_s: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.cache_path = cache_path
        self.model_id = ""
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def start(self, wait_s: float) -> None:
        if not wait_for_server(self.url, wait_s):
            raise RuntimeError(f"server embedding {self.url} tidak siap dalam {wait_s:.0f} detik")
        try:
            models = _http_json(self.url + "/v1/models", None, timeout=5)
            self.model_id = str((models.get("data") or [{}])[0].get("id", ""))
        except Exception:
            self.model_id = "unknown"
        if self.cache_path.is_file():
            try:
                raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if raw.get("model") == self.model_id:
                    self._cache = raw.get("vectors", {})
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str], timeout_s: float | None = None) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        for i in range(0, len(missing), 32):
            batch = missing[i:i + 32]
            res = _http_json(self.url + "/v1/embeddings", {"input": [t for _k, t in batch], "model": "bge-m3"},
                             timeout=timeout_s or self.timeout_s)
            data = sorted(res["data"], key=lambda d: d["index"])
            for (k, _t), d in zip(batch, data):
                vec = np.asarray(d["embedding"], dtype=np.float32)
                vec /= (np.linalg.norm(vec) + 1e-9)
                self._cache[k] = vec.round(6).tolist()
        return np.asarray([self._cache[k] for k in keys], dtype=np.float32)

    def save(self) -> None:
        with self._lock:
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"model": self.model_id, "vectors": self._cache}), encoding="utf-8")
            tmp.replace(self.cache_path)
