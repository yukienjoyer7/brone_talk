"""Wake word "halo brone" (openWakeWord, model kustom ONNX) dan pemantau interupsi.

Desain disepakati 23 Okt 2026: Bron hanya bangun lewat model KWS ini, bukan lewat teks STT.
"halo brone" juga interupsi: saat Bron berpikir/berbicara, pemantau tetap mendengar dan
menghentikan jawaban. Model dilatih di AI-Center (25 Okt 2026, ~/halo_brone/halo_brone_model).

openWakeWord menerima potongan 1280 sampel (80 ms) PCM16 @ 16 kHz; mic memberi blok 512
sampel (jendela VAD), jadi blok dikumpulkan dulu oleh ChunkBuffer.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

LOG = logging.getLogger("brone-talk")
KWS_CHUNK = 1280  # 80 ms @ 16 kHz, ukuran frame openWakeWord

_SELF_WAKE_RE = re.compile(r"\b(?:h?al+o+h?|hel+o)\b.*\bbro+n", re.I)  # halo/alo/haloh/hello ... bron


class ChunkBuffer:
    """Kumpulkan blok float32 sembarang panjang menjadi potongan int16 berukuran tetap."""

    def __init__(self, size: int = KWS_CHUNK) -> None:
        self.size = size
        self._buf = np.zeros(0, dtype=np.int16)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        self._buf = np.concatenate([self._buf, pcm])
        chunks = []
        while len(self._buf) >= self.size:
            chunks.append(self._buf[:self.size])
            self._buf = self._buf[self.size:]
        return chunks

    def clear(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)


class WakeWord:
    """Satu instance model openWakeWord. Tidak thread-safe: pemantau interupsi memakai instance sendiri."""

    def __init__(self, model_path: Path, threshold: float, predictor: Callable[[np.ndarray], float] | None = None):
        self.threshold = threshold
        self.buffer = ChunkBuffer()
        self.last_score = 0.0
        if predictor is not None:  # tes: skor palsu tanpa openwakeword
            self._predict, self._reset = predictor, lambda: None
            return
        from openwakeword.model import Model

        self.model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        self.name = next(iter(self.model.models))
        self._predict = lambda chunk: float(self.model.predict(chunk)[self.name])
        self._reset = self.model.reset

    def reset(self) -> None:
        self.buffer.clear()
        self.last_score = 0.0
        self._reset()

    def feed(self, samples: np.ndarray) -> bool:
        """True bila potongan audio ini memuat "halo brone" (skor >= threshold)."""
        scores = [self._predict(chunk) for chunk in self.buffer.push(samples)]
        if not scores:
            return False
        self.last_score = max(scores)
        return self.last_score >= self.threshold

    def listen(self, blocks: Iterable[np.ndarray]) -> bool:
        """Konsumsi blok sampai wake word terdengar (True) atau sumber habis (False)."""
        self.reset()
        for samples in blocks:
            if self.feed(samples):
                LOG.info("Wake word terdeteksi (skor=%.2f)", self.last_score)
                self.reset()
                return True
        return False


class WakeHeard(Exception):
    """"halo brone" terdengar saat merekam pertanyaan di jendela percakapan (Bron harus "Ya?" lagi)."""


def watch(blocks: Iterable[np.ndarray], detector: WakeWord) -> Iterator[np.ndarray]:
    """Teruskan blok apa adanya ke perekam; WakeHeard bila "halo brone" terdengar di antaranya."""
    detector.reset()
    for samples in blocks:
        if detector.feed(samples):
            LOG.info("Wake word terdeteksi di jendela percakapan (skor=%.2f)", detector.last_score)
            detector.reset()
            raise WakeHeard()
        yield samples


def self_wake(spoken_text: str) -> bool:
    """Kalimat yang sedang diucapkan Bron sendiri memuat "halo ... bron": deteksi diabaikan."""
    return bool(_SELF_WAKE_RE.search(spoken_text or ""))


class InterruptMonitor:
    """Thread yang mendengar "halo brone" selama Bron berpikir/berbicara.

    open_blocks() membuka aliran mic sendiri (PipeWire mengizinkan beberapa klien rekam);
    on_wake() dipanggil sekali saat terdeteksi, kecuali kalimat yang sedang diucapkan Bron
    sendiri memuat "halo brone" (current_text()).
    """

    def __init__(self, detector: WakeWord, open_blocks: Callable[[threading.Event], Iterable[np.ndarray]],
                 on_wake: Callable[[], None], current_text: Callable[[], str]) -> None:
        self.detector = detector
        self.open_blocks = open_blocks
        self.on_wake = on_wake
        self.current_text = current_text
        self.fired = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "InterruptMonitor":
        self.fired.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kws-interrupt", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        try:
            self.detector.reset()
            for samples in self.open_blocks(self._stop):
                if not self.detector.feed(samples):
                    continue
                if self_wake(self.current_text()):
                    LOG.info("Wake word dari suara Bron sendiri diabaikan")
                    self.detector.reset()
                    continue
                LOG.info("Interupsi 'halo brone' (skor=%.2f)", self.detector.last_score)
                self.fired.set()
                self.on_wake()
                return
        except Exception:
            LOG.exception("Pemantau interupsi berhenti")
