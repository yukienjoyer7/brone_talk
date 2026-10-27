"""Endpointing ucapan: satu implementasi untuk jalur sounddevice dan ALSA.

Menggantikan dua loop yang logikanya identik di brone_talk.py v4 (_record_sounddevice dan
_record_alsa): preroll 8 jendela, mulai saat probabilitas VAD >= threshold, berhenti setelah
silence_limit jendela berturut-turut di bawah threshold, atau setelah window_limit jendela.

Sumber audio ditulis sebagai iterator blok sampel. Iterator yang berhenti = batas jendela
tercapai. StopCapture = diminta berhenti. CaptureLost = sumber audio putus.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from brone_talk.audio.constants import VAD_WINDOW

LOG = logging.getLogger("brone-talk")
PREROLL_WINDOWS = 8


class StopCapture(Exception):
    """Perekaman diminta berhenti (stop_event)."""


class CaptureLost(Exception):
    """Sumber audio putus di tengah rekaman (mis. arecord berhenti)."""


class VadModel(Protocol):
    def reset(self) -> None: ...

    def probability(self, samples: np.ndarray) -> float: ...


class Telemetry(Protocol):
    def vad(self, status: str, rms_float: float) -> None: ...

    def stage(self, stage: str, state: str = "", **fields: object) -> None: ...


@dataclass
class Utterance:
    frames: list[np.ndarray] = field(default_factory=list)
    peak_rms: float = 0.0
    speech_start: int = 0  # indeks sampel awal ucapan dalam stream (untuk arah suara)
    speech_end: int = 0  # indeks sampel akhir ucapan dalam stream


class Endpointer:
    def __init__(self, vad: VadModel, telemetry: Telemetry, threshold: float,
                 silence_limit: int, window_limit: int, start_limit: int = 0, speech_limit: int = 0) -> None:
        self.vad = vad
        self.telemetry = telemetry
        self.threshold = threshold
        self.silence_limit = silence_limit
        self.window_limit = window_limit
        self.start_limit = start_limit  # > 0: menyerah bila ucapan belum mulai setelah sekian jendela
        # > 0: ucapan dipotong setelah sekian jendela SEJAK mulai bicara (window_limit ikut menghitung
        # waktu tunggu sebelum bicara, jadi tanpa ini ucapan bisa max + start, 26 Okt 2026)
        self.speech_limit = speech_limit

    def run(self, source: Iterable[np.ndarray]) -> Utterance:
        """Konsumsi jendela dari source sampai ucapan selesai atau batas jendela tercapai."""
        preroll: deque[np.ndarray] = deque(maxlen=PREROLL_WINDOWS)
        utt = Utterance()
        speaking = False
        silent_windows = 0
        ticks = 0
        speech_tick = 0
        try:
            for tick, samples in enumerate(source):
                ticks = tick + 1
                rms = float(np.sqrt(np.mean(np.square(samples))))
                utt.peak_rms = max(utt.peak_rms, rms)
                probability = self.vad.probability(samples)

                if tick % 8 == 0:
                    self.telemetry.vad("speaking" if probability >= self.threshold else "silence", rms)

                if not speaking:
                    preroll.append(samples)
                    if probability >= self.threshold:
                        speaking = True
                        speech_tick = tick
                        utt.frames.extend(preroll)
                        utt.speech_start = (tick - len(preroll) + 1) * VAD_WINDOW
                        silent_windows = 0
                        self.telemetry.stage("listening", "speech", rms=round(rms * 32768, 1))
                        LOG.info("Ucapan terdeteksi")
                else:
                    utt.frames.append(samples)
                    if probability < self.threshold:
                        silent_windows += 1
                        if silent_windows >= self.silence_limit:
                            break
                    else:
                        silent_windows = 0
                if speaking and tick - speech_tick >= self.speech_limit > 0:
                    break
                if ticks >= self.window_limit or (not speaking and ticks >= self.start_limit > 0):
                    break
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
        utt.speech_end = ticks * VAD_WINDOW
        return utt
