"""Keputusan per ucapan: kapan robot diam, kapan minta diulang, kapan menjawab.

Aturan:
  - "Brone" di awal kalimat atau akhir ucapan -> pertanyaannya dijawab langsung. Nama di tengah
    kalimat bukan panggilan (nlu.wake.call_text, 20 Okt 2026).
  - Hanya nama ("Brone", "Halo Brone") -> bila BRONE_AWAIT_SECONDS > 0: jawab "Ya?", lalu SATU
    ucapan berikutnya tanpa nama dijawab dalam jendela itu. Bawaan 0 (sejak 20 Okt 2026): robot
    hanya mengingatkan agar nama disebut, ucapan tanpa nama tidak pernah dijawab.
  - Bron bertanya balik ("Kamu sendiri sudah makan?") -> SATU ucapan berikutnya tanpa nama boleh
    ditanggapi dalam BRONE_SOCIAL_AWAIT_SECONDS; bila ujungnya penolakan, Bron diam (app.py).
  - Tidak disebut nama dan tidak sedang menunggu -> diabaikan (standby).
  - Kata stop saat menunggu pertanyaan -> kembali standby.
Tidak ada kunci penanya (sidik suara/arah): dihapus atas permintaan pengguna (19 Okt 2026).

decide() murni: tidak memanggil audio, LLM, atau MQTT. Efek samping ada di app.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from brone_talk.nlu.wake import STOP_PATTERNS, matches, wake_question


class Decision(Enum):
    STOP = "stop"  # kata stop saat menunggu pertanyaan
    STANDBY = "standby"  # tidak ditujukan ke robot
    CONFIRM = "confirm"  # hanya nama: jawab "Ya?" dan tunggu pertanyaan
    ANSWER = "answer"  # ada pertanyaan yang harus dijawab


@dataclass(frozen=True)
class Outcome:
    decision: Decision
    question: str = ""  # teks yang dijawab (untuk ANSWER)
    wake: bool = False  # True bila ANSWER berasal dari ucapan yang menyebut nama


@dataclass
class Session:
    """State antar giliran. awaiting_question: sudah dipanggil, menunggu pertanyaan.
    unclear_said: pesan "suara kurang jelas" sudah diucapkan pada giliran menunggu ini."""

    awaiting_question: bool = False
    unclear_said: bool = False
    awaiting_since: float = 0.0  # time.monotonic() saat "Ya?" diucapkan
    await_window: float = 0.0  # panjang jendela penantian ini (0 = pakai BRONE_AWAIT_SECONDS)
    social: bool = False  # penantian ini dari pertanyaan balik Bron, bukan dari "Ya?"
    bot_text: str = ""  # kata-kata Bron yang ditanggapi (untuk LLM obrolan)

    def start_wait(self, window_s: float, social: bool = False, bot_text: str = "") -> None:
        self.awaiting_question, self.awaiting_since, self.unclear_said = True, time.monotonic(), False
        self.await_window, self.social, self.bot_text = window_s, social, bot_text

    def end_wait(self) -> None:
        """Penantian selesai (dijawab, dihentikan, atau habis): ucapan berikutnya harus menyebut nama."""
        self.awaiting_question = False
        self.social = False

    def expired(self, now: float, default_window_s: float) -> bool:
        """Sedang menunggu dan jendelanya sudah lewat pada waktu `now` (time.monotonic())."""
        window = self.await_window or default_window_s
        return self.awaiting_question and now - self.awaiting_since > window

    def should_say_unclear(self) -> bool:
        """Pesan "suara kurang jelas" hanya sekali per penantian "Ya?" (tidak untuk tanya-balik)."""
        if not self.awaiting_question or self.social or self.unclear_said:
            return False
        self.unclear_said = True
        return True


def decide(text: str, *, called: bool, awaiting: bool, stop_patterns: list[str] = STOP_PATTERNS) -> Outcome:
    if awaiting and matches(stop_patterns, text):
        return Outcome(Decision.STOP)
    # Tanpa nama, robot hanya menjawab bila sedang menunggu pertanyaan setelah "Ya?".
    if not called and not awaiting:
        return Outcome(Decision.STANDBY)
    if called:
        question = wake_question(text)
        if not question:
            return Outcome(Decision.CONFIRM)
        return Outcome(Decision.ANSWER, question=question, wake=True)
    return Outcome(Decision.ANSWER, question=text, wake=False)
