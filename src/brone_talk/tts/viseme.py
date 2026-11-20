"""Jadwal viseme (bentuk mulut) dari alignment fonem Piper, untuk BroneGUI-Tauri.

Piper (>= 1.8, model dipatch dengan `python3 -m piper.patch_voice_with_alignment`) melaporkan jumlah
sampel audio per fonem. Angka itu keluaran duration predictor VITS yang menentukan panjang audio, jadi
jadwal di sini mengikuti audio yang benar-benar diputar, bukan perkiraan dari teks.

Viseme GUI (docs/CONTRACT.md BroneGUI-Tauri): A, I, U, E, O, REST (bibir rapat / jeda), DIAM (selesai).
Aturan koartikulasi mengikuti README GUI: konsonan biasa + vokal -> langsung bentuk vokal; bilabial
(p, b, m) -> REST. Konsonan di akhir suku kata (sebelum jeda/spasi) mempertahankan vokal sebelumnya.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

VOWEL = {"a": "A", "i": "I", "u": "U", "e": "E", "ɛ": "E", "ə": "E", "o": "O", "ɔ": "O"}
BILABIAL = {"p", "b", "m"}
PAUSE = {"^", "$", ",", ".", ";", ":", "!", "?", "—", "…"}  # awal/akhir kalimat dan tanda baca: bibir rapat
MODIFIER = {"ˈ", "ˌ", "ː"}  # tekanan/panjang: waktunya digabung ke fonem sesudahnya
GUI_EXPRESSIONS = {  # nama yang diterima petakan_emosi() di src-tauri/src/connector/protocol.rs
    "idle", "netral", "neutral", "happy", "senang", "happier", "sangat_senang", "happier_transisi",
    "sangat_senang_transisi", "sad", "sedih", "cry", "menangis", "shock", "terkejut", "confused",
    "thinking", "berpikir", "timeout", "excited",
}


def _segments(chunks: Iterable[Sequence[tuple[str, int]]]) -> list[tuple[str, int, int]]:
    """(fonem, sampel_mulai, jumlah_sampel) berurutan di seluruh audio; modifier digabung ke fonem berikutnya."""
    out: list[tuple[str, int, int]] = []
    pos = 0
    for alignments in chunks:
        carry_start = None
        for phoneme, n in alignments:
            n = int(n)
            if phoneme in MODIFIER:
                carry_start = pos if carry_start is None else carry_start
            else:
                start = pos if carry_start is None else carry_start
                out.append((phoneme, start, pos + n - start))
                carry_start = None
            pos += n
        if carry_start is not None:  # modifier di ujung chunk: tempel ke segmen terakhir
            p, s, _ = out[-1]
            out[-1] = (p, s, pos - s)
    return out


def _label(segs: list[tuple[str, int, int]]) -> list[str]:
    raw = [VOWEL.get(p) or ("REST" if p in BILABIAL or p in PAUSE else None) for p, _, _ in segs]
    labels: list[str] = []
    for i, (p, _, _) in enumerate(segs):
        if raw[i] is not None:
            labels.append(raw[i])
            continue
        nxt = None
        if p != " ":  # konsonan pembuka suku kata: ikut vokal berikutnya dalam kata yang sama
            for j in range(i + 1, len(segs)):
                if segs[j][0] == " " or raw[j] == "REST":
                    break
                if raw[j] is not None:
                    nxt = raw[j]
                    break
        labels.append(nxt or (labels[-1] if labels else "REST"))
    return labels


def schedule(chunks: Iterable[Sequence[tuple[str, int]]], sample_rate: int, offset_s: float = 0.0) -> list[dict]:
    """Daftar [{"t": detik, "v": viseme}] untuk payload `viseme` di robot/expression.

    chunks: alignment per chunk audio Piper, berurutan seperti ditulis ke WAV ([(fonem, jumlah_sampel)]).
    offset_s: geser semua t (positif = mulut lebih lambat), untuk kompensasi latensi yang diukur.
    """
    segs = _segments(chunks)
    if not segs:
        return []
    jadwal: list[dict] = []
    for (_, start, _), v in zip(segs, _label(segs)):
        if jadwal and jadwal[-1]["v"] == v:
            continue
        jadwal.append({"t": max(0.0, round(start / sample_rate + offset_s, 3)), "v": v})
    end = segs[-1][1] + segs[-1][2]
    jadwal.append({"t": max(0.0, round(end / sample_rate + offset_s, 3)), "v": "DIAM"})
    return jadwal


def speech_payload(expression: str, duration_s: float, jadwal: list[dict] | None) -> dict:
    """Paket robot/expression saat audio mulai. Ekspresi yang tidak dikenal GUI membuat seluruh pesan ditolak,
    jadi diganti "speaking" (GUI: bicara mulai, mulut tetap jalan). jadwal None/kosong -> tanpa field viseme."""
    name = expression if expression and expression.lower() in GUI_EXPRESSIONS else "speaking"
    payload = {"expression": name, "duration": round(duration_s, 2)}
    if jadwal:
        payload["viseme"] = jadwal
    return payload


# --- Mode "kalimat" (GUI visemeMode "streaming", docs/CONTRACT.md BroneGUI-Tauri, 20 Nov 2026):
# satu paket {"expression", "duration", "sentence"} per kalimat, dipublish saat kalimat itu mulai terdengar.

# Sama dengan pemecahan kalimat espeak-ng di Piper: "!" / "?" selalu memecah, "." hanya bila kata berikutnya
# tidak diawali huruf kecil. Teruji 233/234 teks FAQ jumlahnya sama dan 233/233 batasnya identik (fonem per
# kalimat sama); sisanya (singkatan gelar "MAcc., DBA.") jatuh ke satu paket utuh.
_SENTENCE_END = re.compile(r"(?<=[!?])\s+|(?<=\.)\s+(?=[^\sa-z])")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
SENTENCE_MAX_CHARS = 4096


def split_sentences(text: str) -> list[str]:
    """Kalimat per baris, dengan urutan yang sama seperti chunk audio Piper."""
    out: list[str] = []
    for line in text.splitlines():
        out += [s.strip() for s in _SENTENCE_END.split(line.strip()) if s.strip()]
    return out


def _clean_sentence(text: str) -> str:
    return " ".join(_CONTROL.sub(" ", text).split())[:SENTENCE_MAX_CHARS]


def sentence_packets(expression: str, text: str, chunk_samples: Sequence[int] | None, sample_rate: int,
                     duration_s: float) -> list[tuple[float, dict]]:
    """[(detik_mulai_di_audio, payload)] untuk mode kalimat.

    chunk_samples: panjang tiap chunk audio Piper (satu chunk = satu kalimat espeak). Bila jumlahnya tidak sama
    dengan kalimat hasil split_sentences (atau tidak ada, mis. jalur CLI), dikirim satu paket berisi seluruh teks
    dengan durasi total, sehingga durasi tetap benar walau pembagian kalimat tidak diketahui.
    """
    name = expression.lower() if expression and expression.lower() in GUI_EXPRESSIONS else "idle"
    sentences = split_sentences(text)
    if not chunk_samples or len(sentences) != len(chunk_samples) or any(n <= 0 for n in chunk_samples):
        whole = _clean_sentence(text)
        return [(0.0, {"expression": name, "duration": round(duration_s, 3), "sentence": whole})] if whole else []
    packets, start = [], 0
    for sentence, n in zip(sentences, chunk_samples):
        packets.append((start / sample_rate,
                        {"expression": name, "duration": round(n / sample_rate, 3), "sentence": _clean_sentence(sentence)}))
        start += n
    return packets
