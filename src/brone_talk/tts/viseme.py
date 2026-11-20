"""Jadwal viseme (bentuk mulut) dari alignment fonem Piper, untuk BroneGUI-Tauri.

Piper (>= 1.8, model dipatch dengan `python3 -m piper.patch_voice_with_alignment`) melaporkan jumlah
sampel audio per fonem. Angka itu keluaran duration predictor VITS yang menentukan panjang audio, jadi
jadwal di sini mengikuti audio yang benar-benar diputar, bukan perkiraan dari teks.

Viseme GUI (docs/CONTRACT.md BroneGUI-Tauri): A, I, U, E, O, REST (bibir rapat / jeda), DIAM (selesai).
Aturan koartikulasi mengikuti README GUI: konsonan biasa + vokal -> langsung bentuk vokal; bilabial
(p, b, m) -> REST. Konsonan di akhir suku kata (sebelum jeda/spasi) mempertahankan vokal sebelumnya.
"""

from __future__ import annotations

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


def speech_payload(expression: str, duration_s: float, jadwal: list[dict]) -> dict:
    """Paket gabungan robot/expression. Ekspresi yang tidak dikenal GUI membuat seluruh pesan ditolak,
    jadi diganti "speaking" (GUI: bicara mulai + jadwal tetap jalan; ekspresi asli tetap dikirim sesudahnya)."""
    name = expression if expression and expression.lower() in GUI_EXPRESSIONS else "speaking"
    return {"expression": name, "duration": round(duration_s, 2), "viseme": jadwal}
