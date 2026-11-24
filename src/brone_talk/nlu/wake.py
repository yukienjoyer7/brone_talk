"""Wake word, pengenalan panggilan, dan jawaban jam untuk BRONE.

Semua fungsi di sini murni.
Bentuk variasi nama ("Brone", "Bron", salah-dengar STT) hanya didefinisikan sekali di sini.
"""

from __future__ import annotations

import re
import time

from brone_talk.names import BRONE_VARIANTS
from brone_talk.text import stt_tokens

_NAME = BRONE_VARIANTS

NAME_RE = re.compile(rf"\b{_NAME}\b", re.I)

_CALL_WORDS = ("halo", "hallo", "hai", "hei", "hey", "hi", "hello", "eh", "oh", "ya", "iya",
               "selamat", "pagi", "siang", "sore", "malam", "permisi", "tolong")

_CALL = "|".join(_CALL_WORDS)

_HEAD_CALL_RE = re.compile(rf"^[\W_]*(?:(?:{_CALL})\b[\W_]*)*{_NAME}\b(?!\s*filkom)[\s,.!?-]*", re.I)

_TAIL_CALL_RE = re.compile(rf"\s*,\s*{_NAME}\s*([.!?]*)\s*$", re.I)

_HEAD_NAME_RE = re.compile(rf"^[\W_]*(?:(?:{_CALL})\b[\W_]*)*{_NAME}\b", re.I)

_TAIL_NAME_RE = re.compile(rf"\b{_NAME}[\W_]*$", re.I)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")

def call_text(text: str) -> str:
    """Bagian ucapan yang memanggil Bron, atau "" bila nama tidak dipakai sebagai panggilan.

    Panggilan = nama di awal kalimat (boleh didahului sapaan) atau di akhir ucapan:
    "Brone, UB ada di mana?", "UB ada di mana, Brone?" -> utuh;
    "Oke. Satu, dua. BRONE, siapa kamu?" -> "BRONE, siapa kamu?" (STT menggabung beberapa
    kalimat dalam satu klip; yang dijawab mulai dari panggilan terakhir);
    "kode BRONE talk itu", "Enggak, BRONE. Saya belum ke FIB. Oh, iya." -> "".
    20 Okt 2026: tim menyebut "BRONE" di tengah obrolan kerja dan robot menjawab penolakan
    18 kali sejak 19 Okt; sejak itu nama di tengah kalimat bukan panggilan.
    """
    text = (text or "").strip()
    pieces = [p for p in _SENTENCE_SPLIT_RE.split(text) if p]
    for i in range(len(pieces) - 1, -1, -1):
        if _HEAD_NAME_RE.match(pieces[i]):
            return " ".join(pieces[i:]) if i else text
    return text if _TAIL_NAME_RE.search(text) else ""

def addressed(text: str) -> bool:
    return bool(call_text(text))

def wake_question(text: str) -> str:
    """Pertanyaan dari ucapan yang memanggil Bron (lihat call_text untuk posisi nama).

    "Brone, UB ada di mana?" dan "UB ada di mana, Brone?" -> "UB ada di mana?";
    "Siapa yang membuat Brone?" -> utuh (nama di tengah kalimat bagian dari pertanyaan).
    Kosong bila nama tidak disebut, atau ucapan hanya panggilan tanpa isi
    (< 2 kata selain nama/sapaan, mis. "Brone" atau "Halo Brone").
    """
    if not NAME_RE.search(text or ""):
        return ""
    question = _HEAD_CALL_RE.sub("", text.strip(), count=1)
    question = _TAIL_CALL_RE.sub(r"\1", question).strip(" ,.!-")
    content = [w for w in re.findall(r"\w+", question.lower())
               if w not in _CALL_WORDS and not NAME_RE.fullmatch(w)]
    return question if len(content) >= 2 else ""

def collapse_repeats(text: str) -> str:
    """Buang panggilan yang diulang: "Halo, BRONE. Halo, BRONE. Halo, BRONE." -> "Halo, BRONE.",
    "Brone, Brone, UB di mana?" -> "Brone, UB di mana?".

    Orang yang tidak langsung dijawab memanggil Bron berkali-kali; tanpa ini klipnya
    dibuang stt_garbage_reason sebagai "frasa berulang" (journal 6 Okt 2026).

    13 Okt 2026: hanya ulangan yang memuat nama Bron yang diringkas. Dulu semua bagian
    berkoma ikut diringkas, jadi halusinasi derau "BRONE, Rector, Rector, Rector," menjadi
    "BRONE, Rector," dan "…AGITO, FUK, FUK, FUK" lolos filter lalu dijawab.
    """
    parts: list[str] = []
    for part in re.split(r"(?<=[.!?,])\s+", text.strip()):
        if parts and stt_tokens(part) == stt_tokens(parts[-1]) and NAME_RE.search(part):
            continue
        parts.append(part)
    keys = [stt_tokens(p) for p in parts]
    for period in range(1, len(parts) // 2 + 1):
        if (all(keys[i] == keys[i % period] for i in range(len(keys)))
                and NAME_RE.search(" ".join(parts[:period]))):
            return " ".join(parts[:period])
    return " ".join(parts)

_HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

_BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
          "September", "Oktober", "November", "Desember"]

_CLOCK_FILLER = {"bron", "brone", "brun", "broni", "bronny", "brony", "berone", "beran", "beron", "done",
                 "halo", "hallo", "hai", "hello", "hi",
                 "baiklah", "baik", "ya", "sekarang", "dong", "kah", "tolong", "sih", "nih", "oke", "coba",
                 "kasih", "tau", "tahu", "boleh", "saya", "aku", "mau", "tanya"}

_CLOCK_TIME = re.compile(r"(jam|pukul) berapa|what time is it")

_CLOCK_DAY = re.compile(r"(hari ini )?hari (ini )?apa( hari ini| ini)?")

_CLOCK_DATE = re.compile(r"(hari ini )?tanggal (berapa|hari ini)( hari ini)?")

def clock_answer(text: str, now: float | None = None) -> str:
    """Jawaban jam/hari/tanggal dari jam sistem, atau "" bila bukan pertanyaan waktu murni.

    Model 0.8B tidak punya jam dan mengarang ("Jam 12:00 WIB", "hari Jumat" pada hari Minggu).
    Hanya dipakai bila seluruh kalimat (tanpa sapaan/pengisi) adalah pertanyaan waktu, supaya
    "perpustakaan buka jam berapa?" tetap ke RAG.
    """
    words = re.findall(r"[a-z]+", (text or "").lower())
    core = " ".join(w for w in words if w not in _CLOCK_FILLER)
    t = time.localtime(now) if now is not None else time.localtime()
    if _CLOCK_TIME.fullmatch(core):
        return f"Sekarang pukul {t.tm_hour:02d}.{t.tm_min:02d} WIB."
    if _CLOCK_DAY.fullmatch(core) or _CLOCK_DATE.fullmatch(core):
        return f"Hari ini hari {_HARI[t.tm_wday]}, {t.tm_mday} {_BULAN[t.tm_mon - 1]} {t.tm_year}."
    return ""

STOP_PATTERNS = [r"\bselesai\b", r"\bdiam\b", r"\bstop\b", r"\bsampai jumpa\b"]

# 9 Nov 2026: sesi (mode wake word) berakhir HANYA lewat "selesai brone" (nama wajib), dicocokkan
# dari teks STT Mai, bukan dari model KWS. Kata "selesai/diam/stop" tanpa nama tidak mengakhiri sesi.
# 10 Nov 2026 (uji sintesis STT-E-04/E-06/E-10, runtime/uji-end-wake): versi lama mencocokkan
# "selesai, Brone" di mana saja, sehingga "Kapan skripsi saya selesai, Brone?" (18/20 klip) dan gema
# sapaan Bron sendiri ("kalau selesai tinggal bilang 'selesai brone'", 8/8) menidurkan sesi. Sekarang
# seluruh ucapan harus berupa frasa akhir, boleh diapit kata pengisi pendek (oke, terima kasih, ya, ...).
# Whisper sering mendengar "selesai" sebagai "Selasai"/"Silasai"/"Selosai"/"Sosai" -> varian ejaan diterima,
# hanya di sini (nama wajib menyusul), tidak di koreksi STT umum.
_END_WORD = r"(?:s[eio]l?[aeo]s[ae]i|sosai)"
_END_PREFIX = (r"(?:oke|ok|okay|okey|baik|baiklah|ya|ia|iya|yaudah|ya\s+sudah|sudah|udah|terima\s+kasih|"
               rf"makasih|trims|halo|hai|nah|jadi|kalau\s+begitu|kalo\s+gitu|{_NAME})")
_END_SUFFIX = (r"(?:terima\s+kasih|makasih|trims|ya|yah|ya\s+sudah|sampai\s+jumpa|dadah|da|bye|dah|oke|ok|"
               r"deh|aja|saja|dulu|dulu\s+ya)")
_SESSION_END_STRICT = re.compile(
    rf"^[\W_]*(?:{_END_PREFIX}\b[\W_]*){{0,4}}{_END_WORD}\b[\W_]*{_NAME}\b[\W_]*(?:{_END_SUFFIX}\b[\W_]*){{0,4}}$",
    re.I,
)

# 20 Nov 2026: regex di atas melewatkan 107/200 klip "selesai brone" di runtime/uji-end-wake (derau, jarak jauh).
# Whisper mendengar "selesai" sebagai Kelasai/Selakai/Lakai/Selatai/Suasai/Kelastai/Filasai, nama sebagai
# Broning/Bronie/Bronya/Kebrone, dan menambah ekor gema prompt ("UB, Universitas Brawijaya") atau
# "sampai jumpa" yang rusak ("sampai jintak"). Pencocok token di bawah tetap mewajibkan pasangan
# KATA-AKHIR + NAMA yang bersebelahan dan seluruh ucapan sebelum pasangan itu hanya kata pengisi,
# sehingga "Kapan skripsi saya selesai, Brone?" dan gema sapaan Bron sendiri tetap tidak mengakhiri sesi.
_END_FILLERS = frozenset((
    "oke ok okay okey baik baiklah ya yak ia iya yaudah sudah udah terima kasih makasih trims halo hai nah jadi "
    "kalau kalo begitu gitu sampai jumpa dadah da bye dah deh aja saja dulu yah eh akhirnya sekarang pokoknya "
    "sip siap sangat bike").split())  # "bike" = Whisper untuk "baik"
# gema daftar istilah prompt STT yang sering menempel di ekor ucapan pendek
_END_ECHO = frozenset(
    "ub universitas brawijaya kota malang filkom veteran lowokwaru fakultas program studi mahasiswa kampus "
    "rektorat".split())
# kata ini tidak boleh muncul sama sekali: tanda pertanyaan/pernyataan biasa, bukan pamit
_END_BLOCK = frozenset(
    "apa apakah kapan siapa berapa mana dimana gimana bagaimana kenapa mengapa belum jam kok saya aku kamu dia "
    "kami kita mereka tugas skripsi kuliah tinggal bilang".split())
# kata umum berakhiran -ai yang bukan salah dengar "selesai"
_AI_WORDS = frozenset(
    "sampai santai pakai pantai lantai damai ramai pandai sungai badai tupai rantai cerai gulai abai bantai belai "
    "nilai mulai bagai tirai landai capai rinai lerai kedai gerai ngantai".split())
_NAME_TOKEN = re.compile(rf"^(?:ke)?(?:{_NAME}|br[ao]n\w{{0,4}})$", re.I)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_end_token(tok: str) -> bool:
    """"selesai", "usai", atau salah dengar Whisper-nya (berakhiran -ai, bukan kata umum)."""
    if tok in ("selesai", "usai") or re.fullmatch(_END_WORD, tok):
        return True
    if tok in _AI_WORDS:
        return False
    if 4 <= len(tok) <= 10 and tok.endswith("ai"):
        return True
    return 5 <= len(tok) <= 9 and _edit_distance(tok, "selesai") <= 2


def is_session_end(text: str) -> bool:
    """Ucapan berisi "selesai brone" (boleh salah eja STT) yang diapit kata pengisi saja."""
    if _SESSION_END_STRICT.search(text or ""):
        return True
    toks = re.findall(r"[a-z]+", (text or "").lower())
    toks = [t for t in toks if t not in _END_ECHO]
    if not toks or len(toks) > 24 or any(t in _END_BLOCK for t in toks):
        return False
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        if _is_end_token(a) and _NAME_TOKEN.match(b):
            pair_end = i + 2
        elif _NAME_TOKEN.match(a) and _is_end_token(b):
            pair_end = i + 2
        else:
            continue
        head = toks[:i]
        if len(head) > 4 or not all(t in _END_FILLERS or _NAME_TOKEN.match(t) or _is_end_token(t) for t in head):
            return False
        tail = toks[pair_end:]
        # ekor: pengisi/nama/kata-akhir (ulangan) bebas, plus paling banyak 2 kata sisa "sampai jumpa" yang rusak
        unknown = [t for t in tail if not (t in _END_FILLERS or _NAME_TOKEN.match(t) or _is_end_token(t))]
        return len(unknown) <= 2
    return False


class _SessionEndMatcher:
    """Kompatibel dengan pemakaian lama SESSION_END_RE.search(text)."""

    def search(self, text: str) -> bool:
        return is_session_end(text)


SESSION_END_RE = _SessionEndMatcher()

def matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)
