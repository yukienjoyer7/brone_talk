"""Normalisasi teks untuk TTS: angka, akronim, URL, tanggal, dan singkatan dibaca sebagai kata.

Dipindah apa adanya dari brone_talk.py (v4). Fungsi di sini murni (tanpa I/O),
sehingga bisa diuji lewat tests/test_tts_normalize.py dengan fixture golden.
"""

from __future__ import annotations

import math
import re
from typing import Any


_ID_LETTER_NAMES = {
    "a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "ef", "g": "ge",
    "h": "ha", "i": "i", "j": "je", "k": "ka", "l": "el", "m": "em", "n": "en",
    "o": "o", "p": "pe", "q": "ki", "r": "er", "s": "es", "t": "te", "u": "u",
    "v": "ve", "w": "we", "x": "eks", "y": "ye", "z": "zet",
}

_EN_LETTER_NAMES = {
    "a": "ei", "b": "bi", "c": "si", "d": "di", "e": "i", "f": "ef", "g": "ji",
    "h": "ej", "i": "ai", "j": "jei", "k": "kei", "l": "el", "m": "em", "n": "en",
    "o": "o", "p": "pi", "q": "kyu", "r": "ar", "s": "es", "t": "ti", "u": "yu",
    "v": "vi", "w": "dabelyu", "x": "eks", "y": "wai", "z": "zi",
}


def _spell_out(token: str, letters: dict[str, str]) -> str:
    return " ".join(letters[c] for c in token.lower())


# Singkatan lembaga/kode lokal UB -- tidak enak diucap sebagai kata, dieja
# pakai nama alfabet Indonesia (contoh terverifikasi: UB -> "u be",
# SMUB -> "es em u be").
_ID_SPELLED_ACRONYMS = [
    "UB", "SMUB", "SNBT", "SNBP", "UTBK", "FEB", "FH", "FK", "FKG", "FKH",
    "FKHP", "FIA", "FISIP", "FIKES", "FTP", "FTAB", "FT", "PTNBH", "UKM",
    "UKT", "RSUB", "PSDKU", "BEI", "BRI", "KRI", "WR", "QA", "SK", "PT",
    "UI", "UGM", "RI", "TSG", "AL", "ADL", "CKL", "GL", "LG", "SPUB",
    "RT", "RW",
]

# Singkatan asal Inggris/internasional -- dieja pakai nama alfabet gaya
# Inggris (contoh terverifikasi: PhD -> "pi ej di", AI -> "ei ai").
_EN_SPELLED_ACRONYMS = ["PhD", "AI", "BISS", "QS", "THE", "WUR", "AUN", "ISO", "AWS", "SES"]

# Akronim yang memang enak diucap sebagai kata (FILKOM, PIMNAS, IPI) sengaja
# TIDAK dimasukkan -- dibiarkan dibaca apa adanya, tanpa dieja per huruf.
PRONOUNCE_MAP: dict[str, str] = {}
for _tok in _ID_SPELLED_ACRONYMS:
    PRONOUNCE_MAP[_tok] = _spell_out(_tok, _ID_LETTER_NAMES)
for _tok in _EN_SPELLED_ACRONYMS:
    PRONOUNCE_MAP[_tok] = _spell_out(_tok, _EN_LETTER_NAMES)

# Istilah/kata pinjaman Inggris -- direspell fonetis ke ejaan Indonesia
# supaya Piper (grafem Indonesia) mendekati lafal Inggris aslinya, bukan
# dibaca literal (mis. "cloud" tanpa respelling dibaca "klo-ud", "research"
# dibaca dengan akhiran serak/kh karena "c" Indonesia selalu jadi /tʃ/).
PRONOUNCE_MAP.update({
    "FMIPA": "ef mipa",
    "cloud": "klaud",
    "WiFi": "waifai",
    "email": "im-el",
    "software": "softwer",
    "computing": "kom-pyu-ting",
    "research": "riserj",
    "scholarship": "skolarsyip",
    "exchange": "ikscéinj",
    "publication": "pa-bli-ke-syen",
    "coverage": "kaverej",
    "services": "servis",
    "online": "on-lain",
    "startup": "start ap",
    "wellbeing": "wel-biyying",
    "platform": "plat-form",
    "learning": "ler-ning",
    # "machine" tanpa respelling dibaca "ma-khi-ne" (maχˈinə). Frasa lengkap
    # didahulukan (pola diurutkan dari yang terpanjang) supaya "learning" di
    # dalamnya pakai "lerning" satu tekanan: mˈɛçin lˈɛrniŋ.
    "Machine Learning": "mesyin lerning",
    "machine": "mesyin",
    "university": "yuniversiti",
    "ranking": "rengking",
    "rankings": "rengkings",
    "world": "worlt",
    "news": "nyus",
    "antaranews": "antaranyus",
    # Moto UB (bagian 1 knowledge base): "Building Up Noble Future".
    "Building Up Noble Future": "bilding ap,,, noubel fyucer",
})

_PRONOUNCE_LOWER_MAP = {k.lower(): v for k, v in PRONOUNCE_MAP.items()}
_PRONOUNCE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(PRONOUNCE_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def apply_pronunciation_fixes(text: str) -> str:
    """Ganti singkatan/istilah asing dengan ejaan yang benar didengar Piper.

    Pencocokan case-insensitive (LLM bisa menulis kapitalisasi berbeda-beda),
    tapi tetap pakai batas kata supaya tidak mengenai potongan kata lain
    (mis. "AI" tidak kena di dalam "SANTAI").
    """
    return _PRONOUNCE_PATTERN.sub(lambda m: _PRONOUNCE_LOWER_MAP[m.group(0).lower()], text)


# Pelafalan alamat website (mis. "selma.ub.ac.id"). Terverifikasi lewat
# dengar langsung: setiap titik dibaca ",,, dot " (bukan "titik" -- lebih
# jelas kedengaran sebagai alamat web), dan potongan TLD pendek yang tidak
# enak diucap dieja per huruf. Subdomain lain (selma, ub, rsub, dst.) dicek
# ke PRONOUNCE_MAP juga supaya konsisten dengan singkatan di luar website.
#
# Hasil eja website (mis. "ai di" utk ".id") DILINDUNGI dari pemrosesan
# ulang lewat placeholder -- tanpa ini, kata "ai" hasil eja ".id" akan
# tertangkap ulang oleh aturan singkatan "AI" dan berubah jadi "ei ai di".
_TLD_SPOKEN = {
    "ac": "a ce",
    "id": "ai di",
    "co": "ce o",
    "go": "ge o",
    "www": "we we we",
    "com": "kom",
}

_URL_PATTERN = re.compile(
    r"\b(?:[a-zA-Z0-9][a-zA-Z0-9-]*\.)+(?:ac\.id|co\.id|go\.id|id|com|org|net)\b",
    re.IGNORECASE,
)

_URL_PLACEHOLDER = "\x00W{}\x00"


def apply_website_pronunciation(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def repl(match: "re.Match[str]") -> str:
        labels = match.group(0).split(".")
        spoken = [
            _TLD_SPOKEN.get(label.lower(), _PRONOUNCE_LOWER_MAP.get(label.lower(), label))
            for label in labels
        ]
        key = _URL_PLACEHOLDER.format(len(placeholders))
        placeholders[key] = ",,, dot ".join(spoken)
        return key

    return _URL_PATTERN.sub(repl, text), placeholders


# ─── Normalisasi singkatan & angka untuk TTS ─────────────────────────────
# System prompt sudah menyuruh LLM menulis kata lengkap dan memberi penanda
# ("nomor", "telepon", "kode pos") di depan angka -- tapi model 0.8B tetap
# bisa bocor ("Jl. Veteran", "No. 12"), dan knowledge base memuat singkatan
# yang bisa tersalin. Lapisan ini deterministik: singkatan diperluas, lalu
# angka yang BUKAN besaran (telepon, kode pos, nomor rumah/surat, RT/RW,
# NIM/NIK) dieja per digit dengan 0 = "kosong". Tahun, jumlah, harga, umur
# sengaja TIDAK disentuh -- tetap dibaca Piper sebagai bilangan.
_DIGIT_WORDS = ("kosong", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan")


def _spell_digits(digits: str) -> str:
    return " ".join(_DIGIT_WORDS[int(d)] for d in digits if d.isdigit())


def _spell_number_token(token: str) -> str:
    """'12A' -> 'satu dua a' (huruf blok/rumah dieja pakai alfabet Indonesia)."""
    out = []
    for ch in token:
        if ch.isdigit():
            out.append(_DIGIT_WORDS[int(ch)])
        elif ch.lower() in _ID_LETTER_NAMES:
            out.append(_ID_LETTER_NAMES[ch.lower()])
    return " ".join(out)


def _chunk_digits(digits: str) -> list[str]:
    """Bagi deret digit panjang jadi kelompok 3-4 supaya enak didengar."""
    if len(digits) <= 4:
        return [digits]
    n = math.ceil(len(digits) / 4)
    base, extra = divmod(len(digits), n)
    sizes = [base] * (n - extra) + [base + 1] * extra
    chunks, start = [], 0
    for size in sizes:
        chunks.append(digits[start:start + size])
        start += size
    return chunks


def _spell_phone(raw: str) -> str:
    """'(0341) 551611' -> 'kosong tiga empat satu, lima lima satu, enam satu satu'."""
    prefix = "plus " if raw.lstrip().startswith("+") else ""
    groups = [g for g in re.split(r"[\s()\-+.]+", raw) if g]
    spoken = [_spell_digits(chunk) for group in groups for chunk in _chunk_digits(group)]
    return prefix + ", ".join(spoken)


# (singkatan, kata lengkap). Diperluas bila diikuti titik, atau bila ditulis
# kapital dan diikuti spasi + kata kapital/angka ("Jl Veteran", "Lt 2").
_ADDRESS_ABBREVIATIONS = (
    ("Jln", "Jalan"), ("Jl", "Jalan"), ("Gg", "Gang"), ("Kec", "Kecamatan"),
    ("Kel", "Kelurahan"), ("Kab", "Kabupaten"), ("Prov", "Provinsi"),
    ("Lt", "Lantai"), ("Telp", "telepon"), ("Tlp", "telepon"), ("Abd", "Abdul"),
)
# Gelar depan: peka huruf besar ("Dr." Doktor vs "dr." dokter).
_TITLE_ABBREVIATIONS = (
    ("Prof", "Profesor"), ("Drs", "Doktorandus"), ("Dra", "Doktoranda"),
    ("Dr", "Doktor"), ("dr", "dokter"), ("Ir", "Insinyur"),
)
# Singkatan yang bisa menutup kalimat -- titiknya dipertahankan sebagai
# titik kalimat bila memang di akhir kalimat.
_TRAILING_ABBREVIATIONS = (
    ("dll", "dan lain-lain"), ("dsb", "dan sebagainya"), ("dst", "dan seterusnya"),
    ("dkk", "dan kawan-kawan"), ("tsb", "tersebut"),
)
_MONTH_ABBREVIATIONS = {
    "Jan": "Januari", "Feb": "Februari", "Mar": "Maret", "Apr": "April",
    "Jun": "Juni", "Jul": "Juli", "Agu": "Agustus", "Agt": "Agustus",
    "Ags": "Agustus", "Agus": "Agustus", "Sep": "September", "Sept": "September",
    "Okt": "Oktober", "Nov": "November", "Nop": "November", "Des": "Desember",
}
_MONTH_ALT = "|".join(sorted(_MONTH_ABBREVIATIONS, key=len, reverse=True))

_ABBREVIATION_RULES: list[tuple["re.Pattern[str]", Any]] = [
    # "Jl"/"Jln" bukan kata biasa, jadi tanpa titik pun tetap "Jalan" apa pun
    # kapitalisasinya ("jl veteran", "JL VETERAN") -- jangan sampai dieja "je el".
    (re.compile(r"\bJln?\b\.?\s*(?=\w)", re.I), "Jalan "),
]
for _abbr, _full in _ADDRESS_ABBREVIATIONS:
    _ABBREVIATION_RULES.append((re.compile(rf"\b{_abbr}\.\s*", re.I), _full + " "))
    _ABBREVIATION_RULES.append((re.compile(rf"\b{_abbr}\s+(?=[A-Z0-9])"), _full + " "))
for _abbr, _full in _TITLE_ABBREVIATIONS:
    _ABBREVIATION_RULES.append((re.compile(rf"\b{_abbr}\.\s*"), _full + " "))
    _ABBREVIATION_RULES.append((re.compile(rf"\b{_abbr}\s+(?=[A-Z])"), _full + " "))
for _abbr, _full in _TRAILING_ABBREVIATIONS:
    _ABBREVIATION_RULES.append((
        re.compile(rf"\b{_abbr}\b(\.)?(?=(\s*$|\s+[A-Z])?)", re.I),
        lambda m, full=_full: full + ("." if m.group(1) and m.group(2) is not None else ""),
    ))
_ABBREVIATION_RULES += [
    # "No. 12" / "No 12" -> "nomor 12" (hanya bila diikuti angka; "no" biasa aman).
    (re.compile(r"\bNo\.\s*(?=\d)", re.I), "nomor "),
    (re.compile(r"\bNo\s+(?=\d)"), "nomor "),
    (re.compile(r"\bext\.?\s*(?=\d)", re.I), "ekstensi "),
    # Bulan hanya diperluas di konteks tanggal supaya nama/kata lain aman.
    (re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\b\.?"),
     lambda m: f"{m.group(1)} {_MONTH_ABBREVIATIONS[m.group(2)]}"),
    (re.compile(rf"\b({_MONTH_ALT})\.?\s+(?=\d{{4}}\b)"),
     lambda m: _MONTH_ABBREVIATIONS[m.group(1)] + " "),
    # "Rp 40–67,5 juta" -> "40–67,5 juta rupiah" (satuan pindah ke belakang).
    (re.compile(r"\bRp\.?\s*(\d(?:[\d.,]*\d)?(?:\s*[–-]\s*\d(?:[\d.,]*\d)?)?(?:\s*(?:ribu|juta|miliar|triliun)\b)?)", re.I),
     r"\1 rupiah"),
    (re.compile(r"±\s*"), "sekitar "),
    (re.compile(r"(\d)\s*ha\b"), r"\1 hektare"),
]

# Gelar belakang ("S.Si.", "M.Si.", "Ph.D.Med.Sc", "M.A.B.") -- minimal dua
# segmen bertitik, tiap segmen diawali huruf kapital. Dieja per huruf dengan
# alfabet Indonesia; "Ph.D" dieja gaya Inggris ("pi ej di", sama seperti PhD
# di PRONOUNCE_MAP).
_DEGREE_PATTERN = re.compile(r"\b(?:[A-Z][A-Za-z]{0,3}\.)+[A-Z][A-Za-z]{0,3}\b(\.)?")


def _spell_degree(match: "re.Match[str]") -> str:
    segments = [s for s in match.group(0).split(".") if s]
    spoken, i = [], 0
    while i < len(segments):
        if segments[i] == "Ph" and i + 1 < len(segments) and segments[i + 1] == "D":
            spoken.append(_spell_out("PhD", _EN_LETTER_NAMES))
            i += 2
            continue
        spoken.append(_spell_out(segments[i], _ID_LETTER_NAMES))
        i += 1
    # Titik terakhir dipertahankan hanya bila gelar menutup kalimat.
    rest = match.string[match.end():]
    end_dot = "." if match.group(1) and (not rest.strip() or re.match(r"\s+[A-Z]", rest)) else ""
    return " ".join(spoken) + end_dot


_RANK_BEFORE = re.compile(r"\b(peringkat|ranking|rangking|juara|urutan)\s*$", re.I)


def _nomor_repl(match: "re.Match[str]") -> str:
    # "peringkat nomor 12" itu urutan, bukan kode -- dibaca sebagai bilangan.
    if _RANK_BEFORE.search(match.string[: match.start()]):
        return match.group(0)
    word, number, tail = match.group(1), match.group(2), match.group(3) or ""
    spoken = f"{word} {_spell_number_token(number)}"
    # Nomor surat "27/1981" atau "957/SK/BAN-PT/Ak.Ppj/PT/XI/2023": kode huruf
    # di tengah tidak dibacakan, cukup nomor + tahunnya.
    year = re.search(r"/(\d{4})$", tail)
    if year:
        spoken += f" tahun {year.group(1)}"
    return spoken


# Nomor rumah/jalan/surat. Titik hanya boleh DI DALAM segmen kode ("Ak.Ppj")
# supaya titik akhir kalimat tidak ikut tertelan. Dijalankan SEBELUM ejaan
# gelar, karena kode surat seperti "Ak.Ppj" juga cocok dengan pola gelar.
_NOMOR_PATTERN = re.compile(r"\b(nomor)\s+(\d+[A-Za-z]?)\b((?:/[\w\-]+(?:\.[\w\-]+)*)*)", re.I)

_NUMBER_RULES: list[tuple["re.Pattern[str]", Any]] = [
    # RT/RW: "RT 03" -> "er te kosong tiga".
    (re.compile(r"\b(RT|RW)\.?\s*(\d+)\b"),
     lambda m: f"{_spell_out(m.group(1), _ID_LETTER_NAMES)} {_spell_digits(m.group(2))}"),
    # NIM/NIK/NIP/NPWP/kode pos dengan penanda eksplisit.
    (re.compile(r"\b(NIM|NIK|NIP|NPWP|nomor induk(?: \w+)?|kode pos)\s*:?\s*(\d[\d.\-\s]*\d)", re.I),
     lambda m: f"{m.group(1)} {_spell_digits(m.group(2))}"),
    # Ekstensi telepon: "ekstensi 117/118".
    (re.compile(r"\b(ekstensi)\s+(\d+)(?:/(\d+))?", re.I),
     lambda m: f"{m.group(1)} {_spell_digits(m.group(2))}"
               + (f" atau {_spell_digits(m.group(3))}" if m.group(3) else "")),
    # Telepon: "(0341) 551611", "0341-551611", "0812 3456 7890", "+62 812...".
    (re.compile(
        r"(?<![\w.,:/])(?:\+62[\s-]?\d{2,4}|\(0\d{2,4}\)|0\d{2,12})(?:[\s-]\d{2,8}){0,3}(?![\w]|[.,:/]\d)"),
     lambda m: _spell_phone(m.group(0))
               if sum(c.isdigit() for c in m.group(0)) >= 7 else m.group(0)),
    # Kode pos: kata kapital (nama kota/provinsi) + 5 digit, "Malang 65145".
    (re.compile(
        r"(?<=[A-Za-z])(,?\s+)([1-9]\d{4})\b(?![.,]\d)"
        r"(?!\s*(?:rupiah|orang|mahasiswa|pendaftar|hektare|meter|km|kali|buah|unit|dosen))"),
     lambda m: f"{m.group(1)}{_spell_digits(m.group(2))}"
               if re.search(r"\b[A-Z][a-z]+$", m.string[: m.start()]) else m.group(0)),
    # Deret angka polos >= 10 digit bukan besaran (NIK, nomor rekening, dsb.).
    (re.compile(r"(?<![\d.,])\d{10,}(?![\d.,])"), lambda m: _spell_digits(m.group(0))),
]


def normalize_for_speech(text: str) -> str:
    """Perluas singkatan dan eja angka non-besaran supaya dilafalkan benar."""
    for pattern, repl in _ABBREVIATION_RULES:
        text = pattern.sub(repl, text)
    text = _NOMOR_PATTERN.sub(_nomor_repl, text)
    text = _DEGREE_PATTERN.sub(_spell_degree, text)
    for pattern, repl in _NUMBER_RULES:
        text = pattern.sub(repl, text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def strip_markdown(text: str) -> str:
    """Buang markdown dari jawaban LLM supaya tidak ikut dibacakan."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*[#>\-*+]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def clean_for_speech(text: str) -> str:
    """Make model output pleasant for TTS without reading markdown aloud."""
    text = strip_markdown(text)
    text, url_placeholders = apply_website_pronunciation(text)
    # Sebelum apply_pronunciation_fixes: hasil eja di sini ("er te", "es es i",
    # "kosong tiga") tidak mengandung kunci PRONOUNCE_MAP, jadi aman.
    text = normalize_for_speech(text)
    text = apply_pronunciation_fixes(text)
    for key, spoken in url_placeholders.items():
        text = text.replace(key, spoken)
    return text


