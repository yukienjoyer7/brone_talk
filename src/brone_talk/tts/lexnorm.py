"""Teks -> bentuk ucapan untuk suara Piper hasil fine-tune BRONE (D21).

Suara D21 dilatih dengan transkrip dari textnorm.to_spoken (repo dataset brone_voxcpm_dataset), jadi teks
saat deploy harus melewati normalisasi yang sama: singkatan dari lexicon.tsv ("es en be te", "be-a-en-pe-te"),
angka/jam/rupiah ditulis huruf, "gigi" -> "gi gi", website "u be dot a ce dot i de".
textnorm.py + lexicon.tsv ikut model di models/voices/brone_D21_teks/ (BRONE_TTS_TEXTNORM).
BRONE_TTS_TEXTNORM kosong (deploy/tts_voice.sh lama): Speaker memakai clean_for_speech (suara news_tts lama).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Callable

from brone_talk.tts.normalize import (
    PRONOUNCE_MAP,
    _EN_SPELLED_ACRONYMS,
    _ID_SPELLED_ACRONYMS,
    normalize_for_speech,
    strip_markdown,
)


_IYA_SENDIRI = re.compile(r"(^|[.!?]\s+)i?ya\s*[.!?]*(?=\s|$)", re.I)  # "Iya." dan sapaan bangun "Ya?"


def load_spoken(textnorm_dir: Path) -> Callable[[str], str]:
    """Bangun fungsi teks -> ucapan dari textnorm.py di <textnorm_dir>/tools (lexicon.tsv di <textnorm_dir>)."""
    source = textnorm_dir / "tools" / "textnorm.py"
    spec = importlib.util.spec_from_file_location("brone_textnorm", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"textnorm tidak ditemukan: {source}")
    textnorm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(textnorm)
    textnorm.to_spoken("UB")  # muat lexicon.tsv sekarang: path salah ketahuan saat start, bukan saat bicara

    # Entri PRONOUNCE_MAP lama yang tidak ada di lexicon (BRI, UGM, PhD, cloud, machine learning, ...) tetap dipakai,
    # SETELAH to_spoken supaya tidak mendahului lexicon (mis. "PT" di dalam "BAN-PT").
    # Singkatan dicocokkan huruf besar persis: hasil to_spoken sudah huruf kecil ("es ka"), jadi tidak kena ulang.
    lexicon = {word for word, _ in textnorm.load_lexicon()}
    extra = {k: v for k, v in PRONOUNCE_MAP.items() if k not in lexicon}
    acronyms = {k: v for k, v in extra.items() if k in _ID_SPELLED_ACRONYMS or k in _EN_SPELLED_ACRONYMS}
    loans = {k.lower(): v for k, v in extra.items() if k not in acronyms}
    acronym_alt = "|".join(sorted(map(re.escape, acronyms), key=len, reverse=True))
    acronym_pattern = re.compile(r"(?<![\w-])(" + acronym_alt + r")(?![\w-])")
    loan_pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, loans), key=len, reverse=True)) + r")\b", re.I)

    def spoken(text: str) -> str:
        text = strip_markdown(text)
        text = re.sub(r"(?<![\w.])@(?=\w)", "", text)  # akun IG "@filkomub" -> "filkom u be" (email tetap)
        text = normalize_for_speech(text)  # Jl./No./gelar, telepon/NIM/RT dieja per digit
        text = textnorm.to_spoken(text)
        text = acronym_pattern.sub(lambda m: acronyms[m.group(0)], text)
        text = loan_pattern.sub(lambda m: loans[m.group(0).lower()], text)
        # "Iya." sebagai kalimat sendiri terlalu pendek & pelan di D21 (terdengar "a" saja; "iya" jarang di dataset).
        # Pilihan uji dengar 26 Okt 2026: dipisah + "a" dipanjangkan + seru. "Iya, ..." di dalam kalimat sudah jelas.
        return _IYA_SENDIRI.sub(r"\1I yaa!", text)

    return spoken
