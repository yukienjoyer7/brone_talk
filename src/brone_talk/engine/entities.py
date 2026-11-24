"""entities: bagian mesin jawaban v4 (dipindah dari brone_v4.py tanpa perubahan logika)."""

from __future__ import annotations

import re


from brone_talk.engine.util import norm_text

_WH = {"siapa": "siapa", "kapan": "kapan", "mana": "mana", "dimana": "mana", "berapa": "berapa",
       "bagaimana": "bagaimana", "gimana": "bagaimana", "kenapa": "kenapa", "mengapa": "kenapa"}

def wh_words(text: str) -> set[str]:
    """Kata tanya utama (siapa/kapan/mana/berapa/bagaimana/kenapa) dalam teks."""
    return {_WH[t] for t in norm_text(text).split() if t in _WH}

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

_ROLES = ["sekretaris universitas", "wakil rektor", "wakil dekan", "wakil direktur", "ketua departemen",
          "rektor", "dekan", "direktur", "warek"]

_ROLE_CANON = {"warek": "wakil rektor"}

class EntityMatcher:
    def __init__(self, entities: dict[str, dict]) -> None:
        pairs = [(a, code) for code, e in entities.items() for a in e["alias"]]
        pairs.sort(key=lambda p: len(p[0]), reverse=True)  # frasa terpanjang menang
        self._pairs = pairs

    def faculties(self, text: str) -> set[str]:
        t = f" {norm_text(text)} "
        found: set[str] = set()
        for alias, code in self._pairs:
            pat = f" {alias} "
            while pat in t:
                found.add(code)
                t = t.replace(pat, " # ", 1)
        return found

    def strip(self, text: str) -> tuple[str, set[str]]:
        """(teks ternormalisasi dengan sebutan fakultas diganti '#', kode fakultas yang disebut)."""
        t = f" {norm_text(text)} "
        found: set[str] = set()
        for alias, code in self._pairs:
            pat = f" {alias} "
            while pat in t:
                found.add(code)
                t = t.replace(pat, " # ", 1)
        return " ".join(t.split()), found

    def aliases(self, code: str) -> list[str]:
        return [a for a, c in self._pairs if c == code]

    @staticmethod
    def roles(text: str) -> set[str]:
        t = f" {norm_text(text)} "
        found: set[str] = set()
        for role in _ROLES:
            pat = f" {role} "
            while pat in t:
                found.add(_ROLE_CANON.get(role, role))
                t = t.replace(pat, " # ", 1)
        return found
