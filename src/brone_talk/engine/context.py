"""Konteks percakapan satu giliran: pertanyaan lanjutan yang belum lengkap dilengkapi dari pertanyaan sebelumnya.

Contoh: "di mana lokasi Filkom?" lalu "kalau FT?" -> "Di mana Fakultas Teknik?".

Yang disimpan hanya dua slot dari SATU giliran terakhir: entitas (fakultas) dan atribut (lokasi, dekan, prodi, ...),
dalam bentuk yang sudah dilengkapi, sehingga rantai ("kalau FT?" lalu "dekannya?") tetap jalan. Tanpa embedding dan
tanpa LLM. Prinsip: bila ragu, JANGAN mewarisi (sama dengan perilaku sebelum fitur ini).

Aturan (query baru; N = jumlah kata, batas followup_max_words):
  entitas + atribut                      -> lengkap, topik baru, tidak mewarisi
  hanya entitas ("kalau FT?")            -> mewarisi ATRIBUT lama (sisa kata hanya kata pengisi)
  hanya atribut fakultas ("dekannya?")   -> mewarisi ENTITAS lama (sisa kata hanya kata pengisi/tanya, dan ada
                                            penanda lanjutan: akhiran -nya, "di sana/situ/tersebut", "dan/kalau/terus")
  selain itu (atribut universitas, kata tanya + kata lain yang tak dikenal, >1 entitas/atribut) -> tidak mewarisi
Slot kedaluwarsa setelah followup_ttl_s detik dan dibersihkan saat sesi berakhir (AnswerEngine.reset).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from brone_talk.engine.entities import EntityMatcher, wh_words
from brone_talk.engine.util import norm_text

# atribut terikat fakultas: kata kunci (satu kata atau frasa) dan kalimat baku untuk menyusun pertanyaan utuh
ATTRS: dict[str, dict] = {
    "lokasi": {"kw": ["lokasi", "alamat", "letak", "posisi", "tempat"], "tpl": "Di mana {nama}?"},
    "dekan": {"kw": ["dekan", "pimpinan", "pemimpin"], "tpl": "Siapa dekan {nama}?"},
    "wakil-dekan": {"kw": ["wakil dekan"], "tpl": "Siapa saja wakil dekan {nama}?"},
    "prodi": {"kw": ["prodi", "jurusan", "program studi", "departemen"], "tpl": "Apa saja program studi di {nama}?"},
    "sejarah": {"kw": ["sejarah", "riwayat", "berdiri", "didirikan", "awal mula"], "tpl": "Bagaimana sejarah {nama}?"},
    "akreditasi": {"kw": ["akreditasi"], "tpl": None},
    "lab": {"kw": ["lab", "laboratorium"], "tpl": "Apa saja laboratorium di {nama}?"},
    "kontak": {"kw": ["telepon", "kontak", "hubungi", "nomor"], "tpl": None},
    "biaya": {"kw": ["biaya", "ukt", "ipi", "uang kuliah"], "tpl": None},
    "ketua-departemen": {"kw": ["ketua departemen"], "tpl": None},
}
# kata yang menandakan pertanyaan tingkat universitas: tidak pernah mewarisi entitas
_GLOBAL = ("ub", "universitas", "brawijaya", "rektor", "rektorat", "sekretaris universitas")
# kata pengisi yang boleh menyertai entitas saja ("kalau FT?", "bagaimana dengan FT juga?")
_FILLER = {"kalau", "kalo", "bagaimana", "gimana", "bagimana", "dengan", "dan", "terus", "lalu", "juga", "kok", "nah",
           "untuk", "ya", "dong", "deh", "sih", "kalau", "yang", "itu", "tadi", "nya", "fakultas", "sedangkan", "sementara"}
# penanda bahwa kalimat melanjutkan topik sebelumnya (wajib ada agar atribut saja mewarisi entitas)
_POINTER = {"sana", "situ", "tersebut"}
_CONTINUE = {"kalau", "kalo", "dan", "terus", "lalu", "juga", "bagaimana", "gimana", "bagimana"}
# kata yang boleh menyertai atribut saja ("alamatnya di mana?", "ada prodi apa saja di sana?")
_ATTR_PAD = {"di", "ke", "dari", "apa", "saja", "ada", "siapa", "berapa", "mana", "dimana", "sana", "situ", "itu", "ini",
             "tersebut", "kuliah", "nama", "yang", "adalah", "kalau", "dan", "juga", "ya", "kapan", "ceritakan", "jelaskan",
             "sebutkan", "tolong", "coba", "dong", "lengkap", "tepatnya", "persisnya", "sekarang"} | _FILLER


_KW_WORDS = {w for spec in ATTRS.values() for kw in spec["kw"] for w in kw.split()}


def _tokens(norm: str) -> list[str]:
    out = []
    for w in norm.split():
        if w != "#" and w.endswith("nya") and (len(w) > 6 or w[:-3] in _KW_WORDS):
            w = w[:-3]
        out.append(w)
    return out


def _find_attrs(norm: str, roles: set[str]) -> list[str]:
    """Atribut fakultas yang disebut (kata kunci atau peran); kata tanya 'mana' saja dihitung lokasi."""
    padded = f" {' '.join(_tokens(norm))} "
    found = []
    for name, spec in ATTRS.items():
        if any(f" {kw} " in padded for kw in spec["kw"]):
            found.append(name)
    if "dekan" in found and ("wakil dekan" in roles or " wakil dekan " in padded):
        found.remove("dekan")
    if "ketua-departemen" in found and "prodi" in found:
        found.remove("prodi")  # "ketua departemen" memuat kata departemen
    if not found and "mana" in wh_words(norm):
        found.append("lokasi")
    return found


@dataclass
class Resolution:
    original: str
    text: str                   # teks yang dipakai mesin (utuh bila mewarisi, selain itu = original)
    mode: str | None = None     # "atribut" | "entitas" | None
    entity: str | None = None
    attribute: str | None = None

    @property
    def inherited(self) -> bool:
        return self.mode is not None

    def as_log(self) -> dict:
        return {"mode": self.mode, "asli": self.original, "lengkap": self.text, "entitas": self.entity,
                "atribut": self.attribute}


class FollowUpContext:
    def __init__(self, entities: dict[str, dict], matcher: EntityMatcher, ttl_s: float = 90.0, max_words: int = 7,
                 clock=time.monotonic, faqs: list[dict] | None = None) -> None:
        self.entities = entities
        self.matcher = matcher
        self.grid: dict[tuple[str, str], str] = {}
        self._build_grid(faqs or [])
        self.ttl_s = ttl_s
        self.max_words = max_words
        self._clock = clock
        self.slots: dict | None = None

    def reset(self) -> None:
        self.slots = None

    def _build_grid(self, faqs: list[dict]) -> None:
        """(atribut, fakultas) -> pertanyaan utama FAQ yang menjawabnya, supaya hasil lengkap memilih FAQ itu
        persis (menghindari FAQ tetangga yang mirip, mis. 'prodi' vs 'akreditasi prodi')."""
        by_id, by_text = {}, {}
        for f in faqs:
            code = f.get("entitas")
            if f.get("guard") not in ("ketat", "longgar") or code not in self.entities:
                continue
            prefix = str(f.get("id", "")).split("-")[0]
            if prefix in ATTRS:
                by_id.setdefault((prefix, code), f["q"])
            else:
                _c, fa, _n = self.parse(f.get("q", ""))
                if len(fa) == 1:
                    by_text.setdefault((fa[0], code), f["q"])
        self.grid = {**by_text, **by_id}

    # -- parsing
    def parse(self, text: str) -> tuple[set[str], list[str], str]:
        norm_ent, codes = self.matcher.strip(text)
        attrs = _find_attrs(norm_ent, EntityMatcher.roles(text))
        return codes, attrs, norm_ent

    def _name(self, code: str) -> str:
        # singkatan (FT, FILKOM): sebagian FAQ manual hanya memakai singkatan, dan variasi FAQ otomatis memuat keduanya
        return self.entities[code]["singkatan"]

    def _compose(self, attr: str | None, code: str, original: str, prev_text: str, prev_code: str | None) -> str | None:
        if (attr, code) in self.grid:
            return self.grid[(attr, code)]
        tpl = ATTRS.get(attr or "", {}).get("tpl")
        if tpl:
            return tpl.format(nama=self._name(code))
        if prev_text and prev_code and prev_code != code:  # ganti sebutan entitas lama di kalimat lama
            for alias in sorted(self.matcher.aliases(prev_code), key=len, reverse=True):
                pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
                if pat.search(prev_text):
                    return pat.sub(self._name(code), prev_text, count=1)
        return None

    # -- inti
    def resolve(self, text: str) -> Resolution:
        res = Resolution(original=text, text=text)
        s = self.slots
        if not s or self._clock() - s["t"] > self.ttl_s:
            return res
        codes, attrs, norm_ent = self.parse(text)
        words = [w for w in _tokens(norm_ent)]
        if not words or len(words) > self.max_words or len(codes) > 1 or len(attrs) > 1:
            return res
        if any(f" {g} " in f" {norm_ent} " for g in _GLOBAL):
            return res
        rest = [w for w in words if w != "#"]
        if len(codes) == 1 and not attrs:
            code = next(iter(codes))
            # "bagaimana/gimana" di sini kata sambung ("kalau FT bagaimana?"), kata tanya lain ("siapa FT?") bukan lanjutan
            if code == s["entity"] or not s.get("attr") or (wh_words(text) - {"bagaimana"}):
                return res
            if any(w not in _FILLER for w in rest):
                return res
            full = self._compose(s["attr"], code, text, s.get("text", ""), s["entity"])
            if full:
                res.text, res.mode, res.entity, res.attribute = full, "atribut", code, s["attr"]
        elif not codes and len(attrs) == 1 and s.get("entity"):
            attr = attrs[0]
            if any(w not in _ATTR_PAD and w not in _KW_WORDS for w in rest):
                return res
            raw = norm_ent.split()
            marker = (any(w in _POINTER or w in _CONTINUE for w in raw)
                      or any(w.endswith("nya") and (len(w) > 6 or w[:-3] in _KW_WORDS) for w in raw))
            if not marker:
                return res
            code = s["entity"]
            tpl = ATTRS[attr]["tpl"]
            full = self.grid.get((attr, code)) or (tpl.format(nama=self._name(code)) if tpl
                                                   else f"{text.rstrip(' ?.!')} {self.entities[code]['singkatan']}?")
            res.text, res.mode, res.entity, res.attribute = full, "entitas", code, attr
        return res

    def update(self, resolved_text: str, route: str, faq: dict | None = None) -> None:
        """Simpan slot dari pertanyaan yang BARU dijawab (sudah lengkap). Basa-basi tidak mengubah slot."""
        if route in ("obrolan", "abaikan", "tidak_dimengerti", "gerakan") or route.startswith("obrolan_"):
            return
        codes, attrs, _ = self.parse(resolved_text)
        code = next(iter(codes)) if len(codes) == 1 else None
        attr = attrs[0] if len(attrs) == 1 else None
        if faq is not None:  # petunjuk dari FAQ yang menjawab, bila teksnya sendiri tidak cukup
            if code is None and faq.get("guard") in ("ketat", "longgar") and faq.get("entitas") in self.entities:
                code = faq["entitas"]
            if attr is None:
                prefix = str(faq.get("id", "")).split("-")[0]
                if prefix in ATTRS:
                    attr = prefix
                else:
                    _c, fa, _n = self.parse(faq.get("q", ""))
                    attr = fa[0] if len(fa) == 1 else None
        if code is None and attr is None:
            self.slots = None  # pertanyaan tingkat universitas / basa-basi: topik fakultas berakhir
        else:
            self.slots = {"entity": code, "attr": attr, "text": resolved_text, "t": self._clock()}
