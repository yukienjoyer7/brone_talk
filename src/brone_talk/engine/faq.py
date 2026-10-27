"""faq: bagian mesin jawaban v4 (dipindah dari brone_v4.py tanpa perubahan logika)."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from brone_talk.engine.servers import Embedder
from brone_talk.engine.entities import EntityMatcher, _YEAR_RE, wh_words
from brone_talk.engine.util import LOG, warn_stale

@dataclass
class FaqHit:
    faq: dict
    score: float
    matched: str
    margin: float
    rejected: list[tuple[str, float, str]] = field(default_factory=list)

def load_smalltalk(path: Path) -> list[dict]:
    """kb/smalltalk.yaml -> rekaman berbentuk FAQ (guard umum; satu jawaban dipilih acak per giliran)."""
    if not path.is_file():
        return []
    rows = []
    for o in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("obrolan", []):
        answers = [str(a).strip() for a in o["jawaban"]]
        rows.append({"id": o["id"], "q": o["q"], "variasi": [str(v) for v in o.get("variasi", [])],
                     "a": answers[0], "jawaban": answers, "guard": "umum", "entitas": None,
                     "berlaku_hingga": None, "tanya_balik": bool(o.get("tanya_balik")),
                     "ekspresi": o.get("ekspresi", "happy")})
    return rows

class FaqIndex:
    def __init__(self, faq_path: Path | None, embedder: Embedder, entities: EntityMatcher,
                 min_sim: float, min_margin: float, records: list[dict] | None = None) -> None:
        self.embedder = embedder
        self.entities = entities
        self.min_sim = min_sim
        self.min_margin = min_margin
        today = _dt.date.today().isoformat()
        self.faqs: list[dict] = []
        if records is None:
            records = [json.loads(line) for line in faq_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        warn_stale("FAQ", records, today)
        for f in records:
            if f.get("berlaku_hingga") and str(f["berlaku_hingga"]) < today:
                continue  # data basi tidak dijawab dari FAQ
            f["_roles"] = EntityMatcher.roles(f["q"])
            texts = [f["q"], *f.get("variasi", [])]
            f["_wh"] = set().union(*(wh_words(t) for t in texts))
            f["_years"] = set(_YEAR_RE.findall(" ".join([*texts, f["a"]])))
            self.faqs.append(f)
        self._texts: list[str] = []
        self._owner: list[int] = []
        for i, f in enumerate(self.faqs):
            for t in [f["q"], *f.get("variasi", [])]:
                self._texts.append(t)
                self._owner.append(i)
        self._mat: np.ndarray | None = None

    @property
    def sentence_count(self) -> int:
        """Jumlah kalimat ter-embed (pertanyaan + variasinya)."""
        return len(self._texts)

    def build(self) -> None:
        self._mat = self.embedder.embed(self._texts, timeout_s=120)
        self._owner_arr = np.asarray(self._owner)

    def _guard(self, faq: dict, query: str) -> str:
        """'' bila lolos, atau alasan penolakan."""
        q_fac = self.entities.faculties(query)
        guard, ent = faq.get("guard", "umum"), faq.get("entitas")
        if guard == "umum" and q_fac:
            return f"FAQ umum, pertanyaan menyebut {sorted(q_fac)}"
        if guard == "longgar" and not q_fac <= {ent}:
            return f"FAQ {ent}, pertanyaan menyebut {sorted(q_fac)}"
        if guard == "ketat" and q_fac != {ent}:
            return f"FAQ {ent}, pertanyaan menyebut {sorted(q_fac) or 'tidak ada fakultas'}"
        q_roles = EntityMatcher.roles(query)
        if q_roles and q_roles != faq["_roles"]:
            return f"peran {sorted(q_roles)} != {sorted(faq['_roles'])}"
        # "kapan FT didirikan" tidak boleh dijawab FAQ "di mana FT"
        q_wh = wh_words(query)
        if q_wh and faq["_wh"] and not (q_wh & faq["_wh"]):
            return f"kata tanya {sorted(q_wh)} bukan {sorted(faq['_wh'])}"
        # "rektor tahun 2010" tidak boleh dijawab FAQ rektor sekarang
        q_years = set(_YEAR_RE.findall(query))
        if q_years - faq["_years"]:
            return f"tahun {sorted(q_years - faq['_years'])} tidak ada di FAQ"
        return ""

    def match(self, query: str, qvec: np.ndarray, min_sim: float | None = None) -> FaqHit | None:
        """min_sim: ambang lebih ketat untuk giliran ini (tidak pernah lebih longgar dari self.min_sim)."""
        assert self._mat is not None
        threshold = self.min_sim if min_sim is None else max(self.min_sim, min_sim)
        sims = self._mat @ qvec
        best: dict[int, tuple[float, int]] = {}
        for j in np.argsort(-sims)[:40]:
            o = int(self._owner_arr[j])
            if o not in best:
                best[o] = (float(sims[j]), int(j))
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        rejected: list[tuple[str, float, str]] = []
        blocked: set[int] = set()
        for o, (score, j) in ranked[:5]:
            if score < threshold:
                break
            faq = self.faqs[o]
            reason = self._guard(faq, query)
            if reason:
                rejected.append((faq["id"], round(score, 4), reason))
                blocked.add(o)
                continue
            # selisih dengan FAQ lain yang juga lolos pengaman; kecil = ambigu
            others = [s for oo, (s, _j) in ranked if oo != o and oo not in blocked]
            margin = score - (max(others) if others else 0.0)
            if margin < self.min_margin:
                rejected.append((faq["id"], round(score, 4), f"margin {margin:.3f} < {self.min_margin}"))
                return None
            return FaqHit(faq, score, self._texts[j], margin, rejected)
        if rejected:
            LOG.info("FAQ ditolak pengaman: %s", rejected)
        return None
