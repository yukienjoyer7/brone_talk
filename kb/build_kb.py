#!/usr/bin/env python3
"""Bangun knowledge base BRONE dari knowledge.yaml.

Output:
  chunks.jsonl   chunk kecil (satu fakta/topik per chunk) + breadcrumb + sumber, untuk RAG versi baru
  knowledge.txt  format lama (BAGIAN N / [ Fxx ] / FAQ di BAGIAN 19) agar RagLite2/3 yang sekarang tetap jalan

Pemakaian:  python build_kb.py            (bangun + validasi)
            python build_kb.py --check    (validasi saja)
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import re
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "knowledge.yaml"
UKT_CSV = ROOT / "data" / "ukt_mandiri_2026.csv"
IPI_CSV = ROOT / "data" / "ipi_2026.csv"
OUT_JSONL = ROOT / "chunks.jsonl"
OUT_TXT = ROOT / "knowledge.txt"
OUT_FAQ = ROOT / "faq.jsonl"            # FAQ untuk pencocokan semantik (brone_v4.py)
OUT_ENT = ROOT / "entities.json"        # alias fakultas untuk pengaman FAQ
OUT_REVIEW = ROOT / "faq_review.md"     # daftar FAQ untuk direview manusia

# Nilai salah dari KB lama yang tidak boleh muncul lagi di teks mana pun.
FORBIDDEN = [
    r"KOREKSI", r"5,12 [Tt]riliun", r"12 golongan", r"PTNBH\W+sejak 2023", r"Kampus Biru",
    r"(?<!Profesor )Doktor dokter Wisnu", r"221\+", r"halusinasi", r"Agroteknologi",
]
TXT_UNIT_WARN = 2800      # RagLite2 memotong konteks di 3000 karakter
CHUNK_WARN = 900          # chunk baru idealnya 200-600 karakter


def rupiah(v: int) -> str:
    if v % 100_000 == 0:
        return f"{v / 1_000_000:g}".replace(".", ",") + " juta"
    return f"{v:,}".replace(",", ".")


def rentang(lo: int, hi: int) -> str:
    return f"{rupiah(lo)} rupiah" if lo == hi else f"{rupiah(lo)} sampai {rupiah(hi)} rupiah"


def load_csv(path: Path, key: str) -> dict[str, list[int]]:
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vals = [int(v) for k, v in row.items() if k.startswith(("ukt_", "kel_"))]
            rows[row[key].strip()] = vals
    return rows


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def norm_text(text: str) -> str:
    """Huruf kecil, tanpa tanda baca. Harus sama dengan normalisasi di brone_v4.py."""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(text).lower())).strip()


def _is_degree(seg: str) -> bool:
    if re.match(r"(menjabat|periode|sebelumnya|kembali|yaitu)\b", seg):
        return False
    if " " not in seg and "." in seg:          # satu token bertitik: S.T., M.Biotech., Ph.D.
        return True
    pieces = [p for p in re.split(r"[\s.()]+", seg) if p]
    return bool(pieces) and all(len(p) <= 6 for p in pieces) and ("." in seg or seg.isupper() or "(K)" in seg)


def strip_degrees(text: str) -> str:
    """'Profesor X, S.T., M.T., Ph.D., periode 2025 sampai 2030' -> 'Profesor X, periode 2025 sampai 2030'.

    Gelar belakang dibuang supaya jawaban FAQ yang diucapkan tidak panjang; gelar depan tetap.
    """
    segs = [s.strip() for s in str(text).split(",")]
    kept = [segs[0]] + [s for s in segs[1:] if not _is_degree(s)]
    return ", ".join(kept)


def faq_review(faqs: list[dict], meta: dict) -> str:
    guard_text = {"umum": "umum (ditolak bila pertanyaan menyebut fakultas)", "bebas": "bebas",
                  "longgar": "fakultas {e} (boleh tanpa menyebut fakultas)", "ketat": "fakultas {e} (wajib disebut)"}
    out = [f"# Review FAQ BRONE (KB v{meta['versi']})", "",
           f"{len(faqs)} FAQ. Centang yang sudah benar; koreksi jawaban di knowledge.yaml (FAQ manual) "
           "atau di data fakultas (FAQ otomatis), lalu jalankan `python build_kb.py`.", ""]
    for group, title in [(False, "FAQ manual"), (True, "FAQ otomatis dari data fakultas")]:
        out += [f"## {title}", ""]
        for i, f in enumerate([f for f in faqs if f["otomatis"] == group], 1):
            out.append(f"### {i}. [ ] {f['q']}  `{f['id']}`")
            out.append(f"- **Jawaban:** {f['a']}")
            out.append(f"- **Variasi:** " + "; ".join(f["variasi"]))
            extra = guard_text[f["guard"]].format(e=f["entitas"])
            if f.get("berlaku_hingga"):
                extra += f"; berlaku hingga {f['berlaku_hingga']}"
            out.append(f"- **Pengaman:** {extra}")
            out.append("")
    return "\n".join(out)


def main(check_only: bool = False) -> int:
    kb = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    sources = kb["sumber"]
    ukt = load_csv(UKT_CSV, "prodi")
    ipi = load_csv(IPI_CSV, "prodi")
    errors: list[str] = []
    warnings: list[str] = []

    def src_urls(ids, where):
        out = []
        for sid in ids or []:
            if sid not in sources:
                errors.append(f"{where}: sumber '{sid}' tidak ada di daftar sumber")
            else:
                out.append(sources[sid].get("url") or sources[sid]["judul"])
        if not ids:
            errors.append(f"{where}: fakta tanpa sumber")
        return out

    chunks: list[dict] = []

    def add_chunk(cid, judul, teks, sumber_ids, tipe="fakta", **extra):
        teks = clean(teks)
        chunk = {"id": cid, "judul": judul, "teks": teks, "tipe": tipe,
                 "sumber": src_urls(sumber_ids, cid), **{k: v for k, v in extra.items() if v}}
        if len(teks) > CHUNK_WARN:
            warnings.append(f"chunk {cid} panjang {len(teks)} karakter")
        chunks.append(chunk)
        return chunk

    # --- BAGIAN umum
    for bag in kb["bagian"]:
        judul_bag = bag["judul"].title().replace("Ub", "UB").replace("Ukt", "UKT").replace("Ipi", "IPI") \
            .replace("Filkom", "FILKOM").replace("Brone", "BRONE")
        for i, f in enumerate(bag["fakta"], 1):
            add_chunk(f"b{bag['nomor']:02d}.{i:02d}", f"{judul_bag} › {f['sub']}", f["teks"], f.get("sumber"),
                      berlaku_hingga=f.get("berlaku_hingga"))

    # --- FAKULTAS
    fak_txt_blocks = []
    for fk in kb["fakultas"]:
        nama = f"{fk['nama']} ({fk['singkatan']})"
        crumb = f"Fakultas › {fk['singkatan']}"
        sid = fk.get("sumber")
        lama = f" Sebelum 2 Mei 2026 bernama {fk['nama_lama']}." if fk.get("nama_lama") else ""
        kontak = [f"{nama} UB.{lama}"]
        if fk.get("alamat"):
            kontak.append(f"Alamat: {fk['alamat']}.")
        if fk.get("telepon"):
            kontak.append(f"Telepon {fk['telepon']}.")
        if fk.get("email"):
            kontak.append(f"Email {fk['email']}.")
        kontak.append(f"Website {fk['situs']}.")
        add_chunk(f"f.{fk['singkatan'].lower()}.profil", f"{crumb} › Profil dan kontak", " ".join(kontak), sid)

        pimpinan = clean(fk["dekan"])
        if not pimpinan.startswith("Sekolah"):
            dulu = f" (dahulu {fk['nama_lama']})" if fk.get("nama_lama") else ""
            pimpinan = f"Dekan {fk['nama']} ({fk['singkatan']}){dulu} adalah {pimpinan}"
        pimpinan = pimpinan.rstrip(".") + "."
        if fk.get("wakil_dekan"):
            pimpinan += " " + " ".join(clean(w).rstrip(".") + "." for w in fk["wakil_dekan"])
        add_chunk(f"f.{fk['singkatan'].lower()}.pimpinan", f"{crumb} › Pimpinan", pimpinan, sid)

        prodi = ""
        if fk.get("departemen"):
            prodi += f"{fk['singkatan']} memiliki {clean(fk['departemen']).rstrip('.')}. "
        prodi += f"Program studi di {fk['singkatan']}: {clean(fk['prodi'])}"
        add_chunk(f"f.{fk['singkatan'].lower()}.prodi", f"{crumb} › Departemen dan program studi", prodi, sid)

        if fk.get("sejarah"):
            add_chunk(f"f.{fk['singkatan'].lower()}.sejarah", f"{crumb} › Sejarah", fk["sejarah"], sid)

        biaya_lines = []
        for p in fk.get("biaya_prodi") or []:
            if p not in ukt and p not in ipi:
                errors.append(f"{fk['singkatan']}: prodi biaya '{p}' tidak ada di CSV UKT/IPI")
                continue
            parts = []
            if p in ukt:
                parts.append(f"UKT {rentang(min(ukt[p]), max(ukt[p]))} per semester")
            if p in ipi:
                parts.append(f"IPI {rentang(min(ipi[p]), max(ipi[p]))} sekali bayar")
            biaya_lines.append(f"{p}: " + ", ".join(parts) + ".")
        if biaya_lines:
            head = (f"Biaya mahasiswa baru jalur mandiri 2026 di {fk['singkatan']}, menurut Peraturan Rektor "
                    f"118 dan 123 Tahun 2025: ")
            # pecah agar tiap chunk tetap pendek
            group, n = [], 0
            for line in biaya_lines:
                group.append(line)
                if len(" ".join(group)) > 520:
                    n += 1
                    add_chunk(f"f.{fk['singkatan'].lower()}.biaya{n}", f"{crumb} › Biaya jalur mandiri 2026",
                              head + " ".join(group), ["selma_ukt", "selma_ipi", "detik_ukt2026"],
                              berlaku_hingga="2027-06-30")
                    group = []
            if group:
                n += 1
                add_chunk(f"f.{fk['singkatan'].lower()}.biaya{n}", f"{crumb} › Biaya jalur mandiri 2026",
                          head + " ".join(group), ["selma_ukt", "selma_ipi", "detik_ukt2026"],
                          berlaku_hingga="2027-06-30")

        # blok [ Fxx ] untuk knowledge.txt (ringkas: kontak, pimpinan, prodi, sejarah, biaya)
        # judul blok ikut memuat nama lama + nama prodi: RagLite2 memberi bonus skor untuk kecocokan judul
        prodi_judul = []
        for p in fk.get("biaya_prodi") or []:
            base = re.sub(r"\s*\(.*?\)", "", p).strip()
            if base not in prodi_judul:
                prodi_judul.append(base)
        title = f"[ {fk['kode']} ] {fk['nama'].upper()} ({fk['singkatan']})"
        if fk.get("nama_lama"):
            title += f" d.h. {fk['nama_lama']}"
        if prodi_judul:
            title += " | prodi: " + ", ".join(prodi_judul)
        lines = [title]
        for c in chunks:
            if c["id"].startswith(f"f.{fk['singkatan'].lower()}."):
                lines.append(c["teks"])
        # baris judul TIDAK dibungkus: RagLite2 hanya membaca baris pertama sebagai judul
        fak_txt_blocks.append("\n".join([lines[0]] + [textwrap.fill(l, 110) for l in lines[1:]]))

    # --- FAQ manual (juga ikut jadi chunk RAG)
    alias = kb.get("alias_fakultas") or {}
    kode_fak = {fk["kode"] for fk in kb["fakultas"]}
    for k in alias:
        if k not in kode_fak:
            errors.append(f"alias_fakultas: kode '{k}' tidak ada di daftar fakultas")
    faqs: list[dict] = []
    for q in kb["faq"]:
        fid = q.get("id") or ""
        where = f"faq '{fid or q.get('q')}'"
        if not fid:
            errors.append(f"{where}: tanpa id")
        entitas = q.get("entitas")
        if entitas not in (None, "bebas") and entitas not in kode_fak:
            errors.append(f"{where}: entitas '{entitas}' bukan kode fakultas atau 'bebas'")
        if len(q.get("variasi") or []) < 3:
            warnings.append(f"{where}: variasi kurang dari 3")
        add_chunk(f"faq.{fid}", f"FAQ › {q['q']}", q["a"], q.get("sumber"), tipe="faq",
                  pertanyaan=q["q"], berlaku_hingga=q.get("berlaku_hingga"))
        faqs.append({
            "id": fid, "q": clean(q["q"]), "variasi": [clean(v) for v in q.get("variasi") or []],
            "a": clean(q["a"]), "sumber": src_urls(q.get("sumber"), where),
            "berlaku_hingga": q.get("berlaku_hingga"), "entitas": None if entitas == "bebas" else entitas,
            "guard": {None: "umum", "bebas": "bebas"}.get(entitas, "longgar"), "otomatis": False,
        })

    # --- FAQ otomatis per fakultas (dekan, program studi, lokasi) dari data fakultas
    for fk in kb["fakultas"]:
        nama, s = fk["nama"], fk["singkatan"]
        sid = fk.get("sumber")
        lamas = re.split(r"\s+atau\s+", fk["nama_lama"]) if fk.get("nama_lama") else []
        lama = lamas[0] if lamas else ""
        base = {"sumber": src_urls(sid, f"faq otomatis {s}"), "berlaku_hingga": None,
                "entitas": fk["kode"], "guard": "ketat", "otomatis": True}
        if fk["kode"] == "F18":
            faqs.append({**base, "id": f"pimpinan-{s.lower()}", "q": f"Siapa direktur {nama}?",
                         "variasi": [f"siapa pimpinan {nama}", "direktur pascasarjana UB siapa",
                                     f"siapa yang memimpin {s}", "siapa direktur pascasarjana"],
                         "a": clean(strip_degrees(fk["dekan"])).rstrip(".") + "."})
        else:
            var = [f"dekan {s} siapa", f"siapa dekan {s}", f"siapa pimpinan {nama}", f"siapa yang memimpin {s}",
                   f"{nama} dekannya siapa"]
            var += [f"siapa dekan {x}" for x in lamas]
            faqs.append({**base, "id": f"dekan-{s.lower()}", "q": f"Siapa dekan {nama}?", "variasi": var,
                         "a": f"Dekan {nama} adalah {clean(strip_degrees(fk['dekan'])).rstrip('.')}."})
        var = [f"jurusan apa saja di {s}", f"{nama} punya prodi apa saja", f"prodi di {s} apa saja",
               f"di {s} ada jurusan apa", f"kalau kuliah di {nama} bisa ambil jurusan apa"]
        var += [f"jurusan di {x} apa saja" for x in lamas]
        faqs.append({**base, "id": f"prodi-{s.lower()}", "q": f"Apa saja program studi di {nama}?", "variasi": var,
                     "a": f"Program studi di {nama}: {clean(fk['prodi']).rstrip('.')}."})
        if fk.get("alamat"):
            a = f"{nama} beralamat di {clean(fk['alamat']).rstrip('.')}."
            var = [f"lokasi {s} di mana", f"alamat {nama}", f"{s} letaknya di mana", f"gedung {nama} ada di mana"]
            if fk.get("telepon"):
                a += f" Teleponnya {fk['telepon']}."
                var.append(f"nomor telepon {nama}")
            var += [f"{x} di mana" for x in lamas]
            faqs.append({**base, "id": f"lokasi-{s.lower()}", "q": f"Di mana {nama}?", "variasi": var, "a": a})

    manual_ids = {f["id"] for f in faqs if not f["otomatis"]}
    for fk in kb["fakultas"]:
        nama, s = fk["nama"], fk["singkatan"]
        if not fk.get("sejarah") or f"sejarah-{s.lower()}" in manual_ids:
            continue
        lamas = re.split(r"\s+atau\s+", fk["nama_lama"]) if fk.get("nama_lama") else []
        var = [f"sejarah {s}", f"asal usul {nama}", f"bagaimana sejarah {s}"] + [f"sejarah {x}" for x in lamas]
        # pertanyaan "kapan berdiri" hanya bila teks sejarah memuat tahun pendirian (sebelum 2026)
        if any(int(y) < 2026 for y in re.findall(r"\b(?:19|20)\d{2}\b", fk["sejarah"])):
            var += [f"kapan {nama} didirikan", f"kapan {s} berdiri", f"{nama} berdiri tahun berapa"]
        faqs.append({"id": f"sejarah-{s.lower()}", "q": f"Bagaimana sejarah {nama}?", "variasi": var,
                     "a": clean(fk["sejarah"]), "sumber": src_urls(fk.get("sumber"), f"faq sejarah {s}"),
                     "berlaku_hingga": None, "entitas": fk["kode"], "guard": "ketat", "otomatis": True})

    seen: dict[str, str] = {}
    ids: set[str] = set()
    for f in faqs:
        if f["id"] in ids:
            errors.append(f"faq id ganda: {f['id']}")
        ids.add(f["id"])
        for t in [f["q"], *f["variasi"]]:
            key = norm_text(t)
            if key in seen and seen[key] != f["id"]:
                errors.append(f"kalimat FAQ '{t}' dipakai di '{seen[key]}' dan '{f['id']}'")
            seen[key] = f["id"]

    # --- validasi isi terlarang
    blob = json.dumps(chunks, ensure_ascii=False)
    for pat in FORBIDDEN:
        for m in re.finditer(pat, blob):
            errors.append(f"teks terlarang '{pat}' ditemukan: …{blob[max(0, m.start() - 60):m.end() + 40]}…")

    # --- knowledge.txt (format lama)
    meta = kb["meta"]
    out = [
        "=" * 80,
        f"  KNOWLEDGE BASE UNIVERSITAS BRAWIJAYA - BRONE v{meta['versi']}",
        f"  Terakhir diperbarui: {meta['diperbarui']}. Dibangun otomatis dari knowledge.yaml; jangan edit file ini langsung.",
        "=" * 80,
        "",
    ]
    by_no = {b["nomor"]: b for b in kb["bagian"]}
    for no in sorted(set(by_no) | {9}):
        if no == 9:
            out.append("BAGIAN 9: STRUKTUR AKADEMIK 17 FAKULTAS DAN SEKOLAH PASCASARJANA")
            out.append("")
            for block in fak_txt_blocks:
                out += [block, ""]
            continue
        bag = by_no[no]
        out.append(f"BAGIAN {no}: {bag['judul']}")
        out.append("")
        for f in bag["fakta"]:
            out.append(f"{f['sub'].upper()}")
            out.append(textwrap.fill(clean(f["teks"]), 110))
            out.append("")
    out.append("BAGIAN 19: PANDUAN JAWABAN CEPAT UNTUK ROBOT BRONE")
    out.append("")
    for q in kb["faq"]:
        out.append(f"Q: {q['q']}")
        out.append("A: " + textwrap.fill(clean(q["a"]), 108, subsequent_indent="   "))
        out.append("")
    out.append("CATATAN PENTING UNTUK ROBOT BRONE")
    for i, n in enumerate(kb["catatan_robot"], 1):
        out.append(f"{i}. {n}")
    out += ["", "=" * 80, f"  KNOWLEDGE BASE BRONE v{meta['versi']} | Universitas Brawijaya", "=" * 80, ""]
    txt = "\n".join(out)

    # ukuran unit sebagaimana dipotong RagLite2 (penanda BAGIAN N: / [ Fxx ])
    marks = [m.start() for m in re.finditer(r"(?m)^\s*(BAGIAN \d+:|\[\s*F\d+\s*\])", txt)] + [len(txt)]
    for a, b in zip(marks, marks[1:]):
        unit = txt[a:b].strip()
        title = unit.split("\n", 1)[0]
        if not title.startswith("BAGIAN 19") and not title.startswith("BAGIAN 9:") and len(unit) > TXT_UNIT_WARN:
            warnings.append(f"unit knowledge.txt '{title[:50]}' {len(unit)} karakter (> {TXT_UNIT_WARN})")

    # data bertanggal: yang sudah basi tidak dipakai mesin; yang hampir basi perlu dijadwalkan
    today = _dt.date.today().isoformat()
    soon_limit = (_dt.date.today() + _dt.timedelta(days=14)).isoformat()
    dated: dict[str, str] = {}
    for rec in [*chunks, *faqs]:
        if rec.get("berlaku_hingga"):
            dated[rec["id"]] = str(rec["berlaku_hingga"])
    for rid, until in sorted(dated.items(), key=lambda kv: kv[1]):
        if until < today:
            warnings.append(f"KEDALUWARSA sejak {until}: {rid} tidak dipakai mesin, perbarui datanya")
        elif until <= soon_limit:
            warnings.append(f"akan kedaluwarsa {until}: {rid}")

    for w in warnings:
        print("PERINGATAN:", w)
    for e in errors:
        print("ERROR:", e)
    if errors:
        return 1
    entities = {}
    for fk in kb["fakultas"]:
        names = [fk["nama"], fk["singkatan"], *(re.split(r"\s+atau\s+", fk["nama_lama"]) if fk.get("nama_lama") else [])]
        entities[fk["kode"]] = {
            "nama": fk["nama"], "singkatan": fk["singkatan"],
            "alias": sorted({norm_text(a) for a in [*names, *(alias.get(fk["kode"]) or [])] if norm_text(a)}),
        }

    if not check_only:
        OUT_JSONL.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8")
        OUT_TXT.write_text(txt, encoding="utf-8")
        OUT_FAQ.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in faqs) + "\n", encoding="utf-8")
        OUT_ENT.write_text(json.dumps(entities, ensure_ascii=False, indent=1), encoding="utf-8")
        OUT_REVIEW.write_text(faq_review(faqs, kb["meta"]), encoding="utf-8")
    sizes = sorted(len(c["teks"]) for c in chunks)
    n_auto = sum(f["otomatis"] for f in faqs)
    print(f"OK: {len(chunks)} chunk (median {sizes[len(sizes) // 2]} karakter, maks {sizes[-1]}), "
          f"{len(kb['fakultas'])} fakultas, {len(faqs)} FAQ ({len(faqs) - n_auto} manual + {n_auto} otomatis, "
          f"{sum(len(f['variasi']) for f in faqs)} variasi), {len(sources)} sumber; knowledge.txt {len(txt)} karakter")
    return 0


if __name__ == "__main__":
    sys.exit(main(check_only="--check" in sys.argv))
