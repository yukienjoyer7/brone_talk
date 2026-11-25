"""Harness uji STT BRONE (SPS tab Husein: STT-A/B/S/N/F/P dan STT-E "selesai brone").

Semua klip dijalankan dengan konfigurasi PRODUKSI: nilai Environment= di unit systemd
~/.config/systemd/user/brone-talk.service ditambah rahasia dari EnvironmentFile (brone.env).

Set klip (manifest.csv):
  Q  pertanyaan rekaman XVF3800 (~/stt_eval_27okt/clips/pertanyaan): 15 bersih + 270 derau/obrolan/keduanya
     siap pakai + klip pelengkap scripts/make_stt_aug.py (white/babble/lab/rt/echo/noiseonly)
  H  39 rekaman "halo brone"
  E  388 klip sintesis "selesai brone" (runtime/uji-end-wake, scripts/make_end_wake_synth.py)

Langkah:
  .venv/bin/python scripts/stt_eval.py manifest
  .venv/bin/python scripts/stt_eval.py run --variant mai            # bisa diulang: klip yang sudah ada dilewati
  .venv/bin/python scripts/stt_eval.py run --variant whisper
  .venv/bin/python scripts/stt_eval.py run --variant mai_nophrase --sets Q --conds base
  .venv/bin/python scripts/stt_eval.py faults                       # F-01..F-04, P-03 (mock jaringan/kunci)
  .venv/bin/python scripts/stt_eval.py score                        # rute jawaban, metrik, ringkasan
Varian: mai | mai_nophrase | mai_bigphrase | whisper | whisper_noprompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "uji-stt"
CLIPS27 = Path(os.path.expanduser("~/stt_eval_27okt/clips"))
ENDWAKE = ROOT / "runtime" / "uji-end-wake"
UNIT = Path(os.path.expanduser("~/.config/systemd/user/brone-talk.service"))
PROD_MAI_TIMEOUT_S = 3.5

REF = {
    "q1_rektor": "Siapa rektor UB sekarang?",
    "q2_filkom": "Di mana gedung FILKOM?",
    "q3_biaya": "Berapa biaya kuliah di UB?",
    "q4_kepanjangan": "Apa kepanjangan BRONE?",
    "q5_wisuda": "Kapan wisuda UB berikutnya?",
    "halobrone": "Halo BRONE.",
    "noiseonly": "",
}
DIST = {"075m": "0,75 m", "150m": "1,5 m", "225m": "2,25 m"}


# ----------------------------------------------------------------------------- konfigurasi produksi
def prod_env(**override: str) -> dict[str, str]:
    """os.environ + Environment= unit systemd + EnvironmentFile (rahasia). Nilai tidak pernah dicetak."""
    env = dict(os.environ)
    home = os.path.expanduser("~")
    for line in UNIT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("Environment="):
            body = line[len("Environment="):].strip().strip('"')
            key, _, val = body.partition("=")
            env[key] = val.replace("%h", home)
        elif line.startswith("EnvironmentFile="):
            path = Path(line.split("=", 1)[1].lstrip("-").replace("%h", home))
            if path.is_file():
                for raw in path.read_text(encoding="utf-8").splitlines():
                    raw = raw.strip()
                    if raw and not raw.startswith("#") and "=" in raw:
                        k, _, v = raw.partition("=")
                        env[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    env.update(override)
    return env


class NullTelemetry:
    def __getattr__(self, name):
        return lambda *a, **k: None


def make_transcriber(**override: str):
    from brone_talk.config import Settings
    from brone_talk.stt.transcriber import Transcriber

    s = Settings.from_env(prod_env(**override))
    return Transcriber(s.stt, s.mai, NullTelemetry())


# ----------------------------------------------------------------------------- metrik teks
def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Levenshtein dengan jejak: daftar (op, kata_ref, kata_hyp); op = ok/sub/del/ins."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm(ref).split(), norm(hyp).split()
    return sum(op != "ok" for op, *_ in align(r, h)), len(r)


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = list(norm(ref)), list(norm(hyp))
    return sum(op != "ok" for op, *_ in align(r, h)), len(r)


def biased_errors(ref: str, hyp: str, bias: set[str]) -> dict[str, int]:
    """Le et al. (2021): sub/del ke kata acuan, sisipan ke kata hipotesis -> B (daftar bias) / U (lainnya)."""
    out = {"b_err": 0, "b_n": 0, "u_err": 0, "u_n": 0}
    for op, rw, hw in align(norm(ref).split(), norm(hyp).split()):
        word = rw if rw is not None else hw
        side = "b" if word in bias else "u"
        if rw is not None:
            out[f"{side}_n"] += 1
        if op != "ok":
            out[f"{side}_err"] += 1
    return out


def bias_words() -> set[str]:
    from brone_talk.stt.transcriber import _MAI_PHRASES

    tr = make_transcriber(BRONE_STT_BACKEND="whisper")
    words = {w for p in list(_MAI_PHRASES) + tr.hotwords.split(",") for w in norm(p).split()}
    return words


def big_phrase_list() -> list[str]:
    """_MAI_PHRASES + nama, singkatan, alias entitas kb/entities.json (uji STT-B-06)."""
    from brone_talk.stt.transcriber import _MAI_PHRASES

    ents = json.loads((ROOT / "kb" / "entities.json").read_text(encoding="utf-8"))
    out = list(_MAI_PHRASES)
    for e in ents.values():
        out += [e.get("nama", ""), e.get("singkatan", "")] + list(e.get("alias", []))
    seen, uniq = set(), []
    for p in out:
        p = p.replace(",", " ").strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            uniq.append(p)
    return uniq


# ----------------------------------------------------------------------------- manifest
def cmd_manifest(_args) -> None:
    rows = []
    for p in sorted((CLIPS27 / "pertanyaan").glob("*_bersih.wav")):
        dist, a, b = p.stem.split("_")[:3]
        rows.append({"set": "Q", "path": str(p), "clip": p.name, "dist": dist, "qid": f"{a}_{b}", "cond": "clean",
                     "source": "siap", "ref": REF[f"{a}_{b}"]})
    for p in sorted((CLIPS27 / "pertanyaan" / "bernoise").glob("*.wav")):
        dist, a, b, kind, snr, var = p.stem.split("_")
        rows.append({"set": "Q", "path": str(p), "clip": p.name, "dist": dist, "qid": f"{a}_{b}",
                     "cond": f"{kind}{int(snr[3:])}", "var": var, "source": "siap", "ref": REF[f"{a}_{b}"]})
    aug = OUT / "aug_manifest.csv"
    if aug.is_file():
        for r in csv.DictReader(aug.open(encoding="utf-8")):
            rows.append({"set": "Q", "path": str(OUT / r["file"]), "clip": Path(r["file"]).name, "dist": r["dist"],
                         "qid": r["qid"], "cond": r["cond"], "source": "aug", "ref": REF[r["qid"]],
                         "interferer": r.get("interferer", "")})
    for p in sorted((CLIPS27 / "halo brone").glob("*.wav")):
        rows.append({"set": "H", "path": str(p), "clip": p.name, "qid": "halobrone", "cond": "clean",
                     "source": "siap", "ref": REF["halobrone"]})
    for r in csv.DictReader((ENDWAKE / "manifest.csv").open(encoding="utf-8")):
        rows.append({"set": "E", "path": str(ENDWAKE / r["file"]), "clip": Path(r["file"]).name,
                     "qid": r["phrase_id"], "cond": r["condition"], "source": "synth", "ref": r["text"],
                     "label": r["label"], "category": r["category"], "voice": r["voice"], "speed": r["speed"]})
    fields = ["set", "clip", "path", "dist", "qid", "cond", "var", "source", "ref", "interferer",
              "label", "category", "voice", "speed"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(int)
    for r in rows:
        by[r["set"]] += 1
    print(f"{len(rows)} klip: {dict(by)} -> {OUT / 'manifest.csv'}")


def load_manifest() -> list[dict]:
    return list(csv.DictReader((OUT / "manifest.csv").open(encoding="utf-8")))


# ----------------------------------------------------------------------------- run
RUN_FIELDS = ["set", "clip", "variant", "engine", "raw", "final", "ms", "err", "stt_mean_prob", "stt_min_prob",
              "max_cr", "mean_lp"]


def cmd_run(args) -> None:
    from brone_talk.stt.corrections import _mai_fix, _stt_fix

    variant = args.variant
    if variant.startswith("mai"):
        over = {"BRONE_STT_BACKEND": "mai", "BRONE_MAI_TIMEOUT_SECONDS": "15"}
        if variant == "mai_nophrase":
            over["BRONE_MAI_PHRASES"] = "-"
        elif variant == "mai_bigphrase":
            over["BRONE_MAI_PHRASES"] = ",".join(big_phrase_list())
        tr = make_transcriber(**over)
        print(f"MAI phrase list: {len(tr.mai_phrases)} frasa; kunci {'ada' if tr.mai_api_key else 'TIDAK ADA'}")
    else:
        tr = make_transcriber(BRONE_STT_BACKEND="whisper")
        if variant == "whisper_noprompt":
            tr.prompt, tr.hotwords = "", ""
        tr.load()

    rows = [r for r in load_manifest() if r["set"] in args.sets.split(",")]
    if args.conds == "base":  # ablasi: hanya data siap pakai (bersih + derau/obrolan/keduanya)
        rows = [r for r in rows if r["source"] == "siap" and r["set"] == "Q"]
    out_path = OUT / f"raw_{variant}.csv"
    done = set()
    if out_path.is_file():
        done = {(r["set"], r["clip"]) for r in csv.DictReader(out_path.open(encoding="utf-8")) if not r["err"]}
    todo = [r for r in rows if (r["set"], r["clip"]) not in done]
    print(f"{variant}: {len(todo)} klip (sudah {len(done)})", flush=True)
    new_file = not out_path.is_file()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new_file:
            w.writeheader()
        for i, r in enumerate(todo, 1):
            path = Path(r["path"])
            t = time.perf_counter()
            err, raw, final, stats = "", "", "", {}
            try:
                if variant.startswith("mai"):
                    raw = tr._transcribe_voicelive(path)
                    tr._set_mai_stats([])
                    final = _stt_fix(_mai_fix(raw))
                else:
                    raw = tr._transcribe_whisper(path)
                    final = _stt_fix(raw)
                stats = dict(tr.last_stats)
            except Exception as exc:  # dicatat, klip diulang pada run berikutnya
                err = f"{type(exc).__name__}: {str(exc)[:150]}"
            ms = round((time.perf_counter() - t) * 1000)
            w.writerow({"set": r["set"], "clip": r["clip"], "variant": variant,
                        "engine": "mai" if variant.startswith("mai") else "whisper", "raw": raw, "final": final,
                        "ms": ms, "err": err, **{k: stats.get(k, "") for k in RUN_FIELDS[8:]}})
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {variant} {i}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- fault (F-01..F-04, P-03)
def cmd_faults(args) -> None:
    import brone_talk.stt.transcriber as tmod

    clip = CLIPS27 / "pertanyaan" / "075m_q1_rektor_bersih.wav"  # = seg_39 (0,75 m, q1)
    res: dict = {}
    wh = make_transcriber(BRONE_STT_BACKEND="whisper")
    wh.load()
    wh.transcribe(clip)  # pemanasan
    t = time.perf_counter()
    res["whisper_only"] = {"text": wh.transcribe(clip), "ms": round((time.perf_counter() - t) * 1000)}

    def trial(tr, reset_cooldown: bool) -> dict:
        if reset_cooldown:
            tr.mai_skip_until = 0.0
        t = time.perf_counter()
        text = tr.transcribe(clip)
        return {"ms": round((time.perf_counter() - t) * 1000), "engine": tr.last_engine, "text": text,
                "same_as_whisper": text == res["whisper_only"]["text"]}

    # F-01 / P-03: endpoint tak terjangkau (IP blackhole: koneksi menggantung) dan DNS gagal
    for name, endpoint in (("blackhole", "https://10.255.255.1"), ("dns_gagal", "https://mai-uji.invalid")):
        tr = make_transcriber(BRONE_STT_BACKEND="mai", BRONE_MAI_ENDPOINT=endpoint)
        tr.model = wh.model  # model whisper yang sama, tanpa memuat ulang
        forced = [trial(tr, reset_cooldown=True) for _ in range(10)]   # tiap percobaan menunggu tenggat
        natural = [trial(tr, reset_cooldown=False) for _ in range(3)]  # cooldown 30 s: langsung whisper
        res[f"F01_{name}"] = {"forced": forced, "natural_after": natural,
                              "max_ms": max(x["ms"] for x in forced),
                              "p50_ms": statistics.median(x["ms"] for x in forced)}
        print(name, res[f"F01_{name}"]["max_ms"], flush=True)

    # F-04: kunci salah. Hitung percobaan MAI (ucapan + probe penjaga) selama 40 s.
    tr = make_transcriber(BRONE_STT_BACKEND="mai", BRONE_MAI_API_KEY="kunci-uji-salah-0000")
    tr.model = wh.model
    calls = {"utter": 0, "open": 0}
    orig_open = tr._voicelive_open

    def counted_open(timeout):
        calls["open"] += 1
        return orig_open(timeout)

    tr._voicelive_open = counted_open
    first = trial(tr, reset_cooldown=False)
    exc_info = {"skip_auth": tr._mai_skip_auth, "cooldown_s": round(tr.mai_skip_until - time.monotonic())}
    tr.load()  # penjaga latar mulai (probe tiap 5 s)
    later = []
    t_end = time.monotonic() + 40
    while time.monotonic() < t_end:
        later.append(trial(tr, reset_cooldown=False))
        time.sleep(8)
    res["F04_kunci_salah"] = {"first": first, **exc_info, "later": later,
                              "voicelive_open_attempts_40s": calls["open"]}
    # penyebab: jenis exception dari handshake ditolak
    try:
        tr2 = make_transcriber(BRONE_STT_BACKEND="mai", BRONE_MAI_API_KEY="kunci-uji-salah-0000")
        tr2._voicelive_open(6.0)
    except Exception as exc:
        res["F04_kunci_salah"]["exception"] = {
            "type": type(exc).__name__, "msg": str(exc)[:160], "code_attr": getattr(exc, "code", None),
            "status": getattr(getattr(exc, "response", None), "status_code", None)}
    print("F04", json.dumps(res["F04_kunci_salah"], ensure_ascii=False)[:300], flush=True)

    # F-02 / F-03 (simulasi): rute default hilang (WiFi mati) lalu kembali, penjaga produksi jalan
    tr = make_transcriber(BRONE_STT_BACKEND="mai")
    tr.model = wh.model
    tr.load()
    t0 = time.monotonic()
    while tr._vl_ws is None and time.monotonic() - t0 < 20:
        time.sleep(0.1)
    online_before = tr.cloud_online
    orig_route = tmod.has_default_route
    tmod.has_default_route = lambda *a, **k: False
    t0 = time.monotonic()
    while tr.cloud_online and time.monotonic() - t0 < 30:
        time.sleep(0.05)
    offline_after_s = round(time.monotonic() - t0, 2)
    utter_offline = trial(tr, reset_cooldown=False)
    tmod.has_default_route = orig_route
    t0 = time.monotonic()
    while not tr.cloud_online and time.monotonic() - t0 < 30:
        time.sleep(0.05)
    online_after_s = round(time.monotonic() - t0, 2)
    utter_online = trial(tr, reset_cooldown=False)
    res["F02_F03_simulasi"] = {"online_before": online_before, "detik_sampai_offline": offline_after_s,
                               "ucapan_saat_offline": utter_offline, "detik_sampai_online": online_after_s,
                               "ucapan_setelah_pulih": utter_online, "keepalive_s": tr.mai_keepalive_s}
    print("F02/F03", offline_after_s, online_after_s, flush=True)
    (OUT / "faults.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", OUT / "faults.json")


def cmd_latency(args) -> None:
    """Jalur produksi utuh (penjaga koneksi aktif, tenggat 3,5 s, fallback whisper) pada 15 klip bersih."""
    tr = make_transcriber(BRONE_STT_BACKEND="mai")
    tr.load()
    t0 = time.monotonic()
    while tr._vl_ws is None and time.monotonic() - t0 < 20:
        time.sleep(0.1)
    clips = sorted((CLIPS27 / "pertanyaan").glob("*_bersih.wav"))
    rows = []
    for rep in range(args.reps):
        for p in clips:
            t = time.perf_counter()
            text = tr.transcribe(p)
            rows.append({"rep": rep, "clip": p.name, "ms": round((time.perf_counter() - t) * 1000),
                         "engine": tr.last_engine, "text": text})
            time.sleep(1.0)  # jeda antar ucapan seperti percakapan
    ms = sorted(r["ms"] for r in rows)
    out = {"n": len(rows), "p50": ms[len(ms) // 2], "p90": ms[int(0.9 * (len(ms) - 1))], "max": ms[-1],
           "fallback_whisper": sum(r["engine"] != "mai" for r in rows),
           "per_jarak": {d: sorted(r["ms"] for r in rows if r["clip"].startswith(d))[len(rows) // 6]
                         for d in DIST}, "rows": rows}
    (OUT / "latency_p01.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"})


# ----------------------------------------------------------------------------- rute (pipeline produksi)
class Router:
    """Ulangi TalkService._hear + _turn_kws (sesi sudah aktif) tanpa audio, lalu mesin jawaban mode faq."""

    def __init__(self) -> None:
        from brone_talk.app import engine_config
        from brone_talk.config import Settings
        from brone_talk.engine.answer import AnswerEngine

        settings = Settings.from_env(prod_env(BRONE_MODE="faq"))
        rt = OUT / "engine-runtime"
        rt.mkdir(parents=True, exist_ok=True)
        cache = ROOT / "runtime" / "embcache.json"
        if cache.is_file() and not (rt / "embcache.json").is_file():
            shutil.copy(cache, rt / "embcache.json")
        self.engine = AnswerEngine(engine_config(settings, rt))
        self.engine.start()
        self.tr = make_transcriber(BRONE_STT_BACKEND="whisper")  # vocab & istilah prompt (filter produksi)
        self._cache: dict = {}

    def route(self, text: str, stats: dict | None, end_only: bool = False) -> str:
        from brone_talk.app import TalkService
        from brone_talk.audio.kws import self_wake
        from brone_talk.nlu.wake import NAME_RE, SESSION_END_RE, clock_answer, collapse_repeats
        from brone_talk.session import Decision, Outcome, decide
        from brone_talk.stt.filters import stt_garbage_reason, stt_strip_prompt_sentence

        if not text:
            return "kosong"
        if NAME_RE.search(text):
            text = collapse_repeats(text)
        text, called = TalkService._addressing(text)
        reason = stt_garbage_reason(text, stats, self.tr.vocab, self.tr.prompt_terms)
        if reason:
            return f"dibuang_filter:{reason}"
        cleaned = stt_strip_prompt_sentence(text)
        if cleaned != text:
            if not cleaned:
                return "dibuang_filter:gema kalimat prompt"
            text, called = TalkService._addressing(cleaned)
        if SESSION_END_RE.search(text):
            return "selesai_sesi"
        if end_only:
            return "lanjut"
        outcome = decide(text, called=called, awaiting=True, stop_patterns=[])
        if outcome.decision is Decision.CONFIRM:
            if self_wake(text):
                return "wake_ulang"
            outcome = Outcome(Decision.ANSWER, question=text)
        q = outcome.question
        if clock_answer(q):
            return "jam"
        key = (q, json.dumps(stats, sort_keys=True) if stats else "")
        if key not in self._cache:
            self.engine.reset()
            res = self.engine.answer(q, stats)
            r, meta = res["route"], res["meta"]
            if r == "faq":
                r = f"faq:{meta['faq']['id']}"
            elif r == "obrolan":
                r = f"obrolan:{meta['obrolan']['id']}"
            elif r == "tolak_mode_faq":
                r = "rag_llm"
            self._cache[key] = (r, meta.get("faq", {}).get("score"))
        return self._cache[key][0]

    def embed(self, texts: list[str]):
        return self.engine.embedder.embed(texts)


def stats_of(r: dict) -> dict | None:
    if r["engine"] == "mai":
        return {"stt_mean_prob": float(r["stt_mean_prob"] or 0.9), "stt_min_prob": float(r["stt_min_prob"] or 0.9)}
    out = {}
    for k in ("stt_mean_prob", "stt_min_prob", "max_cr", "mean_lp"):
        if r.get(k) not in ("", None):
            out[k] = float(r[k])
    return out or None


# ----------------------------------------------------------------------------- score
def pct(a: float, b: float) -> float | None:
    return round(100 * a / b, 1) if b else None


def cmd_score(_args) -> None:  # noqa: C901 - satu tempat untuk semua metrik
    import numpy as np
    from scipy.stats import spearmanr

    from brone_talk.stt.filters import stt_garbage_reason

    man = {(r["set"], r["clip"]): r for r in load_manifest()}
    runs: dict[str, dict] = {}
    for p in sorted(OUT.glob("raw_*.csv")):
        v = p.stem[4:]
        runs[v] = {}
        for r in csv.DictReader(p.open(encoding="utf-8")):
            # galat (DNS/jaringan saat uji) tidak dihitung sebagai transkrip kosong; MAI efektif produksi
            # di bawah tetap memperlakukannya sebagai fallback whisper
            if not r["err"] or (v == "mai" and (r["set"], r["clip"]) not in runs[v]):
                runs[v][(r["set"], r["clip"])] = r
    # MAI efektif produksi: lewat 3,5 s atau galat -> teks whisper klip yang sama
    if "mai" in runs and "whisper" in runs:
        prod = {}
        for k, r in runs["mai"].items():
            w = runs["whisper"].get(k)
            if (r["err"] or int(r["ms"]) > PROD_MAI_TIMEOUT_S * 1000) and w:
                prod[k] = {**w, "variant": "mai_prod", "ms": str(int(PROD_MAI_TIMEOUT_S * 1000) + int(w["ms"]))}
            else:
                prod[k] = {**r, "variant": "mai_prod"}
        runs["mai_prod"] = prod

    router = Router()
    bias = bias_words()
    refroute = {qid: router.route(ref, None) for qid, ref in REF.items() if ref}
    scored = []
    for v, rows in runs.items():
        for k, r in rows.items():
            m = man.get(k)
            if m is None:
                continue
            ref = m["ref"]
            st = stats_of(r)
            o = {**m, "variant": v, "engine": r["engine"], "raw": r["raw"], "final": r["final"], "ms": int(r["ms"]),
                 "err": r["err"], "stt_mean_prob": r["stt_mean_prob"]}
            if ref:
                we, wn = word_errors(ref, r["final"])
                ce, cn = char_errors(ref, r["final"])
                rwe, _ = word_errors(ref, r["raw"])
                o.update(w_err=we, w_n=wn, c_err=ce, c_n=cn, w_err_raw=rwe, **biased_errors(ref, r["final"], bias))
            if m["set"] == "E":
                o["route"] = router.route(r["final"], st, end_only=True)
                o["ended"] = int(o["route"] == "selesai_sesi")
            else:
                o["route"] = router.route(r["final"], st)
                o["ref_route"] = refroute.get(m["qid"], "")
                o["filtered"] = int(o["route"].startswith("dibuang_filter") or o["route"] == "kosong")
                o["garbage"] = stt_garbage_reason(r["final"], st, router.tr.vocab, router.tr.prompt_terms) \
                    if r["final"] else "tanpa kata"
            scored.append(o)
    from brone_talk.nlu.wake import SESSION_END_RE

    for o in scored:
        if o["set"] == "E":
            o["ended_regex_only"] = int(bool(SESSION_END_RE.search(o["final"])))

    # SemDist (bge-m3, embedding yang sama dengan FAQ)
    q_rows = [o for o in scored if o["set"] in ("Q", "H") and o["ref"] and o["variant"] in ("mai", "whisper")]
    texts = sorted({o["ref"] for o in q_rows} | {o["final"] for o in q_rows if o["final"]})
    vecs = {}
    for i in range(0, len(texts), 64):
        for t, vec in zip(texts[i:i + 64], router.embed(texts[i:i + 64]), strict=True):
            a = np.asarray(vec, dtype=float)
            vecs[t] = a / (np.linalg.norm(a) + 1e-12)
    for o in q_rows:
        o["semdist"] = round(1 - float(vecs[o["ref"]] @ vecs[o["final"]]), 4) if o["final"] else 1.0

    fields = sorted({k for o in scored for k in o}, key=lambda k: (k not in ("set", "clip", "variant"), k))
    with (OUT / "scored.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(scored)

    S: dict = {"ref_routes": refroute, "bias_words": sorted(bias), "n_bias_words": len(bias)}

    def sel(variant: str, set_: str = "Q", **eq) -> list[dict]:
        return [o for o in scored if o["variant"] == variant and o["set"] == set_
                and all(o.get(k) == v if not callable(v) else v(o.get(k)) for k, v in eq.items())]

    def wer_of(rows: list[dict], raw: bool = False) -> float | None:
        return pct(sum(o["w_err_raw" if raw else "w_err"] for o in rows), sum(o["w_n"] for o in rows))

    def cer_of(rows):
        return pct(sum(o["c_err"] for o in rows), sum(o["c_n"] for o in rows))

    def route_ok(rows):
        return sum(o["route"] == o["ref_route"] for o in rows), len(rows)

    def lat(rows):
        xs = sorted(o["ms"] for o in rows if not o["err"])
        if not xs:
            return None
        return {"p50": xs[len(xs) // 2], "p90": xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))],
                "max": xs[-1], "n": len(xs)}

    variants = [v for v in ("mai", "whisper", "mai_prod", "mai_nophrase", "mai_bigphrase", "whisper_noprompt")
                if v in runs]
    # A-01..A-04: per jarak, klip bersih
    S["A"] = {}
    for v in variants:
        S["A"][v] = {}
        for d in ("075m", "150m", "225m", None):
            rows = sel(v, dist=d, cond="clean") if d else sel(v, cond="clean")
            if rows:
                ok, n = route_ok(rows)
                S["A"][v][d or "semua"] = {"wer": wer_of(rows), "cer": cer_of(rows), "n": len(rows),
                                           "route_ok": f"{ok}/{n}"}
    # A-05: koreksi regex (semua klip Q+H yang punya acuan)
    S["A05"] = {}
    for v in ("mai", "whisper"):
        rows = [o for o in scored if o["variant"] == v and o["set"] in ("Q", "H") and o.get("w_n")]
        worse = [o for o in rows if o["w_err"] > o["w_err_raw"]]
        better = [o for o in rows if o["w_err"] < o["w_err_raw"]]
        S["A05"][v] = {"wer_raw": wer_of(rows, raw=True), "wer_final": wer_of(rows), "n": len(rows),
                       "lebih_baik": len(better), "memburuk": len(worse),
                       "contoh_memburuk": [(o["raw"], o["final"]) for o in worse[:5]]}
    # A-06 / B-02 / B-06: ablasi pada data siap pakai (bersih + derau/obrolan/keduanya)
    def ablation(v):
        rows = [o for o in scored if o["variant"] == v and o["set"] == "Q" and o["source"] == "siap"]
        clean = [o for o in rows if o["cond"] == "clean"]
        far = [o for o in clean if o["dist"] == "225m"]
        b = {k: sum(o[k] for o in rows) for k in ("b_err", "b_n", "u_err", "u_n")}
        bc = {k: sum(o[k] for o in clean) for k in ("b_err", "b_n", "u_err", "u_n")}
        echo = sum(bool(o["garbage"].startswith("gema")) for o in rows)
        filkom_far = [o["final"] for o in far if o["qid"] == "q2_filkom"]
        return {"n": len(rows), "wer": wer_of(rows), "wer_clean": wer_of(clean),
                "b_wer": pct(b["b_err"], b["b_n"]), "u_wer": pct(b["u_err"], b["u_n"]),
                "b_wer_clean": pct(bc["b_err"], bc["b_n"]), "u_wer_clean": pct(bc["u_err"], bc["u_n"]),
                "route_ok": "%d/%d" % route_ok(rows), "gema_prompt": echo, "filkom_225m": filkom_far}
    S["ablasi"] = {v: ablation(v) for v in variants if v != "mai_prod"}
    # B-01 per jarak (klip bersih) dan per kondisi
    S["B01"] = {}
    for v in ("mai", "whisper"):
        S["B01"][v] = {}
        for d in ("075m", "150m", "225m"):
            rows = sel(v, dist=d, cond="clean")
            b = {k: sum(o[k] for o in rows) for k in ("b_err", "b_n", "u_err", "u_n")}
            S["B01"][v][d] = {"b_wer": pct(b["b_err"], b["b_n"]), "u_wer": pct(b["u_err"], b["u_n"])}
    # B-03..B-05: token domain
    def has(tok, text):
        return tok in norm(text).split()
    S["B03"], S["B04"], S["B05"] = {}, {}, {}
    for v in variants:
        q4 = sel(v, qid="q4_kepanjangan", cond="clean")
        halo = sel(v, "H")
        S["B03"][v] = {"q4_clean": f"{sum(has('brone', o['final']) for o in q4)}/{len(q4)}",
                       "halo": f"{sum(has('brone', o['final']) for o in halo)}/{len(halo)}",
                       "q4_semua_kondisi": "%d/%d" % (sum(has('brone', o['final']) for o in sel(v, qid='q4_kepanjangan')),
                                                       len(sel(v, qid='q4_kepanjangan')))}
        ub = [o for o in sel(v, cond="clean") if o["qid"] in ("q1_rektor", "q3_biaya", "q5_wisuda")]
        fk = sel(v, qid="q2_filkom", cond="clean")
        S["B04"][v] = {"UB_clean": f"{sum(has('ub', o['final']) for o in ub)}/{len(ub)}",
                       "FILKOM_clean": f"{sum(has('filkom', o['final']) for o in fk)}/{len(fk)}",
                       "UB_semua": "%d/%d" % (sum(has('ub', o['final']) for o in sel(v) if o['qid'] in
                                                  ('q1_rektor', 'q3_biaya', 'q5_wisuda')),
                                              len([o for o in sel(v) if o['qid'] in ('q1_rektor', 'q3_biaya', 'q5_wisuda')])),
                       "FILKOM_semua": "%d/%d" % (sum(has('filkom', o['final']) for o in sel(v, qid='q2_filkom')),
                                                  len(sel(v, qid='q2_filkom')))}
        wq = sel(v, qid="q5_wisuda", cond="clean")
        wall = sel(v, qid="q5_wisuda")
        S["B05"][v] = {"wisuda_clean_benar": f"{sum(has('wisuda', o['final']) for o in wq)}/{len(wq)}",
                       "wisuda_semua_benar": f"{sum(has('wisuda', o['final']) for o in wall)}/{len(wall)}",
                       "salah_dengar": sorted({o['final'] for o in wq if not has('wisuda', o['final'])})[:6]}
    # S-01..S-05
    S["S"] = {}
    for v in variants:
        rows = [o for o in sel(v) if o["qid"] != "noiseonly"]
        clean = [o for o in rows if o["cond"] == "clean"]
        wrong_faq = [o for o in rows if o["route"].startswith("faq:") and o["route"] != o["ref_route"]]
        wer_pos_ok = [o for o in rows if o["w_err"] > 0 and o["route"] == o["ref_route"]]
        low = [o for o in rows if o["engine"] == "whisper" and o["stt_mean_prob"] not in ("", None)
               and float(o["stt_mean_prob"]) < 0.5]
        low_faq = [o for o in low if o["route"].startswith("faq:")]
        S["S"][v] = {"route_clean": "%d/%d" % route_ok(clean), "route_all": "%d/%d" % route_ok(rows),
                     "faq_salah": len(wrong_faq),
                     "faq_salah_contoh": [(o["clip"], o["final"], o["route"], o["ref_route"]) for o in wrong_faq[:8]],
                     "wer_pos_rute_benar": len(wer_pos_ok), "wer_pos": sum(o["w_err"] > 0 for o in rows),
                     "lowconf_n": len(low), "lowconf_faq": len(low_faq),
                     "lowconf_faq_salah": sum(o["route"] != o["ref_route"] for o in low_faq),
                     "contoh_wer_pos_ok": [(o["final"]) for o in wer_pos_ok[:6]]}
    rows = [o for o in q_rows if o["set"] == "Q" and o["qid"] != "noiseonly"]
    fail = [int(o["route"] != o["ref_route"]) for o in rows]
    rs = spearmanr([o["semdist"] for o in rows], fail).statistic
    rw = spearmanr([o["w_err"] / max(1, o["w_n"]) for o in rows], fail).statistic
    ok_sd = sorted(o["semdist"] for o, f in zip(rows, fail, strict=True) if not f)
    bad_sd = sorted(o["semdist"] for o, f in zip(rows, fail, strict=True) if f)
    best = max(((t, sum(s <= t for s in ok_sd) + sum(s > t for s in bad_sd)) for t in
                sorted(set(ok_sd + bad_sd))), key=lambda p: p[1], default=(None, 0))
    S["S02"] = {"n": len(rows), "rho_semdist": round(float(rs), 3), "rho_wer": round(float(rw), 3),
                "ambang_semdist": best[0], "akurasi_ambang": pct(best[1], len(rows)),
                "semdist_median_rute_benar": statistics.median(ok_sd) if ok_sd else None,
                "semdist_median_rute_salah": statistics.median(bad_sd) if bad_sd else None}
    # N-01..N-06: per kondisi
    S["N"] = {}
    for v in variants:
        S["N"][v] = {}
        conds = sorted({o["cond"] for o in sel(v) if o["qid"] != "noiseonly"})
        for c in conds:
            rows = sel(v, cond=c)
            ok, n = route_ok(rows)
            S["N"][v][c] = {"wer": wer_of(rows), "route_ok": f"{ok}/{n}", "route_pct": pct(ok, n), "n": n}
    # N-04: halusinasi saat derau (WER >= 100% dan teks tidak kosong) yang lolos filter
    S["N04"] = {}
    for v in variants:
        noisy = [o for o in sel(v) if o["cond"] in ("babble5", "babble0", "obrolan3", "keduanya3")]
        hall = [o for o in noisy if o["final"] and o["w_err"] >= o["w_n"]]
        passed = [o for o in hall if not o["filtered"]]
        to_answer = [o for o in passed if o["route"].startswith(("faq:", "obrolan:", "rag_llm"))]
        S["N04"][v] = {"n": len(noisy), "halusinasi": len(hall), "lolos_filter": len(passed),
                       "diteruskan_faq_llm": len(to_answer),
                       "contoh": [(o["cond"], o["final"], o["route"]) for o in to_answer[:8]]}
    # N-05: kalimat Bron ikut tertranskripsi
    S["N05"] = {}
    for v in variants:
        rows = [o for o in sel(v) if o["cond"].startswith("echo")]
        leak = []
        for o in rows:
            ref_w = set(norm(o["ref"]).split())
            bron_w = set(norm(o["interferer"]).split()) - ref_w
            extra = [w for w in norm(o["final"]).split() if w in bron_w and w not in ref_w]
            if len(extra) >= 2:
                leak.append(o)
        S["N05"][v] = {c: {"wer": wer_of([o for o in rows if o["cond"] == c]),
                           "bocor": sum(o["cond"] == c for o in leak), "n": sum(o["cond"] == c for o in rows)}
                       for c in ("echo20", "echo10", "echo5")}
        S["N05"][v]["contoh_bocor"] = [(o["cond"], o["final"], o["route"]) for o in leak[:6]]
    # F-05: tanpa ucapan
    S["F05"] = {}
    for v in variants:
        rows = sel(v, qid="noiseonly")
        answered = [o for o in rows if not (o["filtered"] or o["route"] == "abaikan")]
        S["F05"][v] = {"n": len(rows), "dijawab": len(answered),
                       "detail": [(o["cond"], o["final"], o["route"]) for o in rows]}
    # P-01 / P-02
    S["P"] = {v: {"clean15": lat(sel(v, cond="clean")), "semua_Q": lat(sel(v)), "E": lat(sel(v, "E"))}
              for v in ("mai", "whisper") if v in runs}
    S["P"]["mai_lewat_3_5s"] = {"Q": sum(o["ms"] > 3500 or bool(o["err"]) for o in sel("mai")),
                                "nQ": len(sel("mai")),
                                "E": sum(o["ms"] > 3500 or bool(o["err"]) for o in sel("mai", "E")),
                                "nE": len(sel("mai", "E"))}
    # STT-E
    S["E"] = {}
    for v in [x for x in ("mai", "whisper", "mai_prod") if x in runs]:
        rows = sel(v, "E")
        end = [o for o in rows if o["label"] == "END"]
        neg = [o for o in rows if o["label"] == "NO_END"]

        def rec(rr):
            return f"{sum(o['ended'] for o in rr)}/{len(rr)} = {pct(sum(o['ended'] for o in rr), len(rr))}%"
        e = {"recall_END": rec(end), "false_end": rec(neg),
             "recall_END_regex_saja": f"{sum(o['ended_regex_only'] for o in end)}/{len(end)}",
             "per_kondisi": {c: rec([o for o in end if o["cond"] == c]) for c in sorted({o["cond"] for o in end})},
             "per_kecepatan_d21": {s: rec([o for o in end if o["speed"] == s and o["voice"] == "d21"])
                                   for s in ("lambat", "normal", "cepat")},
             "per_suara": {s: rec([o for o in end if o["voice"] == s]) for s in ("d21", "news")},
             "per_suara_normal": {s: rec([o for o in end if o["voice"] == s and o["speed"] == "normal"])
                                  for s in ("d21", "news")},
             "per_kategori_NO_END": {c: rec([o for o in neg if o["category"] == c])
                                     for c in sorted({o["category"] for o in neg})},
             "per_kategori_END_clean": {c: rec([o for o in end if o["category"] == c and o["cond"] == "clean"])
                                        for c in sorted({o["category"] for o in end})},
             "false_end_contoh": [(o["clip"], o["final"]) for o in neg if o["ended"]][:10],
             "miss_clean_contoh": [(o["clip"], o["final"]) for o in end if not o["ended"] and o["cond"] == "clean"][:12],
             "latensi_END": lat(end), "wer": wer_of(rows)}
        S["E"][v] = e
    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(S, ensure_ascii=False, indent=1)[:6000])
    print("->", OUT / "summary.json", OUT / "scored.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    r = sub.add_parser("run")
    r.add_argument("--variant", required=True,
                   choices=["mai", "mai_nophrase", "mai_bigphrase", "whisper", "whisper_noprompt"])
    r.add_argument("--sets", default="Q,H,E")
    r.add_argument("--conds", default="all", choices=["all", "base"])
    r.set_defaults(fn=cmd_run)
    sub.add_parser("faults").set_defaults(fn=cmd_faults)
    lt = sub.add_parser("latency", help="P-01: 15 klip bersih x N ulangan lewat jalur produksi (tenggat 3,5 s)")
    lt.add_argument("--reps", type=int, default=5)
    lt.set_defaults(fn=cmd_latency)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
