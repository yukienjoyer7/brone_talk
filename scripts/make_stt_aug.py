"""Klip pelengkap untuk uji STT (SPS tab Husein, STT-N-01..N-06, STT-F-05).

Sumber: 15 rekaman pertanyaan bersih di ~/stt_eval_27okt/clips/pertanyaan (XVF3800, 3 jarak x 5 pertanyaan).
Data siap pakai (270 klip derau/obrolan/keduanya SNR 10 dan 3 dB) tidak dibuat ulang; skrip ini hanya
menambah kondisi yang diminta rencana uji tetapi belum ada:

  white{20,10,5,0}   derau putih Gauss (N-01)
  babble{20,10,5,0}  6 ucapan lain ditumpuk acak (N-02; pengganti pub noise)
  rt{30,60,90}       gema ruangan RT60 0,3/0,6/0,9 s, RIR eksponensial sintetis (N-03)
  echo{20,10,5}      suara Bron (Piper D21, gema RT60 0,3 s) sebagai pengganggu, SIR 20/10/5 dB (N-05)
  lab{20,10,5,0}     derau lab asli direkam lewat XVF3800 (N-06), bila runtime/uji-stt/src/lab_noise_raw.wav ada
  noiseonly_*        klip 3 s tanpa ucapan (F-05)

  .venv/bin/python scripts/make_stt_aug.py [--out runtime/uji-stt]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_end_wake_synth import load16k, mix_at_snr, piper, reverb, rms  # noqa: E402

from brone_talk import messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000
SEED = 20261110
CLIPS = Path(os.path.expanduser("~/stt_eval_27okt/clips"))
SNRS = (20, 10, 5, 0)


def save(path: Path, y: np.ndarray) -> float:
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 0.99:  # cegah clipping; level relatif sudah dikunci SNR
        y = y / peak * 0.99
    wavfile.write(path, SR, (y * 32767).astype(np.int16))
    return round(len(y) / SR, 2)


def babble(n: int, pool: list[np.ndarray], rng: np.random.Generator, k: int = 6) -> np.ndarray:
    mix = np.zeros(n, np.float32)
    for i in rng.choice(len(pool), size=k, replace=False):
        c = pool[i] / (rms(pool[i]) + 1e-9)  # tiap penutur sekeras sama
        mix += np.resize(np.roll(c, int(rng.integers(0, len(c)))), n)
    return mix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "runtime" / "uji-stt")
    args = ap.parse_args()
    wav_dir = args.out / "aug"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    clean = {p.stem.removesuffix("_bersih"): load16k(p) for p in sorted((CLIPS / "pertanyaan").glob("*_bersih.wav"))}
    halo = [load16k(p) for p in sorted((CLIPS / "halo brone").glob("*.wav"))]
    synth = [load16k(p) for p in sorted((ROOT / "runtime" / "uji-end-wake" / "wav").glob("*_normal_clean.wav"))]

    # suara Bron sebagai pengganggu: sapaan sesi + satu jawaban FAQ yang panjang
    faq = [json.loads(line) for line in (ROOT / "kb" / "faq.jsonl").open(encoding="utf-8")]
    bron_lines = [messages.SAY_WAKE_SESSION, next(f["a"] for f in faq if f["id"] == "siapa-kamu")]
    with tempfile.TemporaryDirectory() as tmp:
        bron = []
        for i, line in enumerate(bron_lines):
            out = Path(tmp) / f"bron{i}.wav"
            piper(line, "id_ID-brone_D21-medium", 1.0, out)
            bron.append(reverb(load16k(out), 0.3, rng))
    bron_audio = np.concatenate(bron)

    lab_path = args.out / "src" / "lab_noise_raw.wav"
    lab = load16k(lab_path) if lab_path.is_file() else None
    if lab is not None:
        lab = lab[int(0.5 * SR):]  # buang klik awal perekaman
        lab -= lab.mean()

    rows = []
    for stem, x in clean.items():
        dist, q1, q2 = stem.split("_")[:3]
        qid = f"{q1}_{q2}"
        others = [v for k, v in clean.items() if f"{k.split('_')[1]}_{k.split('_')[2]}" != qid]
        pool = others + halo + synth

        def add(cond: str, y: np.ndarray, **extra) -> None:
            name = f"{stem}_{cond}.wav"
            dur = save(wav_dir / name, y)
            rows.append({"file": f"aug/{name}", "dist": dist, "qid": qid, "cond": cond, "duration_s": dur, **extra})

        for snr in SNRS:
            add(f"white{snr}", mix_at_snr(x, rng.standard_normal(len(x)).astype(np.float32), snr))
            add(f"babble{snr}", mix_at_snr(x, babble(len(x), pool, rng), snr))
            if lab is not None:
                off = int(rng.integers(0, max(1, len(lab) - len(x))))
                add(f"lab{snr}", mix_at_snr(x, np.resize(lab[off:], len(x)), snr))
        for rt in (0.3, 0.6, 0.9):
            add(f"rt{int(rt * 100)}", reverb(x, rt, rng))
        for sir in (20, 10, 5):
            off = int(rng.integers(0, max(1, len(bron_audio) - len(x))))
            add(f"echo{sir}", mix_at_snr(x, np.resize(bron_audio[off:], len(x)), sir),
                interferer=" ".join(bron_lines))

    # F-05: klip tanpa ucapan, 3 s. Level derau setara derau pada SNR 10 dB terhadap rekaman 0,75 m.
    ref_level = float(np.mean([rms(v[np.abs(v) > 0.01 * np.max(np.abs(v))]) for k, v in clean.items()
                               if k.startswith("075m")]))
    n = 3 * SR
    noise_only = {"white": lambda: rng.standard_normal(n).astype(np.float32),
                  "babble": lambda: babble(n, halo + synth + list(clean.values()), rng)}
    if lab is not None:
        noise_only["lab"] = lambda: np.resize(lab[int(rng.integers(0, len(lab) - n)):], n)
    resid_src = CLIPS / "pertanyaan" / "bernoise"
    for kind in ("derau", "obrolan", "keduanya"):  # derau dari data siap pakai: bernoise - bersih
        def resid(kind: str = kind) -> np.ndarray:
            stem = list(clean)[int(rng.integers(0, len(clean)))]
            v = int(rng.integers(1, 4))
            noisy = load16k(resid_src / f"{stem}_{kind}_snr10_v{v}.wav")
            r = noisy[: len(clean[stem])] - clean[stem][: len(noisy)]
            return np.resize(r, n)
        noise_only[f"siap_{kind}"] = resid
    for kind, make in noise_only.items():
        for v in (1, 2):
            y = make()
            y = y / (rms(y) + 1e-9) * ref_level / (10 ** (10 / 20))
            name = f"noiseonly_{kind}_v{v}.wav"
            rows.append({"file": f"aug/{name}", "dist": "", "qid": "noiseonly", "cond": f"noiseonly_{kind}",
                         "duration_s": save(wav_dir / name, y)})
    name = "noiseonly_senyap_v1.wav"  # hampir senyap (-60 dBFS)
    rows.append({"file": f"aug/{name}", "dist": "", "qid": "noiseonly", "cond": "noiseonly_senyap",
                 "duration_s": save(wav_dir / name, rng.standard_normal(n).astype(np.float32) * 1e-3)})

    fields = ["file", "dist", "qid", "cond", "duration_s", "interferer"]
    with (args.out / "aug_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} klip -> {wav_dir} (lab noise: {'ada' if lab is not None else 'TIDAK ADA'})")


if __name__ == "__main__":
    main()
