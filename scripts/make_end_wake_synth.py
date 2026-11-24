"""Data sintesis uji "selesai brone" (akhir sesi mode wake word).

Piper (dua suara) -> 16 kHz mono -> augmentasi (jarak/gema, derau putih, babble).
Keluaran: runtime/uji-end-wake/{wav/*.wav, manifest.csv}. Deterministik (seed tetap).

  .venv/bin/python scripts/make_end_wake_synth.py [--out runtime/uji-end-wake]

Label:
  END     ucapan harus mengakhiri sesi ("selesai brone", nama wajib).
  NO_END  ucapan TIDAK boleh mengakhiri sesi (selesai/diam/stop tanpa nama, pertanyaan biasa,
          "halo brone", dan kalimat jebakan yang memuat kata "selesai" atau nama robot).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000
SEED = 20261110

# (id, teks, label, kategori)
PHRASES: list[tuple[str, str, str, str]] = [
    ("E01", "selesai brone", "END", "baku"),
    ("E02", "Selesai, Brone.", "END", "baku"),
    ("E03", "selesai bron", "END", "varian_nama"),
    ("E04", "oke selesai brone", "END", "awalan"),
    ("E05", "sudah selesai brone", "END", "awalan"),
    ("E06", "terima kasih, selesai brone", "END", "awalan"),
    ("E07", "selesai brone, terima kasih", "END", "akhiran"),
    ("E08", "baik brone, selesai brone", "END", "awalan"),
    ("E09", "selesai brone sampai jumpa", "END", "akhiran"),
    ("E10", "ya selesai brone ya", "END", "awalan"),
    ("N01", "selesai", "NO_END", "tanpa_nama"),
    ("N02", "diam", "NO_END", "tanpa_nama"),
    ("N03", "stop", "NO_END", "tanpa_nama"),
    ("N04", "sampai jumpa", "NO_END", "tanpa_nama"),
    ("N05", "brone", "NO_END", "nama_saja"),
    ("N06", "halo brone", "NO_END", "wake"),
    ("N07", "siapa rektor UB sekarang", "NO_END", "pertanyaan"),
    ("N08", "di mana gedung FILKOM", "NO_END", "pertanyaan"),
    ("N09", "kapan wisuda UB berikutnya", "NO_END", "pertanyaan"),
    ("N10", "apa kepanjangan brone", "NO_END", "pertanyaan"),
    ("N11", "jam berapa kuliah selesai", "NO_END", "jebakan_selesai"),
    ("N12", "kapan pendaftaran selesai", "NO_END", "jebakan_selesai"),
    ("N13", "tugas saya sudah selesai", "NO_END", "jebakan_selesai"),
    ("N14", "brone sudah selesai belum", "NO_END", "jebakan_selesai"),
    ("N15", "siapa yang membuat brone selesai", "NO_END", "jebakan_selesai"),
    ("N16", "saya sudah selesai sekarang giliran teman saya", "NO_END", "jebakan_selesai"),
    ("N17", "berapa lama kuliah di UB sampai selesai, brone", "NO_END", "jebakan_selesai_nama"),
    ("N18", "kapan skripsi saya selesai, brone", "NO_END", "jebakan_selesai_nama"),
    # sapaan Bron sendiri (messages.SAY_WAKE_SESSION) memuat frasa akhir; bila gemanya bocor ke mik
    # dan terdengar STT, sesi tidak boleh berakhir. Hanya dibuat di kondisi ECHO_CONDITIONS.
    ("N19", "hai, Brone di sini. kalau selesai tinggal bilang 'selesai brone'.", "NO_END", "gema_sapaan_sendiri"),
    ("N20", "kalau selesai tinggal bilang selesai brone", "NO_END", "gema_sapaan_sendiri"),
]
ECHO_PHRASES = {"N19", "N20"}

VOICES = {
    "d21": ("id_ID-brone_D21-medium", 1.0),
    "news": ("id_ID-news_tts-medium", 1.0),
}
# kecepatan bicara tambahan (length_scale Piper): 1.0 normal, 0.85 cepat, 1.2 lambat
SPEEDS = {"normal": 1.0, "cepat": 0.85, "lambat": 1.2}

# kondisi augmentasi: nama -> parameter
CONDITIONS = ["clean", "far", "noise10", "noise5", "babble10"]
ECHO_CONDITIONS = ["far", "noise10"]  # gema sapaan: jauh/bergema atau bercampur derau, bukan klip bersih


def piper(text: str, voice: str, length_scale: float, out: Path) -> None:
    model = ROOT / "models" / "voices" / f"{voice}.onnx"
    cmd = [
        str(Path(sys.executable).parent / "piper"),
        "--model", str(model),
        "--config", str(model) + ".json",
        "--length_scale", str(length_scale),
        "--output-file", str(out),
    ]
    done = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, timeout=60, check=False)
    if done.returncode != 0 or not out.is_file():
        raise RuntimeError(f"Piper gagal untuk {text!r}: {done.stderr.decode('utf-8', 'replace')[-200:]}")


def load16k(path: Path) -> np.ndarray:
    sr, data = wavfile.read(path)
    x = data.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        x /= np.iinfo(data.dtype).max
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def pad(x: np.ndarray, lead_s: float = 0.35, tail_s: float = 0.6) -> np.ndarray:
    """Jeda senyap di depan/belakang, seperti klip hasil endpointing mikrofon."""
    return np.concatenate([np.zeros(int(lead_s * SR), np.float32), x, np.zeros(int(tail_s * SR), np.float32)])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    active = speech[np.abs(speech) > 0.01 * np.max(np.abs(speech))]
    p_speech = rms(active) if active.size else rms(speech)
    noise = np.resize(noise, speech.shape)
    gain = p_speech / (rms(noise) * 10 ** (snr_db / 20))
    return speech + noise * gain


def reverb(x: np.ndarray, rt60: float, rng: np.random.Generator) -> np.ndarray:
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir ** 2))
    return signal.fftconvolve(x, rir)[: len(x)].astype(np.float32)


def augment(x: np.ndarray, cond: str, babble_pool: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if cond == "clean":
        y = x
    elif cond == "far":  # ~2,25 m: gema ruangan RT60 0,6 s + redaman + sedikit derau ruang
        y = reverb(x, 0.6, rng) * 0.25
        y = mix_at_snr(y, rng.standard_normal(len(y)).astype(np.float32), 25)
    elif cond.startswith("noise"):
        y = mix_at_snr(x, rng.standard_normal(len(x)).astype(np.float32), float(cond[5:]))
    elif cond.startswith("babble"):  # 4 ucapan lain ditumpuk acak (pengganti derau keramaian)
        mix = np.zeros(len(x), np.float32)
        for clip in rng.choice(len(babble_pool), size=4, replace=False):
            c = babble_pool[clip]
            off = int(rng.integers(0, max(1, len(c))))
            mix += np.resize(np.roll(c, off), len(x))
        y = mix_at_snr(x, mix, float(cond[6:]))
    else:
        raise ValueError(cond)
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak * 0.8).astype(np.float32)  # puncak seragam; level relatif sudah dikunci SNR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "runtime" / "uji-end-wake")
    args = ap.parse_args()
    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    base: list[tuple[str, str, str, str, str, str, np.ndarray]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for pid, text, label, cat in PHRASES:
            for vkey, (vname, _) in VOICES.items():
                # END: tiga kecepatan (suara D21); lainnya hanya kecepatan normal supaya jumlah klip wajar
                speeds = SPEEDS if (label == "END" and vkey == "d21") else {"normal": 1.0}
                for sname, scale in speeds.items():
                    tmp_wav = Path(tmp) / f"{pid}_{vkey}_{sname}.wav"
                    piper(text, vname, scale, tmp_wav)
                    base.append((pid, text, label, cat, vkey, sname, pad(load16k(tmp_wav))))

    pool = [b[6] for b in base]
    rows = []
    for pid, text, label, cat, vkey, sname, x in base:
        for cond in (ECHO_CONDITIONS if pid in ECHO_PHRASES else CONDITIONS):
            y = augment(x, cond, pool, rng)
            name = f"{pid}_{vkey}_{sname}_{cond}.wav"
            wavfile.write(wav_dir / name, SR, (y * 32767).astype(np.int16))
            rows.append({
                "file": f"wav/{name}", "phrase_id": pid, "text": text, "label": label, "category": cat,
                "voice": vkey, "speed": sname, "condition": cond, "duration_s": round(len(y) / SR, 2),
            })

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_end = sum(r["label"] == "END" for r in rows)
    print(f"{len(rows)} klip -> {args.out}  (END={n_end}, NO_END={len(rows) - n_end})")


if __name__ == "__main__":
    main()
