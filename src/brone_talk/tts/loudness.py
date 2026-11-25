"""Penyetaraan kenyaringan ucapan sebelum diputar.

Uji SYM-015 (9 Nov 2026): semua WAV Piper memuncak di skala penuh (true peak s.d. +0,14 dBTP,
batas EBU R128 -1 dBTP) dan kenyaringan antar kalimat bervariasi (12 dari 795 file jauh lebih pelan).
Di sini setiap ucapan disetarakan ke TARGET_LUFS (ITU-R BS.1770-4: K-weighting + gating -70 LUFS
absolut / -10 LU relatif), lalu dibatasi agar puncak sampel <= PEAK_DBFS (selisih true peak vs puncak
sampel pada suara D21 terukur <= 0,14 dB, jadi -1,5 dBFS memberi true peak < -1 dBTP).

17 Nov 2026: kriteria "kalimat pelan" = lebih dari 1 LU di bawah target (EBU R128: toleransi +-1 LU
untuk siaran langsung). Versi 9 Nov membatasi gain agar puncak tidak lewat, sehingga 4/795 file berhenti
> 1 LU di bawah target. Sekarang gain penuh ke target dipakai, dan hanya puncak sesaat yang melewati
PEAK_DBFS diredam oleh limiter lookahead (maksimal MAX_LIMIT_DB, agar suara tidak rusak).

Tanpa dependensi baru: filter K-weighting dihitung dari parameter analog yang sama dengan pyloudnorm
(shelf 1500 Hz +4 dB, high-pass 38 Hz) lewat rumus RBJ, sehingga berlaku untuk sample rate apa pun.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter

TARGET_LUFS = -17.0  # 9 Nov 2026: -16 -> -17 agar file yang pelan tetap terjangkau tanpa melewati puncak (99,6% dalam ±1 LU)
PEAK_DBFS = -1.5
MAX_LIMIT_DB = 6.0      # limiter tidak boleh meredam lebih dari ini; sisanya gain diturunkan seperti versi lama
SMOOTH_S = 0.020        # limiter mulai meredam ~20 ms sebelum puncak dan kembali normal ~20 ms sesudahnya


def _biquad(kind: str, fc: float, q: float, gain_db: float, fs: int) -> tuple[np.ndarray, np.ndarray]:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    if kind == "high_shelf":
        b = [a * ((a + 1) + (a - 1) * c + 2 * np.sqrt(a) * alpha),
             -2 * a * ((a - 1) + (a + 1) * c),
             a * ((a + 1) + (a - 1) * c - 2 * np.sqrt(a) * alpha)]
        den = [(a + 1) - (a - 1) * c + 2 * np.sqrt(a) * alpha,
               2 * ((a - 1) - (a + 1) * c),
               (a + 1) - (a - 1) * c - 2 * np.sqrt(a) * alpha]
    else:  # high_pass
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        den = [1 + alpha, -2 * c, 1 - alpha]
    b, den = np.asarray(b), np.asarray(den)
    return b / den[0], den / den[0]


def integrated_loudness(x: np.ndarray, fs: int) -> float:
    """Loudness terintegrasi (LUFS) sinyal mono float. -inf bila terlalu pendek/hening."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    for kind, fc, q, g in (("high_shelf", 1500.0, 1 / np.sqrt(2), 4.0), ("high_pass", 38.0, 0.5, 0.0)):
        b, a = _biquad(kind, fc, q, g, fs)
        x = lfilter(b, a, x)
    block, step = int(0.4 * fs), int(0.1 * fs)
    if len(x) < block:
        return float("-inf")
    starts = np.arange(0, len(x) - block + 1, step)
    z = np.array([np.mean(x[s:s + block] ** 2) for s in starts])
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(z)
    z = z[lk >= -70.0]
    if len(z) == 0:
        return float("-inf")
    rel = -0.691 + 10 * np.log10(np.mean(z)) - 10.0
    z = z[(-0.691 + 10 * np.log10(z)) >= rel]
    return float(-0.691 + 10 * np.log10(np.mean(z))) if len(z) else float("-inf")


def limiter_gain(mono: np.ndarray, fs: int, ceiling: float) -> np.ndarray:
    """Kurva gain (<= 1) agar |mono * gain| <= ceiling di setiap sampel, dengan transisi halus ~SMOOTH_S sebelum dan sesudah puncak.

    need[i] = gain maksimal yang diizinkan di sampel i. Minimum-filter selebar 2L+1 lalu rata-rata selebar
    <= L+1 menjamin hasilnya tetap <= need[i] (setiap nilai yang dirata-rata adalah minimum atas jendela
    yang memuat i), sehingga puncak tidak pernah lewat ceiling.
    """
    peak = np.abs(mono)
    with np.errstate(divide="ignore"):
        need = np.where(peak > ceiling, ceiling / peak, 1.0)
    if need.min() >= 1.0:
        return np.ones_like(mono)
    L = max(1, int(SMOOTH_S * fs))
    env = minimum_filter1d(need, size=2 * L + 1, mode="nearest")
    return uniform_filter1d(env, size=L + 1, mode="nearest")


def level_for_playback(data: np.ndarray, fs: int) -> np.ndarray:
    """Setarakan ke TARGET_LUFS; puncak sampel <= PEAK_DBFS lewat limiter. data: float, skala [-1, 1]."""
    mono = data.mean(axis=1) if data.ndim == 2 else data
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak <= 0.0:
        return data
    loud = integrated_loudness(mono, fs)
    gain = 10 ** ((TARGET_LUFS - loud) / 20) if np.isfinite(loud) else 1.0
    ceiling = 10 ** (PEAK_DBFS / 20)
    # limiter hanya boleh meredam paling banyak MAX_LIMIT_DB; di luar itu gain keseluruhan diturunkan
    gain = min(gain, ceiling / peak * 10 ** (MAX_LIMIT_DB / 20))
    y = data * gain
    ymono = y.mean(axis=1) if y.ndim == 2 else y
    g = limiter_gain(ymono, fs, ceiling)
    y = y * (g[:, None] if y.ndim == 2 else g)
    return y.astype(np.float32)
