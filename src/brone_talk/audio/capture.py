"""Perekaman audio: mic dipilih dari PipeWire (ALSA) atau sounddevice, lalu endpointing ucapan.

Dua jalur (ALSA lewat PipeWire, dan sounddevice) memakai satu Endpointer dan satu finalisasi.
Rekaman 4 mic mentah untuk arah suara (kunci penanya) dihapus bersama kunci penanya (19 Okt 2026).
stream() memberi aliran tanpa batas untuk wake word "halo brone" dan pemantau interupsi (Okt 2026).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from threading import Event

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from brone_talk.audio.constants import SAMPLE_RATE, VAD_WINDOW
from brone_talk.audio.devices import device_name, parse_card_nodes, resolve_device
from brone_talk.audio.endpointing import CaptureLost, Endpointer, StopCapture, Utterance
from brone_talk.audio.kws import WakeHeard, WakeWord, watch
from brone_talk.audio.vad import SileroVad
from brone_talk.config import AudioConfig

LOG = logging.getLogger("brone-talk")
SETTLE_SILENCE_S = 0.3  # setelah wake word: tunggu jeda sebelum "Ya?" (ucapan sebelum "Ya?" dibuang)
SETTLE_MAX_S = 2.0


@dataclass
class Capture:
    path: Path
    duration_s: float
    peak_rms: float
    audio: np.ndarray | None = None  # mono float32 @ SAMPLE_RATE (cli mic-test menghitung RMS-nya)


class Recorder:
    def __init__(self, cfg: AudioConfig, telemetry, stop_event: Event, vad: SileroVad,
                 runtime_dir: Path) -> None:
        self.cfg = cfg
        self.telemetry = telemetry
        self.stop_event = stop_event
        self.vad = vad
        self.runtime_dir = runtime_dir
        self.threshold = cfg.vad_threshold
        self.max_duration_s = cfg.max_utterance_s
        self.silence_s = cfg.end_silence_s
        self.min_speech_s = cfg.min_speech_s
        self.min_utterance_rms = cfg.min_utterance_rms  # gate energi (ambient ruangan ~0,01-0,05)
        self.backend = cfg.backend
        self.input_device: int | None = None
        self.alsa_device = ""
        self.arecord_bin = ""
        self.pw_node = ""
        self.noise_reduce_enabled = cfg.noise_reduce
        self.noise_reduce_prop = cfg.noise_prop_decrease
        self.noise_reduce_stationary = cfg.noise_stationary
        self.noise_reducer = None
        if self.noise_reduce_enabled:
            try:
                import noisereduce as _nr

                self.noise_reducer = _nr
                LOG.info("Denoise in-code aktif (prop_decrease=%.2f)", self.noise_reduce_prop)
            except Exception as exc:  # pragma: no cover
                LOG.warning("noisereduce tidak tersedia: %s", exc)
                self.noise_reduce_enabled = False
        if self.backend == "alsa":
            self.alsa_device = cfg.alsa_device
            # mic dipilih tiap rekaman dari kartu yang tersambung; urutan = prioritas.
            # pw_node hanya cadangan kalau tak ada yang cocok.
            self.pw_node = cfg.pw_node.strip()
            self.pw_nodes = parse_card_nodes(cfg.pw_nodes)
            self._active_pw_node = ""
            self.arecord_bin = shutil.which("arecord") or ""
            if not self.arecord_bin:
                raise RuntimeError("arecord tidak tersedia untuk input ALSA")
            cards = Path("/proc/asound/cards").read_text(encoding="utf-8")
            if "BRIO" in self.alsa_device.upper() and "[BRIO" not in cards:
                raise RuntimeError(f"ALSA capture device tidak tersedia: {self.alsa_device}")
        elif self.backend == "sounddevice":
            self.input_device = resolve_device(cfg.input_device, input_device=True)
            sd.check_input_settings(device=self.input_device, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        else:
            raise RuntimeError("BRONE_AUDIO_INPUT_BACKEND harus 'alsa' atau 'sounddevice'")

    def _endpointer(self, start_s: float = 0.0) -> tuple[Endpointer, int]:
        silence_limit = max(1, int(self.silence_s * SAMPLE_RATE / VAD_WINDOW))
        window_limit = max(1, int((self.max_duration_s + start_s) * SAMPLE_RATE / VAD_WINDOW))
        start_limit = int(start_s * SAMPLE_RATE / VAD_WINDOW)
        speech_limit = max(1, int(self.max_duration_s * SAMPLE_RATE / VAD_WINDOW))
        endpointer = Endpointer(self.vad, self.telemetry, self.threshold, silence_limit, window_limit, start_limit,
                                speech_limit)
        return endpointer, window_limit

    def record(self, start_s: float = 0.0, wake: WakeWord | None = None) -> Capture | None:
        """Satu ucapan. start_s > 0: menyerah (None) bila ucapan belum mulai dalam start_s detik.

        wake: model "halo brone" ikut mendengar selama merekam (jendela percakapan); terdengar ->
        tunggu penutur diam lalu WakeHeard.
        """
        self.vad.reset()
        if self.backend == "alsa":
            return self._record_alsa(start_s, wake)
        return self._record_sounddevice(start_s, wake)

    # -- aliran tanpa batas (wake word, interupsi) ---------------------------------
    def stream(self, stop: Event | None = None) -> Iterator[np.ndarray]:
        """Blok VAD_WINDOW sampel sampai stop_event atau `stop` diset. Membuka mic sendiri."""
        stops = [self.stop_event] + ([stop] if stop is not None else [])
        if self.backend == "sounddevice":
            with sd.InputStream(device=self.input_device, samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=VAD_WINDOW) as stream:
                while not any(e.is_set() for e in stops):
                    yield stream.read(VAD_WINDOW)[0][:, 0].copy()
            return
        process = self._spawn_arecord(self._select_pw_node(Path("/proc/asound/cards").read_text(encoding="utf-8")))
        try:
            while not any(e.is_set() for e in stops):
                raw = process.stdout.read(VAD_WINDOW * 2)
                if len(raw) != VAD_WINDOW * 2:
                    raise CaptureLost()
                yield np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        finally:
            self._kill(process)

    def wait_for_wake(self, detector) -> bool:
        """Tidur sampai "halo brone" terdengar, lalu tunggu jeda bicara. False bila layanan berhenti."""
        self.telemetry.stage("listening", "sleep")
        blocks = self.stream()
        try:
            if not detector.listen(blocks):
                return False
            self._settle(blocks)
            return True
        except CaptureLost:
            device = self._next_capture_device()
            if not device:
                raise RuntimeError("arecord berhenti saat menunggu wake word") from None
            LOG.warning("Capture %s gagal, coba %s", self.alsa_device, device)
            self.alsa_device = device
            return False
        finally:
            blocks.close()

    def wait_silence(self) -> None:
        """Tunggu penutur diam sejenak (mis. setelah interupsi "halo brone" di tengah jawaban)."""
        blocks = self.stream()
        try:
            self._settle(blocks)
        except CaptureLost:
            pass
        finally:
            blocks.close()

    def _settle(self, blocks: Iterator[np.ndarray]) -> None:
        self.vad.reset()
        need = max(1, int(SETTLE_SILENCE_S * SAMPLE_RATE / VAD_WINDOW))
        quiet = 0
        for tick, samples in enumerate(blocks, 1):
            quiet = quiet + 1 if self.vad.probability(samples) < self.threshold else 0
            if quiet >= need or tick * VAD_WINDOW >= SETTLE_MAX_S * SAMPLE_RATE:
                return

    # -- sumber audio ---------------------------------------------------------
    def _sounddevice_blocks(self, window_limit: int):
        with sd.InputStream(device=self.input_device, samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=VAD_WINDOW) as stream:
            for _ in range(window_limit):
                if self.stop_event.is_set():
                    raise StopCapture()
                block, overflowed = stream.read(VAD_WINDOW)
                if overflowed:
                    LOG.warning("Input audio overflow")
                yield block[:, 0].copy()

    def _pipe_blocks(self, process: subprocess.Popen, window_limit: int):
        for _ in range(window_limit):
            if self.stop_event.is_set():
                raise StopCapture()
            raw = process.stdout.read(VAD_WINDOW * 2)
            if len(raw) != VAD_WINDOW * 2:
                raise CaptureLost()
            yield np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    # -- jalur perekaman --------------------------------------------------------
    def _record_sounddevice(self, start_s: float = 0.0, wake: WakeWord | None = None) -> Capture | None:
        endpointer, window_limit = self._endpointer(start_s)
        self.telemetry.stage("listening", "start", device=device_name(self.input_device), sample_rate=SAMPLE_RATE)
        LOG.info("Mendengarkan dari %s", device_name(self.input_device))
        blocks = self._sounddevice_blocks(window_limit)
        try:
            utt = endpointer.run(watch(blocks, wake) if wake else blocks)
        except StopCapture:
            return None
        return self._finish(utt)

    def _record_alsa(self, start_s: float = 0.0, wake: WakeWord | None = None) -> Capture | None:
        endpointer, window_limit = self._endpointer(start_s)
        cards = Path("/proc/asound/cards").read_text(encoding="utf-8")
        pw_node = self._select_pw_node(cards)
        self.telemetry.stage("listening", "start", device=self.alsa_device, sample_rate=SAMPLE_RATE)
        LOG.info("Mendengarkan dari ALSA %s", self.alsa_device)
        process = self._spawn_arecord(pw_node)
        utt: Utterance | None = None
        stopped = lost = woke = False
        try:
            blocks = self._pipe_blocks(process, window_limit)
            utt = endpointer.run(watch(blocks, wake) if wake else blocks)
        except StopCapture:
            stopped = True
        except CaptureLost:
            lost = True
        except WakeHeard:
            woke = True
            try:  # sisa ucapan setelah "halo brone" dibuang, sama seperti saat bangun dari tidur
                self._settle(self._pipe_blocks(process, window_limit))
            except (StopCapture, CaptureLost):
                pass
        finally:
            self._kill(process)
        if woke:
            raise WakeHeard()
        if stopped:
            return None
        if lost:
            device = self._next_capture_device()
            if device:
                LOG.warning("Capture %s gagal, coba %s", self.alsa_device, device)
                self.alsa_device = device
                return self._record_alsa(start_s, wake)
            raise RuntimeError("arecord berhenti sebelum satu window audio diterima")
        return self._finish(utt)

    # -- finalisasi bersama -------------------------------------------------------
    def _finish(self, utt: Utterance) -> Capture | None:
        if not utt.frames:
            self.telemetry.vad("silence", utt.peak_rms)
            self.telemetry.stage("listening", "timeout", peak_rms=round(utt.peak_rms * 32768, 1))
            LOG.info("Tidak ada suara terdeteksi dalam %.0fs (peak_rms=%.4f) — cek mic/gain jika ini terus terjadi",
                     self.max_duration_s, utt.peak_rms)
            return None
        audio = self._denoise(np.concatenate(utt.frames))
        duration_s = len(audio) / SAMPLE_RATE
        if duration_s < self.min_speech_s:
            self.telemetry.stage("listening", "too_short", audio_s=round(duration_s, 2))
            return None
        # buang klip tanpa energi bicara nyata (sisa derau ruangan)
        utterance_rms = float(np.sqrt(np.mean(np.square(audio))))
        if utterance_rms < self.min_utterance_rms:
            self.telemetry.stage("listening", "low_energy", rms=round(utterance_rms, 4))
            LOG.info("Klip energi rendah (rms=%.4f) diabaikan", utterance_rms)
            return None

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        path = self.runtime_dir / f"input-{uuid.uuid4().hex}.wav"
        wavfile.write(path, SAMPLE_RATE, (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))
        self.telemetry.vad("stopped", utt.peak_rms)
        self.telemetry.stage(
            "listening", "done",
            audio_s=round(duration_s, 2),
            peak_rms=round(utt.peak_rms * 32768, 1),
        )
        return Capture(path=path, duration_s=duration_s, peak_rms=utt.peak_rms, audio=audio)

    # -- helper -------------------------------------------------------------------
    def _spawn_arecord(self, pw_node: str) -> subprocess.Popen:
        command = [self.arecord_bin, "--quiet", "--device", self.alsa_device,
                   "--format=S16_LE", f"--rate={SAMPLE_RATE}", "--channels=1", "--file-type=raw"]
        env = os.environ.copy()
        if pw_node:
            env["PIPEWIRE_NODE"] = pw_node
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, env=env)

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _next_capture_device(self) -> str:
        """Kandidat device berikutnya kalau arecord gagal membuka device."""
        # hw:Array (XVF3800) dipegang PipeWire, jadi fallback plughw langsung
        # pasti "busy"; "default" tetap lewat PipeWire ke default source.
        order = ["pipewire", "default"]
        tried = getattr(self, "_tried_devices", set())
        tried.add(self.alsa_device)
        self._tried_devices = tried
        for cand in order:
            if cand not in tried:
                return cand
        return ""

    def _select_pw_node(self, cards: str) -> str:
        node = next((n for card, n in self.pw_nodes if card in cards), self.pw_node)
        if node != self._active_pw_node:
            LOG.info("Mic aktif: PipeWire node %s", node)
            self.telemetry.stage("listening", "mic", node=node)
            self._active_pw_node = node
        return node

    def _denoise(self, audio: "np.ndarray") -> "np.ndarray":
        if not self.noise_reduce_enabled or self.noise_reducer is None:
            return audio
        try:
            return self.noise_reducer.reduce_noise(
                y=audio, sr=SAMPLE_RATE,
                prop_decrease=self.noise_reduce_prop,
                stationary=self.noise_reduce_stationary,
            )
        except Exception as exc:  # pragma: no cover
            LOG.warning("Denoise gagal, pakai audio asli: %s", exc)
            return audio
