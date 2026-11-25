"""Suara robot: sintesis Piper (ONNX) lalu playback lewat sounddevice.

Dipindah dari brone_talk.py (v4). Hanya Piper: Soniox TTS sudah dihapus di versi lama.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from brone_talk.audio.devices import resolve_device
from brone_talk.config import AudioConfig, TtsConfig
from brone_talk.telemetry import Telemetry
from brone_talk.tts.lexnorm import load_spoken
from brone_talk.tts.loudness import level_for_playback
from brone_talk.tts.normalize import clean_for_speech
from brone_talk.tts.viseme import (estimate_total_s, face_payload, schedule as viseme_schedule, sentence_packets,
                                   speech_payload, split_sentences)

try:  # piper-tts >= 1.3: library yang sama dengan CLI piper
    from piper import PiperVoice, SynthesisConfig
except ImportError:  # pragma: no cover
    PiperVoice = SynthesisConfig = None

LOG = logging.getLogger("brone-talk")
SENTENCE_LEAD_S = 0.2  # mode kalimat: paket kalimat berikutnya masuk antrean GUI sebelum gilirannya
STREAM_BLOCK = 2048  # streaming TTS: sampel per tulis (~93 ms), interupsi dicek di antara blok


def _tanpa_diam(jadwal: list[dict]) -> list[dict]:
    """Streaming: jadwal per kalimat ditutup REST (bibir rapat), bukan DIAM. DIAM tepat di batas kalimat berbalapan
    dengan jadwal kalimat berikutnya (kedip ~15 ms di GUI bawaan, simulasi 24 Nov); DIAM akhir datang dari tts/done."""
    if jadwal and jadwal[-1]["v"] == "DIAM":
        return jadwal[:-1] + [{"t": jadwal[-1]["t"], "v": "REST"}]
    return jadwal


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


class Speaker:
    def __init__(self, cfg: TtsConfig, audio: AudioConfig, telemetry: Telemetry, runtime_dir: Path) -> None:
        self.telemetry = telemetry
        self.runtime_dir = runtime_dir
        self.output_device = resolve_device(audio.output_device, input_device=False)
        self.playback_enabled = audio.playback
        self.backend = "piper"  # dibaca warmup (telemetri)
        self.piper_bin = Path(cfg.piper_bin)
        self.voice_model = cfg.voice_model
        self.voice_config = cfg.voice_config
        # suara fine-tune D21 harus menerima teks berbentuk sama dengan transkrip training-nya
        self.prepare = load_spoken(Path(cfg.textnorm_dir)) if cfg.textnorm_dir else clean_for_speech
        if not self.piper_bin.is_file():
            raise RuntimeError(f"Piper CLI tidak ditemukan: {self.piper_bin}")
        # 17 Nov 2026: model dimuat SEKALI (CLI memuat ulang di setiap jawaban, biaya tetap ±0,66 s, uji SYM-001).
        # Alur sintesis meniru piper/__main__.py (per baris, SynthesisConfig default) sehingga audionya sama.
        # BRONE_PIPER_INPROC=0 -> kembali ke CLI lama; gagal memuat -> juga kembali ke CLI.
        self._voice = None
        self._voice_lock = threading.Lock()
        if PiperVoice is not None and os.environ.get("BRONE_PIPER_INPROC", "1") != "0":
            try:
                # 20 Nov 2026: model yang sudah dipatch (python3 -m piper.patch_voice_with_alignment) memberi jumlah
                # sampel per fonem -> jadwal viseme GUI. Model belum dipatch tetap jalan, hanya tanpa viseme.
                self._voice = PiperVoice.load(self.voice_model, config_path=self.voice_config)
                LOG.info("Piper dimuat sekali (in-process): %s", self.voice_model)
            except Exception:
                LOG.exception("Gagal memuat Piper in-process; memakai CLI")
                self._voice = None
        self._alignment_rate = 22_050
        self._alignments: list[list[tuple[str, int]]] | None = None  # alignment sintesis terakhir (None = tidak ada)
        # geser jadwal viseme (ms, positif = mulut lebih lambat); diisi dari hasil uji sinkron kamera di robot
        self.viseme_offset_s = float(os.environ.get("BRONE_VISEME_OFFSET_MS", "0")) / 1000
        # Format pesan wajah ke robot/expression (mode mulut BroneGUI dipilih di GUI dengan F4):
        #   timestamp: {"expression","duration","viseme"} sekali per jawaban -> GUI "bawaan" / "fallback"
        #   kalimat: {"expression","duration","sentence"} per kalimat saat kalimat mulai -> GUI "streaming"
        #   keduanya (default): timestamp dulu, lalu paket kalimat; tiap mode GUI hanya memakai format miliknya
        #     (bawaan/fallback mengabaikan "sentence", streaming mengabaikan "viseme"), jadi GUI bebas ganti mode.
        self.face_mode = os.environ.get("BRONE_FACE_MODE", "keduanya").strip().lower()
        if self.face_mode not in {"timestamp", "kalimat", "keduanya"}:
            LOG.warning("BRONE_FACE_MODE=%r tidak dikenal; memakai keduanya", self.face_mode)
            self.face_mode = "keduanya"
        self._chunk_samples: list[int] | None = None  # panjang tiap chunk (kalimat) audio terakhir
        self._spoken_text = ""  # teks yang benar-benar disintesis (setelah normalisasi)
        self._face_stop = threading.Event()  # menghentikan publish kalimat yang belum tiba gilirannya
        # 24 Nov 2026: streaming TTS. Kalimat 1 diputar begitu selesai disintesis; kalimat berikutnya disintesis sambil
        # kalimat sebelumnya bunyi (uji hasil_llm_panjang 24 Nov: suara pertama p95 3,61 -> 0,80 s untuk jawaban
        # <= 6 kalimat, kenyaringan tetap -17 LUFS +-1 LU, tanpa jeda antar kalimat).
        # BRONE_TTS_STREAM=0 -> alur lama (seluruh jawaban disintesis dulu, lalu diputar). Butuh Piper in-process.
        self.tts_stream = os.environ.get("BRONE_TTS_STREAM", "1") != "0"
        self._stream = None  # OutputStream streaming TTS yang sedang dipakai
        self.current_text = ""  # kalimat yang sedang diucapkan (penjaga wake word dari suara sendiri)
        self.cancelled = threading.Event()  # diset oleh interupsi "halo brone": playback dihentikan
        self.sample_rate = 22_050  # id_ID-news_tts-medium.onnx.json -> audio.sample_rate
        sd.check_output_settings(device=self.output_device, samplerate=self.sample_rate, channels=1, dtype="float32")

    def synthesize(self, text: str) -> tuple[Path, float, float]:
        text = self.prepare(text)
        if not text:
            raise ValueError("Tidak ada teks untuk disintesis")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        output = self.runtime_dir / f"reply-{uuid.uuid4().hex}.wav"
        self._alignments = None
        self._chunk_samples = None
        self._spoken_text = text
        started = time.perf_counter()
        try:
            self._synthesize_piper(text, output)
        except Exception:
            output.unlink(missing_ok=True)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        return output, elapsed_ms, wav_duration(output)

    def _synthesize_piper(self, text: str, output: Path) -> None:
        if self._voice is not None:
            self._synthesize_inproc(text, output)
            return
        command = [
            str(self.piper_bin),
            "--model", str(self.voice_model),
            "--config", str(self.voice_config),
            "--output-file", str(output),
        ]
        completed = subprocess.run(
            command,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            error = completed.stderr.decode("utf-8", "replace")[-300:]
            raise RuntimeError(f"Piper gagal (rc={completed.returncode}): {error}")

    def _synthesize_inproc(self, text: str, output: Path) -> None:
        """Sama dengan `piper --output-file` tanpa opsi tambahan: tiap baris non-kosong disintesis, tanpa jeda tambahan."""
        syn_config = SynthesisConfig()
        params_set = False
        alignments: list[list[tuple[str, int]]] | None = []
        chunk_samples: list[int] = []
        with self._voice_lock, wave.open(str(output), "wb") as wav_file:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                for chunk in self._voice.synthesize(line, syn_config, include_alignments=True):
                    if not params_set:
                        wav_file.setframerate(chunk.sample_rate)
                        self._alignment_rate = chunk.sample_rate
                        wav_file.setsampwidth(chunk.sample_width)
                        wav_file.setnchannels(chunk.sample_channels)
                        params_set = True
                    wav_file.writeframes(chunk.audio_int16_bytes)
                    chunk_samples.append(len(chunk.audio_int16_array))
                    al = chunk.phoneme_alignments
                    # dipakai hanya bila total sampel alignment == panjang chunk (jadwal tidak boleh meleset)
                    if alignments is not None and al and sum(a.num_samples for a in al) == len(chunk.audio_int16_array):
                        alignments.append([(a.phoneme, int(a.num_samples)) for a in al])
                    else:
                        alignments = None
        if not params_set:
            output.unlink(missing_ok=True)
            raise RuntimeError("Piper tidak menghasilkan audio")
        self._alignments = alignments or None
        self._chunk_samples = chunk_samples

    def play(self, path: Path, on_start=None) -> None:
        """on_start(latency_s): dipanggil sesaat setelah audio mulai dikirim ke perangkat; latency_s = latensi
        keluaran yang dilaporkan PortAudio dikurangi waktu yang sudah lewat sejak sd.play (>= 0)."""
        sample_rate, data = wavfile.read(path)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)
        data = level_for_playback(data, sample_rate)  # -17 LUFS, puncak <= -1,5 dBFS (uji SYM-015)
        if self.cancelled.is_set():
            return
        if on_start is None:
            sd.play(data, sample_rate, device=self.output_device, blocking=True)
            return
        started = time.perf_counter()
        sd.play(data, sample_rate, device=self.output_device, blocking=False)
        try:
            latency = float(sd.get_stream().latency)
        except Exception:
            latency = 0.0
        try:
            on_start(max(0.0, latency - (time.perf_counter() - started)))
        except Exception:
            LOG.exception("Publish viseme gagal")  # suara tetap jalan walau wajah gagal
        sd.wait()

    def stop(self) -> None:
        """Hentikan ucapan yang sedang/akan diputar (dipanggil dari thread pemantau interupsi)."""
        self.cancelled.set()
        self._face_stop.set()
        sd.stop()

    def _publish_sentences(self, packets: list[tuple[float, dict]], t0: float) -> None:
        """Publish tiap paket kalimat saat kalimat itu mulai terdengar (t0 = saat audio mulai terdengar).

        Kalimat ke-2 dst. dikirim SENTENCE_LEAD_S lebih awal: GUI mengantrekannya dan memulainya tepat saat durasi
        kalimat sebelumnya habis. Tanpa itu paket tiba bersamaan dengan akhir kalimat sebelumnya di GUI (selisih
        terukur ~1 ms di simulasi), sehingga mulut bisa sempat DIAM di antara kalimat.
        """
        for i, (start_s, payload) in enumerate(packets):
            lead = SENTENCE_LEAD_S if i else 0.0
            if self._face_stop.wait(max(0.0, t0 + start_s - lead - time.perf_counter())):
                return
            self.telemetry.publish("robot/expression", payload)

    def say(self, text: str, expression: str) -> bool:
        if self.tts_stream and self._voice is not None and self.playback_enabled:
            return self._say_stream(text, expression)
        return self._say_full(text, expression)

    # ---- streaming TTS ------------------------------------------------------------------------------------
    def _produce_chunks(self, lines: list[str], out: queue.Queue, stop: threading.Event) -> None:
        """Thread sintesis: satu item per kalimat (chunk Piper), kenyaringan disetarakan per kalimat."""
        try:
            syn_config = SynthesisConfig()
            with self._voice_lock:
                for line in lines:
                    for chunk in self._voice.synthesize(line, syn_config, include_alignments=True):
                        if stop.is_set() or self.cancelled.is_set():
                            return
                        x = np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1)
                        y = np.asarray(level_for_playback(x, chunk.sample_rate), dtype=np.float32).reshape(-1)
                        al = chunk.phoneme_alignments
                        al = ([(a.phoneme, int(a.num_samples)) for a in al]
                              if al and sum(a.num_samples for a in al) == len(x) else None)
                        out.put({"y": y, "sr": chunk.sample_rate, "al": al})
        except BaseException as exc:  # diteruskan ke thread pemutar
            out.put(exc)
        finally:
            out.put(None)

    def _face_worker(self, items: queue.Queue) -> None:
        """Publish paket wajah pada waktunya (item = (waktu perf_counter, payload)), berurutan."""
        while True:
            item = items.get()
            if item is None:
                return
            due, payload = item
            if self._face_stop.wait(max(0.0, due - time.perf_counter())):
                return
            self.telemetry.publish("robot/expression", payload)

    def _say_stream(self, text: str, expression: str) -> bool:
        if self.cancelled.is_set():  # jawaban sudah diinterupsi: jangan sintesis lagi
            return False
        self.telemetry.stage("tts", "start")
        started = time.perf_counter()
        announced = False
        stop_synth = threading.Event()
        chunks: queue.Queue = queue.Queue()
        faces: queue.Queue = queue.Queue()
        face_thread = threading.Thread(target=self._face_worker, args=(faces,), name="tts-wajah", daemon=True)
        stream = None
        try:
            self.current_text = text
            spoken = self.prepare(text)
            if not spoken:
                raise ValueError("Tidak ada teks untuk disintesis")
            lines = [ln.strip() for ln in spoken.splitlines() if ln.strip()]
            sentences = split_sentences(spoken)
            # paket kalimat GUI hanya bila pemecahan kalimat sama dengan chunk Piper (espeak); uji: 141/141 FAQ
            cocok = sum(1 for ln in lines for ph in self._voice.phonemize(ln) if ph) == len(sentences)
            threading.Thread(target=self._produce_chunks, args=(lines, chunks, stop_synth), name="tts-sintesis",
                             daemon=True).start()
            self._face_stop.clear()
            face_thread.start()
            k, written, sr, first_ms, latency, t0 = 0, 0, self._alignment_rate, None, 0.0, 0.0
            while not self.cancelled.is_set():
                item = chunks.get()
                if isinstance(item, BaseException):
                    raise item
                if item is None:
                    break
                y, sr, al = item["y"], item["sr"], item["al"]
                now = time.perf_counter()
                if stream is None:
                    first_ms = (now - started) * 1000
                    try:
                        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", device=self.output_device)
                        stream.start()
                    except Exception:
                        LOG.exception("Streaming TTS: perangkat audio gagal dibuka; memakai alur lama")
                        stop_synth.set()
                        if stream is not None:
                            stream.close()
                        stream = None
                        announced = True  # _say_full mengirim ekspresinya sendiri
                        return self._say_full(text, expression)
                    self._stream = stream
                    latency = float(getattr(stream, "latency", 0.0) or 0.0)
                    t0 = now
                # sampel ke-n terdengar pada t0 + n/sr + latensi; sintesis terlambat (buffer kosong) menggeser jangkar
                due = t0 + written / sr
                if now > due:
                    t0 += now - due
                start = t0 + written / sr
                offset = latency + self.viseme_offset_s
                d = len(y) / sr
                if self.face_mode in ("timestamp", "keduanya"):
                    if k == 0:  # saat audio mulai ditulis; jadwal digeser sebesar latensi (sama dengan alur lama)
                        jadwal = _tanpa_diam(viseme_schedule([al], sr, offset_s=offset)) if al else None
                        rest = " ".join(sentences[1:]) if cocok else spoken
                        payload, due = face_payload(expression, jadwal, estimate_total_s(d, rest)), start
                    else:
                        # kalimat 2 dst.: tepat saat terdengar, bersamaan dengan paket kalimatnya. Bila lebih awal
                        # sebesar latensi, GUI streaming sempat menutup mulut (DIAM ~30 ms per kalimat di simulasi).
                        jadwal = _tanpa_diam(viseme_schedule([al], sr, offset_s=0.0)) if al else None
                        payload, due = face_payload(expression, jadwal), start + offset
                    if payload:
                        faces.put((due, payload))
                        announced = True
                if self.face_mode in ("kalimat", "keduanya") and cocok:
                    pkt = sentence_packets(expression, sentences[k], [len(y)], sr, d)
                    if pkt:  # dikirim saat kalimat terdengar, setelah paket ekspresi kalimat yang sama
                        faces.put((start + offset, pkt[0][1]))
                        announced = True
                for i in range(0, len(y), STREAM_BLOCK):
                    if self.cancelled.is_set():
                        break
                    stream.write(y[i:i + STREAM_BLOCK].reshape(-1, 1))
                written += len(y)
                k += 1
            played = stream is not None
            if stream is not None:
                if self.cancelled.is_set():
                    stream.abort()  # interupsi "halo brone": sisa buffer dibuang
                else:
                    stream.stop()  # tunggu sisa buffer selesai diputar
            faces.put(None)
            face_thread.join(timeout=1.0)  # tts/done (pembatal antrean GUI) dikirim setelah paket kalimat terakhir
            audio_s = written / sr
            self.telemetry.stage("tts", "done", ms=round(first_ms or 0), audio_s=round(audio_s, 2), played=played)
            LOG.info("TTS stream %.2fs sampai suara pertama (audio=%.2fs, %d kalimat, played=%s): %r",
                     (first_ms or 0) / 1000, audio_s, k, played, text[:120])
            return played
        except Exception as exc:
            self.telemetry.stage("error", "tts", detail=str(exc)[:200])
            LOG.exception("TTS streaming atau playback gagal")
            return False
        finally:
            stop_synth.set()
            self._face_stop.set()  # paket yang belum tiba gilirannya dibatalkan
            faces.put(None)
            if face_thread.is_alive():
                face_thread.join(timeout=1.0)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self._stream = None
            self.current_text = ""
            if not announced:
                self.telemetry.expression(expression)

    # ---- alur lama (BRONE_TTS_STREAM=0): seluruh jawaban disintesis ke WAV, lalu diputar ------------------
    def _say_full(self, text: str, expression: str) -> bool:
        if self.cancelled.is_set():  # jawaban sudah diinterupsi: jangan sintesis lagi
            return False
        self.telemetry.stage("tts", "start")
        path: Path | None = None
        announced = False
        face_thread: threading.Thread | None = None
        try:
            self.current_text = text
            path, synth_ms, duration_s = self.synthesize(text)
            alignments, sample_rate = self._alignments, self._alignment_rate
            chunk_samples, spoken = self._chunk_samples, self._spoken_text

            def announce(latency_s: float = 0.0) -> None:
                nonlocal announced, face_thread
                t_heard = time.perf_counter() + latency_s + self.viseme_offset_s
                if self.face_mode in ("timestamp", "keduanya"):
                    # ekspresi asli + durasi (2 desimal) saat audio mulai; jadwal viseme bila ada alignment.
                    # Tanpa alignment (CLI / model belum dipatch) GUI memakai mode mulutnya sendiri (fallback).
                    # Pada "keduanya" WAJIB sebelum paket kalimat: di GUI streaming, pesan ekspresi menghapus antrean.
                    jadwal = (viseme_schedule(alignments, sample_rate, offset_s=latency_s + self.viseme_offset_s)
                              if alignments else None)
                    self.telemetry.publish("robot/expression", speech_payload(expression, duration_s, jadwal))
                    announced = True
                if self.face_mode in ("kalimat", "keduanya"):
                    packets = sentence_packets(expression, spoken, chunk_samples, sample_rate, duration_s)
                    self._face_stop.clear()
                    face_thread = threading.Thread(target=self._publish_sentences, name="tts-kalimat",
                                                   daemon=True, args=(packets, t_heard))
                    face_thread.start()
                    announced = announced or bool(packets)

            played = False
            if self.playback_enabled and not self.cancelled.is_set():
                # publish tepat saat audio mulai keluar, bukan sebelum penyetaraan kenyaringan (30-234 ms)
                self.play(path, on_start=announce)
                played = True
            else:
                announce()
            if face_thread is not None and self.playback_enabled:
                # audio selesai/dihentikan: tts/done (pembatal antrean GUI) dikirim setelah paket kalimat terakhir
                self._face_stop.set()
                face_thread.join(timeout=1.0)
            self.telemetry.stage(
                "tts",
                "done",
                ms=round(synth_ms),
                audio_s=round(duration_s, 2),
                played=played,
            )
            LOG.info("TTS %.2fs (audio=%.2fs, played=%s): %r", synth_ms / 1000, duration_s, played, text[:120])
            return played
        except Exception as exc:
            self.telemetry.stage("error", "tts", detail=str(exc)[:200])
            LOG.exception("TTS atau playback gagal")
            return False
        finally:
            self.current_text = ""
            if face_thread is not None:  # audio sudah selesai/dihentikan: kalimat yang belum terkirim dibatalkan
                self._face_stop.set()
                face_thread.join(timeout=1.0)
            if path is not None:
                path.unlink(missing_ok=True)
            # 20 Nov 2026: ekspresi sudah dikirim saat audio mulai; mengirim lagi di sini membuat wajah
            # berkedip idle -> ekspresi -> idle di GUI. Hanya dikirim bila belum sempat (sintesis gagal / dibatalkan).
            if not announced:
                self.telemetry.expression(expression)
