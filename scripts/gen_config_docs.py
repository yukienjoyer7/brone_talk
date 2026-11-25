#!/usr/bin/env python3
"""Tulis docs/configuration.md dari config.py, supaya dokumen tidak menyimpang dari kode.

Setiap field di config.py yang membaca env wajib punya komentar "# NAMA_ENV ..." di baris yang sama.
  python scripts/gen_config_docs.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "brone_talk" / "config.py"
OUT = ROOT / "docs" / "configuration.md"

GROUPS = {
    "MqttConfig": "MQTT (telemetri ke Jetson)",
    "AudioConfig": "Audio: mic dan speaker",
    "SttConfig": "STT (whisper lokal)",
    "MaiConfig": "STT cloud (MAI-Transcribe-2 via Azure Voice Live)",
    "TtsConfig": "TTS (Piper)",
    "EngineSettings": "Mesin jawaban (FAQ, RAG, gerbang kemiripan, LLM)",
}


def main() -> None:
    src = CONFIG.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    out = [
        "# Konfigurasi BRONE Talk",
        "",
        "Dibangkitkan otomatis oleh `scripts/gen_config_docs.py` dari `src/brone_talk/config.py`.",
        "Jangan edit berkas ini; ubah `config.py` lalu jalankan ulang skripnya.",
        "",
        "Default di tabel = **nilai produksi efektif** (lihat `docs/baseline-config.md`). Beberapa berbeda",
        "dengan default kode versi lama; selisihnya ditandai pada kolom catatan.",
        "",
        "Variabel warisan yang tidak lagi berpengaruh (RAG v1-v3, Ollama, Soniox, KB format lama)",
        "tidak dibawa ke sini; lihat `docs/decisions.md`.",
        "",
    ]
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in GROUPS:
            continue
        out += [f"## {GROUPS[node.name]}", "", f"`{node.name}`", "",
                "| Field | Default | Variabel env | Keterangan |", "|---|---|---|---|"]
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            line = lines[stmt.lineno - 1]
            m = re.search(r"#\s*(BRONE_[A-Z0-9_]+)\s*(.*)$", line)
            default = ast.get_source_segment(src, stmt.value) if stmt.value is not None else ""
            default = default.replace("|", "\\|")
            if m:
                env, note = m.group(1), m.group(2).strip()
            else:
                env, note = "(tetap)", ""
            out.append(f"| `{stmt.target.id}` | `{default}` | `{env}` | {note} |")
        out.append("")
    out += [
        "## Pengaturan lain",
        "",
        "| Variabel | Default | Keterangan |",
        "|---|---|---|",
        "| `BRONE_WARMUP_LLM` | `1` | validasi mesin jawaban saat start (\"Halo Bron\") |",
        "| `BRONE_LOG_LEVEL` | `INFO` | tingkat log |",
        "| `BRONE_AWAIT_SECONDS` | `0` | jendela satu pertanyaan tanpa nama setelah \"Ya?\"; 0 = mati |",
        "| `BRONE_SOCIAL_AWAIT_SECONDS` | `8` | jendela satu tanggapan setelah Bron bertanya balik |",
        "",
        "Rahasia (`BRONE_DEEPSEEK_API_KEY`, `BRONE_MAI_API_KEY`) hanya di `~/.config/brone-talk/brone.env`.",
        "",
    ]
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"ditulis {OUT}")


if __name__ == "__main__":
    main()
