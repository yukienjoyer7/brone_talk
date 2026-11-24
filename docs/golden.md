# Fixture golden

Dibangkitkan oleh `scripts/make_golden.py` dari kode v4 lama (`~/brone-talk-v4`). Hasilnya di
`tests/golden/`, dan **tidak di-commit** karena berisi transkrip STT asli dari mikrofon robot.

| File | Isi | Jumlah |
|---|---|---|
| tts_normalize.jsonl | clean_for_speech atas jawaban nyata | 117 |
| nlu_wake.jsonl | collapse_repeats, wake_question, clock_answer, stop | 587 |
| stt_filters.jsonl | _stt_fix, _mai_fix, stt_garbage_reason, stt_strip_prompt_sentence | 587 |
| engine_routes.jsonl | rute FAQ dan kemiripan chunk teratas, 60 pertanyaan eval | 60 |

## Keterbatasan yang sudah diketahui

- **stt_filters.garbage_reason kosong untuk semua baris.** Fixture dibuat tanpa `stats` dan `vocab`,
  sehingga filter hampir tidak aktif. Fixture ini belum cukup sebagai bukti filter sampah.
- **Satu mismatch FAQ di baseline:** "siapa dekan fakultas mipa" diharapkan `faq:dekan-fstem`, tetapi
  tidak cocok dengan FAQ mana pun. Ini sudah ada di versi lama dan tidak diperbaiki di sini.
- Rute dicatat dengan `BRONE_MODE=faq`, jadi jalur LLM tidak tercakup.
- `engine_routes` bergantung pada server bge-m3 (8082) yang hidup saat merekam.
