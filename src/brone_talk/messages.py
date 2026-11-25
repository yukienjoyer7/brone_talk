"""Kalimat tetap yang diucapkan robot saat menolak atau gagal. Dipindah dari brone_v4.py."""

from __future__ import annotations


MSG_UNCLEAR_AUDIO = "Maaf, suaranya kurang jelas. Bisa diulang pertanyaannya?"

MSG_NOT_UNDERSTOOD = "Maaf, Bron kurang paham pertanyaannya. Bisa diulang dengan kalimat lain?"

MSG_NO_INFO = "Maaf, Bron belum punya informasi tentang itu. Informasi lengkap bisa dilihat di www.ub.ac.id."

MSG_MOTION = "Maaf, kontrol gerakan Bron belum diaktifkan."

MSG_SERVICE_DOWN = "Maaf, layanan percakapan Bron sedang bermasalah."

# jawaban obrolan dari LLM menyebut nama/angka yang tidak ada di ucapan (lihat AnswerEngine._chat_or_refuse)
MSG_CHAT_FALLBACK = "Hehe, Bron belum bisa menjawab yang itu. Kalau soal Universitas Brawijaya, tanya saja!"

# Kalimat yang diucapkan saat alur percakapan (bukan penolakan mesin jawaban).
SAY_CONFIRM = "Ya?"
# 9 Nov 2026: sapaan saat "halo brone" membangunkan Bron; sesi aktif sampai "selesai brone".
SAY_WAKE_SESSION = "hai, Brone di sini. kalau selesai tinggal bilang 'selesai brone'."
SAY_CALL_HINT = "Ya? Sebut nama Bron di awal atau akhir pertanyaan, ya."  # bila jendela susulan mati
SAY_STOP = "Oke, sampai jumpa!"  # 20 Nov 2026: jawaban "selesai brone"
SAY_SERVICE_ERROR = "Maaf, layanan percakapan Bron sedang bermasalah."
SAY_FALLBACK = "Maaf, saya tidak mengerti."  # jawaban kosong dari mesin (sama dengan produksi)
