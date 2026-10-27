# Perintah umum BRONE Talk. Lihat README.md dan docs/.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PYTHONPATH := src

.PHONY: install test golden ask self-test run lint kb-check clean

install:  # pasang venv dan paket (tanpa menjalankan layanan)
	bash deploy/install.sh

test:  # tes unit dan golden (golden dilewati bila fixture belum dibangkitkan)
	$(PY) -m unittest discover -s tests -v

golden:  # bangkitkan fixture golden dari kode v4 lama (butuh ~/brone-talk-v4 dan server 8082/8083)
	$(PY) scripts/make_golden.py

ask:  # uji mesin jawaban lewat teks: make ask Q="siapa dekan filkom"
	$(PY) -m brone_talk ask "$(Q)"

self-test:  # validasi model dan mesin tanpa mic/speaker
	$(PY) -m brone_talk self-test

run:  # jalankan di terminal (di NUC biasanya lewat systemd)
	$(PY) -m brone_talk run

lint:  # pemeriksaan statis (ruff; pasang dengan: .venv/bin/pip install -e '.[dev]')
	$(PY) -m ruff check src tests scripts

kb-check:  # validasi KB + daftar data bertanggal yang sudah/hampir kedaluwarsa (jalankan tiap minggu)
	$(PY) kb/build_kb.py --check

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
