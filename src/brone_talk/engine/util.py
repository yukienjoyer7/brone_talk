"""Utilitas bersama mesin jawaban: normalisasi teks, HTTP JSON, dan tunggu server siap.

Helper env ada di config.py.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any


LOG = logging.getLogger("brone-talk.v4")


def norm_text(text: str) -> str:
    """Huruf kecil, tanpa tanda baca. Harus sama dengan kb/build_kb.py:norm_text."""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(text).lower())).strip()

def expiry_report(records: list[dict], today: str, warn_days: int = 14) -> tuple[list[str], list[str]]:
    """(id yang sudah kedaluwarsa, "id (tanggal)" yang kedaluwarsa dalam warn_days hari) dari rekaman KB."""
    limit = (_dt.date.fromisoformat(today) + _dt.timedelta(days=warn_days)).isoformat()
    expired: list[str] = []
    soon: list[str] = []
    for r in records:
        until = r.get("berlaku_hingga")
        if not until:
            continue
        if str(until) < today:
            expired.append(str(r.get("id")))
        elif str(until) <= limit:
            soon.append(f"{r.get('id')} ({until})")
    return expired, soon


def warn_stale(kind: str, records: list[dict], today: str) -> None:
    """Catat di log entri KB yang sudah/hampir kedaluwarsa, supaya tidak mati diam-diam."""
    expired, soon = expiry_report(records, today)
    if expired:
        LOG.warning("KB: %d %s kedaluwarsa dan tidak dipakai (perbarui kb/knowledge.yaml): %s",
                    len(expired), kind, ", ".join(expired))
    if soon:
        LOG.warning("KB: %d %s kedaluwarsa dalam 14 hari: %s", len(soon), kind, ", ".join(soon))


def _http_json(url: str, body: dict | None, timeout: float, headers: dict | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def wait_for_server(base_url: str, timeout_s: float) -> bool:
    """Tunggu llama-server siap (/health 200) maksimal timeout_s detik."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _http_json(base_url.rstrip("/") + "/health", None, timeout=3)
            return True
        except Exception:
            if time.monotonic() > deadline:
                return False
            time.sleep(1)
