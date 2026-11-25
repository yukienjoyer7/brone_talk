"""Evaluasi "selesai brone" pada data sintesis (scripts/make_end_wake_synth.py).

Tiap klip -> Transcriber produksi (whisper; MAI bila BRONE_STT_BACKEND=mai dan kunci ada)
-> koreksi STT -> SESSION_END_RE (aturan akhir sesi di app.py). Keluaran: results.csv + ringkasan.

  .venv/bin/python scripts/eval_end_wake.py [--dir runtime/uji-end-wake] [--limit N]
  .venv/bin/python scripts/eval_end_wake.py --rescore   # nilai ulang results.csv dengan aturan terbaru
                                                         # (teks STT sama) -> results_rescored.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from brone_talk.config import Settings
from brone_talk.nlu.wake import SESSION_END_RE
from brone_talk.stt.transcriber import Transcriber

ROOT = Path(__file__).resolve().parents[1]


class NullTelemetry:
    def __getattr__(self, name):
        return lambda *a, **k: None


def rate(num: int, den: int) -> str:
    return f"{num}/{den} = {100 * num / den:.1f}%" if den else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "runtime" / "uji-end-wake")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rescore", action="store_true", help="tanpa STT: terapkan SESSION_END_RE terbaru ke results.csv")
    args = ap.parse_args()

    if args.rescore:
        out = []
        for r in csv.DictReader((args.dir / "results.csv").open(encoding="utf-8")):
            ended = bool(SESSION_END_RE.search(r["stt_text"]))
            out.append({**r, "stt_ms": int(r["stt_ms"]), "ended": int(ended),
                        "correct": int(ended == (r["label"] == "END"))})
        write(args.dir / "results_rescored.csv", out)
        summarize(out)
        return

    settings = Settings.from_env()
    stt = Transcriber(settings.stt, settings.mai, NullTelemetry())
    stt.load()
    rows = list(csv.DictReader((args.dir / "manifest.csv").open(encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    out = []
    for i, r in enumerate(rows, 1):
        started = time.perf_counter()
        text = stt.transcribe(args.dir / r["file"])
        ms = round((time.perf_counter() - started) * 1000)
        ended = bool(SESSION_END_RE.search(text))
        ok = ended == (r["label"] == "END")
        out.append({**r, "stt_engine": getattr(stt, "last_engine", "whisper"), "stt_text": text,
                    "ended": int(ended), "correct": int(ok), "stt_ms": ms})
        if i % 20 == 0:
            print(f"{i}/{len(rows)}", flush=True)

    write(args.dir / "results.csv", out)
    summarize(out)


def write(path: Path, out: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def summarize(out: list[dict]) -> None:
    for o in out:
        o["ended"] = int(o["ended"])
    end = [o for o in out if o["label"] == "END"]
    neg = [o for o in out if o["label"] == "NO_END"]
    print(f"\nengine: {sorted({o['stt_engine'] for o in out})}")
    print(f"recall END (klip END yang mengakhiri sesi): {rate(sum(o['ended'] for o in end), len(end))}")
    print(f"false end (klip NO_END yang salah mengakhiri): {rate(sum(o['ended'] for o in neg), len(neg))}")
    lat = sorted(o["stt_ms"] for o in out)
    print(f"STT p50/p90/max: {lat[len(lat) // 2]} / {lat[int(len(lat) * 0.9)]} / {lat[-1]} ms")

    for key in ("condition", "voice", "speed"):
        print(f"\nrecall END per {key}:")
        g = defaultdict(list)
        for o in end:
            g[o[key]].append(o["ended"])
        for k, v in sorted(g.items()):
            print(f"  {k:10s} {rate(sum(v), len(v))}")
    print("\nfalse end per kategori NO_END:")
    g = defaultdict(list)
    for o in neg:
        g[o["category"]].append(o["ended"])
    for k, v in sorted(g.items()):
        print(f"  {k:22s} {rate(sum(v), len(v))}")


if __name__ == "__main__":
    main()
