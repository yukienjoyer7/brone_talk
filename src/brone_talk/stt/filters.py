"""Filter teks STT yang dibuang sebelum wake word/LLM. Dipindah dari brone_talk.py (v4)."""

import re

from brone_talk.text import stt_tokens


_STT_HALLUCINATIONS = (
    "terima kasih kerana menonton", "terima kasih karena menonton", "terima kasih telah menonton",
    "terima kasih sudah menonton", "terima kasih telah mendengarkan", "jangan lupa subscribe",
    "jangan lupa like", "sampai jumpa di video", "thank you for watching", "thanks for watching",
    "subtitle by", "subtitles by", "amara.org", "go to the intro",
)

# 20 Nov 2026: ucapan yang HANYA berisi "terima kasih" (boleh + nama Bron) lebih sering derau atau salah dengar
# STT daripada ucapan sungguhan, jadi dibuang. "Terima kasih, selesai brone" / "terima kasih infonya" tetap lolos.
_THANKS_ONLY = re.compile(
    r"^[\W_]*(?:(?:brone|bron|brown|brian)[\W_]*)?(?:(?:terima\s*kasih|terimakasih|makasih|trims|thank\s*you|thanks)(?:\s+(?:banyak|ya|yah))?[\W_]*"
    r"(?:(?:brone|bron|brown|brian)[\W_]*)?)+$",
    re.IGNORECASE,
)

_STT_WAKE_STOP = ("halo", "hallo", "hai", "hey", "hello", "hi", "selesai", "diam", "stop")

_PROMPT_SENTENCE_ECHO = re.compile(
    r"(?:\b(?:robot\s+)?maskot\s+)?(?:\buniversitas\s+brawijaya[\s,]+)?\bub[\s,]+di\s+kota\s+malang\b[.,;!?]*"
    r"|(?:\brobot\s+)?\bmaskot\s+universitas\s+brawijaya[\s,]+ub\b[.,;!?]*",
    re.IGNORECASE,
)

def _stt_chunk_norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def stt_prompt_echo(text, terms):
    """13 Okt 2026: Whisper menyalin daftar istilah prompt saat hanya mendengar derau.

    Karena daftar itu memuat "BRONE", gemanya dulu dianggap panggilan (18 dari 31 panggilan
    di journal 7-13 Okt). Gema = >=3 istilah UNIK yang masing-masing berdiri sendiri di antara
    koma, dan istilah itu >=50% potongan teks atau >=3 berturut-turut. Nama robot tidak dihitung,
    jadi "BRONE, apa itu UKT, SNBP?" tetap lolos. Diuji pada 3.704 STT nyata: 24/24 gema-panggilan
    tertangkap, 0 panggilan asli ikut terbuang.
    """
    chunks = [_stt_chunk_norm(c) for c in re.split(r"[,.;!?]+", text)]
    chunks = [c for c in chunks if c and c not in ("brone", "bron")]
    hits = [c in terms for c in chunks]
    if len({c for c, h in zip(chunks, hits) if h}) < 3:
        return False
    run = best = 0
    for h in hits:
        run = run + 1 if h else 0
        best = max(best, run)
    return sum(hits) / len(chunks) >= 0.5 or best >= 3

def stt_garbage_reason(text, stats=None, vocab=frozenset(), terms=frozenset()):  # noqa: C901 (aturan filter berurutan)
    """Alasan teks STT harus dibuang sebelum wake-word/LLM, atau "" kalau layak diproses."""
    raw = text.strip()
    low = raw.lower()
    toks = stt_tokens(raw)
    if not toks:
        return "tanpa kata"
    if any(p in low for p in _STT_HALLUCINATIONS):
        return "halusinasi whisper"
    if _THANKS_ONLY.match(raw):
        return "hanya terima kasih"
    if re.search(r"[぀-ヿ㐀-鿿가-힯]", raw):
        return "karakter asing"
    if re.search(r"([a-z])\1{4,}", low) or re.search(r"([a-z]{2,4})\1{3,}", low):
        return "huruf berulang"
    if stats:
        if stats.get("max_cr", 0) > 2.4:
            return f"berulang (compression {stats['max_cr']:.1f})"
        if stats.get("mean_lp", 0) < -1.5:
            return f"tidak yakin (logprob {stats['mean_lp']:.2f})"
    if re.search(r"[—–-]\s*\d+\s*fakt", low):
        return "gema judul prompt"
    if terms and stt_prompt_echo(raw, terms):
        return "gema daftar istilah prompt"
    short_command = len(toks) <= 3 and any(t in _STT_WAKE_STOP for t in toks)
    if short_command:
        return ""
    if len(toks) == 1 and len(toks[0]) <= 4:
        return "satu kata"
    if len(toks) >= 4 and len(set(toks)) / len(toks) <= 0.5:
        return "frasa berulang"
    if len(toks) >= 8 and max(toks.count(t) for t in set(toks)) / len(toks) >= 0.35:
        return "kata berulang"
    grams = [" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)]
    if grams and max(grams.count(g) for g in grams) >= 3:
        return "frasa berulang"
    if vocab and any(grams.count(g) >= 2 and all(w in vocab for w in g.split()) for g in grams):
        return "gema kosakata prompt"
    if vocab and len(toks) >= 3:
        in_vocab = sum(t in vocab for t in toks) / len(toks)
        letters = [c for c in raw if c.isalpha()]
        upper = sum(c.isupper() for c in letters) / max(1, len(letters))
        if in_vocab >= 0.75 and upper >= 0.6:
            return "gema kosakata prompt"
    return ""

def stt_strip_prompt_sentence(text):
    """Teks tanpa gema kalimat pembuka prompt Whisper; "" bila isinya hanya gema.

    Dipanggil SETELAH stt_garbage_reason pada teks asli: membuang gema lebih dulu membuat 8 baris sampah
    yang tadinya tertangkap filter lain (mis. 'UB di Kota Malang, PIMNAS, FMIPA.') ikut lolos.
    """
    if not _PROMPT_SENTENCE_ECHO.search(text):
        return text
    rest = re.sub(r"\s{2,}", " ", _PROMPT_SENTENCE_ECHO.sub(" ", text)).strip(" ,.;")
    return rest if stt_tokens(rest) else ""


def stt_vocab(*texts):
    return {t for text in texts for t in stt_tokens(text or "") if len(t) > 1}


def stt_prompt_terms(prompt, hotwords):
    """Istilah daftar prompt/hotwords Whisper (per koma), tanpa nama robot."""
    terms = set()
    for part in re.split(r"[,.;:]", f"{prompt or ''},{hotwords or ''}"):
        norm = _stt_chunk_norm(part)
        if norm and len(norm.split()) <= 8:
            terms.add(norm)
    return frozenset(terms - {"brone", "bron", "istilah"})
