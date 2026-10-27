#!/usr/bin/env bash
# Ganti suara TTS brone-talk lalu restart servisnya:
#   bash deploy/tts_voice.sh baru   # default: Piper fine-tune BRONE D21 + normalisasi teks training-nya (textnorm)
#   bash deploy/tts_voice.sh lama   # id_ID-news_tts-medium + clean_for_speech (seperti sebelum 26 Okt 2026)
# "lama" lewat drop-in systemd; "baru" = drop-in dihapus (default config.py).
set -euo pipefail
D="$HOME/.config/systemd/user/brone-talk.service.d"
F="$D/suara.conf"
case "${1:-}" in
  baru) rm -f "$F" ;;
  lama)
    mkdir -p "$D"
    cat > "$F" <<'EOF'
[Service]
Environment=BRONE_PIPER_VOICE=%h/brone_talk/models/voices/id_ID-news_tts-medium.onnx
Environment=BRONE_PIPER_CONFIG=%h/brone_talk/models/voices/id_ID-news_tts-medium.onnx.json
Environment=BRONE_TTS_TEXTNORM=
EOF
    ;;
  *) echo "pakai: bash deploy/tts_voice.sh baru|lama"; exit 1 ;;
esac
systemctl --user daemon-reload
systemctl --user restart brone-talk
sleep 3
systemctl --user --no-pager status brone-talk | head -n 5
if [ -f "$F" ]; then echo "suara: lama (news_tts)"; else echo "suara: baru (D21, default)"; fi
