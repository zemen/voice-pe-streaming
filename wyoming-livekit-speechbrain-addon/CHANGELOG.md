# Changelog

## 0.12.0

Initial public release.

- LiveKit wake word detection (Wyoming protocol, port 10400)
- SpeechBrain ECAPA speaker verification with cosine-similarity threshold
- Mid-sentence false-positive rejection (pre-wake-word ring buffer)
- Detection event stream over WebSocket (`/api/events`, port 10401) for the
  `voice_pe_speaker` Home Assistant integration
- Web UI for speaker enrollment, live pipeline debugging, and wake-word
  sample collection/labeling for fine-tuning
- Per-source sample retention quotas
- Single-threaded ONNX/torch configuration for low-power hosts
