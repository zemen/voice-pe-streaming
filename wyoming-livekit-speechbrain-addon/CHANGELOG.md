# Changelog

## 0.12.1

- Bundled `sputnik.onnx`: all training metadata (torch export stack traces,
  embedded build-environment paths) fully stripped. Weights are byte-identical
  to 0.12.0 — verified by content hash and inference equivalence — the file is
  now ~873 KB and contains no toolchain provenance strings.
- `sputnik_eval.json` replaced with the model's own training-run evaluation
  (33.38 h validation, 24k positive / 60k negative clips): recall 0.9331,
  FPPH 0.0899, accuracy 0.9665, eval-derived optimal threshold **0.5**.
- Default `sensitivity` aligned to that optimal threshold: 0.37 → 0.5.
  Existing installs keep their configured value; tune to your environment.

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
