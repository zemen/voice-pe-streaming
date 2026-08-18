# Wyoming LiveKit + SpeechBrain

Streaming wake word detection via [livekit-wakeword](https://github.com/livekit/livekit-wakeword) plus offline speaker verification via [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), exposed over the Wyoming protocol so Home Assistant Voice can use it as a wake word service.

Fully offline once the SpeechBrain model is cached on first run. No access keys, no per-device quota.

## First-time setup

The add-on bundles the **sputnik** wake word model. On first start, if `/share/wakewords/` contains no models, the bundled `sputnik.onnx` is seeded there automatically. The default sensitivity `0.5` is the model's eval-derived optimal threshold (see `sputnik_eval.json`).

1. Start the add-on. Watch the log for `Wake words: sputnik`.
2. In Home Assistant: **Settings -> Devices & Services -> Add Integration -> Wyoming Protocol** -> Host: `localhost` (same host) or `homeassistant.local`, Port: `10400`.
3. **Settings -> Voice Assistants -> your assistant -> Wake word** dropdown -> select `sputnik`.

To use different ONNX wake-word classifiers, drop the files into `/share/wakewords/` — every `*.onnx` there is loaded on start, with wake-word names derived from filename stems. Compatible with both livekit-wakeword and openWakeWord ONNX classifiers (input `(1, 16, 96)` -> output `(1, 1)`).

## Speaker verification (optional)

1. Enroll each speaker by computing an ECAPA embedding from one or more 16 kHz mono 16-bit WAV samples. From a host with `livekit-wakeword`'s package installed:
   ```bash
   python -m wyoming_livekit_speechbrain.enrollment \
     --name alice \
     --audio /path/to/alice_1.wav /path/to/alice_2.wav \
     --out /share/profiles/alice.npy
   ```
2. In the add-on config:
   ```yaml
   enable_speaker_verification: true
   speaker_profiles:
     - /share/profiles/alice.npy
   speaker_threshold: 0.25   # cosine similarity, raise for stricter match
   ```
3. Restart. Detection events now carry a `speaker` field with the matched profile name (`alice`).

## Mid-sentence rejection

When speaker verification is enabled, the server compares the speaker before and during the wake word. If it's the same speaker, the detection is rejected as a mid-sentence false positive. Disable via `enable_mid_sentence_detection: false`.

## Options reference

| Key | Default | Description |
|---|---|---|
| `wake_word_models` | `[/share/wakewords/sputnik.onnx]` | Required. Paths to ONNX wake-word classifiers (under `/share/`). |
| `wake_words` | `[sputnik]` | Optional labels (defaults to model file stem if omitted). Pairs 1:1 with `wake_word_models`. |
| `sensitivity` | `0.5` | Detection threshold 0.0-1.0, applied to every wake-word model. Default matches the bundled model's eval-derived optimal threshold. Same value applies to all wake words; for per-model sensitivity, invoke the addon manually with repeated `--sensitivity` flags. |
| `save_sample_threshold` | `0.25` | Second, lower threshold (0.0-1.0 or omit/null to disable). When the classifier scores in `[save_sample_threshold, sensitivity)`, the 2 s window is persisted to the sample store with `source="auto-below-threshold"` WITHOUT firing a detection. Mines hard negatives + quiet TPs near the decision boundary for fine-tuning. Lower = more captures + more storage + more manual labeling. Default 0.25 is conservative on a trained sputnik model. |
| `predict_interval_seconds` | `0.5` | Minimum seconds between `WakeWordModel.predict()` calls. Lower = lower detection latency, higher CPU. `0.0` disables throttling (predict on every 80 ms frame ≈ 12 calls/sec). |
| `enable_speaker_verification` | `false` | Toggle SpeechBrain ECAPA verification. |
| `speaker_profiles` | `[]` | Paths to `.npy` ECAPA embeddings under `/share/`. |
| `speaker_threshold` | `0.25` | Cosine similarity threshold. |
| `speaker_model` | `speechbrain/spkrec-ecapa-voxceleb` | HuggingFace model id. |
| `enable_mid_sentence_detection` | `true` | Reject WW when same speaker before AND during. |
| `debug_logging` | `false` | `--debug` flag for the Wyoming server. |

## First boot caches the SpeechBrain model

The first start downloads `speechbrain/spkrec-ecapa-voxceleb` (~17 MB) into `/data/speechbrain/`. This persists across container restarts. Subsequent starts are fully offline.

## Web UI

The add-on exposes a web UI via HA Ingress. Click "Wake Word" in the HA sidebar (icon: account-voice) to open it. Three tabs:

- **Enroll** — Record speaker samples from your browser mic or from a selected Voice PE device, compute an ECAPA embedding, save as a profile under `/share/profiles/`. List, rename, re-enroll, delete existing profiles. Verify a fresh recording against enrolled speakers and see per-speaker cosine similarity scores.
- **Debug** — Open two WebSocket streams to the add-on: one renders a live wake-word score chart, the other panels speaker verification + mid-sentence rejection + a trigger log. Useful to watch the pipeline run in real time and understand why detections fire or don't.
- **Samples** — Every wake-word trigger is automatically captured as a 2-second WAV. Browse them with inline playback, mark each as False Positive or True Positive, retest against the current detector ("would this still trigger after I lowered the threshold?"), or download all FPs as a ZIP. Also has a "Capture from Voice PE" panel that records arbitrary audio from a selected active Voice PE session for fine-tuning data collection.

## Fine-tuning the wake word with collected samples

The Samples tab is your data-collection UI. The retraining cycle:

1. **Collect** — use the device normally. The addon auto-captures every wake-word fire (TP + FP) as a 2 s WAV.
2. **Label** — Samples tab → for each row, mark TP / FP / leave Unlabeled. Bulk-select with shift-click is supported. Unlabeled samples are excluded from the export.
3. **Download** — "Download fine-tune dataset" button at the top of the Samples tab. Produces `wakeword_fine_tune_dataset.zip` with `tp/<id>.wav` and `fp/<id>.wav` directories.
4. **Import into your livekit-wakeword workspace**. The ZIP layout doesn't drop in directly — livekit-wakeword expects `^clip_\d{6}\.wav$` filenames in `output/<model>/{positive,negative}_{train,test}/`. Rename the WAVs to the next free clip indices (`tp/` → `positive_*`, `fp/` → `negative_*`) with an ~80/20 train/test split. The ZIP's `README.md` documents the expected layout.

5. **Retrain** from the workspace root:

   ```bash
   cd /path/to/wakeword_workspace
   livekit-wakeword augment configs/prod.yaml    # picks up the new clips
   livekit-wakeword train configs/prod.yaml       # retrains classifier head
   livekit-wakeword export configs/prod.yaml      # → output/sputnik/sputnik.onnx
   livekit-wakeword eval configs/prod.yaml        # check new FPPH + recall
   ```

   The bundled model is trained on raw audio (no full-window RMS normalization in training or inference), with gain augmentation (`Gain(min_gain_db=-30.0, max_gain_db=3.0, p=1.0)` plus a peak limiter) covering the empirical Voice PE active-RMS range of -44 to -14 dBFS. Training on raw audio preserves the natural amplitude distribution the model sees in production.

6. **Deploy**: copy the new `output/sputnik/sputnik.onnx` to `/share/wakewords/` on your HA host (Samba/SSH add-on, or the File editor), then restart the add-on.

7. **Verify** the new `output/<model>/<model>_eval.json` shows lower FPPH and same-or-higher recall — the classifier has now learned that your specific captured FP patterns should NOT trigger.

This is **not in-place fine-tuning** — livekit-wakeword retrains the classifier head from scratch each time. The pre-trained mel + embedding models are frozen, so retraining is fast (minutes to a couple hours depending on `steps` and dataset size).

## Sample retention

The add-on keeps the newest 1000 samples by default (`--max-samples`). Each sample is ~64 KB so the cap is ~64 MB on disk. Older samples are evicted automatically when new ones arrive. Adjust via the CLI flag if you want more history.
