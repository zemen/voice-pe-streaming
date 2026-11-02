# Retraining the wake word

Recipe that produced the bundled `sputnik.onnx`: the stock
[livekit-wakeword](https://github.com/livekit/livekit-wakeword) pipeline (**v0.2.3**; avoid
PyPI 0.3.0, its `augment` CLI is broken), plus your own labelled clips from the add-on's
**Samples** tab mixed into every training batch. Real-room clips are what make the difference:
the same recipe on synthetic speech alone barely transfers to real microphones.

Run everything from a directory holding `sputnik.yaml`, inside livekit-wakeword's
environment (`uv sync --extra train --extra export --extra eval`). A CUDA GPU is needed for
generation and training. Augmentation is CPU-only.

```sh
livekit-wakeword setup --data-dir ./data          # public features, TTS, RIRs, noise (~18 GB)
livekit-wakeword generate sputnik.yaml            # synthetic positives + confusables (GPU)
python augment.py sputnik.yaml --workers 4        # parallel, resumable augmentation + features

# Add-on web UI -> Samples -> label TP / FP -> download fine-tune dataset, unzip to ./samples
python real_clips.py ./samples ./output/sputnik --data-dir ./data --before 2025-01-01

python train.py sputnik.yaml --tag sputnik --size medium --real-pos 25 --real-neg 100 --steps 100000
livekit-wakeword eval sputnik.yaml               # synthetic validation; also test on held-out samples
```

- `--before`: keep your newest labelled samples out of training and use them to compare the
  new model with the current one. Only the newest samples are a fair test: older ones were
  collected under older models.
- `train.py` is the stock trainer with two extra batch classes (`real_positive`,
  `real_negative`). It checkpoints every 1000 steps and resumes when re-run with the same
  arguments. `--real-pos 0 --real-neg 0` reproduces stock training.
- Medium beat both small and large in our tests. 100k steps gave a model that works at the
  default 0.5 threshold.
- Copy `output/sputnik/sputnik.onnx` into `/share/wakewords/` (or replace the bundled model)
  and restart the add-on.
