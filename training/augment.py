"""Parallel, resumable augmentation + feature extraction (drop-in for `livekit-wakeword augment`).

Upstream augments clips in a single process and extracts features with default ONNX Runtime
sessions, which oversubscribe small CPU boxes. Here: N worker processes with one ORT thread
each, and per-shard .npy outputs so an interrupted run resumes where it stopped. Typically
~30x faster on 4 cores.

Each worker reproduces upstream exactly for one clip:
  round 0: augment_clip -> apply_rir -> mix_with_background -> align (pos: end+jitter, neg: center)
  round r>0: same chain applied to the round r-1 output (compounding, as upstream)
and the features of EVERY round's output (upstream extracts all _rN.wav).
Only difference: WAVs of intermediate rounds are not written to disk (kept in memory),
features are identical in distribution. Optional --gain-db adds a final random gain
(uniform in [lo, hi] dB) + peak limiter at 0.99 FS, applied to the final audio of each round.

Usage: python augment.py <config.yaml> [--workers 4] [--gain-db -12 6]
Writes <model_dir>/{split}_features_{train|test}.npy exactly where `train` expects them.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

SPLITS = [
    ("positive_train", "positive_features_train.npy", True),
    ("positive_test", "positive_features_test.npy", True),
    ("negative_train", "negative_features_train.npy", False),
    ("negative_test", "negative_features_test.npy", False),
    ("background_train", "background_noise_features_train.npy", False),
    ("background_test", "background_noise_features_test.npy", False),
]
SR, TARGET = 16000, 32000

_W = {}


def _init(bg_dirs, rir_dirs, gain_db, seed):
    import onnxruntime as ort

    from livekit.wakeword.data.augment import AudioAugmentor
    from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
    from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    wid = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    random.seed(seed + wid)
    np.random.seed(seed + wid)
    _W["aug"] = AudioAugmentor(background_paths=[Path(p) for p in bg_dirs], rir_paths=[Path(p) for p in rir_dirs])
    _W["mel"] = MelSpectrogramFrontend(onnx_path=get_mel_model_path(), sess_options=so)
    _W["emb"] = SpeechEmbedding(onnx_path=get_embedding_model_path(), sess_options=so)
    _W["gain"] = gain_db


def _align(audio, is_pos):
    from livekit.wakeword.data.augment import align_clip_to_end

    if is_pos:
        return align_clip_to_end(audio, TARGET)
    if len(audio) < TARGET:
        out = np.zeros(TARGET, np.float32)
        s = (TARGET - len(audio)) // 2
        out[s:s + len(audio)] = audio
        return out
    s = (len(audio) - TARGET) // 2
    return audio[s:s + TARGET]


def _gain(audio):
    g = _W["gain"]
    if g is None:
        return audio
    audio = audio * (10 ** (random.uniform(g[0], g[1]) / 20))
    pk = np.abs(audio).max()
    return (audio * (0.99 / pk) if pk > 0.99 else audio).astype(np.float32)


def _feat(audio):
    from livekit.wakeword.data.features import _pad_or_truncate

    mel = _W["mel"](audio.astype(np.float32))
    return _pad_or_truncate(_W["emb"].extract_embeddings(mel)[0])


def _work(args):
    path, is_pos, rounds = args
    import soundfile as sf

    a, _ = sf.read(path)
    if a.ndim > 1:
        a = a[:, 0]
    a = a.astype(np.float32)
    aug = _W["aug"]
    feats = []
    for r in range(rounds):
        a = aug.mix_with_background(aug.apply_rir(aug.augment_clip(a)))
        if r == 0:
            a = _align(a, is_pos)
        feats.append(_feat(_gain(a)))
    return np.stack(feats)  # (rounds, 16, 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gain-db", type=float, nargs=2, default=None)
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    from livekit.wakeword.config import load_config

    cfg = load_config(a.config)
    md = cfg.model_output_dir
    rounds = cfg.augmentation.rounds
    src_re = re.compile(r"^clip_\d{6}\.wav$")
    pool = mp.get_context("fork").Pool(
        a.workers, initializer=_init,
        initargs=([str(p) for p in cfg.augmentation.background_paths],
                  [str(p) for p in cfg.augmentation.rir_paths], a.gain_db, a.seed))
    for split, out_name, is_pos in SPLITS:
        d = md / split
        if not d.exists():
            continue
        out = md / out_name
        if out.exists():
            print(f"[skip] {out_name} exists", flush=True)
            continue
        files = sorted(str(p) for p in d.glob("*.wav") if src_re.match(p.name))
        shard_dir = md / f".shards_{split}"
        shard_dir.mkdir(exist_ok=True)
        t0 = time.time()
        for si in range(0, len(files), a.shard):
            sp = shard_dir / f"{si:07d}.npy"
            if sp.exists():
                continue
            chunk = files[si:si + a.shard]
            res = pool.map(_work, [(f, is_pos, rounds) for f in chunk], chunksize=16)
            np.save(sp, np.stack(res).astype(np.float32))  # (n, rounds, 16, 96)
            done = si + len(chunk)
            rate = done / max(1e-6, time.time() - t0)
            print(f"{split}: {done}/{len(files)} clips  {rate:.1f} clips/s  eta {(len(files)-done)/max(rate,1e-6)/60:.0f} min", flush=True)
        parts = [np.load(p) for p in sorted(shard_dir.glob("*.npy"))]
        F = np.concatenate(parts)  # (n, rounds, 16, 96)
        # upstream order: all clips of r0, then r1, ... (each round extracted per clip)
        F = np.concatenate([F[:, r] for r in range(rounds)])
        np.save(out, F)
        del parts, F
        print(f"[done] {out_name}", flush=True)
    pool.close()


if __name__ == "__main__":
    sys.exit(main())
