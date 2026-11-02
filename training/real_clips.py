"""Turn the add-on's labelled samples into extra training classes for train.py.

Input: the add-on's "Download fine-tune dataset" zip, unpacked (tp/ and fp/ folders of
2 s WAV + JSON). Every clip is embedded with livekit's frontend, plus K augmented copies
(livekit augmentation chain + random gain in [-12, +6] dB, 0.99 peak limiter). The clips
already end at the trigger, so no alignment is applied.

Keep a recent slice of your samples out of training (--before) so you can evaluate on it.

Outputs in <run_dir>:  real_pos_train.npy, real_neg_train.npy   (n, 16, 96)
Usage: python real_clips.py <dataset_dir> <run_dir> --data-dir <livekit data dir>
           [--before YYYY-MM-DD] [--k-pos 30] [--k-neg 3]
"""
import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

_W = {}


def _init(seed, data_dir):
    import onnxruntime as ort

    from livekit.wakeword.data.augment import AudioAugmentor
    from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
    from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path

    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    wid = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    random.seed(seed + wid); np.random.seed(seed + wid)
    d = Path(data_dir)
    _W["aug"] = AudioAugmentor([d / "backgrounds"], [d / "rirs"])
    _W["mel"] = MelSpectrogramFrontend(get_mel_model_path(), so)
    _W["emb"] = SpeechEmbedding(get_embedding_model_path(), so)


def _feat(a):
    from livekit.wakeword.data.features import _pad_or_truncate
    return _pad_or_truncate(_W["emb"].extract_embeddings(_W["mel"](a.astype(np.float32)))[0])


def _work(args):
    path, k = args
    import soundfile as sf
    a, _ = sf.read(path, dtype="float32")
    out = [_feat(a)]
    aug = _W["aug"]
    for _ in range(k):
        b = aug.mix_with_background(aug.apply_rir(aug.augment_clip(a.copy())), snr_db_range=(5.0, 20.0))
        b = b * (10 ** (random.uniform(-12, 6) / 20))
        pk = np.abs(b).max()
        if pk > 0.99:
            b = b * (0.99 / pk)
        out.append(_feat(b.astype(np.float32)))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("run_dir")
    ap.add_argument("--data-dir", required=True, help="livekit-wakeword data dir (backgrounds/, rirs/)")
    ap.add_argument("--before", default=None, help="only use clips captured before this date (YYYY-MM-DD, UTC)")
    ap.add_argument("--k-pos", type=int, default=30)
    ap.add_argument("--k-neg", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import datetime as dt
    cut = (dt.datetime.fromisoformat(a.before).replace(tzinfo=dt.timezone.utc).timestamp()
           if a.before else float("inf"))
    ds = Path(a.dataset_dir)

    def pick(label):
        out = []
        for js in sorted((ds / label).glob("*.json")):
            m = json.load(open(js))
            if m.get("timestamp", 0) < cut and js.with_suffix(".wav").exists():
                out.append(str(js.with_suffix(".wav")))
        return out

    pos, neg = pick("tp"), pick("fp")
    print(f"training clips: {len(pos)} positive, {len(neg)} negative", flush=True)
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(77, a.data_dir)) as pool:
        P = np.concatenate(pool.map(_work, [(p, a.k_pos) for p in pos], chunksize=4))
        N = np.concatenate(pool.map(_work, [(p, a.k_neg) for p in neg], chunksize=16))
    out = Path(a.run_dir)
    np.save(out / "real_pos_train.npy", P.astype(np.float32))
    np.save(out / "real_neg_train.npy", N.astype(np.float32))
    print("saved", P.shape, N.shape)


if __name__ == "__main__":
    main()
