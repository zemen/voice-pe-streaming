"""Speaker embeddings for verification: WeSpeaker CAM++ (ONNX) by default, SpeechBrain ECAPA optional.

CAM++ (Wespeaker/wespeaker-voxceleb-campplus, Apache-2.0) runs on onnxruntime only and
separates enrolled humans from pet/bird vocalisations and other voices better than ECAPA.
Profiles are model-specific: a profile stores which backend produced it, and profiles from
another backend are ignored (re-enroll after switching).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

CAMPP = "wespeaker/campplus"
ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SPEAKER_MODEL = CAMPP

_CAMPP_REPO = "Wespeaker/wespeaker-voxceleb-campplus"
_CAMPP_FILE = "voxceleb_CAM++.onnx"


def _kaldi_fbank(audio: np.ndarray) -> np.ndarray:
    """80-dim Kaldi fbank (25 ms / 10 ms, hamming, no dither) with per-utterance mean removal."""
    import torch
    import torchaudio.compliance.kaldi as kaldi

    wav = torch.from_numpy(audio.astype(np.float32) * (1 << 15)).unsqueeze(0)
    feats = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
    )
    feats = feats - feats.mean(dim=0, keepdim=True)
    return feats.numpy()


class SpeakerEmbedder:
    """Maps 16 kHz mono float audio in [-1, 1] to an L2-normalised speaker embedding."""

    def __init__(self, model_source: str = DEFAULT_SPEAKER_MODEL, savedir: Optional[Path] = None):
        self.model_source = model_source
        self._savedir = savedir
        if model_source == CAMPP:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                _CAMPP_REPO, _CAMPP_FILE, cache_dir=str(savedir) if savedir is not None else None
            )
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                path, sess_options=so, providers=["CPUExecutionProvider"]
            )
            self._input = self._session.get_inputs()[0].name
        else:
            from speechbrain.inference.speaker import SpeakerRecognition

            self._session = SpeakerRecognition.from_hparams(
                source=model_source, savedir=str(savedir) if savedir is not None else None
            )

    def embed(self, audio: np.ndarray) -> np.ndarray:
        if self.model_source == CAMPP:
            feats = _kaldi_fbank(audio)[None]
            emb = self._session.run(None, {self._input: feats})[0][0]
        else:
            import torch

            emb = (
                self._session.encode_batch(torch.from_numpy(audio.astype(np.float32)).reshape(1, -1))
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-9)


def save_profile(path: Path, embedding: np.ndarray, model_source: str) -> None:
    """Save as .npy with the backend tag stored alongside (``<name>.npy`` + ``<name>.model``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding.astype(np.float32))
    path.with_suffix(".model").write_text(model_source)


def profile_model(path: Path) -> str:
    """Backend that produced a profile; untagged legacy profiles are ECAPA."""
    tag = path.with_suffix(".model")
    return tag.read_text().strip() if tag.exists() else ECAPA
