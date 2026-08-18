"""Speaker enrollment helpers for SpeechBrain profiles."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    SpeakerRecognition = None


def _load_wav_audio(audio_path: Path) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width} for {audio_path}")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz WAV, got {sample_rate} Hz for {audio_path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio / 32768.0


def enroll_speaker(
    audio_paths: List[Path],
    output_profile: Path,
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
    savedir: Optional[Path] = None,
) -> None:
    """Compute mean ECAPA embedding from enrollment WAV files and save as .npy.

    SpeechBrain ECAPA-TDNN performs sentence-level mean normalization on log-mel
    features internally (recipe: ``InputNormalization(norm_type='sentence', std_norm=False)``),
    which analytically cancels constant amplitude gain in the log domain. The
    pretrained inference path consumes raw 16 kHz mono audio with no waveform
    normalization step. Pre-normalizing here would add clipping/noise-floor
    artifacts that break the model's designed gain invariance.
    """
    if not audio_paths:
        raise ValueError("At least one enrollment audio path is required")
    if SpeakerRecognition is None or torch is None:
        raise ImportError("speechbrain not installed")

    model = SpeakerRecognition.from_hparams(
        source=model_source,
        savedir=str(savedir) if savedir is not None else None,
    )

    embeddings = []
    for audio_path in audio_paths:
        audio = _load_wav_audio(audio_path)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        if hasattr(audio_tensor, "reshape"):
            audio_tensor = audio_tensor.reshape(1, -1)
        embedding = model.encode_batch(audio_tensor).squeeze(0).squeeze(0)
        if hasattr(embedding, "detach"):
            embedding = embedding.detach()
        if hasattr(embedding, "cpu"):
            embedding = embedding.cpu()
        if hasattr(embedding, "numpy"):
            embeddings.append(embedding.numpy())
        else:
            embeddings.append(np.asarray(embedding, dtype=np.float32))

    mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    output_profile.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_profile, mean_embedding)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="wyoming-livekit-speechbrain-enrollment")
    parser.add_argument("--name", type=str, required=True, help="Speaker name")
    parser.add_argument(
        "--audio",
        type=Path,
        nargs="+",
        required=True,
        help="Enrollment WAV file(s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npy profile path (default: <name>.profile.npy)",
    )
    parser.add_argument(
        "--speaker-model",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain model source",
    )
    parser.add_argument(
        "--speaker-savedir",
        type=Path,
        default=None,
        help="SpeechBrain model cache directory",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_profile = args.out or Path(f"{args.name}.profile.npy")
    enroll_speaker(
        audio_paths=args.audio,
        output_profile=output_profile,
        model_source=args.speaker_model,
        savedir=args.speaker_savedir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
