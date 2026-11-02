"""Speaker enrollment helpers (CAM++ by default, SpeechBrain ECAPA optional)."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np

from .speaker_embedding import DEFAULT_SPEAKER_MODEL, SpeakerEmbedder, save_profile


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
    model_source: str = DEFAULT_SPEAKER_MODEL,
    savedir: Optional[Path] = None,
) -> None:
    """Average the (L2-normalised) speaker embeddings of the enrollment WAVs and save as .npy.

    Raw 16 kHz mono audio goes in unnormalised: both CAM++ (per-utterance fbank mean removal)
    and ECAPA (sentence-level mean normalisation) cancel constant gain in the log domain.
    The backend name is stored next to the profile so a verifier never mixes models.
    """
    if not audio_paths:
        raise ValueError("At least one enrollment audio path is required")

    embedder = SpeakerEmbedder(model_source, savedir)
    embeddings = [embedder.embed(_load_wav_audio(p)) for p in audio_paths]
    mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    save_profile(output_profile, mean_embedding, model_source)


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
        default=DEFAULT_SPEAKER_MODEL,
        help="Speaker embedding model: wespeaker/campplus (default) or a SpeechBrain source",
    )
    parser.add_argument(
        "--speaker-savedir",
        type=Path,
        default=None,
        help="Speaker model cache directory",
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
