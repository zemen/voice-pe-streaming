"""Tests for SpeechBrain speaker enrollment helper."""

import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault("speechbrain", MagicMock())
sys.modules.setdefault("speechbrain.inference", MagicMock())
sys.modules.setdefault("speechbrain.inference.speaker", MagicMock())

from wyoming_livekit_speechbrain.enrollment import enroll_speaker  # noqa: E402


def _write_test_wav(path: Path, samples: int = 32000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.zeros(samples, dtype=np.int16)).tobytes())


class TestEnrollSpeaker:
    @patch("wyoming_livekit_speechbrain.enrollment.torch")
    @patch("wyoming_livekit_speechbrain.enrollment.SpeakerRecognition")
    def test_enroll_writes_npy(self, mock_sr, mock_torch, tmp_path):
        wav_a = tmp_path / "a.wav"
        _write_test_wav(wav_a)

        mock_model = MagicMock()
        mock_sr.from_hparams.return_value = mock_model
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.float32 = "float32"

        embedding = MagicMock()
        embedding.detach.return_value.cpu.return_value.numpy.return_value = (
            np.zeros(192, dtype=np.float32)
        )
        mock_model.encode_batch.return_value.squeeze.return_value.squeeze.return_value = embedding

        out = tmp_path / "alice.npy"
        enroll_speaker([wav_a], out)
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape == (192,)

    def test_empty_audio_list_raises(self, tmp_path):
        out = tmp_path / "x.npy"
        with pytest.raises(ValueError, match="(?i)at least one"):
            enroll_speaker([], out)

    @patch("wyoming_livekit_speechbrain.enrollment.SpeakerRecognition", None)
    @patch("wyoming_livekit_speechbrain.enrollment.torch", None)
    def test_missing_sdk_raises(self, tmp_path):
        wav_a = tmp_path / "a.wav"
        _write_test_wav(wav_a)
        with pytest.raises(ImportError, match="speechbrain not installed"):
            enroll_speaker([wav_a], tmp_path / "out.npy")
