"""Tests for the speaker enrollment helper."""

import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from wyoming_livekit_speechbrain.enrollment import enroll_speaker
from wyoming_livekit_speechbrain.speaker_embedding import CAMPP, profile_model


def _write_test_wav(path: Path, samples: int = 32000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.zeros(samples, dtype=np.int16)).tobytes())


class TestEnrollSpeaker:
    @patch("wyoming_livekit_speechbrain.enrollment.SpeakerEmbedder")
    def test_enroll_writes_tagged_npy(self, mock_emb, tmp_path):
        wavs = []
        for n in ("a.wav", "b.wav"):
            _write_test_wav(tmp_path / n)
            wavs.append(tmp_path / n)
        mock_emb.return_value.embed.side_effect = [np.ones(512, np.float32), np.zeros(512, np.float32)]

        out = tmp_path / "alice.npy"
        enroll_speaker(wavs, out)
        loaded = np.load(out)
        assert loaded.shape == (512,)
        assert np.allclose(loaded, 0.5)
        assert profile_model(out) == CAMPP
        mock_emb.assert_called_once_with(CAMPP, None)

    def test_empty_audio_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="(?i)at least one"):
            enroll_speaker([], tmp_path / "x.npy")
