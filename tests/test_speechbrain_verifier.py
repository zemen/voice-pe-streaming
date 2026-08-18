"""Tests for SpeechBrain speaker verification wrapper."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub heavy SDKs so import works in CI without them
sys.modules.setdefault("speechbrain", MagicMock())
sys.modules.setdefault("speechbrain.inference", MagicMock())
sys.modules.setdefault("speechbrain.inference.speaker", MagicMock())

from wyoming_livekit_speechbrain.speechbrain_verifier import (  # noqa: E402
    SpeechBrainConfig,
    SpeechBrainVerifier,
)


class TestSpeechBrainConfig:
    def test_default_threshold(self):
        c = SpeechBrainConfig(profile_paths=[Path("a.npy")])
        assert c.threshold == 0.25

    def test_custom_threshold(self):
        c = SpeechBrainConfig(profile_paths=[], threshold=0.4)
        assert c.threshold == 0.4

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="Threshold must be between"):
            SpeechBrainConfig(profile_paths=[], threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="Threshold must be between"):
            SpeechBrainConfig(profile_paths=[], threshold=1.5)

    def test_string_paths_converted(self):
        c = SpeechBrainConfig(profile_paths=["x.npy", "y.npy"])
        assert all(isinstance(p, Path) for p in c.profile_paths)

    def test_default_model_source(self):
        c = SpeechBrainConfig(profile_paths=[])
        assert c.model_source == "speechbrain/spkrec-ecapa-voxceleb"


class TestSpeechBrainVerifier:
    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.torch")
    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.SpeakerRecognition")
    def test_creates_with_profile(self, mock_sr, mock_torch, tmp_path):
        profile_path = tmp_path / "alice.npy"
        np.save(profile_path, np.random.rand(192).astype(np.float32))

        mock_model = MagicMock()
        mock_sr.from_hparams.return_value = mock_model
        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.float32 = "float32"

        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[profile_path]))
        mock_sr.from_hparams.assert_called_once()
        assert "alice" in v._embeddings

    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.SpeakerRecognition", None)
    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.torch", None)
    def test_missing_sdk_raises(self):
        with pytest.raises(ImportError, match="speechbrain not installed"):
            SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[]))

    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.torch")
    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.SpeakerRecognition")
    def test_missing_profile_file_raises(self, mock_sr, mock_torch):
        mock_sr.from_hparams.return_value = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.float32 = "float32"

        with pytest.raises(FileNotFoundError, match="Profile file not found"):
            SpeechBrainVerifier(
                SpeechBrainConfig(profile_paths=[Path("/no/such/file.npy")])
            )

    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.torch")
    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.SpeakerRecognition")
    def test_speaker_name_extraction(self, mock_sr, mock_torch, tmp_path):
        for filename in ["bob.npy", "carol.profile.npy"]:
            np.save(tmp_path / filename, np.random.rand(192).astype(np.float32))

        mock_sr.from_hparams.return_value = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.float32 = "float32"

        v = SpeechBrainVerifier(
            SpeechBrainConfig(
                profile_paths=[tmp_path / "bob.npy", tmp_path / "carol.profile.npy"]
            )
        )
        assert set(v._embeddings.keys()) == {"bob", "carol"}

    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.torch")
    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.SpeakerRecognition")
    def test_verify_speaker_too_short_returns_none(self, mock_sr, mock_torch, tmp_path):
        profile_path = tmp_path / "alice.npy"
        np.save(profile_path, np.random.rand(192).astype(np.float32))

        mock_sr.from_hparams.return_value = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.float32 = "float32"

        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[profile_path]))
        # < 4000 samples = < 0.25s = too short
        result = v.verify_speaker(b"\x00\x00" * 100)
        assert result is None

    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.torch")
    @patch("wyoming_livekit_speechbrain.speechbrain_verifier.SpeakerRecognition")
    def test_verify_speaker_no_profiles_returns_none(self, mock_sr, mock_torch):
        mock_sr.from_hparams.return_value = MagicMock()
        mock_torch.tensor.return_value = MagicMock()
        mock_torch.float32 = "float32"
        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[]))
        # Below threshold - empty embeddings short-circuit returns None
        result = v.verify_speaker(b"\x00\x00" * 8000)
        assert result is None

    def test_reset_is_noop(self):
        # Reset doesn't need SDK; just verify it doesn't raise
        with patch("wyoming_livekit_speechbrain.speechbrain_verifier.torch", MagicMock()), \
             patch("wyoming_livekit_speechbrain.speechbrain_verifier.SpeakerRecognition", MagicMock()):
            v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[]))
            assert v.reset() is None
