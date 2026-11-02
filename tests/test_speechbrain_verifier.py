"""Tests for speaker verification (CAM++ default backend, model-tagged profiles)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wyoming_livekit_speechbrain.speaker_embedding import CAMPP, ECAPA, save_profile
from wyoming_livekit_speechbrain.speechbrain_verifier import (
    SpeechBrainConfig,
    SpeechBrainVerifier,
)

EMB = "wyoming_livekit_speechbrain.speechbrain_verifier.SpeakerEmbedder"


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _profile(path: Path, vec, model=CAMPP) -> Path:
    save_profile(path, _unit(vec), model)
    return path


class TestSpeechBrainConfig:
    def test_default_threshold(self):
        assert SpeechBrainConfig(profile_paths=[Path("a.npy")]).threshold == 0.45

    def test_custom_threshold(self):
        assert SpeechBrainConfig(profile_paths=[], threshold=0.4).threshold == 0.4

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="Threshold must be between"):
            SpeechBrainConfig(profile_paths=[], threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="Threshold must be between"):
            SpeechBrainConfig(profile_paths=[], threshold=1.5)

    def test_string_paths_converted(self):
        c = SpeechBrainConfig(profile_paths=["x.npy", "y.npy"])
        assert all(isinstance(p, Path) for p in c.profile_paths)

    def test_default_model_source_is_campp(self):
        assert SpeechBrainConfig(profile_paths=[]).model_source == CAMPP


@patch(EMB)
class TestSpeechBrainVerifier:
    def test_creates_with_profile(self, mock_emb, tmp_path):
        p = _profile(tmp_path / "alice.npy", np.ones(512))
        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[p]))
        mock_emb.assert_called_once_with(CAMPP, None)
        assert "alice" in v._embeddings

    def test_missing_profile_file_raises(self, mock_emb):
        with pytest.raises(FileNotFoundError, match="Profile file not found"):
            SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[Path("/no/such/file.npy")]))

    def test_speaker_name_extraction(self, mock_emb, tmp_path):
        paths = [_profile(tmp_path / f, np.random.rand(512)) for f in ("bob.npy", "carol.profile.npy")]
        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=paths))
        assert set(v._embeddings) == {"bob", "carol"}

    def test_profile_from_other_model_is_skipped(self, mock_emb, tmp_path):
        legacy = tmp_path / "old.npy"
        np.save(legacy, np.random.rand(192).astype(np.float32))  # untagged = ECAPA
        tagged = _profile(tmp_path / "new.npy", np.random.rand(512))
        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[legacy, tagged]))
        assert set(v._embeddings) == {"new"}

    def test_ecapa_backend_accepts_legacy_profiles(self, mock_emb, tmp_path):
        legacy = tmp_path / "old.npy"
        np.save(legacy, np.random.rand(192).astype(np.float32))
        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[legacy], model_source=ECAPA))
        assert set(v._embeddings) == {"old"}

    def test_scores_are_cosine_and_threshold_applies(self, mock_emb, tmp_path):
        a = _profile(tmp_path / "alice.npy", [1, 0, 0])
        b = _profile(tmp_path / "bob.npy", [0, 1, 0])
        mock_emb.return_value.embed.return_value = _unit([1, 1, 0])
        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[a, b], threshold=0.6))
        scores = v.verify_with_scores(b"\x01\x00" * 8000)
        assert scores == pytest.approx({"alice": 0.7071, "bob": 0.7071}, abs=1e-3)
        assert v.verify_speaker(b"\x01\x00" * 8000) in {"alice", "bob"}
        v.config.threshold = 0.8
        assert v.verify_speaker(b"\x01\x00" * 8000) is None

    def test_verify_speaker_too_short_returns_none(self, mock_emb, tmp_path):
        p = _profile(tmp_path / "alice.npy", np.ones(512))
        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[p]))
        assert v.verify_speaker(b"\x00\x00" * 100) is None

    def test_verify_speaker_no_profiles_returns_none(self, mock_emb):
        v = SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[]))
        assert v.verify_speaker(b"\x00\x00" * 8000) is None

    def test_reset_is_noop(self, mock_emb):
        assert SpeechBrainVerifier(SpeechBrainConfig(profile_paths=[])).reset() is None
