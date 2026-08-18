"""SpeechBrain speaker verification integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    SpeakerRecognition = None

_LOGGER = logging.getLogger(__name__)


def _extract_speaker_name(profile_path: Path) -> str:
    """Strip every suffix so ``alice.profile.npy`` and ``alice.npy`` both map to ``alice``."""
    name = profile_path.name
    while True:
        stem, ext = name.rsplit(".", 1) if "." in name else (name, "")
        if not ext:
            return name
        name = stem


@dataclass
class SpeechBrainConfig:
    """Configuration for SpeechBrain speaker verification."""

    profile_paths: List[Path] = field(default_factory=list)
    threshold: float = 0.25
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: Optional[Path] = None

    def __post_init__(self):
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")

        self.profile_paths = [Path(path) if not isinstance(path, Path) else path for path in self.profile_paths]
        if self.savedir is not None and not isinstance(self.savedir, Path):
            self.savedir = Path(self.savedir)


@dataclass
class SpeechBrainVerifier:
    """SpeechBrain verifier using ECAPA embeddings + cosine similarity."""

    config: SpeechBrainConfig
    _model: Optional[object] = field(default=None, init=False, repr=False)
    _embeddings: Dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if SpeakerRecognition is None or torch is None:
            raise ImportError("speechbrain not installed")

        savedir = str(self.config.savedir) if self.config.savedir is not None else None
        self._model = SpeakerRecognition.from_hparams(
            source=self.config.model_source,
            savedir=savedir,
        )

        for profile_path in self.config.profile_paths:
            if not profile_path.exists():
                raise FileNotFoundError(f"Profile file not found: {profile_path}")

            embedding = np.load(profile_path)
            profile_tensor = torch.tensor(embedding, dtype=torch.float32)
            if hasattr(profile_tensor, "reshape"):
                profile_tensor = profile_tensor.reshape(1, -1)
            speaker_name = _extract_speaker_name(profile_path)
            self._embeddings[speaker_name] = profile_tensor

    def verify_speaker(self, audio_bytes: bytes) -> Optional[str]:
        """Verify speaker from audio and return profile name if authorized."""
        scored = self.verify_with_scores(audio_bytes)
        if not scored:
            return None
        best_name, best_score = max(scored.items(), key=lambda kv: kv[1])
        if best_score < self.config.threshold:
            return None
        return best_name

    def verify_with_scores(self, audio_bytes: bytes) -> Dict[str, float]:
        """Return cosine similarity per enrolled speaker. Empty if model unavailable."""
        if self._model is None or not self._embeddings:
            return {}

        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if samples.size < 4000:
            return {}

        normalized = samples.astype(np.float32) / 32768.0
        audio_tensor = torch.tensor(normalized, dtype=torch.float32)
        if hasattr(audio_tensor, "reshape"):
            audio_tensor = audio_tensor.reshape(1, -1)

        test_embedding = self._model.encode_batch(audio_tensor).squeeze(0).squeeze(0)

        scores: Dict[str, float] = {}
        cosine_similarity = torch.nn.functional.cosine_similarity
        for name, profile_embedding in self._embeddings.items():
            profile_vector = (
                profile_embedding.squeeze(0)
                if hasattr(profile_embedding, "squeeze")
                else profile_embedding
            )
            similarity = cosine_similarity(test_embedding, profile_vector, dim=-1)
            score = float(similarity.item() if hasattr(similarity, "item") else similarity)
            scores[name] = score
        return scores

    def reload_profiles(self, profile_paths: Optional[List[Path]] = None) -> List[str]:
        """Reload embeddings from disk. If paths omitted, re-reads the config list.

        Returns the list of speaker names that were successfully loaded.
        Used by the web UI after enrollment / delete so changes apply without restart.
        """
        if profile_paths is not None:
            self.config.profile_paths = [
                Path(p) if not isinstance(p, Path) else p for p in profile_paths
            ]

        self._embeddings.clear()
        loaded: List[str] = []
        for profile_path in self.config.profile_paths:
            if not profile_path.exists():
                _LOGGER.warning("Profile path missing during reload: %s", profile_path)
                continue
            try:
                embedding = np.load(profile_path)
                if torch is None:
                    raise ImportError("torch not installed")
                profile_tensor = torch.tensor(embedding, dtype=torch.float32)
                if hasattr(profile_tensor, "reshape"):
                    profile_tensor = profile_tensor.reshape(1, -1)
                name = _extract_speaker_name(profile_path)
                self._embeddings[name] = profile_tensor
                loaded.append(name)
            except Exception as exc:
                _LOGGER.error("Failed to load profile %s: %s", profile_path, exc)
        return loaded

    def reset(self) -> None:
        """Reset verifier state."""
        return None
