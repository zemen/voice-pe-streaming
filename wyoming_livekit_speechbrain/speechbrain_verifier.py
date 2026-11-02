"""Speaker verification: enrolled-profile cosine similarity on CAM++ (default) or ECAPA embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .speaker_embedding import DEFAULT_SPEAKER_MODEL, SpeakerEmbedder, profile_model

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
    """Configuration for speaker verification (name kept for backwards compatibility)."""

    profile_paths: List[Path] = field(default_factory=list)
    threshold: float = 0.45
    model_source: str = DEFAULT_SPEAKER_MODEL
    savedir: Optional[Path] = None

    def __post_init__(self):
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")

        self.profile_paths = [Path(path) if not isinstance(path, Path) else path for path in self.profile_paths]
        if self.savedir is not None and not isinstance(self.savedir, Path):
            self.savedir = Path(self.savedir)


@dataclass
class SpeechBrainVerifier:
    """Cosine similarity between the wake-word window's embedding and each enrolled profile."""

    config: SpeechBrainConfig
    _embedder: Optional[SpeakerEmbedder] = field(default=None, init=False, repr=False)
    _embeddings: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._embedder = SpeakerEmbedder(self.config.model_source, self.config.savedir)
        for profile_path in self.config.profile_paths:
            if not profile_path.exists():
                raise FileNotFoundError(f"Profile file not found: {profile_path}")
        self.reload_profiles()

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
        """Return cosine similarity per enrolled speaker. Empty if nothing is enrolled."""
        if self._embedder is None or not self._embeddings:
            return {}

        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if samples.size < 4000:
            return {}

        test = self._embedder.embed(samples.astype(np.float32) / 32768.0)
        return {name: float(np.dot(test, prof)) for name, prof in self._embeddings.items()}

    def reload_profiles(self, profile_paths: Optional[List[Path]] = None) -> List[str]:
        """Reload embeddings from disk. If paths omitted, re-reads the config list.

        Profiles enrolled with a different embedding model are skipped (re-enroll them).
        Returns the list of speaker names that were successfully loaded.
        """
        if profile_paths is not None:
            self.config.profile_paths = [Path(p) if not isinstance(p, Path) else p for p in profile_paths]

        self._embeddings.clear()
        loaded: List[str] = []
        for profile_path in self.config.profile_paths:
            if not profile_path.exists():
                _LOGGER.warning("Profile path missing during reload: %s", profile_path)
                continue
            model = profile_model(profile_path)
            if model != self.config.model_source:
                _LOGGER.warning(
                    "Profile %s was enrolled with %s, verifier uses %s - re-enroll this speaker",
                    profile_path.name,
                    model,
                    self.config.model_source,
                )
                continue
            try:
                emb = np.load(profile_path).astype(np.float32).reshape(-1)
                name = _extract_speaker_name(profile_path)
                self._embeddings[name] = emb / (np.linalg.norm(emb) + 1e-9)
                loaded.append(name)
            except Exception as exc:
                _LOGGER.error("Failed to load profile %s: %s", profile_path, exc)
        return loaded

    def reset(self) -> None:
        """Reset verifier state."""
        return None
