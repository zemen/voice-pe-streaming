"""Ring buffer for audio history and mid-sentence detection."""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


def _compute_rms_normalized(audio_bytes: bytes) -> float:
    """Compute RMS of int16 PCM bytes, normalized to [0.0, 1.0] (1.0 = full scale)."""
    if not audio_bytes or len(audio_bytes) < 2:
        return 0.0
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


@dataclass
class RingBufferConfig:
    """Configuration for ring buffer holding pre_ww + WW window.

    Layout at moment of detection (T_end = end of the WW window the model
    classified, regardless of inference wall-clock latency):

        [T_end - 3s ────── T_end - 2s ────── T_end]
        │   pre_ww (1s)  │   WW window (2s)       │
        │   ───────────  │   ───────────────────  │
        │   get_pre_ww_  │   get_ww_audio()       │
        │   audio()      │   (= what LiveKit      │
        │                │   classified)          │
        get_all_audio() returns all 3s.

    buffer_seconds: length of the pre_ww segment (oldest).
    gap_ms: length of the WW window (newest). Named 'gap' for backward
        compatibility - was originally intended as a safety gap but is now
        semantically the WW window itself. Should equal the detector's
        window length (CHUNK_SECONDS = 2.0 s in LiveKitStream).
    """

    buffer_seconds: float = 1.0
    gap_ms: int = 2000
    sample_rate: int = 16000
    sample_width: int = 2

    def __post_init__(self):
        """Validate configuration."""
        if self.buffer_seconds <= 0:
            raise ValueError(f"buffer_seconds must be positive, got {self.buffer_seconds}")
        if self.gap_ms < 0:
            raise ValueError(f"gap_ms must be non-negative, got {self.gap_ms}")

    @property
    def buffer_size_bytes(self) -> int:
        """Calculate total buffer size in bytes."""
        total_seconds = self.buffer_seconds + (self.gap_ms / 1000.0)
        return int(total_seconds * self.sample_rate * self.sample_width)

    @property
    def gap_size_bytes(self) -> int:
        """Calculate gap size in bytes."""
        gap_seconds = self.gap_ms / 1000.0
        return int(gap_seconds * self.sample_rate * self.sample_width)

    @property
    def pre_ww_size_bytes(self) -> int:
        """Calculate pre-wake-word segment size in bytes (buffer - gap)."""
        return int(self.buffer_seconds * self.sample_rate * self.sample_width)


@dataclass
class RingBuffer:
    """Ring buffer for maintaining audio history."""

    config: RingBufferConfig
    _buffer: deque = field(default_factory=deque, init=False, repr=False)
    _total_bytes_written: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Initialize buffer with max length."""
        self._buffer = deque(maxlen=self.config.buffer_size_bytes)

    def write(self, audio_chunk: bytes) -> None:
        """Write audio chunk to ring buffer."""
        for byte in audio_chunk:
            self._buffer.append(byte)
        self._total_bytes_written += len(audio_chunk)

    def get_pre_ww_audio(self) -> Optional[bytes]:
        """Return oldest `buffer_seconds` of buffer = pre-WW segment, or None if buffer not full."""
        if len(self._buffer) < self.config.buffer_size_bytes:
            return None

        buffer_bytes = bytes(self._buffer)
        return buffer_bytes[0 : self.config.pre_ww_size_bytes]

    def get_ww_audio(self) -> Optional[bytes]:
        """Return newest `gap_ms` of buffer = the WW window (what the detector classified)."""
        if len(self._buffer) < self.config.buffer_size_bytes:
            return None

        buffer_bytes = bytes(self._buffer)
        return buffer_bytes[self.config.pre_ww_size_bytes :]

    def get_recent_audio(self, duration_ms: int) -> Optional[bytes]:
        """Get most recent audio of specified duration."""
        duration_bytes = int(
            (duration_ms / 1000.0) * self.config.sample_rate * self.config.sample_width
        )

        if len(self._buffer) < duration_bytes:
            return None

        buffer_bytes = bytes(self._buffer)
        return buffer_bytes[-duration_bytes:]

    def get_all_audio(self) -> Optional[bytes]:
        """Return everything currently buffered."""
        if not self._buffer:
            return None
        return bytes(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_bytes_written = 0

    @property
    def size(self) -> int:
        """Current buffer size in bytes."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached maximum capacity."""
        return len(self._buffer) >= self.config.buffer_size_bytes


@dataclass
class MidSentenceDetector:
    """Detects mid-sentence wake words using speaker comparison.

    Logic: reject the wake word if the same speaker was *confidently* speaking
    in the 1-second pre-WW window. We deliberately apply a stricter threshold
    to the pre-WW check than the regular speaker-verification threshold,
    because pre-WW audio is usually silence / ambient, and the enrolled
    speaker's profile weakly matches almost any human-band signal at the
    regular threshold (0.25) -- causing constant false rejections. Pre-WW
    must clear at least 0.5 cosine similarity (or `speaker_threshold + 0.15`,
    whichever is higher) before being considered "actively speaking".
    """

    ring_buffer: RingBuffer
    speaker_verifier: object
    silence_rms_threshold: float = 0.05

    def should_reject_detection(self, ww_audio: bytes) -> bool:
        """Reject only if the SAME enrolled speaker was actively speaking pre-WW.

        Two-stage gate:
          1. Energy: if pre-WW audio is below the silence RMS threshold, accept
             (no one was speaking, can't be mid-sentence). This is the cheap
             VAD check that prevents ECAPA from misidentifying digital silence
             as the enrolled speaker.
          2. Identity: if pre-WW was loud enough, use the regular speaker
             threshold (whatever the user configured, default 0.25) to check
             whether the same speaker who said the wake word was already
             speaking. If yes -> reject as mid-sentence.
        """
        pre_ww_audio = self.ring_buffer.get_pre_ww_audio()
        if pre_ww_audio is None:
            _LOGGER.info("mid-sentence: no pre_ww audio yet (buffer not full) -> accept")
            return False

        pre_rms = _compute_rms_normalized(pre_ww_audio)
        ww_rms = _compute_rms_normalized(ww_audio)

        if self.silence_rms_threshold > 0 and pre_rms < self.silence_rms_threshold:
            _LOGGER.info(
                "mid-sentence: pre_ww silence (rms=%.5f < %.5f) -> accept",
                pre_rms,
                self.silence_rms_threshold,
            )
            return False

        get_scores = getattr(self.speaker_verifier, "verify_with_scores", None)
        pre_scores: dict = {}
        ww_scores: dict = {}
        if callable(get_scores):
            try:
                r = get_scores(pre_ww_audio)
                if isinstance(r, dict):
                    pre_scores = r
                r = get_scores(ww_audio)
                if isinstance(r, dict):
                    ww_scores = r
            except Exception:
                pass

        pre_ww_speaker = self.speaker_verifier.verify_speaker(pre_ww_audio)
        ww_speaker = self.speaker_verifier.verify_speaker(ww_audio)

        _LOGGER.info(
            "mid-sentence: pre_ww rms=%.5f speaker=%s scores=%s | ww rms=%.5f speaker=%s scores=%s",
            pre_rms,
            pre_ww_speaker,
            {k: round(v, 3) for k, v in pre_scores.items()},
            ww_rms,
            ww_speaker,
            {k: round(v, 3) for k, v in ww_scores.items()},
        )

        if pre_ww_speaker is None or ww_speaker is None:
            return False
        return pre_ww_speaker == ww_speaker

    def reset(self) -> None:
        """Reset detector state."""
        self.ring_buffer.clear()
