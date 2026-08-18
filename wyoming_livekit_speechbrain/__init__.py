"""Wyoming LiveKit + SpeechBrain integration for Voice PE."""

ADDON_VERSION = "0.12.0"

from .buffer import MidSentenceDetector, RingBuffer, RingBufferConfig
from .enrollment import enroll_speaker
from .handler import LiveKitEventHandler
from .livekit_detector import LiveKitDetector, LiveKitDetectorConfig
from .speechbrain_verifier import SpeechBrainConfig, SpeechBrainVerifier

__all__ = [
    "LiveKitDetector",
    "LiveKitDetectorConfig",
    "LiveKitEventHandler",
    "MidSentenceDetector",
    "RingBuffer",
    "RingBufferConfig",
    "SpeechBrainConfig",
    "SpeechBrainVerifier",
    "enroll_speaker",
]
