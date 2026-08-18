"""Voice assistant and wake word simulation for ESPHome Voice PE.

Simulates:
- Micro wake word detection with multiple models
- Voice assistant state machine (idle → listening → thinking → replying)
- Mute integration (muted = ignore wake word)
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable
from enum import Enum

from .states import VoiceAssistantState, VoiceAssistantPhase


class WakeWordModel(Enum):
    """Available wake word models."""

    OKAY_NABU = "okay_nabu"
    HEY_JARVIS = "hey_jarvis"
    HEY_MYCROFT = "hey_mycroft"
    STOP = "stop"


@dataclass
class MicroWakeWordSimulator:
    """Simulates micro wake word detection.

    Tracks enabled models and simulates detection events.
    Does not process actual audio - detection is triggered manually.
    """

    enabled_models: Set[str] = field(default_factory=set)
    is_muted: bool = False
    on_detection: Optional[Callable[[str], None]] = None

    def __post_init__(self):
        """Initialize with default models."""
        # Start with user models enabled
        self.enabled_models = {
            WakeWordModel.OKAY_NABU.value,
            WakeWordModel.HEY_JARVIS.value,
            WakeWordModel.HEY_MYCROFT.value,
        }

    def enable_model(self, model: str) -> None:
        """Enable a wake word model."""
        self.enabled_models.add(model)

    def disable_model(self, model: str) -> None:
        """Disable a wake word model."""
        self.enabled_models.discard(model)

    def is_model_enabled(self, model: str) -> bool:
        """Check if a model is enabled."""
        return model in self.enabled_models

    def set_muted(self, muted: bool) -> None:
        """Set mute state."""
        self.is_muted = muted

    def detect_wake_word(self, model: str) -> bool:
        """Simulate wake word detection.

        Args:
            model: Wake word model that detected

        Returns:
            True if detection was processed, False if ignored (muted or disabled)
        """
        # Ignore if muted
        if self.is_muted:
            return False

        # Ignore if model not enabled
        if model not in self.enabled_models:
            return False

        # Emit detection event
        if self.on_detection:
            self.on_detection(model)

        return True

    def reset(self) -> None:
        """Reset to initial state."""
        self.enabled_models = {
            WakeWordModel.OKAY_NABU.value,
            WakeWordModel.HEY_JARVIS.value,
            WakeWordModel.HEY_MYCROFT.value,
        }
        self.is_muted = False


@dataclass
class VoiceAssistantSimulator:
    """Simulates voice assistant state machine.

    Manages phase transitions:
    - idle → listening (on wake word)
    - listening → thinking (on STT end)
    - thinking → replying (on intent end)
    - replying → idle (on TTS end)
    """

    state: VoiceAssistantState = field(default_factory=VoiceAssistantState)
    wake_word_sim: MicroWakeWordSimulator = field(default_factory=MicroWakeWordSimulator)
    on_phase_change: Optional[Callable[[VoiceAssistantPhase, VoiceAssistantPhase], None]] = None

    def __post_init__(self):
        """Connect wake word simulator."""
        self.wake_word_sim.on_detection = self._on_wake_word_detected

    def _on_wake_word_detected(self, model: str) -> None:
        """Handle wake word detection."""
        if self.state.phase == VoiceAssistantPhase.IDLE:
            self.set_phase(VoiceAssistantPhase.LISTENING)

    def set_phase(self, phase: VoiceAssistantPhase) -> None:
        """Set voice assistant phase."""
        old_phase = self.state.phase
        self.state.phase = phase

        if self.on_phase_change and old_phase != phase:
            self.on_phase_change(old_phase, phase)

    def get_phase(self) -> VoiceAssistantPhase:
        """Get current phase."""
        return self.state.phase

    def detect_wake_word(self, model: str) -> bool:
        """Simulate wake word detection.

        Args:
            model: Wake word model that detected

        Returns:
            True if detection was processed
        """
        return self.wake_word_sim.detect_wake_word(model)

    def on_stt_end(self) -> None:
        """Handle STT end event."""
        if self.state.phase == VoiceAssistantPhase.LISTENING:
            self.set_phase(VoiceAssistantPhase.THINKING)

    def on_intent_end(self) -> None:
        """Handle intent end event."""
        if self.state.phase == VoiceAssistantPhase.THINKING:
            self.set_phase(VoiceAssistantPhase.REPLYING)

    def on_tts_end(self) -> None:
        """Handle TTS end event."""
        if self.state.phase == VoiceAssistantPhase.REPLYING:
            self.set_phase(VoiceAssistantPhase.IDLE)

    def on_error(self) -> None:
        """Handle error event."""
        self.set_phase(VoiceAssistantPhase.IDLE)

    def set_muted(self, muted: bool) -> None:
        """Set mute state."""
        self.wake_word_sim.set_muted(muted)

    def is_muted(self) -> bool:
        """Check if muted."""
        return self.wake_word_sim.is_muted

    def enable_model(self, model: str) -> None:
        """Enable a wake word model."""
        self.wake_word_sim.enable_model(model)

    def disable_model(self, model: str) -> None:
        """Disable a wake word model."""
        self.wake_word_sim.disable_model(model)

    def is_model_enabled(self, model: str) -> bool:
        """Check if a model is enabled."""
        return self.wake_word_sim.is_model_enabled(model)

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = VoiceAssistantState()
        self.wake_word_sim.reset()

    def snapshot(self) -> Dict:
        """Capture current state."""
        return {
            "phase": self.state.phase.value,
            "is_muted": self.wake_word_sim.is_muted,
            "enabled_models": list(self.wake_word_sim.enabled_models),
        }
