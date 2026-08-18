"""State containers for ESPHome component simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class VoiceAssistantPhase(IntEnum):
    """Voice assistant phases matching Voice PE firmware substitution values."""

    IDLE = 1
    WAITING_FOR_COMMAND = 2
    LISTENING = 3
    THINKING = 4
    REPLYING = 5
    NOT_READY = 10
    ERROR = 11


class ComponentState:
    """Base class for all component states."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class LightState(ComponentState):
    on: bool = False
    brightness: float = 0.0
    color: tuple[int, int, int] = (0, 0, 0)
    effect: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "on": self.on,
            "brightness": self.brightness,
            "color": self.color,
            "effect": self.effect,
        }


@dataclass
class SwitchState(ComponentState):
    on: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"on": self.on}


@dataclass
class SensorState(ComponentState):
    value: float = 0.0
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit}


@dataclass
class VoiceAssistantState(ComponentState):
    phase: VoiceAssistantPhase = VoiceAssistantPhase.IDLE
    is_streaming: bool = False
    is_in_session: bool = False
    use_wake_word: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.name.lower(),
            "is_streaming": self.is_streaming,
            "is_in_session": self.is_in_session,
            "use_wake_word": self.use_wake_word,
        }


@dataclass
class MicroWakeWordState(ComponentState):
    """Runtime state for the micro_wake_word component + per-model enable/disable."""

    running: bool = False
    enabled_models: dict[str, bool] = field(default_factory=dict)

    def is_model_enabled(self, model_id: str) -> bool:
        return self.enabled_models.get(model_id, True)

    def to_dict(self) -> dict[str, Any]:
        return {"running": self.running, "enabled_models": dict(self.enabled_models)}


@dataclass
class MediaPlayerState(ComponentState):
    is_announcing: bool = False
    is_muted: bool = False
    volume: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {"is_announcing": self.is_announcing, "is_muted": self.is_muted, "volume": self.volume}


@dataclass
class TimerState(ComponentState):
    active: bool = False
    remaining: float = 0.0
    total: float = 0.0

    @property
    def progress(self) -> float:
        if self.total <= 0.0:
            return 0.0
        return 1.0 - (self.remaining / self.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "remaining": self.remaining,
            "total": self.total,
        }
