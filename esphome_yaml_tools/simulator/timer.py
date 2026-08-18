"""Timer simulator for Voice PE multi-timer support.

Simulates the Voice PE timer system including:
- Multiple simultaneous timers
- Timer events: started, finished, cancelled, updated, tick
- Ringing behavior with audio ducking, stop WW auto-enable, 15-min auto-stop
- LED ring integration for countdown display

References:
- Voice PE YAML lines 169-176 (timer globals)
- Voice PE YAML lines 222-262 (timer_ringing switch)
- Voice PE YAML lines 1382-1459 (timer scripts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from esphome_yaml_tools.simulator.led_ring import LEDRingSimulator


# 15-minute auto-stop safety limit (matches Voice PE firmware)
_AUTO_STOP_SECONDS = 900.0


class TimerEventType(Enum):
    """Timer event types matching Voice PE on_timer_* events."""

    STARTED = "started"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    UPDATED = "updated"
    TICK = "tick"


@dataclass
class TimerEvent:
    """A timer event with type, timer_id, and optional data."""

    event_type: TimerEventType
    timer_id: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimerInfo:
    """State container for a single timer."""

    timer_id: str
    total_seconds: float
    remaining_seconds: float = 0.0
    is_active: bool = True
    is_ringing: bool = False

    def __post_init__(self) -> None:
        if self.remaining_seconds == 0.0 and self.is_active:
            self.remaining_seconds = self.total_seconds

    @property
    def progress(self) -> float:
        """Fraction of timer elapsed (0.0 = just started, 1.0 = finished)."""
        if self.total_seconds <= 0.0:
            return 0.0
        return 1.0 - (self.remaining_seconds / self.total_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timer_id": self.timer_id,
            "total_seconds": self.total_seconds,
            "remaining_seconds": self.remaining_seconds,
            "is_active": self.is_active,
            "is_ringing": self.is_ringing,
            "progress": self.progress,
        }


EventCallback = Callable[[TimerEvent], None]


class TimerSimulator:
    """Simulates Voice PE timer functionality.

    Manages multiple simultaneous timers with events, ringing behavior,
    stop wake word integration, audio ducking, and LED ring countdown.
    """

    def __init__(self, led_ring: LEDRingSimulator | None = None) -> None:
        self._timers: dict[str, TimerInfo] = {}
        self._next_id: int = 0
        self._event_history: list[TimerEvent] = []
        self._callbacks: dict[TimerEventType, list[EventCallback]] = {}
        self._led_ring = led_ring

        # Ringing state
        self._is_ringing: bool = False
        self._ringing_elapsed: float = 0.0
        self._stop_ww_enabled: bool = False
        self._audio_ducking_active: bool = False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_ringing(self) -> bool:
        return self._is_ringing

    @property
    def ringing_elapsed(self) -> float:
        return self._ringing_elapsed

    @property
    def stop_ww_enabled(self) -> bool:
        return self._stop_ww_enabled

    @property
    def audio_ducking_active(self) -> bool:
        return self._audio_ducking_active

    @property
    def is_any_timer_active(self) -> bool:
        return any(t.is_active for t in self._timers.values())

    # -------------------------------------------------------------------------
    # Timer Lifecycle
    # -------------------------------------------------------------------------

    def start_timer(self, duration: float, timer_id: str | None = None) -> str:
        """Start a new timer. Returns the timer ID."""
        if timer_id is None:
            timer_id = self._generate_id()

        info = TimerInfo(timer_id=timer_id, total_seconds=duration)
        self._timers[timer_id] = info
        self._emit(TimerEventType.STARTED, timer_id)

        # Zero-duration timer finishes immediately
        if duration <= 0.0:
            self._finish_timer(timer_id)
        else:
            self._update_led()

        return timer_id

    def cancel_timer(self, timer_id: str) -> None:
        """Cancel a timer by ID."""
        if timer_id not in self._timers:
            raise KeyError(f"Timer '{timer_id}' not found")
        self._timers[timer_id].is_active = False
        self._emit(TimerEventType.CANCELLED, timer_id)
        self._update_led()

    def cancel_all_timers(self) -> None:
        """Cancel all timers."""
        for timer_id in list(self._timers):
            if self._timers[timer_id].is_active:
                self._timers[timer_id].is_active = False
                self._emit(TimerEventType.CANCELLED, timer_id)
        self._update_led()

    def update_timer(self, timer_id: str, new_duration: float) -> None:
        """Update a timer's total duration, recalculating remaining time."""
        if timer_id not in self._timers:
            raise KeyError(f"Timer '{timer_id}' not found")
        info = self._timers[timer_id]
        elapsed = info.total_seconds - info.remaining_seconds
        info.total_seconds = new_duration
        info.remaining_seconds = max(0.0, new_duration - elapsed)
        self._emit(TimerEventType.UPDATED, timer_id)
        self._update_led()

    def stop_ringing(self) -> None:
        """Stop the ringing state (equivalent to timer_ringing switch off)."""
        if not self._is_ringing:
            return
        self._is_ringing = False
        self._ringing_elapsed = 0.0
        self._stop_ww_enabled = False
        self._audio_ducking_active = False
        self._update_led()

    # -------------------------------------------------------------------------
    # Time Advancement
    # -------------------------------------------------------------------------

    def advance_time(self, seconds: float) -> None:
        """Advance time for all active timers."""
        if seconds <= 0.0:
            return

        # Update ringing elapsed & check auto-stop
        if self._is_ringing:
            self._ringing_elapsed += seconds
            if self._ringing_elapsed >= _AUTO_STOP_SECONDS:
                self.stop_ringing()
                return

        # Update active timers
        for timer_id in list(self._timers):
            info = self._timers[timer_id]
            if not info.is_active:
                continue

            info.remaining_seconds -= seconds
            if info.remaining_seconds <= 0.0:
                info.remaining_seconds = 0.0
                info.is_active = False
                self._finish_timer(timer_id)
            else:
                self._emit(
                    TimerEventType.TICK,
                    timer_id,
                    {"seconds_left": info.remaining_seconds},
                )

        self._update_led()

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_timer(self, timer_id: str) -> TimerInfo | None:
        return self._timers.get(timer_id)

    def get_active_timers(self) -> list[TimerInfo]:
        return [t for t in self._timers.values() if t.is_active]

    def get_all_timers(self) -> list[TimerInfo]:
        return list(self._timers.values())

    def get_first_active_timer(self) -> TimerInfo | None:
        """Get the active timer with the least remaining time (Voice PE pattern)."""
        active = self.get_active_timers()
        if not active:
            return None
        return min(active, key=lambda t: t.remaining_seconds)

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def on_event(self, event_type: TimerEventType, callback: EventCallback) -> None:
        """Register a callback for a timer event type."""
        self._callbacks.setdefault(event_type, []).append(callback)

    def get_event_history(self) -> list[TimerEvent]:
        return list(self._event_history)

    def clear_event_history(self) -> None:
        self._event_history.clear()

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all timer state."""
        self._timers.clear()
        self._event_history.clear()
        self._is_ringing = False
        self._ringing_elapsed = 0.0
        self._stop_ww_enabled = False
        self._audio_ducking_active = False
        self._next_id = 0
        if self._led_ring is not None:
            self._led_ring.stop_effect()

    def snapshot(self) -> dict[str, Any]:
        """Serializable snapshot of timer state."""
        return {
            "timers": [t.to_dict() for t in self._timers.values()],
            "is_ringing": self._is_ringing,
            "ringing_elapsed": self._ringing_elapsed,
            "stop_ww_enabled": self._stop_ww_enabled,
            "audio_ducking_active": self._audio_ducking_active,
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _generate_id(self) -> str:
        self._next_id += 1
        return f"timer_{self._next_id}"

    def _emit(
        self,
        event_type: TimerEventType,
        timer_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        event = TimerEvent(event_type=event_type, timer_id=timer_id, data=data or {})
        self._event_history.append(event)
        for cb in self._callbacks.get(event_type, []):
            cb(event)

    def _finish_timer(self, timer_id: str) -> None:
        """Handle timer completion: emit event, start ringing."""
        self._emit(TimerEventType.FINISHED, timer_id)

        # Start ringing (Voice PE: switch.turn_on timer_ringing)
        if not self._is_ringing:
            self._is_ringing = True
            self._ringing_elapsed = 0.0
        self._stop_ww_enabled = True
        self._audio_ducking_active = True

    def _update_led(self) -> None:
        """Update LED ring based on current timer state."""
        if self._led_ring is None:
            return

        if self._is_ringing:
            if self._led_ring.current_effect != "timer_ring":
                self._led_ring.set_effect("timer_ring")
            self._led_ring.update(0.0)
            return

        first = self.get_first_active_timer()
        if first is not None:
            progress = first.progress
            if self._led_ring.current_effect != "timer_tick":
                self._led_ring.set_effect("timer_tick", progress=progress)
            else:
                self._led_ring._effect_params["progress"] = progress
            self._led_ring.update(0.0)
        else:
            # No active timers and not ringing - clear LED
            self._led_ring.stop_effect()
