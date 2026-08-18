"""Button, rotary dial, mute switch, and jack sensor simulator.

Simulates Voice PE physical controls:
- Center button (GPIO0): single/double/triple press, long press, factory reset
- Rotary dial (GPIO16/18): volume when unpressed, hue when pressed
- Mute switch (GPIO3): microphone mute toggle
- Jack sensor (GPIO17): audio jack plug detection

References:
- Voice PE YAML lines 272-492 (button automations)
- Voice PE YAML lines 960-1010 (rotary dial)
- Voice PE YAML lines 494-551 (mute, jack)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ControlEventType(Enum):
    PRESSED = "pressed"
    RELEASED = "released"
    SINGLE_PRESS = "single_press"
    DOUBLE_PRESS = "double_press"
    TRIPLE_PRESS = "triple_press"
    LONG_PRESS = "long_press"
    FACTORY_RESET_WARNING = "factory_reset_warning"
    FACTORY_RESET_CONFIRM = "factory_reset_confirm"
    VOLUME_CHANGED = "volume_changed"
    HUE_CHANGED = "hue_changed"
    MUTED = "muted"
    UNMUTED = "unmuted"
    JACK_PLUGGED = "jack_plugged"
    JACK_UNPLUGGED = "jack_unplugged"


@dataclass
class ControlEvent:
    event_type: ControlEventType
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[ControlEvent], None]

_FACTORY_RESET_WARNING_THRESHOLD = 10.0
_FACTORY_RESET_CONFIRM_THRESHOLD = 22.0

_MULTI_CLICK_EVENT_MAP = {
    1: ControlEventType.SINGLE_PRESS,
    2: ControlEventType.DOUBLE_PRESS,
    3: ControlEventType.TRIPLE_PRESS,
}


class CenterButton:
    def __init__(
        self,
        multi_click_window: float = 0.25,
        long_press_threshold: float = 1.0,
    ) -> None:
        self.multi_click_window = multi_click_window
        self.long_press_threshold = long_press_threshold

        self._is_pressed: bool = False
        self._press_start_time: float = 0.0
        self._current_time: float = 0.0
        self._hold_time: float = 0.0

        self._click_count: int = 0
        self._last_release_time: float = -1.0
        self._last_press_duration: float = 0.0

        self._long_press_emitted: bool = False
        self._factory_warning_emitted: bool = False
        self._factory_confirm_emitted: bool = False

        self._event_history: list[ControlEvent] = []
        self._callbacks: dict[ControlEventType, list[EventCallback]] = {}

    @property
    def is_pressed(self) -> bool:
        return self._is_pressed

    def press(self) -> None:
        if self._is_pressed:
            return
        self._is_pressed = True
        self._press_start_time = self._current_time
        self._hold_time = 0.0
        self._long_press_emitted = False
        self._factory_warning_emitted = False
        self._factory_confirm_emitted = False
        self._emit(ControlEventType.PRESSED, {"time": self._current_time})

    def release(self) -> None:
        if not self._is_pressed:
            return
        self._is_pressed = False
        duration = self._hold_time
        self._emit(ControlEventType.RELEASED, {"duration": duration})

        if self._long_press_emitted or self._factory_warning_emitted:
            return

        self._click_count += 1
        self._last_press_duration = duration
        self._last_release_time = self._current_time

    def press_and_release(self, duration: float = 0.1) -> None:
        self.press()
        self.advance_time(duration)
        self.release()

    def advance_time(self, seconds: float) -> None:
        self._current_time += seconds

        if self._is_pressed:
            self._hold_time += seconds
            self._check_hold_thresholds()

        if not self._is_pressed and self._click_count > 0:
            elapsed_since_release = self._current_time - self._last_release_time
            if elapsed_since_release >= self.multi_click_window:
                self._resolve_clicks()

    def on_event(self, event_type: ControlEventType, callback: EventCallback) -> None:
        self._callbacks.setdefault(event_type, []).append(callback)

    def get_event_history(self) -> list[ControlEvent]:
        return list(self._event_history)

    def reset(self) -> None:
        self._is_pressed = False
        self._press_start_time = 0.0
        self._current_time = 0.0
        self._hold_time = 0.0
        self._click_count = 0
        self._last_release_time = -1.0
        self._last_press_duration = 0.0
        self._long_press_emitted = False
        self._factory_warning_emitted = False
        self._factory_confirm_emitted = False
        self._event_history.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "is_pressed": self._is_pressed,
            "pending_clicks": self._click_count,
            "hold_time": self._hold_time,
            "current_time": self._current_time,
        }

    def _check_hold_thresholds(self) -> None:
        if (
            not self._factory_confirm_emitted
            and self._hold_time >= _FACTORY_RESET_CONFIRM_THRESHOLD
        ):
            if not self._factory_warning_emitted:
                self._factory_warning_emitted = True
                self._emit(
                    ControlEventType.FACTORY_RESET_WARNING,
                    {"hold_time": self._hold_time},
                )
            self._factory_confirm_emitted = True
            self._emit(
                ControlEventType.FACTORY_RESET_CONFIRM,
                {"hold_time": self._hold_time},
            )
            return

        if (
            not self._factory_warning_emitted
            and self._hold_time >= _FACTORY_RESET_WARNING_THRESHOLD
        ):
            self._factory_warning_emitted = True
            self._emit(
                ControlEventType.FACTORY_RESET_WARNING,
                {"hold_time": self._hold_time},
            )
            return

        if not self._long_press_emitted and self._hold_time >= self.long_press_threshold:
            self._long_press_emitted = True
            self._emit(
                ControlEventType.LONG_PRESS,
                {"duration": self._hold_time},
            )

    def _resolve_clicks(self) -> None:
        count = self._click_count
        self._click_count = 0

        event_type = _MULTI_CLICK_EVENT_MAP.get(count)
        if event_type is None:
            event_type = ControlEventType.TRIPLE_PRESS
            count = 3

        self._emit(event_type, {"click_count": count, "duration": self._last_press_duration})

    def _emit(self, event_type: ControlEventType, data: dict[str, Any] | None = None) -> None:
        event = ControlEvent(
            event_type=event_type,
            source="center_button",
            data=data or {},
        )
        self._event_history.append(event)
        for cb in self._callbacks.get(event_type, []):
            cb(event)


class RotaryDial:
    def __init__(
        self,
        initial_volume: float = 0.5,
        volume_step: float = 0.05,
        hue_step: float = 15.0,
    ) -> None:
        self._initial_volume = initial_volume
        self.volume_step = volume_step
        self.hue_step = hue_step

        self._volume: float = initial_volume
        self.hue: float = 0.0
        self._button_pressed: bool = False

        self._event_history: list[ControlEvent] = []
        self._callbacks: dict[ControlEventType, list[EventCallback]] = {}

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(1.0, value))

    def set_button_pressed(self, pressed: bool) -> None:
        self._button_pressed = pressed

    def rotate_clockwise(self) -> None:
        self._rotate(1)

    def rotate_anticlockwise(self) -> None:
        self._rotate(-1)

    def rotate(self, direction: str = "clockwise") -> None:
        if direction == "clockwise":
            self.rotate_clockwise()
        else:
            self.rotate_anticlockwise()

    def on_event(self, event_type: ControlEventType, callback: EventCallback) -> None:
        self._callbacks.setdefault(event_type, []).append(callback)

    def get_event_history(self) -> list[ControlEvent]:
        return list(self._event_history)

    def reset(self) -> None:
        self._volume = self._initial_volume
        self.hue = 0.0
        self._button_pressed = False
        self._event_history.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "volume": self._volume,
            "hue": self.hue,
            "button_pressed": self._button_pressed,
        }

    def _rotate(self, direction: int) -> None:
        dir_str = "clockwise" if direction > 0 else "anticlockwise"

        if self._button_pressed:
            self.hue = (self.hue + direction * self.hue_step) % 360.0
            self._emit(ControlEventType.HUE_CHANGED, {"hue": self.hue, "direction": dir_str})
        else:
            self._volume = max(0.0, min(1.0, self._volume + direction * self.volume_step))
            self._emit(
                ControlEventType.VOLUME_CHANGED,
                {"volume": self._volume, "direction": dir_str},
            )

    def _emit(self, event_type: ControlEventType, data: dict[str, Any] | None = None) -> None:
        event = ControlEvent(
            event_type=event_type,
            source="rotary_dial",
            data=data or {},
        )
        self._event_history.append(event)
        for cb in self._callbacks.get(event_type, []):
            cb(event)


class MuteSwitch:
    def __init__(self) -> None:
        self._is_muted: bool = False
        self._event_history: list[ControlEvent] = []
        self._callbacks: dict[ControlEventType, list[EventCallback]] = {}

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    def set_muted(self, muted: bool) -> None:
        if muted == self._is_muted:
            return
        self._is_muted = muted
        event_type = ControlEventType.MUTED if muted else ControlEventType.UNMUTED
        self._emit(event_type, {"is_muted": muted})

    def toggle(self) -> None:
        self.set_muted(not self._is_muted)

    def on_event(self, event_type: ControlEventType, callback: EventCallback) -> None:
        self._callbacks.setdefault(event_type, []).append(callback)

    def get_event_history(self) -> list[ControlEvent]:
        return list(self._event_history)

    def reset(self) -> None:
        self._is_muted = False
        self._event_history.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"is_muted": self._is_muted}

    def _emit(self, event_type: ControlEventType, data: dict[str, Any] | None = None) -> None:
        event = ControlEvent(
            event_type=event_type,
            source="mute_switch",
            data=data or {},
        )
        self._event_history.append(event)
        for cb in self._callbacks.get(event_type, []):
            cb(event)


class JackSensor:
    def __init__(self) -> None:
        self._is_plugged: bool = False
        self._event_history: list[ControlEvent] = []
        self._callbacks: dict[ControlEventType, list[EventCallback]] = {}

    @property
    def is_plugged(self) -> bool:
        return self._is_plugged

    def set_plugged(self, plugged: bool) -> None:
        if plugged == self._is_plugged:
            return
        self._is_plugged = plugged
        event_type = ControlEventType.JACK_PLUGGED if plugged else ControlEventType.JACK_UNPLUGGED
        self._emit(event_type, {"is_plugged": plugged})

    def on_event(self, event_type: ControlEventType, callback: EventCallback) -> None:
        self._callbacks.setdefault(event_type, []).append(callback)

    def get_event_history(self) -> list[ControlEvent]:
        return list(self._event_history)

    def reset(self) -> None:
        self._is_plugged = False
        self._event_history.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"is_plugged": self._is_plugged}

    def _emit(self, event_type: ControlEventType, data: dict[str, Any] | None = None) -> None:
        event = ControlEvent(
            event_type=event_type,
            source="jack_sensor",
            data=data or {},
        )
        self._event_history.append(event)
        for cb in self._callbacks.get(event_type, []):
            cb(event)
