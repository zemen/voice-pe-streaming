"""LED ring simulator for Voice PE 12-LED WS2812 patterns.

Simulates all 17 Voice PE LED effects with per-LED color state tracking.
Uses simulated time (dt parameter) for deterministic animation playback.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

LEDColor = tuple[int, int, int]

_BLACK: LEDColor = (0, 0, 0)
_RED: LEDColor = (255, 0, 0)
_DEFAULT_USER_COLOR: LEDColor = (255, 255, 255)

_MIC_MUTE_POSITIONS = (3, 9)
_SPEAKER_MUTE_POSITION = 6


def _clamp(value: int, lo: int = 0, hi: int = 255) -> int:
    return max(lo, min(hi, value))


def _scale_color(color: LEDColor, factor: float) -> LEDColor:
    return (
        _clamp(int(color[0] * factor)),
        _clamp(int(color[1] * factor)),
        _clamp(int(color[2] * factor)),
    )


def _hsv_to_rgb(h: float, s: float, v: float) -> LEDColor:
    """Convert HSV (h=0..360, s=0..1, v=0..1) to RGB tuple."""
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0.0
    elif h < 120:
        r, g, b = x, c, 0.0
    elif h < 180:
        r, g, b = 0.0, c, x
    elif h < 240:
        r, g, b = 0.0, x, c
    elif h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return (
        _clamp(int((r + m) * 255)),
        _clamp(int((g + m) * 255)),
        _clamp(int((b + m) * 255)),
    )


LEDEffect = Callable[["LEDRingSimulator", float], None]

_EFFECT_REGISTRY: dict[str, LEDEffect] = {}


def _register_effect(name: str) -> Callable[[LEDEffect], LEDEffect]:
    def decorator(fn: LEDEffect) -> LEDEffect:
        _EFFECT_REGISTRY[name] = fn
        return fn

    return decorator


class LEDRingSimulator:
    def __init__(self, num_leds: int = 12) -> None:
        self._num_leds = num_leds
        self._leds: list[LEDColor] = [_BLACK] * num_leds
        self._brightness: float = 1.0
        self._user_color: LEDColor = _DEFAULT_USER_COLOR
        self._current_effect: str | None = None
        self._effect_params: dict[str, Any] = {}
        self._elapsed: float = 0.0
        self._effect_state: dict[str, Any] = {}
        self._rng = random.Random(42)

    @property
    def brightness(self) -> float:
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        self._brightness = max(0.0, min(1.0, value))

    @property
    def user_color(self) -> LEDColor:
        return self._user_color

    @user_color.setter
    def user_color(self, value: LEDColor) -> None:
        self._user_color = value

    @property
    def current_effect(self) -> str | None:
        return self._current_effect

    @property
    def elapsed_time(self) -> float:
        return self._elapsed

    def get_led_state(self) -> list[LEDColor]:
        if self._brightness >= 1.0:
            return list(self._leds)
        return [_scale_color(led, self._brightness) for led in self._leds]

    def set_led(self, index: int, color: LEDColor) -> None:
        self._leds[index % self._num_leds] = color

    def set_all(self, color: LEDColor) -> None:
        self._leds = [color] * self._num_leds

    def clear(self) -> None:
        self._leds = [_BLACK] * self._num_leds

    def set_effect(self, name: str | None, **params: Any) -> None:
        if name is None:
            self.stop_effect()
            return
        if name not in _EFFECT_REGISTRY:
            raise ValueError(f"Unknown effect: {name}")
        self._current_effect = name
        self._effect_params = params
        self._elapsed = 0.0
        self._effect_state = {}
        self.clear()

    def stop_effect(self) -> None:
        self._current_effect = None
        self._effect_params = {}
        self._elapsed = 0.0
        self._effect_state = {}
        self.clear()

    def update(self, dt: float) -> None:
        self._elapsed += dt
        if self._current_effect is None:
            return
        fn = _EFFECT_REGISTRY.get(self._current_effect)
        if fn:
            fn(self, dt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "leds": self.get_led_state(),
            "effect": self._current_effect,
            "brightness": self._brightness,
            "elapsed": self._elapsed,
        }

    def _apply_mute_overlay(self) -> None:
        mic_muted = self._effect_params.get("mic_muted", False)
        speaker_muted = self._effect_params.get("speaker_muted", False)
        if mic_muted:
            for pos in _MIC_MUTE_POSITIONS:
                self._leds[pos] = _RED
        if speaker_muted:
            self._leds[_SPEAKER_MUTE_POSITION] = _RED


def _spin_pattern(
    ring: LEDRingSimulator,
    dt: float,
    update_interval: float,
    clockwise: bool,
) -> None:
    """Shared logic for spinning effects (waiting, listening, replying).

    2 groups of 3 LEDs with gradient tail, positioned 6 apart.
    """
    state = ring._effect_state
    if "accumulated" not in state:
        state["accumulated"] = 0.0
        state["position"] = 0

    state["accumulated"] += dt
    if state["accumulated"] >= update_interval:
        steps = int(state["accumulated"] / update_interval)
        state["accumulated"] -= steps * update_interval
        direction = 1 if clockwise else -1
        state["position"] = (state["position"] + steps * direction) % ring._num_leds

    ring.clear()
    color = ring._user_color
    pos = state["position"]
    n = ring._num_leds
    gradients = [1.0, 0.6, 0.3]

    for group_offset in (0, n // 2):
        for i, brightness in enumerate(gradients):
            if clockwise:
                idx = (pos + group_offset - i) % n
            else:
                idx = (pos + group_offset + i) % n
            ring._leds[idx] = _scale_color(color, brightness)


@_register_effect("waiting_for_command")
def _effect_waiting_for_command(ring: LEDRingSimulator, dt: float) -> None:
    _spin_pattern(ring, dt, update_interval=0.1, clockwise=True)


@_register_effect("listening")
def _effect_listening(ring: LEDRingSimulator, dt: float) -> None:
    _spin_pattern(ring, dt, update_interval=0.05, clockwise=True)


@_register_effect("thinking")
def _effect_thinking(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "accumulated" not in state:
        state["accumulated"] = 0.0
        state["phase"] = 0.0

    state["accumulated"] += dt
    if state["accumulated"] >= 0.01:
        steps = int(state["accumulated"] / 0.01)
        state["accumulated"] -= steps * 0.01
        state["phase"] += steps * 0.05

    ring.clear()
    brightness = (math.sin(state["phase"]) + 1.0) / 2.0
    brightness = max(0.1, brightness)
    color = _scale_color(ring._user_color, brightness)

    ring._leds[0] = color
    ring._leds[6] = color


@_register_effect("replying")
def _effect_replying(ring: LEDRingSimulator, dt: float) -> None:
    _spin_pattern(ring, dt, update_interval=0.05, clockwise=False)


@_register_effect("muted")
def _effect_muted(ring: LEDRingSimulator, dt: float) -> None:
    color = ring._user_color
    ring.set_all(color)
    ring._apply_mute_overlay()


@_register_effect("startup_failed")
def _effect_startup_failed(ring: LEDRingSimulator, dt: float) -> None:
    ring.clear()
    for i in range(0, ring._num_leds, 3):
        ring._leds[i] = _RED


@_register_effect("volume_display")
def _effect_volume_display(ring: LEDRingSimulator, dt: float) -> None:
    ring.clear()
    volume = ring._effect_params.get("volume", 0.0)
    num_lit = round(volume * ring._num_leds)

    if volume <= 0.0:
        ring._leds[0] = _RED
        return

    color = ring._user_color
    for i in range(num_lit):
        ring._leds[i] = color


@_register_effect("center_button_touched")
def _effect_center_button_touched(ring: LEDRingSimulator, dt: float) -> None:
    boosted = _scale_color(ring._user_color, 1.0)
    ring.set_all(boosted)


@_register_effect("twinkle")
def _effect_twinkle(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "accumulated" not in state:
        state["accumulated"] = 0.0

    state["accumulated"] += dt
    if state["accumulated"] < 0.05:
        return
    state["accumulated"] = 0.0

    color = ring._effect_params.get("color", ring._user_color)
    for i in range(ring._num_leds):
        if ring._rng.random() < 0.5:
            brightness = ring._rng.uniform(0.3, 1.0)
            ring._leds[i] = _scale_color(color, brightness)
        else:
            ring._leds[i] = _BLACK


@_register_effect("error")
def _effect_error(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "accumulated" not in state:
        state["accumulated"] = 0.0
        state["phase"] = 0.0

    state["accumulated"] += dt
    if state["accumulated"] >= 0.01:
        steps = int(state["accumulated"] / 0.01)
        state["accumulated"] -= steps * 0.01
        state["phase"] += steps * 0.05

    brightness = (math.sin(state["phase"]) + 1.0) / 2.0
    brightness = max(0.15, brightness)
    color = _scale_color(_RED, brightness)
    ring.set_all(color)


@_register_effect("timer_ring")
def _effect_timer_ring(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "accumulated" not in state:
        state["accumulated"] = 0.0
        state["phase"] = 0.0

    state["accumulated"] += dt
    if state["accumulated"] >= 0.01:
        steps = int(state["accumulated"] / 0.01)
        state["accumulated"] -= steps * 0.01
        state["phase"] += steps * 0.05

    brightness = (math.sin(state["phase"]) + 1.0) / 2.0
    brightness = max(0.15, brightness)
    color = _scale_color(ring._user_color, brightness)
    ring.set_all(color)
    ring._apply_mute_overlay()


@_register_effect("timer_tick")
def _effect_timer_tick(ring: LEDRingSimulator, dt: float) -> None:
    ring.clear()
    progress = ring._effect_params.get("progress", 0.0)
    remaining_fraction = 1.0 - progress
    num_lit = round(remaining_fraction * ring._num_leds)
    num_lit = max(0, min(ring._num_leds, num_lit))

    color = ring._user_color
    for i in range(num_lit):
        ring._leds[i] = color

    ring._apply_mute_overlay()


@_register_effect("rainbow")
def _effect_rainbow(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "offset" not in state:
        state["offset"] = 0.0
        state["accumulated"] = 0.0

    state["accumulated"] += dt
    if state["accumulated"] >= 0.05:
        steps = int(state["accumulated"] / 0.05)
        state["accumulated"] -= steps * 0.05
        state["offset"] += steps * 10.0

    for i in range(ring._num_leds):
        hue = (state["offset"] + i * (360.0 / ring._num_leds)) % 360.0
        ring._leds[i] = _hsv_to_rgb(hue, 1.0, 1.0)


@_register_effect("tick")
def _effect_tick(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "remaining" not in state:
        state["remaining"] = ring._num_leds
        state["accumulated"] = 0.0
        for i in range(ring._num_leds):
            ring._leds[i] = _RED
        return

    state["accumulated"] += dt
    if state["accumulated"] >= 0.333:
        ticks = int(state["accumulated"] / 0.333)
        state["accumulated"] -= ticks * 0.333
        state["remaining"] = max(0, state["remaining"] - ticks)

    ring.clear()
    for i in range(state["remaining"]):
        ring._leds[i] = _RED


@_register_effect("factory_reset")
def _effect_factory_reset(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "filled" not in state:
        state["filled"] = 0
        state["accumulated"] = 0.0
        ring.clear()
        return

    state["accumulated"] += dt
    if state["accumulated"] >= 1.0:
        new_fills = int(state["accumulated"] / 1.0)
        state["accumulated"] -= new_fills * 1.0
        state["filled"] = min(ring._num_leds, state["filled"] + new_fills)

    ring.clear()
    for i in range(state["filled"]):
        ring._leds[i] = _RED


@_register_effect("jack_plugged")
def _effect_jack_plugged(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "radius" not in state:
        state["radius"] = 0
        state["accumulated"] = 0.0

    state["accumulated"] += dt
    if state["accumulated"] >= 0.04:
        steps = int(state["accumulated"] / 0.04)
        state["accumulated"] -= steps * 0.04
        state["radius"] = min(ring._num_leds // 2, state["radius"] + steps)

    ring.clear()
    color = ring._user_color
    origin = 0
    for r in range(state["radius"] + 1):
        fade = 1.0 - (r / max(1, ring._num_leds // 2)) * 0.5
        c = _scale_color(color, fade)
        ring._leds[(origin + r) % ring._num_leds] = c
        if r > 0:
            ring._leds[(origin - r) % ring._num_leds] = c


@_register_effect("jack_unplugged")
def _effect_jack_unplugged(ring: LEDRingSimulator, dt: float) -> None:
    state = ring._effect_state
    if "radius" not in state:
        state["radius"] = 0
        state["accumulated"] = 0.0

    state["accumulated"] += dt
    if state["accumulated"] >= 0.04:
        steps = int(state["accumulated"] / 0.04)
        state["accumulated"] -= steps * 0.04
        state["radius"] = min(ring._num_leds // 2, state["radius"] + steps)

    ring.clear()
    color = ring._user_color
    origin = 6
    for r in range(state["radius"] + 1):
        fade = 1.0 - (r / max(1, ring._num_leds // 2)) * 0.5
        c = _scale_color(color, fade)
        ring._leds[(origin + r) % ring._num_leds] = c
        if r > 0:
            ring._leds[(origin - r) % ring._num_leds] = c
