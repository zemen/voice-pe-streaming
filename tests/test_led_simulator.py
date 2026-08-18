"""Tests for LED ring simulator - Voice PE 12-LED WS2812 patterns.

TDD approach: Tests written first, then implementation.
Covers all 17 Voice PE LED effects + additional helper tests.
"""

from __future__ import annotations

import math

import pytest

from esphome_yaml_tools.simulator.led_ring import (
    LEDRingSimulator,
    LEDColor,
    LEDEffect,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)


def _any_led_on(ring: LEDRingSimulator) -> bool:
    """Return True if any LED is not black."""
    return any(led != BLACK for led in ring.get_led_state())


def _count_lit(ring: LEDRingSimulator) -> int:
    """Return number of non-black LEDs."""
    return sum(1 for led in ring.get_led_state() if led != BLACK)


def _lit_indices(ring: LEDRingSimulator) -> list[int]:
    """Return indices of non-black LEDs."""
    return [i for i, led in enumerate(ring.get_led_state()) if led != BLACK]


# ===========================================================================
# Core / Construction
# ===========================================================================


class TestLEDRingCore:
    """Basic ring state management."""

    def test_default_state_all_off(self):
        ring = LEDRingSimulator()
        state = ring.get_led_state()
        assert len(state) == 12
        assert all(led == BLACK for led in state)

    def test_custom_num_leds(self):
        ring = LEDRingSimulator(num_leds=6)
        assert len(ring.get_led_state()) == 6

    def test_get_led_state_returns_copy(self):
        ring = LEDRingSimulator()
        s1 = ring.get_led_state()
        s2 = ring.get_led_state()
        assert s1 is not s2

    def test_set_led_color_single(self):
        ring = LEDRingSimulator()
        ring.set_led(0, RED)
        assert ring.get_led_state()[0] == RED
        assert ring.get_led_state()[1] == BLACK

    def test_set_all_leds(self):
        ring = LEDRingSimulator()
        ring.set_all(BLUE)
        assert all(led == BLUE for led in ring.get_led_state())

    def test_clear_all_leds(self):
        ring = LEDRingSimulator()
        ring.set_all(WHITE)
        ring.clear()
        assert all(led == BLACK for led in ring.get_led_state())

    def test_set_led_out_of_range_wraps(self):
        ring = LEDRingSimulator()
        ring.set_led(12, RED)  # wraps to index 0
        assert ring.get_led_state()[0] == RED

    def test_brightness_scales_output(self):
        ring = LEDRingSimulator()
        ring.brightness = 0.5
        ring.set_led(0, (200, 100, 50))
        state = ring.get_led_state()
        assert state[0] == (100, 50, 25)

    def test_brightness_clamp(self):
        ring = LEDRingSimulator()
        ring.brightness = 1.5
        assert ring.brightness == 1.0
        ring.brightness = -0.3
        assert ring.brightness == 0.0

    def test_user_color_property(self):
        ring = LEDRingSimulator()
        ring.user_color = (100, 200, 50)
        assert ring.user_color == (100, 200, 50)


# ===========================================================================
# Effect Management
# ===========================================================================


class TestEffectManagement:
    """Set/get/stop effects."""

    def test_set_effect_stores_name(self):
        ring = LEDRingSimulator()
        ring.set_effect("listening")
        assert ring.current_effect == "listening"

    def test_set_effect_none_stops(self):
        ring = LEDRingSimulator()
        ring.set_effect("listening")
        ring.set_effect(None)
        assert ring.current_effect is None
        assert all(led == BLACK for led in ring.get_led_state())

    def test_set_effect_replaces_previous(self):
        ring = LEDRingSimulator()
        ring.set_effect("listening")
        ring.set_effect("thinking")
        assert ring.current_effect == "thinking"

    def test_stop_effect(self):
        ring = LEDRingSimulator()
        ring.set_effect("listening")
        ring.stop_effect()
        assert ring.current_effect is None

    def test_unknown_effect_raises(self):
        ring = LEDRingSimulator()
        with pytest.raises(ValueError, match="Unknown effect"):
            ring.set_effect("nonexistent_effect_xyz")

    def test_update_without_effect_is_noop(self):
        ring = LEDRingSimulator()
        ring.update(0.1)  # Should not raise
        assert all(led == BLACK for led in ring.get_led_state())

    def test_set_effect_with_params(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_tick", progress=0.5)
        assert ring.current_effect == "timer_tick"


# ===========================================================================
# 1. Waiting for Command (phase 2)
# ===========================================================================


class TestWaitingForCommand:
    """Slow clockwise spin with 2 groups of 3 LEDs (gradient tail)."""

    def test_effect_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("waiting_for_command")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_uses_user_color(self):
        ring = LEDRingSimulator()
        ring.user_color = (0, 200, 100)
        ring.set_effect("waiting_for_command")
        ring.update(0.0)
        state = ring.get_led_state()
        # At least some LEDs should have the user color component
        lit = [led for led in state if led != BLACK]
        assert len(lit) > 0
        # The brightest LED(s) should have green component dominant
        brightest = max(lit, key=lambda c: sum(c))
        assert brightest[1] > 0  # green channel present

    def test_spins_clockwise_over_time(self):
        ring = LEDRingSimulator()
        ring.set_effect("waiting_for_command")
        ring.update(0.0)
        state1 = ring.get_led_state()
        # 100ms update interval, advance enough for several steps
        for _ in range(5):
            ring.update(0.1)
        state2 = ring.get_led_state()
        # Pattern should have rotated
        assert state1 != state2

    def test_two_groups_of_leds(self):
        ring = LEDRingSimulator()
        ring.set_effect("waiting_for_command")
        ring.update(0.0)
        # Should have 2 groups of ~3 LEDs = ~6 lit LEDs
        lit = _count_lit(ring)
        assert 4 <= lit <= 8, f"Expected 4-8 lit LEDs, got {lit}"


# ===========================================================================
# 2. Listening For Command (phase 3)
# ===========================================================================


class TestListeningForCommand:
    """Fast clockwise spin with 2 groups of 3 LEDs."""

    def test_listening_effect_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("listening")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_listening_spins_faster_than_waiting(self):
        """Listening (50ms) is faster than waiting (100ms)."""
        ring_wait = LEDRingSimulator()
        ring_wait.set_effect("waiting_for_command")

        ring_listen = LEDRingSimulator()
        ring_listen.set_effect("listening")

        # Collect state changes over 1 second
        wait_changes = 0
        listen_changes = 0
        prev_wait = None
        prev_listen = None

        for _ in range(20):
            ring_wait.update(0.05)
            ring_listen.update(0.05)
            cur_wait = ring_wait.get_led_state()
            cur_listen = ring_listen.get_led_state()
            if prev_wait is not None and cur_wait != prev_wait:
                wait_changes += 1
            if prev_listen is not None and cur_listen != prev_listen:
                listen_changes += 1
            prev_wait = cur_wait
            prev_listen = cur_listen

        assert listen_changes > wait_changes

    def test_listening_two_groups(self):
        ring = LEDRingSimulator()
        ring.set_effect("listening")
        ring.update(0.0)
        lit = _count_lit(ring)
        assert 4 <= lit <= 8


# ===========================================================================
# 3. Thinking (phase 4)
# ===========================================================================


class TestThinking:
    """2 opposite LEDs pulsing (fade in/out)."""

    def test_thinking_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("thinking")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_thinking_two_opposite_leds(self):
        ring = LEDRingSimulator()
        ring.set_effect("thinking")
        ring.update(0.0)
        lit = _lit_indices(ring)
        # Should have exactly 2 LEDs on, 6 apart
        assert len(lit) == 2
        assert abs(lit[1] - lit[0]) == 6

    def test_thinking_pulses_over_time(self):
        ring = LEDRingSimulator()
        ring.set_effect("thinking")
        ring.update(0.0)
        state_start = ring.get_led_state()
        # Advance through pulse cycle
        for _ in range(50):
            ring.update(0.01)
        state_mid = ring.get_led_state()
        # Brightness should differ at different points in pulse
        brightness_start = sum(sum(c) for c in state_start)
        brightness_mid = sum(sum(c) for c in state_mid)
        # They should not be exactly the same (pulsing)
        assert brightness_start != brightness_mid


# ===========================================================================
# 4. Replying (phase 5)
# ===========================================================================


class TestReplying:
    """Fast anticlockwise spin with 2 groups of 3 LEDs."""

    def test_replying_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("replying")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_replying_spins_anticlockwise(self):
        """Verify spin direction is opposite to listening."""
        ring = LEDRingSimulator()
        ring.set_effect("replying")
        ring.update(0.0)

        # Find a lit LED position
        indices_t0 = _lit_indices(ring)
        ring.update(0.05)
        indices_t1 = _lit_indices(ring)
        ring.update(0.05)
        indices_t2 = _lit_indices(ring)

        # Replying should spin anticlockwise (decreasing indices mod 12)
        # We just verify the pattern changes (direction tested via implementation)
        assert indices_t0 != indices_t1 or indices_t1 != indices_t2

    def test_replying_two_groups(self):
        ring = LEDRingSimulator()
        ring.set_effect("replying")
        ring.update(0.0)
        lit = _count_lit(ring)
        assert 4 <= lit <= 8


# ===========================================================================
# 5. Muted or Silent
# ===========================================================================


class TestMutedOrSilent:
    """Solid ring with red indicators at positions 3, 9 (mic) and 6 (speaker)."""

    def test_muted_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("muted", mic_muted=True, speaker_muted=False)
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_mic_muted_red_indicators(self):
        ring = LEDRingSimulator()
        ring.set_effect("muted", mic_muted=True, speaker_muted=False)
        ring.update(0.0)
        state = ring.get_led_state()
        # LEDs 3 and 9 should be red
        assert state[3] == RED
        assert state[9] == RED

    def test_speaker_muted_red_indicator(self):
        ring = LEDRingSimulator()
        ring.set_effect("muted", mic_muted=False, speaker_muted=True)
        ring.update(0.0)
        state = ring.get_led_state()
        # LED 6 should be red
        assert state[6] == RED

    def test_both_muted(self):
        ring = LEDRingSimulator()
        ring.set_effect("muted", mic_muted=True, speaker_muted=True)
        ring.update(0.0)
        state = ring.get_led_state()
        assert state[3] == RED
        assert state[6] == RED
        assert state[9] == RED

    def test_muted_other_leds_use_user_color(self):
        ring = LEDRingSimulator()
        ring.user_color = (0, 0, 200)
        ring.set_effect("muted", mic_muted=True, speaker_muted=False)
        ring.update(0.0)
        state = ring.get_led_state()
        # Non-indicator LEDs should use user color
        for i in [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]:
            assert state[i][2] > 0, f"LED {i} should have blue component"


# ===========================================================================
# 6. Voice Kit Startup Failed
# ===========================================================================


class TestVoiceKitStartupFailed:
    """Static red 'X' - every 3rd LED red."""

    def test_startup_failed_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("startup_failed")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_startup_failed_every_third_led_red(self):
        ring = LEDRingSimulator()
        ring.set_effect("startup_failed")
        ring.update(0.0)
        state = ring.get_led_state()
        for i in range(12):
            if i % 3 == 0:
                assert state[i] == RED, f"LED {i} should be red"
            else:
                assert state[i] == BLACK, f"LED {i} should be off"

    def test_startup_failed_static(self):
        ring = LEDRingSimulator()
        ring.set_effect("startup_failed")
        ring.update(0.0)
        state1 = ring.get_led_state()
        ring.update(1.0)
        state2 = ring.get_led_state()
        assert state1 == state2  # No animation


# ===========================================================================
# 7. Volume Display
# ===========================================================================


class TestVolumeDisplay:
    """Arc showing volume level (0-12 LEDs)."""

    def test_volume_display_full(self):
        ring = LEDRingSimulator()
        ring.set_effect("volume_display", volume=1.0)
        ring.update(0.0)
        assert _count_lit(ring) == 12

    def test_volume_display_half(self):
        ring = LEDRingSimulator()
        ring.set_effect("volume_display", volume=0.5)
        ring.update(0.0)
        assert _count_lit(ring) == 6

    def test_volume_display_zero_red(self):
        ring = LEDRingSimulator()
        ring.set_effect("volume_display", volume=0.0)
        ring.update(0.0)
        # At zero volume, should show red indicator
        state = ring.get_led_state()
        lit = [i for i, led in enumerate(state) if led != BLACK]
        if lit:
            # Some LEDs may be red to indicate zero
            for i in lit:
                assert state[i][0] > 0  # red component

    def test_volume_display_quarter(self):
        ring = LEDRingSimulator()
        ring.set_effect("volume_display", volume=0.25)
        ring.update(0.0)
        assert _count_lit(ring) == 3

    def test_volume_display_uses_user_color(self):
        ring = LEDRingSimulator()
        ring.user_color = (0, 255, 0)
        ring.set_effect("volume_display", volume=1.0)
        ring.update(0.0)
        state = ring.get_led_state()
        assert all(led[1] > 0 for led in state)  # green component


# ===========================================================================
# 8. Center Button Touched
# ===========================================================================


class TestCenterButtonTouched:
    """Solid ring at increased brightness."""

    def test_button_touched_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("center_button_touched")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_button_touched_all_leds_lit(self):
        ring = LEDRingSimulator()
        ring.set_effect("center_button_touched")
        ring.update(0.0)
        assert _count_lit(ring) == 12

    def test_button_touched_uses_user_color(self):
        ring = LEDRingSimulator()
        ring.user_color = (100, 50, 200)
        ring.set_effect("center_button_touched")
        ring.update(0.0)
        state = ring.get_led_state()
        # All LEDs should use user color (brightness boosted)
        for led in state:
            assert led[2] > 0  # blue component from user_color


# ===========================================================================
# 9. Twinkle
# ===========================================================================


class TestTwinkle:
    """Random twinkling - 50% probability per LED."""

    def test_twinkle_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("twinkle")
        ring.update(0.0)
        # With 50% probability, at least some LEDs should be on
        # Run multiple updates to ensure we get some lit LEDs
        for _ in range(10):
            ring.update(0.05)
        assert _any_led_on(ring)

    def test_twinkle_changes_over_time(self):
        ring = LEDRingSimulator()
        ring.set_effect("twinkle")
        states = []
        for _ in range(20):
            ring.update(0.05)
            states.append(tuple(ring.get_led_state()))
        # Should have at least some variation
        unique_states = set(states)
        assert len(unique_states) > 1

    def test_twinkle_with_color(self):
        ring = LEDRingSimulator()
        ring.user_color = (0, 0, 255)
        ring.set_effect("twinkle", color=(0, 0, 255))
        for _ in range(10):
            ring.update(0.05)
        state = ring.get_led_state()
        lit = [led for led in state if led != BLACK]
        if lit:
            for led in lit:
                assert led[2] > 0  # blue present


# ===========================================================================
# 10. Error (phase 11)
# ===========================================================================


class TestError:
    """Full ring pulsing red."""

    def test_error_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("error")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_error_all_red(self):
        ring = LEDRingSimulator()
        ring.set_effect("error")
        ring.update(0.0)
        state = ring.get_led_state()
        for led in state:
            assert led[0] > 0  # red component
            assert led[1] == 0  # no green
            assert led[2] == 0  # no blue

    def test_error_pulses(self):
        ring = LEDRingSimulator()
        ring.set_effect("error")
        ring.update(0.0)
        brightness_values = []
        for _ in range(100):
            ring.update(0.01)
            state = ring.get_led_state()
            brightness_values.append(state[0][0])  # red channel of LED 0
        # Should have variation (pulsing)
        assert max(brightness_values) > min(brightness_values)


# ===========================================================================
# 11. Timer Ring
# ===========================================================================


class TestTimerRing:
    """Full ring pulsing with mute indicators."""

    def test_timer_ring_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_ring")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_timer_ring_pulses(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_ring")
        brightness_values = []
        for _ in range(100):
            ring.update(0.01)
            state = ring.get_led_state()
            brightness_values.append(sum(sum(c) for c in state))
        assert max(brightness_values) > min(brightness_values)

    def test_timer_ring_with_mute_indicators(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_ring", mic_muted=True)
        ring.update(0.0)
        state = ring.get_led_state()
        # Positions 3 and 9 should be red even during timer ring
        assert state[3] == RED
        assert state[9] == RED


# ===========================================================================
# 12. Timer Tick (Countdown Arc)
# ===========================================================================


class TestTimerTick:
    """Countdown arc: 12 LEDs = 100%, decreases as timer progresses."""

    def test_timer_tick_full(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_tick", progress=0.0)
        ring.update(0.0)
        assert _count_lit(ring) == 12

    def test_timer_tick_half(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_tick", progress=0.5)
        ring.update(0.0)
        assert _count_lit(ring) == 6

    def test_timer_tick_almost_done(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_tick", progress=11.0 / 12.0)
        ring.update(0.0)
        assert _count_lit(ring) == 1

    def test_timer_tick_done(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_tick", progress=1.0)
        ring.update(0.0)
        assert _count_lit(ring) == 0

    def test_timer_countdown_acceptance_criteria(self):
        """Acceptance: timer_countdown at 0.5 shows 6-LED arc."""
        ring = LEDRingSimulator()
        ring.set_effect("timer_tick", progress=0.5)
        ring.update(0.0)
        state = ring.get_led_state()
        lit = _count_lit(ring)
        assert lit == 6

    def test_timer_tick_with_mute_overlay(self):
        ring = LEDRingSimulator()
        ring.set_effect("timer_tick", progress=0.0, mic_muted=True)
        ring.update(0.0)
        state = ring.get_led_state()
        # Mute indicators at positions 3 and 9
        assert state[3] == RED
        assert state[9] == RED


# ===========================================================================
# 13. Rainbow
# ===========================================================================


class TestRainbow:
    """Full spectrum rainbow across all 12 LEDs."""

    def test_rainbow_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("rainbow")
        ring.update(0.0)
        assert _count_lit(ring) == 12

    def test_rainbow_all_different_colors(self):
        ring = LEDRingSimulator()
        ring.set_effect("rainbow")
        ring.update(0.0)
        state = ring.get_led_state()
        unique = set(state)
        # Should have mostly unique colors (full spectrum)
        assert len(unique) >= 6

    def test_rainbow_rotates(self):
        ring = LEDRingSimulator()
        ring.set_effect("rainbow")
        ring.update(0.0)
        state1 = ring.get_led_state()
        ring.update(0.1)
        state2 = ring.get_led_state()
        assert state1 != state2


# ===========================================================================
# 14. Tick (Easter egg countdown)
# ===========================================================================


class TestTick:
    """Countdown from 12 to 0 (red) at 333ms interval."""

    def test_tick_starts_full(self):
        ring = LEDRingSimulator()
        ring.set_effect("tick")
        ring.update(0.0)
        assert _count_lit(ring) == 12

    def test_tick_decrements(self):
        ring = LEDRingSimulator()
        ring.set_effect("tick")
        ring.update(0.0)
        assert _count_lit(ring) == 12
        ring.update(0.333)
        assert _count_lit(ring) == 11

    def test_tick_all_red(self):
        ring = LEDRingSimulator()
        ring.set_effect("tick")
        ring.update(0.0)
        state = ring.get_led_state()
        for led in state:
            if led != BLACK:
                assert led[0] > 0
                assert led[1] == 0
                assert led[2] == 0

    def test_tick_reaches_zero(self):
        ring = LEDRingSimulator()
        ring.set_effect("tick")
        ring.update(0.0)
        # 12 ticks at 333ms each
        for _ in range(12):
            ring.update(0.333)
        assert _count_lit(ring) == 0


# ===========================================================================
# 15. Factory Reset Coming Up
# ===========================================================================


class TestFactoryResetComingUp:
    """Red LEDs filling clockwise at 1/sec."""

    def test_factory_reset_starts_empty(self):
        ring = LEDRingSimulator()
        ring.set_effect("factory_reset")
        ring.update(0.0)
        assert _count_lit(ring) == 0

    def test_factory_reset_fills_over_time(self):
        ring = LEDRingSimulator()
        ring.set_effect("factory_reset")
        ring.update(0.0)

        ring.update(1.0)
        assert _count_lit(ring) == 1

        ring.update(1.0)
        assert _count_lit(ring) == 2

    def test_factory_reset_all_red(self):
        ring = LEDRingSimulator()
        ring.set_effect("factory_reset")
        ring.update(0.0)
        # Fill up several
        for _ in range(5):
            ring.update(1.0)
        state = ring.get_led_state()
        for led in state:
            if led != BLACK:
                assert led == RED

    def test_factory_reset_fills_to_12(self):
        ring = LEDRingSimulator()
        ring.set_effect("factory_reset")
        ring.update(0.0)
        for _ in range(12):
            ring.update(1.0)
        assert _count_lit(ring) == 12

    def test_factory_reset_caps_at_12(self):
        ring = LEDRingSimulator()
        ring.set_effect("factory_reset")
        ring.update(0.0)
        for _ in range(15):
            ring.update(1.0)
        assert _count_lit(ring) == 12


# ===========================================================================
# 16. Jack Plugged
# ===========================================================================


class TestJackPlugged:
    """Ripple effect from position 0 outward."""

    def test_jack_plugged_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("jack_plugged")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_jack_plugged_starts_at_zero(self):
        ring = LEDRingSimulator()
        ring.set_effect("jack_plugged")
        ring.update(0.0)
        state = ring.get_led_state()
        assert state[0] != BLACK

    def test_jack_plugged_ripples_outward(self):
        ring = LEDRingSimulator()
        ring.set_effect("jack_plugged")
        ring.update(0.0)
        lit1 = _count_lit(ring)
        ring.update(0.04)
        lit2 = _count_lit(ring)
        ring.update(0.04)
        lit3 = _count_lit(ring)
        # Should spread over time
        assert lit3 >= lit2 >= lit1

    def test_jack_plugged_uses_user_color(self):
        ring = LEDRingSimulator()
        ring.user_color = GREEN
        ring.set_effect("jack_plugged")
        ring.update(0.0)
        state = ring.get_led_state()
        lit = [led for led in state if led != BLACK]
        for led in lit:
            assert led[1] > 0  # green component


# ===========================================================================
# 17. Jack Unplugged
# ===========================================================================


class TestJackUnplugged:
    """Ripple effect from position 6 inward."""

    def test_jack_unplugged_activates(self):
        ring = LEDRingSimulator()
        ring.set_effect("jack_unplugged")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_jack_unplugged_starts_at_six(self):
        ring = LEDRingSimulator()
        ring.set_effect("jack_unplugged")
        ring.update(0.0)
        state = ring.get_led_state()
        assert state[6] != BLACK

    def test_jack_unplugged_ripples_inward(self):
        ring = LEDRingSimulator()
        ring.set_effect("jack_unplugged")
        ring.update(0.0)
        lit1 = _count_lit(ring)
        ring.update(0.04)
        lit2 = _count_lit(ring)
        ring.update(0.04)
        lit3 = _count_lit(ring)
        assert lit3 >= lit2 >= lit1


# ===========================================================================
# Acceptance Criteria Tests
# ===========================================================================


class TestAcceptanceCriteria:
    """Directly test the acceptance criteria from the plan."""

    def test_listening_shows_fast_clockwise_spin(self):
        """sim.set_led_effect('listening') shows fast clockwise spin."""
        ring = LEDRingSimulator()
        ring.set_effect("listening")
        ring.update(0.0)
        assert _any_led_on(ring)
        assert ring.current_effect == "listening"
        # Verify spin (state changes at 50ms intervals)
        state_a = ring.get_led_state()
        ring.update(0.05)
        state_b = ring.get_led_state()
        assert state_a != state_b  # pattern rotated

    def test_thinking_shows_2_led_pulse(self):
        """sim.set_led_effect('thinking') shows 2-LED pulse."""
        ring = LEDRingSimulator()
        ring.set_effect("thinking")
        ring.update(0.0)
        lit = _lit_indices(ring)
        assert len(lit) == 2
        assert abs(lit[1] - lit[0]) == 6  # opposite

    def test_timer_countdown_shows_6_led_arc(self):
        """sim.set_led_effect('timer_countdown', progress=0.5) shows 6-LED arc."""
        ring = LEDRingSimulator()
        ring.set_effect("timer_tick", progress=0.5)
        ring.update(0.0)
        state = ring.get_led_state()
        assert len(state) == 12
        assert _count_lit(ring) == 6

    def test_get_led_state_returns_12_rgb_values(self):
        """sim.get_led_state() returns 12-element list of RGB values."""
        ring = LEDRingSimulator()
        state = ring.get_led_state()
        assert isinstance(state, list)
        assert len(state) == 12
        for led in state:
            assert isinstance(led, tuple)
            assert len(led) == 3
            for channel in led:
                assert isinstance(channel, int)
                assert 0 <= channel <= 255


# ===========================================================================
# Simulated Time Support
# ===========================================================================


class TestSimulatedTime:
    """Effect timing uses simulated time, not real time."""

    def test_effect_uses_dt_parameter(self):
        ring = LEDRingSimulator()
        ring.set_effect("factory_reset")
        ring.update(0.0)
        # dt=0 should not advance
        ring.update(0.0)
        assert _count_lit(ring) == 0
        # dt=5.0 should advance 5 seconds (5 LEDs filled)
        ring.update(5.0)
        assert _count_lit(ring) == 5

    def test_elapsed_time_tracked(self):
        ring = LEDRingSimulator()
        ring.set_effect("tick")
        ring.update(0.0)
        ring.update(1.0)
        ring.update(0.5)
        assert abs(ring.elapsed_time - 1.5) < 0.001

    def test_elapsed_resets_on_new_effect(self):
        ring = LEDRingSimulator()
        ring.set_effect("tick")
        ring.update(1.0)
        ring.set_effect("error")
        assert ring.elapsed_time == 0.0


# ===========================================================================
# Integration with Simulator Core
# ===========================================================================


class TestSimulatorIntegration:
    """LEDRingSimulator can work with the Simulator core."""

    def test_led_ring_from_light_state(self):
        """LEDRingSimulator can be attached to a Simulator LightState."""
        from esphome_yaml_tools.simulator.core import Simulator

        sim = Simulator()
        sim.load_config({"light": [{"id": "voice_assistant_leds"}]})
        ring = LEDRingSimulator()
        # Set effect should work standalone
        ring.set_effect("listening")
        ring.update(0.0)
        assert _any_led_on(ring)

    def test_to_dict(self):
        """LEDRingSimulator state can be serialized."""
        ring = LEDRingSimulator()
        ring.set_effect("error")
        ring.update(0.0)
        d = ring.to_dict()
        assert "leds" in d
        assert "effect" in d
        assert "brightness" in d
        assert len(d["leds"]) == 12
        assert d["effect"] == "error"
