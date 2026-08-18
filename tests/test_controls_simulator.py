"""Tests for button, rotary dial, mute switch, and jack sensor simulator.

TDD approach: tests written first, implementation follows.

Voice PE controls reference:
- Center button (GPIO0): single/double/triple press, long press, factory reset
- Rotary dial (GPIO16/18): volume/hue adjustment
- Mute switch (GPIO3): mic mute
- Jack sensor (GPIO17): audio jack detection
"""

from __future__ import annotations

import pytest

from esphome_yaml_tools.simulator.controls import (
    CenterButton,
    RotaryDial,
    MuteSwitch,
    JackSensor,
    ControlEvent,
    ControlEventType,
)


# =============================================================================
# CenterButton Tests
# =============================================================================


class TestCenterButtonInit:
    """Test CenterButton initialization and defaults."""

    def test_initial_state_not_pressed(self) -> None:
        btn = CenterButton()
        assert btn.is_pressed is False

    def test_initial_event_history_empty(self) -> None:
        btn = CenterButton()
        assert btn.get_event_history() == []

    def test_default_multi_click_window(self) -> None:
        btn = CenterButton()
        assert btn.multi_click_window == 0.25  # 250ms

    def test_default_long_press_threshold(self) -> None:
        btn = CenterButton()
        assert btn.long_press_threshold == 1.0  # 1 second

    def test_custom_multi_click_window(self) -> None:
        btn = CenterButton(multi_click_window=0.5)
        assert btn.multi_click_window == 0.5

    def test_custom_long_press_threshold(self) -> None:
        btn = CenterButton(long_press_threshold=2.0)
        assert btn.long_press_threshold == 2.0


class TestCenterButtonSinglePress:
    """Test single press detection."""

    def test_single_press_short_duration(self) -> None:
        """Short press (< 1s) followed by multi_click_window timeout = single press."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(0.1)
        btn.release()
        # Wait for multi-click window to expire
        btn.advance_time(0.3)
        events = btn.get_event_history()
        single_events = [e for e in events if e.event_type == ControlEventType.SINGLE_PRESS]
        assert len(single_events) == 1

    def test_single_press_duration_500ms(self) -> None:
        """Acceptance: sim.press_button(duration=0.5) triggers single press."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(0.5)
        btn.release()
        btn.advance_time(0.3)
        events = btn.get_event_history()
        single_events = [e for e in events if e.event_type == ControlEventType.SINGLE_PRESS]
        assert len(single_events) == 1

    def test_press_and_release_convenience(self) -> None:
        """Test press_and_release convenience method."""
        btn = CenterButton()
        btn.press_and_release(duration=0.5)
        btn.advance_time(0.3)
        events = btn.get_event_history()
        single_events = [e for e in events if e.event_type == ControlEventType.SINGLE_PRESS]
        assert len(single_events) == 1

    def test_single_press_emits_pressed_event(self) -> None:
        """Pressing the button should emit a PRESSED event."""
        btn = CenterButton()
        btn.press()
        events = btn.get_event_history()
        pressed = [e for e in events if e.event_type == ControlEventType.PRESSED]
        assert len(pressed) == 1

    def test_single_press_emits_released_event(self) -> None:
        """Releasing the button should emit a RELEASED event."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(0.1)
        btn.release()
        events = btn.get_event_history()
        released = [e for e in events if e.event_type == ControlEventType.RELEASED]
        assert len(released) == 1


class TestCenterButtonDoublePress:
    """Test double press detection (2 presses within multi_click_window)."""

    def test_double_press(self) -> None:
        """Two presses within 250ms = double press."""
        btn = CenterButton()
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)  # gap < 250ms
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.3)  # wait for window to expire
        events = btn.get_event_history()
        double_events = [e for e in events if e.event_type == ControlEventType.DOUBLE_PRESS]
        assert len(double_events) == 1

    def test_double_press_no_single(self) -> None:
        """Double press should NOT emit single press."""
        btn = CenterButton()
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.3)
        events = btn.get_event_history()
        single_events = [e for e in events if e.event_type == ControlEventType.SINGLE_PRESS]
        assert len(single_events) == 0

    def test_double_press_at_window_boundary(self) -> None:
        """Two presses right at the 250ms boundary."""
        btn = CenterButton()
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.24)  # just under 250ms
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.3)
        events = btn.get_event_history()
        double_events = [e for e in events if e.event_type == ControlEventType.DOUBLE_PRESS]
        assert len(double_events) == 1


class TestCenterButtonTriplePress:
    """Test triple press detection (3 presses within multi_click_window)."""

    def test_triple_press(self) -> None:
        """Three presses within multi_click_window = triple press."""
        btn = CenterButton()
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.3)
        events = btn.get_event_history()
        triple_events = [e for e in events if e.event_type == ControlEventType.TRIPLE_PRESS]
        assert len(triple_events) == 1

    def test_triple_press_no_single_or_double(self) -> None:
        """Triple press should NOT emit single or double."""
        btn = CenterButton()
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.3)
        events = btn.get_event_history()
        single = [e for e in events if e.event_type == ControlEventType.SINGLE_PRESS]
        double = [e for e in events if e.event_type == ControlEventType.DOUBLE_PRESS]
        assert len(single) == 0
        assert len(double) == 0


class TestCenterButtonLongPress:
    """Test long press detection (hold >= 1s)."""

    def test_long_press(self) -> None:
        """Hold for 1+ seconds = long press."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(1.1)
        btn.release()
        btn.advance_time(0.3)
        events = btn.get_event_history()
        long_events = [e for e in events if e.event_type == ControlEventType.LONG_PRESS]
        assert len(long_events) == 1

    def test_long_press_emitted_on_release(self) -> None:
        """Long press event emitted when button released after threshold."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(1.5)
        # Not yet released - check for long press
        events_before = [
            e for e in btn.get_event_history() if e.event_type == ControlEventType.LONG_PRESS
        ]
        assert len(events_before) == 1  # emitted when threshold crossed during advance_time
        btn.release()

    def test_long_press_not_single(self) -> None:
        """Long press should NOT emit single press."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(1.1)
        btn.release()
        btn.advance_time(0.3)
        events = btn.get_event_history()
        single = [e for e in events if e.event_type == ControlEventType.SINGLE_PRESS]
        assert len(single) == 0

    def test_exact_threshold_is_long_press(self) -> None:
        """Holding for exactly 1.0s = long press."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(1.0)
        events = btn.get_event_history()
        long_events = [e for e in events if e.event_type == ControlEventType.LONG_PRESS]
        assert len(long_events) == 1


class TestCenterButtonFactoryReset:
    """Test factory reset sequence (10-22s warning, 22s+ confirm)."""

    def test_factory_reset_warning_at_10s(self) -> None:
        """Holding 10+ seconds emits factory reset warning."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(10.0)
        events = btn.get_event_history()
        warnings = [e for e in events if e.event_type == ControlEventType.FACTORY_RESET_WARNING]
        assert len(warnings) == 1

    def test_factory_reset_confirm_at_22s(self) -> None:
        """Holding 22+ seconds emits factory reset confirm."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(22.0)
        events = btn.get_event_history()
        confirms = [e for e in events if e.event_type == ControlEventType.FACTORY_RESET_CONFIRM]
        assert len(confirms) == 1

    def test_factory_reset_warning_before_confirm(self) -> None:
        """Warning must come before confirm."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(22.0)
        events = btn.get_event_history()
        warning_idx = next(
            i
            for i, e in enumerate(events)
            if e.event_type == ControlEventType.FACTORY_RESET_WARNING
        )
        confirm_idx = next(
            i
            for i, e in enumerate(events)
            if e.event_type == ControlEventType.FACTORY_RESET_CONFIRM
        )
        assert warning_idx < confirm_idx

    def test_release_before_22s_no_confirm(self) -> None:
        """Releasing between 10-22s: warning but no confirm."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(15.0)
        btn.release()
        events = btn.get_event_history()
        warnings = [e for e in events if e.event_type == ControlEventType.FACTORY_RESET_WARNING]
        confirms = [e for e in events if e.event_type == ControlEventType.FACTORY_RESET_CONFIRM]
        assert len(warnings) == 1
        assert len(confirms) == 0

    def test_release_before_10s_no_warning(self) -> None:
        """Releasing before 10s: no warning or confirm."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(5.0)
        btn.release()
        events = btn.get_event_history()
        warnings = [e for e in events if e.event_type == ControlEventType.FACTORY_RESET_WARNING]
        confirms = [e for e in events if e.event_type == ControlEventType.FACTORY_RESET_CONFIRM]
        assert len(warnings) == 0
        assert len(confirms) == 0

    def test_factory_reset_no_duplicate_warning(self) -> None:
        """Only one warning event even if time advances past 10s multiple times."""
        btn = CenterButton()
        btn.press()
        btn.advance_time(10.0)
        btn.advance_time(2.0)
        btn.advance_time(2.0)
        events = btn.get_event_history()
        warnings = [e for e in events if e.event_type == ControlEventType.FACTORY_RESET_WARNING]
        assert len(warnings) == 1


class TestCenterButtonCallbacks:
    """Test event callback system."""

    def test_callback_on_single_press(self) -> None:
        received: list[ControlEvent] = []
        btn = CenterButton()
        btn.on_event(ControlEventType.SINGLE_PRESS, lambda e: received.append(e))
        btn.press_and_release(duration=0.1)
        btn.advance_time(0.3)
        assert len(received) == 1
        assert received[0].event_type == ControlEventType.SINGLE_PRESS

    def test_callback_on_double_press(self) -> None:
        received: list[ControlEvent] = []
        btn = CenterButton()
        btn.on_event(ControlEventType.DOUBLE_PRESS, lambda e: received.append(e))
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.3)
        assert len(received) == 1

    def test_callback_on_long_press(self) -> None:
        received: list[ControlEvent] = []
        btn = CenterButton()
        btn.on_event(ControlEventType.LONG_PRESS, lambda e: received.append(e))
        btn.press()
        btn.advance_time(1.5)
        assert len(received) == 1

    def test_multiple_callbacks_for_same_event(self) -> None:
        count = [0, 0]
        btn = CenterButton()
        btn.on_event(ControlEventType.SINGLE_PRESS, lambda e: count.__setitem__(0, count[0] + 1))
        btn.on_event(ControlEventType.SINGLE_PRESS, lambda e: count.__setitem__(1, count[1] + 1))
        btn.press_and_release(duration=0.1)
        btn.advance_time(0.3)
        assert count == [1, 1]


class TestCenterButtonEdgeCases:
    """Edge case tests for center button."""

    def test_release_without_press_is_noop(self) -> None:
        btn = CenterButton()
        btn.release()  # should not raise
        assert btn.get_event_history() == []

    def test_double_press_then_single_press(self) -> None:
        """Double press followed by separate single press."""
        btn = CenterButton()
        # Double press
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.3)

        # Single press
        btn.press_and_release(duration=0.1)
        btn.advance_time(0.3)

        events = btn.get_event_history()
        double = [e for e in events if e.event_type == ControlEventType.DOUBLE_PRESS]
        single = [e for e in events if e.event_type == ControlEventType.SINGLE_PRESS]
        assert len(double) == 1
        assert len(single) == 1

    def test_press_while_already_pressed_is_noop(self) -> None:
        btn = CenterButton()
        btn.press()
        btn.press()  # Should be ignored
        btn.advance_time(0.1)
        btn.release()
        btn.advance_time(0.3)
        pressed = [e for e in btn.get_event_history() if e.event_type == ControlEventType.PRESSED]
        assert len(pressed) == 1

    def test_reset_clears_state(self) -> None:
        btn = CenterButton()
        btn.press_and_release(duration=0.1)
        btn.advance_time(0.3)
        btn.reset()
        assert btn.is_pressed is False
        assert btn.get_event_history() == []

    def test_snapshot(self) -> None:
        btn = CenterButton()
        snap = btn.snapshot()
        assert "is_pressed" in snap
        assert "pending_clicks" in snap
        assert "hold_time" in snap


class TestCenterButtonEventData:
    """Test event data content."""

    def test_single_press_event_has_duration(self) -> None:
        btn = CenterButton()
        btn.press_and_release(duration=0.3)
        btn.advance_time(0.3)
        events = [
            e for e in btn.get_event_history() if e.event_type == ControlEventType.SINGLE_PRESS
        ]
        assert len(events) == 1
        assert "duration" in events[0].data

    def test_long_press_event_has_duration(self) -> None:
        btn = CenterButton()
        btn.press()
        btn.advance_time(1.5)
        btn.release()
        events = [e for e in btn.get_event_history() if e.event_type == ControlEventType.LONG_PRESS]
        assert len(events) == 1
        assert "duration" in events[0].data
        assert events[0].data["duration"] >= 1.0

    def test_double_press_event_has_click_count(self) -> None:
        btn = CenterButton()
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.1)
        btn.press_and_release(duration=0.05)
        btn.advance_time(0.3)
        events = [
            e for e in btn.get_event_history() if e.event_type == ControlEventType.DOUBLE_PRESS
        ]
        assert len(events) == 1
        assert events[0].data.get("click_count") == 2


# =============================================================================
# RotaryDial Tests
# =============================================================================


class TestRotaryDialInit:
    """Test RotaryDial initialization."""

    def test_default_volume(self) -> None:
        dial = RotaryDial()
        assert dial.volume == 0.5  # 50%

    def test_default_hue(self) -> None:
        dial = RotaryDial()
        assert dial.hue == 0.0

    def test_custom_initial_volume(self) -> None:
        dial = RotaryDial(initial_volume=0.8)
        assert dial.volume == 0.8

    def test_default_step_size(self) -> None:
        dial = RotaryDial()
        assert dial.volume_step == 0.05  # 5% matching Voice PE


class TestRotaryDialVolume:
    """Test volume adjustment via dial rotation."""

    def test_clockwise_increases_volume(self) -> None:
        dial = RotaryDial(initial_volume=0.5)
        dial.rotate_clockwise()
        assert dial.volume == pytest.approx(0.55, abs=0.001)

    def test_anticlockwise_decreases_volume(self) -> None:
        dial = RotaryDial(initial_volume=0.5)
        dial.rotate_anticlockwise()
        assert dial.volume == pytest.approx(0.45, abs=0.001)

    def test_volume_clamped_at_max(self) -> None:
        dial = RotaryDial(initial_volume=0.98)
        dial.rotate_clockwise()
        assert dial.volume == 1.0

    def test_volume_clamped_at_min(self) -> None:
        dial = RotaryDial(initial_volume=0.02)
        dial.rotate_anticlockwise()
        assert dial.volume == 0.0

    def test_multiple_rotations(self) -> None:
        dial = RotaryDial(initial_volume=0.5)
        dial.rotate_clockwise()
        dial.rotate_clockwise()
        dial.rotate_clockwise()
        assert dial.volume == pytest.approx(0.65, abs=0.001)

    def test_volume_change_emits_event(self) -> None:
        dial = RotaryDial()
        dial.rotate_clockwise()
        events = dial.get_event_history()
        vol_events = [e for e in events if e.event_type == ControlEventType.VOLUME_CHANGED]
        assert len(vol_events) == 1

    def test_volume_event_has_data(self) -> None:
        dial = RotaryDial(initial_volume=0.5)
        dial.rotate_clockwise()
        events = [
            e for e in dial.get_event_history() if e.event_type == ControlEventType.VOLUME_CHANGED
        ]
        assert events[0].data["volume"] == pytest.approx(0.55, abs=0.001)
        assert events[0].data["direction"] == "clockwise"

    def test_rotate_direction_string(self) -> None:
        """Test rotate() with direction string."""
        dial = RotaryDial(initial_volume=0.5)
        dial.rotate(direction="clockwise")
        assert dial.volume == pytest.approx(0.55, abs=0.001)
        dial.rotate(direction="anticlockwise")
        assert dial.volume == pytest.approx(0.5, abs=0.001)


class TestRotaryDialHue:
    """Test hue adjustment when button is pressed."""

    def test_hue_clockwise_while_pressed(self) -> None:
        dial = RotaryDial()
        dial.set_button_pressed(True)
        dial.rotate_clockwise()
        assert dial.hue > 0.0

    def test_hue_anticlockwise_while_pressed(self) -> None:
        dial = RotaryDial()
        dial.hue = 180.0
        dial.set_button_pressed(True)
        dial.rotate_anticlockwise()
        assert dial.hue < 180.0

    def test_hue_wraps_around_360(self) -> None:
        dial = RotaryDial()
        dial.hue = 350.0
        dial.set_button_pressed(True)
        dial.rotate_clockwise()
        # Should wrap
        assert 0.0 <= dial.hue < 360.0

    def test_hue_wraps_below_zero(self) -> None:
        dial = RotaryDial()
        dial.hue = 5.0
        dial.set_button_pressed(True)
        dial.rotate_anticlockwise()
        assert 0.0 <= dial.hue < 360.0

    def test_hue_emits_event(self) -> None:
        dial = RotaryDial()
        dial.set_button_pressed(True)
        dial.rotate_clockwise()
        events = [
            e for e in dial.get_event_history() if e.event_type == ControlEventType.HUE_CHANGED
        ]
        assert len(events) == 1
        assert "hue" in events[0].data

    def test_no_volume_change_when_pressed(self) -> None:
        """Rotation while pressed changes hue, not volume."""
        dial = RotaryDial(initial_volume=0.5)
        dial.set_button_pressed(True)
        dial.rotate_clockwise()
        assert dial.volume == 0.5  # unchanged

    def test_volume_resumes_after_unpress(self) -> None:
        """After releasing button, rotation changes volume again."""
        dial = RotaryDial(initial_volume=0.5)
        dial.set_button_pressed(True)
        dial.rotate_clockwise()  # hue change
        dial.set_button_pressed(False)
        dial.rotate_clockwise()  # volume change
        assert dial.volume == pytest.approx(0.55, abs=0.001)


class TestRotaryDialCallbacks:
    """Test rotary dial callbacks."""

    def test_callback_on_volume_change(self) -> None:
        received: list[ControlEvent] = []
        dial = RotaryDial()
        dial.on_event(ControlEventType.VOLUME_CHANGED, lambda e: received.append(e))
        dial.rotate_clockwise()
        assert len(received) == 1

    def test_callback_on_hue_change(self) -> None:
        received: list[ControlEvent] = []
        dial = RotaryDial()
        dial.on_event(ControlEventType.HUE_CHANGED, lambda e: received.append(e))
        dial.set_button_pressed(True)
        dial.rotate_clockwise()
        assert len(received) == 1


class TestRotaryDialEdgeCases:
    """Edge cases for rotary dial."""

    def test_reset(self) -> None:
        dial = RotaryDial(initial_volume=0.5)
        dial.rotate_clockwise()
        dial.reset()
        assert dial.volume == 0.5
        assert dial.get_event_history() == []

    def test_snapshot(self) -> None:
        dial = RotaryDial()
        snap = dial.snapshot()
        assert "volume" in snap
        assert "hue" in snap
        assert "button_pressed" in snap

    def test_custom_volume_step(self) -> None:
        dial = RotaryDial(volume_step=0.10)
        dial.rotate_clockwise()
        assert dial.volume == pytest.approx(0.6, abs=0.001)

    def test_custom_hue_step(self) -> None:
        dial = RotaryDial(hue_step=30.0)
        dial.set_button_pressed(True)
        dial.rotate_clockwise()
        assert dial.hue == pytest.approx(30.0, abs=0.1)


# =============================================================================
# MuteSwitch Tests
# =============================================================================


class TestMuteSwitchInit:
    """Test MuteSwitch initialization."""

    def test_default_not_muted(self) -> None:
        mute = MuteSwitch()
        assert mute.is_muted is False

    def test_event_history_empty(self) -> None:
        mute = MuteSwitch()
        assert mute.get_event_history() == []


class TestMuteSwitchBehavior:
    """Test mute switch behavior."""

    def test_set_muted_true(self) -> None:
        mute = MuteSwitch()
        mute.set_muted(True)
        assert mute.is_muted is True

    def test_set_muted_false(self) -> None:
        mute = MuteSwitch()
        mute.set_muted(True)
        mute.set_muted(False)
        assert mute.is_muted is False

    def test_mute_emits_event(self) -> None:
        mute = MuteSwitch()
        mute.set_muted(True)
        events = mute.get_event_history()
        mute_events = [e for e in events if e.event_type == ControlEventType.MUTED]
        assert len(mute_events) == 1

    def test_unmute_emits_event(self) -> None:
        mute = MuteSwitch()
        mute.set_muted(True)
        mute.set_muted(False)
        events = mute.get_event_history()
        unmute_events = [e for e in events if e.event_type == ControlEventType.UNMUTED]
        assert len(unmute_events) == 1

    def test_toggle_mute(self) -> None:
        mute = MuteSwitch()
        mute.toggle()
        assert mute.is_muted is True
        mute.toggle()
        assert mute.is_muted is False

    def test_set_same_state_no_event(self) -> None:
        """Setting mute to same state should not emit event."""
        mute = MuteSwitch()
        mute.set_muted(False)  # already False
        assert mute.get_event_history() == []

    def test_set_muted_true_when_already_muted_no_event(self) -> None:
        mute = MuteSwitch()
        mute.set_muted(True)
        mute.set_muted(True)  # already True
        events = [e for e in mute.get_event_history() if e.event_type == ControlEventType.MUTED]
        assert len(events) == 1  # only one event


class TestMuteSwitchCallbacks:
    """Test mute switch callbacks."""

    def test_callback_on_mute(self) -> None:
        received: list[ControlEvent] = []
        mute = MuteSwitch()
        mute.on_event(ControlEventType.MUTED, lambda e: received.append(e))
        mute.set_muted(True)
        assert len(received) == 1

    def test_callback_on_unmute(self) -> None:
        received: list[ControlEvent] = []
        mute = MuteSwitch()
        mute.on_event(ControlEventType.UNMUTED, lambda e: received.append(e))
        mute.set_muted(True)
        mute.set_muted(False)
        assert len(received) == 1


class TestMuteSwitchEdgeCases:
    """Edge cases for mute switch."""

    def test_reset(self) -> None:
        mute = MuteSwitch()
        mute.set_muted(True)
        mute.reset()
        assert mute.is_muted is False
        assert mute.get_event_history() == []

    def test_snapshot(self) -> None:
        mute = MuteSwitch()
        mute.set_muted(True)
        snap = mute.snapshot()
        assert snap == {"is_muted": True}


# =============================================================================
# JackSensor Tests
# =============================================================================


class TestJackSensorInit:
    """Test JackSensor initialization."""

    def test_default_not_plugged(self) -> None:
        jack = JackSensor()
        assert jack.is_plugged is False

    def test_event_history_empty(self) -> None:
        jack = JackSensor()
        assert jack.get_event_history() == []


class TestJackSensorBehavior:
    """Test jack sensor behavior."""

    def test_plug_in(self) -> None:
        jack = JackSensor()
        jack.set_plugged(True)
        assert jack.is_plugged is True

    def test_unplug(self) -> None:
        jack = JackSensor()
        jack.set_plugged(True)
        jack.set_plugged(False)
        assert jack.is_plugged is False

    def test_plug_emits_event(self) -> None:
        jack = JackSensor()
        jack.set_plugged(True)
        events = jack.get_event_history()
        plugged = [e for e in events if e.event_type == ControlEventType.JACK_PLUGGED]
        assert len(plugged) == 1

    def test_unplug_emits_event(self) -> None:
        jack = JackSensor()
        jack.set_plugged(True)
        jack.set_plugged(False)
        events = jack.get_event_history()
        unplugged = [e for e in events if e.event_type == ControlEventType.JACK_UNPLUGGED]
        assert len(unplugged) == 1

    def test_set_same_state_no_event(self) -> None:
        """Setting same plug state should not emit event."""
        jack = JackSensor()
        jack.set_plugged(False)  # already False
        assert jack.get_event_history() == []


class TestJackSensorCallbacks:
    """Test jack sensor callbacks."""

    def test_callback_on_plugged(self) -> None:
        received: list[ControlEvent] = []
        jack = JackSensor()
        jack.on_event(ControlEventType.JACK_PLUGGED, lambda e: received.append(e))
        jack.set_plugged(True)
        assert len(received) == 1

    def test_callback_on_unplugged(self) -> None:
        received: list[ControlEvent] = []
        jack = JackSensor()
        jack.on_event(ControlEventType.JACK_UNPLUGGED, lambda e: received.append(e))
        jack.set_plugged(True)
        jack.set_plugged(False)
        assert len(received) == 1


class TestJackSensorEdgeCases:
    """Edge cases for jack sensor."""

    def test_reset(self) -> None:
        jack = JackSensor()
        jack.set_plugged(True)
        jack.reset()
        assert jack.is_plugged is False
        assert jack.get_event_history() == []

    def test_snapshot(self) -> None:
        jack = JackSensor()
        jack.set_plugged(True)
        snap = jack.snapshot()
        assert snap == {"is_plugged": True}


# =============================================================================
# Integration with Simulator event dispatch
# =============================================================================


class TestSimulatorIntegration:
    """Test controls integration with Simulator event dispatch."""

    def test_center_button_dispatches_to_simulator(self) -> None:
        from esphome_yaml_tools.simulator.core import Simulator

        sim = Simulator()
        btn = CenterButton()
        dispatched: list[str] = []

        # Wire button events to simulator dispatch
        btn.on_event(
            ControlEventType.SINGLE_PRESS,
            lambda e: sim.dispatch_event("button_single_press", e.data),
        )
        sim.register_event_handler("button_single_press", lambda ev: dispatched.append(ev.name))

        btn.press_and_release(duration=0.1)
        btn.advance_time(0.3)

        assert "button_single_press" in dispatched

    def test_mute_dispatches_to_simulator(self) -> None:
        from esphome_yaml_tools.simulator.core import Simulator

        sim = Simulator()
        mute = MuteSwitch()

        dispatched: list[str] = []
        mute.on_event(
            ControlEventType.MUTED,
            lambda e: sim.dispatch_event("mute_on", e.data),
        )
        sim.register_event_handler("mute_on", lambda ev: dispatched.append(ev.name))

        mute.set_muted(True)
        assert "mute_on" in dispatched

    def test_dial_volume_dispatches_to_simulator(self) -> None:
        from esphome_yaml_tools.simulator.core import Simulator

        sim = Simulator()
        dial = RotaryDial()

        dispatched: list[dict] = []
        dial.on_event(
            ControlEventType.VOLUME_CHANGED,
            lambda e: sim.dispatch_event("volume_changed", e.data),
        )
        sim.register_event_handler("volume_changed", lambda ev: dispatched.append(ev.data))

        dial.rotate_clockwise()
        assert len(dispatched) == 1
        assert "volume" in dispatched[0]

    def test_jack_dispatches_to_simulator(self) -> None:
        from esphome_yaml_tools.simulator.core import Simulator

        sim = Simulator()
        jack = JackSensor()

        dispatched: list[str] = []
        jack.on_event(
            ControlEventType.JACK_PLUGGED,
            lambda e: sim.dispatch_event("jack_plugged", e.data),
        )
        sim.register_event_handler("jack_plugged", lambda ev: dispatched.append(ev.name))

        jack.set_plugged(True)
        assert "jack_plugged" in dispatched


# =============================================================================
# ControlEvent type tests
# =============================================================================


class TestControlEventType:
    """Test ControlEventType enum completeness."""

    def test_all_expected_types_exist(self) -> None:
        assert hasattr(ControlEventType, "PRESSED")
        assert hasattr(ControlEventType, "RELEASED")
        assert hasattr(ControlEventType, "SINGLE_PRESS")
        assert hasattr(ControlEventType, "DOUBLE_PRESS")
        assert hasattr(ControlEventType, "TRIPLE_PRESS")
        assert hasattr(ControlEventType, "LONG_PRESS")
        assert hasattr(ControlEventType, "FACTORY_RESET_WARNING")
        assert hasattr(ControlEventType, "FACTORY_RESET_CONFIRM")
        assert hasattr(ControlEventType, "VOLUME_CHANGED")
        assert hasattr(ControlEventType, "HUE_CHANGED")
        assert hasattr(ControlEventType, "MUTED")
        assert hasattr(ControlEventType, "UNMUTED")
        assert hasattr(ControlEventType, "JACK_PLUGGED")
        assert hasattr(ControlEventType, "JACK_UNPLUGGED")

    def test_control_event_has_source(self) -> None:
        event = ControlEvent(
            event_type=ControlEventType.SINGLE_PRESS,
            source="center_button",
        )
        assert event.source == "center_button"
        assert event.data == {}
