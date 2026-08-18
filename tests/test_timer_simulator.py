"""Comprehensive tests for Voice PE timer simulator.

TDD approach: Tests define expected timer behavior matching Voice PE firmware.
Covers: multiple timers, events, ringing, stop WW, auto-stop, LED integration.
"""

from __future__ import annotations

import pytest

from esphome_yaml_tools.simulator.timer import (
    TimerSimulator,
    TimerEvent,
    TimerEventType,
    TimerInfo,
)
from esphome_yaml_tools.simulator.led_ring import LEDRingSimulator


# =============================================================================
# TimerInfo Tests
# =============================================================================


class TestTimerInfo:
    """Tests for TimerInfo data container."""

    def test_default_state(self):
        info = TimerInfo(timer_id="timer1", total_seconds=300.0)
        assert info.timer_id == "timer1"
        assert info.total_seconds == 300.0
        assert info.remaining_seconds == 300.0
        assert info.is_active is True
        assert info.is_ringing is False

    def test_progress_calculation(self):
        info = TimerInfo(timer_id="t1", total_seconds=100.0, remaining_seconds=75.0)
        assert info.progress == pytest.approx(0.25)

    def test_progress_zero_total(self):
        info = TimerInfo(timer_id="t1", total_seconds=0.0)
        assert info.progress == 0.0

    def test_progress_completed(self):
        info = TimerInfo(
            timer_id="t1",
            total_seconds=60.0,
            remaining_seconds=0.0,
            is_active=False,
        )
        assert info.progress == pytest.approx(1.0)

    def test_to_dict(self):
        info = TimerInfo(timer_id="t1", total_seconds=300.0, remaining_seconds=200.0)
        d = info.to_dict()
        assert d["timer_id"] == "t1"
        assert d["total_seconds"] == 300.0
        assert d["remaining_seconds"] == 200.0
        assert d["is_active"] is True
        assert d["is_ringing"] is False
        assert d["progress"] == pytest.approx(1 / 3)


# =============================================================================
# TimerEvent Tests
# =============================================================================


class TestTimerEvent:
    """Tests for TimerEvent data container."""

    def test_event_creation(self):
        event = TimerEvent(
            event_type=TimerEventType.STARTED,
            timer_id="t1",
        )
        assert event.event_type == TimerEventType.STARTED
        assert event.timer_id == "t1"

    def test_all_event_types(self):
        """All Voice PE timer event types should exist."""
        assert TimerEventType.STARTED.value == "started"
        assert TimerEventType.FINISHED.value == "finished"
        assert TimerEventType.CANCELLED.value == "cancelled"
        assert TimerEventType.UPDATED.value == "updated"
        assert TimerEventType.TICK.value == "tick"


# =============================================================================
# Timer Creation and Basic Operations
# =============================================================================


class TestTimerCreation:
    """Tests for creating and managing timers."""

    def test_start_timer_creates_state(self):
        sim = TimerSimulator()
        sim.start_timer(duration=300)
        timers = sim.get_active_timers()
        assert len(timers) == 1
        assert timers[0].total_seconds == 300.0
        assert timers[0].is_active is True

    def test_start_timer_with_custom_id(self):
        sim = TimerSimulator()
        sim.start_timer(duration=60, timer_id="my_timer")
        info = sim.get_timer("my_timer")
        assert info is not None
        assert info.timer_id == "my_timer"
        assert info.total_seconds == 60.0

    def test_start_timer_auto_generates_id(self):
        sim = TimerSimulator()
        timer_id = sim.start_timer(duration=120)
        assert timer_id is not None
        assert isinstance(timer_id, str)
        assert len(timer_id) > 0

    def test_start_multiple_timers(self):
        sim = TimerSimulator()
        id1 = sim.start_timer(duration=300)
        id2 = sim.start_timer(duration=600)
        id3 = sim.start_timer(duration=60)
        timers = sim.get_active_timers()
        assert len(timers) == 3
        assert {t.timer_id for t in timers} == {id1, id2, id3}

    def test_start_timer_fires_started_event(self):
        sim = TimerSimulator()
        sim.start_timer(duration=300)
        events = sim.get_event_history()
        started = [e for e in events if e.event_type == TimerEventType.STARTED]
        assert len(started) == 1

    def test_get_timer_nonexistent_returns_none(self):
        sim = TimerSimulator()
        assert sim.get_timer("nonexistent") is None

    def test_no_timers_initially(self):
        sim = TimerSimulator()
        assert sim.get_active_timers() == []
        assert sim.get_all_timers() == []
        assert sim.is_any_timer_active is False


# =============================================================================
# Timer Advance Time and Tick Events
# =============================================================================


class TestTimerAdvanceTime:
    """Tests for advancing time and tick events."""

    def test_advance_time_reduces_remaining(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=300)
        sim.advance_time(1.0)
        info = sim.get_timer(tid)
        assert info.remaining_seconds == pytest.approx(299.0)

    def test_advance_time_triggers_tick_event(self):
        sim = TimerSimulator()
        sim.start_timer(duration=300)
        sim.advance_time(1.0)
        events = sim.get_event_history()
        ticks = [e for e in events if e.event_type == TimerEventType.TICK]
        assert len(ticks) >= 1

    def test_advance_time_tick_has_seconds_left(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=300)
        sim.advance_time(1.0)
        events = sim.get_event_history()
        ticks = [e for e in events if e.event_type == TimerEventType.TICK]
        assert ticks[-1].data["seconds_left"] == pytest.approx(299.0)

    def test_advance_multiple_seconds(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=100)
        sim.advance_time(50.0)
        info = sim.get_timer(tid)
        assert info.remaining_seconds == pytest.approx(50.0)
        assert info.progress == pytest.approx(0.5)

    def test_multiple_timers_all_tick(self):
        sim = TimerSimulator()
        t1 = sim.start_timer(duration=300)
        t2 = sim.start_timer(duration=600)
        sim.advance_time(10.0)
        assert sim.get_timer(t1).remaining_seconds == pytest.approx(290.0)
        assert sim.get_timer(t2).remaining_seconds == pytest.approx(590.0)

    def test_advance_time_incremental(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=100)
        sim.advance_time(1.0)
        sim.advance_time(1.0)
        sim.advance_time(1.0)
        assert sim.get_timer(tid).remaining_seconds == pytest.approx(97.0)


# =============================================================================
# Timer Finish
# =============================================================================


class TestTimerFinish:
    """Tests for timer completion."""

    def test_timer_finishes_when_remaining_zero(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=10)
        sim.advance_time(10.0)
        info = sim.get_timer(tid)
        assert info.remaining_seconds == 0.0
        assert info.is_active is False

    def test_timer_finish_fires_finished_event(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=5)
        sim.advance_time(5.0)
        events = sim.get_event_history()
        finished = [e for e in events if e.event_type == TimerEventType.FINISHED]
        assert len(finished) == 1
        assert finished[0].timer_id == tid

    def test_timer_finish_starts_ringing(self):
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert sim.is_ringing is True

    def test_timer_finish_enables_stop_ww(self):
        """On timer finish, stop wake word model should be auto-enabled."""
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert sim.stop_ww_enabled is True

    def test_timer_finish_enables_audio_ducking(self):
        """On timer finish, media audio should be ducked (-20dB)."""
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert sim.audio_ducking_active is True

    def test_overshoot_time_still_finishes(self):
        """Timer finishes even if advance_time overshoots."""
        sim = TimerSimulator()
        tid = sim.start_timer(duration=5)
        sim.advance_time(100.0)
        info = sim.get_timer(tid)
        assert info.remaining_seconds == 0.0
        assert info.is_active is False

    def test_multiple_timers_finish_independently(self):
        sim = TimerSimulator()
        t1 = sim.start_timer(duration=5)
        t2 = sim.start_timer(duration=10)
        sim.advance_time(5.0)
        assert sim.get_timer(t1).is_active is False
        assert sim.get_timer(t2).is_active is True
        assert sim.get_timer(t2).remaining_seconds == pytest.approx(5.0)

    def test_second_timer_finish_still_ringing(self):
        """If first timer finished (ringing) and second finishes, still ringing."""
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.start_timer(duration=10)
        sim.advance_time(5.0)
        assert sim.is_ringing is True
        sim.advance_time(5.0)
        assert sim.is_ringing is True


# =============================================================================
# Timer Cancel
# =============================================================================


class TestTimerCancel:
    """Tests for cancelling timers."""

    def test_cancel_timer(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=300)
        sim.cancel_timer(tid)
        info = sim.get_timer(tid)
        assert info.is_active is False

    def test_cancel_fires_cancelled_event(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=300)
        sim.cancel_timer(tid)
        events = sim.get_event_history()
        cancelled = [e for e in events if e.event_type == TimerEventType.CANCELLED]
        assert len(cancelled) == 1
        assert cancelled[0].timer_id == tid

    def test_cancel_nonexistent_raises(self):
        sim = TimerSimulator()
        with pytest.raises(KeyError):
            sim.cancel_timer("nonexistent")

    def test_cancel_one_of_multiple(self):
        sim = TimerSimulator()
        t1 = sim.start_timer(duration=300)
        t2 = sim.start_timer(duration=600)
        sim.cancel_timer(t1)
        assert sim.get_timer(t1).is_active is False
        assert sim.get_timer(t2).is_active is True
        assert len(sim.get_active_timers()) == 1

    def test_cancel_all_timers(self):
        sim = TimerSimulator()
        t1 = sim.start_timer(duration=300)
        t2 = sim.start_timer(duration=600)
        sim.cancel_all_timers()
        assert len(sim.get_active_timers()) == 0


# =============================================================================
# Timer Update
# =============================================================================


class TestTimerUpdate:
    """Tests for updating timer duration."""

    def test_update_timer_duration(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=300)
        sim.advance_time(100.0)
        sim.update_timer(tid, new_duration=600)
        info = sim.get_timer(tid)
        assert info.total_seconds == 600.0
        # Remaining should be recalculated: new_duration - elapsed
        assert info.remaining_seconds == pytest.approx(500.0)

    def test_update_timer_fires_updated_event(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=300)
        sim.update_timer(tid, new_duration=600)
        events = sim.get_event_history()
        updated = [e for e in events if e.event_type == TimerEventType.UPDATED]
        assert len(updated) == 1
        assert updated[0].timer_id == tid

    def test_update_nonexistent_raises(self):
        sim = TimerSimulator()
        with pytest.raises(KeyError):
            sim.update_timer("nonexistent", new_duration=100)


# =============================================================================
# Timer Ringing Behavior
# =============================================================================


class TestTimerRinging:
    """Tests for timer ringing behavior matching Voice PE firmware."""

    def test_not_ringing_initially(self):
        sim = TimerSimulator()
        assert sim.is_ringing is False

    def test_ringing_after_timer_finish(self):
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert sim.is_ringing is True

    def test_stop_timer_cancels_ringing(self):
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert sim.is_ringing is True
        sim.stop_ringing()
        assert sim.is_ringing is False

    def test_stop_timer_disables_stop_ww(self):
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert sim.stop_ww_enabled is True
        sim.stop_ringing()
        assert sim.stop_ww_enabled is False

    def test_stop_timer_disables_audio_ducking(self):
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert sim.audio_ducking_active is True
        sim.stop_ringing()
        assert sim.audio_ducking_active is False

    def test_auto_stop_after_15_minutes(self):
        """Timer ringing auto-stops after 15 minutes (safety limit)."""
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)  # Timer finishes, starts ringing
        assert sim.is_ringing is True
        # Advance 15 minutes (900 seconds)
        sim.advance_time(900.0)
        assert sim.is_ringing is False
        assert sim.stop_ww_enabled is False

    def test_ringing_tracks_elapsed(self):
        """Ringing state tracks how long it has been ringing."""
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        sim.advance_time(10.0)
        assert sim.ringing_elapsed == pytest.approx(10.0)

    def test_multiple_finishes_reset_ringing_elapsed(self):
        """Each new finish while already ringing doesn't reset elapsed."""
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.start_timer(duration=10)
        sim.advance_time(5.0)  # First finishes
        sim.advance_time(3.0)  # 3s of ringing
        # Ringing elapsed should be 3s
        assert sim.ringing_elapsed == pytest.approx(3.0)


# =============================================================================
# Stop Wake Word Integration
# =============================================================================


class TestStopWakeWord:
    """Tests for stop wake word auto-enable/disable."""

    def test_stop_ww_off_initially(self):
        sim = TimerSimulator()
        assert sim.stop_ww_enabled is False

    def test_stop_ww_enabled_on_ringing(self):
        sim = TimerSimulator()
        sim.start_timer(duration=1)
        sim.advance_time(1.0)
        assert sim.stop_ww_enabled is True

    def test_stop_ww_disabled_on_stop_ringing(self):
        sim = TimerSimulator()
        sim.start_timer(duration=1)
        sim.advance_time(1.0)
        sim.stop_ringing()
        assert sim.stop_ww_enabled is False

    def test_stop_ww_disabled_on_auto_stop(self):
        sim = TimerSimulator()
        sim.start_timer(duration=1)
        sim.advance_time(1.0)
        sim.advance_time(900.0)  # 15 min auto-stop
        assert sim.stop_ww_enabled is False


# =============================================================================
# LED Integration
# =============================================================================


class TestTimerLEDIntegration:
    """Tests for timer-LED ring integration."""

    def test_timer_tick_updates_led(self):
        """LED countdown updates each tick."""
        led = LEDRingSimulator()
        sim = TimerSimulator(led_ring=led)
        sim.start_timer(duration=100)
        sim.advance_time(50.0)
        # LED should show timer_tick effect with progress ~0.5
        assert led.current_effect == "timer_tick"
        leds = led.get_led_state()
        # ~6 LEDs should be lit (50% remaining = 6 out of 12)
        lit = [c for c in leds if c != (0, 0, 0)]
        assert len(lit) == 6

    def test_timer_tick_led_progress_updates(self):
        """LED arc shrinks as timer progresses."""
        led = LEDRingSimulator()
        sim = TimerSimulator(led_ring=led)
        sim.start_timer(duration=120)
        sim.advance_time(30.0)  # 25% done
        leds_25 = led.get_led_state()
        lit_25 = sum(1 for c in leds_25 if c != (0, 0, 0))

        sim.advance_time(30.0)  # 50% done
        leds_50 = led.get_led_state()
        lit_50 = sum(1 for c in leds_50 if c != (0, 0, 0))

        assert lit_25 > lit_50  # More LEDs lit at 25% vs 50% progress

    def test_timer_finish_switches_to_ring_effect(self):
        """When timer finishes (ringing), LED switches to timer_ring effect."""
        led = LEDRingSimulator()
        sim = TimerSimulator(led_ring=led)
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert led.current_effect == "timer_ring"

    def test_stop_ringing_clears_led(self):
        """When ringing stops, LED effect is cleared."""
        led = LEDRingSimulator()
        sim = TimerSimulator(led_ring=led)
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        sim.stop_ringing()
        assert led.current_effect is None

    def test_no_led_without_ring(self):
        """TimerSimulator works without LED ring (no crash)."""
        sim = TimerSimulator()  # No led_ring
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert sim.is_ringing is True

    def test_timer_tick_led_first_active_timer(self):
        """LED shows progress of the first (shortest remaining) timer."""
        led = LEDRingSimulator()
        sim = TimerSimulator(led_ring=led)
        sim.start_timer(duration=120)  # 2 min
        sim.start_timer(duration=60)  # 1 min (shorter)
        sim.advance_time(30.0)
        # The shortest timer has 30s left out of 60 = 50% progress
        # LED should show ~6 LEDs (50% remaining)
        leds = led.get_led_state()
        lit = [c for c in leds if c != (0, 0, 0)]
        assert len(lit) == 6

    def test_cancel_all_clears_led(self):
        """Cancelling all timers clears LED."""
        led = LEDRingSimulator()
        sim = TimerSimulator(led_ring=led)
        sim.start_timer(duration=60)
        sim.advance_time(1.0)
        assert led.current_effect == "timer_tick"
        sim.cancel_all_timers()
        assert led.current_effect is None


# =============================================================================
# Event History and Callbacks
# =============================================================================


class TestTimerEvents:
    """Tests for event history and callbacks."""

    def test_event_history_records_all(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=10)
        sim.advance_time(5.0)
        sim.cancel_timer(tid)
        history = sim.get_event_history()
        types = [e.event_type for e in history]
        assert TimerEventType.STARTED in types
        assert TimerEventType.TICK in types
        assert TimerEventType.CANCELLED in types

    def test_event_callback(self):
        """Register callback for specific event type."""
        sim = TimerSimulator()
        received = []
        sim.on_event(TimerEventType.STARTED, lambda e: received.append(e))
        sim.start_timer(duration=60)
        assert len(received) == 1
        assert received[0].event_type == TimerEventType.STARTED

    def test_multiple_callbacks(self):
        sim = TimerSimulator()
        started = []
        finished = []
        sim.on_event(TimerEventType.STARTED, lambda e: started.append(e))
        sim.on_event(TimerEventType.FINISHED, lambda e: finished.append(e))
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        assert len(started) == 1
        assert len(finished) == 1

    def test_tick_callback_fires_per_advance(self):
        sim = TimerSimulator()
        ticks = []
        sim.on_event(TimerEventType.TICK, lambda e: ticks.append(e))
        sim.start_timer(duration=100)
        sim.advance_time(1.0)
        sim.advance_time(1.0)
        sim.advance_time(1.0)
        assert len(ticks) == 3

    def test_clear_event_history(self):
        sim = TimerSimulator()
        sim.start_timer(duration=10)
        sim.advance_time(1.0)
        sim.clear_event_history()
        assert sim.get_event_history() == []


# =============================================================================
# Fetch First Active Timer (Voice PE pattern)
# =============================================================================


class TestFetchFirstActiveTimer:
    """Tests for fetch_first_active_timer (Voice PE firmware pattern)."""

    def test_first_active_timer_single(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=300)
        first = sim.get_first_active_timer()
        assert first is not None
        assert first.timer_id == tid

    def test_first_active_timer_returns_shortest_remaining(self):
        sim = TimerSimulator()
        sim.start_timer(duration=600)
        t_short = sim.start_timer(duration=60)
        sim.start_timer(duration=300)
        first = sim.get_first_active_timer()
        assert first.timer_id == t_short

    def test_first_active_timer_none_when_empty(self):
        sim = TimerSimulator()
        assert sim.get_first_active_timer() is None

    def test_first_active_timer_skips_finished(self):
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        t2 = sim.start_timer(duration=60)
        sim.advance_time(5.0)  # First timer finishes
        first = sim.get_first_active_timer()
        assert first is not None
        assert first.timer_id == t2


# =============================================================================
# Core Simulator Integration
# =============================================================================


class TestCoreSimulatorIntegration:
    """Tests for integration with core Simulator (core.py)."""

    def test_timer_simulator_wraps_core_timers(self):
        """TimerSimulator can be used alongside core Simulator."""
        from esphome_yaml_tools.simulator.core import Simulator

        core = Simulator()
        timer_sim = TimerSimulator()

        # Both should work independently
        core.start_timer("core_t1", 300)
        tid = timer_sim.start_timer(duration=300)
        assert core.get_timer("core_t1").active is True
        assert timer_sim.get_timer(tid).is_active is True

    def test_timer_simulator_with_core_events(self):
        """TimerSimulator events can bridge to core Simulator."""
        from esphome_yaml_tools.simulator.core import Simulator

        core = Simulator()
        timer_sim = TimerSimulator()

        # Bridge: forward timer events to core dispatch
        timer_sim.on_event(
            TimerEventType.FINISHED,
            lambda e: core.dispatch_event("timer_finished", {"timer_id": e.timer_id}),
        )

        timer_sim.start_timer(duration=5)
        timer_sim.advance_time(5.0)

        core_events = core.get_event_history()
        assert any(e.name == "timer_finished" for e in core_events)


# =============================================================================
# Reset and State Management
# =============================================================================


class TestTimerReset:
    """Tests for resetting timer simulator state."""

    def test_reset_clears_all_timers(self):
        sim = TimerSimulator()
        sim.start_timer(duration=300)
        sim.start_timer(duration=600)
        sim.reset()
        assert sim.get_all_timers() == []
        assert sim.is_ringing is False

    def test_reset_clears_ringing(self):
        sim = TimerSimulator()
        sim.start_timer(duration=1)
        sim.advance_time(1.0)
        assert sim.is_ringing is True
        sim.reset()
        assert sim.is_ringing is False
        assert sim.stop_ww_enabled is False
        assert sim.audio_ducking_active is False

    def test_reset_clears_event_history(self):
        sim = TimerSimulator()
        sim.start_timer(duration=10)
        sim.advance_time(1.0)
        sim.reset()
        assert sim.get_event_history() == []

    def test_reset_clears_led(self):
        led = LEDRingSimulator()
        sim = TimerSimulator(led_ring=led)
        sim.start_timer(duration=60)
        sim.advance_time(1.0)
        sim.reset()
        assert led.current_effect is None


# =============================================================================
# Snapshot/Serialization
# =============================================================================


class TestTimerSnapshot:
    """Tests for timer state serialization."""

    def test_snapshot(self):
        sim = TimerSimulator()
        t1 = sim.start_timer(duration=300)
        t2 = sim.start_timer(duration=60)
        sim.advance_time(10.0)
        snap = sim.snapshot()
        assert "timers" in snap
        assert len(snap["timers"]) == 2
        assert "is_ringing" in snap
        assert "stop_ww_enabled" in snap
        assert "audio_ducking_active" in snap

    def test_snapshot_captures_ringing(self):
        sim = TimerSimulator()
        sim.start_timer(duration=1)
        sim.advance_time(1.0)
        snap = sim.snapshot()
        assert snap["is_ringing"] is True
        assert snap["stop_ww_enabled"] is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestTimerEdgeCases:
    """Edge case tests."""

    def test_zero_duration_timer(self):
        """Timer with 0 duration should immediately finish."""
        sim = TimerSimulator()
        sim.start_timer(duration=0)
        # Should have both STARTED and FINISHED events
        events = sim.get_event_history()
        types = [e.event_type for e in events]
        assert TimerEventType.STARTED in types
        assert TimerEventType.FINISHED in types

    def test_very_small_advance(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=100)
        sim.advance_time(0.001)
        assert sim.get_timer(tid).remaining_seconds == pytest.approx(99.999)

    def test_negative_advance_ignored(self):
        sim = TimerSimulator()
        tid = sim.start_timer(duration=100)
        sim.advance_time(-5.0)
        assert sim.get_timer(tid).remaining_seconds == pytest.approx(100.0)

    def test_stop_ringing_when_not_ringing(self):
        """Calling stop_ringing when not ringing should be a no-op."""
        sim = TimerSimulator()
        sim.stop_ringing()  # Should not raise
        assert sim.is_ringing is False

    def test_advance_time_after_all_finished(self):
        """Advancing time after all timers finished should not crash."""
        sim = TimerSimulator()
        sim.start_timer(duration=5)
        sim.advance_time(5.0)
        sim.advance_time(5.0)  # No active timers, but ringing
        assert sim.is_ringing is True

    def test_cancel_already_finished_timer(self):
        """Cancelling a finished timer should work (already inactive)."""
        sim = TimerSimulator()
        tid = sim.start_timer(duration=5)
        sim.advance_time(5.0)
        sim.cancel_timer(tid)
        assert sim.get_timer(tid).is_active is False
