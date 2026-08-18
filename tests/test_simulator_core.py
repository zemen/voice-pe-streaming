"""Comprehensive tests for ESPHome state machine simulator core.

TDD approach: These tests define the expected behavior of the simulator.
"""

from __future__ import annotations

import pytest

from esphome_yaml_tools.simulator.states import (
    ComponentState,
    LightState,
    SwitchState,
    SensorState,
    VoiceAssistantState,
    TimerState,
    VoiceAssistantPhase,
)
from esphome_yaml_tools.simulator.core import (
    Simulator,
    SimulatorEvent,
    Automation,
    Action,
    Condition,
)


# =============================================================================
# State Container Tests
# =============================================================================


class TestLightState:
    """Tests for LightState container."""

    def test_default_state(self):
        state = LightState()
        assert state.on is False
        assert state.brightness == 0.0
        assert state.color == (0, 0, 0)
        assert state.effect == ""

    def test_custom_state(self):
        state = LightState(on=True, brightness=0.8, color=(255, 128, 0), effect="rainbow")
        assert state.on is True
        assert state.brightness == 0.8
        assert state.color == (255, 128, 0)
        assert state.effect == "rainbow"

    def test_equality(self):
        a = LightState(on=True, brightness=0.5)
        b = LightState(on=True, brightness=0.5)
        assert a == b

    def test_to_dict(self):
        state = LightState(on=True, brightness=1.0, color=(255, 0, 0), effect="pulse")
        d = state.to_dict()
        assert d["on"] is True
        assert d["brightness"] == 1.0
        assert d["color"] == (255, 0, 0)
        assert d["effect"] == "pulse"


class TestSwitchState:
    """Tests for SwitchState container."""

    def test_default_state(self):
        state = SwitchState()
        assert state.on is False

    def test_on_state(self):
        state = SwitchState(on=True)
        assert state.on is True

    def test_to_dict(self):
        state = SwitchState(on=True)
        assert state.to_dict() == {"on": True}


class TestSensorState:
    """Tests for SensorState container."""

    def test_default_state(self):
        state = SensorState()
        assert state.value == 0.0
        assert state.unit == ""

    def test_custom_state(self):
        state = SensorState(value=23.5, unit="°C")
        assert state.value == 23.5
        assert state.unit == "°C"

    def test_to_dict(self):
        state = SensorState(value=42.0, unit="lux")
        d = state.to_dict()
        assert d["value"] == 42.0
        assert d["unit"] == "lux"


class TestVoiceAssistantState:
    """Tests for VoiceAssistantState container."""

    def test_default_state(self):
        state = VoiceAssistantState()
        assert state.phase == VoiceAssistantPhase.IDLE

    def test_custom_phase(self):
        state = VoiceAssistantState(phase=VoiceAssistantPhase.LISTENING)
        assert state.phase == VoiceAssistantPhase.LISTENING

    def test_phase_enum_values(self):
        """Verify phase enum matches Voice PE firmware values."""
        assert VoiceAssistantPhase.IDLE.value == 1
        assert VoiceAssistantPhase.WAITING_FOR_COMMAND.value == 2
        assert VoiceAssistantPhase.LISTENING.value == 3
        assert VoiceAssistantPhase.THINKING.value == 4
        assert VoiceAssistantPhase.REPLYING.value == 5
        assert VoiceAssistantPhase.NOT_READY.value == 10
        assert VoiceAssistantPhase.ERROR.value == 11

    def test_to_dict(self):
        state = VoiceAssistantState(phase=VoiceAssistantPhase.THINKING)
        d = state.to_dict()
        assert d["phase"] == "thinking"


class TestTimerState:
    """Tests for TimerState container."""

    def test_default_state(self):
        state = TimerState()
        assert state.active is False
        assert state.remaining == 0.0
        assert state.total == 0.0

    def test_active_timer(self):
        state = TimerState(active=True, remaining=45.0, total=60.0)
        assert state.active is True
        assert state.remaining == 45.0
        assert state.total == 60.0

    def test_progress(self):
        """Timer progress: 0.0 = just started, 1.0 = complete."""
        state = TimerState(active=True, remaining=30.0, total=60.0)
        assert state.progress == 0.5

    def test_progress_zero_total(self):
        """Progress should be 0.0 for inactive timer."""
        state = TimerState()
        assert state.progress == 0.0

    def test_to_dict(self):
        state = TimerState(active=True, remaining=10.0, total=60.0)
        d = state.to_dict()
        assert d["active"] is True
        assert d["remaining"] == 10.0
        assert d["total"] == 60.0


# =============================================================================
# Simulator Event Tests
# =============================================================================


class TestSimulatorEvent:
    """Tests for SimulatorEvent data class."""

    def test_create_event(self):
        event = SimulatorEvent(name="button_pressed")
        assert event.name == "button_pressed"
        assert event.data == {}

    def test_event_with_data(self):
        event = SimulatorEvent(name="timer_tick", data={"timer_id": "t1", "seconds_left": 30})
        assert event.data["timer_id"] == "t1"
        assert event.data["seconds_left"] == 30


# =============================================================================
# Simulator Core Tests
# =============================================================================


class TestSimulatorInit:
    """Tests for Simulator initialization."""

    def test_create_empty_simulator(self):
        sim = Simulator()
        assert sim is not None

    def test_load_config_dict(self):
        """Simulator can load a parsed YAML config dict."""
        config = {
            "light": [{"platform": "esp32_rmt_led_strip", "id": "led_ring", "num_leds": 12}],
            "switch": [{"platform": "template", "id": "timer_ringing"}],
        }
        sim = Simulator()
        sim.load_config(config)
        # Components should be registered
        assert sim.has_component("light.led_ring")
        assert sim.has_component("switch.timer_ringing")

    def test_load_config_initializes_states(self):
        """Loading config creates default state for each component."""
        config = {
            "light": [{"platform": "partition", "id": "led_ring"}],
            "switch": [{"platform": "template", "id": "mute"}],
            "sensor": [{"platform": "rotary_encoder", "id": "dial"}],
        }
        sim = Simulator()
        sim.load_config(config)

        light_state = sim.get_state("light.led_ring")
        assert isinstance(light_state, LightState)
        assert light_state.on is False

        switch_state = sim.get_state("switch.mute")
        assert isinstance(switch_state, SwitchState)
        assert switch_state.on is False

        sensor_state = sim.get_state("sensor.dial")
        assert isinstance(sensor_state, SensorState)

    def test_load_config_voice_assistant(self):
        """Voice assistant gets VoiceAssistantState."""
        config = {"voice_assistant": {"id": "va"}}
        sim = Simulator()
        sim.load_config(config)
        state = sim.get_state("voice_assistant.va")
        assert isinstance(state, VoiceAssistantState)
        assert state.phase == VoiceAssistantPhase.IDLE

    def test_load_config_binary_sensor(self):
        """Binary sensor gets SwitchState (on/off)."""
        config = {"binary_sensor": [{"platform": "gpio", "id": "center_button"}]}
        sim = Simulator()
        sim.load_config(config)
        state = sim.get_state("binary_sensor.center_button")
        assert isinstance(state, SwitchState)
        assert state.on is False

    def test_get_state_nonexistent(self):
        """Getting state for unknown component returns None."""
        sim = Simulator()
        assert sim.get_state("light.nonexistent") is None


class TestSimulatorStateManipulation:
    """Tests for setting/getting component states."""

    def test_set_light_state(self):
        config = {"light": [{"id": "led_ring", "platform": "esp32_rmt_led_strip"}]}
        sim = Simulator()
        sim.load_config(config)

        sim.set_state("light.led_ring", LightState(on=True, brightness=1.0, color=(255, 0, 0)))
        state = sim.get_state("light.led_ring")
        assert state.on is True
        assert state.brightness == 1.0
        assert state.color == (255, 0, 0)

    def test_set_switch_state(self):
        config = {"switch": [{"id": "timer_ringing", "platform": "template"}]}
        sim = Simulator()
        sim.load_config(config)

        sim.set_state("switch.timer_ringing", SwitchState(on=True))
        assert sim.get_state("switch.timer_ringing").on is True

    def test_set_sensor_value(self):
        config = {"sensor": [{"id": "dial", "platform": "rotary_encoder"}]}
        sim = Simulator()
        sim.load_config(config)

        sim.set_state("sensor.dial", SensorState(value=50.0))
        assert sim.get_state("sensor.dial").value == 50.0

    def test_set_voice_assistant_phase(self):
        config = {"voice_assistant": {"id": "va"}}
        sim = Simulator()
        sim.load_config(config)

        sim.set_state(
            "voice_assistant.va",
            VoiceAssistantState(phase=VoiceAssistantPhase.LISTENING),
        )
        state = sim.get_state("voice_assistant.va")
        assert state.phase == VoiceAssistantPhase.LISTENING

    def test_get_all_states(self):
        config = {
            "light": [{"id": "a", "platform": "x"}],
            "switch": [{"id": "b", "platform": "x"}],
        }
        sim = Simulator()
        sim.load_config(config)
        states = sim.get_all_states()
        assert "light.a" in states
        assert "switch.b" in states


# =============================================================================
# Event Dispatch Tests
# =============================================================================


class TestEventDispatch:
    """Tests for event dispatch system."""

    def test_dispatch_event_basic(self):
        """Dispatching event should trigger registered automations."""
        sim = Simulator()
        sim.load_config({})

        # Track that the callback was called
        called = []

        def on_button(event: SimulatorEvent):
            called.append(event.name)

        sim.register_event_handler("button_pressed", on_button)
        sim.dispatch_event("button_pressed")

        assert called == ["button_pressed"]

    def test_dispatch_event_with_data(self):
        """Event data should be passed to handler."""
        sim = Simulator()
        sim.load_config({})

        received_data = {}

        def on_timer(event: SimulatorEvent):
            received_data.update(event.data)

        sim.register_event_handler("timer_tick", on_timer)
        sim.dispatch_event("timer_tick", data={"seconds_left": 30})

        assert received_data["seconds_left"] == 30

    def test_multiple_handlers_same_event(self):
        """Multiple handlers for same event all get called."""
        sim = Simulator()
        sim.load_config({})

        calls = []
        sim.register_event_handler("wake_word", lambda e: calls.append("handler1"))
        sim.register_event_handler("wake_word", lambda e: calls.append("handler2"))
        sim.dispatch_event("wake_word")

        assert len(calls) == 2
        assert "handler1" in calls
        assert "handler2" in calls

    def test_dispatch_unregistered_event(self):
        """Dispatching event with no handlers should not raise."""
        sim = Simulator()
        sim.load_config({})
        # Should not raise
        sim.dispatch_event("nonexistent_event")

    def test_event_history(self):
        """Simulator tracks event history."""
        sim = Simulator()
        sim.load_config({})
        sim.dispatch_event("button_pressed")
        sim.dispatch_event("timer_tick")

        history = sim.get_event_history()
        assert len(history) == 2
        assert history[0].name == "button_pressed"
        assert history[1].name == "timer_tick"


# =============================================================================
# Automation Tests
# =============================================================================


class TestAutomation:
    """Tests for automation (trigger → condition → action) execution."""

    def test_automation_creation(self):
        auto = Automation(
            trigger="button_pressed",
            actions=[Action(type="switch.turn_on", target="switch.timer_ringing")],
        )
        assert auto.trigger == "button_pressed"
        assert len(auto.actions) == 1

    def test_automation_with_condition(self):
        auto = Automation(
            trigger="wake_word_detected",
            conditions=[Condition(type="switch.is_off", target="switch.mute")],
            actions=[Action(type="voice_assistant.start", target="voice_assistant.va")],
        )
        assert len(auto.conditions) == 1
        assert auto.conditions[0].type == "switch.is_off"

    def test_register_and_execute_automation(self):
        """Registering an automation and dispatching its trigger event executes actions."""
        config = {
            "switch": [{"id": "timer_ringing", "platform": "template"}],
        }
        sim = Simulator()
        sim.load_config(config)

        auto = Automation(
            trigger="timer_finished",
            actions=[Action(type="switch.turn_on", target="switch.timer_ringing")],
        )
        sim.register_automation(auto)

        # Timer should be off initially
        assert sim.get_state("switch.timer_ringing").on is False

        # Dispatch the trigger
        sim.dispatch_event("timer_finished")

        # Automation should have turned the switch on
        assert sim.get_state("switch.timer_ringing").on is True

    def test_automation_switch_turn_off(self):
        config = {"switch": [{"id": "mute", "platform": "template"}]}
        sim = Simulator()
        sim.load_config(config)

        # Start muted
        sim.set_state("switch.mute", SwitchState(on=True))

        auto = Automation(
            trigger="unmute",
            actions=[Action(type="switch.turn_off", target="switch.mute")],
        )
        sim.register_automation(auto)
        sim.dispatch_event("unmute")

        assert sim.get_state("switch.mute").on is False

    def test_automation_light_turn_on(self):
        config = {"light": [{"id": "led_ring", "platform": "partition"}]}
        sim = Simulator()
        sim.load_config(config)

        auto = Automation(
            trigger="activate_leds",
            actions=[
                Action(
                    type="light.turn_on",
                    target="light.led_ring",
                    params={"brightness": 1.0, "color": (0, 255, 0), "effect": "rainbow"},
                )
            ],
        )
        sim.register_automation(auto)
        sim.dispatch_event("activate_leds")

        state = sim.get_state("light.led_ring")
        assert state.on is True
        assert state.brightness == 1.0
        assert state.color == (0, 255, 0)
        assert state.effect == "rainbow"

    def test_automation_light_turn_off(self):
        config = {"light": [{"id": "led_ring", "platform": "partition"}]}
        sim = Simulator()
        sim.load_config(config)

        sim.set_state("light.led_ring", LightState(on=True, brightness=1.0))

        auto = Automation(
            trigger="deactivate_leds",
            actions=[Action(type="light.turn_off", target="light.led_ring")],
        )
        sim.register_automation(auto)
        sim.dispatch_event("deactivate_leds")

        assert sim.get_state("light.led_ring").on is False

    def test_automation_condition_blocks(self):
        """Automation with failing condition should NOT execute actions."""
        config = {
            "switch": [
                {"id": "mute", "platform": "template"},
                {"id": "target", "platform": "template"},
            ],
        }
        sim = Simulator()
        sim.load_config(config)

        # Mute is OFF
        auto = Automation(
            trigger="do_thing",
            conditions=[Condition(type="switch.is_on", target="switch.mute")],
            actions=[Action(type="switch.turn_on", target="switch.target")],
        )
        sim.register_automation(auto)
        sim.dispatch_event("do_thing")

        # Condition failed (mute is off), so target should remain off
        assert sim.get_state("switch.target").on is False

    def test_automation_condition_passes(self):
        """Automation with passing condition SHOULD execute actions."""
        config = {
            "switch": [
                {"id": "mute", "platform": "template"},
                {"id": "target", "platform": "template"},
            ],
        }
        sim = Simulator()
        sim.load_config(config)

        sim.set_state("switch.mute", SwitchState(on=True))

        auto = Automation(
            trigger="do_thing",
            conditions=[Condition(type="switch.is_on", target="switch.mute")],
            actions=[Action(type="switch.turn_on", target="switch.target")],
        )
        sim.register_automation(auto)
        sim.dispatch_event("do_thing")

        assert sim.get_state("switch.target").on is True

    def test_multiple_automations_same_trigger(self):
        """Multiple automations on the same trigger all execute."""
        config = {
            "switch": [
                {"id": "a", "platform": "template"},
                {"id": "b", "platform": "template"},
            ]
        }
        sim = Simulator()
        sim.load_config(config)

        sim.register_automation(
            Automation(
                trigger="go",
                actions=[Action(type="switch.turn_on", target="switch.a")],
            )
        )
        sim.register_automation(
            Automation(
                trigger="go",
                actions=[Action(type="switch.turn_on", target="switch.b")],
            )
        )
        sim.dispatch_event("go")

        assert sim.get_state("switch.a").on is True
        assert sim.get_state("switch.b").on is True

    def test_automation_multiple_actions(self):
        """Single automation with multiple actions executes all."""
        config = {
            "switch": [
                {"id": "a", "platform": "template"},
                {"id": "b", "platform": "template"},
            ],
            "light": [{"id": "c", "platform": "partition"}],
        }
        sim = Simulator()
        sim.load_config(config)

        auto = Automation(
            trigger="start",
            actions=[
                Action(type="switch.turn_on", target="switch.a"),
                Action(type="switch.turn_on", target="switch.b"),
                Action(type="light.turn_on", target="light.c", params={"brightness": 0.5}),
            ],
        )
        sim.register_automation(auto)
        sim.dispatch_event("start")

        assert sim.get_state("switch.a").on is True
        assert sim.get_state("switch.b").on is True
        assert sim.get_state("light.c").on is True
        assert sim.get_state("light.c").brightness == 0.5


# =============================================================================
# Delay / Simulated Time Tests
# =============================================================================


class TestDelayAction:
    """Tests for delay action and simulated time advancement."""

    def test_delay_action_not_immediate(self):
        """Action with delay should NOT execute immediately."""
        config = {"switch": [{"id": "target", "platform": "template"}]}
        sim = Simulator()
        sim.load_config(config)

        auto = Automation(
            trigger="delayed_on",
            actions=[
                Action(type="delay", params={"seconds": 2.0}),
                Action(type="switch.turn_on", target="switch.target"),
            ],
        )
        sim.register_automation(auto)
        sim.dispatch_event("delayed_on")

        # Target should NOT be on yet - delay hasn't passed
        assert sim.get_state("switch.target").on is False

    def test_advance_time_executes_delayed(self):
        """Advancing time past delay should execute pending actions."""
        config = {"switch": [{"id": "target", "platform": "template"}]}
        sim = Simulator()
        sim.load_config(config)

        auto = Automation(
            trigger="delayed_on",
            actions=[
                Action(type="delay", params={"seconds": 2.0}),
                Action(type="switch.turn_on", target="switch.target"),
            ],
        )
        sim.register_automation(auto)
        sim.dispatch_event("delayed_on")

        # Advance 1 second - not enough
        sim.advance_time(1.0)
        assert sim.get_state("switch.target").on is False

        # Advance another 1.5 seconds - now past the 2s delay
        sim.advance_time(1.5)
        assert sim.get_state("switch.target").on is True

    def test_actions_before_delay_execute_immediately(self):
        """Actions before a delay should execute right away."""
        config = {
            "switch": [
                {"id": "immediate", "platform": "template"},
                {"id": "delayed", "platform": "template"},
            ]
        }
        sim = Simulator()
        sim.load_config(config)

        auto = Automation(
            trigger="mixed",
            actions=[
                Action(type="switch.turn_on", target="switch.immediate"),
                Action(type="delay", params={"seconds": 5.0}),
                Action(type="switch.turn_on", target="switch.delayed"),
            ],
        )
        sim.register_automation(auto)
        sim.dispatch_event("mixed")

        # Immediate action executed, delayed has not
        assert sim.get_state("switch.immediate").on is True
        assert sim.get_state("switch.delayed").on is False

        # Now advance time
        sim.advance_time(5.0)
        assert sim.get_state("switch.delayed").on is True

    def test_multiple_delays(self):
        """Multiple delays in sequence should accumulate."""
        config = {
            "switch": [
                {"id": "a", "platform": "template"},
                {"id": "b", "platform": "template"},
            ]
        }
        sim = Simulator()
        sim.load_config(config)

        auto = Automation(
            trigger="multi_delay",
            actions=[
                Action(type="delay", params={"seconds": 1.0}),
                Action(type="switch.turn_on", target="switch.a"),
                Action(type="delay", params={"seconds": 2.0}),
                Action(type="switch.turn_on", target="switch.b"),
            ],
        )
        sim.register_automation(auto)
        sim.dispatch_event("multi_delay")

        # Nothing yet
        assert sim.get_state("switch.a").on is False
        assert sim.get_state("switch.b").on is False

        # After 1s: a fires
        sim.advance_time(1.0)
        assert sim.get_state("switch.a").on is True
        assert sim.get_state("switch.b").on is False

        # After another 2s: b fires (total 3s)
        sim.advance_time(2.0)
        assert sim.get_state("switch.b").on is True

    def test_current_time(self):
        """Simulator tracks current simulated time."""
        sim = Simulator()
        sim.load_config({})
        assert sim.current_time == 0.0

        sim.advance_time(1.5)
        assert sim.current_time == 1.5

        sim.advance_time(3.0)
        assert sim.current_time == 4.5


# =============================================================================
# Timer State Tests
# =============================================================================


class TestTimerSimulation:
    """Tests for timer simulation behavior."""

    def test_start_timer(self):
        """Starting a timer sets it to active with full remaining time."""
        sim = Simulator()
        sim.load_config({})
        sim.start_timer("timer_1", duration=60.0)

        timer = sim.get_timer("timer_1")
        assert timer.active is True
        assert timer.total == 60.0
        assert timer.remaining == 60.0

    def test_advance_time_ticks_timer(self):
        """Advancing time reduces timer remaining."""
        sim = Simulator()
        sim.load_config({})
        sim.start_timer("timer_1", duration=60.0)

        sim.advance_time(10.0)
        timer = sim.get_timer("timer_1")
        assert timer.remaining == 50.0
        assert timer.active is True

    def test_timer_finishes(self):
        """Timer finishing dispatches timer_finished event."""
        sim = Simulator()
        sim.load_config({})

        finished_timers = []
        sim.register_event_handler(
            "timer_finished", lambda e: finished_timers.append(e.data.get("timer_id"))
        )

        sim.start_timer("timer_1", duration=5.0)
        sim.advance_time(5.0)

        timer = sim.get_timer("timer_1")
        assert timer.active is False
        assert timer.remaining == 0.0
        assert "timer_1" in finished_timers

    def test_cancel_timer(self):
        """Cancelling a timer dispatches timer_cancelled event."""
        sim = Simulator()
        sim.load_config({})

        cancelled = []
        sim.register_event_handler(
            "timer_cancelled", lambda e: cancelled.append(e.data.get("timer_id"))
        )

        sim.start_timer("timer_1", duration=60.0)
        sim.cancel_timer("timer_1")

        timer = sim.get_timer("timer_1")
        assert timer.active is False
        assert "timer_1" in cancelled

    def test_multiple_timers(self):
        """Multiple concurrent timers track independently."""
        sim = Simulator()
        sim.load_config({})

        sim.start_timer("timer_1", duration=10.0)
        sim.start_timer("timer_2", duration=20.0)

        sim.advance_time(10.0)

        t1 = sim.get_timer("timer_1")
        t2 = sim.get_timer("timer_2")

        assert t1.active is False  # finished
        assert t2.active is True
        assert t2.remaining == 10.0

    def test_get_timer_nonexistent(self):
        """Getting nonexistent timer returns inactive TimerState."""
        sim = Simulator()
        sim.load_config({})
        timer = sim.get_timer("nonexistent")
        assert timer.active is False

    def test_timer_tick_events(self):
        """Timer dispatches tick events on each advance_time while active."""
        sim = Simulator()
        sim.load_config({})

        ticks = []
        sim.register_event_handler("timer_tick", lambda e: ticks.append(e.data))

        sim.start_timer("timer_1", duration=5.0)

        sim.advance_time(1.0)
        sim.advance_time(1.0)

        # Should have 2 tick events
        assert len(ticks) == 2
        assert ticks[0]["timer_id"] == "timer_1"


# =============================================================================
# Voice Assistant State Transitions
# =============================================================================


class TestVoiceAssistantPhases:
    """Tests for voice assistant phase transitions."""

    def test_set_phase(self):
        config = {"voice_assistant": {"id": "va"}}
        sim = Simulator()
        sim.load_config(config)

        sim.set_voice_assistant_phase("voice_assistant.va", VoiceAssistantPhase.LISTENING)
        state = sim.get_state("voice_assistant.va")
        assert state.phase == VoiceAssistantPhase.LISTENING

    def test_phase_change_dispatches_event(self):
        config = {"voice_assistant": {"id": "va"}}
        sim = Simulator()
        sim.load_config(config)

        events = []
        sim.register_event_handler("voice_assistant_phase_changed", lambda e: events.append(e.data))

        sim.set_voice_assistant_phase("voice_assistant.va", VoiceAssistantPhase.THINKING)

        assert len(events) == 1
        assert events[0]["new_phase"] == VoiceAssistantPhase.THINKING
        assert events[0]["old_phase"] == VoiceAssistantPhase.IDLE


# =============================================================================
# Config Loading with Automations from YAML
# =============================================================================


class TestConfigAutomations:
    """Tests for loading automations from parsed YAML config."""

    def test_load_automation_from_config(self):
        """Automations defined in YAML config are auto-registered."""
        config = {
            "switch": [{"id": "timer_ringing", "platform": "template"}],
            "automation": [
                {
                    "trigger": "timer_finished",
                    "actions": [{"type": "switch.turn_on", "target": "switch.timer_ringing"}],
                }
            ],
        }
        sim = Simulator()
        sim.load_config(config)

        sim.dispatch_event("timer_finished")
        assert sim.get_state("switch.timer_ringing").on is True

    def test_load_automation_with_condition(self):
        """Conditions in YAML automations work correctly."""
        config = {
            "switch": [
                {"id": "mute", "platform": "template"},
                {"id": "target", "platform": "template"},
            ],
            "automation": [
                {
                    "trigger": "activate",
                    "conditions": [{"type": "switch.is_off", "target": "switch.mute"}],
                    "actions": [{"type": "switch.turn_on", "target": "switch.target"}],
                }
            ],
        }
        sim = Simulator()
        sim.load_config(config)

        # Mute is off → condition passes
        sim.dispatch_event("activate")
        assert sim.get_state("switch.target").on is True


# =============================================================================
# Reset / Snapshot Tests
# =============================================================================


class TestSimulatorReset:
    """Tests for simulator reset and state snapshots."""

    def test_reset(self):
        """Reset restores all states to defaults."""
        config = {
            "switch": [{"id": "a", "platform": "template"}],
            "light": [{"id": "b", "platform": "partition"}],
        }
        sim = Simulator()
        sim.load_config(config)

        sim.set_state("switch.a", SwitchState(on=True))
        sim.set_state("light.b", LightState(on=True, brightness=1.0))

        sim.reset()

        assert sim.get_state("switch.a").on is False
        assert sim.get_state("light.b").on is False
        assert sim.current_time == 0.0

    def test_snapshot(self):
        """Snapshot captures current state."""
        config = {
            "switch": [{"id": "a", "platform": "template"}],
        }
        sim = Simulator()
        sim.load_config(config)

        snapshot1 = sim.snapshot()
        assert snapshot1["switch.a"]["on"] is False

        sim.set_state("switch.a", SwitchState(on=True))
        snapshot2 = sim.snapshot()
        assert snapshot2["switch.a"]["on"] is True

        # Original snapshot unchanged
        assert snapshot1["switch.a"]["on"] is False


# =============================================================================
# Integration Test
# =============================================================================


class TestIntegration:
    """Integration tests simulating real Voice PE scenarios."""

    def test_timer_ring_workflow(self):
        """Simulates: timer finishes → switch on → LED effect."""
        config = {
            "switch": [{"id": "timer_ringing", "platform": "template"}],
            "light": [{"id": "led_ring", "platform": "partition"}],
        }
        sim = Simulator()
        sim.load_config(config)

        # Automation: timer_finished → turn on timer_ringing + LED effect
        sim.register_automation(
            Automation(
                trigger="timer_finished",
                actions=[
                    Action(type="switch.turn_on", target="switch.timer_ringing"),
                    Action(
                        type="light.turn_on",
                        target="light.led_ring",
                        params={"brightness": 1.0, "effect": "Timer Ring"},
                    ),
                ],
            )
        )

        # Start a timer
        sim.start_timer("cooking", duration=3.0)

        # Advance past timer duration
        sim.advance_time(3.0)

        # Timer finished → automation should have fired
        assert sim.get_state("switch.timer_ringing").on is True
        state = sim.get_state("light.led_ring")
        assert state.on is True
        assert state.effect == "Timer Ring"

    def test_button_press_stops_timer(self):
        """Simulates: button press while timer ringing → stop timer."""
        config = {
            "switch": [{"id": "timer_ringing", "platform": "template"}],
        }
        sim = Simulator()
        sim.load_config(config)

        # Set timer ringing
        sim.set_state("switch.timer_ringing", SwitchState(on=True))

        # Automation: button_pressed + timer_ringing is on → turn off
        sim.register_automation(
            Automation(
                trigger="button_pressed",
                conditions=[Condition(type="switch.is_on", target="switch.timer_ringing")],
                actions=[Action(type="switch.turn_off", target="switch.timer_ringing")],
            )
        )

        sim.dispatch_event("button_pressed")
        assert sim.get_state("switch.timer_ringing").on is False

    def test_voice_assistant_full_cycle(self):
        """Simulates: wake word → listening → thinking → replying → idle."""
        config = {"voice_assistant": {"id": "va"}}
        sim = Simulator()
        sim.load_config(config)

        phases = []
        sim.register_event_handler(
            "voice_assistant_phase_changed",
            lambda e: phases.append(e.data["new_phase"]),
        )

        sim.set_voice_assistant_phase("voice_assistant.va", VoiceAssistantPhase.WAITING_FOR_COMMAND)
        sim.set_voice_assistant_phase("voice_assistant.va", VoiceAssistantPhase.LISTENING)
        sim.set_voice_assistant_phase("voice_assistant.va", VoiceAssistantPhase.THINKING)
        sim.set_voice_assistant_phase("voice_assistant.va", VoiceAssistantPhase.REPLYING)
        sim.set_voice_assistant_phase("voice_assistant.va", VoiceAssistantPhase.IDLE)

        assert phases == [
            VoiceAssistantPhase.WAITING_FOR_COMMAND,
            VoiceAssistantPhase.LISTENING,
            VoiceAssistantPhase.THINKING,
            VoiceAssistantPhase.REPLYING,
            VoiceAssistantPhase.IDLE,
        ]

    def test_delayed_automation_in_integration(self):
        """Simulates: action with delay in a realistic scenario."""
        config = {
            "switch": [{"id": "amp", "platform": "gpio"}],
            "light": [{"id": "leds", "platform": "partition"}],
        }
        sim = Simulator()
        sim.load_config(config)

        # Automation: startup → turn on amp, wait 1s, then turn on LEDs
        sim.register_automation(
            Automation(
                trigger="startup",
                actions=[
                    Action(type="switch.turn_on", target="switch.amp"),
                    Action(type="delay", params={"seconds": 1.0}),
                    Action(
                        type="light.turn_on",
                        target="light.leds",
                        params={"brightness": 0.8, "effect": "Twinkle"},
                    ),
                ],
            )
        )

        sim.dispatch_event("startup")

        # Amp on immediately, LEDs not yet
        assert sim.get_state("switch.amp").on is True
        assert sim.get_state("light.leds").on is False

        # Advance past delay
        sim.advance_time(1.0)
        assert sim.get_state("light.leds").on is True
        assert sim.get_state("light.leds").effect == "Twinkle"
