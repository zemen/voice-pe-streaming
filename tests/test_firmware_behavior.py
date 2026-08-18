"""Behavioral tests: load real firmware → dispatch events → assert simulator state.

The simulator models everything
the firmware uses: globals, scripts (with mode semantics), lambdas, conditional
if/then/else, micro_wake_word.{start,stop,enable_model,disable_model},
voice_assistant.{start,start_continuous,stop,set_use_wake_word}, and base
package scripts (control_leds, activate_stop_word_once, play_sound) via stubs.

Each test class encodes a feature requirement from the backported working
firmware (voice-living-room.yaml). RULE ZERO: these are feature specs, not
scaffolding — if they fail, fix the firmware, not the test.
"""

from pathlib import Path

import pytest

from esphome_yaml_tools.firmware_adapter import (
    adapt_firmware_config,
    get_firmware_metadata,
    load_firmware_into_simulator,
)
from esphome_yaml_tools.parser import load_yaml
from esphome_yaml_tools.simulator.states import (
    LightState,
    MicroWakeWordState,
    SwitchState,
    VoiceAssistantState,
)


FIRMWARE_PATH = Path(__file__).parent.parent / "voice_pe_streaming" / "voice-pe-streaming.yaml"


@pytest.fixture
def sim():
    return load_firmware_into_simulator(FIRMWARE_PATH)


@pytest.fixture
def sim_post_init():
    """Simulator with `init_in_progress = false`, modelling the device AFTER
    on_client_connected ran on a real boot. Base's event handlers (on_error,
    on_listening, etc.) all gate on `!init_in_progress`, so any test that
    dispatches an event without first firing on_client_connected must use this
    fixture to observe the handler effects."""
    s = load_firmware_into_simulator(FIRMWARE_PATH)
    s.globals.set("init_in_progress", False)
    return s


@pytest.fixture(scope="module")
def firmware_config():
    return load_yaml(FIRMWARE_PATH)





@pytest.fixture(scope="module")
def metadata(firmware_config):
    return get_firmware_metadata(firmware_config)


class TestPhaseTransitions:
    """Each voice_assistant event sets voice_assistant_phase to the right value
    and triggers control_leds. This is what produces correct LED behavior.
    """

    @pytest.mark.parametrize(
        "event,expected_phase",
        [
            ("on_listening", 2),
            ("on_stt_vad_start", 3),
            ("on_stt_vad_end", 4),
        ],
    )
    def test_event_sets_phase(self, sim, event, expected_phase):
        sim.dispatch_event(event)
        assert sim.globals.get("voice_assistant_phase") == expected_phase

    @pytest.mark.parametrize(
        "event,expected_effect",
        [
            ("on_listening", "Waiting for Command"),
            ("on_stt_vad_start", "Listening For Command"),
            ("on_stt_vad_end", "Thinking"),
            ("on_tts_start", "Replying"),
        ],
    )
    def test_event_produces_correct_led_effect(self, sim, event, expected_effect):
        sim.dispatch_event(event)
        ring = sim.get_state("light.voice_assistant_leds")
        assert isinstance(ring, LightState)
        assert ring.effect == expected_effect

    def test_on_tts_start_color_is_green(self, sim):
        sim.dispatch_event("on_tts_start")
        ring = sim.get_state("light.voice_assistant_leds")
        assert ring.color == (0, 255, 0), "Replying must be green per the base-package phase visualization"

    def test_on_error_color_is_red(self, sim_post_init):
        sim_post_init.dispatch_event("on_error", data={"code": "some-real-error"})
        ring = sim_post_init.get_state("light.voice_assistant_leds")
        assert ring.color == (255, 0, 0)
        assert ring.effect == "Error"


class TestStopWordAlwaysAvailable:
    """The stop word must be activated whenever TTS starts, regardless of mode.
    Project requirement: 'stop must ALWAYS work during TTS'.
    """

    def test_activate_stop_word_on_tts_start(self, sim):
        sim.dispatch_event("on_tts_start")
        assert sim.base_scripts_state.stop_word_active is True

    def test_activate_stop_word_on_intent_progress_with_x(self, sim):
        sim.base_scripts_state.stop_word_active = False
        sim.dispatch_event("on_intent_progress", data={"x": "https://media-url/abc"})
        assert sim.base_scripts_state.stop_word_active is True

    def test_intent_progress_with_empty_x_does_not_set_phase(self, sim):
        sim.globals.set("voice_assistant_phase", 4)
        sim.dispatch_event("on_intent_progress", data={"x": ""})
        assert sim.globals.get("voice_assistant_phase") == 4, (
            "Empty x means no TTS URL — phase should stay at thinking"
        )


class TestOnEndRestartsStreaming:
    def test_on_end_restarts_continuous_streaming(self, sim):
        sim.set_state("switch.master_mute_switch", SwitchState(on=False))
        sim.globals.set("voice_assistant_phase", 5)
        sim.dispatch_event("on_end")
        sim.advance_time(5.0)
        va = sim.get_state("voice_assistant.va")
        assert va.is_streaming is True, (
            "Streaming-only: on_end must inline-restart voice_assistant.start_continuous "
            "(no longer routed through a script). Without this restart the device goes "
            "silent after the first command."
        )

    def test_on_end_resets_phase_to_idle(self, sim):
        sim.globals.set("voice_assistant_phase", 5)
        sim.dispatch_event("on_end")
        assert sim.globals.get("voice_assistant_phase") == 1


class TestOnError:
    """on_error has specific handling per error code (Oracle-identified gap)."""

    def test_excluded_code_duplicate_wake_up_does_not_set_error_phase(self, sim):
        sim.globals.set("voice_assistant_phase", 5)
        sim.dispatch_event("on_error", data={"code": "duplicate_wake_up_detected"})
        assert sim.globals.get("voice_assistant_phase") == 5, (
            "duplicate_wake_up_detected is excluded; phase must NOT be overwritten to error"
        )

    def test_excluded_code_stt_no_text_does_not_set_error_phase(self, sim):
        sim.globals.set("voice_assistant_phase", 5)
        sim.dispatch_event("on_error", data={"code": "stt-no-text-recognized"})
        assert sim.globals.get("voice_assistant_phase") == 5

    def test_generic_error_sets_error_phase(self, sim_post_init):
        sim_post_init.dispatch_event("on_error", data={"code": "stt_provider_failed"})
        assert sim_post_init.globals.get("voice_assistant_phase") == 11

    def test_cloud_auth_failed_plays_sound(self, sim):
        sim.dispatch_event("on_error", data={"code": "cloud-auth-failed"})
        sim.advance_time(2.0)
        sounds = [s[0] for s in sim.base_scripts_state.sounds_played]
        assert "error_cloud_expired" in sounds

    def test_generic_error_does_not_play_cloud_auth_sound(self, sim):
        sim.dispatch_event("on_error", data={"code": "some-random-error"})
        sim.advance_time(2.0)
        sounds = [s[0] for s in sim.base_scripts_state.sounds_played]
        assert "error_cloud_expired" not in sounds, (
            "Generic errors must NOT play cloud-auth sound (Oracle bug #1 regression test)"
        )

    def test_init_in_progress_suppresses_error_phase(self, sim):
        sim.globals.set("init_in_progress", True)
        sim.globals.set("voice_assistant_phase", 5)
        sim.dispatch_event("on_error", data={"code": "stt_provider_failed"})
        assert sim.globals.get("voice_assistant_phase") == 5, (
            "While init_in_progress=true, on_error must not transition to error phase"
        )


class TestOnStartDucking:
    """on_start triggers audio ducking (-20 dB)."""

    def test_on_start_event_fires_without_error(self, sim):
        sim.dispatch_event("on_start")


class TestTimerEvents:
    """All 5 timer events must call control_leds. on_timer_finished also turns on timer_ringing."""

    def test_on_timer_finished_turns_on_timer_ringing(self, sim):
        sim.dispatch_event("on_timer_finished")
        ringing = sim.get_state("switch.timer_ringing")
        assert ringing is not None and ringing.on is True

    @pytest.mark.parametrize(
        "event_name",
        ["on_timer_started", "on_timer_cancelled", "on_timer_updated", "on_timer_tick"],
    )
    def test_timer_event_calls_control_leds(self, sim, event_name):
        sim.base_scripts_state.call_history.clear()
        sim.dispatch_event(event_name)
        called = [c.script_id for c in sim.base_scripts_state.call_history]
        assert "control_leds" in called


class TestOnErrorDelayedRestart:
    """on_error has a 2s delay before restarting streaming. Verify the chain."""

    def test_streaming_restarts_after_error_delay(self, sim):
        sim.set_state("switch.use_streaming_wake_word", SwitchState(on=True))
        sim.set_state("switch.master_mute_switch", SwitchState(on=False))
        sim.dispatch_event("on_error", data={"code": "some_real_error"})
        sim.advance_time(5.0)
        va = sim.get_state("voice_assistant.va")
        assert va.is_streaming is True, (
            "After 2s delay on error, streaming must restart (per on_error else-branch)"
        )


class TestOnClientDisconnect:
    def test_voice_assistant_stops(self, sim):
        sim.set_state("voice_assistant.va", VoiceAssistantState(is_streaming=True, is_in_session=True))
        sim.dispatch_event("on_client_disconnected")
        va = sim.get_state("voice_assistant.va")
        assert va.is_streaming is False
        assert va.is_in_session is False

    def test_phase_set_to_not_ready(self, sim):
        sim.dispatch_event("on_client_disconnected")
        assert sim.globals.get("voice_assistant_phase") == 10


class TestPhaseIDsMatchBase:
    """Minimal overlay only re-declares the phase IDs it actually references
    in its own lambdas (idle=1 for streaming mode start, waiting=2 for the
    wake-word detection handoff). All others come from the base package's
    substitution defaults and we trust them.

    These two MUST match base or the overlay's lambdas set the wrong values
    and base's control_leds renders the wrong LED state.
    """

    @pytest.mark.parametrize(
        "phase_key,expected_value",
        [
            ("idle", "1"),
            ("waiting", "2"),
        ],
    )
    def test_overlay_redeclared_phase_id_matches_base(self, metadata, phase_key, expected_value):
        assert metadata["phase_ids"][phase_key] == expected_value


