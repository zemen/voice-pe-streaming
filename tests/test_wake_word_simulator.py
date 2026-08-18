"""Tests for wake word and voice assistant simulation."""

import pytest
from esphome_yaml_tools.simulator.voice_assistant import (
    MicroWakeWordSimulator,
    VoiceAssistantSimulator,
    WakeWordModel,
)
from esphome_yaml_tools.simulator.states import VoiceAssistantPhase


class TestMicroWakeWordSimulator:
    """Test micro wake word simulator."""

    def test_default_models_enabled(self):
        sim = MicroWakeWordSimulator()
        assert sim.is_model_enabled(WakeWordModel.OKAY_NABU.value)
        assert sim.is_model_enabled(WakeWordModel.HEY_JARVIS.value)
        assert sim.is_model_enabled(WakeWordModel.HEY_MYCROFT.value)
        assert not sim.is_model_enabled(WakeWordModel.STOP.value)

    def test_enable_model(self):
        sim = MicroWakeWordSimulator()
        sim.enable_model(WakeWordModel.STOP.value)
        assert sim.is_model_enabled(WakeWordModel.STOP.value)

    def test_disable_model(self):
        sim = MicroWakeWordSimulator()
        sim.disable_model(WakeWordModel.OKAY_NABU.value)
        assert not sim.is_model_enabled(WakeWordModel.OKAY_NABU.value)

    def test_detection_when_enabled(self):
        sim = MicroWakeWordSimulator()
        detected = []
        sim.on_detection = lambda model: detected.append(model)

        result = sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        assert result is True
        assert detected == [WakeWordModel.OKAY_NABU.value]

    def test_detection_when_disabled(self):
        sim = MicroWakeWordSimulator()
        detected = []
        sim.on_detection = lambda model: detected.append(model)
        sim.disable_model(WakeWordModel.OKAY_NABU.value)

        result = sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        assert result is False
        assert detected == []

    def test_detection_when_muted(self):
        sim = MicroWakeWordSimulator()
        detected = []
        sim.on_detection = lambda model: detected.append(model)
        sim.set_muted(True)

        result = sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        assert result is False
        assert detected == []

    def test_unmute_allows_detection(self):
        sim = MicroWakeWordSimulator()
        detected = []
        sim.on_detection = lambda model: detected.append(model)
        sim.set_muted(True)
        sim.set_muted(False)

        result = sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        assert result is True
        assert detected == [WakeWordModel.OKAY_NABU.value]

    def test_reset(self):
        sim = MicroWakeWordSimulator()
        sim.disable_model(WakeWordModel.OKAY_NABU.value)
        sim.enable_model(WakeWordModel.STOP.value)
        sim.set_muted(True)

        sim.reset()

        assert sim.is_model_enabled(WakeWordModel.OKAY_NABU.value)
        assert not sim.is_model_enabled(WakeWordModel.STOP.value)
        assert not sim.is_muted


class TestVoiceAssistantSimulator:
    """Test voice assistant simulator."""

    def test_initial_phase_is_idle(self):
        sim = VoiceAssistantSimulator()
        assert sim.get_phase() == VoiceAssistantPhase.IDLE

    def test_wake_word_triggers_listening(self):
        sim = VoiceAssistantSimulator()

        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        assert sim.get_phase() == VoiceAssistantPhase.LISTENING

    def test_wake_word_when_muted_ignored(self):
        sim = VoiceAssistantSimulator()
        sim.set_muted(True)

        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        assert sim.get_phase() == VoiceAssistantPhase.IDLE

    def test_wake_word_when_disabled_ignored(self):
        sim = VoiceAssistantSimulator()
        sim.disable_model(WakeWordModel.OKAY_NABU.value)

        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        assert sim.get_phase() == VoiceAssistantPhase.IDLE

    def test_stt_end_triggers_thinking(self):
        sim = VoiceAssistantSimulator()
        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        sim.on_stt_end()

        assert sim.get_phase() == VoiceAssistantPhase.THINKING

    def test_intent_end_triggers_replying(self):
        sim = VoiceAssistantSimulator()
        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)
        sim.on_stt_end()

        sim.on_intent_end()

        assert sim.get_phase() == VoiceAssistantPhase.REPLYING

    def test_tts_end_triggers_idle(self):
        sim = VoiceAssistantSimulator()
        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)
        sim.on_stt_end()
        sim.on_intent_end()

        sim.on_tts_end()

        assert sim.get_phase() == VoiceAssistantPhase.IDLE

    def test_full_cycle(self):
        sim = VoiceAssistantSimulator()
        phases = []
        sim.on_phase_change = lambda old, new: phases.append((old, new))

        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)
        sim.on_stt_end()
        sim.on_intent_end()
        sim.on_tts_end()

        assert phases == [
            (VoiceAssistantPhase.IDLE, VoiceAssistantPhase.LISTENING),
            (VoiceAssistantPhase.LISTENING, VoiceAssistantPhase.THINKING),
            (VoiceAssistantPhase.THINKING, VoiceAssistantPhase.REPLYING),
            (VoiceAssistantPhase.REPLYING, VoiceAssistantPhase.IDLE),
        ]

    def test_error_returns_to_idle(self):
        sim = VoiceAssistantSimulator()
        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)
        sim.on_stt_end()

        sim.on_error()

        assert sim.get_phase() == VoiceAssistantPhase.IDLE

    def test_wake_word_during_listening_ignored(self):
        sim = VoiceAssistantSimulator()
        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)

        sim.detect_wake_word(WakeWordModel.HEY_JARVIS.value)

        assert sim.get_phase() == VoiceAssistantPhase.LISTENING

    def test_enable_stop_model(self):
        sim = VoiceAssistantSimulator()

        sim.enable_model(WakeWordModel.STOP.value)

        assert sim.is_model_enabled(WakeWordModel.STOP.value)

    def test_stop_model_detection(self):
        sim = VoiceAssistantSimulator()
        sim.enable_model(WakeWordModel.STOP.value)

        result = sim.detect_wake_word(WakeWordModel.STOP.value)

        assert result is True
        assert sim.get_phase() == VoiceAssistantPhase.LISTENING

    def test_reset(self):
        sim = VoiceAssistantSimulator()
        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)
        sim.on_stt_end()
        sim.set_muted(True)
        sim.enable_model(WakeWordModel.STOP.value)

        sim.reset()

        assert sim.get_phase() == VoiceAssistantPhase.IDLE
        assert not sim.is_muted()
        assert not sim.is_model_enabled(WakeWordModel.STOP.value)

    def test_snapshot(self):
        sim = VoiceAssistantSimulator()
        sim.detect_wake_word(WakeWordModel.OKAY_NABU.value)
        sim.set_muted(True)
        sim.enable_model(WakeWordModel.STOP.value)

        snapshot = sim.snapshot()

        assert snapshot["phase"] == VoiceAssistantPhase.LISTENING.value
        assert snapshot["is_muted"] is True
        assert WakeWordModel.STOP.value in snapshot["enabled_models"]


class TestAcceptanceCriteria:
    """Test acceptance criteria from plan."""

    def test_detect_wake_word_triggers_event(self):
        sim = VoiceAssistantSimulator()

        sim.detect_wake_word("okay_nabu")

        assert sim.get_phase() == VoiceAssistantPhase.LISTENING

    def test_detection_when_muted_is_ignored(self):
        sim = VoiceAssistantSimulator()
        sim.set_muted(True)

        sim.detect_wake_word("okay_nabu")

        assert sim.get_phase() == VoiceAssistantPhase.IDLE

    def test_enable_model_works(self):
        sim = VoiceAssistantSimulator()

        sim.enable_model("stop")

        assert sim.is_model_enabled("stop")

    def test_voice_assistant_transitions_through_all_phases(self):
        sim = VoiceAssistantSimulator()

        assert sim.get_phase() == VoiceAssistantPhase.IDLE

        sim.detect_wake_word("okay_nabu")
        assert sim.get_phase() == VoiceAssistantPhase.LISTENING

        sim.on_stt_end()
        assert sim.get_phase() == VoiceAssistantPhase.THINKING

        sim.on_intent_end()
        assert sim.get_phase() == VoiceAssistantPhase.REPLYING

        sim.on_tts_end()
        assert sim.get_phase() == VoiceAssistantPhase.IDLE
