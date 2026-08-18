"""Full integration tests for Voice PE firmware with streaming WW."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from esphome_yaml_tools.simulator.led_ring import LEDRingSimulator
from esphome_yaml_tools.simulator.timer import TimerSimulator
from esphome_yaml_tools.simulator.controls import (
    CenterButton,
    ControlEventType,
    RotaryDial,
    MuteSwitch,
    JackSensor,
)
from esphome_yaml_tools.simulator.voice_assistant import (
    MicroWakeWordSimulator,
    VoiceAssistantSimulator,
)
from esphome_yaml_tools.simulator.states import VoiceAssistantPhase
from wyoming_livekit_speechbrain.livekit_detector import LiveKitDetectorConfig, LiveKitDetector
from wyoming_livekit_speechbrain.speechbrain_verifier import SpeechBrainConfig, SpeechBrainVerifier
from wyoming_livekit_speechbrain.buffer import RingBufferConfig, RingBuffer, MidSentenceDetector


class TestLEDPatternIntegration:
    """Test all LED patterns work correctly."""

    def test_all_voice_assistant_led_patterns(self):
        led_ring = LEDRingSimulator()

        patterns = [
            "waiting_for_command",
            "listening",
            "thinking",
            "replying",
            "muted",
            "error",
        ]

        for pattern in patterns:
            led_ring.set_effect(pattern)
            led_ring.update(0.1)
            state = led_ring.get_led_state()
            assert len(state) == 12
            assert all(isinstance(led, tuple) and len(led) == 3 for led in state)

    def test_timer_led_patterns(self):
        led_ring = LEDRingSimulator()

        led_ring.set_effect("timer_ring")
        led_ring.update(0.1)
        state = led_ring.get_led_state()
        assert any(led != (0, 0, 0) for led in state)

        led_ring.set_effect("timer_tick", progress=0.5)
        led_ring.update(0.1)
        state = led_ring.get_led_state()
        lit_count = sum(1 for led in state if led != (0, 0, 0))
        assert 5 <= lit_count <= 7

    def test_control_led_patterns(self):
        led_ring = LEDRingSimulator()

        led_ring.set_effect("volume_display", volume=0.5)
        led_ring.update(0.1)
        state = led_ring.get_led_state()
        lit_count = sum(1 for led in state if led != (0, 0, 0))
        assert 5 <= lit_count <= 7

        led_ring.set_effect("center_button_touched")
        led_ring.update(0.1)
        state = led_ring.get_led_state()
        assert all(led != (0, 0, 0) for led in state)


class TestTimerIntegration:
    """Test timer lifecycle with LED integration."""

    def test_timer_full_lifecycle(self):
        timer = TimerSimulator()
        led_ring = LEDRingSimulator()

        timer_id = timer.start_timer(duration=10, timer_id="test")
        assert timer.is_any_timer_active

        info = timer.get_timer("test")
        led_ring.set_effect("timer_tick", progress=info.progress)
        led_ring.update(0.1)

        timer.advance_time(5)
        info = timer.get_timer("test")
        assert info.remaining_seconds == pytest.approx(5, abs=0.1)

        led_ring.set_effect("timer_tick", progress=info.progress)
        led_ring.update(0.1)
        state = led_ring.get_led_state()
        lit_count = sum(1 for led in state if led != (0, 0, 0))
        assert 5 <= lit_count <= 7

        timer.advance_time(5)
        assert not timer.is_any_timer_active

    def test_timer_stop_during_ring(self):
        timer = TimerSimulator()

        timer.start_timer(duration=1, timer_id="test")
        timer.advance_time(1.1)

        assert timer.is_ringing

        timer.stop_ringing()
        assert not timer.is_ringing
        assert not timer.is_any_timer_active


class TestControlsIntegration:
    """Test button and dial controls."""

    def test_button_sequences(self):
        button = CenterButton()
        events = []

        button.on_event(ControlEventType.SINGLE_PRESS, lambda e: events.append(e))
        button.on_event(ControlEventType.DOUBLE_PRESS, lambda e: events.append(e))
        button.on_event(ControlEventType.LONG_PRESS, lambda e: events.append(e))

        button.press_and_release(duration=0.1)
        button.advance_time(0.3)
        assert any(e.event_type == ControlEventType.SINGLE_PRESS for e in events)

        events.clear()
        button.press_and_release(duration=0.1)
        button.press_and_release(duration=0.1)
        button.advance_time(0.3)
        assert any(e.event_type == ControlEventType.DOUBLE_PRESS for e in events)

        events.clear()
        button.press_and_release(duration=1.5)
        assert any(e.event_type == ControlEventType.LONG_PRESS for e in events)

    def test_rotary_dial_volume_control(self):
        dial = RotaryDial(initial_volume=0.5)

        dial.rotate_clockwise()
        assert dial.volume == pytest.approx(0.55, abs=0.01)

        dial.rotate_anticlockwise()
        assert dial.volume == pytest.approx(0.5, abs=0.01)

    def test_rotary_dial_hue_control(self):
        dial = RotaryDial()
        dial.set_button_pressed(True)

        initial_hue = dial.hue
        dial.rotate_clockwise()
        assert dial.hue == (initial_hue + 15) % 360

    def test_mute_switch_integration(self):
        mute = MuteSwitch()
        events = []

        mute.on_event(ControlEventType.MUTED, lambda e: events.append(e))
        mute.on_event(ControlEventType.UNMUTED, lambda e: events.append(e))

        mute.set_muted(True)
        assert any(e.event_type == ControlEventType.MUTED for e in events)

        mute.set_muted(False)
        assert any(e.event_type == ControlEventType.UNMUTED for e in events)


class TestWakeWordIntegration:
    """Test wake word detection with voice assistant."""

    def test_micro_wake_word_triggers_voice_assistant(self):
        mww = MicroWakeWordSimulator()
        va = VoiceAssistantSimulator(wake_word_sim=mww)

        assert va.get_phase() == VoiceAssistantPhase.IDLE

        mww.detect_wake_word("okay_nabu")
        assert va.get_phase() == VoiceAssistantPhase.LISTENING

        va.on_stt_end()
        assert va.get_phase() == VoiceAssistantPhase.THINKING

        va.on_intent_end()
        assert va.get_phase() == VoiceAssistantPhase.REPLYING

        va.on_tts_end()
        assert va.get_phase() == VoiceAssistantPhase.IDLE

    def test_mute_blocks_wake_word(self):
        mww = MicroWakeWordSimulator()
        va = VoiceAssistantSimulator(wake_word_sim=mww)

        mww.set_muted(True)
        result = mww.detect_wake_word("okay_nabu")

        assert result is False
        assert va.get_phase() == VoiceAssistantPhase.IDLE

    def test_stop_model_works_independently(self):
        mww = MicroWakeWordSimulator()

        mww.enable_model("stop")
        result = mww.detect_wake_word("stop")

        assert result is True


class TestStreamingWWIntegration:
    """Test streaming wake word with SpeechBrain and mid-sentence detection."""

    def test_livekit_detection_flow(self):
        from unittest.mock import MagicMock, patch

        from wyoming_livekit_speechbrain.livekit_detector import (
            CHUNK_FRAMES,
            FRAME_BYTES,
            LiveKitStream,
        )

        with patch(
            "wyoming_livekit_speechbrain.livekit_detector.WakeWordModel"
        ) as mock_cls:
            model = MagicMock()
            model.predict.return_value = {"okay_nabu": 0.95}
            mock_cls.return_value = model

            detector = LiveKitDetector(
                LiveKitDetectorConfig(
                    model_paths=[Path("/fake/model.onnx")],
                    keywords=["okay_nabu"],
                )
            )
            stream = LiveKitStream(detector=detector)
            hit, candidate = stream.process_audio(b"\x00\x01" * (FRAME_BYTES * CHUNK_FRAMES // 2))
            assert hit is not None
            assert hit[0] == 0
            assert hit[1] == "okay_nabu"
            assert hit[2] == 0.95
            assert isinstance(hit[3], bytes)
            assert candidate is None

    def test_mid_sentence_detection_rejects_same_speaker(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_with_scores.return_value = {"john": 0.7}
        verifier.verify_speaker.return_value = "john"

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        should_reject = detector.should_reject_detection(b"ww_audio")
        assert should_reject is True

    def test_mid_sentence_detection_accepts_different_speaker(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.side_effect = ["john", "jane"]

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        should_reject = detector.should_reject_detection(b"ww_audio")
        assert should_reject is False


class TestFullVoicePEWorkflow:
    """Test complete Voice PE workflow scenarios."""

    def test_timer_with_stop_wake_word(self):
        timer = TimerSimulator()
        mww = MicroWakeWordSimulator()

        timer.start_timer(duration=10, timer_id="test")
        timer.advance_time(10.1)

        assert timer.is_ringing

        mww.enable_model("stop")
        result = mww.detect_wake_word("stop")

        assert result is True

        timer.stop_ringing()
        assert not timer.is_ringing

    def test_mute_blocks_wake_word_but_not_controls(self):
        mww = MicroWakeWordSimulator()
        dial = RotaryDial(initial_volume=0.5)

        mww.set_muted(True)

        ww_result = mww.detect_wake_word("okay_nabu")
        assert ww_result is False

        dial.rotate_clockwise()
        assert dial.volume == pytest.approx(0.55, abs=0.01)

    def test_voice_assistant_led_sync(self):
        mww = MicroWakeWordSimulator()
        va = VoiceAssistantSimulator(wake_word_sim=mww)
        led_ring = LEDRingSimulator()

        mww.detect_wake_word("okay_nabu")
        led_ring.set_effect("listening")
        led_ring.update(0.1)
        state = led_ring.get_led_state()
        assert any(led != (0, 0, 0) for led in state)

        va.on_stt_end()
        led_ring.set_effect("thinking")
        led_ring.update(0.1)
        state = led_ring.get_led_state()
        assert any(led != (0, 0, 0) for led in state)


class TestAcceptanceCriteria:
    """Verify all acceptance criteria from plan."""

    def test_all_led_patterns_render(self):
        led_ring = LEDRingSimulator()

        all_patterns = [
            "waiting_for_command",
            "listening",
            "thinking",
            "replying",
            "muted",
            "startup_failed",
            "volume_display",
            "center_button_touched",
            "twinkle",
            "error",
            "timer_ring",
            "timer_tick",
            "rainbow",
            "tick",
            "factory_reset",
            "jack_plugged",
            "jack_unplugged",
        ]

        for pattern in all_patterns:
            led_ring.set_effect(pattern)
            led_ring.update(0.1)
            state = led_ring.get_led_state()
            assert len(state) == 12

    def test_timer_lifecycle_complete(self):
        timer = TimerSimulator()

        timer.start_timer(duration=5, timer_id="test")
        assert timer.is_any_timer_active

        timer.advance_time(2)
        info = timer.get_timer("test")
        assert info.remaining_seconds == pytest.approx(3, abs=0.1)

        timer.advance_time(3)
        assert not timer.is_any_timer_active

    def test_all_button_sequences_work(self):
        button = CenterButton()
        events = []

        button.on_event(ControlEventType.SINGLE_PRESS, lambda e: events.append(e))
        button.on_event(ControlEventType.DOUBLE_PRESS, lambda e: events.append(e))
        button.on_event(ControlEventType.LONG_PRESS, lambda e: events.append(e))

        button.press_and_release(0.1)
        button.advance_time(0.3)
        assert any(e.event_type == ControlEventType.SINGLE_PRESS for e in events)

        events.clear()
        button.press_and_release(0.1)
        button.press_and_release(0.1)
        button.advance_time(0.3)
        assert any(e.event_type == ControlEventType.DOUBLE_PRESS for e in events)

        events.clear()
        button.press_and_release(1.5)
        assert any(e.event_type == ControlEventType.LONG_PRESS for e in events)

    def test_rotary_dial_both_modes(self):
        dial = RotaryDial(initial_volume=0.5)

        dial.rotate_clockwise()
        assert dial.volume > 0.5

        dial.set_button_pressed(True)
        initial_hue = dial.hue
        dial.rotate_clockwise()
        assert dial.hue != initial_hue

    def test_mute_and_jack_detection(self):
        mute = MuteSwitch()
        jack = JackSensor()

        mute_events = []
        jack_events = []

        mute.on_event(ControlEventType.MUTED, lambda e: mute_events.append(e))
        mute.on_event(ControlEventType.UNMUTED, lambda e: mute_events.append(e))
        jack.on_event(ControlEventType.JACK_PLUGGED, lambda e: jack_events.append(e))

        mute.set_muted(True)
        assert any(e.event_type == ControlEventType.MUTED for e in mute_events)

        jack.set_plugged(True)
        assert any(e.event_type == ControlEventType.JACK_PLUGGED for e in jack_events)

    def test_wake_word_full_cycle(self):
        mww = MicroWakeWordSimulator()
        va = VoiceAssistantSimulator(wake_word_sim=mww)

        mww.detect_wake_word("okay_nabu")
        assert va.get_phase() == VoiceAssistantPhase.LISTENING

        va.on_stt_end()
        va.on_intent_end()
        va.on_tts_end()
        assert va.get_phase() == VoiceAssistantPhase.IDLE
