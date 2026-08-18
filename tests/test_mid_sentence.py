"""Tests for ring buffer and mid-sentence detection."""

import pytest
from unittest.mock import Mock
from wyoming_livekit_speechbrain.buffer import (
    RingBufferConfig,
    RingBuffer,
    MidSentenceDetector,
)


class TestRingBufferConfig:
    def test_creates_with_defaults(self):
        config = RingBufferConfig()
        assert config.buffer_seconds == 1.0
        assert config.gap_ms == 2000
        assert config.sample_rate == 16000
        assert config.sample_width == 2

    def test_creates_with_custom_values(self):
        config = RingBufferConfig(
            buffer_seconds=2.0,
            gap_ms=300,
            sample_rate=48000,
            sample_width=4,
        )
        assert config.buffer_seconds == 2.0
        assert config.gap_ms == 300
        assert config.sample_rate == 48000
        assert config.sample_width == 4

    def test_negative_buffer_seconds_raises(self):
        with pytest.raises(ValueError, match="buffer_seconds must be positive"):
            RingBufferConfig(buffer_seconds=-1.0)

    def test_zero_buffer_seconds_raises(self):
        with pytest.raises(ValueError, match="buffer_seconds must be positive"):
            RingBufferConfig(buffer_seconds=0.0)

    def test_negative_gap_ms_raises(self):
        with pytest.raises(ValueError, match="gap_ms must be non-negative"):
            RingBufferConfig(gap_ms=-1)

    def test_buffer_size_bytes_calculation(self):
        config = RingBufferConfig(buffer_seconds=1.0, gap_ms=200, sample_rate=16000, sample_width=2)
        expected = int((1.0 + 0.2) * 16000 * 2)
        assert config.buffer_size_bytes == expected

    def test_gap_size_bytes_calculation(self):
        config = RingBufferConfig(gap_ms=200, sample_rate=16000, sample_width=2)
        expected = int(0.2 * 16000 * 2)
        assert config.gap_size_bytes == expected

    def test_pre_ww_size_bytes_calculation(self):
        config = RingBufferConfig(buffer_seconds=1.0, sample_rate=16000, sample_width=2)
        expected = int(1.0 * 16000 * 2)
        assert config.pre_ww_size_bytes == expected


class TestRingBuffer:
    def test_creates_empty(self):
        config = RingBufferConfig()
        buffer = RingBuffer(config)
        assert buffer.size == 0
        assert not buffer.is_full

    def test_write_audio_chunk(self):
        config = RingBufferConfig()
        buffer = RingBuffer(config)

        audio = b"\x00\x01\x02\x03"
        buffer.write(audio)

        assert buffer.size == 4

    def test_write_multiple_chunks(self):
        config = RingBufferConfig()
        buffer = RingBuffer(config)

        buffer.write(b"\x00\x01")
        buffer.write(b"\x02\x03")
        buffer.write(b"\x04\x05")

        assert buffer.size == 6

    def test_buffer_wraps_when_full(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=16000, sample_width=2)
        buffer = RingBuffer(config)

        max_size = config.buffer_size_bytes
        buffer.write(b"\xff" * (max_size + 10))

        assert buffer.size == max_size
        assert buffer.is_full

    def test_get_pre_ww_audio_returns_none_when_not_full(self):
        config = RingBufferConfig()
        buffer = RingBuffer(config)

        buffer.write(b"\x00\x01\x02\x03")

        assert buffer.get_pre_ww_audio() is None

    def test_get_pre_ww_audio_returns_segment_when_full(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        buffer = RingBuffer(config)

        audio = bytes(range(256))
        buffer.write(audio)

        pre_ww = buffer.get_pre_ww_audio()
        assert pre_ww is not None
        assert len(pre_ww) == config.pre_ww_size_bytes

    def test_get_pre_ww_audio_excludes_gap(self):
        config = RingBufferConfig(
            buffer_seconds=0.001, gap_ms=100, sample_rate=1000, sample_width=1
        )
        buffer = RingBuffer(config)

        total_size = config.buffer_size_bytes
        buffer.write(bytes(range(total_size)))

        pre_ww = buffer.get_pre_ww_audio()
        assert pre_ww is not None
        assert len(pre_ww) == config.pre_ww_size_bytes
        assert len(pre_ww) < total_size

    def test_get_recent_audio_returns_none_when_insufficient(self):
        config = RingBufferConfig()
        buffer = RingBuffer(config)

        buffer.write(b"\x00\x01")

        assert buffer.get_recent_audio(1000) is None

    def test_get_recent_audio_returns_segment(self):
        config = RingBufferConfig(sample_rate=1000, sample_width=1)
        buffer = RingBuffer(config)

        buffer.write(bytes(range(200)))

        recent = buffer.get_recent_audio(100)
        assert recent is not None
        assert len(recent) == 100

    def test_get_recent_audio_returns_most_recent(self):
        config = RingBufferConfig(sample_rate=1000, sample_width=1)
        buffer = RingBuffer(config)

        buffer.write(bytes(range(200)))

        recent = buffer.get_recent_audio(50)
        assert recent == bytes(range(150, 200))

    def test_clear_resets_buffer(self):
        config = RingBufferConfig()
        buffer = RingBuffer(config)

        buffer.write(b"\x00\x01\x02\x03")
        buffer.clear()

        assert buffer.size == 0
        assert not buffer.is_full

    def test_is_full_property(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        buffer = RingBuffer(config)

        assert not buffer.is_full

        buffer.write(b"\xff" * config.buffer_size_bytes)

        assert buffer.is_full


class TestMidSentenceDetector:
    def test_creates_with_ring_buffer_and_verifier(self):
        config = RingBufferConfig()
        ring_buffer = RingBuffer(config)
        verifier = Mock()

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        assert detector.ring_buffer is ring_buffer
        assert detector.speaker_verifier is verifier

    def test_should_reject_when_same_speaker(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.return_value = "john"

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        result = detector.should_reject_detection(b"ww_audio")

        assert result is True
        assert verifier.verify_speaker.call_count == 2

    def test_should_accept_when_different_speaker(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.side_effect = ["john", "jane"]

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        result = detector.should_reject_detection(b"ww_audio")

        assert result is False

    def test_should_accept_when_pre_ww_speaker_none(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.side_effect = [None, "john"]

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        result = detector.should_reject_detection(b"ww_audio")

        assert result is False

    def test_should_accept_when_ww_speaker_none(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.side_effect = ["john", None]

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        result = detector.should_reject_detection(b"ww_audio")

        assert result is False

    def test_should_accept_when_both_speakers_none(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.return_value = None

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        result = detector.should_reject_detection(b"ww_audio")

        assert result is False

    def test_should_accept_when_buffer_not_full(self):
        config = RingBufferConfig()
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\x00\x01")

        verifier = Mock()

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        result = detector.should_reject_detection(b"ww_audio")

        assert result is False
        verifier.verify_speaker.assert_not_called()

    def test_reset_clears_ring_buffer(self):
        config = RingBufferConfig()
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\x00\x01\x02\x03")

        verifier = Mock()
        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        detector.reset()

        assert ring_buffer.size == 0


class TestAcceptanceCriteria:
    def test_ring_buffer_maintains_1_2s_of_audio_history(self):
        config = RingBufferConfig(buffer_seconds=1.0, gap_ms=200, sample_rate=16000, sample_width=2)
        buffer = RingBuffer(config)

        expected_size = int(1.2 * 16000 * 2)
        assert config.buffer_size_bytes == expected_size

        buffer.write(b"\xff" * expected_size)
        assert buffer.is_full
        assert buffer.size == expected_size

    def test_pre_ww_audio_extracted_correctly_on_detection(self):
        config = RingBufferConfig(buffer_seconds=1.0, gap_ms=200, sample_rate=16000, sample_width=2)
        buffer = RingBuffer(config)

        total_size = config.buffer_size_bytes
        buffer.write(bytes(range(256)) * (total_size // 256 + 1))

        pre_ww = buffer.get_pre_ww_audio()

        assert pre_ww is not None
        assert len(pre_ww) == config.pre_ww_size_bytes
        assert len(pre_ww) == int(1.0 * 16000 * 2)

    def test_same_speaker_before_and_during_ww_rejected(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.return_value = "alice"

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        should_reject = detector.should_reject_detection(b"ww_audio")

        assert should_reject is True

    def test_different_speaker_before_ww_accepted(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.side_effect = ["alice", "bob"]

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        should_reject = detector.should_reject_detection(b"ww_audio")

        assert should_reject is False

    def test_silence_before_ww_accepted(self):
        config = RingBufferConfig(buffer_seconds=0.001, gap_ms=0, sample_rate=1000, sample_width=1)
        ring_buffer = RingBuffer(config)
        ring_buffer.write(b"\xff" * config.buffer_size_bytes)

        verifier = Mock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.side_effect = [None, "alice"]

        detector = MidSentenceDetector(ring_buffer, verifier, silence_rms_threshold=0.0)

        should_reject = detector.should_reject_detection(b"ww_audio")

        assert should_reject is False
