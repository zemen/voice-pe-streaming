"""Tests for the Wyoming LiveKit + SpeechBrain event handler.

Mocks the per-handler ``LiveKitStream`` directly via ``handler._stream`` so
detection-vs-no-detection can be controlled deterministically without driving
the full 2-second audio window through the real model. The shared
``LiveKitDetector`` is stubbed but its internals are never exercised here.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.modules.setdefault("livekit", MagicMock())
sys.modules.setdefault("livekit.wakeword", MagicMock())
sys.modules.setdefault("speechbrain", MagicMock())
sys.modules.setdefault("speechbrain.inference", MagicMock())
sys.modules.setdefault("speechbrain.inference.speaker", MagicMock())

from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402
from wyoming.info import Describe, Info  # noqa: E402
from wyoming.wake import Detect, Detection, NotDetected  # noqa: E402

from wyoming_livekit_speechbrain.handler import LiveKitEventHandler  # noqa: E402


def _make_detector() -> MagicMock:
    detector = MagicMock()
    detector.frame_length = 1280
    detector.sample_rate = 16000
    detector.chunk_frames = 25
    detector.predict.return_value = None
    detector._model = object()
    return detector


def _make_handler(
    *,
    speaker_verifier=None,
    sample_store=None,
    enable_mid_sentence_detection: bool = True,
    stream_result=None,
    stream_candidate=None,
):
    """Build a LiveKitEventHandler whose internal stream is fully mocked.

    `stream_result` and `stream_candidate` are each the legacy single-hit 4-tuple
    `(idx, name, score, pcm)` or `None`. They are wrapped into the new
    `(hit, candidate)` shape returned by `LiveKitStream.process_audio()` in. Both `None` ⇒ no detection; set `stream_candidate` only to
    exercise the below-threshold save path without firing a Wyoming Detection.
    """
    reader = AsyncMock(spec=asyncio.StreamReader)
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.write = MagicMock()
    writer.writelines = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()

    detector = _make_detector()
    handler = LiveKitEventHandler(
        reader=reader,
        writer=writer,
        wake_word_names=["hey_livekit"],
        model_paths=[Path("/fake/hey_livekit.onnx")],
        livekit=detector,
        speaker_verifier=speaker_verifier,
        enable_mid_sentence_detection=enable_mid_sentence_detection,
        sample_store=sample_store,
    )
    stream_mock = MagicMock()
    stream_mock.process_audio.return_value = (stream_result, stream_candidate)
    stream_mock.process_audio_async = AsyncMock(return_value=(stream_result, stream_candidate))
    stream_mock.reset = MagicMock()
    handler._stream = stream_mock
    return handler, stream_mock, writer


class TestDescribe:
    @pytest.mark.asyncio
    async def test_describe_emits_info(self):
        handler, _, writer = _make_handler()
        await handler.handle_event(Describe().event())
        assert writer.drain.called

    @pytest.mark.asyncio
    async def test_describe_advertises_livekit_attribution(self):
        handler, _, _ = _make_handler()
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))
        await handler.handle_event(Describe().event())
        assert len(captured) == 1
        info = Info.from_event(captured[0])
        assert len(info.wake) == 1
        assert info.wake[0].name == "livekit-wakeword"
        assert info.wake[0].attribution.name == "LiveKit"
        assert "livekit" in info.wake[0].attribution.url


class TestDetectFlow:
    @pytest.mark.asyncio
    async def test_detect_starts_detection_mode(self):
        handler, _, _ = _make_handler()
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        assert handler.is_detecting is True

    @pytest.mark.asyncio
    async def test_audio_chunk_passed_to_stream(self):
        handler, stream, _ = _make_handler()
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x01" * 100, timestamp=0)
        await handler.handle_event(chunk.event())
        stream.process_audio_async.assert_called_with(b"\x00\x01" * 100)

    @pytest.mark.asyncio
    async def test_audio_stop_emits_not_detected(self):
        handler, _, _ = _make_handler()
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(AudioStop().event())
        nd_events = [ev for ev in captured if NotDetected.is_type(ev.type)]
        assert len(nd_events) == 1


class TestDetectionEmission:
    @pytest.mark.asyncio
    async def test_detection_emits_detection_event(self):
        handler, _, _ = _make_handler(stream_result=(0, "hey_livekit", 0.95, b"\x00" * 64000))
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00" * 200, timestamp=42).event()
        )
        det_events = [ev for ev in captured if Detection.is_type(ev.type)]
        assert len(det_events) == 1
        d = Detection.from_event(det_events[0])
        assert d.name == "hey_livekit"
        assert d.timestamp == 42
        assert d.speaker is None


class TestSpeakerVerification:
    @pytest.mark.asyncio
    async def test_verified_speaker_attached(self):
        verifier = MagicMock()
        verifier.config.threshold = 0.25
        verifier.verify_with_scores.return_value = {"alice": 0.60}
        handler, _, _ = _make_handler(
            speaker_verifier=verifier,
            enable_mid_sentence_detection=False,
            stream_result=(0, "hey_livekit", 0.95, b"\x00" * 64000),
        )
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00" * 800, timestamp=10).event()
        )
        det_events = [ev for ev in captured if Detection.is_type(ev.type)]
        assert len(det_events) == 1
        d = Detection.from_event(det_events[0])
        assert d.speaker == "alice"

    @pytest.mark.asyncio
    async def test_unauthorized_speaker_rejects(self):
        verifier = MagicMock()
        verifier.config.threshold = 0.25
        verifier.verify_with_scores.return_value = {"alice": 0.10}
        handler, _, _ = _make_handler(
            speaker_verifier=verifier,
            enable_mid_sentence_detection=False,
            stream_result=(0, "hey_livekit", 0.95, b"\x00" * 64000),
        )
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00" * 800, timestamp=10).event()
        )
        det_events = [ev for ev in captured if Detection.is_type(ev.type)]
        assert len(det_events) == 0

    @pytest.mark.asyncio
    async def test_speaker_verifier_creates_ring_buffer(self):
        verifier = MagicMock()
        handler, _, _ = _make_handler(speaker_verifier=verifier)
        assert handler._ring_buffer is not None
        assert handler.mid_sentence_detector is not None

    @pytest.mark.asyncio
    async def test_no_speaker_verifier_no_mid_sentence_detector(self):
        handler, _, _ = _make_handler()
        assert handler._ring_buffer is not None
        assert handler.mid_sentence_detector is None


class TestMidSentenceRejection:
    @pytest.mark.asyncio
    async def test_mid_sentence_detection_rejects(self):
        # Real timing: in production the ring buffer (1.2s) fills before LiveKit's
        # 2s window fires a detection, so mid-sentence sees pre-WW audio. Mirror
        # that here by returning None until enough audio has been buffered.
        verifier = MagicMock()
        verifier.config.threshold = 0.25
        verifier.verify_speaker.return_value = "bob"
        handler, stream, _ = _make_handler(
            speaker_verifier=verifier,
            enable_mid_sentence_detection=True,
        )
        if handler.mid_sentence_detector is not None:
            handler.mid_sentence_detector.silence_rms_threshold = 0.0
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))

        # Ring buffer is 3s at 16k/16-bit = 96000 bytes (default gap_ms=2000).
        # Need ≥96000 bytes fed BEFORE the chunk that triggers detection.
        n_silent_chunks = 96
        results: list = [(None, None)] * n_silent_chunks + [
            ((0, "hey_livekit", 0.95, b"\x00" * 64000), None)
        ]
        stream.process_audio.side_effect = results

        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        for _ in range(n_silent_chunks + 1):
            await handler.handle_event(
                AudioChunk(
                    rate=16000, width=2, channels=1, audio=b"\xff" * 1024, timestamp=0
                ).event()
            )
        det_events = [ev for ev in captured if Detection.is_type(ev.type)]
        assert len(det_events) == 0


class TestSharedDetectorIsolation:
    @pytest.mark.asyncio
    async def test_two_handlers_share_detector_independently(self):
        # Verifies Oracle BLOCKER #4 fix: the shared LiveKitDetector instance
        # must not have mutable per-connection state. Each handler builds its
        # own LiveKitStream from the shared detector.
        reader_a = AsyncMock(spec=asyncio.StreamReader)
        writer_a = MagicMock(spec=asyncio.StreamWriter)
        writer_a.drain = AsyncMock()
        reader_b = AsyncMock(spec=asyncio.StreamReader)
        writer_b = MagicMock(spec=asyncio.StreamWriter)
        writer_b.drain = AsyncMock()

        detector = _make_detector()
        handler_a = LiveKitEventHandler(
            reader=reader_a,
            writer=writer_a,
            wake_word_names=["hey_livekit"],
            model_paths=[Path("/fake/m.onnx")],
            livekit=detector,
        )
        handler_b = LiveKitEventHandler(
            reader=reader_b,
            writer=writer_b,
            wake_word_names=["hey_livekit"],
            model_paths=[Path("/fake/m.onnx")],
            livekit=detector,
        )
        # Each handler has its own stream — not the same object, not sharing the buffer.
        assert handler_a._stream is not handler_b._stream
        # Both reference the same shared detector.
        assert handler_a._stream.detector is detector
        assert handler_b._stream.detector is detector


class TestBelowThresholdSamplePersistence:
    """Feature: when LiveKitStream surfaces a candidate (sub-threshold
    near-miss) and a sample_store is wired, the handler MUST persist it with
    source='auto-below-threshold' and MUST NOT emit any Wyoming Detection event.
    Speaker verifier and mid-sentence detector are bypassed on this path — we
    want all near-misses (target speaker, other speakers, noise) for mining.
    """

    @pytest.mark.asyncio
    async def test_candidate_persists_sample_no_detection_emitted(self):
        sample_store = MagicMock()
        handler, stream, _ = _make_handler(
            sample_store=sample_store,
            stream_candidate=(0, "hey_livekit", 0.30, b"\x00" * 64000),
        )
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00" * 200, timestamp=99).event()
        )

        assert sample_store.add.call_count == 1
        _pcm, meta = sample_store.add.call_args[0]
        assert meta.source == "auto-below-threshold"
        assert meta.keyword == "hey_livekit"
        assert meta.score == pytest.approx(0.30)
        assert meta.accepted is False
        assert meta.speaker is None
        det_events = [ev for ev in captured if Detection.is_type(ev.type)]
        assert len(det_events) == 0, "candidate must NOT emit a Wyoming Detection"

    @pytest.mark.asyncio
    async def test_candidate_without_sample_store_is_silent(self):
        handler, stream, _ = _make_handler(
            stream_candidate=(0, "hey_livekit", 0.30, b"\x00" * 64000),
        )
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00" * 200, timestamp=0).event()
        )

        det_events = [ev for ev in captured if Detection.is_type(ev.type)]
        assert len(det_events) == 0
        assert handler.is_detecting is True, "candidate must not affect detection state"

    @pytest.mark.asyncio
    async def test_hit_path_still_works_with_sample_store(self):
        sample_store = MagicMock()
        handler, stream, _ = _make_handler(
            sample_store=sample_store,
            stream_result=(0, "hey_livekit", 0.95, b"\x00" * 64000),
        )
        captured = []
        handler.write_event = AsyncMock(side_effect=lambda ev: captured.append(ev))
        await handler.handle_event(Detect(names=["hey_livekit"]).event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00" * 200, timestamp=0).event()
        )

        assert sample_store.add.call_count == 1
        _pcm, meta = sample_store.add.call_args[0]
        assert meta.source == "auto"
        det_events = [ev for ev in captured if Detection.is_type(ev.type)]
        assert len(det_events) == 1
