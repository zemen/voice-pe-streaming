"""Tests for LiveKit wake word detection wrapper.

Verifies the upstream contract from livekit-wakeword: ``WakeWordModel.predict()``
is stateless and expects a ~2 second window of int16 PCM at 16 kHz. The
``LiveKitStream`` accumulates Wyoming chunks into 80 ms frames and slides
a 25-frame (~2 s) window before invoking the shared model.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault("livekit", MagicMock())
sys.modules.setdefault("livekit.wakeword", MagicMock())

from wyoming_livekit_speechbrain.livekit_detector import (  # noqa: E402
    CHUNK_FRAMES,
    FRAME_BYTES,
    LiveKitDetector,
    LiveKitDetectorConfig,
    LiveKitStream,
)


class TestLiveKitDetectorConfig:
    def test_creates_with_models_and_keywords(self):
        cfg = LiveKitDetectorConfig(
            model_paths=[Path("a.onnx"), Path("b.onnx")],
            keywords=["okay_nabu", "hey_jarvis"],
        )
        assert cfg.keywords == ["okay_nabu", "hey_jarvis"]

    def test_default_sensitivities(self):
        cfg = LiveKitDetectorConfig(
            model_paths=[Path("a.onnx"), Path("b.onnx")],
            keywords=["a", "b"],
        )
        assert cfg.sensitivities == [0.5, 0.5]

    def test_mismatched_models_keywords_raises(self):
        with pytest.raises(ValueError, match="must match"):
            LiveKitDetectorConfig(model_paths=[Path("a.onnx")], keywords=["a", "b"])

    def test_mismatched_sensitivities_raises(self):
        with pytest.raises(ValueError, match="sensitivities"):
            LiveKitDetectorConfig(
                model_paths=[Path("a.onnx"), Path("b.onnx")],
                keywords=["a", "b"],
                sensitivities=[0.5],
            )


class TestLiveKitDetector:
    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_creates_model(self, mock_cls):
        mock_cls.return_value = MagicMock()
        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        mock_cls.assert_called_once()
        assert det.frame_length == 1280
        assert det.sample_rate == 16000
        assert det.chunk_frames == 25

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel", None)
    def test_missing_sdk_does_not_raise(self):
        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("a.onnx")], keywords=["a"])
        )
        assert det.predict(np.zeros(32000, dtype=np.int16)) == (None, None)

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_predict_returns_match_above_threshold(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.95}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        hit, candidate = det.predict(np.zeros(32000, dtype=np.int16))
        assert hit is not None
        assert hit[0] == 0
        assert hit[1] == "hey"
        assert hit[2] == pytest.approx(0.95)
        assert candidate is None

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_predict_returns_highest_when_multiple_fire(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"alpha": 0.60, "beta": 0.95}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(
                model_paths=[Path("alpha.onnx"), Path("beta.onnx")],
                keywords=["alpha", "beta"],
            )
        )
        hit, candidate = det.predict(np.zeros(32000, dtype=np.int16))
        assert hit is not None
        assert hit[1] == "beta"
        assert candidate is None

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_predict_uses_model_stem_when_keyword_missing(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey_livekit": 0.95}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(
                model_paths=[Path("hey_livekit.onnx")],
                keywords=["custom_label"],
            )
        )
        hit, candidate = det.predict(np.zeros(32000, dtype=np.int16))
        assert hit is not None
        assert hit[1] == "custom_label"
        assert candidate is None


class TestLiveKitStream:
    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_stream_buffers_until_full_window(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.0}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        stream = LiveKitStream(detector=det)
        for _ in range(CHUNK_FRAMES - 1):
            assert stream.process_audio(b"\x00" * FRAME_BYTES) == (None, None)
        assert model.predict.call_count == 0
        assert stream.process_audio(b"\x00" * FRAME_BYTES) == (None, None)
        assert model.predict.call_count == 1

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_stream_returns_detection_tuple(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.95}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        stream = LiveKitStream(detector=det)
        hit, candidate = stream.process_audio(b"\x00" * FRAME_BYTES * CHUNK_FRAMES)
        assert hit is not None
        assert hit[0] == 0
        assert hit[1] == "hey"
        assert hit[2] == 0.95
        assert isinstance(hit[3], bytes) and len(hit[3]) > 0
        assert candidate is None

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_stream_handles_variable_chunk_sizes(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.0}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        stream = LiveKitStream(detector=det)
        stream.process_audio(b"\x00" * 333)
        stream.process_audio(b"\x00" * 777)
        stream.process_audio(b"\x00" * (FRAME_BYTES * CHUNK_FRAMES - 333 - 777))
        assert model.predict.call_count == 1

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_stream_clears_after_detection(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.95}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        stream = LiveKitStream(detector=det)
        stream.process_audio(b"\x00" * FRAME_BYTES * CHUNK_FRAMES)
        assert len(stream._frame_buffer) == 0
        assert len(stream._byte_buffer) == 0

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_reset_clears_buffers(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.0}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        stream = LiveKitStream(detector=det)
        stream.process_audio(b"\x00" * FRAME_BYTES)
        assert len(stream._frame_buffer) == 1
        stream.reset()
        assert len(stream._frame_buffer) == 0
        assert len(stream._byte_buffer) == 0

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel", None)
    def test_stream_no_sdk_returns_none(self):
        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        stream = LiveKitStream(detector=det)
        assert stream.process_audio(b"\x00" * FRAME_BYTES * CHUNK_FRAMES) == (None, None)

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_two_streams_share_one_detector_independently(self, mock_cls):
        # Verifies Oracle's BLOCKER #4 fix: shared detector, per-handler buffer state
        model = MagicMock()
        model.predict.return_value = {"hey": 0.0}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        stream_a = LiveKitStream(detector=det)
        stream_b = LiveKitStream(detector=det)
        stream_a.process_audio(b"\x00" * FRAME_BYTES)
        assert len(stream_a._frame_buffer) == 1
        assert len(stream_b._frame_buffer) == 0

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_detection_returns_raw_window_unmodified(self, mock_cls):
        """Feature: on detection, the returned PCM is the as-captured mic audio,
        byte-for-byte identical to the input chunk. Downstream consumers (sample
        store, fine-tune export, speaker verifier, mid-sentence detector) MUST
        see the true input distribution. amplitude normalization is intentionally absent
        entirely so this is now the trivial identity invariant — but the test
        stays as a regression guard against any future processing layer being
        inserted between capture and storage.
        """
        model = MagicMock()
        model.predict.return_value = {"hey": 0.95}
        mock_cls.return_value = model

        cfg = LiveKitDetectorConfig(
            model_paths=[Path("hey.onnx")],
            keywords=["hey"],
        )
        det = LiveKitDetector(cfg)
        stream = LiveKitStream(detector=det)

        rng = np.random.RandomState(42)
        quiet_f32 = rng.randn(FRAME_BYTES * CHUNK_FRAMES // 2).astype(np.float32) * 0.01
        quiet_int16 = (quiet_f32 * 32767).astype(np.int16)
        raw_bytes = quiet_int16.tobytes()
        in_rms = float(np.sqrt(np.mean((quiet_int16.astype(np.float32) / 32768.0) ** 2)))

        hit, candidate = stream.process_audio(raw_bytes)
        assert hit is not None
        _, _, _, returned_pcm = hit
        assert candidate is None

        assert returned_pcm == raw_bytes, "stored PCM must equal raw input bytes"

        returned_int16 = np.frombuffer(returned_pcm, dtype=np.int16)
        returned_rms = float(np.sqrt(np.mean((returned_int16.astype(np.float32) / 32768.0) ** 2)))
        assert returned_rms == pytest.approx(in_rms, rel=1e-6)

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_predict_receives_raw_window(self, mock_cls):
        """Feature: predict() receives the raw int16 window. the model is trained
        the model on raw audio (no full-window RMS norm), so inference MUST
        match: predict input RMS equals input RMS, byte-equivalent buffer.
        """
        model = MagicMock()
        model.predict.return_value = {"hey": 0.0}
        mock_cls.return_value = model

        cfg = LiveKitDetectorConfig(
            model_paths=[Path("hey.onnx")],
            keywords=["hey"],
        )
        det = LiveKitDetector(cfg)
        stream = LiveKitStream(detector=det)

        rng = np.random.RandomState(7)
        quiet = (rng.randn(FRAME_BYTES * CHUNK_FRAMES // 2).astype(np.float32) * 0.01)
        quiet_int16 = (quiet * 32767).astype(np.int16)
        in_rms = float(np.sqrt(np.mean(quiet * quiet)))

        stream.process_audio(quiet_int16.tobytes())

        assert model.predict.call_count == 1
        predict_input = model.predict.call_args[0][0]
        predict_rms = float(np.sqrt(np.mean((predict_input.astype(np.float32) / 32768.0) ** 2)))
        assert predict_rms == pytest.approx(in_rms, rel=1e-2), (
            f"predict() input RMS ({predict_rms:.5f}) must equal raw RMS "
            f"({in_rms:.5f}) — there is no pre-predict normalization. "
            f"Tolerance 1% covers int16-quantization roundtrip; any actual "
            f"normalization would shift RMS by >10x for this input."
        )


class TestSaveSampleThreshold:
    """Feature: a SECOND, lower threshold causes near-miss captures
    for fine-tune dataset mining WITHOUT firing Wyoming detection. predict()
    must return ``(hit, candidate)`` with hit-wins-over-candidate precedence,
    and LiveKitStream.process_audio() must NOT clear its buffer on candidate
    (so a real fire 200 ms later still triggers).
    """

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_predict_returns_candidate_when_score_between_save_and_sensitivity(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.30}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(
                model_paths=[Path("hey.onnx")],
                keywords=["hey"],
                sensitivities=[0.50],
                save_sample_threshold=0.25,
            )
        )
        hit, candidate = det.predict(np.zeros(32000, dtype=np.int16))
        assert hit is None
        assert candidate is not None
        assert candidate[1] == "hey"
        assert candidate[2] == pytest.approx(0.30)

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_predict_hit_wins_over_candidate_for_same_window(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"alpha": 0.30, "beta": 0.60}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(
                model_paths=[Path("alpha.onnx"), Path("beta.onnx")],
                keywords=["alpha", "beta"],
                sensitivities=[0.50, 0.50],
                save_sample_threshold=0.25,
            )
        )
        hit, candidate = det.predict(np.zeros(32000, dtype=np.int16))
        assert hit is not None
        assert hit[1] == "beta"
        assert candidate is None, "candidate path must be suppressed when any hit fires"

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_predict_returns_none_when_save_threshold_disabled(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.30}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(
                model_paths=[Path("hey.onnx")],
                keywords=["hey"],
                sensitivities=[0.50],
                save_sample_threshold=None,
            )
        )
        hit, candidate = det.predict(np.zeros(32000, dtype=np.int16))
        assert hit is None
        assert candidate is None, "candidate must be None when save_sample_threshold disabled"

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_predict_no_candidate_when_score_below_save_threshold(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.10}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(
                model_paths=[Path("hey.onnx")],
                keywords=["hey"],
                sensitivities=[0.50],
                save_sample_threshold=0.25,
            )
        )
        hit, candidate = det.predict(np.zeros(32000, dtype=np.int16))
        assert hit is None and candidate is None

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_stream_does_not_clear_buffer_on_candidate(self, mock_cls):
        """Critical safety invariant: candidate must NOT clear the frame buffer.
        If the wake word completes 200 ms after a near-miss, clearing here would
        silently drop the real fire. Verified by checking the buffer stays
        populated after a candidate, contrasted with the hit path which clears.
        """
        model = MagicMock()
        model.predict.return_value = {"hey": 0.30}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(
                model_paths=[Path("hey.onnx")],
                keywords=["hey"],
                sensitivities=[0.50],
                save_sample_threshold=0.25,
                predict_interval_seconds=0.0,
            )
        )
        stream = LiveKitStream(detector=det)
        hit, candidate = stream.process_audio(b"\x00\x01" * (FRAME_BYTES * CHUNK_FRAMES // 2))
        assert hit is None
        assert candidate is not None
        assert candidate[1] == "hey"
        assert candidate[2] == pytest.approx(0.30)
        assert len(stream._frame_buffer) == CHUNK_FRAMES, (
            "candidate must NOT clear frame_buffer — otherwise a real fire "
            "completing within the same 2 s window is silently lost"
        )

    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    def test_stream_clears_buffer_on_hit_even_with_save_threshold(self, mock_cls):
        model = MagicMock()
        model.predict.return_value = {"hey": 0.95}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(
                model_paths=[Path("hey.onnx")],
                keywords=["hey"],
                sensitivities=[0.50],
                save_sample_threshold=0.25,
                predict_interval_seconds=0.0,
            )
        )
        stream = LiveKitStream(detector=det)
        hit, candidate = stream.process_audio(b"\x00\x01" * (FRAME_BYTES * CHUNK_FRAMES // 2))
        assert hit is not None
        assert candidate is None
        assert len(stream._frame_buffer) == 0
        assert len(stream._byte_buffer) == 0


class TestPredictAsyncExecutor:
    """Feature: `predict_async()` MUST dispatch the synchronous ONNX
    `predict()` call to the loop's default executor via `run_in_executor`, so a
    slow predict cannot block other handlers' chunk reads on the asyncio loop.
    Root cause class: HA-side pipeline-restart storms produced many concurrent
    Wyoming sessions whose synchronous predicts serialized the event loop and
    caused snowball-drop on long-lived streaming sessions (a detection
    blackouts under concurrent streams).
    """

    @pytest.mark.asyncio
    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    async def test_predict_async_uses_executor(self, mock_cls):
        import asyncio as _asyncio

        model = MagicMock()
        model.predict.return_value = {"hey": 0.0}
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )
        loop = _asyncio.get_running_loop()
        original_run_in_executor = loop.run_in_executor
        calls = []

        def spy_run_in_executor(executor, func, *args):
            calls.append((executor, func.__name__ if hasattr(func, "__name__") else str(func)))
            return original_run_in_executor(executor, func, *args)

        loop.run_in_executor = spy_run_in_executor
        try:
            hit, candidate = await det.predict_async(np.zeros(32000, dtype=np.int16))
        finally:
            loop.run_in_executor = original_run_in_executor

        assert hit is None and candidate is None
        assert len(calls) >= 1, "predict_async MUST dispatch through run_in_executor"
        assert any(name == "predict" for _, name in calls), (
            "the dispatched callable MUST be self.predict"
        )

    @pytest.mark.asyncio
    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    async def test_event_loop_not_blocked_during_predict(self, mock_cls):
        import asyncio as _asyncio

        model = MagicMock()

        def slow_predict(_window):
            time.sleep(0.3)
            return {"hey": 0.0}

        model.predict.side_effect = slow_predict
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )

        ticks = [0]

        async def ticker():
            for _ in range(20):
                await _asyncio.sleep(0.05)
                ticks[0] += 1

        predict_task = _asyncio.create_task(det.predict_async(np.zeros(32000, dtype=np.int16)))
        ticker_task = _asyncio.create_task(ticker())
        await _asyncio.gather(predict_task, ticker_task)

        assert ticks[0] >= 4, (
            f"ticker should advance >=4 times during 300ms blocking predict; "
            f"got {ticks[0]} ticks (loop is blocked by sync predict)"
        )


class TestPredictAsyncScalesWithoutCap:
    """Feature: predict_async() does NOT impose an artificial concurrency
    cap. Natural backpressure comes from Python's default ThreadPoolExecutor
    (min(32, cpu+4) workers) and the ORT single-thread monkey-patch. This is the
    DESIGN CHOICE for multi-device deployments — capping concurrent predicts
    serializes long-lived streaming sessions and breaks scaling beyond ~3-4 devices.
    If a future maintainer adds a Semaphore here, they would re-introduce the
    detection blackouts at scale.
    """

    @pytest.mark.asyncio
    @patch("wyoming_livekit_speechbrain.livekit_detector.WakeWordModel")
    async def test_many_predicts_run_concurrently_without_artificial_cap(self, mock_cls):
        import asyncio as _asyncio
        import threading

        model = MagicMock()
        in_flight = [0]
        max_seen = [0]
        lock = threading.Lock()

        def slow_predict(_window):
            with lock:
                in_flight[0] += 1
                if in_flight[0] > max_seen[0]:
                    max_seen[0] = in_flight[0]
            time.sleep(0.15)
            with lock:
                in_flight[0] -= 1
            return {"hey": 0.0}

        model.predict.side_effect = slow_predict
        mock_cls.return_value = model

        det = LiveKitDetector(
            LiveKitDetectorConfig(model_paths=[Path("hey.onnx")], keywords=["hey"])
        )

        tasks = [
            _asyncio.create_task(det.predict_async(np.zeros(32000, dtype=np.int16)))
            for _ in range(8)
        ]
        await _asyncio.gather(*tasks)

        assert max_seen[0] >= 4, (
            f"Expected >=4 concurrent predicts (no artificial cap); got max in-flight "
            f"{max_seen[0]}. If a Semaphore was added, remove it — capping breaks "
            f"multi-device scaling. Backpressure is ThreadPoolExecutor + ORT 1-thread."
        )
