"""Tests for web_server.py audio-path raw-pcm invariants.

Amplitude normalization is intentionally absent. The web UI debug
endpoints (`/api/samples/{id}/retest`, `/api/score-stream`) MUST feed RAW
audio to both `model.predict()` and `verifier.verify_with_scores()` so
their behavior matches the live Wyoming pipeline (which also receives
raw audio — the model was retrained on raw, no full-window RMS norm).
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

sys.modules.setdefault("livekit", MagicMock())
sys.modules.setdefault("livekit.wakeword", MagicMock())
sys.modules.setdefault("speechbrain", MagicMock())
sys.modules.setdefault("speechbrain.inference", MagicMock())
sys.modules.setdefault("speechbrain.inference.speaker", MagicMock())
sys.modules.setdefault("torch", MagicMock())

from wyoming_livekit_speechbrain.web_server import (  # noqa: E402
    WebServerContext,
    handle_sample_retest,
    handle_score_stream_ws,
)


def _quiet_int16_pcm(num_samples: int = 32000, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    f32 = (rng.randn(num_samples) * 0.01).astype(np.float32)
    int16 = (f32 * 32767).astype(np.int16)
    return int16.tobytes()


def _rms_of_pcm(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


def _build_ctx():
    detector = MagicMock()
    detector.config.keywords = ["hey_livekit"]
    detector.config.sensitivities = [0.5]
    detector._model.predict.return_value = {"hey_livekit": 0.7}

    verifier = MagicMock()
    verifier.verify_with_scores.return_value = {"alice": 0.42}

    sample_store = MagicMock()
    ctx = WebServerContext(
        detector=detector,
        verifier=verifier,
        profiles_dir=Path("/tmp/_test_profiles_does_not_exist_intentionally"),
        speaker_model_source="speechbrain/spkrec-ecapa-voxceleb",
        speaker_savedir=None,
        speaker_threshold=0.25,
        sample_store=sample_store,
    )
    return ctx, detector, verifier, sample_store


def _make_request(ctx, sample_id: str = "fake-sample-id") -> MagicMock:
    req = MagicMock()
    req.app = {"ctx": ctx}
    req.match_info = {"id": sample_id}
    return req


class TestSampleRetestRawAudioInvariant:
    """`handle_sample_retest` MUST feed raw stored PCM byte-equivalent to both
    predict and verifier. amplitude normalization is intentionally absent, so any
    tampering between sample_store and predict would silently diverge retest
    scores from live-pipeline scores.
    """

    @pytest.mark.asyncio
    async def test_predict_receives_raw_audio(self):
        raw_pcm = _quiet_int16_pcm()
        raw_rms = _rms_of_pcm(raw_pcm)

        ctx, detector, verifier, sample_store = _build_ctx()
        sample_store.get_pcm.return_value = raw_pcm

        await handle_sample_retest(_make_request(ctx))

        assert detector._model.predict.call_count == 1
        predict_arg = detector._model.predict.call_args[0][0]
        predict_rms = float(
            np.sqrt(np.mean((predict_arg.astype(np.float32) / 32768.0) ** 2))
        )
        assert predict_rms == pytest.approx(raw_rms, rel=1e-3), (
            f"predict() must see raw audio (RMS {raw_rms:.5f}); got {predict_rms:.5f}"
        )

    @pytest.mark.asyncio
    async def test_verify_receives_raw_audio(self):
        raw_pcm = _quiet_int16_pcm()
        ctx, detector, verifier, sample_store = _build_ctx()
        sample_store.get_pcm.return_value = raw_pcm

        await handle_sample_retest(_make_request(ctx))

        assert verifier.verify_with_scores.call_count == 1
        verify_arg = verifier.verify_with_scores.call_args[0][0]
        assert verify_arg == raw_pcm, "verifier MUST receive raw stored PCM bytes unchanged"


class TestScoreStreamRawAudioInvariant:
    """`handle_score_stream_ws` MUST feed raw audio to both predict and verifier.
    End-to-end aiohttp WebSocket test: stream a chunk, capture predict + verify args.
    """

    @pytest.mark.asyncio
    async def test_score_stream_predict_and_verifier_both_receive_raw(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from wyoming_livekit_speechbrain.livekit_detector import FRAME_BYTES
        from wyoming_livekit_speechbrain.web_server import handle_score_stream_ws

        ctx, detector, verifier, sample_store = _build_ctx()
        detector.chunk_frames = 25
        detector.config.predict_interval_seconds = 0.0

        captured_predict = []
        captured_verify = []

        def capture_predict(audio):
            captured_predict.append(np.array(audio, copy=True))
            return {"hey_livekit": 0.1}

        def capture_verify(pcm):
            captured_verify.append(bytes(pcm))
            return {"alice": 0.42}

        detector._model.predict.side_effect = capture_predict
        verifier.verify_with_scores.side_effect = capture_verify

        app = web.Application()
        app["ctx"] = ctx
        app.router.add_get("/api/score-stream", handle_score_stream_ws)

        chunk_samples = 25 * (FRAME_BYTES // 2)
        rng = np.random.RandomState(123)
        target_raw_rms = 0.06
        random_walk = rng.randn(chunk_samples).astype(np.float32)
        random_walk *= target_raw_rms / float(np.sqrt(np.mean(random_walk * random_walk)))
        audio_int16 = (random_walk * 32767).astype(np.int16)
        raw_bytes_full = audio_int16.tobytes()
        raw_rms = float(np.sqrt(np.mean((audio_int16.astype(np.float32) / 32768.0) ** 2)))
        assert raw_rms > 0.05, (
            f"test setup: raw RMS ({raw_rms:.5f}) must be > VAD silence "
            f"threshold (0.05) so the verifier branch executes"
        )

        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/score-stream")
            await ws.receive_json()
            chunk_bytes = FRAME_BYTES
            for i in range(0, len(raw_bytes_full), chunk_bytes):
                await ws.send_bytes(raw_bytes_full[i : i + chunk_bytes])
            for _ in range(20):
                msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                if msg.type.name == "TEXT" and msg.json().get("event") == "ww_score":
                    break
            await ws.close()

        assert len(captured_predict) == 1
        predict_arr = captured_predict[0]
        predict_rms = float(np.sqrt(np.mean((predict_arr.astype(np.float32) / 32768.0) ** 2)))
        assert predict_rms == pytest.approx(raw_rms, rel=1e-3), (
            f"predict() must see raw audio (RMS {raw_rms:.5f}); got {predict_rms:.5f}"
        )

        assert len(captured_verify) == 1
        assert captured_verify[0] == raw_bytes_full, "verifier MUST receive raw window bytes"
