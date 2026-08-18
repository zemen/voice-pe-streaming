"""Detection event bus + /api/events WebSocket feature tests.

Feature requirement: every ACCEPTED wake-word detection (post mid-sentence
rejection, post speaker verification) must be published on the event bus with
keyword + speaker + score, and be consumable over the web server's
/api/events WebSocket so the voice_pe_speaker HA integration can expose
"who spoke last" as a sensor. Rejected detections must NOT be published.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.modules.setdefault("livekit", MagicMock())
sys.modules.setdefault("livekit.wakeword", MagicMock())
sys.modules.setdefault("speechbrain", MagicMock())
sys.modules.setdefault("speechbrain.inference", MagicMock())
sys.modules.setdefault("speechbrain.inference.speaker", MagicMock())
sys.modules.setdefault("torch", MagicMock())

from wyoming_livekit_speechbrain.events import DetectionEvent, DetectionEventBus  # noqa: E402


class TestDetectionEventBus:
    def test_publish_reaches_all_subscribers(self):
        bus = DetectionEventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        event = DetectionEvent(keyword="sputnik", speaker="alice", score=0.9)
        bus.publish(event)
        assert q1.get_nowait() is event
        assert q2.get_nowait() is event

    def test_last_event_retained(self):
        bus = DetectionEventBus()
        assert bus.last_event is None
        event = DetectionEvent(keyword="sputnik", speaker="bob")
        bus.publish(event)
        assert bus.last_event is event

    def test_unsubscribe_stops_delivery(self):
        bus = DetectionEventBus()
        q = bus.subscribe()
        bus.unsubscribe(q)
        bus.publish(DetectionEvent(keyword="sputnik"))
        assert q.empty()
        assert bus.subscriber_count == 0

    def test_full_queue_drops_oldest_not_newest(self):
        bus = DetectionEventBus()
        q = bus.subscribe()
        for i in range(20):
            bus.publish(DetectionEvent(keyword=f"kw{i}"))
        drained = []
        while not q.empty():
            drained.append(q.get_nowait())
        assert drained[-1].keyword == "kw19"
        assert len(drained) <= 16

    def test_publish_without_subscribers_is_noop(self):
        bus = DetectionEventBus()
        bus.publish(DetectionEvent(keyword="sputnik"))
        assert bus.last_event.keyword == "sputnik"

    def test_event_to_dict_shape(self):
        event = DetectionEvent(keyword="sputnik", speaker="alice", score=0.87, timestamp=123)
        d = event.to_dict()
        assert d["type"] == "detection"
        assert d["keyword"] == "sputnik"
        assert d["speaker"] == "alice"
        assert d["score"] == 0.87
        assert d["timestamp"] == 123
        assert isinstance(d["detected_at"], float)


def _make_handler(event_bus=None, speaker_verifier=None):
    from wyoming_livekit_speechbrain.handler import LiveKitEventHandler

    reader = MagicMock()
    writer = MagicMock()
    writer.get_extra_info.return_value = ("127.0.0.1", 55555)
    detector = MagicMock()
    handler = LiveKitEventHandler.__new__(LiveKitEventHandler)
    handler.wake_word_names = ["sputnik"]
    handler.model_paths = [Path("sputnik.onnx")]
    handler.is_detecting = True
    handler.chunks_received = 0
    handler.audio_rate = None
    handler.audio_width = None
    handler.audio_channels = None
    handler._detect_context = None
    handler.livekit = detector
    handler._stream = MagicMock()
    handler.speaker_verifier = speaker_verifier
    handler._ring_buffer = None
    handler.mid_sentence_detector = None
    handler._last_ww_name = None
    handler._last_ww_speaker = None
    handler._last_audio_monotonic = None
    handler._sample_store = None
    handler._device_registry = None
    handler._device_id = None
    handler._remote_addr = "test"
    handler._event_log = []
    handler._event_bus = event_bus
    handler.write_event = AsyncMock()
    return handler


class TestHandlerPublishesToEventBus:
    @pytest.mark.asyncio
    async def test_accepted_detection_published(self):
        bus = DetectionEventBus()
        q = bus.subscribe()
        handler = _make_handler(event_bus=bus)

        await handler._process_detection("sputnik", timestamp=100, fire_score=0.91)

        event = q.get_nowait()
        assert event.keyword == "sputnik"
        assert event.speaker is None
        assert event.score == 0.91
        assert event.timestamp == 100
        assert handler.write_event.call_count == 1

    @pytest.mark.asyncio
    async def test_verified_speaker_in_published_event(self):
        bus = DetectionEventBus()
        q = bus.subscribe()
        verifier = MagicMock()
        verifier.verify_with_scores.return_value = {"alice": 0.8, "bob": 0.1}
        verifier.config.threshold = 0.25
        handler = _make_handler(event_bus=bus, speaker_verifier=verifier)

        await handler._process_detection(
            "sputnik", timestamp=100, fire_score=0.91, fire_window_pcm=b"\x00" * 32000
        )

        event = q.get_nowait()
        assert event.speaker == "alice"

    @pytest.mark.asyncio
    async def test_rejected_speaker_not_published(self):
        bus = DetectionEventBus()
        q = bus.subscribe()
        verifier = MagicMock()
        verifier.verify_with_scores.return_value = {"alice": 0.05}
        verifier.config.threshold = 0.25
        handler = _make_handler(event_bus=bus, speaker_verifier=verifier)

        await handler._process_detection(
            "sputnik", timestamp=100, fire_score=0.91, fire_window_pcm=b"\x00" * 32000
        )

        assert q.empty()
        assert handler.write_event.call_count == 0

    @pytest.mark.asyncio
    async def test_no_bus_still_emits_detection(self):
        handler = _make_handler(event_bus=None)
        await handler._process_detection("sputnik", timestamp=100, fire_score=0.91)
        assert handler.write_event.call_count == 1


class TestEventsWebSocket:
    def _make_app(self, bus):
        from aiohttp import web

        from wyoming_livekit_speechbrain.web_server import handle_events_ws

        app = web.Application()
        ctx = MagicMock()
        ctx.event_bus = bus
        app["ctx"] = ctx
        app.router.add_get("/api/events", handle_events_ws)
        return app

    @pytest.mark.asyncio
    async def test_receives_published_events(self):
        from aiohttp.test_utils import TestClient, TestServer

        bus = DetectionEventBus()
        app = self._make_app(bus)
        async with TestClient(TestServer(app)) as client:
            async with client.ws_connect("/api/events") as ws:
                bus.publish(DetectionEvent(keyword="sputnik", speaker="alice", score=0.9))
                msg = await asyncio.wait_for(ws.receive_json(), timeout=5)
                assert msg["type"] == "detection"
                assert msg["keyword"] == "sputnik"
                assert msg["speaker"] == "alice"

    @pytest.mark.asyncio
    async def test_last_event_replayed_on_connect(self):
        from aiohttp.test_utils import TestClient, TestServer

        bus = DetectionEventBus()
        bus.publish(DetectionEvent(keyword="sputnik", speaker="bob", score=0.7))
        app = self._make_app(bus)
        async with TestClient(TestServer(app)) as client:
            async with client.ws_connect("/api/events") as ws:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=5)
                assert msg["speaker"] == "bob"

    @pytest.mark.asyncio
    async def test_ping_gets_pong(self):
        from aiohttp.test_utils import TestClient, TestServer

        bus = DetectionEventBus()
        app = self._make_app(bus)
        async with TestClient(TestServer(app)) as client:
            async with client.ws_connect("/api/events") as ws:
                await ws.send_str(json.dumps({"type": "ping"}))
                msg = await asyncio.wait_for(ws.receive_json(), timeout=5)
                assert msg["type"] == "pong"

    @pytest.mark.asyncio
    async def test_subscriber_cleaned_up_on_disconnect(self):
        from aiohttp.test_utils import TestClient, TestServer

        bus = DetectionEventBus()
        app = self._make_app(bus)
        async with TestClient(TestServer(app)) as client:
            async with client.ws_connect("/api/events") as ws:
                await ws.close()
            await asyncio.sleep(0.1)
        assert bus.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_no_bus_closes_connection(self):
        from aiohttp import WSMsgType
        from aiohttp.test_utils import TestClient, TestServer

        app = self._make_app(None)
        async with TestClient(TestServer(app)) as client:
            async with client.ws_connect("/api/events") as ws:
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
                assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED)
