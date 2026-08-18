"""Pure-Python tests for the Voice PE Speaker push client."""

from datetime import datetime, timezone
import json
import sys
from types import ModuleType
from typing import Generic, TypeVar
from unittest.mock import MagicMock

homeassistant = ModuleType("homeassistant")
homeassistant.__path__ = []
homeassistant_core = ModuleType("homeassistant.core")
homeassistant_helpers = ModuleType("homeassistant.helpers")
homeassistant_helpers.__path__ = []
homeassistant_aiohttp_client = ModuleType("homeassistant.helpers.aiohttp_client")
homeassistant_config_entries = ModuleType("homeassistant.config_entries")
homeassistant_const = ModuleType("homeassistant.const")

RuntimeDataT = TypeVar("RuntimeDataT")


class MockConfigEntry(Generic[RuntimeDataT]):
    pass

setattr(homeassistant_core, "HomeAssistant", MagicMock)
setattr(homeassistant_aiohttp_client, "async_get_clientsession", MagicMock())
setattr(homeassistant_helpers, "aiohttp_client", homeassistant_aiohttp_client)
setattr(homeassistant_config_entries, "ConfigEntry", MockConfigEntry)
setattr(homeassistant_const, "CONF_HOST", "host")
setattr(homeassistant_const, "CONF_PORT", "port")
setattr(homeassistant_const, "Platform", MagicMock(SENSOR="sensor"))

sys.modules.setdefault("homeassistant", homeassistant)
sys.modules.setdefault("homeassistant.core", homeassistant_core)
sys.modules.setdefault("homeassistant.helpers", homeassistant_helpers)
sys.modules.setdefault("homeassistant.helpers.aiohttp_client", homeassistant_aiohttp_client)
sys.modules.setdefault("homeassistant.config_entries", homeassistant_config_entries)
sys.modules.setdefault("homeassistant.const", homeassistant_const)

from custom_components.voice_pe_speaker.coordinator import (  # noqa: E402
    MAX_BACKOFF,
    VoicePESpeakerClient,
)


def _make_client() -> VoicePESpeakerClient:
    return VoicePESpeakerClient(MagicMock(), "example.local", 10401)


def test_detection_message_updates_state() -> None:
    client = _make_client()
    detected_at = 1_725_000_000.25

    processed = client.process_message(
        json.dumps(
            {
                "type": "detection",
                "keyword": "hey_livekit",
                "speaker": "alice",
                "score": 0.82,
                "timestamp": 1234,
                "detected_at": detected_at,
            }
        )
    )

    assert processed is True
    assert client.detection is not None
    assert client.detection.keyword == "hey_livekit"
    assert client.detection.speaker == "alice"
    assert client.detection.score == 0.82
    assert client.detection.detected_at == datetime.fromtimestamp(
        detected_at, timezone.utc
    ).isoformat()


def test_malformed_json_is_ignored() -> None:
    client = _make_client()

    assert client.process_message("{not-json") is False
    assert client.detection is None


def test_pong_message_is_ignored() -> None:
    client = _make_client()

    assert client.process_message('{"type":"pong"}') is False
    assert client.detection is None


def test_backoff_doubles_up_to_cap() -> None:
    delays = [1.0]
    for _ in range(8):
        delays.append(VoicePESpeakerClient.next_backoff(delays[-1]))

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]
    assert VoicePESpeakerClient.next_backoff(MAX_BACKOFF) == MAX_BACKOFF


def test_detection_fires_listener_callbacks() -> None:
    client = _make_client()
    first = MagicMock()
    second = MagicMock()
    remove_first = client.add_listener(first)
    client.add_listener(second)

    client.process_message(
        json.dumps(
            {
                "type": "detection",
                "keyword": "hey_livekit",
                "speaker": None,
                "score": None,
                "detected_at": 1_725_000_000,
            }
        )
    )

    first.assert_called_once_with()
    second.assert_called_once_with()
    remove_first()
    client.process_message(
        json.dumps(
            {
                "type": "detection",
                "keyword": "hey_livekit",
                "speaker": "bob",
                "score": 0.7,
                "detected_at": 1_725_000_001,
            }
        )
    )
    first.assert_called_once_with()
    assert second.call_count == 2
