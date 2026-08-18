# pyright: reportMissingImports=false
"""Push client for Voice PE speaker detection events."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from typing import Any

from aiohttp import ClientError, WSMsgType

from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from .const import WS_PATH

_LOGGER = logging.getLogger(__name__)

INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0


@dataclass(frozen=True, slots=True)
class SpeakerDetection:
    """An accepted wake-word detection received from the event stream."""

    keyword: str
    speaker: str | None
    score: float | None
    detected_at: str | None


class VoicePESpeakerClient:
    """Maintain the Voice PE event WebSocket and notify listeners."""

    def __init__(self, hass: HomeAssistant, host: str, port: int) -> None:
        self._hass = hass
        self._url = f"ws://{host}:{port}{WS_PATH}"
        self._listeners: set[Callable[[], None]] = set()
        self._task: asyncio.Task[None] | None = None
        self.connected = False
        self.detection: SpeakerDetection | None = None

    def start(self) -> None:
        """Start the long-running WebSocket task."""
        if self._task is None or self._task.done():
            self._task = self._hass.async_create_task(
                self._run(), f"{self.__class__.__name__} event stream"
            )

    async def stop(self) -> None:
        """Stop the WebSocket task."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._set_connected(False)

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Register a listener and return its unsubscribe callback."""
        self._listeners.add(listener)

        def remove_listener() -> None:
            self._listeners.discard(listener)

        return remove_listener

    @staticmethod
    def next_backoff(current: float) -> float:
        """Return the next reconnect delay."""
        return min(current * 2, MAX_BACKOFF)

    def process_message(self, raw_message: str) -> bool:
        """Parse one text frame and publish it when it is a detection."""
        try:
            payload = json.loads(raw_message)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(payload, dict) or payload.get("type") != "detection":
            return False

        keyword = payload.get("keyword")
        if not isinstance(keyword, str):
            return False
        speaker = payload.get("speaker")
        if not isinstance(speaker, str):
            speaker = None
        score_value = payload.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None
        detected_at = self._format_detected_at(payload.get("detected_at"))
        self.detection = SpeakerDetection(keyword, speaker, score, detected_at)
        self._notify_listeners()
        return True

    @staticmethod
    def _format_detected_at(value: Any) -> str | None:
        if not isinstance(value, (int, float)):
            return None
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()

    async def _run(self) -> None:
        session = aiohttp_client.async_get_clientsession(self._hass)
        backoff = INITIAL_BACKOFF
        while True:
            try:
                async with session.ws_connect(self._url, heartbeat=30) as websocket:
                    self._set_connected(True)
                    backoff = INITIAL_BACKOFF
                    async for message in websocket:
                        if message.type == WSMsgType.TEXT:
                            self.process_message(message.data)
                        elif message.type in (
                            WSMsgType.CLOSE,
                            WSMsgType.CLOSED,
                            WSMsgType.CLOSING,
                            WSMsgType.ERROR,
                        ):
                            break
            except asyncio.CancelledError:
                raise
            except (ClientError, OSError, TimeoutError):
                _LOGGER.debug("Voice PE event stream disconnected", exc_info=True)
            except Exception:
                _LOGGER.exception("Unexpected Voice PE event stream error")
            finally:
                self._set_connected(False)

            await asyncio.sleep(backoff)
            backoff = self.next_backoff(backoff)

    def _set_connected(self, connected: bool) -> None:
        if self.connected == connected:
            return
        self.connected = connected
        self._notify_listeners()

    def _notify_listeners(self) -> None:
        for listener in tuple(self._listeners):
            listener()
