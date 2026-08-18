"""Tracks active Wyoming connections so the web UI can list/select Voice PE devices.

Each Wyoming connection corresponds to a HA voice-pipeline session, typically
sourced from one Voice PE device. We expose a registry of currently-active
handlers so the web UI can:

- List active sessions ("Voice PE living room", "Voice PE kitchen", ...)
- Trigger manual audio capture from a specific session ("record 5s of mic audio")

Handlers register themselves on construction and unregister on disconnect.
The registry generates stable per-session IDs for the lifetime of the
connection.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class _ManualCapture:
    """Per-handler state for an in-progress manual audio capture."""

    duration_bytes: int
    started_at: float = field(default_factory=time.monotonic)
    _buffer: bytearray = field(default_factory=bytearray)
    _done: asyncio.Event = field(default_factory=asyncio.Event)

    def feed(self, audio: bytes) -> None:
        """Add incoming audio bytes to the capture buffer."""
        if self._done.is_set():
            return
        self._buffer.extend(audio)
        if len(self._buffer) >= self.duration_bytes:
            self._done.set()

    async def wait(self, timeout_seconds: float = 30.0) -> bytes:
        """Block until enough audio has been collected, or the timeout elapses."""
        try:
            await asyncio.wait_for(self._done.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            if not self._buffer:
                raise TimeoutError(
                    "No audio received within timeout; is the device streaming?"
                )
        return bytes(self._buffer[: self.duration_bytes] or self._buffer)


@dataclass
class DeviceInfo:
    """Public snapshot of an active Wyoming connection."""

    id: str
    remote_addr: str
    connected_at: float
    last_audio_at: Optional[float] = None
    chunks_received: int = 0
    audio_rate: Optional[int] = None
    audio_width: Optional[int] = None
    audio_channels: Optional[int] = None
    is_detecting: bool = False
    is_capturing: bool = False
    is_debug_claimed: bool = False
    detect_context: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "remote_addr": self.remote_addr,
            "connected_at": self.connected_at,
            "last_audio_at": self.last_audio_at,
            "chunks_received": self.chunks_received,
            "audio_rate": self.audio_rate,
            "audio_width": self.audio_width,
            "audio_channels": self.audio_channels,
            "is_detecting": self.is_detecting,
            "is_capturing": self.is_capturing,
            "is_debug_claimed": self.is_debug_claimed,
            "detect_context": self.detect_context,
        }


class DeviceRegistry:
    """Process-wide registry of active LiveKitEventHandler instances."""

    def __init__(self) -> None:
        self._handlers: Dict[str, "object"] = {}

    def register(self, handler: "object") -> str:
        """Register a handler. Returns the generated device id."""
        device_id = uuid.uuid4().hex[:12]
        self._handlers[device_id] = handler
        _LOGGER.debug("Device registered: %s (total=%d)", device_id, len(self._handlers))
        return device_id

    def unregister(self, device_id: str) -> None:
        if device_id in self._handlers:
            del self._handlers[device_id]
            _LOGGER.debug(
                "Device unregistered: %s (total=%d)", device_id, len(self._handlers)
            )

    def get(self, device_id: str) -> Optional["object"]:
        return self._handlers.get(device_id)

    def list_info(self) -> List[DeviceInfo]:
        """Snapshot every active handler's public state."""
        out: List[DeviceInfo] = []
        for device_id, handler in self._handlers.items():
            info = getattr(handler, "_make_device_info", None)
            if info is None:
                continue
            try:
                out.append(info())
            except Exception as exc:
                _LOGGER.warning("Failed to snapshot device %s: %s", device_id, exc)
        out.sort(key=lambda d: d.connected_at, reverse=True)
        return out
