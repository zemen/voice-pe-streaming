"""Process-wide detection event bus.

The Wyoming handler publishes an event for every accepted wake-word detection
(after mid-sentence rejection and speaker verification). The web server exposes
these events over a WebSocket (``/api/events``) so external consumers - most
importantly the ``voice_pe_speaker`` Home Assistant integration - can surface
"who spoke last" as an HA sensor without long-lived tokens or REST polling.

Design notes:
- Fan-out via per-subscriber ``asyncio.Queue`` (bounded). Slow consumers drop
  the oldest event instead of blocking the audio pipeline.
- ``last_event`` is retained so a freshly-connected subscriber can restore
  state immediately (HA restart should not blank the sensor).
- Publishing is best-effort and never raises into the detection path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

_QUEUE_SIZE = 16


@dataclass
class DetectionEvent:
    """A single accepted wake-word detection."""

    keyword: str
    speaker: Optional[str] = None
    score: Optional[float] = None
    timestamp: Optional[int] = None
    """Wyoming audio-chunk timestamp (ms), if the pipeline provided one."""
    detected_at: float = field(default_factory=time.time)
    """Wall-clock unix time the event was published."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "detection",
            "keyword": self.keyword,
            "speaker": self.speaker,
            "score": self.score,
            "timestamp": self.timestamp,
            "detected_at": self.detected_at,
        }


class DetectionEventBus:
    """Bounded fan-out pub/sub for :class:`DetectionEvent`."""

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []
        self.last_event: Optional[DetectionEvent] = None

    def publish(self, event: DetectionEvent) -> None:
        """Publish an event to all subscribers. Never raises."""
        self.last_event = event
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()  # drop oldest
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning("Failed to publish detection event: %s", exc)

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_SIZE)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
