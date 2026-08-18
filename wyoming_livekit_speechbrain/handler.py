"""Handler for Wyoming LiveKit + SpeechBrain wake word detection."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler
from wyoming.wake import Detect, Detection, NotDetected

from .buffer import MidSentenceDetector, RingBuffer, RingBufferConfig
from .device_registry import DeviceInfo, DeviceRegistry, _ManualCapture
from .events import DetectionEvent, DetectionEventBus
from .livekit_detector import LiveKitDetector, LiveKitStream
from .sample_store import SampleMeta, SampleStore
from .speechbrain_verifier import SpeechBrainVerifier

_LOGGER = logging.getLogger(__name__)


class LiveKitEventHandler(AsyncEventHandler):
    """Wyoming event handler for LiveKit wake word detection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wake_word_names: List[str],
        model_paths: List[Path],
        livekit: LiveKitDetector,
        speaker_verifier: Optional[SpeechBrainVerifier] = None,
        enable_mid_sentence_detection: bool = True,
        device_registry: Optional[DeviceRegistry] = None,
        sample_store: Optional[SampleStore] = None,
        event_bus: Optional[DetectionEventBus] = None,
    ) -> None:
        super().__init__(reader, writer)
        self._event_bus = event_bus
        self.wake_word_names = wake_word_names
        self.model_paths = model_paths

        self.is_detecting: bool = False
        self.chunks_received: int = 0
        self.audio_rate: Optional[int] = None
        self.audio_width: Optional[int] = None
        self.audio_channels: Optional[int] = None

        self._detect_context: Optional[dict] = None

        self.livekit = livekit
        self._stream = LiveKitStream(detector=livekit)

        self.speaker_verifier = speaker_verifier
        self._ring_buffer: Optional[RingBuffer] = RingBuffer(RingBufferConfig())
        self.mid_sentence_detector: Optional[MidSentenceDetector] = None
        if self.speaker_verifier is not None and enable_mid_sentence_detection:
            self.mid_sentence_detector = MidSentenceDetector(
                self._ring_buffer, self.speaker_verifier
            )

        self._last_ww_name: Optional[str] = None
        self._last_ww_speaker: Optional[str] = None
        self._last_audio_monotonic: Optional[float] = None

        self._sample_store = sample_store
        self._device_registry = device_registry
        self._device_id: Optional[str] = None
        self._connected_at = time.monotonic()
        self._remote_addr: str = "unknown"
        try:
            peer = writer.get_extra_info("peername")
            if peer:
                self._remote_addr = f"{peer[0]}:{peer[1]}"
        except Exception:
            pass

        self._manual_capture: Optional[_ManualCapture] = None

        self._debug_claimed: bool = False
        self._debug_subscribers: List[asyncio.Queue] = []

        self._event_log: List[tuple[float, str]] = []
        self._audio_gap_warned: bool = False
        self._last_snapshot_log_at: float = 0.0
        self._watchdog_task: Optional[asyncio.Task] = None

        if self._device_registry is not None:
            self._device_id = self._device_registry.register(self)
        self._log_event(f"connected from {self._remote_addr}")
        self._watchdog_task = asyncio.create_task(self._state_watchdog())

    def _short_id(self) -> str:
        if self._device_id:
            return self._device_id[:8]
        return self._remote_addr

    def _log_event(self, event: str) -> None:
        now = time.monotonic()
        self._event_log.append((now, event))
        if len(self._event_log) > 100:
            self._event_log = self._event_log[-100:]
        _LOGGER.info("[%s] %s", self._short_id(), event)

    async def _state_watchdog(self) -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                now = time.monotonic()
                if self.is_detecting and self._last_audio_monotonic is not None:
                    gap = now - self._last_audio_monotonic
                    if gap > 2.0 and not self._audio_gap_warned:
                        self._audio_gap_warned = True
                        _LOGGER.warning(
                            "[%s] AUDIO STALLED: is_detecting=True but no chunks for %.1fs "
                            "(chunks_received=%d, last_ww=%s). Voice PE may be stuck in non-streaming state.",
                            self._short_id(),
                            gap,
                            self.chunks_received,
                            self._last_ww_name,
                        )
                    elif gap < 1.0 and self._audio_gap_warned:
                        self._audio_gap_warned = False
                        self._log_event(f"audio resumed after stall (chunks_received={self.chunks_received})")
                if self.is_detecting and (now - self._last_snapshot_log_at) >= 30.0:
                    self._last_snapshot_log_at = now
                    age = (now - self._last_audio_monotonic) if self._last_audio_monotonic else None
                    age_str = f"{age:.1f}s" if age is not None else "n/a"
                    _LOGGER.info(
                        "[%s] snapshot: detecting=%s chunks=%d audio_age=%s last_ww=%s last_speaker=%s",
                        self._short_id(),
                        self.is_detecting,
                        self.chunks_received,
                        age_str,
                        self._last_ww_name,
                        self._last_ww_speaker,
                    )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            _LOGGER.error("[%s] watchdog crashed: %s", self._short_id(), exc, exc_info=True)

    def _make_device_info(self) -> DeviceInfo:
        """Build a public snapshot of this handler for the device registry."""
        return DeviceInfo(
            id=self._device_id or "",
            remote_addr=self._remote_addr,
            connected_at=self._connected_at,
            last_audio_at=self._last_audio_monotonic,
            chunks_received=self.chunks_received,
            audio_rate=self.audio_rate,
            audio_width=self.audio_width,
            audio_channels=self.audio_channels,
            is_detecting=self.is_detecting,
            is_capturing=self._manual_capture is not None,
            is_debug_claimed=self._debug_claimed,
            detect_context=self._detect_context,
        )

    def claim_for_debug(self) -> None:
        """Mark this device as claimed by the web debug UI.

        While claimed: incoming audio is forwarded to debug-subscriber queues
        AND the Wyoming WW pipeline is bypassed (no `Detection` events emitted
        to HA, so STT/TTS do not run on this device's audio while you inspect).
        """
        self._debug_claimed = True

    def release_from_debug(self) -> None:
        """Stop debug claim. Wyoming pipeline resumes normal WW detection + Detection emission."""
        self._debug_claimed = False
        for q in list(self._debug_subscribers):
            try:
                q.put_nowait(None)
            except Exception:
                pass

    def subscribe_debug_audio(self) -> asyncio.Queue:
        """Subscribe to this device's incoming audio chunks. Returns a Queue.

        Each chunk pushed as bytes. A None sentinel is pushed when the device
        disconnects or releases the debug claim so consumers can clean up.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._debug_subscribers.append(q)
        return q

    def unsubscribe_debug_audio(self, q: asyncio.Queue) -> None:
        if q in self._debug_subscribers:
            self._debug_subscribers.remove(q)

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming event."""
        if not AudioChunk.is_type(event.type):
            _LOGGER.debug("Received event: %s", event.type)

        if Describe.is_type(event.type):
            await self._handle_describe()
            return True

        if Detect.is_type(event.type):
            await self._handle_detect(event)
            return True

        if AudioStart.is_type(event.type):
            await self._handle_audio_start(event)
            return True

        if AudioChunk.is_type(event.type):
            await self._handle_audio_chunk(event)
            return True

        if AudioStop.is_type(event.type):
            await self._handle_audio_stop()
            return True

        _LOGGER.warning("Unhandled event type: %s", event.type)
        return True

    async def disconnect(self) -> None:
        duration = time.monotonic() - self._connected_at
        self._log_event(
            f"disconnected after {duration:.1f}s (chunks_received={self.chunks_received}, "
            f"last_ww={self._last_ww_name}, last_speaker={self._last_ww_speaker})"
        )
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
        self.release_from_debug()
        if self._device_registry is not None and self._device_id is not None:
            self._device_registry.unregister(self._device_id)
            self._device_id = None

    async def capture_for_duration(self, duration_seconds: float) -> bytes:
        """Capture the next ``duration_seconds`` of incoming audio. Raises if a capture is already in progress.

        Returns raw 16 kHz mono int16 PCM bytes. Used by the web UI's "Record
        from Voice PE" feature to grab arbitrary audio from this specific
        Wyoming session for fine-tuning data collection or browser-replay
        verification.
        """
        if self._manual_capture is not None:
            raise RuntimeError("Manual capture already in progress for this device")
        if not self.is_detecting:
            raise RuntimeError(
                "Device is not actively streaming audio (no Detect session). "
                "Wake the device first (say the wake word), then retry within a few seconds."
            )

        duration_bytes = int(duration_seconds * 16000 * 2)
        capture = _ManualCapture(duration_bytes=duration_bytes)
        self._manual_capture = capture
        try:
            return await capture.wait(timeout_seconds=max(duration_seconds + 5.0, 10.0))
        finally:
            self._manual_capture = None

    async def _handle_describe(self) -> None:
        """Handle Describe event by sending service info."""
        wake_models = [
            WakeModel(
                name=name,
                attribution=Attribution(
                    name="LiveKit",
                    url="https://github.com/livekit/livekit-wakeword",
                ),
                installed=True,
                description=f"livekit-wakeword: {name}",
                version=None,
                languages=["en"],
                phrase=name,
            )
            for name in self.wake_word_names
        ]

        wake_program = WakeProgram(
            name="livekit-wakeword",
            attribution=Attribution(
                name="LiveKit",
                url="https://github.com/livekit/livekit-wakeword",
            ),
            installed=True,
            description="LiveKit wake word detection",
            version=None,
            models=wake_models,
        )

        info = Info(wake=[wake_program])
        await self.write_event(info.event())

    async def _handle_detect(self, event: Event) -> None:
        detect = Detect.from_event(event)
        self._detect_context = detect.context

        was_detecting = self.is_detecting
        self.is_detecting = True
        self.chunks_received = 0
        self.audio_rate = None
        self.audio_width = None
        self.audio_channels = None
        self._audio_gap_warned = False
        self._last_snapshot_log_at = time.monotonic()
        self._reset_detection_state()
        self._log_event(f"Detect mode entered (was_detecting={was_detecting})")
        if was_detecting:
            _LOGGER.warning(
                "[%s] Detect received while ALREADY in detect mode -- HA pipeline may have restarted "
                "without sending AudioStop. Voice PE state machine may be out of sync.",
                self._short_id(),
            )

    async def _handle_audio_start(self, event: Event) -> None:
        if not self.is_detecting:
            _LOGGER.warning("[%s] Received AudioStart without Detect", self._short_id())
            return

        audio_start = AudioStart.from_event(event)
        self.audio_rate = audio_start.rate
        self.audio_width = audio_start.width
        self.audio_channels = audio_start.channels
        self._log_event(
            f"AudioStart rate={audio_start.rate} width={audio_start.width} "
            f"channels={audio_start.channels}"
        )
        if (audio_start.rate, audio_start.width, audio_start.channels) != (16000, 2, 1):
            _LOGGER.error(
                "AUDIO FORMAT MISMATCH: got %d/%d/%d, model needs 16000/2/1. "
                "Detection scores will be garbage. Source firmware or HA pipeline is misconfigured.",
                audio_start.rate,
                audio_start.width,
                audio_start.channels,
            )

    async def _handle_audio_chunk(self, event: Event) -> None:
        """Handle AudioChunk event."""
        if not self.is_detecting:
            _LOGGER.warning("Received AudioChunk without Detect")
            return

        chunk = AudioChunk.from_event(event)
        self.chunks_received += 1
        self._last_audio_monotonic = time.monotonic()

        for q in self._debug_subscribers:
            try:
                q.put_nowait(chunk.audio)
            except asyncio.QueueFull:
                pass

        if self._debug_claimed:
            return

        if self._ring_buffer is not None:
            self._ring_buffer.write(chunk.audio)

        if self._manual_capture is not None:
            self._manual_capture.feed(chunk.audio)

        hit, candidate = await self._stream.process_audio_async(chunk.audio)
        if hit is not None:
            _keyword_index, keyword, score, window_pcm = hit
            self._last_ww_name = keyword
            _LOGGER.info("LiveKit detected: %s (score=%.4f)", keyword, score)
            await self._process_detection(
                keyword,
                chunk.timestamp,
                fire_score=score,
                fire_window_pcm=window_pcm,
            )
        elif candidate is not None and self._sample_store is not None:
            _keyword_index, keyword, score, window_pcm = candidate
            self._persist_sample(
                keyword=keyword,
                score=score,
                speaker=None,
                accepted=False,
                pcm=window_pcm,
                source="auto-below-threshold",
            )

    async def _handle_audio_stop(self) -> None:
        if not self.is_detecting:
            _LOGGER.warning("[%s] Received AudioStop without Detect", self._short_id())
            return
        self._log_event(f"AudioStop received (chunks_received={self.chunks_received})")
        await self._emit_not_detected()

    async def _emit_not_detected(self) -> None:
        """Emit NotDetected event when audio ends without detection."""
        not_detected = NotDetected(context=self._detect_context)
        await self.write_event(not_detected.event())

        self.is_detecting = False
        self._detect_context = None
        self._reset_detection_state()

    async def _process_detection(
        self,
        keyword: str,
        timestamp: Optional[int] = None,
        fire_score: Optional[float] = None,
        fire_window_pcm: Optional[bytes] = None,
    ) -> None:
        """Verify and emit a pending wake word detection, persisting a sample if a store is configured."""
        ww_audio: Optional[bytes] = fire_window_pcm

        if self.mid_sentence_detector is not None and ww_audio is not None:
            if self.mid_sentence_detector.should_reject_detection(ww_audio):
                _LOGGER.info("Mid-sentence detection rejected: %s", keyword)
                self._reset_detection_state()
                return

        speaker_name: Optional[str] = None
        if self.speaker_verifier is not None:
            if ww_audio is None:
                _LOGGER.warning(
                    "SpeechBrain enabled but no audio available for verification - rejecting detection"
                )
                self._reset_detection_state()
                return

            scored = self.speaker_verifier.verify_with_scores(ww_audio)
            best_name = max(scored, key=scored.get) if scored else None
            best_score = scored[best_name] if best_name is not None else None
            threshold = self.speaker_verifier.config.threshold
            scores_repr = {k: round(v, 4) for k, v in scored.items()}
            if best_name is None or best_score < threshold:
                _LOGGER.warning(
                    "Speaker verification failed: scores=%s best=%s threshold=%.3f - rejecting",
                    scores_repr,
                    f"{best_name}@{best_score:.4f}" if best_name else "none",
                    threshold,
                )
                if self._sample_store is not None and fire_window_pcm is not None:
                    self._persist_sample(
                        keyword=keyword,
                        score=fire_score,
                        speaker=None,
                        accepted=False,
                        pcm=fire_window_pcm,
                        source="auto-rejected-speaker",
                    )
                self._reset_detection_state()
                return

            speaker_name = best_name
            _LOGGER.info(
                "Speaker verified: %s (score=%.4f, threshold=%.3f, all_scores=%s)",
                speaker_name,
                best_score,
                threshold,
                scores_repr,
            )
            self._last_ww_speaker = speaker_name

        if self._sample_store is not None and fire_window_pcm is not None:
            self._persist_sample(
                keyword=keyword,
                score=fire_score,
                speaker=speaker_name,
                accepted=True,
                pcm=fire_window_pcm,
                source="auto",
            )

        if self._event_bus is not None:
            self._event_bus.publish(
                DetectionEvent(
                    keyword=keyword,
                    speaker=speaker_name,
                    score=fire_score,
                    timestamp=timestamp,
                )
            )

        await self._emit_detection(keyword, timestamp=timestamp, speaker=speaker_name)
        self._reset_detection_state()

    def _persist_sample(
        self,
        keyword: str,
        score: Optional[float],
        speaker: Optional[str],
        accepted: bool,
        pcm: bytes,
        source: str,
    ) -> None:
        """Dump a captured WW window to the sample store. Best-effort, never raises."""
        if self._sample_store is None:
            return
        try:
            meta = SampleMeta(
                id="",
                timestamp=time.time(),
                keyword=keyword,
                score=score,
                speaker=speaker,
                accepted=accepted,
                source=source,
                device_id=self._device_id,
            )
            self._sample_store.add(pcm, meta)
        except Exception as exc:
            _LOGGER.warning("Failed to persist sample: %s", exc)

    def _reset_detection_state(self) -> None:
        """Clear per-detection buffers so successive detections don't see stale data."""
        if self._ring_buffer is not None:
            self._ring_buffer.clear()
        self._stream.reset()

    async def _emit_detection(
        self,
        name: str,
        timestamp: Optional[int] = None,
        speaker: Optional[str] = None,
    ) -> None:
        """Emit Detection event when wake word is detected."""
        _LOGGER.info("Wake word detected: %s at %s (speaker: %s)", name, timestamp, speaker)
        detection = Detection(
            name=name,
            timestamp=timestamp,
            speaker=speaker,
            context=self._detect_context,
        )
        await self.write_event(detection.event())

        self.is_detecting = False
        self._detect_context = None
