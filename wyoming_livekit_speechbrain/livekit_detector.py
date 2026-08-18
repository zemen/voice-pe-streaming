"""LiveKit wake word detection integration."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

try:
    from livekit.wakeword import WakeWordModel
except ImportError:
    WakeWordModel = None

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280
FRAME_BYTES = FRAME_SAMPLES * 2
CHUNK_SECONDS = 2.0
CHUNK_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)
DEFAULT_PREDICT_INTERVAL = 0.5


@dataclass
class LiveKitDetectorConfig:
    """Configuration for LiveKit wake word detection."""

    model_paths: List[Path]
    keywords: List[str]
    sensitivities: List[float] = field(default_factory=list)
    predict_interval_seconds: float = DEFAULT_PREDICT_INTERVAL
    save_sample_threshold: Optional[float] = None

    def __post_init__(self):
        if not self.sensitivities:
            self.sensitivities = [0.5] * len(self.model_paths)
        if len(self.model_paths) != len(self.keywords):
            raise ValueError("Number of model paths must match number of keywords")
        if len(self.sensitivities) != len(self.model_paths):
            raise ValueError("Number of sensitivities must match number of models")
        if self.predict_interval_seconds < 0:
            raise ValueError(
                f"predict_interval_seconds must be >= 0, got {self.predict_interval_seconds}"
            )


@dataclass
class LiveKitDetector:
    """Shared, stateless wake-word model wrapper.

    Holds the (expensive-to-load) `WakeWordModel` and applies per-keyword
    sensitivity thresholds. One instance per server, safe to share across
    Wyoming handlers because the underlying model is stateless and audio
    buffering lives in :class:`LiveKitStream` (per handler).
    """

    config: LiveKitDetectorConfig
    sample_rate: int = SAMPLE_RATE
    frame_length: int = FRAME_SAMPLES
    chunk_frames: int = CHUNK_FRAMES

    _model: Optional[object] = field(default=None, init=False, repr=False)
    _model_names: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._model_names = [path.stem for path in self.config.model_paths]
        if WakeWordModel is None:
            _LOGGER.warning("livekit-wakeword not installed - detection disabled")
            return
        try:
            self._model = WakeWordModel(models=[str(path) for path in self.config.model_paths])
        except Exception as exc:
            _LOGGER.error("Failed to initialize LiveKit model: %s", exc, exc_info=True)
            raise RuntimeError(f"Failed to initialize LiveKit detector: {exc}") from exc

    async def predict_async(
        self, audio_window: np.ndarray
    ) -> Tuple[Optional[Tuple[int, str, float]], Optional[Tuple[int, str, float]]]:
        """Async wrapper around :meth:`predict` that dispatches the synchronous ONNX
        call via ``loop.run_in_executor``. The asyncio loop stays responsive during
        inference even with many concurrent handler sessions — prevents a loop-starvation that caused detection blackouts under HA-side pipeline storms.

        No explicit concurrency cap: the default ``ThreadPoolExecutor`` already
        bounds parallelism (Python uses ``min(32, os.cpu_count() + 4)`` workers), and
        the ORT single-thread monkey-patch in ``__main__.py`` keeps each predict to
        one CPU thread. Scales linearly with device count without per-instance tuning.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, audio_window)

    def predict(
        self, audio_window: np.ndarray
    ) -> Tuple[Optional[Tuple[int, str, float]], Optional[Tuple[int, str, float]]]:
        """Score a ~2 s audio window and surface BOTH a winning detection and a sub-threshold near-miss.

        Returns ``(hit, candidate)``:
          - ``hit`` is ``(index, name, score)`` for the highest-scoring keyword whose score
            is at or above its configured sensitivity, else ``None``.
          - ``candidate`` is ``(index, name, score)`` for the highest-scoring keyword whose
            score is below sensitivity but at or above ``config.save_sample_threshold``,
            else ``None``. Only populated when ``save_sample_threshold`` is set AND no hit
            exists (a real hit always wins over a near-miss for the same window).

        The near-miss path lets the handler persist mining samples for fine-tuning without
        firing Wyoming detection. Single ``self._model.predict()`` call powers both
        branches — zero CPU cost over the hit-only flow.
        """
        if self._model is None:
            return None, None
        scores = self._model.predict(audio_window)
        if not isinstance(scores, dict) or not scores:
            return None, None
        best_hit_index: Optional[int] = None
        best_hit_score = float("-inf")
        best_cand_index: Optional[int] = None
        best_cand_score = float("-inf")
        save_threshold = self.config.save_sample_threshold
        for index, keyword in enumerate(self.config.keywords):
            score = self._lookup_score(scores, index, keyword)
            if score is None:
                continue
            sensitivity = self.config.sensitivities[index]
            if score >= sensitivity:
                if score > best_hit_score:
                    best_hit_index = index
                    best_hit_score = score
            elif save_threshold is not None and score >= save_threshold:
                if score > best_cand_score:
                    best_cand_index = index
                    best_cand_score = score
        if best_hit_index is not None:
            _LOGGER.info(
                "predict FIRED: keyword=%s score=%.4f threshold=%.4f all_scores=%s",
                self.config.keywords[best_hit_index],
                best_hit_score,
                self.config.sensitivities[best_hit_index],
                scores,
            )
            return (best_hit_index, self.config.keywords[best_hit_index], best_hit_score), None
        if best_cand_index is not None:
            _LOGGER.debug(
                "predict candidate (below sensitivity): keyword=%s score=%.4f save_threshold=%.4f",
                self.config.keywords[best_cand_index],
                best_cand_score,
                save_threshold,
            )
            return None, (
                best_cand_index,
                self.config.keywords[best_cand_index],
                best_cand_score,
            )
        peak = int(np.abs(audio_window).max()) if audio_window.size else 0
        _LOGGER.debug(
            "predict: peak=%d scores=%s",
            peak,
            {k: round(v, 4) for k, v in scores.items()},
        )
        return None, None

    def _lookup_score(self, scores: Dict[str, float], index: int, keyword: str) -> Optional[float]:
        if keyword in scores:
            return float(scores[keyword])
        if index < len(self._model_names) and self._model_names[index] in scores:
            return float(scores[self._model_names[index]])
        values = list(scores.values())
        if index < len(values):
            return float(values[index])
        return None


@dataclass
class LiveKitStream:
    """Per-connection audio buffer that drives :class:`LiveKitDetector`.

    Wyoming chunks arrive at variable sizes; this class accumulates them
    into 80 ms (1280-sample) frames, slides a 25-frame (~2 s) window, and
    invokes ``detector.predict()`` at most once per
    ``predict_interval_seconds`` (default 0.5s) so the heavy ONNX pipeline
    is not re-run on every 80 ms frame.
    """

    detector: LiveKitDetector
    _byte_buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _frame_buffer: deque = field(init=False, repr=False)
    _last_predict_monotonic: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        self._frame_buffer = deque(maxlen=self.detector.chunk_frames)

    @property
    def frame_length(self) -> int:
        return self.detector.frame_length

    @property
    def sample_rate(self) -> int:
        return self.detector.sample_rate

    def process_audio(
        self, audio_chunk: bytes
    ) -> Tuple[
        Optional[Tuple[int, str, float, bytes]],
        Optional[Tuple[int, str, float, bytes]],
    ]:
        """Feed audio bytes; return ``(hit, candidate)`` per predict, both with their PCM windows.

        Each element, when present, is ``(keyword_index, keyword_name, score, window_pcm)``.
        ``hit`` is a real detection (score >= sensitivity); ``candidate`` is a near-miss
        (sensitivity > score >= save_sample_threshold). They are mutually exclusive per window:
        a real hit suppresses the candidate path. ``window_pcm`` is the as-captured 2 s int16
        buffer (no normalization, no transform).

        Buffer policy:
          - On hit: clear both frame_buffer and byte_buffer (existing behavior — prevents
            double-firing on the same window).
          - On candidate: do NOT clear. The wake word might still complete in the next 200 ms
            and produce a real hit; clearing here would silently drop it. Throttling by
            ``predict_interval_seconds`` rate-limits candidate emissions to ~1/interval per
            stream, and the handler's per-source retention bucket caps total storage.

        Snowball-prevention: `_last_predict_monotonic` is updated to the END of
        the predict() call, not the start. If predict() takes longer than the
        configured throttle interval (CPU contention, long synchronous ONNX
        inference), the accumulated byte_buffer backlog is dropped so we
        always score the most recent audio window, never a stale one.
        """
        if self.detector._model is None:
            return None, None
        self._byte_buffer.extend(audio_chunk)
        interval = self.detector.config.predict_interval_seconds
        while len(self._byte_buffer) >= FRAME_BYTES:
            frame_bytes = bytes(self._byte_buffer[:FRAME_BYTES])
            del self._byte_buffer[:FRAME_BYTES]
            frame = np.frombuffer(frame_bytes, dtype=np.int16)
            self._frame_buffer.append(frame)
            if len(self._frame_buffer) < self.detector.chunk_frames:
                continue
            now = time.monotonic()
            if interval > 0 and (now - self._last_predict_monotonic) < interval:
                continue
            predict_start = time.monotonic()
            window = np.concatenate(list(self._frame_buffer))
            raw_window_pcm = window.tobytes()
            hit, candidate = self.detector.predict(window)
            predict_end = time.monotonic()
            elapsed = predict_end - predict_start
            self._last_predict_monotonic = predict_end
            if interval > 0 and elapsed > interval:
                _LOGGER.warning(
                    "predict took %.3fs (> %.3fs throttle); dropping backlog to prevent snowball",
                    elapsed,
                    interval,
                )
                self._byte_buffer = bytearray()
            if hit is not None:
                keyword_index, keyword_name, score = hit
                self._frame_buffer.clear()
                self._byte_buffer = bytearray()
                return (keyword_index, keyword_name, score, raw_window_pcm), None
            if candidate is not None:
                keyword_index, keyword_name, score = candidate
                return None, (keyword_index, keyword_name, score, raw_window_pcm)
        return None, None

    async def process_audio_async(
        self, audio_chunk: bytes
    ) -> Tuple[
        Optional[Tuple[int, str, float, bytes]],
        Optional[Tuple[int, str, float, bytes]],
    ]:
        """Async sibling of :meth:`process_audio` that dispatches predict via
        :meth:`LiveKitDetector.predict_async`. Identical buffer + throttle policy,
        identical return shape. Use this from asyncio handlers (Wyoming) to avoid
        blocking the event loop during ONNX inference. Sync ``process_audio`` is
        preserved for callers that drive the stream from a thread (web_server.py
        debug endpoints) or run in their own executor.
        """
        if self.detector._model is None:
            return None, None
        self._byte_buffer.extend(audio_chunk)
        interval = self.detector.config.predict_interval_seconds
        while len(self._byte_buffer) >= FRAME_BYTES:
            frame_bytes = bytes(self._byte_buffer[:FRAME_BYTES])
            del self._byte_buffer[:FRAME_BYTES]
            frame = np.frombuffer(frame_bytes, dtype=np.int16)
            self._frame_buffer.append(frame)
            if len(self._frame_buffer) < self.detector.chunk_frames:
                continue
            now = time.monotonic()
            if interval > 0 and (now - self._last_predict_monotonic) < interval:
                continue
            predict_start = time.monotonic()
            window = np.concatenate(list(self._frame_buffer))
            raw_window_pcm = window.tobytes()
            hit, candidate = await self.detector.predict_async(window)
            predict_end = time.monotonic()
            elapsed = predict_end - predict_start
            self._last_predict_monotonic = predict_end
            if interval > 0 and elapsed > interval:
                _LOGGER.warning(
                    "predict took %.3fs (> %.3fs throttle); dropping backlog to prevent snowball",
                    elapsed,
                    interval,
                )
                self._byte_buffer = bytearray()
            if hit is not None:
                keyword_index, keyword_name, score = hit
                self._frame_buffer.clear()
                self._byte_buffer = bytearray()
                return (keyword_index, keyword_name, score, raw_window_pcm), None
            if candidate is not None:
                keyword_index, keyword_name, score = candidate
                return None, (keyword_index, keyword_name, score, raw_window_pcm)
        return None, None

    def reset(self) -> None:
        self._byte_buffer = bytearray()
        self._frame_buffer.clear()
        self._last_predict_monotonic = 0.0
