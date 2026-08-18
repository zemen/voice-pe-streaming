"""Main entry point for Wyoming LiveKit + SpeechBrain server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import ADDON_VERSION

# ONNX Runtime's per-session intra/inter-op thread pool is NOT configurable via
# env vars (OMP_NUM_THREADS only limits OpenMP-backed matrix ops, not the ORT
# thread pool itself). Multiple ORT worker pools waste cycles on thread
# contention for these small mel + embedding + classifier models. Patch
# the InferenceSession constructor so every session in this process defaults
# to single-thread unless the caller passes explicit sess_options.
try:
    import onnxruntime as _ort

    _ort_original_init = _ort.InferenceSession.__init__

    def _ort_single_thread_init(self, *args, **kwargs):
        if "sess_options" not in kwargs:
            opts = _ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            kwargs["sess_options"] = opts
        _ort_original_init(self, *args, **kwargs)

    _ort.InferenceSession.__init__ = _ort_single_thread_init
except ImportError:
    pass

try:
    import torch as _torch

    _torch.set_num_threads(1)
    _torch.set_num_interop_threads(1)
except ImportError:
    pass

from wyoming.server import AsyncTcpServer

from .device_registry import DeviceRegistry
from .events import DetectionEventBus
from .handler import LiveKitEventHandler
from .livekit_detector import LiveKitDetector, LiveKitDetectorConfig
from .sample_store import DEFAULT_MAX_SAMPLES, DEFAULT_SAMPLES_DIR, SampleStore
from .speechbrain_verifier import SpeechBrainConfig, SpeechBrainVerifier
from .web_server import DEFAULT_PROFILES_DIR, WebServerContext, run_web_server

_LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="wyoming-livekit-speechbrain",
        description="Wyoming server for LiveKit wake word detection and SpeechBrain speaker verification",
    )

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=10400, help="TCP port to listen on")
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        dest="models",
        required=True,
        help="Path to wake word ONNX model (repeatable)",
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        action="append",
        dest="wake_words",
        help="Wake word name for each model (repeatable)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        action="append",
        dest="sensitivities",
        help="Per-model detection sensitivity threshold (repeatable)",
    )
    parser.add_argument(
        "--predict-interval-seconds",
        type=float,
        default=0.5,
        help=(
            "Minimum seconds between WakeWordModel.predict() calls "
            "(default: 0.5). Lower = lower detection latency, higher CPU. "
            "0.0 disables throttling (predict on every 80 ms frame)."
        ),
    )
    parser.add_argument(
        "--save-sample-threshold",
        type=float,
        default=None,
        help=(
            "Second, lower wake-word score threshold for SAVE-ONLY captures (no detection "
            "fired). When the classifier scores in [save-sample-threshold, sensitivity), "
            "the 2 s window is persisted to the sample store with source='auto-below-threshold' "
            "for fine-tune dataset mining of near-misses. Pass without value or omit to disable. "
            "Recommended: 0.25 (32%% below default sensitivity 0.37). Lower = more samples = "
            "more storage + more manual labeling work."
        ),
    )
    parser.add_argument(
        "--speaker-profile",
        type=str,
        action="append",
        dest="speaker_profiles",
        help="Path to SpeechBrain speaker profile (.npy) (repeatable)",
    )
    parser.add_argument(
        "--speaker-threshold",
        type=float,
        default=0.25,
        help="SpeechBrain cosine similarity threshold (default: 0.25)",
    )
    parser.add_argument(
        "--speaker-model",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain model source",
    )
    parser.add_argument(
        "--speaker-savedir",
        type=str,
        default=None,
        help="SpeechBrain model cache directory",
    )
    parser.add_argument(
        "--mid-sentence-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mid-sentence wake word rejection",
    )
    parser.add_argument(
        "--mid-sentence-silence-rms",
        type=float,
        default=0.05,
        help=(
            "Pre-WW RMS energy below this is treated as silence (skip mid-sentence rejection). "
            "Tune per-room: quieter rooms can go lower (0.01), noisier rooms higher (0.10). "
            "Default 0.05 = roughly -26 dBFS. Set 0 to disable the energy gate."
        ),
    )
    parser.add_argument(
        "--enable-mid-sentence-detection",
        dest="mid_sentence_detection",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--web-host",
        type=str,
        default="0.0.0.0",
        help="HTTP/WebSocket bind host for the web UI (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=10401,
        help="HTTP/WebSocket port for the web UI (default: 10401)",
    )
    parser.add_argument(
        "--disable-web-ui",
        action="store_true",
        help="Do not start the web UI server",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=DEFAULT_PROFILES_DIR,
        help="Directory where speaker .npy profiles are stored (auto-discovered)",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory where captured wake-word audio samples are saved",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Default max samples per source (oldest evicted, 0 = unlimited). Each source gets its own bucket of this size unless overridden via --max-samples-source.",
    )
    parser.add_argument(
        "--max-samples-source",
        action="append",
        default=[],
        metavar="SOURCE:N",
        help=(
            "Per-source override for retention quota. Repeat per source. Example: "
            "--max-samples-source auto-rejected-speaker:200000 --max-samples-source auto:1000. "
            "Sources without an override fall back to --max-samples."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    if args.wake_words is None:
        args.wake_words = [Path(model).stem for model in args.models]
    if len(args.models) != len(args.wake_words):
        parser.error("--model and --wake-word must be provided 1:1")
    if args.sensitivities is not None and len(args.sensitivities) != len(args.models):
        parser.error("--sensitivity must be provided 1:1 with --model")
    return args


def create_handler_factory(
    wake_word_names: List[str],
    model_paths: List[Path],
    livekit: LiveKitDetector,
    speaker_verifier: Optional[SpeechBrainVerifier] = None,
    enable_mid_sentence_detection: bool = True,
    device_registry: Optional[DeviceRegistry] = None,
    sample_store: Optional[SampleStore] = None,
    event_bus: Optional[DetectionEventBus] = None,
):
    """Create a handler factory using shared detector instances."""

    def factory(reader, writer):
        return LiveKitEventHandler(
            reader=reader,
            writer=writer,
            wake_word_names=wake_word_names,
            model_paths=model_paths,
            livekit=livekit,
            speaker_verifier=speaker_verifier,
            enable_mid_sentence_detection=enable_mid_sentence_detection,
            device_registry=device_registry,
            sample_store=sample_store,
            event_bus=event_bus,
        )

    return factory


async def main_async() -> int:
    """Async main entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model_paths = [Path(path) for path in args.models]
    for model_path in model_paths:
        if not model_path.exists():
            _LOGGER.warning("Model path does not exist: %s", model_path)

    wake_word_names = list(args.wake_words)
    sensitivities = list(args.sensitivities) if args.sensitivities else []

    speaker_profile_paths = [Path(path) for path in args.speaker_profiles or []]
    for profile_path in speaker_profile_paths:
        if not profile_path.exists():
            _LOGGER.warning("Speaker profile path does not exist: %s", profile_path)

    args.profiles_dir.mkdir(parents=True, exist_ok=True)
    auto_discovered = sorted(args.profiles_dir.glob("*.npy"))
    if auto_discovered:
        existing = {p.resolve() for p in speaker_profile_paths}
        for npy in auto_discovered:
            if npy.resolve() not in existing:
                speaker_profile_paths.append(npy)
        _LOGGER.info(
            "Auto-discovered %d profile(s) in %s",
            len(auto_discovered),
            args.profiles_dir,
        )

    _LOGGER.info(
        "wyoming-livekit-speechbrain v%s starting on %s:%d", ADDON_VERSION, args.host, args.port
    )
    _LOGGER.info(
        "Threading: ORT intra_op=1 inter_op=1, torch threads=1 "
        "(single-threaded for predictable low-power-host performance)"
    )
    _LOGGER.info(
        "Predict interval: %.2fs (throttle); sensitivities: %s",
        args.predict_interval_seconds,
        sensitivities or [0.5] * len(model_paths),
    )
    _LOGGER.info("Wake words: %s", ", ".join(wake_word_names))
    _LOGGER.info("Model paths: %s", ", ".join(str(path) for path in model_paths))

    livekit = LiveKitDetector(
        LiveKitDetectorConfig(
            model_paths=model_paths,
            keywords=wake_word_names,
            sensitivities=sensitivities,
            predict_interval_seconds=args.predict_interval_seconds,
            save_sample_threshold=args.save_sample_threshold,
        )
    )

    speaker_verifier: Optional[SpeechBrainVerifier] = None
    if speaker_profile_paths:
        speaker_verifier = SpeechBrainVerifier(
            SpeechBrainConfig(
                profile_paths=speaker_profile_paths,
                threshold=args.speaker_threshold,
                model_source=args.speaker_model,
                savedir=Path(args.speaker_savedir) if args.speaker_savedir else None,
            )
        )

    device_registry = DeviceRegistry()
    max_samples_per_source: dict[str, int] = {}
    for spec in args.max_samples_source:
        if ":" not in spec:
            parser.error(f"--max-samples-source must be SOURCE:N, got: {spec!r}")
        src, n_str = spec.rsplit(":", 1)
        try:
            max_samples_per_source[src] = int(n_str)
        except ValueError:
            parser.error(f"--max-samples-source N must be int, got: {n_str!r} (spec={spec!r})")
    sample_store = SampleStore(
        samples_dir=args.samples_dir,
        max_samples=args.max_samples,
        max_samples_per_source=max_samples_per_source,
    )
    _LOGGER.info(
        "Sample store at %s (max %d samples, %d currently)",
        sample_store.samples_dir,
        sample_store.max_samples,
        len(sample_store.list()),
    )

    event_bus = DetectionEventBus()

    server = AsyncTcpServer(args.host, args.port)
    handler_factory = create_handler_factory(
        wake_word_names=wake_word_names,
        model_paths=model_paths,
        livekit=livekit,
        speaker_verifier=speaker_verifier,
        enable_mid_sentence_detection=args.mid_sentence_detection,
        device_registry=device_registry,
        sample_store=sample_store,
        event_bus=event_bus,
    )

    tasks = [asyncio.create_task(server.run(handler_factory), name="wyoming-tcp")]
    if not args.disable_web_ui:
        web_ctx = WebServerContext(
            detector=livekit,
            verifier=speaker_verifier,
            profiles_dir=args.profiles_dir,
            speaker_model_source=args.speaker_model,
            speaker_savedir=Path(args.speaker_savedir) if args.speaker_savedir else None,
            speaker_threshold=args.speaker_threshold,
            device_registry=device_registry,
            sample_store=sample_store,
            event_bus=event_bus,
        )
        tasks.append(
            asyncio.create_task(
                run_web_server(web_ctx, host=args.web_host, port=args.web_port),
                name="web-ui",
            )
        )

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc:
                _LOGGER.error("Server task %s crashed: %s", task.get_name(), exc, exc_info=exc)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped by user")
    finally:
        await server.stop()

    return 0


def main() -> int:
    """Main entry point."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
