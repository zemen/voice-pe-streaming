"""HTTP + WebSocket server for in-HA-UI speaker enrollment and pipeline debugging."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from aiohttp import WSMsgType, web

from .device_registry import DeviceRegistry
from .enrollment import enroll_speaker
from .events import DetectionEventBus
from .livekit_detector import FRAME_BYTES, LiveKitDetector, LiveKitStream
from .sample_store import SampleMeta, SampleStore
from .speechbrain_verifier import SpeechBrainVerifier

_LOGGER = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = Path("/share/profiles")
WEB_STATIC_DIR = Path(__file__).resolve().parent / "web"

# Speaker name allow list - filename-safe characters only. Constrains user input
# to avoid path traversal (../) and filesystem injection when saving profiles.
_VALID_SPEAKER_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")


def _profile_path(profiles_dir: Path, name: str) -> Path:
    """Map a speaker name to its .npy path under profiles_dir."""
    if not _VALID_SPEAKER_NAME.match(name):
        raise web.HTTPBadRequest(
            reason=f"Invalid speaker name '{name}': must match {_VALID_SPEAKER_NAME.pattern}"
        )
    return profiles_dir / f"{name}.npy"


def _list_profiles(profiles_dir: Path) -> List[Dict[str, Any]]:
    """Return metadata for every .npy in the profiles dir."""
    if not profiles_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for npy in sorted(profiles_dir.glob("*.npy")):
        name = npy.name
        while "." in name:
            name = name.rsplit(".", 1)[0]
        try:
            stat = npy.stat()
            out.append(
                {
                    "name": name,
                    "path": str(npy),
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
        except OSError:
            continue
    return out


def _save_wav_from_pcm(pcm_bytes: bytes, out_path: Path, sample_rate: int = 16000) -> None:
    """Wrap raw 16-bit mono PCM in a WAV header so enroll_speaker can read it."""
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)


class WebServerContext:
    """Shared state across HTTP handlers - detector, verifier, profile dir, device registry, sample store."""

    def __init__(
        self,
        detector: LiveKitDetector,
        verifier: Optional[SpeechBrainVerifier],
        profiles_dir: Path,
        speaker_model_source: str,
        speaker_savedir: Optional[Path],
        speaker_threshold: float,
        device_registry: Optional[DeviceRegistry] = None,
        sample_store: Optional[SampleStore] = None,
        event_bus: Optional[DetectionEventBus] = None,
    ):
        self.event_bus = event_bus
        self.detector = detector
        self.verifier = verifier
        self.profiles_dir = profiles_dir
        self.speaker_model_source = speaker_model_source
        self.speaker_savedir = speaker_savedir
        self.speaker_threshold = speaker_threshold
        self.device_registry = device_registry
        self.sample_store = sample_store
        self.profiles_dir.mkdir(parents=True, exist_ok=True)


async def handle_index(request: web.Request) -> web.Response:
    from . import ADDON_VERSION

    html = (WEB_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="static/styles.css"', f'href="static/styles.css?v={ADDON_VERSION}"')
    html = html.replace('src="static/app.js"', f'src="static/app.js?v={ADDON_VERSION}"')
    return web.Response(
        text=html,
        content_type="text/html",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


async def handle_version(request: web.Request) -> web.Response:
    """GET /api/version - returns the running addon version."""
    from . import ADDON_VERSION

    return web.json_response({"version": ADDON_VERSION})


async def handle_speakers_list(request: web.Request) -> web.Response:
    """GET /api/speakers -> list of enrolled speaker profiles."""
    ctx: WebServerContext = request.app["ctx"]
    return web.json_response(
        {
            "speakers": _list_profiles(ctx.profiles_dir),
            "threshold": ctx.speaker_threshold,
            "verifier_active": ctx.verifier is not None,
        }
    )


async def handle_speaker_enroll(request: web.Request) -> web.Response:
    """POST /api/speakers/{name}/enroll - body: multipart with WAV/PCM samples."""
    ctx: WebServerContext = request.app["ctx"]
    name = request.match_info["name"]
    out_path = _profile_path(ctx.profiles_dir, name)

    reader = await request.multipart()
    saved_wavs: List[Path] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="wls_enroll_"))
    try:
        idx = 0
        async for part in reader:
            if part.name != "sample":
                continue
            pcm = await part.read()
            if not pcm:
                continue
            wav_path = tmpdir / f"sample_{idx}.wav"
            _save_wav_from_pcm(pcm, wav_path)
            saved_wavs.append(wav_path)
            idx += 1

        if not saved_wavs:
            raise web.HTTPBadRequest(reason="No 'sample' parts in multipart body")

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: enroll_speaker(
                audio_paths=saved_wavs,
                output_profile=out_path,
                model_source=ctx.speaker_model_source,
                savedir=ctx.speaker_savedir,
            ),
        )

        if ctx.verifier is not None:
            all_npy = sorted(ctx.profiles_dir.glob("*.npy"))
            ctx.verifier.reload_profiles(all_npy)

        return web.json_response(
            {
                "ok": True,
                "name": name,
                "samples_used": len(saved_wavs),
                "profile_path": str(out_path),
                "size_bytes": out_path.stat().st_size if out_path.exists() else 0,
            }
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def handle_speaker_delete(request: web.Request) -> web.Response:
    """DELETE /api/speakers/{name} - remove the .npy and hot-reload verifier."""
    ctx: WebServerContext = request.app["ctx"]
    name = request.match_info["name"]
    profile_path = _profile_path(ctx.profiles_dir, name)
    if not profile_path.exists():
        raise web.HTTPNotFound(reason=f"No such profile: {name}")
    profile_path.unlink()

    if ctx.verifier is not None:
        all_npy = sorted(ctx.profiles_dir.glob("*.npy"))
        ctx.verifier.reload_profiles(all_npy)

    return web.json_response({"ok": True, "deleted": name})


async def handle_speaker_rename(request: web.Request) -> web.Response:
    """POST /api/speakers/{name}/rename - body: {new_name: str}."""
    ctx: WebServerContext = request.app["ctx"]
    old_name = request.match_info["name"]
    body = await request.json()
    new_name = body.get("new_name", "").strip()
    old_path = _profile_path(ctx.profiles_dir, old_name)
    new_path = _profile_path(ctx.profiles_dir, new_name)

    if not old_path.exists():
        raise web.HTTPNotFound(reason=f"No such profile: {old_name}")
    if new_path.exists():
        raise web.HTTPConflict(reason=f"Profile already exists: {new_name}")
    old_path.rename(new_path)

    if ctx.verifier is not None:
        all_npy = sorted(ctx.profiles_dir.glob("*.npy"))
        ctx.verifier.reload_profiles(all_npy)

    return web.json_response({"ok": True, "old_name": old_name, "new_name": new_name})


async def handle_verify(request: web.Request) -> web.Response:
    """POST /api/verify - body: raw 16k mono int16 PCM. Returns scores per speaker."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.verifier is None:
        return web.json_response(
            {"error": "Speaker verification is not enabled"}, status=400
        )
    pcm = await request.read()
    if not pcm:
        raise web.HTTPBadRequest(reason="Empty body")
    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(None, ctx.verifier.verify_with_scores, pcm)
    best_name: Optional[str] = None
    best_score = float("-inf")
    for name, score in scores.items():
        if score > best_score:
            best_name = name
            best_score = score
    return web.json_response(
        {
            "scores": scores,
            "best_speaker": best_name,
            "best_score": best_score if best_name else None,
            "threshold": ctx.speaker_threshold,
            "accepted": best_name is not None and best_score >= ctx.speaker_threshold,
        }
    )


async def handle_debug_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket /api/debug-stream - receive int16 PCM, stream pipeline scores live.

    Protocol:
      Client -> binary frames of 16k mono int16 PCM (any chunk size).
      Server -> JSON frames with at most one of:
        {"event":"hello", ...}
        {"event":"ww_score", "score": float, "scores": {kw: score}, "fired": bool, "peak": int}
        {"event":"speaker", "scores": {name: float}, "best": name|null, "accepted": bool}
        {"event":"mid_sentence", "rejected": bool}
        {"event":"error", "message": str}
    """
    ctx: WebServerContext = request.app["ctx"]
    ws = web.WebSocketResponse(heartbeat=15.0)
    await ws.prepare(request)

    stream = LiveKitStream(detector=ctx.detector)
    await ws.send_json(
        {
            "event": "hello",
            "wake_words": list(ctx.detector.config.keywords),
            "sensitivities": list(ctx.detector.config.sensitivities),
            "predict_interval_seconds": ctx.detector.config.predict_interval_seconds,
            "speaker_verifier_active": ctx.verifier is not None,
            "speaker_threshold": ctx.speaker_threshold,
        }
    )

    pending_verify_audio = bytearray()
    pending_verify_limit_bytes = 16000 * 2 * 2

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                audio = bytes(msg.data)
                pending_verify_audio.extend(audio)
                if len(pending_verify_audio) > pending_verify_limit_bytes:
                    drop = len(pending_verify_audio) - pending_verify_limit_bytes
                    del pending_verify_audio[:drop]

                hit = stream.process_audio(audio)
                if hit is not None:
                    _, keyword = hit
                    await ws.send_json(
                        {
                            "event": "ww_score",
                            "fired": True,
                            "keyword": keyword,
                        }
                    )
                    if ctx.verifier is not None:
                        scores = await asyncio.get_event_loop().run_in_executor(
                            None, ctx.verifier.verify_with_scores, bytes(pending_verify_audio)
                        )
                        best_name: Optional[str] = None
                        best_score = float("-inf")
                        for name, score in scores.items():
                            if score > best_score:
                                best_name = name
                                best_score = score
                        accepted = (
                            best_name is not None and best_score >= ctx.speaker_threshold
                        )
                        await ws.send_json(
                            {
                                "event": "speaker",
                                "scores": scores,
                                "best": best_name,
                                "accepted": accepted,
                            }
                        )

            elif msg.type == WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if cmd.get("cmd") == "verify_now" and ctx.verifier is not None:
                    scores = await asyncio.get_event_loop().run_in_executor(
                        None, ctx.verifier.verify_with_scores, bytes(pending_verify_audio)
                    )
                    best_name = None
                    best_score = float("-inf")
                    for name, score in scores.items():
                        if score > best_score:
                            best_name = name
                            best_score = score
                    accepted = (
                        best_name is not None and best_score >= ctx.speaker_threshold
                    )
                    await ws.send_json(
                        {
                            "event": "speaker",
                            "scores": scores,
                            "best": best_name,
                            "accepted": accepted,
                            "triggered_manually": True,
                        }
                    )
                elif cmd.get("cmd") == "reset_buffers":
                    stream.reset()
                    pending_verify_audio.clear()

            elif msg.type == WSMsgType.ERROR:
                _LOGGER.error("WebSocket error: %s", ws.exception())
    finally:
        await ws.close()
    return ws


async def handle_score_stream_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket /api/score-stream - live WW scores + speaker scores + VAD bit per predict.

    Text commands (client -> server):
      {"cmd": "subscribe_device", "device_id": "..."} -- consume audio from a
        Wyoming-connected Voice PE instead of from binary WS frames.
      {"cmd": "unsubscribe_device"} -- stop consuming the device queue; resume
        consuming binary frames from the browser.

    Binary frames (client -> server): raw 16k mono int16 PCM. Ignored while
    subscribed to a device.

    Server -> client events:
      {"event":"hello", ...}
      {"event":"source", "source":"browser"|"voice-pe", "device_id":"..."}
      {"event":"ww_score", "scores":{kw:float}, "peak":int, "fired_keyword":...,
       "speaker_scores":{name:float}, "vad_active":bool, "rms":float,
       "timestamp":float}
    """
    import time as _time

    ctx: WebServerContext = request.app["ctx"]
    ws = web.WebSocketResponse(heartbeat=15.0)
    await ws.prepare(request)

    chunk_frames = ctx.detector.chunk_frames
    interval = ctx.detector.config.predict_interval_seconds
    silence_threshold = 0.05

    state: Dict[str, Any] = {
        "byte_buffer": bytearray(),
        "frame_buffer": [],
        "last_predict": 0.0,
        "source": "browser",
        "device_id": None,
        "subscription_q": None,
        "subscription_handler": None,
    }

    await ws.send_json(
        {
            "event": "hello",
            "wake_words": list(ctx.detector.config.keywords),
            "sensitivities": list(ctx.detector.config.sensitivities),
            "predict_interval_seconds": interval,
            "speaker_verifier_active": ctx.verifier is not None,
            "speaker_threshold": ctx.speaker_threshold,
        }
    )

    async def process_audio_bytes(audio: bytes) -> None:
        state["byte_buffer"].extend(audio)
        while len(state["byte_buffer"]) >= FRAME_BYTES:
            frame_bytes = bytes(state["byte_buffer"][:FRAME_BYTES])
            del state["byte_buffer"][:FRAME_BYTES]
            frame = np.frombuffer(frame_bytes, dtype=np.int16)
            fb = state["frame_buffer"]
            fb.append(frame)
            if len(fb) > chunk_frames:
                fb.pop(0)
            if len(fb) < chunk_frames:
                continue
            now = _time.monotonic()
            if interval > 0 and (now - state["last_predict"]) < interval:
                continue
            state["last_predict"] = now
            window = np.concatenate(fb)
            loop = asyncio.get_event_loop()
            raw_scores = await loop.run_in_executor(None, ctx.detector._model.predict, window)
            peak = int(np.abs(window).max())
            rms = float(np.sqrt(np.mean((window.astype(np.float32) / 32768.0) ** 2)))
            vad_active = rms >= silence_threshold
            fired_keyword = None
            for idx, kw in enumerate(ctx.detector.config.keywords):
                score = raw_scores.get(kw, 0.0) if isinstance(raw_scores, dict) else 0.0
                if score >= ctx.detector.config.sensitivities[idx]:
                    fired_keyword = kw
                    break
            speaker_scores: Dict[str, float] = {}
            if ctx.verifier is not None and vad_active:
                try:
                    speaker_scores = await loop.run_in_executor(
                        None, ctx.verifier.verify_with_scores, window.tobytes()
                    )
                except Exception:
                    speaker_scores = {}
            try:
                await ws.send_json(
                    {
                        "event": "ww_score",
                        "scores": raw_scores if isinstance(raw_scores, dict) else {},
                        "peak": peak,
                        "rms": rms,
                        "vad_active": vad_active,
                        "speaker_scores": speaker_scores,
                        "fired_keyword": fired_keyword,
                        "timestamp": now,
                    }
                )
            except ConnectionResetError:
                return

    def reset_buffers() -> None:
        state["byte_buffer"] = bytearray()
        state["frame_buffer"] = []
        state["last_predict"] = 0.0

    def release_subscription() -> None:
        handler = state["subscription_handler"]
        q = state["subscription_q"]
        if handler is not None and q is not None:
            try:
                handler.unsubscribe_debug_audio(q)
            except Exception:
                pass
        state["subscription_handler"] = None
        state["subscription_q"] = None
        state["device_id"] = None
        state["source"] = "browser"

    async def device_audio_reader() -> None:
        while not ws.closed:
            q = state["subscription_q"]
            if q is None:
                await asyncio.sleep(0.1)
                continue
            try:
                audio = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if audio is None:
                release_subscription()
                try:
                    await ws.send_json({"event": "source", "source": "browser", "device_id": None})
                except ConnectionResetError:
                    return
                reset_buffers()
                continue
            await process_audio_bytes(audio)

    reader_task = asyncio.create_task(device_audio_reader())

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                if state["source"] == "browser":
                    await process_audio_bytes(bytes(msg.data))
            elif msg.type == WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                op = cmd.get("cmd")
                if op == "subscribe_device":
                    if ctx.device_registry is None:
                        continue
                    device_id = cmd.get("device_id")
                    handler = ctx.device_registry.get(device_id) if device_id else None
                    if handler is None:
                        await ws.send_json(
                            {"event": "error", "message": f"No active device: {device_id}"}
                        )
                        continue
                    release_subscription()
                    reset_buffers()
                    state["subscription_handler"] = handler
                    state["subscription_q"] = handler.subscribe_debug_audio()
                    state["device_id"] = device_id
                    state["source"] = "voice-pe"
                    await ws.send_json({"event": "source", "source": "voice-pe", "device_id": device_id})
                elif op == "unsubscribe_device":
                    release_subscription()
                    reset_buffers()
                    await ws.send_json({"event": "source", "source": "browser", "device_id": None})
    finally:
        release_subscription()
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
        await ws.close()
    return ws


async def handle_devices_list(request: web.Request) -> web.Response:
    """GET /api/devices - list active Wyoming connections (Voice PE sessions)."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.device_registry is None:
        return web.json_response({"devices": []})
    devices = [info.to_dict() for info in ctx.device_registry.list_info()]
    return web.json_response({"devices": devices})


async def handle_device_claim(request: web.Request) -> web.Response:
    """POST /api/devices/{id}/debug-claim - pause Wyoming WW pipeline; route audio to debug subscribers."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.device_registry is None:
        raise web.HTTPServiceUnavailable()
    device_id = request.match_info["id"]
    handler = ctx.device_registry.get(device_id)
    if handler is None:
        raise web.HTTPNotFound(reason=f"No active device: {device_id}")
    handler.claim_for_debug()
    return web.json_response({"ok": True, "claimed": True, "device_id": device_id})


async def handle_device_release(request: web.Request) -> web.Response:
    """POST /api/devices/{id}/debug-release - resume Wyoming WW pipeline."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.device_registry is None:
        raise web.HTTPServiceUnavailable()
    device_id = request.match_info["id"]
    handler = ctx.device_registry.get(device_id)
    if handler is None:
        raise web.HTTPNotFound(reason=f"No active device: {device_id}")
    handler.release_from_debug()
    return web.json_response({"ok": True, "claimed": False, "device_id": device_id})


async def handle_device_capture(request: web.Request) -> web.Response:
    """POST /api/devices/{id}/capture?duration=5 - record N seconds from one device, save as sample."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.device_registry is None or ctx.sample_store is None:
        raise web.HTTPServiceUnavailable(reason="Device capture not available")

    device_id = request.match_info["id"]
    handler = ctx.device_registry.get(device_id)
    if handler is None:
        raise web.HTTPNotFound(reason=f"No active device: {device_id}")

    try:
        duration = float(request.query.get("duration", "5"))
    except ValueError:
        raise web.HTTPBadRequest(reason="duration must be a number of seconds")
    if not 0.5 <= duration <= 60.0:
        raise web.HTTPBadRequest(reason="duration must be 0.5-60 seconds")

    label_param = request.query.get("label", None)
    if label_param not in (None, "fp", "tp"):
        raise web.HTTPBadRequest(reason="label must be 'fp', 'tp', or omitted")

    try:
        pcm = await handler.capture_for_duration(duration)
    except RuntimeError as exc:
        raise web.HTTPConflict(reason=str(exc))
    except TimeoutError as exc:
        raise web.HTTPGatewayTimeout(reason=str(exc))

    import time as _time

    meta = SampleMeta(
        id="",
        timestamp=_time.time(),
        keyword=None,
        score=None,
        speaker=None,
        accepted=None,
        source="manual",
        device_id=device_id,
        label=label_param,
    )
    saved = ctx.sample_store.add(pcm, meta)
    return web.json_response({"ok": True, "sample": saved.to_dict()})


async def handle_samples_list(request: web.Request) -> web.Response:
    """GET /api/samples?label=fp|tp|all - list captured samples newest-first."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        return web.json_response({"samples": []})
    label = request.query.get("label", "all")
    limit_q = request.query.get("limit")
    limit = int(limit_q) if limit_q else None
    samples = [s.to_dict() for s in ctx.sample_store.list(label=label, limit=limit)]
    return web.json_response(
        {
            "samples": samples,
            "total_size_bytes": ctx.sample_store.total_size_bytes(),
            "max_samples": ctx.sample_store.max_samples,
        }
    )


async def handle_sample_audio(request: web.Request) -> web.Response:
    """GET /api/samples/{id}/audio.wav - serve the WAV for inline browser playback."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        raise web.HTTPNotFound()
    wav_path = ctx.sample_store.get_wav_path(request.match_info["id"])
    if wav_path is None:
        raise web.HTTPNotFound(reason="No such sample")
    return web.FileResponse(wav_path, headers={"Content-Type": "audio/wav"})


async def handle_sample_label(request: web.Request) -> web.Response:
    """POST /api/samples/{id}/label - body: {"label": "fp"|"tp"|null} - set sample label."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        raise web.HTTPServiceUnavailable()
    body = await request.json()
    label = body.get("label")
    if label not in (None, "fp", "tp"):
        raise web.HTTPBadRequest(reason="label must be 'fp', 'tp', or null")
    meta = ctx.sample_store.update_label(request.match_info["id"], label)
    if meta is None:
        raise web.HTTPNotFound(reason="No such sample")
    return web.json_response({"ok": True, "sample": meta.to_dict()})


async def handle_samples_bulk_label(request: web.Request) -> web.Response:
    """POST /api/samples/bulk-label - body: {ids: [...], label: "fp"|"tp"|null} - label many at once."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        raise web.HTTPServiceUnavailable()
    body = await request.json()
    ids = body.get("ids")
    label = body.get("label")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise web.HTTPBadRequest(reason="ids must be a list of strings")
    if label not in (None, "fp", "tp"):
        raise web.HTTPBadRequest(reason="label must be 'fp', 'tp', or null")
    updated = 0
    not_found: list = []
    for sid in ids:
        meta = ctx.sample_store.update_label(sid, label)
        if meta is None:
            not_found.append(sid)
        else:
            updated += 1
    return web.json_response({"ok": True, "updated": updated, "not_found": not_found, "label": label})


async def handle_samples_bulk_delete(request: web.Request) -> web.Response:
    """POST /api/samples/bulk-delete - body: {ids: [...]} - delete many at once."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        raise web.HTTPServiceUnavailable()
    body = await request.json()
    ids = body.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise web.HTTPBadRequest(reason="ids must be a list of strings")
    deleted = 0
    for sid in ids:
        if ctx.sample_store.delete(sid):
            deleted += 1
    return web.json_response({"ok": True, "deleted": deleted, "requested": len(ids)})


async def handle_sample_delete(request: web.Request) -> web.Response:
    """DELETE /api/samples/{id}."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        raise web.HTTPServiceUnavailable()
    deleted = ctx.sample_store.delete(request.match_info["id"])
    if not deleted:
        raise web.HTTPNotFound(reason="No such sample")
    return web.json_response({"ok": True, "deleted": request.match_info["id"]})


async def handle_sample_retest(request: web.Request) -> web.Response:
    """GET /api/samples/{id}/retest - run the saved audio through the current detector, return score."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        raise web.HTTPNotFound()
    pcm = ctx.sample_store.get_pcm(request.match_info["id"])
    if pcm is None:
        raise web.HTTPNotFound(reason="No such sample")
    audio = np.frombuffer(pcm, dtype=np.int16)
    expected = 32000
    if audio.size < expected:
        audio = np.pad(audio, (0, expected - audio.size))
    elif audio.size > expected:
        audio = audio[-expected:]

    loop = asyncio.get_event_loop()
    raw_scores = await loop.run_in_executor(None, ctx.detector._model.predict, audio)
    scored = {k: float(v) for k, v in (raw_scores or {}).items()}

    speaker_scores: Dict[str, float] = {}
    if ctx.verifier is not None:
        speaker_scores = await loop.run_in_executor(
            None, ctx.verifier.verify_with_scores, pcm
        )

    return web.json_response(
        {
            "ww_scores": scored,
            "speaker_scores": speaker_scores,
            "speaker_threshold": ctx.speaker_threshold,
        }
    )


async def handle_samples_zip(request: web.Request) -> web.Response:
    """GET /api/samples/zip?label=fp|tp|all - download a zip of selected samples."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        raise web.HTTPNotFound()
    label = request.query.get("label", "all")
    if label == "all":
        label = None
    zip_bytes = ctx.sample_store.export_zip(label=label)
    filename = f"wakeword_samples_{label or 'all'}.zip"
    return web.Response(
        body=zip_bytes,
        headers={
            "Content-Type": "application/zip",
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


async def handle_samples_fine_tune_zip(request: web.Request) -> web.Response:
    """GET /api/samples/fine-tune.zip - download tp/+fp/ structured zip for retraining."""
    ctx: WebServerContext = request.app["ctx"]
    if ctx.sample_store is None:
        raise web.HTTPNotFound()
    zip_bytes = ctx.sample_store.export_fine_tune_zip()
    return web.Response(
        body=zip_bytes,
        headers={
            "Content-Type": "application/zip",
            "Content-Disposition": 'attachment; filename="wakeword_fine_tune_dataset.zip"',
        },
    )


async def handle_events_ws(request: web.Request) -> web.WebSocketResponse:
    """Stream accepted wake-word detections as JSON over WebSocket.

    Protocol: on connect, the retained last event (if any) is sent immediately
    so reconnecting consumers (the HA integration) restore state. Then one JSON
    object per detection. Incoming text frames are answered with {"type":"pong"}
    for liveness checks.
    """
    ctx: WebServerContext = request.app["ctx"]
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)

    if ctx.event_bus is None:
        await ws.close(code=1011, message=b"event bus not configured")
        return ws

    queue = ctx.event_bus.subscribe()
    _LOGGER.info(
        "Events WS client connected (subscribers=%d)", ctx.event_bus.subscriber_count
    )
    try:
        if ctx.event_bus.last_event is not None:
            await ws.send_json(ctx.event_bus.last_event.to_dict())

        recv_task = asyncio.create_task(ws.receive())
        queue_task = asyncio.create_task(queue.get())
        pending = {recv_task, queue_task}
        while True:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            if recv_task in done:
                msg = recv_task.result()
                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
                if msg.type == WSMsgType.TEXT:
                    await ws.send_json({"type": "pong"})
                recv_task = asyncio.create_task(ws.receive())
                pending.add(recv_task)

            if queue_task in done:
                event = queue_task.result()
                await ws.send_json(event.to_dict())
                queue_task = asyncio.create_task(queue.get())
                pending.add(queue_task)
    finally:
        ctx.event_bus.unsubscribe(queue)
        for task in pending:
            task.cancel()
        _LOGGER.info(
            "Events WS client disconnected (subscribers=%d)",
            ctx.event_bus.subscriber_count,
        )
    return ws


def create_app(ctx: WebServerContext) -> web.Application:
    """Build the aiohttp app with all routes and static file handling."""
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["ctx"] = ctx

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/version", handle_version)
    app.router.add_get("/api/speakers", handle_speakers_list)
    app.router.add_post("/api/speakers/{name}/enroll", handle_speaker_enroll)
    app.router.add_delete("/api/speakers/{name}", handle_speaker_delete)
    app.router.add_post("/api/speakers/{name}/rename", handle_speaker_rename)
    app.router.add_post("/api/verify", handle_verify)
    app.router.add_get("/api/devices", handle_devices_list)
    app.router.add_post("/api/devices/{id}/capture", handle_device_capture)
    app.router.add_post("/api/devices/{id}/debug-claim", handle_device_claim)
    app.router.add_post("/api/devices/{id}/debug-release", handle_device_release)
    app.router.add_get("/api/samples", handle_samples_list)
    app.router.add_get("/api/samples/zip", handle_samples_zip)
    app.router.add_get("/api/samples/fine-tune.zip", handle_samples_fine_tune_zip)
    app.router.add_post("/api/samples/bulk-label", handle_samples_bulk_label)
    app.router.add_post("/api/samples/bulk-delete", handle_samples_bulk_delete)
    app.router.add_get("/api/samples/{id}/audio.wav", handle_sample_audio)
    app.router.add_post("/api/samples/{id}/label", handle_sample_label)
    app.router.add_delete("/api/samples/{id}", handle_sample_delete)
    app.router.add_get("/api/samples/{id}/retest", handle_sample_retest)
    app.router.add_get("/api/events", handle_events_ws)
    app.router.add_get("/api/debug-stream", handle_debug_ws)
    app.router.add_get("/api/score-stream", handle_score_stream_ws)

    if WEB_STATIC_DIR.exists():
        app.router.add_static("/static/", path=WEB_STATIC_DIR, name="static")

    return app


async def run_web_server(
    ctx: WebServerContext,
    host: str = "0.0.0.0",
    port: int = 10401,
) -> None:
    """Run the aiohttp server forever. Cancellation cleanly shuts down."""
    app = create_app(ctx)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    _LOGGER.info("Web UI listening on http://%s:%d/", host, port)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()
