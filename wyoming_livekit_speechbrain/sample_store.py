"""On-disk store for captured wake-word audio samples with metadata + retention."""

from __future__ import annotations

import io
import json
import logging
import secrets
import time
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

DEFAULT_SAMPLES_DIR = Path("/share/wakeword_samples")
DEFAULT_MAX_SAMPLES = 1000
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

# Sidecar JSON written next to every .wav. Keep schema flat and forward-compatible.
# Label is one of: None (unlabeled), "fp" (false positive), "tp" (true positive).
SCHEMA_VERSION = 1


@dataclass
class SampleMeta:
    """Metadata for a captured wake-word audio sample."""

    id: str
    timestamp: float
    keyword: Optional[str] = None
    score: Optional[float] = None
    speaker: Optional[str] = None
    accepted: Optional[bool] = None
    source: str = "auto"
    device_id: Optional[str] = None
    label: Optional[str] = None
    duration_seconds: Optional[float] = None
    size_bytes: int = 0
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "keyword": self.keyword,
            "score": self.score,
            "speaker": self.speaker,
            "accepted": self.accepted,
            "source": self.source,
            "device_id": self.device_id,
            "label": self.label,
            "duration_seconds": self.duration_seconds,
            "size_bytes": self.size_bytes,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleMeta":
        return cls(
            id=data["id"],
            timestamp=float(data.get("timestamp", 0.0)),
            keyword=data.get("keyword"),
            score=data.get("score"),
            speaker=data.get("speaker"),
            accepted=data.get("accepted"),
            source=data.get("source", "auto"),
            device_id=data.get("device_id"),
            label=data.get("label"),
            duration_seconds=data.get("duration_seconds"),
            size_bytes=int(data.get("size_bytes", 0)),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


def _wrap_pcm_to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM into a WAV-formatted byte string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _read_wav_pcm(path: Path) -> bytes:
    """Read a WAV file and return raw 16-bit mono PCM bytes (no header)."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != SAMPLE_WIDTH or wav.getnchannels() != 1:
            raise ValueError(
                f"{path} is not 16-bit mono PCM (got {wav.getsampwidth()*8}-bit, {wav.getnchannels()} ch)"
            )
        return wav.readframes(wav.getnframes())


@dataclass
class SampleStore:
    """Manages WAV + JSON sidecar files for captured wake-word samples.

    Layout::

        samples_dir/
          1734000123_456-sputnik-0.854.wav    # raw 16k mono int16 PCM in WAV container
          1734000123_456-sputnik-0.854.json   # sidecar metadata (SampleMeta as JSON)

    Retention is enforced lazily after every add(): newest N samples kept, rest evicted.
    Quota is per-source — `max_samples_per_source` lets a chatty source
    (e.g. `auto-rejected-speaker` in a noisy environment, 1000s/day) have a
    bigger bucket than rare-but-valuable sources (e.g. `auto` accepted triggers).
    Sources without an explicit override fall back to `max_samples`.
    """

    samples_dir: Path = field(default_factory=lambda: DEFAULT_SAMPLES_DIR)
    max_samples: int = DEFAULT_MAX_SAMPLES
    max_samples_per_source: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def _wav_path(self, sample_id: str) -> Path:
        return self.samples_dir / f"{sample_id}.wav"

    def _json_path(self, sample_id: str) -> Path:
        return self.samples_dir / f"{sample_id}.json"

    def _make_id(self, keyword: Optional[str], score: Optional[float]) -> str:
        ts_ms = int(time.time() * 1000)
        parts = [str(ts_ms)]
        if keyword:
            parts.append(_safe_token(keyword))
        if score is not None:
            parts.append(f"{score:.3f}")
        parts.append(secrets.token_hex(2))
        return "-".join(parts)

    def add(
        self,
        pcm: bytes,
        meta: SampleMeta,
    ) -> SampleMeta:
        """Persist a sample to disk. Returns the meta with id/size populated."""
        if not meta.id:
            meta.id = self._make_id(meta.keyword, meta.score)
        wav_bytes = _wrap_pcm_to_wav_bytes(pcm)
        meta.size_bytes = len(wav_bytes)
        meta.duration_seconds = round(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH), 3)

        wav_path = self._wav_path(meta.id)
        json_path = self._json_path(meta.id)
        wav_path.write_bytes(wav_bytes)
        json_path.write_text(json.dumps(meta.to_dict(), indent=2))

        self._enforce_retention()
        return meta

    def list(
        self,
        label: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SampleMeta]:
        """List samples newest-first. Optionally filter by label.

        Filter semantics:
          - None / "all" -> no filter
          - "fp" / "tp" -> match meta.label exactly
          - "unlabeled" -> match meta.label is None (samples that haven't been
            classified yet). Note: SampleMeta.label is None for unlabeled, not
            the string "unlabeled" — so a direct equality check would miss them.
        """
        out: List[SampleMeta] = []
        for json_path in sorted(
            self.samples_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                meta = SampleMeta.from_dict(json.loads(json_path.read_text()))
            except (json.JSONDecodeError, KeyError) as exc:
                _LOGGER.warning("Bad sample sidecar %s: %s", json_path, exc)
                continue
            if label is not None and label != "all":
                if label == "unlabeled":
                    if meta.label is not None:
                        continue
                elif meta.label != label:
                    continue
            out.append(meta)
            if limit is not None and len(out) >= limit:
                break
        return out

    def get(self, sample_id: str) -> Optional[SampleMeta]:
        json_path = self._json_path(sample_id)
        if not json_path.exists():
            return None
        try:
            return SampleMeta.from_dict(json.loads(json_path.read_text()))
        except (json.JSONDecodeError, KeyError) as exc:
            _LOGGER.warning("Bad sample sidecar %s: %s", json_path, exc)
            return None

    def get_wav_path(self, sample_id: str) -> Optional[Path]:
        wav_path = self._wav_path(sample_id)
        return wav_path if wav_path.exists() else None

    def get_pcm(self, sample_id: str) -> Optional[bytes]:
        """Read back a sample as raw int16 PCM (no WAV header)."""
        wav_path = self._wav_path(sample_id)
        if not wav_path.exists():
            return None
        return _read_wav_pcm(wav_path)

    def update_label(self, sample_id: str, label: Optional[str]) -> Optional[SampleMeta]:
        """Set / clear the FP/TP label on a sample. Valid: 'fp', 'tp', None."""
        if label not in (None, "fp", "tp"):
            raise ValueError(f"Invalid label: {label!r} (expected 'fp', 'tp', or None)")
        meta = self.get(sample_id)
        if meta is None:
            return None
        meta.label = label
        self._json_path(sample_id).write_text(json.dumps(meta.to_dict(), indent=2))
        return meta

    def delete(self, sample_id: str) -> bool:
        wav_path = self._wav_path(sample_id)
        json_path = self._json_path(sample_id)
        existed = wav_path.exists() or json_path.exists()
        wav_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        return existed

    def export_zip(self, label: Optional[str] = None) -> bytes:
        """Pack all samples (optionally filtered by label) into an in-memory ZIP."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for meta in self.list(label=label):
                wav_path = self._wav_path(meta.id)
                if wav_path.exists():
                    zf.write(wav_path, arcname=f"{meta.id}.wav")
                json_path = self._json_path(meta.id)
                if json_path.exists():
                    zf.write(json_path, arcname=f"{meta.id}.json")
        return buf.getvalue()

    def export_fine_tune_zip(self) -> bytes:
        """Pack a structured dataset for livekit-wakeword fine-tuning.

        Layout::

            tp/                  # True positives — these SHOULD trigger
              <id>.wav
              <id>.json
            fp/                  # False positives — these should NOT trigger
              <id>.wav
              <id>.json
            manifest.json        # Summary: counts, schema, generated_at
            README.md            # How to feed into wakeword_workspace training

        Drop both directories into your training workspace. TPs are
        additional positive utterances of the wake word in your acoustic
        environment; FPs become additional negative training data so the
        retrained classifier learns "these specific patterns are NOT the
        wake word".
        """
        buf = io.BytesIO()
        tp_samples = self.list(label="tp")
        fp_samples = self.list(label="fp")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for meta in tp_samples:
                wav_path = self._wav_path(meta.id)
                json_path = self._json_path(meta.id)
                if wav_path.exists():
                    zf.write(wav_path, arcname=f"tp/{meta.id}.wav")
                if json_path.exists():
                    zf.write(json_path, arcname=f"tp/{meta.id}.json")
            for meta in fp_samples:
                wav_path = self._wav_path(meta.id)
                json_path = self._json_path(meta.id)
                if wav_path.exists():
                    zf.write(wav_path, arcname=f"fp/{meta.id}.wav")
                if json_path.exists():
                    zf.write(json_path, arcname=f"fp/{meta.id}.json")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "generated_at": time.time(),
                "tp_count": len(tp_samples),
                "fp_count": len(fp_samples),
                "sample_rate": SAMPLE_RATE,
                "sample_width": SAMPLE_WIDTH,
                "channels": 1,
                "format": "wav_int16",
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr(
                "README.md",
                _FINE_TUNE_README.format(
                    tp_count=len(tp_samples),
                    fp_count=len(fp_samples),
                ),
            )
        return buf.getvalue()

    def total_size_bytes(self) -> int:
        total = 0
        for p in self.samples_dir.glob("*.wav"):
            try:
                total += p.stat().st_size
            except OSError:
                continue
        return total

    def _enforce_retention(self) -> None:
        """Evict oldest samples per source until each source is within its quota.

        Quota resolution:
          1. `self.max_samples_per_source[source]` if present and > 0  → use that
          2. else `self.max_samples` if > 0                            → use that
          3. else                                                       → no cap for this source

        Per-source quotas prevent a chatty source (e.g.
        `auto-rejected-speaker` at 1000s/day) from evicting rarer-but-more-
        valuable sources (e.g. `auto` accepted triggers for fine-tuning).
        """
        any_quota_active = self.max_samples > 0 or any(
            v > 0 for v in self.max_samples_per_source.values()
        )
        if not any_quota_active:
            return
        by_source: Dict[str, List[Path]] = {}
        for json_path in self.samples_dir.glob("*.json"):
            try:
                meta = SampleMeta.from_dict(json.loads(json_path.read_text()))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
            by_source.setdefault(meta.source, []).append(json_path)
        for source, source_paths in by_source.items():
            quota = self.max_samples_per_source.get(source, self.max_samples)
            if quota <= 0:
                continue
            source_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for stale in source_paths[quota:]:
                self.delete(stale.stem)


def _safe_token(s: str) -> str:
    """Sanitize a string for use as a filename token."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)[:32]


_FINE_TUNE_README = """# Wake-word fine-tune dataset

Generated by wyoming-livekit-speechbrain. Contains audio samples captured from
production deployment, labeled by the operator as either:

- `tp/` — True positives ({tp_count}): correctly-fired wake words. Add to
  training as additional **positive** examples (utterances in the actual
  acoustic environment / mic / room conditions of deployment).
- `fp/` — False positives ({fp_count}): triggers that should not have fired.
  Add to training as **negative** examples so the next model learns these
  specific patterns are NOT the wake word.

Each sample is a 2-second 16 kHz mono 16-bit PCM WAV plus a sidecar JSON with
timestamp, original WW score, speaker (if speaker verification was on at the
time), and source.

## Integrate into livekit-wakeword training

livekit-wakeword does NOT read this ZIP layout directly — it expects WAVs at
`output/<model>/positive_train/clip_NNNNNN.wav` (and matching `_test`,
`negative_train`, `negative_test` dirs), with filenames matching the regex
`^clip_\\d{{6}}\\.wav$`. Sidecars and `manifest.json` are ignored by training.

Rename and copy the WAVs into your workspace accordingly: pick the next free
clip index (so new clips don't collide with the TTS-synthesized
`clip_000000..clip_{{n_samples-1}}.wav` originals), put `tp/` clips into
`positive_*` and `fp/` clips into `negative_*`, split roughly 80/20
train/test.

Then from the workspace root re-run:

    livekit-wakeword augment configs/<model>.yaml
    livekit-wakeword train configs/<model>.yaml
    livekit-wakeword export configs/<model>.yaml
    livekit-wakeword eval configs/<model>.yaml   # verify new FPPH/recall

The fresh `output/<model>/<model>.onnx` has learned from your real-world
FPs/TPs. Drop it into `/share/wakewords/` on HA and restart the addon.
"""
