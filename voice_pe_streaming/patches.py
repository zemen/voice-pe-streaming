"""Pinned base + production patch list re-export.

The patch list and feature catalog live in `features.py` (this is the patches-
only architecture: every change is a Patch, no overlay.yaml). This module is the
stable entry point for the CLI generator and for tests.

To bump upstream:
    1. Update BASE.revision with the new commit SHA.
    2. Fetch the new YAML and update BASE.sha256.
    3. Regenerate: `python scripts/generate_firmware.py`. Any patch whose
       `expected` no longer matches raises PatchError -> review the diff and
       update the patch in `features.py`.
    4. `pytest tests/test_firmware*.py` must still pass.
"""

from __future__ import annotations

from pathlib import Path

from esphome_yaml_tools.firmware_generator import BasePackage

from .features import PRODUCTION_PATCHES as _PRODUCTION_PATCHES

_ROOT = Path(__file__).parent

BASE = BasePackage(
    url="https://raw.githubusercontent.com/esphome/home-assistant-voice-pe/0579e7b9d8504264719c593474c85447253c9dc1/home-assistant-voice.yaml",
    revision="0579e7b9d8504264719c593474c85447253c9dc1",
    sha256="bede4eb905231c63be841313fce200dd1a85c1740e286acc8f9741c7e7d34e76",
    cache_path=_ROOT / "base_pinned.yaml",
)

OUTPUT_PATH = _ROOT / "voice-pe-streaming.yaml"

PATCHES = _PRODUCTION_PATCHES


HEADER_COMMENT = (
    "# AUTO-GENERATED -- DO NOT EDIT.\n"
    f"# Source: {BASE.url}\n"
    f"#         (revision {BASE.revision})\n"
    f"#         (sha256   {BASE.sha256})\n"
    "# Plus patches declared in voice_pe_streaming/features.py.\n"
    "# Regenerate with: python scripts/generate_firmware.py\n"
)
