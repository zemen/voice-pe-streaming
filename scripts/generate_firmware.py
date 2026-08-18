#!/usr/bin/env python3
"""CLI entry point: regenerates voice_pe_streaming/voice-pe-streaming.yaml.

Usage:
    python scripts/generate_firmware.py             # use cached base
    python scripts/generate_firmware.py --refresh   # re-download base + verify SHA-256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esphome_yaml_tools.firmware_generator import PatchError, generate_firmware
from voice_pe_streaming import patches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download base from upstream URL and re-verify SHA-256",
    )
    args = parser.parse_args()

    try:
        result = generate_firmware(
            base=patches.BASE,
            patches=patches.PATCHES,
            output_path=patches.OUTPUT_PATH,
            refresh_base=args.refresh,
            header=patches.HEADER_COMMENT,
        )
    except PatchError as e:
        print(f"GENERATION FAILED:\n{e}", file=sys.stderr)
        return 1

    print(f"Wrote {result.output_path} ({len(result.yaml_text)} bytes)")
    print(f"  base revision: {result.base_revision}")
    print(f"  base sha256:   {result.base_sha256}")
    print(f"  patches:       {len(patches.PATCHES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
