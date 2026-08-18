"""ESPHome YAML schemas for Voice PE components.

This module provides voluptuous-based schemas for validating
ESPHome YAML configurations, specifically for Voice PE firmware.
"""

from esphome_yaml_tools.schemas.core import (
    ESPHOME_SCHEMA,
    SUBSTITUTIONS_SCHEMA,
    BINARY_SENSOR_SCHEMA,
    SWITCH_SCHEMA,
)
from esphome_yaml_tools.schemas.voice_assistant import VOICE_ASSISTANT_SCHEMA
from esphome_yaml_tools.schemas.micro_wake_word import MICRO_WAKE_WORD_SCHEMA
from esphome_yaml_tools.schemas.light import LIGHT_SCHEMA

__all__ = [
    "ESPHOME_SCHEMA",
    "SUBSTITUTIONS_SCHEMA",
    "BINARY_SENSOR_SCHEMA",
    "SWITCH_SCHEMA",
    "VOICE_ASSISTANT_SCHEMA",
    "MICRO_WAKE_WORD_SCHEMA",
    "LIGHT_SCHEMA",
]
