"""ESPHome YAML tools for Voice PE firmware."""

__version__ = "0.1.0"
__author__ = "Voice PE Team"
__license__ = "MIT"

from esphome_yaml_tools.validator import (
    validate,
    ValidationError,
    ESPHOME_SCHEMA,
    VOICE_ASSISTANT_SCHEMA,
    MICRO_WAKE_WORD_SCHEMA,
    LIGHT_SCHEMA,
    BINARY_SENSOR_SCHEMA,
    SWITCH_SCHEMA,
    SUBSTITUTIONS_SCHEMA,
)

__all__ = [
    "__version__",
    "validate",
    "ValidationError",
    "ESPHOME_SCHEMA",
    "VOICE_ASSISTANT_SCHEMA",
    "MICRO_WAKE_WORD_SCHEMA",
    "LIGHT_SCHEMA",
    "BINARY_SENSOR_SCHEMA",
    "SWITCH_SCHEMA",
    "SUBSTITUTIONS_SCHEMA",
]
