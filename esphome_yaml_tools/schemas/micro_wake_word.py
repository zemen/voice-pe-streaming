"""Micro Wake Word component schema.

Based on ESPHome micro_wake_word component.
"""

import voluptuous as vol
from esphome_yaml_tools.validator import (
    Optional,
    Required,
    VolAny,
    Schema,
    string,
    boolean,
    float_range,
)

# =============================================================================
# Model Schema
# =============================================================================

_WAKE_WORD_MODEL_SCHEMA = Schema(
    {
        Required("model"): string,  # Model name or URL
        Optional("id"): string,
        Optional("probability_cutoff"): vol.Coerce(float),
        Optional("sliding_window_size"): vol.Coerce(int),
        Optional("internal"): boolean,
    },
    extra=vol.ALLOW_EXTRA,
)


# =============================================================================
# Micro Wake Word Schema
# =============================================================================

_MICRO_WAKE_WORD_BLOCK_SCHEMA = Schema(
    {
        Optional("id"): string,
        Required("models"): [_WAKE_WORD_MODEL_SCHEMA],
        Optional("microphone"): string,  # ID reference to microphone component
        Optional("on_wake_word_detected"): list,
        Optional("vad_threshold"): vol.Coerce(int),
        Optional("on_wake_word"): list,  # Deprecated but still used
    },
    extra=vol.ALLOW_EXTRA,
)

MICRO_WAKE_WORD_SCHEMA = Schema(
    {
        Optional("micro_wake_word"): _MICRO_WAKE_WORD_BLOCK_SCHEMA,
    }
)
