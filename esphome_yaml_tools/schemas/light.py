"""Light component schema.

Supports esp32_rmt_led_strip, partition, and other light platforms.
"""

import voluptuous as vol
from esphome_yaml_tools.validator import (
    Optional,
    Required,
    VolAny,
    Schema,
    string,
    boolean,
    positive_int,
    gpio_pin,
)

# =============================================================================
# LED Strip Schema (esp32_rmt_led_strip)
# =============================================================================

_LED_STRIP_SCHEMA = Schema(
    {
        Required("platform"): vol.In(["esp32_rmt_led_strip"]),
        Optional("id"): string,
        Optional("name"): string,
        Required("pin"): VolAny(positive_int, string),
        Required("num_leds"): positive_int,
        Optional("rgb_order"): string,
        Optional("chipset"): string,
        Optional("rmt_channel"): positive_int,
        Optional("max_refresh_rate"): string,
        Optional("internal"): boolean,
        Optional("effects"): list,
        Optional("default_transition_length"): VolAny(string, dict),
        Optional("restore_mode"): string,
        Optional("on_turn_on"): list,
        Optional("on_turn_off"): list,
        Optional("on_state"): list,
        Optional("power_supply"): string,  # ID reference
    },
    extra=vol.ALLOW_EXTRA,
)


# =============================================================================
# Partition Light Schema
# =============================================================================

_PARTITION_SEGMENT_SCHEMA = Schema(
    {
        Required("id"): string,  # ID reference to light
        Required("from"): positive_int,
        Required("to"): positive_int,
    }
)

_PARTITION_SCHEMA = Schema(
    {
        Required("platform"): vol.In(["partition"]),
        Optional("id"): string,
        Optional("name"): string,
        Required("segments"): [_PARTITION_SEGMENT_SCHEMA],
        Optional("internal"): boolean,
        Optional("effects"): list,
        Optional("default_transition_length"): VolAny(string, dict),
        Optional("on_turn_on"): list,
        Optional("on_turn_off"): list,
        Optional("on_state"): list,
    },
    extra=vol.ALLOW_EXTRA,
)


# =============================================================================
# Generic Light Schema (for other platforms)
# =============================================================================

_GENERIC_LIGHT_SCHEMA = Schema(
    {
        Required("platform"): string,
        Optional("id"): string,
        Optional("name"): string,
        Optional("internal"): boolean,
        Optional("effects"): list,
        Optional("default_transition_length"): VolAny(string, dict),
        Optional("on_turn_on"): list,
        Optional("on_turn_off"): list,
        Optional("on_state"): list,
    },
    extra=vol.ALLOW_EXTRA,
)


# =============================================================================
# Combined Light Schema
# =============================================================================


def _validate_light_item(value):
    """Validate a single light item based on platform."""
    if not isinstance(value, dict):
        raise vol.Invalid("Light must be a dictionary")

    platform = value.get("platform")
    if platform == "esp32_rmt_led_strip":
        return _LED_STRIP_SCHEMA(value)
    elif platform == "partition":
        return _PARTITION_SCHEMA(value)
    else:
        return _GENERIC_LIGHT_SCHEMA(value)


LIGHT_SCHEMA = Schema(
    {
        Optional("light"): [_validate_light_item],
    }
)
