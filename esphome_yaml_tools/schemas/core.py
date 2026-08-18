"""Core ESPHome schemas (esphome block, substitutions, binary_sensor, switch).

Based on ESPHome config validation patterns.
"""

from typing import Any, Dict, List, Union
import voluptuous as vol

from esphome_yaml_tools.validator import (
    gpio_pin,
    Optional,
    Required,
    All,
    VolAny,
    Coerce,
    Schema,
    string,
    boolean,
    positive_int,
)

# =============================================================================
# ESPHome Block Schema
# =============================================================================

_ESPHOME_BLOCK_SCHEMA = Schema(
    {
        Required("name"): string,
        Required("platform"): string,
        Required("board"): string,
        Optional("min_version"): string,
        Optional("friendly_name"): string,
        Optional("comment"): string,
        Optional("on_boot"): list,
        Optional("on_shutdown"): list,
        Optional("on_loop"): list,
        Optional("platformio_options"): dict,
        Optional("includes"): list,
        Optional("libraries"): list,
    }
)

ESPHOME_SCHEMA = Schema(
    {
        Required("esphome"): _ESPHOME_BLOCK_SCHEMA,
    }
)


# =============================================================================
# Substitutions Block Schema
# =============================================================================

# Substitutions are simple string -> string mappings
_SUBSTITUTIONS_BLOCK_SCHEMA = Schema(
    {
        # Allow any string key with any string value
        Optional(str): VolAny(str, int, float, bool),
    }
)

SUBSTITUTIONS_SCHEMA = Schema(
    {
        Optional("substitutions"): _SUBSTITUTIONS_BLOCK_SCHEMA,
    }
)


# =============================================================================
# Binary Sensor Schema
# =============================================================================

_GPIO_PIN_SCHEMA = Schema(
    VolAny(
        positive_int,
        string,
        {
            Required("number"): VolAny(positive_int, string),
            Optional("inverted"): boolean,
            Optional("mode"): VolAny(string, dict),
        },
    )
)

_BINARY_SENSOR_ITEM_SCHEMA = Schema(
    {
        Required("platform"): string,
        Optional("id"): string,
        Optional("name"): string,
        Optional("pin"): _GPIO_PIN_SCHEMA,
        Optional("internal"): boolean,
        Optional("disabled_by_default"): boolean,
        Optional("device_class"): string,
        Optional("filters"): list,
        Optional("on_press"): list,
        Optional("on_release"): list,
        Optional("on_click"): list,
        Optional("on_double_click"): list,
        Optional("on_multi_click"): list,
        Optional("on_state"): list,
    },
    extra=vol.ALLOW_EXTRA,
)

BINARY_SENSOR_SCHEMA = Schema(
    {
        Optional("binary_sensor"): [_BINARY_SENSOR_ITEM_SCHEMA],
    }
)


# =============================================================================
# Switch Schema
# =============================================================================

_SWITCH_ITEM_SCHEMA = Schema(
    {
        Required("platform"): string,
        Optional("id"): string,
        Optional("name"): string,
        Optional("pin"): _GPIO_PIN_SCHEMA,
        Optional("internal"): boolean,
        Optional("disabled_by_default"): boolean,
        Optional("restore_mode"): string,
        Optional("on_turn_on"): list,
        Optional("on_turn_off"): list,
        Optional("optimistic"): boolean,
        Optional("assumed_state"): boolean,
        Optional("lambda"): VolAny(str, object),  # Lambda can be string or parsed object
        Optional("turn_on_action"): list,
        Optional("turn_off_action"): list,
    },
    extra=vol.ALLOW_EXTRA,
)

SWITCH_SCHEMA = Schema(
    {
        Optional("switch"): [_SWITCH_ITEM_SCHEMA],
    }
)
