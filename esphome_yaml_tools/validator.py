"""Validator for ESPHome YAML configurations.

Provides schema-based validation using voluptuous for Voice PE components.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional as OptionalType, Set, Union
import re

import voluptuous as vol
from voluptuous import Schema, Required, Optional, All, Coerce, Invalid

# Voluptuous Any renamed to avoid conflict with typing.Any
VolAny = vol.Any

# Re-export voluptuous primitives for schema modules
__all__ = [
    # Classes
    "ValidationError",
    # Functions
    "validate",
    # Type validators
    "int_range",
    "float_range",
    "percentage",
    "time_period",
    "gpio_pin",
    # Helpers
    "string",
    "boolean",
    "positive_int",
    # Voluptuous re-exports
    "Schema",
    "Required",
    "Optional",
    "All",
    "VolAny",
    "Coerce",
    # Schemas
    "ESPHOME_SCHEMA",
    "VOICE_ASSISTANT_SCHEMA",
    "MICRO_WAKE_WORD_SCHEMA",
    "LIGHT_SCHEMA",
    "BINARY_SENSOR_SCHEMA",
    "SWITCH_SCHEMA",
    "SUBSTITUTIONS_SCHEMA",
]


# =============================================================================
# ValidationError
# =============================================================================


@dataclass
class ValidationError:
    """Represents a validation error with path and message.

    Attributes:
        path: Dot-separated path to the invalid value (e.g., "esphome.name")
        message: Human-readable error message
    """

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"

    def __repr__(self) -> str:
        return f"ValidationError(path={self.path!r}, message={self.message!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValidationError):
            return NotImplemented
        return self.path == other.path and self.message == other.message


# =============================================================================
# Type Validators
# =============================================================================


def string(value: Any) -> str:
    """Validate string value (no coercion from other types)."""
    if isinstance(value, str):
        return value
    raise Invalid(f"Expected string, got {type(value).__name__}")


def boolean(value: Any) -> bool:
    """Validate and coerce to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "on", "1"):
            return True
        if value.lower() in ("false", "no", "off", "0"):
            return False
    if isinstance(value, int):
        return bool(value)
    raise Invalid(f"Expected boolean, got {type(value).__name__}")


def positive_int(value: Any) -> int:
    """Validate and coerce to positive integer."""
    if isinstance(value, bool):
        raise Invalid("Boolean is not a valid integer")
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            raise Invalid(f"Cannot convert '{value}' to integer")
    if isinstance(value, float):
        if value != int(value):
            raise Invalid(f"Expected integer, got float {value}")
        value = int(value)
    if not isinstance(value, int):
        raise Invalid(f"Expected integer, got {type(value).__name__}")
    if value < 0:
        raise Invalid(f"Expected positive integer, got {value}")
    return value


def int_range(min: OptionalType[int] = None, max: OptionalType[int] = None) -> Callable[[Any], int]:
    """Create a validator for integers within a range.

    Args:
        min: Minimum value (inclusive)
        max: Maximum value (inclusive)

    Returns:
        Validator function that returns the int value
    """

    def validator(value: Any) -> int:
        # Coerce string to int
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise Invalid(f"Cannot convert '{value}' to integer")

        if not isinstance(value, int) or isinstance(value, bool):
            raise Invalid(f"Expected integer, got {type(value).__name__}")

        if min is not None and value < min:
            raise Invalid(f"Value {value} is below minimum {min}")
        if max is not None and value > max:
            raise Invalid(f"Value {value} is above maximum {max}")

        return value

    return validator


def float_range(
    min: OptionalType[float] = None, max: OptionalType[float] = None
) -> Callable[[Any], float]:
    """Create a validator for floats within a range.

    Args:
        min: Minimum value (inclusive)
        max: Maximum value (inclusive)

    Returns:
        Validator function that returns the float value
    """

    def validator(value: Any) -> float:
        # Coerce to float
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise Invalid(f"Cannot convert '{value}' to float")
        elif isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        elif not isinstance(value, float):
            raise Invalid(f"Expected float, got {type(value).__name__}")

        if min is not None and value < min:
            raise Invalid(f"Value {value} is below minimum {min}")
        if max is not None and value > max:
            raise Invalid(f"Value {value} is above maximum {max}")

        return value

    return validator


def percentage(value: Any) -> int:
    """Validate a percentage value (0-100 or "X%").

    Args:
        value: Integer 0-100, float 0-100, or string "X%"

    Returns:
        Integer percentage value
    """
    if isinstance(value, str):
        if value.endswith("%"):
            try:
                value = int(value[:-1])
            except ValueError:
                raise Invalid(f"Invalid percentage string: {value}")
        else:
            try:
                value = int(value)
            except ValueError:
                raise Invalid(f"Cannot convert '{value}' to percentage")
    elif isinstance(value, float):
        value = int(value)
    elif not isinstance(value, int) or isinstance(value, bool):
        raise Invalid(f"Expected percentage, got {type(value).__name__}")

    if value < 0:
        raise Invalid(f"Percentage {value} cannot be negative")
    if value > 100:
        raise Invalid(f"Percentage {value} cannot exceed 100")

    return value


# Time period regex patterns
_TIME_PERIOD_PATTERNS = [
    (r"^(\d+(?:\.\d+)?)\s*ms$", "milliseconds"),
    (r"^(\d+(?:\.\d+)?)\s*s$", "seconds"),
    (r"^(\d+(?:\.\d+)?)\s*min$", "minutes"),
    (r"^(\d+(?:\.\d+)?)\s*h$", "hours"),
    (r"^(\d+(?:\.\d+)?)\s*d$", "days"),
]


def time_period(value: Any) -> Dict[str, Union[int, float]]:
    """Validate a time period value.

    Args:
        value: String like "100ms", "5s", "2min", "1h" or dict like {"seconds": 10}

    Returns:
        Dict with appropriate unit key and value
    """
    if isinstance(value, dict):
        # Validate dict has a valid time key
        valid_keys = {"milliseconds", "seconds", "minutes", "hours", "days"}
        if not any(k in value for k in valid_keys):
            raise Invalid(f"Time period dict must have one of: {valid_keys}")
        return value

    if not isinstance(value, str):
        raise Invalid(f"Expected time period string or dict, got {type(value).__name__}")

    value = value.strip()

    for pattern, unit in _TIME_PERIOD_PATTERNS:
        match = re.match(pattern, value, re.IGNORECASE)
        if match:
            num = float(match.group(1))
            # Convert to int if whole number
            if num == int(num):
                num = int(num)
            return {unit: num}

    raise Invalid(f"Invalid time period format: {value}")


def gpio_pin(value: Any) -> Union[int, Dict[str, Any]]:
    """Validate a GPIO pin value.

    Args:
        value: Integer pin number, "GPION" string, or dict with "number" key

    Returns:
        Integer pin number or dict with pin config
    """
    if isinstance(value, dict):
        if "number" not in value:
            raise Invalid("GPIO pin dict must have 'number' key")
        # Recursively validate the number
        number = gpio_pin(value["number"])
        result = dict(value)
        result["number"] = number
        return result

    if isinstance(value, int) and not isinstance(value, bool):
        return value

    if isinstance(value, str):
        # Handle "GPION" format
        match = re.match(r"^GPIO(\d+)$", value, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Try direct conversion
        try:
            return int(value)
        except ValueError:
            raise Invalid(f"Invalid GPIO pin: {value}")

    raise Invalid(f"Expected GPIO pin, got {type(value).__name__}")


# =============================================================================
# Schema Building Helpers
# =============================================================================


def _path_to_string(path: List[Any]) -> str:
    """Convert voluptuous path to dot-separated string."""
    parts = []
    for p in path:
        if isinstance(p, int):
            parts.append(f"[{p}]")
        else:
            parts.append(str(p))

    result = ""
    for part in parts:
        if part.startswith("["):
            result += part
        elif result:
            result += "." + part
        else:
            result = part

    return result


def _collect_ids(data: Any, path: str = "", in_segments: bool = False) -> Dict[str, str]:
    """Recursively collect all IDs from a YAML structure.

    Returns:
        Dict mapping ID to component type
    """
    ids = {}

    if isinstance(data, dict):
        if "id" in data and not in_segments:
            id_val = data["id"]
            if isinstance(id_val, str):
                component = path.split(".")[0] if path else "unknown"
                ids[id_val] = component

        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            is_segments = key == "segments"
            ids.update(_collect_ids(value, new_path, in_segments=is_segments))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]"
            ids.update(_collect_ids(item, new_path, in_segments=in_segments))

    return ids


# ID reference fields that point to other components
_ID_REFERENCE_FIELDS = {
    "microphone",
    "speaker",
    "media_player",
    "power_supply",
}

# Partition segment id field
_SEGMENT_ID_FIELDS = {"id"}


def _collect_references(data: Any, path: str = "") -> List[tuple]:
    """Recursively collect all ID references from a YAML structure.

    Returns:
        List of (path, referenced_id) tuples
    """
    refs = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key

            # Check if this is a known ID reference field
            if key in _ID_REFERENCE_FIELDS and isinstance(value, str):
                refs.append((new_path, value))

            # Check for partition segment id references
            elif key == "segments" and isinstance(value, list):
                for i, segment in enumerate(value):
                    if isinstance(segment, dict) and "id" in segment:
                        seg_path = f"{new_path}[{i}].id"
                        refs.append((seg_path, segment["id"]))

            else:
                refs.extend(_collect_references(value, new_path))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]"
            refs.extend(_collect_references(item, new_path))

    return refs


# =============================================================================
# Main Validation Function
# =============================================================================


def validate(
    data: Dict[str, Any],
    schema: OptionalType[Schema] = None,
    cross_reference: bool = False,
) -> List[ValidationError]:
    """Validate YAML data against schema(s).

    Args:
        data: Parsed YAML data to validate
        schema: Specific schema to validate against. If None, validates
                against all known component schemas.
        cross_reference: If True, also validate ID references

    Returns:
        List of ValidationError objects (empty if valid)
    """
    errors: List[ValidationError] = []

    # If no specific schema, use combined validation
    if schema is None:
        # Must have esphome block
        if "esphome" not in data:
            errors.append(
                ValidationError(path="esphome", message="Required 'esphome' block is missing")
            )

        # Validate each component that exists
        component_schemas = {
            "esphome": ESPHOME_SCHEMA,
            "substitutions": SUBSTITUTIONS_SCHEMA,
            "voice_assistant": VOICE_ASSISTANT_SCHEMA,
            "micro_wake_word": MICRO_WAKE_WORD_SCHEMA,
            "light": LIGHT_SCHEMA,
            "binary_sensor": BINARY_SENSOR_SCHEMA,
            "switch": SWITCH_SCHEMA,
        }

        for component, comp_schema in component_schemas.items():
            if component in data:
                comp_data = {component: data[component]}
                try:
                    comp_schema(comp_data)
                except vol.MultipleInvalid as e:
                    for err in e.errors:
                        path = _path_to_string(err.path)
                        errors.append(ValidationError(path=path, message=str(err.msg)))
                except vol.Invalid as e:
                    path = _path_to_string(e.path)
                    errors.append(ValidationError(path=path, message=str(e.msg)))
    else:
        # Validate against specific schema
        try:
            schema(data)
        except vol.MultipleInvalid as e:
            for err in e.errors:
                path = _path_to_string(err.path)
                errors.append(ValidationError(path=path, message=str(err.msg)))
        except vol.Invalid as e:
            path = _path_to_string(e.path)
            errors.append(ValidationError(path=path, message=str(e.msg)))

    # Cross-reference validation
    if cross_reference:
        defined_ids = _collect_ids(data)
        references = _collect_references(data)

        for ref_path, ref_id in references:
            if ref_id not in defined_ids:
                errors.append(
                    ValidationError(
                        path=ref_path, message=f"Reference to non-existent ID: '{ref_id}'"
                    )
                )

    return errors


# =============================================================================
# Import schemas (circular import resolution)
# =============================================================================


# Deferred imports to avoid circular dependencies
def _import_schemas():
    """Import schemas after module initialization."""
    global ESPHOME_SCHEMA, VOICE_ASSISTANT_SCHEMA, MICRO_WAKE_WORD_SCHEMA
    global LIGHT_SCHEMA, BINARY_SENSOR_SCHEMA, SWITCH_SCHEMA, SUBSTITUTIONS_SCHEMA

    from esphome_yaml_tools.schemas.core import (
        ESPHOME_SCHEMA as _ESPHOME,
        SUBSTITUTIONS_SCHEMA as _SUBS,
        BINARY_SENSOR_SCHEMA as _BINARY,
        SWITCH_SCHEMA as _SWITCH,
    )
    from esphome_yaml_tools.schemas.voice_assistant import (
        VOICE_ASSISTANT_SCHEMA as _VA,
    )
    from esphome_yaml_tools.schemas.micro_wake_word import (
        MICRO_WAKE_WORD_SCHEMA as _MWW,
    )
    from esphome_yaml_tools.schemas.light import (
        LIGHT_SCHEMA as _LIGHT,
    )

    ESPHOME_SCHEMA = _ESPHOME
    VOICE_ASSISTANT_SCHEMA = _VA
    MICRO_WAKE_WORD_SCHEMA = _MWW
    LIGHT_SCHEMA = _LIGHT
    BINARY_SENSOR_SCHEMA = _BINARY
    SWITCH_SCHEMA = _SWITCH
    SUBSTITUTIONS_SCHEMA = _SUBS


# Placeholder schemas (will be replaced by _import_schemas)
ESPHOME_SCHEMA: Schema = None  # type: ignore
VOICE_ASSISTANT_SCHEMA: Schema = None  # type: ignore
MICRO_WAKE_WORD_SCHEMA: Schema = None  # type: ignore
LIGHT_SCHEMA: Schema = None  # type: ignore
BINARY_SENSOR_SCHEMA: Schema = None  # type: ignore
SWITCH_SCHEMA: Schema = None  # type: ignore
SUBSTITUTIONS_SCHEMA: Schema = None  # type: ignore

# Import schemas at module load time
_import_schemas()
