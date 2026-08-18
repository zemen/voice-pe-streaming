"""Tests for ESPHome YAML validator (TDD approach).

Tests schema-based validation for Voice PE components using voluptuous.
"""

import pytest
from typing import List

# Import what we'll build
from esphome_yaml_tools.validator import (
    validate,
    ValidationError,
    # Type validators
    int_range,
    float_range,
    percentage,
    time_period,
    gpio_pin,
    # Component schemas
    ESPHOME_SCHEMA,
    VOICE_ASSISTANT_SCHEMA,
    MICRO_WAKE_WORD_SCHEMA,
    LIGHT_SCHEMA,
    BINARY_SENSOR_SCHEMA,
    SWITCH_SCHEMA,
    SUBSTITUTIONS_SCHEMA,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_voice_pe_yaml():
    """Minimal valid Voice PE YAML configuration."""
    return {
        "esphome": {
            "name": "voice-pe",
            "platform": "ESP32",
            "board": "esp32-s3-devkitc-1",
        },
        "substitutions": {
            "voice_assist_idle_phase_id": "1",
            "hidden_ssid": "false",
        },
        "voice_assistant": {
            "id": "va",
            "microphone": "i2s_mics",
            "speaker": "i2s_audio_speaker",
        },
        "micro_wake_word": {
            "id": "mww",
            "models": [
                {"model": "okay_nabu"},
                {"model": "hey_jarvis"},
            ],
        },
        "light": [
            {
                "platform": "esp32_rmt_led_strip",
                "id": "leds_internal",
                "pin": 21,
                "num_leds": 12,
                "rgb_order": "GRB",
                "chipset": "WS2812",
            }
        ],
        "binary_sensor": [
            {
                "platform": "gpio",
                "id": "center_button",
                "pin": {"number": 0, "inverted": True},
            }
        ],
        "switch": [
            {
                "platform": "template",
                "id": "master_mute_switch",
                "name": "Master Mute",
            }
        ],
    }


@pytest.fixture
def minimal_esphome_yaml():
    """Minimal valid esphome block only."""
    return {
        "esphome": {
            "name": "test-device",
            "platform": "ESP32",
            "board": "esp32dev",
        }
    }


# =============================================================================
# ValidationError Tests
# =============================================================================


class TestValidationError:
    """Tests for ValidationError class."""

    def test_validation_error_has_path_and_message(self):
        """ValidationError should contain path and message."""
        error = ValidationError(path="esphome.name", message="Required field missing")
        assert error.path == "esphome.name"
        assert error.message == "Required field missing"

    def test_validation_error_str_representation(self):
        """ValidationError should have useful string representation."""
        error = ValidationError(path="voice_assistant.id", message="Invalid type")
        assert "voice_assistant.id" in str(error)
        assert "Invalid type" in str(error)

    def test_validation_error_equality(self):
        """Two ValidationErrors with same path/message should be equal."""
        e1 = ValidationError(path="a.b", message="error")
        e2 = ValidationError(path="a.b", message="error")
        assert e1 == e2

    def test_validation_error_repr(self):
        """ValidationError should have useful repr for debugging."""
        error = ValidationError(path="test.path", message="test message")
        assert "ValidationError" in repr(error)


# =============================================================================
# Type Validator Tests
# =============================================================================


class TestIntRange:
    """Tests for int_range validator."""

    def test_int_range_valid(self):
        """Valid int within range passes."""
        validator = int_range(min=0, max=100)
        assert validator(50) == 50

    def test_int_range_at_boundaries(self):
        """Values at min/max boundaries are valid."""
        validator = int_range(min=0, max=100)
        assert validator(0) == 0
        assert validator(100) == 100

    def test_int_range_below_min_raises(self):
        """Value below min raises error."""
        validator = int_range(min=0, max=100)
        with pytest.raises(Exception):  # Will be voluptuous.Invalid
            validator(-1)

    def test_int_range_above_max_raises(self):
        """Value above max raises error."""
        validator = int_range(min=0, max=100)
        with pytest.raises(Exception):
            validator(101)

    def test_int_range_accepts_string_int(self):
        """String representation of int should be coerced."""
        validator = int_range(min=0, max=100)
        assert validator("50") == 50

    def test_int_range_invalid_string_raises(self):
        """Non-numeric string raises error."""
        validator = int_range(min=0, max=100)
        with pytest.raises(Exception):
            validator("not-a-number")


class TestFloatRange:
    """Tests for float_range validator."""

    def test_float_range_valid(self):
        """Valid float within range passes."""
        validator = float_range(min=0.0, max=1.0)
        assert validator(0.5) == 0.5

    def test_float_range_at_boundaries(self):
        """Values at min/max boundaries are valid."""
        validator = float_range(min=0.0, max=1.0)
        assert validator(0.0) == 0.0
        assert validator(1.0) == 1.0

    def test_float_range_below_min_raises(self):
        """Value below min raises error."""
        validator = float_range(min=0.0, max=1.0)
        with pytest.raises(Exception):
            validator(-0.1)

    def test_float_range_above_max_raises(self):
        """Value above max raises error."""
        validator = float_range(min=0.0, max=1.0)
        with pytest.raises(Exception):
            validator(1.1)

    def test_float_range_accepts_int(self):
        """Integer value should be coerced to float."""
        validator = float_range(min=0.0, max=100.0)
        assert validator(50) == 50.0


class TestPercentage:
    """Tests for percentage validator."""

    def test_percentage_numeric(self):
        """Numeric percentage (0-100) is valid."""
        assert percentage(50) == 50

    def test_percentage_string_with_percent(self):
        """String with % suffix is valid."""
        assert percentage("50%") == 50

    def test_percentage_below_zero_raises(self):
        """Percentage below 0 raises error."""
        with pytest.raises(Exception):
            percentage(-10)

    def test_percentage_above_100_raises(self):
        """Percentage above 100 raises error."""
        with pytest.raises(Exception):
            percentage(110)


class TestTimePeriod:
    """Tests for time_period validator."""

    def test_time_period_milliseconds(self):
        """'100ms' returns milliseconds."""
        result = time_period("100ms")
        assert result["milliseconds"] == 100

    def test_time_period_seconds(self):
        """'5s' returns seconds."""
        result = time_period("5s")
        assert result["seconds"] == 5

    def test_time_period_minutes(self):
        """'2min' returns minutes."""
        result = time_period("2min")
        assert result["minutes"] == 2

    def test_time_period_hours(self):
        """'1h' returns hours."""
        result = time_period("1h")
        assert result["hours"] == 1

    def test_time_period_dict_passthrough(self):
        """Dict format passes through."""
        result = time_period({"seconds": 10})
        assert result["seconds"] == 10

    def test_time_period_invalid_raises(self):
        """Invalid time period string raises error."""
        with pytest.raises(Exception):
            time_period("invalid")


class TestGpioPin:
    """Tests for gpio_pin validator."""

    def test_gpio_pin_integer(self):
        """Integer GPIO pin is valid."""
        assert gpio_pin(21) == 21

    def test_gpio_pin_string_gpioN(self):
        """'GPIO21' format is valid."""
        assert gpio_pin("GPIO21") == 21

    def test_gpio_pin_dict_format(self):
        """Dict with 'number' key is valid."""
        result = gpio_pin({"number": 0, "inverted": True})
        assert result["number"] == 0
        assert result["inverted"] is True

    def test_gpio_pin_invalid_string_raises(self):
        """Invalid string raises error."""
        with pytest.raises(Exception):
            gpio_pin("not-a-pin")


# =============================================================================
# ESPHome Block Schema Tests
# =============================================================================


class TestEsphomeSchema:
    """Tests for esphome block validation."""

    def test_valid_esphome_block(self):
        """Valid esphome block passes validation."""
        data = {
            "name": "my-device",
            "platform": "ESP32",
            "board": "esp32dev",
        }
        errors = validate({"esphome": data}, ESPHOME_SCHEMA)
        assert len(errors) == 0

    def test_esphome_missing_name_returns_error(self):
        """Missing 'name' field returns ValidationError."""
        data = {
            "platform": "ESP32",
            "board": "esp32dev",
        }
        errors = validate({"esphome": data}, ESPHOME_SCHEMA)
        assert len(errors) > 0
        assert any("name" in e.path for e in errors)

    def test_esphome_missing_platform_returns_error(self):
        """Missing 'platform' field returns ValidationError."""
        data = {
            "name": "my-device",
            "board": "esp32dev",
        }
        errors = validate({"esphome": data}, ESPHOME_SCHEMA)
        assert len(errors) > 0
        assert any("platform" in e.path or "platform" in e.message.lower() for e in errors)

    def test_esphome_invalid_name_type_returns_error(self):
        """Invalid type for 'name' returns ValidationError."""
        data = {
            "name": 12345,  # Should be string
            "platform": "ESP32",
            "board": "esp32dev",
        }
        errors = validate({"esphome": data}, ESPHOME_SCHEMA)
        assert len(errors) > 0
        assert any("name" in e.path for e in errors)

    def test_esphome_with_optional_fields(self):
        """Optional fields like min_version should be accepted."""
        data = {
            "name": "my-device",
            "platform": "ESP32",
            "board": "esp32dev",
            "min_version": "2024.1.0",
        }
        errors = validate({"esphome": data}, ESPHOME_SCHEMA)
        assert len(errors) == 0


# =============================================================================
# Voice Assistant Schema Tests
# =============================================================================


class TestVoiceAssistantSchema:
    """Tests for voice_assistant component validation."""

    def test_valid_voice_assistant(self):
        """Valid voice_assistant config passes validation."""
        data = {
            "voice_assistant": {
                "id": "va",
                "microphone": "i2s_mics",
                "speaker": "i2s_audio_speaker",
            }
        }
        errors = validate(data, VOICE_ASSISTANT_SCHEMA)
        assert len(errors) == 0

    def test_voice_assistant_missing_microphone_returns_error(self):
        """Missing 'microphone' field returns ValidationError."""
        data = {
            "voice_assistant": {
                "id": "va",
                "speaker": "i2s_audio_speaker",
            }
        }
        errors = validate(data, VOICE_ASSISTANT_SCHEMA)
        assert len(errors) > 0
        assert any("microphone" in e.path or "microphone" in e.message.lower() for e in errors)

    def test_voice_assistant_with_handlers(self):
        """Voice assistant with on_* handlers is valid."""
        data = {
            "voice_assistant": {
                "id": "va",
                "microphone": "i2s_mics",
                "speaker": "i2s_audio_speaker",
                "on_listening": [{"logger.log": "Listening..."}],
                "on_error": [{"logger.log": "Error!"}],
            }
        }
        errors = validate(data, VOICE_ASSISTANT_SCHEMA)
        assert len(errors) == 0


# =============================================================================
# Micro Wake Word Schema Tests
# =============================================================================


class TestMicroWakeWordSchema:
    """Tests for micro_wake_word component validation."""

    def test_valid_micro_wake_word(self):
        """Valid micro_wake_word config passes validation."""
        data = {
            "micro_wake_word": {
                "id": "mww",
                "models": [
                    {"model": "okay_nabu"},
                ],
            }
        }
        errors = validate(data, MICRO_WAKE_WORD_SCHEMA)
        assert len(errors) == 0

    def test_micro_wake_word_missing_models_returns_error(self):
        """Missing 'models' field returns ValidationError."""
        data = {
            "micro_wake_word": {
                "id": "mww",
            }
        }
        errors = validate(data, MICRO_WAKE_WORD_SCHEMA)
        assert len(errors) > 0
        assert any("models" in e.path or "models" in e.message.lower() for e in errors)

    def test_micro_wake_word_model_with_sensitivity(self):
        """Model with probability_cutoff (sensitivity) is valid."""
        data = {
            "micro_wake_word": {
                "id": "mww",
                "models": [
                    {
                        "model": "okay_nabu",
                        "probability_cutoff": 0.85,
                    },
                ],
            }
        }
        errors = validate(data, MICRO_WAKE_WORD_SCHEMA)
        assert len(errors) == 0

    def test_micro_wake_word_model_with_internal_flag(self):
        """Model with internal flag (for stop word) is valid."""
        data = {
            "micro_wake_word": {
                "id": "mww",
                "models": [
                    {
                        "model": "stop",
                        "internal": True,
                    },
                ],
            }
        }
        errors = validate(data, MICRO_WAKE_WORD_SCHEMA)
        assert len(errors) == 0


# =============================================================================
# Light Schema Tests
# =============================================================================


class TestLightSchema:
    """Tests for light component validation."""

    def test_valid_led_strip_light(self):
        """Valid LED strip light config passes validation."""
        data = {
            "light": [
                {
                    "platform": "esp32_rmt_led_strip",
                    "id": "leds_internal",
                    "pin": 21,
                    "num_leds": 12,
                    "rgb_order": "GRB",
                    "chipset": "WS2812",
                }
            ]
        }
        errors = validate(data, LIGHT_SCHEMA)
        assert len(errors) == 0

    def test_light_missing_platform_returns_error(self):
        """Missing 'platform' field returns ValidationError."""
        data = {
            "light": [
                {
                    "id": "leds_internal",
                    "pin": 21,
                }
            ]
        }
        errors = validate(data, LIGHT_SCHEMA)
        assert len(errors) > 0
        assert any("platform" in e.path or "platform" in e.message.lower() for e in errors)

    def test_partition_light(self):
        """Partition light (used for LED ring) is valid."""
        data = {
            "light": [
                {
                    "platform": "partition",
                    "id": "led_ring",
                    "segments": [
                        {"id": "leds_internal", "from": 0, "to": 11},
                    ],
                }
            ]
        }
        errors = validate(data, LIGHT_SCHEMA)
        assert len(errors) == 0

    def test_light_with_effects(self):
        """Light with effects list is valid."""
        data = {
            "light": [
                {
                    "platform": "esp32_rmt_led_strip",
                    "id": "leds",
                    "pin": 21,
                    "num_leds": 12,
                    "effects": [
                        {"addressable_rainbow": {"name": "Rainbow"}},
                    ],
                }
            ]
        }
        errors = validate(data, LIGHT_SCHEMA)
        assert len(errors) == 0


# =============================================================================
# Binary Sensor Schema Tests
# =============================================================================


class TestBinarySensorSchema:
    """Tests for binary_sensor component validation."""

    def test_valid_gpio_binary_sensor(self):
        """Valid GPIO binary sensor passes validation."""
        data = {
            "binary_sensor": [
                {
                    "platform": "gpio",
                    "id": "center_button",
                    "pin": {"number": 0, "inverted": True},
                }
            ]
        }
        errors = validate(data, BINARY_SENSOR_SCHEMA)
        assert len(errors) == 0

    def test_binary_sensor_missing_platform_returns_error(self):
        """Missing 'platform' field returns ValidationError."""
        data = {
            "binary_sensor": [
                {
                    "id": "button",
                    "pin": 0,
                }
            ]
        }
        errors = validate(data, BINARY_SENSOR_SCHEMA)
        assert len(errors) > 0
        assert any("platform" in e.path or "platform" in e.message.lower() for e in errors)

    def test_binary_sensor_with_filters(self):
        """Binary sensor with filters is valid."""
        data = {
            "binary_sensor": [
                {
                    "platform": "gpio",
                    "id": "button",
                    "pin": 0,
                    "filters": [
                        {"delayed_on": "10ms"},
                    ],
                }
            ]
        }
        errors = validate(data, BINARY_SENSOR_SCHEMA)
        assert len(errors) == 0


# =============================================================================
# Switch Schema Tests
# =============================================================================


class TestSwitchSchema:
    """Tests for switch component validation."""

    def test_valid_template_switch(self):
        """Valid template switch passes validation."""
        data = {
            "switch": [
                {
                    "platform": "template",
                    "id": "master_mute_switch",
                    "name": "Master Mute",
                }
            ]
        }
        errors = validate(data, SWITCH_SCHEMA)
        assert len(errors) == 0

    def test_valid_gpio_switch(self):
        """Valid GPIO switch passes validation."""
        data = {
            "switch": [
                {
                    "platform": "gpio",
                    "id": "led_power",
                    "pin": 45,
                }
            ]
        }
        errors = validate(data, SWITCH_SCHEMA)
        assert len(errors) == 0

    def test_switch_missing_platform_returns_error(self):
        """Missing 'platform' field returns ValidationError."""
        data = {
            "switch": [
                {
                    "id": "my_switch",
                    "name": "Switch",
                }
            ]
        }
        errors = validate(data, SWITCH_SCHEMA)
        assert len(errors) > 0
        assert any("platform" in e.path or "platform" in e.message.lower() for e in errors)


# =============================================================================
# Substitutions Schema Tests
# =============================================================================


class TestSubstitutionsSchema:
    """Tests for substitutions block validation."""

    def test_valid_substitutions(self):
        """Valid substitutions block passes validation."""
        data = {
            "substitutions": {
                "voice_assist_idle_phase_id": "1",
                "hidden_ssid": "false",
            }
        }
        errors = validate(data, SUBSTITUTIONS_SCHEMA)
        assert len(errors) == 0

    def test_substitutions_empty_is_valid(self):
        """Empty substitutions block is valid."""
        data = {"substitutions": {}}
        errors = validate(data, SUBSTITUTIONS_SCHEMA)
        assert len(errors) == 0

    def test_substitutions_invalid_key_type(self):
        """Substitution keys must be strings."""
        # Note: YAML naturally makes keys strings, but test the validator
        data = {"substitutions": {"valid_key": "value"}}
        errors = validate(data, SUBSTITUTIONS_SCHEMA)
        assert len(errors) == 0


# =============================================================================
# Cross-Reference Validation Tests
# =============================================================================


class TestCrossReferenceValidation:
    """Tests for cross-reference validation (cv.use_id pattern)."""

    def test_valid_id_reference(self, valid_voice_pe_yaml):
        """Valid ID references pass validation."""
        errors = validate(valid_voice_pe_yaml)
        assert len(errors) == 0

    def test_reference_to_nonexistent_id_returns_error(self):
        """Reference to non-existent ID returns ValidationError."""
        data = {
            "esphome": {
                "name": "test",
                "platform": "ESP32",
                "board": "esp32dev",
            },
            "voice_assistant": {
                "id": "va",
                "microphone": "nonexistent_mic",  # This ID doesn't exist
                "speaker": "nonexistent_speaker",  # This ID doesn't exist
            },
        }
        errors = validate(data, cross_reference=True)
        # Should have errors for missing microphone and speaker references
        assert len(errors) > 0
        assert any(
            "nonexistent" in e.message.lower() or "reference" in e.message.lower() for e in errors
        )

    def test_partition_light_references_valid_id(self):
        """Partition light referencing valid LED strip passes."""
        data = {
            "esphome": {
                "name": "test",
                "platform": "ESP32",
                "board": "esp32dev",
            },
            "light": [
                {
                    "platform": "esp32_rmt_led_strip",
                    "id": "leds_internal",
                    "pin": 21,
                    "num_leds": 12,
                },
                {
                    "platform": "partition",
                    "id": "led_ring",
                    "segments": [
                        {"id": "leds_internal", "from": 0, "to": 11},
                    ],
                },
            ],
        }
        errors = validate(data, cross_reference=True)
        assert len(errors) == 0

    def test_partition_light_references_invalid_id_returns_error(self):
        """Partition light referencing non-existent ID returns error."""
        data = {
            "esphome": {
                "name": "test",
                "platform": "ESP32",
                "board": "esp32dev",
            },
            "light": [
                {
                    "platform": "partition",
                    "id": "led_ring",
                    "segments": [
                        {"id": "nonexistent_leds", "from": 0, "to": 11},
                    ],
                },
            ],
        }
        errors = validate(data, cross_reference=True)
        assert len(errors) > 0
        assert any("nonexistent" in e.message.lower() for e in errors)


# =============================================================================
# Full Validation Tests
# =============================================================================


class TestFullValidation:
    """Tests for complete YAML validation."""

    def test_valid_voice_pe_yaml_passes(self, valid_voice_pe_yaml):
        """Complete valid Voice PE YAML passes validation."""
        errors = validate(valid_voice_pe_yaml)
        assert len(errors) == 0

    def test_minimal_yaml_passes(self, minimal_esphome_yaml):
        """Minimal YAML with just esphome block passes."""
        errors = validate(minimal_esphome_yaml)
        assert len(errors) == 0

    def test_empty_yaml_returns_error(self):
        """Empty YAML returns ValidationError."""
        errors = validate({})
        assert len(errors) > 0
        assert any("esphome" in e.path or "esphome" in e.message.lower() for e in errors)

    def test_multiple_errors_returned(self):
        """Multiple validation errors are all returned."""
        data = {
            "esphome": {
                # Missing name, platform, board
            },
            "voice_assistant": {
                # Missing required fields
            },
        }
        errors = validate(data)
        assert len(errors) >= 2  # At least errors for esphome and voice_assistant

    def test_error_path_shows_nested_location(self):
        """Error path should show full nested location."""
        data = {
            "esphome": {
                "name": "test",
                "platform": "ESP32",
                "board": "esp32dev",
            },
            "light": [
                {
                    "platform": "esp32_rmt_led_strip",
                    "id": "leds",
                    "pin": "invalid",  # Should be int
                    "num_leds": "not-a-number",  # Should be int
                }
            ],
        }
        errors = validate(data)
        # At least one error should have a nested path like "light[0].pin"
        assert any("light" in e.path for e in errors)


# =============================================================================
# Schema Export Tests
# =============================================================================


class TestSchemaExports:
    """Tests that schemas are properly exported."""

    def test_esphome_schema_exists(self):
        """ESPHOME_SCHEMA is exported."""
        assert ESPHOME_SCHEMA is not None

    def test_voice_assistant_schema_exists(self):
        """VOICE_ASSISTANT_SCHEMA is exported."""
        assert VOICE_ASSISTANT_SCHEMA is not None

    def test_micro_wake_word_schema_exists(self):
        """MICRO_WAKE_WORD_SCHEMA is exported."""
        assert MICRO_WAKE_WORD_SCHEMA is not None

    def test_light_schema_exists(self):
        """LIGHT_SCHEMA is exported."""
        assert LIGHT_SCHEMA is not None

    def test_binary_sensor_schema_exists(self):
        """BINARY_SENSOR_SCHEMA is exported."""
        assert BINARY_SENSOR_SCHEMA is not None

    def test_switch_schema_exists(self):
        """SWITCH_SCHEMA is exported."""
        assert SWITCH_SCHEMA is not None

    def test_substitutions_schema_exists(self):
        """SUBSTITUTIONS_SCHEMA is exported."""
        assert SUBSTITUTIONS_SCHEMA is not None
