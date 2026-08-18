"""Comprehensive test suite for ESPHome YAML parser (TDD approach).

Tests cover:
- !include (simple and with vars)
- !secret
- !lambda
- !extend
- !remove
- !literal
- !env_var
- Substitution resolution (${var} and $var)
- << merge key
"""

import os
import pytest
from pathlib import Path
from textwrap import dedent

from esphome_yaml_tools.parser import (
    ESPHomeLoader,
    Lambda,
    Extend,
    Remove,
    load_yaml,
    resolve_substitutions,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Test: !include directive
# =============================================================================

class TestIncludeDirective:
    """Tests for !include directive - both simple and with vars."""

    def test_include_simple_file(self, tmp_path: Path):
        """!include other.yaml loads and merges file content."""
        # Create included file
        included = tmp_path / "wifi_config.yaml"
        included.write_text(dedent("""
            ssid: "MyNetwork"
            password: "secret123"
        """).strip())
        
        # Create main file
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            esphome:
              name: test
            wifi: !include wifi_config.yaml
        """).strip())
        
        result = load_yaml(main)
        
        assert result["wifi"]["ssid"] == "MyNetwork"
        assert result["wifi"]["password"] == "secret123"

    def test_include_with_vars(self, tmp_path: Path):
        """!include {file: x.yaml, vars: {name: test}} substitutes variables."""
        # Create included file with variable placeholder
        included = tmp_path / "button.yaml"
        included.write_text(dedent("""
            platform: template
            name: "${button_name}"
            id: "${button_id}"
        """).strip())
        
        # Create main file with vars
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            button:
              - !include
                file: button.yaml
                vars:
                  button_name: "My Button"
                  button_id: my_button
        """).strip())
        
        result = load_yaml(main)
        
        assert result["button"][0]["name"] == "My Button"
        assert result["button"][0]["id"] == "my_button"

    def test_include_relative_path(self, tmp_path: Path):
        """Include resolves paths relative to config directory."""
        subdir = tmp_path / "common"
        subdir.mkdir()
        
        common_wifi = subdir / "wifi.yaml"
        common_wifi.write_text(dedent("""
            ssid: "CommonNetwork"
        """).strip())
        
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            wifi: !include common/wifi.yaml
        """).strip())
        
        result = load_yaml(main)
        assert result["wifi"]["ssid"] == "CommonNetwork"

    def test_include_nested(self, tmp_path: Path):
        """Nested includes are resolved correctly."""
        # Create deeply nested structure
        common = tmp_path / "common"
        common.mkdir()
        
        wifi = common / "wifi.yaml"
        wifi.write_text(dedent("""
            ssid: !include ../network/ssid.yaml
        """).strip())
        
        network = tmp_path / "network"
        network.mkdir()
        
        ssid_file = network / "ssid.yaml"
        ssid_file.write_text('"NestedNetwork"')
        
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            wifi: !include common/wifi.yaml
        """).strip())
        
        result = load_yaml(main)
        assert result["wifi"]["ssid"] == "NestedNetwork"

    def test_include_missing_file_raises_error(self, tmp_path: Path):
        """Missing include file raises meaningful error."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            wifi: !include nonexistent.yaml
        """).strip())
        
        with pytest.raises(FileNotFoundError):
            load_yaml(main)


# =============================================================================
# Test: !secret directive
# =============================================================================

class TestSecretDirective:
    """Tests for !secret directive loading from secrets.yaml."""

    def test_secret_loads_value(self, tmp_path: Path):
        """!secret loads value from secrets.yaml."""
        secrets = tmp_path / "secrets.yaml"
        secrets.write_text(dedent("""
            wifi_password: supersecret123
        """).strip())
        
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            wifi:
              password: !secret wifi_password
        """).strip())
        
        result = load_yaml(main)
        assert result["wifi"]["password"] == "supersecret123"

    def test_secret_missing_key_raises_error(self, tmp_path: Path):
        """Missing secret key raises meaningful error."""
        secrets = tmp_path / "secrets.yaml"
        secrets.write_text(dedent("""
            wifi_password: secret
        """).strip())
        
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            api:
              key: !secret nonexistent_key
        """).strip())
        
        with pytest.raises(KeyError):
            load_yaml(main)

    def test_secret_missing_file_raises_error(self, tmp_path: Path):
        """Missing secrets.yaml raises meaningful error."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            wifi:
              password: !secret wifi_password
        """).strip())
        
        with pytest.raises(FileNotFoundError):
            load_yaml(main)


# =============================================================================
# Test: !lambda directive
# =============================================================================

class TestLambdaDirective:
    """Tests for !lambda directive creating Lambda objects."""

    def test_lambda_creates_object(self, tmp_path: Path):
        """!lambda creates Lambda object with code string."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            sensor:
              lambda: !lambda return x + 1;
        """).strip())
        
        result = load_yaml(main)
        
        assert isinstance(result["sensor"]["lambda"], Lambda)
        assert result["sensor"]["lambda"].code == "return x + 1;"

    def test_lambda_multiline(self, tmp_path: Path):
        """!lambda preserves multiline code."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent('''
            sensor:
              lambda: !lambda |-
                float value = id(temp_sensor).state;
                if (value > 25.0) {
                  return value * 1.1;
                }
                return value;
        ''').strip())
        
        result = load_yaml(main)
        
        assert isinstance(result["sensor"]["lambda"], Lambda)
        assert "float value" in result["sensor"]["lambda"].code
        assert "return value" in result["sensor"]["lambda"].code

    def test_lambda_str_representation(self, tmp_path: Path):
        """Lambda __str__ returns useful representation."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            sensor:
              lambda: !lambda return 42;
        """).strip())
        
        result = load_yaml(main)
        
        # Should have meaningful str representation
        assert "return 42" in str(result["sensor"]["lambda"])


# =============================================================================
# Test: !extend directive
# =============================================================================

class TestExtendDirective:
    """Tests for !extend directive creating Extend markers."""

    def test_extend_creates_marker(self, tmp_path: Path):
        """!extend creates Extend marker with component ID."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            light:
              - id: !extend led_ring
                effects:
                  - pulse:
        """).strip())
        
        result = load_yaml(main)
        
        assert isinstance(result["light"][0]["id"], Extend)
        assert result["light"][0]["id"].value == "led_ring"

    def test_extend_str_representation(self, tmp_path: Path):
        """Extend __str__ returns useful representation."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            light:
              - id: !extend my_light
        """).strip())
        
        result = load_yaml(main)
        
        assert "extend" in str(result["light"][0]["id"]).lower()
        assert "my_light" in str(result["light"][0]["id"])


# =============================================================================
# Test: !remove directive
# =============================================================================

class TestRemoveDirective:
    """Tests for !remove directive creating Remove markers."""

    def test_remove_creates_marker(self, tmp_path: Path):
        """!remove creates Remove marker."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            button:
              - id: boot_button
                disabled_by_default: !remove
        """).strip())
        
        result = load_yaml(main)
        
        assert isinstance(result["button"][0]["disabled_by_default"], Remove)

    def test_remove_with_value(self, tmp_path: Path):
        """!remove with value stores it for reference."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            button:
              - !remove boot_button
        """).strip())
        
        result = load_yaml(main)
        
        assert isinstance(result["button"][0], Remove)
        assert result["button"][0].value == "boot_button"

    def test_remove_str_representation(self, tmp_path: Path):
        """Remove __str__ returns useful representation."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            item: !remove test_item
        """).strip())
        
        result = load_yaml(main)
        
        assert "remove" in str(result["item"]).lower()


# =============================================================================
# Test: !literal directive
# =============================================================================

class TestLiteralDirective:
    """Tests for !literal directive for raw/unprocessed values."""

    def test_literal_preserves_value(self, tmp_path: Path):
        """!literal preserves value without processing."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            value: !literal "0x1234"
        """).strip())
        
        result = load_yaml(main)
        assert result["value"] == "0x1234"

    def test_literal_with_special_chars(self, tmp_path: Path):
        """!literal preserves special characters."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            pattern: !literal "${not_a_substitution}"
        """).strip())
        
        # Should NOT try to substitute
        result = load_yaml(main)
        assert result["pattern"] == "${not_a_substitution}"


# =============================================================================
# Test: !env_var directive
# =============================================================================

class TestEnvVarDirective:
    """Tests for !env_var directive reading environment variables."""

    def test_env_var_reads_value(self, tmp_path: Path, monkeypatch):
        """!env_var reads from environment."""
        monkeypatch.setenv("TEST_WIFI_SSID", "EnvNetwork")
        
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            wifi:
              ssid: !env_var TEST_WIFI_SSID
        """).strip())
        
        result = load_yaml(main)
        assert result["wifi"]["ssid"] == "EnvNetwork"

    def test_env_var_with_default(self, tmp_path: Path, monkeypatch):
        """!env_var uses default when env var not set."""
        # Ensure var is NOT set
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            wifi:
              ssid: !env_var NONEXISTENT_VAR DefaultSSID
        """).strip())
        
        result = load_yaml(main)
        assert result["wifi"]["ssid"] == "DefaultSSID"

    def test_env_var_missing_raises_error(self, tmp_path: Path, monkeypatch):
        """Missing env var without default raises error."""
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            api:
              key: !env_var NONEXISTENT_VAR
        """).strip())
        
        with pytest.raises(KeyError):
            load_yaml(main)


# =============================================================================
# Test: Substitution resolution
# =============================================================================

class TestSubstitutionResolution:
    """Tests for ${var} and $var substitution resolution."""

    def test_substitution_braced_syntax(self, tmp_path: Path):
        """${var} syntax is resolved from substitutions block."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            substitutions:
              device_name: voice-assistant
            
            esphome:
              name: ${device_name}
        """).strip())
        
        result = load_yaml(main)
        assert result["esphome"]["name"] == "voice-assistant"

    def test_substitution_bare_syntax(self, tmp_path: Path):
        """$var syntax (without braces) is also resolved."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            substitutions:
              device_name: voice-pe
            
            esphome:
              name: $device_name
        """).strip())
        
        result = load_yaml(main)
        assert result["esphome"]["name"] == "voice-pe"

    def test_substitution_in_string(self, tmp_path: Path):
        """Substitutions work within longer strings."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            substitutions:
              room: kitchen
            
            sensor:
              name: "Temperature ${room} sensor"
        """).strip())
        
        result = load_yaml(main)
        assert result["sensor"]["name"] == "Temperature kitchen sensor"

    def test_substitution_recursive(self, tmp_path: Path):
        """Substitutions can reference other substitutions."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            substitutions:
              device_base: voice
              device_suffix: pe
              device_name: ${device_base}-${device_suffix}
            
            esphome:
              name: ${device_name}
        """).strip())
        
        result = load_yaml(main)
        assert result["esphome"]["name"] == "voice-pe"

    def test_substitution_missing_raises_error(self, tmp_path: Path):
        """Missing substitution raises meaningful error."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            substitutions:
              device_name: test
            
            esphome:
              name: ${undefined_var}
        """).strip())
        
        with pytest.raises(KeyError):
            load_yaml(main)

    def test_substitution_in_nested_structure(self, tmp_path: Path):
        """Substitutions work in deeply nested structures."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            substitutions:
              pin: "GPIO5"
            
            sensor:
              - platform: adc
                pin: ${pin}
                filters:
                  - offset: 0.0
        """).strip())
        
        result = load_yaml(main)
        assert result["sensor"][0]["pin"] == "GPIO5"

    def test_no_substitutions_block(self, tmp_path: Path):
        """Config without substitutions block works fine."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            esphome:
              name: static-name
        """).strip())
        
        result = load_yaml(main)
        assert result["esphome"]["name"] == "static-name"


# =============================================================================
# Test: << merge key
# =============================================================================

class TestMergeKey:
    """Tests for << YAML merge key support."""

    def test_merge_key_basic(self, tmp_path: Path):
        """<< merge key merges anchor content."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            defaults: &sensor_defaults
              update_interval: 60s
              accuracy_decimals: 2
            
            sensor:
              - platform: template
                <<: *sensor_defaults
                name: "Temperature"
        """).strip())
        
        result = load_yaml(main)
        
        assert result["sensor"][0]["update_interval"] == "60s"
        assert result["sensor"][0]["accuracy_decimals"] == 2
        assert result["sensor"][0]["name"] == "Temperature"

    def test_merge_key_override(self, tmp_path: Path):
        """Local keys override merged content."""
        main = tmp_path / "main.yaml"
        main.write_text(dedent("""
            defaults: &defaults
              name: default_name
              value: 100
            
            item:
              <<: *defaults
              name: override_name
        """).strip())
        
        result = load_yaml(main)
        
        assert result["item"]["name"] == "override_name"
        assert result["item"]["value"] == 100


# =============================================================================
# Test: Integration tests with real fixtures
# =============================================================================

class TestIntegration:
    """Integration tests using fixture files."""

    def test_load_main_fixture(self):
        """Load and parse the main.yaml fixture."""
        result = load_yaml(FIXTURES_DIR / "main.yaml")
        
        # Check substitutions were applied
        assert result["esphome"]["name"] == "voice-assistant"
        assert result["esphome"]["friendly_name"] == "Voice PE"
        
        # Check include worked
        assert "wifi" in result
        
        # Check lambda was parsed
        assert isinstance(result["sensor"][0]["lambda"], Lambda)
        
        # Check extend was parsed
        assert isinstance(result["light"][0]["id"], Extend)

    def test_secrets_integration(self):
        """Secrets are loaded correctly from fixtures."""
        result = load_yaml(FIXTURES_DIR / "main.yaml")
        
        # The included wifi config references a secret
        # After full resolution, password should be the secret value
        assert result["wifi"]["password"] == "supersecretpassword123"


# =============================================================================
# Test: Edge cases and error handling
# =============================================================================

class TestEdgeCases:
    """Edge cases and error handling tests."""

    def test_empty_file(self, tmp_path: Path):
        """Empty YAML file returns None or empty dict."""
        main = tmp_path / "empty.yaml"
        main.write_text("")
        
        result = load_yaml(main)
        assert result is None or result == {}

    def test_scalar_only_file(self, tmp_path: Path):
        """File with only scalar value works."""
        main = tmp_path / "scalar.yaml"
        main.write_text('"just a string"')
        
        result = load_yaml(main)
        assert result == "just a string"

    def test_list_only_file(self, tmp_path: Path):
        """File with only list works."""
        main = tmp_path / "list.yaml"
        main.write_text(dedent("""
            - item1
            - item2
        """).strip())
        
        result = load_yaml(main)
        assert result == ["item1", "item2"]

    def test_invalid_yaml_raises_error(self, tmp_path: Path):
        """Invalid YAML syntax raises meaningful error."""
        main = tmp_path / "invalid.yaml"
        main.write_text(dedent("""
            key: value
              bad_indent: oops
        """).strip())
        
        with pytest.raises(Exception):  # yaml.YAMLError or subclass
            load_yaml(main)

    def test_circular_include_detected(self, tmp_path: Path):
        """Circular includes are detected and raise error."""
        file_a = tmp_path / "a.yaml"
        file_b = tmp_path / "b.yaml"
        
        file_a.write_text("content: !include b.yaml")
        file_b.write_text("content: !include a.yaml")
        
        with pytest.raises(RecursionError):
            load_yaml(file_a)


# =============================================================================
# Test: resolve_substitutions function directly
# =============================================================================

class TestResolveSubstitutionsFunction:
    """Direct tests for resolve_substitutions function."""

    def test_resolve_dict(self):
        """Resolves substitutions in dict values."""
        subs = {"name": "test", "value": "123"}
        data = {"key": "${name}", "num": "${value}"}
        
        result = resolve_substitutions(data, subs)
        
        assert result["key"] == "test"
        assert result["num"] == "123"

    def test_resolve_list(self):
        """Resolves substitutions in list items."""
        subs = {"item": "resolved"}
        data = ["${item}", "static", "${item}"]
        
        result = resolve_substitutions(data, subs)
        
        assert result == ["resolved", "static", "resolved"]

    def test_resolve_nested(self):
        """Resolves substitutions in nested structures."""
        subs = {"x": "X", "y": "Y"}
        data = {
            "level1": {
                "level2": "${x}",
                "list": ["${y}", {"deep": "${x}"}]
            }
        }
        
        result = resolve_substitutions(data, subs)
        
        assert result["level1"]["level2"] == "X"
        assert result["level1"]["list"][0] == "Y"
        assert result["level1"]["list"][1]["deep"] == "X"

    def test_resolve_preserves_non_strings(self):
        """Non-string values are preserved."""
        subs = {"x": "test"}
        data = {"int": 42, "float": 3.14, "bool": True, "none": None}
        
        result = resolve_substitutions(data, subs)
        
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None

    def test_resolve_lambda_not_modified(self):
        """Lambda objects are not modified by substitution."""
        subs = {"x": "test"}
        lam = Lambda("return ${x};")  # Should NOT substitute inside lambda
        data = {"lambda": lam}
        
        result = resolve_substitutions(data, subs)
        
        # Lambda code should be preserved as-is
        assert result["lambda"].code == "return ${x};"
