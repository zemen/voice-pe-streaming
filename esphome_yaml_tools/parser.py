"""ESPHome YAML parser with support for custom directives.

Implements custom YAML tags:
- !include - Include other YAML files
- !secret - Load secrets from secrets.yaml
- !lambda - C++ lambda code strings
- !extend - Extend marker for package modifications
- !remove - Remove marker for package modifications
- !literal - Raw/unprocessed values
- !env_var - Environment variable loading

Also handles:
- ${var} and $var substitution syntax
- << YAML merge key
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


# Constants
SECRET_YAML = "secrets.yaml"

# Regex for variable substitution: matches ${var} and $var
# ${var} - braced form
# $var - bare form (only word characters)
VARIABLE_PATTERN = re.compile(r"\$(?:\{([^}]+)\}|([a-zA-Z_][a-zA-Z0-9_]*))")


# =============================================================================
# Marker classes for special directives
# =============================================================================


@dataclass
class Lambda:
    """Marker for !lambda directive containing C++ code."""

    code: str

    def __str__(self) -> str:
        return f"!lambda {self.code}"

    def __repr__(self) -> str:
        return f"Lambda({self.code!r})"


@dataclass
class Extend:
    """Marker for !extend directive used in package modifications."""

    value: str

    def __str__(self) -> str:
        return f"!extend {self.value}"

    def __repr__(self) -> str:
        return f"Extend({self.value!r})"


@dataclass
class Remove:
    """Marker for !remove directive used in package modifications."""

    value: str | None = None

    def __str__(self) -> str:
        if self.value:
            return f"!remove {self.value}"
        return "!remove"

    def __repr__(self) -> str:
        return f"Remove({self.value!r})"


# =============================================================================
# ESPHome YAML Loader
# =============================================================================


class ESPHomeLoader(yaml.SafeLoader):
    """Custom YAML loader supporting ESPHome directives.

    Extends SafeLoader with custom tag constructors for ESPHome-specific
    YAML directives like !include, !secret, !lambda, etc.
    """

    # Track include stack for circular reference detection
    _include_stack: list[Path] = []

    def __init__(
        self,
        stream,
        *,
        config_path: Path | None = None,
        yaml_loader: Callable[[Path], Any] | None = None,
    ):
        """Initialize the loader.

        Args:
            stream: YAML content stream (file or string)
            config_path: Path to the config file (for relative path resolution)
            yaml_loader: Callable to load included YAML files (for recursion)
        """
        super().__init__(stream)
        self._config_path = config_path or Path(".")
        self._yaml_loader = yaml_loader or _default_yaml_loader

    def _rel_path(self, *args: str) -> Path:
        """Resolve path relative to config directory."""
        return self._config_path.parent / Path(*args)

    # -------------------------------------------------------------------------
    # !include directive
    # -------------------------------------------------------------------------

    def construct_include(self, node: yaml.Node) -> Any:
        """Construct !include directive.

        Supports two forms:
        - !include filename.yaml
        - !include {file: filename.yaml, vars: {key: value}}
        """
        if isinstance(node, yaml.MappingNode):
            # Dict form: !include {file: x.yaml, vars: {...}}
            # Manually extract file and vars to handle nested mappings correctly
            file_path = None
            vars_dict = {}

            for key_node, value_node in node.value:
                key = self.construct_scalar(key_node)
                if key == "file":
                    file_path = self.construct_scalar(value_node)
                elif key == "vars":
                    # Construct the vars mapping with deep=True to get nested values
                    vars_dict = self.construct_mapping(value_node, deep=True)

            if file_path is None:
                raise yaml.MarkedYAMLError(
                    "!include dict form must have 'file' key", node.start_mark
                )
        else:
            # Simple form: !include filename.yaml
            file_path = self.construct_scalar(node)
            vars_dict = {}

        # Resolve the file path
        full_path = self._rel_path(file_path).resolve()

        # Check for circular includes
        if full_path in ESPHomeLoader._include_stack:
            raise RecursionError(
                f"Circular include detected: {full_path}\n"
                f"Include stack: {' -> '.join(str(p) for p in ESPHomeLoader._include_stack)}"
            )

        # Check file exists
        if not full_path.exists():
            raise FileNotFoundError(f"Include file not found: {full_path}")

        # Load the included file with variable substitution
        try:
            ESPHomeLoader._include_stack.append(full_path)
            # If vars are provided, substitute them in the file content BEFORE parsing
            if vars_dict:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Substitute variables in the raw content
                content = _substitute_string(content, vars_dict, 0)
                # Parse the substituted content
                loader = ESPHomeLoader(
                    content,
                    config_path=full_path,
                    yaml_loader=self._yaml_loader,
                )
                try:
                    result = loader.get_single_data()
                finally:
                    loader.dispose()
            else:
                result = self._yaml_loader(full_path)
        finally:
            ESPHomeLoader._include_stack.pop()

        return result

    # -------------------------------------------------------------------------
    # !secret directive
    # -------------------------------------------------------------------------

    def construct_secret(self, node: yaml.Node) -> str:
        """Construct !secret directive.

        Loads the secret value from secrets.yaml in the config directory.
        """
        secret_key = self.construct_scalar(node)

        # Load secrets.yaml
        secrets_path = self._rel_path(SECRET_YAML)
        if not secrets_path.exists():
            raise FileNotFoundError(f"secrets.yaml not found at {secrets_path}")

        secrets = self._yaml_loader(secrets_path)

        if secret_key not in secrets:
            raise KeyError(f"Secret '{secret_key}' not found in {SECRET_YAML}")

        return secrets[secret_key]

    # -------------------------------------------------------------------------
    # !lambda directive
    # -------------------------------------------------------------------------

    def construct_lambda(self, node: yaml.Node) -> Lambda:
        """Construct !lambda directive.

        Returns a Lambda object containing the C++ code string.
        """
        code = self.construct_scalar(node)
        return Lambda(code)

    # -------------------------------------------------------------------------
    # !extend directive
    # -------------------------------------------------------------------------

    def construct_extend(self, node: yaml.Node) -> Extend:
        """Construct !extend directive.

        Returns an Extend marker with the component ID.
        """
        value = self.construct_scalar(node)
        return Extend(value)

    # -------------------------------------------------------------------------
    # !remove directive
    # -------------------------------------------------------------------------

    def construct_remove(self, node: yaml.Node) -> Remove:
        """Construct !remove directive.

        Returns a Remove marker, optionally with a value.
        """
        value = self.construct_scalar(node)
        return Remove(value if value else None)

    # -------------------------------------------------------------------------
    # !literal directive
    # -------------------------------------------------------------------------

    def construct_literal(self, node: yaml.Node) -> str:
        """Construct !literal directive.

        Returns the raw value without any processing.
        """
        return self.construct_scalar(node)

    # -------------------------------------------------------------------------
    # !env_var directive
    # -------------------------------------------------------------------------

    def construct_env_var(self, node: yaml.Node) -> str:
        """Construct !env_var directive.

        Reads value from environment variable. Supports default value:
        !env_var VAR_NAME default_value
        """
        args = self.construct_scalar(node).split(maxsplit=1)
        var_name = args[0]

        if len(args) > 1:
            # Default value provided
            default_value = args[1]
            return os.environ.get(var_name, default_value)

        # No default - must exist
        if var_name not in os.environ:
            raise KeyError(f"Environment variable '{var_name}' not set")

        return os.environ[var_name]


# Register all custom constructors
def _register_constructors(loader_class: type) -> None:
    """Register all ESPHome-specific constructors."""
    loader_class.add_constructor("!include", loader_class.construct_include)
    loader_class.add_constructor("!secret", loader_class.construct_secret)
    loader_class.add_constructor("!lambda", loader_class.construct_lambda)
    loader_class.add_constructor("!extend", loader_class.construct_extend)
    loader_class.add_constructor("!remove", loader_class.construct_remove)
    loader_class.add_constructor("!literal", loader_class.construct_literal)
    loader_class.add_constructor("!env_var", loader_class.construct_env_var)


_register_constructors(ESPHomeLoader)


# =============================================================================
# Substitution resolution
# =============================================================================


def resolve_substitutions(
    data: Any,
    substitutions: dict[str, str],
    *,
    _recursive_depth: int = 0,
) -> Any:
    """Resolve ${var} and $var substitutions in data.

    Args:
        data: The data structure to process (dict, list, str, or scalar)
        substitutions: Dictionary of variable name -> value mappings
        _recursive_depth: Internal counter to prevent infinite recursion

    Returns:
        Data with all substitutions resolved

    Raises:
        KeyError: If a referenced variable is not in substitutions
        RecursionError: If recursive substitutions exceed max depth
    """
    max_depth = 100
    if _recursive_depth > max_depth:
        raise RecursionError(f"Substitution recursion exceeded {max_depth} levels")

    if isinstance(data, dict):
        return {
            key: resolve_substitutions(value, substitutions, _recursive_depth=_recursive_depth)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            resolve_substitutions(item, substitutions, _recursive_depth=_recursive_depth)
            for item in data
        ]

    if isinstance(data, str):
        return _substitute_string(data, substitutions, _recursive_depth)

    # Preserve other types (int, float, bool, None, Lambda, Extend, Remove)
    return data


def _substitute_string(
    value: str,
    substitutions: dict[str, str],
    recursive_depth: int,
) -> str:
    """Substitute variables in a string value.

    Handles both ${var} and $var syntax.
    """
    if "$" not in value:
        return value

    def replace_match(match: re.Match) -> str:
        # Group 1 is braced form ${var}, Group 2 is bare form $var
        var_name = match.group(1) or match.group(2)

        if var_name not in substitutions:
            raise KeyError(f"Substitution variable '{var_name}' not found")

        replacement = substitutions[var_name]

        # Recursively resolve if the replacement contains substitutions
        if isinstance(replacement, str) and "$" in replacement:
            replacement = _substitute_string(replacement, substitutions, recursive_depth + 1)

        return replacement

    result = VARIABLE_PATTERN.sub(replace_match, value)
    return result


# =============================================================================
# Main API
# =============================================================================


def _default_yaml_loader(path: Path) -> Any:
    """Default YAML loader function for recursion."""
    return load_yaml(path)


def load_yaml(path: Path | str) -> Any:
    """Load and parse an ESPHome YAML configuration file.

    This is the main entry point for parsing ESPHome configs.
    Handles all custom directives and resolves substitutions.

    Args:
        path: Path to the YAML file

    Returns:
        Parsed configuration data with substitutions resolved

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        KeyError: If substitutions or secrets reference undefined variables
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return None

    # Create loader with file context
    loader = ESPHomeLoader(content, config_path=path, yaml_loader=_default_yaml_loader)

    try:
        data = loader.get_single_data()
    finally:
        loader.dispose()

    # Process substitutions if present
    if isinstance(data, dict):
        substitutions = data.get("substitutions", {})
        if substitutions:
            # First resolve substitutions within the substitutions block itself
            resolved_subs = resolve_substitutions(substitutions, substitutions)
            # Then resolve throughout the config (but not inside Lambda)
            data = resolve_substitutions(data, resolved_subs)

    return data
