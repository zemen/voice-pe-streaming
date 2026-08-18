"""ESPHome `globals:` registry for the simulator.

The base Voice PE package defines globals like `voice_assistant_phase` (int)
and `init_in_progress` (bool). Our overlay firmware reads/writes them via
lambdas: `id(voice_assistant_phase) = 5;` or `return id(voice_assistant_phase)
== 4;`. The simulator needs to track these so behavioral tests can assert
phase transitions without running an actual ESPHome compiler.

This module is intentionally minimal: type-untyped storage with a fluent API.
The lambda parser writes here; condition evaluators read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GlobalsRegistry:
    _values: dict[str, Any] = field(default_factory=dict)
    _types: dict[str, str] = field(default_factory=dict)

    def declare(self, name: str, type_: str, initial: Any = None) -> None:
        """Register a global with a declared type and initial value.

        ESPHome's `globals:` block declares type + initial_value; we mirror
        that here. Subsequent set()/get() may be type-checked (warn-only)
        against the declared type.
        """
        self._types[name] = type_
        if initial is not None or name not in self._values:
            self._values[name] = initial

    def set(self, name: str, value: Any) -> None:
        self._values[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def has(self, name: str) -> bool:
        return name in self._values

    def all(self) -> dict[str, Any]:
        return dict(self._values)

    def reset(self) -> None:
        """Reset all globals to their declared initials (test isolation)."""
        self._values.clear()
