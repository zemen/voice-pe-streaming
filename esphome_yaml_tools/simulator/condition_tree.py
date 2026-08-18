"""Recursive condition evaluation for ESPHome-style nested conditions.

The condition tree supports arbitrary nesting beyond a flat implicit-AND list:

    condition:
      and:
        - switch.is_on: use_streaming_wake_word
        - switch.is_off: master_mute_switch
        - api.connected:
        - not:
            voice_assistant.is_running:
        - lambda: 'return id(voice_assistant_phase) != 2;'

This module replaces the flat list with a Condition tree that handles
`and`, `or`, `not`, `lambda`, and the various leaf condition types ESPHome
exposes (switch.is_on, switch.is_off, api.connected, voice_assistant.is_running,
media_player.is_announcing, light.is_on, binary_sensor.is_on, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)


@dataclass
class ConditionNode:
    """Tree node representing one condition.

    `op` discriminates the kind:
      "and"|"or"|"not" — composite, children populated
      "leaf"           — atomic condition: leaf_type + target/params

    Leaf types we recognize:
      switch.is_on / switch.is_off
      light.is_on / light.is_off
      binary_sensor.is_on / binary_sensor.is_off
      api.connected
      voice_assistant.is_running
      media_player.is_announcing
      lambda
    """

    op: str = "leaf"
    leaf_type: str | None = None
    target: str | None = None
    lambda_body: str | None = None
    children: list["ConditionNode"] = field(default_factory=list)


@dataclass
class ConditionEvaluator:
    """Evaluate ConditionNodes against simulator state.

    The simulator passes itself in via `state_lookup` (a callable that returns
    a component's state given "domain.id"). `lambda_eval` evaluates lambda
    conditions through the lambda_parser module.

    For leaf types whose state isn't modeled in the simulator (e.g.,
    api.connected when there's no API simulation), we use sensible defaults:
      api.connected            → True (assume HA is connected in tests)
      voice_assistant.is_running → reads voice_assistant state
      media_player.is_announcing → False unless test sets it
    """

    state_lookup: Callable[[str], Any]
    lambda_eval: Callable[[str], bool]

    def evaluate(self, node: ConditionNode) -> bool:
        if node.op == "and":
            return all(self.evaluate(c) for c in node.children)
        if node.op == "or":
            return any(self.evaluate(c) for c in node.children)
        if node.op == "not":
            return not self.evaluate(node.children[0])
        return self._evaluate_leaf(node)

    def _evaluate_leaf(self, node: ConditionNode) -> bool:
        lt = node.leaf_type
        target = node.target

        if lt == "lambda" and node.lambda_body is not None:
            return self.lambda_eval(node.lambda_body)

        if lt == "switch.is_on":
            state = self.state_lookup(f"switch.{target}")
            return bool(getattr(state, "on", False)) if state else False
        if lt == "switch.is_off":
            state = self.state_lookup(f"switch.{target}")
            return not bool(getattr(state, "on", False)) if state else True

        if lt == "light.is_on":
            state = self.state_lookup(f"light.{target}")
            return bool(getattr(state, "on", False)) if state else False
        if lt == "light.is_off":
            state = self.state_lookup(f"light.{target}")
            return not bool(getattr(state, "on", False)) if state else True

        if lt == "binary_sensor.is_on":
            state = self.state_lookup(f"binary_sensor.{target}")
            return bool(getattr(state, "on", False)) if state else False
        if lt == "binary_sensor.is_off":
            state = self.state_lookup(f"binary_sensor.{target}")
            return not bool(getattr(state, "on", False)) if state else True

        if lt == "api.connected":
            return True

        if lt == "voice_assistant.is_running":
            state = self.state_lookup(f"voice_assistant.{target or 'va'}")
            return bool(getattr(state, "is_streaming", False) or getattr(state, "is_in_session", False)) if state else False

        if lt == "media_player.is_announcing":
            state = self.state_lookup(f"media_player.{target}")
            return bool(getattr(state, "is_announcing", False)) if state else False

        _LOGGER.warning("ConditionEvaluator: unknown leaf type %r, defaulting to False", lt)
        return False


def parse_condition_yaml(node: Any) -> ConditionNode | None:
    """Recursively convert a parsed-YAML condition into a ConditionNode tree.

    Handles the YAML shapes the firmware uses:
      {and: [c1, c2, ...]}
      {or: [c1, c2, ...]}
      {not: cN}
      {switch.is_on: 'foo'} or {switch: { is_on: 'foo' }}
      {lambda: 'return ...;'}
      api.connected (bare key, value None)
      voice_assistant.is_running (bare key)
      media_player.is_announcing: { id: foo }

    Returns None for unparseable nodes (caller may default-to-True).
    """
    if node is None:
        return None
    if isinstance(node, list):
        return ConditionNode(
            op="and",
            children=[c for c in (parse_condition_yaml(x) for x in node) if c],
        )
    if not isinstance(node, dict):
        return None

    if "and" in node:
        return ConditionNode(
            op="and",
            children=[c for c in (parse_condition_yaml(x) for x in node["and"]) if c],
        )
    if "or" in node:
        return ConditionNode(
            op="or",
            children=[c for c in (parse_condition_yaml(x) for x in node["or"]) if c],
        )
    if "not" in node:
        child = parse_condition_yaml(node["not"])
        if child is None:
            return None
        return ConditionNode(op="not", children=[child])

    for key in (
        "switch.is_on", "switch.is_off",
        "light.is_on", "light.is_off",
        "binary_sensor.is_on", "binary_sensor.is_off",
    ):
        if key in node:
            value = node[key]
            target = value if isinstance(value, str) else (value.get("id") if isinstance(value, dict) else "")
            return ConditionNode(op="leaf", leaf_type=key, target=target)

    if "lambda" in node:
        body = node["lambda"]
        body_text = body.value if hasattr(body, "value") else str(body)
        return ConditionNode(op="leaf", leaf_type="lambda", lambda_body=body_text)

    if "api.connected" in node:
        return ConditionNode(op="leaf", leaf_type="api.connected")

    if "voice_assistant.is_running" in node:
        value = node["voice_assistant.is_running"]
        target = value.get("id") if isinstance(value, dict) else (value if isinstance(value, str) else "va")
        return ConditionNode(op="leaf", leaf_type="voice_assistant.is_running", target=target)

    if "media_player.is_announcing" in node:
        value = node["media_player.is_announcing"]
        target = value.get("id") if isinstance(value, dict) else (value if isinstance(value, str) else "")
        return ConditionNode(op="leaf", leaf_type="media_player.is_announcing", target=target)

    _LOGGER.warning("parse_condition_yaml: unrecognized condition shape: %s", node)
    return None
