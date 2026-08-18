"""Adapts ESPHome firmware YAML into simulator-compatible config.

Capabilities:
- Translates every action type the firmware uses (script.execute, script.stop,
  lambda, wait_until, voice_assistant.{start, start_continuous, stop, set_use_wake_word},
  micro_wake_word.{start, stop, enable_model, disable_model}, mixer_speaker.apply_ducking,
  conditional `{if: {...}}` structures)
- Parses `script:` definitions and registers them with the simulator
- Parses `globals:` definitions and seeds the GlobalsRegistry
- Builds ConditionNode trees for nested and/or/not/lambda/leaf conditions
- Adds all 16 voice_assistant event handlers to va_events list
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from esphome_yaml_tools.parser import Extend, Lambda, Remove, load_yaml
from esphome_yaml_tools.simulator.condition_tree import (
    ConditionNode,
    parse_condition_yaml,
)
from esphome_yaml_tools.simulator.core import Action, Automation, Condition, Simulator


def _lambda_body(value: Any) -> str:
    if isinstance(value, Lambda):
        return value.value
    return str(value) if value is not None else ""


def _parse_color_percent(value: str | int | float) -> int:
    if isinstance(value, str) and value.endswith("%"):
        return int(float(value.rstrip("%")) / 100.0 * 255)
    return int(float(value) * 255) if isinstance(value, float) and value <= 1.0 else int(value)


def _parse_delay(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        val = value.strip()
        if val.endswith("ms"):
            return float(val[:-2]) / 1000.0
        if val.endswith("s"):
            return float(val[:-1])
        if val.endswith("min"):
            return float(val[:-3]) * 60.0
        return float(val)
    return 0.0


def _translate_action(
    raw: Any,
    va_id: str = "va",
    mww_id: str = "mww",
) -> Action | None:
    """Convert one YAML action dict (or string shorthand) into a simulator Action.

    Returns None for unrecognized action types; callers skip them. The
    `va_id` and `mww_id` arguments wire voice_assistant.* and
    micro_wake_word.* actions to the right target component id.
    """
    if isinstance(raw, str):
        return None
    if not isinstance(raw, dict):
        return None

    if "if" in raw and isinstance(raw["if"], dict):
        if_block = raw["if"]
        condition = parse_condition_yaml(if_block.get("condition"))
        then_actions = [
            a for a in (_translate_action(x, va_id=va_id, mww_id=mww_id) for x in if_block.get("then", []) or [])
            if a is not None
        ]
        else_actions = [
            a for a in (_translate_action(x, va_id=va_id, mww_id=mww_id) for x in if_block.get("else", []) or [])
            if a is not None
        ]
        return Action(
            type="if",
            params={
                "condition": condition,
                "then": then_actions,
                "else": else_actions,
            },
        )

    if "wait_until" in raw:
        body = raw["wait_until"]
        if isinstance(body, dict) and "condition" in body:
            condition = parse_condition_yaml(body["condition"])
            timeout = _parse_delay(body.get("timeout", 0)) if body.get("timeout") else 0.0
        else:
            condition = parse_condition_yaml(body)
            timeout = 0.0
        return Action(type="wait_until", params={"condition": condition, "timeout": timeout})

    for action_type, params in raw.items():
        if action_type == "light.turn_on":
            if isinstance(params, dict):
                target_id = params.get("id", "")
                color = (
                    _parse_color_percent(params.get("red", 0)),
                    _parse_color_percent(params.get("green", 0)),
                    _parse_color_percent(params.get("blue", 0)),
                )
                return Action(
                    type="light.turn_on",
                    target=f"light.{target_id}",
                    params={
                        "effect": params.get("effect", ""),
                        "color": color,
                        "brightness": 1.0,
                    },
                )

        elif action_type == "light.turn_off":
            target_id = params if isinstance(params, str) else (params.get("id", "") if isinstance(params, dict) else "")
            return Action(type="light.turn_off", target=f"light.{target_id}")

        elif action_type == "delay":
            return Action(type="delay", params={"seconds": _parse_delay(params)})

        elif action_type == "logger.log":
            return None

        elif action_type == "voice_assistant.start_continuous":
            return Action(type="voice_assistant.start_continuous", target=f"voice_assistant.{va_id}")

        elif action_type == "voice_assistant.start":
            return Action(type="voice_assistant.start", target=f"voice_assistant.{va_id}")

        elif action_type == "voice_assistant.stop":
            return Action(type="voice_assistant.stop", target=f"voice_assistant.{va_id}")

        elif action_type == "micro_wake_word.start":
            return Action(type="micro_wake_word.start", target=f"micro_wake_word.{mww_id}")

        elif action_type == "micro_wake_word.stop":
            return Action(type="micro_wake_word.stop", target=f"micro_wake_word.{mww_id}")

        elif action_type == "micro_wake_word.enable_model":
            model_id = params if isinstance(params, str) else (params.get("id", "") if isinstance(params, dict) else "")
            return Action(
                type="micro_wake_word.enable_model",
                target=f"micro_wake_word.{mww_id}",
                params={"model_id": model_id},
            )

        elif action_type == "micro_wake_word.disable_model":
            model_id = params if isinstance(params, str) else (params.get("id", "") if isinstance(params, dict) else "")
            return Action(
                type="micro_wake_word.disable_model",
                target=f"micro_wake_word.{mww_id}",
                params={"model_id": model_id},
            )

        elif action_type == "lambda":
            return Action(type="lambda", params={"body": _lambda_body(params)})

        elif action_type == "script.execute":
            if isinstance(params, str):
                return Action(type="script.execute", target=params, params={"id": params})
            if isinstance(params, dict):
                script_id = params.get("id", "")
                args = {k: v for k, v in params.items() if k != "id"}
                merged = {"id": script_id, **args}
                return Action(type="script.execute", target=script_id, params=merged)

        elif action_type == "script.stop":
            script_id = params if isinstance(params, str) else (params.get("id", "") if isinstance(params, dict) else "")
            return Action(type="script.stop", target=script_id, params={"id": script_id})

        elif action_type == "mixer_speaker.apply_ducking":
            return Action(type="mixer_speaker.apply_ducking", params=params if isinstance(params, dict) else {})

        elif action_type == "switch.turn_on":
            target_id = params if isinstance(params, str) else (params.get("id", "") if isinstance(params, dict) else "")
            return Action(type="switch.turn_on", target=f"switch.{target_id}")

        elif action_type == "switch.turn_off":
            target_id = params if isinstance(params, str) else (params.get("id", "") if isinstance(params, dict) else "")
            return Action(type="switch.turn_off", target=f"switch.{target_id}")

        elif action_type == "media_player.stop":
            target_id = ""
            if isinstance(params, dict):
                target_id = params.get("id", "")
            return Action(type="media_player.stop", target=f"media_player.{target_id}", params=params if isinstance(params, dict) else {})

    return None


_VA_EVENTS = (
    "on_client_connected",
    "on_client_disconnected",
    "on_wake_word_detected",
    "on_listening",
    "on_start",
    "on_stt_vad_start",
    "on_stt_vad_end",
    "on_stt_end",
    "on_intent_progress",
    "on_tts_start",
    "on_tts_end",
    "on_end",
    "on_error",
    "on_timer_finished",
    "on_timer_started",
    "on_timer_cancelled",
    "on_timer_updated",
    "on_timer_tick",
)


def adapt_firmware_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert ESPHome firmware YAML dict into a config the new Simulator can load.

    Output keys:
      voice_assistant: [{id}]
      micro_wake_word: [{id}]
      light:           [{id}, ...]
      switch:          [{id}, ...]
      media_player:    [{id}, ...]
      script:          [{id, mode, actions}, ...]
      globals:         [{id, type, initial_value}, ...]
      automation:      [{trigger, actions, condition_tree?}, ...]
    """
    sim_config: dict[str, Any] = {}
    automations: list[dict[str, Any]] = []

    va = config.get("voice_assistant", {}) or {}
    va_id = va.get("id", "va") if isinstance(va, dict) else "va"
    mww_id = "mww"
    if isinstance(config.get("micro_wake_word"), dict):
        mww_id = config["micro_wake_word"].get("id", "mww")

    sim_config["voice_assistant"] = [{"id": va_id}]
    sim_config["micro_wake_word"] = [{"id": mww_id}]

    if isinstance(va, dict):
        for event_name in _VA_EVENTS:
            raw_actions = va.get(event_name, [])
            if raw_actions is None:
                continue
            if not isinstance(raw_actions, list):
                raw_actions = [raw_actions]

            translated: list[Action] = []
            for raw in raw_actions:
                action = _translate_action(raw, va_id=va_id, mww_id=mww_id)
                if action is not None:
                    translated.append(action)

            if translated:
                automations.append({"trigger": event_name, "actions": translated})

    scripts_out: list[dict[str, Any]] = []
    for s in config.get("script", []) or []:
        if not isinstance(s, dict) or "id" not in s:
            continue
        actions_translated: list[Action] = []
        for raw in s.get("then", []) or []:
            action = _translate_action(raw, va_id=va_id, mww_id=mww_id)
            if action is not None:
                actions_translated.append(action)
        scripts_out.append(
            {
                "id": s["id"],
                "mode": s.get("mode", "single"),
                "actions": actions_translated,
                "parameters": s.get("parameters", {}),
            }
        )
    if scripts_out:
        sim_config["script"] = scripts_out

    globals_out: list[dict[str, Any]] = []
    for g in config.get("globals", []) or []:
        if isinstance(g, dict) and "id" in g:
            globals_out.append(
                {
                    "id": g["id"],
                    "type": g.get("type", "any"),
                    "initial_value": _coerce_initial(g.get("initial_value")),
                }
            )
    if globals_out:
        sim_config["globals"] = globals_out

    subs = config.get("substitutions", {}) or {}
    if isinstance(subs, dict):
        for sub_key, global_id, type_ in (
            ("voice_assist_idle_phase_id", "voice_assistant_phase", "int"),
        ):
            try:
                value = int(subs.get(sub_key))
            except (TypeError, ValueError):
                continue
            globals_out.append(
                {"id": global_id, "type": type_, "initial_value": value}
            )
    if globals_out and "globals" not in sim_config:
        sim_config["globals"] = globals_out

    switches_out: list[dict[str, Any]] = []
    for s in config.get("switch", []) or []:
        if not isinstance(s, dict):
            continue
        sid_raw = s.get("id")
        sid = sid_raw.value if isinstance(sid_raw, Extend) else sid_raw
        if sid:
            switches_out.append({"id": sid})
    if switches_out:
        sim_config["switch"] = switches_out

    lights_referenced = {"led_ring", "voice_assistant_leds"}
    for auto in automations:
        for act in auto["actions"]:
            if act.type in ("light.turn_on", "light.turn_off"):
                if act.target.startswith("light."):
                    lights_referenced.add(act.target.split(".", 1)[1])
    if lights_referenced:
        sim_config["light"] = [{"id": lid} for lid in sorted(lights_referenced)]

    sim_config["media_player"] = [{"id": "external_media_player"}]
    sim_config["automation"] = automations
    return sim_config


def _coerce_initial(value: Any) -> Any:
    """Convert YAML scalar string into Python int/bool when possible."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().strip("'\"")
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
    return value


def load_firmware_into_simulator(firmware_path: Path | str) -> Simulator:
    raw_config = load_yaml(Path(firmware_path))
    sim_config = adapt_firmware_config(raw_config)
    sim = Simulator()
    sim.load_config(sim_config)
    return sim


def get_firmware_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Extract testable metadata from firmware config (compatibility shape)."""
    metadata: dict[str, Any] = {}

    va = config.get("voice_assistant", {}) or {}
    if isinstance(va, dict):
        metadata["use_wake_word"] = va.get("use_wake_word", False)
        metadata["va_id"] = va.get("id", "")

    switches = config.get("switch", []) or []
    if isinstance(switches, list):
        streaming_switch = None
        for s in switches:
            if not isinstance(s, dict):
                continue
            sid = s.get("id")
            sid_value = sid.value if isinstance(sid, Extend) else sid
            if sid_value in ("use_streaming_wake_word", "enable_streaming_ww"):
                streaming_switch = s
                break
        metadata["has_streaming_switch"] = streaming_switch is not None
        if streaming_switch:
            metadata["streaming_switch_optimistic"] = streaming_switch.get("optimistic", False)

    subs = config.get("substitutions", {}) or {}
    if isinstance(subs, dict):
        metadata["phase_ids"] = {
            "idle": subs.get("voice_assist_idle_phase_id"),
            "waiting": subs.get("voice_assist_waiting_for_command_phase_id"),
            "listening": subs.get("voice_assist_listening_for_command_phase_id"),
            "thinking": subs.get("voice_assist_thinking_phase_id"),
            "replying": subs.get("voice_assist_replying_phase_id"),
            "error": subs.get("voice_assist_error_phase_id"),
            "not_ready": subs.get("voice_assist_not_ready_phase_id"),
        }

    scripts = config.get("script", []) or []
    script_ids = set()
    for s in scripts:
        if isinstance(s, dict) and "id" in s:
            script_ids.add(s["id"])
    metadata["script_ids"] = script_ids

    return metadata
