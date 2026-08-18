"""Python stubs for base-package scripts our firmware references.

The base ESPHome Voice PE package defines several scripts that our overlay
firmware invokes via `script.execute:`. The simulator can't load the base
YAML (it lives on GitHub), so we hand-code Python equivalents that match
the base's documented behavior:

- `control_leds`     — priority-based dispatcher mapping voice_assistant_phase
                       and other globals to a specific LED effect on the
                       voice_assistant_leds light component.
- `activate_stop_word_once` — enables the stop mWW model briefly during TTS.
                       Modeled as a tracked call (sets a flag); tests assert
                       the flag was set.
- `play_sound`       — plays a named sound file. Modeled as tracked call;
                       tests assert sound name was queued.

Each stub is registered with the simulator's ScriptRegistry at init time.
When the firmware does `script.execute: control_leds`, the simulator's
action dispatcher looks up the script and either runs the stub (for base
scripts) or runs the registered actions (for firmware-defined scripts).

These are NOT meant to be perfect ESPHome replicas — they're sufficient to
test the behavioral assertions our firmware tests make (e.g., "after
phase=5 + control_leds, the led_ring's tracked effect is 'Replying'").
The verbatim base YAML is at base_voice_pe_internals.md (reference doc).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from esphome_yaml_tools.simulator.states import LightState

_LOGGER = logging.getLogger(__name__)


PHASE_EFFECT_MAP = {
    2: "Waiting for Command",
    3: "Listening For Command",
    4: "Thinking",
    5: "Replying",
    11: "Error",
}

MUTED_OR_SILENT_EFFECT = "Muted or Silent"
NOT_READY_EFFECT = "Twinkle"


@dataclass
class BaseScriptCallRecord:
    """One invocation of a base script. Tests can read history to assert calls."""

    script_id: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseScriptsState:
    """Tracks per-run state of all base script stubs (test-readable).

    `stop_word_active` tracks activate_stop_word_once invocations (the
    firmware's "stop must always work during TTS" requirement is verified
    by checking this flag was set during on_tts_start / on_intent_progress).

    `sounds_played` is a list of (sound_file, priority) tuples recorded by
    play_sound stub invocations — tests assert "wake_word_triggered_sound
    was played when wake_sound switch was on".
    """

    stop_word_active: bool = False
    sounds_played: list[tuple[str, bool]] = field(default_factory=list)
    call_history: list[BaseScriptCallRecord] = field(default_factory=list)


def control_leds_stub(
    globals_get: Callable[[str], Any],
    state_lookup: Callable[[str], Any],
    set_light_state: Callable[[str, LightState], None],
    call_history: list[BaseScriptCallRecord],
) -> None:
    """Simulate the base's control_leds priority dispatch.

    Priority order (matches base YAML lines 1038-1089):
      1. voice_kit_component.is_failed     → "Voice kit startup failed"
      2. improv_ble_in_progress            → improv visualization (unmodeled, fallback to idle)
      3. init_in_progress                  → init state (unmodeled, fallback to idle)
      4. !wifi_id.is_connected || !api_id.is_connected → no_ha_connection (unmodeled)
      5. center_button.state               → "Center Button Touched"
      6. jack_plugged_recently             → jack plugged
      7. jack_unplugged_recently           → jack unplugged
      8. dial_touched                      → dial touched
      9. timer_ringing.state               → timer ring effect
     10. voice_assistant_phase = 2 (waiting)  → "Waiting for Command"
     11. voice_assistant_phase = 3 (listening) → "Listening For Command"
     12. voice_assistant_phase = 4 (thinking)  → "Thinking"
     13. voice_assistant_phase = 5 (replying)  → "Replying"
     14. voice_assistant_phase = 11 (error)    → "Error"
     15. voice_assistant_phase = 10 (not_ready) → "Twinkle" (red)
     16. is_timer_active                       → timer tick
     17. master_mute_switch.state              → "Muted or Silent"
     18. external_media_player.volume==0 || muted → "Muted or Silent"
     19. voice_assistant_phase = 1 (idle)      → off / leds_internal default

    The simulator only models phases (10-15) and mute (17). Other states
    fallback to phase-based dispatch. This is sufficient for the firmware
    behavioral tests which exercise phase transitions + mute.

    Writes the resolved effect into `light.voice_assistant_leds`. Tests
    look there.
    """
    call_history.append(BaseScriptCallRecord(script_id="control_leds"))

    mute_state = state_lookup("switch.master_mute_switch")
    if mute_state and getattr(mute_state, "on", False):
        set_light_state(
            "light.voice_assistant_leds",
            LightState(on=True, brightness=0.2, color=(255, 0, 0), effect=MUTED_OR_SILENT_EFFECT),
        )
        return

    phase = globals_get("voice_assistant_phase")
    try:
        phase_int = int(phase) if phase is not None else 0
    except (TypeError, ValueError):
        phase_int = 0

    if phase_int in PHASE_EFFECT_MAP:
        if phase_int == 11:
            color = (255, 0, 0)
        elif phase_int == 5:
            color = (0, 255, 0)
        else:
            color = (255, 102, 51)
        set_light_state(
            "light.voice_assistant_leds",
            LightState(on=True, brightness=0.66, color=color, effect=PHASE_EFFECT_MAP[phase_int]),
        )
        return

    if phase_int == 10:
        set_light_state(
            "light.voice_assistant_leds",
            LightState(on=True, brightness=0.66, color=(255, 0, 0), effect=NOT_READY_EFFECT),
        )
        return

    if phase_int == 1:
        set_light_state(
            "light.voice_assistant_leds",
            LightState(on=False, brightness=0.0, color=(0, 0, 0), effect=""),
        )
        return

    _LOGGER.debug("control_leds: unhandled phase=%r, defaulting to off", phase)
    set_light_state(
        "light.voice_assistant_leds",
        LightState(on=False, brightness=0.0, color=(0, 0, 0), effect=""),
    )


def activate_stop_word_once_stub(state: BaseScriptsState) -> None:
    """Mark that the stop word was enabled during TTS / replying phase.

    Real base script: waits for media_player.is_announcing, then enables stop
    mWW model briefly. The simulator doesn't simulate timing for this; it
    just records that the stop word was activated. Tests assert the flag was
    set as proof that the on_tts_start / on_intent_progress event handlers
    correctly called the script.
    """
    state.stop_word_active = True
    state.call_history.append(BaseScriptCallRecord(script_id="activate_stop_word_once"))


def play_sound_stub(state: BaseScriptsState, sound_file: str, priority: bool = False) -> None:
    """Record a play_sound invocation. Tests verify expected sound names play."""
    state.sounds_played.append((sound_file, bool(priority)))
    state.call_history.append(
        BaseScriptCallRecord(script_id="play_sound", args={"sound_file": sound_file, "priority": priority})
    )


def base_voice_assistant_automations(va_id: str = "va", mww_id: str = "mww") -> list:
    """Return the list of automations the base ESPHome Voice PE package defines.

    These mirror github://esphome/home-assistant-voice-pe/home-assistant-voice.yaml@dev
    `voice_assistant:` block verbatim (source-cited in the librarian research at
    commit 1efed69c, file home-assistant-voice.yaml lines 1804-1908).

    The simulator pre-registers these so behavioral tests still work when our
    overlay firmware DOESN'T reproduce them. ESPHome's runtime concatenates
    package list-values, so both base and overlay handlers fire for the same
    event — this matches that semantics in test.
    """
    from esphome_yaml_tools.simulator.condition_tree import ConditionNode
    from esphome_yaml_tools.simulator.core import Action, Automation

    return [
        Automation(
            trigger="on_client_connected",
            actions=[
                Action(type="lambda", params={"body": "id(init_in_progress) = false;"}),
                Action(type="micro_wake_word.start", target=f"micro_wake_word.{mww_id}"),
                Action(type="lambda", params={"body": "id(voice_assistant_phase) = 1;"}),
                Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
            ],
        ),
        Automation(
            trigger="on_client_disconnected",
            actions=[
                Action(type="voice_assistant.stop", target=f"voice_assistant.{va_id}"),
                Action(type="lambda", params={"body": "id(voice_assistant_phase) = 10;"}),
                Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
            ],
        ),
        Automation(
            trigger="on_listening",
            actions=[
                Action(type="lambda", params={"body": "id(voice_assistant_phase) = 2;"}),
                Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
            ],
        ),
        Automation(
            trigger="on_start",
            actions=[
                Action(
                    type="mixer_speaker.apply_ducking",
                    params={"id": "media_mixing_input", "decibel_reduction": 20, "duration": 0.0},
                ),
            ],
        ),
        Automation(
            trigger="on_stt_vad_start",
            actions=[
                Action(type="lambda", params={"body": "id(voice_assistant_phase) = 3;"}),
                Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
            ],
        ),
        Automation(
            trigger="on_stt_vad_end",
            actions=[
                Action(type="lambda", params={"body": "id(voice_assistant_phase) = 4;"}),
                Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
            ],
        ),
        Automation(
            trigger="on_intent_progress",
            actions=[
                Action(
                    type="if",
                    params={
                        "condition": ConditionNode(
                            op="leaf",
                            leaf_type="lambda",
                            lambda_body="return !x.empty();",
                        ),
                        "then": [
                            Action(type="lambda", params={"body": "id(voice_assistant_phase) = 5;"}),
                            Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
                            Action(
                                type="script.execute",
                                target="activate_stop_word_once",
                                params={"id": "activate_stop_word_once"},
                            ),
                        ],
                        "else": [],
                    },
                ),
            ],
        ),
        Automation(
            trigger="on_tts_start",
            actions=[
                Action(
                    type="if",
                    params={
                        "condition": ConditionNode(
                            op="leaf",
                            leaf_type="lambda",
                            lambda_body="return id(voice_assistant_phase) != 5;",
                        ),
                        "then": [
                            Action(type="lambda", params={"body": "id(voice_assistant_phase) = 5;"}),
                            Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
                            Action(
                                type="script.execute",
                                target="activate_stop_word_once",
                                params={"id": "activate_stop_word_once"},
                            ),
                        ],
                        "else": [],
                    },
                ),
            ],
        ),
        Automation(
            trigger="on_end",
            actions=[
                Action(
                    type="wait_until",
                    params={
                        "condition": ConditionNode(
                            op="not",
                            children=[
                                ConditionNode(
                                    op="leaf",
                                    leaf_type="voice_assistant.is_running",
                                    target=va_id,
                                ),
                            ],
                        ),
                        "timeout": 0.0,
                    },
                ),
                Action(
                    type="mixer_speaker.apply_ducking",
                    params={"id": "media_mixing_input", "decibel_reduction": 0, "duration": 1.0},
                ),
                Action(
                    type="if",
                    params={
                        "condition": ConditionNode(
                            op="leaf",
                            leaf_type="lambda",
                            lambda_body="return id(voice_assistant_phase) == 11;",
                        ),
                        "then": [Action(type="delay", params={"seconds": 1.0})],
                        "else": [],
                    },
                ),
                Action(type="lambda", params={"body": "id(voice_assistant_phase) = 1;"}),
                Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
            ],
        ),
        Automation(
            trigger="on_error",
            actions=[
                Action(
                    type="if",
                    params={
                        "condition": ConditionNode(
                            op="and",
                            children=[
                                ConditionNode(
                                    op="leaf",
                                    leaf_type="lambda",
                                    lambda_body="return !id(init_in_progress);",
                                ),
                                ConditionNode(
                                    op="leaf",
                                    leaf_type="lambda",
                                    lambda_body='return code != "duplicate_wake_up_detected";',
                                ),
                                ConditionNode(
                                    op="leaf",
                                    leaf_type="lambda",
                                    lambda_body='return code != "stt-no-text-recognized";',
                                ),
                            ],
                        ),
                        "then": [
                            Action(type="lambda", params={"body": "id(voice_assistant_phase) = 11;"}),
                            Action(type="script.execute", target="control_leds", params={"id": "control_leds"}),
                        ],
                        "else": [],
                    },
                ),
                Action(
                    type="if",
                    params={
                        "condition": ConditionNode(
                            op="leaf",
                            leaf_type="lambda",
                            lambda_body='return code == "cloud-auth-failed";',
                        ),
                        "then": [
                            Action(
                                type="script.execute",
                                target="play_sound",
                                params={"id": "play_sound", "priority": True, "sound_file": "error_cloud_expired"},
                            ),
                        ],
                        "else": [],
                    },
                ),
            ],
        ),
        Automation(
            trigger="on_timer_finished",
            actions=[Action(type="switch.turn_on", target="switch.timer_ringing")],
        ),
        Automation(
            trigger="on_timer_started",
            actions=[Action(type="script.execute", target="control_leds", params={"id": "control_leds"})],
        ),
        Automation(
            trigger="on_timer_cancelled",
            actions=[Action(type="script.execute", target="control_leds", params={"id": "control_leds"})],
        ),
        Automation(
            trigger="on_timer_updated",
            actions=[Action(type="script.execute", target="control_leds", params={"id": "control_leds"})],
        ),
        Automation(
            trigger="on_timer_tick",
            actions=[Action(type="script.execute", target="control_leds", params={"id": "control_leds"})],
        ),
    ]
