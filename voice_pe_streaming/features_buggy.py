"""Buggy variant catalog: one mutation per feature that minimally regresses it.

Each variant returns a modified patches list. The test infrastructure asserts:
    1. After applying the broken patches, the targeted Feature's invariant FAILS.
    2. Every other Feature's invariant STILL HOLDS (modulo `expected_collateral`).

When two features share a patch (e.g., F6/F7/F8 all share PATCH_ON_CLIENT_CONNECTED),
the mutation is surgical: it replaces the shared patch with a new one whose value
is identical except for the one detail that establishes the targeted feature.
This is how isolation is preserved without splitting shared patches.
"""

from __future__ import annotations

from esphome_yaml_tools.firmware_generator import MISSING, Patch
from voice_pe_streaming.features import (
    BuggyVariant,
    PATCH_DROP_SENSITIVITY,
    PATCH_ESPHOME_NAMING,
    PATCH_HANDOFF_SCRIPT,
    PATCH_MIC_MONO,
    PATCH_MWW_STOP_ONLY,
    PATCH_ON_CLIENT_CONNECTED,
    PATCH_ON_END,
    PATCH_ON_ERROR,
    PATCH_ON_WAKE_WORD_DETECTED,
    PATCH_PTT_CENTER_BUTTON,
    PATCH_PTT_GLOBAL,
    PATCH_SUBSTITUTIONS_NAMING,
    PATCH_USE_WAKE_WORD,
    _RESTART_GUARDS,
)


def _drop_patch(patches: list[Patch], target: Patch) -> list[Patch]:
    return [p for p in patches if p is not target]


def _swap_patch(patches: list[Patch], old: Patch, new: Patch) -> list[Patch]:
    return [new if p is old else p for p in patches]


def _b1_drop_mic_patch(patches: list[Patch]) -> list[Patch]:
    return _drop_patch(patches, PATCH_MIC_MONO)


def _b2_drop_mww_patch(patches: list[Patch]) -> list[Patch]:
    return _drop_patch(patches, PATCH_MWW_STOP_ONLY)


def _b3_drop_sensitivity_patch(patches: list[Patch]) -> list[Patch]:
    return _drop_patch(patches, PATCH_DROP_SENSITIVITY)


def _b4_drop_use_wake_word_patch(patches: list[Patch]) -> list[Patch]:
    return _drop_patch(patches, PATCH_USE_WAKE_WORD)


def _b5_drop_naming_patches(patches: list[Patch]) -> list[Patch]:
    return [p for p in patches if p not in (PATCH_SUBSTITUTIONS_NAMING, PATCH_ESPHOME_NAMING)]


_OCC_NO_START_CONTINUOUS = [
    {"delay": "1s"},
    {
        "if": {
            "condition": {"not": {"voice_assistant.is_running": None}},
            "then": [{"script.execute": "control_leds"}],
        }
    },
]

_OCC_NO_DELAY = [
    {
        "if": {
            "condition": {"not": {"voice_assistant.is_running": None}},
            "then": [{"voice_assistant.start_continuous": None}],
        }
    },
]

_OCC_NO_GUARD = [
    {"delay": "1s"},
    {"voice_assistant.start_continuous": None},
]


def _b6_remove_start_continuous(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_CLIENT_CONNECTED,
        Patch(
            op="append_list",
            path="/voice_assistant/on_client_connected",
            rationale="BUGGY: omits start_continuous",
            expected=PATCH_ON_CLIENT_CONNECTED.expected,
            value=_OCC_NO_START_CONTINUOUS,
        ),
    )


def _b7_remove_settle_delay(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_CLIENT_CONNECTED,
        Patch(
            op="append_list",
            path="/voice_assistant/on_client_connected",
            rationale="BUGGY: omits 1s settle delay",
            expected=PATCH_ON_CLIENT_CONNECTED.expected,
            value=_OCC_NO_DELAY,
        ),
    )


def _b8_remove_idempotency_guard(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_CLIENT_CONNECTED,
        Patch(
            op="append_list",
            path="/voice_assistant/on_client_connected",
            rationale="BUGGY: omits if-not-running idempotency guard",
            expected=PATCH_ON_CLIENT_CONNECTED.expected,
            value=_OCC_NO_GUARD,
        ),
    )


_ON_END_PTT_BRANCH = {
    "if": {
        "condition": {"lambda": "return id(button_initiated_wake);"},
        "then": [
            {"lambda": "id(button_initiated_wake) = false;"},
            {"voice_assistant.stop": None},
            {"wait_until": {"not": {"voice_assistant.is_running": None}}},
            {"lambda": "id(va).set_use_wake_word(false);"},
            {"voice_assistant.start": None},
        ],
        "else": [
            {"lambda": "id(va).set_use_wake_word(true);"},
            {"voice_assistant.start_continuous": None},
        ],
    }
}

_ON_END_NO_RESTART = [
    {"delay": "500ms"},
    {
        "if": {
            "condition": {"and": list(_RESTART_GUARDS)},
            "then": [
                {
                    "if": {
                        "condition": {"lambda": "return id(button_initiated_wake);"},
                        "then": [
                            {"lambda": "id(button_initiated_wake) = false;"},
                            {"voice_assistant.start": None},
                        ],
                        "else": [
                            {"lambda": "id(va).set_use_wake_word(true);"},
                            {"script.execute": "control_leds"},
                        ],
                    }
                }
            ],
        }
    },
]

_ON_END_NO_DEBOUNCE = [
    {
        "if": {
            "condition": {"and": list(_RESTART_GUARDS)},
            "then": [_ON_END_PTT_BRANCH],
        }
    },
]

_ON_END_NO_GUARDS = [
    {"delay": "500ms"},
    _ON_END_PTT_BRANCH,
]

_ON_END_NO_PTT_BRANCH = [
    {"delay": "500ms"},
    {
        "if": {
            "condition": {"and": list(_RESTART_GUARDS)},
            "then": [
                {"lambda": "id(va).set_use_wake_word(true);"},
                {"voice_assistant.start_continuous": None},
            ],
        }
    },
]


def _b9_remove_restart(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_END,
        Patch(
            op="append_list",
            path="/voice_assistant/on_end",
            rationale="BUGGY: removes start_continuous from on_end restart path",
            expected=PATCH_ON_END.expected,
            value=_ON_END_NO_RESTART,
        ),
    )


def _b10_remove_debounce(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_END,
        Patch(
            op="append_list",
            path="/voice_assistant/on_end",
            rationale="BUGGY: omits 500ms debounce before on_end restart",
            expected=PATCH_ON_END.expected,
            value=_ON_END_NO_DEBOUNCE,
        ),
    )


def _b11_remove_end_guards(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_END,
        Patch(
            op="append_list",
            path="/voice_assistant/on_end",
            rationale="BUGGY: omits triple guard (mute/api/not_running) on on_end restart",
            expected=PATCH_ON_END.expected,
            value=_ON_END_NO_GUARDS,
        ),
    )


def _b21_remove_ptt_branch(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_END,
        Patch(
            op="append_list",
            path="/voice_assistant/on_end",
            rationale="BUGGY: omits PTT branch in on_end; button press just restarts continuous (no chime/handoff)",
            expected=PATCH_ON_END.expected,
            value=_ON_END_NO_PTT_BRANCH,
        ),
    )


_ON_ERROR_NO_RESTART = [
    {"delay": "2s"},
    {
        "if": {
            "condition": {"and": list(_RESTART_GUARDS)},
            "then": [{"lambda": "id(va).set_use_wake_word(true);"}],
        }
    },
]

_ON_ERROR_NO_DELAY = [
    {
        "if": {
            "condition": {"and": list(_RESTART_GUARDS)},
            "then": [
                {"lambda": "id(va).set_use_wake_word(true);"},
                {"voice_assistant.start_continuous": None},
            ],
        }
    },
]

_ON_ERROR_NO_GUARDS = [
    {"delay": "2s"},
    {"lambda": "id(va).set_use_wake_word(true);"},
    {"voice_assistant.start_continuous": None},
]


def _b12_remove_error_restart(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_ERROR,
        Patch(
            op="append_list",
            path="/voice_assistant/on_error",
            rationale="BUGGY: omits start_continuous from on_error restart",
            expected=PATCH_ON_ERROR.expected,
            value=_ON_ERROR_NO_RESTART,
        ),
    )


def _b13_remove_error_delay(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_ERROR,
        Patch(
            op="append_list",
            path="/voice_assistant/on_error",
            rationale="BUGGY: omits 2s delay before on_error restart",
            expected=PATCH_ON_ERROR.expected,
            value=_ON_ERROR_NO_DELAY,
        ),
    )


def _b14_remove_error_guards(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_ERROR,
        Patch(
            op="append_list",
            path="/voice_assistant/on_error",
            rationale="BUGGY: omits triple guard on on_error restart",
            expected=PATCH_ON_ERROR.expected,
            value=_ON_ERROR_NO_GUARDS,
        ),
    )


_OWWD_CALL_HANDOFF = lambda guards: [
    {
        "if": {
            "condition": {"and": guards},
            "then": [{"script.execute": "handle_session_handoff"}],
        }
    }
]

_GUARD_MUTE = {"switch.is_off": "master_mute_switch"}
_GUARD_TIMER = {"switch.is_off": "timer_ringing"}
_GUARD_ANNOUNCING = {"not": {"media_player.is_announcing": {"id": "external_media_player"}}}


_HANDOFF_NO_CHIME = [
    {
        "id": "handle_session_handoff",
        "mode": "single",
        "then": [
            {"lambda": "id(button_initiated_wake) = false;"},
            {"lambda": "id(voice_assistant_phase) = 2;"},
            {"script.execute": "control_leds"},
            {"voice_assistant.stop": None},
            {"wait_until": {"not": {"voice_assistant.is_running": None}}},
            {"delay": "100ms"},
            {"lambda": "id(va).set_use_wake_word(false);"},
            {"voice_assistant.start": None},
        ],
    }
]


def _b15_remove_chime(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_HANDOFF_SCRIPT,
        Patch(
            op="append_list",
            path="/script",
            rationale="BUGGY: handoff script omits the wake chime play_sound action",
            value=_HANDOFF_NO_CHIME,
        ),
    )


def _b16_remove_announcing_guard(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_WAKE_WORD_DETECTED,
        Patch(
            op="set_key",
            path="/voice_assistant/on_wake_word_detected",
            rationale="BUGGY: handoff fires during TTS announcement -> chime echo loop",
            expected=MISSING,
            value=_OWWD_CALL_HANDOFF([_GUARD_MUTE, _GUARD_TIMER]),
        ),
    )


def _b17_remove_timer_guard(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_WAKE_WORD_DETECTED,
        Patch(
            op="set_key",
            path="/voice_assistant/on_wake_word_detected",
            rationale="BUGGY: handoff fires during timer ringing -> chime echo loop",
            expected=MISSING,
            value=_OWWD_CALL_HANDOFF([_GUARD_MUTE, _GUARD_ANNOUNCING]),
        ),
    )


def _b18_remove_mute_guard(patches: list[Patch]) -> list[Patch]:
    return _swap_patch(
        patches,
        PATCH_ON_WAKE_WORD_DETECTED,
        Patch(
            op="set_key",
            path="/voice_assistant/on_wake_word_detected",
            rationale="BUGGY: handoff fires when muted",
            expected=MISSING,
            value=_OWWD_CALL_HANDOFF([_GUARD_TIMER, _GUARD_ANNOUNCING]),
        ),
    )


def _b19_drop_ptt_global(patches: list[Patch]) -> list[Patch]:
    return _drop_patch(patches, PATCH_PTT_GLOBAL)


def _b20_drop_button_extension(patches: list[Patch]) -> list[Patch]:
    return _drop_patch(patches, PATCH_PTT_CENTER_BUTTON)


_OWWD_CHIME_ONLY_NO_HANDOFF = [
    {
        "if": {
            "condition": {
                "and": [
                    {"switch.is_off": "master_mute_switch"},
                    {"switch.is_off": "timer_ringing"},
                    {"not": {"media_player.is_announcing": {"id": "external_media_player"}}},
                    {"switch.is_on": "wake_sound"},
                ]
            },
            "then": [
                {
                    "script.execute": {
                        "id": "play_sound",
                        "priority": True,
                        "sound_file": "wake_word_triggered_sound",
                    }
                }
            ],
        }
    }
]


def _b22_drop_handoff(patches: list[Patch]) -> list[Patch]:
    """Models a broken firmware variant: chime-only on_wake_word_detected,
    no voice_assistant.stop / set_use_wake_word(false) / voice_assistant.start handoff."""
    out = _drop_patch(patches, PATCH_HANDOFF_SCRIPT)
    return _swap_patch(
        out,
        PATCH_ON_WAKE_WORD_DETECTED,
        Patch(
            op="set_key",
            path="/voice_assistant/on_wake_word_detected",
            rationale="BUGGY: chime-only path that leaves voice_assistant in continuous mode",
            expected=MISSING,
            value=_OWWD_CHIME_ONLY_NO_HANDOFF,
        ),
    )


BUGGY_VARIANTS: list[BuggyVariant] = [
    BuggyVariant(
        feature_id="F1_MIC_MONO",
        name="drop_mic_patch",
        rationale="Without PATCH_MIC_MONO, voice_assistant.microphone reverts to base [c0,c1] -> HA pipeline-restart bug returns.",
        mutate=_b1_drop_mic_patch,
    ),
    BuggyVariant(
        feature_id="F2_MWW_STOP_ONLY",
        name="drop_mww_filter",
        rationale="All 4 mWW models present -> stale wake words flash storage with no test coverage of streaming-only path.",
        mutate=_b2_drop_mww_patch,
    ),
    BuggyVariant(
        feature_id="F3_DROP_SENSITIVITY",
        name="keep_sensitivity_select",
        rationale="wake_word_sensitivity stays; its lambda references id(okay_nabu) etc. -> ESPHome codegen error.",
        mutate=_b3_drop_sensitivity_patch,
    ),
    BuggyVariant(
        feature_id="F4_USE_WAKE_WORD_TRUE",
        name="drop_use_wake_word_true",
        rationale="Reverts to base default false; HA never gets the wake-word signal -> no sessions ever start.",
        mutate=_b4_drop_use_wake_word_patch,
    ),
    BuggyVariant(
        feature_id="F5_PROJECT_NAMING",
        name="drop_project_naming",
        rationale="Device shows as 'home-assistant-voice' in HA; project metadata missing.",
        mutate=_b5_drop_naming_patches,
    ),
    BuggyVariant(
        feature_id="F6_START_CONTINUOUS_ON_CONNECT",
        name="remove_start_continuous_from_connect",
        rationale="After API connect, audio stream never starts -> device silent until first error/end event.",
        mutate=_b6_remove_start_continuous,
    ),
    BuggyVariant(
        feature_id="F7_API_SETTLE_DELAY",
        name="remove_1s_settle_delay",
        rationale="start_continuous races API init; observed flakiness on initial connection.",
        mutate=_b7_remove_settle_delay,
    ),
    BuggyVariant(
        feature_id="F8_CONNECT_IDEMPOTENCY_GUARD",
        name="remove_connect_idempotency_guard",
        rationale="Wake-word session started during 1s settle gets torn down by start_continuous.",
        mutate=_b8_remove_idempotency_guard,
    ),
    BuggyVariant(
        feature_id="F9_RESTART_ON_END",
        name="remove_on_end_restart",
        rationale="After each session ends, no restart -> device silent until next reconnect.",
        mutate=_b9_remove_restart,
    ),
    BuggyVariant(
        feature_id="F10_END_DEBOUNCE",
        name="remove_on_end_debounce",
        rationale="Restart races base on_end's wait_until/unduck/phase reset.",
        mutate=_b10_remove_debounce,
    ),
    BuggyVariant(
        feature_id="F11_END_GUARDS",
        name="remove_on_end_guards",
        rationale="Restart fires during mute / without API / mid-session -> wasted CPU + corrupted state.",
        mutate=_b11_remove_end_guards,
    ),
    BuggyVariant(
        feature_id="F12_RESTART_ON_ERROR",
        name="remove_on_error_restart",
        rationale="Transient HA errors brick the device until next reconnect.",
        mutate=_b12_remove_error_restart,
    ),
    BuggyVariant(
        feature_id="F13_ERROR_DELAY",
        name="remove_on_error_delay",
        rationale="Tight restart loop on persistent errors.",
        mutate=_b13_remove_error_delay,
    ),
    BuggyVariant(
        feature_id="F14_ERROR_GUARDS",
        name="remove_on_error_guards",
        rationale="Restart fires under invalid preconditions.",
        mutate=_b14_remove_error_guards,
    ),
    BuggyVariant(
        feature_id="F15_WAKE_CHIME",
        name="remove_wake_chime",
        rationale="User has no audible confirmation that the device heard the wake word.",
        mutate=_b15_remove_chime,
    ),
    BuggyVariant(
        feature_id="F16_CHIME_SUPPRESS_ANNOUNCING",
        name="remove_announcing_suppression",
        rationale="Chime fires during TTS -> echo loop bug returns .",
        mutate=_b16_remove_announcing_guard,
    ),
    BuggyVariant(
        feature_id="F17_CHIME_SUPPRESS_TIMER",
        name="remove_timer_suppression",
        rationale="Chime fires during timer ringing -> echo loop bug.",
        mutate=_b17_remove_timer_guard,
    ),
    BuggyVariant(
        feature_id="F18_CHIME_SUPPRESS_MUTED",
        name="remove_mute_suppression",
        rationale="Chime fires when user has muted the device.",
        mutate=_b18_remove_mute_guard,
    ),
    BuggyVariant(
        feature_id="F19_PTT_GLOBAL",
        name="drop_ptt_global_declaration",
        rationale="button_initiated_wake undefined -> ESPHome codegen failure; PTT can't be armed.",
        mutate=_b19_drop_ptt_global,
    ),
    BuggyVariant(
        feature_id="F20_PTT_BUTTON_HANDLER",
        name="drop_center_button_ptt_extension",
        rationale="Button press never sets the PTT flag; user loses push-to-talk feature.",
        mutate=_b20_drop_button_extension,
    ),
    BuggyVariant(
        feature_id="F21_PTT_ON_END_BRANCH",
        name="remove_on_end_ptt_branch",
        rationale="on_end always restarts continuous instead of routing PTT through chime+single-cmd handoff -> button press is silent restart.",
        mutate=_b21_remove_ptt_branch,
    ),
    BuggyVariant(
        feature_id="F22_SESSION_HANDOFF",
        name="chime_only_no_handoff_2026_05_30_regression",
        rationale=(
            "Models a broken firmware variant: on_wake_word_detected plays the "
            "chime but doesn't transition voice_assistant out of continuous mode. "
            "Result: phase transitions stall (device stuck in listening), stop word "
            "doesn't fire during timer ring (mWW path inert because session never "
            "ends cleanly), LED stuck in replying after timer set (phase never resets "
            "to idle). The fix is the handoff script + on_wake_word_detected calling it."
        ),
        mutate=_b22_drop_handoff,
    ),
]
