"""Feature catalog for the streaming-only Voice PE firmware.

Each `Feature` bundles:
    - patches that establish the behavior
    - an invariant that asserts the behavior holds in the generated config

Each `BuggyVariant` mutates the production patch list to minimally break one
feature. Tests assert the targeted feature's invariant fails while all other
features' invariants still pass, providing a no-silent-regressions guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from esphome_yaml_tools.firmware_generator import MISSING, Patch
from esphome_yaml_tools.parser import Lambda


@dataclass(frozen=True)
class Feature:
    id: str
    name: str
    rationale: str
    patches: tuple[Patch, ...]
    invariant: Callable[[dict], bool]


@dataclass(frozen=True)
class BuggyVariant:
    feature_id: str
    name: str
    rationale: str
    mutate: Callable[[list[Patch]], list[Patch]]
    expected_collateral: frozenset[str] = frozenset()


def _flatten_actions(root: Any) -> list[dict]:
    """Yield every dict node reachable through `then`/`else` branches of `if`
    blocks (action tree walk). Used by invariants to scan for specific actions."""
    out: list[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            out.append(node)
            for key in ("then", "else"):
                if key in node:
                    walk(node[key])
            if "if" in node and isinstance(node["if"], dict):
                for key in ("then", "else"):
                    if key in node["if"]:
                        walk(node["if"][key])

    walk(root)
    return out


def _conditions_text(action: dict) -> str:
    """Stringify an action's `if.condition` for substring matching in invariants."""
    if not isinstance(action, dict) or "if" not in action:
        return ""
    return str(action["if"].get("condition", ""))


def _keep_stop_model_only(models: list[dict]) -> list[dict]:
    kept = [m for m in models if m.get("id") == "stop"]
    if len(kept) != 1:
        raise ValueError(
            f"Expected exactly one model with id='stop', found {len(kept)}. "
            f"Upstream removed/renamed the stop model; review the transform."
        )
    return kept


def _drop_wake_word_sensitivity(items: list[dict]) -> list[dict]:
    kept = [s for s in items if s.get("id") != "wake_word_sensitivity"]
    if len(kept) == len(items):
        raise ValueError(
            "wake_word_sensitivity select not found; upstream removed/renamed it."
        )
    return kept


def _append_center_button_ptt(items: list[dict]) -> list[dict]:
    """Append the push-to-talk trigger to center_button.on_press. The trigger
    sets `button_initiated_wake = true` ONLY when device is in streaming idle
    (no timer, no announcement, no playing media, no mute, phase == idle).
    Without these guards, button presses during TTS or media playback would
    spuriously arm PTT."""
    for item in items:
        if isinstance(item, dict) and item.get("id") == "center_button":
            existing = item.get("on_press", [])
            trigger = {
                "if": {
                    "condition": {
                        "and": [
                            {"switch.is_off": "master_mute_switch"},
                            {"switch.is_off": "timer_ringing"},
                            {"not": {"media_player.is_announcing": {"id": "external_media_player"}}},
                            {"not": {"media_player.is_playing": {"id": "external_media_player"}}},
                            {"lambda": "return id(voice_assistant_phase) == 1;"},
                        ]
                    },
                    "then": [
                        {"lambda": "id(button_initiated_wake) = true;"},
                    ],
                }
            }
            item["on_press"] = list(existing) + [trigger]
            return items
    raise ValueError("center_button not found in binary_sensor list")


_RESTART_GUARDS = [
    {"switch.is_off": "master_mute_switch"},
    {"api.connected": None},
    {"not": {"voice_assistant.is_running": None}},
]


_ON_CLIENT_CONNECTED_APPEND = [
    {"delay": "1s"},
    {
        "if": {
            "condition": {"not": {"voice_assistant.is_running": None}},
            "then": [
                {"voice_assistant.start_continuous": None},
            ],
        }
    },
]


_ON_END_APPEND = [
    {"delay": "500ms"},
    {
        "if": {
            "condition": {"and": list(_RESTART_GUARDS)},
            "then": [
                {
                    "if": {
                        "condition": {"lambda": "return id(button_initiated_wake);"},
                        "then": [
                            {"script.execute": "handle_session_handoff"},
                        ],
                        "else": [
                            {"lambda": "id(va).set_use_wake_word(true);"},
                            {"voice_assistant.start_continuous": None},
                        ],
                    }
                }
            ],
        }
    },
]


_ON_ERROR_APPEND = [
    {"delay": "2s"},
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


_HANDOFF_SCRIPT = [
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
            {
                "if": {
                    "condition": {"switch.is_on": "wake_sound"},
                    "then": [
                        {
                            "script.execute": {
                                "id": "play_sound",
                                "priority": True,
                                "sound_file": "wake_word_triggered_sound",
                            }
                        },
                        {"delay": "300ms"},
                    ],
                }
            },
            {"voice_assistant.start": None},
        ],
    }
]


_ON_WAKE_WORD_DETECTED = [
    {
        "if": {
            "condition": {
                "and": [
                    {"switch.is_off": "master_mute_switch"},
                    {"switch.is_off": "timer_ringing"},
                    {"not": {"media_player.is_announcing": {"id": "external_media_player"}}},
                ]
            },
            "then": [
                {"script.execute": "handle_session_handoff"},
            ],
        }
    }
]


_PTT_GLOBAL = [
    {
        "id": "button_initiated_wake",
        "type": "bool",
        "restore_value": False,
        "initial_value": "false",
    }
]


PATCH_MIC_MONO = Patch(
    op="replace",
    path="/voice_assistant/microphone",
    rationale=(
        "Mono ch0 feed for HA Assist API. Upstream config subscribes to BOTH "
        "XMOS channels; ESPHome's STREAMING_MICROPHONE OR-loop "
        "(voice_assistant.cpp:303) emits msg with empty data when ch0 ring "
        "buffer is starved while ch1 has data; aioesphomeapi has no data2 "
        "field; HA assist_satellite.py:706 reads `if not chunk: break` and "
        "terminates the pipeline -> ~3 sessions/sec restart storm. "
        "micro_wake_word still uses ch1 for on-device stop word."
    ),
    expected=[
        {"microphone": "i2s_mics", "channels": 0},
        {"microphone": "i2s_mics", "channels": 1},
    ],
    value=[{"microphone": "i2s_mics", "channels": 0}],
)

PATCH_MWW_STOP_ONLY = Patch(
    op="transform",
    path="/micro_wake_word/models",
    rationale=(
        "Streaming-only: HA detects wake words server-side. The 3 on-device "
        "models (okay_nabu, hey_jarvis, hey_mycroft) are dead weight (~600KB "
        "each) and never fire. Keep only the 'stop' model (internal:true) for "
        "TTS/timer-interrupt detection. transform (vs replace): upstream may "
        "bump the stop model's URL; we want to inherit that automatically."
    ),
    transform=_keep_stop_model_only,
)

PATCH_DROP_SENSITIVITY = Patch(
    op="transform",
    path="/select",
    rationale=(
        "wake_word_sensitivity exposes a probability-cutoff slider for "
        "okay_nabu/hey_jarvis/hey_mycroft (removed by PATCH_MWW_STOP_ONLY). "
        "Its on_value lambda calls id(okay_nabu).set_probability_cutoff(...); "
        "with those IDs removed, ESPHome codegen fails: 'id does not exist'."
    ),
    transform=_drop_wake_word_sensitivity,
)

PATCH_USE_WAKE_WORD = Patch(
    op="replace",
    path="/voice_assistant/use_wake_word",
    rationale=(
        "Streaming-only: HA-side wake word detection. Upstream defaults to "
        "false because on-device mWW drove the wake path. Flip to true so HA "
        "detects from streamed audio."
    ),
    expected=False,
    value=True,
)

PATCH_SUBSTITUTIONS_NAMING = Patch(
    op="merge_dict",
    path="/substitutions",
    rationale=(
        "Adds project identity substitutions referenced by esphome.name/friendly_name."
    ),
    value={"name": "voice-pe-streaming", "friendly_name": "Voice PE Streaming"},
)

PATCH_ESPHOME_NAMING = Patch(
    op="merge_dict",
    path="/esphome",
    rationale=(
        "Overrides upstream's hard-coded 'home-assistant-voice' identity and "
        "adds project metadata so flashed devices show as Voice PE Streaming "
        "in HA and report the correct project name in dashboards."
    ),
    value={
        "name": "${name}",
        "friendly_name": "${friendly_name}",
        "project": {"name": "esphome.voice-pe-streaming", "version": "1.0"},
    },
)

PATCH_PTT_GLOBAL = Patch(
    op="append_list",
    path="/globals",
    rationale=(
        "Adds button_initiated_wake flag. Set by center_button.on_press during "
        "streaming idle; read by voice_assistant.on_end to route the resulting "
        "session-end into a push-to-talk handoff (chime + single-cmd session) "
        "instead of a normal continuous-streaming restart."
    ),
    value=_PTT_GLOBAL,
)

PATCH_PTT_CENTER_BUTTON = Patch(
    op="transform",
    path="/binary_sensor",
    rationale=(
        "Push-to-talk trigger: extends center_button.on_press to set "
        "button_initiated_wake=true ONLY when device is in streaming idle "
        "(no mute, no timer, no announcement, no playing media, phase==idle). "
        "Without these guards, button presses during TTS or media would "
        "spuriously arm PTT."
    ),
    transform=_append_center_button_ptt,
)

PATCH_HANDOFF_SCRIPT = Patch(
    op="append_list",
    path="/script",
    rationale=(
        "Session handoff: when HA detects a wake word in the continuous audio "
        "stream (or the center button arms PTT), this script switches voice_assistant "
        "out of continuous mode and into a single-command session. Without this "
        "handoff, the device-side event chain (on_listening / on_stt_vad_start / "
        "on_stt_vad_end / on_tts_start / on_end) does not fire reliably for sessions "
        "embedded in continuous + use_wake_word=true mode, so phase transitions get "
        "stuck and the stop word can't be enabled by base's timer_ringing path "
        "The script restores "
        "handle_streaming_wake_word semantics within the patches-only architecture."
    ),
    value=_HANDOFF_SCRIPT,
)

PATCH_ON_CLIENT_CONNECTED = Patch(
    op="append_list",
    path="/voice_assistant/on_client_connected",
    rationale=(
        "Appends streaming bootstrap after base's init handlers. delay:1s "
        "lets the API connection settle; the if-not-running guard is the "
        "idempotency guarantee against wake-word events firing during the 1s "
        "window (without it the wake-word session gets torn down)."
    ),
    expected=[
        {"lambda": "id(init_in_progress) = false;"},
        {"micro_wake_word.start": None},
        {"lambda": "id(voice_assistant_phase) = 1;"},
        {"script.execute": "control_leds"},
    ],
    value=_ON_CLIENT_CONNECTED_APPEND,
)

PATCH_ON_END = Patch(
    op="append_list",
    path="/voice_assistant/on_end",
    rationale=(
        "Appends streaming-restart after base's on_end (wait_until + unduck + "
        "phase reset). delay:500ms debounces base's settling. Triple guards "
        "prevent restart during mute, without API, or while a new session "
        "already started. PTT branch: if button_initiated_wake was set by "
        "center_button.on_press during streaming idle, route through a "
        "single-cmd handoff (chime + voice_assistant.start) instead of "
        "continuous restart. Else branch sets use_wake_word=true defensively "
        "(in case PTT handoff toggled it to false on a prior session)."
    ),
    expected=[
        {"wait_until": {"not": {"voice_assistant.is_running": None}}},
        {
            "mixer_speaker.apply_ducking": {
                "id": "media_mixing_input",
                "decibel_reduction": 0,
                "duration": "1.0s",
            }
        },
        {
            "if": {
                "condition": {
                    "lambda": "return id(voice_assistant_phase) == 11;"
                },
                "then": [{"delay": "1s"}],
            }
        },
        {"lambda": "id(voice_assistant_phase) = 1;"},
        {"script.execute": "control_leds"},
    ],
    value=_ON_END_APPEND,
)

PATCH_ON_ERROR = Patch(
    op="append_list",
    path="/voice_assistant/on_error",
    rationale=(
        "Appends streaming-restart after base's on_error (sets error phase, "
        "plays cloud-auth sound if applicable). delay:2s lets transient HA "
        "errors resolve before reconnect attempt; same triple guards. "
        "set_use_wake_word(true) defensively restores streaming mode in case "
        "PTT toggled it to false."
    ),
    expected=[
        {
            "if": {
                "condition": {
                    "and": [
                        {"lambda": "return !id(init_in_progress);"},
                        {"lambda": 'return code != "duplicate_wake_up_detected";'},
                        {"lambda": 'return code != "stt-no-text-recognized";'},
                    ]
                },
                "then": [
                    {"lambda": "id(voice_assistant_phase) = 11;"},
                    {"script.execute": "control_leds"},
                ],
            }
        },
        {
            "if": {
                "condition": [{"lambda": 'return code == "cloud-auth-failed";'}],
                "then": [
                    {
                        "script.execute": {
                            "id": "play_sound",
                            "priority": True,
                            "sound_file": "error_cloud_expired",
                        }
                    }
                ],
            }
        },
    ],
    value=_ON_ERROR_APPEND,
)

PATCH_ON_WAKE_WORD_DETECTED = Patch(
    op="set_key",
    path="/voice_assistant/on_wake_word_detected",
    rationale=(
        "voice_assistant.on_wake_word_detected fires when HA detects a wake "
        "word in the streamed audio. The chime gives audible confirmation, "
        "BUT must be suppressed when media_player is announcing (TTS) or "
        "timer is ringing to break the echo loop: chime audio -> speaker -> "
        "mic -> HA detects another 'wake word' from the chime echo -> chime "
        "fires again -> infinite loop. "
        "Also suppressed when muted (no chime if user explicitly silenced)."
    ),
    expected=MISSING,
    value=_ON_WAKE_WORD_DETECTED,
)


def _has_action_containing(actions: list, key_substring: str) -> bool:
    return any(
        any(key_substring in str(k) for k in node.keys())
        for node in _flatten_actions(actions)
    )


def _has_delay(actions: list, value: str) -> bool:
    return any(
        isinstance(n, dict) and n.get("delay") == value
        for n in _flatten_actions(actions)
    )


def _has_lambda_containing(actions: list, code_substring: str) -> bool:
    for node in _flatten_actions(actions):
        for v in node.values():
            if isinstance(v, Lambda) and code_substring in v.code:
                return True
            if isinstance(v, str) and code_substring in v:
                return True
    return False


def _restart_in_on_end(cfg: dict) -> list[dict]:
    return [
        a
        for a in _flatten_actions(cfg["voice_assistant"]["on_end"])
        if "voice_assistant.start_continuous" in a
    ]


def _has_restart_guards(cfg: dict, key: str) -> bool:
    actions = cfg["voice_assistant"].get(key, [])
    for node in actions:
        condition_text = _conditions_text(node)
        if (
            "master_mute_switch" in condition_text
            and "api.connected" in condition_text
            and "voice_assistant.is_running" in condition_text
        ):
            return True
    return False


FEATURES: list[Feature] = [
    Feature(
        id="F1_MIC_MONO",
        name="Microphone mono ch0",
        rationale="Prevents HA pipeline-restart bug from asymmetric chunks.",
        patches=(PATCH_MIC_MONO,),
        invariant=lambda cfg: cfg["voice_assistant"]["microphone"]
        == [{"microphone": "i2s_mics", "channels": 0}],
    ),
    Feature(
        id="F2_MWW_STOP_ONLY",
        name="MWW keeps only stop model",
        rationale="Drops 3 unused on-device models; preserves stop for TTS interrupt.",
        patches=(PATCH_MWW_STOP_ONLY,),
        invariant=lambda cfg: [
            m.get("id") for m in cfg["micro_wake_word"]["models"]
        ] == ["stop"],
    ),
    Feature(
        id="F3_DROP_SENSITIVITY",
        name="Drop wake_word_sensitivity select",
        rationale="Its lambda references removed model IDs; codegen failure if kept.",
        patches=(PATCH_DROP_SENSITIVITY,),
        invariant=lambda cfg: "wake_word_sensitivity"
        not in {s.get("id") for s in cfg.get("select", [])},
    ),
    Feature(
        id="F4_USE_WAKE_WORD_TRUE",
        name="voice_assistant.use_wake_word = true",
        rationale="HA does wake word detection on streamed audio.",
        patches=(PATCH_USE_WAKE_WORD,),
        invariant=lambda cfg: cfg["voice_assistant"]["use_wake_word"] is True,
    ),
    Feature(
        id="F5_PROJECT_NAMING",
        name="Project identity overrides base",
        rationale="Device appears as Voice PE Streaming in HA, not 'home-assistant-voice'.",
        patches=(PATCH_SUBSTITUTIONS_NAMING, PATCH_ESPHOME_NAMING),
        invariant=lambda cfg: (
            cfg["substitutions"].get("name") == "voice-pe-streaming"
            and cfg["substitutions"].get("friendly_name") == "Voice PE Streaming"
            and cfg["esphome"].get("name") == "voice-pe-streaming"
            and cfg["esphome"].get("friendly_name") == "Voice PE Streaming"
            and cfg["esphome"].get("project", {}).get("name") == "esphome.voice-pe-streaming"
        ),
    ),
    Feature(
        id="F6_START_CONTINUOUS_ON_CONNECT",
        name="start_continuous called on API connect",
        rationale="Without this, use_wake_word=true alone doesn't stream audio.",
        patches=(PATCH_ON_CLIENT_CONNECTED,),
        invariant=lambda cfg: _has_action_containing(
            cfg["voice_assistant"]["on_client_connected"], "voice_assistant.start_continuous"
        ),
    ),
    Feature(
        id="F7_API_SETTLE_DELAY",
        name="1s API-settle delay before start_continuous on connect",
        rationale="Gives the API connection time to settle so start_continuous doesn't race init.",
        patches=(PATCH_ON_CLIENT_CONNECTED,),
        invariant=lambda cfg: _has_delay(
            cfg["voice_assistant"]["on_client_connected"], "1s"
        ),
    ),
    Feature(
        id="F8_CONNECT_IDEMPOTENCY_GUARD",
        name="Connect-time start_continuous guarded on not running",
        rationale="Without the guard, a wake-word event firing in the 1s settle window has its session torn down by start_continuous.",
        patches=(PATCH_ON_CLIENT_CONNECTED,),
        invariant=lambda cfg: any(
            "voice_assistant.is_running" in _conditions_text(node)
            for node in cfg["voice_assistant"]["on_client_connected"]
        ),
    ),
    Feature(
        id="F9_RESTART_ON_END",
        name="on_end calls start_continuous restart",
        rationale="Without restart, device goes silent after each session.",
        patches=(PATCH_ON_END,),
        invariant=lambda cfg: len(_restart_in_on_end(cfg)) >= 1,
    ),
    Feature(
        id="F10_END_DEBOUNCE",
        name="500ms delay before on_end restart",
        rationale="Lets base's wait_until/unduck/phase-reset settle before our restart.",
        patches=(PATCH_ON_END,),
        invariant=lambda cfg: _has_delay(
            cfg["voice_assistant"]["on_end"], "500ms"
        ),
    ),
    Feature(
        id="F11_END_GUARDS",
        name="on_end restart triple-guarded",
        rationale="Prevents restart during mute, without API, or while a session is already active.",
        patches=(PATCH_ON_END,),
        invariant=lambda cfg: _has_restart_guards(cfg, "on_end"),
    ),
    Feature(
        id="F12_RESTART_ON_ERROR",
        name="on_error calls start_continuous restart",
        rationale="Without restart, transient errors brick the device until next reconnect.",
        patches=(PATCH_ON_ERROR,),
        invariant=lambda cfg: _has_action_containing(
            cfg["voice_assistant"]["on_error"], "voice_assistant.start_continuous"
        ),
    ),
    Feature(
        id="F13_ERROR_DELAY",
        name="2s delay before on_error restart",
        rationale="Transient errors may self-resolve; prevents tight restart loops on persistent failures.",
        patches=(PATCH_ON_ERROR,),
        invariant=lambda cfg: _has_delay(cfg["voice_assistant"]["on_error"], "2s"),
    ),
    Feature(
        id="F14_ERROR_GUARDS",
        name="on_error restart triple-guarded",
        rationale="Same guards as on_end for the same reasons.",
        patches=(PATCH_ON_ERROR,),
        invariant=lambda cfg: _has_restart_guards(cfg, "on_error"),
    ),
    Feature(
        id="F15_WAKE_CHIME",
        name="Wake chime on HA-side wake detection",
        rationale="Audible confirmation that the device heard the user (in handoff script or inline).",
        patches=(PATCH_HANDOFF_SCRIPT,),
        invariant=lambda cfg: "wake_word_triggered_sound"
        in (
            str(cfg.get("script", []))
            + str(cfg["voice_assistant"].get("on_wake_word_detected", []))
        ),
    ),
    Feature(
        id="F16_CHIME_SUPPRESS_ANNOUNCING",
        name="Chime suppressed during media announcement",
        rationale="Without this, chime echo from TTS triggers another wake detection -> infinite loop.",
        patches=(PATCH_ON_WAKE_WORD_DETECTED,),
        invariant=lambda cfg: any(
            "media_player.is_announcing" in _conditions_text(node)
            for node in cfg["voice_assistant"]["on_wake_word_detected"]
        ),
    ),
    Feature(
        id="F17_CHIME_SUPPRESS_TIMER",
        name="Chime suppressed during timer ringing",
        rationale="Same echo-loop concern as F16 for timer sound.",
        patches=(PATCH_ON_WAKE_WORD_DETECTED,),
        invariant=lambda cfg: any(
            "timer_ringing" in _conditions_text(node)
            for node in cfg["voice_assistant"]["on_wake_word_detected"]
        ),
    ),
    Feature(
        id="F18_CHIME_SUPPRESS_MUTED",
        name="Chime suppressed when muted",
        rationale="No chime if user explicitly silenced the device.",
        patches=(PATCH_ON_WAKE_WORD_DETECTED,),
        invariant=lambda cfg: any(
            "master_mute_switch" in _conditions_text(node)
            for node in cfg["voice_assistant"]["on_wake_word_detected"]
        ),
    ),
    Feature(
        id="F19_PTT_GLOBAL",
        name="button_initiated_wake global declared",
        rationale="Required so PTT center-button can communicate intent to on_end.",
        patches=(PATCH_PTT_GLOBAL,),
        invariant=lambda cfg: "button_initiated_wake"
        in {g.get("id") for g in cfg.get("globals", [])},
    ),
    Feature(
        id="F20_PTT_BUTTON_HANDLER",
        name="center_button.on_press sets PTT flag when streaming-idle",
        rationale="The button press path. Guards prevent PTT firing during TTS/timer/media (would interfere with active flow).",
        patches=(PATCH_PTT_CENTER_BUTTON,),
        invariant=lambda cfg: any(
            _has_lambda_containing(item.get("on_press", []), "id(button_initiated_wake) = true")
            for item in cfg.get("binary_sensor", [])
            if isinstance(item, dict) and item.get("id") == "center_button"
        ),
    ),
    Feature(
        id="F21_PTT_ON_END_BRANCH",
        name="on_end branches on PTT flag for handoff",
        rationale="Without the branch, button press just stops + restarts continuous (no chime, no single-cmd session) -- PTT feature gone.",
        patches=(PATCH_ON_END,),
        invariant=lambda cfg: "button_initiated_wake"
        in str(cfg["voice_assistant"]["on_end"]),
    ),
    Feature(
        id="F22_SESSION_HANDOFF",
        name="HA wake event triggers single-cmd session handoff",
        rationale=(
            "voice_assistant in continuous + use_wake_word=true mode does NOT "
            "drive the device-side event chain (on_listening/on_stt_vad_*/on_tts_*/on_end) "
            "reliably for sessions embedded in continuous mode. The "
            "failure mode: phase stuck in listening, stop word inert in timer, LED stuck "
            "in replying when timer set. Fix: on HA wake detection, transition to "
            "single-cmd mode (stop continuous, set_use_wake_word(false), voice_assistant.start) "
            "so events fire normally; on_end restores continuous + use_wake_word(true)."
        ),
        patches=(PATCH_HANDOFF_SCRIPT, PATCH_ON_WAKE_WORD_DETECTED),
        invariant=lambda cfg: (
            any(
                isinstance(s, dict) and s.get("id") == "handle_session_handoff"
                for s in cfg.get("script", [])
            )
            and any(
                _has_action_containing(s.get("then", []), "voice_assistant.stop")
                and _has_action_containing(s.get("then", []), "voice_assistant.start")
                and _has_lambda_containing(s.get("then", []), "set_use_wake_word(false)")
                for s in cfg.get("script", [])
                if isinstance(s, dict) and s.get("id") == "handle_session_handoff"
            )
            and any(
                isinstance(node.get("script.execute"), str)
                and node["script.execute"] == "handle_session_handoff"
                for node in _flatten_actions(cfg["voice_assistant"]["on_wake_word_detected"])
            )
        ),
    ),
]


def _dedupe_preserve_order(patches: list[Patch]) -> list[Patch]:
    seen: set[int] = set()
    out: list[Patch] = []
    for p in patches:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


PRODUCTION_PATCHES: list[Patch] = _dedupe_preserve_order(
    [p for f in FEATURES for p in f.patches]
)
