"""Voice Assistant component schema.

Based on ESPHome voice_assistant/__init__.py lines 89-177.
"""

import voluptuous as vol
from esphome_yaml_tools.validator import (
    Optional,
    Required,
    VolAny,
    Schema,
    string,
    boolean,
)

# =============================================================================
# Voice Assistant Schema
# =============================================================================

_VOICE_ASSISTANT_BLOCK_SCHEMA = Schema(
    {
        Optional("id"): string,
        Required("microphone"): string,  # ID reference to microphone component
        Optional("speaker"): string,  # ID reference to speaker component
        Optional("media_player"): string,  # ID reference to media player
        Optional("noise_suppression_level"): vol.Coerce(int),
        Optional("auto_gain"): vol.Coerce(int),
        Optional("volume_multiplier"): vol.Coerce(float),
        Optional("use_wake_word"): boolean,
        Optional("vad_threshold"): vol.Coerce(int),
        # Event handlers
        Optional("on_listening"): list,
        Optional("on_start"): list,
        Optional("on_wake_word_detected"): list,
        Optional("on_stt_vad_start"): list,
        Optional("on_stt_vad_end"): list,
        Optional("on_stt_end"): list,
        Optional("on_tts_start"): list,
        Optional("on_tts_stream_start"): list,
        Optional("on_tts_stream_end"): list,
        Optional("on_tts_end"): list,
        Optional("on_end"): list,
        Optional("on_error"): list,
        Optional("on_client_connected"): list,
        Optional("on_client_disconnected"): list,
        Optional("on_intent_start"): list,
        Optional("on_intent_end"): list,
        Optional("on_intent_progress"): list,
        # Timer events
        Optional("on_timer_started"): list,
        Optional("on_timer_updated"): list,
        Optional("on_timer_cancelled"): list,
        Optional("on_timer_finished"): list,
        Optional("on_timer_tick"): list,
        # Announce event
        Optional("on_announce"): list,
    },
    extra=vol.ALLOW_EXTRA,
)

VOICE_ASSISTANT_SCHEMA = Schema(
    {
        Optional("voice_assistant"): _VOICE_ASSISTANT_BLOCK_SCHEMA,
    }
)
