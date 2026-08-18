"""Structural tests for the streaming-only Voice PE firmware.

The streaming-only commitment removed the on-device/
streaming mode-toggle machinery: `use_streaming_wake_word` switch, 6
mode-machine scripts, `in_mode_transition`/`button_initiated_wake` globals,
`center_button.on_press` and `master_mute_switch.on_turn_off` `!extend` blocks.
What remains:
- `overlay.yaml` contributes: substitutions, esphome name/project,
  voice_assistant.on_client_connected (single `voice_assistant.start_continuous`),
  voice_assistant.on_end (inline restart if mute off + api connected + not
  running), voice_assistant.on_error (2s delay then inline restart).
- `patches.py` applies 4 surgical replacements: microphone mono, mww models =
  [stop only], select drops wake_word_sensitivity, voice_assistant.use_wake_word
  flipped to true.
"""

from pathlib import Path

import pytest

from esphome_yaml_tools.parser import Extend, load_yaml


FIRMWARE_PATH = Path(__file__).parent.parent / "voice_pe_streaming" / "voice-pe-streaming.yaml"
OVERLAY_PATH = Path(__file__).parent.parent / "voice_pe_streaming" / "overlay.yaml"


@pytest.fixture(scope="module")
def config():
    return load_yaml(FIRMWARE_PATH)


def _flatten_actions(root):
    out = []

    def walk(node):
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


class TestFirmwareStructure:
    def test_firmware_file_exists(self):
        assert FIRMWARE_PATH.exists()

    def test_has_substitutions(self, config):
        assert config["substitutions"]["name"] == "voice-pe-streaming"

    def test_generated_file_has_pin_header(self):
        text = FIRMWARE_PATH.read_text(encoding="utf-8")
        head = text.split("\n", 10)[:10]
        assert any("AUTO-GENERATED" in line for line in head), (
            "Generated firmware must declare itself AUTO-GENERATED to prevent hand-edits"
        )
        assert any("revision " in line for line in head)
        assert any("sha256" in line for line in head)

    def test_pin_points_to_voice_pe_repo(self):
        from voice_pe_streaming import patches
        import re

        assert "home-assistant-voice-pe" in patches.BASE.url
        assert re.fullmatch(r"[0-9a-f]{7,40}", patches.BASE.revision)
        assert re.fullmatch(r"[0-9a-f]{64}", patches.BASE.sha256)


class TestPatchEffects:
    """Each patch in patches.py should leave a visible, asserted fingerprint in
    the generated YAML. If a patch silently no-ops on a future upstream, these
    tests catch it."""

    def test_microphone_is_mono_channel_zero(self, config):
        mic = config["voice_assistant"]["microphone"]
        assert mic == [{"microphone": "i2s_mics", "channels": 0}]

    def test_mww_keeps_only_stop_model(self, config):
        ids = [m.get("id") for m in config["micro_wake_word"]["models"]]
        assert ids == ["stop"], (
            f"Streaming-only: only the internal 'stop' model is kept, got {ids}"
        )

    def test_mww_stop_model_is_internal(self, config):
        stop = next(m for m in config["micro_wake_word"]["models"] if m.get("id") == "stop")
        assert stop.get("internal") is True, (
            "stop model must be internal so HA never sees it as a wake word; it's "
            "armed only briefly via activate_stop_word_once during TTS/timer ring"
        )

    def test_mww_still_uses_channel_1(self, config):
        mww_mic = config["micro_wake_word"]["microphone"]
        assert mww_mic == {"microphone": "i2s_mics", "channels": 1, "gain_factor": 4}, (
            "mww must keep ch1 with gain_factor:4 for the stop model; only voice_assistant "
            "was forced to mono ch0"
        )

    def test_select_does_not_contain_wake_word_sensitivity(self, config):
        ids = [s.get("id") for s in config.get("select", [])]
        assert "wake_word_sensitivity" not in ids, (
            "wake_word_sensitivity references id(okay_nabu)/id(hey_jarvis)/id(hey_mycroft) "
            "in its on_value lambda; with those models removed by patches, the lambda "
            "fails ESPHome's codegen 'id does not exist' validation"
        )

    def test_voice_assistant_use_wake_word_is_true(self, config):
        assert config["voice_assistant"]["use_wake_word"] is True, (
            "Streaming-only requires HA-side wake word detection; without use_wake_word=true "
            "voice_assistant.start_continuous streams audio but HA never triggers a session"
        )


class TestOverlayContributions:
    def test_on_client_connected_triggers_start_continuous(self, config):
        actions = _flatten_actions(config["voice_assistant"].get("on_client_connected", []))
        has_start = any(
            "voice_assistant.start_continuous" in a for a in actions
        )
        assert has_start, (
            "Without voice_assistant.start_continuous on connect, use_wake_word=true alone "
            "does not start the audio stream to HA"
        )

    def test_on_end_restarts_continuous_when_preconditions_hold(self, config):
        on_end = config["voice_assistant"].get("on_end", [])
        actions = _flatten_actions(on_end)
        has_restart = any("voice_assistant.start_continuous" in a for a in actions)
        assert has_restart, (
            "Streaming sessions end after each command; without on_end restart the device "
            "would go silent until the next API reconnect"
        )

    def test_on_end_guards_on_mute_and_api_and_not_running(self, config):
        on_end = config["voice_assistant"].get("on_end", [])
        joined_conditions = " ".join(
            str(a.get("if", {}).get("condition", ""))
            for a in on_end
            if isinstance(a, dict)
        )
        for guard in ("master_mute_switch", "api.connected", "voice_assistant.is_running"):
            assert guard in joined_conditions, (
                f"on_end restart must guard on {guard!r} -- otherwise we either restart "
                f"during mute (wasted CPU), without API (will fail), or while the next "
                f"session already started (double-trigger)"
            )

    def test_on_error_uses_two_second_delay_before_restart(self, config):
        on_error = config["voice_assistant"].get("on_error", [])
        delay = next(
            (a for a in on_error if isinstance(a, dict) and "delay" in a),
            None,
        )
        assert delay is not None and delay["delay"] == "2s", (
            "on_error must wait 2s before restart so transient errors (network blip, HA "
            "pipeline race) don't cause tight restart loops; if you change this, also "
            "update the rationale + the buggy_009_error_delay fixture"
        )

    def test_overlay_file_must_not_exist(self):
        """Patches-only architecture : every change is a Patch in
        voice_pe_streaming/features.py. If someone reintroduces an overlay
        file, the generator would silently re-merge it on top of base, breaking
        the single-source-of-truth contract."""
        assert not OVERLAY_PATH.exists(), (
            f"overlay.yaml must not exist; all changes belong in features.py.\n"
            f"Found at: {OVERLAY_PATH}"
        )

    def test_no_state_machine_patches_in_features_catalog(self):
        """Streaming-only commitment guard: the FEATURES list must not contain
        patches whose path targets removed state-machine constructs. If someone
        re-adds the use_streaming_wake_word switch or the mode-dispatch scripts,
        this catches it before the patches reach base."""
        from voice_pe_streaming.features import FEATURES  # noqa: F401

        forbidden_substrings = (
            "use_streaming_wake_word",
            "apply_wake_word_mode",
            "start_streaming_wake_word_mode",
            "handle_streaming_wake_word",
            "start_ondevice_wake_word_mode",
            "streaming_restart_after_va_end",
            "streaming_restart_after_va_error",
            "in_mode_transition",
        )
        for feature in FEATURES:
            for patch in feature.patches:
                for needle in forbidden_substrings:
                    assert needle not in patch.path, (
                        f"Feature {feature.id} has patch targeting forbidden path "
                        f"containing {needle!r}: {patch.path}"
                    )
