"""Tests for the firmware generator engine.

These are unit tests for `esphome_yaml_tools.firmware_generator` covering:
- `merge_config` matches ESPHome's package-merge semantics
- `resolve_extend_remove` resolves !extend/!remove markers correctly
- `apply_patches` enforces `expected`-value safety checks
- `BasePackage.load` rejects content with mismatched SHA-256
- `dump_yaml` preserves !lambda tags and refuses unresolved markers
- End-to-end: generate_firmware on the real overlay + base + patch produces
  a YAML that the existing firmware test suite already exercises (no separate
  re-verification needed here)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from esphome_yaml_tools.firmware_generator import (
    BasePackage,
    Patch,
    PatchError,
    apply_patches,
    dump_yaml,
    merge_config,
    resolve_extend_remove,
)
from esphome_yaml_tools.parser import Extend, Lambda, Remove


class TestMergeConfig:
    """Behaviour must match `esphome/config_helpers.py::merge_config`."""

    def test_dict_recursive_merge(self):
        old = {"a": 1, "b": {"c": 2, "d": 3}}
        new = {"b": {"c": 99}, "e": 4}
        assert merge_config(old, new) == {"a": 1, "b": {"c": 99, "d": 3}, "e": 4}

    def test_list_concatenates_not_replaces(self):
        old = [1, 2]
        new = [3, 4]
        assert merge_config(old, new) == [1, 2, 3, 4]

    def test_list_concat_at_nested_level(self):
        old = {"x": [1, 2]}
        new = {"x": [3]}
        assert merge_config(old, new) == {"x": [1, 2, 3]}

    def test_remove_sentinel_propagates(self):
        old = {"a": 1, "b": 2}
        new = {"a": Remove()}
        merged = merge_config(old, new)
        assert isinstance(merged["a"], Remove)
        assert merged["b"] == 2

    def test_remove_in_new_replaces_old_list(self):
        old = [1, 2, 3]
        new = Remove()
        assert merge_config(old, new) is new

    def test_none_in_new_keeps_old(self):
        assert merge_config(5, None) == 5

    def test_scalar_new_replaces_old(self):
        assert merge_config("old_value", "new_value") == "new_value"

    def test_type_mismatch_new_wins(self):
        assert merge_config({"a": 1}, [1, 2]) == [1, 2]
        assert merge_config([1, 2], {"a": 1}) == {"a": 1}


class TestResolveExtendRemove:
    def test_extend_merges_into_matching_id(self):
        config = {
            "switch": [
                {"id": "foo", "name": "Foo", "on_press": [{"action": "base"}]},
                {"id": Extend("foo"), "on_press": [{"action": "overlay"}]},
            ]
        }
        resolve_extend_remove(config)
        assert config["switch"] == [
            {
                "id": "foo",
                "name": "Foo",
                "on_press": [{"action": "base"}, {"action": "overlay"}],
            }
        ]

    def test_extend_with_missing_id_raises(self):
        config = {"switch": [{"id": Extend("nonexistent"), "on_press": [{"a": "x"}]}]}
        with pytest.raises(PatchError, match="not found in base list"):
            resolve_extend_remove(config)

    def test_remove_drops_dict_key(self):
        config = {"foo": Remove(), "bar": 1}
        resolve_extend_remove(config)
        assert config == {"bar": 1}

    def test_remove_with_id_drops_list_item(self):
        config = {
            "switch": [
                {"id": "keep_me", "x": 1},
                {"id": "drop_me", "x": 2},
                {"id": Remove("drop_me")},
            ]
        }
        resolve_extend_remove(config)
        ids = [s.get("id") for s in config["switch"]]
        assert ids == ["keep_me"]

    def test_recurses_into_nested(self):
        config = {
            "voice_assistant": {
                "microphone": [
                    {"id": "src1", "channels": 0},
                    {"id": Extend("src1"), "gain_factor": 4},
                ]
            }
        }
        resolve_extend_remove(config)
        assert config["voice_assistant"]["microphone"] == [
            {"id": "src1", "channels": 0, "gain_factor": 4}
        ]


class TestApplyPatches:
    def test_replace_with_matching_expected_succeeds(self):
        config = {"a": {"b": [1, 2]}}
        patch = Patch(
            op="replace",
            path="/a/b",
            rationale="test",
            expected=[1, 2],
            value=[3],
        )
        apply_patches(config, [patch])
        assert config == {"a": {"b": [3]}}

    def test_replace_with_mismatched_expected_raises(self):
        config = {"a": {"b": [1, 2]}}
        patch = Patch(
            op="replace",
            path="/a/b",
            rationale="test rationale",
            expected=[1, 2, 3],
            value=[3],
        )
        with pytest.raises(PatchError, match="test rationale"):
            apply_patches(config, [patch])

    def test_replace_mismatch_message_includes_path_and_diff(self):
        config = {"a": "actual"}
        patch = Patch(op="replace", path="/a", rationale="r", expected="expected", value="new")
        with pytest.raises(PatchError) as exc_info:
            apply_patches(config, [patch])
        msg = str(exc_info.value)
        assert "/a" in msg
        assert "expected" in msg
        assert "actual" in msg

    def test_transform_applies_function(self):
        config = {"items": [1, 2, 3, 4]}
        patch = Patch(
            op="transform",
            path="/items",
            rationale="keep evens",
            transform=lambda lst: [x for x in lst if x % 2 == 0],
        )
        apply_patches(config, [patch])
        assert config == {"items": [2, 4]}

    def test_invalid_path_key_raises(self):
        config = {"a": 1}
        patch = Patch(op="replace", path="/nonexistent", rationale="r", expected=None, value=None)
        with pytest.raises(PatchError, match="not found"):
            apply_patches(config, [patch])

    def test_invalid_path_list_index_raises(self):
        config = {"a": [1, 2]}
        patch = Patch(op="replace", path="/a/99", rationale="r", expected=None, value=None)
        with pytest.raises(PatchError, match="out of range"):
            apply_patches(config, [patch])

    def test_unknown_op_raises(self):
        config = {"a": 1}
        patch = Patch(op="explode", path="/a", rationale="r")
        with pytest.raises(PatchError, match="unknown op"):
            apply_patches(config, [patch])

    def test_patches_run_in_order(self):
        config = {"x": 1}
        patches = [
            Patch(op="replace", path="/x", rationale="step1", expected=1, value=2),
            Patch(op="replace", path="/x", rationale="step2", expected=2, value=3),
        ]
        apply_patches(config, patches)
        assert config == {"x": 3}

    def test_patches_failure_message_shows_position(self):
        config = {"x": 1}
        patches = [
            Patch(op="replace", path="/x", rationale="ok", expected=1, value=2),
            Patch(op="replace", path="/x", rationale="bad", expected=99, value=3),
        ]
        with pytest.raises(PatchError, match=r"Patch 2/2"):
            apply_patches(config, patches)


class TestBasePackage:
    def test_load_from_cache_succeeds_when_sha_matches(self, tmp_path):
        content = "test content"
        sha = hashlib.sha256(content.encode()).hexdigest()
        cache = tmp_path / "cached.yaml"
        cache.write_text(content)
        bp = BasePackage(url="https://example/x", revision="abc", sha256=sha, cache_path=cache)
        assert bp.load() == content

    def test_load_rejects_mismatched_sha(self, tmp_path):
        content = "test content"
        cache = tmp_path / "cached.yaml"
        cache.write_text(content)
        bp = BasePackage(
            url="https://example/x",
            revision="abc",
            sha256="0" * 64,
            cache_path=cache,
        )
        with pytest.raises(PatchError, match="SHA-256 mismatch"):
            bp.load()

    def test_load_message_includes_url_and_revision(self, tmp_path):
        cache = tmp_path / "cached.yaml"
        cache.write_text("any")
        bp = BasePackage(
            url="https://my/specific/url",
            revision="my-revision",
            sha256="0" * 64,
            cache_path=cache,
        )
        with pytest.raises(PatchError) as exc:
            bp.load()
        msg = str(exc.value)
        assert "https://my/specific/url" in msg
        assert "my-revision" in msg


class TestDumper:
    def test_lambda_emitted_as_tag(self):
        config = {"action": Lambda("return id(foo);")}
        out = dump_yaml(config)
        assert "!lambda" in out
        assert "return id(foo);" in out

    def test_unresolved_extend_raises(self):
        config = {"id": Extend("foo")}
        with pytest.raises(PatchError, match="!extend"):
            dump_yaml(config)

    def test_unresolved_remove_raises(self):
        config = {"foo": Remove()}
        with pytest.raises(PatchError, match="!remove"):
            dump_yaml(config)

    def test_roundtrip_preserves_lambdas(self):
        from esphome_yaml_tools.parser import load_yaml

        original = {"script": [{"id": "x", "then": [{"lambda": Lambda("id(y) = 1;")}]}]}
        text = dump_yaml(original)
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(text)
            tmp = Path(f.name)
        try:
            reloaded = load_yaml(tmp)
        finally:
            tmp.unlink()
        assert isinstance(reloaded["script"][0]["then"][0]["lambda"], Lambda)
        assert reloaded["script"][0]["then"][0]["lambda"].code == "id(y) = 1;"


class TestRealGeneration:
    """Smoke tests: the real generator config (cached base + overlay + patches)
    produces a YAML the simulator can load and the firmware tests already
    exercise."""

    def test_generated_file_exists(self):
        from voice_pe_streaming import patches

        assert patches.OUTPUT_PATH.exists(), (
            "voice-pe-streaming.yaml must be checked in (run scripts/generate_firmware.py)"
        )

    def test_generated_file_starts_with_header(self):
        from voice_pe_streaming import patches

        text = patches.OUTPUT_PATH.read_text(encoding="utf-8")
        assert text.startswith("# AUTO-GENERATED")

    def test_microphone_patch_applied(self):
        from voice_pe_streaming import patches
        from esphome_yaml_tools.parser import load_yaml

        config = load_yaml(patches.OUTPUT_PATH)
        mic = config["voice_assistant"]["microphone"]
        assert len(mic) == 1, f"Expected single mic source post-patch, got {mic!r}"
        assert mic[0] == {"microphone": "i2s_mics", "channels": 0}

    def test_micro_wake_word_still_uses_channel_1(self):
        """The fix removes ch1 from voice_assistant only; mww must keep ch1 for the stop word."""
        from voice_pe_streaming import patches
        from esphome_yaml_tools.parser import load_yaml

        config = load_yaml(patches.OUTPUT_PATH)
        mww_mic = config["micro_wake_word"]["microphone"]
        assert mww_mic["channels"] == 1
        assert mww_mic["microphone"] == "i2s_mics"

    def test_base_sha_matches_cached_file(self):
        from voice_pe_streaming import patches

        actual = hashlib.sha256(
            patches.BASE.cache_path.read_bytes()
        ).hexdigest()
        assert actual == patches.BASE.sha256, (
            f"Cached base SHA-256 ({actual}) does not match patches.BASE.sha256. "
            "Either upstream was refreshed without updating the pin, or someone "
            "hand-edited base_pinned.yaml."
        )

    def test_patches_expected_values_still_match_cached_base(self):
        """The generator's safety checks must succeed on the cached base.
        If this fails, upstream changed and the patches need a manual review."""
        from voice_pe_streaming import patches
        from esphome_yaml_tools.firmware_generator import generate_firmware

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            generate_firmware(
                base=patches.BASE,
                patches=patches.PATCHES,
                output_path=tmp,
            )
        finally:
            tmp.unlink(missing_ok=True)
