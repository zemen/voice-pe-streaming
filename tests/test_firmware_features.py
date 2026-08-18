"""Per-feature invariant tests + per-buggy-variant isolation tests.

This is the regression guarantee that was missing from the old buggy_*.yaml
fixtures. Each Feature in `voice_pe_streaming.features` has an invariant; each
BuggyVariant mutates the production patch list to break exactly one feature.
Tests:
    - TestFeatures: every feature's invariant must hold on production config.
    - TestBuggyVariants: for each variant, the targeted feature's invariant
      must fail AND every other feature's invariant must still hold (isolation).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from esphome_yaml_tools.firmware_generator import (
    PatchError,
    generate_firmware,
)
from esphome_yaml_tools.parser import load_yaml
from voice_pe_streaming import patches
from voice_pe_streaming.features import FEATURES, PRODUCTION_PATCHES
from voice_pe_streaming.features_buggy import BUGGY_VARIANTS


@pytest.fixture(scope="module")
def production_config() -> dict:
    return load_yaml(patches.OUTPUT_PATH)


def _generate_to_temp(patches_list) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp = Path(f.name)
    try:
        generate_firmware(
            base=patches.BASE,
            patches=patches_list,
            output_path=tmp,
        )
        return load_yaml(tmp)
    finally:
        tmp.unlink(missing_ok=True)


class TestProductionPatchesNoDup:
    def test_no_duplicate_patches_in_production(self):
        assert len(PRODUCTION_PATCHES) == len(set(id(p) for p in PRODUCTION_PATCHES)), (
            "PRODUCTION_PATCHES contains duplicate patch instances; dedup failed"
        )


class TestFeatures:
    @pytest.mark.parametrize("feature", FEATURES, ids=lambda f: f.id)
    def test_invariant_holds_on_production(self, feature, production_config):
        assert feature.invariant(production_config), (
            f"Feature {feature.id} ({feature.name}) invariant FAILED on production config.\n"
            f"Rationale: {feature.rationale}"
        )


class TestBuggyVariants:
    @pytest.mark.parametrize(
        "variant", BUGGY_VARIANTS, ids=lambda v: f"{v.feature_id}__{v.name}"
    )
    def test_variant_breaks_only_targeted_feature(self, variant):
        broken_patches = variant.mutate(list(PRODUCTION_PATCHES))
        try:
            broken_config = _generate_to_temp(broken_patches)
        except PatchError as e:
            pytest.fail(
                f"Buggy variant {variant.name} produced a generation error rather than "
                f"a clean feature regression. The mutation should produce a VALID firmware "
                f"that simply lacks the targeted feature. Error:\n{e}"
            )

        target = next(f for f in FEATURES if f.id == variant.feature_id)
        assert not target.invariant(broken_config), (
            f"Buggy variant {variant.name} targets {variant.feature_id} but the "
            f"target's invariant still passes -- the mutation failed to break the feature."
        )

        for other in FEATURES:
            if other.id == variant.feature_id:
                continue
            if other.id in variant.expected_collateral:
                continue
            assert other.invariant(broken_config), (
                f"Buggy variant {variant.name} broke unrelated feature {other.id} "
                f"({other.name}). Either the mutation is too coarse, or declare "
                f"'{other.id}' in expected_collateral for this variant."
            )


class TestEveryFeatureHasAtLeastOneBuggyVariant:
    def test_coverage(self):
        targeted = {v.feature_id for v in BUGGY_VARIANTS}
        all_ids = {f.id for f in FEATURES}
        missing = all_ids - targeted
        assert not missing, (
            f"Features without buggy variants: {missing}. Every feature must have at "
            f"least one BuggyVariant proving the invariant catches its regression."
        )
