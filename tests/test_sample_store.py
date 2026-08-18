"""Tests for SampleStore filter / retention semantics."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from wyoming_livekit_speechbrain.sample_store import SampleMeta, SampleStore


def _make_store(tmpdir: Path, max_samples: int = 1000) -> SampleStore:
    return SampleStore(samples_dir=tmpdir, max_samples=max_samples)


def _add(store: SampleStore, label, keyword="sputnik", source="auto"):
    pcm = b"\x00\x00" * 16000
    return store.add(
        pcm,
        SampleMeta(
            id="",
            timestamp=time.time(),
            keyword=keyword,
            score=0.5,
            speaker=None,
            accepted=label is None or label == "tp",
            source=source,
            label=label,
        ),
    )


class TestListFilter:
    def test_unlabeled_filter_matches_label_None_samples(self, tmp_path):
        store = _make_store(tmp_path)
        m1 = _add(store, label=None)
        m2 = _add(store, label="fp")
        m3 = _add(store, label="tp")
        m4 = _add(store, label=None)

        result = [m.id for m in store.list(label="unlabeled")]
        assert m1.id in result
        assert m4.id in result
        assert m2.id not in result
        assert m3.id not in result
        assert len(result) == 2

    def test_fp_filter_matches_label_fp_only(self, tmp_path):
        store = _make_store(tmp_path)
        _add(store, label=None)
        m_fp = _add(store, label="fp")
        _add(store, label="tp")

        result = [m.id for m in store.list(label="fp")]
        assert result == [m_fp.id]

    def test_tp_filter_matches_label_tp_only(self, tmp_path):
        store = _make_store(tmp_path)
        _add(store, label=None)
        _add(store, label="fp")
        m_tp = _add(store, label="tp")

        result = [m.id for m in store.list(label="tp")]
        assert result == [m_tp.id]

    def test_all_filter_returns_everything(self, tmp_path):
        store = _make_store(tmp_path)
        _add(store, label=None)
        _add(store, label="fp")
        _add(store, label="tp")

        assert len(store.list(label="all")) == 3
        assert len(store.list(label=None)) == 3

    def test_none_filter_returns_everything(self, tmp_path):
        store = _make_store(tmp_path)
        _add(store, label="fp")
        _add(store, label=None)
        assert len(store.list()) == 2


class TestBulkLabelViaUpdate:
    def test_update_label_then_filter_unlabeled(self, tmp_path):
        store = _make_store(tmp_path)
        m = _add(store, label=None)

        assert m.id in [s.id for s in store.list(label="unlabeled")]
        store.update_label(m.id, "fp")
        assert m.id not in [s.id for s in store.list(label="unlabeled")]
        assert m.id in [s.id for s in store.list(label="fp")]
        store.update_label(m.id, None)
        assert m.id in [s.id for s in store.list(label="unlabeled")]


class TestPerSourceRetention:
    def test_chatty_source_does_not_evict_other_sources(self, tmp_path):
        """auto-rejected-speaker firing thousands of times must NOT evict rare 'auto' TPs.

        Regression test for a retention bug: when global max_samples=1000
        was hit by a flood of speaker-rej samples in a noisy environment, older
        accepted-trigger samples (the actual TP fine-tune data) got evicted.
        """
        store = _make_store(tmp_path, max_samples=5)
        rare_ids = [_add(store, label="tp", source="auto").id for _ in range(3)]
        for _ in range(20):
            _add(store, label=None, source="auto-rejected-speaker")

        surviving = {m.id for m in store.list()}
        for rid in rare_ids:
            assert rid in surviving, "rare 'auto' TP got evicted by speaker-rej flood"

        rejected = [m for m in store.list() if m.source == "auto-rejected-speaker"]
        assert len(rejected) == 5

    def test_quota_applied_per_source_independently(self, tmp_path):
        store = _make_store(tmp_path, max_samples=3)
        for _ in range(10):
            _add(store, label=None, source="manual")
        for _ in range(10):
            _add(store, label=None, source="auto")
        for _ in range(10):
            _add(store, label=None, source="auto-rejected-speaker")

        by_source: dict[str, int] = {}
        for m in store.list():
            by_source[m.source] = by_source.get(m.source, 0) + 1
        assert by_source == {"manual": 3, "auto": 3, "auto-rejected-speaker": 3}

    def test_within_source_newest_kept(self, tmp_path):
        store = _make_store(tmp_path, max_samples=2)
        old1 = _add(store, label=None, source="auto", keyword="first")
        time.sleep(0.01)
        old2 = _add(store, label=None, source="auto", keyword="second")
        time.sleep(0.01)
        new1 = _add(store, label=None, source="auto", keyword="third")
        time.sleep(0.01)
        new2 = _add(store, label=None, source="auto", keyword="fourth")

        surviving = {m.id for m in store.list()}
        assert new1.id in surviving
        assert new2.id in surviving
        assert old1.id not in surviving
        assert old2.id not in surviving

    def test_single_source_backward_compat(self, tmp_path):
        """When only one source is used, per-source retention behaves like global."""
        store = _make_store(tmp_path, max_samples=5)
        for _ in range(10):
            _add(store, label=None, source="auto")
        assert len(store.list()) == 5

    def test_max_samples_zero_disables_retention(self, tmp_path):
        store = _make_store(tmp_path, max_samples=0)
        for _ in range(10):
            _add(store, label=None, source="auto")
        for _ in range(10):
            _add(store, label=None, source="auto-rejected-speaker")
        assert len(store.list()) == 20


class TestPerSourceQuotaOverrides:
    def test_chatty_source_keeps_more_than_default(self, tmp_path):
        """200k cap for auto-rejected-speaker, 1k for everything else.

        A high-volume setting: big bucket for the FP-firehose,
        small bucket for the rare-but-precious accepted triggers.
        """
        store = SampleStore(
            samples_dir=tmp_path,
            max_samples=5,
            max_samples_per_source={"auto-rejected-speaker": 20},
        )
        for _ in range(30):
            _add(store, label=None, source="auto-rejected-speaker")
        for _ in range(30):
            _add(store, label=None, source="auto")

        by_source: dict[str, int] = {}
        for m in store.list():
            by_source[m.source] = by_source.get(m.source, 0) + 1
        assert by_source == {"auto-rejected-speaker": 20, "auto": 5}

    def test_unspecified_source_falls_back_to_default(self, tmp_path):
        store = SampleStore(
            samples_dir=tmp_path,
            max_samples=3,
            max_samples_per_source={"auto-rejected-speaker": 100},
        )
        for _ in range(10):
            _add(store, label=None, source="manual")
        assert len([m for m in store.list() if m.source == "manual"]) == 3

    def test_per_source_quota_zero_means_unlimited_for_that_source(self, tmp_path):
        store = SampleStore(
            samples_dir=tmp_path,
            max_samples=3,
            max_samples_per_source={"auto": 0},
        )
        for _ in range(10):
            _add(store, label=None, source="auto")
        for _ in range(10):
            _add(store, label=None, source="manual")
        assert len([m for m in store.list() if m.source == "auto"]) == 10
        assert len([m for m in store.list() if m.source == "manual"]) == 3

    def test_per_source_overrides_when_global_is_disabled(self, tmp_path):
        """max_samples=0 globally + explicit per-source still enforces that source's quota."""
        store = SampleStore(
            samples_dir=tmp_path,
            max_samples=0,
            max_samples_per_source={"auto-rejected-speaker": 5},
        )
        for _ in range(20):
            _add(store, label=None, source="auto-rejected-speaker")
        for _ in range(20):
            _add(store, label=None, source="auto")
        assert len([m for m in store.list() if m.source == "auto-rejected-speaker"]) == 5
        assert len([m for m in store.list() if m.source == "auto"]) == 20
