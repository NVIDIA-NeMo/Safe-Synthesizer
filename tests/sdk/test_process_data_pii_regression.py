# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for process_data behavior when PII replacement is enabled.

Covers two invariants:

1. ``training.csv`` and the evaluator reference must use the **original**
   (pre-PII) training data so that privacy metrics remain valid.
2. ``train()`` and ``run()`` must raise after ``load_from_save_path()``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

MODULE = "nemo_safe_synthesizer.sdk.library_builder"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_original_df() -> pd.DataFrame:
    """50-row DataFrame with recognisable content."""
    return pd.DataFrame({"name": [f"Person_{i}" for i in range(50)], "value": range(50)})


def _make_replaced_df() -> pd.DataFrame:
    """Same shape, obviously different content."""
    return pd.DataFrame({"name": [f"ANON_{i}" for i in range(50)], "value": range(50)})


def _patch_heavy_deps(monkeypatch, original_df: pd.DataFrame, replaced_df: pd.DataFrame) -> None:
    """Patch Holdout, NemoPII, AutoConfigResolver, and ModelMetadata so
    ``process_data`` can run without GPUs or real models.
    """
    # Holdout: pass through without splitting; return original + empty test
    mock_holdout_instance = MagicMock()
    mock_holdout_instance.train_test_split.return_value = (
        original_df.copy(),
        pd.DataFrame(columns=original_df.columns),
    )
    monkeypatch.setattr(f"{MODULE}.Holdout", lambda config: mock_holdout_instance)

    # NemoPII: simulate PII replacement producing a different DataFrame
    mock_pii = MagicMock()
    mock_pii.result = SimpleNamespace(
        transformed_df=replaced_df.copy(),
        column_statistics={"name": {}},
    )
    mock_pii.elapsed_time = 0.1
    monkeypatch.setattr(f"{MODULE}.NemoPII", lambda config: mock_pii)

    # AutoConfigResolver: identity
    monkeypatch.setattr(f"{MODULE}.AutoConfigResolver", lambda df, config: lambda: config)

    # ModelMetadata: stub
    monkeypatch.setattr(
        f"{MODULE}.ModelMetadata",
        type("MM", (), {"from_config": staticmethod(lambda c, workdir: None)}),
    )


# ---------------------------------------------------------------------------
# PII replacement: original data must survive for evaluation
# ---------------------------------------------------------------------------


class TestProcessDataPreservesOriginalForEvaluation:
    """After PII replacement, the original (pre-replacement) training data
    must be what lands in ``training.csv`` and what the evaluator receives.
    """

    @pytest.fixture
    def builder_after_process_data(
        self, tmp_path: Path, monkeypatch
    ) -> tuple[SafeSynthesizer, pd.DataFrame, pd.DataFrame]:
        original = _make_original_df()
        replaced = _make_replaced_df()
        _patch_heavy_deps(monkeypatch, original, replaced)

        builder = (
            SafeSynthesizer(save_path=tmp_path)
            .with_data_source(original)
            .with_replace_pii(config=PiiReplacerConfig.get_default_config())
            .synthesize()
            .resolve()
        )
        builder.process_data()
        return builder, original, replaced

    def test_training_csv_contains_original_data(self, builder_after_process_data):
        """training.csv should persist the original training split, not the
        PII-replaced version, so that ``load_from_save_path`` can reload the
        canonical data for evaluation.
        """
        builder, original, _replaced = builder_after_process_data

        saved = pd.read_csv(builder._workdir.dataset.training)
        pd.testing.assert_frame_equal(saved, original, check_dtype=False)

    def test_original_train_df_is_preserved(self, builder_after_process_data):
        """``_original_train_df`` should hold the pre-PII data that
        ``evaluate()`` passes to the Evaluator.
        """
        builder, original, _replaced = builder_after_process_data

        pd.testing.assert_frame_equal(builder._original_train_df, original)

    def test_train_df_is_pii_replaced(self, builder_after_process_data):
        """``_train_df`` (used for training) should hold the PII-replaced
        version after process_data runs with PII enabled.
        """
        builder, _original, replaced = builder_after_process_data

        pd.testing.assert_frame_equal(builder._train_df, replaced)


# ---------------------------------------------------------------------------
# Guard: train()/run() must not be called after load_from_save_path()
# ---------------------------------------------------------------------------


class TestLoadFromSavePathGuard:
    """Calling ``train()`` or ``run()`` after ``load_from_save_path()``
    is a misuse -- the resume path is generate-then-evaluate only.
    """

    @pytest.fixture
    def loaded_builder(self, tmp_path: Path, monkeypatch) -> SafeSynthesizer:
        """Create a builder that has gone through load_from_save_path()."""
        original = _make_original_df()
        replaced = _make_replaced_df()
        _patch_heavy_deps(monkeypatch, original, replaced)

        builder = (
            SafeSynthesizer(save_path=tmp_path)
            .with_data_source(original)
            .with_replace_pii(config=PiiReplacerConfig.get_default_config())
            .synthesize()
            .resolve()
        )
        # Run process_data to create the workdir artifacts
        builder.process_data()

        # Now simulate a fresh builder that loads from that save path.
        # Don't call .resolve() -- that requires a data source, but the
        # resume path loads data from disk via load_from_save_path().
        loaded = SafeSynthesizer(save_path=tmp_path)

        # Patch metadata loading so load_from_save_path doesn't fail
        assert loaded._workdir is not None
        metadata_file = loaded._workdir.metadata_file
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text("{}")
        monkeypatch.setattr(
            f"{MODULE}.ModelMetadata",
            type(
                "MM",
                (),
                {
                    "from_config": staticmethod(lambda c, workdir: None),
                    "from_metadata_json": staticmethod(lambda f, workdir: None),
                },
            ),
        )
        # Write the config file
        config_path = loaded._workdir.source_config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        assert builder._nss_config is not None
        config_path.write_text(builder._nss_config.model_dump_json(indent=2))

        # Write cached train/test CSVs so load_from_save_path finds them
        src_ds = loaded._workdir.source_dataset
        training_csv = Path(src_ds.training)
        training_csv.parent.mkdir(parents=True, exist_ok=True)
        original.to_csv(training_csv, index=False)
        replaced.to_csv(Path(src_ds.test), index=False)

        loaded.load_from_save_path()
        return loaded

    def test_train_raises_after_load(self, loaded_builder: SafeSynthesizer):
        with pytest.raises(RuntimeError, match="train.*cannot be called after load_from_save_path"):
            loaded_builder.train()

    def test_run_raises_after_load(self, loaded_builder: SafeSynthesizer):
        with pytest.raises(RuntimeError, match="run.*cannot be called after load_from_save_path"):
            loaded_builder.run()
