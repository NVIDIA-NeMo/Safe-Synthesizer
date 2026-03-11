# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the original-vs-PII-replaced training data split in process_data.

When PII replacement is enabled, ``process_data`` must preserve the original
training split in ``_original_train_df`` (used by evaluation) while storing
the PII-replaced version in ``_train_df`` (used by model training).  These
tests verify the separation, persistence, and round-trip through
``load_from_save_path``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_workdir(tmp_path: Path) -> Workdir:
    return Workdir(base_path=tmp_path, config_name="test", dataset_name="data")


def _patch_process_data_deps(
    fixture_sample_patient_dataframe: pd.DataFrame,
    fixture_sample_patient_redacted_dataframe: pd.DataFrame | None,
    fixture_workdir: Workdir,
    enable_replace_pii: bool,
):
    """Build a SafeSynthesizer with mocked heavy dependencies.

    Returns the builder *before* calling ``process_data()`` so callers
    can inspect state at each stage.
    """
    original_df = fixture_sample_patient_dataframe.copy()
    if fixture_sample_patient_redacted_dataframe is not None:
        pii_replaced_df = fixture_sample_patient_redacted_dataframe.head(100).copy()
    else:
        pii_replaced_df = None

    # Holdout returns a deterministic split
    train_split = original_df.head(100).copy()
    test_split = original_df.tail(100).copy()

    config = SafeSynthesizerParameters()

    builder = SafeSynthesizer(config=config, workdir=fixture_workdir)
    builder._data_source = original_df
    builder._nss_config.enable_replace_pii = enable_replace_pii
    if enable_replace_pii:
        from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig

        builder._nss_config.replace_pii = PiiReplacerConfig.get_default_config()

    # Mock the PII replacer to return our pre-built replaced df
    mock_replacer_instance = MagicMock()
    mock_replacer_instance.result.transformed_df = pii_replaced_df
    mock_replacer_instance.result.column_statistics = {
        "patient_name": MagicMock(),
        "timestamp": MagicMock(),
        "age": MagicMock(),
    }
    mock_replacer_instance.elapsed_time = 1.5

    return builder, train_split, test_split, pii_replaced_df, mock_replacer_instance


# ---------------------------------------------------------------------------
# Tests: process_data
# ---------------------------------------------------------------------------


class TestProcessDataPiiSeparation:
    """Verify that process_data keeps original and PII-replaced dfs separate."""

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_without_pii_replacement_sets_original_train_df(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
        fixture_workdir,
    ):
        """Without PII replacement, _original_train_df matches the training split."""
        builder, train_split, test_split, _, _ = _patch_process_data_deps(
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
            fixture_workdir,
            enable_replace_pii=False,
        )
        mock_holdout_cls.return_value.train_test_split.return_value = (train_split, test_split)
        mock_resolver_cls.return_value.return_value = builder._nss_config
        mock_metadata_cls.from_config.return_value = MagicMock()

        builder.process_data()

        pd.testing.assert_frame_equal(builder._original_train_df, train_split)
        assert builder._train_df is not None
        pd.testing.assert_frame_equal(builder._train_df, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.NemoPII")
    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_with_pii_replacement_preserves_original(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        mock_pii_cls,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
        fixture_workdir,
    ):
        """With PII replacement, _original_train_df holds the pre-PII data."""
        builder, train_split, test_split, pii_replaced_df, mock_replacer = _patch_process_data_deps(
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
            fixture_workdir,
            enable_replace_pii=True,
        )
        mock_holdout_cls.return_value.train_test_split.return_value = (train_split, test_split)
        mock_resolver_cls.return_value.return_value = builder._nss_config
        mock_metadata_cls.from_config.return_value = MagicMock()
        mock_pii_cls.return_value = mock_replacer

        builder.process_data()

        # _train_df should be the PII-replaced version (used for training)
        pd.testing.assert_frame_equal(builder._train_df, pii_replaced_df)
        # _original_train_df should be the original (used for evaluation)
        pd.testing.assert_frame_equal(builder._original_train_df, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.NemoPII")
    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_with_pii_replacement_persists_original_and_transformed(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        mock_pii_cls,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
        fixture_workdir,
    ):
        """When PII is enabled, training holds original and transformed_training holds PII output."""
        builder, train_split, test_split, pii_replaced_df, mock_replacer = _patch_process_data_deps(
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
            fixture_workdir,
            enable_replace_pii=True,
        )
        mock_holdout_cls.return_value.train_test_split.return_value = (train_split, test_split)
        mock_resolver_cls.return_value.return_value = builder._nss_config
        mock_metadata_cls.from_config.return_value = MagicMock()
        mock_pii_cls.return_value = mock_replacer

        builder.process_data()

        training_csv = fixture_workdir.dataset.training
        transformed_csv = fixture_workdir.dataset.transformed_training

        assert training_csv.exists()
        assert transformed_csv.exists()

        # training.csv always contains the original training split
        saved_training = pd.read_csv(training_csv)
        pd.testing.assert_frame_equal(saved_training, train_split)

        # transformed_training.csv contains the PII-replaced data (inspection only)
        saved_transformed = pd.read_csv(transformed_csv)
        pd.testing.assert_frame_equal(saved_transformed, pii_replaced_df)

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_without_pii_replacement_does_not_write_transformed_training(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
        fixture_workdir,
    ):
        """Without PII replacement, training.csv holds the original data and no transformed_training.csv is created."""
        builder, train_split, test_split, _, _ = _patch_process_data_deps(
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
            fixture_workdir,
            enable_replace_pii=False,
        )
        mock_holdout_cls.return_value.train_test_split.return_value = (train_split, test_split)
        mock_resolver_cls.return_value.return_value = builder._nss_config
        mock_metadata_cls.from_config.return_value = MagicMock()

        builder.process_data()

        training_csv = fixture_workdir.dataset.training
        assert training_csv.exists()
        saved_training = pd.read_csv(training_csv)
        pd.testing.assert_frame_equal(saved_training, train_split)

        transformed_csv = fixture_workdir.dataset.transformed_training
        assert not transformed_csv.exists()


# ---------------------------------------------------------------------------
# Tests: evaluate uses correct reference
# ---------------------------------------------------------------------------


class TestEvaluateUsesOriginalTrainDf:
    """Verify that evaluate() passes the original training data to the Evaluator."""

    @patch("nemo_safe_synthesizer.sdk.library_builder.make_nss_results")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Evaluator")
    def test_evaluate_with_pii_replacement_uses_original_train_df(
        self,
        mock_evaluator_cls,
        mock_make_results,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
        fixture_workdir,
    ):
        """When _original_train_df is set, evaluate passes it as train_df."""
        builder, train_split, test_split, pii_replaced_df, _ = _patch_process_data_deps(
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
            fixture_workdir,
            enable_replace_pii=True,
        )
        builder._train_df = pii_replaced_df
        builder._original_train_df = train_split
        builder._test_df = test_split
        builder._total_start = 0.0

        mock_gen = MagicMock()
        mock_gen.gen_results.elapsed_time = 1.0
        builder.generator = mock_gen

        mock_evaluator_cls.return_value.evaluation_time = 0.5
        mock_evaluator_cls.return_value.report = MagicMock()

        builder.evaluate()

        # The Evaluator should have been called with the original df, not the PII-replaced one
        call_kwargs = mock_evaluator_cls.call_args[1]
        pd.testing.assert_frame_equal(call_kwargs["train_df"], train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.make_nss_results")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Evaluator")
    def test_evaluate_without_pii_replacement_uses_original_train_df(
        self,
        mock_evaluator_cls,
        mock_make_results,
        fixture_sample_patient_dataframe,
        fixture_workdir,
    ):
        """Without PII, evaluate uses _original_train_df (same content as _train_df)."""
        builder, train_split, test_split, _, _ = _patch_process_data_deps(
            fixture_sample_patient_dataframe,
            None,
            fixture_workdir,
            enable_replace_pii=False,
        )
        builder._train_df = train_split
        builder._original_train_df = train_split
        builder._test_df = test_split
        builder._total_start = 0.0

        mock_gen = MagicMock()
        mock_gen.gen_results.elapsed_time = 1.0
        builder.generator = mock_gen

        mock_evaluator_cls.return_value.evaluation_time = 0.5
        mock_evaluator_cls.return_value.report = MagicMock()

        builder.evaluate()

        call_kwargs = mock_evaluator_cls.call_args[1]
        pd.testing.assert_frame_equal(call_kwargs["train_df"], train_split)


# ---------------------------------------------------------------------------
# Tests: load_from_save_path round-trip
# ---------------------------------------------------------------------------


class TestLoadFromSavePath:
    """Verify that load_from_save_path loads the original training split."""

    def _prepare_workdir(
        self,
        tmp_path: Path,
        fixture_sample_patient_dataframe: pd.DataFrame,
        fixture_sample_patient_redacted_dataframe: pd.DataFrame,
    ) -> tuple[Workdir, pd.DataFrame, pd.DataFrame]:
        """Create a workdir with cached dataset files on disk.

        Writes the original training data to ``training.csv``.
        """
        workdir = Workdir(base_path=tmp_path, config_name="test", dataset_name="data")
        workdir.ensure_directories()

        _, train_split, test_split, _, _ = _patch_process_data_deps(
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
            workdir,
            enable_replace_pii=False,
        )

        train_split.to_csv(workdir.dataset.training, index=False)
        test_split.to_csv(workdir.dataset.test, index=False)

        # Write minimal config so load_from_save_path can parse it
        config = SafeSynthesizerParameters()
        config_path = workdir.config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config.model_dump_json())

        # Create the metadata file so load_from_save_path doesn't raise
        metadata_path = workdir.metadata_file
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text("{}")

        return workdir, train_split, test_split

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_restores_training_split(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """training.csv is loaded into _original_train_df in resume flow."""
        workdir, train_split, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()

        assert builder._train_df is None  # We don't need any transformed training data in the evaluation phase
        pd.testing.assert_frame_equal(builder._original_train_df, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_process_data_skips_when_cached_splits_loaded(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """After loading cached splits, process_data() returns early without a data source."""
        workdir, train_split, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()
        builder.process_data()

        assert builder._train_df is None  # We don't need any transformed training data in the evaluation phase
        pd.testing.assert_frame_equal(builder._original_train_df, train_split)
