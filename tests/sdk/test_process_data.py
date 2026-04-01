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
from pydantic import ValidationError

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_workdir(tmp_path: Path) -> Workdir:
    return Workdir(base_path=tmp_path, config_name="test", dataset_name="data")


@pytest.fixture
def fixture_process_data_setup_without_pii(
    fixture_sample_patient_dataframe: pd.DataFrame,
    fixture_sample_patient_redacted_dataframe: pd.DataFrame | None,
    fixture_workdir: Workdir,
) -> tuple[SafeSynthesizer, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, MagicMock]:
    """Build a SafeSynthesizer with mocked heavy dependencies (PII disabled).

    Returns the builder *before* calling ``process_data()`` so callers
    can inspect state at each stage.

    For tests that need PII replacement enabled, use
    ``fixture_process_data_setup_with_pii``.
    """
    return _create_process_data_setup(
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
        fixture_workdir,
        replace_pii=False,
    )


@pytest.fixture
def fixture_process_data_setup_with_pii(
    fixture_sample_patient_dataframe: pd.DataFrame,
    fixture_sample_patient_redacted_dataframe: pd.DataFrame | None,
    fixture_workdir: Workdir,
) -> tuple[SafeSynthesizer, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, MagicMock]:
    """Build a SafeSynthesizer with mocked heavy dependencies (PII enabled).

    Returns the builder *before* calling ``process_data()`` so callers
    can inspect state at each stage.
    """
    return _create_process_data_setup(
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
        fixture_workdir,
        replace_pii=True,
    )


def _create_process_data_setup(
    fixture_sample_patient_dataframe: pd.DataFrame,
    fixture_sample_patient_redacted_dataframe: pd.DataFrame | None,
    fixture_workdir: Workdir,
    *,
    replace_pii: bool = True,
) -> tuple[SafeSynthesizer, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, MagicMock]:
    """Shared factory for the ``fixture_process_data_setup_*`` fixtures.

    Builds a ``SafeSynthesizer`` wired with deterministic train/test splits
    and a pre-built PII replacer mock, bypassing real NER models.  The
    builder is returned before ``process_data()`` runs so each test
    controls when -- and whether -- the method is called.
    """
    original_df = fixture_sample_patient_dataframe.copy()
    if fixture_sample_patient_redacted_dataframe is not None:
        pii_replaced_df = fixture_sample_patient_redacted_dataframe.head(100).copy()
    else:
        pii_replaced_df = None

    # Returns a deterministic train/test split
    train_split = original_df.head(100).copy()
    test_split = original_df.tail(100).copy()

    config = SafeSynthesizerParameters()

    builder = SafeSynthesizer(config=config, workdir=fixture_workdir)
    builder._data_source = original_df
    assert builder._nss_config is not None
    if replace_pii:
        from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig

        builder._nss_config.replace_pii = PiiReplacerConfig.get_default_config()
    else:
        builder._nss_config.replace_pii = None

    # Stub just enough of NemoPII's interface to satisfy process_data
    mock_replacer_instance = MagicMock()
    mock_replacer_instance.result.transformed_df = pii_replaced_df
    mock_replacer_instance.result.column_statistics = {
        "patient_name": MagicMock(),
        "timestamp": MagicMock(),
        "patient_age": MagicMock(),
    }
    mock_replacer_instance.elapsed_time = 1.5

    return builder, train_split, test_split, pii_replaced_df, mock_replacer_instance


def _wire_process_data_mocks(
    mock_holdout_cls: MagicMock,
    mock_resolver_cls: MagicMock,
    mock_metadata_cls: MagicMock,
    builder: SafeSynthesizer,
    train_split: pd.DataFrame,
    test_split: pd.DataFrame,
) -> None:
    """Configure the three mocks that ``process_data`` always invokes.

    These are separate from the fixture because they come from ``@patch``
    decorators on the test method, which pytest injects as positional args
    that fixtures cannot access.
    """
    mock_holdout_cls.return_value.train_test_split.return_value = (train_split, test_split)
    mock_resolver_cls.return_value.return_value = builder._nss_config
    mock_metadata_cls.from_config.return_value = MagicMock()


# ---------------------------------------------------------------------------
# Tests: process_data
# ---------------------------------------------------------------------------


class TestProcessDataPiiSeparation:
    """``process_data`` must keep original and PII-replaced DataFrames separate."""

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_without_pii_replacement_sets_original_train_df(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        fixture_process_data_setup_without_pii,
    ):
        """Without PII replacement, ``_original_train_df`` matches the training split."""
        builder, train_split, test_split, _, _ = fixture_process_data_setup_without_pii
        _wire_process_data_mocks(
            mock_holdout_cls, mock_resolver_cls, mock_metadata_cls, builder, train_split, test_split
        )

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
        fixture_process_data_setup_with_pii,
    ):
        """With PII replacement, ``_original_train_df`` preserves the pre-PII data."""
        builder, train_split, test_split, pii_replaced_df, mock_replacer = fixture_process_data_setup_with_pii
        _wire_process_data_mocks(
            mock_holdout_cls, mock_resolver_cls, mock_metadata_cls, builder, train_split, test_split
        )
        mock_pii_cls.return_value = mock_replacer

        builder.process_data()

        # Training uses the PII-replaced data; evaluation uses the original
        pd.testing.assert_frame_equal(builder._train_df, pii_replaced_df)
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
        fixture_process_data_setup_with_pii,
        fixture_workdir,
    ):
        """``training.csv`` persists the original split (pre-PII data)."""
        builder, train_split, test_split, pii_replaced_df, mock_replacer = fixture_process_data_setup_with_pii
        _wire_process_data_mocks(
            mock_holdout_cls, mock_resolver_cls, mock_metadata_cls, builder, train_split, test_split
        )
        mock_pii_cls.return_value = mock_replacer

        builder.process_data()

        training_csv = fixture_workdir.dataset.training
        assert training_csv.exists()

        # ``training.csv`` always contains the original training split
        saved_training = pd.read_csv(training_csv)
        pd.testing.assert_frame_equal(saved_training, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_without_pii_replacement_does_not_write_transformed_training(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        fixture_process_data_setup_without_pii,
        fixture_workdir,
    ):
        """Without PII replacement, ``training.csv`` contains the original split."""
        builder, train_split, test_split, _, _ = fixture_process_data_setup_without_pii
        _wire_process_data_mocks(
            mock_holdout_cls, mock_resolver_cls, mock_metadata_cls, builder, train_split, test_split
        )

        builder.process_data()

        training_csv = fixture_workdir.dataset.training
        assert training_csv.exists()
        saved_training = pd.read_csv(training_csv)
        pd.testing.assert_frame_equal(saved_training, train_split)


# ---------------------------------------------------------------------------
# Tests: evaluate uses correct reference
# ---------------------------------------------------------------------------


class TestEvaluateUsesOriginalTrainDf:
    """``evaluate()`` must always pass the original (pre-PII) data to ``Evaluator``."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "fixture_process_data_setup_with_pii",
            "fixture_process_data_setup_without_pii",
        ],
        ids=["with_pii_replacement", "without_pii_replacement"],
    )
    @patch("nemo_safe_synthesizer.sdk.library_builder.make_nss_results")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Evaluator")
    def test_evaluate_uses_original_train_df(
        self,
        mock_evaluator_cls,
        mock_make_results,
        fixture_name,
        request: pytest.FixtureRequest,
    ):
        """Evaluate always passes ``_original_train_df`` as ``train_df``."""
        setup = request.getfixturevalue(fixture_name)
        builder, train_split, test_split, pii_replaced_df, _ = setup
        has_pii = fixture_name == "fixture_process_data_setup_with_pii"
        builder._train_df = pii_replaced_df if has_pii else train_split
        builder._original_train_df = train_split
        builder._test_df = test_split
        builder._total_start = 0.0

        mock_gen = MagicMock()
        mock_gen.gen_results.elapsed_time = 1.0
        builder.generator = mock_gen

        mock_evaluator_cls.return_value.evaluation_time = 0.5
        mock_evaluator_cls.return_value.report = MagicMock()

        builder.evaluate()

        # Evaluation metrics must reflect real data, not PII-replaced tokens
        call_kwargs = mock_evaluator_cls.call_args[1]
        pd.testing.assert_frame_equal(call_kwargs["train_df"], train_split)


# ---------------------------------------------------------------------------
# Tests: load_from_save_path round-trip
# ---------------------------------------------------------------------------


class TestLoadFromSavePath:
    """``load_from_save_path`` round-trip must restore the original training split."""

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

        _, train_split, test_split, _, _ = _create_process_data_setup(
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
            workdir,
            replace_pii=False,
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
        """``training.csv`` is loaded into ``_original_train_df`` in the resume flow."""
        workdir, train_split, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()

        assert builder._original_train_df is not None
        assert builder._train_df is not None
        pd.testing.assert_frame_equal(builder._original_train_df, train_split)
        # _train_df is also set so process_data's early-return guard works
        pd.testing.assert_frame_equal(builder._train_df, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_process_data_skips_when_cached_splits_loaded(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """After loading cached splits, ``process_data()`` is a no-op."""
        workdir, train_split, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()
        builder.process_data()

        assert builder._original_train_df is not None
        assert builder._train_df is not None
        pd.testing.assert_frame_equal(builder._original_train_df, train_split)
        pd.testing.assert_frame_equal(builder._train_df, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_train_after_load_from_save_path_raises(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """``train()`` is not valid in the resume path -- it should fail immediately."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()

        with pytest.raises(RuntimeError, match="train.*cannot be called after load_from_save_path"):
            builder.train()

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_run_after_load_from_save_path_raises(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """``run()`` includes ``train()`` and is not valid in the resume path."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()

        with pytest.raises(RuntimeError, match="run.*cannot be called after load_from_save_path"):
            builder.run()


# ---------------------------------------------------------------------------
# Tests: config validation at process_data entry
# ---------------------------------------------------------------------------


class TestProcessDataConfigValidation:
    """``process_data`` must validate configuration before doing any I/O.

    Incompatible settings that are supplied via the builder's ``with_*``
    methods after construction are not visible to the Pydantic validator
    until ``_resolve_nss_config()`` is called.  ``process_data`` must
    call it at the top of the method so invalid configs are caught
    immediately -- before holdout split, PII replacement, or any disk I/O.
    """

    def test_dp_and_explicit_unsloth_raises_at_process_data(self, fixture_workdir: Workdir) -> None:
        """DP + explicit ``use_unsloth=True`` raises before any data is processed.

        Pydantic wraps the inner ``ParameterError`` in a ``ValidationError``.
        """
        ss = (
            SafeSynthesizer(workdir=fixture_workdir)
            .with_train(use_unsloth=True)
            .with_differential_privacy(dp_enabled=True)
        )
        with pytest.raises(ValidationError, match="not compatible with DP"):
            ss.process_data()
