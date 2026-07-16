# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the original-vs-PII-replaced training data split in process_data.

When PII replacement is enabled, ``process_data`` must preserve the original
training split in ``_original_training_df`` (used by evaluation) while storing
the PII-replaced version in ``_training_df`` (used by model training).  These
tests verify the separation, persistence, and round-trip through
``load_from_save_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.generation.results import GenerateJobResults
from nemo_safe_synthesizer.generation.utils import GenerationStatus
from nemo_safe_synthesizer.preflight import PreflightReport, PreflightStage
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

_EMPTY_PREFLIGHT = PreflightReport(checks=[])

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

    @patch("nemo_safe_synthesizer.sdk.library_builder.run_preflight", return_value=_EMPTY_PREFLIGHT)
    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_without_pii_replacement_sets_original_training_df(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        mock_preflight,
        fixture_process_data_setup_without_pii,
    ):
        """Without PII replacement, ``_original_training_df`` matches the training split."""
        builder, train_split, test_split, _, _ = fixture_process_data_setup_without_pii
        _wire_process_data_mocks(
            mock_holdout_cls, mock_resolver_cls, mock_metadata_cls, builder, train_split, test_split
        )

        builder.process_data()

        pd.testing.assert_frame_equal(builder._original_training_df, train_split)
        assert builder._training_df is not None
        pd.testing.assert_frame_equal(builder._training_df, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.run_preflight", return_value=_EMPTY_PREFLIGHT)
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
        mock_preflight,
        fixture_process_data_setup_with_pii,
    ):
        """With PII replacement, ``_original_training_df`` preserves the pre-PII data."""
        builder, train_split, test_split, pii_replaced_df, mock_replacer = fixture_process_data_setup_with_pii
        _wire_process_data_mocks(
            mock_holdout_cls, mock_resolver_cls, mock_metadata_cls, builder, train_split, test_split
        )
        mock_pii_cls.return_value = mock_replacer

        builder.process_data()

        # Training uses the PII-replaced data; evaluation uses the original
        pd.testing.assert_frame_equal(builder._training_df, pii_replaced_df)
        pd.testing.assert_frame_equal(builder._original_training_df, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.run_preflight", return_value=_EMPTY_PREFLIGHT)
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
        mock_preflight,
        fixture_process_data_setup_with_pii,
        fixture_workdir,
    ):
        """``training.csv`` persists the original split; ``transformed_training.csv`` persists the PII-replaced data."""
        builder, train_split, test_split, pii_replaced_df, mock_replacer = fixture_process_data_setup_with_pii
        _wire_process_data_mocks(
            mock_holdout_cls, mock_resolver_cls, mock_metadata_cls, builder, train_split, test_split
        )
        mock_pii_cls.return_value = mock_replacer

        builder.process_data()

        training_csv = fixture_workdir.dataset.training
        transformed_csv = fixture_workdir.dataset.transformed_training

        assert training_csv.exists()
        assert transformed_csv.exists()

        # ``training.csv`` always contains the original training split
        saved_training = pd.read_csv(training_csv)
        pd.testing.assert_frame_equal(saved_training, train_split)

        # ``transformed_training.csv`` contains the PII-replaced data (inspection only)
        saved_transformed = pd.read_csv(transformed_csv)
        pd.testing.assert_frame_equal(saved_transformed, pii_replaced_df)

    @patch("nemo_safe_synthesizer.sdk.library_builder.run_preflight", return_value=_EMPTY_PREFLIGHT)
    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_without_pii_replacement_does_not_write_transformed_training(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        mock_preflight,
        fixture_process_data_setup_without_pii,
        fixture_workdir,
    ):
        """Without PII replacement, no ``transformed_training.csv`` is written."""
        builder, train_split, test_split, _, _ = fixture_process_data_setup_without_pii
        _wire_process_data_mocks(
            mock_holdout_cls, mock_resolver_cls, mock_metadata_cls, builder, train_split, test_split
        )

        builder.process_data()

        training_csv = fixture_workdir.dataset.training
        assert training_csv.exists()
        saved_training = pd.read_csv(training_csv)
        pd.testing.assert_frame_equal(saved_training, train_split)

        transformed_csv = fixture_workdir.dataset.transformed_training
        assert not transformed_csv.exists()


# ---------------------------------------------------------------------------
# Tests: metadata lifecycle in validate mode
# ---------------------------------------------------------------------------


class TestProcessDataMetadataLifecycle:
    """Validate-mode metadata fallback should not persist stub metadata."""

    @patch("nemo_safe_synthesizer.sdk.library_builder.run_preflight", return_value=_EMPTY_PREFLIGHT)
    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_check_only_stub_metadata_not_persisted_for_followup_run(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        mock_preflight,
        fixture_process_data_setup_without_pii,
    ):
        """A check-only stub does not block rebuilding metadata for later full runs."""
        builder, train_split, test_split, _, _ = fixture_process_data_setup_without_pii
        mock_holdout_cls.return_value.train_test_split.return_value = (train_split, test_split)
        mock_resolver_cls.return_value.return_value = builder._nss_config

        stub_metadata = MagicMock(name="stub_metadata")
        rebuilt_metadata = MagicMock(name="rebuilt_metadata")
        mock_metadata_cls.stub.return_value = stub_metadata
        mock_metadata_cls.from_config.side_effect = [RuntimeError("offline"), rebuilt_metadata]

        builder.process_data(check_only=True)

        assert builder._llm_metadata is None
        full_preflight_calls = [call for call in mock_preflight.call_args_list if "stages" not in call.kwargs]
        assert len(full_preflight_calls) == 1
        assert full_preflight_calls[0].args[2] is stub_metadata

        builder.process_data(check_only=False)

        assert builder._llm_metadata is rebuilt_metadata
        full_preflight_calls = [call for call in mock_preflight.call_args_list if "stages" not in call.kwargs]
        assert len(full_preflight_calls) == 2
        assert full_preflight_calls[1].args[2] is rebuilt_metadata
        early_preflight_calls = [call for call in mock_preflight.call_args_list if "stages" in call.kwargs]
        assert len(early_preflight_calls) == 2
        assert all(
            call.kwargs["stages"] == frozenset({PreflightStage.CONFIG, PreflightStage.DATAFRAME})
            for call in early_preflight_calls
        )
        assert mock_metadata_cls.from_config.call_count == 2


# ---------------------------------------------------------------------------
# Tests: evaluate uses correct reference
# ---------------------------------------------------------------------------


class TestEvaluateUsesOriginalTrainingDf:
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
    def test_evaluate_uses_original_training_df(
        self,
        mock_evaluator_cls,
        mock_make_results,
        fixture_name,
        request: pytest.FixtureRequest,
    ):
        """Evaluate always passes ``_original_training_df`` as ``training_df``."""
        setup = request.getfixturevalue(fixture_name)
        builder, train_split, test_split, pii_replaced_df, _ = setup
        has_pii = fixture_name == "fixture_process_data_setup_with_pii"
        builder._training_df = pii_replaced_df if has_pii else train_split
        builder._original_training_df = train_split
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
        pd.testing.assert_frame_equal(call_kwargs["training_df"], train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.Evaluator")
    def test_evaluate_disabled_skips_evaluator_and_builds_results(
        self,
        mock_evaluator_cls,
        fixture_process_data_setup_without_pii,
    ):
        """When evaluation is disabled, ``evaluate()`` still prepares saveable results."""
        builder, train_split, test_split, _, _ = fixture_process_data_setup_without_pii
        assert builder._nss_config is not None
        builder._nss_config.evaluation.enabled = False
        builder._training_df = train_split
        builder._original_training_df = train_split
        builder._test_df = test_split
        builder._total_start = 0.0

        synthetic_df = pd.DataFrame({"patient_name": ["Synthetic Person"], "patient_age": [42]})
        mock_gen = MagicMock()
        mock_gen.gen_results = GenerateJobResults(
            df=synthetic_df,
            status=GenerationStatus.COMPLETE,
            num_valid_records=1,
            num_invalid_records=0,
            num_prompts=1,
            valid_record_fraction=1.0,
            batch_valid_record_fractions=[1.0],
            elapsed_time=1.0,
        )
        builder.generator = mock_gen

        builder.evaluate()

        mock_evaluator_cls.assert_not_called()
        assert builder.evaluator is None
        assert builder.results.evaluation_report_html is None
        assert builder.results.summary.timing.evaluation_time_sec is None
        pd.testing.assert_frame_equal(builder.results.synthetic_data, synthetic_df)


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
        """``training.csv`` is loaded into ``_original_training_df`` in the resume flow."""
        workdir, train_split, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()

        assert builder._training_df is None  # generation-evaluation resume path doesn't need the transformed df
        assert builder._original_training_df is not None
        pd.testing.assert_frame_equal(builder._original_training_df, train_split)

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_rejects_unknown_legacy_saved_fields_by_default(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """Saved configs retain strict validation when no resume opt-out is set."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        saved_config = SafeSynthesizerParameters().model_dump(mode="json")
        saved_config.pop("strict_config")
        saved_config["training"]["epoch"] = 1
        workdir.config.write_text(json.dumps(saved_config))
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)

        with pytest.raises(ValidationError, match="epoch"):
            builder.load_from_save_path()

    @pytest.mark.parametrize("use_runtime_config", [False, True], ids=["builder-policy", "runtime-policy"])
    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_can_ignore_unknown_legacy_saved_fields(
        self,
        mock_metadata_cls,
        use_runtime_config,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """An explicit non-strict resume policy applies before saved-config validation."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        saved_config = SafeSynthesizerParameters().model_dump(mode="json")
        saved_config.pop("strict_config")
        saved_config["training"]["epoch"] = 1
        workdir.config.write_text(json.dumps(saved_config))
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        runtime_config = SafeSynthesizerParameters.model_validate({"strict_config": False})
        builder = SafeSynthesizer(config=runtime_config if use_runtime_config else None, workdir=workdir)
        if not use_runtime_config:
            builder.with_strict_config(False)

        builder.load_from_save_path(runtime_config=runtime_config if use_runtime_config else None)

        assert builder._nss_config is not None
        assert builder._nss_config.strict_config is False
        assert not hasattr(builder._nss_config.training, "epoch")

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_applies_runtime_generation_and_evaluation_config(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """Resume keeps train config from disk while honoring generation overrides."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        saved_config = SafeSynthesizerParameters()
        saved_config.training.batch_size = 8
        saved_config.generation.num_records = 3000
        saved_config.generation.structured_generation.enabled = True
        saved_config.generation.structured_generation.schema_method = "auto"
        workdir.config.write_text(saved_config.model_dump_json())

        runtime_config = SafeSynthesizerParameters()
        runtime_config.training.batch_size = 32
        runtime_config.generation.num_records = 100
        runtime_config.generation.structured_generation.schema_method = "auto"
        runtime_config.evaluation.enabled = False
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=runtime_config, workdir=workdir)
        builder.load_from_save_path(runtime_config=runtime_config)

        assert builder._nss_config is not None
        assert builder._nss_config.training.batch_size == 8
        assert builder._nss_config.generation.num_records == 100
        # Saved value not re-specified at runtime is preserved (field-level merge).
        assert builder._nss_config.generation.structured_generation.enabled is True
        assert builder._nss_config.generation.structured_generation.schema_method == "auto"
        assert builder._nss_config.evaluation.enabled is False

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_preserves_saved_generation_when_runtime_config_has_no_overrides(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """Default runtime config must not reset saved generation settings."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        saved_config = SafeSynthesizerParameters()
        saved_config.generation.num_records = 3000
        saved_config.generation.structured_generation.enabled = True
        saved_config.generation.structured_generation.schema_method = "auto"
        saved_config.evaluation.enabled = True
        workdir.config.write_text(saved_config.model_dump_json())

        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path(runtime_config=SafeSynthesizerParameters())

        assert builder._nss_config is not None
        assert builder._nss_config.generation.num_records == 3000
        assert builder._nss_config.generation.structured_generation.enabled is True
        assert builder._nss_config.generation.structured_generation.schema_method == "auto"
        assert builder._nss_config.evaluation.enabled is True

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_deep_merges_nested_validation_overrides(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """Resume merges nested generation.validation fields without dropping saved siblings."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        saved_config = SafeSynthesizerParameters()
        saved_config.generation.num_records = 3000
        saved_config.generation.validation.group_by_ignore_invalid_records = True
        workdir.config.write_text(saved_config.model_dump_json())

        runtime_config = SafeSynthesizerParameters()
        runtime_config.generation.validation.group_by_fix_unordered_records = True
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=runtime_config, workdir=workdir)
        builder.load_from_save_path(runtime_config=runtime_config)

        assert builder._nss_config is not None
        # Nested override applied.
        assert builder._nss_config.generation.validation.group_by_fix_unordered_records is True
        # Saved sibling in the same nested group preserved.
        assert builder._nss_config.generation.validation.group_by_ignore_invalid_records is True
        # Unrelated saved generation field preserved.
        assert builder._nss_config.generation.num_records == 3000

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_full_runtime_config_replaces_supported_runtime_sections(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """A fully materialized generate-time config replaces generation/evaluation sections."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        saved_config = SafeSynthesizerParameters()
        saved_config.training.batch_size = 8
        saved_config.generation.num_records = 3000
        saved_config.generation.structured_generation.enabled = True
        saved_config.generation.structured_generation.schema_method = "auto"
        saved_config.evaluation.enabled = True
        workdir.config.write_text(saved_config.model_dump_json())

        # model_validate(model_dump(...)) marks every field as explicitly set, so
        # the whole generation/evaluation sections override the saved values.
        runtime_config = SafeSynthesizerParameters.model_validate(SafeSynthesizerParameters().model_dump(mode="json"))
        runtime_config.generation.num_records = 100
        runtime_config.evaluation.enabled = False
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=runtime_config, workdir=workdir)
        builder.load_from_save_path(runtime_config=runtime_config)

        assert builder._nss_config is not None
        assert builder._nss_config.training.batch_size == 8
        assert builder._nss_config.generation.num_records == 100
        assert builder._nss_config.generation.structured_generation.enabled is False
        assert builder._nss_config.generation.structured_generation.schema_method == "auto"
        assert builder._nss_config.evaluation.enabled is False

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_warns_when_saved_values_differ_from_current_defaults(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
        fixture_sample_patient_redacted_dataframe,
    ):
        """Default drift warnings fire because saved configs carry no provenance metadata."""
        workdir, _, _ = self._prepare_workdir(
            tmp_path,
            fixture_sample_patient_dataframe,
            fixture_sample_patient_redacted_dataframe,
        )
        saved_config = SafeSynthesizerParameters()
        saved_config.generation.num_records = 3000
        workdir.config.write_text(saved_config.model_dump_json())
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        with patch("nemo_safe_synthesizer.sdk.library_builder.logger") as mock_logger:
            builder.load_from_save_path()

        warning_messages = [call.args[0] for call in mock_logger.user.warning.call_args_list]
        assert any("generation.num_records" in message for message in warning_messages)
        assert builder._nss_config is not None
        assert builder._nss_config.generation.num_records == 3000

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

        assert builder._training_df is None  # generation-evaluation resume path doesn't need the transformed df
        assert builder._original_training_df is not None
        pd.testing.assert_frame_equal(builder._original_training_df, train_split)

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
# Tests: load_from_save_path with holdout=0
# ---------------------------------------------------------------------------


class TestLoadFromSavePathHoldoutZero:
    """``load_from_save_path`` and ``process_data`` must handle holdout=0 gracefully.

    When ``holdout=0``, the train/test split returns ``test_df=None`` and no
    test set is produced.  Previously, ``process_data`` would ``touch()`` an
    empty ``test.csv``, and ``load_from_save_path`` would unconditionally call
    ``pd.read_csv`` on it, raising ``EmptyDataError``.

    These tests verify:

    * ``process_data`` does not write ``test.csv`` when the holdout split
      produces no test set.
    * ``load_from_save_path`` succeeds when ``test.csv`` is absent (new runs
      with ``holdout=0``).
    * ``load_from_save_path`` succeeds when ``test.csv`` is an empty 0-byte
      file (backward compatibility with runs created before the fix).
    """

    def _prepare_workdir_no_holdout(
        self,
        tmp_path: Path,
        fixture_sample_patient_dataframe: pd.DataFrame,
    ) -> tuple[Workdir, pd.DataFrame]:
        """Create a workdir with only training.csv (no test.csv), simulating holdout=0."""
        workdir = Workdir(base_path=tmp_path, config_name="test", dataset_name="data")
        workdir.ensure_directories()

        train_split = fixture_sample_patient_dataframe.copy()
        train_split.to_csv(workdir.dataset.training, index=False)

        config = SafeSynthesizerParameters()
        config_path = workdir.config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config.model_dump_json())

        metadata_path = workdir.metadata_file
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text("{}")

        return workdir, train_split

    @patch("nemo_safe_synthesizer.sdk.library_builder.run_preflight", return_value=_EMPTY_PREFLIGHT)
    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    @patch("nemo_safe_synthesizer.sdk.library_builder.AutoConfigResolver")
    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_process_data_no_test_csv_when_holdout_zero(
        self,
        mock_holdout_cls,
        mock_resolver_cls,
        mock_metadata_cls,
        mock_preflight,
        fixture_workdir,
        fixture_sample_patient_dataframe,
    ):
        """No ``test.csv`` is written when the holdout split yields no test set.

        Mocks ``Holdout.train_test_split`` to return ``(train_df, None)`` --
        the same value it returns for ``holdout=0`` -- and asserts that
        ``process_data`` leaves the dataset directory without a ``test.csv``
        file and keeps ``_test_df`` as ``None``.
        """
        train_split = fixture_sample_patient_dataframe.copy()
        builder = SafeSynthesizer(
            config=SafeSynthesizerParameters(replace_pii=None),
            workdir=fixture_workdir,
        )
        builder._data_source = fixture_sample_patient_dataframe

        mock_holdout_cls.return_value.train_test_split.return_value = (train_split, None)
        mock_resolver_cls.return_value.return_value = builder._nss_config
        mock_metadata_cls.from_config.return_value = MagicMock()

        builder.process_data()

        assert not fixture_workdir.dataset.test.exists()
        assert builder._test_df is None

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_succeeds_without_test_csv(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
    ):
        """Resume succeeds when ``test.csv`` does not exist on disk.

        Prepares a workdir that only contains ``training.csv`` (no
        ``test.csv``), simulating a new run with ``holdout=0``.  Verifies
        that ``load_from_save_path`` loads the training split, sets
        ``_test_df`` to ``None``, and marks the load as complete.
        """
        workdir, train_split = self._prepare_workdir_no_holdout(tmp_path, fixture_sample_patient_dataframe)
        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()

        assert builder._original_training_df is not None
        pd.testing.assert_frame_equal(builder._original_training_df, train_split)
        assert builder._test_df is None
        assert builder._loaded_from_save_path is True

    @patch("nemo_safe_synthesizer.sdk.library_builder.ModelMetadata")
    def test_load_handles_empty_test_csv_from_old_runs(
        self,
        mock_metadata_cls,
        tmp_path,
        fixture_sample_patient_dataframe,
    ):
        """Resume succeeds when ``test.csv`` is an empty 0-byte file.

        Before the fix, ``process_data`` called ``touch()`` to create an
        empty ``test.csv`` when there was no holdout.  Saved run directories
        from those older runs still have the empty file on disk.  This test
        ensures ``load_from_save_path`` treats it the same as a missing file
        rather than crashing with ``EmptyDataError``.
        """
        workdir, train_split = self._prepare_workdir_no_holdout(tmp_path, fixture_sample_patient_dataframe)
        # Simulate old behavior: empty 0-byte test.csv
        test_csv = workdir.dataset.test
        assert isinstance(test_csv, Path)
        test_csv.touch()

        mock_metadata_cls.from_metadata_json.return_value = MagicMock()

        builder = SafeSynthesizer(config=SafeSynthesizerParameters(), workdir=workdir)
        builder.load_from_save_path()

        assert builder._original_training_df is not None
        pd.testing.assert_frame_equal(builder._original_training_df, train_split)
        assert builder._test_df is None
        assert builder._loaded_from_save_path is True


# ---------------------------------------------------------------------------
# Tests: config validation at process_data entry
# ---------------------------------------------------------------------------


class TestProcessDataConfigValidation:
    """``process_data`` must validate configuration before doing any I/O.

    Incompatible settings that are supplied via the builder's ``with_*``
    methods after construction are not visible to the Pydantic validator
    until ``_resolve_nss_config()`` is called.  ``process_data`` must
    call it at the top of the method so invalid configs are caught
    immediately. It also validates configured group/order columns against
    the input dataset before holdout split, autoconfig resolution, PII
    replacement, or any disk I/O.
    """

    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_invalid_groupby_raises_before_holdout(
        self,
        mock_holdout_cls,
        fixture_workdir: Workdir,
        fixture_sample_patient_dataframe: pd.DataFrame,
    ) -> None:
        """Missing group-by column raises immediately during ``process_data``.

        This catches invalid ``group_training_examples_by`` before holdout split
        or autoconfig runs, ensuring a clear ``ParameterError`` instead of a
        downstream ``KeyError``.
        """
        ss = SafeSynthesizer(
            config=SafeSynthesizerParameters.from_params(group_training_examples_by="non_existent_group"),
            workdir=fixture_workdir,
        ).with_data_source(fixture_sample_patient_dataframe)

        with pytest.raises(ParameterError, match="Group by column 'non_existent_group' not found"):
            ss.process_data()

        assert ss.preflight_report is not None
        assert any(
            issue.check == "columns.groupby" and issue.code == "column_not_found"
            for issue in ss.preflight_report.errors
        )
        mock_holdout_cls.assert_not_called()

    @patch("nemo_safe_synthesizer.sdk.library_builder.Holdout")
    def test_invalid_orderby_raises_before_holdout(
        self,
        mock_holdout_cls,
        fixture_workdir: Workdir,
        fixture_sample_patient_dataframe: pd.DataFrame,
    ) -> None:
        """Missing order-by column raises immediately during ``process_data``.

        This catches invalid ``order_training_examples_by`` before holdout split
        or autoconfig runs, ensuring a clear ``ParameterError`` instead of a
        downstream pandas error.
        """
        ss = SafeSynthesizer(
            config=SafeSynthesizerParameters.from_params(
                group_training_examples_by="patient_name",
                order_training_examples_by="non_existent_order",
            ),
            workdir=fixture_workdir,
        ).with_data_source(fixture_sample_patient_dataframe)

        with pytest.raises(ParameterError, match="Order by column 'non_existent_order' not found"):
            ss.process_data()

        assert ss.preflight_report is not None
        assert any(
            issue.check == "columns.orderby" and issue.code == "column_not_found"
            for issue in ss.preflight_report.errors
        )
        mock_holdout_cls.assert_not_called()
