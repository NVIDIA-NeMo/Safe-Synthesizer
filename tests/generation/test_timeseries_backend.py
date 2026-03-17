# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TimeseriesBackend class private methods."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from transformers import PretrainedConfig

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import (
    DataParameters,
    GenerateParameters,
    SafeSynthesizerParameters,
    TimeSeriesParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.defaults import DEFAULT_MAX_SEQ_LENGTH, PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.generation.processors import TimeSeriesDataProcessor
from nemo_safe_synthesizer.generation.timeseries_backend import (
    GroupState,
    TimeseriesBackend,
)
from nemo_safe_synthesizer.llm.metadata import LLMPromptConfig, ModelMetadata

PROMPT_TEMPLATE = "[INST] {instruction} {schema} [/INST]"


@pytest.fixture(scope="session")
def fixture_autoconfig() -> PretrainedConfig:
    """Create a PretrainedConfig for testing."""
    config = PretrainedConfig()
    config.max_position_embeddings = DEFAULT_MAX_SEQ_LENGTH
    return config


@pytest.fixture
def timeseries_model_metadata(fixture_session_cache_dir, fixture_tokenizer, fixture_autoconfig, mock_workdir):
    """Create a real ModelMetadata object for timeseries backend testing."""
    metadata = ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        workdir=mock_workdir,
    )
    # TimeseriesBackend requires initial_prefill to be a dict mapping group -> prefill
    metadata.initial_prefill = {
        "group_A": '{"timestamp": "2024-01-01 00:00:00", "value": 1}\n',
        "group_B": '{"timestamp": "2024-01-01 00:00:00", "value": 2}\n',
    }
    return metadata


@pytest.fixture
def timeseries_base_params():
    """Create basic SafeSynthesizerParameters for timeseries testing."""
    return SafeSynthesizerParameters(
        data=DataParameters(
            group_training_examples_by="group_id",
            order_training_examples_by="timestamp",
        ),
        training=TrainingHyperparams(
            num_input_records_to_sample=100,
            batch_size=2,
            gradient_accumulation_steps=4,
            validation_ratio=0.0,
            pretrained_model="test-model",
            quantize_model=False,
            lora_r=16,
            lora_alpha_over_r=1.0,
            lora_target_modules=["q_proj", "v_proj"],
        ),
        generation=GenerateParameters(
            num_records=100,
            use_structured_generation=False,
        ),
        time_series=TimeSeriesParameters(
            is_timeseries=True,
            timestamp_column="timestamp",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            timestamp_interval_seconds=3600,
            start_timestamp="2024-01-01 00:00:00",
            stop_timestamp="2024-01-01 03:00:00",
        ),
    )


@pytest.fixture
def timeseries_elapsed_params():
    """Create params for elapsed time format testing."""
    return SafeSynthesizerParameters(
        data=DataParameters(
            group_training_examples_by="group_id",
            order_training_examples_by="elapsed_seconds",
        ),
        training=TrainingHyperparams(
            num_input_records_to_sample=100,
            batch_size=2,
            gradient_accumulation_steps=4,
            validation_ratio=0.0,
            pretrained_model="test-model",
            quantize_model=False,
            lora_r=16,
            lora_alpha_over_r=1.0,
            lora_target_modules=["q_proj", "v_proj"],
        ),
        generation=GenerateParameters(
            num_records=100,
            use_structured_generation=False,
        ),
        time_series=TimeSeriesParameters(
            is_timeseries=True,
            timestamp_column="elapsed_seconds",
            timestamp_format="elapsed_seconds",
            timestamp_interval_seconds=3600,
            start_timestamp="0",
            stop_timestamp="10800",
        ),
    )


@pytest.fixture
def mock_workdir(fixture_session_cache_dir):
    """Create a real Workdir with actual directories for testing."""
    workdir = Workdir(
        base_path=fixture_session_cache_dir,
        dataset_name="test-dataset",
        config_name="test-config",
        run_name="2026-01-15T12:00:00",
        _current_phase="train",
    )

    # Create all directories
    workdir.ensure_directories()

    return workdir


def create_timeseries_backend(config: SafeSynthesizerParameters, model_metadata, workdir, schema=None):
    """Helper to create a TimeseriesBackend instance with patched file dependencies."""
    if schema is None:
        schema = {
            "properties": {
                "timestamp": {"type": "string"},
                "value": {"type": "integer"},
            }
        }

    # Create the real processor
    processor = TimeSeriesDataProcessor(
        schema=schema,
        config=config.generation.validation,
        time_column=config.time_series.timestamp_column,
        interval_seconds=config.time_series.timestamp_interval_seconds,
        time_format=config.time_series.timestamp_format,
    )

    with (
        patch(
            "nemo_safe_synthesizer.generation.vllm_backend.load_json",
            return_value=schema,
        ),
        patch(
            "nemo_safe_synthesizer.generation.vllm_backend.utils.create_schema_prompt",
            return_value="test prompt",
        ),
        patch(
            "nemo_safe_synthesizer.generation.vllm_backend.create_processor",
            return_value=processor,
        ),
    ):
        return TimeseriesBackend(config=config, model_metadata=model_metadata, workdir=workdir)


class TestParseTimestampSeconds:
    """Tests for the _parse_timestamp_seconds method."""

    def test_parses_datetime_format(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test parsing datetime string to seconds."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        result = backend._parse_timestamp_seconds("2024-01-01 00:00:00")

        assert result is not None
        assert isinstance(result, int)

    def test_parses_elapsed_time_format(self, timeseries_elapsed_params, timeseries_model_metadata, mock_workdir):
        """Test parsing elapsed time values (int, str, float)."""
        backend = create_timeseries_backend(timeseries_elapsed_params, timeseries_model_metadata, mock_workdir)

        # Test integer input
        assert backend._parse_timestamp_seconds(3600) == 3600
        # Test string input
        assert backend._parse_timestamp_seconds("3600") == 3600
        # Test float input (truncated)
        assert backend._parse_timestamp_seconds(3600.5) == 3600

    def test_returns_none_for_invalid_values(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that invalid values return None."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        assert backend._parse_timestamp_seconds(None) is None
        assert backend._parse_timestamp_seconds("invalid") is None


class TestAdvanceExpectedTime:
    """Tests for _advance_expected_time method."""

    def test_advance_expected_time(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test advancing timestamp by interval."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        assert backend._advance_expected_time(0) == 3600
        assert backend._advance_expected_time(3600) == 7200


class TestHasReachedStopTime:
    """Tests for the _has_reached_stop_time method."""

    def test_returns_true_at_or_past_stop(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test returns True when record is at or past stop time."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        # At stop time
        records_at_stop = [{"timestamp": "2024-01-01 03:00:00", "value": 1}]
        assert backend._has_reached_stop_time(records_at_stop) is True

        # Past stop time
        records_past = [{"timestamp": "2024-01-01 04:00:00", "value": 1}]
        assert backend._has_reached_stop_time(records_past) is True

    def test_returns_false_before_stop(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test returns False when record is before stop time."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        records_before = [{"timestamp": "2024-01-01 02:00:00", "value": 1}]
        assert backend._has_reached_stop_time(records_before) is False

    def test_returns_false_for_empty_records(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test returns False for empty or None records."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        assert backend._has_reached_stop_time([]) is False
        assert backend._has_reached_stop_time(None) is False


class TestComputeExpectedRecordsPerGroup:
    """Tests for the _compute_expected_records_per_group method."""

    def test_computes_expected_records(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test expected records calculation: (stop - start) / interval + 1."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        # 3 hours = 3 intervals + 1 = 4 records
        result = backend._compute_expected_records_per_group()
        assert result == 4

    def test_with_elapsed_time(self, timeseries_elapsed_params, timeseries_model_metadata, mock_workdir):
        """Test calculation with elapsed time format."""
        backend = create_timeseries_backend(timeseries_elapsed_params, timeseries_model_metadata, mock_workdir)

        # 10800 seconds / 3600 interval + 1 = 4 records
        result = backend._compute_expected_records_per_group()
        assert result == 4


class TestSortDataframe:
    """Tests for the _sort_dataframe method."""

    def test_sorts_by_group_and_timestamp(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test sorting by group then timestamp."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        df = pd.DataFrame(
            {
                "group_id": ["B", "A", "B", "A"],
                "timestamp": [
                    "2024-01-01 02:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                ],
                "value": [1, 2, 3, 4],
            }
        )

        df_sorted = backend._sort_dataframe(df)

        assert list(df_sorted["group_id"]) == ["A", "A", "B", "B"]
        assert list(df_sorted["timestamp"]) == [
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
        ]

    def test_removes_pseudo_group_column(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that PSEUDO_GROUP_COLUMN is removed from output."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        df = pd.DataFrame(
            {
                "group_id": ["A", "A"],
                "timestamp": ["2024-01-01 01:00:00", "2024-01-01 02:00:00"],
                PSEUDO_GROUP_COLUMN: [0, 0],
            }
        )

        df_sorted = backend._sort_dataframe(df)

        assert PSEUDO_GROUP_COLUMN not in df_sorted.columns

    def test_handles_empty_dataframe(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that empty DataFrame is handled correctly."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        df_empty = pd.DataFrame()
        df_result = backend._sort_dataframe(df_empty)

        assert df_result.empty


class TestBuildProgressSnapshots:
    """Tests for the _build_progress_snapshots method."""

    def test_creates_snapshots_at_percentages(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that snapshots are created at 25%, 50%, 75%."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        snapshots = backend._build_progress_snapshots(total=100)

        assert len(snapshots) == 3
        assert snapshots[0].threshold == 25
        assert snapshots[1].threshold == 50
        assert snapshots[2].threshold == 75

    def test_returns_empty_for_zero_total(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that empty list is returned for total <= 0."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        assert backend._build_progress_snapshots(total=0) == []
        assert backend._build_progress_snapshots(total=-1) == []

    def test_deduplicates_thresholds(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that duplicate thresholds are deduplicated for small totals."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        # For total=2: 25%=1, 50%=1, 75%=2 -> deduped to 1, 2
        snapshots = backend._build_progress_snapshots(total=2)

        thresholds = [c.threshold for c in snapshots]
        assert len(thresholds) == len(set(thresholds))


class TestUpdateGroupState:
    """Tests for the _update_group_state method."""

    def test_appends_records_and_updates_prefill(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that records are appended and prefill is regenerated."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        state = GroupState(
            group_id="test",
            initial_prefill="",
            current_prefill="",
            expected_records=10,
        )

        records = [
            {"timestamp": "2024-01-01 00:00:00", "value": 1},
            {"timestamp": "2024-01-01 01:00:00", "value": 2},
        ]

        backend._update_group_state(state, records)

        assert len(state.recent_records) == 2
        assert '"timestamp"' in state.current_prefill
        assert state.last_timestamp_seconds is not None


class TestGetTimestampFromPrefill:
    """Tests for the _get_timestamp_from_prefill method."""

    def test_extracts_timestamp_from_prefill(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test extracting timestamp from prefill string."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        prefill = '{"timestamp": "2024-01-01 00:00:00", "value": 1}\n{"timestamp": "2024-01-01 01:00:00", "value": 2}\n'
        result = backend._get_timestamp_from_prefill(prefill)

        assert result is not None
        assert isinstance(result, int)

    def test_returns_none_for_empty_prefill(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that empty prefill returns None."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        assert backend._get_timestamp_from_prefill("") is None
        assert backend._get_timestamp_from_prefill(None) is None
