# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TimeseriesBackend class private methods."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from transformers import PretrainedConfig
from vllm.sampling_params import SamplingParams

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import (
    DataParameters,
    GenerateParameters,
    SafeSynthesizerParameters,
    TimeSeriesParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.data_processing.actions.utils import MetadataColumns
from nemo_safe_synthesizer.data_processing.assembler import Example
from nemo_safe_synthesizer.data_processing.record_utils import ParsedRecord
from nemo_safe_synthesizer.defaults import DEFAULT_MAX_SEQ_LENGTH, PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.generation.batch import Batch
from nemo_safe_synthesizer.generation.processors import TimeSeriesDataProcessor
from nemo_safe_synthesizer.generation.results import GenerationBatches, GenerationStatus
from nemo_safe_synthesizer.generation.timeseries_backend import (
    GroupProcessingResult,
    GroupState,
    RecordPromptState,
    TimeseriesBackend,
    _ResolvedTimeseriesSettings,
)
from nemo_safe_synthesizer.llm.metadata import (
    GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER,
    LLMPromptConfig,
    ModelMetadata,
)

PROMPT_TEMPLATE = "[INST] {instruction} {schema} [/INST]{prefill}"


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
    metadata.timeseries_group_values = ["group_A", "group_B"]
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
        timestamp_column = config.time_series.timestamp_column
        assert timestamp_column is not None
        timestamp_type = "integer" if config.time_series.timestamp_format == "elapsed_seconds" else "string"
        schema = {
            "properties": {
                "group_id": {"type": "string"},
                timestamp_column: {"type": timestamp_type},
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

    def test_single_timestamp_without_interval_has_one_expected_record(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        """One-record groups remain bounded when no interval can be inferred."""
        timeseries_base_params.time_series.timestamp_interval_seconds = None
        timeseries_base_params.time_series.stop_timestamp = timeseries_base_params.time_series.start_timestamp
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        assert backend._compute_expected_records_per_group() == 1


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

    def test_restores_source_column_order(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Generated output should match the user's input column order."""
        timeseries_model_metadata.timeseries_source_columns = ["value", "group_id", "timestamp"]
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        df = pd.DataFrame(
            {
                "group_id": ["A"],
                "timestamp": ["2024-01-01 00:00:00"],
                "value": [1],
            }
        )

        result = backend._sort_dataframe(df)

        assert list(result.columns) == ["value", "group_id", "timestamp"]


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


class TestInitGroupState:
    """Tests for partial-record group initialization."""

    @pytest.mark.parametrize(
        ("attribute", "expected_name"),
        [
            pytest.param("timestamp_column", "timestamp column", id="timestamp-column"),
            pytest.param("start_timestamp", "start timestamp", id="start-timestamp"),
            pytest.param("group_training_examples_by", "group column", id="group-column"),
        ],
    )
    def test_requires_resolved_timeseries_settings(
        self,
        timeseries_base_params,
        attribute,
        expected_name,
    ):
        """Generation rejects configs that skipped time-series preprocessing."""
        target = (
            timeseries_base_params.data
            if attribute == "group_training_examples_by"
            else timeseries_base_params.time_series
        )
        setattr(target, attribute, None)

        with pytest.raises(GenerationError, match=expected_name):
            _ResolvedTimeseriesSettings.from_config(timeseries_base_params)

    def test_requires_group_registry(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Artifacts without typed group values must be retrained."""
        timeseries_model_metadata.timeseries_group_values = None

        with pytest.raises(GenerationError, match="no time-series group registry"):
            create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

    def test_initializes_with_partial_record_prefix(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        state = backend._init_group_state("group_A")

        expected = '{"group_id":"group_A","timestamp":"2024-01-01 00:00:00","'
        assert state.prompt_state.prefix == expected
        assert state.prompt_state.prompt_segments == expected
        assert state.prompt_state.completion_prefix == expected
        assert state.group_ordinal == 1
        assert state.last_timestamp_seconds is None
        assert backend._groups == ["group_A", "group_B"]
        assert backend._history_window_size == 3

    def test_missing_prefix_error_uses_group_ordinal(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        sensitive_group = "ACCT-SENSITIVE-1234"
        timeseries_model_metadata.timeseries_group_values = [sensitive_group]
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        del backend._group_prefixes[sensitive_group]

        with pytest.raises(GenerationError) as exc_info:
            backend._init_group_state(sensitive_group)

        assert sensitive_group not in str(exc_info.value)
        assert "group 1" in str(exc_info.value)

    def test_prompt_tokens_match_training_example_prefix(
        self,
        timeseries_base_params,
        timeseries_model_metadata,
        mock_workdir,
        fixture_tokenizer,
    ):
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend.llm = MagicMock()
        backend.llm.get_tokenizer.return_value = fixture_tokenizer
        partial_prefix = backend._init_group_state("group_A").prompt_state.prefix
        complete_record = f'{partial_prefix}value":3}}\n'
        example = Example(
            prompt=backend.prompt,
            tokenizer=fixture_tokenizer,
            metadata=timeseries_model_metadata,
        )
        record_ids = fixture_tokenizer.encode(complete_record, add_special_tokens=False)
        example.add_sequence(
            {"input_ids": record_ids, "attention_mask": [1] * len(record_ids)},
            add_special_tokens=True,
        )

        prompt_only_ids = backend._build_prompt_token_ids("")
        partial_prefix_ids = fixture_tokenizer.encode(partial_prefix, add_special_tokens=False)
        generation_ids = backend._build_prompt_token_ids(partial_prefix)

        assert generation_ids == [*prompt_only_ids, *partial_prefix_ids]
        assert example.input_ids[: len(generation_ids)] == generation_ids

    def test_history_prompt_tokens_match_open_training_sequence(
        self,
        timeseries_base_params,
        timeseries_model_metadata,
        mock_workdir,
        fixture_tokenizer,
    ):
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend.llm = MagicMock()
        backend.llm.get_tokenizer.return_value = fixture_tokenizer
        records = [
            '{"group_id":"group_A","timestamp":"2024-01-01 00:00:00","value":1}\n',
            '{"group_id":"group_A","timestamp":"2024-01-02 00:00:00","value":2}\n',
        ]
        record_ids: list[int] = []
        for record in records:
            record_ids.extend(fixture_tokenizer.encode(record, add_special_tokens=False))
        example = Example(
            prompt=backend.prompt,
            tokenizer=fixture_tokenizer,
            metadata=timeseries_model_metadata,
        )
        example.add_sequence(
            {"input_ids": record_ids, "attention_mask": [1] * len(record_ids)},
            add_special_tokens=True,
        )

        generation_ids = backend._build_prompt_token_ids(records)

        assert generation_ids == example.input_ids[:-1]


class TestUpdateGroupState:
    """Tests for the _update_group_state method."""

    def test_appends_records_and_updates_history(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """Test that records are appended and history is regenerated."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        state = GroupState(
            group_id="test",
            group_ordinal=1,
            prompt_state=RecordPromptState(prefix='{"group_id":"test","timestamp":"2024-01-01 00:00:00","'),
            expected_records=10,
        )

        records = [
            ParsedRecord(
                text='{"timestamp":"2024-01-01 00:00:00","value":1}',
                parsed={"timestamp": "2024-01-01 00:00:00", "value": 1},
            ),
            ParsedRecord(
                text='{"timestamp":"2024-01-01 01:00:00","value":2}',
                parsed={"timestamp": "2024-01-01 01:00:00", "value": 2},
            ),
        ]

        backend._update_group_state(state, records)

        assert len(state.prompt_state.history) == 2
        # The rebuilt history must reuse the records' emitted text verbatim
        # (training dialect): no leading whitespace, newline-terminated records,
        # and single newlines between them.
        assert state.prompt_state.history_text == (
            '{"timestamp":"2024-01-01 00:00:00","value":1}\n{"timestamp":"2024-01-01 01:00:00","value":2}\n'
        )
        assert state.last_timestamp_seconds is not None
        assert state.prompt_state.completion_prefix == ""

    def test_keeps_prefix_when_no_records_are_accepted(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        state = backend._init_group_state("group_A")

        backend._update_group_state(state, [])

        assert state.prompt_state.using_prefix is True
        assert state.prompt_state.completion_prefix == state.prompt_state.prefix

    def test_truncates_context_without_reserializing_records(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend._history_window_size = 1
        state = backend._init_group_state("group_A")
        records = [
            ParsedRecord(
                text='{"group_id":"group_A","timestamp":"2024-01-01 00:00:00","value":1}',
                parsed={"group_id": "group_A", "timestamp": "2024-01-01 00:00:00", "value": 1},
            ),
            ParsedRecord(
                text='{"group_id":"group_A","timestamp":"2024-01-01 01:00:00","value":2}',
                parsed={"group_id": "group_A", "timestamp": "2024-01-01 01:00:00", "value": 2},
            ),
        ]

        backend._update_group_state(state, records)

        assert state.prompt_state.history == [records[-1]]
        assert state.prompt_state.history_text == f"{records[-1].text}\n"

    def test_updates_timestamp_for_empty_string_column_name(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        """Test that an empty-string timestamp column name remains valid."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend._time_column = ""
        state = GroupState(
            group_id="test",
            group_ordinal=1,
            prompt_state=RecordPromptState(prefix=""),
            expected_records=10,
            last_timestamp_seconds=0,
        )
        timestamp = "2024-01-01 01:00:00"
        records = [ParsedRecord(text='{"":"2024-01-01 01:00:00"}', parsed={"": timestamp})]

        backend._update_group_state(state, records)

        assert state.last_timestamp_seconds == backend._parse_timestamp_seconds(timestamp)


class TestProcessGroupResult:
    """Tests for post-processed group progress and retry accounting."""

    @staticmethod
    def _batch(*, valid_records: int = 1, invalid_records: int = 0) -> Batch:
        batch = MagicMock(spec=Batch)
        batch.num_valid_records = valid_records
        batch.num_invalid_records = invalid_records
        total_records = valid_records + invalid_records
        batch.valid_record_fraction = valid_records / total_records if total_records else 0
        return batch

    @staticmethod
    def _record(timestamp: str) -> ParsedRecord:
        return ParsedRecord(
            text=f'{{"timestamp":"{timestamp}","value":1}}',
            parsed={"timestamp": timestamp, "value": 1},
        )

    def test_timestamp_progress_resets_no_progress_count(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        state = backend._init_group_state("group_A")
        state.no_progress_count = 1
        records = [self._record("2024-01-01 01:00:00")]

        result = backend._process_group_result(
            state,
            self._batch(),
            records,
            invalid_fraction_threshold=0.8,
        )

        assert result == GroupProcessingResult.IN_PROGRESS
        assert state.no_progress_count == 0
        assert state.total_valid_records == 1

    def test_fails_after_consecutive_batches_without_timestamp_progress(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        timeseries_base_params.generation.patience = 2
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        state = backend._init_group_state("group_A")
        state.last_timestamp_seconds = backend._parse_timestamp_seconds("2024-01-01 00:00:00")
        records = [self._record("2024-01-01 00:00:00")]

        first_result = backend._process_group_result(
            state,
            self._batch(),
            records,
            invalid_fraction_threshold=0.8,
        )
        second_result = backend._process_group_result(
            state,
            self._batch(),
            records,
            invalid_fraction_threshold=0.8,
        )

        assert first_result == GroupProcessingResult.IN_PROGRESS
        assert second_result == GroupProcessingResult.FAILED
        assert state.no_progress_count == 2
        assert state.failed is True


class TestBuildModifiedSamplingParamsStopPropagation:
    """Tests that _build_modified_sampling_params propagates ignore_eos and other stop fields."""

    def test_propagates_ignore_eos_false(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """ignore_eos=False from the upstream VllmBackend must survive the rebuild."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        original = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            min_p=0.0,
            max_tokens=2048,
            repetition_penalty=1.0,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            ignore_eos=False,
        )

        modified, _ = backend._build_modified_sampling_params(original, num_active=2)

        assert modified.ignore_eos is False
        assert modified.stop == []
        assert modified.stop_token_ids == []

    def test_propagates_none_stop_values(self, timeseries_base_params, timeseries_model_metadata, mock_workdir):
        """When original has no stop values, the rebuilt params should also have none."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        original = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            min_p=0.0,
            max_tokens=2048,
            repetition_penalty=1.0,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            ignore_eos=False,
        )

        modified, _ = backend._build_modified_sampling_params(original, num_active=2)

        assert modified.ignore_eos is False
        assert modified.stop == []
        assert modified.stop_token_ids == []

    def test_clamps_max_tokens_to_current_rolling_prompt(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        modified, _ = backend._build_modified_sampling_params(
            SamplingParams(max_tokens=1000),
            num_active=2,
            max_prompt_tokens=1800,
        )

        assert modified.max_tokens == 248

    def test_rejects_rolling_prompt_that_fills_context(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        with pytest.raises(GenerationError, match="leaves no room"):
            backend._build_modified_sampling_params(
                SamplingParams(max_tokens=1000),
                num_active=2,
                max_prompt_tokens=timeseries_model_metadata.max_seq_length,
            )


class TestGenerateParallelGroups:
    """Tests for processing vLLM completions during parallel time-series generation."""

    def test_data_action_rejection_prevents_completion_and_redacts_diagnostics(
        self,
        timeseries_base_params,
        timeseries_model_metadata,
        mock_workdir,
        fixture_tokenizer,
    ):
        sensitive_group = "ACCT-SENSITIVE-1234"
        timeseries_base_params.generation.patience = 2
        timeseries_base_params.time_series.stop_timestamp = timeseries_base_params.time_series.start_timestamp
        timeseries_model_metadata.timeseries_group_values = [sensitive_group]
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend.llm = MagicMock()
        backend.llm.get_tokenizer.return_value = fixture_tokenizer
        backend.llm.generate.return_value = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        finish_reason="stop",
                        text='value":3}\n',
                        token_ids=[1, 2, 3],
                    )
                ]
            )
        ]

        def reject_all_records(batch: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            assert df.empty
            rejected_df = batch.copy()
            rejected_df[MetadataColumns.REJECT_REASON.value] = "test rejection"
            return batch.iloc[0:0].copy(), rejected_df

        batches = GenerationBatches(target_num_records=100, data_actions_fn=reject_all_records)
        with patch("nemo_safe_synthesizer.generation.timeseries_backend.logger") as mock_logger:
            all_groups_succeeded = backend._generate_parallel_groups(
                batches=batches,
                sampling_params=SamplingParams(max_tokens=10),
                progress_snapshots=[],
            )

        assert all_groups_succeeded is False
        assert backend.llm.generate.call_count == 2
        assert batches.num_valid_records == 0
        assert batches.status == GenerationStatus.IN_PROGRESS
        assert sensitive_group not in repr(mock_logger.method_calls)
        warning_extras = [call.kwargs.get("extra", {}) for call in mock_logger.warning.call_args_list]
        assert any(extra.get("group_ordinal") == 1 for extra in warning_extras)
        assert any(extra.get("retry_count") == 2 for extra in warning_extras)

    def test_completion_log_uses_group_ordinal(
        self,
        timeseries_base_params,
        timeseries_model_metadata,
        mock_workdir,
        fixture_tokenizer,
    ):
        sensitive_group = "ACCT-SENSITIVE-1234"
        timeseries_base_params.time_series.stop_timestamp = timeseries_base_params.time_series.start_timestamp
        timeseries_model_metadata.timeseries_group_values = [sensitive_group]
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend.llm = MagicMock()
        backend.llm.get_tokenizer.return_value = fixture_tokenizer
        backend.llm.generate.return_value = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        finish_reason="stop",
                        text='value":3}\n',
                        token_ids=[1, 2, 3],
                    )
                ]
            )
        ]
        batches = GenerationBatches(target_num_records=1)

        with patch("nemo_safe_synthesizer.generation.timeseries_backend.logger") as mock_logger:
            all_groups_succeeded = backend._generate_parallel_groups(
                batches=batches,
                sampling_params=SamplingParams(max_tokens=10),
                progress_snapshots=[],
            )

        assert all_groups_succeeded is True
        assert batches.num_valid_records == 1
        assert sensitive_group not in repr(mock_logger.method_calls)
        completion_call = next(
            call
            for call in mock_logger.info.call_args_list
            if call.args[0] == "Time-series group completed after reaching the stop timestamp."
        )
        assert completion_call.kwargs["extra"] == {
            "group_ordinal": 1,
            "groups_completed": 1,
            "total_groups": 1,
        }

    def test_uses_per_group_retries_instead_of_global_zero_record_stop(
        self,
        timeseries_base_params,
        timeseries_model_metadata,
        mock_workdir,
        fixture_tokenizer,
    ):
        timeseries_base_params.generation.patience = 2
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend.llm = MagicMock()
        backend.llm.get_tokenizer.return_value = fixture_tokenizer
        backend.llm.generate.return_value = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        finish_reason="stop",
                        text="not-json",
                        token_ids=[1, 2, 3],
                    )
                ]
            )
        ]
        backend._groups = ["group_A"]
        batches = GenerationBatches(target_num_records=100)

        all_groups_succeeded = backend._generate_parallel_groups(
            batches=batches,
            sampling_params=SamplingParams(max_tokens=10),
            progress_snapshots=[],
        )

        assert all_groups_succeeded is False
        assert backend.llm.generate.call_count == 2
        assert batches.status == GenerationStatus.IN_PROGRESS

    def test_records_completion_finish_reasons(
        self,
        timeseries_base_params,
        timeseries_model_metadata,
        mock_workdir,
        fixture_tokenizer,
    ):
        """The generated batch should preserve vLLM finish reasons for stopping logic."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend.llm = MagicMock()
        backend.llm.get_tokenizer.return_value = fixture_tokenizer
        backend.llm.generate.return_value = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        finish_reason="length",
                        text='value":3}\n',
                        token_ids=[1, 2, 3],
                    )
                ]
            )
        ]
        backend._groups = ["group_A"]

        captured = {}

        def _capture_result(state, batch, accepted_records, invalid_fraction_threshold):  # noqa: ARG001
            captured["state"] = state
            captured["batch"] = batch
            captured["accepted_records"] = accepted_records
            return GroupProcessingResult.COMPLETED

        backend._process_group_result = _capture_result

        batches = GenerationBatches(target_num_records=100)
        sampling_params = SamplingParams(max_tokens=10)

        backend._generate_parallel_groups(
            batches=batches,
            sampling_params=sampling_params,
            progress_snapshots=[],
        )

        assert captured["batch"].finish_reasons["length"] == 1
        assert captured["batch"]._responses[0].valid_records == [
            {
                "group_id": "group_A",
                "timestamp": "2024-01-01 00:00:00",
                "value": 3,
            }
        ]
        assert captured["state"].prompt_state.completion_prefix.startswith('{"group_id":"group_A"')
        prompts = backend.llm.generate.call_args.kwargs["prompts"]
        assert prompts == [
            {"prompt_token_ids": backend._build_prompt_token_ids(captured["state"].prompt_state.prompt_segments)}
        ]
        assert isinstance(backend.llm.generate.call_args.kwargs["sampling_params"], SamplingParams)
        assert batches.num_length_truncated_completions == 1


class TestGenerationMaxTokensPlumbing:
    """``SamplingParams.max_tokens`` is sourced from ``metadata.generation_max_tokens_for``."""

    class _RecordingTokenizer:
        """Record encoded text and make copied wide rows overflow the context."""

        def __init__(self, forbidden_text: str, overflow_token_count: int):
            self.forbidden_text = forbidden_text
            self.overflow_token_count = overflow_token_count
            self.encoded_text: list[str] = []

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            self.encoded_text.append(text)
            token_count = self.overflow_token_count if self.forbidden_text in text else max(1, len(text.split()))
            return list(range(token_count))

    @staticmethod
    def _capture_sampling_params(
        backend,
        *,
        prompt_token_count: int | None = None,
    ) -> SamplingParams:
        """Invoke ``generate`` with ``_generate_parallel_groups`` stubbed out.

        Returns the ``SamplingParams`` the backend constructed.
        ``prompt_token_count`` overrides the backend's prompt-token helper to
        drive the prompt-length clamp deterministically without standing up a
        real vLLM engine.
        """
        captured: dict[str, SamplingParams] = {}

        def _capture(*, batches, sampling_params, progress_snapshots):  # noqa: ARG001
            captured["sp"] = sampling_params
            raise StopIteration("short-circuit")

        backend._generate_parallel_groups = _capture
        backend._groups = []
        if prompt_token_count is not None:
            backend._get_prompt_token_count = MagicMock(return_value=prompt_token_count)

        with pytest.raises(StopIteration):
            backend.generate()

        return captured["sp"]

    def test_uses_helper_when_stat_set_and_prompt_short(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        """When the prompt is short, the scaled stat drives ``max_tokens``."""
        timeseries_model_metadata.max_tokens_per_example = 1000
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        sp = self._capture_sampling_params(backend)

        expected = timeseries_model_metadata.generation_max_tokens_for(0)
        assert sp.max_tokens == expected
        assert sp.max_tokens == int(1000 * GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER)

    def test_falls_back_to_remaining_context_when_stat_unset(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        """Without the stat, SamplingParams.max_tokens == ``max_seq_length - prompt_len``."""
        assert timeseries_model_metadata.max_tokens_per_example is None
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        sp = self._capture_sampling_params(backend)

        # No engine -> prompt-token count is 0 -> clamp gives full window back.
        assert sp.max_tokens == timeseries_model_metadata.max_seq_length

    def test_prompt_length_clamps_below_scaled_stat(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        """Long prompts trigger the ``max_seq_length - prompt_len`` clamp instead of the stat."""
        # Stat * 1.2 = 1800; clamp = 2048 - 1500 = 548 wins.
        timeseries_model_metadata.max_tokens_per_example = 1500
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)

        sp = self._capture_sampling_params(backend, prompt_token_count=1500)

        assert sp.max_tokens is not None
        assert sp.max_tokens == timeseries_model_metadata.max_seq_length - 1500
        assert sp.max_tokens < int(1500 * GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER)

    def test_partial_prefix_does_not_copy_wide_training_rows_into_prompt(
        self, fixture_pems_sf_sample_dataset, timeseries_elapsed_params, timeseries_model_metadata, mock_workdir
    ):
        """A partial prefix leaves generation room even when saved training rows are wide."""
        pems_df = fixture_pems_sf_sample_dataset.to_pandas()
        timeseries_elapsed_params.data.group_training_examples_by = "e_id"
        timeseries_elapsed_params.data.order_training_examples_by = "s_index"
        timeseries_elapsed_params.time_series.timestamp_column = "s_index"
        timeseries_elapsed_params.time_series.stop_timestamp = "4"
        timeseries_model_metadata.max_tokens_per_example = None
        saved_prefill = " " + pems_df.head(3).to_json(orient="records", lines=True)
        timeseries_model_metadata.timeseries_group_values = [0]
        ordered_columns = [
            "e_id",
            "s_index",
            *(column for column in pems_df.columns if column not in {"e_id", "s_index"}),
        ]
        schema = {"properties": {column: {"type": "number"} for column in ordered_columns}}
        backend = create_timeseries_backend(timeseries_elapsed_params, timeseries_model_metadata, mock_workdir, schema)
        backend.llm = MagicMock()
        tokenizer = self._RecordingTokenizer(
            forbidden_text=saved_prefill,
            overflow_token_count=timeseries_model_metadata.max_seq_length + 1,
        )
        backend.llm.get_tokenizer.return_value = tokenizer

        prompt_len = backend._get_prompt_token_count()

        assert backend._group_prefixes == {0: '{"e_id":0.0,"s_index":0.0,"'}
        assert prompt_len < timeseries_model_metadata.max_seq_length
        assert timeseries_model_metadata.generation_max_tokens_for(prompt_len) > 0
        assert tokenizer.encoded_text
        assert all(saved_prefill not in encoded_text for encoded_text in tokenizer.encoded_text)
