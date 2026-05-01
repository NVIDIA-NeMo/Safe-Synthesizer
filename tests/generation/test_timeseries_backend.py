# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TimeseriesBackend class private methods."""

import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from transformers import PretrainedConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

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
from nemo_safe_synthesizer.generation.results import GenerationBatches
from nemo_safe_synthesizer.generation.timeseries_backend import (
    GroupState,
    GroupProcessingResult,
    TimeseriesBackend,
)
from nemo_safe_synthesizer.llm.metadata import (
    GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER,
    LLMPromptConfig,
    ModelMetadata,
)

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


class TestGenerateParallelGroups:
    """Tests for processing vLLM completions during parallel time-series generation."""

    def test_records_completion_finish_reasons(
        self, timeseries_base_params, timeseries_model_metadata, mock_workdir
    ):
        """The generated batch should preserve vLLM finish reasons for stopping logic."""
        backend = create_timeseries_backend(timeseries_base_params, timeseries_model_metadata, mock_workdir)
        backend.llm = MagicMock()
        backend.llm.generate.return_value = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        finish_reason="length",
                        text='{"timestamp": "2024-01-01 01:00:00", "value": 3}\n',
                        token_ids=[1, 2, 3],
                    )
                ]
            )
        ]
        backend._groups = ["group_A"]

        captured = {}

        def _capture_result(state, batch, invalid_fraction_threshold):  # noqa: ARG001
            captured["batch"] = batch
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
        assert batches.num_length_truncated_completions == 1


class TestGenerationMaxTokensPlumbing:
    """``SamplingParams.max_tokens`` is sourced from ``metadata.generation_max_tokens_for``."""

    class _WhitespaceTokenizer:
        """Tiny tokenizer stand-in that makes long text payloads countable."""

        @staticmethod
        def encode(text: str) -> list[str]:
            return re.compile(r"\s+").split(text)

    @staticmethod
    def _jsonl_prefill_from_dataframe(df: pd.DataFrame, rows: int = 3) -> str:
        """Serialize seed rows like a time-series initial prefill."""
        return " " + "\n".join(json.dumps(record) for record in df.head(rows).to_dict("records"))

    @staticmethod
    def _capture_sampling_params(
        backend,
        *,
        groups: list[str] | None = None,
        group_prefills: dict[str, str] | None = None,
        prompt_token_count: int | None = None,
    ) -> SamplingParams:
        """Invoke ``generate`` with ``_generate_parallel_groups`` stubbed out.

        Returns the ``SamplingParams`` the backend constructed. ``groups``
        and ``group_prefills`` seed minimal state so ``generate()`` reaches
        SamplingParams construction; ``prompt_token_count`` overrides the
        backend's prompt-token helper to drive the prompt-length clamp
        deterministically without standing up a real vLLM engine.
        """
        captured: dict[str, SamplingParams] = {}

        def _capture(*, batches, sampling_params, progress_snapshots):  # noqa: ARG001
            captured["sp"] = sampling_params
            raise StopIteration("short-circuit")

        backend._generate_parallel_groups = _capture
        backend._groups = groups if groups is not None else []
        backend._group_prefills = group_prefills if group_prefills is not None else {}
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

    def test_dataset_shaped_prefill_can_consume_entire_context(
        self, fixture_pems_sf_sample_dataset, timeseries_elapsed_params, timeseries_model_metadata, mock_workdir
    ):
        """The wide PEMS-SF fixture can make three seed rows consume the full context."""
        pems_df = fixture_pems_sf_sample_dataset.to_pandas()
        timeseries_elapsed_params.data.group_training_examples_by = "e_id"
        timeseries_elapsed_params.data.order_training_examples_by = "s_index"
        timeseries_elapsed_params.time_series.timestamp_column = "s_index"
        timeseries_elapsed_params.time_series.stop_timestamp = "4"
        timeseries_model_metadata.max_tokens_per_example = None
        timeseries_model_metadata.initial_prefill = {"0": self._jsonl_prefill_from_dataframe(pems_df)}
        schema = {"properties": {column: {"type": "number"} for column in pems_df.columns}}
        backend = create_timeseries_backend(timeseries_elapsed_params, timeseries_model_metadata, mock_workdir, schema)
        backend.llm = MagicMock()
        backend.llm.get_tokenizer.return_value = self._WhitespaceTokenizer()

        prompt_len = backend._get_prompt_token_count()

        assert prompt_len >= timeseries_model_metadata.max_seq_length
        assert timeseries_model_metadata.generation_max_tokens_for(prompt_len) == 0
        with pytest.raises(VLLMValidationError, match="max_tokens must be at least 1"):
            backend.generate()
