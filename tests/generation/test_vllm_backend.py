# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the VllmBackend class private methods."""

from unittest.mock import MagicMock, patch

import pytest

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import (
    DataParameters,
    GenerateParameters,
    SafeSynthesizerParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.defaults import DEFAULT_SAMPLING_PARAMETERS
from nemo_safe_synthesizer.generation.vllm_backend import VllmBackend  # noqa: F401


@pytest.fixture
def mock_model_metadata(fixture_session_cache_dir):
    """Create a mock model metadata object."""
    metadata = MagicMock()
    metadata.save_path = fixture_session_cache_dir / "save"
    metadata.adapter_path = fixture_session_cache_dir / "adapter"
    metadata.instruction = "Generate data"
    metadata.prompt_config = MagicMock()
    metadata.prompt_config.template = "[INST] {instruction} {schema} [/INST]"
    metadata.prompt_config.bos_token = "<s>"
    metadata.prompt_config.eos_token = "</s>"
    return metadata


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

    # Verify directories exist
    assert workdir.project_dir.exists(), f"Project dir not created: {workdir.project_dir}"
    assert workdir.run_dir.exists(), f"Run dir not created: {workdir.run_dir}"
    assert workdir.train.path.exists(), f"Train dir not created: {workdir.train.path}"
    assert workdir.generate.path.exists(), f"Generate dir not created: {workdir.generate.path}"
    assert workdir.train.adapter.path.exists(), f"Adapter path not created: {workdir.train.adapter.path}"

    return workdir


@pytest.fixture
def base_params():
    """Create basic SafeSynthesizerParameters for testing."""
    return SafeSynthesizerParameters(
        data=DataParameters(
            group_training_examples_by=None,
            order_training_examples_by=None,
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
    )


@pytest.fixture
def params_with_structured_generation_regex(base_params):
    """Create params with structured generation enabled using regex."""
    base_params.generation.use_structured_generation = True
    base_params.generation.structured_generation_schema_method = "regex"
    base_params.generation.structured_generation_backend = "xgrammar"
    return base_params


@pytest.fixture
def params_with_structured_generation_json(base_params):
    """Create params with structured generation enabled using json_schema."""
    base_params.generation.use_structured_generation = True
    base_params.generation.structured_generation_schema_method = "json_schema"
    base_params.generation.structured_generation_backend = "xgrammar"
    return base_params


@pytest.fixture
def mock_schema():
    """Create a mock JSON schema."""
    return {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        }
    }


def create_backend(config, model_metadata, schema, workdir):
    """Helper to create a VllmBackend instance with mocked dependencies."""
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
            return_value=MagicMock(),
        ),
    ):
        from nemo_safe_synthesizer.generation.vllm_backend import VllmBackend

        return VllmBackend(config=config, model_metadata=model_metadata, workdir=workdir)


class TestBuildStructuredOutputParams:
    """Tests for the _build_structured_output_params method."""

    def test_returns_none_when_structured_generation_disabled(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that None is returned when structured generation is disabled."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        result = backend._build_structured_output_params()

        assert result is None

    def test_returns_params_with_regex_when_regex_method(
        self,
        params_with_structured_generation_regex,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
    ):
        """Test that StructuredOutputsParams with regex is returned when regex method is used."""
        backend = create_backend(
            params_with_structured_generation_regex,
            mock_model_metadata,
            mock_schema,
            mock_workdir,
        )

        with patch(
            "nemo_safe_synthesizer.generation.vllm_backend.build_json_based_regex",
            return_value="test_regex_pattern",
        ) as mock_build_regex:
            result = backend._build_structured_output_params()
            mock_build_regex.assert_called_once_with(
                mock_schema,
                params_with_structured_generation_regex,
                mock_model_metadata.prompt_config.bos_token,
                mock_model_metadata.prompt_config.eos_token,
            )
            assert result is not None
            assert result.regex == "test_regex_pattern"

    def test_returns_params_with_json_when_json_schema_method(
        self,
        params_with_structured_generation_json,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
    ):
        """Test that StructuredOutputsParams with json is returned when json_schema method is used."""
        backend = create_backend(
            params_with_structured_generation_json,
            mock_model_metadata,
            mock_schema,
            mock_workdir,
        )

        result = backend._build_structured_output_params()

        assert result is not None
        assert result.json == mock_schema

    def test_config_with_grouping_passed_to_build_regex(
        self, params_with_structured_generation_regex, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that config with group_training_examples_by set is passed to build_json_based_regex."""
        params_with_structured_generation_regex.data.group_training_examples_by = "category"
        backend = create_backend(
            params_with_structured_generation_regex,
            mock_model_metadata,
            mock_schema,
            mock_workdir,
        )

        with patch(
            "nemo_safe_synthesizer.generation.vllm_backend.build_json_based_regex",
            return_value="test_regex_pattern",
        ) as mock_build_regex:
            backend._build_structured_output_params()
            mock_build_regex.assert_called_once()
            call_args, _ = mock_build_regex.call_args
            assert call_args[1].data.group_training_examples_by == "category"


class TestResolveTemperature:
    """Tests for the _resolve_temperature method."""

    def test_raises_when_do_sample_false_and_temperature_nonzero(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that ValueError is raised when do_sample is False but temperature > 0."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        with pytest.raises(ValueError, match="Cannot set a nonzero temperature"):
            backend._resolve_temperature({"do_sample": False, "temperature": 0.5})

    def test_returns_zero_when_do_sample_false(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that 0.0 is returned when do_sample is False (greedy decoding)."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        result = backend._resolve_temperature({"do_sample": False})

        assert result == 0.0

    def test_returns_provided_temperature(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that the provided temperature value is returned."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        result = backend._resolve_temperature({"temperature": 0.7})

        assert result == 0.7

    def test_returns_default_when_temperature_undefined(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that default temperature is returned when not provided."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        result = backend._resolve_temperature({})

        assert result == DEFAULT_SAMPLING_PARAMETERS["temperature"]

    def test_do_sample_false_takes_precedence_over_zero_temp(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that do_sample=False with temperature=0.0 returns 0.0 without error."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        # This should not raise because temp == 0.0 is fine with do_sample=False
        result = backend._resolve_temperature({"do_sample": False, "temperature": 0.0})

        assert result == 0.0


class TestGetApiParamMapping:
    """Tests for the _get_api_param_mapping method."""

    def test_mapping_includes_expected_keys(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that the mapping includes all expected parameter keys."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)

        expected_keys = {
            "max_new_tokens",
            "eos_token_id",
            "typical_p",
            "temperature",
            "num_beams",
            "early_stopping",
        }
        assert set(mapping.keys()) == expected_keys

    def test_max_new_tokens_maps_to_max_tokens(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that max_new_tokens is mapped to max_tokens."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)
        key, value = mapping["max_new_tokens"](100)

        assert key == "max_tokens"
        assert value == 100

    def test_eos_token_id_maps_to_stop_token_ids_single(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that a single eos_token_id is converted to a list."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)
        key, value = mapping["eos_token_id"](42)

        assert key == "stop_token_ids"
        assert value == [42]

    def test_eos_token_id_maps_to_stop_token_ids_list(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that a list eos_token_id stays as a list."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)
        key, value = mapping["eos_token_id"]([1, 2, 3])

        assert key == "stop_token_ids"
        assert value == [1, 2, 3]

    def test_temperature_uses_resolved_value(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that temperature mapping uses the resolved temperature value."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.8)
        key, value = mapping["temperature"](1.0)  # Input value is ignored

        assert key == "temperature"
        assert value == 0.8  # Uses resolved, not input

    def test_num_beams_greater_than_one_maps_to_beam_width(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that num_beams > 1 maps to beam_width."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)
        key, value = mapping["num_beams"](4)

        assert key == "beam_width"
        assert value == 4

    def test_num_beams_one_returns_none(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that num_beams == 1 returns (None, None) to exclude from params."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)
        key, value = mapping["num_beams"](1)

        assert key is None
        assert value is None

    def test_early_stopping_returns_none(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that early_stopping returns (None, None) as it's not used in vLLM."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)
        key, value = mapping["early_stopping"](True)

        assert key is None
        assert value is None

    def test_typical_p_creates_logits_processor(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that typical_p creates a TypicalLogitsWarperWrapper."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)
        key, value = mapping["typical_p"](0.95)

        assert key == "logits_processors"
        assert len(value) == 1
        # Verify it's a TypicalLogitsWarperWrapper
        from nemo_safe_synthesizer.generation.vllm_backend import (
            TypicalLogitsWarperWrapper,
        )

        assert isinstance(value[0], TypicalLogitsWarperWrapper)


class TestTransformKwargsToSamplingParams:
    """Tests for the _transform_kwargs_to_sampling_params method."""

    def test_transforms_known_params_using_mapping(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that known parameters are transformed using the mapping."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        api_mapping = {
            "max_new_tokens": lambda x: ("max_tokens", x),
        }

        result = backend._transform_kwargs_to_sampling_params(kwargs={"max_new_tokens": 256}, api_mapping=api_mapping)

        assert "max_tokens" in result
        assert result["max_tokens"] == 256

    def test_passes_through_unknown_params(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that unknown parameters are passed through unchanged."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        api_mapping = {}

        result = backend._transform_kwargs_to_sampling_params(
            kwargs={"custom_param": "custom_value"}, api_mapping=api_mapping
        )

        assert "custom_param" in result
        assert result["custom_param"] == "custom_value"

    def test_excludes_params_when_mapping_returns_none(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Test that params are excluded when mapping returns (None, None)."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        api_mapping = {
            "early_stopping": lambda x: (None, None),
        }

        result = backend._transform_kwargs_to_sampling_params(
            kwargs={"early_stopping": True, "other": "value"},
            api_mapping=api_mapping,
        )

        # Parameters mapped to (None, None) should be excluded from the result
        assert None not in result
        assert "early_stopping" not in result
        assert result.get("other") == "value"

    def test_handles_multiple_transforms(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that multiple parameters are transformed correctly."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        api_mapping = {
            "max_new_tokens": lambda x: ("max_tokens", x),
            "eos_token_id": lambda x: ("stop_token_ids", [x]),
        }

        result = backend._transform_kwargs_to_sampling_params(
            kwargs={
                "max_new_tokens": 512,
                "eos_token_id": 2,
                "top_p": 0.9,
            },
            api_mapping=api_mapping,
        )

        assert result["max_tokens"] == 512
        assert result["stop_token_ids"] == [2]
        assert result["top_p"] == 0.9

    def test_empty_kwargs_returns_empty_dict(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that empty kwargs returns an empty dict."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        result = backend._transform_kwargs_to_sampling_params(kwargs={}, api_mapping={})

        assert result == {}
