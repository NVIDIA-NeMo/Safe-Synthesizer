# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the VllmBackend class private methods and module-level side effects."""

import os
from functools import partial
from unittest.mock import MagicMock, patch

import pytest

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import (
    DataParameters,
    GenerateParameters,
    SafeSynthesizerParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.config.generate import ValidationParameters
from nemo_safe_synthesizer.defaults import DEFAULT_SAMPLING_PARAMETERS
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.generation.processors import TabularDataProcessor
from nemo_safe_synthesizer.generation.vllm_observability import GenerationObservability
from nemo_safe_synthesizer.llm.metadata import ModelMetadata, RopeScaling


@pytest.fixture
def fixture_cached_nvidia_snapshot(hf_cached_snapshot_factory):
    """Realistic Hugging Face cache layout for a cached trusted model."""
    return hf_cached_snapshot_factory("nvidia/Nemotron-Mini-4B-Instruct")


@pytest.fixture
def mock_model_metadata(fixture_session_cache_dir):
    """Spec'd mock for ``ModelMetadata`` -- guards against signature drift on the helper."""
    metadata = MagicMock(spec=ModelMetadata)
    metadata.adapter_path = fixture_session_cache_dir / "adapter"
    metadata.instruction = "Generate data"
    metadata.prompt_config = MagicMock()
    metadata.prompt_config.template = "[INST] {instruction} {schema} [/INST]"
    metadata.prompt_config.bos_token = "<s>"
    metadata.prompt_config.eos_token = "</s>"
    # Default: generation uses the full context window (mirrors the real
    # helper's fallback when ``max_tokens_per_example`` is unset).
    # Individual tests override per-prompt return values where needed.
    metadata.max_seq_length = 2048
    metadata.base_max_seq_length = 2048
    metadata.rope_scaling = None
    metadata.max_tokens_per_example = None
    metadata.max_records_per_group = None
    metadata.generation_max_tokens_for.return_value = 2048
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
    workdir.dataset_profile_file.write_text(
        """
        {
          "columns": {
            "value": {
              "name": "value",
              "nullable": false,
              "constraints": {
                "kind": "integer",
                "min_value": 1,
                "max_value": 2
              }
            }
          }
        }
        """
    )

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
        ),
    )


@pytest.fixture
def params_with_structured_generation_auto(base_params):
    """Create params with structured generation enabled using auto schema method."""
    base_params.generation.structured_generation.enabled = True
    base_params.generation.structured_generation.schema_method = "auto"
    return base_params


@pytest.fixture
def params_with_structured_generation_regex(base_params):
    """Create params with structured generation enabled using regex."""
    base_params.generation.structured_generation.enabled = True
    base_params.generation.structured_generation.schema_method = "regex"
    base_params.generation.structured_generation.backend = "xgrammar"
    return base_params


@pytest.fixture
def params_with_structured_generation_json(base_params):
    """Create params with structured generation enabled using json_schema."""
    base_params.generation.structured_generation.enabled = True
    base_params.generation.structured_generation.schema_method = "json_schema"
    base_params.generation.structured_generation.backend = "xgrammar"
    return base_params


@pytest.fixture
def params_with_structured_generation_structural_tag(base_params):
    """Create params with structured generation enabled using structural_tag."""
    base_params.generation.structured_generation.enabled = True
    base_params.generation.structured_generation.schema_method = "structural_tag"
    base_params.generation.structured_generation.backend = "xgrammar"
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
    """Create a backend while retaining the legacy schema fixtures for unit tests."""
    with (
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

        backend = VllmBackend(config=config, model_metadata=model_metadata, workdir=workdir)
    backend.schema = schema
    backend.columns = list(schema["properties"])
    return backend


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
                bos_token=mock_model_metadata.prompt_config.bos_token,
                eos_token=mock_model_metadata.prompt_config.eos_token,
                default_max_records_per_group=mock_model_metadata.max_records_per_group,
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

    def test_returns_params_with_structural_tag_when_structural_tag_method(
        self,
        params_with_structured_generation_structural_tag,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
    ):
        """Test that structural_tag uses vLLM's Structural Tag constraint."""
        backend = create_backend(
            params_with_structured_generation_structural_tag,
            mock_model_metadata,
            mock_schema,
            mock_workdir,
        )

        with patch(
            "nemo_safe_synthesizer.generation.vllm_backend.build_json_structural_tag",
            return_value='{"type":"structural_tag","format":{"type":"json_schema","json_schema":{}}}',
        ) as mock_build_structural_tag:
            result = backend._build_structured_output_params()
            mock_build_structural_tag.assert_called_once_with(
                mock_schema,
                params_with_structured_generation_structural_tag,
                bos_token=mock_model_metadata.prompt_config.bos_token,
                eos_token=mock_model_metadata.prompt_config.eos_token,
                default_max_records_per_group=mock_model_metadata.max_records_per_group,
            )
            assert result is not None
            assert result.structural_tag == '{"type":"structural_tag","format":{"type":"json_schema","json_schema":{}}}'

    @pytest.mark.parametrize("backend", ["auto", "xgrammar"])
    def test_auto_resolves_to_structural_tag_on_xgrammar_backends(
        self,
        params_with_structured_generation_auto,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
        backend,
    ):
        """Auto schema method uses structural_tag on xgrammar-capable backends."""
        params_with_structured_generation_auto.generation.structured_generation.backend = backend
        backend_instance = create_backend(
            params_with_structured_generation_auto,
            mock_model_metadata,
            mock_schema,
            mock_workdir,
        )

        with patch(
            "nemo_safe_synthesizer.generation.vllm_backend.build_json_structural_tag",
            return_value='{"type":"structural_tag","format":{"type":"json_schema","json_schema":{}}}',
        ) as mock_build_structural_tag:
            result = backend_instance._build_structured_output_params()
            mock_build_structural_tag.assert_called_once_with(
                mock_schema,
                params_with_structured_generation_auto,
                bos_token=mock_model_metadata.prompt_config.bos_token,
                eos_token=mock_model_metadata.prompt_config.eos_token,
                default_max_records_per_group=mock_model_metadata.max_records_per_group,
            )
            assert result is not None
            assert result.structural_tag is not None

    @pytest.mark.parametrize("backend", ["guidance", "outlines", "lm-format-enforcer"])
    def test_auto_resolves_to_regex_on_other_backends(
        self,
        params_with_structured_generation_auto,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
        backend,
    ):
        """Auto schema method falls back to regex on non-xgrammar backends."""
        params_with_structured_generation_auto.generation.structured_generation.backend = backend
        backend_instance = create_backend(
            params_with_structured_generation_auto,
            mock_model_metadata,
            mock_schema,
            mock_workdir,
        )

        with patch(
            "nemo_safe_synthesizer.generation.vllm_backend.build_json_based_regex",
            return_value="test_regex_pattern",
        ) as mock_build_regex:
            result = backend_instance._build_structured_output_params()
            mock_build_regex.assert_called_once_with(
                mock_schema,
                params_with_structured_generation_auto,
                bos_token=mock_model_metadata.prompt_config.bos_token,
                eos_token=mock_model_metadata.prompt_config.eos_token,
                default_max_records_per_group=mock_model_metadata.max_records_per_group,
            )
            assert result is not None
            assert result.regex == "test_regex_pattern"

    @pytest.mark.parametrize("backend", ["guidance", "outlines", "lm-format-enforcer"])
    def test_structural_tag_rejects_non_xgrammar_backends(
        self,
        params_with_structured_generation_structural_tag,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
        backend,
    ):
        """Structural Tag requires vLLM's xgrammar backend."""
        params_with_structured_generation_structural_tag.generation.structured_generation.backend = backend
        backend_instance = create_backend(
            params_with_structured_generation_structural_tag,
            mock_model_metadata,
            mock_schema,
            mock_workdir,
        )

        with pytest.raises(ParameterError, match="requires `backend`"):
            backend_instance._build_structured_output_params()

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


class TestInitializeModelRef:
    """Tests intentionally tied to HF cache layout through ``ModelRef``."""

    def test_initialize_passes_cached_snapshot_target_and_trust_to_vllm(
        self,
        base_params,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
        fixture_cached_nvidia_snapshot,
    ):
        """Cached snapshot target selection intentionally follows HF cache design."""
        cache_root, snapshot = fixture_cached_nvidia_snapshot
        base_params.training.pretrained_model = "nvidia/Nemotron-Mini-4B-Instruct"
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        mock_llm = MagicMock()
        mock_llm.get_tokenizer.return_value = MagicMock()

        with (
            patch(
                "nemo_safe_synthesizer.generation.vllm_backend.ModelRef._default_hf_cache_root",
                return_value=cache_root,
            ),
            patch("nemo_safe_synthesizer.generation.vllm_backend.vLLM", return_value=mock_llm) as mock_vllm,
            patch("nemo_safe_synthesizer.generation.vllm_backend.get_max_vram", return_value={0: 0.8}),
            patch("nemo_safe_synthesizer.generation.vllm_backend.create_processor", return_value=MagicMock()),
        ):
            backend.initialize()

        assert backend.llm is mock_llm
        assert mock_vllm.call_args.kwargs["model"] == str(snapshot)
        assert mock_vllm.call_args.kwargs["max_model_len"] == mock_model_metadata.max_seq_length
        assert mock_vllm.call_args.kwargs["hf_overrides"] is None
        assert mock_vllm.call_args.kwargs["trust_remote_code"] is True

    def test_initialize_passes_rope_hf_overrides_for_extended_context(
        self,
        base_params,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
        fixture_cached_nvidia_snapshot,
    ):
        """VLLM 0.24 requires config overrides when max_model_len exceeds the base context."""
        cache_root, snapshot = fixture_cached_nvidia_snapshot
        base_params.training.pretrained_model = "nvidia/Nemotron-Mini-4B-Instruct"
        mock_model_metadata.base_max_seq_length = 2048
        mock_model_metadata.max_seq_length = 4096
        mock_model_metadata.rope_scaling = RopeScaling(
            rope_type="linear",
            factor=2.0,
            theta=10000.0,
            rope_parameters={"rope_type": "linear", "rope_theta": 10000.0, "low_freq_factor": 1.0},
        )
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        mock_llm = MagicMock()
        mock_llm.get_tokenizer.return_value = MagicMock()

        with (
            patch(
                "nemo_safe_synthesizer.generation.vllm_backend.ModelRef._default_hf_cache_root",
                return_value=cache_root,
            ),
            patch("nemo_safe_synthesizer.generation.vllm_backend.vLLM", return_value=mock_llm) as mock_vllm,
            patch("nemo_safe_synthesizer.generation.vllm_backend.get_max_vram", return_value={0: 0.8}),
            patch("nemo_safe_synthesizer.generation.vllm_backend.create_processor", return_value=MagicMock()),
        ):
            backend.initialize()

        assert mock_vllm.call_args.kwargs["model"] == str(snapshot)
        assert mock_vllm.call_args.kwargs["max_model_len"] == 4096
        assert mock_vllm.call_args.kwargs["hf_overrides"] == {
            "rope_parameters": {
                "rope_type": "linear",
                "factor": 2.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 2048,
                "rope_theta": 10000.0,
            },
        }

    def test_initialize_caches_engine_runtime_config(
        self,
        base_params,
        mock_model_metadata,
        mock_schema,
        mock_workdir,
        fixture_cached_nvidia_snapshot,
    ):
        """``initialize()`` probes the engine once and caches the effective runtime config.

        The cached dict is the source the generation-complete event reads
        at end of generation, so the init-time wiring is part of the
        observability contract.
        """
        cache_root, _ = fixture_cached_nvidia_snapshot
        base_params.training.pretrained_model = "nvidia/Nemotron-Mini-4B-Instruct"
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        # Pre-condition: nothing cached before initialize().
        assert backend._engine_runtime_config == {}

        mock_llm = MagicMock()
        mock_llm.get_tokenizer.return_value = MagicMock()

        with (
            patch(
                "nemo_safe_synthesizer.generation.vllm_backend.ModelRef._default_hf_cache_root",
                return_value=cache_root,
            ),
            patch("nemo_safe_synthesizer.generation.vllm_backend.vLLM", return_value=mock_llm),
            patch("nemo_safe_synthesizer.generation.vllm_backend.get_max_vram", return_value={0: 0.8}),
            patch("nemo_safe_synthesizer.generation.vllm_backend.create_processor", return_value=MagicMock()),
            patch(
                "nemo_safe_synthesizer.generation.vllm_backend.probe_engine_runtime_config",
                return_value={"max_num_seqs": 256, "enable_prefix_caching": True},
            ) as mock_probe,
        ):
            backend.initialize()

        mock_probe.assert_called_once_with(mock_llm)
        assert backend._engine_runtime_config == {"max_num_seqs": 256, "enable_prefix_caching": True}


class TestGenerationObservabilityEmission:
    """The generation-complete production emission contract.

    Covers the path that backend sampling-plumbing tests skip: that
    ``generate()`` always runs the finalizer, and that the finalizer
    assembles and routes a ``GenerationObservability`` without ever breaking
    generation.
    """

    def test_generate_runs_finalizer_even_when_body_fails(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """A failure inside the generation loop must still emit the generation event."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        backend.prepare_params = MagicMock(side_effect=StopIteration("short-circuit"))

        with patch.object(backend, "_emit_generation_observability") as mock_emit:
            with pytest.raises(StopIteration):
                backend.generate()

        mock_emit.assert_called_once()

    def test_emit_assembles_and_routes_event(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """The finalizer builds a GenerationObservability from probes and routes it to logs + wandb."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        backend.llm = None
        backend._engine_runtime_config = {"enable_prefix_caching": True}

        sampler = MagicMock()
        sampler.peak_gb = 12.5

        with (
            patch(
                "nemo_safe_synthesizer.generation.vllm_backend.read_vllm_runtime_metrics",
                return_value={"kv_cache_usage_perc": 0.4, "prefix_cache_hit_rate": 0.9, "spec_accept_rate": None},
            ),
            patch("nemo_safe_synthesizer.generation.vllm_backend.read_loadavg", return_value=(1.0, 2.0, 3.0)),
            patch("nemo_safe_synthesizer.generation.vllm_backend.log_observability_event") as mock_log_wandb,
            patch("nemo_safe_synthesizer.generation.vllm_backend.logger") as mock_logger,
        ):
            backend._emit_generation_observability(sampler, (0.5, 0.6, 0.7))

        mock_log_wandb.assert_called_once()
        (event,) = mock_log_wandb.call_args.args
        assert mock_log_wandb.call_args.kwargs == {"prefix": "vllm_gen"}
        assert isinstance(event, GenerationObservability)
        assert event.peak_vram_gb == 12.5
        assert event.kv_cache_usage_perc == 0.4
        assert event.prefix_cache_hit_rate == 0.9
        assert event.spec_accept_rate is None
        assert event.engine_runtime_config == {"enable_prefix_caching": True}
        # ``loadavg_pre`` is the value captured before generation; ``loadavg_post``
        # is read fresh inside the finalizer.
        assert event.loadavg_pre == (0.5, 0.6, 0.7)
        assert event.loadavg_post == (1.0, 2.0, 3.0)
        # The same event is mirrored to structured logs as "vLLM generation complete".
        mock_logger.runtime.info.assert_called_once_with("vLLM generation complete", extra={"ctx": event.model_dump()})

    def test_emit_swallows_failures(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """A failure inside emission must not propagate — observability is best-effort."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        backend.llm = None

        sampler = MagicMock()
        sampler.peak_gb = 1.0

        with patch(
            "nemo_safe_synthesizer.generation.vllm_backend.read_vllm_runtime_metrics",
            side_effect=RuntimeError("probe blew up"),
        ):
            # Must not raise.
            backend._emit_generation_observability(sampler, None)


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

    def test_num_beams_greater_than_one_is_omitted(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """Test that num_beams is omitted because vLLM 0.24 removed beam_width."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        mapping = backend._get_api_param_mapping(resolved_temperature=0.5)
        key, value = mapping["num_beams"](4)

        assert key is None
        assert value is None

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


class TestGenerateDispatch:
    """Tests for vLLM generation dispatch."""

    def test_generate_passes_flat_token_ids_through_prompts(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Flat token IDs use the vLLM 0.24 ``prompts=`` token prompt API."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        captured = {}

        def fake_generate(**kwargs):
            captured.update(kwargs)
            return ["ok"]

        backend._gen_method = partial(fake_generate, sampling_params=object())

        assert backend._generate(input_ids=[1, 2, 3]) == ["ok"]
        assert captured["prompts"] == {"prompt_token_ids": [1, 2, 3]}
        assert "prompt_token_ids" not in captured

    def test_generate_passes_batched_token_ids_through_prompts(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Batched token IDs use one token prompt per batch element."""
        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        captured = {}

        def fake_generate(**kwargs):
            captured.update(kwargs)
            return ["ok"]

        backend._gen_method = partial(fake_generate, sampling_params=object())

        assert backend._generate(input_ids=[[1, 2], [3, 4]]) == ["ok"]
        assert captured["prompts"] == [{"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [3, 4]}]
        assert "prompt_token_ids" not in captured


class TestNoopRemoteCacheBackend:
    """Tests for the _NoopRemoteCacheBackend and conditional installation logic."""

    def test_noop_backend_get_returns_none(self):
        from nemo_safe_synthesizer.generation.vllm_backend import _NoopRemoteCacheBackend

        backend = _NoopRemoteCacheBackend()
        assert backend.get("any-key") is None

    def test_noop_backend_put_is_silent(self):
        from nemo_safe_synthesizer.generation.vllm_backend import _NoopRemoteCacheBackend

        backend = _NoopRemoteCacheBackend()
        backend.put("key", b"data")  # should not raise

    def test_install_skipped_when_redis_available(self):
        """When redis is importable, the override must not be installed."""
        pytest.importorskip("torch._inductor.remote_cache")
        from torch._inductor.remote_cache import RemoteAutotuneCache

        from nemo_safe_synthesizer.generation.vllm_backend import (
            _install_noop_remote_cache_backends,
        )

        original = RemoteAutotuneCache.backend_override_cls
        try:
            RemoteAutotuneCache.backend_override_cls = None
            with patch("nemo_safe_synthesizer.generation.vllm_backend._is_redis_available", return_value=True):
                _install_noop_remote_cache_backends()
            assert RemoteAutotuneCache.backend_override_cls is None
        finally:
            RemoteAutotuneCache.backend_override_cls = original

    def test_install_applies_when_redis_unavailable(self):
        """When redis is not importable, RemoteAutotuneCache gets the no-op backend."""
        pytest.importorskip("torch._inductor.remote_cache")
        from torch._inductor.remote_cache import RemoteAutotuneCache

        from nemo_safe_synthesizer.generation.vllm_backend import (
            _install_noop_remote_cache_backends,
            _NoopRemoteCacheBackend,
        )

        original = RemoteAutotuneCache.backend_override_cls
        try:
            RemoteAutotuneCache.backend_override_cls = None
            with patch("nemo_safe_synthesizer.generation.vllm_backend._is_redis_available", return_value=False):
                _install_noop_remote_cache_backends()
            assert RemoteAutotuneCache.backend_override_cls is _NoopRemoteCacheBackend
        finally:
            RemoteAutotuneCache.backend_override_cls = original

    def test_is_redis_available_returns_false_when_missing(self):
        """_is_redis_available returns False when redis cannot be imported."""
        from nemo_safe_synthesizer.generation.vllm_backend import _is_redis_available

        with patch.dict("sys.modules", {"redis": None}):
            assert _is_redis_available() is False

    def test_is_redis_available_returns_true_when_present(self):
        """_is_redis_available returns True when redis is importable."""
        from nemo_safe_synthesizer.generation.vllm_backend import _is_redis_available

        fake_redis = MagicMock()
        with patch.dict("sys.modules", {"redis": fake_redis}):
            assert _is_redis_available() is True


class TestSecureOutlinesCacheDir:
    """Tests for the CVE-2025-69872 outlines diskcache hardening."""

    def test_chmods_existing_cache_dir_to_0700(self, tmp_path, monkeypatch):
        """``_secure_outlines_cache_dir`` tightens permissions on a permissive dir.

        Exercises the explicit-OUTLINES_CACHE_DIR branch: simulates a co-tenant-
        writable cache directory (mode 0777) and asserts the helper locks it down
        to 0700, which is the precondition CVE-2025-69872 needs to fail.
        """
        import stat

        from nemo_safe_synthesizer.generation.vllm_backend import _secure_outlines_cache_dir

        cache_dir = tmp_path / "outlines-cache"
        cache_dir.mkdir()
        cache_dir.chmod(0o777)
        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o777, "precondition: dir starts world-writable"

        monkeypatch.setenv("OUTLINES_CACHE_DIR", str(cache_dir))

        _secure_outlines_cache_dir()

        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
        assert os.environ["OUTLINES_CACHE_DIR"] == str(cache_dir)

    def test_vllm_outlines_diskcache_is_disabled(self):
        """Module import must hard-disable the vLLM opt-in diskcache."""
        from nemo_safe_synthesizer.generation import vllm_backend  # noqa: F401  -- ensure module is imported

        assert os.environ.get("VLLM_V1_USE_OUTLINES_CACHE") == "0"


class TestGroupedGenerationStopKwargs:
    """Tests that grouped generation relies on native EOS stopping (ignore_eos=False)
    rather than explicit stop/stop_token_ids kwargs.
    """

    def test_native_eos_stopping_for_grouped_processor(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """When processor is not TabularDataProcessor, ignore_eos must be False and
        no explicit stop/stop_token_ids kwargs are passed.
        """
        mock_model_metadata.prompt_config.eos_token = "</s>"
        mock_model_metadata.prompt_config.eos_token_id = 2
        mock_model_metadata.max_seq_length = 2048
        mock_model_metadata.generation_max_tokens_for.return_value = 2048

        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        assert not isinstance(backend.processor, TabularDataProcessor)

        captured = {}

        def capture_and_stop(**kwargs):
            captured.update(kwargs)
            raise StopIteration("short-circuit")

        backend.prepare_params = capture_and_stop

        with pytest.raises(StopIteration):
            backend.generate()

        assert "stop" not in captured
        assert "stop_token_ids" not in captured
        assert captured["ignore_eos"] is False
        assert captured["include_stop_str_in_output"] is True

    def test_no_stop_kwargs_for_tabular_processor(self, base_params, mock_model_metadata, mock_schema, mock_workdir):
        """When processor is TabularDataProcessor, no explicit stop/stop_token_ids
        kwargs are passed, ignore_eos is False, and special-token outputs are off.
        """
        mock_model_metadata.max_seq_length = 2048
        mock_model_metadata.generation_max_tokens_for.return_value = 2048

        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        backend.processor = TabularDataProcessor(schema=mock_schema, config=ValidationParameters())
        assert isinstance(backend.processor, TabularDataProcessor)

        captured = {}

        def capture_and_stop(**kwargs):
            captured.update(kwargs)
            raise StopIteration("short-circuit")

        backend.prepare_params = capture_and_stop

        with pytest.raises(StopIteration):
            backend.generate()

        assert "stop" not in captured
        assert "stop_token_ids" not in captured
        assert captured["ignore_eos"] is False
        assert captured["skip_special_tokens"] is True
        assert captured["include_stop_str_in_output"] is False

    def test_large_context_grouped_generation_has_eos_stop(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """Regression: large-context models (e.g. SmolLM3) need explicit EOS stop tokens
        for grouped generation, otherwise generation runs away to max_tokens.

        With small context windows (e.g. TinyLlama 2048) the max_tokens limit masks
        the missing stop condition. A large window like 8192+ exposes the bug.
        """
        mock_model_metadata.prompt_config.eos_token = "</s>"
        mock_model_metadata.prompt_config.eos_token_id = 2
        mock_model_metadata.max_seq_length = 8192
        mock_model_metadata.generation_max_tokens_for.return_value = 8192

        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)
        assert not isinstance(backend.processor, TabularDataProcessor)

        captured = {}

        def capture_and_stop(**kwargs):
            captured.update(kwargs)
            raise StopIteration("short-circuit")

        backend.prepare_params = capture_and_stop

        with pytest.raises(StopIteration):
            backend.generate()

        assert captured["max_tokens"] == 8192
        assert "stop" not in captured
        assert "stop_token_ids" not in captured
        assert captured["ignore_eos"] is False


class TestGenerationMaxTokensPlumbing:
    """``SamplingParams.max_tokens`` is sourced from ``metadata.generation_max_tokens_for``."""

    def test_uses_generation_max_tokens_for_with_cached_prompt_len(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """The helper drives ``max_tokens`` and is invoked with the cached prompt-token count."""
        mock_model_metadata.max_seq_length = 12_288
        mock_model_metadata.generation_max_tokens_for.return_value = 4_200

        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        captured = {}

        def capture_and_stop(**kwargs):
            captured.update(kwargs)
            raise StopIteration("short-circuit")

        backend.prepare_params = capture_and_stop

        with pytest.raises(StopIteration):
            backend.generate()

        assert captured["max_tokens"] == 4_200
        # Engine is not initialized in this plumbing test, so the cached
        # prompt-token count falls back to 0; the helper is still called
        # exactly once with that value plus the configured budget multiplier.
        mock_model_metadata.generation_max_tokens_for.assert_called_once_with(
            0, multiplier=base_params.generation.max_tokens_multiplier
        )

    def test_passes_cached_prompt_token_count_when_engine_initialized(
        self, base_params, mock_model_metadata, mock_schema, mock_workdir
    ):
        """When the vLLM engine exists, the cached prompt-token count is forwarded to the helper."""
        mock_model_metadata.max_seq_length = 12_288
        mock_model_metadata.generation_max_tokens_for.return_value = 4_096

        backend = create_backend(base_params, mock_model_metadata, mock_schema, mock_workdir)

        # Stand in for an initialized engine; the backend tokenizes its templated
        # prompt once via ``llm.get_tokenizer().encode`` and caches the count.
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = list(range(37))
        backend.llm = MagicMock()
        backend.llm.get_tokenizer.return_value = fake_tokenizer

        captured = {}

        def capture_and_stop(**kwargs):
            captured.update(kwargs)
            raise StopIteration("short-circuit")

        backend.prepare_params = capture_and_stop

        with pytest.raises(StopIteration):
            backend.generate()

        assert captured["max_tokens"] == 4_096
        mock_model_metadata.generation_max_tokens_for.assert_called_once_with(
            37, multiplier=base_params.generation.max_tokens_multiplier
        )
        # Cached: a second access does not retokenize.
        assert backend._get_prompt_token_count() == 37
        fake_tokenizer.encode.assert_called_once_with(backend.prompt)
