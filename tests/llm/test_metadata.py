# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLM metadata module.

Uses pytest parametrization patterns from:
- https://docs.pytest.org/en/stable/example/parametrize.html
- https://docs.pytest.org/en/stable/how-to/parametrize.html
"""

import pytest

# Skip all tests in this module if transformers is not available
pytest.importorskip(
    "transformers", reason="transformers is required for these tests (install with: uv sync --extra cpu)"
)

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from transformers import PretrainedConfig

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.defaults import (
    DEFAULT_INSTRUCTION,
    MAX_ROPE_SCALING_FACTOR,
    PROMPT_TEMPLATE,
)
from nemo_safe_synthesizer.llm.metadata import (
    DEFAULT_MAX_SEQ_LENGTH,
    GLOBAL_MAX_SEQ_LENGTH,
    Llama32,
    LLMPromptConfig,
    Mistral,
    ModelMetadata,
    Nemotron,
    Qwen,
    RopeScaling,
    SmolLM2,
    SmolLM3,
    TinyLlama,
    resolve_rope_scaling_factor,
)


@dataclass(frozen=True)
class ModelDetectionScenario:
    """Scenario for testing model detection from path."""

    id: str
    model_path: str
    expected_class: type


@dataclass(frozen=True)
class ModelInitScenario:
    """Scenario for testing model initialization."""

    id: str
    model_class: type
    model_path: str
    expected_template: str
    expected_add_bos: bool
    expected_add_eos: bool
    expected_bos_token: str | None  # None means use default from tokenizer
    expected_bos_token_id: int | None
    use_global_max_seq: bool = False
    custom_max_position_embeddings: int | None = None


@dataclass(frozen=True)
class RopeScalingScenario:
    """Scenario for testing rope scaling factor resolution."""

    id: str
    factor: RopeScaling | dict | int | float | None
    autoconfig: bool  # Whether to provide autoconfig
    rope_theta: float | None  # Theta value to set on autoconfig
    expected_result_type: type | None  # None or RopeScaling
    expected_factor: float | None
    expected_theta: float | None
    raises: type[Exception] | None = None
    raises_match: str | None = None


# Model detection scenarios - maps model paths to expected classes
MODEL_DETECTION_SCENARIOS = [
    ModelDetectionScenario("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", TinyLlama),
    ModelDetectionScenario("qwen", "Qwen/Qwen2-0.5B", Qwen),
    ModelDetectionScenario("llama32", "meta-llama/Llama32-1B", Llama32),
    ModelDetectionScenario("smollm2", "HuggingFaceTB/SmolLM2-135M", SmolLM2),
    ModelDetectionScenario("smollm3", "HuggingFaceTB/SmolLM3-3B", SmolLM3),
    ModelDetectionScenario("mistral", "mistralai/Mistral-7B-v0.1", Mistral),
    ModelDetectionScenario("nemotron", "nvidia/Nemotron-4-340B", Nemotron),
]

# Model initialization scenarios - tests each model's specific configuration
MODEL_INIT_SCENARIOS = [
    ModelInitScenario(
        id="tinyllama",
        model_class=TinyLlama,
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        expected_template=PROMPT_TEMPLATE,
        expected_add_bos=True,
        expected_add_eos=True,
        expected_bos_token=None,  # Uses tokenizer default
        expected_bos_token_id=None,
    ),
    ModelInitScenario(
        id="qwen",
        model_class=Qwen,
        model_path="Qwen/Qwen2-0.5B",
        expected_template="user\n {instruction} {schema} \n assistant\n{prefill}",
        expected_add_bos=True,
        expected_add_eos=False,
        expected_bos_token=None,
        expected_bos_token_id=None,
    ),
    ModelInitScenario(
        id="llama32",
        model_class=Llama32,
        model_path="meta-llama/Llama32-1B",
        expected_template="user\n {instruction} {schema} \n assistant\n{prefill}",
        expected_add_bos=False,
        expected_add_eos=False,
        expected_bos_token="<|im_start|>",
        expected_bos_token_id=151644,
    ),
    ModelInitScenario(
        id="smollm2",
        model_class=SmolLM2,
        model_path="HuggingFaceTB/SmolLM2-135M",
        expected_template="user\n {instruction} {schema} \n assistant\n{prefill}",
        expected_add_bos=False,
        expected_add_eos=False,
        expected_bos_token="<|im_start|>",
        expected_bos_token_id=151644,
        custom_max_position_embeddings=8192,
    ),
    ModelInitScenario(
        id="smollm3",
        model_class=SmolLM3,
        model_path="HuggingFaceTB/SmolLM3-3B",
        expected_template="user\n {instruction} {schema} <|im_end|> \n <|im_start|>assistant\n{prefill}",
        expected_add_bos=True,
        expected_add_eos=True,
        expected_bos_token="<|im_start|>",
        expected_bos_token_id=128011,
        use_global_max_seq=True,
    ),
    ModelInitScenario(
        id="mistral",
        model_class=Mistral,
        model_path="mistralai/Mistral-7B-v0.1",
        expected_template="[INST] {instruction} \n\n {schema} [/INST]{prefill}",
        expected_add_bos=True,
        expected_add_eos=True,
        expected_bos_token=None,
        expected_bos_token_id=None,
        use_global_max_seq=True,
    ),
    ModelInitScenario(
        id="nemotron",
        model_class=Nemotron,
        model_path="nvidia/Nemotron-4-340B-Base",
        expected_template="[INST] {instruction} \n\n {schema} [/INST]{prefill}",
        expected_add_bos=True,
        expected_add_eos=True,
        expected_bos_token=None,
        expected_bos_token_id=None,
        custom_max_position_embeddings=4096,
    ),
]

# Rope scaling resolution scenarios
ROPE_SCALING_SCENARIOS = [
    RopeScalingScenario(
        id="none_returns_none",
        factor=None,
        autoconfig=False,
        rope_theta=None,
        expected_result_type=None,
        expected_factor=None,
        expected_theta=None,
    ),
    RopeScalingScenario(
        id="int_one_returns_none",
        factor=1,
        autoconfig=False,
        rope_theta=None,
        expected_result_type=None,
        expected_factor=None,
        expected_theta=None,
    ),
    RopeScalingScenario(
        id="float_one_returns_none",
        factor=1.0,
        autoconfig=False,
        rope_theta=None,
        expected_result_type=None,
        expected_factor=None,
        expected_theta=None,
    ),
    RopeScalingScenario(
        id="rope_scaling_unchanged",
        factor=RopeScaling(rope_type="linear", factor=2.0, theta=10000.0),
        autoconfig=False,
        rope_theta=None,
        expected_result_type=RopeScaling,
        expected_factor=2.0,
        expected_theta=10000.0,
    ),
    RopeScalingScenario(
        id="dict_to_rope_scaling",
        factor={"rope_type": "dynamic", "factor": 4.0, "theta": 50000.0},
        autoconfig=False,
        rope_theta=None,
        expected_result_type=RopeScaling,
        expected_factor=4.0,
        expected_theta=50000.0,
    ),
    RopeScalingScenario(
        id="int_with_autoconfig",
        factor=2,
        autoconfig=True,
        rope_theta=20000.0,
        expected_result_type=RopeScaling,
        expected_factor=2,
        expected_theta=20000.0,
    ),
    RopeScalingScenario(
        id="float_with_autoconfig",
        factor=2.5,
        autoconfig=True,
        rope_theta=15000.0,
        expected_result_type=RopeScaling,
        expected_factor=2.5,
        expected_theta=15000.0,
    ),
    RopeScalingScenario(
        id="int_without_autoconfig_raises",
        factor=2,
        autoconfig=False,
        rope_theta=None,
        expected_result_type=None,
        expected_factor=None,
        expected_theta=None,
        raises=ValueError,
        raises_match="autoconfig is required when factor is an int or float",
    ),
    RopeScalingScenario(
        id="float_without_autoconfig_raises",
        factor=2.0,
        autoconfig=False,
        rope_theta=None,
        expected_result_type=None,
        expected_factor=None,
        expected_theta=None,
        raises=ValueError,
        raises_match="autoconfig is required when factor is an int or float",
    ),
    RopeScalingScenario(
        id="default_theta_when_not_in_autoconfig",
        factor=3,
        autoconfig=True,
        rope_theta=None,  # Will delete rope_theta from autoconfig
        expected_result_type=RopeScaling,
        expected_factor=3,
        expected_theta=10000.0,  # Default value
    ),
    RopeScalingScenario(
        id="factor_exceeds_max_capped_int",
        factor=10,  # Greater than MAX_ROPE_SCALING_FACTOR (6)
        autoconfig=True,
        rope_theta=20000.0,
        expected_result_type=RopeScaling,
        expected_factor=MAX_ROPE_SCALING_FACTOR,  # Should be capped to 6
        expected_theta=20000.0,
    ),
    RopeScalingScenario(
        id="factor_exceeds_max_capped_float",
        factor=10.5,  # Greater than MAX_ROPE_SCALING_FACTOR (6)
        autoconfig=True,
        rope_theta=15000.0,
        expected_result_type=RopeScaling,
        expected_factor=MAX_ROPE_SCALING_FACTOR,  # Should be capped to 6
        expected_theta=15000.0,
    ),
]


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = 1
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def mock_autoconfig_obj():
    """Create a mock AutoConfig for testing.

    Uses a real PretrainedConfig instance so it passes Pydantic isinstance validation.
    """
    config = PretrainedConfig()
    config.max_position_embeddings = DEFAULT_MAX_SEQ_LENGTH
    return config


@pytest.fixture
def sample_prompt_config():
    """Create a sample LLMPromptConfig for testing."""
    return LLMPromptConfig(
        template="[INST] {instruction} {schema} [/INST]",
        add_bos_token_to_prompt=True,
        add_eos_token_to_prompt=True,
        bos_token="<s>",
        bos_token_id=1,
        eos_token="</s>",
        eos_token_id=2,
    )


@pytest.fixture
def sample_workdir(tmp_path):
    """Create a sample Workdir for testing."""
    workdir = Workdir(
        base_path=tmp_path,
        dataset_name="test-dataset",
        config_name="test-config",
        run_name="2026-01-15T12:00:00",
        _current_phase="train",
    )
    workdir.ensure_directories()
    return workdir


@pytest.fixture
def sample_model_metadata(sample_prompt_config, mock_autoconfig_obj, sample_workdir):
    """Create a sample ModelMetadata for testing."""
    return ModelMetadata(
        model_name_or_path="test-model",
        prompt_config=sample_prompt_config,
        autoconfig=mock_autoconfig_obj,
        base_max_seq_length=2048,
        workdir=sample_workdir,
    )


@pytest.fixture(params=MODEL_DETECTION_SCENARIOS, ids=lambda s: s.id)
def model_detection_scenario(request) -> ModelDetectionScenario:
    """Parametrized fixture for model detection scenarios."""
    return request.param


@pytest.fixture(params=MODEL_INIT_SCENARIOS, ids=lambda s: s.id)
def model_init_scenario(request) -> ModelInitScenario:
    """Parametrized fixture for model initialization scenarios."""
    return request.param


@pytest.fixture(params=ROPE_SCALING_SCENARIOS, ids=lambda s: s.id)
def rope_scaling_scenario(request) -> RopeScalingScenario:
    """Parametrized fixture for rope scaling scenarios."""
    return request.param


class TestLLMPromptConfig:
    """Tests for the LLMPromptConfig class."""

    def test_create_prompt_config(self, sample_prompt_config):
        """Test that LLMPromptConfig can be created with valid parameters."""
        assert sample_prompt_config.template == "[INST] {instruction} {schema} [/INST]"
        assert sample_prompt_config.add_bos_token_to_prompt is True
        assert sample_prompt_config.add_eos_token_to_prompt is True
        assert sample_prompt_config.bos_token == "<s>"
        assert sample_prompt_config.bos_token_id == 1
        assert sample_prompt_config.eos_token == "</s>"
        assert sample_prompt_config.eos_token_id == 2

    @patch("nemo_safe_synthesizer.llm.metadata.AutoTokenizer")
    def test_from_tokenizer(self, mock_auto_tokenizer, sample_prompt_config):
        """Test the from_tokenizer method creates a new config from tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "<bos>"
        mock_tokenizer.bos_token_id = 10
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 20
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        new_config = sample_prompt_config.from_tokenizer("test-model")

        assert new_config.bos_token == "<bos>"
        assert new_config.bos_token_id == 10
        assert new_config.eos_token == "<eos>"
        assert new_config.eos_token_id == 20
        assert new_config.template == PROMPT_TEMPLATE
        assert new_config.add_bos_token_to_prompt is True
        assert new_config.add_eos_token_to_prompt is True

    @patch("nemo_safe_synthesizer.llm.metadata.AutoTokenizer")
    def test_from_tokenizer_with_kwargs_override(self, mock_auto_tokenizer, sample_prompt_config):
        """Test that kwargs can override default values in from_tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "<bos>"
        mock_tokenizer.bos_token_id = 10
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 20
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        new_config = sample_prompt_config.from_tokenizer(
            "test-model",
            add_bos_token_to_prompt=False,
            template="custom template",
        )

        assert new_config.add_bos_token_to_prompt is False
        assert new_config.template == "custom template"


class TestResolveRopeScalingFactor:
    """Tests for the resolve_rope_scaling_factor function."""

    def test_resolve_rope_scaling(self, rope_scaling_scenario: RopeScalingScenario, mock_autoconfig_obj):
        """Test rope scaling factor resolution for all scenarios."""
        # Setup autoconfig if needed
        autoconfig = None
        if rope_scaling_scenario.autoconfig:
            autoconfig = mock_autoconfig_obj
            if rope_scaling_scenario.rope_theta is not None:
                autoconfig.rope_theta = rope_scaling_scenario.rope_theta
            elif hasattr(autoconfig, "rope_theta"):
                delattr(autoconfig, "rope_theta")

        # Handle expected errors
        if rope_scaling_scenario.raises:
            with pytest.raises(rope_scaling_scenario.raises, match=rope_scaling_scenario.raises_match):
                resolve_rope_scaling_factor(rope_scaling_scenario.factor, autoconfig=autoconfig)
            return

        # Normal case
        result = resolve_rope_scaling_factor(rope_scaling_scenario.factor, autoconfig=autoconfig)

        if rope_scaling_scenario.expected_result_type is None:
            assert result is None
        else:
            assert isinstance(result, RopeScaling)
            assert result.factor == rope_scaling_scenario.expected_factor
            assert result.theta == rope_scaling_scenario.expected_theta


class TestModelMetadata:
    """Tests for the ModelMetadata class."""

    def test_create_model_metadata(self, sample_model_metadata):
        """Test that ModelMetadata can be created with valid parameters."""
        assert sample_model_metadata.model_name_or_path == "test-model"
        assert sample_model_metadata.base_max_seq_length == 2048
        assert sample_model_metadata.is_adapter is False
        assert sample_model_metadata.instruction == DEFAULT_INSTRUCTION

    def test_adapter_path_property(self, sample_model_metadata, sample_workdir):
        """Test the adapter_path property returns the correct path."""
        expected_path = sample_workdir.adapter_path.resolve()
        assert sample_model_metadata.adapter_path == expected_path

    def test_metadata_path_property(self, sample_model_metadata, sample_workdir):
        """Test the metadata_path property returns the correct path."""
        expected_path = sample_workdir.metadata_file
        assert sample_model_metadata.metadata_path == expected_path

    def test_adapter_path_raises_without_workdir(self, sample_prompt_config, mock_autoconfig_obj):
        """Test that adapter_path raises ValueError when workdir is not set."""
        metadata = ModelMetadata(
            model_name_or_path="test-model",
            prompt_config=sample_prompt_config,
            autoconfig=mock_autoconfig_obj,
            base_max_seq_length=2048,
        )
        with pytest.raises(ValueError, match="workdir is not set"):
            _ = metadata.adapter_path

    def test_metadata_path_raises_without_workdir(self, sample_prompt_config, mock_autoconfig_obj):
        """Test that metadata_path raises ValueError when workdir is not set."""
        metadata = ModelMetadata(
            model_name_or_path="test-model",
            prompt_config=sample_prompt_config,
            autoconfig=mock_autoconfig_obj,
            base_max_seq_length=2048,
        )
        with pytest.raises(ValueError, match="workdir is not set"):
            _ = metadata.metadata_path

    @pytest.mark.parametrize(
        "rope_scaling, expected_length, expected_factor",
        [
            pytest.param(None, 2048, None, id="no_rope_scaling"),
            pytest.param(RopeScaling(factor=1.0), 2048, 1.0, id="factor_1"),
            pytest.param(RopeScaling(factor=2.0, theta=10000.0), 4096, 2.0, id="factor_2"),
        ],
    )
    def test_max_seq_length_with_rope_scaling(
        self, sample_prompt_config, mock_autoconfig_obj, sample_workdir, rope_scaling, expected_length, expected_factor
    ):
        """Test max_seq_length calculation with various rope scaling factors."""
        metadata = ModelMetadata(
            model_name_or_path="test-model",
            prompt_config=sample_prompt_config,
            autoconfig=mock_autoconfig_obj,
            base_max_seq_length=2048,
            rope_scaling=rope_scaling,
            workdir=sample_workdir,
        )
        assert metadata.max_seq_length == expected_length
        if expected_factor:
            assert metadata.rope_scaling_factor == expected_factor

    def test_save_metadata_default_path(self, sample_model_metadata):
        """Test save_metadata saves to the default metadata_path."""
        sample_model_metadata.save_metadata()
        assert sample_model_metadata.metadata_path.exists()

    def test_save_metadata_raises_without_workdir(self, sample_prompt_config, mock_autoconfig_obj):
        """Test that save_metadata raises ValueError when workdir is not set."""
        metadata = ModelMetadata(
            model_name_or_path="test-model",
            prompt_config=sample_prompt_config,
            autoconfig=mock_autoconfig_obj,
            base_max_seq_length=2048,
        )
        with pytest.raises(ValueError, match="workdir is not set"):
            metadata.save_metadata()

    @patch("nemo_safe_synthesizer.llm.metadata.AutoConfig")
    @patch("nemo_safe_synthesizer.llm.metadata.load_json")
    def test_from_metadata_json(
        self, mock_load_json, mock_auto_config, sample_prompt_config, mock_autoconfig_obj, tmp_path, sample_workdir
    ):
        """Test from_metadata_json loads metadata from a JSON file."""
        mock_auto_config.from_pretrained.return_value = mock_autoconfig_obj
        mock_load_json.return_value = {
            "model_name_or_path": "loaded-model",
            "prompt_config": sample_prompt_config.model_dump(),
            "base_max_seq_length": 2048,
            "rope_scaling": {"rope_type": "linear", "factor": 2.0, "theta": 10000.0},
        }

        metadata = ModelMetadata.from_metadata_json(tmp_path / "metadata.json", workdir=sample_workdir)

        assert metadata.model_name_or_path == "loaded-model"
        assert metadata.base_max_seq_length == 2048
        assert metadata.rope_scaling and metadata.rope_scaling.factor == 2.0
        assert metadata.max_seq_length == 4096
        assert metadata.workdir == sample_workdir

    def test_from_str_or_path_raises_for_unknown_model(self):
        """Test from_str_or_path raises ValueError for unknown model names."""
        with pytest.raises(ValueError, match="Unknown model name or path"):
            ModelMetadata.from_str_or_path("unknown-model-xyz")


class TestResolveModelClass:
    """Tests for ModelMetadata._resolve_model_class (model name → class, no instantiation)."""

    @pytest.mark.parametrize(
        "model_name_or_path",
        [
            "google/gemma-2-27b",
            "SomeRandomModel/1B",
        ],
        ids=["gemma", "random_model"],
    )
    def test_resolve_model_class_raises_for_unknown_model(self, model_name_or_path):
        """When the model name does not match any valid ModelMetadata subclass, raise ValueError.

        Covers failed jobs where the configured model is not in the expected set
        (TinyLlama, Qwen, Llama32, SmolLM2, SmolLM3, Mistral, Nemotron, Granite).
        """
        with pytest.raises(ValueError, match=r"Unknown model name or path"):
            ModelMetadata._resolve_model_class(model_name_or_path)

    def test_resolve_model_class_returns_class_for_known_model(self):
        """When the model name matches a valid subclass name, return that class (no instantiation)."""
        assert ModelMetadata._resolve_model_class("HuggingFaceTB/SmolLM3-3B") is SmolLM3
        assert ModelMetadata._resolve_model_class("mistralai/Mistral-7B-v0.1") is Mistral
        assert ModelMetadata._resolve_model_class("TinyLlama/TinyLlama-1.1B-Chat-v1.0") is TinyLlama


class TestModelDetection:
    """Tests for ModelMetadata.from_str_or_path model detection."""

    @patch("nemo_safe_synthesizer.llm.metadata.AutoConfig")
    @patch("nemo_safe_synthesizer.llm.metadata.AutoTokenizer")
    def test_model_detection(
        self,
        mock_auto_tokenizer,
        mock_auto_config,
        mock_tokenizer,
        mock_autoconfig_obj,
        model_detection_scenario: ModelDetectionScenario,
    ):
        """Test that models are correctly detected from their paths."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_config.from_pretrained.return_value = mock_autoconfig_obj

        metadata = ModelMetadata.from_str_or_path(model_detection_scenario.model_path)

        assert isinstance(metadata, model_detection_scenario.expected_class)


class TestModelInitialization:
    """Tests for individual model class initialization."""

    @patch("nemo_safe_synthesizer.llm.metadata.AutoConfig")
    @patch("nemo_safe_synthesizer.llm.metadata.AutoTokenizer")
    def test_model_initialization(
        self,
        mock_auto_tokenizer,
        mock_auto_config,
        mock_tokenizer,
        mock_autoconfig_obj,
        model_init_scenario: ModelInitScenario,
    ):
        """Test model initialization with expected configuration values."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_config.from_pretrained.return_value = mock_autoconfig_obj

        # Set custom max_position_embeddings if specified
        if model_init_scenario.custom_max_position_embeddings:
            mock_autoconfig_obj.max_position_embeddings = model_init_scenario.custom_max_position_embeddings
        elif model_init_scenario.use_global_max_seq:
            mock_autoconfig_obj.max_position_embeddings = GLOBAL_MAX_SEQ_LENGTH

        metadata = model_init_scenario.model_class(model_name_or_path=model_init_scenario.model_path)

        # Verify common attributes
        assert metadata.instruction == DEFAULT_INSTRUCTION
        assert metadata.prompt_config.template == model_init_scenario.expected_template
        assert metadata.prompt_config.add_bos_token_to_prompt is model_init_scenario.expected_add_bos
        assert metadata.prompt_config.add_eos_token_to_prompt is model_init_scenario.expected_add_eos

        # Verify BOS token if explicitly expected
        if model_init_scenario.expected_bos_token is not None:
            assert metadata.prompt_config.bos_token == model_init_scenario.expected_bos_token
        if model_init_scenario.expected_bos_token_id is not None:
            assert metadata.prompt_config.bos_token_id == model_init_scenario.expected_bos_token_id

        # Verify max_seq_length
        if model_init_scenario.use_global_max_seq:
            assert metadata.base_max_seq_length == GLOBAL_MAX_SEQ_LENGTH
        elif model_init_scenario.custom_max_position_embeddings:
            assert metadata.base_max_seq_length == model_init_scenario.custom_max_position_embeddings
        else:
            assert metadata.base_max_seq_length == DEFAULT_MAX_SEQ_LENGTH


class TestFromConfig:
    """Tests for ModelMetadata.from_config method."""

    @patch("nemo_safe_synthesizer.llm.metadata.AutoConfig")
    @patch("nemo_safe_synthesizer.llm.metadata.AutoTokenizer")
    @pytest.mark.parametrize(
        "rope_factor, max_seq_per_example, expected_rope_factor, expected_max_seq",
        [
            pytest.param(2, None, 2, None, id="with_rope_no_dp"),
            pytest.param(None, 1, None, 1, id="no_rope_with_dp"),
            pytest.param(None, None, None, None, id="no_rope_no_dp"),
        ],
    )
    def test_from_config(
        self,
        mock_auto_tokenizer,
        mock_auto_config,
        mock_tokenizer,
        mock_autoconfig_obj,
        rope_factor,
        max_seq_per_example,
        expected_rope_factor,
        expected_max_seq,
    ):
        """Test from_config creates metadata from SafeSynthesizerParameters."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_config.from_pretrained.return_value = mock_autoconfig_obj

        mock_config = MagicMock()
        mock_config.training.pretrained_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        mock_config.training.rope_scaling_factor = rope_factor
        mock_config.data.max_sequences_per_example = max_seq_per_example

        metadata = ModelMetadata.from_config(mock_config)

        assert isinstance(metadata, TinyLlama)
        if expected_rope_factor is not None:
            assert metadata.rope_scaling is not None
            assert isinstance(metadata.rope_scaling, RopeScaling)
            assert metadata.rope_scaling.factor == expected_rope_factor
        assert metadata.max_sequences_per_example == expected_max_seq


class TestModelMetadataKwargsPassthrough:
    """Tests for kwargs passthrough in model classes."""

    @patch("nemo_safe_synthesizer.llm.metadata.AutoConfig")
    @patch("nemo_safe_synthesizer.llm.metadata.AutoTokenizer")
    @pytest.mark.parametrize(
        "kwarg_name, kwarg_value, attr_name, expected_value",
        [
            pytest.param("is_adapter", True, "is_adapter", True, id="is_adapter"),
            pytest.param("max_sequences_per_example", 1, "max_sequences_per_example", 1, id="max_seq_per_example"),
        ],
    )
    def test_kwargs_passthrough(
        self,
        mock_auto_tokenizer,
        mock_auto_config,
        mock_tokenizer,
        mock_autoconfig_obj,
        kwarg_name,
        kwarg_value,
        attr_name,
        expected_value,
    ):
        """Test that kwargs are correctly passed through to model metadata."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_config.from_pretrained.return_value = mock_autoconfig_obj

        kwargs = {kwarg_name: kwarg_value}
        metadata = TinyLlama(model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", **kwargs)

        assert getattr(metadata, attr_name) == expected_value

    @patch("nemo_safe_synthesizer.llm.metadata.AutoConfig")
    @patch("nemo_safe_synthesizer.llm.metadata.AutoTokenizer")
    def test_workdir_passthrough(
        self, mock_auto_tokenizer, mock_auto_config, mock_tokenizer, mock_autoconfig_obj, sample_workdir
    ):
        """Test that workdir can be passed through kwargs."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_config.from_pretrained.return_value = mock_autoconfig_obj

        metadata = TinyLlama(model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", workdir=sample_workdir)

        assert metadata.workdir == sample_workdir

    @patch("nemo_safe_synthesizer.llm.metadata.AutoConfig")
    @patch("nemo_safe_synthesizer.llm.metadata.AutoTokenizer")
    def test_rope_scaling_factor_passthrough(
        self, mock_auto_tokenizer, mock_auto_config, mock_tokenizer, mock_autoconfig_obj
    ):
        """Test that rope_scaling_factor can be passed through kwargs."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_config.from_pretrained.return_value = mock_autoconfig_obj

        metadata = TinyLlama(model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", rope_scaling_factor=4)

        assert metadata.rope_scaling is not None
        assert isinstance(metadata.rope_scaling, RopeScaling)
        assert metadata.rope_scaling.factor == 4
        assert metadata.rope_scaling_factor == 4
        assert metadata.max_seq_length == 4 * DEFAULT_MAX_SEQ_LENGTH


class TestTinyLlamaWithTokenizer:
    """Additional TinyLlama-specific tests."""

    @patch("nemo_safe_synthesizer.llm.metadata.AutoConfig")
    def test_tinyllama_with_provided_tokenizer(self, mock_auto_config, mock_tokenizer, mock_autoconfig_obj):
        """Test TinyLlama initialization with a provided tokenizer."""
        mock_auto_config.from_pretrained.return_value = mock_autoconfig_obj

        metadata = TinyLlama(
            model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            tokenizer=mock_tokenizer,
        )

        assert metadata.prompt_config.bos_token == mock_tokenizer.bos_token
        assert metadata.prompt_config.eos_token == mock_tokenizer.eos_token
