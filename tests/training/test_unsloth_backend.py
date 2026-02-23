# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    GenerateParameters,
    SafeSynthesizerParameters,
    TrainingHyperparams,
)


@pytest.fixture(autouse=True)
def mock_autoconfig_from_pretrained():
    """Mock AutoConfig.from_pretrained to prevent HuggingFace API calls during init."""
    mock_config = MagicMock()
    mock_config.max_position_embeddings = 2048
    with patch(
        "nemo_safe_synthesizer.training.huggingface_backend.AutoConfig.from_pretrained",
        return_value=mock_config,
    ):
        yield mock_config


@pytest.fixture
def mock_workdir(fixture_session_cache_dir: Path) -> Workdir:
    """Create a mock workdir for testing."""
    workdir = Workdir(
        base_path=fixture_session_cache_dir,
        config_name="test-config",
        dataset_name="test-dataset",
        run_name="2026-01-15T12:00:00",
    )
    workdir.ensure_directories()
    return workdir


@pytest.fixture
def mock_model_metadata(mock_workdir: Workdir):
    """Create a mock model metadata object."""
    metadata = MagicMock()
    metadata.adapter_path = mock_workdir.train.adapter.path
    metadata.save_path = mock_workdir.run_dir
    metadata.metadata_path = mock_workdir.metadata_file
    return metadata


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
        generation=GenerateParameters(num_records=100),
        privacy=DifferentialPrivacyHyperparams(dp_enabled=False),
    )


@pytest.fixture
def params_with_quantization(base_params):
    """Create params with quantization enabled."""
    base_params.training.quantize_model = True
    base_params.training.quantization_bits = 4
    return base_params


@pytest.fixture
def mock_fast_language_model():
    """Create a mock FastLanguageModel class."""
    mock_flm = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    mock_flm.get_peft_model.return_value = mock_model
    return mock_flm


class TestUnslothTrainerInit:
    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_raises_when_cuda_not_available(
        self, mock_prepare_config, mock_cuda_available, base_params, mock_model_metadata, mock_workdir
    ):
        """Test that RuntimeError is raised when CUDA is not available."""
        mock_cuda_available.return_value = False

        with patch.dict("sys.modules", {"unsloth": MagicMock()}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            with pytest.raises(RuntimeError, match="Cannot use unsloth without GPU"):
                UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)

    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_initializes_with_cuda(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test successful initialization when CUDA is available."""
        mock_cuda_available.return_value = True

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)

            assert backend.model_loader_type == mock_fast_language_model
            mock_prepare_config.assert_called()


class TestUpdateForUnsloth:
    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_renames_pretrained_model_to_model_name(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that pretrained_model_name_or_path is renamed to model_name."""
        mock_cuda_available.return_value = True

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            # Manually set framework_load_params to simulate prepare_config
            backend.framework_load_params = {"pretrained_model_name_or_path": "test-model"}

            backend._update_for_unsloth()

            assert "model_name" in backend.framework_load_params
            assert backend.framework_load_params["model_name"] == "test-model"
            assert "pretrained_model_name_or_path" not in backend.framework_load_params

    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_uses_model_metadata_max_seq_length(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that max_seq_length is set from model_metadata.max_seq_length."""
        mock_cuda_available.return_value = True
        mock_model_metadata.max_seq_length = 2048

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {"max_position_embeddings": 1024}  # Should be ignored

            backend._update_for_unsloth()

            assert "max_seq_length" in backend.framework_load_params
            assert backend.framework_load_params["max_seq_length"] == 2048
            assert "max_position_embeddings" not in backend.framework_load_params

    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_max_seq_length_includes_rope_scaling_from_metadata(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that max_seq_length from model_metadata already includes rope_scaling_factor."""
        mock_cuda_available.return_value = True
        # model_metadata.max_seq_length already includes rope scaling (base 2048 * factor 2 = 4096)
        mock_model_metadata.max_seq_length = 4096

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {}

            backend._update_for_unsloth()

            # max_seq_length should be 4096 (already includes rope scaling from model_metadata)
            assert backend.framework_load_params["max_seq_length"] == 4096

    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_sets_load_in_4bit_when_quantization_enabled(
        self,
        mock_prepare_config,
        mock_cuda_available,
        params_with_quantization,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that load_in_4bit is set when 4-bit quantization is enabled."""
        mock_cuda_available.return_value = True

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(
                params=params_with_quantization, model_metadata=mock_model_metadata, workdir=mock_workdir
            )
            backend.framework_load_params = {}

            backend._update_for_unsloth(quantization_config={"some": "config"})

            assert backend.framework_load_params.get("load_in_4bit") is True
            assert backend.framework_load_params.get("bias") == "none"
            assert backend.framework_load_params.get("use_gradient_checkpointing") == "unsloth"

    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_sets_load_in_8bit_when_8bit_quantization(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that load_in_8bit is set when 8-bit quantization is enabled."""
        mock_cuda_available.return_value = True
        base_params.training.quantize_model = True
        base_params.training.quantization_bits = 8

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {}

            backend._update_for_unsloth(quantization_config={"some": "config"})

            assert backend.framework_load_params.get("load_in_8bit") is True

    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_disables_quantization_when_not_enabled(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that load_in_4bit and load_in_8bit are False when quantization is disabled."""
        mock_cuda_available.return_value = True

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {}

            backend._update_for_unsloth()

            assert backend.framework_load_params.get("load_in_4bit") is False
            assert backend.framework_load_params.get("load_in_8bit") is False

    @patch.dict("os.environ", {"LOCAL_FILES_ONLY": "true"})
    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_reads_local_files_only_from_environment(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that LOCAL_FILES_ONLY environment variable is respected."""
        mock_cuda_available.return_value = True

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {}

            backend._update_for_unsloth()

            assert backend.framework_load_params.get("local_files_only") is True


class TestMaybeQuantize:
    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend._prepare_quantize_base")
    def test_calls_get_peft_model(
        self,
        mock_prepare_quantize_base,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that FastLanguageModel.get_peft_model is called."""
        mock_cuda_available.return_value = True

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.model = MagicMock()
            backend.quant_params = {"task_type": "CAUSAL_LM", "r": 16}

            backend.maybe_quantize()

            mock_fast_language_model.get_peft_model.assert_called_once()

    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend._prepare_quantize_base")
    def test_removes_task_type_from_quant_params(
        self,
        mock_prepare_quantize_base,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that task_type is removed from quant_params for Unsloth."""
        mock_cuda_available.return_value = True

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.model = MagicMock()
            backend.quant_params = {"task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 16}

            backend.maybe_quantize()

            # Verify task_type was not passed to get_peft_model
            call_kwargs = mock_fast_language_model.get_peft_model.call_args[1]
            assert "task_type" not in call_kwargs


class TestLoadPretrainedModel:
    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.add_bos_eos_tokens_to_tokenizer")
    def test_loads_model_and_tokenizer(
        self,
        mock_add_tokens,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that model and tokenizer are loaded from FastLanguageModel."""
        mock_cuda_available.return_value = True
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_fast_language_model.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_add_tokens.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {"model_name": "test-model"}

            backend._load_pretrained_model()

            mock_fast_language_model.from_pretrained.assert_called_once_with(**backend.framework_load_params)
            mock_add_tokens.assert_called_once_with(mock_tokenizer)
            assert backend.model == mock_model
            assert backend.tokenizer == mock_tokenizer


class TestLoadModel:
    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_training_data")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_params")
    def test_load_model_sequence(
        self,
        mock_prepare_params,
        mock_prepare_training_data,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that load_model calls methods in correct sequence."""
        mock_cuda_available.return_value = True

        with patch.dict(
            "sys.modules",
            {
                "unsloth": MagicMock(FastLanguageModel=mock_fast_language_model),
                "unsloth.models": MagicMock(),
                "unsloth.models.loader": MagicMock(),
            },
        ):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {"model_name": "test-model"}
            backend.quant_params = {}

            # Use patch.object for cleaner method patching
            with (
                patch.object(backend, "_load_pretrained_model") as mock_load,
                patch.object(backend, "maybe_quantize") as mock_quantize,
            ):
                backend.load_model()

                # Verify the sequence of calls
                mock_load.assert_called_once()
                mock_quantize.assert_called_once()

    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_disables_llama32_support(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that SUPPORTS_LLAMA32 is set to False to avoid HuggingFace requests."""
        mock_cuda_available.return_value = True
        # Create mock_loader that will be used by the module
        mock_loader = MagicMock()
        mock_loader.SUPPORTS_LLAMA32 = True

        mock_unsloth_models = MagicMock()
        mock_unsloth_models.loader = mock_loader

        with patch.dict(
            "sys.modules",
            {
                "unsloth": MagicMock(FastLanguageModel=mock_fast_language_model),
                "unsloth.models": mock_unsloth_models,
                "unsloth.models.loader": mock_loader,
            },
        ):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {"model_name": "test-model"}
            backend.quant_params = {}

            # Use patch.object for cleaner method patching
            with (
                patch.object(backend, "_load_pretrained_model"),
                patch.object(backend, "maybe_quantize"),
                patch.object(backend, "prepare_training_data"),
                patch.object(backend, "prepare_params"),
            ):
                backend.load_model()

                # Verify SUPPORTS_LLAMA32 was set to False
                # The loader is imported from unsloth.models inside load_model
                import sys

                actual_loader = sys.modules["unsloth.models.loader"]
                assert actual_loader.SUPPORTS_LLAMA32 is False  # type: ignore[union-attr]


class TestInvalidQuantizationBits:
    @patch("nemo_safe_synthesizer.training.unsloth_backend.torch.cuda.is_available")
    @patch("nemo_safe_synthesizer.training.unsloth_backend.HuggingFaceBackend.prepare_config")
    def test_raises_for_invalid_quantization_bits(
        self,
        mock_prepare_config,
        mock_cuda_available,
        base_params,
        mock_model_metadata,
        mock_workdir,
        mock_fast_language_model,
    ):
        """Test that ValueError is raised for invalid quantization bits."""
        mock_cuda_available.return_value = True
        base_params.training.quantize_model = True
        base_params.training.quantization_bits = 16  # Invalid - only 4 and 8 are supported

        with patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=mock_fast_language_model)}):
            from nemo_safe_synthesizer.training.unsloth_backend import UnslothTrainer

            backend = UnslothTrainer(params=base_params, model_metadata=mock_model_metadata, workdir=mock_workdir)
            backend.framework_load_params = {}

            with pytest.raises(ValueError, match="Invalid quantization bits"):
                backend._update_for_unsloth(quantization_config={"some": "config"})
