# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RemoteBackend HTTP generation backend.

These tests never contact a live server: ``httpx`` and the processor are
mocked so the suite stays fast, deterministic, and GPU-free.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config import (
    DataParameters,
    GenerateParameters,
    RemoteParameters,
    SafeSynthesizerParameters,
    TimeSeriesParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.errors import GenerationError, InternalError, ParameterError
from nemo_safe_synthesizer.generation.backend import GeneratorBackend
from nemo_safe_synthesizer.generation.batch import Batch
from nemo_safe_synthesizer.generation.processors import TabularDataProcessor, create_processor
from nemo_safe_synthesizer.generation.remote_backend import RemoteBackend
from nemo_safe_synthesizer.llm.metadata import ModelMetadata

MODULE = "nemo_safe_synthesizer.generation.remote_backend"


@pytest.fixture
def mock_schema():
    return {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}


@pytest.fixture
def mock_model_metadata():
    metadata = MagicMock(spec=ModelMetadata)
    metadata.instruction = "Generate data"
    metadata.prompt_config = MagicMock()
    metadata.prompt_config.template = "[INST] {instruction} {schema} [/INST]"
    metadata.prompt_config.bos_token = "<s>"
    metadata.prompt_config.eos_token = "</s>"
    return metadata


def make_params(
    *,
    endpoint_url: str = "http://localhost:8000/v1",
    model: str = "my-lora",
    api_key_env: str | None = None,
    max_concurrency: int = 16,
) -> SafeSynthesizerParameters:
    """Build params with a configured remote endpoint."""
    return SafeSynthesizerParameters(
        data=DataParameters(group_training_examples_by=None, order_training_examples_by=None),
        training=TrainingHyperparams(pretrained_model="test-model", lora_r=16),
        generation=GenerateParameters(
            num_records=10,
            use_structured_generation=False,
            remote=RemoteParameters(
                endpoint_url=endpoint_url,
                model=model,
                api_key_env=api_key_env,
                max_concurrency=max_concurrency,
            ),
        ),
    )


def make_backend(config, model_metadata, schema, processor=None) -> RemoteBackend:
    """Construct a RemoteBackend with patched schema/prompt/processor helpers."""
    with (
        patch(f"{MODULE}.load_json", return_value=schema),
        patch(f"{MODULE}.utils.create_schema_prompt", return_value="test prompt"),
        patch(f"{MODULE}.create_processor", return_value=processor or MagicMock()),
    ):
        return RemoteBackend(config=config, model_metadata=model_metadata, workdir=MagicMock())


def make_http_response(text: str, completion_tokens: int, finish_reason: str = "stop") -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = {
        "choices": [{"text": text, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": completion_tokens},
    }
    response.raise_for_status.return_value = None
    return response


class TestConstruction:
    def test_sets_remote_flag_and_prompt(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        assert backend.remote is True
        assert backend.prompt == "test prompt"
        assert backend.columns == ["name", "age"]

    def test_is_concrete_generator_backend(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        assert isinstance(backend, GeneratorBackend)

    def test_prompt_token_count_is_zero(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        assert backend._get_prompt_token_count() == 0


class TestInitialize:
    def test_creates_client_and_pool(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend.initialize()
        try:
            client, pool = backend._client, backend._pool
            assert client is not None
            assert pool is not None
            assert "Authorization" not in client.headers
        finally:
            backend.teardown()

    def test_api_key_header_set_from_env(self, mock_model_metadata, mock_schema, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret-token")
        backend = make_backend(make_params(api_key_env="MY_KEY"), mock_model_metadata, mock_schema)
        backend.initialize()
        try:
            client = backend._client
            assert client is not None
            assert client.headers["Authorization"] == "Bearer secret-token"
        finally:
            backend.teardown()

    def test_missing_api_key_raises(self, mock_model_metadata, mock_schema, monkeypatch):
        monkeypatch.delenv("MY_KEY", raising=False)
        backend = make_backend(make_params(api_key_env="MY_KEY"), mock_model_metadata, mock_schema)
        with pytest.raises(ParameterError, match="MY_KEY"):
            backend.initialize()


class TestPrepareParams:
    def _sampling_kwargs(self, **overrides):
        kwargs = dict(
            temperature=0.9,
            top_p=1.0,
            max_tokens=128,
            repetition_penalty=1.0,
            top_k=-1,
            min_p=0,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            ignore_eos=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_builds_request_body(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend.prepare_params(**self._sampling_kwargs())
        body = backend._request_body
        assert body is not None
        assert body["model"] == "my-lora"
        assert body["n"] == 1
        assert body["temperature"] == 0.9
        assert body["max_tokens"] == 128
        # vLLM protocol extensions are present.
        assert body["repetition_penalty"] == 1.0
        assert body["top_k"] == -1
        assert body["min_p"] == 0
        # No structured generation -> no guided fields.
        assert "guided_regex" not in body
        assert "guided_json" not in body

    def test_guided_json_when_json_schema_method(self, mock_model_metadata, mock_schema):
        config = make_params()
        config.generation.use_structured_generation = True
        config.generation.structured_generation_schema_method = "json_schema"
        backend = make_backend(config, mock_model_metadata, mock_schema)
        backend.prepare_params(**self._sampling_kwargs())
        body = backend._request_body
        assert body is not None
        assert body["guided_json"] == mock_schema

    def test_guided_regex_when_regex_method(self, mock_model_metadata, mock_schema):
        config = make_params()
        config.generation.use_structured_generation = True
        config.generation.structured_generation_schema_method = "regex"
        config.generation.structured_generation_backend = "outlines"
        backend = make_backend(config, mock_model_metadata, mock_schema)
        with patch(f"{MODULE}.build_json_based_regex", return_value="REGEX") as build_regex:
            backend.prepare_params(**self._sampling_kwargs())
        build_regex.assert_called_once()
        body = backend._request_body
        assert body is not None
        assert body["guided_regex"] == "REGEX"

    def test_structural_tag_rejected(self, mock_model_metadata, mock_schema):
        config = make_params()
        config.generation.use_structured_generation = True
        config.generation.structured_generation_schema_method = "structural_tag"
        config.generation.structured_generation_backend = "xgrammar"
        backend = make_backend(config, mock_model_metadata, mock_schema)
        with pytest.raises(ParameterError, match="structural_tag"):
            backend.prepare_params(**self._sampling_kwargs())


class TestCompleteOne:
    def test_parses_text_tokens_and_finish_reason(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend._client = MagicMock()
        backend._client.post.return_value = make_http_response('{"name": "A", "age": 1}', 7, "stop")
        backend._request_body = {"model": "my-lora"}
        text, tokens, finish = backend._complete_one()
        assert text == '{"name": "A", "age": 1}'
        assert tokens == 7
        assert finish == "stop"
        # The per-request prompt is injected into the body.
        _, kwargs = backend._client.post.call_args
        assert kwargs["json"]["prompt"] == "test prompt"

    def test_http_status_error_becomes_generation_error(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend._client = MagicMock()
        error_response = MagicMock(status_code=500, text="boom")
        backend._client.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=error_response
        )
        backend._request_body = {"model": "my-lora"}
        with pytest.raises(GenerationError, match="500"):
            backend._complete_one()

    def test_complete_one_before_init_raises(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        with pytest.raises(InternalError):
            backend._complete_one()


class TestGenerateBatch:
    def test_dispatches_n_requests_and_counts(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend.initialize()
        try:
            with patch.object(backend, "_complete_one", return_value=("rec", 5, "stop")) as complete:
                batch = backend._generate_batch(num_prompts_per_batch=3, batch=Batch(processor=MagicMock()))
            assert complete.call_count == 3
            assert batch.finish_reasons["stop"] == 3
            assert batch.total_completion_tokens == 15
        finally:
            backend.teardown()

    def test_generate_batch_before_init_raises(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        with pytest.raises(InternalError):
            backend._generate_batch(num_prompts_per_batch=1, batch=Batch(processor=MagicMock()))


class TestTeardown:
    def test_idempotent(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend.initialize()
        pool = backend._pool
        assert pool is not None
        backend.teardown()
        backend.teardown()  # second call must be a no-op, not an error
        assert backend._client is None
        assert backend._pool is None
        # underlying resources were released
        assert pool._shutdown


class TestResponseEdgeCases:
    def test_missing_usage_defaults_tokens_to_zero(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend._client = MagicMock()
        response = MagicMock(spec=httpx.Response)
        response.json.return_value = {"choices": [{"text": "x", "finish_reason": "stop"}]}  # no usage key
        response.raise_for_status.return_value = None
        backend._client.post.return_value = response
        backend._request_body = {"model": "my-lora"}
        _, tokens, _ = backend._complete_one()
        assert tokens == 0

    def test_malformed_response_becomes_generation_error(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend._client = MagicMock()
        response = MagicMock(spec=httpx.Response)
        response.json.return_value = {"unexpected": "shape"}  # no "choices"
        response.raise_for_status.return_value = None
        response.text = '{"unexpected": "shape"}'
        backend._client.post.return_value = response
        backend._request_body = {"model": "my-lora"}
        with pytest.raises(GenerationError, match="unexpected response shape"):
            backend._complete_one()

    def test_none_finish_reason_counted_as_unknown(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend.initialize()
        try:
            with patch.object(backend, "_complete_one", return_value=("rec", 1, None)):
                batch = backend._generate_batch(num_prompts_per_batch=2, batch=Batch(processor=MagicMock()))
            assert batch.finish_reasons["unknown"] == 2
        finally:
            backend.teardown()


class TestGenerateEndToEnd:
    """Exercise the shared GeneratorBackend.generate() template loop via RemoteBackend.

    Network is mocked at ``_complete_one``, but the real TabularDataProcessor,
    Batch, and result aggregation run -- so this guards the base-class refactor.
    """

    def test_full_loop_with_real_processor(self, mock_model_metadata, mock_schema):
        config = make_params()
        config.generation.num_records = 3
        mock_model_metadata.generation_max_tokens_for.return_value = 128
        processor = create_processor(mock_schema, mock_model_metadata, config)
        assert isinstance(processor, TabularDataProcessor)

        backend = make_backend(config, mock_model_metadata, mock_schema, processor=processor)
        backend.initialize()
        try:
            with patch.object(backend, "_complete_one", return_value=('{"name": "Alice", "age": 30}', 8, "stop")):
                results = backend.generate()
        finally:
            backend.teardown()

        assert results.df is not None
        assert len(results.df) >= 3
        assert list(results.df.columns) == ["name", "age"]


class TestRemoteParametersValidation:
    def test_requires_endpoint_and_model(self):
        with pytest.raises(ValidationError):
            RemoteParameters.model_validate({})

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            RemoteParameters(endpoint_url="http://x/v1", model="m", timeout_seconds=0)

    def test_max_concurrency_at_least_one(self):
        with pytest.raises(ValidationError):
            RemoteParameters(endpoint_url="http://x/v1", model="m", max_concurrency=0)


class TestBackendSelectionValidation:
    def test_remote_with_timeseries_rejected_at_config_time(self):
        # The ParameterError raised in the model validator is wrapped by pydantic.
        with pytest.raises(ValidationError, match="time-series"):
            SafeSynthesizerParameters(
                data=DataParameters(group_training_examples_by="g", order_training_examples_by=None),
                training=TrainingHyperparams(pretrained_model="test-model", lora_r=16),
                generation=GenerateParameters(remote=RemoteParameters(endpoint_url="http://x/v1", model="m")),
                time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="t"),
            )
