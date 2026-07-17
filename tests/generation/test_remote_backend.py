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
    StructuredGenerationParameters,
    TimeSeriesParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.config.generate import RemoteDialect
from nemo_safe_synthesizer.errors import GenerationError, InternalError, ParameterError
from nemo_safe_synthesizer.generation.backend import GeneratorBackend
from nemo_safe_synthesizer.generation.batch import Batch
from nemo_safe_synthesizer.generation.processors import TabularDataProcessor, create_processor
from nemo_safe_synthesizer.generation.remote_backend import (
    RemoteBackend,
    _backoff_delay,
    _coerce_token_count,
    _compact_json_completion,
    _parse_retry_after,
)
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
    # The resume / offline-remote path carries no local tokenizer.
    metadata.tokenizer = None
    return metadata


def make_params(
    *,
    endpoint_url: str = "http://localhost:8000/v1",
    model: str = "my-lora",
    api_key_env: str | None = None,
    max_concurrency: int = 16,
    max_retries: int = 4,
    dialect: RemoteDialect = "vllm",
) -> SafeSynthesizerParameters:
    """Build params with a configured remote endpoint."""
    return SafeSynthesizerParameters(
        data=DataParameters(group_training_examples_by=None, order_training_examples_by=None),
        training=TrainingHyperparams(pretrained_model="test-model", lora_r=16),
        generation=GenerateParameters(
            num_records=10,
            structured_generation=StructuredGenerationParameters(enabled=False),
            remote=RemoteParameters(
                endpoint_url=endpoint_url,
                model=model,
                api_key_env=api_key_env,
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                dialect=dialect,
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
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"text": text, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": completion_tokens},
    }
    response.raise_for_status.return_value = None
    return response


def make_status_response(status_code: int, text: str = "", *, retry_after: str | None = None) -> MagicMock:
    """A response with an explicit status code, for retry/error-path tests."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = text
    response.headers = {"Retry-After": retry_after} if retry_after else {}
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            str(status_code), request=MagicMock(), response=response
        )
    else:
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

    def test_prompt_token_count_is_zero_without_tokenizer(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        assert backend._get_prompt_token_count() == 0

    def test_prompt_token_count_uses_tokenizer_when_present(self, mock_model_metadata, mock_schema):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_model_metadata.tokenizer = tokenizer
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        assert backend._get_prompt_token_count() == 5
        tokenizer.encode.assert_called_once_with("test prompt")
        # Result is cached: the tokenizer is not re-invoked.
        assert backend._get_prompt_token_count() == 5
        tokenizer.encode.assert_called_once()


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
        # No structured generation -> no structured_outputs field.
        assert "structured_outputs" not in body

    def test_openai_dialect_omits_vllm_extensions(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(dialect="openai"), mock_model_metadata, mock_schema)
        backend.prepare_params(**self._sampling_kwargs())
        body = backend._request_body
        assert body is not None
        # Universal OpenAI fields are still present.
        assert body["temperature"] == 0.9
        assert body["top_p"] == 1.0
        assert body["max_tokens"] == 128
        assert body["n"] == 1
        # The vLLM-only extensions that strict servers (NIM/TRT-LLM) reject are dropped.
        for field in (
            "repetition_penalty",
            "top_k",
            "min_p",
            "skip_special_tokens",
            "include_stop_str_in_output",
            "ignore_eos",
        ):
            assert field not in body

    def test_structured_outputs_json_vllm_dialect_disables_whitespace(self, mock_model_metadata, mock_schema):
        config = make_params(dialect="vllm")
        config.generation.structured_generation.enabled = True
        config.generation.structured_generation.schema_method = "json_schema"
        backend = make_backend(config, mock_model_metadata, mock_schema)
        backend.prepare_params(**self._sampling_kwargs())
        body = backend._request_body
        assert body is not None
        # Source fix: vLLM/xgrammar emits compact JSON when whitespace is disabled.
        assert body["structured_outputs"] == {"json": mock_schema, "disable_any_whitespace": True}
        # Compaction stays armed as a safety net even with the source fix on.
        assert backend._compact_json is True

    def test_structured_outputs_json_openai_dialect_omits_vllm_extension(self, mock_model_metadata, mock_schema):
        config = make_params(dialect="openai")
        config.generation.structured_generation.enabled = True
        config.generation.structured_generation.schema_method = "json_schema"
        backend = make_backend(config, mock_model_metadata, mock_schema)
        backend.prepare_params(**self._sampling_kwargs())
        body = backend._request_body
        assert body is not None
        # The vLLM-only field would 400 on strict OpenAI servers, so it is dropped;
        # the post-process compaction is the portable fallback.
        assert body["structured_outputs"] == {"json": mock_schema}
        assert backend._compact_json is True

    def test_compact_json_not_armed_for_other_methods(self, mock_model_metadata, mock_schema):
        config = make_params()
        config.generation.structured_generation.enabled = True
        config.generation.structured_generation.schema_method = "regex"
        config.generation.structured_generation.backend = "outlines"
        backend = make_backend(config, mock_model_metadata, mock_schema)
        with patch(f"{MODULE}.build_json_based_regex", return_value="REGEX"):
            backend.prepare_params(**self._sampling_kwargs())
        assert backend._compact_json is False

    def test_compact_json_not_armed_without_structured_generation(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend.prepare_params(**self._sampling_kwargs())
        assert backend._compact_json is False

    def test_structured_outputs_regex_when_regex_method(self, mock_model_metadata, mock_schema):
        config = make_params()
        config.generation.structured_generation.enabled = True
        config.generation.structured_generation.schema_method = "regex"
        config.generation.structured_generation.backend = "outlines"
        backend = make_backend(config, mock_model_metadata, mock_schema)
        with patch(f"{MODULE}.build_json_based_regex", return_value="REGEX") as build_regex:
            backend.prepare_params(**self._sampling_kwargs())
        build_regex.assert_called_once()
        body = backend._request_body
        assert body is not None
        assert body["structured_outputs"] == {"regex": "REGEX"}

    def test_structured_outputs_structural_tag_when_structural_tag_method(self, mock_model_metadata, mock_schema):
        config = make_params()
        config.generation.structured_generation.enabled = True
        config.generation.structured_generation.schema_method = "structural_tag"
        config.generation.structured_generation.backend = "xgrammar"
        backend = make_backend(config, mock_model_metadata, mock_schema)
        with patch(f"{MODULE}.build_json_structural_tag", return_value="TAG") as build_tag:
            backend.prepare_params(**self._sampling_kwargs())
        build_tag.assert_called_once()
        body = backend._request_body
        assert body is not None
        assert body["structured_outputs"] == {"structural_tag": "TAG"}


class TestCompleteOne:
    def test_parses_text_tokens_and_finish_reason(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend._client = MagicMock()
        backend._client.post.return_value = make_http_response('{"name": "A", "age": 1}', 7, "stop")
        backend._request_body = {"model": "my-lora", "prompt": "test prompt"}
        text, tokens, finish = backend._complete_one()
        assert text == '{"name": "A", "age": 1}'
        assert tokens == 7
        assert finish == "stop"
        # The prompt is baked into the request body by prepare_params and posted as-is.
        _, kwargs = backend._client.post.call_args
        assert kwargs["json"] == {"model": "my-lora", "prompt": "test prompt"}

    def test_complete_one_before_init_raises(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        with pytest.raises(InternalError):
            backend._complete_one()


class TestPostCompletionRetries:
    """Transient-failure resilience: the network path must survive blips, not abort the run."""

    @staticmethod
    def _backend_and_client(mock_model_metadata, mock_schema, *, max_retries: int) -> tuple[RemoteBackend, MagicMock]:
        backend = make_backend(make_params(max_retries=max_retries), mock_model_metadata, mock_schema)
        client = MagicMock()
        backend._client = client
        backend._request_body = {"model": "m", "prompt": "p"}
        return backend, client

    def test_retries_retryable_status_then_succeeds(self, mock_model_metadata, mock_schema):
        backend, client = self._backend_and_client(mock_model_metadata, mock_schema, max_retries=3)
        client.post.side_effect = [
            make_status_response(503, "busy"),
            make_http_response('{"name": "A"}', 3, "stop"),
        ]
        with patch(f"{MODULE}.time.sleep") as sleep:
            text, tokens, finish = backend._complete_one()
        assert (text, tokens, finish) == ('{"name": "A"}', 3, "stop")
        assert client.post.call_count == 2
        sleep.assert_called_once()

    def test_retries_transient_connection_error(self, mock_model_metadata, mock_schema):
        backend, client = self._backend_and_client(mock_model_metadata, mock_schema, max_retries=3)
        client.post.side_effect = [
            httpx.ConnectError("connection refused"),
            make_http_response('{"name": "A"}', 1, "stop"),
        ]
        with patch(f"{MODULE}.time.sleep"):
            backend._complete_one()
        assert client.post.call_count == 2

    def test_persistent_retryable_status_exhausts_attempts(self, mock_model_metadata, mock_schema):
        backend, client = self._backend_and_client(mock_model_metadata, mock_schema, max_retries=2)
        client.post.return_value = make_status_response(503, "down")
        with patch(f"{MODULE}.time.sleep") as sleep, pytest.raises(GenerationError, match="after 3 attempt"):
            backend._complete_one()
        assert client.post.call_count == 3  # initial + 2 retries
        assert sleep.call_count == 2

    def test_non_retryable_status_fails_fast(self, mock_model_metadata, mock_schema):
        backend, client = self._backend_and_client(mock_model_metadata, mock_schema, max_retries=5)
        client.post.return_value = make_status_response(400, "bad request")
        with patch(f"{MODULE}.time.sleep") as sleep, pytest.raises(GenerationError, match="400"):
            backend._complete_one()
        assert client.post.call_count == 1  # no retries on a permanent error
        sleep.assert_not_called()

    def test_zero_retries_disables_retry(self, mock_model_metadata, mock_schema):
        backend, client = self._backend_and_client(mock_model_metadata, mock_schema, max_retries=0)
        client.post.return_value = make_status_response(503, "down")
        with patch(f"{MODULE}.time.sleep") as sleep, pytest.raises(GenerationError, match="after 1 attempt"):
            backend._complete_one()
        assert client.post.call_count == 1
        sleep.assert_not_called()

    def test_honors_retry_after_header(self, mock_model_metadata, mock_schema):
        backend, client = self._backend_and_client(mock_model_metadata, mock_schema, max_retries=1)
        client.post.side_effect = [
            make_status_response(429, "slow down", retry_after="2.5"),
            make_http_response('{"name": "A"}', 1, "stop"),
        ]
        with patch(f"{MODULE}.time.sleep") as sleep:
            backend._complete_one()
        sleep.assert_called_once_with(2.5)


class TestResilienceHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (5, 5),
            (0, 0),
            (-3, 0),
            (None, 0),
            (True, 0),
            (False, 0),
            ("7", 7),
            (7.9, 7),
            ("bad", 0),
            ([], 0),
            ({}, 0),
        ],
    )
    def test_coerce_token_count(self, value, expected):
        assert _coerce_token_count(value) == expected

    def test_parse_retry_after_numeric_seconds(self):
        response = MagicMock(spec=httpx.Response)
        response.headers = {"Retry-After": "3"}
        assert _parse_retry_after(response) == 3.0

    def test_parse_retry_after_absent(self):
        response = MagicMock(spec=httpx.Response)
        response.headers = {}
        assert _parse_retry_after(response) is None

    def test_parse_retry_after_http_date_ignored(self):
        response = MagicMock(spec=httpx.Response)
        response.headers = {"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"}
        assert _parse_retry_after(response) is None

    def test_backoff_delay_uses_retry_after_capped(self):
        assert _backoff_delay(0, 999.0) == 30.0
        assert _backoff_delay(5, 3.0) == 3.0

    def test_backoff_delay_full_jitter_within_bounds(self):
        for attempt in range(7):
            assert 0.0 <= _backoff_delay(attempt, None) <= 30.0


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

    def test_compacts_pretty_printed_json_before_processing(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend._compact_json = True
        backend.initialize()
        pretty = '{\n  "name": "Alice",\n  "age": 30\n}'
        try:
            with patch.object(backend, "_complete_one", return_value=(pretty, 8, "stop")):
                batch = Batch(processor=MagicMock())
                with patch.object(batch, "process") as process:
                    backend._generate_batch(num_prompts_per_batch=1, batch=batch)
        finally:
            backend.teardown()
        _, kwargs = process.call_args
        # The multi-line object is collapsed to single-line JSONL the extractor can match.
        assert process.call_args[0][1] == '{"name":"Alice","age":30}'
        assert kwargs["completion_tokens"] == 8

    def test_no_compaction_when_flag_disabled(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend.initialize()
        pretty = '{\n  "name": "Alice"\n}'
        try:
            with patch.object(backend, "_complete_one", return_value=(pretty, 8, "stop")):
                batch = Batch(processor=MagicMock())
                with patch.object(batch, "process") as process:
                    backend._generate_batch(num_prompts_per_batch=1, batch=batch)
        finally:
            backend.teardown()
        # Text passes through untouched when compaction is not armed.
        assert process.call_args[0][1] == pretty


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
        response.status_code = 200
        response.json.return_value = {"choices": [{"text": "x", "finish_reason": "stop"}]}  # no usage key
        response.raise_for_status.return_value = None
        backend._client.post.return_value = response
        backend._request_body = {"model": "my-lora", "prompt": "p"}
        _, tokens, _ = backend._complete_one()
        assert tokens == 0

    def test_null_completion_tokens_defaults_to_zero(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend._client = MagicMock()
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        # A null token count must degrade to 0, not crash the completion.
        response.json.return_value = {
            "choices": [{"text": "x", "finish_reason": "stop"}],
            "usage": {"completion_tokens": None},
        }
        response.raise_for_status.return_value = None
        backend._client.post.return_value = response
        backend._request_body = {"model": "my-lora", "prompt": "p"}
        _, tokens, _ = backend._complete_one()
        assert tokens == 0

    def test_malformed_response_becomes_generation_error(self, mock_model_metadata, mock_schema):
        backend = make_backend(make_params(), mock_model_metadata, mock_schema)
        backend._client = MagicMock()
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"unexpected": "shape"}  # no "choices"
        response.raise_for_status.return_value = None
        response.text = '{"unexpected": "shape"}'
        backend._client.post.return_value = response
        backend._request_body = {"model": "my-lora", "prompt": "p"}
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


class TestCompactJsonCompletion:
    def test_collapses_multiline_object(self):
        pretty = '{\n  "name": "Alice",\n  "age": 30\n}'
        assert _compact_json_completion(pretty) == '{"name":"Alice","age":30}'

    def test_strips_surrounding_whitespace(self):
        assert _compact_json_completion('  \n {"a": 1}\n  ') == '{"a":1}'

    def test_preserves_non_ascii(self):
        assert _compact_json_completion('{"city": "São Paulo"}') == '{"city":"São Paulo"}'

    def test_already_compact_object_unchanged(self):
        assert _compact_json_completion('{"a":1,"b":2}') == '{"a":1,"b":2}'

    def test_non_json_returned_unchanged(self):
        assert _compact_json_completion("not json at all") == "not json at all"

    def test_empty_returned_unchanged(self):
        assert _compact_json_completion("   ") == "   "

    def test_multiple_objects_returned_unchanged(self):
        # Two JSONL records don't parse as one object; the line-oriented
        # extractor already handles them, so leave the text untouched.
        jsonl = '{"a": 1}\n{"a": 2}'
        assert _compact_json_completion(jsonl) == jsonl

    def test_non_object_json_returned_unchanged(self):
        # A bare array is valid JSON but not a record object; don't reshape it.
        assert _compact_json_completion("[1, 2, 3]") == "[1, 2, 3]"


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
