# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from nemo_safe_synthesizer.config.replace_pii import LLMConfig
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.llm_client import InferenceSettings, InvalidRequestSettingsError
from nemo_safe_synthesizer.pii_replacer.local_inference import (
    InvalidLocalServerSettingsError,
    LocalServerRequest,
    load_profile,
    planning_inference_settings,
    resolve_planning_inference,
)

from ..conftest import FakeLocalServer

GPT_OSS = "openai/gpt-oss-120b"
NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def _local_request(config: LLMConfig, environ: dict[str, str]) -> LocalServerRequest:
    request = resolve_planning_inference(config, environ=environ)
    assert isinstance(request, LocalServerRequest)
    return request


@pytest.mark.unit
class TestResolvePlanningInference:
    @pytest.mark.parametrize(
        "endpoint",
        ["https://integrate.api.nvidia.com/v1", "http://127.0.0.1:8000/v1"],
        ids=["hosted", "own-local-server"],
    )
    def test_endpoint_without_profile_selects_that_endpoint(self, endpoint: str) -> None:
        environ = {"NSS_INFERENCE_ENDPOINT": endpoint, "NSS_INFERENCE_TIMEOUT": "90"}

        settings = resolve_planning_inference(LLMConfig(model_id="served/model"), environ=environ)

        assert isinstance(settings, InferenceSettings)
        assert (settings.endpoint_url, settings.model_id, settings.timeout_seconds) == (endpoint, "served/model", 90)

    @pytest.mark.parametrize(
        "environ",
        [{}, {"NSS_INFERENCE_KEY": "hosted-key"}],
        ids=["nothing-set", "key-alone"],
    )
    def test_without_endpoint_the_default_is_a_local_nemotron_server(self, environ: dict[str, str]) -> None:
        request = _local_request(LLMConfig(), environ)

        assert (request.host, request.port, request.profile.served_name) == ("127.0.0.1", None, NEMOTRON)

    @pytest.mark.parametrize(
        ("config", "environ"),
        [
            pytest.param(LLMConfig(model_id=GPT_OSS), {}, id="yaml-model"),
            pytest.param(LLMConfig(), {"NSS_INFERENCE_MODEL": GPT_OSS}, id="env-model"),
        ],
    )
    def test_model_id_selects_the_bundled_profile(self, config: LLMConfig, environ: dict[str, str]) -> None:
        assert _local_request(config, environ).profile == load_profile("gpt-oss-120b")

    @pytest.mark.parametrize(
        ("config", "environ"),
        [
            pytest.param(LLMConfig(model_id="nvidia/nemotron-3-ultra-550b-a55b"), {}, id="yaml-model"),
            pytest.param(LLMConfig(), {"NSS_INFERENCE_MODEL": "org/unbundled"}, id="env-model"),
        ],
    )
    def test_model_without_bundled_profile_points_to_an_endpoint_or_profile(
        self,
        config: LLMConfig,
        environ: dict[str, str],
    ) -> None:
        with pytest.raises(
            InvalidLocalServerSettingsError, match="no bundled profile serves it(.|\\n)*NSS_INFERENCE_LOCAL_PROFILE"
        ):
            resolve_planning_inference(config, environ=environ)

    @pytest.mark.parametrize(
        ("endpoint", "host", "port"),
        [
            pytest.param(None, "127.0.0.1", None, id="unset-picks-free-port"),
            pytest.param("http://127.0.0.1:8123/v1", "127.0.0.1", 8123, id="ipv4-loopback"),
            pytest.param("http://localhost:8000/v1/", "localhost", 8000, id="localhost-trailing-slash"),
            pytest.param("http://[::1]:9000/v1", "::1", 9000, id="ipv6-loopback"),
        ],
    )
    def test_loopback_endpoint_selects_the_listening_address(
        self,
        endpoint: str | None,
        host: str,
        port: int | None,
    ) -> None:
        environ = {"NSS_INFERENCE_LOCAL_PROFILE": "gpt-oss-120b"}
        if endpoint is not None:
            environ["NSS_INFERENCE_ENDPOINT"] = endpoint

        request = _local_request(LLMConfig(), environ)

        assert (request.host, request.port, request.profile.served_name) == (host, port, GPT_OSS)

    @pytest.mark.parametrize(
        "endpoint",
        [
            pytest.param("https://127.0.0.1:8000/v1", id="https"),
            pytest.param("http://inference.example.com:8000/v1", id="remote-host"),
            pytest.param("http://127.0.0.1/v1", id="missing-port"),
            pytest.param("http://127.0.0.1:8000", id="missing-v1-path"),
            pytest.param("http://[::1/v1", id="malformed-ipv6"),
        ],
    )
    def test_endpoint_that_cannot_host_the_managed_server_is_rejected(self, endpoint: str) -> None:
        environ = {"NSS_INFERENCE_LOCAL_PROFILE": "gpt-oss-120b", "NSS_INFERENCE_ENDPOINT": endpoint}

        with pytest.raises(InvalidLocalServerSettingsError, match="NSS_INFERENCE_ENDPOINT must be"):
            resolve_planning_inference(LLMConfig(), environ=environ)

    @pytest.mark.parametrize(
        ("config", "environ", "match"),
        [
            pytest.param(LLMConfig(model_id="other/model"), {}, "replace_pii.llm.model_id", id="yaml-model"),
            pytest.param(LLMConfig(), {"NSS_INFERENCE_MODEL": "other/model"}, "NSS_INFERENCE_MODEL", id="env-model"),
        ],
    )
    def test_conflicting_model_name_is_rejected_before_launch(
        self,
        config: LLMConfig,
        environ: dict[str, str],
        match: str,
    ) -> None:
        with pytest.raises(InvalidLocalServerSettingsError, match=match):
            resolve_planning_inference(config, environ={"NSS_INFERENCE_LOCAL_PROFILE": "gpt-oss-120b", **environ})

    def test_invalid_profile_reference_is_a_local_server_settings_error(self) -> None:
        with pytest.raises(InvalidLocalServerSettingsError, match="neither a bundled profile"):
            resolve_planning_inference(LLMConfig(), environ={"NSS_INFERENCE_LOCAL_PROFILE": "missing"})

    @pytest.mark.parametrize(
        ("environ", "expected"),
        [({}, 120.0), ({"NSS_INFERENCE_TIMEOUT": "30"}, 30.0), ({"NSS_INFERENCE_TIMEOUT": "  "}, 120.0)],
        ids=["profile-timeout", "explicit-timeout-wins", "blank-keeps-profile"],
    )
    def test_planner_timeout_comes_from_profile_unless_set_explicitly(
        self, environ: dict[str, str], expected: float
    ) -> None:
        request = _local_request(LLMConfig(model_id=GPT_OSS), environ)

        assert request.timeout_seconds == expected

    @pytest.mark.parametrize(
        ("environ", "expected_options"),
        [
            ({}, {"temperature": 1.0, "top_p": 1.0}),
            ({"NSS_INFERENCE_REQUEST_OPTIONS": '{"temperature": 0.2}'}, {"temperature": 0.2, "top_p": 1.0}),
            ({"NSS_INFERENCE_REQUEST_OPTIONS": '{"top_p": null}'}, {"temperature": 1.0}),
            ({"NSS_INFERENCE_REQUEST_OPTIONS": "  "}, {"temperature": 1.0, "top_p": 1.0}),
        ],
        ids=["profile-options", "explicit-field-overrides", "null-drops-profile-field", "blank-options-keep-profile"],
    )
    def test_request_options_merge_profile_and_explicit_fields(
        self, environ: dict[str, str], expected_options: dict[str, object]
    ) -> None:
        request = _local_request(LLMConfig(model_id=GPT_OSS), environ)

        assert request.request_options == expected_options

    @pytest.mark.parametrize(
        "environ",
        [{"NSS_INFERENCE_TIMEOUT": "soon"}, {"NSS_INFERENCE_REQUEST_OPTIONS": "[1]"}],
        ids=["timeout", "request-options"],
    )
    def test_invalid_request_settings_are_rejected_before_launch(self, environ: dict[str, str]) -> None:
        with pytest.raises(InvalidRequestSettingsError):
            resolve_planning_inference(LLMConfig(), environ=environ)


@pytest.mark.unit
class TestPlanningInferenceSettings:
    def test_explicit_endpoint_yields_its_settings_without_a_server(
        self, fixture_fake_local_servers: list[FakeLocalServer]
    ) -> None:
        environ = {"NSS_INFERENCE_ENDPOINT": "https://hosted.example/v1"}

        with planning_inference_settings(LLMConfig(model_id="served/model"), environ=environ) as settings:
            assert (settings.endpoint_url, settings.model_id) == ("https://hosted.example/v1", "served/model")

        assert fixture_fake_local_servers == []

    @pytest.mark.parametrize(
        ("endpoint", "raises"),
        [
            ("https://hosted.example:8443/v1?key=secret", False),
            ("https://user:secret@hosted.example:8443/v1", True),
        ],
        ids=["query", "credentials"],
    )
    def test_destination_log_omits_secrets_in_the_endpoint(
        self,
        fixture_fake_local_servers: list[FakeLocalServer],
        caplog: pytest.LogCaptureFixture,
        endpoint: str,
        raises: bool,
    ) -> None:
        caplog.set_level(logging.INFO)
        environ = {"NSS_INFERENCE_ENDPOINT": endpoint}

        if raises:
            with pytest.raises(ParameterError, match="must not contain credentials"):
                with planning_inference_settings(LLMConfig(model_id="m"), environ=environ):
                    pass
        else:
            with planning_inference_settings(LLMConfig(model_id="m"), environ=environ):
                pass
            assert "hosted.example:8443" in caplog.text
        assert "secret" not in caplog.text

    def test_with_profile_runs_the_server_only_inside_the_block(
        self, fixture_fake_local_servers: list[FakeLocalServer]
    ) -> None:
        environ = {
            "NSS_INFERENCE_LOCAL_PROFILE": "gpt-oss-120b",
            "NSS_INFERENCE_ENDPOINT": "http://127.0.0.1:8123/v1",
            "NSS_INFERENCE_TIMEOUT": "45",
        }

        with planning_inference_settings(LLMConfig(max_workers=3), environ=environ) as settings:
            (server,) = fixture_fake_local_servers
            assert server.running
            assert (settings.model_id, settings.max_workers, settings.timeout_seconds) == (GPT_OSS, 3, 45.0)

        assert (server.host, server.port) == ("127.0.0.1", 8123)
        assert server.request_options == {"temperature": 1.0, "top_p": 1.0}
        assert not server.running

    def test_unused_key_is_reported_when_a_local_server_runs(
        self, fixture_fake_local_servers: list[FakeLocalServer], caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)

        with planning_inference_settings(LLMConfig(), environ={"NSS_INFERENCE_KEY": "hosted-key"}):
            pass

        assert "NSS_INFERENCE_KEY is set, but PII planning runs in a local vLLM server" in caplog.text
