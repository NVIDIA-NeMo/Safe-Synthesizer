# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import TracebackType

import pytest

from nemo_safe_synthesizer.config.replace_pii import LLMConfig
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.local_inference import (
    LocalVllmProfile,
    load_profile,
    planning_inference_environment,
    resolve_local_server_request,
)
from nemo_safe_synthesizer.pii_replacer.local_inference import managed as managed_module

GPT_OSS = "openai/gpt-oss-120b"
NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


class FakeServer:
    instances: list[FakeServer] = []

    def __init__(
        self,
        profile: LocalVllmProfile,
        *,
        host: str,
        port: int | None,
        environ: dict[str, str] | None,
    ) -> None:
        self.profile = profile
        self.host = host
        self.port = port
        self.running = False
        FakeServer.instances.append(self)

    def __enter__(self) -> FakeServer:
        self.running = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.running = False

    def inference_environ(self) -> dict[str, str]:
        return {"NSS_INFERENCE_ENDPOINT": "http://127.0.0.1:1/v1", "NSS_INFERENCE_MODEL": self.profile.served_name}


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> type[FakeServer]:
    FakeServer.instances = []
    monkeypatch.setattr(managed_module, "LocalVllmServer", FakeServer)
    return FakeServer


@pytest.mark.unit
class TestResolveLocalServerRequest:
    @pytest.mark.parametrize(
        "endpoint",
        ["https://integrate.api.nvidia.com/v1", "http://127.0.0.1:8000/v1"],
        ids=["hosted", "own-local-server"],
    )
    def test_endpoint_without_profile_opts_out_of_the_managed_server(self, endpoint: str) -> None:
        # Explicit-endpoint settings are validated with that endpoint, not as a local profile.
        environ = {"NSS_INFERENCE_ENDPOINT": endpoint, "NSS_INFERENCE_TIMEOUT": "not-a-number"}

        assert resolve_local_server_request(LLMConfig(), environ=environ) is None

    @pytest.mark.parametrize(
        "environ",
        [{}, {"NSS_INFERENCE_KEY": "hosted-key"}],
        ids=["nothing-set", "key-alone"],
    )
    def test_without_endpoint_the_default_is_a_local_nemotron_server(self, environ: dict[str, str]) -> None:
        request = resolve_local_server_request(LLMConfig(), environ=environ)

        assert request is not None
        assert (request.host, request.port, request.profile.served_name) == ("127.0.0.1", None, NEMOTRON)

    @pytest.mark.parametrize(
        ("config", "environ"),
        [
            pytest.param(LLMConfig(model_id=GPT_OSS), {}, id="yaml-model"),
            pytest.param(LLMConfig(), {"NSS_INFERENCE_MODEL": GPT_OSS}, id="env-model"),
        ],
    )
    def test_model_id_selects_the_bundled_profile(self, config: LLMConfig, environ: dict[str, str]) -> None:
        request = resolve_local_server_request(config, environ=environ)

        assert request is not None
        assert request.profile == load_profile("gpt-oss-120b")

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
        with pytest.raises(ParameterError, match="no bundled profile serves it(.|\\n)*NSS_INFERENCE_LOCAL_PROFILE"):
            resolve_local_server_request(config, environ=environ)

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

        request = resolve_local_server_request(LLMConfig(), environ=environ)

        assert request is not None
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

        with pytest.raises(ParameterError, match="NSS_INFERENCE_ENDPOINT must be"):
            resolve_local_server_request(LLMConfig(), environ=environ)

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
        with pytest.raises(ParameterError, match=match):
            resolve_local_server_request(config, environ={"NSS_INFERENCE_LOCAL_PROFILE": "gpt-oss-120b", **environ})

    def test_invalid_profile_reference_is_a_parameter_error(self) -> None:
        with pytest.raises(ParameterError, match="neither a bundled profile"):
            resolve_local_server_request(LLMConfig(), environ={"NSS_INFERENCE_LOCAL_PROFILE": "missing"})


@pytest.mark.unit
class TestPlanningInferenceEnvironment:
    def test_without_profile_yields_the_given_environment(self, fake_server: type[FakeServer]) -> None:
        environ = {"NSS_INFERENCE_ENDPOINT": "https://hosted.example/v1"}

        with planning_inference_environment(LLMConfig(), environ=environ) as yielded:
            assert yielded is environ

        assert fake_server.instances == []

    def test_with_profile_runs_the_server_only_inside_the_block(self, fake_server: type[FakeServer]) -> None:
        environ = {"NSS_INFERENCE_LOCAL_PROFILE": "gpt-oss-120b", "NSS_INFERENCE_ENDPOINT": "http://127.0.0.1:8123/v1"}

        with planning_inference_environment(LLMConfig(), environ=environ) as yielded:
            (server,) = fake_server.instances
            assert server.running
            assert yielded == server.inference_environ()

        assert (server.host, server.port) == ("127.0.0.1", 8123)
        assert not server.running
