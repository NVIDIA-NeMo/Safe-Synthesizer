# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.local_inference import (
    LocalVllmProfile,
    bundled_profile_for_model,
    bundled_profile_names,
    load_profile,
)


@pytest.mark.unit
class TestBundledProfiles:
    @pytest.mark.parametrize("name", bundled_profile_names())
    def test_every_bundled_profile_loads_with_a_pinned_commit(self, name: str) -> None:
        profile = load_profile(name)

        assert re.fullmatch(r"[0-9a-f]{40}", profile.revision)

    def test_gpt_oss_120b_fits_planner_requests_and_parses_reasoning(self) -> None:
        profile = load_profile("gpt-oss-120b")

        assert profile.served_name == "openai/gpt-oss-120b"
        assert profile.max_model_len is not None and profile.max_model_len >= 32768
        assert "--reasoning-parser=openai_gptoss" in profile.extra_args

    def test_bundled_profile_is_found_by_served_model_name(self) -> None:
        assert bundled_profile_for_model("openai/gpt-oss-120b") == load_profile("gpt-oss-120b")
        assert bundled_profile_for_model("gpt-oss-120b") is None


@pytest.mark.unit
class TestLoadProfile:
    def test_loads_profile_file_by_path(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.yaml"
        path.write_text("model_id: org/tiny\nrevision: abc123\nserved_model_name: tiny\n")

        profile = load_profile(str(path))

        assert profile == LocalVllmProfile(model_id="org/tiny", revision="abc123", served_model_name="tiny")
        assert profile.served_name == "tiny"

    def test_unknown_reference_lists_bundled_profiles(self) -> None:
        with pytest.raises(ParameterError, match="gpt-oss-120b"):
            load_profile("no-such-profile")

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            pytest.param("- a\n- b\n", "must be a YAML mapping", id="not-a-mapping"),
            pytest.param("model_id: [unclosed\n", "Could not read", id="bad-yaml"),
            pytest.param("model_id: org/tiny\n", "revision", id="missing-revision"),
            pytest.param(
                "model_id: org/tiny\nrevision: abc\nport: 8000\n", "port", id="address-is-not-a-profile-field"
            ),
        ],
    )
    def test_invalid_profile_files_raise_parameter_error(self, tmp_path: Path, content: str, match: str) -> None:
        path = tmp_path / "profile.yaml"
        path.write_text(content)

        with pytest.raises(ParameterError, match=match):
            load_profile(path)


@pytest.mark.unit
class TestExtraArgs:
    @pytest.mark.parametrize(
        "argument",
        ["--port=9000", "--host", "--api_key", "--enable-log-requests", "--enable-log-req", "--served-model-name=x"],
    )
    def test_rejects_options_nss_manages(self, argument: str) -> None:
        with pytest.raises(ValidationError, match="managed by NSS"):
            LocalVllmProfile(model_id="org/tiny", revision="abc", extra_args=(argument,))

    @pytest.mark.parametrize(
        "argument", ["--reasoning-parser=openai_gptoss", "--enforce-eager", "--api-server-count=2"]
    )
    def test_accepts_other_vllm_options(self, argument: str) -> None:
        profile = LocalVllmProfile(model_id="org/tiny", revision="abc", extra_args=(argument,))

        assert profile.extra_args == (argument,)
