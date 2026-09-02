# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plan-only PII replacement SDK interface."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import EntityType, PiiColumnPlan, PiiReplacementPlan, ReplacePiiConfig
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import load_plan
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer


def test_importing_safe_synthesizer_does_not_load_model_stack() -> None:
    code = (
        "import sys;"
        "from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer;"
        "forbidden = sorted(name for name in ('torch', 'torchao', 'transformers', 'vllm') if name in sys.modules);"
        "print(','.join(forbidden));"
        "sys.exit(1 if forbidden else 0)"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, (
        "Importing SafeSynthesizer loaded model dependencies that are not required "
        f"for plan-only workflows: {result.stdout.strip()}\nstderr: {result.stderr}"
    )


def test_plan_pii_replacement_delegates_full_input_without_pipeline_stages(tmp_path: Path) -> None:
    dataframe = pd.DataFrame({"row_id": range(100), "email": [f"person-{i}@example.com" for i in range(100)]})
    config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig())
    expected = PiiReplacementPlan()
    nss = SafeSynthesizer(config=config, save_path=tmp_path).with_data_source(dataframe)

    with (
        patch("nemo_safe_synthesizer.pii_replacer.planning.resolve_plan", return_value=expected) as resolve_plan,
        patch.object(nss, "process_data") as process_data,
        patch.object(nss, "train") as train,
        patch.object(nss, "generate") as generate,
        patch.object(nss, "evaluate") as evaluate,
    ):
        result = nss.plan_pii_replacement()

    assert result is expected
    args = resolve_plan.call_args.args
    assert args[0] is dataframe
    assert args[1] == config.replace_pii
    assert args[2] == config.data
    assert args[3] == config.time_series
    assert resolve_plan.call_args.kwargs == {"output_path": None}
    process_data.assert_not_called()
    train.assert_not_called()
    generate.assert_not_called()
    evaluate.assert_not_called()


def test_plan_pii_replacement_persists_when_output_path_is_supplied(tmp_path: Path) -> None:
    dataframe = pd.DataFrame({"email": ["ada@example.com", "grace@example.com"]})
    inline_plan = PiiReplacementPlan(
        columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
    )
    config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(replacement_plan=inline_plan))
    output_path = tmp_path / "review" / "pii_replacement_plan.yaml"

    result = (
        SafeSynthesizer(config=config, save_path=tmp_path / "artifacts")
        .with_data_source(dataframe)
        .plan_pii_replacement(output_path=output_path)
    )

    assert result == inline_plan
    assert load_plan(output_path) == inline_plan


def test_plan_pii_replacement_rejects_disabled_pii(tmp_path: Path) -> None:
    config = SafeSynthesizerParameters(replace_pii=None)
    nss = SafeSynthesizer(config=config, save_path=tmp_path).with_data_source(
        pd.DataFrame({"email": ["a@example.com"]})
    )

    with pytest.raises(ParameterError, match="PII replacement is disabled"):
        nss.plan_pii_replacement()
