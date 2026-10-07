# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone PII planning and replacement SDK interfaces."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiSamplerBackend,
    PiiSamplerConfig,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.data_processing.csv_source import CsvSource
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer import TabularPiiReplacer, TransformResult
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.telemetry import NSSTrainingAndGenerationEvent, TaskStatusEnum


@pytest.fixture(autouse=True)
def fixture_no_telemetry_network(monkeypatch: pytest.MonkeyPatch) -> list[NSSTrainingAndGenerationEvent]:
    """Capture telemetry events instead of sending them; tests read the returned list."""
    events: list[NSSTrainingAndGenerationEvent] = []

    class _CapturingTelemetryHandler:
        def __init__(self, source_client_version: str) -> None:
            pass

        def enqueue(self, event: NSSTrainingAndGenerationEvent) -> None:
            events.append(event)

        def stop(self) -> None:
            pass

    monkeypatch.setattr("nemo_safe_synthesizer.sdk.library_builder.TelemetryHandler", _CapturingTelemetryHandler)
    return events


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
    expected = ReplacePiiConfig(replacement_plan=PiiReplacementPlan())
    nss = SafeSynthesizer(config=config, save_path=tmp_path).with_data_source(dataframe)

    with (
        patch(
            "nemo_safe_synthesizer.pii_replacer.planning.resolve_replacement_config",
            return_value=expected,
        ) as resolve_config,
        patch.object(nss, "process_data") as process_data,
        patch.object(nss, "train") as train,
        patch.object(nss, "generate") as generate,
        patch.object(nss, "evaluate") as evaluate,
    ):
        result = nss.plan_pii_replacement()

    assert result is expected
    args = resolve_config.call_args.args
    assert args[0] is dataframe
    assert args[1] == config.replace_pii
    assert args[2] == config.data
    assert args[3] == config.time_series
    assert resolve_config.call_args.kwargs == {}
    process_data.assert_not_called()
    train.assert_not_called()
    generate.assert_not_called()
    evaluate.assert_not_called()


def test_plan_pii_replacement_persists_when_output_path_is_supplied(tmp_path: Path) -> None:
    dataframe = pd.DataFrame({"email": ["ada@example.com", "grace@example.com"], "visit_id": ["V1", "V2"]})
    inline_plan = PiiReplacementPlan(
        columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)],
    )
    config = SafeSynthesizerParameters(
        replace_pii=ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER), replacement_plan=inline_plan
        )
    )
    output_path = tmp_path / "review" / "safe-synthesizer-config.yaml"

    result = (
        SafeSynthesizer(config=config, save_path=tmp_path / "artifacts")
        .with_data_source(dataframe)
        .plan_pii_replacement(output_path=output_path)
    )

    assert result.inline_plan == inline_plan
    assert result.inline_plan is not None
    assert result.inline_plan.data_to_sampler_value_mapping == {}
    assert SafeSynthesizerParameters.from_yaml(output_path).replace_pii == result
    plan_text = (output_path.parent / "pii_replacement_plan.yaml").read_text()
    assert plan_text.startswith(
        "# PII replacement plan for this dataset, written for the faker sampler in locale en_US."
    )
    assert "Draft review" not in plan_text


def test_plan_pii_replacement_rejects_an_output_path_named_like_the_plan_file(tmp_path: Path) -> None:
    nss = SafeSynthesizer(save_path=tmp_path).with_data_source(pd.DataFrame({"email": ["a@example.com"]}))
    nss.with_replace_pii(replacement_plan={"columns_to_replace": []})

    with pytest.raises(ParameterError, match="output_path cannot be named pii_replacement_plan.yaml"):
        nss.plan_pii_replacement(output_path=tmp_path / "pii_replacement_plan.yaml")


def test_plan_pii_replacement_rejects_disabled_pii(tmp_path: Path) -> None:
    config = SafeSynthesizerParameters(replace_pii=None)
    nss = SafeSynthesizer(config=config, save_path=tmp_path).with_data_source(
        pd.DataFrame({"email": ["a@example.com"]})
    )

    with pytest.raises(ParameterError, match="PII replacement is disabled"):
        nss.plan_pii_replacement()


def test_replace_pii_resolves_and_writes_the_complete_input(tmp_path: Path) -> None:
    dataframe = pd.DataFrame({"email": ["ada@example.com", "grace@example.com"]})
    resolved = ReplacePiiConfig(
        sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        replacement_plan=PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)],
        ),
    )
    data_output = tmp_path / "output" / "pii_replaced.csv"
    config_output = tmp_path / "output" / "safe-synthesizer-config.yaml"
    nss = SafeSynthesizer(
        config=SafeSynthesizerParameters(replace_pii=resolved),
        save_path=tmp_path / "artifacts",
    ).with_data_source(dataframe)

    result = nss.replace_pii(data_output, config_output_path=config_output)

    assert result.replacement_plan == resolved.inline_plan
    assert result.transformed_df["email"].tolist() != dataframe["email"].tolist()
    assert result.transformed_df["email"].str.contains("@").all()
    pd.testing.assert_frame_equal(pd.read_csv(data_output), result.transformed_df)
    assert SafeSynthesizerParameters.from_yaml(config_output).replace_pii == result.resolved_config
    assert nss._column_statistics == result.column_statistics
    assert nss._pii_replacer_time == result.replacement_time_seconds


def test_replace_pii_writes_unchanged_cells_as_in_the_source_csv(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("email,zip,count,note\nada@example.com,02054,7,N/A\n,10001,,None\n")
    resolved = ReplacePiiConfig(
        sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        replacement_plan=PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)],
        ),
    )
    data_output = tmp_path / "pii_replaced.csv"
    nss = SafeSynthesizer(
        config=SafeSynthesizerParameters(replace_pii=resolved),
        save_path=tmp_path / "artifacts",
    ).with_data_source(str(source))

    result = nss.replace_pii(data_output)

    written = pd.read_csv(data_output, dtype=str, na_filter=False)
    assert written["email"].tolist() == [result.transformed_df.loc[0, "email"], ""]
    assert written["email"][0] != "ada@example.com"
    assert written.drop(columns="email").to_dict("list") == {
        "zip": ["02054", "10001"],
        "count": ["7", ""],
        "note": ["N/A", "None"],
    }


def _email_replacement_config() -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        replace_pii=ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)],
            ),
        )
    )


def test_replace_pii_takes_unchanged_cells_from_the_bytes_it_replaced(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("email,zip\nada@example.com,02054\n")
    nss = SafeSynthesizer(config=_email_replacement_config(), save_path=tmp_path / "artifacts").with_data_source(
        str(source)
    )
    original_replace = TabularPiiReplacer.replace

    def replace_then_change_the_file(replacer: TabularPiiReplacer, dataframe: pd.DataFrame) -> TransformResult:
        source.write_text("email,zip\nother@example.com,99999\nmore@example.com,11111\n")
        return original_replace(replacer, dataframe)

    with patch.object(TabularPiiReplacer, "replace", replace_then_change_the_file):
        nss.replace_pii(tmp_path / "pii_replaced.csv")

    written = pd.read_csv(tmp_path / "pii_replaced.csv", dtype=str, na_filter=False)
    assert written["zip"].tolist() == ["02054"]


def test_replace_pii_accepts_a_csv_source(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("email;zip\nada@example.com;02054\n")
    nss = SafeSynthesizer(config=_email_replacement_config(), save_path=tmp_path / "artifacts").with_data_source(
        CsvSource.load(str(source), {"sep": ";"})
    )

    result = nss.replace_pii(tmp_path / "pii_replaced.csv")

    written = pd.read_csv(tmp_path / "pii_replaced.csv", dtype=str, na_filter=False)
    assert result.transformed_df["zip"].tolist() == [2054]
    assert written["zip"].tolist() == ["02054"]


@pytest.mark.filterwarnings("ignore:Support for nested sequences for 'parse_dates':FutureWarning")
def test_replace_pii_keeps_dates_combined_from_csv_columns(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("email,date,time,zip\nada@example.com,2024-01-02,03:04,02054\n")
    nss = SafeSynthesizer(config=_email_replacement_config(), save_path=tmp_path / "artifacts").with_data_source(
        CsvSource.load(str(source), {"parse_dates": {"timestamp": ["date", "time"]}})
    )

    nss.replace_pii(tmp_path / "pii_replaced.csv")

    written = pd.read_csv(tmp_path / "pii_replaced.csv", dtype=str, na_filter=False)
    assert written.columns.tolist() == ["timestamp", "email", "zip"]
    assert written["timestamp"].tolist() == ["2024-01-02 03:04:00"]
    assert written["zip"].tolist() == ["02054"]


def test_replace_pii_rejects_source_text_with_other_columns(tmp_path: Path) -> None:
    resolved = ReplacePiiConfig(
        sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        replacement_plan=PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)],
        ),
    )
    nss = SafeSynthesizer(
        config=SafeSynthesizerParameters(replace_pii=resolved),
        save_path=tmp_path / "artifacts",
    ).with_data_source(pd.DataFrame({"email": ["ada@example.com"]}))

    with pytest.raises(ParameterError, match="same columns and row count"):
        nss.replace_pii(tmp_path / "out.csv", source_text=pd.DataFrame({"other": ["x"]}))


def test_replace_pii_writes_parquet_output_with_its_column_types(tmp_path: Path) -> None:
    source = tmp_path / "input.parquet"
    pd.DataFrame(
        {
            "email": ["ada@example.com", "grace@example.com"],
            "zip": ["02054", "10001"],
            "visits": pd.array([3, None], dtype="Int64"),
        }
    ).to_parquet(source)
    resolved = ReplacePiiConfig(
        sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        replacement_plan=PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)],
        ),
    )
    output = tmp_path / "pii_replaced.parquet"
    nss = SafeSynthesizer(
        config=SafeSynthesizerParameters(replace_pii=resolved),
        save_path=tmp_path / "artifacts",
    ).with_data_source(str(source))

    result = nss.replace_pii(output)

    written = pd.read_parquet(output)
    pd.testing.assert_frame_equal(written, result.transformed_df.reset_index(drop=True), check_dtype=False)
    assert written["zip"].tolist() == ["02054", "10001"]
    assert str(written["visits"].dtype) == "Int64"
    assert not set(written["email"]) & {"ada@example.com", "grace@example.com"}


def test_replace_pii_emits_completed_telemetry_with_sampler_backend(
    tmp_path: Path, fixture_no_telemetry_network: list[NSSTrainingAndGenerationEvent]
) -> None:
    dataframe = pd.DataFrame({"email": ["ada@example.com", "grace@example.com"]})
    replace_pii = ReplacePiiConfig(
        replacement_plan=PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
        ),
        sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
    )
    nss = SafeSynthesizer(
        config=SafeSynthesizerParameters(replace_pii=replace_pii, emit_telemetry=True),
        save_path=tmp_path / "artifacts",
    ).with_data_source(dataframe)

    nss.replace_pii()

    assert len(fixture_no_telemetry_network) == 1
    event = fixture_no_telemetry_network[0]
    assert event.task == "replace_pii"
    assert event.task_status == TaskStatusEnum.COMPLETED
    assert event.replace_pii_enabled is True
    assert event.pii_sampler_backend == "faker"
    assert event.job_duration_sec >= 0.0


def test_replace_pii_emits_error_telemetry_when_pii_is_disabled(
    tmp_path: Path, fixture_no_telemetry_network: list[NSSTrainingAndGenerationEvent]
) -> None:
    nss = SafeSynthesizer(
        config=SafeSynthesizerParameters(replace_pii=None, emit_telemetry=True),
        save_path=tmp_path,
    ).with_data_source(pd.DataFrame({"email": ["a@example.com"]}))

    with pytest.raises(ParameterError, match="PII replacement is disabled"):
        nss.replace_pii()

    assert [(e.task, e.task_status, e.pii_sampler_backend) for e in fixture_no_telemetry_network] == [
        ("replace_pii", TaskStatusEnum.ERROR, "undefined")
    ]


def test_plan_pii_replacement_does_not_emit_telemetry(
    tmp_path: Path, fixture_no_telemetry_network: list[NSSTrainingAndGenerationEvent]
) -> None:
    nss = SafeSynthesizer(
        config=SafeSynthesizerParameters(replace_pii=ReplacePiiConfig(), emit_telemetry=True),
        save_path=tmp_path,
    ).with_data_source(pd.DataFrame({"email": ["a@example.com", "b@example.com"]}))

    nss.plan_pii_replacement()

    assert fixture_no_telemetry_network == []
