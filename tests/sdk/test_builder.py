# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import GenerateParameters, SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import (
    DEFAULT_PII_TRANSFORM_CONFIG,
    PiiReplacerConfig,
)
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

_SMALL_DF = pd.DataFrame({"a": [1, 2, 3]})
_REPORT_HTML = "<html><body>report</body></html>"

PATCH_PREFIX = "nemo_safe_synthesizer.sdk.builder"


def test_safe_synthesizer_builder_sanity():
    SafeSynthesizer(config=SafeSynthesizerParameters())


@pytest.fixture
def fixture_base_builder() -> SafeSynthesizer:
    builder = SafeSynthesizer().with_data_source(pd.DataFrame(data={"name": ["John", "Jane", "Jim"]}))
    return builder


def test_pii_replacer_only_builder(fixture_base_builder: SafeSynthesizer):
    builder = fixture_base_builder.with_replace_pii(
        config={
            "globals": {
                "classify": {
                    "enable_classify": True,
                },
                "locales": ["en_US"],
            },
            "steps": [
                {
                    "vars": {
                        "row_seed": "random.random()",
                    },
                    "rows": {
                        "update": [
                            {
                                "condition": 'column.entity == "first_name" and not (this | isna)',
                                "value": "fake.persona(row_index=vars.row_seed + index).first_name",
                            }
                        ]
                    },
                }
            ],
        }
    ).resolve()

    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None
    assert builder._nss_config.replace_pii.globals.classify.enable_classify is True


def test_all_builder():
    builder = (
        SafeSynthesizer()
        .with_data_source(pd.DataFrame({"name": ["John", "Jane", "Jim"]}))
        .with_replace_pii(
            config={
                "globals": {
                    "classify": {
                        "enable_classify": True,
                    },
                    "locales": ["en_US"],
                },
                "steps": [
                    {
                        "vars": {
                            "row_seed": "random.random()",
                        },
                        "rows": {
                            "update": [
                                {
                                    "condition": 'column.entity == "first_name" and not (this | isna)',
                                    "value": "fake.persona(row_index=vars.row_seed + index).first_name",
                                }
                            ]
                        },
                    }
                ],
            }
        )
        .resolve()
    )

    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None
    assert builder._nss_config.training.num_input_records_to_sample == "auto"


def test_builder_change_training_params_with_dict():
    builder = (
        SafeSynthesizer()
        .with_data_source(pd.DataFrame({"name": ["John", "Jane", "Jim"]}))
        .with_train(
            config={
                "batch_size": 128,
            }
        )
        .resolve()
    )
    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None
    assert builder._nss_config.training.batch_size == 128


def test_builder_change_training_params_with_kwargs():
    builder = (
        SafeSynthesizer()
        .with_data_source(pd.DataFrame(data={"name": ["John", "Jane", "Jim"]}))
        .with_train(num_input_records_to_sample=5000)
        .resolve()
    )

    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None
    assert builder._nss_config.training.num_input_records_to_sample == 5000


def test_builder_change_generation_params_with_object(fixture_base_builder: SafeSynthesizer):
    builder = (
        fixture_base_builder.with_data_source(pd.DataFrame(data={"name": ["John", "Jane", "Jim"]}))
        .with_generate(config=GenerateParameters(num_records=10000))
        .resolve()
    )

    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None
    assert builder._nss_config.training.num_input_records_to_sample == "auto"
    assert builder._nss_config.generation.num_records == 10000


def test_builder_change_generation_params_with_kwargs(fixture_base_builder):
    builder = (
        fixture_base_builder.with_data_source(pd.DataFrame(data={"name": ["John", "Jane", "Jim"]}))
        .with_generate(config=None, num_records=10000, patience=42)
        .resolve()
    )
    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None
    assert builder._nss_config.generation.patience == 42
    assert builder._nss_config.training.num_input_records_to_sample == "auto"
    assert builder._nss_config.generation.num_records == 10000


def test_pii_replacer_with_default_config_object(fixture_base_builder):
    """Test PII replacer configuration using PiiReplacerConfig.get_default_config()"""
    default_config = PiiReplacerConfig.get_default_config()

    builder = fixture_base_builder.with_replace_pii(config=default_config).resolve()
    assert builder._nss_config is not None
    assert default_config == builder._nss_config.replace_pii


def test_builder_with_all_parameters_customized():
    """Test builder with all parameters customized (training, generation, PII, data, evaluation)"""
    data = pd.DataFrame(
        {
            "id": range(100),
            "name": [f"Person_{i}" for i in range(100)],
            "value": range(100, 200),
        }
    )

    builder = (
        SafeSynthesizer()
        .with_data_source(data)
        .with_replace_pii(config=PiiReplacerConfig.get_default_config())
        .with_train(
            batch_size=64,
            learning_rate=0.001,
            num_input_records_to_sample=50,
            validation_ratio=0.2,
        )
        .with_generate(
            num_records=5000,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.2,
        )
        .with_evaluate(
            mia_enabled=False,
            aia_enabled=True,
            sqs_report_rows=1000,
        )
        .resolve()
    )
    config = builder._nss_config
    assert config is not None

    assert config.replace_pii is not None

    # Training params
    assert config.training.batch_size == 64
    assert config.training.learning_rate == 0.001
    assert config.training.num_input_records_to_sample == 50
    assert config.training.validation_ratio == 0.2

    # Generation params
    assert config.generation.num_records == 5000
    assert config.generation.temperature == 0.8
    assert config.generation.top_p == 0.95
    assert config.generation.repetition_penalty == 1.2

    # Evaluation params
    assert config.evaluation.mia_enabled is False
    assert config.evaluation.aia_enabled is True
    assert config.evaluation.sqs_report_rows == 1000


def test_pii_config_equality():
    """Test that PiiReplacerConfig objects can be compared for equality"""
    config1 = PiiReplacerConfig.get_default_config()
    config2 = PiiReplacerConfig.get_default_config()
    yaml_config = PiiReplacerConfig.from_yaml_str(DEFAULT_PII_TRANSFORM_CONFIG)

    # Same configs should be equal
    assert config1.model_dump() == config2.model_dump()
    assert config1.model_dump() == yaml_config.model_dump()

    # Modified config should not be equal
    # Need to properly construct nested models to avoid serialization warnings
    modified_globals = config1.globals.model_copy(deep=True)
    modified_globals.classify.enable_classify = False
    modified_config = config1.model_copy(update={"globals": modified_globals}, deep=True)
    assert config1.model_dump() != modified_config.model_dump()


def test_pii_replacer_from_yaml_str(fixture_base_builder):
    """Test creating PiiReplacerConfig directly from YAML string"""
    config = PiiReplacerConfig.from_yaml_str(DEFAULT_PII_TRANSFORM_CONFIG)

    builder = fixture_base_builder.with_replace_pii(config=config).resolve()

    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None


def test_builder_with_evaluation_config(fixture_base_builder):
    """Test builder with evaluation configuration"""
    builder = fixture_base_builder.with_evaluate(
        mia_enabled=True,
        aia_enabled=True,
        sqs_report_columns=100,
        sqs_report_rows=2000,
        pii_replay_enabled=True,
        pii_replay_entities=["email", "phone_number"],
    ).resolve()
    assert builder._nss_config is not None
    config = builder._nss_config
    assert config.evaluation.mia_enabled is True
    assert config.evaluation.aia_enabled is True
    assert config.evaluation.sqs_report_columns == 100
    assert config.evaluation.sqs_report_rows == 2000
    assert config.evaluation.pii_replay_enabled is True
    assert config.evaluation.pii_replay_entities == ["email", "phone_number"]


def test_builder_with_data_config():
    """Test builder with data configuration parameters"""
    builder = (
        SafeSynthesizer()
        .with_data_source(pd.DataFrame({"col": range(100)}))
        .with_data(
            holdout=0.1,
            max_holdout=1000,
            random_state=42,
            group_training_examples_by="col",
        )
        .resolve()
    )

    assert builder._nss_config is not None
    config = builder._nss_config
    assert config.data.holdout == 0.1
    assert config.data.max_holdout == 1000
    assert config.data.random_state == 42
    assert config.data.group_training_examples_by == "col"


def test_default_builder_has_pii_enabled():
    builder = SafeSynthesizer().with_data_source(_SMALL_DF).resolve()
    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None


def test_with_replace_pii_enable_false_disables_pii():
    builder = SafeSynthesizer().with_data_source(_SMALL_DF).with_replace_pii(enable=False).resolve()
    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is None


def test_with_train_still_enables_pii_by_default():
    builder = SafeSynthesizer().with_data_source(_SMALL_DF).with_train().resolve()
    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None


# Regression tests for https://github.com/NVIDIA/NeMo-Safe-Synthesizer/issues/132
# Root cause: ConfigBuilder had a stale default that silently skipped PII replacement
# when no with_replace_pii() call was made.


def test_regression_132_sdk_no_with_replace_pii_call_still_enables_pii():
    """Bare SDK builder with no with_replace_pii() call must resolve to PII enabled.

    This is the exact scenario from issue #132: the old code had a stale default
    that resolved to None, silently skipping PII.
    """
    config = SafeSynthesizer().with_data_source(_SMALL_DF).resolve()._nss_config
    assert config is not None
    assert config.replace_pii is not None


def test_regression_132_cli_path_no_overrides_enables_pii():
    """CLI path (model_validate with empty overrides) must also default to PII enabled.

    Ensures SDK and CLI agree on the default: both produce a populated replace_pii
    config when no PII flags are passed.
    """
    config = SafeSynthesizerParameters.model_validate({})
    assert config.replace_pii is not None


def test_builder_seeded_from_config_with_pii_disabled():
    """SafeSynthesizer(config=existing) must propagate replace_pii=None from the seed config.

    The __init__ branch that reads from an existing SafeSynthesizerParameters
    seeds _replace_pii_config from config.replace_pii directly. This test
    confirms that a disabled seed stays disabled after resolve().
    """
    existing = SafeSynthesizerParameters(replace_pii=None)
    config = SafeSynthesizer(config=existing).with_data_source(_SMALL_DF).resolve()._nss_config
    assert config is not None
    assert config.replace_pii is None


def test_builder_seeded_from_config_with_pii_enabled():
    """SafeSynthesizer(config=existing) must propagate a populated replace_pii from the seed config."""
    existing = SafeSynthesizerParameters()
    config = SafeSynthesizer(config=existing).with_data_source(_SMALL_DF).resolve()._nss_config
    assert config is not None
    assert config.replace_pii is not None


def test_with_replace_pii_reenable_after_disable():
    """Calling with_replace_pii() after with_replace_pii(enable=False) re-enables PII."""
    config = (
        SafeSynthesizer()
        .with_data_source(_SMALL_DF)
        .with_replace_pii(enable=False)
        .with_replace_pii()
        .resolve()
        ._nss_config
    )
    assert config is not None
    assert config.replace_pii is not None


_METRICS_JSON = '{"timing": {}}'


def _builder_with_mock_results(tmp_path: Path) -> SafeSynthesizer:
    """Create a SafeSynthesizer with mocked results for save_results testing."""
    nss = SafeSynthesizer(save_path=tmp_path / "artifacts")
    nss.results = MagicMock()
    nss.results.synthetic_data = _SMALL_DF
    nss.results.evaluation_report_html = _REPORT_HTML
    nss.results.summary.model_dump_json.return_value = _METRICS_JSON
    return nss


class TestSaveResults:
    """Verify save_results persists CSV, HTML, and evaluation metrics."""

    def test_saves_to_default_workdir(self, tmp_path: Path):
        nss = _builder_with_mock_results(tmp_path)

        nss.save_results()
        assert nss._workdir is not None

        csv_path = nss._workdir.output_file
        report_path = nss._workdir.evaluation_report
        metrics_path = nss._workdir.evaluation_metrics
        assert csv_path.exists()
        assert report_path.exists()
        assert metrics_path.exists()
        assert pd.read_csv(csv_path).equals(_SMALL_DF)
        assert report_path.read_text() == _REPORT_HTML
        assert metrics_path.read_text() == _METRICS_JSON

    def test_output_file_override_writes_csv_to_custom_path(self, tmp_path: Path):
        nss = _builder_with_mock_results(tmp_path)
        custom_csv = tmp_path / "custom" / "output.csv"

        nss.save_results(output_file=custom_csv)

        assert custom_csv.exists()
        assert pd.read_csv(custom_csv).equals(_SMALL_DF)
        # Report still goes to the workdir regardless of output_file
        assert nss._workdir is not None
        assert nss._workdir.evaluation_report.exists()
        assert nss._workdir.evaluation_report.read_text() == _REPORT_HTML

    def test_skips_report_when_html_is_none(self, tmp_path: Path):
        nss = _builder_with_mock_results(tmp_path)
        nss.results.evaluation_report_html = None

        nss.save_results()
        assert nss._workdir is not None

        assert nss._workdir.output_file.exists()
        assert not nss._workdir.evaluation_report.exists()
        assert not nss._workdir.evaluation_metrics.exists()
