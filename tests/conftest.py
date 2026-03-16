# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset, load_dataset


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Auto-marks tests based on their location:
    - Tests in gpu_integration/ get the 'gpu_integration' marker
    - Tests in e2e/ directories get the 'e2e' marker
    - Tests in integration/ directories get the 'integration' marker
    - Tests without category markers get the 'unit' marker
    """
    category_markers = {
        "unit",
        "e2e",
        "integration",
        "gpu_integration",
    }

    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        path_str = str(item.fspath)

        # Auto-mark tests in gpu_integration/ directory
        if "/gpu_integration/" in path_str:
            if "gpu_integration" not in marker_names:
                item.add_marker(pytest.mark.gpu_integration)
                marker_names.add("gpu_integration")

        if "/e2e/" in path_str:
            if "e2e" not in marker_names:
                item.add_marker(pytest.mark.e2e)
                marker_names.add("e2e")

        elif "/integration/" in path_str:
            if "integration" not in marker_names:
                item.add_marker(pytest.mark.integration)
                marker_names.add("integration")

        if not marker_names.intersection(category_markers):
            item.add_marker(pytest.mark.unit)


@pytest.fixture()
def yaml_config_str() -> str:
    return """
data:
  group_training_examples_by: null
  holdout: 0.05
  max_holdout: 2000
  max_sequences_per_example: 2
  order_training_examples_by: null
  random_state: 872602
evaluation:
  aia_enabled: true
  enabled: true
  mandatory_columns: null
  mia_enabled: true
  pii_replay_columns: null
  pii_replay_enabled: true
  pii_replay_entities: null
  quasi_identifier_count: 3
  sqs_report_columns: 250
  sqs_report_rows: 5000
generation:
  invalid_fraction_threshold: 0.8
  num_records: 1000
  patience: 1
  repetition_penalty: 1.0
  temperature: 0.9
  top_p: 1.0
  use_structured_generation: false
privacy:
  delta: auto
  dp_enabled: false
  epsilon: 1.0
  per_sample_max_grad_norm: 1.0
replace_pii: null
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0005
  lora_alpha_over_r: 1.0
  lora_r: 32
  lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  lr_scheduler: cosine
  num_input_records_to_sample: auto
  pretrained_model: HuggingFaceTB/SmolLM3-3B
  rope_scaling_factor: auto
  use_unsloth: auto
  validation_ratio: 0.0
  validation_steps: 15
  warmup_ratio: 0.05
  weight_decay: 0.01
"""


@pytest.fixture(scope="session")
def fixture_session_cache_dir(tmp_path_factory) -> Path:
    dir = tmp_path_factory.mktemp("nss_pytest_cache")
    return dir


# Purpose: Load a small stub dataset from disk via HuggingFace datasets (train split only).
# Data: CSV files under tests/generation/stub_datasets; cached in the tests directory.
def load_test_dataset(
    dataset_file_name: str,
    fixture_session_cache_dir: Path | None = None,
    data_format: str = "csv",
) -> Dataset:
    dir_path = Path(__file__).parent
    dataset_path = dir_path / "stub_datasets" / dataset_file_name
    data = load_dataset(data_format, data_files=str(dataset_path), cache_dir=None)["train"]
    if isinstance(data, Dataset):
        return data
    raise ValueError(f"Unsupported data type: {type(data)}")


# Purpose: Load a small stub DataFrame from a file for edge-case testing.
# Data: files under ./stub_datasets.
def load_test_dataframe(filename: str) -> pd.DataFrame:
    dataset_path = Path(__file__).parent / "stub_datasets" / filename
    match dataset_path.suffix:
        case ".csv":
            return pd.read_csv(str(dataset_path))
        case ".parquet":
            return pd.read_parquet(dataset_path)

        case ".json":
            return pd.read_json(str(dataset_path))
        case ".jsonl":
            return pd.read_json(str(dataset_path), lines=True)

        case _:
            raise ValueError(f"Unknown dataset format: {dataset_path.suffix}")


@pytest.fixture
def fixture_smollm3_tokenizer() -> str:
    return str(Path(__file__).parent / "test_data" / "tokenizers" / "smollm3b")


# Purpose: Iris dataset fixture (train split) for quick test sampling.
# Used by: fixture_valid_iris_dataset_jsonl_and_schema (internal consumer)
@pytest.fixture
def fixture_iris_dataset() -> Dataset:
    return load_test_dataset("iris.csv")


# Purpose: ChickWeight dataset fixture (train split) for sampling.
# Used by: (no direct test references currently)
@pytest.fixture
def fixture_chickweight_dataset() -> Dataset:
    return load_test_dataset("chickweight.csv")


# Purpose: Dow Jones Index dataset fixture for group-by-order-by tests.
# Used by:
#   - data_processing/test_assembler.py::test_assembler_dow_jones_index_dataset
#   - e2e/test_safe_synthesizer.py::test_dow_jones_index_dataset
@pytest.fixture
def fixture_dow_jones_index_dataset() -> Dataset:
    return load_test_dataset("dow_jones_index_group_size_8.csv")


# Purpose: Sample patient events dataset with multiple groups for grouped tests.
# Used by: (no direct test references currently)
@pytest.fixture
def fixture_sample_patient_dataset() -> Dataset:
    return load_test_dataset("sample-patient-events-12groups-200-records.csv")


@pytest.fixture
def fixture_sample_patient_dataframe() -> pd.DataFrame:
    return load_test_dataframe("sample-patient-events-12groups-200-records.csv")


@pytest.fixture
def fixture_sample_patient_redacted_dataframe(
    fixture_sample_patient_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    redacted = fixture_sample_patient_dataframe.copy()
    redacted["patient_name"] = "REDACTED"
    return redacted


# Purpose: PEMS-SF sample dataset fixture for time-series like tests.
# Used by: (no direct test references currently)
@pytest.fixture
def fixture_pems_sf_sample_dataset() -> Dataset:
    return load_test_dataset("pems_sf_sample.csv")


# Purpose: DataFrame with embedded carriage returns to exercise serialization/regex edge cases.
# Used by: (no direct test references currently)
@pytest.fixture
def fixture_embedded_carriage_return_dataframe() -> pd.DataFrame:
    return load_test_dataframe("embedded_carriage_return.parquet")


# Purpose: Minimal processor returning a predictable ParsedResponse (3 valid, 1 invalid).
# Data: valid_records=[{"some":"value0","other":1},{"some":"value1","other":2},{"some":"value2","other":pd.NA}];
#       invalid_records=["invalidjson"]; errors=["some error msg"]; prompt_number=1.
# Used by:
#   - tests/generation/test_batch.py::test_batch_process
#   - tests/generation/test_batch.py::test_batch_to_dataframe
#   - tests/generation/test_generation.py::test_apply_data_actions
#   - tests/generation/test_generation.py::fixture_stub_batches (as dependency)
@pytest.fixture()
def fixture_mock_processor():
    stub_valid_records = [
        dict(some="value0", other=1),
        dict(some="value1", other=2),
        dict(some="value2", other=pd.NA),
    ]
    from unittest.mock import MagicMock

    from nemo_safe_synthesizer.generation.processors import ParsedResponse

    mock_processor = MagicMock()
    mock_processor.return_value = ParsedResponse(
        valid_records=stub_valid_records,
        invalid_records=["invalidjson"],
        errors=[("some error msg", "some error msg")],
        prompt_number=1,
    )
    return mock_processor


# Purpose: Minimal processor yielding 0 valid, one invalid sentinel, and a generic error.
# Used by:
#   - tests/generation/test_batch.py::test_batch_to_dataframe_without_valid_records
#   - tests/generation/test_generation.py::fixture_stub_batches (as dependency)
@pytest.fixture()
def fixture_mock_processor_without_valid_records():
    from unittest.mock import MagicMock

    from nemo_safe_synthesizer.generation.processors import ParsedResponse

    mock_processor = MagicMock()
    mock_processor.return_value = ParsedResponse(
        valid_records=[],
        invalid_records=["invalidjson"],
        errors=[("some error msg", "some error msg")],
        prompt_number=1,
    )
    return mock_processor


@pytest.fixture
def fixture_lmsys_chat_non_english_dataset() -> pd.DataFrame:
    return load_test_dataframe("lmsys_chat_non_english_sample.jsonl")


@pytest.fixture
def fixture_adobe_sampled_dataset() -> pd.DataFrame:
    return load_test_dataframe("adobe-sampled.csv")


# Purpose: Clinc OOS dataset fixture for free text tests.
# Used by: e2e/test_safe_synthesizer.py::test_clinc_oos_dataset
@pytest.fixture
def fixture_clinc_oos_dataset() -> pd.DataFrame:
    return load_test_dataframe("clinc_oos.csv")
