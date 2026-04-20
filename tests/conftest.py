# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-level pytest fixtures and hooks for the Safe Synthesizer test suite.

Provides directory-path fixtures, stub-dataset loaders, and mock processors
shared across unit, smoke, and e2e tests.
"""

from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset, load_dataset


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests by directory: `/e2e/` -> e2e, `/smoke/` -> smoke, else unit."""
    category_markers = {"unit", "e2e", "smoke"}

    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        path_str = str(item.fspath)

        if "/e2e/" in path_str:
            if "e2e" not in marker_names:
                item.add_marker(pytest.mark.e2e)
                marker_names.add("e2e")

        if "/smoke/" in path_str:
            if "smoke" not in marker_names:
                item.add_marker(pytest.mark.smoke)
                marker_names.add("smoke")

        if not marker_names.intersection(category_markers):
            item.add_marker(pytest.mark.unit)


@pytest.fixture()
def yaml_config_str() -> str:
    """Return a representative YAML config string covering all SafeSynthesizerParameters sections."""
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
def tests_dir(request) -> Path:
    """Return the path to the tests directory; ``request`` is a built-in pytest fixture."""
    return Path(request.config.rootdir / "tests")


@pytest.fixture(scope="session")
def stub_datasets_dir(tests_dir) -> Path:
    """Path to the stub CSV/Parquet/JSON datasets used by test fixtures."""
    return tests_dir / "stub_datasets"


@pytest.fixture(scope="session")
def test_data_dir(tests_dir) -> Path:
    """Path to the ``tests/test_data/`` directory (tokenizers, PII data, etc.)."""
    return tests_dir / "test_data"


@pytest.fixture(scope="session")
def tokenizers_dir(test_data_dir) -> Path:
    """Path to the directory containing tokenizer fixtures."""
    return test_data_dir / "tokenizers"


@pytest.fixture(scope="session")
def stub_tokenizer_dir(tests_dir) -> Path:
    """Path to a minimal stub tokenizer used for fast unit tests."""
    return tests_dir / "stub_tokenizer"


@pytest.fixture(scope="session")
def pii_test_data_dir(test_data_dir) -> Path:
    """Path to PII-specific test data (NER fixtures, redaction samples)."""
    return test_data_dir / "pii"


@pytest.fixture(scope="session")
def e2e_config_dir(tests_dir) -> Path:
    """Path to YAML configs required by end-to-end tests."""
    return tests_dir / "e2e" / "required_configs"


@pytest.fixture(scope="session")
def fixture_session_cache_dir(tmp_path_factory) -> Path:
    """Create and return a session-scoped temporary directory for test caches."""
    dir = tmp_path_factory.mktemp("nss_pytest_cache")
    return dir


def load_test_dataset(
    dataset_file_name: str,
    datasets_dir: Path,
    data_format: str = "csv",
) -> Dataset:
    """Load a stub dataset file as a HuggingFace ``Dataset``.

    Args:
        dataset_file_name: Filename relative to *datasets_dir*.
        datasets_dir: Directory containing the dataset file.
        data_format: Format passed to ``load_dataset`` (default ``"csv"``).

    Returns:
        The train split of the loaded dataset.

    Raises:
        ValueError: If the loaded object is not a ``Dataset``.
    """
    dataset_path = datasets_dir / dataset_file_name
    data = load_dataset(data_format, data_files=str(dataset_path), cache_dir=None)["train"]
    if isinstance(data, Dataset):
        return data
    raise ValueError(f"Unsupported data type: {type(data)}")


def load_test_dataframe(filename: str, datasets_dir: Path) -> pd.DataFrame:
    """Load a stub dataset file as a pandas ``DataFrame``.

    Dispatches on the file extension: ``.csv``, ``.parquet``, ``.json``, and
    ``.jsonl`` are supported.

    Args:
        filename: Filename relative to *datasets_dir*.
        datasets_dir: Directory containing the dataset file.

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: If the file extension is not recognized.
    """
    dataset_path = datasets_dir / filename
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


@pytest.fixture(scope="session")
def fixture_smollm3_tokenizer(tokenizers_dir) -> str:
    """Return the path to the SmolLM3 tokenizer directory."""
    return str(tokenizers_dir / "smollm3b")


@pytest.fixture
def fixture_iris_dataset(stub_datasets_dir) -> Dataset:
    """Return the Iris dataset."""
    return load_test_dataset("iris.csv", stub_datasets_dir)


@pytest.fixture
def fixture_chickweight_dataset(stub_datasets_dir) -> Dataset:
    """Return the ChickWeight dataset."""
    return load_test_dataset("chickweight.csv", stub_datasets_dir)


@pytest.fixture
def fixture_dow_jones_index_dataset(stub_datasets_dir) -> Dataset:
    """Dow Jones Index dataset (group size 8) for group-by / order-by tests."""
    return load_test_dataset("dow_jones_index_group_size_8.csv", stub_datasets_dir)


@pytest.fixture
def fixture_sample_patient_dataset(stub_datasets_dir) -> Dataset:
    """Sample patient-events dataset (12 groups, 200 records) for grouped tests."""
    return load_test_dataset("sample-patient-events-12groups-200-records.csv", stub_datasets_dir)


@pytest.fixture
def fixture_sample_patient_dataframe(stub_datasets_dir) -> pd.DataFrame:
    """Sample patient-events dataset as a DataFrame."""
    return load_test_dataframe("sample-patient-events-12groups-200-records.csv", stub_datasets_dir)


@pytest.fixture
def fixture_sample_patient_redacted_dataframe(
    fixture_sample_patient_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Patient-events DataFrame with the ``patient_name`` column replaced by ``"REDACTED"``."""
    redacted = fixture_sample_patient_dataframe.copy()
    redacted["patient_name"] = "REDACTED"
    return redacted


@pytest.fixture
def fixture_pems_sf_sample_dataset(stub_datasets_dir) -> Dataset:
    """PEMS-SF traffic-sensor sample for time-series-like tests."""
    return load_test_dataset("pems_sf_sample.csv", stub_datasets_dir)


@pytest.fixture
def fixture_embedded_carriage_return_dataframe(stub_datasets_dir) -> pd.DataFrame:
    """DataFrame with embedded carriage returns for serialization / regex edge-case tests."""
    return load_test_dataframe("embedded_carriage_return.parquet", stub_datasets_dir)


@pytest.fixture()
def fixture_mock_processor():
    """Mock processor returning a ``ParsedResponse`` with 3 valid and 1 invalid record."""
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


@pytest.fixture()
def fixture_mock_processor_without_valid_records():
    """Mock processor returning a ``ParsedResponse`` with zero valid records."""
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
def fixture_lmsys_chat_non_english_dataset(stub_datasets_dir) -> pd.DataFrame:
    """LMSYS chat sample with non-English conversations."""
    return load_test_dataframe("lmsys_chat_non_english_sample.jsonl", stub_datasets_dir)


@pytest.fixture
def fixture_doc_summaries_dataset(stub_datasets_dir) -> pd.DataFrame:
    """Document-summaries dataset for free-text generation tests."""
    return load_test_dataframe("doc_summaries.csv", stub_datasets_dir)


@pytest.fixture
def fixture_clinc_oos_dataset(stub_datasets_dir) -> pd.DataFrame:
    """CLINC OOS (out-of-scope) intent dataset for free-text tests."""
    return load_test_dataframe("clinc_oos.csv", stub_datasets_dir)


@pytest.fixture
def fixture_tokenizer(fixture_smollm3_tokenizer):
    """Load the SmolLM3 tokenizer via ``AutoTokenizer``.

    Skips the requesting test if ``transformers`` is not installed.
    """
    transformers = pytest.importorskip(
        "transformers", reason="transformers is required (install with: uv sync --extra cpu)"
    )
    return transformers.AutoTokenizer.from_pretrained(fixture_smollm3_tokenizer)


@pytest.fixture
def fixture_lmsys_dataset_jsonl_and_schema(
    fixture_lmsys_chat_non_english_dataset,
) -> tuple[str, dict]:
    """LMSYS non-English chat dataset serialized as JSONL with its inferred JSON schema."""
    from nemo_safe_synthesizer.data_processing.dataset import make_json_schema
    from nemo_safe_synthesizer.data_processing.record_utils import records_to_jsonl

    return records_to_jsonl(fixture_lmsys_chat_non_english_dataset), make_json_schema(
        fixture_lmsys_chat_non_english_dataset
    )


@pytest.fixture
def fixture_valid_iris_dataset_jsonl_and_schema(
    fixture_iris_dataset,
) -> tuple[str, dict]:
    """First 5 Iris rows as JSONL with inferred JSON schema for processor / regex tests."""
    from io import StringIO

    from nemo_safe_synthesizer.data_processing.dataset import make_json_schema

    sample_df = pd.DataFrame(fixture_iris_dataset[:5])
    str_buffer = StringIO()
    sample_df.to_json(str_buffer, orient="records", lines=True)
    return str_buffer.getvalue(), make_json_schema(sample_df)
