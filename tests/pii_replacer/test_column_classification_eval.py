# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column-classification golden tests and optional live benchmark.

Fast local checks:

    uv run --frozen pytest tests/pii_replacer/test_column_classification_eval.py \
        -k "gold_fixture or accuracy_metric or benchmark_matrix or summary_report" -n 0 -vv

Live benchmark:

    export NSS_INFERENCE_KEY=<build.nvidia.com API key>
    # Optional for local HF if models are not cached or require gated access:
    export HF_TOKEN=<huggingface token>

    NSS_RUN_CLASSIFICATION_BENCHMARK=1 \
    NSS_CLASSIFICATION_BENCHMARK_OUTPUT=local_runs/classification-benchmark.json \
    uv run --frozen pytest tests/pii_replacer/test_column_classification_eval.py -m slow -s -n 0

The live benchmark runs the API baseline and local HF models as separate pytest
cases. Use comma-separated filters to narrow runs:

    NSS_CLASSIFICATION_BENCHMARK_BACKENDS=local_hf
    NSS_CLASSIFICATION_BENCHMARK_API_MODELS=default
    NSS_CLASSIFICATION_BENCHMARK_LOCAL_MODELS=smollm3,mistral,tinyllama

Reports:
- Per-case JSON: ``column-classification-benchmark-{backend}-{model}.json``
- Combined JSON: ``column-classification-benchmark-summary.json``
- Combined table: ``column-classification-benchmark-summary.md``

When ``NSS_CLASSIFICATION_BENCHMARK_OUTPUT`` is set, the per-case and summary
files are written next to that path using its stem.
"""

# ruff: noqa: E402
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("torch", reason="torch is required for these tests (install with: uv sync --extra cpu)")

from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.pii_replacer.data_editor.detect import UNKNOWN_ENTITY, DefaultLLMConfig
from nemo_safe_synthesizer.pii_replacer.nemo_pii import classify_config_from_params, get_column_classifier

LOCAL_HF_CLASSIFICATION_BENCHMARK_MODELS = {
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}
API_CLASSIFICATION_BENCHMARK_MODELS = {
    "default": DefaultLLMConfig.DEFAULT_CONFIG_ID,
}
CLASSIFICATION_BENCHMARK_BACKENDS = ("api", "local_hf")
CLASSIFICATION_BENCHMARK_LOCK_PATH = Path(".pytest_cache") / "column-classification-benchmark.lock"
CLASSIFICATION_BENCHMARK_CASE_IDS = (
    "pii_dataset_structured",
    "credit_card_transactions",
    "adult_negative_control",
)


def _load_gold(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["cases"]


def _column_accuracy(predicted: Mapping[str, str | None], expected: Mapping[str, str]) -> float:
    if not expected:
        return 1.0
    _validate_prediction_columns(predicted, expected)
    correct = sum(_normalize_label(predicted.get(column)) == label for column, label in expected.items())
    return correct / len(expected)


def _positive_recall(predicted: Mapping[str, str | None], expected: Mapping[str, str]) -> float:
    _validate_prediction_columns(predicted, expected)
    positives = {column: label for column, label in expected.items() if label != UNKNOWN_ENTITY}
    if not positives:
        return 1.0
    correct = sum(_normalize_label(predicted.get(column)) == label for column, label in positives.items())
    return correct / len(positives)


def _validate_prediction_columns(predicted: Mapping[str, str | None], expected: Mapping[str, str]) -> None:
    missing = set(expected) - set(predicted)
    if missing:
        raise AssertionError(f"Classifier did not return predictions for columns: {sorted(missing)}")


def _normalize_label(label: str | None) -> str:
    return label if label else UNKNOWN_ENTITY


def _expected_positive_labels(cases: list[dict[str, Any]]) -> set[str]:
    return {label for case in cases for label in case["expected_entities"].values() if label != UNKNOWN_ENTITY}


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _benchmark_matrix() -> list[tuple[str, str, str]]:
    backends = _parse_csv_env("NSS_CLASSIFICATION_BENCHMARK_BACKENDS", CLASSIFICATION_BENCHMARK_BACKENDS)
    model_names_by_backend = {
        "api": _parse_csv_env("NSS_CLASSIFICATION_BENCHMARK_API_MODELS", tuple(API_CLASSIFICATION_BENCHMARK_MODELS)),
        "local_hf": _parse_csv_env(
            "NSS_CLASSIFICATION_BENCHMARK_LOCAL_MODELS", tuple(LOCAL_HF_CLASSIFICATION_BENCHMARK_MODELS)
        ),
    }

    unknown_backends = set(backends) - set(CLASSIFICATION_BENCHMARK_BACKENDS)
    if unknown_backends:
        pytest.fail(f"Unknown classification benchmark backends: {sorted(unknown_backends)}")

    unknown_api_models = set(model_names_by_backend["api"]) - set(API_CLASSIFICATION_BENCHMARK_MODELS)
    if unknown_api_models:
        pytest.fail(f"Unknown API classification benchmark models: {sorted(unknown_api_models)}")

    unknown_local_models = set(model_names_by_backend["local_hf"]) - set(LOCAL_HF_CLASSIFICATION_BENCHMARK_MODELS)
    if unknown_local_models:
        pytest.fail(f"Unknown local classification benchmark models: {sorted(unknown_local_models)}")

    return [
        (backend, model_name, _models_for_backend(backend)[model_name])
        for backend in backends
        for model_name in model_names_by_backend[backend]
    ]


def _models_for_backend(backend: str) -> dict[str, str]:
    if backend == "api":
        return API_CLASSIFICATION_BENCHMARK_MODELS
    return LOCAL_HF_CLASSIFICATION_BENCHMARK_MODELS


def _all_benchmark_params() -> list[Any]:
    return [
        pytest.param(backend, model_name, model, id=f"{backend}-{model_name}")
        for backend in CLASSIFICATION_BENCHMARK_BACKENDS
        for model_name, model in _models_for_backend(backend).items()
    ]


def _benchmark_report_path(backend: str, model_name: str) -> Path:
    configured = os.environ.get("NSS_CLASSIFICATION_BENCHMARK_OUTPUT")
    if configured is None:
        return Path(f"column-classification-benchmark-{backend}-{model_name}.json")

    output_path = Path(configured)
    return output_path.with_name(f"{output_path.stem}-{backend}-{model_name}{output_path.suffix}")


def _benchmark_summary_paths() -> tuple[Path, Path]:
    configured = os.environ.get("NSS_CLASSIFICATION_BENCHMARK_OUTPUT")
    if configured is None:
        json_path = Path("column-classification-benchmark-summary.json")
    else:
        output_path = Path(configured)
        json_path = output_path.with_name(f"{output_path.stem}-summary.json")
    return json_path, json_path.with_suffix(".md")


def _sort_result_key(result: dict[str, Any]) -> tuple[int, int]:
    backend = result["backend"]
    model_name = result["model_name"]
    backend_idx = CLASSIFICATION_BENCHMARK_BACKENDS.index(backend)
    model_names = tuple(_models_for_backend(backend))
    model_idx = model_names.index(model_name)
    return backend_idx, model_idx


def _upsert_result(results: list[dict[str, Any]] | Any, result: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = [
        existing
        for existing in results
        if not (existing["backend"] == result["backend"] and existing["model_name"] == result["model_name"])
    ]
    filtered.append(result)
    return sorted(filtered, key=_sort_result_key)


def _render_benchmark_markdown(report: dict[str, Any]) -> str:
    header = ["Backend", "Model", "Macro Accuracy", "Macro Positive Recall", *CLASSIFICATION_BENCHMARK_CASE_IDS]
    lines = [
        "# Column Classification Benchmark",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for result in report["results"]:
        case_scores = {case["case_id"]: case["accuracy"] for case in result["cases"]}
        row = [
            result["backend"],
            result["model_name"],
            f"{result['macro_accuracy']:.3f}",
            f"{result['macro_positive_recall']:.3f}",
            *(
                f"{case_scores[case_id]:.3f}" if case_id in case_scores else "n/a"
                for case_id in CLASSIFICATION_BENCHMARK_CASE_IDS
            ),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_benchmark_reports(result: dict[str, Any]) -> tuple[Path, Path, Path]:
    report = {
        "api_models": API_CLASSIFICATION_BENCHMARK_MODELS,
        "local_hf_models": LOCAL_HF_CLASSIFICATION_BENCHMARK_MODELS,
        "results": [result],
    }
    output_path = _benchmark_report_path(result["backend"], result["model_name"])
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    summary_json_path, summary_md_path = _benchmark_summary_paths()
    if summary_json_path.exists():
        summary = json.loads(summary_json_path.read_text())
    else:
        summary = {
            "api_models": API_CLASSIFICATION_BENCHMARK_MODELS,
            "local_hf_models": LOCAL_HF_CLASSIFICATION_BENCHMARK_MODELS,
            "results": [],
        }
    summary["api_models"] = API_CLASSIFICATION_BENCHMARK_MODELS
    summary["local_hf_models"] = LOCAL_HF_CLASSIFICATION_BENCHMARK_MODELS
    summary["results"] = _upsert_result(summary["results"], result)
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary_md_path.write_text(_render_benchmark_markdown(summary))

    return output_path, summary_json_path, summary_md_path


@contextmanager
def _classification_benchmark_lock():
    """Serialize live benchmark cases even when pytest-xdist is enabled."""
    try:
        import fcntl
    except ImportError:
        pytest.skip("Column classification live benchmark locking requires fcntl.")

    CLASSIFICATION_BENCHMARK_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLASSIFICATION_BENCHMARK_LOCK_PATH.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _evaluate_classifier(
    backend: Literal["api", "local_hf"], model_name: str, model: str, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    config = PiiReplacerConfig.get_default_config()
    config.globals.classify.backend = backend
    config.globals.classify.model = model
    config.globals.classify.entities = sorted(
        classify_config_from_params(config).valid_entities | _expected_positive_labels(cases)
    )
    classify_config = classify_config_from_params(config)
    model_env = {"NSS_INFERENCE_MODEL": model} if backend == "api" else {}

    case_results = []
    with patch.dict("os.environ", model_env, clear=False):
        classifier = None
        try:
            classifier = get_column_classifier(config)
            for case in cases:
                df = pd.DataFrame(case["rows"])
                predicted = classifier.detect_types(df, classify_config.valid_entities)
                accuracy = _column_accuracy(predicted, case["expected_entities"])
                positive_recall = _positive_recall(predicted, case["expected_entities"])
                case_results.append(
                    {
                        "case_id": case["id"],
                        "accuracy": accuracy,
                        "positive_recall": positive_recall,
                        "expected": case["expected_entities"],
                        "predicted": {column: _normalize_label(label) for column, label in predicted.items()},
                    }
                )
        finally:
            if classifier is not None:
                classifier.close()

    macro_accuracy = sum(case_result["accuracy"] for case_result in case_results) / len(case_results)
    macro_positive_recall = sum(case_result["positive_recall"] for case_result in case_results) / len(case_results)
    return {
        "backend": backend,
        "model_name": model_name,
        "model": model,
        "status": "ok",
        "macro_accuracy": macro_accuracy,
        "macro_positive_recall": macro_positive_recall,
        "cases": case_results,
    }


def test_column_classification_gold_fixture_schema(pii_test_data_dir):
    cases = _load_gold(pii_test_data_dir / "column_classification_gold.json")
    valid_entities = classify_config_from_params(PiiReplacerConfig.get_default_config()).valid_entities

    assert {case["id"] for case in cases} == {
        "pii_dataset_structured",
        "credit_card_transactions",
        "adult_negative_control",
    }
    for case in cases:
        df = pd.DataFrame(case["rows"])
        expected = case["expected_entities"]
        assert case["source"].startswith("cleaned/")
        assert set(expected) == set(df.columns)
        assert all(isinstance(label, str) and label for label in expected.values())
        assert set(expected.values()) <= valid_entities | {UNKNOWN_ENTITY}


def test_column_accuracy_metric_normalizes_none_labels():
    expected = {"name": "name", "height": UNKNOWN_ENTITY}
    predicted = {"name": "name", "height": None}

    assert _column_accuracy(predicted, expected) == 1.0


def test_column_accuracy_metric_rejects_missing_predictions():
    expected = {"name": "name", "height": UNKNOWN_ENTITY}
    predicted = {"height": None}

    with pytest.raises(AssertionError, match="name"):
        _column_accuracy(predicted, expected)


def test_positive_recall_ignores_negative_columns():
    expected = {
        "name": "name",
        "email": "email",
        "height": UNKNOWN_ENTITY,
    }
    predicted = {
        "name": "name",
        "email": UNKNOWN_ENTITY,
        "height": UNKNOWN_ENTITY,
    }

    assert _positive_recall(predicted, expected) == 0.5


def test_classification_benchmark_matrix_defaults(monkeypatch):
    monkeypatch.delenv("NSS_CLASSIFICATION_BENCHMARK_BACKENDS", raising=False)
    monkeypatch.delenv("NSS_CLASSIFICATION_BENCHMARK_API_MODELS", raising=False)
    monkeypatch.delenv("NSS_CLASSIFICATION_BENCHMARK_LOCAL_MODELS", raising=False)

    assert _benchmark_matrix() == [
        ("api", "default", DefaultLLMConfig.DEFAULT_CONFIG_ID),
        ("local_hf", "smollm3", "HuggingFaceTB/SmolLM3-3B"),
        ("local_hf", "mistral", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("local_hf", "tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ]


def test_benchmark_summary_report_upserts_and_renders_table(tmp_path, monkeypatch):
    monkeypatch.setenv("NSS_CLASSIFICATION_BENCHMARK_OUTPUT", str(tmp_path / "classification-benchmark.json"))
    result = {
        "backend": "local_hf",
        "model_name": "smollm3",
        "model": LOCAL_HF_CLASSIFICATION_BENCHMARK_MODELS["smollm3"],
        "status": "ok",
        "macro_accuracy": 0.5,
        "macro_positive_recall": 0.25,
        "cases": [
            {"case_id": "pii_dataset_structured", "accuracy": 1.0},
            {"case_id": "credit_card_transactions", "accuracy": 0.5},
            {"case_id": "adult_negative_control", "accuracy": 0.0},
        ],
    }

    _write_benchmark_reports(result)
    result["macro_accuracy"] = 0.75
    _write_benchmark_reports(result)

    summary_json = json.loads((tmp_path / "classification-benchmark-summary.json").read_text())
    assert len(summary_json["results"]) == 1
    assert summary_json["results"][0]["macro_accuracy"] == 0.75
    summary_md = (tmp_path / "classification-benchmark-summary.md").read_text()
    assert (
        "| Backend | Model | Macro Accuracy | Macro Positive Recall | pii_dataset_structured | credit_card_transactions | adult_negative_control |"
        in summary_md
    )
    assert "| local_hf | smollm3 | 0.750 | 0.250 | 1.000 | 0.500 | 0.000 |" in summary_md


@pytest.mark.slow
@pytest.mark.parametrize(("backend", "model_name", "model"), _all_benchmark_params())
def test_live_column_classification_model_comparison(pii_test_data_dir, backend, model_name, model):
    """Optional benchmark for one API/local classifier and model-family pair.

    Set ``NSS_RUN_CLASSIFICATION_BENCHMARK=1`` to run. By default this evaluates
    the production API classifier baseline plus local HF SmolLM3, Mistral, and
    TinyLlama as separate pytest cases. Narrow the
    run with comma-separated ``NSS_CLASSIFICATION_BENCHMARK_BACKENDS``,
    ``NSS_CLASSIFICATION_BENCHMARK_API_MODELS``, and
    ``NSS_CLASSIFICATION_BENCHMARK_LOCAL_MODELS``. Each case writes a JSON report
    named ``column-classification-benchmark-{backend}-{model}.json`` by default.
    The test takes a file lock so benchmark cases run serially under xdist.
    """
    if os.environ.get("NSS_RUN_CLASSIFICATION_BENCHMARK") != "1":
        pytest.skip("Set NSS_RUN_CLASSIFICATION_BENCHMARK=1 to run live column classification benchmark.")
    if (backend, model_name, model) not in _benchmark_matrix():
        pytest.skip("Benchmark case excluded by NSS_CLASSIFICATION_BENCHMARK_* filters.")

    with _classification_benchmark_lock():
        cases = _load_gold(pii_test_data_dir / "column_classification_gold.json")
        result = _evaluate_classifier(backend, model_name, model, cases)

        output_path, summary_json_path, summary_md_path = _write_benchmark_reports(result)

        print(f"Column classification benchmark report written to {output_path}")
        print(f"Column classification benchmark summary written to {summary_json_path}")
        print(f"Column classification benchmark table written to {summary_md_path}")
        assert len(result["cases"]) == len(cases)
