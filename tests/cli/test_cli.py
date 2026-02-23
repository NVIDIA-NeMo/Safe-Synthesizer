# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re

from click.testing import CliRunner

from nemo_safe_synthesizer.cli.cli import cli
from nemo_safe_synthesizer.utils import merge_dicts


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def test_merge_dicts():
    test_data_base = {
        "a": 1,
        "b": {"c": 1, "d": 2},
        "c": {"d": {"e": 0, "f": 1, "p": {"q": 4}}},
        "x": 0,
        "y": {"x": 3},
    }

    test_data_update = {
        "a": 9,
        "b": {"d": 3, "e": 3},
        "c": {"d": {"e": 1, "g": 8, "p": {"r": 5, "s": 6}}, "h": 7},
        "d": 6,
        "e": {"f": 10, "g": 10},
    }

    test_expected_updated_data = {
        "a": 9,
        "b": {"c": 1, "d": 3, "e": 3},
        "c": {"d": {"e": 1, "f": 1, "p": {"q": 4, "r": 5, "s": 6}, "g": 8}, "h": 7},
        "d": 6,
        "e": {"f": 10, "g": 10},
        "x": 0,
        "y": {"x": 3},
    }

    import copy

    test_data_base_copy = copy.deepcopy(test_data_base)
    test_actual_updated_data = merge_dicts(test_data_base, test_data_update)
    assert test_actual_updated_data == test_expected_updated_data
    assert test_data_base == test_data_base_copy


def test_config_keep_default_with_nested_replacement(tmp_path_factory, yaml_config_str):
    """
    Note these tests need to be run without extra pytest capturing
    to see the print statements, e.g. pytest -s or pytest --log-cli-level=DEBUG
    """
    tmp_path = tmp_path_factory.mktemp("data", numbered=True)
    config_path = tmp_path / "test_safe_synth.yaml"
    with open(config_path, "w") as f:
        f.write(yaml_config_str)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "config",
            "modify",
            "--config",
            config_path,
            "--training__rope_scaling_factor",
            "3",
            "--data__group_training_examples_by",
            "something",
        ],
        color=False,
    )
    print(result.stdout)
    clean_output = strip_ansi_codes(result.stdout)
    vals = json.loads(clean_output)  # ensure valid json
    assert result.exit_code == 0
    assert vals["training"]["rope_scaling_factor"] == 3
    assert vals["training"]["weight_decay"] == 0.01
    assert vals["data"]["group_training_examples_by"] == "something"
    assert vals["data"]["max_sequences_per_example"] == 2


def test_config_accurate_replacement(tmp_path_factory, yaml_config_str):
    """
    Note these tests need to be ran without extra pytest capturing
    to see the print statements, e.g. pytest -s or pytest --log-cli-level=DEBUG
    """
    tmp_path = tmp_path_factory.mktemp("data", numbered=True)
    config_path = tmp_path / "test_safe_synth.yaml"
    with open(config_path, "w") as f:
        f.write(yaml_config_str)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "config",
            "modify",
            "--config",
            config_path,
            "--training__rope_scaling_factor",
            "3",
            "--data__max_sequences_per_example",
            "5",
            "--data__group_training_examples_by",
            "something",
        ],
        color=False,
    )
    print(result.stdout)
    clean_output = strip_ansi_codes(result.stdout)
    vals = json.loads(clean_output)  # ensure valid json
    assert vals["training"]["rope_scaling_factor"] == 3
    assert vals["data"]["max_sequences_per_example"] == 5
    assert vals["data"]["group_training_examples_by"] == "something"
    assert result.exit_code == 0


def test_config_accurate_replacement_multiples(tmp_path_factory, yaml_config_str):
    """
    Note these tests need to be ran without extra pytest capturing
    to see the print statements, e.g. pytest -s or pytest --log-cli-level=DEBUG
    """
    tmp_path = tmp_path_factory.mktemp("data", True)
    config_path = tmp_path / "test_safe_synth.yaml"
    with open(config_path, "w") as f:
        f.write(yaml_config_str)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "config",
            "modify",
            "--config",
            config_path,
            "--data__group_training_examples_by",
            "something",
            "--data__order_training_examples_by",
            "else",
        ],
        color=False,
    )
    print(result.stdout)
    clean_output = strip_ansi_codes(result.stdout)
    vals = json.loads(clean_output)  # ensure valid json
    assert vals["training"]["rope_scaling_factor"] == "auto"
    assert vals["data"]["group_training_examples_by"] == "something"
    assert vals["data"]["order_training_examples_by"] == "else"
    assert result.exit_code == 0
