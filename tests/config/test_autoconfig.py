# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AutoConfigResolver class methods.

This module tests the automatic configuration resolution for SafeSynthesizer.

Test cases are defined as `AutoConfigTestCase` instances that pair:
- A `SafeSynthesizerParameters` config (the input)
- An `Expected` object (the expected output/behavior)


Each test case explicitly defines the SafeSynthesizerParameters config (input to AutoConfigResolver)
and the Expected values (what AutoConfigResolver should produce)his makes it explicit what config is being tested and what we expect from it.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    SafeSynthesizerParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.config.autoconfig import POW, AutoConfigResolver


@dataclass(frozen=True)
class Expected:
    """Expected output values after AutoConfigResolver runs.

    Fields:
        use_unsloth: Expected resolved value for use_unsloth
        rope_scaling_factor: Expected value (None = will be auto-resolved to an int)
        num_input_records_to_sample: Expected value (None = will be auto-resolved)
        learning_rate: Expected value (None = will be auto-resolved)
        delta: Expected value (None = will be auto-resolved for DP configs)
        dp_enabled: Whether DP is enabled (affects auto-resolution behavior)
        max_seq: Expected resolved value for max_sequences_per_example
        raises: Exception class expected to be raised (None = no error expected)
        raises_match: Regex pattern to match exception message
    """

    use_unsloth: bool
    rope_scaling_factor: int | None  # None = auto-resolved
    num_input_records_to_sample: int | None  # None = auto-resolved
    learning_rate: float | None  # None = auto-resolved
    delta: float | None  # None = auto-resolved or not set
    dp_enabled: bool
    max_seq: int | None  # Expected resolved value
    raises: type[Exception] | None = None
    raises_match: str | None = None

    @property
    def is_auto(self) -> bool:
        """True if this config uses auto-resolution (rope_scaling_factor=None)."""
        return self.rope_scaling_factor is None

    @property
    def should_fail(self) -> bool:
        """True if this config is expected to raise an exception."""
        return self.raises is not None

    @property
    def contextmanager(self) -> AbstractContextManager:
        """Return pytest.raises context if error expected, else nullcontext."""
        if self.raises is not None:
            return pytest.raises(self.raises, match=self.raises_match)
        return nullcontext()


@dataclass(frozen=True)
class AutoConfigTestCase:
    """A complete test case pairing input config with expected output.

    For error cases where Pydantic validation would fail at construction time,
    `config` can be a callable that returns the SafeSynthesizerParameters.
    """

    name: str
    config: SafeSynthesizerParameters | Callable[[], SafeSynthesizerParameters]
    expected: Expected

    def get_config(self) -> SafeSynthesizerParameters:
        """Get the config, calling it if it's a factory function."""
        if callable(self.config):
            return self.config()  # ty: ignore[call-top-callable] -- dynamic callable
        return self.config


AUTO_NO_DP = AutoConfigTestCase(
    name="auto_no_dp",
    config=SafeSynthesizerParameters(
        training=TrainingHyperparams(
            rope_scaling_factor="auto",
            num_input_records_to_sample="auto",
            learning_rate="auto",
            use_unsloth="auto",
        ),
        data=DataParameters(max_sequences_per_example="auto"),
        privacy=DifferentialPrivacyHyperparams(dp_enabled=False, delta="auto"),
    ),
    expected=Expected(
        use_unsloth=True,  # "auto" resolves to True when DP disabled
        rope_scaling_factor=None,  # Will be auto-resolved to an int
        num_input_records_to_sample=None,  # Will be auto-resolved
        learning_rate=None,  # Will be auto-resolved based on model name
        delta=None,  # Not used (DP disabled)
        dp_enabled=False,
        max_seq=10,  # "auto" with no DP -> 10
    ),
)

AUTO_WITH_DP = AutoConfigTestCase(
    name="auto_with_dp",
    config=SafeSynthesizerParameters(
        training=TrainingHyperparams(
            rope_scaling_factor="auto",
            num_input_records_to_sample="auto",
            learning_rate="auto",
            use_unsloth="auto",
        ),
        data=DataParameters(max_sequences_per_example="auto"),
        privacy=DifferentialPrivacyHyperparams(dp_enabled=True, delta="auto"),
    ),
    expected=Expected(
        use_unsloth=False,  # "auto" resolves to False when DP enabled
        rope_scaling_factor=None,  # Will be auto-resolved to an int
        num_input_records_to_sample=None,  # Will be auto-resolved
        learning_rate=None,  # Will be auto-resolved based on model name
        delta=None,  # Will be auto-resolved based on data size
        dp_enabled=True,
        max_seq=1,  # DP enabled -> always 1
    ),
)

EXPLICIT = AutoConfigTestCase(
    name="explicit",
    config=SafeSynthesizerParameters(
        training=TrainingHyperparams(
            rope_scaling_factor=2,
            num_input_records_to_sample=5000,
            learning_rate=0.001,
            use_unsloth=True,
        ),
        data=DataParameters(max_sequences_per_example=3),
        privacy=DifferentialPrivacyHyperparams(dp_enabled=False, delta=0.001),
    ),
    expected=Expected(
        use_unsloth=True,  # Explicit value preserved
        rope_scaling_factor=2,  # Explicit value preserved
        num_input_records_to_sample=5000,  # Explicit value preserved
        learning_rate=0.001,  # Explicit value preserved
        delta=0.001,  # Explicit value preserved
        dp_enabled=False,
        max_seq=3,  # Explicit value preserved
    ),
)

# Error case: DP enabled with use_unsloth=True is invalid
# Config is a lambda because Pydantic validation would fail at module load time
DP_WITH_UNSLOTH_TRUE = AutoConfigTestCase(
    name="dp_with_unsloth_true",
    config=lambda: SafeSynthesizerParameters(
        training=TrainingHyperparams(
            rope_scaling_factor="auto",
            num_input_records_to_sample="auto",
            learning_rate="auto",
            use_unsloth=True,  # Invalid: explicit True with DP
        ),
        data=DataParameters(max_sequences_per_example="auto"),
        privacy=DifferentialPrivacyHyperparams(dp_enabled=True, delta="auto"),
    ),
    expected=Expected(
        use_unsloth=True,  # This causes the error
        rope_scaling_factor=None,
        num_input_records_to_sample=None,
        learning_rate=None,
        delta=None,
        dp_enabled=True,
        max_seq=1,
        raises=Exception,
        raises_match="Unsloth is currently not compatible with DP|not compatible with DP",
    ),
)


ALL_TEST_CASES: list[AutoConfigTestCase] = [
    AUTO_NO_DP,
    AUTO_WITH_DP,
    EXPLICIT,
    DP_WITH_UNSLOTH_TRUE,
]

# Valid test cases (configs that should pass) for standard tests
VALID_TEST_CASES = [tc for tc in ALL_TEST_CASES if not tc.expected.should_fail]

# Error cases that should fail validation
ERROR_TEST_CASES = [tc for tc in ALL_TEST_CASES if tc.expected.should_fail]


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Standard test DataFrame (100 rows)."""
    return pd.DataFrame({"col_a": range(100), "col_b": ["text"] * 100})


@pytest.fixture(params=[1000, 10000, 50000], ids=lambda n: f"{n}_rows")
def variable_data(request) -> pd.DataFrame:
    """DataFrame with variable row counts for testing data-size dependent behavior."""
    n = request.param
    return pd.DataFrame({"col_a": range(n), "col_b": ["text"] * n})


@pytest.fixture(params=VALID_TEST_CASES, ids=lambda tc: tc.name)
def test_case(request) -> AutoConfigTestCase:
    """Parametrized fixture providing each valid test case."""
    return request.param


@pytest.fixture
def config(test_case: AutoConfigTestCase) -> SafeSynthesizerParameters:
    """Extract config from test_case for convenience."""
    return test_case.get_config()


@pytest.fixture
def expected(test_case: AutoConfigTestCase) -> Expected:
    """Extract expected from test_case for convenience."""
    return test_case.expected


class TestAutoConfigResolver:
    """Tests for AutoConfigResolver class methods.

    Tests are organized by the method under test:
    - _determine_* methods: Test individual auto-resolution logic
    - resolve(): Test full resolution pipeline including expected failures
    """

    def test_determine_rope_scaling_factor(self, sample_data, config, expected):
        """Rope scaling factor should be auto-resolved for auto configs, unchanged for explicit."""
        resolver = AutoConfigResolver(sample_data, config)
        result = resolver._determine_rope_scaling_factor()

        if expected.is_auto:
            assert "rope_scaling_factor" in result
            assert isinstance(result["rope_scaling_factor"], int)
            assert result["rope_scaling_factor"] >= 1
            assert resolver._rope_scaling_factor == result["rope_scaling_factor"]
        else:
            assert result == {}
            assert resolver._rope_scaling_factor is None

    @pytest.mark.parametrize(
        "rope_override, expected_multiplier",
        [
            pytest.param(None, 1, id="default_rope"),
            pytest.param(3, 3, id="rope_3x"),
        ],
    )
    def test_determine_num_input_records_to_sample(
        self, sample_data, config, expected, rope_override, expected_multiplier
    ):
        """Num input records scales with rope_scaling_factor for auto configs."""
        resolver = AutoConfigResolver(sample_data, config)
        if rope_override is not None:
            resolver._rope_scaling_factor = rope_override

        result = resolver._determine_num_input_records_to_sample()

        if expected.is_auto:
            assert result == {"num_input_records_to_sample": 25_000 * expected_multiplier}
        else:
            assert result == {}

    @pytest.mark.parametrize(
        "pretrained_model, expected_lr",
        [
            pytest.param(None, 0.0005, id="default_model"),
            pytest.param("HuggingFaceTB/SmolLM3-3B", 0.0005, id="smollm"),
            pytest.param("mistralai/Mistral-7B-Instruct-v0.3", 0.0001, id="mistral"),
        ],
    )
    def test_determine_learning_rate(self, sample_data, config, expected, pretrained_model, expected_lr):
        """Learning rate is auto configured with pretrained model → 0.0001 for Mistral, 0.0005 otherwise"""
        config_copy = config
        if pretrained_model is not None:
            config_copy = config.model_copy(deep=True)
            config_copy.training.pretrained_model = pretrained_model

        resolver = AutoConfigResolver(sample_data, config_copy)
        result = resolver._determine_learning_rate()

        if expected.is_auto:
            assert result == {"learning_rate": expected_lr}
        else:
            assert result == {}

    def test_determine_use_unsloth(self, sample_data, config, expected):
        """Use_unsloth should be False with DP, True without, or unchanged for explicit."""
        resolver = AutoConfigResolver(sample_data, config)
        result = resolver._determine_use_unsloth()

        if expected.is_auto:
            # Auto configs have use_unsloth="auto" which gets resolved
            assert result == {"use_unsloth": expected.use_unsloth}
        else:
            # Explicit configs don't change use_unsloth
            assert result == {}

    @pytest.mark.parametrize("data_size", [50, 1000, 10000], ids=lambda n: f"{n}_rows")
    def test_determine_delta(self, data_size, test_case, config, expected):
        """Delta should be auto-calculated for DP configs based on data size."""
        data = pd.DataFrame({"col_a": range(data_size), "col_b": ["text"] * data_size})
        resolver = AutoConfigResolver(data, config)
        result = resolver._determine_delta()

        if expected.dp_enabled and expected.is_auto:
            assert "delta" in result
            # Delta formula depends on data size
            if data_size >= 100:
                assert result["delta"] == pytest.approx(1 / (data_size**POW))
            else:
                assert result["delta"] == pytest.approx(0.1 / data_size)
        else:
            assert result == {}

    @pytest.mark.parametrize(
        "max_seq_input, expected_max_seq",
        [
            pytest.param("auto", [1, 10], id="auto_max_seq"),  # dp_enabled=True -> 1, dp_enabled=False -> 10
            pytest.param(5, [1, 5], id="explicit_max_seq"),  # dp_enabled=True -> 1, dp_enabled=False -> 5
            pytest.param(None, [1, None], id="none_max_seq"),  # dp_enabled=True -> 1, dp_enabled=False -> None
        ],
    )
    def test_determine_max_sequences_per_example(self, sample_data, config, max_seq_input, expected_max_seq):
        """Max sequences should be 1 for DP regardless of input; for non-DP, auto -> 10, explicit -> explicit value, None -> None."""
        config_copy = config.model_copy(deep=True)
        config_copy.data.max_sequences_per_example = max_seq_input
        resolver = AutoConfigResolver(sample_data, config_copy)
        result = resolver._determine_max_sequences_per_example()

        expected_index = 0 if config_copy.privacy.dp_enabled else 1
        assert result == {"max_sequences_per_example": expected_max_seq[expected_index]}

    def test_resolve(self, sample_data, config, expected):
        """Full resolution should produce valid SafeSynthesizerParameters.

        Parametrized via `test_case` fixture over VALID_TEST_CASES, which provides
        the `config` and `expected` fixtures for each test case.
        """
        resolver = AutoConfigResolver(sample_data, config)
        result = resolver()

        assert isinstance(result, SafeSynthesizerParameters)

        if expected.is_auto:
            assert isinstance(result.training.rope_scaling_factor, int)
            assert isinstance(result.training.num_input_records_to_sample, int)
            assert isinstance(result.training.learning_rate, float)
            assert result.training.use_unsloth is expected.use_unsloth
            assert result.data.max_sequences_per_example == expected.max_seq
            if expected.dp_enabled:
                assert result.privacy and isinstance(result.privacy.delta, float)
        else:
            assert result.training.rope_scaling_factor == expected.rope_scaling_factor
            assert result.training.num_input_records_to_sample == expected.num_input_records_to_sample
            assert result.training.learning_rate == expected.learning_rate
            assert result.training.use_unsloth is expected.use_unsloth
            assert result.data.max_sequences_per_example == expected.max_seq
            assert result.privacy and result.privacy.delta == expected.delta

    @pytest.mark.parametrize(
        "error_case",
        ERROR_TEST_CASES,
        ids=lambda tc: tc.name,
    )
    def test_invalid_config_combinations(self, sample_data, error_case: AutoConfigTestCase):
        """Test that invalid config combinations raise expected errors. This is done mostly to semantically separate the expected-to-fail cases
        from the above resolve parametrization.
        """
        with error_case.expected.contextmanager:
            config = error_case.get_config()
            resolver = AutoConfigResolver(sample_data, config)
            resolver()

    def test_resolve_with_variable_data_sizes(self, variable_data, config, expected):
        """Resolution should handle various data sizes correctly."""
        resolver = AutoConfigResolver(variable_data, config)
        result = resolver()

        assert isinstance(result, SafeSynthesizerParameters)
        if expected.is_auto:
            assert isinstance(result.training.rope_scaling_factor, int)
