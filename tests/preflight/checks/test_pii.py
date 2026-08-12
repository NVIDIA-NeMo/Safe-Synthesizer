# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``preflight.checks.pii`` (config + plan validity)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import (
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementSettings,
    PiiReplacerConfig,
)
from nemo_safe_synthesizer.preflight import PreflightContext
from nemo_safe_synthesizer.preflight.checks.pii import (
    PiiPlanValidityCheck,
    PiiReplacementConfigCheck,
)
from tests.pii_replacer.helpers import PgmCheckout, pgm_checkout


def test_preflight_faker_locale_error():
    config = SafeSynthesizerParameters(
        replace_pii=PiiReplacerConfig(
            replacement=PiiReplacementSettings(locale="not_a_real_locale"),
            person=PiiPersonConfig(backend=PiiPersonBackend.faker),
        )
    )
    issues = PiiReplacementConfigCheck().run(PreflightContext(config=config, data=pd.DataFrame(), metadata=MagicMock()))
    assert any(i.code == "pii_faker_locale_invalid" for i in issues)


@pytest.mark.parametrize("locale", ["en_SG", "hi_Deva_IN", "hi_Latn_IN"])
def test_preflight_managed_locale_warns_when_faker_needs_fallback(locale: str):
    """Managed locales without Faker providers must not silently pass then crash at apply."""
    config = SafeSynthesizerParameters(
        replace_pii=PiiReplacerConfig(
            replacement=PiiReplacementSettings(locale=locale),
            person=PiiPersonConfig(backend=PiiPersonBackend.managed),
        )
    )
    issues = PiiReplacementConfigCheck().run(PreflightContext(config=config, data=pd.DataFrame(), metadata=MagicMock()))
    assert any(i.code == "pii_managed_faker_locale_fallback" for i in issues)
    assert not any(i.code == "pii_faker_locale_invalid" for i in issues)


def _pgm_preflight_issues(locale: str, src: Path):
    config = SafeSynthesizerParameters(
        replace_pii=PiiReplacerConfig(
            replacement=PiiReplacementSettings(locale=locale),
            person=PiiPersonConfig(backend=PiiPersonBackend.pgm, sdg_pgms_src=str(src)),
        )
    )
    return PiiReplacementConfigCheck().run(PreflightContext(config=config, data=pd.DataFrame(), metadata=MagicMock()))


@pytest.mark.parametrize(
    ("locale", "checkout", "code"),
    [
        pytest.param("ja_JP", "complete", "pii_pgm_locale_invalid", id="unsupported_locale"),
        pytest.param("en_US", "absent", "pii_pgm_src_missing", id="missing_source"),
        pytest.param("en_US", "without_package", "pii_pgm_import_missing", id="source_without_package"),
    ],
)
def test_preflight_reports_an_unusable_pgm_backend(locale: str, checkout: PgmCheckout, code: str, tmp_path: Path):
    """The PGM never falls back, so pre-flight must fail rather than warn."""
    issues = _pgm_preflight_issues(locale, pgm_checkout(tmp_path, checkout))
    issue = next(i for i in issues if i.code == code)
    assert issue.severity == "error"


def test_preflight_reports_a_pgm_source_it_cannot_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A checkout the run cannot stat is an issue to report, not one to raise."""
    src = pgm_checkout(tmp_path, "complete")
    readable_is_dir = Path.is_dir

    def deny(self: Path) -> bool:
        if self == src:
            raise PermissionError(13, "Permission denied", str(self))
        return readable_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", deny)
    assert [i.code for i in _pgm_preflight_issues("en_US", src)] == ["pii_pgm_src_missing"]


def _plan_validity_issues(config: SafeSynthesizerParameters, df: pd.DataFrame):
    return PiiPlanValidityCheck().run(PreflightContext(config=config, data=df, metadata=MagicMock()))


def test_preflight_plan_validity_reports_user_plan_errors(fixture_dob_df: pd.DataFrame):
    """A hand-written plan fails preflight instead of waiting for replacement to start."""
    plan = PiiReplacementPlan.model_validate(
        {
            "standalone_columns_to_replace": [
                {"column_name": "not_a_column", "entity_type": "unique_identifier"},
                {"column_name": "date_of_birth", "entity_type": "date_of_birth", "patterns": ["%Y-%m"]},
            ]
        }
    )
    config = SafeSynthesizerParameters(replace_pii=PiiReplacerConfig(replacement_plan=plan))
    issues = _plan_validity_issues(config, fixture_dob_df)
    assert {i.code for i in issues} == {"pii_plan_column_not_found", "pii_plan_pattern_invalid"}
    assert all(i.severity == "error" for i in issues)


def test_preflight_plan_validity_accepts_a_usable_plan(fixture_dob_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[PiiColumnPlan(column_name="patient_id", entity_type=PiiEntity.unique_identifier)]
    )
    config = SafeSynthesizerParameters(replace_pii=PiiReplacerConfig(replacement_plan=plan))
    assert _plan_validity_issues(config, fixture_dob_df) == []


def test_preflight_plan_validity_warns_on_section_mismatches(fixture_dob_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
                ],
            ),
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
        ],
    )
    config = SafeSynthesizerParameters(replace_pii=PiiReplacerConfig(replacement_plan=plan))
    issues = _plan_validity_issues(config, fixture_dob_df)
    assert {i.code: i.severity for i in issues} == {
        "pii_plan_free_text_under_persona": "warning",
        "pii_plan_persona_column_under_standalone": "warning",
    }


def test_preflight_plan_validity_skips_auto_discovery(fixture_dob_df: pd.DataFrame):
    """Discovery builds its plan from this dataframe, so there is nothing to pre-check."""
    config = SafeSynthesizerParameters(replace_pii=PiiReplacerConfig())
    assert config.replace_pii is not None and config.replace_pii.is_auto_discovery
    assert _plan_validity_issues(config, fixture_dob_df) == []


def test_preflight_plan_validity_reports_unreadable_plan_file(fixture_dob_df: pd.DataFrame, tmp_path):
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("persona_backed_columns: [oops\n")
    config = SafeSynthesizerParameters(replace_pii=PiiReplacerConfig(replacement_plan=str(plan_path)))
    issues = _plan_validity_issues(config, fixture_dob_df)
    assert [i.code for i in issues] == ["pii_plan_unreadable"]


def test_preflight_llm_enhancement_is_error():
    # Bypass config validation so preflight can still report the not-implemented flag
    # if an instance somehow carries llm_enhancement=True.
    config = SafeSynthesizerParameters(
        replace_pii=PiiReplacerConfig.model_construct(
            llm_enhancement=True, person=PiiPersonConfig(backend=PiiPersonBackend.faker)
        )
    )
    issues = PiiReplacementConfigCheck().run(PreflightContext(config=config, data=pd.DataFrame(), metadata=MagicMock()))
    issue = next(i for i in issues if i.code == "pii_llm_not_implemented")
    assert issue.severity == "error"


def test_preflight_rejects_user_plan_with_protected_column():
    from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters

    n = 30
    df = pd.DataFrame(
        {
            "patient_id": [f"pmc-6{i:05d}-1" for i in range(n)],
            "seq_id": [f"S{i:05d}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
        }
    )
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="seq_id", entity_type=PiiEntity.unique_identifier),
        ]
    )
    config = SafeSynthesizerParameters(
        data=DataParameters(
            group_training_examples_by="patient_id",
            order_training_examples_by="seq_id",
        ),
        time_series=TimeSeriesParameters(is_timeseries=False),
        replace_pii=PiiReplacerConfig(replacement_plan=plan),
    )
    issues = _plan_validity_issues(config, df)
    assert {i.code for i in issues} == {"pii_plan_protected_column"}
