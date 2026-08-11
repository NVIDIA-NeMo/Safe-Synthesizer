# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan validation, advisories, and load/save error wrapping."""

from __future__ import annotations

import re

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementScope,
    PiiReplacerConfig,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import (
    iter_plan_advisories,
    iter_plan_issues,
    validate_plan,
)
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
)


def test_tabular_pii_replacer_rejects_date_entity_plan(fixture_dob_df: pd.DataFrame):
    """Public path refuses identify-only ``date``; it never reaches replacement passthrough."""
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="date_of_birth", entity_type=PiiEntity.date),
        ],
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan),
        data_config=DataParameters(),
    )
    with pytest.raises(ParameterError, match="only identified, never replaced"):
        replacer.transform_df(fixture_dob_df)


def test_validate_plan_group_scope_requires_training_group_key(fixture_patient_df: pd.DataFrame):
    plan = PiiReplacementPlan(scope=PiiReplacementScope.group)
    with pytest.raises(ParameterError, match="group_training_examples_by"):
        validate_plan(fixture_patient_df, plan, data_config=DataParameters())


def test_replacement_plan_reports_plan_errors_without_union_noise():
    """A malformed inline plan names its own bad field, not the string half of the union."""
    with pytest.raises(ValidationError) as excinfo:
        PiiReplacerConfig.model_validate(
            {"replacement_plan": {"standalone_columns_to_replace": [{"entity_type": "date_of_birth"}]}}
        )
    message = str(excinfo.value)
    assert "standalone_columns_to_replace.0.column_name" in message
    assert "valid string" not in message


def test_replacement_plan_rejects_values_that_are_neither_plan_nor_string():
    with pytest.raises(ValidationError, match="must be 'auto_discovery', a path to a plan file, or an inline plan"):
        PiiReplacerConfig.model_validate({"replacement_plan": 42})


@pytest.mark.parametrize(
    ("plan_data", "message"),
    [
        pytest.param(
            {
                "standalone_columns_to_replace": [
                    {"column_name": "date_of_birth", "entity_type": "date_of_birth", "patterns": ["%Y-%m"]}
                ]
            },
            "does not parse any of its values",
            id="date_pattern_matches_nothing",
        ),
        pytest.param(
            {
                "standalone_columns_to_replace": [
                    {
                        "column_name": "date_of_birth",
                        "entity_type": "date_of_birth",
                        "patterns": ["%m/%d/%Y", "%m/%d/%Y"],
                    }
                ]
            },
            "more than once",
            id="duplicated_pattern",
        ),
        pytest.param(
            {
                "standalone_columns_to_replace": [
                    {"column_name": "date_of_birth", "entity_type": "date_of_birth", "patterns": ["%m/%d/%Y", ""]}
                ]
            },
            "lists an empty pattern",
            id="empty_pattern",
        ),
        pytest.param(
            {"standalone_columns_to_replace": [{"column_name": "first_name"}]},
            "has no entity_type",
            id="missing_entity_type",
        ),
        pytest.param(
            {"standalone_columns_to_replace": [{"column_name": "date_of_birth", "entity_type": "date"}]},
            "only identified, never replaced",
            id="identify_only_date_entity",
        ),
        pytest.param(
            {
                "standalone_columns_to_replace": [
                    {"column_name": "patient_id", "entity_type": "unique_identifier", "patterns": ["pmc-fixed-1"]}
                ]
            },
            "no variable position",
            id="constant_identifier_template",
        ),
        pytest.param(
            {
                "standalone_columns_to_replace": [
                    {"column_name": "patient_id", "entity_type": "unique_identifier", "patterns": ["###-###"]}
                ]
            },
            "matches none of its values",
            id="identifier_template_matches_nothing",
        ),
        pytest.param(
            {
                "standalone_columns_to_replace": [
                    {"column_name": "notes", "entity_type": "free_text", "patterns": ["%m/%d/%Y"]}
                ]
            },
            "ignores patterns",
            id="pattern_on_free_text",
        ),
        pytest.param(
            {
                "persona_backed_columns": [
                    {
                        "persona": "patient",
                        "columns_to_replace": [{"column_name": "sex", "entity_type": "first_name"}],
                        "match_persona_by": [{"persona_attribute": "gender", "column_name": "sex"}],
                    }
                ]
            },
            "never replaced",
            id="match_persona_by_column_also_replaced",
        ),
    ],
)
def test_validate_plan_rejects_unusable_column_specs(fixture_dob_df: pd.DataFrame, plan_data: dict, message: str):
    plan = PiiReplacementPlan.model_validate(plan_data)
    with pytest.raises(ParameterError, match=message):
        validate_plan(fixture_dob_df, plan, data_config=DataParameters())


@pytest.mark.parametrize(
    ("plan_data", "message"),
    [
        pytest.param(
            {
                "persona_backed_columns": [
                    {
                        "persona": "patient",
                        "columns_to_replace": [{"column_name": "first_name", "entity_type": "first_name"}],
                    }
                ],
                "standalone_columns_to_replace": [
                    {"column_name": "first_name", "entity_type": "first_name"},
                ],
            },
            "appears more than once in replacement plan",
            id="duplicate_column",
        ),
        pytest.param(
            {
                "persona_backed_columns": [
                    {
                        "persona": "patient",
                        "columns_to_replace": [{"column_name": "first_name", "entity_type": "first_name"}],
                    },
                    {
                        "persona": "patient",
                        "columns_to_replace": [{"column_name": "notes", "entity_type": "free_text"}],
                    },
                ],
            },
            "persona 'patient' appears more than once",
            id="duplicate_persona",
        ),
        pytest.param(
            {
                "persona_backed_columns": [
                    {
                        "persona": "patient",
                        "columns_to_replace": [{"column_name": "first_name", "entity_type": "first_name"}],
                        "match_persona_by": [
                            {"persona_attribute": "gender", "column_name": "sex"},
                            {"persona_attribute": "gender", "column_name": "sex"},
                        ],
                    }
                ],
            },
            "persona_attribute 'gender' appears more than once",
            id="duplicate_match_attribute",
        ),
    ],
)
def test_validate_plan_rejects_duplicate_plan_entries(fixture_dob_df: pd.DataFrame, plan_data: dict, message: str):
    plan = PiiReplacementPlan.model_validate(plan_data)
    with pytest.raises(ParameterError, match=re.escape(message)):
        validate_plan(fixture_dob_df, plan, data_config=DataParameters())


def test_iter_plan_issues_reports_every_problem_at_once(fixture_dob_df: pd.DataFrame):
    """Preflight shows a hand-edited plan's mistakes together, not one per run."""
    plan = PiiReplacementPlan.model_validate(
        {
            "persona_backed_columns": [
                {
                    "persona": "patient",
                    "columns_to_replace": [{"column_name": "first_name", "entity_type": "first_name"}],
                    "match_persona_by": [{"persona_attribute": "gender", "column_name": "not_a_column"}],
                }
            ],
            "standalone_columns_to_replace": [
                {"column_name": "missing_column", "entity_type": "unique_identifier"},
                {"column_name": "date_of_birth", "entity_type": "date_of_birth", "patterns": ["%Y-%m"]},
                {"column_name": "notes"},
            ],
        }
    )
    issues = list(iter_plan_issues(fixture_dob_df, plan, data_config=DataParameters()))
    assert [i.code for i in issues] == [
        "pii_plan_column_not_found",
        "pii_plan_pattern_invalid",
        "pii_plan_entity_type_invalid",
        "pii_plan_column_not_found",
    ]


def test_iter_plan_issues_skips_pattern_sampling_for_absent_columns(fixture_dob_df: pd.DataFrame):
    """A pattern on a missing column must report the column, not raise a KeyError."""
    plan = PiiReplacementPlan.model_validate(
        {
            "standalone_columns_to_replace": [
                {"column_name": "not_a_column", "entity_type": "date_of_birth", "patterns": ["%Y-%m"]}
            ]
        }
    )
    assert [i.code for i in iter_plan_issues(fixture_dob_df, plan)] == ["pii_plan_column_not_found"]


def test_iter_plan_advisories_flag_section_mismatches(fixture_dob_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
                    # Standalone-mapped under persona is silent: generation path is unchanged.
                    PiiColumnPlan(column_name="date_of_birth", entity_type=PiiEntity.date_of_birth),
                    PiiColumnPlan(column_name="patient_id", entity_type=PiiEntity.unique_identifier),
                ],
            ),
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
        ],
    )
    codes = {issue.code for issue in iter_plan_advisories(plan, persona_backend="managed")}
    assert codes == {
        "pii_plan_free_text_under_persona",
        "pii_plan_persona_column_under_standalone",
    }
    # Errors stay separate; mis-sectioning never fails validation.
    assert list(iter_plan_issues(fixture_dob_df, plan)) == []
    validate_plan(fixture_dob_df, plan, data_config=DataParameters())


def test_iter_plan_advisories_phone_under_standalone_depends_on_backend(fixture_dob_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.phone_number),
        ],
    )
    managed = [i.code for i in iter_plan_advisories(plan, persona_backend="managed")]
    pgm = [i.code for i in iter_plan_advisories(plan, persona_backend="pgm")]
    assert managed == []
    assert pgm == ["pii_plan_persona_column_under_standalone"]


def test_iter_plan_advisories_silent_for_well_placed_columns(fixture_dob_df: pd.DataFrame):
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            ),
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="patient_id", entity_type=PiiEntity.unique_identifier),
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
            PiiColumnPlan(column_name="date_of_birth", entity_type=PiiEntity.date_of_birth),
        ],
    )
    assert list(iter_plan_advisories(plan, persona_backend="managed")) == []


def test_pii_replacement_logs_section_placement_warnings(fixture_patient_df: pd.DataFrame, caplog):
    import logging

    caplog.set_level(logging.WARNING)
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
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    replacer.transform_df(fixture_patient_df)
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "free_text" in messages
    assert "standalone_columns_to_replace lists 'first_name'" in messages


@pytest.mark.parametrize(
    ("entity", "pattern", "message"),
    [
        (PiiEntity.full_name, "{surname}, {First}", "names no part of a person"),
        (PiiEntity.full_name, "Anonymous", "names no part of a person"),
        (PiiEntity.email, "{f}.{last}", "no '@' and its domain"),
        (PiiEntity.email, "{domain}.{last}@example.com", "before the '@'"),
        (PiiEntity.email, "support@{domain}", "no variable position before the '@'"),
    ],
    ids=["unknown_part", "no_part", "no_domain", "domain_in_local", "constant_local"],
)
def test_validate_plan_reads_a_broken_persona_pattern(entity: PiiEntity, pattern: str, message: str):
    df = pd.DataFrame({"who": [f"Person {i}" for i in range(5)]})
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person",
                columns_to_replace=[PiiColumnPlan(column_name="who", entity_type=entity, patterns=[pattern])],
            )
        ]
    )

    with pytest.raises(ParameterError, match=message):
        validate_plan(df, plan, data_config=DataParameters())


@pytest.mark.parametrize(
    ("entity", "values"),
    [
        pytest.param(PiiEntity.ipv4, [f"10.0.0.{i}" for i in range(20)], id="ipv4"),
        pytest.param(
            PiiEntity.ipv6,
            [f"2001:db8::{i:x}" for i in range(20)],
            id="ipv6",
        ),
    ],
)
def test_a_plan_may_not_shape_an_ip_column(entity: PiiEntity, values: list[str]):
    """A template counts characters, so it cannot promise a valid address."""
    df = pd.DataFrame({"ip_address": values})
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="ip_address", entity_type=entity, patterns=["##.#.#.###"])
        ]
    )

    with pytest.raises(ParameterError, match="octet of at most 255"):
        validate_plan(df, plan, data_config=DataParameters())


@pytest.mark.parametrize(
    ("entity", "column"),
    [
        (PiiEntity.ssn, "ssn"),
        (PiiEntity.national_id, "national_id"),
        (PiiEntity.street_address, "street"),
    ],
)
def test_a_plan_may_not_list_patterns_a_field_ignores(entity: PiiEntity, column: str):
    """SSN, national ID and street do not follow listed character formats."""
    df = pd.DataFrame({column: [f"value-{i}" for i in range(5)]})
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person",
                columns_to_replace=[PiiColumnPlan(column_name=column, entity_type=entity, patterns=["###-##-####"])],
            )
        ]
    )

    with pytest.raises(ParameterError, match="does not follow a listed format"):
        validate_plan(df, plan, data_config=DataParameters())


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        pytest.param("+1-415-555-0100", "no variable position", id="constant_template"),
        pytest.param("###-####", "matches none of its values", id="describes_no_value"),
    ],
)
def test_validate_plan_rejects_unusable_phone_patterns(fixture_phone_df: pd.DataFrame, pattern: str, message: str):
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="phone", entity_type=PiiEntity.phone_number, patterns=[pattern])
        ]
    )
    with pytest.raises(ParameterError, match=message):
        validate_plan(fixture_phone_df, plan, data_config=DataParameters())


def test_a_plan_may_list_only_some_of_a_columns_formats(fixture_dob_df: pd.DataFrame):
    """The values the list misses keep their own format, which is no mistake to report."""
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(
                column_name="date_of_birth",
                entity_type=PiiEntity.date_of_birth,
                patterns=["%m/%d/%Y"],  # the column also holds one ISO date
            )
        ]
    )
    validate_plan(fixture_dob_df, plan, data_config=DataParameters())


def test_a_plan_may_leave_the_local_part_to_the_person():
    """'@{domain}' is the convention of a column that has none: keep the domain, no more."""
    df = pd.DataFrame({"contact": [f"usr{i}x@acme.com" for i in range(5)]})
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person",
                columns_to_replace=[
                    PiiColumnPlan(column_name="contact", entity_type=PiiEntity.email, patterns=["@{domain}"])
                ],
            )
        ]
    )

    validate_plan(df, plan, data_config=DataParameters())


def test_load_plan_from_path_wraps_io_yaml_and_validation_errors(tmp_path):
    import yaml

    from nemo_safe_synthesizer.pii_replacer.planning import load_plan_from_path

    missing = tmp_path / "missing.yaml"
    with pytest.raises(ParameterError, match=r"Could not read PII replacement plan file") as missing_exc:
        load_plan_from_path(str(missing))
    assert isinstance(missing_exc.value.__cause__, OSError)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("persona_backed_columns: [oops\n")
    with pytest.raises(ParameterError, match=r"Invalid YAML in PII replacement plan file") as yaml_exc:
        load_plan_from_path(str(bad_yaml))
    assert isinstance(yaml_exc.value.__cause__, yaml.YAMLError)

    bad_plan = tmp_path / "invalid.yaml"
    bad_plan.write_text("scope: not_a_real_scope\n")
    with pytest.raises(ParameterError, match=r"Invalid PII replacement plan") as plan_exc:
        load_plan_from_path(str(bad_plan))
    assert isinstance(plan_exc.value.__cause__, ValidationError)


def test_hand_plan_still_replaces_oddly_named_email_column():
    """Hand-written plans remain the escape hatch for oddly named columns."""
    n = 20
    df = pd.DataFrame({"contact": [f"user{i}@example.com" for i in range(n)]})
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person_1",
                columns_to_replace=[PiiColumnPlan(column_name="contact", entity_type=PiiEntity.email)],
            )
        ]
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    assert list(replacer.result.transformed_df["contact"]) != list(df["contact"])
    assert all("@" in v for v in replacer.result.transformed_df["contact"])
