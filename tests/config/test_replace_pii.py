# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ALLOWED_DEPENDS_ON,
    AUTO_DISCOVERY,
    ENTITIES,
    ENTITY_BY_TYPE,
    ConditioningColumn,
    EntityAction,
    EntityType,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementScope,
    PiiSamplerBackend,
    ReplacePiiConfig,
    can_condition,
    is_columns_to_replace_type,
)
from nemo_safe_synthesizer.defaults import NSS_MANAGED_ASSETS_PATH_ENV, default_managed_assets_path
from nemo_safe_synthesizer.errors import ParameterError


def _raises(match: str):
    return pytest.raises((ParameterError, ValidationError), match=match)


@pytest.mark.unit
class TestEntityCatalog:
    def test_every_entity_type_has_a_catalog_entry(self) -> None:
        assert set(ENTITY_BY_TYPE) == set(EntityType)
        assert len(ENTITIES) == len(EntityType)

    def test_replace_and_replace_in_text_may_appear_on_columns_to_replace(self) -> None:
        assert is_columns_to_replace_type(EntityType.FIRST_NAME)
        assert is_columns_to_replace_type(EntityType.FREE_TEXT)
        assert not is_columns_to_replace_type(EntityType.GENDER)
        assert not is_columns_to_replace_type(EntityType.DATE)

    def test_conditioners_match_can_condition_flag(self) -> None:
        assert can_condition(EntityType.FIRST_NAME)
        assert can_condition(EntityType.GENDER)
        assert not can_condition(EntityType.EMAIL)
        assert not can_condition(EntityType.FREE_TEXT)

    def test_depends_on_matrix_only_lists_conditionable_types(self) -> None:
        for sources in ALLOWED_DEPENDS_ON.values():
            for source in sources:
                assert can_condition(source)


@pytest.mark.unit
class TestPiiColumnPlan:
    def test_identify_only_entity_type_is_rejected(self) -> None:
        with _raises("identify-only"):
            PiiColumnPlan(column_name="sex", entity_type=EntityType.GENDER)

    def test_pattern_rejected_when_entity_has_no_pattern_syntax(self) -> None:
        with _raises("does not allow pattern"):
            PiiColumnPlan(column_name="ssn", entity_type=EntityType.SSN, pattern="###-##-####")
        with _raises("does not allow pattern"):
            PiiColumnPlan(column_name="addr", entity_type=EntityType.STREET_ADDRESS, pattern="#### Main St")
        with _raises("does not allow pattern"):
            PiiColumnPlan(column_name="notes", entity_type=EntityType.FREE_TEXT, pattern="{First}")

    def test_empty_pattern_is_treated_as_omitted(self) -> None:
        spec = PiiColumnPlan(column_name="dob", entity_type=EntityType.DATE_OF_BIRTH, pattern="  ")
        assert spec.pattern is None

    def test_strftime_and_name_parts_patterns_are_accepted(self) -> None:
        dob = PiiColumnPlan(column_name="dob", entity_type=EntityType.DATE_OF_BIRTH, pattern="%d.%m.%y")
        name = PiiColumnPlan(column_name="first", entity_type=EntityType.FIRST_NAME, pattern="{First}")
        assert dob.pattern == "%d.%m.%y"
        assert name.pattern == "{First}"

    def test_depends_on_rejected_for_entities_not_in_matrix(self) -> None:
        with _raises("does not allow depends_on"):
            PiiColumnPlan(
                column_name="phone",
                entity_type=EntityType.PHONE_NUMBER,
                depends_on=[ConditioningColumn(column_name="gender", entity_type=EntityType.GENDER)],
            )

    def test_depends_on_rejected_when_type_not_in_allowlist(self) -> None:
        with _raises("is not allowed for entity_type 'last_name'"):
            PiiColumnPlan(
                column_name="last",
                entity_type=EntityType.LAST_NAME,
                depends_on=[ConditioningColumn(column_name="gender", entity_type=EntityType.GENDER)],
            )

    def test_email_cannot_be_a_conditioner(self) -> None:
        with _raises("cannot be used in depends_on"):
            ConditioningColumn(column_name="email", entity_type=EntityType.EMAIL)


@pytest.mark.unit
class TestPiiReplacementPlan:
    def test_omitted_depends_on_entity_type_is_inferred(self) -> None:
        plan = PiiReplacementPlan.model_validate(
            {
                "columns_to_replace": [
                    {"column_name": "first_name", "entity_type": "first_name"},
                    {
                        "column_name": "email",
                        "entity_type": "email",
                        "depends_on": [{"column_name": "first_name"}],
                    },
                ]
            }
        )
        assert plan.columns_to_replace[1].depends_on[0].entity_type is EntityType.FIRST_NAME

    def test_inference_does_not_mutate_a_column_plan_reused_across_plans(self) -> None:
        email = PiiColumnPlan(
            column_name="email",
            entity_type=EntityType.EMAIL,
            depends_on=[ConditioningColumn(column_name="person")],
        )

        first = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(column_name="person", entity_type=EntityType.FIRST_NAME),
                email,
            ]
        )
        assert first.columns_to_replace[1].depends_on[0].entity_type is EntityType.FIRST_NAME
        assert email.depends_on[0].entity_type is None

        second = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(column_name="person", entity_type=EntityType.LAST_NAME),
                email,
            ]
        )
        assert second.columns_to_replace[1].depends_on[0].entity_type is EntityType.LAST_NAME
        assert email.depends_on[0].entity_type is None
        assert first.columns_to_replace[1].depends_on[0].entity_type is EntityType.FIRST_NAME

    def test_omitted_type_errors_when_column_is_not_a_replace_target(self) -> None:
        with _raises("omits entity_type but is not listed"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {
                            "column_name": "first_name",
                            "entity_type": "first_name",
                            "depends_on": [{"column_name": "gender"}],
                        }
                    ]
                }
            )

    def test_explicit_type_must_match_replace_entry(self) -> None:
        with _raises("columns_to_replace lists entity_type"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {"column_name": "first_name", "entity_type": "first_name"},
                        {
                            "column_name": "email",
                            "entity_type": "email",
                            "depends_on": [
                                {"column_name": "first_name", "entity_type": "last_name"},
                            ],
                        },
                    ]
                }
            )

    def test_replaceable_conditioner_must_be_a_replace_target(self) -> None:
        with _raises("list it in columns_to_replace"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {
                            "column_name": "email",
                            "entity_type": "email",
                            "depends_on": [
                                {"column_name": "first_name", "entity_type": "first_name"},
                            ],
                        }
                    ]
                }
            )

    def test_duplicate_conditioner_entity_type_on_one_target_is_rejected(self) -> None:
        with _raises("appears more than once"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {
                            "column_name": "first_name",
                            "entity_type": "first_name",
                            "depends_on": [
                                {"column_name": "gender", "entity_type": "gender"},
                                {"column_name": "spouse_gender", "entity_type": "gender"},
                            ],
                        }
                    ]
                }
            )

    def test_email_may_depend_on_name_parts_or_full_name_not_both(self) -> None:
        PiiReplacementPlan.model_validate(
            {
                "columns_to_replace": [
                    {"column_name": "first_name", "entity_type": "first_name"},
                    {"column_name": "last_name", "entity_type": "last_name"},
                    {
                        "column_name": "email",
                        "entity_type": "email",
                        "depends_on": [
                            {"column_name": "first_name"},
                            {"column_name": "last_name"},
                            {"column_name": "employer", "entity_type": "organization"},
                        ],
                    },
                ]
            }
        )
        with _raises("mutually exclusive conditioner groups"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {"column_name": "first_name", "entity_type": "first_name"},
                        {"column_name": "legal_name", "entity_type": "full_name"},
                        {
                            "column_name": "email",
                            "entity_type": "email",
                            "depends_on": [
                                {"column_name": "first_name"},
                                {"column_name": "legal_name"},
                            ],
                        },
                    ]
                }
            )

    def test_full_name_cannot_mix_with_gender_on_the_same_target(self) -> None:
        with _raises("mutually exclusive conditioner groups"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {"column_name": "legal_name", "entity_type": "full_name"},
                        {
                            "column_name": "first_name",
                            "entity_type": "first_name",
                            "depends_on": [
                                {"column_name": "legal_name"},
                                {"column_name": "gender", "entity_type": "gender"},
                            ],
                        },
                    ]
                }
            )

    def test_first_name_may_depend_on_gender_and_ethnicity(self) -> None:
        PiiReplacementPlan.model_validate(
            {
                "columns_to_replace": [
                    {
                        "column_name": "first_name",
                        "entity_type": "first_name",
                        "depends_on": [
                            {"column_name": "gender", "entity_type": "gender"},
                            {"column_name": "ethnicity", "entity_type": "ethnic_background"},
                        ],
                    }
                ]
            }
        )

    def test_zipcode_cannot_mix_with_city(self) -> None:
        with _raises("mutually exclusive conditioner groups"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {
                            "column_name": "street",
                            "entity_type": "street_address",
                            "depends_on": [
                                {"column_name": "zip", "entity_type": "zipcode"},
                                {"column_name": "city", "entity_type": "city"},
                            ],
                        }
                    ]
                }
            )

    def test_inferred_depends_on_must_still_be_in_the_allowlist(self) -> None:
        with _raises("is not allowed for entity_type 'last_name'"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {"column_name": "first_name", "entity_type": "first_name"},
                        {
                            "column_name": "last_name",
                            "entity_type": "last_name",
                            "depends_on": [{"column_name": "first_name"}],
                        },
                    ]
                }
            )

    def test_inferred_depends_on_must_be_conditionable(self) -> None:
        with _raises("cannot be used as a conditioner"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {"column_name": "street", "entity_type": "street_address"},
                        {
                            "column_name": "email",
                            "entity_type": "email",
                            "depends_on": [{"column_name": "street"}],
                        },
                    ]
                }
            )

    def test_duplicate_replace_columns_are_rejected(self) -> None:
        with _raises("duplicate column_name"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {"column_name": "name", "entity_type": "first_name"},
                        {"column_name": "name", "entity_type": "last_name"},
                    ]
                }
            )

    def test_street_address_may_depend_on_city_state_country(self) -> None:
        PiiReplacementPlan.model_validate(
            {
                "columns_to_replace": [
                    {
                        "column_name": "street",
                        "entity_type": "street_address",
                        "depends_on": [
                            {"column_name": "city", "entity_type": "city"},
                            {"column_name": "state", "entity_type": "state"},
                            {"column_name": "country", "entity_type": "country"},
                        ],
                    }
                ]
            }
        )


@pytest.mark.unit
class TestReplacePiiConfig:
    def test_defaults_are_auto_discovery(self) -> None:
        config = ReplacePiiConfig()
        assert config.schema_version == 1
        assert config.replacement_plan == AUTO_DISCOVERY
        assert config.is_auto_discovery
        assert config.plan_path is None
        assert config.inline_plan is None
        assert config.llm is None
        assert config.sampler.backend is PiiSamplerBackend.MANAGED
        assert ENTITY_BY_TYPE[EntityType.FREE_TEXT].action is EntityAction.REPLACE_IN_TEXT

    def test_plan_path_and_inline_plan_properties(self) -> None:
        path_config = ReplacePiiConfig(replacement_plan="/tmp/plan.yaml")
        assert path_config.plan_path == "/tmp/plan.yaml"
        assert path_config.inline_plan is None
        assert not path_config.is_auto_discovery

        plan = PiiReplacementPlan(scope=PiiReplacementScope.RECORD)
        inline = ReplacePiiConfig(replacement_plan=plan)
        assert inline.inline_plan is plan
        assert inline.plan_path is None

    def test_path_object_is_stored_as_string(self) -> None:
        config = ReplacePiiConfig.model_validate({"replacement_plan": Path("plans/pii.yaml")})
        assert config.replacement_plan == "plans/pii.yaml"
        assert config.plan_path == "plans/pii.yaml"

    @pytest.mark.parametrize(
        "value",
        ["plans/pii.yaml", "plan.yaml", "./plan.yaml", "plan"],
    )
    def test_non_sentinel_strings_are_stored_as_plan_path(self, value: str) -> None:
        config = ReplacePiiConfig(replacement_plan=value)
        assert config.plan_path == value
        assert not config.is_auto_discovery

    def test_inline_mapping_is_validated_as_a_plan(self) -> None:
        config = ReplacePiiConfig.model_validate(
            {
                "replacement_plan": {
                    "scope": "record",
                    "columns_to_replace": [
                        {"column_name": "phone", "entity_type": "phone_number"},
                    ],
                }
            }
        )
        assert isinstance(config.inline_plan, PiiReplacementPlan)
        assert config.inline_plan.scope.value == "record"

    def test_malformed_inline_plan_raises_parameter_error(self) -> None:
        with _raises("invalid inline replacement plan"):
            ReplacePiiConfig.model_validate({"replacement_plan": {"scope": "galaxy"}})

    def test_llm_mapping_configures_shared_inference_behavior(self) -> None:
        config = ReplacePiiConfig.model_validate(
            {
                "llm": {
                    "model_id": "local-model",
                }
            }
        )

        assert config.llm == LLMConfig(model_id="local-model")
        assert config.llm is not None
        assert config.llm.max_workers == 8

    def test_llm_endpoint_is_runtime_only(self) -> None:
        with _raises("Unknown configuration field 'replace_pii.llm.endpoint_url'"):
            SafeSynthesizerParameters.model_validate(
                {"replace_pii": {"llm": {"endpoint_url": "http://localhost:8000/v1"}}}
            )

    def test_empty_llm_mapping_enables_inference_defaults(self) -> None:
        config = ReplacePiiConfig.model_validate({"llm": {}})

        assert config.llm == LLMConfig()

    def test_llm_max_workers_must_be_positive(self) -> None:
        with _raises("greater than or equal to 1"):
            ReplacePiiConfig.model_validate({"llm": {"max_workers": 0}})

    def test_resolved_managed_assets_path_uses_override(self, tmp_path: Path) -> None:
        config = ReplacePiiConfig.model_validate(
            {"sampler": {"backend": "faker", "managed_assets_path": str(tmp_path)}}
        )
        assert config.sampler.resolved_managed_assets_path() == tmp_path

    def test_default_managed_assets_path_uses_env_then_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(NSS_MANAGED_ASSETS_PATH_ENV, raising=False)
        assert default_managed_assets_path() == Path.home() / ".data-designer" / "managed-assets"
        monkeypatch.setenv(NSS_MANAGED_ASSETS_PATH_ENV, str(tmp_path))
        assert default_managed_assets_path() == tmp_path
        config = ReplacePiiConfig()
        assert config.sampler.resolved_managed_assets_path() == tmp_path
