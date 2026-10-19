# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated, get_args, get_origin

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ALLOWED_DEPENDS_ON,
    AUTO_DISCOVERY,
    DEFAULT_GLINER2_MODEL_ID,
    DEFAULT_GLINER2_MODEL_REVISION,
    ENTITIES,
    ENTITY_BY_TYPE,
    FREE_TEXT_DETECTION_ENTITY_TYPES,
    GLINER_DETECTION_ENTITY_TYPES,
    ConditioningColumn,
    EntityAction,
    EntityType,
    FreeTextDetectionConfig,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiSamplerBackend,
    ReplacePiiConfig,
    can_condition,
    is_columns_to_replace_type,
)
from nemo_safe_synthesizer.defaults import NSS_NEMOTRON_PERSONAS_PATH_ENV, default_nemotron_personas_path


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
        with pytest.raises(ValidationError, match="identify-only"):
            PiiColumnPlan(column_name="sex", entity_type=EntityType.GENDER)

    def test_pattern_rejected_when_entity_has_no_pattern_syntax(self) -> None:
        with pytest.raises(ValidationError, match="does not allow pattern"):
            PiiColumnPlan(column_name="ssn", entity_type=EntityType.SSN, pattern="###-##-####")
        with pytest.raises(ValidationError, match="does not allow pattern"):
            PiiColumnPlan(column_name="addr", entity_type=EntityType.STREET_ADDRESS, pattern="#### Main St")
        with pytest.raises(ValidationError, match="does not allow pattern"):
            PiiColumnPlan(column_name="notes", entity_type=EntityType.FREE_TEXT, pattern="{First}")

    @pytest.mark.parametrize("pattern", ["", "  "])
    def test_empty_pattern_is_rejected(self, pattern: str) -> None:
        with pytest.raises(ValidationError, match="pattern must be non-empty when provided"):
            PiiColumnPlan(column_name="dob", entity_type=EntityType.DATE_OF_BIRTH, pattern=pattern)

    def test_strftime_and_name_parts_patterns_are_accepted(self) -> None:
        dob = PiiColumnPlan(column_name="dob", entity_type=EntityType.DATE_OF_BIRTH, pattern="%d.%m.%y")
        name = PiiColumnPlan(column_name="first", entity_type=EntityType.FIRST_NAME, pattern="{First}")
        assert dob.pattern == "%d.%m.%y"
        assert name.pattern == "{First}"

    def test_depends_on_rejected_for_entities_not_in_matrix(self) -> None:
        with pytest.raises(ValidationError, match="does not allow depends_on"):
            PiiColumnPlan(
                column_name="phone",
                entity_type=EntityType.PHONE_NUMBER,
                depends_on=[ConditioningColumn(column_name="gender", entity_type=EntityType.GENDER)],
            )

    def test_depends_on_rejected_when_type_not_in_allowlist(self) -> None:
        with pytest.raises(ValidationError, match="is not allowed for entity_type 'last_name'"):
            PiiColumnPlan(
                column_name="last",
                entity_type=EntityType.LAST_NAME,
                depends_on=[ConditioningColumn(column_name="gender", entity_type=EntityType.GENDER)],
            )

    def test_email_cannot_be_a_conditioner(self) -> None:
        with pytest.raises(ValidationError, match="cannot be used in depends_on"):
            ConditioningColumn(column_name="email", entity_type=EntityType.EMAIL)

    def test_self_dependency_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot depend on itself"):
            PiiColumnPlan(
                column_name="name",
                entity_type=EntityType.FULL_NAME,
                depends_on=[ConditioningColumn(column_name="name")],
            )


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
        dependency = plan.columns_to_replace[1].depends_on[0]
        assert dependency.entity_type is EntityType.FIRST_NAME

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
        with pytest.raises(ValidationError, match="'gender' is missing entity_type.*read-only conditioners"):
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

    @pytest.mark.parametrize("entity_type", ["first_name", None])
    def test_replacement_conditioner_may_repeat_a_matching_type(self, entity_type: str | None) -> None:
        plan = PiiReplacementPlan.model_validate(
            {
                "columns_to_replace": [
                    {"column_name": "first_name", "entity_type": "first_name"},
                    {
                        "column_name": "email",
                        "entity_type": "email",
                        "depends_on": [{"column_name": "first_name", "entity_type": entity_type}],
                    },
                ]
            }
        )

        assert plan.columns_to_replace[1].depends_on[0].entity_type is EntityType.FIRST_NAME

    def test_replacement_conditioner_type_must_match_its_entry(self) -> None:
        entity_type = "last_name"
        with pytest.raises(ValidationError, match="but its columns_to_replace entry has 'first_name'"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {"column_name": "first_name", "entity_type": "first_name"},
                        {
                            "column_name": "email",
                            "entity_type": "email",
                            "depends_on": [
                                {"column_name": "first_name", "entity_type": entity_type},
                            ],
                        },
                    ]
                }
            )

    def test_replaceable_conditioner_must_be_a_replace_target(self) -> None:
        with pytest.raises(ValidationError, match="list it in columns_to_replace"):
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
        with pytest.raises(ValidationError, match="appears more than once"):
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
        with pytest.raises(ValidationError, match="mutually exclusive conditioner groups"):
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
        with pytest.raises(ValidationError, match="mutually exclusive conditioner groups"):
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

    def test_inferred_depends_on_must_still_be_in_the_allowlist(self) -> None:
        with pytest.raises(ValidationError, match="is not allowed for entity_type 'last_name'"):
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
        with pytest.raises(ValidationError, match="cannot be used as a conditioner"):
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
        with pytest.raises(ValidationError, match="duplicate column_name"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {"column_name": "name", "entity_type": "first_name"},
                        {"column_name": "name", "entity_type": "last_name"},
                    ]
                }
            )

    def test_dependency_cycles_are_rejected_defensively(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The current catalog is acyclic. Temporarily allow the reverse edge to
        # verify that the plan model still protects future catalog changes.
        monkeypatch.setitem(ALLOWED_DEPENDS_ON, EntityType.FULL_NAME, frozenset({EntityType.FIRST_NAME}))

        with pytest.raises(ValidationError, match="dependencies contain a cycle"):
            PiiReplacementPlan.model_validate(
                {
                    "columns_to_replace": [
                        {
                            "column_name": "first_name",
                            "entity_type": "first_name",
                            "depends_on": [{"column_name": "full_name"}],
                        },
                        {
                            "column_name": "full_name",
                            "entity_type": "full_name",
                            "depends_on": [{"column_name": "first_name"}],
                        },
                    ]
                }
            )

    def test_street_address_takes_no_locality_dependencies(self) -> None:
        # Only the street is replaced; the original locality text stays as written.
        with pytest.raises(ValidationError, match="cannot be used in depends_on"):
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
        assert config.schema_version == 3
        assert config.replacement_plan == AUTO_DISCOVERY
        assert config.locale == "en_US"
        assert config.seed is None
        assert config.is_auto_discovery
        assert config.inline_plan is None
        assert config.llm == LLMConfig()
        assert config.sampler.backend is PiiSamplerBackend.NEMOTRON_PERSONAS
        assert "sampler" not in config.model_fields_set
        assert ENTITY_BY_TYPE[EntityType.FREE_TEXT].action is EntityAction.REPLACE_IN_TEXT

    def test_null_llm_is_rejected_with_the_ways_to_skip_it(self) -> None:
        with pytest.raises(ValidationError, match="give an inline plan in replace_pii.replacement_plan"):
            ReplacePiiConfig.model_validate({"llm": None})

    def test_missing_schema_version_is_v3_and_sparse_serialization_includes_it(self) -> None:
        config = ReplacePiiConfig.model_validate({})

        assert config.schema_version == 3
        assert config.model_dump(exclude_unset=True)["schema_version"] == 3

    def test_sparse_parent_serialization_includes_schema_version(self) -> None:
        config = SafeSynthesizerParameters(replace_pii=ReplacePiiConfig())

        assert config.model_dump(exclude_unset=True)["replace_pii"]["schema_version"] == 3

    def test_config_serialization_omits_empty_and_null_plan_fields(self) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    depends_on=[],
                )
            ]
        )
        serialized = ReplacePiiConfig(replacement_plan=plan).model_dump()
        replacement_plan = serialized["replacement_plan"]

        assert isinstance(replacement_plan, dict)
        serialized_column = replacement_plan["columns_to_replace"][0]
        assert "depends_on" not in serialized_column
        assert "pattern" not in serialized_column
        assert replacement_plan["data_to_sampler_value_mapping"] == {}

    def test_config_serialization_includes_inferred_dependency_type(self) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(column_name="first", entity_type=EntityType.FIRST_NAME),
                PiiColumnPlan(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    depends_on=[
                        ConditioningColumn(column_name="first"),
                        ConditioningColumn(column_name="company", entity_type=EntityType.ORGANIZATION),
                    ],
                ),
            ]
        )

        serialized = ReplacePiiConfig(replacement_plan=plan).model_dump(exclude_unset=False)
        replacement_plan = serialized["replacement_plan"]

        assert isinstance(replacement_plan, dict)
        assert replacement_plan["columns_to_replace"][1]["depends_on"] == [
            {"column_name": "first", "entity_type": EntityType.FIRST_NAME},
            {"column_name": "company", "entity_type": EntityType.ORGANIZATION},
        ]

    @pytest.mark.parametrize("schema_version", [1, 2, 0, -1])
    def test_unsupported_schema_version_is_rejected(self, schema_version: int) -> None:
        with pytest.raises(
            ValidationError, match=f"schema version {schema_version} is unsupported.*supports version 3"
        ):
            ReplacePiiConfig.model_validate({"schema_version": schema_version})

    @pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
    def test_non_integer_schema_version_is_rejected(self, schema_version: object) -> None:
        with pytest.raises(ValidationError, match="schema_version must be an integer"):
            ReplacePiiConfig.model_validate({"schema_version": schema_version})

    def test_inline_plan_property(self) -> None:
        plan = PiiReplacementPlan()
        inline = ReplacePiiConfig(replacement_plan=plan)

        assert inline.inline_plan is plan
        assert not inline.is_auto_discovery

    @pytest.mark.parametrize("value", [42, ["auto_discovery"]])
    def test_a_value_that_is_no_plan_or_path_is_rejected(self, value: object) -> None:
        with pytest.raises(ValidationError, match="must be 'auto_discovery', an inline plan, or the path"):
            ReplacePiiConfig.model_validate({"replacement_plan": value})

    def test_schema_offers_the_sentinel_or_file_path_and_the_inline_plan(self) -> None:
        schema = ReplacePiiConfig.model_json_schema()

        options = schema["properties"]["replacement_plan"]["anyOf"]
        assert {"type": "string"} in options
        assert any(option.get("title") == "PiiReplacementPlan" for option in options)

    def test_inline_mapping_is_validated_as_a_plan(self) -> None:
        config = ReplacePiiConfig.model_validate(
            {
                "replacement_plan": {
                    "columns_to_replace": [
                        {"column_name": "phone", "entity_type": "phone_number"},
                    ],
                }
            }
        )
        assert isinstance(config.inline_plan, PiiReplacementPlan)
        assert config.inline_plan.columns_to_replace[0].column_name == "phone"

    def test_parsed_inline_plan_skips_redundant_union_validation(self) -> None:
        annotation = ReplacePiiConfig.model_fields["replacement_plan"].annotation
        plan_arm = next(member for member in get_args(annotation) if get_origin(member) is Annotated)

        assert get_args(plan_arm)[0] is PiiReplacementPlan
        assert any(type(metadata).__name__ == "SkipValidation" for metadata in get_args(plan_arm)[1:])

    def test_full_config_rejects_unknown_inline_plan_field(self) -> None:
        with pytest.raises(
            ValidationError,
            match="Unknown configuration field 'replace_pii.replacement_plan.unexpected'",
        ):
            SafeSynthesizerParameters.model_validate({"replace_pii": {"replacement_plan": {"unexpected": True}}})

    def test_plan_file_is_read_relative_to_the_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "pii_replacement_plan.yaml").write_text(
            "columns_to_replace:\n- column_name: email\n  entity_type: email\n"
        )
        (config_dir / "config.yaml").write_text("replace_pii:\n  replacement_plan: pii_replacement_plan.yaml\n")
        monkeypatch.chdir(tmp_path)

        replace_pii = SafeSynthesizerParameters.from_yaml(config_dir / "config.yaml").replace_pii

        assert replace_pii is not None
        assert replace_pii.inline_plan is not None
        assert replace_pii.inline_plan.columns_to_replace[0].column_name == "email"

    def test_relative_personas_path_resolves_against_the_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("replace_pii:\n  sampler:\n    nemotron_personas_path: personas\n")
        monkeypatch.chdir(tmp_path)

        replace_pii = SafeSynthesizerParameters.from_yaml(Path("configs") / "config.yaml").replace_pii

        assert replace_pii is not None
        assert replace_pii.sampler.nemotron_personas_path == str(config_dir / "personas")

    def test_missing_plan_file_is_reported_by_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="is not 'auto_discovery' or an inline plan, and the plan file"):
            ReplacePiiConfig.model_validate({"replacement_plan": str(tmp_path / "missing.yaml")})

    def test_llm_mapping_configures_planning_inference_behavior(self) -> None:
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
        with pytest.raises(ValidationError, match="Unknown configuration field 'replace_pii.llm.endpoint_url'"):
            SafeSynthesizerParameters.model_validate(
                {"replace_pii": {"llm": {"endpoint_url": "http://localhost:8000/v1"}}}
            )

    def test_empty_llm_mapping_enables_inference_defaults(self) -> None:
        config = ReplacePiiConfig.model_validate({"llm": {}})

        assert config.llm == LLMConfig()

    def test_llm_max_workers_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ReplacePiiConfig.model_validate({"llm": {"max_workers": 0}})

    def test_free_text_detection_defaults_and_serialization(self) -> None:
        config = ReplacePiiConfig()
        expected_thresholds = {
            EntityType.FIRST_NAME: 0.946,
            EntityType.MIDDLE_NAME: 0.7,
            EntityType.LAST_NAME: 0.618,
            EntityType.FULL_NAME: 0.964,
            EntityType.PHONE_NUMBER: 0.990,
            EntityType.DATE_OF_BIRTH: 0.9998,
            EntityType.STREET_ADDRESS: 0.985,
            EntityType.SSN: 0.050,
            EntityType.NATIONAL_ID: 0.982,
            EntityType.API_KEY: 0.992,
            EntityType.UNIQUE_IDENTIFIER: 0.947,
        }
        expected_exclusions = {
            "body_part": 0.062,
            "family_relationship": 0.062,
            "gender": 0.385,
            "medical_condition": 0.393,
            "organism": 0.051,
        }

        assert config.free_text_detection == FreeTextDetectionConfig(
            model_id=DEFAULT_GLINER2_MODEL_ID,
            model_revision=DEFAULT_GLINER2_MODEL_REVISION,
            entity_thresholds=expected_thresholds,
            name_exclusion_thresholds=expected_exclusions,
            require_middle_name_neighbors=True,
            batch_size=8,
            chunk_length=384,
            chunk_overlap=128,
        )
        assert config.model_dump(mode="json")["free_text_detection"] == {
            "model_id": DEFAULT_GLINER2_MODEL_ID,
            "model_revision": DEFAULT_GLINER2_MODEL_REVISION,
            "entity_thresholds": {
                entity_type.value: threshold for entity_type, threshold in expected_thresholds.items()
            },
            "name_exclusion_thresholds": expected_exclusions,
            "require_middle_name_neighbors": True,
            "batch_size": 8,
            "chunk_length": 384,
            "chunk_overlap": 128,
        }
        assert tuple(config.free_text_detection.entity_thresholds) == GLINER_DETECTION_ENTITY_TYPES
        assert FreeTextDetectionConfig.model_validate(config.model_dump(mode="json")["free_text_detection"]) == (
            config.free_text_detection
        )
        assert set(FREE_TEXT_DETECTION_ENTITY_TYPES) - set(GLINER_DETECTION_ENTITY_TYPES) == {
            EntityType.EMAIL,
            EntityType.CREDIT_DEBIT_CARD,
            EntityType.IPV4,
            EntityType.IPV6,
        }

    @pytest.mark.parametrize(
        ("values", "expected_revision"),
        [
            ({}, DEFAULT_GLINER2_MODEL_REVISION),
            ({"model_revision": None}, None),
            ({"model_id": "example/other-gliner2"}, None),
            ({"model_id": "example/other-gliner2", "model_revision": "v1.0"}, "v1.0"),
            ({"model_id": DEFAULT_GLINER2_MODEL_ID, "model_revision": "main"}, "main"),
        ],
    )
    def test_model_revision_is_pinned_only_for_the_default_model(
        self,
        values: dict[str, str | None],
        expected_revision: str | None,
    ) -> None:
        assert FreeTextDetectionConfig.model_validate(values).model_revision == expected_revision

    @pytest.mark.parametrize("threshold", [-0.01, 1.01])
    def test_free_text_detection_thresholds_must_be_in_closed_unit_interval(self, threshold: float) -> None:
        entity_thresholds = FreeTextDetectionConfig().entity_thresholds | {EntityType.PHONE_NUMBER: threshold}

        with pytest.raises(ValidationError, match="less than or equal|greater than or equal"):
            FreeTextDetectionConfig.model_validate({"entity_thresholds": entity_thresholds})

    def test_free_text_detection_merges_partial_thresholds_with_defaults(self) -> None:
        config = ReplacePiiConfig.model_validate({"free_text_detection": {"entity_thresholds": {"first_name": 0.8}}})

        expected = FreeTextDetectionConfig().entity_thresholds | {EntityType.FIRST_NAME: 0.8}
        assert config.free_text_detection.entity_thresholds == expected

    @pytest.mark.parametrize(
        "entity_type",
        [EntityType.EMAIL, EntityType.CREDIT_DEBIT_CARD, EntityType.IPV4, EntityType.IPV6],
    )
    def test_free_text_detection_rejects_thresholds_for_non_gliner_entities(
        self,
        entity_type: EntityType,
    ) -> None:
        entity_thresholds = FreeTextDetectionConfig().entity_thresholds | {entity_type: 0.5}

        with pytest.raises(ValidationError, match=rf"unsupported: \['{entity_type.value}'\]"):
            FreeTextDetectionConfig(entity_thresholds=entity_thresholds)

    def test_free_text_detection_exclusion_mapping_replaces_the_defaults(self) -> None:
        config = FreeTextDetectionConfig.model_validate({"name_exclusion_thresholds": {"medical_condition": 0.2}})

        assert config.name_exclusion_thresholds == {"medical_condition": 0.2}

    @pytest.mark.parametrize(
        ("labels", "match"),
        [
            ({" ": 0.1}, "must be nonempty"),
            ({"gender": 0.1, "Gender": 0.2}, "must be distinct"),
            ({"first-name": 0.1}, r"detected entity types: \['first_name'\]"),
        ],
    )
    def test_free_text_detection_rejects_invalid_exclusion_labels(self, labels: dict[str, float], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            FreeTextDetectionConfig.model_validate({"name_exclusion_thresholds": labels})

    @pytest.mark.parametrize("field", ["batch_size", "chunk_length"])
    def test_free_text_detection_batch_and_chunk_values_must_be_positive(self, field: str) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            FreeTextDetectionConfig.model_validate({field: 0})

    def test_free_text_detection_overlap_must_be_nonnegative_and_smaller_than_chunk(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            FreeTextDetectionConfig(chunk_overlap=-1)
        with pytest.raises(ValidationError, match="must be smaller than chunk_length"):
            FreeTextDetectionConfig(chunk_length=32, chunk_overlap=32)

        assert FreeTextDetectionConfig(chunk_length=32, chunk_overlap=0).chunk_overlap == 0

    def test_free_text_detection_model_id_must_be_nonempty(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 character"):
            FreeTextDetectionConfig(model_id="")

    def test_resolved_nemotron_personas_path_uses_override(self, tmp_path: Path) -> None:
        config = ReplacePiiConfig.model_validate(
            {"sampler": {"backend": "faker", "nemotron_personas_path": str(tmp_path)}}
        )
        assert config.sampler.resolved_nemotron_personas_path() == tmp_path

    def test_plan_accepts_manual_data_to_sampler_value_mapping(self) -> None:
        plan = PiiReplacementPlan.model_validate(
            {
                "data_to_sampler_value_mapping": {
                    "sex": {"Woman": ["female"], "Non-binary": ["female", "male"]},
                    "race": {"Asian": ["east asian", "south asian"]},
                },
            }
        )

        assert plan.data_to_sampler_value_mapping == {
            "sex": {"Woman": ["female"], "Non-binary": ["female", "male"]},
            "race": {"Asian": ["east asian", "south asian"]},
        }
        assert plan.model_dump()["data_to_sampler_value_mapping"]["sex"]["Non-binary"] == ["female", "male"]

    def test_plan_rejects_null_sampler_values(self) -> None:
        with pytest.raises(ValidationError, match="list_type"):
            PiiReplacementPlan.model_validate({"data_to_sampler_value_mapping": {"sex": {"Non-binary": None}}})

    @pytest.mark.parametrize(
        ("mappings", "error"),
        [
            ({"": {"work": ["personal"]}}, "column names must be non-empty"),
            ({"sex": {"": ["female"]}}, "dataset values must be non-empty"),
            ({"sex": {"Woman": []}}, "sampler value lists must be non-empty"),
            ({"sex": {"Woman": [" "]}}, "sampler values must be non-empty"),
            (
                {"sex": {"Woman": ["female"], "woman": ["female"]}},
                "duplicate dataset values after case-folding",
            ),
            (
                {"sex": {"Woman": ["female", "FEMALE"]}},
                "duplicate sampler values after case-folding",
            ),
        ],
    )
    def test_plan_rejects_invalid_data_to_sampler_value_mapping(
        self,
        mappings: object,
        error: str,
    ) -> None:
        with pytest.raises(ValidationError, match=error):
            PiiReplacementPlan.model_validate({"data_to_sampler_value_mapping": mappings})

    def test_default_nemotron_personas_path_uses_env_then_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(NSS_NEMOTRON_PERSONAS_PATH_ENV, raising=False)
        assert default_nemotron_personas_path() == Path.home() / ".data-designer" / "managed-assets"
        monkeypatch.setenv(NSS_NEMOTRON_PERSONAS_PATH_ENV, str(tmp_path))
        assert default_nemotron_personas_path() == tmp_path
        assert ReplacePiiConfig().sampler.resolved_nemotron_personas_path() == tmp_path
