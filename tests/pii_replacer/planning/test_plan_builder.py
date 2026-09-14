# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import (
    ColumnClassification,
    DependencyCandidate,
    apply_dependencies,
    derive_dependency_candidates,
    pattern_grammar_catalog,
    plan_from_classifications,
)


@pytest.mark.unit
class TestPlanBuilder:
    def test_replacement_membership_is_derived_from_entity_metadata(self) -> None:
        plan = plan_from_classifications(
            [
                ColumnClassification(column_name="patient_id", entity_type=EntityType.UNIQUE_IDENTIFIER),
                ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
                ColumnClassification(column_name="company", entity_type=EntityType.ORGANIZATION),
                ColumnClassification(column_name="notes", entity_type=None),
            ],
            expected_columns=frozenset({"patient_id", "email", "company", "notes"}),
            protected_columns=frozenset({"email"}),
        )

        assert [spec.column_name for spec in plan.columns_to_replace] == ["patient_id"]

    def test_dependency_candidates_are_derived_from_catalog_relationships(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="first_name", entity_type=EntityType.FIRST_NAME),
            ColumnClassification(column_name="company", entity_type=EntityType.ORGANIZATION),
            ColumnClassification(column_name="gender", entity_type=EntityType.GENDER),
        ]
        candidates = derive_dependency_candidates(
            classifications,
            expected_columns=frozenset({"email", "first_name", "company", "gender"}),
        )

        assert candidates == [
            DependencyCandidate(
                target_column="email",
                source_column="first_name",
            ),
            DependencyCandidate(
                target_column="email",
                source_column="company",
            ),
            DependencyCandidate(
                target_column="first_name",
                source_column="gender",
            ),
        ]

    def test_dependency_candidates_apply_protected_column_exclusion(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="company", entity_type=EntityType.ORGANIZATION),
        ]

        candidates = derive_dependency_candidates(
            classifications,
            expected_columns=frozenset({"email", "company"}),
            protected_columns=frozenset({"email"}),
        )

        assert candidates == []

    def test_selected_dependencies_are_applied_to_the_plan(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="company", entity_type=EntityType.ORGANIZATION),
        ]
        expected_columns = frozenset({"email", "company"})
        plan = plan_from_classifications(classifications, expected_columns=expected_columns)
        candidate = derive_dependency_candidates(classifications, expected_columns=expected_columns)[0]

        result = apply_dependencies(plan, [candidate], classifications=classifications)

        assert result.columns_to_replace[0].depends_on[0].column_name == "company"
        assert result.columns_to_replace[0].depends_on[0].entity_type is EntityType.ORGANIZATION

    def test_dependency_candidate_rejects_self_edge(self) -> None:
        with pytest.raises(ParameterError, match="same column"):
            DependencyCandidate(
                target_column="email",
                source_column="email",
            )

    def test_dependencies_must_match_a_replacement_target(self) -> None:
        plan = plan_from_classifications(
            [ColumnClassification(column_name="email", entity_type=EntityType.EMAIL)],
            expected_columns=frozenset({"email"}),
        )
        candidate = DependencyCandidate(
            target_column="name",
            source_column="gender",
        )

        with pytest.raises(ParameterError, match="unknown replacement column"):
            apply_dependencies(
                plan,
                [candidate],
                classifications=[
                    ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
                    ColumnClassification(column_name="gender", entity_type=EntityType.GENDER),
                ],
            )

    def test_dependencies_reject_unknown_and_unclassified_sources(self) -> None:
        classifications = [ColumnClassification(column_name="email", entity_type=EntityType.EMAIL)]
        plan = plan_from_classifications(classifications, expected_columns=frozenset({"email"}))

        with pytest.raises(ParameterError, match="unknown classified column 'missing'"):
            apply_dependencies(
                plan,
                [DependencyCandidate(target_column="email", source_column="missing")],
                classifications=classifications,
            )

        unclassified = [
            *classifications,
            ColumnClassification(column_name="unknown", entity_type=None),
        ]
        with pytest.raises(ParameterError, match="source 'unknown' is not classified as one of the supported entities"):
            apply_dependencies(
                plan,
                [DependencyCandidate(target_column="email", source_column="unknown")],
                classifications=unclassified,
            )

    def test_dependencies_reject_plan_classification_mismatch(self) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="contact", entity_type=EntityType.EMAIL)]
        )
        classifications = [ColumnClassification(column_name="contact", entity_type=EntityType.FULL_NAME)]

        with pytest.raises(ParameterError, match="classified as 'full_name'.*plan uses 'email'"):
            apply_dependencies(plan, [], classifications=classifications)

    def test_dependencies_reject_relationships_outside_the_catalog(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="gender", entity_type=EntityType.GENDER),
        ]
        plan = plan_from_classifications(
            classifications,
            expected_columns=frozenset({"email", "gender"}),
        )

        with pytest.raises(ParameterError, match="is not allowed for entity_type 'email'"):
            apply_dependencies(
                plan,
                [DependencyCandidate(target_column="email", source_column="gender")],
                classifications=classifications,
            )

    def test_selected_exclusive_dependencies_are_rejected(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="first", entity_type=EntityType.FIRST_NAME),
            ColumnClassification(column_name="full", entity_type=EntityType.FULL_NAME),
        ]
        expected_columns = frozenset({"email", "first", "full"})
        plan = plan_from_classifications(classifications, expected_columns=expected_columns)
        candidates = derive_dependency_candidates(classifications, expected_columns=expected_columns)

        assert DependencyCandidate(target_column="email", source_column="first") in candidates
        assert DependencyCandidate(target_column="email", source_column="full") in candidates

        with pytest.raises(ValidationError, match="mutually exclusive conditioner groups"):
            apply_dependencies(
                plan,
                [
                    DependencyCandidate(target_column="email", source_column="first"),
                    DependencyCandidate(target_column="email", source_column="full"),
                ],
                classifications=classifications,
            )

    def test_replacement_conditioner_type_is_inferred_but_read_only_type_is_explicit(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="first", entity_type=EntityType.FIRST_NAME),
            ColumnClassification(column_name="company", entity_type=EntityType.ORGANIZATION),
        ]
        plan = plan_from_classifications(
            classifications,
            expected_columns=frozenset({"email", "first", "company"}),
        )

        result = apply_dependencies(
            plan,
            [
                DependencyCandidate(target_column="email", source_column="first"),
                DependencyCandidate(target_column="email", source_column="company"),
            ],
            classifications=classifications,
        )

        first, company = result.columns_to_replace[0].depends_on
        assert first.entity_type is EntityType.FIRST_NAME
        assert "entity_type" not in first.model_fields_set
        assert company.entity_type is EntityType.ORGANIZATION
        assert "entity_type" in company.model_fields_set

    @pytest.mark.parametrize("operation", ["plan", "derive", "apply"])
    def test_duplicate_classifications_are_rejected(self, operation: str) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="email", entity_type=EntityType.FULL_NAME),
        ]
        plan = PiiReplacementPlan(columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)])

        with pytest.raises(ParameterError, match="duplicate column_name"):
            if operation == "plan":
                plan_from_classifications(classifications, expected_columns=frozenset({"email"}))
            elif operation == "derive":
                derive_dependency_candidates(classifications, expected_columns=frozenset({"email"}))
            else:
                apply_dependencies(plan, [], classifications=classifications)

    @pytest.mark.parametrize("operation", ["plan", "derive"])
    def test_classifications_must_cover_every_expected_column(self, operation: str) -> None:
        classifications = [ColumnClassification(column_name="email", entity_type=EntityType.EMAIL)]

        with pytest.raises(ParameterError, match="missing expected column_name values: 'notes'"):
            if operation == "plan":
                plan_from_classifications(
                    classifications,
                    expected_columns=frozenset({"email", "notes"}),
                )
            else:
                derive_dependency_candidates(
                    classifications,
                    expected_columns=frozenset({"email", "notes"}),
                )

    def test_classifications_reject_unknown_columns(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="invented", entity_type=None),
        ]

        with pytest.raises(ParameterError, match="unknown column_name values: 'invented'"):
            plan_from_classifications(
                classifications,
                expected_columns=frozenset({"email"}),
            )

    @pytest.mark.parametrize("pattern", ["", "  "])
    def test_classification_rejects_blank_pattern(self, pattern: str) -> None:
        with pytest.raises(ValidationError, match="pattern must be non-empty when provided"):
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL, pattern=pattern)

    def test_pattern_grammar_catalog_documents_supported_tokens(self) -> None:
        grammars = pattern_grammar_catalog()

        assert grammars["character_mask"]["tokens"] == {
            "#": "digit 0-9",
            "^": "uppercase letter A-Z",
            "@": "lowercase letter a-z",
            "&": "digit or uppercase letter",
            "%": "digit or lowercase letter",
            "*": "digit or letter",
            "[abc]": "one literal character from the brackets",
            "\\x": r"literal escaped special character; for example, \# matches #",
        }
        name_placeholders = grammars["name_parts"]["placeholders"]
        assert isinstance(name_placeholders, dict)
        assert "{domain}" in name_placeholders
        assert {"{F}", "{M}", "{L}"} <= name_placeholders.keys()
