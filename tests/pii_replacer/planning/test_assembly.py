# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType, PiiReplacementScope
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
class TestPlanAssembly:
    def test_replacement_membership_is_derived_from_entity_metadata(self) -> None:
        plan = plan_from_classifications(
            PiiReplacementScope.GROUP,
            [
                ColumnClassification(column_name="patient_id", entity_type=EntityType.UNIQUE_IDENTIFIER),
                ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
                ColumnClassification(column_name="company", entity_type=EntityType.ORGANIZATION),
                ColumnClassification(column_name="notes", entity_type=None),
            ],
            protected_columns=frozenset({"email"}),
        )

        assert plan.scope is PiiReplacementScope.GROUP
        assert [spec.column_name for spec in plan.columns_to_replace] == ["patient_id"]

    def test_dependency_candidates_are_derived_from_catalog_relationships(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="first_name", entity_type=EntityType.FIRST_NAME),
            ColumnClassification(column_name="company", entity_type=EntityType.ORGANIZATION),
            ColumnClassification(column_name="gender", entity_type=EntityType.GENDER),
        ]
        plan = plan_from_classifications(PiiReplacementScope.DATAFRAME, classifications)

        candidates = derive_dependency_candidates(plan, classifications)

        assert candidates == [
            DependencyCandidate(
                target_column="email",
                target_entity_type=EntityType.EMAIL,
                source_column="first_name",
                source_entity_type=EntityType.FIRST_NAME,
            ),
            DependencyCandidate(
                target_column="email",
                target_entity_type=EntityType.EMAIL,
                source_column="company",
                source_entity_type=EntityType.ORGANIZATION,
            ),
            DependencyCandidate(
                target_column="first_name",
                target_entity_type=EntityType.FIRST_NAME,
                source_column="gender",
                source_entity_type=EntityType.GENDER,
            ),
        ]

    def test_selected_dependencies_are_applied_to_the_plan(self) -> None:
        classifications = [
            ColumnClassification(column_name="email", entity_type=EntityType.EMAIL),
            ColumnClassification(column_name="company", entity_type=EntityType.ORGANIZATION),
        ]
        plan = plan_from_classifications(PiiReplacementScope.DATAFRAME, classifications)
        candidate = derive_dependency_candidates(plan, classifications)[0]

        result = apply_dependencies(plan, [candidate])

        assert result.columns_to_replace[0].depends_on[0].column_name == "company"
        assert result.columns_to_replace[0].depends_on[0].entity_type is EntityType.ORGANIZATION

    def test_dependency_candidates_reject_relationships_outside_the_catalog(self) -> None:
        with pytest.raises(ParameterError, match="cannot condition"):
            DependencyCandidate(
                target_column="email",
                target_entity_type=EntityType.EMAIL,
                source_column="gender",
                source_entity_type=EntityType.GENDER,
            )

    def test_dependencies_must_match_a_replacement_target(self) -> None:
        plan = plan_from_classifications(
            PiiReplacementScope.DATAFRAME,
            [ColumnClassification(column_name="email", entity_type=EntityType.EMAIL)],
        )
        candidate = DependencyCandidate(
            target_column="name",
            target_entity_type=EntityType.FULL_NAME,
            source_column="gender",
            source_entity_type=EntityType.GENDER,
        )

        with pytest.raises(ParameterError, match="unknown replacement column"):
            apply_dependencies(plan, [candidate])

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
            "\\x": "literal x",
        }
        name_placeholders = grammars["name_parts"]["placeholders"]
        assert isinstance(name_placeholders, dict)
        assert "{domain}" in name_placeholders
