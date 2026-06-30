# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Self

import pytest
from pydantic import BaseModel, Field, model_validator

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.patch import CompiledConfigPatch, PatchAssignment
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig, StepDefinition
from nemo_safe_synthesizer.configurator.parameter_paths import ParameterPath, ParameterSchema, UnknownParameterName
from nemo_safe_synthesizer.errors import ParameterError


class _Child(BaseModel):
    count: int = 3
    label: str = "default"


class _PatchTarget(BaseModel):
    child: _Child | None = Field(default_factory=_Child)
    payload: dict[str, object] | None = None
    items: list[dict[str, int]] = Field(default_factory=list)
    validated_count: int = 0

    @model_validator(mode="after")
    def record_top_level_validation(self) -> Self:
        object.__setattr__(self, "validated_count", self.validated_count + 1)
        return self


class _OtherTarget(BaseModel):
    child: _Child | None = None


def _assignment(path: str, value: object, *, origin: str = "test", precedence: int = 0) -> PatchAssignment:
    return PatchAssignment(ParameterPath(tuple(path.split("."))), value, origin, precedence)


def _paths(*assignments: PatchAssignment) -> CompiledConfigPatch[_PatchTarget]:
    return CompiledConfigPatch.from_paths(_PatchTarget, assignments)


def test_mapping_leaf_with_nested_dictionaries_is_atomic_and_isolated() -> None:
    fallback = {"fallback": "name"}
    source = {"vars": {"template": {"given": ["first", fallback]}}}
    patch = CompiledConfigPatch.from_mapping(StepDefinition, source, origin="mapping", precedence=0)
    fallback["fallback"] = "changed"

    first = patch.apply()
    second = patch.apply()
    assert first.vars == {"template": {"given": ["first", {"fallback": "name"}]}}

    first.vars["template"]["given"][1]["fallback"] = "result-change"  # type: ignore[index]
    assert second.vars == {"template": {"given": ["first", {"fallback": "name"}]}}


def test_nested_nss_model_branch_patch_preserves_pii_global_siblings() -> None:
    base = PiiReplacerConfig.get_default_config()
    original_entities = deepcopy(base.globals.classify.entities)
    patch = CompiledConfigPatch.from_mapping(
        PiiReplacerConfig, {"globals": {"seed": 17}}, origin="override", precedence=1
    )

    result = patch.apply(base)

    assert result.globals.seed == 17
    assert result.globals.classify.entities == original_entities
    assert result.steps == base.steps


@pytest.mark.parametrize("path", ["payload.nested", "items.0", "child.count.value"])
def test_mapping_collection_and_scalar_leaves_reject_descendants(path: str) -> None:
    with pytest.raises(ParameterError, match=path.rsplit(".", 1)[0]):
        _paths(_assignment(path, "invalid"))


@pytest.mark.parametrize("reverse", [False, True], ids=["parent-first", "child-first"])
@pytest.mark.parametrize("parent", [{"label": "mapping"}, _Child(label="model")], ids=["mapping", "model"])
def test_compatible_branch_parent_and_child_merge_in_both_orders(parent: object, reverse: bool) -> None:
    assignments = [_assignment("child", parent), _assignment("child.count", 9)]
    if reverse:
        assignments.reverse()

    result = _paths(*assignments).apply()

    assert result.child is not None
    assert result.child.count == 9
    assert result.child.label in {"mapping", "model"}


@pytest.mark.parametrize("parent", [None, 4], ids=["none", "scalar"])
@pytest.mark.parametrize("reverse", [False, True], ids=["parent-first", "child-first"])
def test_atomic_ancestor_conflicts_are_input_order_independent(parent: object, reverse: bool) -> None:
    assignments = [_assignment("child", parent, origin="parent"), _assignment("child.count", 9, origin="child")]
    if reverse:
        assignments.reverse()

    with pytest.raises(ParameterError, match=r"parent/child.*child.*parent"):
        _paths(*assignments)


@pytest.mark.parametrize("reverse", [False, True], ids=["first-second", "second-first"])
def test_exact_duplicate_diagnostic_includes_path_and_origins_independent_of_order(reverse: bool) -> None:
    assignments = [
        _assignment("child.count", 4, origin="first"),
        _assignment("child.count", 5, origin="second"),
    ]
    if reverse:
        assignments.reverse()

    with pytest.raises(ParameterError, match=r"(?i)duplicate.*child\.count.*first.*second"):
        _paths(*assignments)


def test_higher_precedence_child_replaces_lower_atomic_parent() -> None:
    patch = _paths(
        _assignment("child", None, origin="file", precedence=0),
        _assignment("child.count", 11, origin="cli", precedence=1),
    )

    result = patch.apply()

    assert result.child is not None
    assert result.child.count == 11


def test_higher_precedence_atomic_parent_replaces_lower_child() -> None:
    patch = _paths(
        _assignment("child.count", 11, origin="file", precedence=0),
        _assignment("child", None, origin="cli", precedence=1),
    )

    assert patch.apply().child is None


def test_higher_precedence_parent_seed_wins_only_its_overlapping_children() -> None:
    patch = _paths(
        _assignment("child.count", 4, origin="file", precedence=0),
        _assignment("child.label", "file-label", origin="file", precedence=0),
        _assignment("child", {"count": 8}, origin="cli", precedence=1),
    )

    result = patch.apply()

    assert result.child == _Child(count=8, label="file-label")


def test_absence_explicit_none_and_explicit_default_remain_distinct() -> None:
    absent = CompiledConfigPatch.from_mapping(_PatchTarget, {}, origin="empty", precedence=0).apply()
    explicit_none = _paths(_assignment("child", None)).apply()
    explicit_default = _paths(_assignment("child.count", 3)).apply()

    assert "child" not in absent.model_fields_set
    assert explicit_none.child is None
    assert explicit_none.model_fields_set == {"child"}
    assert explicit_default.child is not None
    assert explicit_default.child.model_fields_set == {"count"}


def test_mapping_constructor_ignores_unknown_keys_at_each_model_level() -> None:
    patch = CompiledConfigPatch.from_mapping(
        _PatchTarget,
        {"unknown": True, "child": {"unknown": True}},
        origin="mapping",
        precedence=0,
    )

    result = patch.apply()

    assert result.model_dump(exclude_unset=True) == {"child": {}}


def test_path_constructor_remains_strict_for_unknown_canonical_path() -> None:
    with pytest.raises(ParameterError, match=r"Unknown configuration path 'unknown'"):
        _paths(_assignment("unknown", True))


def test_model_constructor_recursively_extracts_sparse_explicit_fields() -> None:
    source = _PatchTarget()
    assert source.model_fields_set == set()
    assert source.child is not None
    source.child.count = 12

    patch = CompiledConfigPatch.from_model(_PatchTarget, source, origin="base", precedence=0)
    result = patch.apply()

    assert result.child is not None
    assert result.child.count == 12
    assert result.model_dump(exclude_unset=True) == {"child": {"count": 12}}


def test_model_source_and_results_do_not_share_mutable_payloads() -> None:
    source = _PatchTarget(items=[{"value": 1}])
    patch = CompiledConfigPatch.from_model(_PatchTarget, source, origin="model", precedence=0)
    source.items[0]["value"] = 2

    first = patch.apply()
    second = patch.apply()
    first.items[0]["value"] = 3

    assert second.items == [{"value": 1}]


def test_top_level_validator_runs_at_application_boundary() -> None:
    result = _paths(_assignment("child.count", 6)).apply()

    assert result.validated_count == 1


def test_wrong_target_model_is_rejected_for_combine_and_apply() -> None:
    patch = _paths(_assignment("child.count", 6))
    other = CompiledConfigPatch.from_mapping(_OtherTarget, {}, origin="other", precedence=0)

    with pytest.raises(TypeError, match="target model"):
        patch.combine(other)  # ty: ignore[invalid-argument-type] -- runtime rejection is the contract
    with pytest.raises(TypeError, match="target model"):
        patch.apply(_OtherTarget())  # ty: ignore[invalid-argument-type] -- runtime rejection is the contract


def test_patch_schema_does_not_widen_public_pii_name_resolution() -> None:
    CompiledConfigPatch.from_mapping(
        SafeSynthesizerParameters,
        {"replace_pii": {"globals": {"seed": 3}}},
        origin="config",
        precedence=0,
    )

    assert isinstance(
        ParameterSchema.from_model(SafeSynthesizerParameters).resolve("replace_pii.globals.seed"), UnknownParameterName
    )
