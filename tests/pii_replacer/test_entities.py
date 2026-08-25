# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EntitySpec registry ↔ catalog action alignment."""

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.config.replace_pii import ENTITY_BY_TYPE, EntityAction, EntityType
from nemo_safe_synthesizer.errors import InternalError
from nemo_safe_synthesizer.pii_replacer.entities import (
    ENTITY_REGISTRY,
    EntitySpec,
    _validate_apply_paths,
    entity_type_for_label,
    is_identify_only,
    is_propagate,
    spec,
)


@pytest.mark.unit
def test_replace_specs_require_apply_path():
    for entity_type, entity_spec in ENTITY_REGISTRY.items():
        action = ENTITY_BY_TYPE[entity_type].action
        if action is EntityAction.replace:
            assert entity_spec.apply_path in {"persona", "standalone_map"}
        else:
            assert entity_spec.apply_path is None


@pytest.mark.unit
def test_temporal_aliases_map_to_date_overlay():
    assert entity_type_for_label("datetime") is EntityType.date
    assert spec("datetime") is ENTITY_REGISTRY[EntityType.date]
    assert is_identify_only("datetime")


@pytest.mark.unit
def test_identify_only_and_propagate_follow_catalog_action():
    assert is_identify_only("gender")
    assert is_identify_only("date")
    assert not is_identify_only("ssn")
    assert is_propagate("free_text")
    assert not is_propagate("email")


@pytest.mark.unit
def test_validate_apply_paths_rejects_persona_on_identify_only():
    with pytest.raises(InternalError, match="require action=replace"):
        _validate_apply_paths(
            {
                EntityType.gender: EntitySpec(label=EntityType.gender, apply_path="persona"),
            }
        )


@pytest.mark.unit
def test_validate_apply_paths_rejects_missing_path_on_replace():
    with pytest.raises(InternalError, match="apply_path is None"):
        _validate_apply_paths(
            {
                EntityType.ssn: EntitySpec(label=EntityType.ssn, apply_path=None),
            }
        )
