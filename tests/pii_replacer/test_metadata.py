# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_safe_synthesizer.pii_replacer.ner.metadata import (
    DatasetMetadata,
    EntityMetadata,
    EntitySummary,
    FieldAttribute,
    FieldMetadata,
    TypeMetadata,
)


def test_metadata_dict_payloads_preserve_dataclass_shape():
    entity = EntityMetadata(
        label="email",
        count=2,
        f_ratio=0.5,
        approx_cardinality=2,
        sources=["regex"],
        field_label_f_ratio=1.0,
    )
    field = FieldMetadata(
        field="email_address",
        count=2,
        approx_cardinality=2,
        missing=0,
        pct_missing=0.0,
        pct_total_unique=100.0,
        s_score=1.0,
        entities=[entity],
        types=[TypeMetadata(type="str", count=2)],
        field_labels=["email"],
        field_attributes=[FieldAttribute.ID],
    )
    entity_summary = EntitySummary(
        label="email",
        fields=["email_address"],
        count=2,
        approx_distinct_count=2,
        sources=["regex"],
    )
    metadata = DatasetMetadata(project_record_count=2, total_field_count=1)
    metadata.add_field(field)
    metadata.add_entity(entity_summary)

    expected_field = {
        "field": "email_address",
        "count": 2,
        "approx_cardinality": 2,
        "missing": 0,
        "pct_missing": 0.0,
        "pct_total_unique": 100.0,
        "s_score": 1.0,
        "entities": [
            {
                "label": "email",
                "count": 2,
                "f_ratio": 0.5,
                "approx_cardinality": 2,
                "sources": ["regex"],
                "field_label_f_ratio": 1.0,
            }
        ],
        "types": [{"type": "str", "count": 2}],
        "field_labels": ["email"],
        "field_attributes": [FieldAttribute.ID],
    }
    expected_entity_summary = {
        "label": "email",
        "fields": ["email_address"],
        "count": 2,
        "approx_distinct_count": 2,
        "sources": ["regex"],
    }

    assert field.dict() == expected_field
    assert entity_summary.dict() == expected_entity_summary
    assert metadata.to_dict() == {
        "project_record_count": 2,
        "total_field_count": 1,
        "data": {
            "fields": [expected_field],
            "entities": [expected_entity_summary],
        },
    }
