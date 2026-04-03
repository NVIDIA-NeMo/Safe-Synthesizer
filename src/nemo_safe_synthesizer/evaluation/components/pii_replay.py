# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from functools import cached_property

from pydantic import BaseModel, Field

from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger

logger = get_logger(__name__)

UNKNOWN_ENTITY: str = "none"


class PIIReplayData(BaseModel):
    """Per-column PII data listed in the PII Replay section of the SQS report."""

    column_name: str = Field(description="The name of the column with PII data.")
    column_assigned_type: str = Field(
        description="The assigned type for the column (text, unique identifier, date, email, etc.)."
    )
    pii_type: str = Field(
        default=UNKNOWN_ENTITY,
        description="Type of the PII data in the column. For non-text fields, same as column_assigned_type. For text fields, the PII entities detected within the text (race, SSN, address, etc.).",
    )
    total_training_data: int = Field(default=0, description="Total rows in the training data that contain PII values.")
    unique_training_data: int = Field(
        default=0, description="Count of distinct PII values for this entity in the training column."
    )
    total_synthetic_data: int = Field(
        default=0, description="Number of synthetic rows whose column value matches a training PII value."
    )
    unique_synthetic_data: int = Field(
        default=0, description="Count of distinct training PII values that appear in the synthetic column."
    )
    unique_synthetic_data_percentage: float = Field(
        default=0,
        description="Percentage of distinct training PII values replayed in the synthetic data (unique_synthetic_data / unique_training_data * 100).",
    )


class PIIReplay(Component):
    """PII Replay metric -- counts PII values from the training data appearing in the synthetic data.

    For each classified PII entity, reports total and unique replay counts.
    This component does not produce a numeric score; it surfaces PII
    leakage details for the HTML report.
    """

    name: str = Field(default="PII Replay")
    training_total_records: int = Field(default=0, description="Total rows in the training data.")
    synthetic_total_records: int = Field(default=0, description="Total rows in the synthetic data.")
    pii_replay_data: list[PIIReplayData] = Field(
        default=list(), description="Per-column / per-entity replay statistics."
    )

    @cached_property
    def jinja_context(self):
        """Template context with PII replay statistics and entity type list."""
        d = super().jinja_context
        d["training_total_records"] = self.training_total_records
        d["synthetic_total_records"] = self.synthetic_total_records
        d["pii_types"] = list({datum.pii_type for datum in self.pii_replay_data})
        d["pii_replay_data"] = [datum.model_dump() for datum in self.pii_replay_data]
        return d

    @staticmethod
    def from_evaluation_datasets(evaluation_datasets, config: SafeSynthesizerParameters | None = None) -> PIIReplay:
        """Compute PII replay counts from classified entity metadata."""
        if evaluation_datasets.column_statistics is None or len(evaluation_datasets.column_statistics) == 0:
            logger.warning("No classified entities, skipping PII Replay.")
            return PIIReplay(score=EvaluationScore())

        pii_replay_entities = config.get("pii_replay_entities") if config else None
        pii_replay_columns = config.get("pii_replay_columns") if config else None

        # Build up a list of keys of tuple(col_name, entity_type)
        classified_entities = []
        for col, column_statistics in evaluation_datasets.column_statistics.items():
            entity_names = column_statistics.detected_entity_counts.keys()
            entity_assigned_type = column_statistics.assigned_type
            # Scope down to user supplied set of entities if there is one
            if pii_replay_entities:
                entity_names = set(entity_names).intersection(set(pii_replay_entities))
            classified_entities += [(col, entity_name, entity_assigned_type) for entity_name in entity_names]
            # But add user specified set of columns as needed if there is one
            if pii_replay_columns:
                for user_specified_col in set(pii_replay_columns).difference(
                    set(evaluation_datasets.column_statistics.keys())
                ):
                    classified_entities.append((user_specified_col, UNKNOWN_ENTITY))

        pii_replay_data = []
        for col, entity_name, entity_assigned_type in classified_entities:
            # UNKNOWN_ENTITY case, use the entire column
            training_entity_count = len(evaluation_datasets.training[col])
            training_entity_unique_values = evaluation_datasets.training[col].unique()

            if entity_name != UNKNOWN_ENTITY:
                # Ideal case, use the count of detected entities in training for that (col, entity_type).
                # Also get the set of unique values tagged with that entity type.
                training_entity_count = evaluation_datasets.column_statistics[col].detected_entity_counts[entity_name]
                training_entity_unique_values = evaluation_datasets.column_statistics[col].detected_entity_values[
                    entity_name
                ]

            # These are the same in both cases. We want the size of that set of training[col] unique values.
            training_entity_unique_count = len(training_entity_unique_values)
            # We want the total number of rows in synthetic[column] that contain some value from the training[col] unique values.
            synthetic_entity_values = (
                evaluation_datasets.synthetic[col].to_frame().query(f"`{col}` in @training_entity_unique_values")[col]
            )
            synthetic_entity_count = len(synthetic_entity_values)
            # With those query results, we also want the count of unique items in those filtered synthetic results.
            synthetic_entity_unique_count = len(synthetic_entity_values.unique())

            pii_replay_data.append(
                PIIReplayData(
                    column_name=col,
                    column_assigned_type=entity_assigned_type,
                    pii_type=entity_name,
                    total_training_data=training_entity_count,
                    unique_training_data=training_entity_unique_count,
                    total_synthetic_data=synthetic_entity_count,
                    unique_synthetic_data=synthetic_entity_unique_count,
                    unique_synthetic_data_percentage=math.ceil(
                        synthetic_entity_unique_count / training_entity_unique_count * 100
                    ),
                )
            )

        return PIIReplay(
            score=EvaluationScore(),
            training_total_records=evaluation_datasets.training.shape[0],
            synthetic_total_records=evaluation_datasets.synthetic.shape[0],
            pii_replay_data=pii_replay_data,
        )
