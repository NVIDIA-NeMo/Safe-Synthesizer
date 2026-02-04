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
    column_name: str = Field()
    pii_type: str = Field(default=UNKNOWN_ENTITY)
    total_ref_data: int = Field(default=0)
    unique_ref_data: int = Field(default=0)
    total_synth_data: int = Field(default=0)
    unique_synth_data: int = Field(default=0)
    unique_synth_data_percentage: float = Field(default=0)


class PIIReplay(Component):
    name: str = Field(default="PII Replay")
    reference_total_records: int = Field(default=0)
    output_total_records: int = Field(default=0)
    pii_replay_data: list[PIIReplayData] = Field(default=list())

    @cached_property
    def jinja_context(self):
        d = super().jinja_context
        d["reference_total_records"] = self.reference_total_records
        d["output_total_records"] = self.output_total_records
        d["pii_types"] = list({datum.pii_type for datum in self.pii_replay_data})
        d["pii_replay_data"] = [datum.model_dump() for datum in self.pii_replay_data]
        return d

    @staticmethod
    def from_evaluation_dataset(evaluation_dataset, config: SafeSynthesizerParameters | None = None) -> PIIReplay:
        if evaluation_dataset.column_statistics is None or len(evaluation_dataset.column_statistics) == 0:
            logger.warning("No classified entities, skipping PII Replay.")
            return PIIReplay(score=EvaluationScore())

        pii_replay_entities = config.get("pii_replay_entities") if config else None
        pii_replay_columns = config.get("pii_replay_columns") if config else None

        # Build up a list of keys of tuple(col_name, entity_type)
        classified_entities = []
        for col, column_statistics in evaluation_dataset.column_statistics.items():
            entity_names = column_statistics.detected_entity_counts.keys()
            # Scope down to user supplied set of entities if there is one
            if pii_replay_entities:
                entity_names = set(entity_names).intersection(set(pii_replay_entities))
            classified_entities += [(col, entity_name) for entity_name in entity_names]
            # But add user specified set of columns as needed if there is one
            if pii_replay_columns:
                for user_specified_col in set(pii_replay_columns).difference(
                    set(evaluation_dataset.column_statistics.keys())
                ):
                    classified_entities.append((user_specified_col, UNKNOWN_ENTITY))

        pii_replay_data = []
        for ce in classified_entities:
            col, entity_name = ce[0], ce[1]

            # UNKNOWN_ENTITY case, use the entire column
            ref_entity_count = len(evaluation_dataset.reference[col])
            ref_entity_unique_values = evaluation_dataset.reference[col].unique()

            if entity_name != UNKNOWN_ENTITY:
                # Ideal case, use the count of detected entities in ref for that (col, entity_type).
                # Also get the set of unique values tagged with that entity type.
                ref_entity_count = evaluation_dataset.column_statistics[col].detected_entity_counts[entity_name]
                ref_entity_unique_values = evaluation_dataset.column_statistics[col].detected_entity_values[entity_name]

            # These are the same in both cases. We want the size of that set of ref[col] unique values.
            ref_entity_unique_count = len(ref_entity_unique_values)
            # We want the total number of rows in output[column] that contain some value from the ref[col] unique values.
            output_entity_values = (
                evaluation_dataset.output[col].to_frame().query(f"`{col}` in @ref_entity_unique_values")[col]
            )
            output_entity_count = len(output_entity_values)
            # With those query results, we also want to get the count of unique items in those filtered output results.
            output_entity_unique_count = len(output_entity_values.unique())

            pii_replay_data.append(
                PIIReplayData(
                    column_name=col,
                    pii_type=entity_name,
                    total_ref_data=ref_entity_count,
                    unique_ref_data=ref_entity_unique_count,
                    total_synth_data=output_entity_count,
                    unique_synth_data=output_entity_unique_count,
                    unique_synth_data_percentage=math.ceil(output_entity_unique_count / ref_entity_unique_count * 100),
                )
            )

        return PIIReplay(
            score=EvaluationScore(),
            reference_total_records=evaluation_dataset.reference.shape[0],
            output_total_records=evaluation_dataset.output.shape[0],
            pii_replay_data=pii_replay_data,
        )
