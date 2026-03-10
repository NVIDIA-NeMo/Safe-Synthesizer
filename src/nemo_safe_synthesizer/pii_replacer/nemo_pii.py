# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from ..artifacts.analyzers.field_features import describe_field
from ..artifacts.base.fields import FieldType
from ..config.replace_pii import PiiReplacerConfig
from ..pii_replacer.data_editor.edit import Editor, TransformFnAccounting
from ..pii_replacer.transform_result import ColumnStatistics, TransformResult
from .data_editor.detect import (
    DEFAULT_ENTITIES,
    UNKNOWN_ENTITY,
    ClassifyConfig,
    ColumnClassifierLLM,
    EntityExtractor,
    EntityExtractorGliner,
    EntityExtractorMulti,
    EntityExtractorRegexp,
    NerReport,
)


class ColumnClassification(BaseModel):
    """Classification and detected-entity info for a column prior to transform.

    When ``entity`` is ``None`` (e.g. unclassified), ``entity_count``
    is ``None`` and ``entity_values`` is an empty list.
    """

    field_name: str = Field(description="Name of the field/column.")
    column_type: str | None = Field(
        default=None,
        description="Detected column type (e.g. ``text``, ``numeric``).",
    )
    entity: str | None = Field(
        default=None,
        description="Detected entity type (e.g. ``email``, ``phone``), or ``None`` if none.",
    )
    entity_count: int | None = Field(
        default=None,
        description="Number of non-empty values in this field. ``None`` if no entity detected.",
    )
    entity_values: list[Any] = Field(
        default_factory=list,
        description="Unique values for this field. Empty if no entity detected.",
    )


def classify_config_from_params(
    config: PiiReplacerConfig,
) -> ClassifyConfig:
    """Build classification and NER config from PII replacer config.

    Args:
        config: PII replacer config containing globals for classify and NER.

    Returns:
        ``ClassifyConfig`` with valid entities, NER settings, and GLiNER options.
    """
    valid_entities = DEFAULT_ENTITIES

    if config.globals.classify.entities is not None:
        valid_entities = set(config.globals.classify.entities)

    ner_entities = valid_entities
    if config.globals.ner.ner_entities is not None:
        ner_entities = set(config.globals.ner.ner_entities)

    cc = ClassifyConfig(
        valid_entities=valid_entities,
        ner_threshold=config.globals.ner.ner_threshold,
        ner_regexps_enabled=config.globals.ner.enable_regexps,
        ner_entities=ner_entities,
        gliner_enabled=config.globals.ner.gliner.enable_gliner,
        gliner_batch_mode_enabled=config.globals.ner.gliner.enable_batch_mode,
        gliner_batch_mode_chunk_length=config.globals.ner.gliner.chunk_length,
        gliner_batch_mode_batch_size=config.globals.ner.gliner.batch_size,
        gliner_model=config.globals.ner.gliner.gliner_model,
    )

    return cc


def build_entity_extractor(clsfy_cfg: ClassifyConfig) -> EntityExtractor:
    """Build a composite entity extractor from classification config."""
    entity_extractor = EntityExtractorMulti.get_entity_extractor(clsfy_cfg)
    if clsfy_cfg.gliner_enabled:
        entity_extractor.add_entity_extractor(EntityExtractorGliner.get_entity_extractor(clsfy_cfg))
    if clsfy_cfg.ner_regexps_enabled:
        entity_extractor.add_entity_extractor(EntityExtractorRegexp.get_entity_extractor(clsfy_cfg))
    return entity_extractor


def _get_classify_endpoint_url() -> str:
    """Return the inference endpoint URL for column classification.

    Reads ``NIM_ENDPOINT_URL`` from the environment.

    Returns:
        The endpoint URL string.

    Raises:
        Exception: If ``NIM_ENDPOINT_URL`` is not set.
    """
    endpoint = os.environ.get("NIM_ENDPOINT_URL")
    if endpoint:
        return endpoint

    raise Exception("Inference endpoint not configured for classify. Set NIM_ENDPOINT_URL environment variable.")


def get_column_classifier() -> ColumnClassifierLLM:
    """Return a column classifier backed by the NIM endpoint (``NIM_ENDPOINT_URL``, ``NIM_API_KEY``)."""
    classifier = ColumnClassifierLLM()
    classifier._num_samples = 5

    endpoint = _get_classify_endpoint_url()

    # When using Inference Gateway, no API key is needed (gateway handles auth).
    # For legacy direct endpoint, NIM_API_KEY can be provided.
    api_key = os.environ.get("NIM_API_KEY", "not-needed")

    classifier._llm = OpenAI(api_key=api_key, base_url=endpoint)
    return classifier


ACCOUNTING_FUNCTIONS = [
    "re",
    "fake",
    "random",
    "hash",
    "normalize",
    "partial_mask",
    "tld",
    "date_shift",
    "date_time_shift",
    "date_format",
    "date_time_format",
    "detect_entities",
    "redact_entities",
    "label_entities",
    "hash_entities",
    "fake_entities",
    "drop",
]
"""Transform function names tracked for report accounting (which functions were used per column)."""


def _build_column_statistics(
    classifications: list[ColumnClassification],
    transform_fn_accounting: TransformFnAccounting,
    column_report: NerReport,
) -> dict[str, ColumnStatistics]:
    """Build per-column statistics from classification, accounting, and NER report.

    Args:
        classifications: Per-column classification (type, entity, counts, values).
        transform_fn_accounting: Which transform functions were applied per column.
        column_report: NER report with detected entities for text columns.

    Returns:
        Map of column name to ``ColumnStatistics``.
    """
    result = {}
    for field in classifications:
        column_name = field.field_name

        # Prepare entity detection data based on column type
        # column_report only includes the entities detected in text columns.
        # ColumnClassification object includes entities for non text fields.
        detected_counts = {}
        detected_values = {}
        # Assigning detected_entity_counts and detected_entity_values only for text columns or fields with PII entities.
        if field.column_type == "text":
            column_entities = column_report.get(column_name, {})
            for entity_name, entity_report in column_entities.items():
                detected_counts[entity_name] = entity_report.count
                detected_values[entity_name] = entity_report.values
        elif field.entity is not None:
            # Non-text column with detected entity
            detected_counts[field.entity] = field.entity_count
            detected_values[field.entity] = set(field.entity_values)

        # Get transform functions for this column
        transform_fns = transform_fn_accounting.column_fns.get(column_name, set())
        # Create ColumnStatistics for this column and add to result
        result[column_name] = ColumnStatistics(
            assigned_type=field.column_type,
            assigned_entity=field.entity,
            detected_entity_counts=detected_counts,
            detected_entity_values=detected_values,
            is_transformed=bool(transform_fns),
            transform_functions=transform_fns,
        )
    return result


class NemoPII(object):
    """PII replacement over DataFrames via classification, NER, and configurable transforms.

    Call ``classify_df`` to get column classifications, then ``transform_df`` to replace
    PII. The result and per-column statistics are on ``result`` after ``transform_df``.

    Args:
        config: PII replacer config. If ``None``, default config is used.

    Attributes:
        result: Result of the last ``transform_df`` (``TransformResult`` with ``transformed_df`` and ``column_statistics``).

    Example:
        >>> nemo_pii = NemoPII()
        >>> nemo_pii.transform_df(df)
        >>> result = nemo_pii.result
        >>> print(result.transformed_df)
        >>> print(result.column_statistics)
    """

    result: TransformResult

    def __init__(self, config: PiiReplacerConfig | None = None):
        if config:
            self.pii_replacer_config = config
        else:
            self.pii_replacer_config = PiiReplacerConfig.get_default_config()

        self.classify_config = classify_config_from_params(self.pii_replacer_config)

        # TODO: clean up to use pydantic model directly or something typed
        # internally, for now just convert to dict to match existing code.
        self.data_editor_config = self.pii_replacer_config.model_dump()

        self.entity_extractor = build_entity_extractor(self.classify_config)
        self.editor = Editor(self.data_editor_config, self.entity_extractor)
        self.elapsed_time = 0.0

    def classify_df(self, df: pd.DataFrame) -> list[ColumnClassification]:
        """Classify each column (type and entity) using config and optional LLM classifier.

        Args:
            df: DataFrame to classify.

        Returns:
            List of ``ColumnClassification``, one per column, with field name, column type,
            entity, entity count, and unique entity values.
        """
        # Pre-initialize with defaults
        entities = {}
        columns = {item: None for item in df.columns}

        try:
            # Only attempt classification if enabled
            if self.pii_replacer_config.globals.classify.enable_classify is not False:
                column_classifier = None

                # Try to initialize the column classifier
                try:
                    column_classifier = get_column_classifier()
                except Exception:
                    logging.error(
                        "Could not initialize column classifier, falling back to default entities.", exc_info=False
                    )

                # Try to perform classification if we successfully got a classifier
                if column_classifier is not None:
                    try:
                        columns = column_classifier.detect_types(df, self.classify_config.valid_entities)

                        entities = {
                            name: (
                                entity
                                if entity != UNKNOWN_ENTITY and entity in self.classify_config.valid_entities
                                else None
                            )
                            for (name, entity) in columns.items()
                        }
                    except Exception:
                        logging.error("Could not perform classify, falling back to default entities.", exc_info=False)
            else:
                logging.info("Column classification is disabled (enable_classify=False), skipping classify call.")
        finally:
            # Use field type detection to identify text columns if not already
            # assigned an entity. These text columns are where NER is used if
            # enabled during transform_df.
            field_results = []
            fields = [describe_field(field_name, df[field_name]) for field_name in df.columns]
            for field in fields:
                entity_count = field.count
                entity_values = field.unique_values_list
                entity = entities.get(field.name, None)
                existing_type = columns.get(field.name, None)
                # Determine column type
                is_text_without_type = (
                    existing_type is None or existing_type.lower() == "none"
                ) and field.type == FieldType.TEXT
                column_type = "text" if is_text_without_type else existing_type

                # Missing entities, are not expected to have entity_count and entity_values.
                if entity is None:
                    entity_count = None
                    entity_values = []
                field_results.append(
                    ColumnClassification(
                        field_name=field.name,
                        column_type=column_type,
                        entity=entity,  # currently this is None for text fields.
                        entity_count=entity_count,
                        entity_values=entity_values,
                    )
                )
        return field_results

    def transform_df(self, df: pd.DataFrame, classifications: list[ColumnClassification] | None = None) -> None:
        """Replace PII in the DataFrame and set ``self.result``.

        Args:
            df: DataFrame to transform.
            classifications: Optional precomputed classifications. If ``None``,
                ``classify_df`` is run first.
        """
        pii_replacer_start = time.monotonic()
        try:
            if not classifications:
                classifications = self.classify_df(df)

            # Convert classification result to entities and column types dicts for editor
            column_types_dict = {field.field_name: field.column_type for field in classifications}
            entities_dict = {field.field_name: field.entity for field in classifications}
            transform_fn_accounting = TransformFnAccounting(ACCOUNTING_FUNCTIONS)

            transformed_df = self.editor.process_df(
                df, entities_dict, column_types_dict, fnreport=transform_fn_accounting
            )
            self.result = TransformResult(
                transformed_df=transformed_df,
                column_statistics=_build_column_statistics(
                    classifications, transform_fn_accounting, self.entity_extractor.column_report
                ),
            )
        except Exception as e:
            logging.exception("Error transforming dataframe")
            raise e

        finally:
            self.elapsed_time = time.monotonic() - pii_replacer_start
