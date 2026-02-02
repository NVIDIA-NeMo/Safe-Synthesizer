# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time

import pandas as pd
from openai import OpenAI

from nemo_safe_synthesizer.artifacts.analyzers.field_features import describe_field
from nemo_safe_synthesizer.artifacts.base.fields import FieldType
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.pii_replacer.data_editor.detect import (
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
from nemo_safe_synthesizer.pii_replacer.data_editor.edit import Editor, TransformFnAccounting
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics, TransformResult


def classify_config_from_params(
    config: PiiReplacerConfig,
) -> ClassifyConfig:
    """
    Parse out classification / NER parameters from config
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
    entity_extractor = EntityExtractorMulti.get_entity_extractor(clsfy_cfg)
    if clsfy_cfg.gliner_enabled:
        entity_extractor.add_entity_extractor(EntityExtractorGliner.get_entity_extractor(clsfy_cfg))
    if clsfy_cfg.ner_regexps_enabled:
        entity_extractor.add_entity_extractor(EntityExtractorRegexp.get_entity_extractor(clsfy_cfg))
    return entity_extractor


def _get_classify_endpoint_url() -> str:
    """Get the inference endpoint URL for column classification.

    The endpoint is expected to be set in the NIM_ENDPOINT_URL environment variable.

    Returns:
        The inference endpoint URL.

    Raises:
        Exception: If no valid configuration is found.
    """
    endpoint = os.environ.get("NIM_ENDPOINT_URL")
    if endpoint:
        return endpoint

    raise Exception("Inference endpoint not configured for classify. Set NIM_ENDPOINT_URL environment variable.")


def get_column_classifier() -> ColumnClassifierLLM:
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
"""Transform functions tracked for report accounting of which functions were used for which columns"""


def _build_column_statistics(
    classifications: dict | dict[str, dict[str, str]],
    transform_fn_accounting: TransformFnAccounting,
    column_report: NerReport,
) -> dict[str, ColumnStatistics]:
    """Build statistics for each column from various objects used by Editor."""

    columns = set(transform_fn_accounting.column_fns.keys()) | set(column_report.keys())

    return {
        column_name: ColumnStatistics(
            assigned_type=classifications["columns"].get(column_name, None),
            assigned_entity=classifications["entities"].get(column_name, None),
            detected_entity_counts={
                entity_name: entity_report.count
                for entity_name, entity_report in column_report.get(column_name, {}).items()
            },
            detected_entity_values={
                entity_name: entity_report.values
                for entity_name, entity_report in column_report.get(column_name, {}).items()
            },
            is_transformed=len(transform_fn_accounting.column_fns.get(column_name, set())) > 0,
            transform_functions=transform_fn_accounting.column_fns.get(column_name, set()),
        )
        for column_name in columns
    }


class NemoPII(object):
    """Class for performing PII replacement.

    Sample usage:

    ```python
    nemo_pii = NemoPII()
    result = nemo_pii.transform_df(df)
    print(result.transformed_df)
    print(result.column_statistics)
    ```
    """

    result: TransformResult

    def __init__(self, config: PiiReplacerConfig | None = None):
        """Initialize the NemoPII class.

        Args:
            config: Optional configuration for the PII replacer. If not provided, a default configuration will be used.
        """
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

    def classify_df(self, df: pd.DataFrame) -> dict[str, dict[str, str]]:
        """Classify the columns of a dataframe.

        Identifies both a column type and entity name for each column.

        Args:
            df: The dataframe to classify.

        Returns:
            A dictionary with two keys: "columns" and "entities". The "columns"
            key maps to a dictionary with column names as keys and detected
            type as values. The "entities" key maps to a dictionary with column
            names as keys and entity names as values.
        """
        try:
            column_classifier = get_column_classifier()
            columns = column_classifier.detect_types(df, self.classify_config.valid_entities)

            entities = {
                name: (entity if entity != UNKNOWN_ENTITY and entity in self.classify_config.valid_entities else None)
                for (name, entity) in columns.items()
            }
        except Exception:
            logging.error("Could not perform classify, falling back to default entities.", exc_info=False)
            entities = {}
            columns = {item: None for item in df.columns}

        # Use field type detection to identify text columns if not already
        # assigned an entity. These text columns are where NER is used if
        # enabled during transform_df.
        fields = [describe_field(field_name, df[field_name]) for field_name in df.columns]
        for field in fields:
            existing_type = columns.get(field.name, None)
            if (existing_type is None or existing_type.lower() == "none") and field.type == FieldType.TEXT:
                columns[field.name] = "text"

        return {
            "columns": columns,
            "entities": entities,
        }

    def transform_df(self, df: pd.DataFrame, classifications: dict | None = None) -> None:
        """Transform the dataframe by replacing PII.

        Args:
            df: The dataframe to transform.
            classifications: Optional classifications for the columns. If not
                provided, column classification will be performed.

        Returns:
            A TransformResult object containing the transformed dataframe and column statistics.
        """
        pii_replacer_start = time.monotonic()
        try:
            if not classifications:
                classifications = self.classify_df(df)

            transform_fn_accounting = TransformFnAccounting(ACCOUNTING_FUNCTIONS)

            transformed_df = self.editor.process_df(
                df, classifications["entities"], classifications["columns"], fnreport=transform_fn_accounting
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
