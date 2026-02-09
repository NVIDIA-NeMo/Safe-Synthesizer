# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection of default settings for the Nemo Safe Synthesizer implementation."""

from pathlib import Path

# default artifacts path
DEFAULT_ARTIFACTS_PATH = Path("./safe-synthesizer-artifacts").resolve()
DEFAULT_CONFIG_SAVE_NAME = "safe-synthesizer-config.json"

# logging parameters
LOG_DASHES = "-" * 100
LOG_NUM_ERRORS = 10
MAX_ERR_COL_WIDTH = 80
# Human-readable JSON error messages, particularly for general schema validation
# categories in prod. Schema validation messages are based on
# https://json-schema.org/draft/2020-12/json-schema-validation#name-a-vocabulary-for-structural
HUMAN_READABLE_ERR_MSGS = {
    # JSON schema errors
    "type": "Invalid field type",
    "enum": "Invalid field value",
    "const": "Invalid field value",
    "multipleOf": "Field value must be a multiple of a given number",
    "maximum": "Field value must be less than or equal to a given number",
    "exclusiveMaximum": "Field value must be less than a given number",
    "minimum": "Field value must be greater than or equal to a given number",
    "exclusiveMinimum": "Field value must be greater than a given number",
    "maxLength": "Field value must be at most a given length",
    "minLength": "Field value must be at least a given length",
    "pattern": "Field value must match a given pattern",
    "maxItems": "Field value must have at most a given number of items",
    "minItems": "Field value must have at least a given number of items",
    "uniqueItems": "Field value must have unique items",
    "maxContains": "Field value must contain at most a given number of items",
    "minContains": "Field value must contain at least a given number of items",
    "maxProperties": "Field value must have at most a given number of properties",
    "minProperties": "Field value must have at least a given number of properties",
    "required": "Missing required field",
    "dependentRequired": "Missing required field based on another field",
    # JSON decode errors
    "Invalid JSON: Unterminated string starting at": "Invalid JSON: Unterminated string",
    # Groupby errors
    "groupby": "Groupby generation failed",
}

# project paths
PACKAGE_PATH = Path(__file__).parent
PROJECT_PATH = PACKAGE_PATH.parent
DEFAULT_BASE_OUTPUT_PATH = PACKAGE_PATH.parent / "local_data"

# evaluation parameters
EVAL_STEPS = 0.3
DEFAULT_VALID_RECORD_EVAL_BATCH_SIZE = 16
NUM_EVAL_BATCHES_TABULAR = 1
NUM_EVAL_BATCHES_GROUPED = 1

# timeseries

# Pseudo-group column name for single-sequence time series
# This column is added internally when no group column is specified,
# allowing unified processing of grouped and ungrouped time series.
# It is excluded from JSONL conversion so the model never sees it.
PSEUDO_GROUP_COLUMN = "__nss_sequence_id"

# training +  parameters
DEFAULT_BASE_SEQ_LENGTH = 2048
MAX_ROPE_SCALING_FACTOR = 6
DEFAULT_MAX_SEQ_LENGTH = DEFAULT_BASE_SEQ_LENGTH * MAX_ROPE_SCALING_FACTOR

DEFAULT_PRETRAINED_MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0"
PROMPT_TEMPLATE = "[INST] {instruction} {schema} [/INST]{prefill}"
DEFAULT_INSTRUCTION = "Generate a JSONL dataset with the following columns: "
DEFAULT_SAMPLING_PARAMETERS = {
    "repetition_penalty": 1.0,
    "temperature": 0.9,
    "top_k": 0,
    "top_p": 1,
}
MAX_NUM_PROMPTS_PER_BATCH = 100
TRAIN_SET_SIZE_BUFFER = 100

# training examples
NUM_OVERLAP_RECORDS = 3

FIXED_RUNTIME_LORA_ARGS = {"use_rslora": True}
FIXED_RUNTIME_GENERATE_ARGS = {"top_k": -1, "min_p": 0}
RUNTIME_MODEL_ARCHIVE_NAME = "model"
RUNTIME_MODEL_CONFIG_NAME = "safe-synthesizer-config"

# miscellaneous
EPS = 1e-15
NUM_SPECIAL_TOKENS = 2
DEFAULT_CACHE_PREFIX = "safe-synthesizer-dataset-cache"
DEFAULT_ATTN_IMPLEMENTATION = "kernels-community/vllm-flash-attn3"
BACKUP_ATTN_IMPLEMENTATION = "spda"
