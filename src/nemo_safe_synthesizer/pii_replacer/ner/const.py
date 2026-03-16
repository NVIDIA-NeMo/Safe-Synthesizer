# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Shared strings for core modules"""


class ConstDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


const = ConstDict(
    {
        "GRETEL_ID": "_gretel_id",
        "GRETEL_TS": "_gretel_ts",
        "GRETEL_SUB": "_gretel_subscriber",
        "TYPE": "type",
        "STR_VALUE": "string_value",
        "INT_VALUE": "int_value",
        "NUM_VALUE": "number_value",
        "BOOL_VALUE": "bool_value",
        "NULL_VALUE": "null_value",
        "ARR_COUNT": "array_count",
        "VALUE": "value",
        "TEXT": "text",
        "START": "start",
        "END": "end",
        "LABEL": "label",
        "SOURCE": "source",
        "FIELD": "field",
        "HOST": "host",
        "PORT": "port",
        "SSL": "ssl",
        "USER": "user",
        "PASS": "password",
        "VERSION": "version",
        "URL": "url",
        "NAME": "name",
        "ID": "id",
        "RECORD": "record",
        "NUMBER": "number",
        "INT": "integer",
        "ARRAY": "array",
        "STRING": "string",
        "TRUE": "true",
        "FALSE": "false",
        "NULL": "null",
        "ARRAY_POS": "_gretelarray_",
        "BOOL": "boolean",
        "DELIM": ".",
        "ENTITY": "entity",
        "DESCRIPTION": "description",
        "IS_BATCH": "is_batch",
        "CONTEXT": "context",
        "INSIGHTS_TIMEOUT": "insight_timeout",
        "INSIGHTS_BATCH": "insights_batch",
        "INSIGHTS_COUNT": "insights_count",
        "SCORE": "score",
        "NER_SCORE": "ner_score",
    }
)
