# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from io import StringIO

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing.dataset import make_json_schema
from nemo_safe_synthesizer.generation.regex_manager import (
    build_json_based_regex,
)

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


@pytest.fixture
def fixture_safe_synthesizer_config():
    return SafeSynthesizerParameters.from_params(
        group_training_examples_by=None,
        max_sequences_per_example=None,
        structured_generation_use_single_sequence=False,
    )


# Purpose: Validates exact regex generated for the Iris JSONL schema (5 records) and exercises behavior.
# Data: Schema with numeric fields and enum 'variety'; no grouping.
# Asserts: Full regex string equals legacy reference (regression guard) and re.fullmatch accepts valid rows and rejects invalid variants.
def test_build_json_based_regex(fixture_valid_iris_dataset_jsonl_and_schema, fixture_safe_synthesizer_config):
    _, schema = fixture_valid_iris_dataset_jsonl_and_schema

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Regex from older implementation (gretel-2025-02-25 tag)
    assert (
        regex
        == '(\\{"sepal\\.length":((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?,"sepal\\.width":((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?,"petal\\.length":((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?,"petal\\.width":(0\\.2),"variety":("Setosa")\\}\\n)+'
    )

    # Behavior checks: positive and negative matches
    assert (
        re.fullmatch(
            regex,
            '{"sepal.length":5.1,"sepal.width":3.5,"petal.length":1.4,"petal.width":0.2,"variety":"Setosa"}\n',
        )
        is not None
    )
    # Wrong variety should fail
    assert (
        re.fullmatch(
            regex,
            '{"sepal.length":5.1,"sepal.width":3.5,"petal.length":1.4,"petal.width":0.2,"variety":"Versicolor"}\n',
        )
        is None
    )
    # Wrong petal.width should fail (expects 0.2)
    assert (
        re.fullmatch(
            regex,
            '{"sepal.length":5.1,"sepal.width":3.5,"petal.length":1.4,"petal.width":0.3,"variety":"Setosa"}\n',
        )
        is None
    )


# Purpose: Ensures keys with special chars (e.g., '.') are escaped and string length bounds enforced.
# Data: Object with required key 'full.name', minLength=3, maxLength=10.
# Asserts: Equality against expected regex and behavioral matches via re.fullmatch for valid/invalid cases.
def test_build_json_based_regex_key_escaping(fixture_safe_synthesizer_config):
    schema = {
        "type": "object",
        "properties": {
            "full.name": {
                "type": "string",
                "minLength": 3,
                "maxLength": 10,
            }
        },
        "required": ["full.name"],
    }
    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Regex from older implementation (gretel-2025-02-25 tag), updated for outlines STRING_INNER changes:
    assert (
        regex
        == '(\\{"full\\.name":"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\/bfnrt]|\\\\u[0-9a-fA-F]{4}){3,10}"\\}\\n)+'
    )

    # NOTE: if writing re asserts, there are no spaces between keys and values
    # in the constructed regexes, and this is jsonl format, so must end with a
    # newline.
    assert re.fullmatch(regex, """{"full.name":"foo"}\n""") is not None
    assert re.fullmatch(regex, """{"full_name":"foo"}\n""") is None
    assert re.fullmatch(regex, """{"fullsname":"foo"}\n""") is None
    assert re.fullmatch(regex, """{"full.name":"areallylongname"})\n""") is None


# Purpose: Ensures enum string values with dots are escaped properly in the generated regex.
# Data: Object with required key 'foo' and enum values ["a.b", "bar", "bar.baz"].
# Asserts: Exact regex equals legacy reference and behavior via positive/negative re.fullmatch cases.
def test_build_json_based_regex_value_escaping(fixture_safe_synthesizer_config):
    schema = {
        "type": "object",
        "properties": {"foo": {"enum": ["a.b", "bar", "bar.baz"]}},
        "required": ["foo"],
    }
    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Regex from older implementation (gretel-2025-02-25 tag)
    assert regex == '(\\{"foo":("a\\.b"|"bar"|"bar\\.baz")\\}\\n)+'

    assert re.fullmatch(regex, """{"foo":"bar"}\n""") is not None
    assert re.fullmatch(regex, """{"foo":"a.b"}\n""") is not None
    assert re.fullmatch(regex, """{"fooo":"a.b"}\n""") is None
    assert re.fullmatch(regex, """{"foo":"a_b"})\n""") is None
    assert re.fullmatch(regex, """{"foo":"azb"})\n""") is None


# Purpose: Verifies enum handling when empty string is included as a valid value.
# Data: Object with required 'foo' and enum values ["a", "", "hello"].
# Asserts: Exact regex equals legacy reference and matches accept empty string while rejecting invalid forms.
def test_build_json_based_regex_enum_with_empty(fixture_safe_synthesizer_config):
    schema = {
        "type": "object",
        "properties": {"foo": {"enum": ["a", "", "hello"]}},
        "required": ["foo"],
    }
    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Regex from older implementation (gretel-2025-02-25 tag)
    assert regex == '(\\{"foo":("a"|""|"hello")\\}\\n)+'

    assert re.fullmatch(regex, """{"foo":"a"}\n""") is not None
    assert re.fullmatch(regex, """{"foo":""}\n""") is not None
    assert re.fullmatch(regex, """{"foo":"hello"}\n""") is not None
    assert re.fullmatch(regex, """{"foo":"a"}\n{"foo":""}\n""") is not None
    assert re.fullmatch(regex, """{"foo":null}\n""") is None
    assert re.fullmatch(regex, """{"foo":"aa"}\n""") is None
    assert re.fullmatch(regex, """{"foo":a}\n""") is None
    assert re.fullmatch(regex, """{"foo":}\n""") is None
    assert re.fullmatch(regex, """{}\n""") is None


# Purpose: Verifies enum handling when None is present; should be rendered as null (unquoted).
# Data: Object with required 'foo' and enum values ["a", None, "hello"].
# Asserts: Exact regex equals legacy reference and behavior via positive/negative matches for null semantics.
def test_build_json_based_regex_enum_with_none(fixture_safe_synthesizer_config):
    schema = {
        "type": "object",
        "properties": {"foo": {"enum": ["a", None, "hello"]}},
        "required": ["foo"],
    }

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    assert regex == '(\\{"foo":("a"|null|"hello")\\}\\n)+'

    assert re.fullmatch(regex, """{"foo":"a"}\n""") is not None
    assert re.fullmatch(regex, """{"foo":"hello"}\n""") is not None
    assert re.fullmatch(regex, """{"foo":null}\n""") is not None
    assert re.fullmatch(regex, """{"foo":"null"}\n""") is None
    assert re.fullmatch(regex, """{"foo":None}\n""") is None


# Purpose: Ensures optional properties and nullability are reflected when schema is inferred from data with missing values.
# Data: DataFrame with name (always present), age (sometimes None), gender (sometimes None) → inferred optional fields.
# Asserts: Exact regex equals legacy reference plus re.fullmatch behavior checks for accepted and rejected combinations.
def test_build_json_based_regex_with_missing(fixture_safe_synthesizer_config):
    df = pd.DataFrame(
        {
            "name": ["John", "Mary", "Joseph"],
            "age": [43, None, 61],
            "gender": [None, "F", "M"],
        }
    )

    schema = make_json_schema(df)
    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Regex from older implementation (gretel-2025-02-25 tag), updated formatting:
    assert (
        regex
        == '(\\{"name":"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\/bfnrt]|\\\\u[0-9a-fA-F]{4}){3,9}"(,"age":(((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?|null))?(,"gender":("([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\/bfnrt]|\\\\u[0-9a-fA-F]{4})*"|null))?\\}\\n)+'
    )

    # Behavior checks: accept required-only and optional combinations
    assert re.fullmatch(regex, '{"name":"John"}\n') is not None
    assert re.fullmatch(regex, '{"name":"Mary","age":43}\n') is not None
    assert re.fullmatch(regex, '{"name":"Joseph","gender":"M"}\n') is not None
    assert re.fullmatch(regex, '{"name":"Mary","age":null,"gender":null}\n') is not None

    # Negative cases: missing required, wrong types, invalid lengths
    assert re.fullmatch(regex, "{}\n") is None
    assert re.fullmatch(regex, '{"age":43}\n') is None
    assert re.fullmatch(regex, '{"name":"Jo"}\n') is None  # too short
    assert re.fullmatch(regex, '{"name":"ThisIsTooLong"}\n') is None  # too long
    assert re.fullmatch(regex, '{"name":"John","age":"43"}\n') is None  # wrong type


# Purpose: Validates grouped generation mode wraps groups in BOS/EOS and repeats group blocks, with behavior checks.
# Data: Simple enum schema; group_by=True; BOS/EOS tokens provided.
# Asserts: Exact grouped regex string including BOS/EOS and repetition structure; re.fullmatch accepts valid grouped content and rejects malformed/mismatched inner content.
def test_build_json_based_regex_with_groupby(fixture_safe_synthesizer_config):
    fixture_safe_synthesizer_config.data.group_training_examples_by = "id"

    schema = {
        "type": "object",
        "properties": {
            "foo": {"enum": ["a", "b", "c"]},
        },
        "required": ["foo"],
    }

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    assert regex == '(<s>(\\{"foo":("a"|"b"|"c")\\}\\n)+</s>\\n)+'

    # Behavior checks for grouped content
    grouped_ok = '<s>{"foo":"a"}\n{"foo":"b"}\n</s>\n'
    assert re.fullmatch(regex, grouped_ok) is not None

    grouped_bad_inner = '<s>{"foo":"x"}\n</s>\n'
    assert re.fullmatch(regex, grouped_bad_inner) is None

    grouped_missing_eos = '<s>{"foo":"a"}\n'
    assert re.fullmatch(regex, grouped_missing_eos) is None


# Purpose: Validates integer range inference and sign handling across mixed and all-negative datasets.
# Data: Case 1: numbers spanning -45..21; Case 2: all negative numbers -5..-34.
# Asserts: Exact regex equality plus positive/negative matches around inferred boundaries.
def test_build_json_regex_with_int(fixture_safe_synthesizer_config):
    # x_min negative and x_max positive, with -x_min > x_max
    df = pd.DataFrame({"number": [-5, 12, 3, -45, 9, -4, 17, 21, -12]})
    schema = make_json_schema(df)
    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )
    assert regex == '(\\{"number":(-)?(\\d|[1-3]\\d|4[0-5])\\}\\n)+'

    assert re.fullmatch(regex, """{"number":10}\n""") is not None
    assert re.fullmatch(regex, """{"number":0}\n""") is not None
    assert re.fullmatch(regex, """{"number":-10}\n""") is not None
    assert re.fullmatch(regex, """{"number":50}\n""") is None
    assert re.fullmatch(regex, """{"number":-50}\n""") is None

    # All negative
    df = pd.DataFrame({"number": [-5, -7, -9, -21, -34, -17]})
    schema = make_json_schema(df)
    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )
    assert regex == '(\\{"number":(-)([5-9]|[1-2]\\d|3[0-4])\\}\\n)+'

    assert re.fullmatch(regex, """{"number":-30}\n""") is not None
    assert re.fullmatch(regex, """{"number":30}\n""") is None
    assert re.fullmatch(regex, """{"number":-35}\n""") is None
    assert re.fullmatch(regex, """{"number":3}\n""") is None
    assert re.fullmatch(regex, """{"number":-3}\n""") is None


# Purpose: Validates error handling for invalid string length constraints in schema (minLength > maxLength).
# Data: Key 'full.name' with minLength=30 and maxLength=10.
# Asserts: build_json_based_regex raises ValueError.
def test_build_json_schema_str_bad_len(fixture_safe_synthesizer_config):
    schema = {
        "type": "object",
        "properties": {
            "full.name": {
                "type": "string",
                "minLength": 30,
                "maxLength": 10,
            }
        },
        "required": ["full.name"],
    }

    with pytest.raises(ValueError):
        build_json_based_regex(schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN)


# Purpose: Ensures mixed-type enum serialization: numbers unquoted, strings quoted and escaped, None → null.
# Data: Enum values [1, 2.5, "x", None].
# Asserts: Exact regex equals reference and behavior via representative matches.
def test_enum_regex(fixture_safe_synthesizer_config):
    schema = {
        "type": "object",
        "properties": {
            "v": {
                "enum": [1, 2.5, "x", None],
            }
        },
        "required": ["v"],
    }

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Enum should quote strings, escape dots, keep numbers unquoted, and map None -> null
    assert regex == '(\\{"v":(1|2\\.5|"x"|null)\\}\\n)+'

    assert re.fullmatch(regex, '{"v":1}\n') is not None
    assert re.fullmatch(regex, '{"v":2.5}\n') is not None
    assert re.fullmatch(regex, '{"v":"x"}\n') is not None
    assert re.fullmatch(regex, '{"v":null}\n') is not None

    assert re.fullmatch(regex, '{"v":"1"}\n') is None
    assert re.fullmatch(regex, '{"v":true}\n') is None
    assert re.fullmatch(regex, '{"v":}\n') is None
    assert re.fullmatch(regex, "{}\n") is None


# Purpose: Validates string constraints: length bounds and anchored pattern matching.
# Data: (1) minLength=2, maxLength=3; (2) pattern '^foo\\d{2}$'.
# Asserts: Accepts only allowed lengths and anchored patterns; rejects others.
def test_str_regex(fixture_safe_synthesizer_config):
    # Length-constrained string
    schema_len = {
        "type": "object",
        "properties": {"s": {"type": "string", "minLength": 2, "maxLength": 3}},
        "required": ["s"],
    }
    regex_len = build_json_based_regex(
        schema_len, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    assert re.fullmatch(regex_len, '{"s":"ab"}\n') is not None
    assert re.fullmatch(regex_len, '{"s":"abc"}\n') is not None
    assert re.fullmatch(regex_len, '{"s":"a"}\n') is None
    assert re.fullmatch(regex_len, '{"s":"abcd"}\n') is None

    # Pattern-constrained string (anchored)
    schema_pat = {
        "type": "object",
        "properties": {"s": {"type": "string", "pattern": r"^foo\\d{2}$"}},
        "required": ["s"],
    }
    regex_pat = build_json_based_regex(
        schema_pat, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    assert re.fullmatch(regex_pat, '{"s":"foo12"}\n') is not None
    assert re.fullmatch(regex_pat, '{"s":"foo1"}\n') is None
    assert re.fullmatch(regex_pat, '{"s":"bar12"}\n') is None
    assert re.fullmatch(regex_pat, '{"s":"xfoo12"}\n') is None
    assert re.fullmatch(regex_pat, '{"s":"foo12x"}\n') is None


# Purpose: Validates objects with additionalProperties integer type and property count limits.
# Data: Integers in [1,3]; minProperties=1, maxProperties=3.
# Asserts: Accepts 1..3 integer-valued props; rejects empty, too many, wrong type, or out-of-range values.
def test_obj_regex(fixture_safe_synthesizer_config):
    # additionalProperties with integer values in [1,3], between 1 and 3 properties
    schema = {
        "type": "object",
        "additionalProperties": {"type": "integer", "minimum": 1, "maximum": 3},
        "minProperties": 1,
        "maxProperties": 3,
    }

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Valid: 1..3 properties, integer values within range
    assert re.fullmatch(regex, '{"a":1}\n') is not None
    assert re.fullmatch(regex, '{"x":3}\n') is not None
    assert re.fullmatch(regex, '{"a":1,"b":2}\n') is not None
    assert re.fullmatch(regex, '{"a":1,"b":2,"c":3}\n') is not None

    # Invalid: 0 properties, too many properties, wrong types, out of range
    assert re.fullmatch(regex, "{}\n") is None
    assert re.fullmatch(regex, '{"a":1,"b":2,"c":3,"d":1}\n') is None
    assert re.fullmatch(regex, '{"a":"1"}\n') is None
    assert re.fullmatch(regex, '{"a":0}\n') is None
    assert re.fullmatch(regex, '{"a":4}\n') is None


# Purpose: Validates required vs optional property ordering and positioning rules.
# Data: Required 'b'=2; optional 'a'=1 may precede, 'c'=3 may follow.
# Asserts: Accepts allowed orderings; rejects missing required and disallowed optional placements.
def test_property_regex(fixture_safe_synthesizer_config):
    # Required/optional property ordering
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "minimum": 1, "maximum": 1},
            "b": {"type": "integer", "minimum": 2, "maximum": 2},
            "c": {"type": "integer", "minimum": 3, "maximum": 3},
        },
        "required": ["b"],
    }

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Valid: optional 'a' before required 'b', optional 'c' after required 'b'
    assert re.fullmatch(regex, '{"b":2}\n') is not None
    assert re.fullmatch(regex, '{"a":1,"b":2}\n') is not None
    assert re.fullmatch(regex, '{"b":2,"c":3}\n') is not None
    assert re.fullmatch(regex, '{"a":1,"b":2,"c":3}\n') is not None

    # Invalid: required missing, or optional properties in disallowed positions
    assert re.fullmatch(regex, '{"a":1}\n') is None
    assert re.fullmatch(regex, '{"c":3}\n') is None
    assert re.fullmatch(regex, '{"c":3,"b":2}\n') is None
    assert re.fullmatch(regex, '{"b":2,"a":1}\n') is None


# Purpose: Round-trip regression: DataFrame -> schema -> regex must match the original JSONL rows.
# Data: First 5 rows of Iris dataset serialized to JSONL; same schema used to build regex.
# Asserts: Non-grouped JSONL matches; grouped (BOS/EOS-wrapped) matches when group_by=True.
@pytest.mark.parametrize("group_by", [False, True])
def test_round_trip_dataframe_schema_regex_matches_iris(
    group_by,
    fixture_iris_dataset,
    fixture_safe_synthesizer_config,
):
    if group_by:
        fixture_safe_synthesizer_config.data.group_training_examples_by = "id"

    sample_df = pd.DataFrame(fixture_iris_dataset[:5])
    buf = StringIO()
    sample_df.to_json(buf, orient="records", lines=True)
    jsonl_str = buf.getvalue()

    schema = make_json_schema(sample_df)
    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    text = f"{BOS_TOKEN}{jsonl_str}{EOS_TOKEN}\n" if group_by else jsonl_str
    assert re.fullmatch(regex, text) is not None


# Purpose: Round-trip regression for doc summaries dataset (single text column): CSV -> DataFrame -> schema -> regex must match JSONL rows.
# Data: stub_datasets/doc_summaries.csv; each CSV record is quoted text in a single column (long-form string).
# Asserts: Non-grouped JSONL (one object per line) fully matches the produced regex.
def test_round_trip_doc_summaries_schema_regex_matches(fixture_doc_summaries_dataset, fixture_safe_synthesizer_config):
    sample_df = fixture_doc_summaries_dataset.copy()

    buf = StringIO()
    sample_df.to_json(buf, orient="records", lines=True)
    jsonl_str = buf.getvalue()

    schema = make_json_schema(sample_df)
    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    # Expect exact JSONL (without BOS/EOS) to match
    assert re.fullmatch(regex, jsonl_str) is not None


# Purpose: Validates single-sequence mode without grouping produces a bare record regex (no \n, no repetition).
# Data: Simple enum schema; structured_generation_use_single_sequence=True, max_sequences_per_example=1, no grouping.
# Asserts: Regex is the raw record pattern; matches a single record without trailing newline; rejects newline-terminated and multi-record inputs.
def test_single_sequence_no_grouping(fixture_safe_synthesizer_config):
    fixture_safe_synthesizer_config.generation.structured_generation_use_single_sequence = True
    fixture_safe_synthesizer_config.data.max_sequences_per_example = 1

    schema = {
        "type": "object",
        "properties": {"foo": {"enum": ["a", "b", "c"]}},
        "required": ["foo"],
    }

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    assert regex == '\\{"foo":("a"|"b"|"c")\\}'

    assert re.fullmatch(regex, '{"foo":"a"}') is not None

    assert re.fullmatch(regex, '{"foo":"a"}\n') is None
    assert re.fullmatch(regex, '{"foo":"a"}\n{"foo":"b"}\n') is None


# Purpose: Validates single-sequence mode with grouping produces a single BOS/EOS group (no trailing \n, no group repetition).
# Data: Simple enum schema; structured_generation_use_single_sequence=True, max_sequences_per_example=1, group_by="id".
# Asserts: Regex matches exactly one group; rejects trailing newline after EOS and multiple groups.
def test_single_sequence_with_groupby(fixture_safe_synthesizer_config):
    fixture_safe_synthesizer_config.generation.structured_generation_use_single_sequence = True
    fixture_safe_synthesizer_config.data.max_sequences_per_example = 1
    fixture_safe_synthesizer_config.data.group_training_examples_by = "id"

    schema = {
        "type": "object",
        "properties": {"foo": {"enum": ["a", "b", "c"]}},
        "required": ["foo"],
    }

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    assert regex == '<s>(\\{"foo":("a"|"b"|"c")\\}\\n)+</s>'

    assert re.fullmatch(regex, '<s>{"foo":"a"}\n</s>') is not None
    assert re.fullmatch(regex, '<s>{"foo":"a"}\n{"foo":"b"}\n</s>') is not None

    assert re.fullmatch(regex, '<s>{"foo":"a"}\n</s>\n') is None
    assert re.fullmatch(regex, '<s>{"foo":"a"}\n</s>\n<s>{"foo":"b"}\n</s>\n') is None


# Purpose: Validates that structured_generation_use_single_sequence is ignored when max_sequences_per_example != 1.
# Data: Simple enum schema; flag=True but max_sequences_per_example is None or 2.
# Asserts: Regex uses the repeated (...\n)+ form despite the flag being set.
@pytest.mark.parametrize("max_seq", [None, 2])
def test_single_sequence_flag_ignored_when_max_sequences_not_one(max_seq, fixture_safe_synthesizer_config):
    fixture_safe_synthesizer_config.generation.structured_generation_use_single_sequence = True
    fixture_safe_synthesizer_config.data.max_sequences_per_example = max_seq

    schema = {
        "type": "object",
        "properties": {"foo": {"enum": ["a", "b"]}},
        "required": ["foo"],
    }

    regex = build_json_based_regex(
        schema, config=fixture_safe_synthesizer_config, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN
    )

    assert regex == '(\\{"foo":("a"|"b")\\}\\n)+'
    assert re.fullmatch(regex, '{"foo":"a"}\n') is not None
    assert re.fullmatch(regex, '{"foo":"a"}\n{"foo":"b"}\n') is not None


# Purpose: Validates that BOS/EOS tokens containing regex metacharacters are properly escaped via re.escape().
# Data: Simple enum schema with grouping; BOS="[BOS]", EOS="[EOS]" (square brackets are regex metacharacters).
# Asserts: Escaped literals appear in regex; matches literal bracket tokens; rejects inputs where brackets are interpreted as character classes.
def test_bos_eos_tokens_with_special_chars_escaped(fixture_safe_synthesizer_config):
    fixture_safe_synthesizer_config.data.group_training_examples_by = "id"

    schema = {
        "type": "object",
        "properties": {"foo": {"enum": ["x"]}},
        "required": ["foo"],
    }

    regex = build_json_based_regex(
        schema,
        config=fixture_safe_synthesizer_config,
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    assert r"\[BOS\]" in regex
    assert r"\[EOS\]" in regex

    assert re.fullmatch(regex, '[BOS]{"foo":"x"}\n[EOS]\n') is not None

    assert re.fullmatch(regex, 'B{"foo":"x"}\nO\n') is None
