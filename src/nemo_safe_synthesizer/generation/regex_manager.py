# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re

from outlines_core.json_schema import (
    BOOLEAN,
    DATE,
    DATE_TIME,
    NULL,
    NUMBER,
    STRING,
    TIME,
    UUID,
)
from range_regex import bounded_regex_for_range

from ..observability import get_logger

logger = get_logger()


# JSON string inner pattern matching full JSON escape set (RFC 8259)
# Allows: \" \\ \/ \b \f \n \r \t and \uXXXX
JSON_STRING_INNER = r'([^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
JSON_STRING = rf'"{JSON_STRING_INNER}*"'


# Helper method not exported by outlines_core and outlines doesn't have it anymore
# past outlines==0.11.8
def _get_num_items_pattern(min_items, max_items, **kwargs):
    # Helper function for arrays and objects
    min_items = int(min_items or 0)
    if max_items is None:
        return rf"{{{max(min_items - 1, 0)},}}"
    else:
        max_items = int(max_items)
        if max_items < 1:
            return None
        return rf"{{{max(min_items - 1, 0)},{max_items - 1}}}"


def _build_object_key_prefix(name: str, whitespace_pattern: str) -> str:
    key_inner = json.dumps(name, ensure_ascii=True)[1:-1]
    return f'{whitespace_pattern}"{re.escape(key_inner)}"{whitespace_pattern}:{whitespace_pattern}'


def _properties_regex(instance, whitespace_pattern, **kwargs):
    regex = ""
    regex += r"\{"
    properties = instance["properties"]
    required_properties = instance.get("required", [])
    is_required = [item in required_properties for item in properties]
    # If at least one property is required, we include the one in the last position
    # without any comma.
    # For each property before it (optional or required), we add with a comma after the property.
    # For each property after it (optional), we add with a comma before the property.
    if any(is_required):
        last_required_pos = max([i for i, value in enumerate(is_required) if value])
        for i, (name, value) in enumerate(properties.items()):
            # Use JSON-escaped key name to align with df.to_json output
            subregex = _build_object_key_prefix(name, whitespace_pattern)

            # Skip fields that are always null
            if isinstance(value, dict) and "enum" in value and value["enum"] == [None]:
                subregex += "null"
            else:
                subregex += _build_regex(value, whitespace_pattern)

            if i < last_required_pos:
                subregex = f"{subregex}{whitespace_pattern},"
            elif i > last_required_pos:
                subregex = f"{whitespace_pattern},{subregex}"
            regex += subregex if is_required[i] else f"({subregex})?"

    # If no property is required, we have to create a possible pattern for each property in which
    # it's the last one necessarily present. Then, we add the others as optional before and after
    # following the same strategy as described above.
    # The whole block is made optional to allow the case in which no property is returned.
    else:
        property_subregexes = []
        for i, (name, value) in enumerate(properties.items()):
            # Use JSON-escaped key name to align with df.to_json output
            subregex = _build_object_key_prefix(name, whitespace_pattern)
            subregex += _build_regex(value, whitespace_pattern)
            property_subregexes.append(subregex)
        possible_patterns = []
        for i in range(len(property_subregexes)):
            pattern = ""
            for subregex in property_subregexes[:i]:
                pattern += f"({subregex}{whitespace_pattern},)?"
            pattern += property_subregexes[i]
            for subregex in property_subregexes[i + 1 :]:
                pattern += f"({whitespace_pattern},{subregex})?"
            possible_patterns.append(pattern)
        regex += f"({'|'.join(possible_patterns)})?"

    regex += f"{whitespace_pattern}" + r"\}"
    return regex


def _enum_regex(instance, **kwargs):
    """
    The enum keyword is used to restrict a value to a fixed set of values. It
    must be an array with at least one element, where each element is unique.
    """
    choices = []
    for choice in instance["enum"]:
        if isinstance(choice, bool):
            # JSON uses lowercase true/false
            choices.append("true" if choice else "false")
        elif isinstance(choice, (int, float)):
            choices.append(re.escape(str(choice)))
        elif isinstance(choice, str):
            # Match JSON-escaped string content
            inner = json.dumps(choice, ensure_ascii=True)[1:-1]
            choices.append(f'"{re.escape(inner)}"')
        elif choice is None:
            # make_json_schema represents a missing value in an enum type field
            # as None, but when we create json for this it's encoded as
            # "field":null (since we don't allow optional fields in the
            # encoding). So None gets converted to null here. This may not be
            # standard json schema, but is how TabFT works right now.
            choices.append(NULL)
        else:
            raise NotImplementedError(f"Unsupported type={type(choice)} in enum")

    return f"({'|'.join(choices)})"


def _string_type_regex(instance, **kwargs):
    if "maxLength" in instance or "minLength" in instance:
        max_items = instance.get("maxLength", "")
        min_items = instance.get("minLength", "")
        # note that the original code before the change to using outlines-core
        # had a peculiar try-catch in it.
        # https://github.com/dottxt-ai/outlines/commit/1b7ec18a91b6d39e0279dd91cfe9bab415dc80c4#diff-90728716bf5e6de30e84bb0588f3aac4a78a3ec0773564726f7ff65e1017b647
        # the try catch was always eating the ValueError and returning  a regex, as the except was always catching it.

        if int(max_items) < int(min_items):
            raise ValueError("maxLength must be greater than or equal to minLength")
        return f'"{JSON_STRING_INNER}{{{min_items},{max_items}}}"'
    elif pattern := instance.get("pattern"):
        # Normalize double backslashes (common in JSON Schema/text) to single for Python regex engine
        # so that patterns like "^foo\\d{2}$" behave as intended (\\d -> \d)
        normalized = pattern.replace("\\\\", "\\")
        # If schema pattern is anchored, anchor only the content between JSON quotes
        if normalized[0] == "^" and normalized[-1] == "$":
            return rf'"{normalized[1:-1]}"'
        else:
            return rf'"{normalized}"'
    elif format := instance.get("format", None):
        if format == "date-time":
            return DATE_TIME
        elif format == "uuid":
            return UUID
        elif format == "date":
            return DATE
        elif format == "time":
            return TIME
        else:
            raise NotImplementedError(f"Format {format} is not supported")

    return JSON_STRING


def _type_array_regex(instance, whitespace_pattern, **kwargs):
    num_repeats = _get_num_items_pattern(instance.get("minItems"), instance.get("maxItems"))
    if num_repeats is None:
        return rf"\[{whitespace_pattern}\]"

    allow_empty = "?" if int(instance.get("minItems", 0)) == 0 else ""

    if "items" in instance:
        items_regex = _build_regex(instance["items"], whitespace_pattern)
        return rf"\[{whitespace_pattern}(({items_regex})(,{whitespace_pattern}({items_regex})){num_repeats}){allow_empty}{whitespace_pattern}\]"
    else:
        # Here we need to make the choice to exclude generating list of objects
        # if the specification of the object is not given, even though a JSON
        # object that contains an object here would be valid under the specification.
        types = [
            {"type": "boolean"},
            {"type": "null"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "string"},
        ]
        regexes = [_build_regex(t, whitespace_pattern) for t in types]
        return rf"\[{whitespace_pattern}({'|'.join(regexes)})(,{whitespace_pattern}({'|'.join(regexes)})){num_repeats}){allow_empty}{whitespace_pattern}\]"


def _type_object_regex(instance, whitespace_pattern, **kwargs):
    # pattern for json object with values defined by instance["additionalProperties"]
    # enforces value type constraints recursively, "minProperties", and "maxProperties"
    # doesn't enforce "required", "dependencies", "propertyNames" "any/all/on Of"
    num_repeats = _get_num_items_pattern(
        instance.get("minProperties"),
        instance.get("maxProperties"),
    )
    if num_repeats is None:
        return rf"\{{{whitespace_pattern}\}}"

    allow_empty = "?" if int(instance.get("minProperties", 0)) == 0 else ""

    value_pattern = _build_regex(instance["additionalProperties"], whitespace_pattern)
    key_value_pattern = f"{STRING}{whitespace_pattern}:{whitespace_pattern}{value_pattern}"
    key_value_successor_pattern = f"{whitespace_pattern},{whitespace_pattern}{key_value_pattern}"
    multiple_key_value_pattern = f"({key_value_pattern}({key_value_successor_pattern}){num_repeats}){allow_empty}"

    return r"\{" + whitespace_pattern + multiple_key_value_pattern + whitespace_pattern + r"\}"


def _type_int_regex(instance, **kwargs):
    if "minimum" in instance and "maximum" in instance:
        min_int = int(instance["minimum"])
        max_int = int(instance["maximum"])

        # Function has issues with negative min, so we need to handle this case by case
        if min_int >= 0:  # both positive
            regex = bounded_regex_for_range(min_int, max_int).replace("\\b", "")
        elif max_int < 0:  # both negative
            # do minus sign plus x, with -x_max < x < -x_min
            regex = "(-)" + bounded_regex_for_range(-max_int, -min_int).replace("\\b", "")
        else:  # x_min negative and x_max positive
            # do |x| < max(x_max, -x_min)
            regex = "(-)?" + bounded_regex_for_range(0, max(max_int, -min_int)).replace("\\b", "")
    elif "minimum" in instance or "maximum" in instance:
        raise ValueError("both minimum and maximum need to be specified in schema")
    else:
        regex = NUMBER

    return regex


def _type_regex(instance, whitespace_pattern, **kwargs):
    """
    The type keyword may either be a string or an array:
    - If it's a string, it is the name of one of the basic types.
    - If it is an array, it must be an array of strings, where each string is
    the name of one of the basic types, and each element is unique. In this
    case, the JSON snippet is valid if it matches any of the given types.
    """
    instance_type = instance["type"]
    dispatch = {
        "string": _string_type_regex,
        "integer": _type_int_regex,
        "array": _type_array_regex,
        "object": _type_object_regex,
    }

    kwargs = {"instance": instance, "whitespace_pattern": whitespace_pattern}
    match instance_type:
        # match first on the list type as the other will attempt to hash it to index into
        # the dict
        case list(instance_type):
            # Here we need to make the choice to exclude generating an object
            # if the specification of the object is not give, even though a JSON
            # object that contains an object here would be valid under the specification.
            # TODO: it appears a list of types can still have additional
            # constraints like minimum or minLength, but make_json_schema
            # doesn't emit those right now, would be nice to emit those, but
            # then the handling here needs to change, alternately we use the
            # anyOf keyword as at
            # https://cswr.github.io/JsonSchema/spec/multiple_types/
            regexes = [_build_regex({"type": t}, whitespace_pattern) for t in instance_type if t != "object"]
            return rf"({'|'.join(regexes)})"
        case x if x in dispatch:
            return dispatch[x](**kwargs)

        case "number":
            return NUMBER
        case "boolean":
            return BOOLEAN
        case "null":
            return NULL
        case _:
            raise NotImplementedError(f"Unsupported type={instance_type}")


def _build_regex(instance: dict, whitespace_pattern: str, **kwargs) -> str:
    """
    Custom implementation of limited set of json schema to regex.

    We support what's needed for TabFT schemas, but this should not be
    considered a generic json to regex method.

    Notable missing features of JSON schema:
    - Handle `additionalProperties` keyword
    - Handle oneOf, anyOf, allOf keywords
    - Handle $ref keyword


    Copied with modifications from
    https://github.com/dottxt-ai/outlines/blob/0b4d12b0b9998a26e9dbde3bd558e695c51b75be/outlines/fsm/json_schema.py#L99

    Args:
        schema: The dict-based JSON schema used as a base for generating the
            regular expression.
        whitespace_pattern: String pattern to match whitespaces while
            constructing the regex.

    Returns:
        The constructed regex as a string.
    """
    match instance:
        case {"properties": _, **rest}:  # noqa: F841
            return _properties_regex(instance, whitespace_pattern, **kwargs)
        case {"enum": _, **rest}:  # noqa: F841
            return _enum_regex(instance, **kwargs)
        case {"type": _, **rest}:  # noqa: F841
            return _type_regex(instance, whitespace_pattern, **kwargs)
        case _:
            raise NotImplementedError()


def build_json_based_regex(
    schema: dict,
    bos_token: str,
    eos_token: str,
    whitespace_pattern: str | None = None,
    group_by: bool = False,
):
    """
    Builds a regular expression based on the provided JSON schema.

    Args:
        schema: The dict-based JSON schema used as a base for generating the
            regular expression.
        whitespace_pattern: An optional string pattern to match
            whitespaces while constructing the regex.
        group_by: The grouping token used to wrap the regex across
            multiple lines. Based on the tokenizer used outside of this.
    """
    whitespace_pattern = whitespace_pattern or ""

    json_regex = _build_regex(schema, whitespace_pattern)

    if group_by:
        json_lines_regex = rf"({bos_token}({json_regex}\n)+{eos_token}\n)+"
    else:
        json_lines_regex = rf"({json_regex}\n)+"

    return json_lines_regex
