# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from ...data_processing.records.json_record import JsonObject, JSONRecord

JsonRecordInput: TypeAlias = str | JsonObject | JSONRecord
InData: TypeAlias = JsonRecordInput | Sequence[JsonRecordInput]


def input_to_json_records(in_data: InData) -> list[JSONRecord]:
    """Try and convert python objects to a list of Fields"""
    match in_data:
        case JSONRecord() as record:
            return [record]
        case (str() | dict()) as record:
            return [JSONRecord(record)]
        case list() as records:
            out: list[JSONRecord] = []
            for record in records:
                match record:
                    case JSONRecord() as json_record:
                        out.append(json_record)
                    case (str() | dict()) as raw_record:
                        out.append(JSONRecord(raw_record))
                    case _:
                        raise TypeError("Input data not supported.")
            return out
        case _:
            raise TypeError("Input data not supported.")


def is_string_a_number(value) -> bool:
    # Ensure value is a string to prevent iteration errors on non-string types
    value = str(value)
    return all(
        [
            char
            in [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                ".",
                "\n",
                "-",
                "E",
                "+",
                "e",
                " ",
            ]
            for char in value
        ]
    )  # noqa
