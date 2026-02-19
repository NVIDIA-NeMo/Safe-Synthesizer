# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ...data_processing.records.json_record import JSONRecord

"""Valid input data includes a str, tuple or dict"""
InData = str | list | dict | JSONRecord


def input_to_json_records(in_data: InData) -> list[JSONRecord]:
    """Try and convert python objects to a list of Fields"""
    if isinstance(in_data, JSONRecord):
        return [in_data]
    if isinstance(in_data, (str, dict)):
        return [JSONRecord(in_data)]
    if isinstance(in_data, list):
        out = []
        for record in in_data:
            if isinstance(record, JSONRecord):
                out.append(record)
            else:
                out.append(JSONRecord(record))
        return out
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
