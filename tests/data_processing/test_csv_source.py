# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
from pathlib import Path

import httpx
import pytest

from nemo_safe_synthesizer.data_processing.csv_source import CsvSource

_CSV = "zip,note\n02054,N/A\n10001,\n"


@pytest.mark.unit
class TestCsvSource:
    def test_typed_and_text_reads_parse_the_same_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "input.csv"
        path.write_text(_CSV)

        source = CsvSource.load(str(path))
        path.write_text("zip,note\n1,changed\n")

        assert source.read()["zip"].tolist() == [2054, 10001]
        assert source.read_text().to_dict("list") == {"zip": ["02054", "10001"], "note": ["N/A", ""]}

    def test_memory_map_is_ignored_when_parsing_retained_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "input.csv"
        path.write_text(_CSV)

        source = CsvSource.load(str(path), {"memory_map": True})

        assert source.read()["zip"].tolist() == [2054, 10001]
        assert source.read_text()["zip"].tolist() == ["02054", "10001"]

    def test_compression_is_inferred_from_the_file_name(self, tmp_path: Path) -> None:
        path = tmp_path / "input.csv.gz"
        path.write_bytes(gzip.compress(_CSV.encode()))

        source = CsvSource.load(str(path))

        assert source.read_text()["zip"].tolist() == ["02054", "10001"]

    def test_file_url_is_read_once(self, tmp_path: Path) -> None:
        path = tmp_path / "input.csv"
        path.write_text(_CSV)

        source = CsvSource.load(path.as_uri())

        assert source.content == _CSV.encode()

    def test_http_url_is_fetched_once_with_storage_options_as_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests: list[httpx.Request] = []

        def fake_get(url: str, *, headers: dict[str, str] | None, **kwargs: object) -> httpx.Response:
            request = httpx.Request("GET", url, headers=headers)
            requests.append(request)
            return httpx.Response(200, content=gzip.compress(_CSV.encode()), request=request)

        monkeypatch.setattr(httpx, "get", fake_get)

        source = CsvSource.load(
            "https://example.test/data/input.csv.gz?version=2", {"storage_options": {"Authorization": "token"}}
        )

        assert source.read()["zip"].tolist() == [2054, 10001]
        assert source.read_text()["zip"].tolist() == ["02054", "10001"]
        assert [request.headers["Authorization"] for request in requests] == ["token"]

    def test_other_url_schemes_are_left_to_pandas(self) -> None:
        source = CsvSource.load("s3://bucket/input.csv")

        assert source.content is None
