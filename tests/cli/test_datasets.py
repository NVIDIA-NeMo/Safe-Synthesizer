# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from nemo_safe_synthesizer.cli.datasets import DatasetInfo, DatasetRegistry


class TestDatasetInfo:
    """Tests for DatasetInfo class."""

    def test_default_values(self):
        """Test that DatasetInfo has expected defaults."""
        info = DatasetInfo(name="test", url="test.csv")

        assert info.name == "test"
        assert info.url == "test.csv"
        assert info.overrides is None
        assert info.load_args is None
        assert info._registry is None

    def test_all_fields_populated(self):
        """Test DatasetInfo with all fields populated."""
        info = DatasetInfo(
            name="my-dataset",
            url="data/test.csv",
            overrides={"key": "value"},
            load_args={"sep": ";"},
        )

        assert info.name == "my-dataset"
        assert info.url == "data/test.csv"
        assert info.overrides == {"key": "value"}
        assert info.load_args == {"sep": ";"}

    def test_get_url_absolute_path(self):
        """Test get_url returns absolute path as-is."""
        info = DatasetInfo(name="test", url="/absolute/path/data.csv")

        assert info.get_url() == "/absolute/path/data.csv"

    def test_get_url_absolute_http_url(self):
        """Test get_url returns absolute http url as-is."""
        info = DatasetInfo(name="test", url="http://example.com/data.csv")

        assert info.get_url() == "http://example.com/data.csv"

    def test_get_url_absolute_https_url(self):
        """Test get_url returns absolute https url as-is."""
        info = DatasetInfo(name="test", url="https://example.com/data.csv")

        assert info.get_url() == "https://example.com/data.csv"

    def test_get_url_relative_path_no_registry(self):
        """Test get_url returns relative path as-is when no registry."""
        info = DatasetInfo(name="test", url="relative/data.csv")

        assert info.get_url() == "relative/data.csv"

    def test_get_url_relative_path_with_registry_no_base_url(self):
        """Test get_url returns relative path as-is when registry has no base_url."""
        registry = DatasetRegistry(datasets=[], base_url=None)
        info = DatasetInfo(name="test", url="relative/data.csv")
        info._registry = registry

        assert info.get_url() == "relative/data.csv"

    def test_get_url_relative_path_with_registry_and_base_url(self):
        """Test get_url joins with base_url for relative paths."""
        registry = DatasetRegistry(datasets=[], base_url="/base/path")
        info = DatasetInfo(name="test", url="relative/data.csv")
        info._registry = registry

        assert info.get_url() == "/base/path/relative/data.csv"

    def test_get_url_absolute_path_ignores_base_url(self):
        """Test get_url ignores base_url for absolute paths."""
        registry = DatasetRegistry(datasets=[], base_url="/base/path")
        info = DatasetInfo(name="test", url="/absolute/data.csv")
        info._registry = registry

        assert info.get_url() == "/absolute/data.csv"

    def test_fetch_csv_file(self, tmp_path: Path):
        """Test fetch reads CSV file correctly."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")

        info = DatasetInfo(name="test", url=str(csv_file))
        result = info.fetch()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 2

    def test_fetch_txt_file(self, tmp_path: Path):
        """Test fetch reads TXT file as CSV."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("col1,col2\na,b\nc,d\n")

        info = DatasetInfo(name="test", url=str(txt_file))
        result = info.fetch()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 2

    def test_fetch_json_file(self, tmp_path: Path):
        """Test fetch reads JSON file correctly."""
        json_file = tmp_path / "test.json"
        json_file.write_text('[{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]')

        info = DatasetInfo(name="test", url=str(json_file))
        result = info.fetch()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 2

    def test_fetch_jsonl_file(self, tmp_path: Path):
        """Test fetch reads JSONL file with lines=True."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"col1": 1, "col2": 2}\n{"col1": 3, "col2": 4}\n')

        info = DatasetInfo(name="test", url=str(jsonl_file))
        result = info.fetch()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 2

    def test_fetch_parquet_file(self, tmp_path: Path):
        """Test fetch reads Parquet file correctly."""
        parquet_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
        df.to_parquet(parquet_file)

        info = DatasetInfo(name="test", url=str(parquet_file))
        result = info.fetch()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 2

    def test_fetch_unsupported_extension(self, tmp_path: Path):
        """Test fetch raises ValueError for unsupported file extension."""
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<data></data>")

        info = DatasetInfo(name="test", url=str(xml_file))

        with pytest.raises(ValueError, match="Unsupported file extension: xml"):
            info.fetch()

    def test_fetch_no_extension(self, tmp_path: Path):
        """Test fetch raises ValueError when URL has no extension."""
        no_ext_file = tmp_path / "test_data"
        no_ext_file.write_text("some data")

        info = DatasetInfo(name="test", url=str(no_ext_file))

        with pytest.raises(ValueError, match="no extension found"):
            info.fetch()

    def test_fetch_with_custom_load_args(self, tmp_path: Path):
        """Test fetch passes custom load_args to reader."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1;col2\n1;2\n3;4\n")

        info = DatasetInfo(name="test", url=str(csv_file), load_args={"sep": ";"})
        result = info.fetch()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 2

    def test_fetch_uses_registry_base_url(self, tmp_path: Path):
        """Test fetch uses get_url() which includes base_url."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_file = data_dir / "test.csv"
        csv_file.write_text("col1,col2\n1,2\n")

        registry = DatasetRegistry(datasets=[], base_url=str(data_dir))
        info = DatasetInfo(name="test", url="test.csv")
        info._registry = registry

        result = info.fetch()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_get_url_with_http(self):
        """Test get_url returns http url as-is."""
        info = DatasetInfo(name="test", url="http://example.com/data.csv")

        assert info.get_url() == "http://example.com/data.csv"

    def test_get_url_uses_registry_base_http_url(self):
        """Test get_url uses registry base_url with relative urls."""
        registry = DatasetRegistry(datasets=[], base_url="http://example.com")
        info = DatasetInfo(name="test", url="data.csv")
        info._registry = registry

        assert info.get_url() == "http://example.com/data.csv"

    def test_get_url_uses_registry_base_http_url_trailing_slash(self):
        """Test get_url uses registry base_url with relative urls."""
        registry = DatasetRegistry(datasets=[], base_url="http://example.com/storage/")
        info = DatasetInfo(name="test", url="data.csv")
        info._registry = registry

        assert info.get_url() == "http://example.com/storage/data.csv"

    def test_get_url_absolute_http_url_ignores_registry_base_url(self):
        """Test get_url ignores registry base_url for absolute urls."""
        registry = DatasetRegistry(datasets=[], base_url="http://example.com")
        info = DatasetInfo(name="test", url="http://other.com/data.csv")
        info._registry = registry

        assert info.get_url() == "http://other.com/data.csv"

    def test_get_url_absolute_https_url_ignores_registry_base_url(self):
        """Test get_url ignores registry base_url for absolute urls."""
        registry = DatasetRegistry(datasets=[], base_url="/local/path/to/other/data")
        info = DatasetInfo(name="test", url="https://other.com/data.csv")
        info._registry = registry

        assert info.get_url() == "https://other.com/data.csv"

    def test_fetch_file_not_found_raises_error(self, tmp_path: Path):
        """Test fetch raises error when file does not exist."""
        info = DatasetInfo(name="test", url=str(tmp_path / "nonexistent.csv"))

        with pytest.raises(Exception):
            info.fetch()


class TestDatasetRegistry:
    """Tests for DatasetRegistry class."""

    def test_default_values(self):
        """Test that DatasetRegistry has expected defaults."""
        registry = DatasetRegistry()

        assert registry.datasets == []
        assert registry.base_url is None

    def test_with_datasets_and_base_url(self):
        """Test DatasetRegistry with datasets and base_url."""
        datasets = [
            DatasetInfo(name="ds1", url="data1.csv"),
            DatasetInfo(name="ds2", url="data2.csv"),
        ]
        registry = DatasetRegistry(datasets=datasets, base_url="/base/path")

        assert len(registry.datasets) == 2
        assert registry.base_url == "/base/path"

    def test_constructor_sets_registry_reference(self):
        """Test that constructor sets _registry on all datasets."""
        datasets = [
            DatasetInfo(name="ds1", url="data1.csv"),
            DatasetInfo(name="ds2", url="data2.csv"),
        ]
        registry = DatasetRegistry(datasets=datasets)

        for dataset in registry.datasets:
            assert dataset._registry is registry

    def test_get_dataset_returns_existing(self):
        """Test get_dataset returns existing dataset by name."""
        ds1 = DatasetInfo(name="my-dataset", url="data.csv", overrides={"key": "val"})
        registry = DatasetRegistry(datasets=[ds1])

        result = registry.get_dataset("my-dataset")

        assert result is ds1

    def test_get_dataset_creates_new_when_not_found(self):
        """Test get_dataset creates new DatasetInfo when name not found."""
        registry = DatasetRegistry(datasets=[])

        result = registry.get_dataset("new-dataset.csv")

        assert result.name == "new-dataset.csv"
        assert result.url == "new-dataset.csv"
        assert result.overrides is None
        assert result.load_args is None

    def test_get_dataset_adds_new_to_registry(self):
        """Test that newly created dataset is added to registry."""
        registry = DatasetRegistry(datasets=[])

        result = registry.get_dataset("new.csv")

        assert result in registry.datasets
        assert len(registry.datasets) == 1

    def test_get_dataset_new_has_no_registry_reference(self):
        """Test that newly created dataset does NOT have registry reference."""
        registry = DatasetRegistry(datasets=[], base_url="/some/base")

        result = registry.get_dataset("new.csv")

        # Deliberately not set - see docstring in datasets.py
        assert result._registry is None

    def test_get_dataset_existing_has_registry_reference(self):
        """Test that existing dataset has registry reference."""
        ds1 = DatasetInfo(name="existing", url="data.csv")
        registry = DatasetRegistry(datasets=[ds1], base_url="/some/base")

        result = registry.get_dataset("existing")

        assert result._registry is registry


class TestDatasetRegistryFromYaml:
    """Tests for DatasetRegistry.from_yaml() classmethod."""

    def test_from_yaml_loads_valid_file(self, tmp_path: Path):
        """Test from_yaml loads a valid YAML registry file."""
        yaml_file = tmp_path / "registry.yaml"
        yaml_file.write_text("""
datasets:
  - name: dataset1
    url: data/file1.csv
  - name: dataset2
    url: data/file2.json
    overrides:
      key: value
""")

        registry = DatasetRegistry.from_yaml(yaml_file)

        assert len(registry.datasets) == 2
        assert registry.datasets[0].name == "dataset1"
        assert registry.datasets[0].url == "data/file1.csv"
        assert registry.datasets[1].name == "dataset2"
        assert registry.datasets[1].overrides == {"key": "value"}

    def test_from_yaml_with_base_url(self, tmp_path: Path):
        """Test from_yaml loads base_url from file."""
        yaml_file = tmp_path / "registry.yaml"
        yaml_file.write_text("""
base_url: /shared/data
datasets:
  - name: ds1
    url: file1.csv
""")

        registry = DatasetRegistry.from_yaml(yaml_file)

        assert registry.base_url == "/shared/data"
        assert registry.datasets[0].get_url() == "/shared/data/file1.csv"

    def test_from_yaml_empty_datasets(self, tmp_path: Path):
        """Test from_yaml handles empty datasets list."""
        yaml_file = tmp_path / "registry.yaml"
        yaml_file.write_text("""
datasets: []
""")

        registry = DatasetRegistry.from_yaml(yaml_file)

        assert registry.datasets == []

    def test_from_yaml_file_not_found(self, tmp_path: Path):
        """Test from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            DatasetRegistry.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_accepts_string_path(self, tmp_path: Path):
        """Test from_yaml accepts string path argument."""
        yaml_file = tmp_path / "registry.yaml"
        yaml_file.write_text("""
datasets:
  - name: test
    url: test.csv
""")

        # Pass as string instead of Path
        registry = DatasetRegistry.from_yaml(str(yaml_file))

        assert len(registry.datasets) == 1

    def test_from_yaml_with_load_args(self, tmp_path: Path):
        """Test from_yaml loads load_args from file."""
        yaml_file = tmp_path / "registry.yaml"
        yaml_file.write_text("""
datasets:
  - name: semicolon-csv
    url: data.csv
    load_args:
      sep: ";"
      encoding: utf-8
""")

        registry = DatasetRegistry.from_yaml(yaml_file)

        assert registry.datasets[0].load_args == {"sep": ";", "encoding": "utf-8"}

    def test_from_yaml_sets_registry_reference(self, tmp_path: Path):
        """Test from_yaml sets registry reference on loaded datasets."""
        yaml_file = tmp_path / "registry.yaml"
        yaml_file.write_text("""
datasets:
  - name: test
    url: test.csv
""")

        registry = DatasetRegistry.from_yaml(yaml_file)

        assert registry.datasets[0]._registry is registry
