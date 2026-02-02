# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset loading utilities for the CLI.

This module provides the DatasetInfo dataclass for loading datasets from
URLs or file paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ..observability import get_logger

logger = get_logger(__name__)


class DatasetInfo(BaseModel):
    """Entry in the dataset registry."""

    name: str
    """Short name of the dataset.

    Used to fetch the dataset from the registry.
    """

    url: str
    """URL or path to the dataset.

    If a relative path, it is joined with the base_url from the registry if
    present.
    """

    overrides: dict[str, Any] | None = None
    """Config overrides for this dataset.

    These overrides take precendence over the values from the config file in
    the CLI, but are themselves overridden by any CLI args specifying config
    parameters.
    """

    load_args: dict[str, Any] | None = None
    """Extra arguments needed by the data reader for a this dataset."""

    _registry: DatasetRegistry | None = None
    """Private attribute to keep a reference to an associated registry.

    Pydantic does not include this in the model schema, it is not serialized or
    deserialized, etc., so we manually manage it in the DatasetRegistry class.
    """

    def get_url(self) -> str:
        """Get url for the dataset with base_url from registry added if appropriate.

        If self.url is a relative path, it is joined with the base_url from the registry if present.
        Otherwise, self.url is returned as is.

        Returns:
            The realized url for the dataset.
        """
        # pathlib.Path will collapse double slashes, so we need to check for
        # http(s) urls explicitly and not always use Path. For this reason we
        # also return a string instead of a Path object.
        if self.url.startswith(("http://", "https://")):
            # URLs are absolute, so we return them as is.
            return self.url

        url = Path(self.url)
        if self._registry and self._registry.base_url and not url.is_absolute():
            if self._registry.base_url.startswith(("http://", "https://")):
                return self._registry.base_url.rstrip("/") + "/" + str(url)
            else:
                return str(Path(self._registry.base_url) / url)

        return self.url

    def fetch(self) -> pd.DataFrame:
        """Fetch the dataset and return a pandas DataFrame."""
        url = self.get_url()

        logger.info(f"Reading dataset from {url}")

        # Determine the file extension and appropriate reader
        match Path(url).suffix.lstrip("."):
            case "csv" | "txt":
                reader = pd.read_csv
                default_load_args: dict[str, Any] = {}
            case "json":
                reader = pd.read_json
                default_load_args = {}
            case "jsonl":
                reader = pd.read_json
                default_load_args = {"lines": True}
            case "parquet":
                reader = pd.read_parquet
                default_load_args = {}
            case extension:
                if not extension:
                    extension = f"<no extension found on url '{url}'>"
                raise ValueError(f"Unsupported file extension: {extension}")

        # Merge load args: user-provided args override defaults
        final_load_args = {**default_load_args, **(self.load_args or {})}

        try:
            return reader(url, **final_load_args)
        except Exception as e:
            logger.error(f"Error reading dataset from {url}: {e}", exc_info=True)
            raise


class DatasetRegistry(BaseModel):
    """Registry of datasets for easy reference by name."""

    datasets: list[DatasetInfo] = Field(default_factory=list)
    """List of datasets in the registry."""

    base_url: str | None = None
    """Base URL for the registry.

    Any relative paths will be prepended with the base_url before attempting to load the dataset.
    This only applies to the datasets in the registry which have a relative url.
    """

    def __init__(self, **data):
        super().__init__(**data)
        for dataset in self.datasets:
            dataset._registry = self

    def get_dataset(self, url: str) -> DatasetInfo:
        """Get a dataset from the registry.

        Automatically adds a new Dataset with name url if it doesn't exist in the registry.
        """

        for dataset in self.datasets:
            if dataset.name == url:
                return dataset

        # If the dataset is not already in the registry, create and add a new
        # DatasetInfo object to the registry. Deliberately do NOT set the
        # registry reference for this new dataset, this ensures its relative
        # paths resolve against the current working directory, not the
        # registry's base_url (if set).
        new_dataset = DatasetInfo(name=url, url=url, overrides=None, load_args=None)
        self.datasets.append(new_dataset)
        return new_dataset

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a DatasetRegistry from a YAML file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
