# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ...observability import get_logger

logger = get_logger(__name__)

DEFAULT_BUCKET = os.getenv("NSS_OPT_BUCKET", "nss-opt-dev-use2")
"""Default bucket environment variable. If it's not found, use dev."""

DEFAULT_CACHE_DIR = os.getenv("NSS_OPT_CACHE_DIR", ".optcache")
"""``StorageConfig`` default cache directory. Searches the NSS_OPT_CACHE_DIR environment
for a cache directory. By default it will fallback to `.optcache`. If this is set to
``disabled`` files wont be cached to disk.
"""


class Visibility(Enum):
    """Visibility identifiers indicating what access is opened for each model"""

    PUBLIC = "pub"
    """Packages that are open to the public with a public-read ACL"""

    PRIVATE = "priv"
    """Packages available to customers via "paywall" behind an api key"""

    INTERNAL = "int"
    """Only available from internal infrastructure."""


@dataclass
class ObjectRef:
    """Maps individual model files to keys in memory"""

    key: str
    """Lookup key to access model data"""

    file_name: str
    """Remote file name. Used to download model data from storage"""


@dataclass
class ModelManifest:
    """Manages remote model files and versions. These configurations are model specific.

    Models are by default stored in opt, under the convention

        /[vis]/models/[pkg]/[version]/[...files]

    """

    model: str
    """Model identifer. Eg spacy, fasttext, entityruler"""

    version: str
    """Model version"""

    sources: list[ObjectRef]
    """A list of pickled objects to load into memory."""

    visibility: Visibility
    """Package visibility"""

    @property
    def key(self) -> str:
        """A unique key that can be used to store or fetch model sources."""
        return f"{self.model}/{self.version}"


@dataclass
class StorageConfig:
    """Defines where and how ``ModelManifest``s are stored. These configurations are likely
    environment specific.
    """

    bucket: Optional[str] = DEFAULT_BUCKET
    """Remote "opt" bucket model files are stored under"""

    cache_dir: Optional[Path] = None
    """Local file system directory. If this value is None, ``ModelManifest`` files
    wont be cached through to disk.
    """

    @classmethod
    def from_system(cls) -> StorageConfig:
        """Return a default ``StorageConfig`` based on a system's environment variables.

        By convention, it looks for the environment variable ``NSS_OPT_BUCKET`` as
        the bucket location. The default settings from this function are appropriate
        for development without additional configuration.
        """
        if DEFAULT_CACHE_DIR == "disabled":
            cache_dir = None
        else:
            cache_dir = Path(DEFAULT_CACHE_DIR)

        return cls(bucket=DEFAULT_BUCKET, cache_dir=cache_dir)


def get_cache_manager(storage_config: StorageConfig = None) -> CacheManager:
    """Returns a singleton instance of ``CacheManager``."""
    return CacheManager.get_instance(storage_config)


class CacheManager:
    """Handles downloading model files from the "opt" package repo.

    This class will also optionally cache these files to disk. This is useful for
    environments with local persistent state such as a local development laptop.

    Args:
        storage_config: A storage config.
    """

    __instance = None
    """Used to hold a singleton of ``CacheManager``"""

    _cache: dict[str, dict[str, Any]]
    """Holds model object values in memory. Key by ``ModelManifest.key`` by ``ObjectRef.key``"""

    _manifests: dict[str, ModelManifest]
    """Contains each registered model manifest"""

    timings: dict[str, float]
    """Holds timings for each model manifest. Keyed by ``ModelManifest.model``"""

    @staticmethod
    def reset():
        CacheManager.__instance = None

    @classmethod
    def get_instance(cls, storage_config: StorageConfig = None) -> CacheManager:
        """Returns a singleton instance of ``CacheManager``."""
        if not CacheManager.__instance:
            CacheManager(storage_config)
        return CacheManager.__instance

    def __init__(self, storage_config: StorageConfig = None):
        if CacheManager.__instance:
            raise Exception("Cannot instantiate a singleton.")
        else:
            CacheManager.__instance = self

        self.storage_config = storage_config or StorageConfig.from_system()
        logger.info(f"Creating a new instance of CacheManager for {storage_config}")

        self._cache = {}
        self._manifests = {}
        self.timings = {}

    def register_manifest(self, manifest: ModelManifest):
        """Registers a manifest in the cache manager. This will not download the file"""
        if manifest.key not in self._manifests:
            logger.info(f"Registering ModelManifest in cache: {manifest}")
            self._manifests[manifest.key] = manifest

    def set_storage_config(self, storage_config: StorageConfig):
        """Apply a new ``StorageConfig`` to the ``CacheManager``."""
        self.storage_config = storage_config

    def download_and_cache_manifest_data(self):
        """Load each registered manifest into memory"""
        for manifest in self._manifests.values():
            self.resolve(manifest)

    def resolve(self, manifest: ModelManifest, evict: bool = False, skip_pickle: bool = False) -> Optional[dict]:
        """Given a manifest, will return it's resolved data. If the manifest hasn't
        already been registered with the manager, it will be registered automatically.

        The load order is as follows:

            1. From in-memory cache
            2. From FS cache if enabled via a ``StorageConfig``

        Args:
            manifest: The manifest file to resolve and return
            evict: If ``True`` will return the object, but won't store it in the cache.
                If the manifest is already in the cache, it will be removed.
        """
        if manifest.key in self._cache:
            return self._cache.pop(manifest.key) if evict else self._cache[manifest.key]

        self.register_manifest(manifest)

        objs = {}
        for obj_ref in manifest.sources:
            src_obj = None
            start_time = time.perf_counter()
            for step in self.obj_from_fs:
                src_obj = step(manifest, obj_ref, skip_pickle=skip_pickle)
                if src_obj:
                    break
            if not src_obj:
                raise RuntimeError(f"Could note resolve manifest {manifest}. Failed to load {src_obj}")
            else:
                elapsed_time_seconds = time.perf_counter() - start_time
                objs[obj_ref.key] = src_obj
                self.timings[manifest.model] = elapsed_time_seconds

        if not evict:
            self._cache[manifest.key] = objs
        return objs

    def obj_from_fs(self, manifest: ModelManifest, obj_ref: ObjectRef, skip_pickle: bool = False) -> Optional[Any]:
        """Return the source object from the filesystem if it exists. If no file is found
        return ``None``.
        """
        logger.debug(f"Checking local FS for manifest {manifest.model} for {obj_ref.key} at {obj_ref.file_name}...")
        if not self.storage_config.cache_dir:
            logger.debug("Manifest data not found on local FS!")
            return None

        file_path = self.storage_config.cache_dir / manifest.key / obj_ref.file_name
        if not file_path.is_file():
            return None

        with open(file_path, "rb") as cache:
            if skip_pickle:
                return cache.read()
            src_obj = pickle.load(cache)
            return src_obj

    def _cache_to_disk(self, manifest: ModelManifest, obj_ref: ObjectRef, raw_obj: bytes):
        """Write ``raw_obj`` to disk if the CacheManager is configured properly"""
        if self.storage_config.cache_dir:
            logger.info(f"caching {manifest.key}: {obj_ref} to disk")
            target = self.storage_config.cache_dir / manifest.key / obj_ref.file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                out.write(raw_obj)
