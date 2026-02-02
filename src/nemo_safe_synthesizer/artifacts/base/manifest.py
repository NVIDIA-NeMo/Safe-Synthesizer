# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional


class ManifestStatus(StrEnum):
    SUCCESS = "success"
    PENDING = "pending"
    ERROR = "error"


CURRENT_VERSION = "1.2"


@dataclass
class ArtifactManifest:
    """
    Represents calculated manifest for a given artifact.
    The way information is represented is unstructured dict, so that it's easy to modify.
    """

    RECORD_COUNT = "record_count"
    FIELD_COUNT = "field_count"

    READER_TYPE = "reader_type"
    FORMAT_CONFIG = "format_config"

    DUPLICATED_COUNT = "duplicated_count"
    DUPLICATED_PERCENT = "duplicated_percent"

    AVG_RECORD_LENGTH = "avg_record_length"

    artifact_id: str

    version: str = CURRENT_VERSION
    project_id: Optional[str] = None

    # basic info
    file_size: Optional[int] = None
    upload_time: Optional[str] = None
    content_sha256: Optional[str] = None
    contains_nested_json: Optional[bool] = None

    # status
    status: Optional[ManifestStatus] = None
    error_message: Optional[str] = None
    error_trace: Optional[str] = None

    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def fields(self) -> Dict[str, list]:
        return self.manifest.get("fields", {})

    @property
    def types(self) -> List[dict]:
        return self.manifest.get("types", [])

    @fields.setter
    def fields(self, value: Dict[str, list]) -> None:
        self.manifest["fields"] = value

    @property
    def data_check_results(self) -> List[Dict[str, Any]]:
        return self.manifest.get("data_check_results", [])

    @data_check_results.setter
    def data_check_results(self, value: List[Dict[str, Any]]) -> None:
        self.manifest["data_check_results"] = value

    def add_feature(self, name: str, value: Any):
        self.manifest[name] = value

    def dict(self) -> Dict[str, Any]:
        return {key: value for key, value in dataclasses.asdict(self).items() if value is not None}
