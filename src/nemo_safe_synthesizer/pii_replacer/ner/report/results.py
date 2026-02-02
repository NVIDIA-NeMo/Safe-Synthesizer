# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nemo_safe_synthesizer.data_processing.records.value_path import value_path_to_json_path

Prediction = Dict[str, Any]


@dataclass
class RecordResult:
    """
    Represents a NER result for a record in a dataset.

    Args:
        index: index of the record in the original dataset
        entities: Dict representation of NERPrediction (NERPrediction.as_dict)
    """

    index: int
    entities: List[Prediction]

    def __init__(self, index: int, entities: List[Prediction]):
        self.index = index
        self.entities = entities


@dataclass
class ExporterConfig:
    include_json_path: bool = field(default=False)
    """
    If set to True, the exported predictions will include "json_path" field for
    each entity.
    """

    include_match: bool = field(default=False)
    """
    If set to True, the exported prediction will contain text that got matched.
    """


class RecordResultExporter:
    """
    Responsible for creating a representation of RecordResult that is exportable to
    the outside.

    Args:
        config: Configuration for how to export the predictions.
    """

    _keys_to_export: Dict[str, str]
    """
    A mapping from names in the NERPrediction to names that will be exported.

    Note: only the names that are defined in this dict will be exported.

    For example:
     - exporting a prediction {"start": 5, "end": 10, ...}
     - with _keys_to_export={"start": "index_start"}
     - will result in {"index_start": 5} being exported
    """

    _pass_through_fields = {"start", "end", "label", "score", "field", "source"}
    """
    Fields that will be exported with the same name.
    """

    def __init__(self, config: Optional[ExporterConfig] = None):
        if not config:
            config = ExporterConfig()

        self._keys_to_export = {name: name for name in self._pass_through_fields}

        if config.include_json_path:
            self._keys_to_export["json_path"] = "json_path"

        if config.include_match:
            self._keys_to_export["text"] = "match"

    def export(self, record_result: RecordResult) -> dict:
        return {
            "index": record_result.index,
            "entities": [self._export_entity(entity) for entity in record_result.entities],
        }

    def _export_entity(self, entity: Prediction):
        exported_entity = {
            self._keys_to_export[key]: value for key, value in entity.items() if key in self._keys_to_export
        }
        if "json_path" in self._keys_to_export:
            exported_entity["json_path"] = value_path_to_json_path(entity["value_path"])

        return exported_entity


DatasetPredictions = List[RecordResult]


class DatasetClassifyResults:
    """
    Represents classification result on the dataset.

    For now this representation is using whatever is returned from the NER.predict()
    for simplicity and so we don't need to create a separate data structure (as there
    may be a lot of results).
    """

    results: DatasetPredictions

    def __init__(self, results: Optional[DatasetPredictions] = None):
        if not results:
            results = []

        self.results = results

    def add_record_result(self, record_result: RecordResult):
        self.results.append(record_result)

    def jsonl(self, exporter_config: Optional[ExporterConfig] = None) -> str:
        """
        Serializes results into a JSONL-formatted string.

        Args:
            exporter_config: Config to use for the result exporter.

        Returns: JSONL representation of results. Each line contains classification
            result for corresponding row in the input dataset.
        """
        exporter = RecordResultExporter(exporter_config)
        return "\n".join(json.dumps(exporter.export(result)) for result in self.results)
