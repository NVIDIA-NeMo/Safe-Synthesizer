# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiReplacementSettings,
    PiiSamplerBackend,
    PiiSamplerConfig,
)
from nemo_safe_synthesizer.pii_replacer import dependency_labels


@pytest.mark.unit
class TestDependencyLabelCatalog:
    def test_faker_exposes_only_supported_conditioning_labels(self) -> None:
        catalog = dependency_labels.dependency_label_catalog(
            PiiReplacementSettings(),
            PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        )

        assert catalog == {EntityType.GENDER: ("female", "male")}

    def test_managed_reads_only_dependency_columns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_path = tmp_path / "datasets" / "en_US.parquet"
        dataset_path.parent.mkdir()
        dataset_path.touch()
        reads: list[tuple[Path, list[str], str]] = []
        monkeypatch.setattr(
            dependency_labels,
            "_available_parquet_columns",
            lambda _path: frozenset({"first_name", "sex", "ethnic_background"}),
        )

        def read_parquet(path: Path, *, columns: list[str], dtype_backend: str) -> pd.DataFrame:
            reads.append((path, columns, dtype_backend))
            return pd.DataFrame(
                {
                    "sex": ["Female", "Male", "Female"],
                    "ethnic_background": ["White", "East Asian", "White"],
                }
            )

        monkeypatch.setattr(pd, "read_parquet", read_parquet)

        catalog = dependency_labels.dependency_label_catalog(
            PiiReplacementSettings(locale="en_US"),
            PiiSamplerConfig(backend=PiiSamplerBackend.MANAGED, managed_assets_path=str(tmp_path)),
        )

        assert reads == [(dataset_path, ["sex", "ethnic_background"], "pyarrow")]
        assert catalog == {
            EntityType.GENDER: ("female", "male"),
            EntityType.ETHNIC_BACKGROUND: ("east asian", "white"),
        }

    def test_missing_managed_locale_has_no_catalog(self, tmp_path: Path) -> None:
        catalog = dependency_labels.dependency_label_catalog(
            PiiReplacementSettings(locale="fr_FR"),
            PiiSamplerConfig(backend=PiiSamplerBackend.MANAGED, managed_assets_path=str(tmp_path)),
        )

        assert catalog == {}
