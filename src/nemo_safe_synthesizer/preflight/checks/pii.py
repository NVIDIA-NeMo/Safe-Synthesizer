# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PII replacement configuration and plan advisories."""

from __future__ import annotations

from pathlib import Path

from typing_extensions import override

from ...config.pii_replacement import PiiPersonBackend, PiiReplacementPlan
from ...defaults import NSS_MANAGED_ASSETS_PATH_ENV
from ...pii_replacer.plan import load_plan_from_path, unique_id_advisories
from ...pii_replacer.runtime_config import runtime_config_from_replace_pii
from ..base import AdvisoryCheck, ConfigCheck, IssueCollector
from ..types import ConfigView, DataFrameView

__all__ = [
    "PiiReplacementConfigCheck",
    "PiiUniqueIdAdvisoryCheck",
]


def _faker_locale_supported(locale: str) -> bool:
    try:
        from faker.config import AVAILABLE_LOCALES
    except ImportError:
        return True
    return locale in AVAILABLE_LOCALES


class PiiReplacementConfigCheck(ConfigCheck):
    """Validate persona backend and locale settings for tabular PII replacement."""

    name = "pii.replacement_config"
    label = "PII replacement configuration"
    category = "environment"

    @override
    def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
        replace_pii = ctx.config.replace_pii
        if replace_pii is None:
            return

        if replace_pii.llm_enhancement:
            collector.warning(
                "pii_llm_not_implemented",
                "llm_enhancement=True is not implemented in this release; the run will fail when PII replacement starts.",
            )

        locale = replace_pii.replacement.locale
        backend = replace_pii.person.backend

        if backend == PiiPersonBackend.faker and not _faker_locale_supported(locale):
            collector.error(
                "pii_faker_locale_invalid",
                f"replace_pii.replacement.locale {locale!r} is not supported by Faker.",
            )
            return

        if backend == PiiPersonBackend.managed:
            assets_root = replace_pii.person.resolved_managed_assets_path()
            parquet_path = assets_root / "datasets" / f"{locale}.parquet"
            if not parquet_path.exists():
                collector.warning(
                    "pii_managed_assets_missing",
                    f"Managed persona assets not found at {parquet_path}; set "
                    f"{NSS_MANAGED_ASSETS_PATH_ENV} or replace_pii.person.managed_assets_path; "
                    "apply will fall back to Faker.",
                )

        if backend == PiiPersonBackend.pgm:
            src = Path(replace_pii.person.sdg_pgms_src)
            if not src.is_dir():
                collector.warning(
                    "pii_pgm_src_missing",
                    f"person.sdg_pgms_src {src} does not exist; apply will fall back to managed/Faker personas.",
                )
            else:
                init_py = src / "sdg_pgms" / "__init__.py"
                if not init_py.exists():
                    collector.warning(
                        "pii_pgm_import_missing",
                        f"sdg-pgms package not found under {src}; apply will fall back to managed/Faker personas.",
                    )


def _resolved_user_plan(replace_pii) -> PiiReplacementPlan | None:
    inline = replace_pii.inline_plan
    if inline is not None:
        return inline
    plan_path = replace_pii.plan_path
    if plan_path is None:
        return None
    try:
        return load_plan_from_path(plan_path)
    except Exception:
        return None


class PiiUniqueIdAdvisoryCheck(AdvisoryCheck):
    """Advise when a user-supplied plan marks low-cardinality columns as unique_id."""

    name = "pii.unique_id_cardinality"
    label = "PII unique_id cardinality"
    category = "data quality"

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        replace_pii = ctx.config.replace_pii
        if replace_pii is None or replace_pii.is_auto_discovery:
            return

        plan = _resolved_user_plan(replace_pii)
        if plan is None:
            return

        runtime = runtime_config_from_replace_pii(replace_pii)
        for message in unique_id_advisories(ctx.data, plan, runtime):
            collector.warning("pii_unique_id_low_cardinality", message)
