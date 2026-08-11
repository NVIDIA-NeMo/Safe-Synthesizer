# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PII replacement configuration and plan advisories."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError
from typing_extensions import override

from ...config.pii_replacement import PiiPersonBackend, PiiReplacementPlan, ReplacePiiConfig
from ...defaults import NSS_MANAGED_ASSETS_PATH_ENV
from ...errors import ParameterError
from ...pii_replacer.planning import iter_plan_advisories, iter_plan_issues, load_plan_from_path
from ..base import ConfigCheck, DataFrameCheck, IssueCollector
from ..types import ConfigView, DataFrameView

__all__ = [
    "PiiPlanValidityCheck",
    "PiiReplacementConfigCheck",
]


def _faker_locale_supported(locale: str) -> bool:
    try:
        from faker.config import AVAILABLE_LOCALES
    except ImportError:
        return True
    return locale in AVAILABLE_LOCALES


def _pgm_source_state(src: Path) -> Literal["unusable", "no_package", "ok"]:
    """Classify an sdg-pgms checkout without letting the filesystem raise.

    Reporting an unusable path is this check's job, so a path the run cannot even
    stat -- a checkout under another user's home, as on CI -- counts as unusable
    rather than surfacing as a ``PermissionError`` out of pre-flight.
    """
    try:
        if not src.is_dir():
            return "unusable"
        return "ok" if (src / "pgms" / "__init__.py").exists() else "no_package"
    except OSError:
        return "unusable"


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
            collector.error(
                "pii_llm_not_implemented",
                "llm_enhancement=True is not implemented in this release; unset it or set llm_enhancement: false.",
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

        # The PGM is internal-only and never falls back (see replacement.personas._load_pgm_generator),
        # so a missing checkout is an error here rather than a warning about a fallback.
        if backend == PiiPersonBackend.pgm:
            src = Path(replace_pii.person.sdg_pgms_src)
            if locale != "en_US":
                collector.error(
                    "pii_pgm_locale_invalid",
                    f"replace_pii.person.backend 'pgm' supports locale 'en_US' only, but the locale is {locale!r}.",
                )
            match _pgm_source_state(src):
                case "unusable":
                    collector.error(
                        "pii_pgm_src_missing",
                        f"replace_pii.person.sdg_pgms_src {src} is not a readable directory; the 'pgm' backend "
                        "needs a local sdg-pgms checkout.",
                    )
                case "no_package":
                    collector.error(
                        "pii_pgm_import_missing",
                        f"sdg-pgms package not found under {src}; expected a 'pgms' package there.",
                    )
                case "ok":
                    pass


def _load_user_plan(replace_pii: ReplacePiiConfig) -> PiiReplacementPlan | None:
    """Return the user's plan, or ``None`` when the config asks for auto-discovery.

    Raises whatever ``load_plan_from_path`` raises for an unreadable plan file;
    callers decide whether that is an issue to report or one to ignore.
    """
    inline = replace_pii.inline_plan
    if inline is not None:
        return inline
    plan_path = replace_pii.plan_path
    if plan_path is None:
        return None
    return load_plan_from_path(plan_path)


class PiiPlanValidityCheck(DataFrameCheck):
    """Check a user-supplied replacement plan against the dataset it will run on.

    Auto-discovered plans are excluded: discovery builds them from this same
    dataframe, and ``resolve_plan`` validates that output when replacement runs.
    A hand-written plan is the one that can name a column that does not exist or
    carry a pattern its own values do not match, and without this check those
    only surface once replacement starts -- never on the ``--validate`` path,
    which skips PII replacement entirely.

    On a full ``process_data`` run the late (post-holdout) preflight disables this
    check: the early CONFIG/DATAFRAME pass already validated the same user plan
    against the same columns.
    """

    name = "pii.plan_validity"
    label = "PII replacement plan"
    category = "data quality"

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        replace_pii = ctx.config.replace_pii
        if replace_pii is None or replace_pii.is_auto_discovery:
            return

        try:
            plan = _load_user_plan(replace_pii)
        except (OSError, ParameterError, ValidationError, yaml.YAMLError) as exc:
            collector.error(
                "pii_plan_unreadable",
                f"replace_pii.replacement_plan {replace_pii.plan_path!r} could not be loaded: {exc}",
            )
            return
        if plan is None:
            return

        for issue in iter_plan_issues(
            ctx.data,
            plan,
            data_config=ctx.config.data,
            time_series=ctx.config.time_series,
        ):
            collector.error(issue.code, issue.message)

        backend = replace_pii.person.backend.value
        for advisory in iter_plan_advisories(plan, persona_backend=backend):
            collector.warning(advisory.code, advisory.message)
