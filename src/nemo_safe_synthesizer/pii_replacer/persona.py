# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona generation backends."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ..observability import get_logger
from . import core
from .runtime_config import RuntimeConfig

logger = get_logger(__name__)

__all__ = ["PersonaEngine", "load_managed_person_sampler"]


def load_managed_person_sampler(assets_root: str, locale: str) -> pd.DataFrame | None:
    """Load persona parquet assets without data_designer."""
    path = Path(assets_root) / "datasets" / f"{locale}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.runtime.warning(f"Managed assets unreadable at {path} ({exc}); using Faker person sampler.")
        return None


def _row_to_persona(row: pd.Series) -> dict:
    return core._pgm_persona(row)


def _faker_persona(fake: Any, sex: str | None = None) -> dict:
    # Condition the given name on sex so the synthetic name matches the source
    # demographic; fall back to an unconditioned name when sex is unknown.
    if sex == "Male":
        first_name, resolved_sex = fake.first_name_male(), "Male"
    elif sex == "Female":
        first_name, resolved_sex = fake.first_name_female(), "Female"
    else:
        first_name, resolved_sex = fake.first_name(), fake.random_element(["Male", "Female"])
    return {
        "first_name": first_name,
        "last_name": fake.last_name(),
        "sex": resolved_sex,
        "ethnic_background": "white",
        "email_address": fake.email(),
        "phone_number": fake.phone_number(),
        "birth_date": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
        "street_number": fake.building_number(),
        "street_name": fake.street_name(),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "postcode": fake.postcode(),
        "ssn": fake.ssn(),
    }


class PersonaEngine:
    """Assign one synthetic persona per person instance."""

    def __init__(self, runtime: RuntimeConfig, n_instances: int):
        self.runtime = runtime
        self.cfg = core.Config(
            locale=runtime.locale,
            random_seed=runtime.random_seed,
            persona_backend=runtime.persona_backend,
            sdg_pgms_src=runtime.sdg_pgms_src,
            managed_assets_path=runtime.managed_assets_path,
            pool_min_size=runtime.pool_min_size,
            pool_oversample=runtime.pool_oversample,
            use_race_constraint=runtime.use_race_constraint,
        )
        self.backend = runtime.persona_backend if n_instances else "none"
        self.pgm_pool: core.PgmPersonaPool | None = None
        self.managed_df: pd.DataFrame | None = None
        self.source_counts: Counter = Counter()
        if self.backend == "pgm":
            gen = core._load_pgm_generator(self.cfg)
            if gen is not None:
                pool_n = max(runtime.pool_min_size, runtime.pool_oversample * max(1, n_instances))
                self.pgm_pool = core.PgmPersonaPool(gen, runtime.random_seed, pool_n)
            else:
                self.backend = "managed"
        if self.backend == "managed":
            if runtime.managed_assets_path:
                self.managed_df = load_managed_person_sampler(runtime.managed_assets_path, runtime.locale)
            if self.managed_df is None:
                self.backend = "faker"

    def _sample_managed(
        self, n: int, seed: int, sex: str | None = None, sfv: dict | None = None
    ) -> list[dict]:
        df = self.managed_df
        if df is None or df.empty:
            return []
        eth = None
        if sfv:
            vals = sfv.get("ethnic_background")
            if vals:
                eth = {str(v).strip().lower() for v in vals}
        # Prefer matching sex+ethnicity, then relax to sex-only, ethnicity-only, and
        # finally the whole pool -- mirroring the PGM pool's matching (names are
        # conditioned on sex + ethnic_background only).
        sub = df
        for use_sex, use_eth in ((True, True), (True, False), (False, True), (False, False)):
            sub = df
            if use_sex and sex and "sex" in sub.columns:
                sub = sub[sub["sex"] == sex]
            if use_eth and eth and "ethnic_background" in sub.columns:
                sub = sub[sub["ethnic_background"].astype(str).str.strip().str.lower().isin(eth)]
            if not sub.empty:
                break
        if sub.empty:
            sub = df
        sample = sub.sample(n=n, replace=n > len(sub), random_state=seed)
        return [_row_to_persona(row) for _, row in sample.iterrows()]

    def _sample_faker(self, n: int, seed: int, sex: str | None = None) -> list[dict]:
        fake = core._seeded_faker(seed, self.runtime.locale)
        return [_faker_persona(fake, sex) for _ in range(n)]

    def assign(self, instances: list[dict]) -> None:
        if not instances:
            return
        if self.pgm_pool is not None:
            for inst in instances:
                inst["persona"] = self.pgm_pool.match_one(inst["sex"], inst["select_field_values"])
                inst["persona_source"] = "pgm"
                self.source_counts["pgm"] += 1
            return

        # Bucket instances by their demographic constraint (sex, ethnicity) so every
        # instance in a bucket is sampled from the matching sub-population.
        buckets: dict[Any, list[int]] = {}
        for idx, inst in enumerate(instances):
            buckets.setdefault(core._constraint_signature(inst), []).append(idx)
        for b_idx, (sig, idxs) in enumerate(buckets.items()):
            sex, sfv_key = sig
            sfv = {k: list(v) for k, v in sfv_key} or None
            seed = self.runtime.random_seed + b_idx
            if self.backend == "managed" and self.managed_df is not None:
                personas = self._sample_managed(len(idxs), seed, sex, sfv)
                source = "managed"
            else:
                personas = self._sample_faker(len(idxs), seed, sex)
                source = "faker"
            if len(personas) < len(idxs):
                personas.extend(self._sample_faker(len(idxs) - len(personas), seed + 1, sex))
                source = "faker"
            self.source_counts[source] += len(idxs)
            for inst_idx, persona in zip(idxs, personas):
                instances[inst_idx]["persona"] = persona
                instances[inst_idx]["persona_source"] = source
