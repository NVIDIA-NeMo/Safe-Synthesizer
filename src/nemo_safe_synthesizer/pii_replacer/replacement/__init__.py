# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turning a resolved plan into replaced values.

The stages run in the order the modules are listed here: ``scope`` fixes what a
replacement unit is, ``demographics`` reads the row's person into the vocabulary
``personas`` samples from, ``instances`` collects the people the plan finds,
``standalone`` maps the entity-driven columns that have no person behind them,
``free_text`` propagates a unit's replacements into its prose, and ``apply``
writes all of it onto the frame.
"""

from __future__ import annotations

from ..models import ScopedValueMap, StandaloneColMap
from .apply import apply_replacements, run_replacement
from .demographics import ethnicity_to_pgm, fuzzy_category, norm_sex, race_to_sfv
from .free_text import (
    build_text_substituter,
    instance_text_pair_labels,
    instance_text_pairs,
)
from .instances import compute_instance_synthetics, extract_instances
from .personas import (
    PersonaEngine,
    PgmPersonaPool,
    load_managed_person_sampler,
    persona_written,
    synth_value,
)
from .scope import FakerLike, build_scoped_col_map, seeded_faker, stable_hash, unit_key
from .standalone import build_standalone_maps, fake_value, synth_date_value, unique_synthetic

__all__ = [
    "FakerLike",
    "PersonaEngine",
    "PgmPersonaPool",
    "ScopedValueMap",
    "StandaloneColMap",
    "apply_replacements",
    "build_scoped_col_map",
    "build_standalone_maps",
    "build_text_substituter",
    "compute_instance_synthetics",
    "ethnicity_to_pgm",
    "extract_instances",
    "fake_value",
    "fuzzy_category",
    "instance_text_pair_labels",
    "instance_text_pairs",
    "load_managed_person_sampler",
    "norm_sex",
    "persona_written",
    "race_to_sfv",
    "run_replacement",
    "seeded_faker",
    "stable_hash",
    "synth_date_value",
    "synth_value",
    "unique_synthetic",
    "unit_key",
]
