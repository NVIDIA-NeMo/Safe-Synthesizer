# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for whether a value is a locale's personal national ID, using python-stdnum."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from functools import lru_cache
from types import MappingProxyType

__all__ = ["national_id_check"]

_PERSONAL_ID_MODULES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "BE": ("be.nn",),
        "BR": ("br.cpf",),
        "CA": ("ca.sin",),
        "CH": ("ch.ssn",),
        "CL": ("cl.rut",),
        "CN": ("cn.ric",),
        "DE": ("de.idnr",),
        "EC": ("ec.ci",),
        "EE": ("ee.ik",),
        "ES": ("es.dni", "es.nie"),
        "FI": ("fi.hetu",),
        "FR": ("fr.nir",),
        "GR": ("gr.amka",),
        "HR": ("hr.oib",),
        "IE": ("ie.pps",),
        "IL": ("il.idnr",),
        "IN": ("in_.aadhaar",),
        "IS": ("is_.kennitala",),
        "IT": ("it.codicefiscale",),
        "JP": ("jp.in_",),
        "KR": ("kr.rrn",),
        "LT": ("lt.asmens",),
        "MX": ("mx.curp",),
        "MY": ("my.nric",),
        "NO": ("no.fodselsnummer",),
        "PL": ("pl.pesel",),
        "RO": ("ro.cnp",),
        "SE": ("se.personnummer",),
        "SI": ("si.emso",),
        "TH": ("th.pin",),
        "TR": ("tr.tckimlik",),
        "US": ("us.ssn",),
        "ZA": ("za.idnr",),
    }
)
"""python-stdnum modules for each country's personal national IDs, keyed by ISO 3166 country code.

Checks that pass about as many other numbers as real IDs in detected text are left out, such as the Danish CPR
and Australian TFN.
"""

_LOOSE_PERSONAL_ID_MODULES: Mapping[str, tuple[str, ...]] = MappingProxyType({"NL": ("nl.bsn",)})
"""Checks too loose for detection that still tell a column's format: the BSN check passes one 9-digit number in ten."""


@lru_cache
def national_id_check(locale: str, *, loose: bool = False) -> Callable[[str], bool] | None:
    """Return whether a value is a personal national ID of ``locale``'s country, or ``None`` without a check.

    The country is the locale's region, so ``en_CA`` and ``fr_CA`` both check Canadian SINs. Whitespace is
    ignored, since python-stdnum's checks only strip some separators.

    Args:
        locale: Locale whose country's IDs to check, such as ``en_US``.
        loose: Also use checks that many other numbers pass, such as the Dutch BSN. Fine for choosing how to
            replace a value already typed as a national ID, too loose for detecting one.

    Example:
        ``national_id_check("en_US")`` accepts ``"123-45-6789"``, ``"123456789"``, and ``"123 45 6789"``, but not
        ``"912-45-6789"``, an ITIN. ``national_id_check("en_GB")`` is ``None``.
    """
    _, _, region = locale.replace("-", "_").rpartition("_")
    modules = _PERSONAL_ID_MODULES.get(region.upper()) or (
        _LOOSE_PERSONAL_ID_MODULES.get(region.upper()) if loose else None
    )
    if modules is None:
        return None
    validators = tuple(importlib.import_module(f"stdnum.{module}").is_valid for module in modules)

    def is_national_id(value: str) -> bool:
        compact = "".join(value.split())
        return any(_passes(is_valid, compact) for is_valid in validators)

    return is_national_id


def _passes(is_valid: Callable[[str], bool], value: str) -> bool:
    # Some checks raise instead of returning False, such as cn.ric for an unknown region code.
    try:
        return is_valid(value)
    except Exception:
        return False
