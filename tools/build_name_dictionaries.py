#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "openpyxl",
#   "pandas",
#   "pyreadr",
# ]
# ///

"""Build the name dictionaries in ``name_dictionaries/`` from public CC0 and public-domain sources.

Usage:
    uv run --script tools/build_name_dictionaries.py --cache-dir /path/to/downloads

The script downloads each source once into ``--cache-dir`` and writes CSV files under ``--out-dir``
(default: ``name_dictionaries/`` at the repository root).
"""

from __future__ import annotations

import argparse
import re
import tarfile
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ssa.gov blocks automated downloads, so the SSA counts come from the CRAN package that bundles them (1880-2017).
SSA_URL = "https://cran.r-project.org/src/contrib/babynames_1.0.1.tar.gz"
CENSUS_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"
# Harvard Dataverse file ids: Tzioumis (2018) firstnames, doi:10.7910/DVN/TYJKEZ; WGND 2.0, doi:10.7910/DVN/MSEGSJ.
TZIOUMIS_URL = "https://dataverse.harvard.edu/api/access/datafile/3078263?format=original"
WGND_URL = "https://dataverse.harvard.edu/api/access/datafile/4750348?format=original"

# Birth years of today's adult population.
SSA_YEARS = (1930, 2017)
SSA_MIN_BIRTHS = 100
UNISEX_MIN_BIRTHS = 1000
# Lower bound on the female share for each label, checked in order.
SEX_BANDS = ((0.9, "female"), (0.75, "mostly_female"), (0.25, "unisex"), (0.1, "mostly_male"), (-1.0, "male"))
CENSUS_GROUPS = ("white", "black", "hispanic", "api", "aian", "2prace")
# Placeholders that records use when the real name is missing, such as "Baby" or "Unknown" on birth records and "Null"
# or "Test" in Census surnames. Short real names that look similar, such as Son and Na, and the surname Baby are kept.
PLACEHOLDER_NAMES = frozenset(
    {"blank", "female", "infant", "male", "noname", "none", "notnamed", "null", "test", "unk", "unknown", "unnamed"}
)
PLACEHOLDER_FIRST_NAMES = PLACEHOLDER_NAMES | {"baby", "babyboy", "babygirl", "boy", "girl", "twin"}
PLACEHOLDER_LAST_NAMES = PLACEHOLDER_NAMES | {"boy", "daughter", "girl"}
# Rows that aggregate every name below a source's publication threshold.
CATCH_ALL_NAMES = frozenset({"all other names", "all other first names"})
# WGND country codes per Nemotron-Personas locale. en_IN is left out: WGND has 2.5M noisy Indian entries and no
# counts to filter them by.
WGND_LOCALES = {"en_GB": "GB", "de_DE": "DE", "fr_FR": "FR", "pt_BR": "BR", "en_SG": "SG", "ja_JP": "JP", "ko_KR": "KR"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="directory for downloaded sources")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "name_dictionaries")
    args = parser.parse_args()
    cache: Path = args.cache_dir
    out: Path = args.out_dir
    cache.mkdir(parents=True, exist_ok=True)
    (out / "en_US").mkdir(parents=True, exist_ok=True)
    (out / "other_locales").mkdir(parents=True, exist_ok=True)

    wgnd = _read_wgnd(_download(WGND_URL, cache / "wgnd_2_0_name-gender-code.tab"))
    sex = _ssa_first_name_sex(_download(SSA_URL, cache / "babynames_1.0.1.tar.gz"), wgnd)
    sex.to_csv(out / "en_US/first_name_sex.csv", index=False)
    unisex = sex[(sex.label == "unisex") & (sex.births_1930_2017 >= UNISEX_MIN_BIRTHS)]
    unisex.to_csv(out / "en_US/unisex_first_names.csv", index=False)
    print(f"first_name_sex: {len(sex)} names {sex.label.value_counts().to_dict()}; unisex list: {len(unisex)}")

    last = _census_last_name_ethnicity(_download(CENSUS_URL, cache / "names_2010census.zip"))
    last.to_csv(out / "en_US/last_name_ethnicity.csv", index=False)
    print(f"last_name_ethnicity: {len(last)} names")

    first = _tzioumis_first_name_ethnicity(_download(TZIOUMIS_URL, cache / "tzioumis_firstnames.xlsx"))
    first.to_csv(out / "en_US/first_name_ethnicity.csv", index=False)
    print(f"first_name_ethnicity: {len(first)} names")

    for locale, code in WGND_LOCALES.items():
        locale_sex = _wgnd_locale_sex(wgnd, code)
        locale_sex.to_csv(out / f"other_locales/{locale}_first_name_sex.csv", index=False)
        print(f"{locale}: {len(locale_sex)} names")


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response:
            path.write_bytes(response.read())
    return path


def name_key(name: object) -> str:
    """Return the lookup key for a name: casefolded, Latin accents removed, letters of any script, spaces, hyphens,
    and apostrophes kept.

    Example:
        ``"Hélène"`` and ``"Helene"`` both become ``"helene"``, and Hangul such as ``"지우"`` is kept as is.
    """
    decomposed = unicodedata.normalize("NFKD", str(name).casefold().strip())
    # Only the combining diacritics block is removed, so Japanese voicing marks such as the one in "が" are kept.
    kept = "".join(
        character
        for character in decomposed
        if not _COMBINING_DIACRITICS.match(character)
        and (unicodedata.category(character)[0] in "LM" or character in " '-")
    )
    return unicodedata.normalize("NFC", kept)


_COMBINING_DIACRITICS = re.compile("[̀-ͯ]")


def _sex_label(female_share: float) -> str:
    return next(label for lower, label in SEX_BANDS if female_share >= lower)


def _read_wgnd(path: Path) -> pd.DataFrame:
    wgnd = pd.read_csv(path, dtype=str, keep_default_na=False)
    wgnd["wgt"] = pd.to_numeric(wgnd.wgt, errors="coerce")
    wgnd["key"] = wgnd.name.map(name_key)
    return wgnd[~wgnd.key.isin(PLACEHOLDER_FIRST_NAMES)]


def _wgnd_female_shares(wgnd: pd.DataFrame, code: str, by: str = "key") -> pd.DataFrame:
    shares = wgnd[wgnd.code == code].pivot_table(index=by, columns="gender", values="wgt", aggfunc="sum")
    shares = shares.reindex(columns=["F", "M", "?"]).fillna(0.0)
    # Each spelling's shares sum to 1, so dividing by the total averages over spellings that are grouped together.
    known = shares.F + shares.M
    total = known + shares["?"]
    return pd.DataFrame(
        {
            "female_share": (shares.F / known.where(known > 0)).round(4),
            "unknown_share": (shares["?"] / total.where(total > 0)).round(4),
        }
    )


def _wgnd_locale_sex(wgnd: pd.DataFrame, code: str) -> pd.DataFrame:
    # One row per spelling: WGND lists many names both with and without accents, such as "hélène" and "helene".
    # Spellings that differ only in case are merged, and `key` shows which rows a lookup treats as one name.
    spellings = wgnd.assign(spelling=wgnd.name.str.title())
    result = _wgnd_female_shares(spellings, code, by="spelling").dropna(subset=["female_share"])
    result = result.rename_axis("name").reset_index()
    result["key"] = result.name.map(name_key)
    result["label"] = result.female_share.map(_sex_label)
    return result.sort_values(["key", "name"])[["name", "key", "female_share", "unknown_share", "label"]]


def _ssa_first_name_sex(tarball: Path, wgnd: pd.DataFrame) -> pd.DataFrame:
    # pyreadr is a dependency of this script only, not of the project environment that ty checks.
    import pyreadr  # ty: ignore[unresolved-import]

    rda = tarball.with_name("babynames.rda")
    if not rda.exists():
        with tarfile.open(tarball) as archive:
            member = archive.extractfile("babynames/data/babynames.rda")
            if member is None:
                raise FileNotFoundError("babynames.rda is missing from the CRAN tarball")
            rda.write_bytes(member.read())
    ssa = pyreadr.read_r(rda)["babynames"]
    ssa = ssa[ssa.year.between(*SSA_YEARS)].copy()
    ssa["key"] = ssa.name.map(name_key)
    ssa = ssa[~ssa.key.isin(PLACEHOLDER_FIRST_NAMES)]
    ssa["era"] = np.where(ssa.year < 1980, "1930_1979", "1980_2017")
    births = ssa.pivot_table(index="key", columns=["era", "sex"], values="n", aggfunc="sum", fill_value=0)
    female = births.xs("F", level="sex", axis=1).sum(axis=1)
    male = births.xs("M", level="sex", axis=1).sum(axis=1)
    # Show the most common spelling for names that differ only in accents or case.
    spelling = ssa.groupby(["key", "name"]).n.sum().reset_index().sort_values("n").groupby("key").name.last()
    sex = pd.DataFrame(
        {"name": spelling, "births_1930_2017": female + male, "female_births": female, "male_births": male}
    )
    sex["female_share"] = (female / (female + male)).round(4)
    for era in ("1930_1979", "1980_2017"):
        era_female = births.get((era, "F"), 0)
        era_total = era_female + births.get((era, "M"), 0)
        sex[f"female_share_{era}"] = (era_female / era_total).where(era_total >= SSA_MIN_BIRTHS).round(3)
    sex["label"] = sex.female_share.map(_sex_label)
    sex["wgnd_us_female_share"] = _wgnd_female_shares(wgnd, "US").female_share.reindex(sex.index)
    sex = sex[sex.births_1930_2017 >= SSA_MIN_BIRTHS]
    return sex.sort_values("births_1930_2017", ascending=False)


def _census_last_name_ethnicity(archive_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive, archive.open("Names_2010Census.csv") as source:
        census = pd.read_csv(source, dtype=str, keep_default_na=False)
    keys = census.name.map(name_key)
    census = census[~keys.isin(CATCH_ALL_NAMES | PLACEHOLDER_LAST_NAMES)].reset_index(drop=True)
    percents = pd.DataFrame({group: pd.to_numeric(census[f"pct{group}"], errors="coerce") for group in CENSUS_GROUPS})
    suppressed = percents.isna()
    # The Census suppresses small cells as "(S)"; spread the remaining percentage evenly across them.
    remainder = (100 - percents.sum(axis=1)).clip(lower=0) / suppressed.sum(axis=1).replace(0, np.nan)
    percents = percents.mask(suppressed, remainder, axis=0).fillna(0)
    result = pd.DataFrame({"name": census.name.str.title(), "count": census["count"].astype(int)})
    for group in CENSUS_GROUPS:
        result[f"pct_{group}"] = percents[group].round(2)
    result["top_group"] = percents.idxmax(axis=1)
    result["top_pct"] = percents.max(axis=1).round(2)
    result["suppressed_cells"] = suppressed.sum(axis=1)
    return result


def _tzioumis_first_name_ethnicity(path: Path) -> pd.DataFrame:
    tzioumis = pd.read_excel(path, sheet_name="Data")
    tzioumis = tzioumis[~tzioumis.firstname.map(name_key).isin(CATCH_ALL_NAMES | PLACEHOLDER_FIRST_NAMES)]
    result = pd.DataFrame({"name": tzioumis.firstname.astype(str).str.title(), "obs": tzioumis.obs.astype(int)})
    columns = [f"pct_{group}" for group in CENSUS_GROUPS]
    for group in CENSUS_GROUPS:
        result[f"pct_{group}"] = pd.to_numeric(tzioumis[f"pct{group}"], errors="coerce").round(2)
    result["top_group"] = result[columns].idxmax(axis=1).str.removeprefix("pct_")
    result["top_pct"] = result[columns].max(axis=1)
    return result.sort_values("obs", ascending=False)


if __name__ == "__main__":
    main()
