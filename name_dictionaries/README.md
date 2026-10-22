# Name dictionaries for review

Draft dictionaries for inferring the likely sex and race/ethnicity of an original name, so that PII replacement can pick a
matching replacement name when the plan has no gender or ethnicity dependency. Nothing here is wired into the code yet.
This directory is a preview for review, and its final location in the package is still open.

Rebuild the files with:

```bash
uv run --script tools/build_name_dictionaries.py --cache-dir /tmp/name-sources
```

The script downloads each source into `--cache-dir` once. It drops each source's catch-all row, such as "All Other
First Names", and placeholders that records use for missing names, such as "Unknown", "Baby", and "Infant" on birth
records or "Null" and "Test" in Census surnames. The placeholder lists are at the top of the script.

## Sources

All sources are CC0 or US-government public domain, so they can be redistributed with the package.

| Source | Used for | Coverage |
| ------ | -------- | -------- |
| SSA national baby names, 1880–2017, via the CRAN `babynames` 1.0.1 package (CC0). Only births 1930–2017 are used, roughly today's adult population. | en_US first name → sex, unisex list | 35,225 names with at least 100 births |
| US Census 2010 surnames (public domain) | en_US last name → race/ethnicity | 162,245 surnames with at least 100 people |
| Tzioumis (2018), "Demographic aspects of first names", Harvard Dataverse (CC0), from mortgage applications | en_US first name → race/ethnicity | 4,250 first names |
| World Gender Name Dictionary 2.0 (WGND), Harvard Dataverse (CC0) | first name → sex for other locales, and a cross-check for en_US | varies by country, see below |

ssa.gov blocks automated downloads, so the SSA data comes from the CRAN package, which stops at 2017.

SSA, Census, and Tzioumis publish names in plain ASCII, so the en_US files have "Jose", not "José". WGND's few
accented US spellings are mostly typos ("jérry", "ródney"), so none are added. WGND lists many names in other locales
both with and without accents, and the locale files keep both spellings as separate rows, such as "Hélène" and
"Helene", each with its own shares.

A lookup key matches names that differ only in case or Latin accents: "Hélène" and "Helene" both become "helene", and
"Nguyễn" becomes "nguyen". Letters of other scripts are kept, including Japanese voicing marks, so "지우" and "がく" are
their own keys. The locale files show this key in a `key` column.

## Sex labels

Each name gets a label from its female share:

| Label | Female share |
| ----- | ------------ |
| `female` | ≥ 0.90 |
| `mostly_female` | 0.75–0.90 |
| `unisex` | 0.25–0.75 |
| `mostly_male` | 0.10–0.25 |
| `male` | < 0.10 |

The cutoffs are a first guess and are the main thing to review.

## Files

`en_US/first_name_sex.csv` holds one row per first name, sorted by births. Columns:

- `births_1930_2017`, `female_births`, `male_births`, `female_share`
- `female_share_1930_1979` and `female_share_1980_2017` show names whose sex balance shifted. For example, Sidney went
  from 8% to 62% female, and Pat and Jan moved the other way.
- `wgnd_us_female_share` is an independent cross-check. Among names with at least 1,000 births, SSA and WGND give the
  same label 97.6% of the time.

`en_US/unisex_first_names.csv` is the `unisex` subset with at least 1,000 births: 519 names, Jordan and Taylor first.
Lowering the minimum to 100 births gives 1,533 names.

`en_US/last_name_ethnicity.csv` holds Census percentages per surname: `pct_white`, `pct_black`, `pct_hispanic`, `pct_api`
(Asian and Pacific Islander), `pct_aian` (American Indian and Alaska Native), and `pct_2prace` (two or more races), plus
`top_group`. Census suppresses small cells as "(S)". The script spreads the leftover percentage evenly across the
suppressed cells, and `suppressed_cells` counts them.

`en_US/first_name_ethnicity.csv` holds the Tzioumis percentages, with the same groups, per first name.

`other_locales/{locale}_first_name_sex.csv` holds one row per WGND spelling for one locale, with its `key`,
`female_share`, `unknown_share` (the share WGND could not assign to a sex), and `label`.

| Locale | Names | Notes |
| ------ | ----- | ----- |
| en_GB | 72,070 | Good. Some names differ from the US: Jamie is 94% male in the UK and 76% female in the US. |
| en_IN | — | Not committed: 2.5M noisy entries (63 MB), and WGND has shares, not counts, to filter them by. |
| de_DE | 5,429 | 130 spellings with umlauts or accents. |
| ja_JP | 3,473 | Romanized only. Won't match names written in kanji or kana. |
| ko_KR | 10,538 | 7,696 in Hangul and the rest romanized. For example, 서연 is female, 민준 male, and 지우 unisex. |
| fr_FR | 2,963 | 452 names have both an accented and an unaccented row. |
| es_ES | 53,938 | Not a Nemotron-Personas locale; included for Spanish names in any dataset. 65% are compound names such as "José María". Only 311 names have an accented spelling in WGND, so most, such as "Maria Jose", appear only without accents. Latin American countries have fewer than 41 names each in WGND and are not built. |
| pt_BR | 448 | Too thin to use. |
| en_SG | 8 | Unusable. |
| hi_Deva_IN, hi_Latn_IN | — | Not built. WGND has no separate entry for them. |

## Results on dev data

Measured on patient_events (distinct people) and telco_churn. Only aggregates were computed; no input names were stored.

For sex, the SSA file covers 94% of first names in both datasets. Taking the more likely sex matches the dataset's sex
column for 98% of covered names. For names labeled `female` or `male` (89% of names), it matches 99.7% of the time.
Nemotron-Personas-derived counts reached 95%.

For race/ethnicity on patient_events:

- The Census table covers 94.5% of last names; Tzioumis covers 71% of first names.
- The gap between the original and replacement group mix is measured as total variation distance (0 means identical).
  It is 0.58 with unconditioned sampling and 0.15 with first + last name inference. Re-estimating the group mix from the
  column's own names first brings it to 0.095.
- With that re-estimation, 80% of people get a replacement drawn from their actual group.
- Most of the signal comes from last names.
- Names separate Black from White poorly, so Black people are under-counted (6% inferred vs 13% actual).

## Open review points

1. Sex label cutoffs, and the minimum births for the unisex list.
2. Whether to use names labeled `mostly_*` as female/male or treat them like `unisex`.
3. Census/Tzioumis groups are broader than the persona `ethnic_background` values. `api` would map to the
   east/southeast/south Asian and Pacific Islander values, and `hispanic` to all Hispanic subgroups. `2prace` has no
   persona equivalent.
4. Which locales to support beyond en_US and en_GB.
