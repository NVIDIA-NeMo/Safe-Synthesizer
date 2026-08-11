# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Faker / PGM / managed persona backends, and the names and addresses they write."""

from __future__ import annotations

import re
from pathlib import Path
from random import Random

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.pii_replacement import (
    PersonaColumnSet,
    PersonaMatchColumn,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementSettings,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.entities import Config, config_from_replace_pii
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
)
from tests.pii_replacer.helpers import FIRSTS, LASTS, PgmCheckout, column_spec, persona_set, pgm_checkout


def test_faker_persona_names_conditioned_on_sex():
    """Regression: the persona engine must condition synthetic names on the source
    ``sex`` demographic. A bug dropped the (sex, ethnicity) bucket signature so names
    were drawn with a random gender, producing e.g. a female name for a Male row.
    """
    from nemo_safe_synthesizer.pii_replacer.replacement import PersonaEngine, extract_instances

    n = 40
    df = pd.DataFrame(
        {
            "Name": [f"Person {i}" for i in range(n)],
            "Gender": ["Male" if i % 2 else "Female" for i in range(n)],
        }
    )
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="primary_person",
                columns_to_replace=[
                    PiiColumnPlan(column_name="Name", entity_type=PiiEntity.full_name),
                ],
                match_persona_by=[
                    PersonaMatchColumn(persona_attribute="gender", column_name="Gender"),
                ],
            )
        ],
    )
    runtime = config_from_replace_pii(
        ReplacePiiConfig(
            person=PiiPersonConfig(backend=PiiPersonBackend.faker), replacement=PiiReplacementSettings(locale="en_US")
        )
    )

    instances = extract_instances(df, plan, runtime)
    assert len(instances) == n
    engine = PersonaEngine(runtime, len(instances))
    assert engine.backend == "faker"
    engine.assign(instances)

    # Every instance carries the source sex, and its sampled person matches it.
    assert all(inst.sex in ("Male", "Female") for inst in instances)
    assert all(inst.synthetic_person is not None and inst.synthetic_person["sex"] == inst.sex for inst in instances)


@pytest.mark.parametrize(
    ("locale", "checkout", "message"),
    [
        pytest.param("ja_JP", "complete", "locale 'en_US' only", id="unsupported_locale"),
        pytest.param("en_US", "absent", "could not be loaded", id="missing_source"),
    ],
)
def test_pgm_backend_fails_loudly_when_unavailable(locale: str, checkout: PgmCheckout, message: str, tmp_path: Path):
    """Falling back would replace columns by a method the plan never described."""
    from nemo_safe_synthesizer.pii_replacer.replacement import PersonaEngine

    cfg = ReplacePiiConfig(
        replacement=PiiReplacementSettings(locale=locale),
        person=PiiPersonConfig(backend=PiiPersonBackend.pgm, sdg_pgms_src=str(pgm_checkout(tmp_path, checkout))),
    )
    with pytest.raises(ParameterError, match=message):
        PersonaEngine(config_from_replace_pii(cfg), 1)


def test_standalone_persona_entity_uses_real_faker_value():
    """If a persona entity is placed under the standalone section, it should use the real Faker value."""
    from nemo_safe_synthesizer.pii_replacer.replacement import run_replacement

    cfg = Config(
        locale="en_US",
        random_seed=3,
        persona_backend="faker",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )
    df = pd.DataFrame({"first_name": ["Alice", "Bob"], "email": ["alice@example.com", "bob@example.com"]})
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
            PiiColumnPlan(column_name="email", entity_type=PiiEntity.email),
        ]
    )
    out, _ = run_replacement(df, plan, cfg)
    assert list(out["first_name"]) != ["Alice", "Bob"]
    assert all(isinstance(v, str) and v.isalpha() for v in out["first_name"])
    assert all("@" in v and "." in v.split("@", 1)[-1] for v in out["email"])
    assert list(out["email"]) != list(df["email"])


def test_a_row_that_names_nobody_is_not_given_a_name():
    """A persona is written into the cells that held a person, not into the empty ones."""
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker, synth_value

    written = synth_value(
        "full_name",
        "N/A",
        {"first_name": "Robert", "last_name": "Jones"},
        seeded_faker(7, "en_US"),
        ["{LAST}, {First}"],
    )

    assert written is None


def test_an_address_is_built_from_the_person_the_row_names(fixture_numbered_email_df: pd.DataFrame):
    """The address is the one thing that has to agree with the names beside it."""
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker, synth_value

    original = str(fixture_numbered_email_df.loc[0, "contact_email"])
    written = synth_value(
        "email",
        original,
        {"first_name": "Robert", "last_name": "Jones", "email_address": "unrelated@faker.test"},
        seeded_faker(7, "en_US"),
        ["{first}.{last}.###-####@{domain}"],
        {"full_name": str(fixture_numbered_email_df.loc[0, "contact_name"])},
    )

    assert written is not None
    assert re.fullmatch(r"robert\.jones\.\d{3}-\d{4}@example\.invalid", written)


def test_replacement_follows_the_columns_own_conventions(fixture_contact_df: pd.DataFrame):
    """The point of reading the convention: the column still reads like itself."""
    cfg = ReplacePiiConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(fixture_contact_df)
    assert replacer.result is not None
    out = replacer.result.transformed_df

    assert (out["patient_name"] != fixture_contact_df["patient_name"]).all()
    assert (out["patient_email"] != fixture_contact_df["patient_email"]).all()
    for name, email, original in zip(out["patient_name"], out["patient_email"], fixture_contact_df["patient_email"]):
        last, _, first = name.partition(", ")
        assert last.isupper() and first.istitle(), name
        # The address is built from the same person, and keeps the domain it had.
        local, domain = email.split("@")
        assert local == f"{first[0].lower()}.{last.lower()}", (name, email)
        assert domain == original.split("@")[1]


def test_a_name_outside_the_columns_convention_keeps_its_own(fixture_contact_df: pd.DataFrame):
    df = fixture_contact_df.copy()
    df.loc[0, "patient_name"] = "Jane Smith"  # the one row written the other way round
    cfg = ReplacePiiConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(df)
    assert replacer.result is not None

    replaced = replacer.result.transformed_df["patient_name"]
    assert "," not in replaced[0], replaced[0]
    assert all("," in value for value in replaced[1:])


def test_a_middle_name_column_is_replaced(fixture_middle_name_df: pd.DataFrame):
    """Detected but absent from the entity vocabulary, middle names used to survive whole."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    cfg = ReplacePiiConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    plan = discover_plan(fixture_middle_name_df, group_key=None, cfg=config_from_replace_pii(cfg), config=cfg)
    spec = column_spec(persona_set(plan, "person_1").columns_to_replace, "middle_name")
    assert spec is not None and spec.entity_type == PiiEntity.middle_name

    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(fixture_middle_name_df)
    assert replacer.result is not None
    # A persona can draw the very name it replaces, and the engine then leaves that
    # cell alone, so the column changes as a whole rather than row by row.
    changed = replacer.result.transformed_df["middle_name"] != fixture_middle_name_df["middle_name"]
    assert changed.mean() > 0.9


def test_one_person_has_one_middle_name(fixture_middle_name_df: pd.DataFrame):
    """No backend supplies a middle name, so the drawn one has to hold across columns."""
    cfg = ReplacePiiConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(fixture_middle_name_df)
    assert replacer.result is not None
    out = replacer.result.transformed_df

    for full, middle in zip(out["full_name"], out["middle_name"]):
        assert full.split()[1] == middle, (full, middle)


def test_an_address_is_written_from_the_value_even_when_its_column_named_nothing():
    """A plan that lists no pattern still says the column holds addresses.

    Falling back to the persona's own address wrote a stranger's name beside the
    row's person, since only the PGM people carry one and Faker draws an unrelated
    one, so the value itself is read instead.
    """
    from nemo_safe_synthesizer.pii_replacer.replacement import persona_written

    persona = {"first_name": "Danielle", "last_name": "Figueroa", "email_address": "unrelated@faker.test"}

    assert (
        persona_written("email", "jane.smith@acme.com", persona, [], {"full_name": "Jane Smith"}, Random(1))
        == "danielle.figueroa@acme.com"
    )
    handle = persona_written("email", "usr4700@acme.com", persona, [], {"full_name": "Jane Smith"}, Random(1))
    assert handle is not None
    assert re.fullmatch(r"[a-z]{3}\d{4}@acme\.com", handle)  # its own shape, the column having named none
    assert "unrelated" not in handle


def test_an_address_following_no_convention_is_replaced_without_one(fixture_contact_df: pd.DataFrame):
    from nemo_safe_synthesizer.pii_replacer.patterns import value_shape_template

    df = fixture_contact_df.copy()
    df["patient_email"] = [f"usr{i}x{i * 37 % 97}@{'acme' if i % 2 else 'globex'}.com" for i in range(len(df))]
    cfg = ReplacePiiConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(df)
    assert replacer.result is not None
    out = replacer.result.transformed_df

    assert (out["patient_email"] != df["patient_email"]).all()
    for email, original, was in zip(out["patient_email"], fixture_contact_df["patient_name"], df["patient_email"]):
        local, domain = email.split("@")
        assert domain == was.split("@")[1]  # the domain is the part of it that was real
        assert local != was.split("@")[0]
        assert original.split(",")[0].lower() not in local.lower()
        # The handle keeps the shape it had, since that is all it ever said.
        assert value_shape_template(local) == value_shape_template(was.split("@")[0])


def test_a_column_of_several_conventions_keeps_each_of_them():
    """No convention covers the column, so no address is rewritten into another's."""
    rows = []
    for i in range(60):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        if i % 10 < 4:
            local = f"{first[0].lower()}.{last.lower()}"
        elif i % 10 < 7:
            local = f"{first.lower()}.{last.lower()}"
        else:
            local = f"{first.lower()}{last.lower()}"
        rows.append({"patient_name": f"{first} {last}", "patient_email": f"{local}@acme.com"})
    df = pd.DataFrame(rows)
    cfg = ReplacePiiConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(df)
    assert replacer.result is not None
    out = replacer.result.transformed_df

    for i, (name, email) in enumerate(zip(out["patient_name"], out["patient_email"])):
        first, last = (part.lower() for part in name.split())
        expected = f"{first[0]}.{last}" if i % 10 < 4 else (f"{first}.{last}" if i % 10 < 7 else f"{first}{last}")
        assert email == f"{expected}@acme.com", (df["patient_email"][i], name, email)
