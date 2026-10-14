# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from faker.providers.person.en_US import Provider as FakerPersonProvider
from faker.providers.person.fr_FR import Provider as FrenchFakerPersonProvider

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiSamplerBackend,
    PiiSamplerConfig,
)
from nemo_safe_synthesizer.errors import InternalError, ParameterError
from nemo_safe_synthesizer.pii_replacer.replacement import birth_dates
from nemo_safe_synthesizer.pii_replacer.replacement.generation import (
    ReplacementGenerationRequest,
    ReplacementGenerator,
)
from nemo_safe_synthesizer.pii_replacer.replacement.generators import (
    FakerReplacementGenerator,
    NemotronPersonasReplacementGenerator,
)
from nemo_safe_synthesizer.pii_replacer.replacement.types import CanonicalValue


def _construct(factory: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Call ``factory`` with deliberately invalid arguments that its type hints would reject."""
    return factory(*args, **kwargs)


_FAKER_FEMALE_FIRST_NAMES = frozenset(FakerPersonProvider.first_names_female)
_FAKER_MALE_FIRST_NAMES = frozenset(FakerPersonProvider.first_names_male)


def _request(
    entity_type: EntityType = EntityType.FIRST_NAME,
    original_value: str = "Ada",
    *,
    dependencies: tuple[tuple[EntityType, CanonicalValue | None], ...] = (),
    pattern: str | None = None,
    seed: int = 42,
    resolved_values: tuple[tuple[EntityType, tuple[str, ...]], ...] = (),
) -> ReplacementGenerationRequest:
    return ReplacementGenerationRequest(
        entity_type=entity_type,
        original_value=original_value,
        effective_dependency_tuple=dependencies,
        pattern=pattern,
        seed=seed,
        resolved_dependency_values=resolved_values,
    )


def _faker_generator(*, locale: str = "en_US") -> FakerReplacementGenerator:
    return FakerReplacementGenerator(
        sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        locale=locale,
    )


def _nemotron_personas_generator(
    tmp_path: Path,
    people: pd.DataFrame | None = None,
    *,
    locale: str = "en_US",
) -> NemotronPersonasReplacementGenerator:
    """Build a generator over ``people`` written as the locale dataset, or over an empty, unreadable file."""
    asset_path = tmp_path / "datasets" / f"{locale}.parquet"
    asset_path.parent.mkdir(exist_ok=True)
    if people is None:
        asset_path.touch()
    else:
        people.to_parquet(asset_path)
    return NemotronPersonasReplacementGenerator(
        sampler=PiiSamplerConfig(
            backend=PiiSamplerBackend.NEMOTRON_PERSONAS,
            nemotron_personas_path=str(tmp_path),
        ),
        locale=locale,
    )


def _record_faker_fallback(
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> list[ReplacementGenerationRequest]:
    requests: list[ReplacementGenerationRequest] = []

    def generate(
        _generator: FakerReplacementGenerator,
        request: ReplacementGenerationRequest,
    ) -> str:
        requests.append(request)
        return replacement

    monkeypatch.setattr(FakerReplacementGenerator, "generate", generate)
    return requests


@pytest.mark.unit
class TestReplacementGenerator:
    @pytest.mark.parametrize(
        ("generator_class", "backend"),
        [
            (NemotronPersonasReplacementGenerator, PiiSamplerBackend.NEMOTRON_PERSONAS),
            (FakerReplacementGenerator, PiiSamplerBackend.FAKER),
        ],
    )
    def test_named_generator_adapters_declare_their_backend(
        self,
        generator_class: type[ReplacementGenerator],
        backend: PiiSamplerBackend,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "datasets").mkdir()
        pd.DataFrame({"first_name": ["Persona"]}).to_parquet(tmp_path / "datasets" / "en_US.parquet")
        generator = generator_class(
            sampler=PiiSamplerConfig(backend=backend, nemotron_personas_path=str(tmp_path)),
            locale="en_US",
        )

        assert generator.backend is backend
        generated = generator.generate(_request())

        assert generated != "Ada"

    @pytest.mark.parametrize(
        ("generator_class", "backend"),
        [
            (NemotronPersonasReplacementGenerator, PiiSamplerBackend.FAKER),
            (FakerReplacementGenerator, PiiSamplerBackend.NEMOTRON_PERSONAS),
        ],
    )
    def test_named_generator_adapters_reject_the_wrong_backend(
        self,
        generator_class: type[ReplacementGenerator],
        backend: PiiSamplerBackend,
    ) -> None:
        with pytest.raises(InternalError, match="requires the .* sampler backend"):
            generator_class(
                sampler=PiiSamplerConfig(backend=backend),
                locale="en_US",
            )

    @pytest.mark.parametrize(
        ("entity_type", "original"),
        [
            (EntityType.FIRST_NAME, "Ada"),
            (EntityType.MIDDLE_NAME, "Augusta"),
            (EntityType.LAST_NAME, "Lovelace"),
            (EntityType.FULL_NAME, "Ada Lovelace"),
            (EntityType.EMAIL, "ada@example.com"),
            (EntityType.PHONE_NUMBER, "+1-202-555-0101"),
            (EntityType.DATE_OF_BIRTH, "1815-12-10"),
            (EntityType.STREET_ADDRESS, "1 Main Street"),
            (EntityType.SSN, "123-45-6789"),
            (EntityType.NATIONAL_ID, "123-45-6789"),
            (EntityType.CREDIT_DEBIT_CARD, "4111111111111111"),
            (EntityType.API_KEY, "sk-ABC123"),
            (EntityType.IPV4, "192.0.2.1"),
            (EntityType.IPV6, "2001:db8::1"),
            (EntityType.UNIQUE_IDENTIFIER, "550e8400-e29b-41d4-a716-446655440000"),
        ],
    )
    def test_faker_supports_every_structured_entity(self, entity_type: EntityType, original: str) -> None:
        generator = _faker_generator()
        request = _request(entity_type, original)

        replacement = generator.generate(request)

        assert replacement
        assert replacement != original
        assert generator.generate(request) == replacement

    @pytest.mark.parametrize(
        ("original", "shape"),
        [
            ("123-45-6789", r"[0-9]{3}-[0-9]{2}-[0-9]{4}"),
            ("D1234567", r"[A-Z][0-9]{7}"),
            ("12-3456789", r"[0-9]{2}-[0-9]{7}"),
            ("AB 12 34 56 C", r"[A-Z]{2} [0-9]{2} [0-9]{2} [0-9]{2} [A-Z]"),
        ],
    )
    def test_national_id_keeps_the_original_shape(self, original: str, shape: str) -> None:
        generator = _faker_generator()

        for seed in range(20):
            replacement = generator.generate(_request(EntityType.NATIONAL_ID, original, seed=seed))

            assert re.fullmatch(shape, replacement), replacement
            assert replacement != original

    def test_request_is_immutable_and_hides_sensitive_inputs_from_repr(self) -> None:
        request = _request(
            EntityType.FULL_NAME,
            "Ada Lovelace",
            dependencies=(
                (
                    EntityType.ORGANIZATION,
                    CanonicalValue(type_tag="string", normalized_value="Analytical Engines"),
                ),
            ),
        )

        with pytest.raises(FrozenInstanceError):
            setattr(request, "seed", 7)
        assert "Ada Lovelace" not in repr(request)
        assert "Analytical Engines" not in repr(request)

    @pytest.mark.parametrize(
        "dependency_tuple",
        [
            [(EntityType.ORGANIZATION, None)],
            (("organization", None),),
            ((EntityType.ORGANIZATION, "example"),),
            ((EntityType.ORGANIZATION,),),
        ],
    )
    def test_request_rejects_malformed_dependency_tuples(self, dependency_tuple: object) -> None:
        with pytest.raises(InternalError, match="effective_dependency_tuple must"):
            _construct(
                ReplacementGenerationRequest,
                entity_type=EntityType.EMAIL,
                original_value="ada@example.com",
                effective_dependency_tuple=dependency_tuple,
                pattern=None,
                seed=42,
            )

    def test_request_requires_resolved_dependency_values_to_be_a_tuple(self) -> None:
        with pytest.raises(InternalError, match="resolved_dependency_values must be a tuple"):
            _construct(
                ReplacementGenerationRequest,
                entity_type=EntityType.FIRST_NAME,
                original_value="Ada",
                effective_dependency_tuple=(),
                pattern=None,
                seed=42,
                resolved_dependency_values=[],
            )

    def test_importing_the_contract_does_not_load_generator_implementations(self) -> None:
        code = (
            "import sys\n"
            "import nemo_safe_synthesizer.pii_replacer.replacement.generation\n"
            "loaded = [m for m in sys.modules if m.startswith('nemo_safe_synthesizer.pii_replacer.replacement.generators')]\n"
            "assert not loaded, loaded\n"
        )

        subprocess.run([sys.executable, "-c", code], check=True)

    def test_faker_generation_is_deterministic_for_equal_requests(self) -> None:
        generator = _faker_generator()
        request = _request(
            EntityType.UNIQUE_IDENTIFIER,
            "USER-ab12",
            pattern="USR-^^####",
            seed=91,
        )

        assert generator.generate(request) == generator.generate(request)
        assert generator.generate(request).startswith("USR-")

    @pytest.mark.parametrize(
        ("dependency_value", "resolved_values", "allowed_names"),
        [
            ("Female", (), _FAKER_FEMALE_FIRST_NAMES),
            ("Woman", ((EntityType.GENDER, ("female",)),), _FAKER_FEMALE_FIRST_NAMES),
            ("M", ((EntityType.GENDER, ("male",)),), _FAKER_MALE_FIRST_NAMES),
            ("F", ((EntityType.GENDER, ("femme",)),), _FAKER_FEMALE_FIRST_NAMES),
            ("F", ((EntityType.GENDER, ("女",)),), _FAKER_FEMALE_FIRST_NAMES),
            ("M", ((EntityType.GENDER, ("männlich",)),), _FAKER_MALE_FIRST_NAMES),
        ],
    )
    def test_faker_applies_gender_dependency_mappings(
        self,
        dependency_value: str,
        resolved_values: tuple[tuple[EntityType, tuple[str, ...]], ...],
        allowed_names: frozenset[str],
    ) -> None:
        generator = _faker_generator()

        names = {
            generator.generate(
                _request(
                    dependencies=((EntityType.GENDER, CanonicalValue("string", dependency_value)),),
                    resolved_values=resolved_values,
                    seed=seed,
                )
            )
            for seed in range(20)
        }

        assert names <= allowed_names

    @pytest.mark.parametrize(
        ("dependency", "resolved_values"),
        [
            ((EntityType.GENDER, CanonicalValue("string", "Non-binary")), ()),
            (
                (EntityType.ETHNIC_BACKGROUND, CanonicalValue("string", "Asian")),
                ((EntityType.ETHNIC_BACKGROUND, ("east asian",)),),
            ),
        ],
    )
    def test_faker_uses_ungendered_names_without_a_supported_gender(
        self,
        dependency: tuple[EntityType, CanonicalValue],
        resolved_values: tuple[tuple[EntityType, tuple[str, ...]], ...],
    ) -> None:
        generator = _faker_generator()

        names = {
            generator.generate(_request(dependencies=(dependency,), resolved_values=resolved_values, seed=seed))
            for seed in range(40)
        }

        assert names - _FAKER_FEMALE_FIRST_NAMES
        assert names - _FAKER_MALE_FIRST_NAMES

    def test_faker_preserves_uuid_shape(self) -> None:
        generator = _faker_generator()
        request = _request(EntityType.UNIQUE_IDENTIFIER, "550e8400-e29b-41d4-a716-446655440000")

        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            generator.generate(request),
        )

    def test_faker_preserves_unpatterned_api_key_shape(self) -> None:
        generator = _faker_generator()
        request = _request(EntityType.API_KEY, "sk-ABC123")

        assert re.fullmatch(r"[a-z]{2}-[A-Z]{3}\d{3}", generator.generate(request))

    def test_unpatterned_identifier_replaces_non_ascii_letters_and_digits(self) -> None:
        generator = _faker_generator()

        replacement = generator.generate(_request(EntityType.UNIQUE_IDENTIFIER, "张伟-Zoë-٣4"))

        assert re.fullmatch(r"[A-Z]{2}-[A-Z][a-z]{2}-[0-9]{2}", replacement), replacement

    def test_email_pattern_uses_dependencies_and_normalizes_organization(self) -> None:
        generator = _faker_generator()
        request = _request(
            EntityType.EMAIL,
            "ada@original.example",
            dependencies=(
                (EntityType.FIRST_NAME, CanonicalValue("string", "Synthetic")),
                (EntityType.ORGANIZATION, CanonicalValue("string", "Société ACME, Inc.")),
            ),
            pattern="{first}@mail.{organization}.co.uk",
            seed=12,
        )

        assert generator.generate(request) == "synthetic@mail.societe-acme-inc.co.uk"

    @pytest.mark.parametrize(
        ("dependencies", "expected"),
        [
            (
                (
                    (EntityType.FIRST_NAME, CanonicalValue("string", "Synthetic")),
                    (EntityType.LAST_NAME, CanonicalValue("string", "Person")),
                ),
                r"synthetic\.person\d{2}@original\.example",
            ),
            (
                ((EntityType.FULL_NAME, CanonicalValue("string", "Synthetic Person")),),
                r"synthetic\.person\d{2}@original\.example",
            ),
            (((EntityType.FIRST_NAME, CanonicalValue("string", "Synthetic")),), r"synthetic\d{2}@original\.example"),
            (((EntityType.LAST_NAME, CanonicalValue("string", "Person")),), r"person\d{2}@original\.example"),
        ],
    )
    def test_unpatterned_email_uses_name_dependencies(
        self,
        dependencies: tuple[tuple[EntityType, CanonicalValue | None], ...],
        expected: str,
    ) -> None:
        request = _request(EntityType.EMAIL, "hughes_b@original.example", dependencies=dependencies)

        assert re.fullmatch(expected, _faker_generator().generate(request))

    def test_unpatterned_email_without_name_dependencies_is_sampled_whole(self) -> None:
        dependencies = ((EntityType.ORGANIZATION, CanonicalValue("string", "Acme")),)
        request = _request(EntityType.EMAIL, "ada@original.example", dependencies=dependencies)

        replacement = _faker_generator().generate(request)

        assert "@" in replacement
        assert not replacement.endswith("@original.example")

    def test_card_check_digit_respects_a_restricted_final_position(self) -> None:
        generator = _faker_generator()

        for seed in range(50):
            replacement = generator.generate(
                _request(
                    EntityType.CREDIT_DEBIT_CARD, "4111-1111-1111-1110", pattern="4###-####-####-###[05]", seed=seed
                )
            )
            digits = [int(character) for character in replacement if character.isdigit()]
            checksum = sum(digit if index % 2 else sum(divmod(digit * 2, 10)) for index, digit in enumerate(digits))

            assert re.fullmatch(r"4\d{3}-\d{4}-\d{4}-\d{3}[05]", replacement), replacement
            assert checksum % 10 == 0

    def test_birth_date_shift_is_small_nonzero_and_preserves_pattern(self) -> None:
        generator = _faker_generator()
        original = datetime.strptime("12/10/1815", "%m/%d/%Y")

        shifts = []
        for seed in range(2000):
            replacement = generator.generate(
                _request(EntityType.DATE_OF_BIRTH, "12/10/1815", pattern="%m/%d/%Y", seed=seed)
            )
            shifts.append((datetime.strptime(replacement, "%m/%d/%Y") - original).days)

        assert all(1 <= abs(days) <= 180 for days in shifts)
        assert 25 <= statistics.pstdev(shifts) <= 35
        # Redrawing zero instead of moving it leaves +1 and -1 no more likely than +2 and -2.
        assert shifts.count(1) + shifts.count(-1) <= 2 * (shifts.count(2) + shifts.count(-2)) + 10

    @pytest.mark.parametrize(
        ("original", "expected"),
        [("1990-02-14", r"\d{2}/\d{2}/\d{4}"), ("08/2010", r"\d{2}/\d{2}/\d{4}")],
    )
    def test_birth_date_cells_outside_the_pattern_are_still_replaced_in_the_pattern(
        self,
        original: str,
        expected: str,
    ) -> None:
        generator = _faker_generator()

        replacement = generator.generate(_request(EntityType.DATE_OF_BIRTH, original, pattern="%m/%d/%Y"))

        assert re.fullmatch(expected, replacement), replacement
        assert replacement != original

    @pytest.mark.parametrize("pattern", ["%Y-%m-%d %z", "%Y-%m-%dT%H:%M:%S%z", "%d %b %Y %Z"])
    def test_unparseable_birth_date_fallback_keeps_a_timezone_pattern(self, pattern: str) -> None:
        replacement = _faker_generator().generate(_request(EntityType.DATE_OF_BIRTH, "unknown", pattern=pattern))

        datetime.strptime(replacement, pattern)

    def test_birth_date_pattern_supports_named_months(self) -> None:
        request = _request(
            EntityType.DATE_OF_BIRTH,
            "December 10, 1815",
            pattern="%B %d, %Y",
        )

        replacement = _faker_generator().generate(request)

        datetime.strptime(replacement, "%B %d, %Y")
        assert replacement != request.original_value

    @pytest.mark.parametrize("original", ["5 April 1990", "April 5th, 1990", "5 avril 1990"])
    def test_natural_language_birth_date_is_shifted_to_a_safe_complete_date(self, original: str) -> None:
        generator = _faker_generator()
        request = _request(EntityType.DATE_OF_BIRTH, original)

        replacement = generator.generate(request)

        parsed_replacement = datetime.fromisoformat(replacement)
        assert datetime(1989, 4, 5) <= parsed_replacement <= datetime(1991, 4, 5)
        assert replacement != original

    def test_numeric_non_date_birth_value_skips_natural_language_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_parse(*args: object, **kwargs: object) -> None:
            raise AssertionError("dateparser must not run for a value without letters")

        monkeypatch.setattr(birth_dates.dateparser, "parse", fail_parse)

        replacement = _faker_generator().generate(_request(EntityType.DATE_OF_BIRTH, "45"))

        assert datetime(1930, 1, 1) <= datetime.fromisoformat(replacement) <= datetime(2010, 12, 31)

    def test_vague_birth_date_is_replaced_with_a_random_complete_date(self) -> None:
        replacement = _faker_generator().generate(_request(EntityType.DATE_OF_BIRTH, "spring"))

        assert datetime(1930, 1, 1) <= datetime.fromisoformat(replacement) <= datetime(2010, 12, 31)

    def test_card_pattern_produces_a_luhn_valid_number(self) -> None:
        generator = _faker_generator()
        request = _request(
            EntityType.CREDIT_DEBIT_CARD,
            "4111-1111-1111-1111",
            pattern="####-####-####-####",
            seed=3,
        )

        replacement = generator.generate(request)
        digits = [int(character) for character in replacement if character.isdigit()]
        checksum = sum(digit if index % 2 else sum(divmod(digit * 2, 10)) for index, digit in enumerate(digits))

        assert replacement != request.original_value
        assert checksum % 10 == 0

    def test_nemotron_personas_generator_reads_the_configured_locale_asset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "first_name": ["SyntheticAda"],
                "last_name": ["SyntheticLovelace"],
                "email_address": ["synthetic@example.test"],
            }
        )

        assert _nemotron_personas_generator(tmp_path, people).generate(_request()) == "SyntheticAda"

    def test_nemotron_personas_generator_builds_emails_with_faker_from_name_dependencies(
        self,
        tmp_path: Path,
    ) -> None:
        # Persona datasets have no email column; any email column in the data is ignored.
        people = pd.DataFrame({"first_name": ["SampledAda"], "email_address": ["sampled@example.test"]})
        generator = _nemotron_personas_generator(tmp_path, people)
        dependencies = (
            (EntityType.FIRST_NAME, CanonicalValue("string", "Synthetic")),
            (EntityType.LAST_NAME, CanonicalValue("string", "Person")),
        )

        named = generator.generate(_request(EntityType.EMAIL, "padams@hotmail.com", dependencies=dependencies))
        unnamed = generator.generate(_request(EntityType.EMAIL, "padams@hotmail.com"))

        assert re.fullmatch(r"synthetic\.person\d{2}@hotmail\.com", named), named
        assert unnamed != "sampled@example.test"
        assert "@" in unnamed

    def test_nemotron_personas_generator_uses_address_components_from_the_selected_asset_row(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "street_number": ["42"],
                "street_name": ["Analytical Engine Way"],
            }
        )

        generator = _nemotron_personas_generator(tmp_path, people)
        assert generator.generate(_request(EntityType.STREET_ADDRESS, "1 Main Street")) == "42 Analytical Engine Way"

    def test_nemotron_personas_generator_samples_each_value_independently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "first_name": ["FirstRow", "SecondRow"],
                "last_name": ["FirstLast", "SecondLast"],
            }
        )
        generator = _nemotron_personas_generator(tmp_path, people)

        assert generator.generate(_request(seed=0)) == "FirstRow"
        assert generator.generate(_request(EntityType.LAST_NAME, "Lovelace", seed=1)) == "SecondLast"

    def test_nemotron_personas_generator_matches_dependency_values_case_insensitively(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "first_name": ["Selected", "NotSelected"],
                "sex": ["female", "male"],
            }
        )
        generator = _nemotron_personas_generator(tmp_path, people)
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", "Female")),),
            seed=1,
        )

        assert generator.generate(request) == "Selected"

    def test_nemotron_personas_generator_maps_one_dependency_value_to_a_candidate_union(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "first_name": ["East", "Identity", "South"],
                "ethnic_background": ["east asian", "Asian", "south asian"],
            }
        )
        generator = _nemotron_personas_generator(tmp_path, people)
        request = _request(
            dependencies=((EntityType.ETHNIC_BACKGROUND, CanonicalValue("string", "Asian")),),
            seed=1,
            resolved_values=((EntityType.ETHNIC_BACKGROUND, ("east asian", "south asian")),),
        )

        assert generator.generate(request) == "South"

    def test_nemotron_personas_generator_caches_positions_by_resolved_dependency_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "first_name": ["EastWoman", "SouthWoman", "EastMan", "OtherWoman"],
                "sex": ["female", "female", "male", "female"],
                "ethnic_background": ["east asian", "south asian", "east asian", "white"],
            }
        )
        original_intersect = np.intersect1d
        intersection_count = 0

        def count_intersection(
            first: np.ndarray,
            second: np.ndarray,
            *,
            assume_unique: bool = False,
        ) -> np.ndarray:
            nonlocal intersection_count
            intersection_count += 1
            return original_intersect(first, second, assume_unique=assume_unique)

        monkeypatch.setattr(np, "intersect1d", count_intersection)
        generator = _nemotron_personas_generator(tmp_path, people)

        def generate(dataset_value: str, seed: int) -> str:
            return generator.generate(
                _request(
                    dependencies=(
                        (EntityType.GENDER, CanonicalValue("string", "female")),
                        (EntityType.ETHNIC_BACKGROUND, CanonicalValue("string", dataset_value)),
                    ),
                    seed=seed,
                    resolved_values=((EntityType.ETHNIC_BACKGROUND, ("east asian", "south asian")),),
                )
            )

        assert generate("Asian", 0) == "EastWoman"
        assert generate("AAPI", 1) == "SouthWoman"
        assert intersection_count == 1

    def test_nemotron_personas_generator_listing_every_value_allows_any_candidate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "first_name": ["First", "Second"],
                "sex": ["female", "male"],
            }
        )
        generator = _nemotron_personas_generator(tmp_path, people)
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", "Non-Binary")),),
            seed=1,
            resolved_values=((EntityType.GENDER, ("female", "male")),),
        )

        assert generator.generate(request) == "Second"

    def test_nemotron_personas_generator_rejects_a_dependency_with_no_candidates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame({"first_name": ["First"], "sex": ["female"]})
        generator = _nemotron_personas_generator(tmp_path, people)
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", "Secret Source Label")),),
            seed=1,
        )

        with pytest.raises(ParameterError, match="gender.*candidate_count=0") as error:
            generator.generate(request)
        assert "Secret Source Label" not in str(error.value)

    def test_nemotron_personas_generator_requires_its_locale_dataset(self, tmp_path: Path) -> None:
        with pytest.raises(ParameterError, match="no Nemotron-Personas data for locale 'en_US' was found"):
            NemotronPersonasReplacementGenerator(
                sampler=PiiSamplerConfig(
                    backend=PiiSamplerBackend.NEMOTRON_PERSONAS,
                    nemotron_personas_path=str(tmp_path),
                ),
                locale="en_US",
            )

    def test_nemotron_personas_generator_fails_when_its_dataset_cannot_be_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        generator = _nemotron_personas_generator(tmp_path)

        with pytest.raises(ParameterError, match="could not be read") as error:
            generator.generate(_request())
        assert error.value.__cause__ is not None

    def test_nemotron_personas_generator_names_a_missing_dependency_column(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame({"first_name": ["First"], "sex": ["female"]})
        generator = _nemotron_personas_generator(tmp_path, people)
        request = _request(
            dependencies=((EntityType.ETHNIC_BACKGROUND, CanonicalValue("string", "Asian")),),
            resolved_values=((EntityType.ETHNIC_BACKGROUND, ("east asian",)),),
        )

        with pytest.raises(ParameterError, match=r"conditions on 'ethnic_background'.*'en_US' has no column") as error:
            generator.generate(request)
        assert "'ethnicity'" in str(error.value)

    def test_nemotron_personas_generator_reads_selected_columns_and_reuses_candidate_indexes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "first_name": ["Selected", "NotSelected"],
                "sex": ["female", "male"],
                "persona": ["unused", "unused"],
                "detailed_persona": ["unused", "unused"],
            }
        )
        generator = _nemotron_personas_generator(tmp_path, people)
        reads: list[dict[str, object]] = []
        real_read_parquet = pd.read_parquet

        def read_parquet(path: Path, *, columns: list[str], dtype_backend: Literal["pyarrow"]) -> pd.DataFrame:
            reads.append({"columns": columns, "dtype_backend": dtype_backend})
            return real_read_parquet(path, columns=columns, dtype_backend=dtype_backend)

        monkeypatch.setattr(pd, "read_parquet", read_parquet)
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", "female")),),
            seed=0,
        )

        assert generator.generate(request) == "Selected"
        assert generator.generate(request) == "Selected"
        assert reads == [{"columns": ["first_name", "sex"], "dtype_backend": "pyarrow"}]

    def test_nemotron_personas_generator_combines_casefolded_arrow_dependency_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame(
            {
                "first_name": ["First", "Second", "NotSelected"],
                "sex": ["Female", "FEMALE", None],
            }
        ).convert_dtypes(dtype_backend="pyarrow")
        generator = _nemotron_personas_generator(tmp_path, people)

        def generate(seed: int) -> str:
            return generator.generate(
                _request(
                    dependencies=((EntityType.GENDER, CanonicalValue("string", "female")),),
                    seed=seed,
                )
            )

        assert generate(0) == "First"
        assert generate(1) == "Second"

    def test_nemotron_personas_generator_leaves_unchanged_candidate_for_executor_to_resample(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_if_faker_is_used(
            _generator: FakerReplacementGenerator,
            _request: ReplacementGenerationRequest,
        ) -> str:
            pytest.fail("an unchanged Nemotron-Personas value should be resampled, not delegated to Faker")

        monkeypatch.setattr(FakerReplacementGenerator, "generate", fail_if_faker_is_used)
        generator = _nemotron_personas_generator(tmp_path, pd.DataFrame({"first_name": ["Ada"]}))
        request = _request()

        assert generator.generate(request) == request.original_value

    def test_nemotron_personas_generator_delegates_the_same_value_request_when_persona_data_is_incomplete(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame({"first_name": ["PersonaFirst"]})
        delegated_requests = _record_faker_fallback(monkeypatch, "Faker Full Name")
        generator = _nemotron_personas_generator(tmp_path, people)
        request = _request(EntityType.FULL_NAME, "Ada Lovelace")

        assert generator.generate(request) == "Faker Full Name"
        assert delegated_requests == [request]

    def test_nemotron_personas_generator_delegates_patterned_phone_generation_to_faker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        delegated_requests = _record_faker_fallback(monkeypatch, "202-555-0101")
        generator = _nemotron_personas_generator(tmp_path, pd.DataFrame({"first_name": ["Persona"]}))
        request = _request(
            EntityType.PHONE_NUMBER,
            "202-555-9999",
            pattern="###-###-####",
        )

        assert generator.generate(request) == "202-555-0101"
        assert delegated_requests == [request]

    @pytest.mark.parametrize("sampler_value", ["female", "femme", "Female"])
    def test_nemotron_personas_matches_localized_sex_by_female_or_the_locale_word(
        self, tmp_path: Path, sampler_value: str
    ) -> None:
        people = pd.DataFrame(
            {"first_name": ["Claire", "Louis"], "last_name": ["Martin", "Martin"], "sex": ["Femme", "Homme"]}
        )
        generator = _nemotron_personas_generator(tmp_path, people, locale="fr_FR")

        names = {
            generator.generate(
                _request(
                    EntityType.FIRST_NAME,
                    "Marie",
                    dependencies=((EntityType.GENDER, CanonicalValue("string", "F")),),
                    resolved_values=((EntityType.GENDER, (sampler_value,)),),
                    seed=seed,
                )
            )
            for seed in range(10)
        }

        assert names == {"Claire"}

    def test_nemotron_personas_faker_fallback_keeps_localized_sex_conditioning(self, tmp_path: Path) -> None:
        # fr_FR personas write sex as Femme and Homme and have no middle names, so middle names come from Faker.
        people = pd.DataFrame(
            {
                "first_name": ["Claire", "Louis"],
                "middle_name": [None, None],
                "last_name": ["Martin", "Martin"],
                "sex": ["Femme", "Homme"],
            }
        )
        generator = _nemotron_personas_generator(tmp_path, people, locale="fr_FR")

        names = {
            generator.generate(
                _request(
                    EntityType.MIDDLE_NAME,
                    "Marie",
                    dependencies=((EntityType.GENDER, CanonicalValue("string", "F")),),
                    resolved_values=((EntityType.GENDER, ("femme",)),),
                    seed=seed,
                )
            )
            for seed in range(40)
        }

        assert names <= frozenset(FrenchFakerPersonProvider.first_names_female)
