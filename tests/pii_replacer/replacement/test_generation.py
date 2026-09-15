# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiReplacementSettings,
    PiiSamplerBackend,
    PiiSamplerConfig,
)
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.pii_replacer.replacement.generation import (
    ReplacementGenerationRequest,
    ReplacementGenerator,
)
from nemo_safe_synthesizer.pii_replacer.replacement.generators import (
    FakerReplacementGenerator,
    ManagedReplacementGenerator,
)
from nemo_safe_synthesizer.pii_replacer.replacement.types import CanonicalValue


class _FakeGenerator:
    backend = PiiSamplerBackend.FAKER

    def generate(self, request: ReplacementGenerationRequest) -> str:
        return f"synthetic-{request.entity_type.value}"


class _GenderAwareFake:
    def first_name_female(self) -> str:
        return "FEMALE"

    def first_name_male(self) -> str:
        return "MALE"

    def first_name(self) -> str:
        return "GENERIC"

    def last_name(self) -> str:
        return "LAST"


def _generate(generator: ReplacementGenerator, request: ReplacementGenerationRequest) -> str:
    return generator.generate(request)


def _request(
    entity_type: EntityType = EntityType.FIRST_NAME,
    original_value: str = "Ada",
    *,
    dependencies: tuple[tuple[EntityType, CanonicalValue | None], ...] = (),
    pattern: str | None = None,
    seed: int = 42,
    resolved_labels: tuple[tuple[EntityType, tuple[str, ...] | None], ...] = (),
) -> ReplacementGenerationRequest:
    return ReplacementGenerationRequest(
        entity_type=entity_type,
        original_value=original_value,
        effective_dependency_tuple=dependencies,
        pattern=pattern,
        seed=seed,
        resolved_dependency_labels=resolved_labels,
    )


def _faker_generator(*, locale: str = "en_US") -> FakerReplacementGenerator:
    return FakerReplacementGenerator(
        settings=PiiReplacementSettings(locale=locale),
        sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
    )


def _managed_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    people: pd.DataFrame | None = None,
) -> ManagedReplacementGenerator:
    asset_path = tmp_path / "datasets" / "en_US.parquet"
    asset_path.parent.mkdir(exist_ok=True)
    asset_path.touch()
    if people is not None:
        monkeypatch.setattr(pd, "read_parquet", lambda _path: people)
    return ManagedReplacementGenerator(
        settings=PiiReplacementSettings(),
        sampler=PiiSamplerConfig(
            backend=PiiSamplerBackend.MANAGED,
            managed_assets_path=str(tmp_path),
        ),
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
            (ManagedReplacementGenerator, PiiSamplerBackend.MANAGED),
            (FakerReplacementGenerator, PiiSamplerBackend.FAKER),
        ],
    )
    def test_named_generator_adapters_declare_their_backend(
        self,
        generator_class: type[ReplacementGenerator],
        backend: PiiSamplerBackend,
    ) -> None:
        generator = generator_class(
            settings=PiiReplacementSettings(),
            sampler=PiiSamplerConfig(backend=backend),
        )

        assert generator.backend is backend
        generated = generator.generate(_request())

        assert generated != "Ada"

    @pytest.mark.parametrize(
        ("generator_class", "backend"),
        [
            (ManagedReplacementGenerator, PiiSamplerBackend.FAKER),
            (FakerReplacementGenerator, PiiSamplerBackend.MANAGED),
        ],
    )
    def test_named_generator_adapters_reject_the_wrong_backend(
        self,
        generator_class: type[ReplacementGenerator],
        backend: PiiSamplerBackend,
    ) -> None:
        with pytest.raises(ValueError, match="requires the .* sampler backend"):
            generator_class(
                settings=PiiReplacementSettings(),
                sampler=PiiSamplerConfig(backend=backend),
            )

    def test_generator_is_a_structural_interface_for_one_replacement(self) -> None:
        request = _request(
            EntityType.EMAIL,
            "ada@example.com",
            dependencies=((EntityType.ORGANIZATION, CanonicalValue(type_tag="string", normalized_value="example")),),
            pattern="{first_name}.{last_name}@example.com",
        )

        assert _generate(_FakeGenerator(), request) == "synthetic-email"

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
        ("dependency_value", "resolved_labels", "expected"),
        [
            ("Female", (), "FEMALE"),
            ("Woman", ((EntityType.GENDER, ("female",)),), "FEMALE"),
            ("Non-binary", ((EntityType.GENDER, None),), "GENERIC"),
        ],
    )
    def test_faker_applies_gender_dependency_mappings(
        self,
        dependency_value: str,
        resolved_labels: tuple[tuple[EntityType, tuple[str, ...] | None], ...],
        expected: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        generator = _faker_generator()
        monkeypatch.setattr(generator, "_faker", lambda _seed: _GenderAwareFake())
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", dependency_value)),),
            resolved_labels=resolved_labels,
        )

        assert generator.generate(request) == expected

    def test_faker_accepts_and_ignores_unsupported_dependency_mappings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        generator = _faker_generator()
        monkeypatch.setattr(generator, "_faker", lambda _seed: _GenderAwareFake())
        request = _request(
            dependencies=((EntityType.ETHNIC_BACKGROUND, CanonicalValue("string", "Asian")),),
            resolved_labels=((EntityType.ETHNIC_BACKGROUND, ("east asian",)),),
        )

        assert generator.generate(request) == "GENERIC"

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

    def test_birth_date_is_shifted_within_one_year_and_preserves_pattern(self) -> None:
        generator = _faker_generator()
        request = _request(
            EntityType.DATE_OF_BIRTH,
            "12/10/1815",
            pattern="%m/%d/%Y",
        )

        replacement = generator.generate(request)

        delta = datetime.strptime(replacement, "%m/%d/%Y") - datetime.strptime(request.original_value, "%m/%d/%Y")
        assert 1 <= abs(delta.days) <= 365

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

    def test_vague_birth_date_is_not_generation_input(self) -> None:
        generator = _faker_generator()

        with pytest.raises(GenerationError, match="complete parseable date"):
            generator.generate(_request(EntityType.DATE_OF_BIRTH, "spring"))

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

    def test_managed_generator_reads_the_configured_locale_asset(
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

        assert _managed_generator(tmp_path, monkeypatch, people).generate(_request()) == "SyntheticAda"

    def test_managed_generator_uses_address_components_from_the_selected_asset_row(
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

        generator = _managed_generator(tmp_path, monkeypatch, people)
        assert generator.generate(_request(EntityType.STREET_ADDRESS, "1 Main Street")) == "42 Analytical Engine Way"

    def test_managed_generator_samples_each_value_independently(
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
        generator = _managed_generator(tmp_path, monkeypatch, people)

        assert generator.generate(_request(seed=0)) == "FirstRow"
        assert generator.generate(_request(EntityType.LAST_NAME, "Lovelace", seed=1)) == "SecondLast"

    def test_managed_generator_matches_dependency_labels_case_insensitively(
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
        generator = _managed_generator(tmp_path, monkeypatch, people)
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", "Female")),),
            seed=1,
        )

        assert generator.generate(request) == "Selected"

    def test_managed_generator_maps_one_dependency_value_to_a_candidate_union(
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
        generator = _managed_generator(tmp_path, monkeypatch, people)
        request = _request(
            dependencies=((EntityType.ETHNIC_BACKGROUND, CanonicalValue("string", "Asian")),),
            seed=1,
            resolved_labels=((EntityType.ETHNIC_BACKGROUND, ("east asian", "south asian")),),
        )

        assert generator.generate(request) == "South"

    def test_managed_generator_caches_positions_by_resolved_dependency_labels(
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
        generator = _managed_generator(tmp_path, monkeypatch, people)

        def generate(source_label: str, seed: int) -> str:
            return generator.generate(
                _request(
                    dependencies=(
                        (EntityType.GENDER, CanonicalValue("string", "female")),
                        (EntityType.ETHNIC_BACKGROUND, CanonicalValue("string", source_label)),
                    ),
                    seed=seed,
                    resolved_labels=((EntityType.ETHNIC_BACKGROUND, ("east asian", "south asian")),),
                )
            )

        assert generate("Asian", 0) == "EastWoman"
        assert generate("AAPI", 1) == "SouthWoman"
        assert intersection_count == 1

    def test_managed_generator_null_mapping_disables_the_dependency_condition(
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
        generator = _managed_generator(tmp_path, monkeypatch, people)
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", "Non-Binary")),),
            seed=1,
            resolved_labels=((EntityType.GENDER, None),),
        )

        assert generator.generate(request) == "Second"

    def test_managed_generator_rejects_a_dependency_with_no_candidates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame({"first_name": ["First"], "sex": ["female"]})
        generator = _managed_generator(tmp_path, monkeypatch, people)
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", "Secret Source Label")),),
            seed=1,
        )

        with pytest.raises(GenerationError, match="gender.*candidate_count=0") as error:
            generator.generate(request)
        assert "Secret Source Label" not in str(error.value)

    def test_managed_generator_reads_selected_columns_and_reuses_candidate_indexes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        read_columns: list[list[str]] = []
        read_dtype_backends: list[str] = []

        monkeypatch.setattr(
            "nemo_safe_synthesizer.pii_replacer.replacement.generators.managed._available_parquet_columns",
            lambda _path: frozenset({"first_name", "sex", "persona", "detailed_persona"}),
        )

        def read_parquet(_path: Path, *, columns: list[str], dtype_backend: str) -> pd.DataFrame:
            read_columns.append(columns)
            read_dtype_backends.append(dtype_backend)
            return pd.DataFrame(
                {
                    "first_name": ["Selected", "NotSelected"],
                    "sex": ["female", "male"],
                }
            )

        monkeypatch.setattr(pd, "read_parquet", read_parquet)
        generator = _managed_generator(tmp_path, monkeypatch)
        request = _request(
            dependencies=((EntityType.GENDER, CanonicalValue("string", "female")),),
            seed=0,
        )

        assert generator.generate(request) == "Selected"
        assert generator.generate(request) == "Selected"
        assert read_columns == [["first_name", "sex"]]
        assert read_dtype_backends == ["pyarrow"]

    def test_managed_generator_combines_casefolded_arrow_dependency_labels(
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
        generator = _managed_generator(tmp_path, monkeypatch, people)

        def generate(seed: int) -> str:
            return generator.generate(
                _request(
                    dependencies=((EntityType.GENDER, CanonicalValue("string", "female")),),
                    seed=seed,
                )
            )

        assert generate(0) == "First"
        assert generate(1) == "Second"

    def test_managed_generator_leaves_unchanged_candidate_for_executor_to_resample(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_if_faker_is_used(
            _generator: FakerReplacementGenerator,
            _request: ReplacementGenerationRequest,
        ) -> str:
            pytest.fail("an unchanged managed value should be resampled, not delegated to Faker")

        monkeypatch.setattr(FakerReplacementGenerator, "generate", fail_if_faker_is_used)
        generator = _managed_generator(tmp_path, monkeypatch, pd.DataFrame({"first_name": ["Ada"]}))
        request = _request()

        assert generator.generate(request) == request.original_value

    def test_managed_generator_delegates_the_same_value_request_when_managed_data_is_incomplete(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        people = pd.DataFrame({"first_name": ["ManagedFirst"]})
        delegated_requests = _record_faker_fallback(monkeypatch, "Faker Full Name")
        generator = _managed_generator(tmp_path, monkeypatch, people)
        request = _request(EntityType.FULL_NAME, "Ada Lovelace")

        assert generator.generate(request) == "Faker Full Name"
        assert delegated_requests == [request]

    def test_managed_generator_delegates_patterned_phone_generation_to_faker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        delegated_requests = _record_faker_fallback(monkeypatch, "202-555-0101")
        generator = ManagedReplacementGenerator(
            settings=PiiReplacementSettings(),
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.MANAGED),
        )
        request = _request(
            EntityType.PHONE_NUMBER,
            "202-555-9999",
            pattern="###-###-####",
        )

        assert generator.generate(request) == "202-555-0101"
        assert delegated_requests == [request]
