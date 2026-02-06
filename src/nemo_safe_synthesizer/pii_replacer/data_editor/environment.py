# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import random
import re
from datetime import date, datetime, timedelta
from functools import partial
from typing import Any, Callable, List, Optional, Union

import dateutil.parser
import jinja2
import jinja2.nodes
import pandas as pd
import pycountry
import pycountry.db
from anyascii import anyascii
from faker import Faker as VanillaFaker
from faker.config import AVAILABLE_LOCALES
from faker.providers import BaseProvider
from faker.typing import SeedType
from jinja2.sandbox import SandboxedEnvironment

from ..ner.ner import NERPrediction
from .detect import EntityExtractor, EntityExtractorNoop
from .filters import partial_mask


def lookup_country(value: str) -> pycountry.db.Data:
    value = value.strip()
    COUNTRY_MAP: dict[str, str] = {
        "Russia": "Russian Federation",
        "UK": "GB",
    }
    value = COUNTRY_MAP.get(value, value)
    return pycountry.countries.lookup(value)


def lookup_locales(value: str) -> Optional[list[str]]:
    locales = [locale for locale in AVAILABLE_LOCALES if locale.endswith(lookup_country(value).alpha_2)]
    return locales if locales else None


def tld(value: Union[str, pycountry.db.Data]) -> str:
    if isinstance(value, str):
        value = lookup_country(value)
    value = value.alpha_2.lower()
    if value == "gb":
        value = "uk"
    return f".{value}"


def normalize(value, allow: str = "") -> str:
    return re.sub(rf"[^\w{allow}]", "", anyascii(value))


def sha256(default_salt: str, value, salt: Optional[str] = None) -> str:
    """
    default_salt to be set as a partial from within the Environment. salt param
    can override it and is available to the user.

    This way there is a default salt if unspecified in the template:
    `this | hash`
    or it can be specified by the user:
    `this | hash(salt="ABC")`
    """
    if salt is None:
        salt = default_salt
    return hashlib.sha256((str(salt) + str(value)).encode("utf-8")).hexdigest()


class PersonaProvider(BaseProvider):
    __provider__ = "persona"

    def persona(
        self,
        row_index: int = 1,
        email_format: str = "first_name.last_name",
        domain_type: str = "all_domains",
        gender: Optional[str] = None,
    ):
        saved_seed = self.generator._global_seed
        self.generator.seed_instance(row_index)

        is_male = False
        if gender is not None:
            # Infer gender instead of replacing with Female or Male
            if gender.lower() in {"m", "male", "man"}:
                is_male = True
            # Do a random guess on gender if not female
            elif gender.lower() not in {"f", "female", "woman"}:
                is_male = self.generator.boolean()
        else:
            gender = self.random_element(["Female", "Male"])
            is_male = gender == "Male"
        first_name = self.generator.first_name_male() if is_male else self.generator.first_name_female()
        last_name = self.generator.last_name()

        if domain_type == "free_domains":
            domain_name = self.generator.free_email_domain()
        else:
            domain_name = self.generator.domain_name()

        if email_format == "first_name":
            email = f"{first_name.lower()}@{domain_name}"
        elif email_format == "last_name":
            email = f"{last_name.lower()}@{domain_name}"
        elif email_format == "flast_name":
            email = f"{first_name[0].lower()}{last_name.lower()}@{domain_name}"
        else:
            email = f"{first_name.lower()}.{last_name.lower()}@{domain_name}"

        self.generator.seed_instance(saved_seed)

        return {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "gender": gender,
        }


class Faker:
    _fake: VanillaFaker
    global_seed: SeedType
    maybe_seed: Callable[[SeedType], VanillaFaker]

    def __init__(self, locale: Optional[list[str]] = None, seed: Optional[SeedType] = None):
        self._fake = VanillaFaker(locale)
        if locale and len(locale) <= 1:  # Can't proxy multiple locales
            self._fake.add_provider(PersonaProvider)
        if seed:
            self.global_seed = seed
            self.maybe_seed = self._fake_with_seed
        else:
            self.maybe_seed = self._fake_without_seed

    def _fake_with_seed(self, instance_seed: SeedType) -> VanillaFaker:
        self._fake.__class__.seed(f"{self.global_seed!r}{instance_seed!r}")
        return self._fake

    def _fake_without_seed(self, instance_seed: Optional[SeedType] = None) -> VanillaFaker:
        return self._fake

    def __call__(self, locale: Optional[List[str]] = None, seed: Optional[SeedType] = None):
        if locale is None:
            locale = self._fake.locales
        if seed is None:
            seed = self.global_seed
        elif seed is not False:
            self._fake_with_seed(random.random())
        else:
            self._fake_with_seed(seed)
        return self.__class__(locale, seed)

    def __getattr__(self, name):
        return getattr(self._fake, name)


def redact_entities_fn(entity: NERPrediction) -> str:
    """
    Replace entity with its label in angled braces
    """
    return f"<{entity.label}>"


def label_entities_fn(entity: NERPrediction, extended: Optional[bool] = False) -> str:
    """
    Replace entity with its text and label in angled braces
    """
    extra_attrs = ""
    if extended:
        extra_attrs = f''' source="{entity.source}" score="{entity.score}"'''
    return f"""<entity type="{entity.label}" value="{entity.text}"{extra_attrs}>"""


def hash_entities_fn(default_salt: str, entity: NERPrediction, salt: Optional[str] = None) -> str:
    """
    Replace entity with a hash of its text
    """
    return sha256(default_salt, entity.text, salt=salt)[:9]


class SafeSynthesizerFakerMethodNotFound(Exception):
    pass


def fake_entities_fn(
    hash_salt: str,
    fake: Faker,
    entity: NERPrediction,
    on_error: Optional[str] = None,
    extended: Optional[bool] = False,
) -> str:
    """
    Replace entity with faked entity of the same type

    If a fake function is not found for a specific entity, on_error param specifies what to do:
      "redact", "label", and "hash" fall back on associated *_entities_fn
      "raise" will raise an exception causing execution to stop
    """

    try:
        return getattr(fake, entity.label)()
    except AttributeError:
        pass

    if on_error is None:
        on_error = "redact"
    if on_error == "raise":
        raise SafeSynthesizerFakerMethodNotFound(f"""Unable to fake entity '{entity.text}' of type '{entity.label}'""")

    on_error_map = {
        "label": partial(label_entities_fn, extended=extended),
        "hash": partial(hash_entities_fn, hash_salt),
        "redact": redact_entities_fn,
    }

    if on_error in on_error_map:
        return on_error_map[on_error](entity)

    raise Exception(f"""Invalid on_error value: {on_error}""")


class Environment:
    _env: SandboxedEnvironment
    _fake: Faker
    entity_extractor: EntityExtractor

    def __init__(
        self,
        locales: Optional[list[str]],
        seed: SeedType,
        globals_config: Optional[dict[str, Any]] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ):
        if entity_extractor is not None:
            self.entity_extractor = entity_extractor
        else:
            self.entity_extractor = EntityExtractorNoop()
        self._env = SandboxedEnvironment(loader=jinja2.BaseLoader())
        self._fake = Faker(locale=locales, seed=seed)
        self._env.globals["fake"] = self._fake
        self._env.globals["globals"] = globals_config
        self._env.globals["random"] = random
        self._env.globals["re"] = re
        self._env.globals["timedelta"] = timedelta
        self._env.filters["hash"] = partial(sha256, str(seed))
        self._env.filters["isna"] = pd.isna
        self._env.filters["fake"] = lambda faker_type: getattr(self._fake, faker_type)()
        self._env.filters["lookup_country"] = lookup_country
        self._env.filters["lookup_locales"] = lookup_locales
        self._env.filters["normalize"] = normalize
        self._env.filters["partial_mask"] = partial_mask
        self._env.filters["tld"] = tld
        self._env.filters["date_parse"] = dateutil.parser.parse
        self._env.filters["date_shift"] = self.date_shift
        self._env.filters["date_time_shift"] = self.date_time_shift
        self._env.filters["date_format"] = self.date_format
        self._env.filters["date_time_format"] = self.date_time_format
        self._env.filters["detect_entities"] = self.entity_extractor.extract_entity_values
        self._env.filters["redact_entities"] = partial(
            self.entity_extractor.extract_and_replace_entities, redact_entities_fn
        )
        self._env.filters["label_entities"] = lambda text, entities=None, extended=False: (
            self.entity_extractor.extract_and_replace_entities(
                partial(label_entities_fn, extended=extended), text, entities
            )
        )

        self._env.filters["hash_entities"] = lambda text, entities=None, salt=None: (
            self.entity_extractor.extract_and_replace_entities(
                partial(hash_entities_fn, str(seed), salt=salt), text, entities
            )
        )
        self._env.filters["fake_entities"] = lambda text, entities=None, on_error=None, extended=False: (
            entity_extractor.extract_and_replace_entities(
                partial(
                    fake_entities_fn,
                    str(seed),
                    self._fake,
                    on_error=on_error,
                    extended=extended,
                ),
                text,
                entities,
            )
        )
        self.ner_cacheable_filters = set(
            [
                "detect_entities",
                "redact_entities",
                "label_entities",
                "hash_entities",
                "fake_entities",
            ]
        )

    def maybe_seed(self, instance_seed: SeedType):
        self._fake.maybe_seed(instance_seed)

    def template_to_fnames(self, template_str) -> set[str]:
        """
        Parse the template's AST and return set of filter/function names.
        """
        retval = set()
        ast = self._env.parse(f"{{{{{template_str}}}}}")
        fns = [f for f in ast.find_all((jinja2.nodes.Name, jinja2.nodes.Filter))]
        for fn in fns:
            try:
                retval.add(fn.name)
            except AttributeError:
                # This shouldn't happen, but don't crash if node is malformed.
                pass
        return retval

    def make_template(self, template_str: str) -> jinja2.Template:
        return self._env.from_string(
            f"{{{{{template_str}|default(__tmp_literal)}}}}",
            globals={"__tmp_literal": template_str},
        )

    def _get_delta(self, base: Union[date, datetime, str]) -> timedelta:
        if isinstance(base, str):
            base = dateutil.parser.parse(base)
        elif isinstance(base, date):
            base = datetime.combine(base, datetime.min.time())
        # We use the tzinfo if available from the base to avoid
        # offset-naive vs. offset-aware errors
        delta = datetime.today().replace(tzinfo=base.tzinfo) - base
        return delta

    def date_shift(
        self,
        value: Union[date, datetime, str],
        min_offset: Union[date, datetime, timedelta, str, int] = "-30y",
        max_offset: Union[date, datetime, timedelta, str, int] = "today",
    ) -> datetime:
        """
        Pick a random date from interval. Interval defined by input value
        and an offset on either side defined by filter params.

        For instance `2000-01-01 | date_shift('-1y', '+1y')` selects from
        an interval between 1999-01-01 and 2001-01-01, as defined by faker.

        """
        delta = self._get_delta(value)
        try:
            fake_date = self._fake.date_between(min_offset, max_offset)
        except ValueError:
            # This happens if min/max offset are reversed. To avoid
            # parsing all of faker's options, just reverse and try again.
            fake_date = self._fake.date_between(max_offset, min_offset)
        fake_date = fake_date - delta
        return fake_date

    def date_time_shift(
        self,
        value: Union[date, datetime, str],
        min_offset: Union[date, datetime, timedelta, str, int] = "-30y",
        max_offset: Union[date, datetime, timedelta, str, int] = "now",
    ) -> datetime:
        """
        Pick a random datetime from interval. Interval defined by input value
        and an offset on either side defined by filter params.

        For instance `2000-01-01 00:00 | date_time_shift('-1y', '+1y')` selects from
        an interval between 1999-01-01 00:00 and 2001-01-01 00:00, as defined by faker.

        """
        delta = self._get_delta(value)
        try:
            fake_date = self._fake.date_time_between(min_offset, max_offset)
        except ValueError:
            # This happens if min/max offset are reversed. To avoid
            # parsing all of faker's options, just reverse and try again.
            fake_date = self._fake.date_time_between(max_offset, min_offset)
        fake_date = fake_date - delta
        return fake_date

    def date_format(self, value: Union[date, datetime, str], format: str = "%Y-%m-%d"):
        return self.date_time_format(value, format=format)

    def date_time_format(self, value: Union[date, datetime, str], format: str = "%Y-%m-%d %H:%M:%S"):
        if isinstance(value, str):
            value = dateutil.parser.parse(value)
        return value.strftime(format)

    def fake_filter(self, fake_type: str) -> str:
        entity_to_fake_function_mapping = {
            "birthdate": partial(self._fake.date_between, start_date="-100y"),
            "email_address": self._fake.free_email,
            "work_email": self._fake.free_email,
            "street": self._fake.street_name,
            "profession": self._fake.job,
            "weekday": self._fake.day_of_week,
            "full_address": self._fake.address,
            "residence": self._fake.address,
        }

        fake_type = fake_type.lower().replace(" ", "_")
        fn = entity_to_fake_function_mapping.get(fake_type)
        if fn is None:
            return getattr(self._fake, fake_type)()
        else:
            return fn()
