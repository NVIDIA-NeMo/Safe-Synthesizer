# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Any, Self

from faker.config import AVAILABLE_LOCALES
from pydantic import Field, field_validator, model_validator

from ..configurator.parameters import (
    Parameters,
)
from ..observability import get_logger
from .base import NSSBaseModel
from .types import OptionalListOrInt, OptionalListOrStr, OptionalStrList

logger = get_logger(__name__)

__all__ = [
    "PiiReplacerConfig",
    "Globals",
    "StepDefinition",
    "Column",
    "Row",
    "RowActions",
    "GlinerConfig",
    "ColumnActions",
    "ClassifyConfig",
    "DEFAULT_PII_TRANSFORM_CONFIG",
]

MAX_32_BIT_INT = 2**31 - 1


class Column(NSSBaseModel):
    """Rule matcher for selecting columns by name, position, condition, entity, or type."""

    name: str | None = Field(description="Column name.", default=None)

    position: OptionalListOrInt = Field(description="Column position.", default=None)

    condition: str | None = Field(description="Column condition.", default=None)

    value: str | None = Field(description="Rename to value.", default=None)

    entity: OptionalListOrStr = Field(description="Column entity match.", default=None)

    type: OptionalListOrStr = Field(description="Column type match.", default=None)

    @model_validator(mode="before")
    @classmethod
    def identifier_required(cls, values: Any) -> Any:
        """Ensure at least one column identifier field is provided."""
        # Handle both dict and model instance cases (Pydantic v2 compatibility)
        if not isinstance(values, dict):
            return values
        if (
            values.get("name") is None
            and values.get("condition") is None
            and values.get("entity") is None
            and values.get("position") is None
            and values.get("type") is None
        ):
            raise ValueError("column rule must contain one of name, position, entity, type or condition.")
        return values


class ColumnActions(NSSBaseModel):
    """Container for column add, drop, and rename operations."""

    add: list[Column] | None = Field(description="Columns to add.", default=None)

    drop: list[Column] | None = Field(description="Columns to drop.", default=None)

    rename: list[Column] | None = Field(description="Columns to rename.", default=None)


class Row(NSSBaseModel):
    """Rule matcher for selecting rows by name, condition, entity, or type."""

    # eg PrimaryKey or [AddressLine1, AddressLine2]
    name: OptionalListOrStr = Field(description="Row name.", default=None)

    condition: str | None = Field(description="Row condition match.", default=None)

    foreach: str | None = Field(description="Foreach expression.", default=None)

    value: str | None = Field(description="Row value definition.", default=None)

    entity: OptionalListOrStr = Field(description="Row entity match.", default=None)

    type: OptionalListOrStr = Field(description="Row type match.", default=None)

    fallback_value: str | None = Field(description="Row fallback value.", default=None)

    description: str | None = Field(description="Rule description for human consumption.", default=None)

    @model_validator(mode="before")
    @classmethod
    def identifier_required(cls, values: Any) -> Any:
        """Ensure at least one row identifier field is provided."""
        # Handle both dict and model instance cases (Pydantic v2 compatibility)
        if not isinstance(values, dict):
            return values
        if (
            values.get("name") is None
            and values.get("condition") is None
            and values.get("entity") is None
            and values.get("type") is None
        ):
            raise ValueError("row rule must contain one of name, entity, type or condition.")

        if values.get("foreach") is not None and values.get("value") is None:
            raise ValueError(
                "foreach without value field. If a rule contains foreach, it must also "
                "include a value field to iterate on."
            )
        return values


class RowActions(NSSBaseModel):
    """Container for row drop and update operations."""

    drop: list[Row] | None = Field(description="Rows to drop.", default=None)

    update: list[Row] | None = Field(description="Rows to update.", default=None)


class StepDefinition(NSSBaseModel):
    """Single transformation step with optional variables, column actions, and row actions."""

    vars: dict[str, str | dict | list] | None = Field(description="Variable names and templates.", default=None)

    columns: ColumnActions | None = Field(description="Columns transform configuration.", default=None)

    rows: RowActions | None = Field(description="Rows transform configurations.", default=None)


class GlinerConfig(NSSBaseModel):
    """Configuration for the GLiNER named-entity recognition model."""

    enable_gliner: bool = Field(description="Enable GLiNER NER module.", default=True)

    enable_batch_mode: bool = Field(description="Enable GLiNER batch mode.", default=True)

    batch_size: int = Field(description="GLiNER batch size.", default=8)

    chunk_length: int = Field(description="GLiNER batch chunk length in characters.", default=512)

    gliner_model: str = Field(
        description="GLiNER model name.",
        default="gretelai/gretel-gliner-bi-large-v1.0",
    )


class NERConfig(NSSBaseModel):
    """Configuration for Named Entity Recognition."""

    ner_threshold: float = Field(description="NER model threshold.", default=0.3)

    enable_regexps: bool = Field(
        description="Enable NER regular expressions (experimental).",
        default=False,
    )

    gliner: GlinerConfig = Field(
        description="GLiNER NER configuration.",
        default=GlinerConfig(),
    )

    ner_entities: OptionalStrList = Field(
        description="List of entity types to recognize. If unset, classification entity types are used.",
        default=None,
    )


class ClassifyConfig(NSSBaseModel):
    """Configuration for column classification using an LLM."""

    enable_classify: bool | None = Field(default=None, description="Enable column classification.")

    entities: OptionalStrList = Field(default=None, description="List of entity types to classify.")

    num_samples: int | None = Field(description="Number of column values to sample for classification.", default=3)

    classify_model_provider: str | None = Field(
        default=None,
        description="Name of the model provider in the Inference Gateway for column classification. "
        "The job compiler will resolve this to the appropriate endpoint URL.",
    )


class Globals(NSSBaseModel):
    """Global settings for the PII replacer including locales, seed, NER, and classification."""

    locales: list[str] | None = Field(description="List of locales.", examples=["en_US"], default=None)

    seed: int | None = Field(
        lt=MAX_32_BIT_INT,
        gt=-MAX_32_BIT_INT,
        description="Optional random seed.",
        default=None,
    )

    classify: Annotated[ClassifyConfig, Field(description="Column classification configuration.")] = ClassifyConfig()

    ner: Annotated[NERConfig, Field(description="Named Entity Recognition configuration.")] = NERConfig()

    lock_columns: OptionalStrList = Field(
        description="List of columns to preserve as immutable across all transformations.",
        default=None,
    )

    @field_validator("locales")
    @classmethod
    def _validate_locale(cls, locales: list[str] | None) -> list[str] | None:
        """Validate locale strings against Faker's supported locales."""
        if locales is None:
            return locales

        validated_locales = []
        for locale in locales:
            canonical_locale = locale.replace("-", "_")

            # AVAILABLE_LOCALES is in `en_US` format (with underscore).
            supported_locales = set(AVAILABLE_LOCALES)
            if canonical_locale not in supported_locales:
                raise ValueError(f"Invalid locale: {locale}!")

            validated_locales.append(canonical_locale)

        return validated_locales


class PiiReplacerConfig(Parameters):
    """Configuration for PII replacer.

    Defines how PII data should be detected and replaced in a dataset.
    """

    globals: Globals = Field(description="Global configuration options.", default_factory=Globals)

    steps: list[StepDefinition] = Field(
        min_length=1,
        max_length=10,
        description="List of transformation steps to perform on input data.",
    )

    @classmethod
    def get_default_config(cls) -> Self:
        """Return a default configuration loaded from the embedded YAML template."""
        return cls.from_yaml_str(DEFAULT_PII_TRANSFORM_CONFIG)


DEFAULT_PII_TRANSFORM_CONFIG = """
globals:
  classify:
    enable_classify: true
    entities:
      # True identifiers
      - first_name
      - last_name
      - name
      - street_address
      - city
      - state
      - postcode
      - country
      - address
      - latitude
      - longitude
      - coordinate
      - age
      - phone_number
      - fax_number
      - email
      - ssn
      - unique_identifier
      - medical_record_number
      - health_plan_beneficiary_number
      - account_number
      - certificate_license_number
      - vehicle_identifier
      - license_plate
      - device_identifier
      - biometric_identifier
      - url
      - ipv4
      - ipv6
      - national_id
      - tax_id
      - bank_routing_number
      - swift_bic
      - credit_debit_card
      - cvv
      - pin
      - employee_id
      - api_key
      - coordinate
      - customer_id
      - user_name
      - password
      - mac_address
      - http_cookie

      # Quasi identifiers
      - date
      - date_time
      - blood_type
      - gender
      - sexuality
      - political_view
      - race
      - ethnicity
      - religious_belief
      - language
      - education
      - job_title
      - employment_status
      - company_name
  ner:
    ner_threshold: 0.3
    ner_entities:
      # True identifiers
      - first_name
      - last_name
      - name
      - street_address
      - city
      - state
      - postcode
      - country
      - address
      - latitude
      - longitude
      - coordinate
      - age
      - phone_number
      - fax_number
      - email
      - ssn
      - unique_identifier
      - medical_record_number
      - health_plan_beneficiary_number
      - account_number
      - certificate_license_number
      - vehicle_identifier
      - license_plate
      - device_identifier
      - biometric_identifier
      - url
      - ipv4
      - ipv6
      - national_id
      - tax_id
      - bank_routing_number
      - swift_bic
      - credit_debit_card
      - pin
      - employee_id
      - api_key
      - coordinate
      - customer_id
      - user_name
      - password
      - mac_address
      - http_cookie
  locales: [en_US]
steps:
  - vars:
      row_seed: random.random()
    rows:
      update:
        - condition: column.entity == "first_name" and not (this | isna)
          value: fake.persona(row_index=vars.row_seed + index).first_name
        - condition: column.entity == "last_name" and not (this | isna)
          value: fake.persona(row_index=vars.row_seed + index).last_name
        - condition: column.entity == "name" and not (this | isna)
          value: column.entity | fake
        - condition: (column.entity == "street_address" or column.entity == "city" or column.entity == "state" or column.entity == "postcode" or column.entity == "address") and not (this | isna)
          value: column.entity | fake
        - condition: column.entity == "latitude" and not (this | isna)
          value: fake.location_on_land()[0]
        - condition: column.entity == "longitude" and not (this | isna)
          value: fake.location_on_land()[1]
        - condition: column.entity == "coordinate" and not (this | isna)
          value: fake.location_on_land()
        - condition: column.entity == "email" and not (this | isna)
          value: fake.persona(row_index=vars.row_seed + index).email
        - condition: column.entity == "ssn" and not (this | isna)
          value: column.entity | fake
        - condition: column.entity == "phone_number" and not (this | isna)
          value: (fake.random_number(digits=3) | string) + "-" + (fake.random_number(digits=3) | string) + "-" + (fake.random_number(digits=4) | string)
        - condition: column.entity == "fax_number" and not (this | isna)
          value: (fake.random_number(digits=3) | string) + "-" + (fake.random_number(digits=3) |
            string) + "-" + (fake.random_number(digits=4) | string)
        - condition: column.entity == "vehicle_identifier" and not (this | isna)
          value: fake.vin()
        - condition: column.entity == "license_plate" and not (this | isna)
          value: column.entity | fake
        - condition: (column.entity == "unique_identifier" or column.entity == "medical_record_number" or column.entity == "health_plan_beneficiary_number" or column.entity == "account_number" or column.entity == "certificate_license_number" or column.entity == "device_identifier" or column.entity == "biometric_identifier" or column.entity == "bank_routing_number" or column.entity == "swift_bic" or column.entity == "employee_id" or column.entity == "api_key" or column.entity == "customer_id" or column.entity == "user_name" or column.entity == "password" or column.entity == "http_cookie") and not (this | isna)
          value: fake.bothify(re.sub("\\\\d", "#", re.sub("[A-Z]", "?", (this | string))))
        - condition: (column.entity == "url" or column.entity == "ipv4" or column.entity == "ipv6") and not (this | isna)
          value: column.entity | fake
        - condition: (column.entity == "national_id" or column.entity == "tax_id") and not (this | isna)
          value: fake.itin()
        - condition: column.entity == "credit_debit_card" and not (this | isna)
          value: fake.credit_card_number()
        - condition: column.entity == "cvv" and not (this | isna)
          value: fake.credit_card_security_code()
        - condition: column.entity == "pin" and not (this | isna)
          value: fake.random_number(digits=4) | string
        - condition: column.entity == "coordinate" and not (this | isna)
          value: column.entity | fake
        - condition: column.entity == "mac_address" and not (this | isna)
          value: column.entity | fake

        - condition: column.entity is none and column.type == "text"
          value: this | fake_entities
"""
