# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared cross-table replacement store for database-scope runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...config.replace_pii import KeyDomain

# Core person attributes tracked on an identified person bundle.
PERSON_ATTRIBUTE_KEYS: tuple[str, ...] = (
    "first_name",
    "middle_name",
    "last_name",
    "full_name",
    "street_address",
    "phone_number",
    "email",
    "date_of_birth",
    "ssn",
    "national_id",
    "credit_debit_card",
)


@dataclass
class AttributePair:
    """Original and synthetic values for one person attribute."""

    original: str | None = None
    synthetic: str | None = None

    def to_dict(self) -> dict[str, str | None] | None:
        if self.original is None and self.synthetic is None:
            return None
        return {"original": self.original, "synthetic": self.synthetic}

    @classmethod
    def from_dict(cls, raw: object) -> AttributePair | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            return None
        return cls(
            original=None if raw.get("original") is None else str(raw["original"]),
            synthetic=None if raw.get("synthetic") is None else str(raw["synthetic"]),
        )


@dataclass
class PersonBundle:
    """One identified person keyed by a person-reference domain value."""

    domain: str
    original_key: str
    synthetic_key: str | None = None
    match: dict[str, str | None] = field(default_factory=dict)
    attributes: dict[str, AttributePair | None] = field(default_factory=dict)
    free_text_pairs: list[tuple[str, str]] = field(default_factory=list)

    def ensure_attribute_slots(self) -> None:
        for key in PERSON_ATTRIBUTE_KEYS:
            self.attributes.setdefault(key, None)

    def set_attribute(self, label: str, original: str | None, synthetic: str | None) -> None:
        if label not in PERSON_ATTRIBUTE_KEYS:
            return
        self.attributes[label] = AttributePair(original=original, synthetic=synthetic)

    def merge_free_text_pairs(self, pairs: list[tuple[str, str]]) -> None:
        existing = set(self.free_text_pairs)
        for pair in pairs:
            if pair not in existing:
                self.free_text_pairs.append(pair)
                existing.add(pair)


@dataclass
class DomainState:
    """Runtime maps for one key domain."""

    domain_id: str
    person_reference: bool = False
    columns: list[str] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)


@dataclass
class PolymorphicColumnRoute:
    """Per-row router for one polymorphic Id column on the current table."""

    bare_column: str
    type_column: str  # bare
    # type_value -> domain_id
    targets: dict[str, str] = field(default_factory=dict)


@dataclass
class TableRunContext:
    """Per-table context passed into the single-table engine with a shared store.

    Column names here are **bare** (DataFrame columns). Qualified names are
    reconstructed as ``f"{table_name}.{col}"``.
    """

    table_name: str
    # bare column -> domain id for columns that participate in a key domain
    column_domains: dict[str, str] = field(default_factory=dict)
    # bare FK/PK columns that reference a person_reference domain -> that domain id
    person_ref_columns: dict[str, str] = field(default_factory=dict)
    # persona label -> (person_key_domain_id, bare key column on this table)
    # One table may bind several personas to different domains (patient vs provider).
    persona_key_bindings: dict[str, tuple[str, str]] = field(default_factory=dict)
    # polymorphic Id routers (bare column names)
    polymorphic_routes: dict[str, PolymorphicColumnRoute] = field(default_factory=dict)


class SharedRuntimeStore:
    """In-memory shared mapping object for a database-scope run.

    Holds person bundles, per-domain original→synthetic maps, and per-domain
    ``used`` sets. Persist via ``map_io``.
    """

    def __init__(
        self,
        *,
        seed: int,
        locale: str,
        key_domains: list[KeyDomain] | None = None,
    ) -> None:
        self.seed = seed
        self.locale = locale
        self.domains: dict[str, DomainState] = {}
        self.column_to_domain: dict[str, str] = {}
        self.persons: dict[tuple[str, str], PersonBundle] = {}
        if key_domains:
            for kd in key_domains:
                self.register_key_domain(kd)

    def register_key_domain(self, kd: KeyDomain) -> DomainState:
        state = self.domains.get(kd.id)
        if state is None:
            state = DomainState(
                domain_id=kd.id,
                person_reference=kd.person_reference,
                columns=list(kd.columns),
            )
            self.domains[kd.id] = state
        else:
            state.person_reference = state.person_reference or kd.person_reference
            for col in kd.columns:
                if col not in state.columns:
                    state.columns.append(col)
        for col in kd.columns:
            self.column_to_domain[col] = kd.id
        return state

    def domain_for_qualified(self, qualified: str) -> DomainState | None:
        domain_id = self.column_to_domain.get(qualified)
        if domain_id is None:
            return None
        return self.domains.get(domain_id)

    def domain_for_bare(self, table_name: str, bare: str) -> DomainState | None:
        return self.domain_for_qualified(f"{table_name}.{bare}")

    def get_or_create_person(self, domain_id: str, original_key: str) -> PersonBundle:
        key = (domain_id, original_key)
        person = self.persons.get(key)
        if person is None:
            person = PersonBundle(domain=domain_id, original_key=original_key)
            person.ensure_attribute_slots()
            self.persons[key] = person
        return person

    def lookup_person(self, domain_id: str, original_key: str) -> PersonBundle | None:
        return self.persons.get((domain_id, original_key))

    def record_domain_mapping(self, domain_id: str, original: str, synthetic: str) -> None:
        state = self.domains.setdefault(domain_id, DomainState(domain_id=domain_id))
        state.values[original] = synthetic
        state.used.add(original)
        state.used.add(synthetic)

    def merge_domain_map(self, domain_id: str, mapping: dict[str, str]) -> None:
        for original, synthetic in mapping.items():
            self.record_domain_mapping(domain_id, original, synthetic)

    def ingest_persona_instance(
        self,
        *,
        domain_id: str,
        original_key: str,
        synthetic_key: str | None,
        originals_by_label: dict[str, str],
        synthetic_by_label: dict[str, str],
        free_text_pairs: list[tuple[str, str]],
        sex: str | None = None,
        race_raw: str | None = None,
    ) -> PersonBundle:
        person = self.get_or_create_person(domain_id, original_key)
        if synthetic_key:
            person.synthetic_key = synthetic_key
        if sex is not None:
            person.match["sex"] = sex
        if race_raw is not None:
            person.match["ethnic_background"] = race_raw
        for label, original in originals_by_label.items():
            if label not in PERSON_ATTRIBUTE_KEYS:
                continue
            synthetic = synthetic_by_label.get(label)
            existing = person.attributes.get(label)
            if existing is None or existing.original is None:
                person.set_attribute(label, original, synthetic)
            elif synthetic and existing.synthetic is None:
                existing.synthetic = synthetic
        person.merge_free_text_pairs(free_text_pairs)
        if synthetic_key:
            self.record_domain_mapping(domain_id, original_key, synthetic_key)
        return person

    def free_text_pairs_for_key(self, domain_id: str, original_key: str) -> list[tuple[str, str]]:
        person = self.lookup_person(domain_id, original_key)
        if person is None:
            return []
        return list(person.free_text_pairs)

    def to_persist_dict(self) -> dict[str, Any]:
        persons_out: list[dict[str, Any]] = []
        for person in self.persons.values():
            person.ensure_attribute_slots()
            attrs: dict[str, Any] = {}
            for key in PERSON_ATTRIBUTE_KEYS:
                pair = person.attributes.get(key)
                attrs[key] = None if pair is None else pair.to_dict()
            entry: dict[str, Any] = {
                "key": {"domain": person.domain, "original": person.original_key},
                "match": dict(person.match),
                "attributes": attrs,
                "free_text_pairs": [[a, b] for a, b in person.free_text_pairs],
            }
            if person.synthetic_key is not None:
                entry["synthetic_key"] = person.synthetic_key
            persons_out.append(entry)

        domains_out: dict[str, Any] = {}
        for domain_id, state in self.domains.items():
            domains_out[domain_id] = {
                "person_reference": state.person_reference,
                "columns": list(state.columns),
                "values": dict(state.values),
            }

        return {
            "version": 1,
            "scope": "database",
            "seed": self.seed,
            "locale": self.locale,
            "persons": persons_out,
            "key_domains": domains_out,
        }

    @classmethod
    def from_persist_dict(cls, raw: dict[str, Any]) -> SharedRuntimeStore:
        store = cls(seed=int(raw.get("seed", 42)), locale=str(raw.get("locale", "en_US")))
        for domain_id, payload in (raw.get("key_domains") or {}).items():
            state = DomainState(
                domain_id=str(domain_id),
                person_reference=bool(payload.get("person_reference", False)),
                columns=[str(c) for c in (payload.get("columns") or [])],
                values={str(k): str(v) for k, v in (payload.get("values") or {}).items()},
            )
            state.used = set(state.values.keys()) | set(state.values.values())
            store.domains[state.domain_id] = state
            for col in state.columns:
                store.column_to_domain[col] = state.domain_id
        for entry in raw.get("persons") or []:
            key = entry.get("key") or {}
            domain = str(key.get("domain"))
            original = str(key.get("original"))
            person = PersonBundle(
                domain=domain,
                original_key=original,
                synthetic_key=None if entry.get("synthetic_key") is None else str(entry["synthetic_key"]),
                match={str(k): (None if v is None else str(v)) for k, v in (entry.get("match") or {}).items()},
            )
            person.ensure_attribute_slots()
            for label, pair_raw in (entry.get("attributes") or {}).items():
                person.attributes[str(label)] = AttributePair.from_dict(pair_raw)
            pairs_raw = entry.get("free_text_pairs") or []
            person.free_text_pairs = [(str(a), str(b)) for a, b in pairs_raw]
            store.persons[(domain, original)] = person
        return store
