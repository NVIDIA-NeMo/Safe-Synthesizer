# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned persistent specification for NSS tokenizers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from string import hexdigits
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..errors import ParameterError
from .types import JsonObject, JsonValue, TokenizerProbe, canonical_json_bytes

_IMMUTABLE_REMOTE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def validate_native_revision(source: str, revision: str) -> None:
    """Require resolved commit provenance for remote tokenizer sources."""
    if Path(source).exists():
        return
    if _IMMUTABLE_REMOTE_REVISION.fullmatch(revision) is None:
        raise ParameterError("Remote native tokenizer sources require a resolved immutable revision.")


class _StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ProbeInput(_StrictManifestModel):
    text: str
    input_ids: list[int]
    decoded: str


class _NativeInput(_StrictManifestModel):
    source: str
    revision: str
    trust_remote_code: bool
    class_: str = Field(alias="class")
    fingerprint: str
    probes: list[_ProbeInput]


class _EpochInput(_StrictManifestModel):
    prompt: int
    record: int
    delimiter: int
    padding: int
    cache: int


class _SpecInput(_StrictManifestModel):
    spec_version: int
    api_version: int
    workload_kind: str
    implementation_id: str
    implementation_version: str
    implementation_payload: dict[str, object]
    native: _NativeInput
    registry_digest: str
    policy_epochs: _EpochInput


@dataclass(frozen=True, slots=True)
class PolicyEpochs:
    """Semantic compatibility epochs, each starting at zero."""

    prompt: int = 0
    record: int = 0
    delimiter: int = 0
    padding: int = 0
    cache: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ParameterError(f"Policy epoch {name!r} must be non-negative.")


@dataclass(frozen=True, slots=True)
class NssTokenizerSpec:
    """Immutable schema for ``nss_tokenizer.json`` reconstruction."""

    spec_version: int
    api_version: int
    workload_kind: str
    implementation_id: str
    implementation_version: str
    implementation_payload: str
    native_source: str
    native_revision: str
    native_trust_remote_code: bool
    native_class: str
    native_fingerprint: str
    native_probes: tuple[TokenizerProbe, ...]
    registry_digest: str
    policy_epochs: PolicyEpochs

    def __post_init__(self) -> None:
        if not isinstance(self.spec_version, int) or isinstance(self.spec_version, bool) or self.spec_version != 1:
            raise ParameterError(f"Unsupported NSS tokenizer spec version: {self.spec_version}.")
        if not isinstance(self.api_version, int) or isinstance(self.api_version, bool) or self.api_version != 1:
            raise ParameterError(f"Unsupported NSS tokenizer API version: {self.api_version}.")
        if (
            not isinstance(self.workload_kind, str)
            or not isinstance(self.implementation_id, str)
            or not isinstance(self.implementation_version, str)
        ):
            raise ParameterError("Tokenizer workload and implementation identities must be strings.")
        if ":" not in self.implementation_id:
            raise ParameterError("Tokenizer implementation IDs must be namespaced.")
        if (
            not isinstance(self.native_source, str)
            or not isinstance(self.native_revision, str)
            or not self.native_source
            or not self.native_revision
        ):
            raise ParameterError("Native tokenizer source and immutable revision are required.")
        validate_native_revision(self.native_source, self.native_revision)
        if not isinstance(self.native_trust_remote_code, bool) or not isinstance(self.native_class, str):
            raise ParameterError("Native tokenizer trust and class provenance have invalid types.")
        if not all(
            isinstance(digest, str)
            and len(digest) == 64
            and digest == digest.lower()
            and all(character in hexdigits for character in digest)
            for digest in (self.native_fingerprint, self.registry_digest)
        ):
            raise ParameterError("Tokenizer fingerprints and registry digests must be SHA-256 hex digests.")
        if not isinstance(self.policy_epochs, PolicyEpochs) or not all(
            isinstance(probe, TokenizerProbe) for probe in self.native_probes
        ):
            raise ParameterError("Tokenizer probes and policy epochs have invalid types.")
        try:
            payload = json.loads(self.implementation_payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ParameterError("Tokenizer implementation payload is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ParameterError("Tokenizer implementation payload must be a JSON object.")
        canonical = canonical_json_bytes(cast(JsonValue, payload)).decode()
        if canonical != self.implementation_payload:
            raise ParameterError("Tokenizer implementation payload must use canonical JSON.")

    @property
    def key(self) -> tuple[int, str, str]:
        """Return the exact immutable registry key."""
        return self.api_version, self.implementation_id, self.implementation_version

    def to_dict(self) -> JsonObject:
        """Return the tested ``nss_tokenizer.json`` object schema."""
        return {
            "spec_version": self.spec_version,
            "api_version": self.api_version,
            "workload_kind": self.workload_kind,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "implementation_payload": cast(JsonValue, json.loads(self.implementation_payload)),
            "native": {
                "source": self.native_source,
                "revision": self.native_revision,
                "trust_remote_code": self.native_trust_remote_code,
                "class": self.native_class,
                "fingerprint": self.native_fingerprint,
                "probes": [
                    {"text": probe.text, "input_ids": list(probe.input_ids), "decoded": probe.decoded}
                    for probe in self.native_probes
                ],
            },
            "registry_digest": self.registry_digest,
            "policy_epochs": cast(JsonValue, asdict(self.policy_epochs)),
        }

    def canonical_bytes(self) -> bytes:
        """Return canonical UTF-8 manifest bytes."""
        return canonical_json_bytes(cast(JsonValue, self.to_dict()))

    @property
    def cache_identity_fragment(self) -> str:
        """SHA-256 over the canonical serialized specification."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> NssTokenizerSpec:
        """Validate and reconstruct a specification object."""
        try:
            parsed = _SpecInput.model_validate(value)
            native = parsed.native
            probes = tuple(
                TokenizerProbe(
                    text=probe.text,
                    input_ids=tuple(probe.input_ids),
                    decoded=probe.decoded,
                )
                for probe in native.probes
            )
            return cls(
                spec_version=parsed.spec_version,
                api_version=parsed.api_version,
                workload_kind=parsed.workload_kind,
                implementation_id=parsed.implementation_id,
                implementation_version=parsed.implementation_version,
                implementation_payload=canonical_json_bytes(cast(JsonValue, parsed.implementation_payload)).decode(),
                native_source=native.source,
                native_revision=native.revision,
                native_trust_remote_code=native.trust_remote_code,
                native_class=native.class_,
                native_fingerprint=native.fingerprint,
                native_probes=probes,
                registry_digest=parsed.registry_digest,
                policy_epochs=PolicyEpochs(**parsed.policy_epochs.model_dump()),
            )
        except (ParameterError, TypeError, ValueError, ValidationError) as exc:
            raise ParameterError("Invalid nss_tokenizer.json schema.") from exc

    @classmethod
    def from_json_bytes(cls, value: bytes) -> NssTokenizerSpec:
        """Reconstruct a specification from UTF-8 JSON."""
        try:
            decoded = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ParameterError("Invalid UTF-8 nss_tokenizer.json.") from exc
        return cls.from_dict(decoded)
