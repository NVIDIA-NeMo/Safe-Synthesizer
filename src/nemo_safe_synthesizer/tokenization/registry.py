# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable tokenizer factory registry and fail-closed reconstruction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..errors import ParameterError
from ..package_info import __version__
from .base import NssTokenizer
from .spec import NssTokenizerSpec, PolicyEpochs, validate_native_revision
from .tabular import TabularNssTokenizer
from .timeseries import TimeSeriesNssTokenizer
from .types import FramingPolicy, JsonValue, canonical_json_bytes

RegistryKey = tuple[int, str, str]


class NativeTokenizerLoader(Protocol):
    """Loader seam used by worker reconstruction."""

    def __call__(
        self,
        source: str,
        revision: str,
        trust_remote_code: bool,
        /,
    ) -> PreTrainedTokenizerBase: ...


TokenizerFactory = type[NssTokenizer]


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One installed implementation and its provider identity."""

    key: RegistryKey
    distribution_name: str
    distribution_version: str
    factory: TokenizerFactory

    def __post_init__(self) -> None:
        if not isinstance(self.factory, type) or not issubclass(self.factory, NssTokenizer):
            raise ParameterError("Tokenizer registry factories must inherit NssTokenizer.")
        if ":" not in self.key[1]:
            raise ParameterError("Tokenizer implementation IDs must be namespaced.")
        if not self.distribution_name or not self.distribution_version:
            raise ParameterError("Tokenizer registry providers require distribution name and installed version.")
        factory_key = (
            self.factory.API_VERSION,
            self.factory.IMPLEMENTATION_ID,
            self.factory.IMPLEMENTATION_VERSION,
        )
        if self.key != factory_key:
            raise ParameterError(f"Tokenizer registry key {self.key!r} does not match factory key {factory_key!r}.")

    @property
    def normalized_distribution_name(self) -> str:
        """Return the PEP 503-style normalized distribution identity."""
        return _normalize_distribution_name(self.distribution_name)


@dataclass(frozen=True, slots=True)
class NssTokenizerRegistry:
    """Persistent registry snapshot; registration returns a new instance."""

    entries: tuple[RegistryEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not all(isinstance(entry, RegistryEntry) for entry in self.entries):
            raise ParameterError("Tokenizer registry entries must be an immutable tuple.")
        keys = tuple(entry.key for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ParameterError("Duplicate exact tokenizer registry keys are not allowed.")
        if keys != tuple(sorted(keys)):
            raise ParameterError("Tokenizer registry entries must be sorted by their full key.")

    def register(self, entry: RegistryEntry, *, admit_external: bool = False) -> NssTokenizerRegistry:
        """Return a registry containing an explicitly admitted unique entry."""
        if not admit_external:
            raise ParameterError("External tokenizer implementations require explicit admission.")
        return self._append(entry)

    def _append(self, entry: RegistryEntry) -> NssTokenizerRegistry:
        if any(installed.key == entry.key for installed in self.entries):
            raise ParameterError(f"Duplicate tokenizer registry key: {entry.key!r}.")
        entries = tuple(sorted((*self.entries, entry), key=lambda item: item.key))
        return NssTokenizerRegistry(entries)

    @property
    def digest(self) -> str:
        """SHA-256 over sorted keys and normalized provider identities."""
        snapshot: JsonValue = [
            {
                "api_version": entry.key[0],
                "implementation_id": entry.key[1],
                "implementation_version": entry.key[2],
                "distribution_name": entry.normalized_distribution_name,
                "distribution_version": entry.distribution_version,
            }
            for entry in self.entries
        ]
        return hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()

    def resolve(self, key: RegistryKey) -> RegistryEntry:
        """Resolve an exact installed key or fail closed."""
        for entry in self.entries:
            if entry.key == key:
                return entry
        versions = sorted(entry.key for entry in self.entries if entry.key[1] == key[1])
        if versions:
            raise ParameterError(
                f"Unsupported tokenizer API or implementation version {key!r}; installed: {versions!r}."
            )
        raise ParameterError(f"Unknown tokenizer implementation ID: {key[1]!r}.")

    def create(
        self,
        key: RegistryKey,
        native: PreTrainedTokenizerBase,
        *,
        framing: FramingPolicy,
        native_source: str,
        native_revision: str,
        native_trust_remote_code: bool = False,
        policy_epochs: PolicyEpochs | None = None,
        workload_payload: object | None = None,
    ) -> NssTokenizer:
        """Construct a registered tokenizer with this immutable snapshot."""
        entry = self.resolve(key)
        return entry.factory(
            native,
            framing=framing,
            native_source=native_source,
            native_revision=native_revision,
            native_trust_remote_code=native_trust_remote_code,
            registry_digest=self.digest,
            policy_epochs=policy_epochs,
            workload_payload=workload_payload,
        )

    def reconstruct(
        self,
        spec: NssTokenizerSpec,
        *,
        native_loader: NativeTokenizerLoader | None = None,
        admit_remote_code: bool = False,
    ) -> NssTokenizer:
        """Load and verify a tokenizer from a persisted manifest."""
        validate_native_revision(spec.native_source, spec.native_revision)
        if spec.registry_digest != self.digest:
            raise ParameterError("Tokenizer registry drift detected; the persisted registry digest does not match.")
        if spec.native_trust_remote_code and not admit_remote_code:
            raise ParameterError("Remote tokenizer code requires explicit reconstruction-time admission.")
        entry = self.resolve(spec.key)
        if spec.policy_epochs != entry.factory.POLICY_EPOCHS:
            raise ParameterError("Unsupported tokenizer policy epoch for the installed implementation.")
        if spec.workload_kind != entry.factory.WORKLOAD_KIND:
            raise ParameterError("Persisted workload kind does not match the tokenizer implementation.")
        framing, workload_payload = entry.factory.policies_from_spec(spec)
        loader = native_loader or _load_native
        native = loader(spec.native_source, spec.native_revision, spec.native_trust_remote_code)
        return entry.factory(
            native,
            framing=framing,
            native_source=spec.native_source,
            native_revision=spec.native_revision,
            native_trust_remote_code=spec.native_trust_remote_code,
            registry_digest=self.digest,
            policy_epochs=spec.policy_epochs,
            persisted_spec=spec,
            workload_payload=workload_payload,
        )


def _load_native(source: str, revision: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ParameterError("Native tokenizer loader returned an unsupported object.")
    return tokenizer


def builtin_registry() -> NssTokenizerRegistry:
    """Return the deterministic v1 registry containing only built-ins."""
    provider = "nemo-safe-synthesizer"
    registry = NssTokenizerRegistry()
    for implementation in (TabularNssTokenizer, TimeSeriesNssTokenizer):
        registry = registry._append(
            RegistryEntry(
                key=(
                    implementation.API_VERSION,
                    implementation.IMPLEMENTATION_ID,
                    implementation.IMPLEMENTATION_VERSION,
                ),
                distribution_name=provider,
                distribution_version=__version__,
                factory=implementation,
            )
        )
    return registry
