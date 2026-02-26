# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hmac
from abc import ABC, abstractmethod
from hashlib import sha256


class NameAnonymizer(ABC):
    """Used to anonymize field names from customer's datasets."""

    @abstractmethod
    def anonymize(self, name: str) -> str | None: ...

    def anonymize_list(self, names: list[str]) -> list[str]:
        """Anonymizes all of the values in the list, removing ``None``."""
        result = []
        for name in names:
            value = self.anonymize(name)
            if value is not None:
                result.append(value)
        return result


class NoopNameAnonymizer(NameAnonymizer):
    """Returns the name as-is, with no anonymization."""

    def anonymize(self, name: str) -> str | None:
        return name


def _encode_utf8(name):
    return name.encode("utf-8", "ignore")


class HashNameAnonymizer(NameAnonymizer):
    """Returns substring of a SHA256 hash."""

    def __init__(self, length: int = 16):
        self._length = length

    def anonymize(self, name: str) -> str | None:
        return sha256(_encode_utf8(name)).hexdigest()[0 : self._length]


class HMACNameAnonymizer(NameAnonymizer):
    """Uses HMAC with provided secret to create"""

    def __init__(self, secret: bytes, length: int = 16):
        self._secret = secret
        self._length = length

    def anonymize(self, name: str) -> str | None:
        return hmac.digest(self._secret, _encode_utf8(name), "sha256").hex()[0 : self._length]
