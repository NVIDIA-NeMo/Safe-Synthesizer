# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Abstract base class for parameter collections with serialization helpers.

``Parameters`` is the common superclass for every configuration group in
``config/`` (``DataParameters``, ``TrainingHyperparams``, ``GenerateParameters``,
etc.).  It extends ``pydantic.BaseModel`` with:

- Recursive iteration over nested parameter groups.
- Name-based lookup across the full parameter tree (``get()``, ``has()``).
- YAML / JSON round-trip serialization (``from_yaml()``, ``to_yaml()``,
  ``from_json()``).
"""

from __future__ import annotations

import json
import typing
from abc import ABCMeta
from pathlib import Path
from typing import Any, Generator, Iterable, Mapping, get_args

import yaml
from pydantic import (
    BaseModel,
)
from typing_extensions import Self

from ..config.base import (
    pydantic_model_config,
)
from ..observability import get_logger
from .parameter import (
    DataT,
)

__all__ = ["Parameters"]

logger = get_logger(__name__)
PathT = str | Path


class Parameters(BaseModel, metaclass=ABCMeta):
    """Abstract base for parameter collections used throughout the config layer.

    Subclasses define typed fields (often ``Parameter[T]`` or ``AutoParam[T]``)
    and inherit recursive iteration, name-based lookup, and YAML / JSON
    serialization from this class.
    """

    model_config = pydantic_model_config

    def _isparams(self):
        """Marker method used by ``__subclasshook__`` to identify ``Parameters`` subclasses."""
        return True

    @classmethod
    def __subclasshook__(cls, c):
        """Enable ``isinstance()`` checks via duck typing on the ``_isparams`` marker."""
        if cls is Parameters:
            mro = c.__mro__
            for B in mro:
                if "_isparams" in B.__dict__:
                    if B.__dict__["_isparams"] is None:
                        return NotImplemented
                    break
            else:
                return NotImplemented
            return True
        return NotImplemented

    def _iter_subparamgroups(self) -> "Generator[Self, None, None]":
        """Yield nested ``Parameters`` instances that are direct attributes of this collection."""
        only_params_ = [(x, info) for x, info in self.__class__.model_fields.items()]

        for field in only_params_:
            name = field[0]
            info = field[1]
            anno = getattr(info, "annotation", None)
            anno_type = type(anno)
            if getattr(anno_type, "__origin__", None) is typing.Union:
                args = get_args(anno)
                for arg in args:
                    for subtype in get_args(arg):
                        if isinstance(subtype, type) and issubclass(subtype, Parameters):
                            yield getattr(self, name)

            elif isinstance(anno, type) and issubclass(anno, Parameters):
                yield getattr(self, name)

            elif isinstance(getattr(self, name), Parameters):
                yield getattr(self, name)
            else:
                pass

    def _iter_parameters(self, recursive: bool = True) -> Generator[Mapping[str, DataT | Parameters], None, None]:
        """Yield ``{name: value}`` dicts for every parameter in this collection.

        Args:
            recursive: If ``True``, also descend into nested ``Parameters`` groups.

        Yields:
            Single-key dicts mapping field name to serialized value.
        """
        parameters = [{k: v} for k, v in self.model_dump().items()]
        param_groups = self._iter_subparamgroups()
        yield from parameters
        if recursive:
            for pg in param_groups:
                yield from pg._iter_parameters(recursive=True)

    def __iter__(self) -> Iterable[DataT]:
        """Iterate over all parameters, recursing into nested groups."""
        return self._iter_parameters(recursive=True)

    def get(self, name: str, default: Any = None) -> DataT | Any | None:
        """Look up a parameter or sub-group by name across the full tree.

        Checks direct attributes first, then walks nested groups recursively.

        Args:
            name: Field name to search for.
            default: Value returned when ``name`` is not found.

        Returns:
            The parameter value or sub-group if found, otherwise ``default``.
        """
        if (group := getattr(self, name, None)) is not None:
            return group
        for param in self._iter_parameters(recursive=True):
            if name in param:
                return param.get(name)
        return default

    def has(self, name: str) -> bool:
        """Check whether ``name`` exists anywhere in the parameter tree.

        Unlike ``get()``, this does not conflate falsy values (``0``, ``""``,
        ``False``, ``None``) with absence.

        Args:
            name: Field name to search for.

        Returns:
            ``True`` if the parameter or sub-group exists.
        """
        if getattr(self, name, None) is not None:
            return True
        for param in self._iter_parameters(recursive=True):
            if name in param:
                return True
        return False

    @classmethod
    def from_yaml_str(cls, raw: str) -> Self:
        """Construct an instance from a YAML-formatted string.

        Args:
            raw: YAML content as a string.

        Returns:
            A validated ``Parameters`` instance.
        """
        data = yaml.safe_load(raw)
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: PathT, overrides: dict | None = None) -> Self:
        """Load from a JSON file, optionally applying field overrides.

        Args:
            path: Path to the JSON file.
            overrides: Field-level overrides applied via ``model_copy(update=...)``.

        Returns:
            A validated ``Parameters`` instance.
        """
        with open(path, "r") as f:
            data = json.load(f)
        params = cls.model_validate(data)
        if overrides:
            params = params.model_copy(update=overrides)
        return params

    @classmethod
    def from_yaml(cls, path: PathT, overrides: dict | None = None) -> Self:
        """Load from a YAML file, optionally applying field overrides.

        Args:
            path: Path to the YAML file.
            overrides: Field-level overrides applied via ``model_copy(update=...)``.

        Returns:
            A validated ``Parameters`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        pth = Path(path)
        if not pth.exists():
            raise FileNotFoundError(f"File {pth} does not exist")
        with pth.open("r") as f:
            data = yaml.safe_load(f)
        params = cls.model_validate(data)
        if overrides:
            params = params.model_copy(update=overrides)
        return params

    @classmethod
    def from_yaml_or_overrides(cls, path: PathT | None = None, overrides: dict | None = None) -> Self:
        """Load from YAML when ``path`` is provided, otherwise build from ``overrides`` alone.

        Args:
            path: Optional path to a YAML file.
            overrides: Keyword overrides forwarded to ``from_yaml`` or ``from_params``.

        Returns:
            A validated ``Parameters`` instance.
        """
        if path:
            return cls.from_yaml(path, overrides)
        else:
            return cls.from_params(**overrides)

    def to_yaml(self, path: PathT, exclude_unset: bool = True) -> None:
        """Serialize this instance to a YAML file.

        Args:
            path: Destination file path.
            exclude_unset: If ``True``, omit fields that were never explicitly set.
        """
        with open(path, "w") as f:
            j = json.loads(self.model_dump_json(exclude_unset=exclude_unset))
            yaml.safe_dump(j, f)

    @classmethod
    def from_params(cls, **kwargs) -> Self:
        """Construct a ``Parameters`` instance from keyword arguments.

        Args:
            **kwargs: Parameter values passed to ``model_validate``.

        Returns:
            A validated ``Parameters`` instance.
        """
        return cls.model_validate(kwargs)

    def get_auto_params(self) -> Iterable[Any]:
        """Yield parameters whose current value is the ``"auto"`` sentinel."""
        for param in self:
            if param == "auto":
                yield param
