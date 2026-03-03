# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact directory structure for Safe Synthesizer.

Defines the on-disk layout produced by each pipeline run using a declarative
descriptor pattern. ``FileNode`` and ``DirNode`` descriptors declare the
tree shape on ``Workdir``; at runtime they resolve to ``Path`` and ``BoundDir``
objects respectively, giving typed access to every artifact path without
hard-coding strings throughout the CLI.

Typical directory tree::

    $base_path/<config>---<dataset>/<run_name>/
    - safe-synthesizer-config.json
    - train/  ...
    - generate/  ...
    - dataset/  ...

See ``Workdir`` for the full structure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Self, overload

from ..observability import get_logger
from ..utils import write_json

logger = get_logger(__name__)

PathT = str | Path

RUN_NAME_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
"""Format string for auto-generated timestamp-based run names."""


def _try_parse_timestamp(value: str) -> datetime | None:
    """Try to parse a timestamp string, returning None on failure.

    Args:
        value: String to parse as a timestamp

    Returns:
        Parsed datetime or None if parsing failed
    """
    try:
        return datetime.strptime(value, RUN_NAME_DATE_FORMAT)
    except ValueError:
        return None


PROJECT_NAME_DELIMITER = "---"
"""Delimiter used to separate config_name and dataset_name in project names.

Uses triple-dash to avoid ambiguity with single dashes that commonly appear
in config and dataset filenames (e.g., my-config.yaml, training-data.csv).
"""


def _parse_project_name(project_name: str) -> tuple[str, str]:
    """Parse a project name into config_name and dataset_name.

    Uses pattern matching to split on the project name delimiter (---).

    Args:
        project_name: Project name string (e.g., "my-config---my-dataset")

    Returns:
        Tuple of (config_name, dataset_name)
    """
    match project_name.split(PROJECT_NAME_DELIMITER, 1):
        case [config_name, dataset_name]:
            return config_name, dataset_name
        case [config_name]:
            return config_name, "unknown"
        case _:
            return project_name, "unknown"


@dataclass
class RunName:
    """Run name for artifact directories.

    Supports two modes:
    - Auto-generated: Creates a timestamp-based name (default when no value provided)
    - Explicit: Stores an arbitrary string name (from --run-path)

    Examples:
        - Auto-generated: "2026-01-15T12:00:00"
        - Explicit: "unsloth_adult_0", "my-experiment-run"
    """

    _value: str = field(default="")
    _timestamp: datetime | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize the run name, auto-generating timestamp if no value provided."""
        if not self._value:
            self._timestamp = datetime.now()
            self._value = self._timestamp.strftime(RUN_NAME_DATE_FORMAT)

    def to_string(self) -> str:
        """Convert the run name to a string for use in directory names."""
        return self._value

    @classmethod
    def from_string(cls, name: str) -> Self:
        """Parse a run name string into a RunName object.

        Accepts any valid string. If the string matches the timestamp format,
        the timestamp is also stored for potential use.

        Args:
            name: Run name string (e.g., "2026-01-15T12:00:00" or "unsloth_adult_0")

        Returns:
            RunName with the provided name and optional parsed timestamp
        """
        ts = _try_parse_timestamp(name)
        return cls(_value=name, _timestamp=ts)

    @property
    def is_timestamp_based(self) -> bool:
        """Whether this run name was generated from or parsed as a timestamp."""
        return self._timestamp is not None

    @property
    def timestamp(self) -> datetime | None:
        """Parsed timestamp, or None for non-timestamp-based run names."""
        return self._timestamp


class FileNode:
    """Descriptor for file paths within a directory structure.

    When accessed on a class, returns the descriptor itself.
    When accessed on an instance, returns the full Path to the file.
    """

    def __init__(self, name: str):
        """Initialize a FileNode descriptor.

        Args:
            name: The filename (e.g., "config.json")
        """
        self.name = name
        self._attr_name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> FileNode: ...

    @overload
    def __get__(self, obj: BoundDir | Workdir, objtype: type | None = None) -> Path: ...

    def __get__(self, obj: object | None, objtype: type | None = None) -> FileNode | Path:
        """Resolve to the descriptor itself (class access) or a full ``Path`` (instance access)."""
        match obj:
            case None:
                return self
            case BoundDir(_path=parent_path):
                return parent_path / self.name
            case Workdir() as workdir:
                return workdir.run_dir / self.name
            case _:
                raise TypeError(f"FileNode can only be used with BoundDir or Workdir, got {type(obj)}")


class DirNode:
    """Descriptor for directory paths within a directory structure.

    Supports nested children (both FileNode and DirNode).
    When accessed on a class, returns the descriptor itself.
    When accessed on an instance, returns a BoundDir with the resolved path.
    """

    def __init__(self, name: str, **children: FileNode | DirNode):
        """Initialize a DirNode descriptor.

        Args:
            name: The directory name (e.g., "train")
            **children: Child nodes (FileNode or DirNode instances)
        """
        self.name = name
        self.children: dict[str, FileNode | DirNode] = children
        self._attr_name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = name

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> DirNode: ...

    @overload
    def __get__(self, obj: BoundDir | Workdir, objtype: type | None = None) -> BoundDir: ...

    def __get__(self, obj: object | None, objtype: type | None = None) -> DirNode | BoundDir:
        """Resolve to the descriptor itself (class access) or a ``BoundDir`` (instance access)."""
        match obj:
            case None:
                return self
            case BoundDir(_path=parent_path):
                return BoundDir(parent_path / self.name, self.children)
            case Workdir() as workdir:
                return BoundDir(workdir.run_dir / self.name, self.children)
            case _:
                raise TypeError(f"DirNode can only be used with BoundDir or Workdir, got {type(obj)}")


class BoundDir(os.PathLike[str]):
    """Runtime class representing a bound directory path.

    Provides access to child FileNode and DirNode descriptors as attributes.
    Implements os.PathLike[str] so instances can be used wherever paths are expected.
    """

    def __init__(self, path: Path, children: dict[str, FileNode | DirNode]):
        """Initialize a BoundDir.

        Args:
            path: The resolved directory path
            children: Child nodes from the DirNode
        """
        self._path = path
        self._children = children

    @property
    def path(self) -> Path:
        """The resolved directory path."""
        return self._path

    def __fspath__(self) -> str:
        """Support os.fspath() for use with open(), etc."""
        return str(self._path)

    def __str__(self) -> str:
        """Return the path as a string."""
        return str(self._path)

    def __repr__(self) -> str:
        return f"BoundDir({self._path!r})"

    def __eq__(self, other: object) -> bool:
        """Support comparison with Path objects."""
        if isinstance(other, BoundDir):
            return self._path == other._path
        if isinstance(other, Path):
            return self._path == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._path)

    def __getattr__(self, name: str) -> Path | BoundDir:
        """Resolve child ``FileNode`` to ``Path`` or child ``DirNode`` to ``BoundDir``."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if name not in self._children:
            raise AttributeError(f"No child '{name}' in {self._path}")

        # Dispatch on child type using pattern matching
        match self._children[name]:
            case FileNode(name=filename):
                return self._path / filename
            case DirNode(name=dirname, children=children):
                return BoundDir(self._path / dirname, children)
            case other:
                raise TypeError(f"Unknown child type: {type(other)}")


@dataclass
class Workdir:
    """Working directory structure for Safe Synthesizer artifacts.

    This class defines the complete directory layout and provides typed access
    to all paths within the structure. It uses FileNode and DirNode descriptors
    for declarative path definitions.

    Structure:
        $base_path/<config>---<dataset>/<run_name>/
        - safe-synthesizer-config.json
        - train/
          - safe-synthesizer-config.json
          - adapter/
            - adapter_config.json
            - metadata_v2.json
            - dataset_schema.json
        - generate/
          - safe-synthesizer-config.json
          - logs.jsonl
          - synthetic_data.csv
          - evaluation_report.html
        - dataset/
          - training.csv
          - test.csv
          - validation.csv

    Args:
        base_path: The base path for the workdir
        config_name: The name of the config
        dataset_name: The name of the dataset
        run_name: The run name (auto-generated timestamp or explicit name from CLI)
        _current_phase: The current phase of the workdir
        _parent_workdir: The parent workdir
    """

    base_path: Path
    config_name: str
    dataset_name: str
    run_name: str | None = None
    _run_name_obj: RunName = field(default_factory=RunName, repr=False)
    _current_phase: str = field(default="unknown", repr=False)
    _parent_workdir: Workdir | None = field(default=None, repr=False)
    _explicit_run_path: Path | None = field(default=None, repr=False)

    # Root-level config file
    config = FileNode("safe-synthesizer-config.json")

    # WandB run ID file
    wandb_run_id_file = FileNode("wandb_run_id.txt")

    # Train directory structure
    train = DirNode(
        "train",
        config=FileNode("safe-synthesizer-config.json"),
        cache=DirNode(
            "cache",
        ),
        adapter=DirNode(
            "adapter",
            adapter_config=FileNode("adapter_config.json"),
            metadata=FileNode("metadata_v2.json"),
            schema=FileNode("dataset_schema.json"),
        ),
    )

    # Generate directory structure
    generate = DirNode(
        "generate",
        config=FileNode("safe-synthesizer-config.json"),
        logs=FileNode("logs.jsonl"),
        output=FileNode("synthetic_data.csv"),
        report=FileNode("evaluation_report.html"),
        info=FileNode("info.json"),
    )

    # Dataset directory structure
    dataset = DirNode(
        "dataset",
        training=FileNode("training.csv"),
        test=FileNode("test.csv"),
        validation=FileNode("validation.csv"),
    )

    def __post_init__(self) -> None:
        """Initialize the workdir after dataclass fields are set."""
        # Convert string base_path to Path
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)

        # Generate run_name if not provided, otherwise parse it
        # RunName.from_string() accepts both timestamp and arbitrary names
        if self.run_name is None:
            self._run_name_obj = RunName()  # Auto-generates timestamp
            self.run_name = self._run_name_obj.to_string()
        else:
            self._run_name_obj = RunName.from_string(self.run_name)

    @property
    def project_name(self) -> str:
        """Project name in ``<config>---<dataset>`` format."""
        return f"{self.config_name}{PROJECT_NAME_DELIMITER}{self.dataset_name}"

    @property
    def project_dir(self) -> Path:
        """Project directory path (``$base_path/<config>---<dataset>/``).

        Falls back to the parent of ``_explicit_run_path`` when one was provided.
        """
        if self._explicit_run_path is not None:
            return self._explicit_run_path.parent
        return self.base_path / self.project_name

    @property
    def run_dir(self) -> Path:
        """Run directory path (``$base_path/<config>---<dataset>/<run_name>/``).

        Uses ``_explicit_run_path`` directly when one is provided.
        """
        if self._explicit_run_path is not None:
            return self._explicit_run_path
        return self.project_dir / self.run_name

    def phase_dir(self, phase: str | None = None) -> Path:
        """Get the phase directory path.

        Args:
            phase: Phase name (train, generate, etc.). Defaults to _current_phase.

        Returns:
            Path to the phase directory
        """
        phase = phase or self._current_phase
        return self.run_dir / phase

    @property
    def log_file(self) -> Path:
        """Log file path for the current phase."""
        phase = self._current_phase or "unknown"
        if phase == "generate":
            return self.generate.logs  # type: ignore[return-value]
        return self.run_dir / "logs" / f"{phase}.jsonl"

    @property
    def adapter_path(self) -> Path:
        """Shortcut to train.adapter.path (adapter directory).

        When this workdir has a parent (e.g., a generation run spawned from training),
        returns the parent's adapter path since that's where the trained adapter lives.
        """
        if self._parent_workdir is not None:
            return self._parent_workdir.train.adapter.path  # type: ignore[return-value]
        return self.train.adapter.path  # type: ignore[return-value]

    @property
    def metadata_file(self) -> Path:
        """Shortcut to train.adapter.metadata.

        Uses parent workdir's path when available.
        """
        if self._parent_workdir is not None:
            return self._parent_workdir.train.adapter.metadata  # type: ignore[return-value]
        return self.train.adapter.metadata  # type: ignore[return-value]

    @property
    def schema_file(self) -> Path:
        """Shortcut to train.adapter.schema.

        Uses parent workdir's path when available.
        """
        if self._parent_workdir is not None:
            return self._parent_workdir.train.adapter.schema  # type: ignore[return-value]
        return self.train.adapter.schema  # type: ignore[return-value]

    @property
    def dataset_schema_file(self) -> Path:
        """Alias for schema_file (backwards compatibility)."""
        return self.schema_file

    @property
    def output_file(self) -> Path:
        """Shortcut to generate.output."""
        return self.generate.output  # type: ignore[return-value]

    @property
    def evaluation_report(self) -> Path:
        """Shortcut to generate.report."""
        return self.generate.report  # type: ignore[return-value]

    # =========================================================================
    # Source paths (for generation runs that have a parent training run)
    # =========================================================================

    @property
    def source_run_dir(self) -> Path:
        """Source run directory (parent's ``run_dir`` for child generation runs)."""
        if self._parent_workdir is not None:
            return self._parent_workdir.run_dir
        return self.run_dir

    @property
    def source_config(self) -> Path:
        """Source config file path (from parent workdir if available).

        Checks multiple locations for backwards compatibility:
        1. Root level config: ``<run_dir>/safe-synthesizer-config.json``
        2. Train config: ``<run_dir>/train/safe-synthesizer-config.json``
        """
        source_workdir = self._parent_workdir if self._parent_workdir is not None else self

        # First check root level config
        root_config = source_workdir.config
        if root_config.exists():
            return root_config

        # Fallback to train directory config (older training runs)
        train_config = source_workdir.train.config  # type: ignore[return-value]
        if train_config.exists():
            return train_config

        # Return root config path (will fail with appropriate error message)
        return root_config

    @property
    def source_adapter_path(self) -> Path:
        """Source adapter path (from parent workdir if available)."""
        if self._parent_workdir is not None:
            return self._parent_workdir.adapter_path
        return self.adapter_path

    @property
    def source_dataset(self) -> BoundDir:
        """Source dataset directory (from parent workdir if available)."""
        if self._parent_workdir is not None:
            return self._parent_workdir.dataset  # type: ignore[return-value]
        return self.dataset  # type: ignore[return-value]

    @property
    def source_schema_file(self) -> Path:
        """Source schema file path (from parent workdir if available)."""
        if self._parent_workdir is not None:
            return self._parent_workdir.schema_file
        return self.schema_file

    # =========================================================================
    # Methods
    # =========================================================================

    def ensure_directories(self) -> Self:
        """Create directories based on the current phase.

        For training runs: creates train/, generate/, and dataset/ directories
        For generation-only runs: creates only generate/ directory and writes info.txt

        Returns:
            self for method chaining
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if self._current_phase == "generate" and self._parent_workdir is not None:
            # Generation-only run - only create generate directory
            # Train and dataset are in the parent workdir
            self.generate.path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            self._write_generation_info()
        else:
            # Training run or end-to-end - create all directories
            self.train.cache.path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            self.train.adapter.path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            self.generate.path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            self.dataset.path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

        return self

    def _write_generation_info(self) -> None:
        """Write info.json with adapter source information for generation-only runs."""
        if self._parent_workdir is None:
            return

        info_dict = {
            "Safe Synthesizer Generation Run": "=" * 40,
            "Generated": datetime.now().isoformat(),
            "Adapter Source": {
                "Run directory": str(self._parent_workdir.run_dir),
                "Adapter path": str(self._parent_workdir.adapter_path),
            },
            "Source Files": {
                "Config": str(self.source_config),
                "Training data": str(self.source_dataset.training),
                "Test data": str(self.source_dataset.test),
                "Schema": str(self.schema_file),
            },
        }

        info_file = self.generate.info  # type: ignore[attr-defined]
        write_json(info_dict, info_file, indent=2)

    def new_generation_run(self) -> Self:
        """Create a new Workdir for a generation run from this workdir.

        This method is used when resuming from a trained model to run generation.
        The new Workdir shares the same project but gets a new run_name, and
        references this workdir as the parent for loading config/data/adapter.

        Returns:
            New Workdir instance with a fresh timestamp-based run_name and this workdir as parent
        """
        # Create a new RunName with a fresh timestamp (auto-generated)
        new_run_name = RunName()
        logger.info(f"Created new generation run: {new_run_name.to_string()}")
        logger.info(f"Parent workdir (for adapter/config/data): {self.run_dir}")

        return self.__class__(
            base_path=self.base_path,
            config_name=self.config_name,
            dataset_name=self.dataset_name,
            run_name=new_run_name.to_string(),
            _current_phase="generate",
            _parent_workdir=self,
        )

    @classmethod
    def from_explicit_run_path(
        cls,
        run_path: Path,
        config_name: str,
        dataset_name: str,
        current_phase: str = "unknown",
    ) -> Workdir:
        """Create Workdir from an explicit run path (no auto-generated nesting).

        Used when --run-path is provided on the CLI. The path is used directly
        as the run directory, without the normal <project>/<timestamp> nesting.

        Args:
            run_path: Explicit path to use as the run directory
            config_name: Name of the config (used for project naming)
            dataset_name: Name of the dataset (used for project naming)
            current_phase: The current phase (train, generate, end_to_end)

        Returns:
            Workdir with run_dir set to run_path

        Raises:
            ValueError: If run_path already contains a trained adapter
        """
        run_path = Path(run_path).resolve()

        # Check if path already contains a previous run (Option A: error)
        adapter_dir = run_path / "train" / "adapter"
        if adapter_dir.is_dir():
            adapter_files = list(adapter_dir.glob("*.safetensors"))
            if adapter_files:
                raise ValueError(
                    f"--run-path '{run_path}' already contains a training run.\n"
                    f"Use a different path or delete the existing run."
                )

        # For explicit paths, we store the path directly and use _explicit_run_path
        # to override the normal run_dir calculation. The base_path and run_name are
        # set for metadata purposes but won't affect the actual directory location.
        run_name = run_path.name
        base_path = run_path.parent

        logger.info(f"Using explicit run path: {run_path}")

        return cls(
            base_path=base_path,
            config_name=config_name,
            dataset_name=dataset_name,
            run_name=run_name,
            _current_phase=current_phase,
            _explicit_run_path=run_path,
        )

    @classmethod
    def from_path(cls, path: Path) -> Workdir:
        """Load a Workdir from an existing path.

        This method handles three scenarios:
        1. Path is a run_dir (contains train/adapter/ with safetensors) - use it directly
        2. Path is a project_dir - find the latest run within that project
        3. Path is a base_path - find the latest run across all projects

        Args:
            path: Path to run_dir, project_dir, or base_path

        Returns:
            Workdir pointing to the existing run

        Raises:
            ValueError: If path doesn't exist or no valid run found
        """
        if not path.is_dir():
            raise ValueError(f"Invalid path: {path}")

        # Check if this is a run_dir (has train/adapter/ subdirectory with safetensors)
        train_dir = path / "train"
        adapter_dir = train_dir / "adapter" if train_dir.is_dir() else path / "adapter"

        if adapter_dir.is_dir():
            adapter_files = list(adapter_dir.glob("*.safetensors"))
            if adapter_files:
                # This is a run_dir - parse structure from path
                # Path structure: base_path/<config>---<dataset>/<run_name>
                run_name = path.name
                project_dir = path.parent
                base_path = project_dir.parent

                # Parse project name using pattern matching helper
                config_name, dataset_name = _parse_project_name(project_dir.name)

                logger.info(f"Found existing workdir at {path}")
                logger.info(f"Adapter files: {adapter_files}")

                return cls(
                    base_path=base_path,
                    config_name=config_name,
                    dataset_name=dataset_name,
                    run_name=run_name,
                )

        # Check if this is a project_dir - find the latest run with an adapter
        adapter_files = list(path.glob("*/train/adapter/*.safetensors"))
        if adapter_files:
            adapter_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            latest_adapter = adapter_files[0]
            run_dir = latest_adapter.parent.parent.parent  # adapter file -> adapter -> train -> run_dir

            # Parse project name using pattern matching helper
            config_name, dataset_name = _parse_project_name(path.name)

            logger.info(f"Found {len(adapter_files)} runs with adapters in {path}")
            logger.info(f"Using most recent run: {run_dir}")

            return cls(
                base_path=path.parent,
                config_name=config_name,
                dataset_name=dataset_name,
                run_name=run_dir.name,
            )

        # Check if this is a base_path - find the latest run across all projects
        adapter_files = list(path.glob("*/*/train/adapter/*.safetensors"))
        if adapter_files:
            adapter_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            latest_adapter = adapter_files[0]
            # adapter file -> adapter -> train -> run_dir -> project_dir
            run_dir = latest_adapter.parent.parent.parent
            project_dir = run_dir.parent

            # Parse project name using pattern matching helper
            config_name, dataset_name = _parse_project_name(project_dir.name)

            logger.info(f"Found {len(adapter_files)} runs with adapters across all projects in {path}")
            logger.info(f"Usig ggmost recent run: {run_dir}")

            return cls(
                base_path=path,
                config_name=config_name,
                dataset_name=dataset_name,
                run_name=run_dir.name,
            )

        raise ValueError(f"No valid run found in {path}")
