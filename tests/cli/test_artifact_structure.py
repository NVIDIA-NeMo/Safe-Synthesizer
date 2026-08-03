# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the artifact_structure module."""

import ast
import inspect
import os
from pathlib import Path

import pytest

from nemo_safe_synthesizer.cli.artifact_structure import (
    PROJECT_NAME_DELIMITER,
    BoundDir,
    DirNode,
    FileNode,
    RunName,
    Workdir,
    _parse_project_name,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def workdir(fixture_session_cache_dir: Path) -> Workdir:
    """Create a standard Workdir for testing with a fixed base path."""
    return Workdir(
        base_path=fixture_session_cache_dir,
        config_name="default",
        dataset_name="adult",
        run_name="2026-01-15T12:00:00",
    )


@pytest.fixture
def workdir_tmp(fixture_session_cache_dir: Path) -> Workdir:
    """Create a Workdir with a real temporary directory for filesystem tests."""
    return Workdir(
        base_path=fixture_session_cache_dir,
        config_name="test",
        dataset_name="data",
        run_name="2026-01-15T12:00:00",
    )


# =============================================================================
# RunName Tests
# =============================================================================


class TestRunName:
    """Tests for the RunName dataclass."""

    def test_auto_generates_timestamp(self):
        """RunName() auto-generates a timestamp-based name."""
        run_name = RunName()

        # Should have a non-empty value
        assert run_name.to_string()
        # Should be timestamp-based
        assert run_name.is_timestamp_based
        assert run_name.timestamp is not None
        # Should match the expected format (YYYY-MM-DDTHH:MM:SS)
        assert len(run_name.to_string()) == 19
        assert "T" in run_name.to_string()

    def test_from_string_with_timestamp(self):
        """from_string() parses timestamp-formatted strings."""
        timestamp_str = "2026-01-15T12:00:00"
        run_name = RunName.from_string(timestamp_str)

        assert run_name.to_string() == timestamp_str
        assert run_name.is_timestamp_based
        assert run_name.timestamp is not None
        assert run_name.timestamp.year == 2026
        assert run_name.timestamp.month == 1
        assert run_name.timestamp.day == 15

    def test_from_string_with_arbitrary_name(self):
        """from_string() accepts arbitrary non-timestamp strings."""
        arbitrary_name = "my-experiment-run"
        run_name = RunName.from_string(arbitrary_name)

        assert run_name.to_string() == arbitrary_name
        # Should NOT be timestamp-based
        assert not run_name.is_timestamp_based
        assert run_name.timestamp is None

    def test_from_string_with_job_style_name(self):
        """from_string() accepts job-style names."""
        job_name = "config_dataset_3"
        run_name = RunName.from_string(job_name)

        assert run_name.to_string() == job_name
        assert not run_name.is_timestamp_based

    def test_from_string_with_hyphenated_name(self):
        """from_string() accepts hyphenated names."""
        hyphenated = "my-experiment-run"
        run_name = RunName.from_string(hyphenated)

        assert run_name.to_string() == hyphenated
        assert not run_name.is_timestamp_based

    def test_is_timestamp_based_property(self):
        """is_timestamp_based correctly identifies timestamp vs arbitrary names."""
        # Timestamp-based
        ts_name = RunName()
        assert ts_name.is_timestamp_based

        parsed_ts = RunName.from_string("2026-01-15T12:00:00")
        assert parsed_ts.is_timestamp_based

        # Not timestamp-based
        arbitrary = RunName.from_string("arbitrary_name")
        assert not arbitrary.is_timestamp_based


# =============================================================================
# FileNode Tests
# =============================================================================


class TestFileNode:
    """Tests for the FileNode descriptor."""

    def test_class_access_returns_descriptor(self):
        """Accessing FileNode on class returns the descriptor itself."""

        class TestClass:
            config = FileNode("config.json")

        assert isinstance(TestClass.config, FileNode)
        assert TestClass.config.name == "config.json"

    def test_instance_access_returns_path(self, workdir: Workdir):
        """Accessing FileNode on instance returns Path."""
        config_path = workdir.config
        assert isinstance(config_path, Path)
        assert config_path == workdir.run_dir / "safe-synthesizer-config.json"


# =============================================================================
# DirNode Tests
# =============================================================================


class TestDirNode:
    """Tests for the DirNode descriptor."""

    def test_class_access_returns_descriptor(self):
        """Accessing DirNode on class returns the descriptor itself."""

        class TestClass:
            train = DirNode("train", config=FileNode("config.json"))

        assert isinstance(TestClass.train, DirNode)
        assert TestClass.train.name == "train"
        assert "config" in TestClass.train.children

    def test_instance_access_returns_bound_dir(self, workdir: Workdir):
        """Accessing DirNode on instance returns BoundDir."""
        train = workdir.train
        assert isinstance(train, BoundDir)
        assert train.path == workdir.run_dir / "train"


# =============================================================================
# BoundDir Tests
# =============================================================================


class TestBoundDir:
    """Tests for the BoundDir runtime class."""

    def test_path_property(self):
        """BoundDir.path returns the directory Path."""
        bound = BoundDir(Path("/test/path"), {})
        assert bound.path == Path("/test/path")

    def test_fspath_compatibility(self):
        """BoundDir works with os.fspath()."""
        bound = BoundDir(Path("/test/path"), {})
        assert os.fspath(bound) == "/test/path"

    def test_str_returns_path_string(self):
        """str(BoundDir) returns the path as string."""
        bound = BoundDir(Path("/test/path"), {})
        assert str(bound) == "/test/path"

    def test_file_child_access(self):
        """Accessing a FileNode child returns a Path."""
        bound = BoundDir(Path("/test"), {"config": FileNode("config.json")})
        assert bound.config == Path("/test/config.json")

    def test_dir_child_access(self):
        """Accessing a DirNode child returns another BoundDir."""
        bound = BoundDir(
            Path("/test"),
            {"subdir": DirNode("subdir", file=FileNode("file.txt"))},
        )
        subdir = bound.subdir
        assert isinstance(subdir, BoundDir)
        assert subdir.path == Path("/test/subdir")

    def test_nested_child_access(self):
        """Can access nested children through chain."""
        bound = BoundDir(
            Path("/test"),
            {
                "level1": DirNode(
                    "level1",
                    level2=DirNode("level2", file=FileNode("deep.txt")),
                )
            },
        )
        assert bound.level1.level2.file == Path("/test/level1/level2/deep.txt")  # ty: ignore[unresolved-attribute] -- BoundDir delegates via __getattr__

    def test_missing_child_raises_attribute_error(self):
        """Accessing non-existent child raises AttributeError."""
        bound = BoundDir(Path("/test"), {"exists": FileNode("exists.txt")})
        with pytest.raises(AttributeError, match="No child 'missing'"):
            _ = bound.missing

    def test_private_attr_raises_attribute_error(self):
        """Accessing private attributes raises AttributeError."""
        bound = BoundDir(Path("/test"), {})
        with pytest.raises(AttributeError):
            _ = bound._private


# =============================================================================
# Workdir Tests
# =============================================================================


class TestWorkdir:
    """Tests for the Workdir class."""

    def test_base_path_string_conversion(self):
        """base_path is converted from string to Path."""
        workdir = Workdir(
            base_path=Path("/string/path"),
            config_name="cfg",
            dataset_name="data",
            run_name="2026-01-15T12:00:00",
        )
        assert isinstance(workdir.base_path, Path)
        assert workdir.base_path == Path("/string/path")

    def test_project_name(self, workdir: Workdir):
        """project_name combines config and dataset."""
        assert workdir.project_name == "default---adult"

    def test_project_dir(self, workdir: Workdir):
        """project_dir is base_path/project_name."""
        assert workdir.project_dir == workdir.base_path / workdir.project_name

    def test_run_dir(self, workdir: Workdir):
        """run_dir is project_dir/run_name."""
        assert workdir.run_name, "run_name should be populated by __post_init__"
        assert workdir.run_dir == workdir.project_dir / workdir.run_name

    def test_workdir_with_arbitrary_run_name(self, fixture_session_cache_dir: Path):
        """Workdir accepts arbitrary (non-timestamp) run names."""
        workdir = Workdir(
            base_path=fixture_session_cache_dir,
            config_name="test",
            dataset_name="data",
            run_name="my-experiment-run",  # Job-style name
        )

        assert workdir.run_name == "my-experiment-run"
        assert workdir.run_dir == workdir.project_dir / "my-experiment-run"
        # Should have a RunName object that's not timestamp-based
        assert not workdir._run_name_obj.is_timestamp_based

    def test_workdir_with_hyphenated_run_name(self, fixture_session_cache_dir: Path):
        """Workdir accepts hyphenated run names."""
        workdir = Workdir(
            base_path=fixture_session_cache_dir,
            config_name="test",
            dataset_name="data",
            run_name="my-custom-experiment",
        )

        assert workdir.run_name == "my-custom-experiment"
        assert not workdir._run_name_obj.is_timestamp_based

    def test_workdir_with_timestamp_run_name_is_recognized(self, fixture_session_cache_dir: Path):
        """Workdir correctly identifies timestamp-based run names."""
        workdir = Workdir(
            base_path=fixture_session_cache_dir,
            config_name="test",
            dataset_name="data",
            run_name="2026-01-15T12:00:00",
        )

        assert workdir.run_name == "2026-01-15T12:00:00"
        assert workdir._run_name_obj.is_timestamp_based
        assert workdir._run_name_obj.timestamp is not None

    # =========================================================================
    # Train directory structure
    # =========================================================================

    def test_train_path(self, workdir: Workdir):
        """train.path is run_dir/train."""
        assert workdir.train.path == workdir.run_dir / "train"

    def test_train_config(self, workdir: Workdir):
        """train.config is the config file in train/."""
        assert workdir.train.config == workdir.train.path / "safe-synthesizer-config.json"

    def test_train_adapter_path(self, workdir: Workdir):
        """train.adapter.path is train/adapter."""
        assert workdir.train.adapter.path == workdir.train.path / "adapter"

    def test_train_model_files(self, workdir: Workdir):
        """Adapter directory contains expected files."""
        adapter = workdir.train.adapter
        assert adapter.adapter_config == adapter.path / "adapter_config.json"
        assert adapter.nss_tokenizer == adapter.path / "nss_tokenizer.json"
        assert adapter.metadata == adapter.path / "metadata_v2.json"
        assert adapter.schema == adapter.path / "dataset_schema.json"

    # =========================================================================
    # Generate directory structure
    # =========================================================================

    def test_generate_path(self, workdir: Workdir):
        """generate.path is run_dir/generate."""
        assert workdir.generate.path == workdir.run_dir / "generate"

    def test_generate_files(self, workdir: Workdir):
        """Generate directory contains expected files."""
        gen = workdir.generate
        assert gen.logs == gen.path / "logs.jsonl"
        assert gen.output == gen.path / "synthetic_data.csv"
        assert gen.report == gen.path / "evaluation_report.html"
        assert gen.evaluation_metrics == gen.path / "evaluation_metrics.json"

    # =========================================================================
    # Dataset directory structure
    # =========================================================================

    def test_dataset_path(self, workdir: Workdir):
        """dataset.path is run_dir/dataset."""
        assert workdir.dataset.path == workdir.run_dir / "dataset"

    def test_dataset_files(self, workdir: Workdir):
        """Dataset directory contains expected files."""
        ds = workdir.dataset
        assert ds.training == ds.path / "training.csv"
        assert ds.test == ds.path / "test.csv"
        assert ds.validation == ds.path / "validation.csv"

    # =========================================================================
    # Convenience aliases
    # =========================================================================

    def test_adapter_path_alias(self, workdir: Workdir):
        """adapter_path shortcut matches full path."""
        assert workdir.adapter_path == workdir.train.adapter.path

    def test_metadata_file_alias(self, workdir: Workdir):
        """metadata_file shortcut matches full path."""
        assert workdir.metadata_file == workdir.train.adapter.metadata

    def test_schema_file_alias(self, workdir: Workdir):
        """schema_file shortcut matches full path."""
        assert workdir.schema_file == workdir.train.adapter.schema

    def test_output_file_alias(self, workdir: Workdir):
        """output_file shortcut matches full path."""
        assert workdir.output_file == workdir.generate.output

    def test_evaluation_report_alias(self, workdir: Workdir):
        """evaluation_report shortcut matches full path."""
        assert workdir.evaluation_report == workdir.generate.report

    def test_evaluation_metrics_alias(self, workdir: Workdir):
        """evaluation_metrics shortcut matches full path."""
        assert workdir.evaluation_metrics == workdir.generate.evaluation_metrics

    # =========================================================================
    # Directory creation
    # =========================================================================

    def test_ensure_directories_creates_structure(self, workdir_tmp: Workdir):
        """ensure_directories creates the expected directory structure."""
        workdir_tmp.ensure_directories()

        assert workdir_tmp.train.adapter.path.is_dir()
        assert workdir_tmp.generate.path.is_dir()
        assert workdir_tmp.dataset.path.is_dir()

    def test_ensure_directories_returns_self(self, workdir_tmp: Workdir):
        """ensure_directories returns self for method chaining."""
        result = workdir_tmp.ensure_directories()
        assert result is workdir_tmp

    def test_ensure_directories_idempotent(self, workdir_tmp: Workdir):
        """ensure_directories can be called multiple times safely."""
        workdir_tmp.ensure_directories()
        workdir_tmp.ensure_directories()  # Should not raise

        assert workdir_tmp.train.adapter.path.is_dir()

    # =========================================================================
    # from_path tests
    # =========================================================================

    def test_from_path_with_run_dir(self, workdir_tmp: Workdir):
        """from_path correctly loads a workdir from a run_dir."""
        # Create adapter directory and dummy safetensors file
        adapter_dir = workdir_tmp.train.adapter.path
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        adapter_file.touch()

        loaded = Workdir.from_path(workdir_tmp.run_dir)

        assert loaded.run_dir == workdir_tmp.run_dir
        assert loaded.config_name == workdir_tmp.config_name
        assert loaded.dataset_name == workdir_tmp.dataset_name

    def test_from_path_with_project_dir(self, workdir_tmp: Workdir):
        """from_path finds latest run when given a project_dir."""
        # Create adapter directory and dummy safetensors file
        adapter_dir = workdir_tmp.train.adapter.path
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        adapter_file.touch()

        loaded = Workdir.from_path(workdir_tmp.project_dir)

        assert loaded.run_dir == workdir_tmp.run_dir
        assert loaded.config_name == workdir_tmp.config_name
        assert loaded.dataset_name == workdir_tmp.dataset_name

    def test_from_path_with_base_path(self, workdir_tmp: Workdir):
        """from_path finds latest run across all projects when given a base_path."""
        # Create adapter directory and dummy safetensors file
        adapter_dir = workdir_tmp.train.adapter.path
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        adapter_file.touch()

        loaded = Workdir.from_path(workdir_tmp.base_path)

        assert loaded.run_dir == workdir_tmp.run_dir
        assert loaded.config_name == workdir_tmp.config_name
        assert loaded.dataset_name == workdir_tmp.dataset_name

    def test_from_path_invalid_raises(self, tmp_path: Path):
        """from_path raises ValueError for invalid paths."""
        with pytest.raises(ValueError, match="Invalid path"):
            Workdir.from_path(tmp_path / "nonexistent")

    def test_from_path_no_adapter_raises(self, tmp_path: Path):
        """from_path raises ValueError when no adapter found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No valid run found"):
            Workdir.from_path(empty_dir)

    # =========================================================================
    # from_explicit_run_path tests
    # =========================================================================

    def test_from_explicit_run_path_creates_workdir(self, tmp_path: Path):
        """from_explicit_run_path creates workdir with run_dir set to explicit path."""
        run_path = tmp_path / "my-explicit-run"

        workdir = Workdir.from_explicit_run_path(
            run_path=run_path,
            config_name="test-config",
            dataset_name="test-data",
            current_phase="train",
        )

        assert workdir.run_dir == run_path
        assert workdir.config_name == "test-config"
        assert workdir.dataset_name == "test-data"
        assert workdir._current_phase == "train"

    def test_from_explicit_run_path_resolves_path(self, tmp_path: Path):
        """from_explicit_run_path resolves the path to absolute."""
        # Use a relative-like path structure
        run_path = tmp_path / "subdir" / ".." / "my-run"

        workdir = Workdir.from_explicit_run_path(
            run_path=run_path,
            config_name="cfg",
            dataset_name="data",
        )

        # run_dir should be resolved
        assert workdir.run_dir == (tmp_path / "my-run").resolve()
        assert workdir.run_dir.is_absolute()

    def test_from_explicit_run_path_errors_if_adapter_exists(self, tmp_path: Path):
        """from_explicit_run_path raises ValueError if adapter already exists."""
        run_path = tmp_path / "existing-run"
        adapter_path = run_path / "train" / "adapter"
        adapter_path.mkdir(parents=True)
        (adapter_path / "model.safetensors").touch()

        with pytest.raises(ValueError, match="already contains a training run"):
            Workdir.from_explicit_run_path(
                run_path=run_path,
                config_name="cfg",
                dataset_name="data",
            )

    def test_from_explicit_run_path_allows_empty_existing_dir(self, tmp_path: Path):
        """from_explicit_run_path allows existing empty directory."""
        run_path = tmp_path / "empty-run"
        run_path.mkdir()

        workdir = Workdir.from_explicit_run_path(
            run_path=run_path,
            config_name="cfg",
            dataset_name="data",
        )

        assert workdir.run_dir == run_path

    def test_from_explicit_run_path_ensure_directories_works(self, tmp_path: Path):
        """ensure_directories creates structure in explicit run path."""
        run_path = tmp_path / "new-run"

        workdir = Workdir.from_explicit_run_path(
            run_path=run_path,
            config_name="cfg",
            dataset_name="data",
            current_phase="train",
        )
        workdir.ensure_directories()

        assert (run_path / "train" / "adapter").is_dir()
        assert (run_path / "generate").is_dir()
        assert (run_path / "dataset").is_dir()

    # =========================================================================
    # new_generation_run tests
    # =========================================================================

    def test_new_generation_run_sets_parent(self, workdir_tmp: Workdir):
        """new_generation_run creates child workdir with parent reference."""
        child = workdir_tmp.new_generation_run()

        assert child._parent_workdir is workdir_tmp
        assert child.run_name != workdir_tmp.run_name
        assert child.project_dir == workdir_tmp.project_dir

    def test_new_generation_run_adapter_path_uses_parent(self, workdir_tmp: Workdir):
        """Child workdir's adapter_path returns parent's adapter path."""
        # Create adapter in parent
        adapter_dir = workdir_tmp.train.adapter.path
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        adapter_file.touch()

        child = workdir_tmp.new_generation_run()

        # Child's adapter_path should point to parent's adapter
        assert child.adapter_path == workdir_tmp.train.adapter.path
        assert child.adapter_path.exists()

    def test_new_generation_run_source_config_uses_parent(self, workdir_tmp: Workdir):
        """Child workdir's source_config returns parent's config path."""
        child = workdir_tmp.new_generation_run()

        assert child.source_config == workdir_tmp.config
        assert child.source_config != child.config  # Different run dirs

    def test_new_generation_run_source_dataset_uses_parent(self, workdir_tmp: Workdir):
        """Child workdir's source_dataset returns parent's dataset."""
        child = workdir_tmp.new_generation_run()

        assert child.source_dataset.training == workdir_tmp.dataset.training
        assert child.source_dataset.test == workdir_tmp.dataset.test

    def test_new_generation_run_ensure_directories_only_creates_generate(self, workdir_tmp: Workdir):
        """Generation-only run should only create generate/ directory, not train/."""
        child = workdir_tmp.new_generation_run()
        child.ensure_directories()

        # Generate directory should exist
        assert child.generate.path.is_dir()
        # Train directory should NOT exist in child's run_dir
        assert not child.train.path.is_dir()
        # Dataset directory should NOT exist in child's run_dir
        assert not child.dataset.path.is_dir()


# =============================================================================
# OS Path Integration Tests
# =============================================================================


class TestOsPathIntegration:
    """Tests for Path/os.path integration."""

    def test_bound_dir_with_open(self, workdir_tmp: Workdir):
        """BoundDir can be used with open() via os.fspath."""
        workdir_tmp.ensure_directories()

        # Write using the path
        test_content = "test content"
        with open(workdir_tmp.generate.output, "w") as f:
            f.write(test_content)

        # Read back
        with open(workdir_tmp.generate.output) as f:
            assert f.read() == test_content

    def test_path_join_with_bound_dir(self, workdir: Workdir):
        """Path / BoundDir works correctly."""
        extra = Path(os.fspath(workdir.train)) / "extra"
        assert extra == workdir.train.path / "extra"


class TestParseProjectName:
    """Tests for _parse_project_name function."""

    def test_simple_names(self):
        """Simple config and dataset names are parsed correctly."""
        config, dataset = _parse_project_name(f"config{PROJECT_NAME_DELIMITER}dataset")
        assert config == "config"
        assert dataset == "dataset"

    def test_dashed_config_name(self):
        """Config names with dashes are preserved."""
        config, dataset = _parse_project_name(f"my-config{PROJECT_NAME_DELIMITER}dataset")
        assert config == "my-config"
        assert dataset == "dataset"

    def test_dashed_dataset_name(self):
        """Dataset names with dashes are preserved."""
        config, dataset = _parse_project_name(f"config{PROJECT_NAME_DELIMITER}my-dataset")
        assert config == "config"
        assert dataset == "my-dataset"

    def test_both_dashed(self):
        """Both config and dataset can have dashes."""
        config, dataset = _parse_project_name(f"my-config{PROJECT_NAME_DELIMITER}my-dataset")
        assert config == "my-config"
        assert dataset == "my-dataset"

    def test_no_delimiter_returns_unknown_dataset(self):
        """Project name without delimiter returns unknown for dataset."""
        config, dataset = _parse_project_name("config-only")
        assert config == "config-only"
        assert dataset == "unknown"

    def test_workdir_project_name_roundtrip(self, fixture_session_cache_dir: Path):
        """Workdir project_name can be parsed back correctly."""
        workdir = Workdir(
            base_path=fixture_session_cache_dir,
            config_name="test-config",
            dataset_name="test-dataset",
            run_name="run-1",
        )
        # Verify the project name uses the correct delimiter
        assert PROJECT_NAME_DELIMITER in workdir.project_name
        # Verify we can parse it back
        config, dataset = _parse_project_name(workdir.project_name)
        assert config == "test-config"
        assert dataset == "test-dataset"


# =============================================================================
# Typed-View Drift Guard
# =============================================================================
#
# The `Workdir` class declares its directory tree using `DirNode[_FooDir](...)`.
# `_FooDir` is a `BoundDir` subclass whose body lives under `if TYPE_CHECKING:`
# and exists only to declare the typed children of that subtree so `ty` (and
# editors) can resolve attribute access through `BoundDir.__getattr__`. We
# call these subclasses "typed views" internally (see the section comment in
# `artifact_structure.py` for context and prior art).
#
# If someone adds or renames a child in a `DirNode(...)` tree but forgets to
# update the matching typed-view subclass (or vice versa), type checking will
# silently weaken: the code still runs (thanks to `__getattr__`) but static
# analysis loses precision and ``ty: ignore`` comments start creeping back.
#
# This test reads `artifact_structure.py` with `ast` (typed-view annotations
# are gated by `TYPE_CHECKING` and therefore absent at runtime) and asserts
# that every `DirNode[_Foo](...)` call's keyword-argument names exactly match
# the attribute names declared on `_Foo`'s `if TYPE_CHECKING:` block.


def _typed_view_annotations(cls_node: ast.ClassDef) -> set[str]:
    """Extract attribute names from a typed-view class's ``if TYPE_CHECKING:`` body."""
    names: set[str] = set()
    for stmt in cls_node.body:
        if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "TYPE_CHECKING":
            for inner in stmt.body:
                if isinstance(inner, ast.AnnAssign) and isinstance(inner.target, ast.Name):
                    names.add(inner.target.id)
    return names


def _dir_node_view_and_children(call: ast.Call) -> tuple[str, set[str]] | None:
    """Return ``(view_name, child_kwarg_names)`` for ``DirNode[_Foo](name, **kwargs)``.

    Returns ``None`` for bare ``DirNode(...)`` calls (no typed-view type parameter).
    """
    func = call.func
    if (
        isinstance(func, ast.Subscript)
        and isinstance(func.value, ast.Name)
        and func.value.id == "DirNode"
        and isinstance(func.slice, ast.Name)
    ):
        view_name = func.slice.id
        kwargs = {kw.arg for kw in call.keywords if kw.arg is not None}
        return view_name, kwargs
    return None


def _collect_dir_node_calls(node: ast.AST) -> list[ast.Call]:
    """Walk an AST subtree and return every ``ast.Call`` node."""
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def _compute_drift_mismatches(
    view_annotations: dict[str, set[str]],
    dir_node_usages: list[tuple[str, set[str]]],
) -> list[str]:
    """Diff each ``DirNode[_Foo](...)`` call's kwargs against ``_Foo``'s annotations.

    Returns an empty list when every usage's kwargs exactly equal the
    annotations on its referenced typed view, otherwise one human-readable
    diagnostic per mismatch.
    """
    mismatches: list[str] = []
    for view_name, child_kwargs in dir_node_usages:
        annotated = view_annotations.get(view_name)
        if annotated is None:
            mismatches.append(f"DirNode[{view_name}] references unknown typed-view class")
            continue
        if annotated != child_kwargs:
            only_in_view = sorted(annotated - child_kwargs)
            only_in_dir_node = sorted(child_kwargs - annotated)
            mismatches.append(
                f"DirNode[{view_name}]: "
                f"only in {view_name} annotations = {only_in_view}; "
                f"only in DirNode kwargs = {only_in_dir_node}"
            )
    return mismatches


class TestTypedViewDrift:
    """Guards that `DirNode[_Foo](...)` keys match `_Foo`'s typed attributes.

    Failures here mean somebody added/renamed a child on one side of the
    ``DirNode[_FooDir]`` ↔ typed-view-subclass pair without updating the
    other. ``artifact_structure.py`` documents the convention.
    """

    @pytest.fixture(scope="class")
    def module_ast(self) -> ast.Module:
        module = inspect.getmodule(Workdir)
        assert module is not None, "Workdir must be importable from a real source file"
        return ast.parse(inspect.getsource(module))

    @pytest.fixture(scope="class")
    def typed_view_annotations(self, module_ast: ast.Module) -> dict[str, set[str]]:
        """Map typed-view class name → attribute names declared under ``TYPE_CHECKING``."""
        return {
            node.name: _typed_view_annotations(node)
            for node in module_ast.body
            if isinstance(node, ast.ClassDef) and node.name.startswith("_") and node.name.endswith("Dir")
        }

    @pytest.fixture(scope="class")
    def dir_node_usages(self, module_ast: ast.Module) -> list[tuple[str, set[str]]]:
        """List of every ``DirNode[_Foo](...)`` call's (typed-view name, child kwarg names)."""
        usages: list[tuple[str, set[str]]] = []
        for call in _collect_dir_node_calls(module_ast):
            result = _dir_node_view_and_children(call)
            if result is not None:
                usages.append(result)
        return usages

    def test_typed_view_classes_discovered(self, typed_view_annotations: dict[str, set[str]]):
        """Sanity check: ensure we actually found the typed-view subclasses."""
        assert {"_AdapterDir", "_TrainDir", "_GenerateDir", "_DatasetDir"} <= set(typed_view_annotations)

    def test_dir_node_usages_discovered(self, dir_node_usages: list[tuple[str, set[str]]]):
        """Sanity check: ensure we actually found at least the top-level typed DirNodes."""
        views_used = {view for view, _ in dir_node_usages}
        assert {"_TrainDir", "_AdapterDir", "_GenerateDir", "_DatasetDir"} <= views_used

    def test_dir_node_keys_match_typed_view_annotations(
        self,
        typed_view_annotations: dict[str, set[str]],
        dir_node_usages: list[tuple[str, set[str]]],
    ):
        """Every ``DirNode[_Foo](...)`` kwargs must equal ``_Foo``'s TYPE_CHECKING attrs."""
        mismatches = _compute_drift_mismatches(typed_view_annotations, dir_node_usages)
        assert not mismatches, "Typed-view / DirNode drift:\n  " + "\n  ".join(mismatches)


class TestTypedViewDriftHelpers:
    """Direct unit tests for the typed-view drift helper functions.

    These exercise the error-reporting branches that the real module never
    hits (unknown typed view, kwargs mismatch, bare ``DirNode(...)`` calls).
    """

    @staticmethod
    def _parse_single_call(source: str) -> ast.Call:
        """Parse a one-expression snippet and return its single ``ast.Call`` node."""
        tree = ast.parse(source, mode="eval")
        assert isinstance(tree.body, ast.Call)
        return tree.body

    def test_dir_node_view_and_children_extracts_subscripted_call(self):
        """``DirNode[_Foo](name, a=..., b=...)`` → ``('_Foo', {'a', 'b'})``."""
        call = self._parse_single_call('DirNode[_FooDir]("foo", a=1, b=2)')
        assert _dir_node_view_and_children(call) == ("_FooDir", {"a", "b"})

    def test_dir_node_view_and_children_returns_none_for_bare_dir_node(self):
        """Unsubscripted ``DirNode(...)`` calls yield ``None`` (typed view unknown)."""
        call = self._parse_single_call('DirNode("foo", a=1)')
        assert _dir_node_view_and_children(call) is None

    def test_dir_node_view_and_children_returns_none_for_other_calls(self):
        """Calls that aren't ``DirNode[...](...)`` at all yield ``None``."""
        call = self._parse_single_call("FileNode('foo.json')")
        assert _dir_node_view_and_children(call) is None

    def test_compute_drift_mismatches_clean(self):
        """No mismatches when every usage lines up with its typed-view annotations."""
        views = {"_FooDir": {"a", "b"}}
        usages = [("_FooDir", {"a", "b"})]
        assert _compute_drift_mismatches(views, usages) == []

    def test_compute_drift_mismatches_unknown_view(self):
        """``DirNode[_Missing]`` with no matching class is reported by name."""
        mismatches = _compute_drift_mismatches(
            view_annotations={},
            dir_node_usages=[("_Missing", {"x"})],
        )
        assert mismatches == ["DirNode[_Missing] references unknown typed-view class"]

    def test_compute_drift_mismatches_reports_both_sides(self):
        """Diagnostic splits extras into 'only in typed view' vs 'only in DirNode kwargs'."""
        views = {"_FooDir": {"a", "b", "c"}}
        usages = [("_FooDir", {"a", "d"})]
        mismatches = _compute_drift_mismatches(views, usages)
        assert len(mismatches) == 1
        msg = mismatches[0]
        assert "only in _FooDir annotations = ['b', 'c']" in msg
        assert "only in DirNode kwargs = ['d']" in msg
