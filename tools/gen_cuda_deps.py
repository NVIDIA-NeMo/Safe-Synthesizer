#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "click>=8",
#     "packaging>=24",
#     "pydantic",
#     "structlog",
#     "tomlkit",
# ]
# ///
"""Generate CUDA-related pyproject.toml sections from cuda_deps.toml.

Developer orientation:

This script has three intentionally readable layers.

1. Command entrypoint and presentation
   Click owns argument parsing, logging setup, stdout/stderr behavior, and exit
   codes. It should delegate quickly to `run_generation_command()` and avoid
   knowing TOML structure.

2. High-level API
   The stable spine is:
   `load_cuda_deps_config()` ->
   `build_cuda_pyproject_fragment()` ->
   `apply_cuda_fragment_to_pyproject()` / command output handling.
   These functions describe what the tool does in domain language and are the
   first place future agents should read.

3. Mid-level processing
   The lower layer collects uv sources/indexes, builds the generated TOML
   fragment, and splices that fragment into an existing pyproject. Keep
   pyproject mutation and generated-marker mechanics separate from dependency
   rendering.

The CUDA variant object model also has a deliberate split:

* `CudaVariantContext` owns one variant's template context and dependency stack.
* `CudaVariantDependencyRenderer` renders PEP 508 dependency strings and package
  names.
* `CudaVariantUvRouter` materializes `[tool.uv.sources]` and `[[tool.uv.index]]`
  routing for that variant.

When changing this file, prefer preserving those boundaries over adding a single
shortcut method to whichever class is nearby.
"""

import sys
import tomllib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, NamedTuple, Self

import click
import structlog
import tomlkit
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tomlkit import TOMLDocument
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.items import AoT, Array, Table

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Generated pyproject marker model
# ---------------------------------------------------------------------------


class GenStatus(StrEnum):
    """CLI result status."""

    ok = "ok"
    changed = "changed"
    error = "error"


GENERATED_BEGIN_PREFIX = "# >>> BEGIN GENERATED CUDA "
GENERATED_END_PREFIX = "# <<< END GENERATED CUDA "
GENERATED_MARKER_SUFFIX = " - DO NOT EDIT"
GENERATED_MARKER_BODY = (
    "# Source of truth: cuda_deps.toml.",
    "# Regenerate with: uv run --script tools/gen_cuda_deps.py cuda_deps.toml --pyproject pyproject.toml",
    "# Manual edits inside this block will be overwritten.",
)
GENERATED_MARKER_COMMENT_PREFIXES = (
    "# Generated extras in this block:",
    "# Generated uv section:",
    "# Generated uv sections:",
)


class GeneratedBlock(NamedTuple):
    """Line range that should be wrapped in a generated-code marker."""

    label: str
    start: int
    end: int
    detail: str


@dataclass(frozen=True)
class GeneratedBlocks:
    """Generated pyproject marker blocks computed from current line positions."""

    lines: Sequence[str]
    extras: Sequence[str]

    def __iter__(self) -> Iterator[GeneratedBlock]:
        runtime_start = _find_assignment(self.lines, self.extras[0])
        conflicts_start = _find_assignment(self.lines, "conflicts")
        yield GeneratedBlock(
            label="RUNTIME EXTRAS",
            start=runtime_start,
            end=_find_array_end(self.lines, _find_assignment(self.lines, self.extras[-1])),
            detail=f"# Generated extras in this block: {', '.join(self.extras)}.",
        )
        yield GeneratedBlock(
            label="UV CONFLICTS",
            start=conflicts_start,
            end=_find_array_end(self.lines, conflicts_start),
            detail="# Generated uv section: tool.uv.conflicts.",
        )
        yield GeneratedBlock(
            label="UV SOURCES AND INDEXES",
            start=_find_section(self.lines, "[tool.uv.sources]"),
            end=_last_generated_index_line(self.lines),
            detail="# Generated uv sections: tool.uv.sources and tool.uv.index.",
        )

    def reversed(self) -> list[GeneratedBlock]:
        return sorted(self, key=lambda block: block.start, reverse=True)


# ---------------------------------------------------------------------------
# cuda_deps.toml domain model
# ---------------------------------------------------------------------------


class SourceKind(StrEnum):
    """Symbolic source routes resolved from CUDA variant metadata."""

    pytorch = "pytorch"
    flashinfer = "flashinfer"


class StrictModel(BaseModel):
    """Pydantic base for validated TOML records."""

    model_config = ConfigDict(extra="forbid")


class SourceSpec(StrictModel):
    """One [tool.uv.sources] entry for a package."""

    index: str = Field(description="uv index name.")
    extra: str | None = Field(default=None, description="Optional dependency extra that activates the source.")
    marker: str | None = Field(default=None, description="PEP 508 marker that activates the source.")

    @model_validator(mode="after")
    def _validate_marker(self) -> Self:
        if self.marker is not None and not _has_template(self.marker):
            Marker(self.marker)
        return self


class IndexSpec(StrictModel):
    """One [[tool.uv.index]] entry."""

    name: str = Field(description="uv index name.")
    url: str = Field(description="Package index URL.")
    explicit: bool = Field(default=True, description="Whether uv should use this index only when requested.")


class DependencySpec(StrictModel):
    """Structured dependency record that renders to a PEP 508 requirement."""

    name: str = Field(description="Package name or name template.")
    version: str | None = Field(default=None, description="Exact version without the == prefix.")
    specifier: str | None = Field(default=None, description="Version specifier, e.g. >=2.0.0 or ==1.2.3.")
    local: str | None = Field(default=None, description="Local version suffix, with or without a leading +.")
    marker: str | None = Field(default=None, description="PEP 508 marker.")
    sys_platform: str | None = Field(default=None, description="Shortcut for sys_platform marker.")
    arch: str | None = Field(default=None, description="Shortcut for platform_machine marker.")
    source_kind: SourceKind | None = Field(default=None, description="Symbolic uv source route for this dependency.")
    index: str | None = Field(default=None, description="Literal uv index name or template for this dependency.")
    source_marker: str | None = Field(default=None, description="Marker for the generated uv source entry.")
    variants: list[str] | None = Field(
        default=None, description="CUDA variant extras that should include this dependency."
    )

    @model_validator(mode="after")
    def _validate_version_fields(self) -> Self:
        if self.version is not None and self.specifier is not None:
            raise ValueError("Use either version or specifier, not both")
        if self.local is not None and self.version is None:
            raise ValueError("local requires version")
        if self.source_kind is not None and self.index is not None:
            raise ValueError("Use either source_kind or index, not both")
        self._validate_packaging_fields()
        return self

    def _validate_packaging_fields(self) -> None:
        if not _has_template(self.name):
            canonicalize_name(self.name)
        if self.version is not None and not _has_template(self.version, self.local):
            Version(f"{self.version}{_local_suffix(self.local)}")
        if self.specifier is not None and not _has_template(self.specifier):
            SpecifierSet(self.specifier)
        for marker in (self.marker, self.source_marker):
            if marker is not None and not _has_template(marker):
                Marker(marker)

    def as_pepstr(self, renderer: "TemplateRenderer") -> str:
        requirement = self.versioned_requirement(renderer)
        marker = " and ".join(self.pep_markers(renderer))
        pepstr = f"{requirement}; {marker}" if marker else requirement
        self.validate_pepstr(pepstr, renderer)
        return pepstr

    def versioned_requirement(self, renderer: "TemplateRenderer") -> str:
        name = renderer.template(self.name)
        match self:
            case DependencySpec(version=str() as version, local=local):
                rendered_version = f"{renderer.template(version)}{_local_suffix(renderer.optional_template(local))}"
                Version(rendered_version)
                return f"{name}=={rendered_version}"
            case DependencySpec(specifier=str() as specifier):
                rendered_specifier = renderer.template(specifier)
                SpecifierSet(rendered_specifier)
                return f"{name}{rendered_specifier}"
        return name

    def pep_markers(self, renderer: "TemplateRenderer") -> Iterator[str]:
        for marker in (
            renderer.optional_template(self.marker),
            _platform_marker("sys_platform", self.sys_platform),
            _platform_marker("platform_machine", self.arch),
        ):
            if marker:
                yield marker

    def validate_pepstr(self, pepstr: str, renderer: "TemplateRenderer") -> None:
        for marker in self.pep_markers(renderer):
            Marker(marker)
        requirement = Requirement(pepstr)
        if canonicalize_name(requirement.name) != canonicalize_name(renderer.template(self.name)):
            raise ValueError(f"Rendered requirement name changed unexpectedly: {pepstr!r}")


DependencyEntry = str | DependencySpec


@dataclass(frozen=True)
class DependencyStack:
    """Ordered dependency layers that render as one stream."""

    groups: tuple[Sequence[DependencyEntry], ...]

    @classmethod
    def of(cls, *groups: Sequence[DependencyEntry]) -> Self:
        return cls(groups=groups)

    def __iter__(self) -> Iterator[DependencyEntry]:
        for group in self.groups:
            yield from group


class NvidiaCudaLibrarySpec(StrictModel):
    """NVIDIA CUDA package source routing metadata."""

    name: str = Field(description="NVIDIA CUDA library package stem without the nvidia- prefix.")
    nvidia_package_suffix: str | None = Field(
        default=None,
        description="Optional package suffix override. Defaults to the CUDA variant suffix.",
    )
    index: str | None = Field(
        default=None,
        description="Optional literal uv index name or template. Defaults to the variant PyTorch index.",
    )


NvidiaCudaLibraryEntry = str | NvidiaCudaLibrarySpec


class CpuExtra(StrictModel):
    """CPU optional dependency extra."""

    dependencies: list[DependencyEntry] = Field(description="Dependency lines for the cpu extra.")


class CudaIndexTemplates(StrictModel):
    """Default index templates for CUDA variants."""

    pytorch: IndexSpec = Field(description="Template for PyTorch CUDA indexes.")
    flashinfer: IndexSpec | None = Field(default=None, description="Template for FlashInfer CUDA indexes.")


class CudaVariant(StrictModel):
    """CUDA optional dependency extra and its package indexes."""

    cuda_package_suffix: str = Field(description="Suffix for nvidia-* packages, e.g. cu12.")
    nvidia_package_suffix: str | None = Field(
        default=None,
        description="Suffix appended to nvidia-* package names. Defaults to cuda_package_suffix; use empty string for unsuffixed packages.",
    )
    torch_local_version: str | None = Field(
        default=None,
        description="Local version suffix for torch packages. Defaults to the extra name.",
    )
    dependencies: list[DependencyEntry] = Field(
        default_factory=list,
        description="Additional variant-specific dependency templates.",
    )
    pytorch_index: IndexSpec | None = Field(default=None, description="Optional PyTorch index override.")
    flashinfer_index: IndexSpec | None = Field(default=None, description="Optional FlashInfer index override.")


class CudaDepsConfig(StrictModel):
    """Source-of-truth CUDA dependency matrix."""

    requires_torch: list[DependencyEntry] = Field(
        description="Dependencies shared by all runtime extras that require the Torch stack."
    )
    runtime_deps: list[DependencyEntry] = Field(
        description="Pipeline/runtime dependencies shared by all runtime extras.",
    )
    cuda_dependencies: list[DependencyEntry] = Field(description="Dependency templates shared by all CUDA extras.")
    generated_extras: list[str] = Field(description="Extras owned by this generator in pyproject.toml.")
    nvidia_cuda_libraries: list[NvidiaCudaLibraryEntry] = Field(
        description="NVIDIA CUDA libraries used for uv source routing."
    )
    conflict_extras: list[str] = Field(default_factory=lambda: ["cpu"], description="Non-CUDA conflicting extras.")
    cpu: CpuExtra = Field(description="CPU dependency extra.")
    indexes: list[IndexSpec] = Field(default_factory=list, description="Shared package indexes.")
    cuda_indexes: CudaIndexTemplates = Field(description="CUDA index templates.")
    sources: dict[str, dict[str, list[SourceSpec]]] = Field(
        default_factory=dict,
        description="Static source entries keyed by extra name, then package name.",
    )
    variants: dict[str, CudaVariant] = Field(description="CUDA variants keyed by extra name.")

    @model_validator(mode="after")
    def _validate_generated_extras(self) -> "CudaDepsConfig":
        missing = [extra for extra in ["cpu", *self.variants] if extra not in self.generated_extras]
        if missing:
            raise ValueError(f"generated_extras is missing generated extras: {', '.join(missing)}")
        return self


# ---------------------------------------------------------------------------
# Mid-level materializers and accumulators
# ---------------------------------------------------------------------------


class UvSourcesAccumulator:
    """Deduplicating accumulator for [tool.uv.sources]."""

    def __init__(self) -> None:
        self._sources: dict[str, list[SourceSpec]] = {}

    def add(self, package: str, spec: SourceSpec) -> None:
        bucket = self._sources.setdefault(package, [])
        if spec not in bucket:
            bucket.append(spec)

    def merge(self, package_sources: dict[str, list[SourceSpec]]) -> None:
        for package, specs in package_sources.items():
            for spec in specs:
                self.add(package, spec)

    def as_dict(self) -> dict[str, list[SourceSpec]]:
        return self._sources


class TemplateRenderer:
    """Base renderer for TOML template strings."""

    def __init__(self, context: dict[str, str]) -> None:
        self.context = context

    def optional_template(self, template: str | None) -> str | None:
        return self.template(template) if template is not None else None

    def template(self, template: str) -> str:
        return _render_template(template, self.context)


class RequirementRenderer(TemplateRenderer):
    """Render dependency entries into PEP 508 requirement strings."""

    def __init__(self, context: dict[str, str], extra: str | None = None) -> None:
        super().__init__(context)
        self.extra = extra

    def render_many(self, dependencies: Iterable[DependencyEntry]) -> list[str]:
        return [self.render(dependency) for dependency in dependencies if self.applies(dependency)]

    def render(self, dependency: DependencyEntry) -> str:
        match dependency:
            case str() as template:
                requirement = self.template(template)
                Requirement(requirement)
                return requirement
            case DependencySpec() as spec:
                return spec.as_pepstr(self)
        raise TypeError(f"Unsupported dependency entry {dependency!r}")

    def applies(self, dependency: DependencyEntry) -> bool:
        match dependency:
            case DependencySpec(variants=list() as variants):
                return self.extra in variants
            case DependencySpec():
                # Structured dependencies without variants apply to every generated extra.
                return True
            case _:
                return True


class CudaVariantContext(TemplateRenderer):
    """Template context and helper objects for one CUDA optional dependency extra."""

    def __init__(self, config: CudaDepsConfig, extra: str, variant: CudaVariant) -> None:
        self.config = config
        self.extra = extra
        self.variant = variant
        super().__init__(
            {
                "extra": extra,
                "cuda_package_suffix": variant.cuda_package_suffix,
                "nvidia_package_suffix": self.nvidia_package_suffix,
                "torch_local_version": variant.torch_local_version or extra,
            }
        )
        self.dependencies = _cuda_dependency_stack(config, variant)
        self.dependency_renderer = CudaVariantDependencyRenderer(self)
        self.uv_router = CudaVariantUvRouter(self)

    @property
    def nvidia_package_suffix(self) -> str:
        suffix = self.variant.nvidia_package_suffix
        if suffix is None:
            suffix = self.variant.cuda_package_suffix
        return _nvidia_package_suffix(suffix)


class CudaVariantDependencyRenderer:
    """Render dependency strings and package names for one CUDA variant."""

    def __init__(self, context: CudaVariantContext) -> None:
        self.context = context
        self.requirements = RequirementRenderer(context.context, extra=context.extra)

    def render_dependencies(self, dependencies: Iterable[DependencyEntry]) -> list[str]:
        return self.requirements.render_many(dependencies)

    def dependency_packages(self, dependencies: Iterable[DependencyEntry]) -> set[str]:
        packages = set()
        for dependency in dependencies:
            match dependency:
                case DependencySpec(name=name) as spec if self.requirements.applies(spec):
                    packages.add(self.context.template(name))
        return packages


class CudaVariantUvRouter:
    """Materialize uv sources and indexes for one CUDA variant."""

    def __init__(self, context: CudaVariantContext) -> None:
        self.context = context

    def add_dependency_sources(
        self,
        sources: UvSourcesAccumulator,
        dependencies: Iterable[DependencyEntry],
    ) -> None:
        for dependency in dependencies:
            match dependency:
                case DependencySpec() as spec if self.context.dependency_renderer.requirements.applies(spec):
                    self.add_dependency_source(sources, spec)

    def add_dependency_source(self, sources: UvSourcesAccumulator, dependency: DependencySpec) -> None:
        index = self.dependency_index(dependency)
        if index is None:
            return
        sources.add(
            self.context.template(dependency.name),
            SourceSpec(
                index=index,
                extra=self.context.extra,
                marker=self.context.optional_template(dependency.source_marker),
            ),
        )

    def add_nvidia_sources(self, sources: UvSourcesAccumulator, *, direct_packages: set[str]) -> None:
        for library in self.context.config.nvidia_cuda_libraries:
            package = self.nvidia_package_name(library)
            extra = self.context.extra if package in direct_packages else None
            sources.add(
                package,
                SourceSpec(index=self.nvidia_source_index(library), extra=extra),
            )

    def nvidia_package_name(self, library: NvidiaCudaLibraryEntry) -> str:
        match library:
            case str() as name:
                return f"nvidia-{name}{self.context.nvidia_package_suffix}"
            case NvidiaCudaLibrarySpec(name=name, nvidia_package_suffix=str() as suffix):
                return f"nvidia-{name}{_nvidia_package_suffix(suffix)}"
            case NvidiaCudaLibrarySpec(name=name):
                return f"nvidia-{name}{self.context.nvidia_package_suffix}"
        raise TypeError(f"Unsupported NVIDIA CUDA library entry {library!r}")

    def nvidia_source_index(self, library: NvidiaCudaLibraryEntry) -> str:
        match library:
            case NvidiaCudaLibrarySpec(index=str() as index):
                return self.context.template(index)
        return self.pytorch_index.name

    @property
    def pytorch_index(self) -> IndexSpec:
        return self.source_kind_index_spec(SourceKind.pytorch)

    @property
    def flashinfer_index(self) -> IndexSpec | None:
        return self.optional_source_kind_index_spec(SourceKind.flashinfer)

    def indexes(self) -> list[IndexSpec]:
        return [index for source_kind in SourceKind if (index := self.optional_source_kind_index_spec(source_kind))]

    def dependency_index(self, dependency: DependencySpec) -> str | None:
        match dependency:
            case DependencySpec(source_kind=SourceKind() as source_kind):
                return self.source_kind_index(source_kind)
            case DependencySpec(index=str() as index):
                return self.context.template(index)
        return None

    def source_kind_index(self, source_kind: SourceKind) -> str:
        return self.source_kind_index_spec(source_kind).name

    def source_kind_index_spec(self, source_kind: SourceKind) -> IndexSpec:
        index = self.optional_source_kind_index_spec(source_kind)
        if index is None:
            raise ValueError(f"CUDA source {source_kind.value!r} needs [cuda_indexes.{source_kind.value}]")
        return index

    def optional_source_kind_index_spec(self, source_kind: SourceKind) -> IndexSpec | None:
        variant_index = getattr(self.context.variant, f"{source_kind.value}_index", None)
        default_index = getattr(self.context.config.cuda_indexes, source_kind.value, None)
        index = variant_index or default_index
        return self.render_index(index) if index is not None else None

    def render_index(self, index: IndexSpec) -> IndexSpec:
        return IndexSpec(
            name=self.context.template(index.name),
            url=self.context.template(index.url),
            explicit=index.explicit,
        )


@dataclass(frozen=True)
class CudaVariantContexts:
    """Lazy iterable of CUDA variant materialization contexts."""

    config: CudaDepsConfig

    def __iter__(self) -> Iterator[CudaVariantContext]:
        for extra, variant in self.config.variants.items():
            yield CudaVariantContext(self.config, extra, variant)


# ---------------------------------------------------------------------------
# High-level API DTOs
# ---------------------------------------------------------------------------


class CudaPyprojectFragment(BaseModel):
    """Generated pyproject TOML fragment plus summary metadata."""

    text: str = Field(description="Generated pyproject.toml section text.")
    rendered_extra_names: list[str] = Field(description="Optional dependency extras rendered in this fragment.")
    managed_extra_names: list[str] = Field(
        description="Optional dependency extras owned by the generator in pyproject.toml."
    )
    indexes: list[str] = Field(description="Generated uv index names.")
    source_packages: list[str] = Field(description="Generated uv source package names.")


class CudaDepsCommandResult(BaseModel):
    """Structured result returned by the command workflow."""

    status: GenStatus = Field(description="Command status.")
    message: str = Field(description="Human-readable result.")
    config: str = Field(description="Input cuda_deps.toml path.")
    output: str | None = Field(default=None, description="Output path, when provided.")
    generated: CudaPyprojectFragment | None = Field(default=None, description="Generated content summary.")


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def load_cuda_deps_config(path: Path) -> CudaDepsConfig:
    """Load and validate a CUDA dependency matrix."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return CudaDepsConfig.model_validate(data)


def build_cuda_pyproject_fragment(config: CudaDepsConfig) -> CudaPyprojectFragment:
    """Build generated CUDA optional-dependency and uv sections."""
    rendered_extra_names = ["cpu", *config.variants]
    sources = _collect_uv_sources(config)
    indexes = _collect_uv_indexes(config)
    doc = _build_cuda_fragment_document(config, sources, indexes)
    text = tomlkit.dumps(doc)
    return CudaPyprojectFragment(
        text=text,
        rendered_extra_names=rendered_extra_names,
        managed_extra_names=config.generated_extras,
        indexes=[index.name for index in indexes],
        source_packages=list(sources),
    )


def run_generation_command(
    config_path: Path,
    output_path: Path | None,
    check: bool,
    pyproject_path: Path | None = None,
) -> CudaDepsCommandResult:
    """Load config, build the fragment, and route the requested output mode."""
    generated = build_cuda_pyproject_fragment(load_cuda_deps_config(config_path))
    if pyproject_path is not None:
        return _update_pyproject(config_path, pyproject_path, check, generated)
    if check:
        return _check_generated(config_path, output_path, generated)
    if output_path is None:
        return _stdout_result(config_path, generated)
    output_path.write_text(generated.text, encoding="utf-8")
    return CudaDepsCommandResult(
        status=GenStatus.ok,
        message=f"Generated CUDA dependency sections at {output_path}",
        config=str(config_path),
        output=str(output_path),
        generated=generated,
    )


def apply_cuda_fragment_to_pyproject(pyproject_text: str, generated: CudaPyprojectFragment) -> str:
    """Return pyproject.toml with generated CUDA sections spliced in."""
    pyproject = tomlkit.parse(pyproject_text)
    generated_doc = tomlkit.parse(generated.text)
    _update_optional_dependencies(pyproject, generated_doc, generated.managed_extra_names)
    _update_uv_sections(pyproject, generated_doc)
    return _mark_generated_sections(tomlkit.dumps(pyproject), generated.rendered_extra_names)


# ---------------------------------------------------------------------------
# Mid-level processing: fragment assembly
# ---------------------------------------------------------------------------


def _build_cuda_fragment_document(
    config: CudaDepsConfig,
    sources: dict[str, list[SourceSpec]],
    indexes: list[IndexSpec],
) -> TOMLDocument:
    doc = tomlkit.document()
    doc.add(tomlkit.comment("Generated by tools/gen_cuda_deps.py from cuda_deps.toml."))
    doc.add(tomlkit.comment("Copy these sections into pyproject.toml or write them with --output."))
    doc.add(tomlkit.nl())
    doc.add("project", _build_project_table(config))
    doc.add("tool", _build_tool_table(config, sources, indexes))
    return doc


def _build_project_table(config: CudaDepsConfig) -> Table:
    project = tomlkit.table()
    optional = tomlkit.table()
    optional.add("cpu", _string_array(_render_cpu_extra_dependencies(config)))
    for context in CudaVariantContexts(config):
        optional.add(
            context.extra,
            _string_array(context.dependency_renderer.render_dependencies(context.dependencies)),
        )
    project.add("optional-dependencies", optional)
    return project


def _build_tool_table(
    config: CudaDepsConfig,
    sources: dict[str, list[SourceSpec]],
    indexes: list[IndexSpec],
) -> Table:
    tool = tomlkit.table()
    uv = tomlkit.table()
    uv.add("conflicts", _conflicts_array([*config.conflict_extras, *config.variants]))
    uv.add("sources", _sources_table(sources))
    uv.add("index", _index_aot(indexes))
    tool.add("uv", uv)
    return tool


# ---------------------------------------------------------------------------
# Mid-level processing: pyproject splicing and generated markers
# ---------------------------------------------------------------------------


def _update_optional_dependencies(
    pyproject: TOMLDocument,
    generated: TOMLDocument,
    generated_extras: list[str],
) -> None:
    optional = _required_table(_required_table(pyproject, "project"), "optional-dependencies")
    generated_optional = _required_table(_required_table(generated, "project"), "optional-dependencies")
    for extra in list(optional):
        if str(extra) in generated_extras:
            del optional[extra]
    for extra, dependencies in generated_optional.items():
        optional[extra] = dependencies


def _update_uv_sections(pyproject: TOMLDocument, generated: TOMLDocument) -> None:
    uv = _required_table(_required_table(pyproject, "tool"), "uv")
    generated_uv = _required_table(_required_table(generated, "tool"), "uv")
    for key in ("conflicts", "sources", "index"):
        uv[key] = generated_uv[key]


def _mark_generated_sections(pyproject_text: str, extras: list[str]) -> str:
    lines = _strip_generated_markers(pyproject_text.splitlines())
    for block in GeneratedBlocks(lines, extras).reversed():
        _insert_generated_marker(lines, block)
    return "\n".join(lines) + "\n"


def _insert_generated_marker(lines: list[str], block: GeneratedBlock) -> None:
    start_marker = [
        f"{GENERATED_BEGIN_PREFIX}{block.label}{GENERATED_MARKER_SUFFIX} <<<",
        *GENERATED_MARKER_BODY,
        block.detail,
    ]
    lines[block.start : block.start] = start_marker
    end = block.end + len(start_marker)
    lines[end + 1 : end + 1] = [f"{GENERATED_END_PREFIX}{block.label}{GENERATED_MARKER_SUFFIX} >>>"]


def _strip_generated_markers(lines: list[str]) -> list[str]:
    stripped = []
    in_marker_header = False
    for line in lines:
        if line.startswith(GENERATED_BEGIN_PREFIX):
            in_marker_header = True
            continue
        if line.startswith(GENERATED_END_PREFIX):
            continue
        if in_marker_header and _is_generated_marker_comment(line):
            continue
        in_marker_header = False
        stripped.append(line)
    return stripped


def _is_generated_marker_comment(line: str) -> bool:
    if line in GENERATED_MARKER_BODY:
        return True
    return line.startswith(GENERATED_MARKER_COMMENT_PREFIXES)


def _find_assignment(lines: Sequence[str], key: str) -> int:
    assignment = f"{key} = ["
    for index, line in enumerate(lines):
        if line == assignment:
            return index
    raise ValueError(f"pyproject.toml: missing generated assignment {assignment!r}")


def _find_section(lines: Sequence[str], section: str) -> int:
    for index, line in enumerate(lines):
        if line == section:
            return index
    raise ValueError(f"pyproject.toml: missing generated section {section!r}")


def _last_generated_index_line(lines: Sequence[str]) -> int:
    index_start = _find_section(lines, "[[tool.uv.index]]")
    next_section = _find_next_non_index_section(lines, index_start + 1)
    return next_section - 1 if next_section is not None else len(lines) - 1


def _find_next_non_index_section(lines: Sequence[str], start_index: int) -> int | None:
    for index in range(start_index, len(lines)):
        if lines[index].startswith("[") and lines[index] != "[[tool.uv.index]]":
            return index
    return None


def _find_array_end(lines: Sequence[str], start_index: int) -> int:
    for index in range(start_index + 1, len(lines)):
        if lines[index] == "]":
            return index
    raise ValueError(f"pyproject.toml: missing closing bracket for generated assignment at line {start_index + 1}")


def _required_table(
    container: TOMLDocument | Table | OutOfOrderTableProxy, key: str
) -> TOMLDocument | Table | OutOfOrderTableProxy:
    value = container.get(key)
    if not isinstance(value, TOMLDocument | Table | OutOfOrderTableProxy):
        raise ValueError(f"pyproject.toml: missing [{key}] table")
    return value


# ---------------------------------------------------------------------------
# Mid-level processing: dependency, source, and index collection
# ---------------------------------------------------------------------------


def _render_cpu_extra_dependencies(config: CudaDepsConfig) -> list[str]:
    dependencies = DependencyStack.of(config.requires_torch, config.runtime_deps, config.cpu.dependencies)
    return RequirementRenderer({}).render_many(dependencies)


def _collect_uv_sources(config: CudaDepsConfig) -> dict[str, list[SourceSpec]]:
    sources = UvSourcesAccumulator()
    for package_sources in config.sources.values():
        sources.merge(package_sources)
    _add_cpu_dependency_sources(sources, config.cpu.dependencies)
    for context in CudaVariantContexts(config):
        context.uv_router.add_dependency_sources(sources, context.dependencies)
        context.uv_router.add_nvidia_sources(
            sources,
            direct_packages=context.dependency_renderer.dependency_packages(context.dependencies),
        )
    return sources.as_dict()


def _add_cpu_dependency_sources(sources: UvSourcesAccumulator, dependencies: Iterable[DependencyEntry]) -> None:
    renderer = RequirementRenderer({}, extra="cpu")
    for dependency in dependencies:
        match dependency:
            case DependencySpec(name=name, index=str() as index, source_marker=source_marker) as spec if (
                renderer.applies(spec)
            ):
                sources.add(
                    renderer.template(name),
                    SourceSpec(
                        index=renderer.template(index),
                        extra="cpu",
                        marker=renderer.optional_template(source_marker),
                    ),
                )


def _collect_uv_indexes(config: CudaDepsConfig) -> list[IndexSpec]:
    indexes: dict[str, IndexSpec] = {}
    for context in CudaVariantContexts(config):
        for index in context.uv_router.indexes():
            _add_index(indexes, index)
    for index in config.indexes:
        _add_index(indexes, index)
    return list(indexes.values())


def _add_index(indexes: dict[str, IndexSpec], index: IndexSpec) -> None:
    existing = indexes.get(index.name)
    if existing is not None and existing != index:
        raise ValueError(
            f"Conflicting uv index definition for {index.name!r}: "
            f"{existing.url!r} explicit={existing.explicit!r} != {index.url!r} explicit={index.explicit!r}"
        )
    indexes[index.name] = index


def _add_index(indexes: dict[str, IndexSpec], index: IndexSpec) -> None:
    existing = indexes.get(index.name)
    if existing is not None and existing != index:
        raise ValueError(
            f"Conflicting uv index definition for {index.name!r}: "
            f"{existing.url!r} explicit={existing.explicit!r} != {index.url!r} explicit={index.explicit!r}"
        )
    indexes[index.name] = index


def _cuda_dependency_stack(config: CudaDepsConfig, variant: CudaVariant) -> DependencyStack:
    return DependencyStack.of(
        config.requires_torch,
        config.runtime_deps,
        config.cuda_dependencies,
        variant.dependencies,
    )


# ---------------------------------------------------------------------------
# Low-level rendering and TOML helpers
# ---------------------------------------------------------------------------


def _local_suffix(local: str | None) -> str:
    match local:
        case None | "":
            return ""
        case str() if local.startswith("+"):
            return local
        case str():
            return f"+{local}"
    raise TypeError(f"Unsupported local version suffix {local!r}")


def _has_template(*values: str | None) -> bool:
    return any(value is not None and "{" in value for value in values)


def _nvidia_package_suffix(suffix: str) -> str:
    return f"-{suffix}" if suffix else ""


def _platform_marker(name: str, value: str | None) -> str | None:
    return f"{name} == '{value}'" if value else None


def _render_template(template: str, context: dict[str, str]) -> str:
    try:
        return template.format_map(context)
    except KeyError as exc:
        key = str(exc).strip("'")
        raise ValueError(f"Unknown template key {key!r} in dependency {template!r}") from exc


def _string_array(items: list[str]) -> Array:
    array = tomlkit.array()
    array.multiline(True)
    for item in items:
        array.append(item)
    return array


def _conflicts_array(extras: list[str]) -> Array:
    group = tomlkit.array()
    group.multiline(True)
    for extra in extras:
        table = tomlkit.inline_table()
        table.add("extra", extra)
        group.append(table)
    outer = tomlkit.array()
    outer.multiline(True)
    outer.append(group)
    return outer


def _sources_table(sources: dict[str, list[SourceSpec]]) -> Table:
    table = tomlkit.table()
    for package, specs in sources.items():
        table.add(package, _source_array(specs))
    return table


def _source_array(specs: list[SourceSpec]) -> Array:
    array = tomlkit.array()
    array.multiline(True)
    for spec in specs:
        table = tomlkit.inline_table()
        table.add("index", spec.index)
        if spec.extra is not None:
            table.add("extra", spec.extra)
        if spec.marker is not None:
            table.add("marker", spec.marker)
        array.append(table)
    return array


def _index_aot(indexes: list[IndexSpec]) -> AoT:
    aot = tomlkit.aot()
    for index in indexes:
        table = tomlkit.table()
        table.add("name", index.name)
        table.add("url", index.url)
        table.add("explicit", index.explicit)
        aot.append(table)
    return aot


# ---------------------------------------------------------------------------
# Command workflow helpers
# ---------------------------------------------------------------------------


def _check_generated(
    config_path: Path,
    output_path: Path | None,
    generated: CudaPyprojectFragment,
) -> CudaDepsCommandResult:
    if output_path is None:
        return CudaDepsCommandResult(
            status=GenStatus.error,
            message="--check requires --output",
            config=str(config_path),
            generated=generated,
        )
    if not output_path.exists():
        return CudaDepsCommandResult(
            status=GenStatus.changed,
            message=f"{output_path} does not exist; run again without --check to generate it",
            config=str(config_path),
            output=str(output_path),
            generated=generated,
        )
    current = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    status = GenStatus.ok if current == generated.text else GenStatus.changed
    message = "Generated CUDA dependency sections are up to date"
    if status is GenStatus.changed:
        message = f"Generated CUDA dependency sections differ from {output_path}"
    return CudaDepsCommandResult(
        status=status,
        message=message,
        config=str(config_path),
        output=str(output_path),
        generated=generated,
    )


def _update_pyproject(
    config_path: Path,
    pyproject_path: Path,
    check: bool,
    generated: CudaPyprojectFragment,
) -> CudaDepsCommandResult:
    current = pyproject_path.read_text(encoding="utf-8")
    updated = apply_cuda_fragment_to_pyproject(current, generated)
    if check:
        return _check_pyproject(config_path, pyproject_path, generated, current, updated)
    pyproject_path.write_text(updated, encoding="utf-8")
    return CudaDepsCommandResult(
        status=GenStatus.ok,
        message=f"Updated generated CUDA dependency sections in {pyproject_path}",
        config=str(config_path),
        output=str(pyproject_path),
        generated=generated,
    )


def _check_pyproject(
    config_path: Path,
    pyproject_path: Path,
    generated: CudaPyprojectFragment,
    current: str,
    updated: str,
) -> CudaDepsCommandResult:
    status = GenStatus.ok if current == updated else GenStatus.changed
    message = "Generated CUDA dependency sections in pyproject.toml are up to date"
    if status is GenStatus.changed:
        message = f"Generated CUDA dependency sections differ from {pyproject_path}"
    return CudaDepsCommandResult(
        status=status,
        message=message,
        config=str(config_path),
        output=str(pyproject_path),
        generated=generated,
    )


def _stdout_result(config_path: Path, generated: CudaPyprojectFragment) -> CudaDepsCommandResult:
    return CudaDepsCommandResult(
        status=GenStatus.ok,
        message="Generated CUDA dependency sections",
        config=str(config_path),
        generated=generated,
    )


# ---------------------------------------------------------------------------
# Command entrypoint and presentation
# ---------------------------------------------------------------------------


def _configure_logging(log_format: Literal["plain", "json"]) -> None:
    renderer = structlog.processors.JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer()
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        processors=[renderer],
    )


def _emit_result(result: CudaDepsCommandResult, json_output: bool) -> None:
    if json_output:
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
        return
    if result.output is None and result.generated is not None and result.status is GenStatus.ok:
        sys.stdout.write(result.generated.text)
        return
    logger.info(result.message)
    return


def _exit_code(result: CudaDepsCommandResult) -> int:
    if result.status is GenStatus.ok:
        return 0
    if result.status is GenStatus.changed:
        return 1
    return 125


@click.command(help=__doc__)
@click.argument("config", required=False, type=click.Path(path_type=Path), default=Path("cuda_deps.toml"))
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write generated TOML sections here.")
@click.option(
    "--pyproject", type=click.Path(path_type=Path), default=None, help="Update generated sections in pyproject.toml."
)
@click.option("--check", is_flag=True, help="Exit non-zero if generated output would change.")
@click.option("--json", "json_output", is_flag=True, help="Emit a JSON command result.")
@click.option("--log-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def _cli(
    config: Path,
    output: Path | None,
    pyproject: Path | None,
    check: bool,
    json_output: bool,
    log_format: Literal["plain", "json"],
) -> None:
    """Generate CUDA dependency TOML sections."""
    _configure_logging(log_format)
    try:
        if output is not None and pyproject is not None:
            raise ValueError("Use either --output or --pyproject, not both")
        result = run_generation_command(config, output, check, pyproject)
    except (OSError, ValueError, ValidationError, tomllib.TOMLDecodeError) as exc:
        logger.error("Generation failed", error=str(exc))
        raise click.exceptions.Exit(125) from exc
    _emit_result(result, json_output)
    raise click.exceptions.Exit(_exit_code(result))


def _run_cli() -> None:
    _cli()


if __name__ == "__main__":
    _run_cli()
