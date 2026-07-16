#!/usr/bin/env -S uv run --frozen
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate CPU and CUDA dependency metadata in pyproject.toml."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, Self

import click
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

# ---------------------------------------------------------------------------
# Generated pyproject marker model
# ---------------------------------------------------------------------------


class GenStatus(StrEnum):
    """CLI result status."""

    ok = "ok"
    changed = "changed"


GENERATED_BEGIN_PREFIX = "# >>> BEGIN GENERATED CUDA "
GENERATED_END_PREFIX = "# <<< END GENERATED CUDA "
GENERATED_MARKER_SUFFIX = " - DO NOT EDIT"
GENERATED_MARKER_BODY = (
    "# Source of truth: cuda_deps.toml.",
    "# Regenerate with: uv run --frozen tools/gen_cuda_deps.py cuda_deps.toml --pyproject pyproject.toml",
    "# Manual edits inside this block will be overwritten.",
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
        optional_dependencies = "[project.optional-dependencies]"
        tool_uv = "[tool.uv]"
        runtime_start = self._find_assignment(self.extras[0], optional_dependencies)
        conflicts_start = self._find_assignment("conflicts", tool_uv)
        yield GeneratedBlock(
            label="RUNTIME EXTRAS",
            start=runtime_start,
            end=self._find_array_end(self._find_assignment(self.extras[-1], optional_dependencies)),
            detail=f"# Generated extras in this block: {', '.join(self.extras)}.",
        )
        yield GeneratedBlock(
            label="UV CONFLICTS",
            start=conflicts_start,
            end=self._find_array_end(conflicts_start),
            detail="# Generated uv section: tool.uv.conflicts.",
        )
        yield GeneratedBlock(
            label="UV SOURCES AND INDEXES",
            start=self._find_section("[tool.uv.sources]"),
            end=self._last_generated_index_line(),
            detail="# Generated uv sections: tool.uv.sources and tool.uv.index.",
        )

    def reversed(self) -> list[GeneratedBlock]:
        return sorted(self, key=lambda block: block.start, reverse=True)

    def _find_assignment(self, key: str, section: str) -> int:
        assignment = f"{key} = ["
        section_start = self._find_section(section)
        for index in range(section_start + 1, len(self.lines)):
            line = self.lines[index]
            if line.startswith("["):
                break
            if line == assignment:
                return index
        raise ValueError(f"pyproject.toml: missing generated assignment {assignment!r} in {section}")

    def _find_section(self, section: str) -> int:
        for index, line in enumerate(self.lines):
            if line == section:
                return index
        raise ValueError(f"pyproject.toml: missing generated section {section!r}")

    def _last_generated_index_line(self) -> int:
        index_start = self._find_section("[[tool.uv.index]]")
        next_section = self._find_next_non_index_section(index_start + 1)
        return next_section - 1 if next_section is not None else len(self.lines) - 1

    def _find_next_non_index_section(self, start_index: int) -> int | None:
        for index in range(start_index, len(self.lines)):
            if self.lines[index].startswith("[") and self.lines[index] != "[[tool.uv.index]]":
                return index
        return None

    def _find_array_end(self, start_index: int) -> int:
        for index in range(start_index + 1, len(self.lines)):
            if self.lines[index] == "]":
                return index
        raise ValueError(f"pyproject.toml: missing closing bracket for generated assignment at line {start_index + 1}")


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
    source_marker: str | None = Field(
        default=None,
        description=(
            "Marker for the generated uv source entry. Defaults to the requirement's own "
            "marker/sys_platform/arch; set this only when the source route needs a condition "
            "that differs from the requirement itself."
        ),
    )
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
        name = renderer.template(self.name)
        markers = list(self.pep_markers(renderer))
        for marker in markers:
            Marker(marker)
        requirement = self._versioned_requirement(name, renderer)
        pepstr = f"{requirement}; {' and '.join(markers)}" if markers else requirement
        self._validate_pepstr(pepstr, name)
        return pepstr

    def pep_markers(self, renderer: "TemplateRenderer") -> Iterator[str]:
        for marker in (
            renderer.optional_template(self.marker),
            _platform_marker("sys_platform", self.sys_platform),
            _platform_marker("platform_machine", self.arch),
        ):
            if marker:
                yield marker

    def effective_source_marker(self, renderer: "TemplateRenderer") -> str | None:
        """Marker for the generated uv source entry: an explicit override, else the requirement's own markers."""
        if self.source_marker is not None:
            marker = renderer.template(self.source_marker)
            Marker(marker)
            return marker
        markers = list(self.pep_markers(renderer))
        return " and ".join(markers) if markers else None

    def _versioned_requirement(self, name: str, renderer: "TemplateRenderer") -> str:
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

    def _validate_pepstr(self, pepstr: str, name: str) -> None:
        requirement = Requirement(pepstr)
        if canonicalize_name(requirement.name) != canonicalize_name(name):
            raise ValueError(f"Rendered requirement name changed unexpectedly: {pepstr!r}")


DependencyEntry = str | DependencySpec


class NvidiaCudaLibrarySpec(StrictModel):
    """NVIDIA CUDA package source routing metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

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
    nvidia_index: str | None = Field(
        default=None,
        description=(
            "Optional default index name for this variant's NVIDIA CUDA library packages. "
            "References an already-declared [[indexes]] entry rather than defining a new one, "
            "unlike pytorch_index/flashinfer_index. A per-library NvidiaCudaLibrarySpec.index "
            "override always wins over this; if neither is set, the variant's pytorch index "
            "is used instead."
        ),
    )


class CudaDepsConfig(StrictModel):
    """Source-of-truth CUDA dependency matrix."""

    base_runtime_deps: list[DependencyEntry] = Field(
        description="Pipeline/runtime dependencies shared by all runtime extras without implying Torch wheels.",
    )
    torch_runtime_deps: list[DependencyEntry] = Field(
        description="Torch-adjacent runtime dependencies shared by CPU and CUDA extras."
    )
    cuda_runtime_deps: list[DependencyEntry] = Field(
        description="CUDA-only runtime dependency templates shared by all CUDA extras."
    )
    torch_wheel_deps: list[DependencyEntry] = Field(
        description="Torch-family dependency templates whose wheels follow the selected CUDA extra."
    )
    managed_extras: list[str] = Field(description="Extras owned by this generator in pyproject.toml.")
    nvidia_cuda_libraries: list[NvidiaCudaLibraryEntry] = Field(
        description="NVIDIA CUDA libraries used for uv source routing."
    )
    conflict_extras: list[str] = Field(default_factory=lambda: ["cpu"], description="Non-CUDA conflicting extras.")
    cpu: CpuExtra = Field(description="CPU dependency extra.")
    indexes: list[IndexSpec] = Field(default_factory=list, description="Shared package indexes.")
    cuda_indexes: CudaIndexTemplates = Field(description="CUDA index templates.")
    sources: dict[str, dict[str, list[SourceSpec]]] = Field(
        default_factory=dict,
        description="Static source entries keyed by owning extra, then package name.",
    )
    variants: dict[str, CudaVariant] = Field(description="CUDA variants keyed by extra name.")

    @model_validator(mode="after")
    def _validate_managed_extras(self) -> "CudaDepsConfig":
        missing = [extra for extra in ["cpu", *self.variants] if extra not in self.managed_extras]
        if missing:
            raise ValueError(f"managed_extras is missing generated extras: {', '.join(missing)}")
        self._validate_static_source_extras()
        return self

    def _validate_static_source_extras(self) -> None:
        known_extras = {"cpu", *self.variants}
        unknown_extras = set(self.sources) - known_extras
        if unknown_extras:
            raise ValueError(f"sources has unknown extras: {', '.join(sorted(unknown_extras))}")
        for extra, package_sources in self.sources.items():
            for specs in package_sources.values():
                for spec in specs:
                    if spec.extra is not None and spec.extra != extra:
                        raise ValueError(f"source extra {spec.extra!r} does not match [sources.{extra}]")


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

    def as_dict(self) -> dict[str, list[SourceSpec]]:
        return self._sources


class TemplateRenderer:
    """Base renderer for TOML template strings."""

    def __init__(self, context: dict[str, str]) -> None:
        self.context = context

    def optional_template(self, template: str | None) -> str | None:
        return self.template(template) if template is not None else None

    def template(self, template: str) -> str:
        try:
            return template.format_map(self.context)
        except KeyError as exc:
            key = str(exc).strip("'")
            raise ValueError(f"Unknown template key {key!r} in dependency {template!r}") from exc


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
        self.dependencies = _cuda_dependencies(config, variant)
        self.requirements = RequirementRenderer(self.context, extra=extra)
        self.uv_router = CudaVariantUvRouter(self)

    @property
    def nvidia_package_suffix(self) -> str:
        suffix = self.variant.nvidia_package_suffix
        if suffix is None:
            suffix = self.variant.cuda_package_suffix
        return _nvidia_package_suffix(suffix)

    def dependency_index(self, dependency: DependencySpec) -> str | None:
        match dependency:
            case DependencySpec(source_kind=SourceKind() as source_kind):
                return self.source_kind_index(source_kind)
            case DependencySpec(index=str() as index):
                return self.template(index)
        return None

    def nvidia_sources(self) -> Iterator[tuple[str, str]]:
        for library in self.config.nvidia_cuda_libraries:
            yield self.nvidia_package_name(library), self.nvidia_source_index(library)

    def nvidia_package_name(self, library: NvidiaCudaLibraryEntry) -> str:
        match library:
            case str() as name:
                return f"nvidia-{name}{self.nvidia_package_suffix}"
            case NvidiaCudaLibrarySpec(name=name, nvidia_package_suffix=str() as suffix):
                return f"nvidia-{name}{_nvidia_package_suffix(suffix)}"
            case NvidiaCudaLibrarySpec(name=name):
                return f"nvidia-{name}{self.nvidia_package_suffix}"
        raise TypeError(f"Unsupported NVIDIA CUDA library entry {library!r}")

    def nvidia_source_index(self, library: NvidiaCudaLibraryEntry) -> str:
        match library:
            case NvidiaCudaLibrarySpec(index=str() as index):
                return self.template(index)
        if self.variant.nvidia_index is not None:
            return self.template(self.variant.nvidia_index)
        return self.pytorch_index.name

    @property
    def pytorch_index(self) -> IndexSpec:
        return self.source_kind_index_spec(SourceKind.pytorch)

    @property
    def flashinfer_index(self) -> IndexSpec | None:
        return self.optional_source_kind_index_spec(SourceKind.flashinfer)

    def indexes(self) -> list[IndexSpec]:
        return [index for source_kind in SourceKind if (index := self.optional_source_kind_index_spec(source_kind))]

    def source_kind_index(self, source_kind: SourceKind) -> str:
        return self.source_kind_index_spec(source_kind).name

    def source_kind_index_spec(self, source_kind: SourceKind) -> IndexSpec:
        index = self.optional_source_kind_index_spec(source_kind)
        if index is None:
            raise ValueError(f"CUDA source {source_kind.value!r} needs [cuda_indexes.{source_kind.value}]")
        return index

    def optional_source_kind_index_spec(self, source_kind: SourceKind) -> IndexSpec | None:
        variant_index = getattr(self.variant, f"{source_kind.value}_index", None)
        default_index = getattr(self.config.cuda_indexes, source_kind.value, None)
        index = variant_index or default_index
        return self.render_index(index) if index is not None else None

    def render_index(self, index: IndexSpec) -> IndexSpec:
        return IndexSpec(
            name=self.template(index.name),
            url=self.template(index.url),
            explicit=index.explicit,
        )


class CudaVariantUvRouter:
    """Populate a uv sources accumulator with one CUDA variant's dependency and NVIDIA library sources."""

    def __init__(self, context: CudaVariantContext) -> None:
        self.context = context

    def add_dependency_sources(
        self,
        sources: UvSourcesAccumulator,
        dependencies: Iterable[DependencyEntry],
    ) -> None:
        for dependency in dependencies:
            match dependency:
                case DependencySpec() as spec if self.context.requirements.applies(spec):
                    self.add_dependency_source(sources, spec)

    def add_dependency_source(self, sources: UvSourcesAccumulator, dependency: DependencySpec) -> None:
        index = self.context.dependency_index(dependency)
        if index is None:
            return
        sources.add(
            self.context.template(dependency.name),
            SourceSpec(
                index=index,
                extra=self.context.extra,
                marker=dependency.effective_source_marker(self.context),
            ),
        )

    def add_nvidia_sources(self, sources: UvSourcesAccumulator) -> None:
        for package, index in self.context.nvidia_sources():
            sources.add(package, SourceSpec(index=index))


def _cuda_variant_contexts(config: CudaDepsConfig) -> Iterator[CudaVariantContext]:
    for extra, variant in config.variants.items():
        yield CudaVariantContext(config, extra, variant)


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


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of updating or checking generated metadata."""

    status: GenStatus
    message: str


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
    _validate_source_indexes(sources, indexes)
    doc = _build_cuda_fragment_document(config, sources, indexes)
    text = tomlkit.dumps(doc)
    return CudaPyprojectFragment(
        text=text,
        rendered_extra_names=rendered_extra_names,
        managed_extra_names=config.managed_extras,
    )


def run_generation_command(config_path: Path, pyproject_path: Path, check: bool) -> GenerationResult:
    """Load config and update or check generated pyproject.toml metadata."""
    generated = build_cuda_pyproject_fragment(load_cuda_deps_config(config_path))
    return _update_pyproject(pyproject_path, check, generated)


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
    doc.add(tomlkit.comment("The complete tool.uv.sources and tool.uv.index sections are generated here."))
    doc.add(tomlkit.nl())
    doc.add("project", _build_project_table(config))
    doc.add("tool", _build_tool_table(config, sources, indexes))
    return doc


def _build_project_table(config: CudaDepsConfig) -> Table:
    project = tomlkit.table()
    optional = tomlkit.table()
    optional.add("cpu", _string_array(_render_cpu_extra_dependencies(config)))
    for context in _cuda_variant_contexts(config):
        optional.add(
            context.extra,
            _string_array(context.requirements.render_many(context.dependencies)),
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
    managed_extras: list[str],
) -> None:
    optional = _required_table(_required_table(pyproject, "project"), "optional-dependencies")
    generated_optional = _required_table(_required_table(generated, "project"), "optional-dependencies")
    for extra in list(optional):
        if str(extra) in managed_extras:
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
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(GENERATED_BEGIN_PREFIX):
            index += 1
            while index < len(lines) and lines[index].startswith("#"):
                index += 1
            continue
        if line.startswith(GENERATED_END_PREFIX):
            index += 1
            continue
        stripped.append(line)
        index += 1
    return stripped


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
    dependencies = (*config.base_runtime_deps, *config.torch_runtime_deps, *config.cpu.dependencies)
    return RequirementRenderer({}).render_many(dependencies)


def _collect_uv_sources(config: CudaDepsConfig) -> dict[str, list[SourceSpec]]:
    sources = UvSourcesAccumulator()
    _add_static_sources(sources, config.sources)
    _add_cpu_dependency_sources(sources, config.cpu.dependencies)
    for context in _cuda_variant_contexts(config):
        context.uv_router.add_dependency_sources(sources, context.dependencies)
        context.uv_router.add_nvidia_sources(sources)
    return sources.as_dict()


def _add_static_sources(
    sources: UvSourcesAccumulator,
    source_groups: dict[str, dict[str, list[SourceSpec]]],
) -> None:
    for extra, package_sources in source_groups.items():
        for package, specs in package_sources.items():
            for spec in specs:
                sources.add(package, spec.model_copy(update={"extra": extra}))


def _add_cpu_dependency_sources(sources: UvSourcesAccumulator, dependencies: Iterable[DependencyEntry]) -> None:
    renderer = RequirementRenderer({}, extra="cpu")
    for dependency in dependencies:
        match dependency:
            case DependencySpec(index=str() as index) as spec if renderer.applies(spec):
                sources.add(
                    renderer.template(spec.name),
                    SourceSpec(
                        index=renderer.template(index),
                        extra="cpu",
                        marker=spec.effective_source_marker(renderer),
                    ),
                )


def _collect_uv_indexes(config: CudaDepsConfig) -> list[IndexSpec]:
    indexes: dict[str, IndexSpec] = {}
    for context in _cuda_variant_contexts(config):
        for index in context.indexes():
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


def _validate_source_indexes(sources: dict[str, list[SourceSpec]], indexes: Iterable[IndexSpec]) -> None:
    known_indexes = {index.name for index in indexes}
    unknown_indexes = {spec.index for specs in sources.values() for spec in specs if spec.index not in known_indexes}
    if unknown_indexes:
        raise ValueError(f"uv sources reference unknown indexes: {', '.join(sorted(unknown_indexes))}")


def _cuda_dependencies(config: CudaDepsConfig, variant: CudaVariant) -> tuple[DependencyEntry, ...]:
    return (
        *config.base_runtime_deps,
        *config.torch_runtime_deps,
        *config.cuda_runtime_deps,
        *config.torch_wheel_deps,
        *variant.dependencies,
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


def _string_array(items: list[str]) -> Array:
    array = tomlkit.array()
    array.multiline(True)
    for item in items:
        array.append(item)
    return array


def _conflicts_array(extras: list[str]) -> Array:
    group = tomlkit.array()
    group.multiline(True)
    group.indent(4)
    for extra in extras:
        table = tomlkit.inline_table()
        table.append(None, tomlkit.ws(" "))
        table.add("extra", extra)
        table.append(None, tomlkit.ws(" "))
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
        table.append(None, tomlkit.ws(" "))
        table.add("index", spec.index)
        if spec.extra is not None:
            table.add("extra", spec.extra)
        if spec.marker is not None:
            table.add("marker", spec.marker)
        table.append(None, tomlkit.ws(" "))
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
# Command workflow
# ---------------------------------------------------------------------------


def _update_pyproject(
    pyproject_path: Path,
    check: bool,
    generated: CudaPyprojectFragment,
) -> GenerationResult:
    current = pyproject_path.read_text(encoding="utf-8")
    updated = apply_cuda_fragment_to_pyproject(current, generated)
    if check or current == updated:
        return _check_pyproject(pyproject_path, current, updated)
    pyproject_path.write_text(updated, encoding="utf-8")
    return GenerationResult(
        status=GenStatus.ok,
        message=f"Updated generated CUDA dependency sections in {pyproject_path}",
    )


def _check_pyproject(pyproject_path: Path, current: str, updated: str) -> GenerationResult:
    status = GenStatus.ok if current == updated else GenStatus.changed
    message = "Generated CUDA dependency sections in pyproject.toml are up to date"
    if status is GenStatus.changed:
        message = f"Generated CUDA dependency sections differ from {pyproject_path}"
    return GenerationResult(
        status=status,
        message=message,
    )


# ---------------------------------------------------------------------------
# Command entrypoint
# ---------------------------------------------------------------------------


def _exit_code(result: GenerationResult) -> int:
    match result.status:
        case GenStatus.ok:
            return 0
        case GenStatus.changed:
            return 1
    raise AssertionError(f"Unsupported generation status: {result.status}")


@click.command(help="Update generated CPU and CUDA dependency metadata in pyproject.toml.")
@click.argument(
    "config",
    required=False,
    # click-stubs omits pathlib.Path from this runtime-supported argument.
    type=click.Path(path_type=Path),  # ty: ignore[invalid-argument-type]
    default=Path("cuda_deps.toml"),
)
@click.option(
    "--pyproject",
    # click-stubs omits pathlib.Path from this runtime-supported argument.
    type=click.Path(path_type=Path),  # ty: ignore[invalid-argument-type]
    default=Path("pyproject.toml"),
    show_default=True,
    help="pyproject.toml file to update or check.",
)
@click.option("--check", is_flag=True, help="Exit non-zero if generated output would change.")
def _cli(
    config: Path,
    pyproject: Path,
    check: bool,
) -> None:
    """Update generated CPU and CUDA dependency metadata."""
    try:
        result = run_generation_command(config, pyproject, check)
    except (OSError, ValueError, ValidationError, tomllib.TOMLDecodeError) as exc:
        click.echo(f"Generation failed: {exc}", err=True)
        raise click.exceptions.Exit(125) from exc
    click.echo(result.message, err=result.status is not GenStatus.ok)
    raise click.exceptions.Exit(_exit_code(result))


def _run_cli() -> None:
    _cli()


if __name__ == "__main__":
    _run_cli()
