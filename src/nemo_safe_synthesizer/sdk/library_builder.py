# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable pipeline for Safe Synthesizer."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset

from ..cli.artifact_structure import Workdir
from ..config import (
    SafeSynthesizerParameters,
)
from ..config.autoconfig import AutoConfigResolver
from ..config.unknown_fields import UnknownFieldBehavior, normalize_unknown_fields
from ..configurator.parameters import Parameters
from ..errors import ParameterError
from ..evaluation.evaluator import Evaluator
from ..generation.timeseries_backend import TimeseriesBackend
from ..generation.vllm_backend import VllmBackend
from ..holdout.holdout import Holdout
from ..llm.metadata import ModelMetadata
from ..llm.utils import get_device_name
from ..observability import LogCategory, configure_logging_from_workdir, get_logger, initialize_observability, traced
from ..package_info import __version__
from ..preflight import PreflightReport, PreflightStage, run_preflight
from ..results import SafeSynthesizerResults, make_nss_results
from ..telemetry import (
    DeploymentTypeEnum,
    NSSTrainingAndGenerationEvent,
    TaskStatusEnum,
    TelemetryHandler,
    _deployment_type,
    _telemetry_enabled,
    bucket_columns,
    bucket_records,
    sanitize_model_for_telemetry,
)
from ..training.huggingface_backend import HuggingFaceBackend
from .config_builder import ConfigBuilder

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..generation.backend import GeneratorBackend
    from ..training.backend import TrainingBackend


def _direct_parameter_values(params: Parameters) -> Iterator[tuple[str, object]]:
    for item in params._iter_parameters(recursive=False):
        yield next(iter(item.items()))


def _default_drift_paths(saved_model: Parameters, default_model: Parameters, prefix: str = "") -> list[str]:
    paths: list[str] = []
    default_values = dict(_direct_parameter_values(default_model))
    for field_name, saved_value in _direct_parameter_values(saved_model):
        default_value = default_values[field_name]
        field_path = f"{prefix}.{field_name}" if prefix else field_name

        saved_attr = saved_model.__dict__[field_name]
        default_attr = default_model.__dict__[field_name]
        if isinstance(saved_attr, Parameters) and isinstance(default_attr, Parameters):
            paths.extend(_default_drift_paths(saved_attr, default_attr, field_path))
        elif saved_value != default_value:
            paths.append(field_path)
    return paths


def _warn_for_saved_default_drift(saved_config: SafeSynthesizerParameters) -> None:
    """Warn when saved materialized values differ from current defaults."""
    current_defaults = SafeSynthesizerParameters()
    for path in _default_drift_paths(saved_config, current_defaults):
        logger.user.warning(
            f"Saved run config value at {path} is non-default; preserving saved value.",
            extra={"config_path": path},
        )


def _build_telemetry_event(ss: SafeSynthesizer, status: TaskStatusEnum) -> NSSTrainingAndGenerationEvent:
    """Build a telemetry event from the current pipeline state."""
    cfg = ss._nss_config

    duration = time.monotonic() - ss._total_start if ss._total_start is not None else -1.0

    num_records = -1
    sqs = -1.0
    dps = -1.0
    if hasattr(ss, "results") and ss.results is not None:
        summary = ss.results.summary
        if summary.num_valid_records is not None:
            num_records = summary.num_valid_records
        if summary.synthetic_data_quality_score is not None:
            sqs = summary.synthetic_data_quality_score
        if summary.data_privacy_score is not None:
            dps = summary.data_privacy_score

    replace_pii = cfg is not None and cfg.replace_pii is not None
    dp_enabled = cfg is not None and cfg.privacy is not None and cfg.privacy.dp_enabled
    ts_enabled = cfg is not None and cfg.time_series.is_timeseries
    group_by = cfg is not None and cfg.data.group_training_examples_by is not None
    model = sanitize_model_for_telemetry(cfg.training.pretrained_model if cfg is not None else None)

    records_bucket = "undefined"
    columns_bucket = "undefined"
    if isinstance(ss._data_source, pd.DataFrame):
        records_bucket = bucket_records(len(ss._data_source))
        columns_bucket = bucket_columns(len(ss._data_source.columns))

    gpu = get_device_name()

    return NSSTrainingAndGenerationEvent(
        task="run",
        task_status=status,
        deployment_type=ss._deployment_type,
        job_duration_sec=duration,
        num_records_generated=num_records,
        replace_pii_enabled=replace_pii,
        differential_privacy_enabled=dp_enabled,
        time_series_enabled=ts_enabled,
        group_by_enabled=group_by,
        input_records_bucket=records_bucket,
        input_columns_bucket=columns_bucket,
        synthetic_quality_score=sqs,
        data_privacy_score=dps,
        model=model,
        gpu=gpu,
    )


def _emit_nss_telemetry(ss: SafeSynthesizer, status: TaskStatusEnum) -> None:
    """Enqueue and immediately flush a single telemetry event. Never raises."""
    try:
        if not ss._emit_telemetry:
            return
        event = _build_telemetry_event(ss, status)
        handler = TelemetryHandler(source_client_version=__version__)
        handler.enqueue(event)
        handler.stop()  # Flushes the queue and sends
    except Exception:  # noqa: BLE001
        pass  # Telemetry is best-effort; never disrupt the pipeline


class SafeSynthesizer(ConfigBuilder):
    """Fluent builder and runner for Safe Synthesizer workflows.

    Extends ``ConfigBuilder`` with artifact management and stepwise
    pipeline execution.  Run all at once via ``run()``, or step by
    step::

        builder = SafeSynthesizer().with_data_source(df)
        builder.process_data().train().generate().evaluate()
        builder.save_results()
        results = builder.results

    ``train()`` uses ``HuggingFaceBackend``. ``generate()`` chooses
    ``TimeseriesBackend`` when ``config.time_series.is_timeseries`` is true and
    ``VllmBackend`` otherwise. Stepwise callers must call ``save_results()``
    themselves after ``evaluate()``; ``run()`` does this automatically.

    Args:
        config: Optional pre-built parameters that seed every
            config section.
        workdir: Explicit artifact directory layout.  When ``None``
            a default ``Workdir`` is created under ``save_path``.
        save_path: Root directory for artifacts when ``workdir``
            is not provided.  Defaults to
            ``"safe-synthesizer-artifacts"``.

    Example::

        builder = (
            SafeSynthesizer()
            .with_data_source(df)
            .with_replace_pii()
            .with_train(learning_rate=0.0001)
            .with_generate(num_records=10000)
        )
        builder.run()
        results = builder.results
    """

    _workdir: Workdir | None
    """Artifact directory layout, always set to a ``Workdir`` instance after ``__init__``."""

    trainer: TrainingBackend
    """Training backend instance, populated after ``train()``."""

    generator: GeneratorBackend
    """Generation backend instance, populated after ``generate()``."""

    evaluator: Evaluator | None
    """Evaluator instance, populated after ``evaluate()`` when evaluation is enabled."""

    results: SafeSynthesizerResults
    """Final pipeline results, populated after ``evaluate()`` or ``run()``."""

    _emit_telemetry: bool
    _deployment_type: DeploymentTypeEnum

    def __init__(
        self,
        config: SafeSynthesizerParameters | None = None,
        workdir: Workdir | None = None,
        save_path: Path | str | None = None,
        emit_telemetry: bool | None = None,
        deployment_type: DeploymentTypeEnum | None = None,
        unknown_fields: UnknownFieldBehavior | None = None,
    ):
        super().__init__(config=config, unknown_fields=unknown_fields)
        self._workdir = workdir
        if self._workdir is None:
            # Create a default workdir when none provided
            # Use "default" for config_name and "data" for dataset_name as fallbacks
            self._workdir = Workdir(
                base_path=Path(save_path) if save_path else Path("safe-synthesizer-artifacts"),
                config_name="default",
                dataset_name="data",
            )
        # Initialize state for pipeline stages
        self._training_df: pd.DataFrame | None = (
            None  # The active training df that might go through transformation, eg. pii replacement
        )
        self._original_training_df: pd.DataFrame | None = (
            None  # The original training df that we save for evaluation at the end
        )
        self._test_df: pd.DataFrame | None = None
        self._column_statistics: dict | None = None
        self._pii_replacer_time: float | None = None
        self._llm_metadata: ModelMetadata | None = None
        self._total_start: float | None = None
        self._loaded_from_save_path: bool = False
        self.preflight_report: PreflightReport | None = None
        self._data_processed: bool = False
        self._preflight_config_path: Path | None = None
        self._emit_telemetry: bool = emit_telemetry if emit_telemetry is not None else self._config_emit_telemetry()
        self._deployment_type: DeploymentTypeEnum = (
            deployment_type if deployment_type is not None else _deployment_type()
        )

    def _config_emit_telemetry(self) -> bool:
        """Return the current config's telemetry setting, defaulting on before resolution."""
        return _telemetry_enabled() if self._nss_config is None else self._nss_config.emit_telemetry

    def _ensure_observability(self) -> None:
        """Initialize structured logging when running via the SDK.

        The CLI path calls ``initialize_observability()`` during
        ``common_setup``.  When the SDK is used directly, the structlog
        processor chain (including table rendering) is never installed,
        so log messages that carry data in ``extra["ctx"]`` render as
        empty lines.  This method mirrors the CLI setup --
        ``configure_logging_from_workdir`` followed by
        ``initialize_observability`` -- and is idempotent: both the
        env-var configuration and the logging initialization are
        skipped on subsequent calls.
        """
        from ..observability import _INITIALIZED_OBSERVABILITY

        if _INITIALIZED_OBSERVABILITY:
            return
        assert self._workdir is not None
        configure_logging_from_workdir(self._workdir)
        initialize_observability()

    @traced("SafeSynthesizer.load_from_save_path", category=LogCategory.RUNTIME)
    def load_from_save_path(self, runtime_config: SafeSynthesizerParameters | None = None) -> SafeSynthesizer:
        """Load the Safe Synthesizer configuration from the save path.

        Loads the configuration from the source run directory's config file.
        When resuming from a trained model for generation, the source paths
        point to the parent workdir that contains the trained adapter.
        Optional ``runtime_config`` values for generation, evaluation,
        telemetry, and the unknown-field policy are applied without mutating
        the persisted train config. An explicitly set ``unknown_fields`` policy
        also controls validation of the saved config itself so legacy fields
        can be ignored for version-skew compatibility.

        Always prefers cached train/test splits from the training run to ensure
        evaluation metrics are consistent and privacy guarantees are maintained.
        Falls back to with_data_source() data only if cached files are missing.

        Returns:
            Self for method chaining.
        """
        self._ensure_observability()
        assert self._workdir is not None
        # Use source paths which point to parent workdir when resuming for generation
        config_file = self._workdir.source_config

        with config_file.open() as file:
            saved_config_input = json.load(file)
        unknown_fields_override = self._unknown_fields_override
        if runtime_config is not None and "unknown_fields" in runtime_config.model_fields_set:
            unknown_fields_override = runtime_config.unknown_fields
        if unknown_fields_override is None:
            saved_config = SafeSynthesizerParameters.model_validate(saved_config_input)
        else:
            normalized_input = (
                normalize_unknown_fields(
                    SafeSynthesizerParameters,
                    saved_config_input,
                    unknown_fields_override,
                )
                if isinstance(saved_config_input, Mapping)
                else saved_config_input
            )
            saved_config = SafeSynthesizerParameters.model_validate(normalized_input)
            saved_config = saved_config.with_config_patch({"unknown_fields": unknown_fields_override})
        _warn_for_saved_default_drift(saved_config)
        if runtime_config is not None:
            saved_config = saved_config.with_runtime_overrides(runtime_config)
        self._nss_config = saved_config
        self._unknown_fields_override = None
        self._generation_config = self._nss_config.generation
        self._evaluation_config = self._nss_config.evaluation
        self._emit_telemetry_config = self._nss_config.emit_telemetry

        # Load model metadata from saved file (contains initial_prefill for timeseries)
        # rather than creating new metadata from config
        metadata_file = self._workdir.metadata_file
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        logger.info(f"Loading model metadata from: {metadata_file}")
        self._llm_metadata = ModelMetadata.from_metadata_json(metadata_file, workdir=self._workdir)

        # Always prefer cached train/test splits to preserve the exact split from training.
        # This ensures evaluation metrics are consistent and privacy guarantees are maintained.
        # Only fall back to with_data_source() data if cached files are missing.
        training_path = self._workdir.source_dataset.training
        test_path = self._workdir.source_dataset.test
        assert isinstance(training_path, Path) and isinstance(test_path, Path)
        if training_path.exists():
            logger.info("Loading cached train/test split from training run")
            # training_path persists the original training split for evaluation.
            self._original_training_df = pd.read_csv(training_path)
            # test.csv may not exist (holdout=0) or may be empty (old runs with holdout=0).
            if test_path.exists() and test_path.stat().st_size > 0:
                self._test_df = pd.read_csv(test_path)
            else:
                logger.info("No test split loaded (holdout was disabled for this run)")
                self._test_df = None
            # Mark that we have fully loaded from the saved run, including cached splits.
            self._loaded_from_save_path = True
        elif self._data_source is not None:
            logger.warning(
                "Cached dataset not found, will use provided data source. "
                "Note: A new train/test split will be created which may differ from the original training split."
            )
            # process_data() will handle the split using self._data_source
        else:
            raise ValueError(
                "Cached train/test split not found and no data source provided. "
                "Call with_data_source() before load_from_save_path(), or ensure the cached dataset exists."
            )
        return self

    @traced("SafeSynthesizer.process_data", category=LogCategory.RUNTIME)
    def process_data(self, check_only: bool = False) -> SafeSynthesizer:
        """Perform train/test split, auto-config resolution, and optional PII replacement.

        Validates configured grouping/ordering columns against the input
        dataset, splits the data via ``Holdout``, runs
        ``AutoConfigResolver`` to resolve ``"auto"`` parameters, applies
        PII replacement to the training set when enabled, and persists the
        splits to the workdir.

        When ``check_only`` is ``True`` (the ``--validate`` path), PII
        replacement is intentionally skipped and CSV writes are elided; a
        resolved config YAML is written instead. Preflight therefore sees
        the *pre-replacement* training split, which is a known gap: PII
        replacement can change token lengths, so a clean ``--validate``
        does not guarantee a full run will pass token-budget checks. See
        the "``--validate`` is best-effort" callout in
        ``docs/user-guide/running.md``.

        Args:
            check_only: If ``True``, run preflight checks only (validation mode).

        Returns:
            Self for method chaining.
        """
        self._total_start = time.monotonic()
        if not os.environ.get("NSS_PHASE"):
            os.environ["NSS_PHASE"] = "process_data"

        self._ensure_observability()

        if self._loaded_from_save_path or getattr(self, "_data_processed", False):
            # Resume path or already-processed data in this builder instance; nothing to do.
            return self

        self._resolve_nss_config()
        self._resolve_datasource()

        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert isinstance(self._data_source, pd.DataFrame)

        # Run the config/dataframe stages before holdout so invalid column
        # settings produce structured preflight issues instead of downstream
        # pandas/sklearn errors. The later full preflight run still uses the
        # final training split and real metadata for split-dependent checks.
        preflight = run_preflight(
            self._data_source,
            self._nss_config,
            ModelMetadata.stub(self._nss_config),
            stages=frozenset({PreflightStage.CONFIG, PreflightStage.DATAFRAME}),
        )
        self.preflight_report = preflight
        if preflight.errors:
            summary = "\n".join(f"  {e.code}: {e.message}" for e in preflight.errors)
            raise ParameterError(f"Pre-flight check failed with {len(preflight.errors)} error(s):\n{summary}")

        holdout = Holdout(self._nss_config)
        original_training_df, self._test_df = holdout.train_test_split(self._data_source)

        self._original_training_df = (
            original_training_df  # The original training df that we use for evaluation at the end
        )
        self._training_df = original_training_df  # The active training df that might go through transformation
        self._column_statistics = None

        resolver = AutoConfigResolver(self._training_df, self._nss_config)
        resolved_config = resolver()
        self._nss_config = resolved_config

        # PII replacement is intentionally unavailable on this removal-only
        # branch. Keep validation usable, but require callers running the
        # pipeline to disable PII explicitly.
        if not check_only and self._nss_config.replace_pii is not None:
            raise ParameterError(
                "PII replacement is not available on this branch. Set replace_pii to null, "
                "pass --no-replace-pii, or call with_replace_pii(enable=False)."
            )

        # Only create new metadata if not already loaded (e.g., from load_from_save_path)
        metadata_for_preflight = self._llm_metadata
        if metadata_for_preflight is None:
            if check_only:
                try:
                    metadata_for_preflight = ModelMetadata.from_config(self._nss_config, workdir=self._workdir)
                    self._llm_metadata = metadata_for_preflight
                except Exception:
                    logger.user.warning(
                        "Could not load model metadata (network/cache); token budget checks will be skipped."
                    )
                    metadata_for_preflight = ModelMetadata.stub(self._nss_config)
            else:
                metadata_for_preflight = ModelMetadata.from_config(self._nss_config, workdir=self._workdir)
                self._llm_metadata = metadata_for_preflight

        # Persist the resolved config before running preflight so that on
        # preflight failure the CLI error report can still point at the
        # config YAML.  ``_preflight_config_path`` is set here (not after
        # ``run_preflight``) so the error path has a valid location.
        if check_only:
            assert self._workdir is not None
            self._workdir.ensure_directories()
            config_path = self._workdir.run_dir / "safe-synthesizer-config.yaml"
            self._nss_config.to_yaml(config_path, exclude_unset=False)
            self._preflight_config_path = config_path

        preflight = run_preflight(self._training_df, self._nss_config, metadata_for_preflight)
        self.preflight_report = preflight
        for issue in preflight.warnings:
            logger.user.warning(issue.message, extra={"preflight_code": issue.code, "preflight_check": issue.check})
        if preflight.errors:
            summary = "\n".join(f"  {e.code}: {e.message}" for e in preflight.errors)
            raise ParameterError(f"Pre-flight check failed with {len(preflight.errors)} error(s):\n{summary}")

        # If we're in check-only mode, we don't need to process the data further and we'll end the program.
        # ``_data_processed`` is intentionally *not* set here: the validate →
        # full-run pattern calls ``process_data(check_only=True)`` followed
        # by ``process_data()`` on the same instance, and the second call
        # must rebuild real metadata and apply PII replacement (see
        # ``TestProcessDataMetadataLifecycle.test_check_only_stub_metadata_not_persisted_for_followup_run``).
        # Callers who repeat ``process_data(check_only=True)`` pay the
        # (cheap) preflight cost twice on purpose.
        if check_only:
            return self

        self._data_processed = True

        # Always persist the original training split -- this is the version
        # reloaded by load_from_save_path and used for evaluation metrics.
        assert self._workdir is not None
        self._workdir.ensure_directories()
        # ``training.csv`` is the canonical persisted original training split.
        self._original_training_df.to_csv(self._workdir.dataset.training, index=False)
        if not self._training_df.equals(self._original_training_df):
            # The transformed (e.g. PII-replaced) training data is saved for
            # inspection only -- we don't need it in the generation or evaluation phase.
            self._training_df.to_csv(self._workdir.dataset.transformed_training, index=False)
        if self._test_df is not None:
            self._test_df.to_csv(self._workdir.dataset.test, index=False)
        return self

    @traced("SafeSynthesizer.train", category=LogCategory.RUNTIME)
    def train(self) -> SafeSynthesizer:
        """Fine-tune the base model on the processed training data.

        Creates the HuggingFace training backend, loads the base model,
        and runs fine-tuning.  Requires ``process_data()`` to have been
        called first.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If called after ``load_from_save_path()`` or
                before ``process_data()``.
        """
        if self._loaded_from_save_path:
            raise RuntimeError(
                "train() cannot be called after load_from_save_path(). "
                "The resume path is for generation and evaluation only: "
                ".load_from_save_path().generate().evaluate()"
            )

        # these are for ty
        if TYPE_CHECKING:
            assert self._training_df is not None
            assert self._nss_config is not None
            assert self._llm_metadata is not None

        if self._total_start is None:
            self._total_start = time.monotonic()
        if not os.environ.get("NSS_PHASE"):
            os.environ["NSS_PHASE"] = "train"

        self.trainer = HuggingFaceBackend(
            params=self._nss_config,
            model_metadata=self._llm_metadata,
            training_dataset=Dataset.from_pandas(self._training_df),
            action_executor=None,
            verbose_logging=True,
            maybe_split_dataset=True,
            artifact_path=None,
            workdir=self._workdir,
        )
        self.trainer.load_model()
        self.trainer.train()

        # Propagate config changes from training (e.g., inferred timestamp_format) to generation
        self._nss_config = self.trainer.params

        return self

    @traced("SafeSynthesizer.generate", category=LogCategory.RUNTIME)
    def generate(self) -> SafeSynthesizer:
        """Generate synthetic data using the trained model.

        Selects the appropriate backend (``VllmBackend`` or
        ``TimeseriesBackend``), initializes it, and generates
        synthetic records.

        Returns:
            Self for method chaining.
        """
        if not os.environ.get("NSS_PHASE"):
            os.environ["NSS_PHASE"] = "generate"
        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert self._llm_metadata is not None
        if self._total_start is None:
            self._total_start = time.monotonic()

        # Clean up trainer model if it exists (only present when train->generate in same session)
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            trainer.teardown()

        assert self._workdir is not None
        # Select backend based on time_series configuration
        if self._nss_config.time_series and self._nss_config.time_series.is_timeseries:
            self.generator = TimeseriesBackend(
                config=self._nss_config, model_metadata=self._llm_metadata, workdir=self._workdir
            )
        else:
            self.generator = VllmBackend(
                config=self._nss_config, model_metadata=self._llm_metadata, workdir=self._workdir
            )

        try:
            self.generator.initialize()
            self.generator.generate()
        finally:
            self.generator.teardown()
        self._generated = True
        return self

    @traced("SafeSynthesizer.evaluate", category=LogCategory.RUNTIME)
    def evaluate(self) -> SafeSynthesizer:
        """Run quality and privacy evaluations and populate ``results``.

        Returns:
            Self for method chaining.
        """
        if not os.environ.get("NSS_PHASE"):
            os.environ["NSS_PHASE"] = "evaluate"
        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert self._original_training_df is not None
            assert self._test_df is not None
            assert self._total_start is not None
            if self._nss_config.replace_pii is not None:
                assert self._pii_replacer_time is not None
                assert self._column_statistics is not None

        evaluation_time = None
        report = None
        if self._nss_config.evaluation.enabled:
            self.evaluator = Evaluator(
                config=self._nss_config,
                generate_results=self.generator.gen_results,
                pii_replacer_time=self._pii_replacer_time,
                column_statistics=self._column_statistics,
                training_df=self._original_training_df,
                test_df=self._test_df,
                workdir=self._workdir,
            )
            self.evaluator.evaluate()
            evaluation_time = self.evaluator.evaluation_time
            report = self.evaluator.report
        else:
            logger.info("Evaluation disabled; skipping evaluation.")
            self.evaluator = None

        training_time = None
        if trainer := getattr(self, "trainer", {}):
            if res := getattr(trainer, "results", None):
                training_time = res.elapsed_time
        generation_time = None
        if generator := getattr(self, "generator", {}):
            if res := getattr(generator, "gen_results", None):
                generation_time = res.elapsed_time

        self.results = make_nss_results(
            total_time=time.monotonic() - self._total_start,
            pii_replacer_time=self._pii_replacer_time,
            training_time=training_time,
            generation_time=generation_time,
            evaluation_time=evaluation_time,
            report=report,
            generate_results=self.generator.gen_results,
        )
        return self

    def run(self, output_file: Path | str | None = None) -> None:
        """Run the full pipeline and save results.

        Executes ``process_data`` -> ``train`` -> ``generate`` ->
        ``evaluate`` -> ``save_results``.  For step-by-step control,
        call the individual methods instead.

        Args:
            output_file: Explicit output path for the synthetic data CSV.
                Falls back to ``workdir.output_file`` when ``None``.

        Raises:
            RuntimeError: If called after ``load_from_save_path()``.
                Use ``.generate().evaluate()`` for the resume path.
        """
        if self._loaded_from_save_path:
            raise RuntimeError(
                "run() cannot be called after load_from_save_path(). "
                "The resume path is for generation and evaluation only: "
                ".load_from_save_path().generate().evaluate()"
            )

        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert isinstance(self._data_source, pd.DataFrame)

        try:
            self.process_data().train().generate().evaluate()
            self.save_results(output_file=output_file)
            _emit_nss_telemetry(self, TaskStatusEnum.COMPLETED)
        except KeyboardInterrupt:
            _emit_nss_telemetry(self, TaskStatusEnum.CANCELED)
            raise
        except Exception:
            _emit_nss_telemetry(self, TaskStatusEnum.ERROR)
            raise

    @traced("SafeSynthesizer.save_results", category=LogCategory.RUNTIME, level="INFO")
    def save_results(self, output_file: Path | str | None = None) -> SafeSynthesizer:
        """Save synthetic data, evaluation report, and metrics to the workdir.

        Writes ``synthetic_data.csv``, ``evaluation_report.html`` (when
        available), and ``evaluation_metrics.json`` into the generate
        directory.  Called automatically by ``run()``.  Call explicitly
        after stepwise execution
        (``process_data().train().generate().evaluate()``).

        Args:
            output_file: Explicit output path for the CSV.  Falls back
                to ``workdir.output_file`` when ``None``.
        """
        if TYPE_CHECKING:
            assert self.results is not None
            assert isinstance(self.results.synthetic_data, pd.DataFrame)

        assert self._workdir is not None
        match output_file:
            case Path() as p:
                output_file = p
            case str() as s:
                output_file = Path(s)
            case _:
                output_file = self._workdir.output_file

        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.results.synthetic_data.to_csv(str(output_file), index=False)
        logger.info(f"Saved synthetic data to {output_file}")

        if self.results.evaluation_report_html:
            report_path = self._workdir.evaluation_report
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(self.results.evaluation_report_html)
            logger.info(f"Saved evaluation report to {report_path}")

            # we only get non-empty results summary when evaluation is run
            metrics_path = self._workdir.evaluation_metrics
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(self.results.summary.model_dump_json(indent=2))
            logger.info(f"Saved evaluation metrics and runtimes to {metrics_path}")

        return self
