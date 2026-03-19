# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable pipeline for Safe Synthesizer.

Extends ``ConfigBuilder`` with the ``SafeSynthesizer`` class, which
adds artifact management (via ``Workdir``) and stepwise pipeline
execution: ``process_data`` -> ``train`` -> ``generate`` -> ``evaluate``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset

from ..cli.artifact_structure import Workdir
from ..config import (
    SafeSynthesizerParameters,
)
from ..config.autoconfig import AutoConfigResolver
from ..evaluation.evaluator import Evaluator
from ..generation.timeseries_backend import TimeseriesBackend
from ..generation.vllm_backend import VllmBackend
from ..holdout.holdout import Holdout
from ..llm.metadata import ModelMetadata
from ..observability import LogCategory, configure_logging_from_workdir, get_logger, initialize_observability, traced
from ..pii_replacer.nemo_pii import NemoPII
from ..results import SafeSynthesizerResults, make_nss_results
from ..training.huggingface_backend import HuggingFaceBackend
from .config_builder import ConfigBuilder

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..generation.backend import GeneratorBackend
    from ..training.backend import TrainingBackend


def _get_unsloth_backend_class() -> type[TrainingBackend]:
    """Lazily import and return the Unsloth training backend class.

    The import is deferred so that the ``unsloth`` extra is only
    required when Unsloth is actually selected as the backend.

    Returns:
        The ``UnslothTrainer`` class.
    """
    from ..training.unsloth_backend import UnslothTrainer

    return UnslothTrainer


def get_training_backend_class(config: SafeSynthesizerParameters) -> type[TrainingBackend]:
    """Select the training backend class based on configuration.

    Returns ``HuggingFaceBackend`` by default, or ``UnslothTrainer``
    when ``config.training.use_unsloth`` is ``True``.

    Args:
        config: Resolved pipeline parameters.

    Returns:
        The training backend class to instantiate.

    Raises:
        ValueError: If the backend identifier is unrecognized.
    """
    class_map = {
        "huggingface": HuggingFaceBackend,
        "unsloth": _get_unsloth_backend_class(),
    }
    logger.user.info(f"Unsloth enabled: {config.training.use_unsloth}")
    cls = "unsloth" if config.training.use_unsloth is True else "huggingface"
    cls = class_map.get(cls)
    if cls is None:
        raise ValueError(f"Unsupported training backend: {config.training.use_unsloth}")
    return cls


class SafeSynthesizer(ConfigBuilder):
    """Fluent builder and runner for Safe Synthesizer workflows.

    Extends ``ConfigBuilder`` with artifact management and stepwise
    pipeline execution.  Run all at once via ``run()``, or step by
    step::

        builder = SafeSynthesizer().with_data_source(df)
        builder.process_data().train().generate().evaluate()
        results = builder.results

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

    trainer: TrainingBackend
    """Training backend instance, populated after ``train()``."""

    generator: GeneratorBackend
    """Generation backend instance, populated after ``generate()``."""

    evaluator: Evaluator
    """Evaluator instance, populated after ``evaluate()``."""

    results: SafeSynthesizerResults
    """Final pipeline results, populated after ``evaluate()`` or ``run()``."""

    def __init__(
        self,
        config: SafeSynthesizerParameters | None = None,
        workdir: Workdir | None = None,
        save_path: Path | str | None = None,
    ):
        super().__init__(config=config)
        self._workdir = workdir
        if self._workdir is None:
            # Create a default workdir when none provided
            # Use "default" for config_name and "data" for dataset_name as fallbacks
            self._workdir = Workdir(
                base_path=Path(save_path) if save_path else Path("safe-synthesizer-artifacts"),
                config_name="default",
                dataset_name="data",
            )
        self._resolve_nss_config()
        # Initialize state for pipeline stages
        self._train_df: pd.DataFrame | None = (
            None  # The active training df that might go through transformation, eg. pii replacement
        )
        self._original_train_df: pd.DataFrame | None = (
            None  # The original training df that we save for evaluation at the end
        )
        self._test_df: pd.DataFrame | None = None
        self._column_statistics: dict | None = None
        self._pii_replacer_time: float | None = None
        self._llm_metadata: ModelMetadata | None = None
        self._total_start: float | None = None
        self._loaded_from_save_path: bool = False

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
        configure_logging_from_workdir(self._workdir)
        initialize_observability()

    @traced("SafeSynthesizer.load_from_save_path", category=LogCategory.RUNTIME)
    def load_from_save_path(self) -> SafeSynthesizer:
        """Load the Safe Synthesizer configuration from the save path.

        Loads the configuration from the source run directory's config file.
        When resuming from a trained model for generation, the source paths
        point to the parent workdir that contains the trained adapter.

        Always prefers cached train/test splits from the training run to ensure
        evaluation metrics are consistent and privacy guarantees are maintained.
        Falls back to with_data_source() data only if cached files are missing.

        Returns:
            Self for method chaining.
        """
        self._ensure_observability()
        # Use source paths which point to parent workdir when resuming for generation
        config_file = self._workdir.source_config

        self._nss_config = SafeSynthesizerParameters.from_json(config_file)

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
        if training_path.exists() and test_path.exists():
            logger.info("Loading cached train/test split from training run")
            # training_path persists the original training split for evaluation.
            self._original_train_df = pd.read_csv(training_path)
            self._test_df = pd.read_csv(test_path)
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
    def process_data(self) -> SafeSynthesizer:
        """Perform train/test split, auto-config resolution, and optional PII replacement.

        Splits the data via ``Holdout``, runs ``AutoConfigResolver`` to
        resolve ``"auto"`` parameters, applies PII replacement to the
        training set when enabled, and persists the splits to the workdir.

        Returns:
            Self for method chaining.
        """
        self._total_start = time.monotonic()
        if not os.environ.get("NSS_PHASE"):
            os.environ["NSS_PHASE"] = "process_data"

        self._ensure_observability()

        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert isinstance(self._data_source, pd.DataFrame)

        if self._loaded_from_save_path or getattr(self, "_data_processed", False):
            # Resume path or already-processed data in this builder instance; nothing to do.
            return self

        self._resolve_datasource()

        holdout = Holdout(self._nss_config)
        original_train_df, self._test_df = holdout.train_test_split(self._data_source)

        self._original_train_df = original_train_df  # The original training df that we use for evaluation at the end
        self._train_df = original_train_df  # The active training df that might go through transformation
        self._column_statistics = None

        resolver = AutoConfigResolver(self._train_df, self._nss_config)
        resolved_config = resolver()
        self._nss_config = resolved_config

        if self._nss_config.replace_pii is not None:
            replacer = NemoPII(self._nss_config.replace_pii)
            replacer.transform_df(original_train_df)
            self._train_df = replacer.result.transformed_df
            self._column_statistics = replacer.result.column_statistics
            self._pii_replacer_time = replacer.elapsed_time
            # We explicitly do not replace PII in the test set so that the
            # privacy metrics are valid.

        # Only create new metadata if not already loaded (e.g., from load_from_save_path)
        if self._llm_metadata is None:
            self._llm_metadata = ModelMetadata.from_config(self._nss_config, workdir=self._workdir)
        self._data_processed = True

        # Always persist the original training split -- this is the version
        # reloaded by load_from_save_path and used for evaluation metrics.
        self._workdir.ensure_directories()
        # ``training.csv`` is the canonical persisted original training split.
        self._original_train_df.to_csv(self._workdir.dataset.training, index=False)
        if not self._train_df.equals(self._original_train_df):
            # The transformed (e.g. PII-replaced) training data is saved for
            # inspection only -- we don't need it in the generation or evaluation phase.
            self._train_df.to_csv(self._workdir.dataset.transformed_training, index=False)
        if self._test_df is not None:
            self._test_df.to_csv(self._workdir.dataset.test, index=False)
        else:
            self._workdir.dataset.test.touch()
        return self

    @traced("SafeSynthesizer.train", category=LogCategory.RUNTIME)
    def train(self) -> SafeSynthesizer:
        """Fine-tune the base model on the processed training data.

        Creates the training backend (HuggingFace or Unsloth), loads
        the base model, and runs fine-tuning.  Requires
        ``process_data()`` to have been called first.

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
            assert self._train_df is not None
            assert self._nss_config is not None
            assert self._llm_metadata is not None

        if self._total_start is None:
            self._total_start = time.monotonic()
        if not os.environ.get("NSS_PHASE"):
            os.environ["NSS_PHASE"] = "train"

        self.trainer = get_training_backend_class(self._nss_config)(
            params=self._nss_config,
            model_metadata=self._llm_metadata,
            training_dataset=Dataset.from_pandas(self._train_df),
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
        if hasattr(self, "trainer") and self.trainer is not None:
            self.trainer.delete_trainable_model()

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
            assert self._original_train_df is not None
            assert self._test_df is not None
            assert self._total_start is not None
            if self._nss_config.replace_pii is not None:
                assert self._pii_replacer_time is not None
                assert self._column_statistics is not None

        self.evaluator = Evaluator(
            config=self._nss_config,
            generate_results=self.generator.gen_results,
            pii_replacer_time=self._pii_replacer_time,
            column_statistics=self._column_statistics,
            train_df=self._original_train_df,
            test_df=self._test_df,
            workdir=self._workdir,
        )
        self.evaluator.evaluate()

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
            training_time=training_time,
            generation_time=generation_time,
            evaluation_time=self.evaluator.evaluation_time,
            report=self.evaluator.report,
            generate_results=self.generator.gen_results,
        )
        return self

    def run(self) -> None:
        """Run the full pipeline: ``process_data`` -> ``train`` -> ``generate`` -> ``evaluate``.

        For step-by-step control, call the individual methods instead.

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

        self.process_data().train().generate().evaluate()
        self.save_results()

    @traced("SafeSynthesizer.save_results", category=LogCategory.RUNTIME, level="INFO")
    def save_results(self, output_file: Path | str | None = None) -> None:
        """Save synthetic data CSV and evaluation report HTML to the workdir.

        Args:
            output_file: Explicit output path for the CSV.  Falls back
                to ``workdir.output_file`` when ``None``.
        """
        if TYPE_CHECKING:
            assert self.results is not None
            assert isinstance(self.results.synthetic_data, pd.DataFrame)

        # Determine output file path for synthetic data
        match output_file:
            case Path() as p:
                output_file = p
            case str() as s:
                output_file = Path(s)
            case _:
                output_file = self._workdir.output_file

        # Save synthetic data CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.results.synthetic_data.to_csv(str(output_file), index=False)
        logger.info(f"Saved synthetic data to {output_file}")

        # Save evaluation report HTML if available
        if self.results.evaluation_report_html:
            report_path = self._workdir.evaluation_report
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(self.results.evaluation_report_html)
            logger.info(f"Saved evaluation report to {report_path}")

        return self
