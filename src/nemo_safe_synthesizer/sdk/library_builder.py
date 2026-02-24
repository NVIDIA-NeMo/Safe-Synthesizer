# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Self

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
from ..observability import LogCategory, get_logger, traced
from ..pii_replacer.nemo_pii import NemoPII
from ..results import SafeSynthesizerResults, make_nss_results
from ..training.huggingface_backend import HuggingFaceBackend
from .config_builder import ConfigBuilder

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..generation.backend import GeneratorBackend
    from ..training.backend import TrainingBackend


def _run_pii_replacer_only(config: SafeSynthesizerParameters, df: pd.DataFrame) -> SafeSynthesizerResults:
    total_start = time.monotonic()

    replacer = NemoPII(config.replace_pii)
    replacer.transform_df(df)

    evaluator = None
    if config.evaluation.enabled:
        evaluator = Evaluator(
            config=config,
            generate_results=replacer.result.transformed_df,
            pii_replacer_time=replacer.elapsed_time if replacer else None,
            column_statistics=replacer.result.column_statistics,
            train_df=df,  # Pass the original df as the reference for evaluation
        )
        evaluator.evaluate()

    total_time_sec = time.monotonic() - total_start
    evaluation_time_sec = evaluator.evaluation_time if evaluator else None

    return make_nss_results(
        total_time=total_time_sec,
        evaluation_time=evaluation_time_sec,
        training_time=None,
        generation_time=None,
        generate_results=replacer.result.transformed_df,
        report=evaluator.report if evaluator else None,
    )


def _get_unsloth_backend_class() -> type[TrainingBackend]:
    """Get the Unsloth training backend class."""
    from ..training.unsloth_backend import UnslothTrainer

    return UnslothTrainer


def get_training_backend_class(config: SafeSynthesizerParameters) -> type[TrainingBackend]:
    """Get the training backend class for the given configuration.
    Args:
        config: SafeSynthesizerParameters object.

    Returns:
        The training backend class.
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
    """Builder for package-only Safe Synthesizer workflows.

    This class provides a fluent interface for building Safe Synthesizer workflows.
    It allows you to configure all the parameters needed to create and run a Safe Synthesizer workflow.

    Each main parameter group method returns the builder instance to allow for method chaining, and most methods
    follow a common api:
        ```python
            >>> def with_<parameter_group>(self, config: ParamT | ParamDict | None = None, **kwargs) -> SafeSynthesizer: pass
        ```
        config: Optional configuration object or dictionary containing <parameter_group> parameters.
      **kwargs: Configuration parameters for <parameter_group>, that will override any overlapping parameters in config and model defaults.

    The workflow can be run either all at once via `run()`, or step-by-step:
        ```python
        >>> builder = SafeSynthesizer().with_data_source(df)
        >>> builder.process_data().train().generate().evaluate()
        >>> results = builder.results
        ```

    Examples:
        ```python
        >>> from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

        >>> builder = (
        >>>     SafeSynthesizer()
        >>>     .with_data_source("your_dataframe")
        >>>     .with_replace_pii()  # Uses default PII replacement settings
        >>>     .synthesize()  # Enables synthesis; not strictly needed if you are already calling training() or generation()
        >>>     .with_train(learning_rate=0.0001)  # Custom training settings
        >>>     .with_generate(num_records=10000)  # Custom generation settings
        >>>     .with_evaluate(enable=False)  # disable evaluation for this job
        >>> )
        >>> builder.run()
        >>> results = builder.results
        ```

         ```python
        >>> from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

        >>> builder = (
        >>>     SafeSynthesizer()
        >>>     .with_data_source("your_dataframe")
        >>>     .with_replace_pii()  # Uses default PII replacement settings
        >>>     .synthesize()  # Enables synthesis; not strictly needed if you are already calling training() or generation()
        >>>     .with_train(learning_rate=0.0001)  # Custom training settings
        >>>     .with_generate(num_records=10000)  # Custom generation settings
        >>>     .with_evaluate(enable=False)  # disable evaluation for this job
        >>>     .process_data()  # Process data
        >>>     .train()  # Train the model
        >>>     .generate()  # Generate synthetic data
        >>>     .evaluate()  # Evaluate the generated data
        >>> )
        >>> builder.run()
        >>> results = builder.results
        ```
    """

    trainer: TrainingBackend
    generator: GeneratorBackend
    evaluator: Evaluator
    results: SafeSynthesizerResults

    def __init__(
        self,
        config: SafeSynthesizerParameters | None = None,
        workdir: Workdir | None = None,
        save_path: Path | str | None = None,
    ):
        super().__init__(config=config)
        self._workdir: Workdir = (
            workdir
            if workdir is not None
            else Workdir(
                base_path=Path(save_path) if save_path else Path("safe-synthesizer-artifacts"),
                config_name="default",
                dataset_name="data",
            )
        )
        self._resolve_nss_config()
        # Initialize state for pipeline stages
        self._train_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None
        self._column_statistics: dict | None = None
        self._pii_replacer_time: float | None = None
        self._llm_metadata: ModelMetadata | None = None
        self._total_start: float | None = None

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
        training_path = Path(self._workdir.source_dataset.training)
        test_path = Path(self._workdir.source_dataset.test)
        if training_path.exists() and test_path.exists():
            logger.info("Loading cached train/test split from training run")
            self._train_df = pd.read_csv(training_path)
            self._test_df = pd.read_csv(test_path)
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
        # match run_dir:
        #     case Path() as p if p.exists():
        #         if not config_file.exists():
        #             raise ValueError(f"Config file does not exist: {config_file}")
        #         self._nss_config = SafeSynthesizerParameters.from_json(config_file)
        #         self._llm_metadata = ModelMetadata.from_config(self._nss_config, workdir=self._workdir_structure)
        #     case Path() as p if not p.exists():
        #         raise ValueError(f"Run directory does not exist: {p}")
        #     case None:
        #         raise ValueError("save_path is required to load an existing Safe Synthesizer adapter")
        #     case _:
        #         raise ValueError(f"Invalid run directory: {run_dir}")
        return self

    @traced("SafeSynthesizer.process_data", category=LogCategory.RUNTIME)
    def process_data(self) -> SafeSynthesizer:
        """Process data: perform train/test split, auto-config resolution, and optional PII replacement.

        This method prepares the data for training by:
        - Splitting data into train/test sets using holdout
        - Resolving auto-configuration parameters
        - Applying PII replacement if enabled (on training data only)

        Returns:
            Self for method chaining.

        """
        self._total_start = time.monotonic()
        if not os.environ.get("NSS_PHASE"):
            os.environ["NSS_PHASE"] = "process_data"

        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert isinstance(self._data_source, pd.DataFrame)

        if self._train_df is not None and self._test_df is not None:
            logger.warning("Data already processed, skipping data processing...")
            return self

        holdout = Holdout(self._nss_config)
        original_train_df, self._test_df = holdout.train_test_split(self._data_source)

        self._train_df = original_train_df
        self._column_statistics = None

        resolver = AutoConfigResolver(self._train_df, self._nss_config)
        resolved_config = resolver()
        self._nss_config = resolved_config

        if self._nss_config.enable_replace_pii:
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
        # Ensure dataset directory exists before writing CSV files
        self._workdir.ensure_directories()
        self._train_df.to_csv(self._workdir.dataset.training, index=False)
        if self._test_df is not None:
            self._test_df.to_csv(self._workdir.dataset.test, index=False)
        else:
            Path(self._workdir.dataset.test).touch()
        return self

    @traced("SafeSynthesizer.train", category=LogCategory.RUNTIME)
    def train(self) -> SafeSynthesizer:
        """Train the model on the processed data.

        This method:
        - Creates the training backend (HuggingFace or Unsloth)
        - Loads the base model
        - Performs fine-tuning on the training data

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If process_data() has not been called first.
        """

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

        This method:
        - Initializes the VLLM generation backend
        - Generates synthetic records based on configuration

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

        self.generator.initialize()
        self.generator.generate(keep_llm_state=False)
        self._generated = True
        return self

    @traced("SafeSynthesizer.evaluate", category=LogCategory.RUNTIME)
    def evaluate(self) -> SafeSynthesizer:
        """Evaluate the generated synthetic data and build final results.

        This method:
        - Runs quality and privacy evaluations on generated data
        - Compiles timing information and evaluation reports
        - Populates the `results` attribute

        Returns:
            Self for method chaining.
        """
        if not os.environ.get("NSS_PHASE"):
            os.environ["NSS_PHASE"] = "evaluate"
        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert self._train_df is not None
            assert self._test_df is not None
            assert self._column_statistics is not None
            assert self._pii_replacer_time is not None
            assert self._total_start is not None

        self.evaluator = Evaluator(
            config=self._nss_config,
            generate_results=self.generator.gen_results,
            pii_replacer_time=self._pii_replacer_time,
            column_statistics=self._column_statistics,
            train_df=self._train_df,
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

    def _run_pii_replacer_only(self) -> SafeSynthesizerResults:
        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert isinstance(self._data_source, pd.DataFrame)

        if self._total_start is None:
            self._total_start = time.monotonic()

        replacer = NemoPII(self._nss_config.replace_pii)
        replacer.transform_df(self._data_source)
        return make_nss_results(
            total_time=time.monotonic() - self._total_start,
            evaluation_time=None,
            training_time=None,
            generation_time=None,
            generate_results=replacer.result.transformed_df,
        )

    def run(self) -> None:
        """Run the Safe Synthesizer workflow end to end.

        This method executes the complete pipeline:
        1. process_data() - Data preparation and PII replacement
        2. train() - Model fine-tuning
        3. generate() - Synthetic data generation
        4. evaluate() - Quality and privacy evaluation

        For PII-replacement-only mode (when enable_synthesis=False),
        this method handles that workflow directly.

        For step-by-step control, call the individual methods instead:
            builder.process_data().train().generate().evaluate()
        """
        if TYPE_CHECKING:
            assert self._nss_config is not None
            assert isinstance(self._data_source, pd.DataFrame)

        if not self._nss_config.enable_synthesis:
            self.results = self._run_pii_replacer_only()
            return  # Exit after PII-replacer-only mode

        self.process_data().train().generate().evaluate()

    @traced("SafeSynthesizer.save_results", category=LogCategory.RUNTIME, level="INFO")
    def save_results(self, output_file: Path | str | None = None) -> Self:
        """Save synthetic data results and evaluation report to the workdir.

        Saves:
            - Synthetic data CSV to output_file or workdir.output_file
            - Evaluation report HTML to workdir.evaluation_report (if available)

        Args:
            output_file: Explicit output path for CSV (takes precedence over workdir default)
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
