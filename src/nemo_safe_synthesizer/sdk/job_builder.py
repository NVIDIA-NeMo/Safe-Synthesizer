# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
import string
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from nemo_microservices import NotFoundError
from typing_extensions import Self

from ..config import (
    SafeSynthesizerJobConfig,
)
from .config_builder import ConfigBuilder
from .job import SafeSynthesizerJob

if TYPE_CHECKING:
    from nemo_microservices import NeMoMicroservices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafeSynthesizerJobBuilder(ConfigBuilder):
    """Builder for Safe Synthesizer Jobs ran with the Nemo Microservice Platform.

    This class provides a fluent interface for building Safe Synthesizer configurations.
    It allows you to configure all the parameters needed to create and run a Safe Synthesizer job
    as defined by the SafeSythesizerJobConfig class.

    Each main parameter group method returns the builder instance to allow for method chaining, and most methods
    follow a common api:
        ```python
            >>> def with_<parameter_group>(self, config: ParamT | ParamDict | None = None, **kwargs) -> SafeSynthesizerJobBuilder: pass
        ```
      config: Optional configuration object or dictionary containing <parameter_group> parameters.
      **kwargs: Configuration parameters for <parameter_group>, that will override any overlapping parameters in config and model defaults.

    Examples:
        ```python
        >>> from nemo_microservices import NeMoMicroservices
        >>> from nemo_microservices.beta.safe_synthesizer.sdk.job_builder import SafeSynthesizerJobBuilder

        >>> client = NeMoMicroservices(base_url=..., inference_base_url=...)
        >>> # Using default PII replacement settings
        >>> builder = (
        >>>     SafeSynthesizerJobBuilder(client)
        >>>     .with_data_source("your_dataframe")
        >>>     .with_replace_pii()  # Uses default PII replacement settings
        >>>     .synthesize()  # Enables synthesis; not strictly needed if you are already calling training() or generation()
        >>>     .with_train(learning_rate=0.0001)  # Custom training settings
        >>>     .with_generate(num_records=10000)  # Custom generation settings
        >>>     .with_evaluate(enable=False)  # disable evaluation for this job
        >>>     .resolve_job_config()  # Finalizes the job configuration - useful for debugging or logging for yourself
        >>> )
        >>> job = builder.create_job()  # Creates and starts the job
    """

    def __init__(self, client: NeMoMicroservices, workspace: str = "default"):
        super().__init__()
        self._client = client
        self._workspace = workspace
        self._final_job_config: SafeSynthesizerJobConfig | None = None

    def _generate_random_string(self, length=6):
        """
        Generates a random string of a specified length using uppercase letters and digits.
        """
        characters = string.ascii_uppercase + string.digits
        random_string = "".join(random.choice(characters) for _ in range(length))
        return random_string

    def _resolve_datasource(self, **kwargs) -> None:
        try:
            match self._data_source:
                case pd.DataFrame():
                    pass
                case str(url):
                    self._data_source: pd.DataFrame = pd.read_csv(url, **kwargs)
                case _:
                    raise ValueError("Data source must be a pandas DataFrame or a URL")

            with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as temp_file:
                # Write the DataFrame to the temporary file
                self._data_source.to_csv(temp_file.name, index=False)
            # Generate unique filename
            file_name = f"dataset{self._generate_random_string()}.csv"
            result = self._upload_to_fileset(
                dataset_path=temp_file.name,
                filename=file_name,
                fileset_name="safe-synthesizer-inputs",
            )
            self._data_source_path = result
        finally:
            Path(temp_file.name).unlink(missing_ok=True)

    def with_classify_model_provider(self, provider_name: str) -> Self:
        """Configure column classification using an Inference Gateway model provider.

        The model provider must exist in the same workspace as the job and should
        be configured to serve an LLM suitable for column classification tasks.

        The provider reference is stored in the job config as 'workspace/provider_name',
        and the service-side job compiler will resolve it to the appropriate endpoint URL.

        Args:
            provider_name: Name of the model provider in the Inference Gateway.
                Must exist in the same workspace as the job.

        Returns:
            The builder instance for method chaining.

        Examples:
            ```python
            >>> builder = (
            >>>     SafeSynthesizerJobBuilder(client)
            >>>     .with_data_source(df)
            >>>     .with_classify_model_provider("my-classify-llm")
            >>>     .with_replace_pii()
            >>> )
            ```
        """
        # Store as workspace/provider_name format for the compiler to resolve
        self._classify_model_provider = f"{self._workspace}/{provider_name}"
        logger.info(f"Configured classify model provider: {self._classify_model_provider}")
        return self

    def with_hf_token_secret(self, secret_name: str) -> Self:
        """Configure HuggingFace authentication using a platform secret.

        The secret must exist in the same workspace as the job and should contain
        a valid HuggingFace token for accessing private models or avoiding rate limits.

        Args:
            secret_name: Name of the platform secret containing the HuggingFace token.
                Must exist in the same workspace as the job.

        Returns:
            The builder instance for method chaining.

        Examples:
            ```python
            >>> # First, create the secret:
            >>> client.secrets.create(name="hf-token", namespace="default", data="hf_...")
            >>>
            >>> # Then reference it in the job:
            >>> builder = (
            >>>     SafeSynthesizerJobBuilder(client)
            >>>     .with_data_source(df)
            >>>     .with_hf_token_secret("hf-token")
            >>>     .with_replace_pii()
            >>> )
            ```
        """
        self._hf_token_secret = secret_name
        return self

    def _resolve_job_config(self):
        self._resolve_datasource()
        self._resolve_nss_config()
        if not self._enable_replace_pii and not self._enable_synthesis:
            raise ValueError("Data synthesis and/or replace PII must be enabled")

        if not self._data_source_path:
            raise ValueError("No data source path found after uploading dataset")

        job_config = SafeSynthesizerJobConfig(
            data_source=self._data_source_path,
            config=self._nss_config,
            hf_token_secret=self._hf_token_secret,
        )
        self._final_job_config = job_config

    def resolve_job_config(self) -> Self:
        """Generate the final job configuration.

        This method compiles all the configurations set through the builder methods
        into a final job configuration that can be used to create and execute a job.

        Returns:
            The final job configuration.
        """
        if self._final_job_config is None:
            self._resolve_job_config()
        return self

    def create_job(self, **kwargs) -> SafeSynthesizerJob:
        """Create and optionally execute the synthetic data generation job.

        Args:
            **kwargs: Additional job creation parameters.

        Returns:
            Job object that can be used to fetch results.

        """
        self._resolve_job_config()
        response = self._client.safe_synthesizer.jobs.create(
            workspace=self._workspace, spec=self._final_job_config.model_dump(), **kwargs
        )
        return SafeSynthesizerJob(response.name, self._client, workspace=self._workspace)

    def _upload_to_fileset(
        self,
        dataset_path: str | Path,
        filename: str,
        fileset_name: str,
    ) -> str:
        """Upload a dataset file to a fileset.

        Args:
            dataset_path: Local path to the dataset file.
            filename: Name to give the file in the fileset.
            fileset_name: Name of the fileset to upload to.

        Returns:
            The fileset:// URL of the uploaded file.
        """
        dataset_path = self._validate_dataset_path(dataset_path)

        # Ensure the fileset exists
        try:
            self._client.filesets.retrieve(fileset_name, workspace=self._workspace)
        except NotFoundError:
            logger.info(f"Creating fileset: {fileset_name}")
            self._client.filesets.create(name=fileset_name, workspace=self._workspace)

        # Upload the file using sdk.filesets.fsspec
        remote_path = f"{self._workspace}/{fileset_name}/{filename}"
        self._client.filesets.fsspec.put_file(str(dataset_path), remote_path)

        return f"fileset://{remote_path}"

    @staticmethod
    def _validate_dataset_path(dataset_path: str | Path) -> Path:
        if not Path(dataset_path).is_file():
            raise ValueError("🛑 To upload a dataset, you must provide a valid file path.")
        if not Path(dataset_path).name.endswith((".parquet", ".csv", ".json", ".jsonl")):
            raise ValueError(
                "🛑 Dataset files must be in `parquet`, `csv`, or `json` (orient='records', lines=True) format."
            )
        return Path(dataset_path)
