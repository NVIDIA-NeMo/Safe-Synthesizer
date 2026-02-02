# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd
from typing_extensions import Self

if TYPE_CHECKING:
    from nemo_microservices import NeMoMicroservices
    from nemo_microservices.types import PlatformJobLog, PlatformJobStatusResponse
    from nemo_microservices.types.safe_synthesizer.jobs.safe_synthesizer_summary import (
        SafeSynthesizerSummary,
    )

logger = logging.getLogger(__name__)


class ReportHtml(object):
    """Class to hold the HTML report string."""

    def __init__(self, html: str):
        self.raw_html = html
        self.as_data_uri = f"data:text/html;base64,{b64encode(self.raw_html.encode()).decode()}"

    def save(self, path: str | Path) -> None:
        """Save the evaluation report to a file.

        Args:
            path: The path to save the report to.
        """
        with open(path, "w") as f:
            f.write(self.raw_html)

    def display_report_in_notebook(self, width="100%", height=1000) -> None:
        """Display the evaluation report in a jupyter notebook.

        Requires the IPython library to be installed.

        Args:
            width: The width of the iframe to display the report in.
            height: The height of the iframe to display the report in.
        """
        try:
            from IPython.display import IFrame, display
        except ImportError:
            logger.warning("IPython is required to display reports in notebooks. Report will not be displayed.")
            return

        display(IFrame(self.as_data_uri, width=width, height=height))

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read the evaluation report from a file.

        Args:
            path: The path to read the report from.
        """
        with open(path, "r") as f:
            raw_html = f.read()
        return cls(raw_html)


class SafeSynthesizerJob:
    """Interface for convenient interaction with Safe Synthesizer jobs.

    This class provides a wrapper around the Safe Synthesizer job SDK to make common operations easier.

    An instance is returned from the `create_job` method of the `SafeSynthesizerJobBuilder` class.
    Or create an instance with a job id and a NeMoMicroservices client.

    Examples:
        ```python
        >>> from nemo_microservices import NeMoMicroservices
        >>> from nemo_microservices.beta.safe_synthesizer.sdk.job import SafeSynthesizerJob

        >>> client = NeMoMicroservices(base_url=..., inference_base_url=...)
        >>> job = SafeSynthesizerJob(job_id=..., client=client, workspace="default")
        >>> job.fetch_status()
        >>> job.wait_for_completion()
        >>> job.fetch_summary()
        >>> df = job.fetch_data()
        >>> job.save_report("./evaluation_report.html")
        ```

        And in a jupyter notebook to display the report inline:
        ```python
        >>> job.display_report_in_notebook()
        ```
    """

    def __init__(self, job_name: str, client: NeMoMicroservices, workspace: str = "default"):
        """Initialize a SafeSynthesizerJob instance.

        Args:
            job_name: The name of the job to interact with (note: despite the parameter name, this should be the job name, not entity ID).
            client: The NeMoMicroservices client to use to interact with the job.
            workspace: The workspace to use to interact with the job.
        """
        self.job_name = job_name
        self._client = client
        self._workspace = workspace

    def fetch_status(self) -> str:
        """Fetch the status of the job.

        Returns:
            The status of the job.
        """
        return self._client.safe_synthesizer.jobs.get_status(self.job_name, workspace=self._workspace).status

    def fetch_status_info(self) -> PlatformJobStatusResponse:
        """Fetch the status information of the job.

        Returns:
            The status information of the job.
        """
        return self._client.safe_synthesizer.jobs.get_status(self.job_name, workspace=self._workspace)

    def wait_for_completion(
        self, poll_interval: int = 10, verbose: bool = True, log_timeout: float | None = None
    ) -> None:
        """Block until the job is completed.

        Prints the logs by default. Uses incremental log fetching via page cursor
        to only fetch new logs on each poll.

        Args:
            poll_interval: The interval in seconds to poll the job status.
            verbose: Gets logs and prints them at this interval. Default: True
            log_timeout: Timeout in seconds for log requests. Defaults to 300s (5 minutes).
        """
        # Track cursor to continue from last position, and seen timestamps for dedup
        last_page_cursor: str | None = None
        seen_log_keys: set[str] = set()
        previous_status_info = None
        current_status_info = self.fetch_status_info()
        while current_status_info.status not in ["completed", "error", "cancelled"]:
            if verbose:
                logging_level = None
                try:
                    httpx_logger = logging.getLogger("httpx")
                    logging_level = httpx_logger.level
                    httpx_logger.setLevel("ERROR")
                    # Fetch logs starting from last cursor position
                    new_logs, last_page_cursor = self._fetch_logs_incremental(
                        page_cursor=last_page_cursor, timeout=log_timeout
                    )
                    for new_log in new_logs:
                        # Deduplicate using timestamp + message hash (cursor may re-fetch last page)
                        log_key = f"{new_log.timestamp}:{hash(new_log.message)}"
                        if log_key not in seen_log_keys:
                            print(new_log.message.strip())
                            seen_log_keys.add(log_key)
                except Exception:
                    logger.exception("Error fetching logs")
                finally:
                    if logging_level:
                        logging.getLogger("httpx").setLevel(logging_level)
            current_status_info = self.fetch_status_info()
            if current_status_info != previous_status_info:
                print(
                    f"Job status changed to status: '{current_status_info.status}',",
                    f"status_details: {current_status_info.status_details},",
                    f"error_details: {current_status_info.error_details}",
                )
                previous_status_info = current_status_info
            time.sleep(poll_interval)

    def fetch_summary(self) -> SafeSynthesizerSummary:
        """Fetch the summary of the job.

        Returns:
            A summary of machine-readable metrics for a completed job. Raises a 404 error if the job is not finished.
        """
        return self._client.safe_synthesizer.jobs.results.download_summary(self.job_name, workspace=self._workspace)

    def fetch_report(self) -> ReportHtml:
        """Fetch the evaluation report of the job as a string of html.

        Recommended to use save_report or display_report_in_notebook for most use cases.

        Returns:
            A string containing the html representation of the evaluation report.
        """
        response = self._client.safe_synthesizer.jobs.results.download_evaluation_report(
            self.job_name, workspace=self._workspace
        )
        return ReportHtml(html=response.read().decode("utf-8"))

    def display_report_in_notebook(self, width="100%", height=1000) -> None:
        """Display the evaluation report in a jupyter notebook.

        Requires the IPython library to be installed.

        Args:
            width: The width of the iframe to display the report in.
            height: The height of the iframe to display the report in.
        """

        report = self.fetch_report()
        # Create a data URI from the report HTML
        report.display_report_in_notebook()

    def save_report(self, path: str | Path) -> None:
        """Save the evaluation report to a file.

        Args:
            path: The path to save the report to.
        """
        report = self.fetch_report()
        report.save(path)

    def fetch_data(self) -> pd.DataFrame:
        """Fetch the synthetic data of the job as a pandas DataFrame.

        Returns:
            A pandas DataFrame containing the synthetic data.
        """
        response = self._client.safe_synthesizer.jobs.results.download_synthetic_data(
            self.job_name, workspace=self._workspace
        )
        return pd.read_csv(BytesIO(response.read()))

    def _fetch_logs_incremental(
        self, page_cursor: str | None = None, timeout: float | None = None
    ) -> tuple[list[PlatformJobLog], str | None]:
        """Fetch logs incrementally starting from a page cursor.

        Fetches all pages from the given cursor position to the end,
        returning the collected logs and a cursor for the next poll.

        The returned cursor points to the last page fetched, so subsequent
        polls will re-fetch that page (to catch any newly appended logs)
        plus any new pages. Use deduplication on the caller side to avoid
        printing duplicate logs.

        Args:
            page_cursor: Cursor to start from (None = start from beginning).
            timeout: Timeout in seconds. Defaults to 300s (5 minutes).

        Returns:
            Tuple of (list of log objects, cursor to use for next poll).
        """
        if timeout is None:
            timeout = 300.0

        all_logs: list[PlatformJobLog] = []
        current_cursor = page_cursor
        last_cursor_with_data: str | None = page_cursor

        while True:
            response = self._client.with_options(timeout=timeout).safe_synthesizer.jobs.get_logs(
                self.job_name, page_cursor=current_cursor, workspace=self._workspace
            )

            if response.data:
                all_logs.extend(response.data)
                # Remember this cursor - it had data
                last_cursor_with_data = current_cursor

            if response.next_page is None:
                # No more pages - return the last cursor that had data
                # so next poll re-fetches that page (in case new logs appended)
                return all_logs, last_cursor_with_data
            current_cursor = response.next_page

    def fetch_logs(self, timeout: float | None = None) -> Iterator[PlatformJobLog]:
        """Fetch the logs of the job as an iterator over log objects.

        Recommended to use print_logs for human-readable output. This method returns an iterator
        over log objects and is useful for programmatic access.

        Args:
            timeout: Timeout in seconds for each log request. Defaults to 300 seconds (5 minutes)
                to handle jobs with many log files. Set to None to use the SDK default (60s).

        Returns:
            A generator for the log objects.
        """
        # Use a longer timeout for log queries since they can be slow when
        # there are many parquet log files to scan
        if timeout is None:
            timeout = 300.0  # 5 minutes default for log queries

        page_cursor = None
        while True:
            response = self._client.with_options(timeout=timeout).safe_synthesizer.jobs.get_logs(
                self.job_name, page_cursor=page_cursor, workspace=self._workspace
            )
            yield from response.data
            if response.next_page is None:
                break
            page_cursor = response.next_page

    def print_logs(self, timeout: float | None = None) -> None:
        """Print the logs of the job to stdout.

        Args:
            timeout: Timeout in seconds for each log request. Defaults to 300s (5 minutes).
        """
        for log in self.fetch_logs(timeout=timeout):
            print(log.message.strip())
