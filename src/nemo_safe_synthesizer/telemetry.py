# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Telemetry handler for NeMo products.

Environment variables:
- NEMO_TELEMETRY_ENABLED: Whether telemetry is enabled.
- NEMO_DEPLOYMENT_TYPE: The deployment type the event came from.
- NEMO_TELEMETRY_ENDPOINT: The endpoint to send the telemetry events to.
- NEMO_SESSION_PREFIX: Optional prefix to add to session IDs.
"""

from __future__ import annotations

import asyncio
import os
import platform
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import BaseModel, Field

from .observability import get_logger

if TYPE_CHECKING:
    import httpx

CLIENT_ID = "184482118588404"
NEMO_TELEMETRY_VERSION = "nemo-telemetry/1.0"
DEFAULT_ENDPOINT = "https://events.telemetry.data.nvidia.com/v1.1/events/json"
MAX_RETRIES = 3
CPU_ARCHITECTURE = platform.uname().machine
LOCAL_MODEL_LABEL = "local_path"
logger = get_logger(__name__)


class NemoSourceEnum(str, Enum):
    SAFE_SYNTHESIZER = "safe-synthesizer"
    UNDEFINED = "undefined"


class DeploymentTypeEnum(str, Enum):
    CLI = "cli"  # Library invoked via the CLI entry point
    SDK = "sdk"  # Library invoked programmatically via the SDK
    NMP = "nmp"  # Deployed through NVIDIA NeMo Platform
    SLURM = "slurm"  # Deployed through SLURM
    UNDEFINED = "undefined"


class TaskStatusEnum(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    CANCELED = "canceled"
    UNDEFINED = "undefined"


def _telemetry_enabled() -> bool:
    return os.getenv("NEMO_TELEMETRY_ENABLED", "true").lower() in ("1", "true", "yes")


def _telemetry_endpoint() -> str:
    return os.getenv("NEMO_TELEMETRY_ENDPOINT", DEFAULT_ENDPOINT)


def _redact_endpoint(endpoint: str) -> str:
    """Redact query parameters before logging telemetry endpoints."""
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return "<invalid-endpoint>"
    query = "<redacted>" if parsed.query else ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))


def _deployment_type() -> DeploymentTypeEnum:
    raw = os.getenv("NEMO_DEPLOYMENT_TYPE", "sdk").lower()
    try:
        return DeploymentTypeEnum(raw)
    except ValueError:
        return DeploymentTypeEnum.UNDEFINED


def sanitize_model_for_telemetry(model: str | None) -> str:
    """Return a telemetry-safe pretrained model label.

    Hugging Face repo IDs are safe to report, but local model paths may embed
    user or machine details. Prefer the coarse local label when the value looks
    path-like or does not satisfy Hugging Face repo ID syntax.
    """
    if model is None:
        return "undefined"

    model = model.strip()
    if not model:
        return "undefined"

    if model.startswith(("/", "./", "../", "~")):
        return LOCAL_MODEL_LABEL
    if "\\" in model or PureWindowsPath(model).drive:
        return LOCAL_MODEL_LABEL
    if model.count("/") > 1:
        return LOCAL_MODEL_LABEL
    if Path(model).expanduser().exists():
        return LOCAL_MODEL_LABEL

    try:
        validate_repo_id(model)
    except HFValidationError:
        return LOCAL_MODEL_LABEL

    return model


def _session_prefix() -> str | None:
    return os.getenv("NEMO_SESSION_PREFIX")


def bucket_records(n: int) -> str:
    """Return a bucketed string label for a count of input records.

    Used to avoid transmitting exact record counts in telemetry.
    """
    if n <= 100:
        return "1-100"
    if n <= 1_000:
        return "101-1000"
    if n <= 10_000:
        return "1001-10000"
    if n <= 100_000:
        return "10001-100000"
    return "100001+"


def bucket_columns(n: int) -> str:
    """Return a bucketed string label for a count of input columns.

    Used to avoid transmitting exact column counts in telemetry.
    """
    if n <= 5:
        return "1-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    if n <= 50:
        return "21-50"
    return "51+"


class NSSTrainingAndGenerationEvent(BaseModel):
    _event_name: ClassVar[str] = "train_and_generation_event"
    _schema_version: ClassVar[str] = "1.4"

    nemo_source: NemoSourceEnum = Field(
        default=NemoSourceEnum.SAFE_SYNTHESIZER,
        alias="nemoSource",
        description="The NeMo product that created the event.",
    )
    task: str = Field(
        ...,
        description="The type of task that was performed (e.g. train, generate, evaluate, run).",
    )
    task_status: TaskStatusEnum = Field(
        ...,
        serialization_alias="taskStatus",
        description="The final status of the task.",
    )
    deployment_type: DeploymentTypeEnum = Field(
        default_factory=_deployment_type,
        alias="deploymentType",
        description="How Safe Synthesizer was invoked (cli, sdk, nmp).",
    )

    # Timing
    job_duration_sec: float = Field(
        default=-1.0,
        alias="jobDurationSec",
        description="Wall-clock duration of the job in seconds. -1.0 if not available.",
    )

    # Generation metrics
    num_records_generated: int = Field(
        default=-1,
        alias="numRecordsGenerated",
        description="Number of valid synthetic records produced. -1 if not available.",
    )
    num_tokens_generated: int = Field(
        default=-1,
        alias="numTokensGenerated",
        description="Number of tokens generated by the model. -1 if not available.",
    )

    # Feature flags
    replace_pii_enabled: bool = Field(
        default=False,
        alias="replacePiiEnabled",
        description="Whether PII replacement was enabled for this run.",
    )
    differential_privacy_enabled: bool = Field(
        default=False,
        alias="differentialPrivacyEnabled",
        description="Whether differential privacy training was enabled for this run.",
    )
    time_series_enabled: bool = Field(
        default=False,
        alias="timeSeriesEnabled",
        description="Whether time-series mode was enabled for this run.",
    )
    group_by_enabled: bool = Field(
        default=False,
        alias="groupByEnabled",
        description="Whether group-by was set on the input data for this run.",
    )

    # Input characteristics (bucketed to avoid transmitting exact counts)
    input_records_bucket: str = Field(
        default="undefined",
        alias="inputRecordsBucket",
        description="Bucketed count of input training records (e.g. '101-1000'). Use bucket_records().",
    )
    input_columns_bucket: str = Field(
        default="undefined",
        alias="inputColumnsBucket",
        description="Bucketed count of input columns (e.g. '6-10'). Use bucket_columns().",
    )

    # Evaluation scores (-1.0 when evaluation was skipped or unavailable)
    synthetic_quality_score: float = Field(
        default=-1.0,
        alias="syntheticQualityScore",
        description="Top-level Synthetic Quality Score from the evaluation report. -1.0 if not available.",
    )
    data_privacy_score: float = Field(
        default=-1.0,
        alias="dataPrivacyScore",
        description="Top-level Data Privacy Score from the evaluation report. -1.0 if not available.",
    )

    # Model and hardware
    model: str = Field(
        default="undefined",
        description="The pretrained model used for training/generation.",
    )
    gpu: str = Field(
        default="undefined",
        description="GPU device name (e.g. 'NVIDIA A100 80GB PCIe'). 'undefined' if not on GPU.",
    )

    model_config = {"populate_by_name": True}


@dataclass
class QueuedEvent:
    event: NSSTrainingAndGenerationEvent
    timestamp: datetime
    retry_count: int = 0


def _get_iso_timestamp(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def build_payload(
    events: list[QueuedEvent], *, source_client_version: str, session_id: str = "undefined"
) -> dict[str, Any]:
    if not events:
        raise ValueError("build_payload requires at least one event")
    return {
        "browserType": "undefined",  # do not change
        "clientId": CLIENT_ID,
        "clientType": "Native",  # do not change
        "clientVariant": "Release",  # do not change
        "clientVer": source_client_version,
        "cpuArchitecture": CPU_ARCHITECTURE,
        "deviceGdprBehOptIn": "None",  # do not change
        "deviceGdprFuncOptIn": "None",  # do not change
        "deviceGdprTechOptIn": "None",  # do not change
        "deviceId": "undefined",  # do not change
        "deviceMake": "undefined",  # do not change
        "deviceModel": "undefined",  # do not change
        "deviceOS": "undefined",  # do not change
        "deviceOSVersion": "undefined",  # do not change
        "deviceType": "undefined",  # do not change
        "eventProtocol": "1.6",  # do not change
        "eventSchemaVer": events[0].event._schema_version,
        "eventSysVer": NEMO_TELEMETRY_VERSION,
        "externalUserId": "undefined",  # do not change
        "gdprBehOptIn": "None",  # do not change
        "gdprFuncOptIn": "None",  # do not change
        "gdprTechOptIn": "None",  # do not change
        "idpId": "undefined",  # do not change
        "integrationId": "undefined",  # do not change
        "productName": "undefined",  # do not change
        "productVersion": "undefined",  # do not change
        "sentTs": _get_iso_timestamp(),
        "sessionId": session_id,
        "userId": "undefined",  # do not change
        "events": [
            {
                "ts": _get_iso_timestamp(queued.timestamp),
                "parameters": queued.event.model_dump(by_alias=True, mode="json"),
                "name": queued.event._event_name,
            }
            for queued in events
        ],
    }


class TelemetryHandler:
    """
    Handles telemetry event batching, flushing, and retry logic for NeMo products.

    Supports two usage patterns:

    - **Background mode**: call ``start()`` (or use ``with handler:``) to spawn
      a daemon thread with its own event loop that drives periodic flushing.
      ``stop()`` schedules a final flush, then stops the loop and joins the thread.
    - **Fire-and-flush mode**: skip ``start()``, ``enqueue()`` events, then call
      ``stop()`` to flush once via ``asyncio.run``. No background thread is created.

    Args:
        flush_interval_seconds (float): The interval in seconds to flush the events.
        max_queue_size (int): The maximum number of events to queue before flushing.
        max_retries (int): The maximum number of times to retry sending an event.
        source_client_version (str): The version of the source client. This should be the version of
            the actual NeMo product that is sending the events, typically the same as the version of
            a PyPi package that a user would install.
        session_id (str): An optional session ID to associate with the events.
            This should be a unique identifier for the session, such as a UUID.
            It is used to group events together.
    """

    def __init__(
        self,
        flush_interval_seconds: float = 120.0,
        max_queue_size: int = 50,
        max_retries: int = MAX_RETRIES,
        source_client_version: str = "undefined",
        session_id: str = "undefined",
    ):
        self._flush_interval = flush_interval_seconds
        self._max_queue_size = max_queue_size
        self._max_retries = max_retries
        self._events: list[QueuedEvent] = []
        self._dlq: list[QueuedEvent] = []  # Dead letter queue for retry
        self._queue_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._flush_signal: asyncio.Event | None = None
        self._timer_task: asyncio.Task | None = None
        self._running = False
        self._source_client_version = source_client_version
        prefix = _session_prefix()
        self._session_id = f"{prefix}{session_id}" if prefix else session_id

    # -- Async API -----------------------------------------------------------

    async def astart(self) -> None:
        """Start the background timer task on the current event loop."""
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._flush_signal = asyncio.Event()
        self._running = True
        self._timer_task = asyncio.create_task(self._timer_loop())

    async def astop(self) -> None:
        """Cancel the timer task and flush any remaining events."""
        if not self._running:
            await self._flush_events()
            return
        self._running = False
        if self._flush_signal is not None:
            self._flush_signal.set()
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass  # expected: we just cancelled the task during shutdown
            self._timer_task = None
        await self._flush_events()
        self._loop = None
        self._flush_signal = None

    async def aflush(self) -> None:
        """Flush all queued events immediately and await completion."""
        await self._flush_events()

    # -- Sync API ------------------------------------------------------------

    def start(self) -> None:
        """Spawn a daemon thread with a persistent event loop for periodic flushing."""
        if self._running:
            return
        ready = threading.Event()
        startup_error: list[BaseException] = []

        def _run() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._flush_signal = asyncio.Event()
                self._timer_task = loop.create_task(self._timer_loop())
                self._running = True
            except BaseException as exc:  # noqa: BLE001
                startup_error.append(exc)
                ready.set()
                return
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, name="nemo-telemetry", daemon=True)
        self._thread.start()
        ready.wait()
        if startup_error:
            self._thread = None
            raise startup_error[0]

    def stop(self) -> None:
        """Flush pending events. If a background thread is running, shut it down and join."""
        if self._running and self._loop is not None and self._thread is not None:
            loop = self._loop
            future = asyncio.run_coroutine_threadsafe(self._astop_inner(), loop)
            try:
                future.result(timeout=30)
            except Exception:  # noqa: BLE001
                pass  # best-effort: telemetry must not disrupt callers
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=5)
            self._thread = None
            self._loop = None
            self._flush_signal = None
            self._timer_task = None
            self._running = False
            return
        # Fire-and-flush: no background thread; flush once on a fresh loop.
        if self._events or self._dlq:
            try:
                asyncio.run(self._flush_events())
            except Exception:  # noqa: BLE001
                pass  # best-effort: telemetry must not disrupt callers

    def flush(self) -> None:
        """Flush all queued events immediately and wait for completion."""
        if self._running and self._loop is not None and self._thread is not None:
            future: Future[None] = asyncio.run_coroutine_threadsafe(self._flush_events(), self._loop)
            try:
                future.result(timeout=30)
            except Exception:  # noqa: BLE001
                pass  # best-effort
            return
        if self._events or self._dlq:
            try:
                asyncio.run(self._flush_events())
            except Exception:  # noqa: BLE001
                pass  # best-effort

    async def _astop_inner(self) -> None:
        """Async shutdown body run on the background loop."""
        self._running = False
        if self._flush_signal is not None:
            self._flush_signal.set()
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass  # expected: we just cancelled the task during shutdown
            self._timer_task = None
        await self._flush_events()

    # -- Enqueue / signalling ------------------------------------------------

    def enqueue(self, event: object) -> None:
        if not _telemetry_enabled():
            return
        if not isinstance(event, NSSTrainingAndGenerationEvent):
            # Silently fail as we prioritize not disrupting upstream call sites and telemetry is best effort
            return
        queued = QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))
        with self._queue_lock:
            self._events.append(queued)
            should_signal = len(self._events) >= self._max_queue_size
        if should_signal:
            self._signal_flush()

    def _signal_flush(self) -> None:
        """Set the flush signal, threadsafe across the background-loop boundary."""
        loop = self._loop
        signal = self._flush_signal
        if loop is None or signal is None:
            return
        try:
            loop.call_soon_threadsafe(signal.set)
        except RuntimeError:
            pass  # loop already closed during shutdown

    # -- Context managers ----------------------------------------------------

    def __enter__(self) -> TelemetryHandler:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    async def __aenter__(self) -> TelemetryHandler:
        await self.astart()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.astop()

    # -- Internal loop -------------------------------------------------------

    async def _timer_loop(self) -> None:
        assert self._flush_signal is not None
        while self._running:
            try:
                await asyncio.wait_for(
                    self._flush_signal.wait(),
                    timeout=self._flush_interval,
                )
            except asyncio.TimeoutError:
                pass  # expected: timeout drives the periodic flush cadence
            self._flush_signal.clear()
            await self._flush_events()

    async def _flush_events(self) -> None:
        with self._queue_lock:
            dlq_events, self._dlq = self._dlq, []
            new_events, self._events = self._events, []
        events_to_send = dlq_events + new_events
        if events_to_send:
            await self._send_events(events_to_send)

    async def _send_events(self, events: list[QueuedEvent]) -> None:
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await self._send_events_with_client(client, events)
        except Exception:  # noqa: BLE001
            # Import or client setup failed; preserve events for retry rather than dropping silently.
            self._add_to_dlq(events)

    async def _send_events_with_client(self, client: httpx.AsyncClient, events: list[QueuedEvent]) -> None:
        if not events:
            return

        payload = build_payload(events, source_client_version=self._source_client_version, session_id=self._session_id)
        endpoint = _telemetry_endpoint()
        logger.runtime.debug(
            "Sending telemetry events",
            extra={
                "ctx": {
                    "endpoint": _redact_endpoint(endpoint),
                    "event_count": len(events),
                    "events": [
                        {
                            "name": queued.event._event_name,
                            "task": queued.event.task,
                            "task_status": queued.event.task_status.value,
                            "deployment_type": queued.event.deployment_type.value,
                            "retry_count": queued.retry_count,
                        }
                        for queued in events
                    ],
                }
            },
        )
        try:
            response = await client.post(endpoint, json=payload)
            # 2xx, 400, 422 are all considered complete (no retry)
            # 400/422 indicate bad payload which retrying won't fix
            if response.status_code in (400, 422) or response.is_success:
                return
            # 413 (payload too large) - split and retry
            if response.status_code == 413:
                if len(events) == 1:
                    # Can't split further, drop the event
                    return
                mid = len(events) // 2
                await self._send_events_with_client(client, events[:mid])
                await self._send_events_with_client(client, events[mid:])
                return
            if response.status_code == 408 or response.status_code >= 500:
                self._add_to_dlq(events)
        except Exception:  # noqa: BLE001
            # Silently swallow all send errors — telemetry is best-effort
            self._add_to_dlq(events)

    def _add_to_dlq(self, events: list[QueuedEvent]) -> None:
        with self._queue_lock:
            for queued in events:
                queued.retry_count += 1
                if queued.retry_count > self._max_retries:
                    continue
                self._dlq.append(queued)
