# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nemo_safe_synthesizer.telemetry import (
    TELEMETRY_ENABLED,
    DeploymentTypeEnum,
    NemoSourceEnum,
    NSSTrainingAndGenerationEvent,
    QueuedEvent,
    TaskStatusEnum,
    TelemetryHandler,
    bucket_columns,
    bucket_records,
    build_payload,
)

# =============================================================================
# Bucket helpers
# =============================================================================


class TestBucketRecords:
    def test_lower_boundary(self):
        assert bucket_records(1) == "1-100"
        assert bucket_records(100) == "1-100"

    def test_mid_buckets(self):
        assert bucket_records(101) == "101-1000"
        assert bucket_records(1000) == "101-1000"
        assert bucket_records(1001) == "1001-10000"
        assert bucket_records(10000) == "1001-10000"
        assert bucket_records(10001) == "10001-100000"
        assert bucket_records(100000) == "10001-100000"

    def test_upper_bucket(self):
        assert bucket_records(100001) == "100001+"
        assert bucket_records(10_000_000) == "100001+"


class TestBucketColumns:
    def test_lower_boundary(self):
        assert bucket_columns(1) == "1-5"
        assert bucket_columns(5) == "1-5"

    def test_mid_buckets(self):
        assert bucket_columns(6) == "6-10"
        assert bucket_columns(10) == "6-10"
        assert bucket_columns(11) == "11-20"
        assert bucket_columns(20) == "11-20"
        assert bucket_columns(21) == "21-50"
        assert bucket_columns(50) == "21-50"

    def test_upper_bucket(self):
        assert bucket_columns(51) == "51+"
        assert bucket_columns(500) == "51+"


# =============================================================================
# NSSTrainingAndGenerationEvent
# =============================================================================


class TestNSSTrainingAndGenerationEvent:
    def test_defaults(self):
        event = NSSTrainingAndGenerationEvent(task="run", task_status=TaskStatusEnum.COMPLETED)
        assert event.nemo_source == NemoSourceEnum.SAFE_SYNTHESIZER
        assert event.deployment_type == DeploymentTypeEnum.SDK
        assert event.job_duration_sec == -1.0
        assert event.num_records_generated == -1
        assert event.num_tokens_generated == -1
        assert event.replace_pii_enabled is False
        assert event.differential_privacy_enabled is False
        assert event.time_series_enabled is False
        assert event.group_by_enabled is False
        assert event.input_records_bucket == "undefined"
        assert event.input_columns_bucket == "undefined"
        assert event.synthetic_quality_score == -1.0
        assert event.data_privacy_score == -1.0
        assert event.model == "undefined"
        assert event.gpu == "undefined"

    def test_all_task_statuses(self):
        for status in TaskStatusEnum:
            event = NSSTrainingAndGenerationEvent(task="run", task_status=status)
            assert event.task_status == status

    def test_all_deployment_types(self):
        for dtype in DeploymentTypeEnum:
            event = NSSTrainingAndGenerationEvent(
                task="run", task_status=TaskStatusEnum.COMPLETED, deployment_type=dtype
            )
            assert event.deployment_type == dtype

    def test_event_name(self):
        assert NSSTrainingAndGenerationEvent._event_name == "train_and_generation_event"

    def test_feature_flags(self):
        event = NSSTrainingAndGenerationEvent(
            task="run",
            task_status=TaskStatusEnum.COMPLETED,
            replace_pii_enabled=True,
            differential_privacy_enabled=True,
            time_series_enabled=True,
            group_by_enabled=True,
        )
        assert event.replace_pii_enabled is True
        assert event.differential_privacy_enabled is True
        assert event.time_series_enabled is True
        assert event.group_by_enabled is True

    def test_metrics(self):
        event = NSSTrainingAndGenerationEvent(
            task="run",
            task_status=TaskStatusEnum.COMPLETED,
            job_duration_sec=42.5,
            num_records_generated=1000,
            num_tokens_generated=50000,
            input_records_bucket=bucket_records(500),
            input_columns_bucket=bucket_columns(8),
            synthetic_quality_score=0.87,
            data_privacy_score=0.95,
            model="meta-llama/Llama-3.1-8B",
            gpu="NVIDIA A100 80GB PCIe",
        )
        assert event.job_duration_sec == 42.5
        assert event.num_records_generated == 1000
        assert event.num_tokens_generated == 50000
        assert event.input_records_bucket == "101-1000"
        assert event.input_columns_bucket == "6-10"
        assert event.synthetic_quality_score == 0.87
        assert event.data_privacy_score == 0.95
        assert event.model == "meta-llama/Llama-3.1-8B"
        assert event.gpu == "NVIDIA A100 80GB PCIe"

    def test_model_dump_by_alias(self):
        event = NSSTrainingAndGenerationEvent(
            task="generate",
            task_status=TaskStatusEnum.COMPLETED,
            replace_pii_enabled=True,
        )
        dumped = event.model_dump(by_alias=True)
        assert "nemoSource" in dumped
        assert "taskStatus" in dumped
        assert "jobDurationSec" in dumped
        assert "numRecordsGenerated" in dumped
        assert "replacePiiEnabled" in dumped
        assert "differentialPrivacyEnabled" in dumped
        assert "timeSeriesEnabled" in dumped
        assert "groupByEnabled" in dumped
        assert "inputRecordsBucket" in dumped
        assert "inputColumnsBucket" in dumped
        assert "syntheticQualityScore" in dumped
        assert "dataPrivacyScore" in dumped
        assert dumped["replacePiiEnabled"] is True
        assert dumped["nemoSource"] == NemoSourceEnum.SAFE_SYNTHESIZER


# =============================================================================
# build_payload
# =============================================================================


class TestBuildPayload:
    def _make_queued(self, task: str = "generate", status: TaskStatusEnum = TaskStatusEnum.COMPLETED) -> QueuedEvent:
        event = NSSTrainingAndGenerationEvent(task=task, task_status=status)
        return QueuedEvent(event=event, timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    def test_structure(self):
        queued = self._make_queued()
        payload = build_payload([queued], source_client_version="1.2.3", session_id="test-session")
        assert payload["clientId"] == "184482118588404"
        assert payload["clientVer"] == "1.2.3"
        assert payload["sessionId"] == "test-session"
        assert len(payload["events"]) == 1

    def test_event_fields(self):
        queued = self._make_queued(task="train", status=TaskStatusEnum.ERROR)
        payload = build_payload([queued], source_client_version="0.0.1")
        event_entry = payload["events"][0]
        assert event_entry["name"] == "train_and_generation_event"
        assert event_entry["ts"] == "2025-01-01T12:00:00.000Z"
        params = event_entry["parameters"]
        assert params["task"] == "train"
        assert params["taskStatus"] == TaskStatusEnum.ERROR
        assert params["nemoSource"] == NemoSourceEnum.SAFE_SYNTHESIZER

    def test_multiple_events(self):
        events = [self._make_queued(task=t) for t in ("train", "generate", "evaluate")]
        payload = build_payload(events, source_client_version="1.0.0")
        assert len(payload["events"]) == 3
        tasks = [e["parameters"]["task"] for e in payload["events"]]
        assert tasks == ["train", "generate", "evaluate"]

    def test_default_session_id(self):
        queued = self._make_queued()
        payload = build_payload([queued], source_client_version="1.0.0")
        assert payload["sessionId"] == "undefined"


# =============================================================================
# TelemetryHandler — telemetry disabled
# =============================================================================


class TestTelemetryDisabled:
    def test_enqueue_noop_when_disabled(self):
        handler = TelemetryHandler()
        event = NSSTrainingAndGenerationEvent(task="generate", task_status=TaskStatusEnum.COMPLETED)
        with patch("nemo_safe_synthesizer.telemetry.TELEMETRY_ENABLED", False):
            handler.enqueue(event)
        assert handler._events == []

    def test_enqueue_noop_for_non_event(self):
        """Silently ignores non-TelemetryEvent objects regardless of TELEMETRY_ENABLED."""
        handler = TelemetryHandler()
        with patch("nemo_safe_synthesizer.telemetry.TELEMETRY_ENABLED", True):
            handler.enqueue("not an event")  # type: ignore[arg-type]
        assert handler._events == []

    def test_telemetry_enabled_env_var(self):
        """Sanity-check: the module-level constant reflects the environment."""
        assert isinstance(TELEMETRY_ENABLED, bool)


# =============================================================================
# TelemetryHandler — enqueue and flush
# =============================================================================


class TestTelemetryHandlerEnqueue:
    def test_enqueue_adds_event(self):
        handler = TelemetryHandler()
        event = NSSTrainingAndGenerationEvent(task="generate", task_status=TaskStatusEnum.COMPLETED)
        with patch("nemo_safe_synthesizer.telemetry.TELEMETRY_ENABLED", True):
            handler.enqueue(event)
        assert len(handler._events) == 1
        assert handler._events[0].event is event

    def test_enqueue_triggers_flush_at_max_queue_size(self):
        handler = TelemetryHandler(max_queue_size=3)
        event = NSSTrainingAndGenerationEvent(task="run", task_status=TaskStatusEnum.COMPLETED)
        with patch("nemo_safe_synthesizer.telemetry.TELEMETRY_ENABLED", True):
            for _ in range(3):
                handler.enqueue(event)
        assert handler._flush_signal.is_set()


# =============================================================================
# TelemetryHandler — send and retry
# =============================================================================


class TestTelemetryHandlerSend:
    def _make_handler(self) -> TelemetryHandler:
        return TelemetryHandler(source_client_version="1.0.0", session_id="s1")

    def _make_queued(self) -> QueuedEvent:
        event = NSSTrainingAndGenerationEvent(task="generate", task_status=TaskStatusEnum.COMPLETED)
        return QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))

    async def test_successful_send_clears_queue(self):
        handler = self._make_handler()
        queued = self._make_queued()
        handler._events.append(queued)

        mock_response = MagicMock(status_code=200, is_success=True)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        await handler._send_events_with_client(mock_client, [queued])
        mock_client.post.assert_awaited_once()
        assert handler._dlq == []

    async def test_500_adds_to_dlq(self):
        handler = self._make_handler()
        queued = self._make_queued()

        mock_response = MagicMock(status_code=500, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        await handler._send_events_with_client(mock_client, [queued])
        assert len(handler._dlq) == 1
        assert handler._dlq[0].retry_count == 1

    async def test_exceeds_max_retries_dropped(self):
        handler = self._make_handler()
        queued = self._make_queued()
        queued.retry_count = handler._max_retries  # already at max

        mock_response = MagicMock(status_code=500, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        await handler._send_events_with_client(mock_client, [queued])
        assert handler._dlq == []  # dropped, not re-queued

    async def test_413_splits_and_retries(self):
        handler = self._make_handler()
        event = NSSTrainingAndGenerationEvent(task="generate", task_status=TaskStatusEnum.COMPLETED)
        events = [
            QueuedEvent(event=event, timestamp=datetime.now(timezone.utc)),
            QueuedEvent(event=event, timestamp=datetime.now(timezone.utc)),
        ]

        success_response = MagicMock(status_code=200, is_success=True)
        too_large_response = MagicMock(status_code=413, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.side_effect = [too_large_response, success_response, success_response]

        await handler._send_events_with_client(mock_client, events)
        assert mock_client.post.await_count == 3  # 1 original + 2 splits

    async def test_session_prefix_applied(self):
        with patch("nemo_safe_synthesizer.telemetry.SESSION_PREFIX", "pfx-"):
            handler = TelemetryHandler(session_id="abc")
        assert handler._session_id == "pfx-abc"
