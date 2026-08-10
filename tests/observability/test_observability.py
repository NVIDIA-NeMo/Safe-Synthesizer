# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import json
import logging
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
import structlog
from rich.table import Table

from nemo_safe_synthesizer.observability import (
    CategoryFilter,
    CategoryLogger,
    LogCategory,
    NSSObservabilitySettings,
    TracedContext,
    _canonicalize_log_event,
    _category_log_processor,
    _convert_rich_table_to_string,
    _current_log_category,
    _initialize_logging,
    _move_category_for_column,
    _render_rich_table,
    _render_table_data_for_console,
    get_logger,
    heartbeat,
    initialize_observability,
    traced,
)

obs = importlib.import_module("nemo_safe_synthesizer.observability")

# =============================================================================
# NSSObservabilitySettings Tests
# =============================================================================


class TestNSSObservabilitySettings:
    """Tests for NSSObservabilitySettings configuration class."""

    @staticmethod
    def _clear_nss_log_env(monkeypatch: pytest.MonkeyPatch) -> None:
        """Defaults must not inherit ``NSS_LOG_*`` left by other tests on xdist workers."""
        for name in ("NSS_LOG_LEVEL", "NSS_LOG_FORMAT", "NSS_LOG_FILE", "NSS_LOG_COLOR"):
            monkeypatch.delenv(name, raising=False)

    def test_default_values_tty(self, monkeypatch):
        """Test that default settings are applied correctly."""
        # we dont' want this test to be affected by the actual terminal being a tty or being run in ci
        self._clear_nss_log_env(monkeypatch)
        with mock.patch("nemo_safe_synthesizer.observability.sys.stdout") as stdout:
            stdout.isatty.return_value = True

            settings = NSSObservabilitySettings()

            assert settings.nss_log_format == "plain"
            assert settings.nss_log_level == "INFO"
            assert settings.otel_service_name == "nemo-safe-synthesizer"

    def test_default_values_no_tty(self, monkeypatch):
        """Test that default settings are applied correctly."""
        # we dont' want this test to be affected by the actual terminal being a tty or being run in ci
        self._clear_nss_log_env(monkeypatch)
        with mock.patch("nemo_safe_synthesizer.observability.sys.stdout") as stdout:
            stdout.isatty.return_value = False

            settings = NSSObservabilitySettings()

            assert settings.nss_log_format == "json"
            assert settings.nss_log_level == "INFO"
            assert settings.otel_service_name == "nemo-safe-synthesizer"

    def test_env_var_override_log_format(self, monkeypatch):
        """Test that NSS_LOG_FORMAT env var overrides default."""
        monkeypatch.setenv("NSS_LOG_FORMAT", "plain")
        settings = NSSObservabilitySettings()
        assert settings.nss_log_format == "plain"

    def test_env_var_override_log_level(self, monkeypatch):
        """Test that NSS_LOG_LEVEL env var overrides default."""
        monkeypatch.setenv("NSS_LOG_LEVEL", "DEBUG")
        settings = NSSObservabilitySettings()
        assert settings.nss_log_level == "DEBUG"

    def test_env_var_override_log_file(self, monkeypatch, tmp_path):
        """Test that NSS_LOG_FILE env var overrides default."""
        log_file = str(tmp_path / "test.log")
        monkeypatch.setenv("NSS_LOG_FILE", log_file)
        settings = NSSObservabilitySettings()
        assert settings.nss_log_file == log_file

    @pytest.mark.parametrize("explicit_value", [False, True])
    @pytest.mark.parametrize("is_tty", [False, True])
    def test_explicit_log_color_bool_overrides_tty_default(self, explicit_value, is_tty):
        """Explicit boolean log-color settings should not be recomputed from stdout."""
        with mock.patch("nemo_safe_synthesizer.observability.sys.stdout") as stdout:
            stdout.isatty.return_value = is_tty

            settings = NSSObservabilitySettings(nss_log_color=explicit_value)

        assert settings.nss_log_color is explicit_value


class TestCategoryFilter:
    """Tests for CategoryFilter logging filter."""

    def test_filter_allows_all_when_no_categories_specified(self):
        """Test that filter allows all logs when include_categories is None."""
        filter_obj = CategoryFilter(include_categories=None)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is True

    def test_filter_allows_matching_category(self):
        """Test that filter allows logs with matching category."""
        filter_obj = CategoryFilter(include_categories={LogCategory.USER})
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.category = LogCategory.USER.value
        assert filter_obj.filter(record) is True

    def test_filter_blocks_non_matching_category(self):
        """Test that filter blocks logs with non-matching category."""
        filter_obj = CategoryFilter(include_categories={LogCategory.USER})
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.category = LogCategory.RUNTIME.value
        assert filter_obj.filter(record) is False


class TestCategoryLogger:
    """Tests for CategoryLogger wrapper class."""

    @pytest.fixture
    def mock_base_logger(self):
        """Create a mock base logger without spec to allow all method calls."""
        logger = MagicMock()
        logger.name = "test_logger"
        # LoggerAdapter checks isEnabledFor before calling underlying methods
        logger.isEnabledFor.return_value = True
        return logger

    def test_debug_method(self, mock_base_logger):
        """Test debug method delegates to underlying logger via log()."""
        category_logger = CategoryLogger(mock_base_logger)
        category_logger.debug("test message")
        # LoggerAdapter calls self.logger.log() internally
        mock_base_logger.log.assert_called()
        call_args = mock_base_logger.log.call_args
        assert call_args[0][0] == logging.DEBUG
        assert call_args[0][1] == "test message"

    def test_info_method(self, mock_base_logger):
        """Test info method delegates to underlying logger via log()."""
        category_logger = CategoryLogger(mock_base_logger)
        category_logger.info("test message")
        mock_base_logger.log.assert_called()
        call_args = mock_base_logger.log.call_args
        assert call_args[0][0] == logging.INFO
        assert call_args[0][1] == "test message"

    def test_warning_method(self, mock_base_logger):
        """Test warning method delegates to underlying logger via log()."""
        category_logger = CategoryLogger(mock_base_logger)
        category_logger.warning("test message")
        mock_base_logger.log.assert_called()
        call_args = mock_base_logger.log.call_args
        assert call_args[0][0] == logging.WARNING
        assert call_args[0][1] == "test message"

    def test_error_method(self, mock_base_logger):
        """Test error method delegates to underlying logger via log()."""
        category_logger = CategoryLogger(mock_base_logger)
        category_logger.error("test message")
        mock_base_logger.log.assert_called()
        call_args = mock_base_logger.log.call_args
        assert call_args[0][0] == logging.ERROR
        assert call_args[0][1] == "test message"

    def test_critical_method(self, mock_base_logger):
        """Test critical method delegates to underlying logger via log()."""
        category_logger = CategoryLogger(mock_base_logger)
        category_logger.critical("test message")
        mock_base_logger.log.assert_called()
        call_args = mock_base_logger.log.call_args
        assert call_args[0][0] == logging.CRITICAL
        assert call_args[0][1] == "test message"

    def test_exception_method(self, mock_base_logger):
        """Test exception method delegates to underlying logger via log()."""
        category_logger = CategoryLogger(mock_base_logger)
        category_logger.exception("test message")
        # exception calls log with ERROR level and exc_info=True
        mock_base_logger.log.assert_called()
        call_args = mock_base_logger.log.call_args
        assert call_args[0][0] == logging.ERROR
        assert call_args[0][1] == "test message"

    def test_log_method(self, mock_base_logger):
        """Test log method delegates to underlying logger."""
        category_logger = CategoryLogger(mock_base_logger)
        category_logger.log(logging.INFO, "test message")
        mock_base_logger.log.assert_called()

    def test_is_enabled_for(self, mock_base_logger):
        """Test isEnabledFor delegates to base logger."""
        mock_base_logger.isEnabledFor.return_value = True
        category_logger = CategoryLogger(mock_base_logger)
        assert category_logger.isEnabledFor(logging.DEBUG) is True
        mock_base_logger.isEnabledFor.assert_called_with(logging.DEBUG)


class TestCategoryLogProcessor:
    """Tests for _category_log_processor."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for processor tests."""
        return MagicMock(spec=logging.Logger)

    def test_adds_category_from_contextvar(self, mock_logger):
        """Test processor adds category from contextvar."""
        _current_log_category.set(LogCategory.USER.value)
        event_dict = {"event": "test"}

        result = _category_log_processor(mock_logger, "info", event_dict)

        assert result["category"] == LogCategory.USER.value
        assert _current_log_category.get() is None  # Reset after use

    def test_canonicalizes_native_and_foreign_structured_context(self, mock_logger):
        native = _canonicalize_log_event(mock_logger, "info", {"extra": {"ctx": {"rows": 1}}})
        foreign = _canonicalize_log_event(mock_logger, "info", {"ctx": {"rows": 1}})

        assert native == {"context": {"rows": 1}}
        assert foreign == {"context": {"rows": 1}}


class TestMoveCategoryForColumn:
    """Tests for _move_category_for_column processor."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for processor tests."""
        return MagicMock(spec=logging.Logger)

    def test_moves_category_to_display_key(self, mock_logger):
        """Test that category is moved to _category_display."""
        event_dict = {"event": "test", "category": LogCategory.USER.value}

        result = _move_category_for_column(mock_logger, "info", event_dict)

        assert "category" not in result
        assert result["_category_display"] == LogCategory.USER.value


class TestRenderRichTable:
    """Tests for _render_rich_table function."""

    def test_renders_flat_dict(self):
        """Test rendering a flat key-value dictionary."""
        data = {"count": 100, "rate": 0.95}
        result = _render_rich_table(data)

        assert "Count" in result
        assert "100" in result
        assert "Rate" in result
        assert "95.00%" in result  # Formatted as percentage

    def test_sec_suffix_not_formatted_as_percentage(self):
        """Keys ending in _sec or _seconds are rendered as plain floats, not percentages."""
        data = {"tokenization_overhead_sec": 0.35, "total_seconds": 0.72, "rate": 0.95}
        result = _render_rich_table(data)

        assert "0.35" in result
        assert "0.72" in result
        assert "95.00%" in result

    def test_fraction_fields_render_as_percentages_when_generation_overshoots(self):
        result = _render_rich_table({"progress_fraction": 1.25})

        assert "125.00%" in result

    def test_renders_mapping_values_without_python_repr(self):
        """Mapping values in flat tables are rendered as compact key-value lists."""
        data = {"num_prompts": 10, "finish_reasons": {"length": 10, "stop": 2}}
        result = _render_rich_table(data)

        assert "Finish Reasons" in result
        assert "length: 10, stop: 2" in result
        assert "{'length': 10" not in result

    def test_renders_nested_dict(self):
        """Test rendering a nested statistics dictionary."""
        data = {
            "col1": {"min": 1, "max": 10},
            "col2": {"min": 5, "max": 20},
        }
        result = _render_rich_table(data)

        assert "Col1" in result
        assert "Col2" in result
        assert "min" in result
        assert "max" in result

    def test_renders_with_title(self):
        """Test rendering with custom title."""
        data = {"value": 42}
        result = _render_rich_table(data, title="Custom Title")

        assert "Custom Title" in result

    def test_converts_rich_table_to_string(self):
        """Test converting a Rich Table to a string."""
        rich_table = Table(title="Custom Title")
        rich_table.add_column("Metric", style="bold")
        rich_table.add_column("Value")
        rich_table.add_row("Count", "100")
        result = _convert_rich_table_to_string(rich_table)
        assert "Custom Title" in result
        assert "Count" in result


class TestRenderTableDataForConsole:
    """Tests for _render_table_data_for_console processor."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for processor tests."""
        return MagicMock(spec=logging.Logger)

    def test_renders_ctx_key(self, mock_logger):
        """Test that ctx key is rendered as table."""
        event_dict = {
            "event": "test",
            "extra": {
                "ctx": {"key1": "value1", "key2": "value2"},
            },
        }

        result = _render_table_data_for_console(mock_logger, "info", event_dict)

        assert "key1" in result["extra"]["ctx"]
        assert "value1" == result["extra"]["ctx"]["key1"]

    def test_creates_filtered_extra_display_without_ctx(self, mock_logger):
        """Plain output keeps ordinary extras without repeating structured context."""
        event_dict = {
            "event": "test",
            "extra": {
                "ctx": {"count": 100},
                "other_key": "other_value",
            },
        }

        result = _render_table_data_for_console(mock_logger, "info", event_dict)

        assert "_extra_display" in result
        assert result["_extra_display"] == {"other_key": "other_value"}
        assert result["extra"]["ctx"] == {"count": 100}

    def test_handles_empty_event_dict(self, mock_logger):
        """Test handling of event dict without table data."""
        event_dict = {"event": "test"}

        result = _render_table_data_for_console(mock_logger, "info", event_dict)

        assert result["event"] == "test"
        assert "_extra_display" not in result


class TestTracedContext:
    """Tests for TracedContext class."""

    def test_requires_name(self):
        """Test that TracedContext requires a name."""
        with pytest.raises(ValueError, match="name is required"):
            TracedContext(name="")

    def test_as_decorator(self):
        """Test TracedContext as a decorator."""
        call_count = 0

        @TracedContext(name="test_operation", log_entry=False, log_exit=False)
        def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        result = test_func()
        assert result == "result"
        assert call_count == 1

    def test_as_context_manager(self):
        """Test TracedContext as a context manager."""
        executed = False

        with TracedContext(name="test_operation", log_entry=False, log_exit=False):
            executed = True

        assert executed is True

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @TracedContext(name="op", log_entry=False, log_exit=False)
        def my_function():
            """My docstring."""
            pass

        assert getattr(my_function, "__name__", None) == "my_function"
        assert getattr(my_function, "__doc__", None) == """My docstring."""

    def test_decorator_handles_exception(self):
        """Test that decorator handles and re-raises exceptions."""

        @TracedContext(name="failing_op", log_entry=False, log_exit=False)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

    def test_context_manager_handles_exception(self):
        """Test that context manager handles and re-raises exceptions."""
        with pytest.raises(ValueError, match="Test error"):
            with TracedContext(name="failing_op", log_entry=False, log_exit=False):
                raise ValueError("Test error")

    def test_context_manager_returns_self(self):
        """Test that context manager returns self on __enter__."""
        ctx = TracedContext(name="test_op", log_entry=False, log_exit=False)
        with ctx as returned:
            assert returned is ctx

    def test_default_category_is_runtime(self):
        """Test that default category is RUNTIME."""
        ctx = TracedContext(name="test")
        assert ctx.category == LogCategory.RUNTIME

    def test_custom_category(self):
        """Test setting a custom category."""
        ctx = TracedContext(name="test", category=LogCategory.USER)
        assert ctx.category == LogCategory.USER


class TestTracedHelpers:
    """Tests for traced helper functions."""

    def test_traced_returns_traced_context(self):
        """Test that traced() returns a TracedContext."""
        ctx = traced(name="test_op")
        assert isinstance(ctx, TracedContext)
        assert ctx.name == "test_op"
        assert ctx.category == LogCategory.RUNTIME

    def test_traced_passes_kwargs(self):
        """Test that traced passes through kwargs."""
        ctx = traced(name="test", log_entry=False, log_exit=False, record_duration=False)
        assert ctx.log_entry is False
        assert ctx.log_exit is False
        assert ctx.record_duration is False


class TestInitializeObservability:
    """Tests for initialize_observability function."""

    def test_idempotent_initialization(self):
        """Test that initialize_observability is idempotent."""
        # Reset state
        obs._INITIALIZED_OBSERVABILITY = False

        # First call should initialize
        initialize_observability()
        assert obs._INITIALIZED_OBSERVABILITY is True

        # Second call should not re-initialize (no error)
        initialize_observability()
        assert obs._INITIALIZED_OBSERVABILITY is True

    def test_initialize_logging_configures_json_format(self, monkeypatch, capsys, tmp_path):
        """Test that initialize_logging() configures JSON format when NSS_LOG_FORMAT=json."""
        # Keep the initialization flag aligned with structlog.reset_defaults()
        # during cleanup; monkeypatch would restore a stale True value.
        obs._INITIALIZED_OBSERVABILITY = False
        monkeypatch.setenv("NSS_LOG_FORMAT", "json")
        monkeypatch.setenv("NSS_LOG_LEVEL", "INFO")
        monkeypatch.setenv("NSS_LOG_FILE", str(tmp_path / "test.log"))
        monkeypatch.setattr(obs, "SETTINGS", NSSObservabilitySettings())

        # Clear handlers and reset structlog for clean test
        structlog.reset_defaults()
        logging.getLogger().handlers.clear()

        # Initialize logging and log a message
        _initialize_logging()
        logger = get_logger("test_json_format")
        logger.info("Test JSON message")

        # Check stdout for JSON output
        captured = capsys.readouterr()
        if captured.out.strip():
            for line in captured.out.strip().split("\n"):
                if line.strip():
                    parsed = json.loads(line)
                    assert "message" in parsed or "event" in parsed

    def test_plain_console_table_does_not_leak_into_json_file(self, monkeypatch, capsys, tmp_path):
        """Console-only table rendering preserves the canonical JSONL event."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level
        log_file = tmp_path / "events.jsonl"

        obs._INITIALIZED_OBSERVABILITY = False
        monkeypatch.setenv("NSS_LOG_FORMAT", "plain")
        monkeypatch.setenv("NSS_LOG_LEVEL", "INFO")
        monkeypatch.setenv("NSS_LOG_COLOR", "false")
        monkeypatch.setenv("NSS_LOG_FILE", str(log_file))
        monkeypatch.setattr(obs, "SETTINGS", NSSObservabilitySettings())
        structlog.reset_defaults()
        root_logger.handlers.clear()

        try:
            _initialize_logging()
            obs._INITIALIZED_OBSERVABILITY = True
            logger = get_logger("test_mixed_handlers")
            logger.user.info(
                "Mixed handler table",
                extra={"ctx": {"render_table": True, "tabular_data": {"records": 2}, "title": "Dogfood table"}},
            )
            logging.getLogger("test_foreign_handlers").info(
                "Foreign %s interpolation",
                "logger",
                extra={"ctx": {"source": "foreign"}},
            )

            console_output = capsys.readouterr().out
            native_record, foreign_record = [
                json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()
            ]
        finally:
            structlog.reset_defaults()
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)
            obs._INITIALIZED_OBSERVABILITY = False

        assert console_output.count("Dogfood table") == 1
        assert "Mixed handler table" in console_output
        assert native_record["message"] == "Mixed handler table"
        assert native_record["category"] == LogCategory.USER.value
        assert native_record["context"] == {
            "render_table": True,
            "tabular_data": {"records": 2},
            "title": "Dogfood table",
        }
        assert native_record["timestamp"] in console_output
        assert native_record["logger"] == "test_mixed_handlers"
        assert native_record["filename"] == Path(__file__).name
        assert native_record["lineno"] is not None
        assert native_record["qual_name"] is not None
        assert "_category_display" not in native_record
        assert "positional_args" not in native_record
        assert "+" not in native_record["message"]
        assert foreign_record["message"] == "Foreign logger interpolation"
        assert foreign_record["category"] == LogCategory.RUNTIME.value
        assert foreign_record["context"] == {"source": "foreign"}
        assert foreign_record["logger"] == "test_foreign_handlers"
        assert "positional_args" not in foreign_record


class TestGetLogger:
    """Tests for get_logger function."""

    @pytest.fixture
    def callsite_test_logging(self, monkeypatch, caplog):
        """Initialize observability while preserving caplog capture.

        Leaves ``_INITIALIZED_OBSERVABILITY`` False and structlog reset on
        teardown so the global state is internally consistent. Tests that need
        an initialized observability stack (e.g. ``TestObservabilityIntegration``)
        re-initialize via their own autouse fixture.
        """
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level

        # Manage _INITIALIZED_OBSERVABILITY manually rather than through
        # monkeypatch.setattr. monkeypatch restores the pre-test value AFTER
        # this fixture's teardown runs, which would re-flip the flag back to
        # True (set by earlier tests) while structlog has just been reset to
        # defaults. That mismatch causes get_logger() to return a structlog
        # BoundLoggerFilteringAtNotset wrapped in a stdlib LoggerAdapter, which
        # blows up on isEnabledFor() in subsequent tests.
        obs._INITIALIZED_OBSERVABILITY = False
        monkeypatch.setenv("NSS_LOG_FORMAT", "plain")
        monkeypatch.setenv("NSS_LOG_LEVEL", "INFO")
        monkeypatch.setenv("NSS_LOG_COLOR", "false")
        monkeypatch.delenv("NSS_LOG_FILE", raising=False)
        monkeypatch.setattr(obs, "SETTINGS", NSSObservabilitySettings())
        structlog.reset_defaults()
        root_logger.handlers.clear()

        yield

        structlog.reset_defaults()
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        obs._INITIALIZED_OBSERVABILITY = False

    @staticmethod
    def _record_callsite(record: logging.LogRecord) -> tuple[str, int]:
        match record.msg:
            case {"filename": filename, "lineno": lineno}:
                return str(filename), int(lineno)
            case _:
                return record.filename, record.lineno

    @staticmethod
    def _assert_log_record_callsite(
        caplog,
        *,
        message: str,
        expected_filename: str,
        expected_lineno: int,
    ) -> None:
        records = [record for record in caplog.records if message in record.getMessage()]
        assert len(records) == 1

        filename, lineno = TestGetLogger._record_callsite(records[0])
        assert filename == expected_filename
        assert lineno == expected_lineno

    def test_returns_category_logger_by_default(self):
        """Test that get_logger returns CategoryLogger by default."""
        logger = get_logger("test_module")
        assert isinstance(logger, CategoryLogger)

    def test_uses_provided_name_when_not_INITIALIZED_OBSERVABILITY(self, monkeypatch):
        """Test that logger uses provided name when logging is not initialized."""
        monkeypatch.setattr(obs, "_INITIALIZED_OBSERVABILITY", False)

        logger = get_logger("my_custom_name")
        # When not initialized, we get a stdlib logger
        assert logger.name == "my_custom_name"

    def test_uses_provided_name_when_INITIALIZED_OBSERVABILITY(self):
        """Test that logger uses provided name when logging is initialized."""
        # Ensure logging is initialized
        if not obs._INITIALIZED_OBSERVABILITY:
            initialize_observability()

        logger = get_logger("my_INITIALIZED_OBSERVABILITY_name")
        # When initialized, we still get the correct name
        assert logger.name == "my_INITIALIZED_OBSERVABILITY_name"

    def test_returns_stdlib_logger_when_not_INITIALIZED_OBSERVABILITY(self, monkeypatch):
        """Test that get_logger returns basic stdlib logger when logging hasn't been initialized.

        This ensures the package can be used as a library without taking over the
        parent application's logging configuration.
        """
        monkeypatch.setattr(obs, "_INITIALIZED_OBSERVABILITY", False)

        logger = get_logger("test_library_mode")

        # Should be a CategoryLogger wrapping a stdlib logger, not structlog
        assert isinstance(logger, CategoryLogger)
        assert isinstance(logger._logger, logging.Logger)
        # stdlib loggers have 'handlers' attribute, structlog BoundLoggers don't
        assert hasattr(logger._logger, "handlers")

    def test_does_not_clobber_existing_handlers_when_not_INITIALIZED_OBSERVABILITY(self, monkeypatch):
        """Test that get_logger() doesn't modify root logger when not initialized.

        When used as a library (without calling initialize_logging()), the package
        should not add handlers or change the logging configuration.
        """
        monkeypatch.setattr(obs, "_INITIALIZED_OBSERVABILITY", False)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level

        # Get a logger without initialization
        logger = get_logger("test_no_clobber")
        logger.info("Test message")

        # Root logger configuration should be unchanged
        assert root_logger.handlers == original_handlers
        assert root_logger.level == original_level

    def test_logger_created_before_observability_configuration_reports_real_callsite(
        self,
        callsite_test_logging,
        caplog,
    ):
        """Logger created before configuration still reports the caller filename."""
        logger = get_logger("test_callsite_before_configuration")

        initialize_observability()
        root_logger = logging.getLogger()
        if caplog.handler not in root_logger.handlers:
            root_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger=logger.name)

        message = "logger before configuration callsite"
        frame = inspect.currentframe()
        assert frame is not None
        expected_lineno = frame.f_lineno + 1
        logger.info(message)

        self._assert_log_record_callsite(
            caplog,
            message=message,
            expected_filename=Path(__file__).name,
            expected_lineno=expected_lineno,
        )

    def test_logger_created_after_observability_configuration_reports_real_callsite(
        self,
        callsite_test_logging,
        caplog,
    ):
        """Logger created after configuration still reports the caller filename."""
        initialize_observability()
        root_logger = logging.getLogger()
        if caplog.handler not in root_logger.handlers:
            root_logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO)

        logger = get_logger("test_callsite_after_configuration")
        message = "logger after configuration callsite"
        frame = inspect.currentframe()
        assert frame is not None
        expected_lineno = frame.f_lineno + 1
        logger.info(message)

        self._assert_log_record_callsite(
            caplog,
            message=message,
            expected_filename=Path(__file__).name,
            expected_lineno=expected_lineno,
        )


class TestObservabilityIntegration:
    """Integration tests for observability components.

    These tests require logging to be initialized, matching the new opt-in behavior
    where entry points must explicitly call initialize_logging().
    """

    @pytest.fixture(autouse=True)
    def ensure_logging_INITIALIZED_OBSERVABILITY(self):
        """Ensure logging is initialized for integration tests."""
        # Initialize logging if not already done
        if not obs._INITIALIZED_OBSERVABILITY:
            initialize_observability()

    def test_category_logger_with_real_logging(self, caplog):
        """Test CategoryLogger with actual log capture."""
        caplog.set_level(logging.DEBUG)

        logger = get_logger("test_integration")
        assert isinstance(logger, CategoryLogger)
        logger.user.info("User message")
        logger.runtime.debug("Runtime message")

        # Verify logs were captured
        assert "User message" in caplog.text

    def test_native_logger_interpolates_printf_arguments(self, caplog):
        caplog.set_level(logging.INFO)

        logger = get_logger("test_native_printf")
        logger.info("graded %s attack evaluations", 360)

        assert "graded 360 attack evaluations" in caplog.text
        assert "positional_args" not in caplog.text

    def test_traced_decorator_logs_entry_exit(self, caplog):
        """Test that traced decorator logs entry and exit."""
        caplog.set_level(logging.DEBUG)

        @traced("test_traced_op")
        def my_traced_function():
            return 42

        result = my_traced_function()
        assert result == 42

        # Entry and exit should be logged
        assert "Entering test_traced_op" in caplog.text
        assert "Exiting test_traced_op" in caplog.text

    def test_traced_context_manager_logs_entry_exit(self, caplog):
        """Test that traced context manager logs entry and exit."""
        caplog.set_level(logging.DEBUG)

        with traced("ctx_test_op"):
            pass

        assert "Entering ctx_test_op" in caplog.text
        assert "Exiting ctx_test_op" in caplog.text

    def test_traced_records_duration(self, caplog):
        """Test that traced records duration."""
        caplog.set_level(logging.DEBUG)

        @traced("duration_test", record_duration=True)
        def slow_function():
            time.sleep(0.01)  # 10ms

        slow_function()

        # Should log duration
        assert "duration_ms" in caplog.text

    def test_traced_logs_errors(self, caplog):
        """Test that traced logs errors on exception."""
        caplog.set_level(logging.DEBUG)

        @traced("error_test")
        def failing_function():
            raise RuntimeError("Test failure")

        with pytest.raises(RuntimeError):
            failing_function()

        assert "Error in error_test" in caplog.text
        assert "RuntimeError" in caplog.text


class TestHeartbeat:
    """Tests for the heartbeat context manager."""

    def test_heartbeat_logs_completion(self, caplog):
        caplog.set_level(logging.INFO)
        with heartbeat("Test op", interval=0.05):
            time.sleep(0.01)

        assert "Test op complete" in caplog.text

    def test_heartbeat_logs_progress_on_long_operation(self, caplog):
        caplog.set_level(logging.INFO)
        with heartbeat("Slow op", interval=0.05):
            time.sleep(0.5)

        assert "Slow op in progress" in caplog.text
        assert "Slow op complete" in caplog.text

    def test_heartbeat_progress_note_on_periodic_logs_only(self, caplog):
        caplog.set_level(logging.INFO)
        message = "Generation"
        progress_note = "Records update only after each batch finishes."
        with heartbeat(message, interval=0.05, progress_note=progress_note):
            time.sleep(0.15)

        assert f"{message} in progress. {progress_note}" in caplog.text
        assert f"{message} complete" in caplog.text
        for record in caplog.records:
            if "complete" in record.getMessage():
                assert progress_note not in record.getMessage()

    def test_heartbeat_includes_extra_fields(self, caplog):
        caplog.set_level(logging.INFO)
        with heartbeat("Loading", interval=0.05, model="test-model"):
            time.sleep(0.2)

        # Extra fields appear on record.ctx (plain logging) or get
        # merged into the structlog event dict (when structlog is
        # initialized). Check both paths.
        has_field = any(
            getattr(r, "ctx", {}).get("model") == "test-model" or "test-model" in getattr(r, "message", r.getMessage())
            for r in caplog.records
        )
        assert has_field, f"model field not found in records: {caplog.text}"

    def test_heartbeat_logs_elapsed_seconds(self, caplog):
        caplog.set_level(logging.INFO)
        with heartbeat("Timed op", interval=0.05):
            time.sleep(0.2)

        has_elapsed = any(
            "elapsed_seconds" in getattr(r, "ctx", {}) or "elapsed_seconds" in getattr(r, "message", r.getMessage())
            for r in caplog.records
        )
        assert has_elapsed, f"elapsed_seconds not found in records: {caplog.text}"

    def test_heartbeat_logs_failure_on_exception(self, caplog):
        def fail() -> None:
            raise RuntimeError("boom")

        caplog.set_level(logging.INFO)
        with pytest.raises(RuntimeError):
            with heartbeat("Failing op", interval=60.0):
                fail()

        assert "Failing op failed" in caplog.text
        assert "Failing op complete" not in caplog.text
        has_error_type = any(
            getattr(r, "ctx", {}).get("error_type") == "RuntimeError"
            or "'error_type': 'RuntimeError'" in getattr(r, "message", r.getMessage())
            for r in caplog.records
        )
        assert has_error_type, f"error_type not found in records: {caplog.text}"
