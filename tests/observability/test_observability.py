# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from unittest import mock
from unittest.mock import MagicMock

import nemo_safe_synthesizer.observability as obs
import pytest
import structlog
from nemo_safe_synthesizer.observability import (
    CategoryFilter,
    CategoryLogger,
    LogCategory,
    NSSObservabilitySettings,
    TracedContext,
    _category_log_processor,
    _convert_rich_table_to_string,
    _current_log_category,
    _initialize_logging,
    _move_category_for_column,
    _render_rich_table,
    _render_table_data_for_console,
    get_logger,
    initialize_observability,
    traced,
)
from rich.table import Table

# =============================================================================
# NSSObservabilitySettings Tests
# =============================================================================


class TestNSSObservabilitySettings:
    """Tests for NSSObservabilitySettings configuration class."""

    def test_default_values_tty(self, monkeypatch):
        """Test that default settings are applied correctly."""
        # we dont' want this test to be affected by the actual terminal being a tty or being run in ci
        with mock.patch("nemo_safe_synthesizer.observability.sys.stdout") as stdout:
            stdout.isatty.return_value = True

            settings = NSSObservabilitySettings()

            assert settings.nss_log_format == "plain"
            assert settings.nss_log_level == "INFO"
            assert settings.otel_service_name == "nemo-safe-synthesizer"

    def test_default_values_no_tty(self, monkeypatch):
        """Test that default settings are applied correctly."""
        # we dont' want this test to be affected by the actual terminal being a tty or being run in ci
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
        assert "95.000%" in result  # Formatted as percentage

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

    def test_creates_filtered_extra_display(self, mock_logger):
        """Test that _extra_display is created without table keys."""
        event_dict = {
            "event": "test",
            "extra": {
                "ctx": {"count": 100},
                "other_key": "other_value",
            },
        }

        result = _render_table_data_for_console(mock_logger, "info", event_dict)

        assert "_extra_display" in result
        assert "other_key" in result["_extra_display"]
        assert "count" not in result["_extra_display"]

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
        # Use monkeypatch.setattr for automatic cleanup of module state
        monkeypatch.setattr(obs, "_INITIALIZED_OBSERVABILITY", False)
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


class TestGetLogger:
    """Tests for get_logger function."""

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
