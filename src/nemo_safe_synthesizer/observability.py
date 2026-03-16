# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability for Safe Synthesizer.

Provides structured logging with category support for batch/CLI operations.

Log categories:

  - RUNTIME: internal operational details (memory, timings, debug info)
  - USER: user-relevant progress and results
  - SYSTEM: system-level events (startup, shutdown, config)
  - BACKEND: logs from dependencies

Configure via environment variables:

  - ``NSS_LOG_FORMAT``: ``"json"`` or ``"plain"`` (default: auto-detect from tty)
  - ``NSS_LOG_LEVEL``: ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``,
    ``"DEBUG_DEPENDENCIES"``, or ``"DEBUG"`` (default: ``"INFO"``)
  - ``NSS_LOG_FILE``: path to file for JSON logs (optional)
  - ``OTEL_SERVICE_NAME``: OpenTelemetry service name
    (default: ``"nemo-safe-synthesizer"``)

Logging is NOT auto-initialized on import. Entry points (CLI, scripts) must
call ``initialize_observability()`` first. When used as a library,
``get_logger()`` returns basic stdlib loggers that integrate with the parent
application's logging configuration.
"""

import contextvars
import inspect
import logging
import os
import sys
import time
import warnings
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

import colorama

if TYPE_CHECKING:
    from .cli.artifact_structure import Workdir
import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from rich import box
from rich.console import Console
from rich.table import Table
from structlog.processors import JSONRenderer
from structlog.typing import EventDict

P = ParamSpec("P")
R = TypeVar("R")


__all__ = [
    "NSSObservabilitySettings",
    "LogCategory",
    "CategoryLogger",
    "CategoryFilter",
    "initialize_observability",
    "configure_logging_from_workdir",
    "get_logger",
    "TracedContext",
    "traced",
    "traced_user",
    "traced_runtime",
    "traced_system",
    "traced_backend",
]


verbosity_mapping = {
    "ERROR": (logging.ERROR, logging.ERROR),
    "CRITICAL": (logging.CRITICAL, logging.CRITICAL),
    "WARNING": (logging.WARNING, logging.ERROR),
    "INFO": (logging.INFO, logging.WARNING),
    "DEBUG": (logging.DEBUG, logging.INFO),
    "DEBUG_DEPENDENCIES": (logging.DEBUG, logging.DEBUG),
}

PACKAGES_TO_SET_TO_WARN = [
    "accelerate",
    "arrow",
    "bitsandbytes",
    "datasets",
    "faiss",
    "httpx",
    "matplotlib",
    "nvidia",
    "numba",
    "peft",
    "torch",
    "torch.distributed",
    "torchao",
    "transformers",
    "unsloth",
    "vllm",
    "wandb",
]


class NSSObservabilitySettings(BaseSettings):
    """Logging configuration read from environment variables or CLI flags."""

    # NSS-specific settings
    nss_log_format: Literal["json", "plain"] | None = None
    nss_log_level: Literal["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG_DEPENDENCIES", "DEBUG"] = "INFO"
    nss_log_file: str | None = Field(default=None, validation_alias="NSS_LOG_FILE")
    nss_log_color: bool = True
    otel_service_name: str = "nemo-safe-synthesizer"
    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}

    @field_validator("nss_log_format", mode="before")
    @classmethod
    def set_log_format_default(cls, value: Any) -> Literal["json", "plain"]:
        """Set nss_log_format default based on whether stdout is a tty at instantiation time."""
        match value:
            case str():
                return value.lower()
            case _:
                return "plain" if sys.stdout.isatty() else "json"

    @field_validator("nss_log_color", mode="before")
    @classmethod
    def set_log_color_default(cls, value: Any) -> bool:
        """Set nss_log_color default based on whether stdout is a tty at instantiation time."""
        match value:
            case str():
                return value.lower() == "true"
            case _ if sys.stdout.isatty():
                warnings.warn("stdout is a tty, setting nss_log_color to True", UserWarning)
                return True
            case _:
                warnings.warn("stdout is not a tty, setting nss_log_color to False", UserWarning)
                return False


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    SETTINGS: NSSObservabilitySettings = NSSObservabilitySettings()


class LogCategory(str, Enum):
    """Categories for log messages."""

    RUNTIME = "runtime"  # Internal operational details (memory, timings, debug)
    USER = "user"  # User-relevant progress and results
    SYSTEM = "system"  # System-level events (startup, shutdown, config)
    BACKEND = "backend"  # logs from dependencies


# Contextvar to pass category from LoggerAdapter to processor without going through 'extra'
_current_log_category: contextvars.ContextVar[str | None] = contextvars.ContextVar(  # ty: ignore[invalid-assignment]
    "_current_log_category", default=None
)


def _category_log_processor(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Ensures all logs have a category, defaulting to RUNTIME."""
    # Read category from contextvar (set by _CategoryLogAdapter)
    category = _current_log_category.get()
    if category is not None:
        event_dict["category"] = category
        # Reset the contextvar after use
        _current_log_category.set(None)
    elif "category" not in event_dict:
        event_dict["category"] = LogCategory.RUNTIME.value
    return event_dict


def _move_category_for_column(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Move category to a render-only key so it doesn't appear in the default column output."""
    # Pop 'category' and store in '_category_display' for the column to use
    # This ensures 'category' won't be rendered by the default column
    category = event_dict.pop("category", LogCategory.RUNTIME.value)
    event_dict["_category_display"] = category
    return event_dict


def _render_rich_table(data: dict, title: str | None = None) -> str:
    """Render a dictionary as a Rich ASCII table string."""
    table = Table(title=title, box=box.ASCII)

    # Check if this is nested stats data (e.g., {"col1": {"min": 1, "max": 2, ...}})
    first_value = next(iter(data.values()), None)
    if isinstance(first_value, dict):
        # Nested format: render as a statistics table
        table.add_column("", style="bold")
        for key in data.keys():
            table.add_column(key.replace("_", " ").title(), justify="right")

        # Get all stat keys from the first item
        stat_keys = list(first_value.keys()) if first_value else []
        for stat_key in stat_keys:
            row_values = [str(data[col].get(stat_key, "")) for col in data.keys()]
            table.add_row(stat_key, *row_values)
    else:
        # Flat format: render as key-value pairs
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        for key, value in data.items():
            display_key = key.replace("_", " ").title()
            # TODO: Refactor this formatting logic to be more generic and maintainable.
            # Currently requires updating this file whenever new metrics are added that
            # need special formatting (e.g., the "loss", "eval_loss" exclusion list).
            if isinstance(value, float):
                if key not in ("loss", "eval_loss") and value < 1 and value > 0:
                    display_value = f"{value:.3%}"
                else:
                    display_value = f"{value:.3f}"
            else:
                display_value = str(value)
            table.add_row(display_key, display_value)

    return _convert_rich_table_to_string(table)


def _convert_rich_table_to_string(rich_table: Table) -> str:
    """Convert a Rich Table to a string."""
    string_io = StringIO()
    console = Console(file=string_io, force_terminal=False, no_color=True, width=120)
    console.print(rich_table)
    return string_io.getvalue()


def _render_table_data_for_console(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
    """Processor that renders table data keys as Rich tables for console output.

    Looks for specific keys in the event_dict (defined in TABLE_DATA_KEYS).
    For each found key, renders the data as a Rich ASCII table and appends to the message.
    Creates a filtered copy of extra (without table keys) for console display,
    preserving the original extra dict for JSON logging.

    This processor should only be used for console/plain output, not JSON.
    """
    tables_to_render = []
    rendered_keys: list[str] = list()
    # Check both locations: ExtraAdder flattens to event_dict["ctx"] for foreign loggers,
    # but native structlog loggers may have it in event_dict["extra"]["ctx"]
    ctx = event_dict.get("ctx") or event_dict.get("extra", {}).get("ctx", {})

    match ctx:
        case {"rich_table": rich_table, **__} if isinstance(rich_table, Table):
            tables_to_render.append(_convert_rich_table_to_string(rich_table))
        case {"render_table": True, "tabular_data": tabular_data, "title": title, **__}:
            tables_to_render.append(_render_rich_table(tabular_data, title=title))
            rendered_keys.extend(list(tabular_data.keys()))
        case {"render_table": True, "tabular_data": tabular_data, **__}:
            tables_to_render.append(_render_rich_table(tabular_data, title="Table Data"))
            rendered_keys.extend(list(tabular_data.keys()))
        case _:
            pass

    if tables_to_render:
        # merge the message with the table
        event_dict["event"] = f"{event_dict.get('event', '')}\n\n" + "\n".join(tables_to_render)
        # Don't set _extra_display when we've rendered a table - the table is the display
        # The original 'extra' dict (with ctx) is preserved for JSON logging
        return event_dict

    # Create a filtered copy of extra for console display (excluding rendered table keys)
    # Keep the original extra intact for JSON logging
    extra = event_dict.get("extra", {})
    if extra:
        filtered_extra = {k: v for k, v in extra.items() if k not in rendered_keys}
        if filtered_extra:
            event_dict["_extra_display"] = filtered_extra
    return event_dict


class DiscardSensitiveMessages(logging.Filter):
    """Discards messages marked as sensitive via the `sensitive` flag."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "sensitive", False)


class CategoryFilter(logging.Filter):
    """Filter logs by category."""

    def __init__(self, include_categories: set[LogCategory] | None = None):
        super().__init__()
        self.include_categories = include_categories

    def filter(self, record: logging.LogRecord) -> bool:
        if self.include_categories is None:
            return True
        category = getattr(record, "category", LogCategory.RUNTIME.value)
        return category in {c.value for c in self.include_categories}


def _prepare_json_logging() -> tuple[
    JSONRenderer, structlog.processors.TimeStamper, list[structlog.processors.CallsiteParameterAdder | Callable]
]:
    """Prepare JSON file logging."""
    # TODO: add info context to the json logging
    # todo: add force ascii to decode unicode to ascii
    json_renderer = structlog.processors.JSONRenderer()
    json_timestamp_processor = structlog.processors.TimeStamper(fmt="iso")
    json_env_processors: list[structlog.processors.CallsiteParameterAdder | Callable] = [
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        _category_log_processor,
    ]

    return json_renderer, json_timestamp_processor, json_env_processors


def _prepare_file_logging() -> logging.Handler | None:
    """Prepare file logging for JSON logs."""
    if SETTINGS and SETTINGS.nss_log_file:
        json_renderer, json_timestamp_processor, json_env_processors = _prepare_json_logging()
        json_foreign_pre_chain = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ExtraAdder(),
            json_timestamp_processor,
            structlog.processors.StackInfoRenderer(),
            # structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            *json_env_processors,
            structlog.processors.EventRenamer(to="message"),
        ]
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=json_foreign_pre_chain,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                json_renderer,
            ],
        )
        Path(SETTINGS.nss_log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(SETTINGS.nss_log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(DiscardSensitiveMessages())
        return file_handler
    return None


def _get_console_columns() -> list:
    """Get the column configuration for console rendering. This is the place to modify the console logging format."""
    global SETTINGS

    if SETTINGS and SETTINGS.nss_log_color:
        dim_white = colorama.Style.DIM + colorama.Fore.LIGHTWHITE_EX
        bright_white = colorama.Style.BRIGHT + colorama.Fore.WHITE
        reset_all = colorama.Style.RESET_ALL
    else:
        dim_white = ""
        bright_white = ""
        reset_all = ""

    timestamp_column = structlog.dev.Column(
        "timestamp",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=dim_white,
            reset_style=reset_all,
            value_repr=str,
            postfix=dim_white + " | Nemo Safe Synthesizer | ",
        ),
    )
    category_column = structlog.dev.Column(
        "_category_display",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            postfix=dim_white + " | ",
            value_style=dim_white,
            reset_style=reset_all,
            value_repr=str,
            width=len(LogCategory.RUNTIME.value),
        ),
    )

    level_column = structlog.dev.Column(
        "level",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=dim_white,
            reset_style=reset_all,
            value_repr=str,
            postfix=dim_white + " | ",
            width=5,
        ),
    )
    filename_column = structlog.dev.Column(
        "filename",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=dim_white,
            reset_style=reset_all,
            value_repr=str,
            postfix=dim_white + ":",
        ),
    )
    qual_name_column = structlog.dev.Column(
        "qual_name",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=dim_white,
            reset_style=reset_all,
            value_repr=str,
            postfix=dim_white + ":",
        ),
    )
    lineno_column = structlog.dev.Column(
        "lineno",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=dim_white,
            reset_style=reset_all,
            value_repr=str,
            postfix=dim_white + "",
        ),
    )
    message_column = structlog.dev.Column(
        "message",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=bright_white,
            reset_style=reset_all,
            value_repr=str,
            prefix="\n",
        ),
    )

    extra_display_column = structlog.dev.Column(
        "_extra_display",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style=dim_white,
            reset_style=reset_all,
            value_repr=str,
            prefix=dim_white + ": ",
        ),
    )
    # Suppress "ctx" - table data is rendered separately by _render_table_data_for_console
    ctx_suppress_column = structlog.dev.Column(
        "ctx",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style="",
            reset_style=reset_all,
            value_repr=lambda _: "",  # Render nothing
        ),
    )
    # Suppress "extra" - the actual extra content is rendered via _extra_display above
    extra_suppress_column = structlog.dev.Column(
        "extra",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style="",
            reset_style=reset_all,
            value_repr=lambda _: "",  # Render nothing
        ),
    )
    # Default formatter for all keys not explicitly mentioned.
    default_column = structlog.dev.Column(
        "",
        structlog.dev.KeyValueColumnFormatter(
            key_style=None,
            value_style="",
            reset_style=reset_all,
            value_repr=lambda _: "",
        ),
    )
    return [
        timestamp_column,
        category_column,
        level_column,
        filename_column,
        qual_name_column,
        lineno_column,
        message_column,
        extra_display_column,
        ctx_suppress_column,
        extra_suppress_column,
        default_column,
    ]


def _remove_category_before_render(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
    """Remove category from event_dict right before rendering to prevent duplicate display."""
    event_dict.pop("category", None)
    return event_dict


def _prepare_console_logging(
    shared_processors: list,
    renderer: structlog.stdlib.ProcessorFormatter | structlog.dev.ConsoleRenderer,
    is_plain: bool = False,
) -> logging.Handler:
    # Console handler formatter
    # For plain format, add a processor to remove 'category' right before the renderer
    # since it's already displayed in the column header
    final_processors = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ]
    if is_plain:
        final_processors.append(_remove_category_before_render)
    final_processors.append(renderer)

    console_formatter = structlog.stdlib.ProcessorFormatter(
        # foreign_pre_chain is for logs from non-structlog loggers  - this gives a unified interface for all logs
        # It should NOT include wrap_for_formatter
        foreign_pre_chain=shared_processors,
        processors=final_processors,
    )
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(DiscardSensitiveMessages())
    return console_handler


def _prepare_common_processors() -> tuple[
    list[structlog.stdlib.ProcessorFormatter],
    Any,
]:
    json_renderer, json_timestamp_processor, json_env_processors = _prepare_json_logging()
    global SETTINGS
    if SETTINGS and SETTINGS.nss_log_format == "json":
        renderer = json_renderer
        timestamp_processor = json_timestamp_processor
        env_processors = json_env_processors
    else:
        renderer = structlog.dev.ConsoleRenderer(
            columns=_get_console_columns(),
        )

        def _stamper(event_dict: EventDict) -> EventDict:
            event_dict["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
            return event_dict

        timestamp_processor = structlog.processors.TimeStamper()
        timestamp_processor._stamper = _stamper  # type: ignore[attr-defined]
        # For console output: render table data as Rich tables, then handle categories
        env_processors = [_category_log_processor, _render_table_data_for_console, _move_category_for_column]

    # Shared processors for both native structlog and foreign loggers
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.QUAL_NAME,
            },
            # Ignore our logging wrappers so callsite shows the actual caller
            # without this you get every log line wrapped in our logging wrappers
            additional_ignores=["nemo_safe_synthesizer.observability"],
        ),
        structlog.stdlib.ExtraAdder(),
        *env_processors,
        timestamp_processor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.EventRenamer(to="message"),
    ]

    return shared_processors, renderer  # type: ignore[return-value]


def _clear_loggers():
    logging.getLogger().handlers.clear()


class _CategoryLogAdapter(logging.LoggerAdapter):
    """Adapter that injects category into all log records."""

    def __init__(self, logger: logging.Logger, category: LogCategory):
        super().__init__(logger, {})
        self.category = category

    def process(self, msg, kwargs):
        # Set category via contextvar to avoid it appearing in ExtraAdder output
        _current_log_category.set(self.category.value)
        return msg, kwargs


class CategoryLogger(logging.Logger):
    """Logger wrapper that adds category support.

    Usage::

        logger = get_logger(__name__)

        # Runtime logs (internal details)
        logger.runtime.debug("Memory allocated", extra={"bytes": 1024})
        logger.runtime.info("Cache hit rate", extra={"rate": 0.95})

        # User-relevant logs (progress, results)
        logger.user.info("Training started", extra={"epochs": 10})
        logger.user.info("Generation complete", extra={"records": 1000})

        # Backend logs
        logger.backend.info("Configuration loaded")

        # Default (runtime)
        logger.info("Some message")
    """

    def __init__(self, base_logger: logging.Logger):
        self._logger = base_logger
        self.runtime = _CategoryLogAdapter(base_logger, LogCategory.RUNTIME)
        self.user = _CategoryLogAdapter(base_logger, LogCategory.USER)
        self.system = _CategoryLogAdapter(base_logger, LogCategory.SYSTEM)
        self.backend = _CategoryLogAdapter(base_logger, LogCategory.BACKEND)
        self.default = self.runtime

    @property
    def name(self) -> str:
        return self._logger.name

    # Delegate standard methods to runtime by default (backwards compatible)
    def log(self, level: int, msg, *args, **kwargs):
        self.default.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.default.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.default.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.default.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.default.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.default.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.default.critical(msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)


def _initialize_logging():
    """Initialize logging for Safe Synthesizer. Note that this is to be called only by initialize_observability()."""
    SETTINGS.__init__()
    program_level, dependencies_level = verbosity_mapping[SETTINGS.nss_log_level.upper()]
    _clear_loggers()
    shared_processors, renderer = _prepare_common_processors()
    is_plain = SETTINGS.nss_log_format == "plain"
    console_handler = _prepare_console_logging(shared_processors, renderer, is_plain=is_plain)

    # Attach handlers to root logger so ALL loggers use the configured formatters
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    file_handler = _prepare_file_logging()
    if file_handler:
        root_logger.addHandler(file_handler)

    root_logger.setLevel(program_level)

    # Configure structlog to use stdlib logging as its backend
    # This ensures structlog.get_logger() returns loggers that go through stdlib
    # Note: wrap_for_formatter IS needed here for native structlog logging
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Quiet verbose loggers
    for package in PACKAGES_TO_SET_TO_WARN:
        logging.getLogger(package).setLevel(dependencies_level)


_INITIALIZED_OBSERVABILITY = False


def initialize_observability():
    """Initialize observability for Safe Synthesizer.

    Central entry point for all observability setup -- currently initializes
    logging only. Must be called explicitly by entry points (CLI, scripts);
    not called automatically on import. Idempotent.
    """
    global _INITIALIZED_OBSERVABILITY
    if _INITIALIZED_OBSERVABILITY:
        return

    _initialize_logging()
    _INITIALIZED_OBSERVABILITY = True


def configure_logging_from_workdir(
    workdir: "Workdir",
    log_level: Literal["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG_DEPENDENCIES", "DEBUG"] = "INFO",
    log_format: Literal["json", "plain"] | None = None,
    log_color: bool = True,
) -> Path:
    """Configure observability settings from a Workdir before initialization.

    This should be called BEFORE initialize_observability() to set the log file path
    and other settings based on the workdir structure. The workdir's log_file path
    will be used for file logging.

    Args:
        workdir: The Workdir that defines artifact paths
        log_level: Log level (default: INFO)
        log_format: Log format - 'json' or 'plain' (default: auto-detect from tty)
        log_color: Whether to colorize console output (default: True)

    Returns:
        The configured log file path

    Example:
        workdir = Workdir(base_path=Path("artifacts"), config_name="default", dataset_name="mydata")
        log_file = configure_logging_from_workdir(workdir, log_level="DEBUG")
        initialize_observability()
        logger = get_logger(__name__)
        logger.info("Logs will be written to", extra={"log_file": str(log_file)})
    """
    # Import here to avoid circular imports
    from .cli.artifact_structure import Workdir as WS

    if not isinstance(workdir, WS):
        raise TypeError(f"Expected Workdir, got {type(workdir)}")

    # Ensure the logs directory exists
    log_file = workdir.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure via environment variables (read by NSSObservabilitySettings)
    os.environ["NSS_LOG_FILE"] = str(log_file)
    os.environ["NSS_LOG_LEVEL"] = log_level
    if log_format:
        os.environ["NSS_LOG_FORMAT"] = log_format
    if not log_color:
        os.environ["NSS_LOG_COLOR"] = "false"

    return log_file


def get_logger(name: str | None = None) -> CategoryLogger:
    """Return a category logger for structured logging.

    Always pass ``__name__`` as the argument. After
    ``initialize_observability()`` is called, returns a structlog-based
    logger with full formatting. Before initialization (e.g. when imported
    as a library), returns a basic stdlib logger that integrates with the
    parent application's logging configuration.
    """
    if _INITIALIZED_OBSERVABILITY:
        return CategoryLogger(structlog.get_logger(name))

    # Return basic stdlib logger when logging hasn't been initialized
    # This allows the package to be used as a library without taking over
    # the parent application's logging configuration
    return CategoryLogger(logging.getLogger(name))


class TracedContext:
    """Traced context usable as both a decorator and a context manager.

    As a decorator::

        @traced("operation_name", category=LogCategory.USER)
        def my_function(): ...

    As a context manager::

        with traced("operation_name", category=LogCategory.USER):
            ...
    """

    def __init__(
        self,
        name: str | None,
        category: LogCategory = LogCategory.RUNTIME,
        log_entry: bool = True,
        log_exit: bool = True,
        record_duration: bool = True,
        logger: CategoryLogger | None = None,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
    ):
        if not name:
            raise ValueError("TracedContext name is required")
        self.name: str = name
        self.category = category
        self.log_entry = log_entry
        self.log_exit = log_exit
        self.record_duration = record_duration
        self._start_time: float | None = None
        self._log_adapter: _CategoryLogAdapter | None = None
        self.level = level

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        """Use as a decorator."""
        operation_name = self.name if self.name else getattr(func, "__qualname__", getattr(func, "__name__", "unknown"))

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            func_logger = get_logger(func.__module__)
            log_adapter = getattr(func_logger, self.category.value)

            if self.log_entry:
                getattr(log_adapter, self.level.lower())(f"Entering {operation_name}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                duration = time.time() - start_time
                if self.log_exit:
                    extra = {"duration_ms": duration * 1000} if self.record_duration else {}
                    getattr(log_adapter, self.level.lower())(f"Exiting {operation_name}", extra=extra)

                return result

            except Exception as e:
                log_adapter.error(f"Error in {operation_name}: {e}", extra={"error_type": type(e).__name__})
                raise

        return wrapper

    def __enter__(self) -> "TracedContext":
        """Enter the context manager."""
        operation_name = self.name

        # Get the caller's module for the logger
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame else None
        module_name = caller_frame.f_globals.get("__name__", __name__) if caller_frame else __name__

        logger = get_logger(module_name)
        self._log_adapter = getattr(logger, self.category.value)

        if self.log_entry:
            self._log_adapter.debug(f"Entering {operation_name}")

        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the context manager."""
        operation_name = self.name or "unnamed_operation"

        if exc_type is not None and self._log_adapter is not None:
            self._log_adapter.error(
                f"Error in {operation_name}: {exc_val}",
                extra={"error_type": exc_type.__name__},
            )
            from .cli.wandb_setup import log_failure_to_wandb

            phase = os.getenv("NSS_PHASE", "unknown")
            log_failure_to_wandb(exc_val, phase)
            return False  # Don't suppress the exception

        if self.log_exit and self._log_adapter is not None:
            duration = time.time() - self._start_time if self._start_time else 0
            extra = {"duration_ms": duration * 1000} if self.record_duration else {}
            self._log_adapter.debug(f"Exiting {operation_name}", extra=extra)

        return False


def traced(
    name: str | None = None,
    category: LogCategory = LogCategory.RUNTIME,
    log_entry: bool = True,
    log_exit: bool = True,
    record_duration: bool = True,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
) -> TracedContext:
    """Create a traced context for logging operation entry/exit.

    Args:
        name: Operation name (defaults to function qualname when used as
            a decorator).
        category: Log category for entry/exit messages.
        log_entry: Whether to log function entry.
        log_exit: Whether to log function exit.
        record_duration: Whether to record duration in the exit log.
        level: Log level for entry/exit messages.

    Example::
        # Usage as a decorator
        @traced("training.epoch", category=LogCategory.USER)
        def train_epoch(self, epoch_num: int): ...


        @traced(category=LogCategory.RUNTIME)  # Internal operation
        def _compute_gradients(self): ...


        # Usage as a context manager
        with traced("data_loading", category=LogCategory.USER):
            data = load_data()
            process(data)
    """
    return TracedContext(
        name=name,
        category=category,
        log_entry=log_entry,
        log_exit=log_exit,
        record_duration=record_duration,
        level=level,
    )


def traced_user(name: str | None = None, **kwargs):
    """Log a user-relevant operation (progress, results)."""
    return traced(name=name, category=LogCategory.USER, **kwargs)


def traced_runtime(name: str | None = None, **kwargs):
    """Log a runtime/internal operation."""
    return traced(name=name, category=LogCategory.RUNTIME, **kwargs)


def traced_system(name: str | None = None, **kwargs):
    """Log a system-level operation."""
    return traced(name=name, category=LogCategory.SYSTEM, **kwargs)


def traced_backend(name: str | None = None, **kwargs):
    """Log a backend operation."""
    return traced(name=name, category=LogCategory.BACKEND, **kwargs)
