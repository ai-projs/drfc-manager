import os
import sys
import logging
import structlog
from typing import List, Optional, Union


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = False,
    json_output: bool = True,
) -> None:
    """
    Configure structlog with the specified settings.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_output: Whether to output logs to console
        json_output: Whether to use JSON format for logs
    """

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")


    console_env = os.environ.get('DRFC_CONSOLE_LOGGING', 'false').lower() in ('1','true','yes')
    emit_console = console_output or console_env


    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)


    handlers: list[Union[logging.StreamHandler, logging.FileHandler]] = []
    if emit_console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(stream_handler)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            handlers.append(file_handler)
        except Exception:
            if not emit_console:
                fallback = logging.StreamHandler(sys.stdout)
                fallback.setFormatter(logging.Formatter("%(message)s"))
                handlers.append(fallback)

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=handlers,
    )

    processors: List[structlog.types.Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]  # type: ignore[list-item]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured structlog logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
