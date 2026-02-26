import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from app.core.config import get_settings


def _make_console_handler(is_production: bool) -> logging.StreamHandler:
    renderer = (
        structlog.processors.JSONRenderer()
        if is_production
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(processor=renderer))
    return handler


def _make_file_handler(path: str, max_bytes: int, backup_count: int) -> RotatingFileHandler:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=structlog.processors.JSONRenderer())
    )
    return handler


def setup_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handlers: list[logging.Handler] = [_make_console_handler(settings.is_production)]
    if settings.log_file:
        handlers.append(
            _make_file_handler(settings.log_file, settings.log_max_bytes, settings.log_backup_count)
        )

    root = logging.getLogger()
    root.handlers = []
    root.setLevel(log_level)
    for handler in handlers:
        handler.setLevel(log_level)
        root.addHandler(handler)


def get_logger(name: str = __name__):
    return structlog.get_logger(name)
