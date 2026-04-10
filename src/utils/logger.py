import logging
import sys
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


_log_format: str = "console"
_structlogger = None


def setup_logger(
    name: str = "openeyes",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    log_format: str = "console",
) -> logging.Logger:
    global _log_format, _structlogger
    _log_format = log_format

    if log_format == "json" and STRUCTLOG_AVAILABLE:
        return _setup_structlog(name, level, log_file)
    return _setup_standard(name, level, log_file, max_bytes, backup_count)


def _setup_structlog(
    name: str,
    level: int,
    log_file: Optional[str],
) -> logging.Logger:
    global _structlogger

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundaryLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _structlogger = structlog.get_logger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

    return logger


def _setup_standard(
    name: str,
    level: int,
    log_file: Optional[str],
    max_bytes: int,
    backup_count: int,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "openeyes") -> logging.Logger:
    return logging.getLogger(name)


def log_json(level: str, message: str, **kwargs: Any) -> None:
    if _log_format != "json" or not STRUCTLOG_AVAILABLE:
        return

    global _structlogger
    if _structlogger is None:
        _structlogger = structlog.get_logger("openeyes")

    log_fn = getattr(_structlogger, level, _structlogger.info)
    log_fn(message, **kwargs)


def log_detection(frame: int, detections: int, latency_ms: float) -> None:
    log_json("info", "detection", frame=frame, detections=detections, latency_ms=round(latency_ms, 2))


def log_error(message: str, **kwargs: Any) -> None:
    log_json("error", message, **kwargs)
