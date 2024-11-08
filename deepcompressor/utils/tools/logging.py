# -*- coding: utf-8 -*-
"""Logging tools."""

import logging
import sys
import typing as tp

from tqdm.contrib.logging import logging_redirect_tqdm

__all__ = [
    "CRITICAL",
    "FATAL",
    "ERROR",
    "WARNING",
    "WARN",
    "INFO",
    "DEBUG",
    "NOTSET",
    "log",
    "info",
    "debug",
    "warning",
    "error",
    "critical",
    "Formatter",
    "basicConfig",
    "setup",
    "getLogger",
    "redirect_tqdm",
]


CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET


redirect_tqdm = logging_redirect_tqdm
shutdown = logging.shutdown
Logger = logging.Logger


def getLogger(name: str | None = None) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name (`str` or `None`, *optional*, defaults to `None`): The name of the logger.

    Returns:
        logging.Logger: The logger.
    """
    return logging.getLogger(name)


def log(level: int, msg: str, logger: logging.Logger | None = None) -> None:
    """Log a message with the given level.

    Args:
        level (`int`): The logging level.
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    if logger is None:
        logger = logging.getLogger()
    if not logger.isEnabledFor(level):
        return
    msg = str(msg)
    if "\n" in msg:
        for line in msg.split("\n"):
            log(level, line, logger)
    else:
        logger.log(level, msg)


def info(msg: str, logger: logging.Logger | None = None):
    """Log a message with the INFO level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.INFO, msg, logger)


def debug(msg: str, logger: logging.Logger | None = None):
    """Log a message with the DEBUG level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.DEBUG, msg, logger)


def warning(msg: str, logger: logging.Logger | None = None):
    """Log a message with the WARNING level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.WARNING, msg, logger)


def error(msg: str, logger: logging.Logger | None = None):
    """Log a message with the ERROR level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.ERROR, msg, logger)


def critical(msg: str, logger: logging.Logger | None = None):
    """Log a message with the CRITICAL level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.CRITICAL, msg, logger)


class Formatter(logging.Formatter):
    """A custom formatter for logging."""

    indent = 0

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, style: tp.Literal["%", "{", "$"] = "%"):
        """Initialize the formatter.

        Args:
            fmt (`str` or `None`, *optional*, defaults to `None`): The format string.
            datefmt (`str` or `None`, *optional*, defaults to `None`): The date format string.
            style (`str`, *optional*, defaults to `"%"`): The format style.
        """
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        """Format the record.

        Args:
            record (`logging.LogRecord`): The log record.

        Returns:
            str: The formatted record.
        """
        record.message = " " * self.indent + record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record)
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)
        return s

    @staticmethod
    def indent_inc(delta: int = 2):
        """Increase the indent."""
        Formatter.indent += delta

    @staticmethod
    def indent_dec(delta: int = 2):
        """Decrease the indent."""
        Formatter.indent -= delta

    @staticmethod
    def indent_reset(indent: int = 0):
        """Reset the indent."""
        Formatter.indent = indent


def basicConfig(**kwargs) -> None:
    """Configure the root logger."""
    fmt = kwargs.pop("format", None)
    datefmt = kwargs.pop("datefmt", None)
    style = kwargs.pop("style", "%")
    logging.basicConfig(**kwargs)
    for h in logging.root.handlers[:]:
        h.setFormatter(Formatter(fmt, datefmt, style))


def setup(
    path: str | None = None,
    level: int = logging.DEBUG,
    format: str = "%(asctime)s | %(levelname).1s | %(message)s",
    datefmt: str = "%y-%m-%d %H:%M:%S",
    **kwargs,
) -> None:
    """Setup the default logging configuration.

    Args:
        path (`str` | `None`, *optional*, defaults to `None`):
            The path to the log file. If `None`, only the console is used.
        level (`int`, *optional*, defaults to `logging.DEBUG`): The logging level.
        format (`str`, *optional*, defaults to `"%(asctime)s | %(levelname).1s | %(message)s"`):
            The format string.
        datefmt (`str`, *optional*, defaults to `"%y-%m-%d %H:%M:%S"`): The date format string.
        **kwargs: Additional keyword arguments.
    """
    handlers = kwargs.pop("handlers", None)
    force = kwargs.pop("force", True)
    if handlers is None:
        handlers = [logging.StreamHandler(sys.stdout)]
        if path is not None:
            handlers.append(logging.FileHandler(path, mode="w"))
    basicConfig(
        level=level,
        format=format,
        datefmt=datefmt,
        handlers=handlers,
        force=force,
    )
