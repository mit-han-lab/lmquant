# -*- coding: utf-8 -*-
"""Logging tools."""

import logging
import sys

__all__ = ["log", "info", "debug", "warning", "error", "critical", "Formatter", "basicConfig", "setup_default_config"]


def log(level, msg: str, logger: logging.Logger = None) -> None:
    """Log a message with the given level.

    Args:
        level: The level to log the message with.
        msg: The message to log.
        logger: The logger to use. If None, the root logger is used.
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


def info(msg: str, logger: logging.Logger = None):
    """Log a message with the INFO level.

    Args:
        msg: The message to log.
        logger: The logger to use. If None, the root logger is used.
    """
    log(logging.INFO, msg, logger)


def debug(msg: str, logger: logging.Logger = None):
    """Log a message with the DEBUG level.

    Args:
        msg: The message to log.
        logger: The logger to use. If None, the root logger is used.
    """
    log(logging.DEBUG, msg, logger)


def warning(msg: str, logger: logging.Logger = None):
    """Log a message with the WARNING level.

    Args:
        msg: The message to log.
        logger: The logger to use. If None, the root logger is used.
    """
    log(logging.WARNING, msg, logger)


def error(msg: str, logger: logging.Logger = None):
    """Log a message with the ERROR level.

    Args:
        msg: The message to log.
        logger: The logger to use. If None, the root logger is used.
    """
    log(logging.ERROR, msg, logger)


def critical(msg: str, logger: logging.Logger = None):
    """Log a message with the CRITICAL level.

    Args:
        msg: The message to log.
        logger: The logger to use. If None, the root logger is used.
    """
    log(logging.CRITICAL, msg, logger)


class Formatter(logging.Formatter):
    """A custom formatter for logging."""

    indent = 0

    def __init__(self, fmt: str = None, datefmt: str = None, style: str = "%"):
        """Initialize the formatter.

        Args:
            fmt: The format string.
            datefmt: The date format string.
            style: The style to use for the format string.
        """
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        """Format the record.

        Args:
            record: The record to format.

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
    path: str = None,
    level: int = logging.DEBUG,
    format: str = "%(asctime)s | %(levelname).1s | %(message)s",
    datefmt: str = "%y-%m-%d %H:%M:%S",
    **kwargs,
) -> None:
    """Setup the default logging configuration.

    Args:
        path (str, optional): The path to the log file. Defaults to ``None``.
        level (int, optional): The logging level. Defaults to ``logging.DEBUG``.
        format (str, optional): The format string. Defaults to ``"%(asctime)s | %(levelname).1s | %(message)s"``.
        datefmt (str, optional): The date format string. Defaults to ``"%y-%m-%d %H:%M:%S"``.
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
