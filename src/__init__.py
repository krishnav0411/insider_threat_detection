# src/__init__.py
"""
Package initialiser. Provides logging configuration shared by all src modules.
"""

import logging
import logging.handlers
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure a root logger that writes to stdout with a minimal format.

    Args:
        level: Logging level (default: INFO).

    Example:
        >>> setup_logging(logging.DEBUG)
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


# Auto-configure on first import so log messages from src modules are visible.
setup_logging()
