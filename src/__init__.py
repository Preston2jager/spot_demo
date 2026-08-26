"""Spot control library.

Importing anything from this package installs the project logging configuration.
"""
from .log_config import setup_logging

setup_logging()

__all__ = ["setup_logging"]
