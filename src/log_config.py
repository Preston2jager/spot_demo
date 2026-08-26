"""Loguru configuration for the whole project.

Kept in its own module (rather than as an import side effect of ``spot_tracker``)
so that no linter can silently delete the import that installs it.
``setup_logging()`` is idempotent and is called once from ``src/__init__.py``.
"""
from datetime import datetime
from pathlib import Path

from loguru import logger

LOG_DIR = Path(__file__).resolve().parent.parent / "runtime" / "logs"

_configured = False


def setup_logging() -> None:
    """Install the console and dated-file sinks. Safe to call more than once."""
    global _configured
    if _configured:
        return
    _configured = True

    logger.remove()
    logger.level("INFO", color="<yellow>")

    # Console: route through print() so SpotTracker can intercept and indent it
    logger.add(
        lambda msg: print(msg.rstrip('\n')),
        format="<level><bold>[[{level}]]</bold></level> {message}",
        colorize=True,
    )

    # File: one runtime/logs/<date>.log per day, runs append to the same file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        LOG_DIR / "{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        colorize=False,
        enqueue=True,
    )

    # Session marker goes straight to the file so the console stays clean
    started = datetime.now()
    with open(LOG_DIR / f"{started:%Y-%m-%d}.log", "a", encoding="utf-8") as f:
        f.write(f"\n===== Run started: {started:%Y-%m-%d %H:%M:%S} =====\n")
