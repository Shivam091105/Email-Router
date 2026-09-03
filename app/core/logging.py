"""
Application-wide logging configuration.

We configure Python's standard `logging` module once, at startup, so every
module can just do `logger = logging.getLogger(__name__)` and get consistent
formatting, without each file reinventing its own setup.

Rule enforced throughout the project: never log secrets (passwords, API
tokens, full connection strings). Log what happened and relevant IDs, not
credentials.
"""

import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Quiet down noisy third-party loggers unless we're debugging.
    if level > logging.DEBUG:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
