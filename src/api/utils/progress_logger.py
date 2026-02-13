from __future__ import annotations

import logging
import sys
from collections.abc import Callable


def get_progress_logger(name: str = "kr_super_momentum.app") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def build_progress_callback(
    logger: logging.Logger, prefix: str = "[BOT]"
) -> Callable[[str], None]:
    def _emit(message: str) -> None:
        logger.info("%s %s", prefix, message)

    return _emit
