import logging
import sys
from typing import Any


class ProjectLogger:
    def __init__(self, name: str) -> None:
        self._name = name

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger(self._name)
        logger.disabled = False
        logger.setLevel(logging.INFO)
        logger.propagate = False

        has_handler = any(
            getattr(handler, "_vlarlkit_handler", False)
            for handler in logger.handlers
        )
        if not has_handler:
            handler = logging.StreamHandler(sys.stdout)
            handler._vlarlkit_handler = True
            handler.setFormatter(
                logging.Formatter(
                    fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            logger.addHandler(handler)
        return logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_logger(), name)


def get_logger(name: str) -> ProjectLogger:
    return ProjectLogger(name)
