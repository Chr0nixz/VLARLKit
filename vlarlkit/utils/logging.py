import logging
import sys
from pathlib import Path
from typing import Any


_HANDLER_KIND_ATTR = "_vlarlkit_handler_kind"
_FILE_PATH_ATTR = "_vlarlkit_file_path"
_LOG_FMT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_configured_log_path: Path | None = None


def _hydra_log_path() -> Path | None:
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
    except Exception:
        return None

    output_dir = getattr(hydra_cfg.runtime, "output_dir", None)
    job_name = getattr(hydra_cfg.job, "name", None)
    if not output_dir or not job_name:
        return None
    return Path(output_dir) / f"{job_name}.log"


def _current_log_path() -> Path | None:
    return _configured_log_path or _hydra_log_path()


def configure_file_logging(log_path: str | Path) -> None:
    global _configured_log_path
    _configured_log_path = Path(log_path).resolve()


def _make_handler(kind: str, path: Path | None = None) -> logging.Handler:
    if kind == "file":
        assert path is not None
        handler = logging.FileHandler(path)
        setattr(handler, _FILE_PATH_ATTR, str(path))
        datefmt = "%Y-%m-%d %H:%M:%S"
    else:
        handler = logging.StreamHandler(sys.stdout)
        datefmt = "%H:%M:%S"

    setattr(handler, _HANDLER_KIND_ATTR, kind)
    handler.setFormatter(
        logging.Formatter(
            fmt=_LOG_FMT,
            datefmt=datefmt,
        )
    )
    return handler


def _ensure_console_handler(logger: logging.Logger) -> None:
    has_console = any(
        getattr(handler, _HANDLER_KIND_ATTR, None) == "console"
        for handler in logger.handlers
    )
    if not has_console:
        logger.addHandler(_make_handler("console"))


def _ensure_file_handler(logger: logging.Logger, log_path: Path | None) -> None:
    if log_path is None:
        return

    path = log_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_KIND_ATTR, None) != "file":
            continue
        if (
            getattr(handler, _FILE_PATH_ATTR, None) == str(path)
            and not getattr(handler, "_closed", False)
        ):
            return
        logger.removeHandler(handler)
        handler.close()

    logger.addHandler(_make_handler("file", path))


class ProjectLogger:
    def __init__(self, name: str) -> None:
        self._name = name

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger(self._name)
        logger.disabled = False
        logger.setLevel(logging.INFO)
        logger.propagate = False

        _ensure_file_handler(logger, _current_log_path())
        _ensure_console_handler(logger)
        return logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_logger(), name)


def get_logger(name: str) -> ProjectLogger:
    return ProjectLogger(name)
