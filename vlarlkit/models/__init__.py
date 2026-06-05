import importlib
from typing import Any


def get_model(cfg: Any):
    backend = cfg.get("backend", None)
    if not backend:
        raise ValueError("Model config must set 'backend', e.g. model.backend=openpi.")

    module_name = f"vlarlkit_{backend}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ImportError(
            f"Could not import model backend '{module_name}'. Install the "
            f"'{backend}' model backend package or run with its uv project."
        ) from exc

    if not hasattr(module, "get_model"):
        raise AttributeError(f"Model backend '{module_name}' does not define get_model(cfg).")
    return module.get_model(cfg)


__all__ = ["get_model"]
