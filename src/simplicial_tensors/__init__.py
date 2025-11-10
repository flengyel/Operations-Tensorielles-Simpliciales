"""Core package for simplicial tensor computations."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["tensor_ops", "symbolic_tensor_ops"]

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from . import tensor_ops as tensor_ops  # type: ignore self-import
    from . import symbolic_tensor_ops as symbolic_tensor_ops


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
