"""Fast tidal harmonic analysis with a UTide-inspired Python API."""

from ._api import (
    Bunch,
    Coefficient,
    CoefficientBatch,
    Tide,
    load,
    reconstruct,
    reconstruct_many,
    save,
    solve,
    solve_many,
)
from ._native import __version__

__all__ = [
    "Bunch",
    "Coefficient",
    "CoefficientBatch",
    "Tide",
    "__version__",
    "load",
    "reconstruct",
    "reconstruct_many",
    "save",
    "solve",
    "solve_many",
]
