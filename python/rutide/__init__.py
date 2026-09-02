"""Fast tidal harmonic analysis with a UTide-inspired Python API."""

from ._api import (
    Bunch,
    Coefficient,
    CoefficientBatch,
    Tide,
    reconstruct,
    reconstruct_many,
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
    "reconstruct",
    "reconstruct_many",
    "solve",
    "solve_many",
]
