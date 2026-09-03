from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str

class Fit:
    @property
    def is_vector(self) -> bool: ...
    def summary(self) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...

class BatchFit:
    @property
    def is_vector(self) -> bool: ...
    @property
    def series_count(self) -> int: ...
    def summary(self) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...

def solve(
    time_mjd: npt.NDArray[np.float64],
    eastward: npt.NDArray[np.float64],
    northward: npt.NDArray[np.float64] | None,
    latitude: float,
    constituent_names: Sequence[str] | None,
    rayleigh_min: float,
    diagnostics: bool,
    diagnostic_min_signal_to_noise: float,
    method_name: str,
    confidence_name: str,
    white: bool,
    trend: bool,
    phase_name: str,
    nodal_name: str,
    monte_carlo_realizations: int,
    monte_carlo_seed: int,
    robust_weight_name: str,
    robust_tuning: float | None,
    robust_tolerance: float,
    robust_max_iterations: int,
    inferred_names: Sequence[str],
    reference_names: Sequence[str],
    inference_ratios: Sequence[float],
    inference_phase_offsets: Sequence[float],
    approximate_inference: bool,
    order_name: str,
    order_names: Sequence[str],
) -> Fit: ...
def reconstruct(
    time_mjd: npt.NDArray[np.float64],
    fit: Fit,
    constituent_names: Sequence[str] | None,
    minimum_signal_to_noise: float | None,
    minimum_percent_energy: float,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]: ...
def solve_many(
    time_mjd: npt.NDArray[np.float64],
    eastward: npt.NDArray[np.float64],
    northward: npt.NDArray[np.float64] | None,
    latitudes: npt.NDArray[np.float64],
    constituent_names: Sequence[str] | None,
    rayleigh_min: float,
    diagnostics: bool,
    diagnostic_min_signal_to_noise: float,
    method_name: str,
    confidence_name: str,
    white: bool,
    trend: bool,
    phase_name: str,
    nodal_name: str,
    monte_carlo_realizations: int,
    monte_carlo_seed: int,
    robust_weight_name: str,
    robust_tuning: float | None,
    robust_tolerance: float,
    robust_max_iterations: int,
    inferred_names: Sequence[str],
    reference_names: Sequence[str],
    inference_ratios: Sequence[float],
    inference_phase_offsets: Sequence[float],
    approximate_inference: bool,
    order_name: str,
    order_names: Sequence[str],
    workers: int | None,
    memory_limit_bytes: int | None,
) -> BatchFit: ...
def reconstruct_many(
    time_mjd: npt.NDArray[np.float64],
    fit: BatchFit,
    constituent_names: Sequence[str] | None,
    minimum_signal_to_noise: float | None,
    minimum_percent_energy: float,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]: ...
def restore_fit(snapshot: dict[str, Any]) -> Fit: ...
def restore_batch(snapshot: dict[str, Any], workers: int | None) -> BatchFit: ...
