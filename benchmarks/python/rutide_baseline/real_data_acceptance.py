"""Exercise an installed RUTide package on observational ADCP and FVCOM currents."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import rutide

from .constants import REPOSITORY_ROOT

SCHEMA_VERSION = 2
CONSTITUENTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
FVCOM_CONSTITUENTS = ["M2", "S2", "N2", "K1", "O1"]
DEFAULT_FVCOM = (
    REPOSITORY_ROOT
    / ".."
    / "projects"
    / "fvcom"
    / "claude_scratchpad"
    / "baroclinic_vikc1701"
    / "run"
    / "frs2f_0001.nc"
).resolve()


def evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    """Return a sorted deterministic selection spanning the complete source axis."""
    if size < 1 or count < 1 or count > size:
        raise ValueError("count must be between one and the source-axis size")
    if count == size:
        return np.arange(size, dtype=np.int64)
    return np.rint(np.linspace(0, size - 1, count)).astype(np.int64)


def numeric_array(value: Any) -> np.ndarray:
    """Convert a NetCDF masked array to contiguous float64 with NaN fill."""
    if np.ma.isMaskedArray(value):
        value = np.ma.filled(value, np.nan)
    return np.ascontiguousarray(value, dtype=np.float64)


def read_time_major_columns(
    variable: Any, indices: np.ndarray, working_memory_mb: float
) -> np.ndarray:
    """Read sparse classic-NetCDF columns through bounded contiguous time slabs."""
    if len(variable.shape) != 2 or working_memory_mb <= 0:
        raise ValueError("a two-dimensional variable and positive memory budget are required")
    time_count, source_columns = variable.shape
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= source_columns):
        raise ValueError("column indices are outside the source variable")
    source_bytes = np.dtype(variable.dtype).itemsize
    budget_bytes = max(1, int(working_memory_mb * 1024 * 1024))
    rows_per_block = max(1, min(time_count, budget_bytes // (source_columns * source_bytes)))
    output = np.empty((time_count, len(indices)), dtype=np.float64)
    for start in range(0, time_count, rows_per_block):
        stop = min(start + rows_per_block, time_count)
        block = variable[start:stop, :]
        if np.ma.isMaskedArray(block):
            block = np.ma.filled(block, np.nan)
        output[start:stop] = np.asarray(block)[:, indices]
    return output


def array_digest(*values: np.ndarray) -> str:
    """Hash dtype, shape, finite mask, and values for materialized results."""
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.uint64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a modest external fixture without loading it as one allocation."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def shared_velocity_units(eastward: Any, northward: Any) -> str:
    """Return matching, non-empty units for a vector-current variable pair."""
    values = []
    for variable in (eastward, northward):
        value = getattr(variable, "units", None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("vector-current variables must both declare non-empty text units")
        values.append(value.strip())
    if values[0] != values[1]:
        raise ValueError(
            f"vector-current units disagree: eastward={values[0]!r}, northward={values[1]!r}"
        )
    return values[0]


def sampling_summary(time_values: np.ndarray) -> dict[str, Any]:
    """Summarize the time axis and reject chronology errors before fitting."""
    values = np.asarray(time_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(
            "the acceptance time axis must be finite, one-dimensional, and non-trivial"
        )
    steps_hours = np.diff(values) * 24.0
    if np.any(steps_hours <= 0.0):
        raise ValueError("the acceptance time axis must be strictly increasing")
    median_step = float(np.median(steps_hours))
    tolerance = max(1e-9, abs(median_step) * 1e-6)
    return {
        "record_span_days": float(values[-1] - values[0]),
        "median_interval_hours": median_step,
        "largest_gap_hours": float(np.max(steps_hours)),
        "non_median_intervals": int(
            np.count_nonzero(np.abs(steps_hours - median_step) > tolerance)
        ),
    }


def finite_range(values: Any) -> dict[str, Any]:
    """Return a JSON-safe finite-value count and range."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise RuntimeError("a required diagnostic field contains no finite values")
    return {
        "finite": int(finite.size),
        "total": int(array.size),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
    }


def diagnostic_summary(coefficient: Any) -> dict[str, Any]:
    """Reduce dense constituent diagnostics to auditable release-gate evidence."""
    diagnostics = coefficient.diagn
    if diagnostics is None:
        raise RuntimeError("constituent-selection diagnostics were not produced")
    condition = np.asarray(diagnostics.K)
    adjusted_snr = np.asarray(diagnostics.SNRallc_over_K)
    percent_all = np.asarray(diagnostics.PTVallc)
    if not (
        np.isfinite(condition).all()
        and np.isfinite(adjusted_snr).all()
        and np.isfinite(percent_all).all()
    ):
        raise RuntimeError("whole-model diagnostic fields must be finite for every series")
    higher_rr = np.asarray(diagnostics.hi.RR)
    higher_rnm = np.asarray(diagnostics.hi.RNM)
    higher_correlation = np.asarray(diagnostics.hi.CorMx)
    return {
        "basis_condition_number": finite_range(condition),
        "condition_adjusted_signal_to_noise": finite_range(adjusted_snr),
        "whole_model_condition_bound_pass_series": int(np.count_nonzero(adjusted_snr > 1.0)),
        "percent_tidal_variance_all": finite_range(percent_all),
        "percent_tidal_variance_significant": finite_range(diagnostics.PTVsnrc),
        "significant_constituents": int(np.count_nonzero(np.asarray(coefficient.SNR) >= 2.0)),
        "adjacent_pairs": int(np.count_nonzero(np.isfinite(higher_rr))),
        "rayleigh_resolved_pairs": int(np.count_nonzero(higher_rr >= 1.0)),
        "noise_modified_resolved_pairs": int(np.count_nonzero(higher_rnm >= 1.0)),
        "correlation_at_most_0_2_pairs": int(np.count_nonzero(higher_correlation <= 0.2)),
        "maximum_neighbor_parameter_correlation": finite_range(higher_correlation),
    }


def assert_unity_prefilter_control(
    *,
    time_values: np.ndarray,
    eastward: np.ndarray,
    northward: np.ndarray,
    latitudes: float | np.ndarray,
    options: dict[str, Any],
    baseline: Any,
) -> dict[str, Any]:
    """Require unity correction to preserve representative fitted results bit-for-bit."""
    series_count = min(4, eastward.shape[1])
    control_latitudes = (
        latitudes if np.isscalar(latitudes) else np.asarray(latitudes)[:series_count]
    )
    control_options = {
        **options,
        "lat": control_latitudes,
        "prefilt": {"frq": [0.0, 0.2], "P": [1.0, 1.0], "rng": [0.01, 2.0]},
    }
    started = time.perf_counter()
    corrected = rutide.solve_many(
        time_values,
        eastward[:, :series_count],
        northward[:, :series_count],
        **control_options,
    )
    elapsed = time.perf_counter() - started
    checked_fields = ["Lsmaj", "Lsmin", "theta", "g", "Lsmaj_ci", "SNR"]
    for field in checked_fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(baseline, field))[:series_count],
            np.asarray(getattr(corrected, field)),
            err_msg=f"unity pre-filter correction changed {field}",
        )
    for field in ["K", "SNRallc", "SNRallc_over_K", "PTVallc", "PTVsnrc"]:
        np.testing.assert_array_equal(
            np.asarray(getattr(baseline.diagn, field))[:series_count],
            np.asarray(getattr(corrected.diagn, field)),
            err_msg=f"unity pre-filter correction changed diagnostic {field}",
        )
    return {
        "series": series_count,
        "seconds": elapsed,
        "bitwise_equal": True,
        "checked_coefficient_fields": checked_fields,
        "checked_diagnostic_fields": [
            "K",
            "SNRallc",
            "SNRallc_over_K",
            "PTVallc",
            "PTVsnrc",
        ],
    }


def analyze_vector_batch(
    *,
    label: str,
    time_values: np.ndarray,
    eastward: np.ndarray,
    northward: np.ndarray,
    latitudes: float | np.ndarray,
    constituents: list[str],
    epoch: str | None,
    workers: int,
    memory_limit_mb: float,
) -> dict[str, Any]:
    """Fit, reconstruct, persist, restore, and validate one real vector batch."""
    if eastward.shape != northward.shape or eastward.shape[0] != len(time_values):
        raise ValueError(f"{label} input arrays have inconsistent shapes")
    joint = np.isfinite(eastward) & np.isfinite(northward)
    minimum_rows = 2 * len(constituents) + 3
    retained_columns = np.count_nonzero(joint, axis=0) > minimum_rows
    if not retained_columns.any():
        raise ValueError(f"{label} has no overdetermined jointly valid current series")
    eastward = np.ascontiguousarray(eastward[:, retained_columns])
    northward = np.ascontiguousarray(northward[:, retained_columns])
    if not np.isscalar(latitudes):
        latitudes = np.ascontiguousarray(np.asarray(latitudes)[retained_columns])

    options = {
        "lat": latitudes,
        "constit": constituents,
        "order_constit": constituents,
        "conf_int": "linear",
        "method": "ols",
        "trend": True,
        "phase": "Greenwich",
        "nodal": True,
        "white": False,
        "diagnostics": True,
        "diagnostic_min_SNR": 2.0,
        "epoch": epoch,
        "workers": workers,
        "memory_limit_mb": memory_limit_mb,
        "verbose": False,
    }
    started = time.perf_counter()
    coefficient = rutide.solve_many(time_values, eastward, northward, **options)
    solve_seconds = time.perf_counter() - started

    started = time.perf_counter()
    tide = rutide.reconstruct_many(
        time_values,
        coefficient,
        epoch=epoch,
        constit=constituents,
        min_SNR=None,
        verbose=False,
    )
    reconstruct_seconds = time.perf_counter() - started
    if tide.u.shape != eastward.shape or tide.v.shape != northward.shape:
        raise RuntimeError(f"{label} reconstruction shape does not match its input")
    if not np.isfinite(coefficient.Lsmaj).all() or not np.isfinite(coefficient.g).all():
        raise RuntimeError(f"{label} produced non-finite primary coefficients")

    prefilter_control = assert_unity_prefilter_control(
        time_values=time_values,
        eastward=eastward,
        northward=northward,
        latitudes=latitudes,
        options=options,
        baseline=coefficient,
    )

    with tempfile.TemporaryDirectory(prefix=f"rutide-{label}-") as temporary:
        archive = Path(temporary) / "coefficients.rutide.npz"
        started = time.perf_counter()
        coefficient.save(archive)
        save_seconds = time.perf_counter() - started
        archive_bytes = archive.stat().st_size
        started = time.perf_counter()
        restored = rutide.load(archive, workers=workers)
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        restored_tide = rutide.reconstruct_many(
            time_values,
            restored,
            epoch=epoch,
            constit=constituents,
            min_SNR=None,
            verbose=False,
        )
        restored_reconstruct_seconds = time.perf_counter() - started
    np.testing.assert_array_equal(tide.u, restored_tide.u)
    np.testing.assert_array_equal(tide.v, restored_tide.v)

    residual_eastward = (eastward - tide.u)[joint[:, retained_columns]]
    residual_northward = (northward - tide.v)[joint[:, retained_columns]]
    return {
        "source_series": int(joint.shape[1]),
        "analyzed_series": int(eastward.shape[1]),
        "samples": int(eastward.shape[0]),
        "joint_valid_observations": int(np.count_nonzero(joint[:, retained_columns])),
        "missing_vector_fraction": float(1.0 - np.mean(joint[:, retained_columns])),
        "sampling": sampling_summary(time_values),
        "constituents": constituents,
        "solve_seconds": solve_seconds,
        "solve_series_per_second": eastward.shape[1] / solve_seconds,
        "reconstruct_seconds": reconstruct_seconds,
        "reconstruct_series_per_second": eastward.shape[1] / reconstruct_seconds,
        "coefficient_archive_bytes": archive_bytes,
        "coefficient_save_seconds": save_seconds,
        "coefficient_load_seconds": load_seconds,
        "restored_reconstruct_seconds": restored_reconstruct_seconds,
        "coefficient_digest": array_digest(
            coefficient.Lsmaj,
            coefficient.Lsmin,
            coefficient.theta,
            coefficient.g,
            coefficient.Lsmaj_ci,
            coefficient.SNR,
        ),
        "reconstruction_digest": array_digest(tide.u, tide.v),
        "tidal_residual_rms": float(
            np.sqrt(np.mean(np.square(residual_eastward) + np.square(residual_northward)))
        ),
        "diagnostics": diagnostic_summary(coefficient),
        "unity_prefilter_control": prefilter_control,
        "persistence_bitwise_equal": True,
    }


def analyze_adcp(path: Path, workers: int, memory_limit_mb: float) -> dict[str, Any]:
    """Analyze every sufficiently populated depth cell in the CCE1 ADCP record."""
    started = time.perf_counter()
    with netCDF4.Dataset(path) as dataset:
        times = numeric_array(dataset.variables["TIME"][:])
        depths = numeric_array(dataset.variables["DEPTH"][:])
        latitude = float(dataset.variables["LATITUDE"][0])
        eastward_variable = dataset.variables["UCUR"]
        northward_variable = dataset.variables["VCUR"]
        velocity_units = shared_velocity_units(eastward_variable, northward_variable)
        eastward = numeric_array(eastward_variable[:])
        northward = numeric_array(northward_variable[:])
        metadata = {
            "title": dataset.getncattr("title"),
            "institution": dataset.getncattr("institution"),
            "time_coverage_start": dataset.getncattr("time_coverage_start"),
            "time_coverage_end": dataset.getncattr("time_coverage_end"),
            "latitude": latitude,
            "depth_min_metres": float(np.min(depths)),
            "depth_max_metres": float(np.max(depths)),
            "velocity_units": velocity_units,
        }
    input_seconds = time.perf_counter() - started
    result = analyze_vector_batch(
        label="adcp",
        time_values=times,
        eastward=eastward,
        northward=northward,
        latitudes=latitude,
        constituents=CONSTITUENTS,
        epoch="1950-01-01",
        workers=workers,
        memory_limit_mb=memory_limit_mb,
    )
    return {
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "source_sha256": file_sha256(path),
        "input_seconds": input_seconds,
        "input_strategy": "complete observational profile",
        "metadata": metadata,
        "analysis": result,
    }


def analyze_fvcom(
    path: Path, series_count: int, workers: int, memory_limit_mb: float
) -> dict[str, Any]:
    """Analyze a domain-spanning current selection from the largest FVCOM fixture."""
    started = time.perf_counter()
    with netCDF4.Dataset(path) as dataset:
        element_count = len(dataset.dimensions["nele"])
        indices = evenly_spaced_indices(element_count, min(series_count, element_count))
        times = numeric_array(dataset.variables["Itime"][:])
        times += numeric_array(dataset.variables["Itime2"][:]) / 86_400_000.0
        latitudes = numeric_array(dataset.variables["latc"][indices])
        input_budget_mb = min(128.0, memory_limit_mb / 2.0)
        eastward_variable = dataset.variables["ua"]
        northward_variable = dataset.variables["va"]
        velocity_units = shared_velocity_units(eastward_variable, northward_variable)
        eastward = read_time_major_columns(eastward_variable, indices, input_budget_mb)
        northward = read_time_major_columns(northward_variable, indices, input_budget_mb)
        metadata = {
            "title": dataset.getncattr("title"),
            "source_model": dataset.getncattr("source"),
            "format": dataset.data_model,
            "elements": element_count,
            "selected_first_element": int(indices[0]),
            "selected_last_element": int(indices[-1]),
            "velocity_units": velocity_units,
        }
    input_seconds = time.perf_counter() - started
    result = analyze_vector_batch(
        label="fvcom",
        time_values=times,
        eastward=eastward,
        northward=northward,
        latitudes=latitudes,
        constituents=FVCOM_CONSTITUENTS,
        epoch=None,
        workers=workers,
        memory_limit_mb=memory_limit_mb,
    )
    return {
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "input_seconds": input_seconds,
        "input_strategy": "bounded contiguous time slabs across classic-NetCDF records",
        "input_working_memory_mb": input_budget_mb,
        "selection_digest": array_digest(indices),
        "metadata": metadata,
        "analysis": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adcp", type=Path, required=True)
    parser.add_argument("--fvcom", type=Path, default=DEFAULT_FVCOM)
    parser.add_argument("--fvcom-series", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--memory-limit-mb", type=float, default=512.0)
    parser.add_argument("--expected-version")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.fvcom_series < 1 or args.workers < 1 or args.memory_limit_mb <= 0:
        parser.error("fvcom-series, workers, and memory-limit-mb must be positive")
    adcp_path = args.adcp.resolve(strict=True)
    fvcom_path = args.fvcom.resolve(strict=True)
    if args.expected_version is not None and rutide.__version__ != args.expected_version:
        parser.error(
            f"installed rutide version {rutide.__version__} does not match {args.expected_version}"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "netcdf4": importlib.metadata.version("netcdf4"),
            "rutide": rutide.__version__,
            "rutide_native_extension": str(rutide._native.__file__),
            "workers": args.workers,
            "memory_limit_mb": args.memory_limit_mb,
        },
        "adcp": analyze_adcp(adcp_path, args.workers, args.memory_limit_mb),
        "fvcom": analyze_fvcom(
            fvcom_path,
            args.fvcom_series,
            args.workers,
            args.memory_limit_mb,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
