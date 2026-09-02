"""Matched benchmark for UTide loops, RUTide loops, and RUTide batch bindings."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rutide
from threadpoolctl import threadpool_info, threadpool_limits

from .constants import DEFAULT_UTIDE_ROOT, EXPECTED_UTIDE_REVISION, REPOSITORY_ROOT
from .runner import load_oracle

SCHEMA_VERSION = 1
_CONSTITUENTS = ["M2", "S2", "N2", "K1", "O1"]
_FREQUENCIES_CPH = np.array(
    [0.080_511_400_7, 0.083_333_333_3, 0.078_999_248_8, 0.041_780_746_2, 0.038_730_654_4]
)
_EPOCH = "1858-11-17"


@dataclass(frozen=True)
class Fixture:
    """One deterministic time-major scalar or vector benchmark fixture."""

    time: np.ndarray
    eastward: np.ndarray
    northward: np.ndarray | None
    latitudes: np.ndarray


def make_fixture(field: str, sampling: str, samples: int, series_count: int) -> Fixture:
    """Generate distinct deterministic series with repeated realistic missing masks."""
    index = np.arange(samples, dtype=np.float64)
    time_mjd = 60_000.0 + index / 24.0
    if sampling == "irregular" and samples > 2:
        time_mjd[1:-1] += 0.002 * np.sin(index[1:-1] * 0.37)
        time_mjd[1:-1] += 0.0007 * np.cos(index[1:-1] * 0.11)
    hours = (time_mjd - np.mean(time_mjd)) * 24.0
    phase = 2.0 * np.pi * hours[:, None] * _FREQUENCIES_CPH[None, :]
    amplitudes = np.array([0.85, 0.24, 0.19, 0.13, 0.09])
    eastward = np.empty((samples, series_count), dtype=np.float64)
    northward = np.empty((samples, series_count), dtype=np.float64) if field == "vector" else None
    for column in range(series_count):
        shifts = 0.11 * column + np.arange(len(_CONSTITUENTS)) * 0.37
        harmonic = np.cos(phase + shifts[None, :]) @ (amplitudes * (1.0 + 0.002 * column))
        eastward[:, column] = (
            0.15
            + 0.0003 * (time_mjd - np.mean(time_mjd))
            + harmonic
            + 0.012 * np.sin(index * (0.23 + 0.0003 * column))
        )
        if northward is not None:
            rotary = np.sin(phase - 0.07 * column + shifts[None, :]) @ (0.55 * amplitudes)
            northward[:, column] = (
                -0.08
                - 0.0002 * (time_mjd - np.mean(time_mjd))
                + rotary
                + 0.009 * np.cos(index * (0.19 + 0.0002 * column))
            )

    if sampling == "irregular" and samples >= 10:
        for column in range(series_count):
            group = column % 4
            positions = {
                max(1, samples // 7 + group),
                max(2, samples // 2 + group - 2),
                min(samples - 2, 5 * samples // 6 - group),
            }
            for offset, position in enumerate(sorted(positions)):
                if northward is None or offset % 2 == 0:
                    eastward[position, column] = np.nan
                else:
                    northward[position, column] = np.nan

    return Fixture(
        time=np.ascontiguousarray(time_mjd),
        eastward=eastward,
        northward=northward,
        latitudes=np.linspace(58.0, 62.0, series_count),
    )


def solve_options(profile: str) -> dict[str, Any]:
    """Return options understood identically by the two public APIs."""
    options: dict[str, Any] = {
        "constit": _CONSTITUENTS,
        "order_constit": _CONSTITUENTS,
        "conf_int": "none" if profile == "ols" else "linear",
        "method": "robust" if profile == "robust-colored" else "ols",
        "trend": True,
        "phase": "Greenwich",
        "nodal": True,
        "white": False,
        "verbose": False,
        "epoch": _EPOCH,
    }
    return options


def solve_loop(module: Any, fixture: Fixture, options: dict[str, Any]) -> list[Any]:
    """Fit each series through one implementation's scalar endpoint."""
    output = []
    for series, latitude in enumerate(fixture.latitudes):
        arguments = [fixture.time, fixture.eastward[:, series]]
        if fixture.northward is not None:
            arguments.append(fixture.northward[:, series])
        output.append(module.solve(*arguments, lat=float(latitude), **options))
    return output


def solve_batch(
    fixture: Fixture,
    options: dict[str, Any],
    workers: int,
    memory_limit_mb: float | None,
) -> rutide.CoefficientBatch:
    """Fit all series through the native time-major endpoint."""
    return rutide.solve_many(
        fixture.time,
        fixture.eastward,
        fixture.northward,
        lat=fixture.latitudes,
        workers=workers,
        memory_limit_mb=memory_limit_mb,
        **options,
    )


def reconstruct_loop(
    module: Any,
    fixture: Fixture,
    coefficients: list[Any],
) -> tuple[np.ndarray, ...]:
    """Reconstruct every series through one implementation's scalar endpoint."""
    eastward = []
    northward = []
    for coefficient in coefficients:
        tide = module.reconstruct(
            fixture.time,
            coefficient,
            epoch=_EPOCH,
            verbose=False,
            constit=_CONSTITUENTS,
        )
        if fixture.northward is None:
            eastward.append(tide.h)
        else:
            eastward.append(tide.u)
            northward.append(tide.v)
    output = (np.column_stack(eastward),)
    if northward:
        output += (np.column_stack(northward),)
    return output


def reconstruct_batch(
    fixture: Fixture, coefficients: rutide.CoefficientBatch
) -> tuple[np.ndarray, ...]:
    """Reconstruct all series through the native time-major endpoint."""
    tide = rutide.reconstruct_many(
        fixture.time,
        coefficients,
        epoch=_EPOCH,
        verbose=False,
        constit=_CONSTITUENTS,
    )
    if fixture.northward is None:
        return (tide.h,)
    return tide.u, tide.v


def measure(
    name: str,
    function: Callable[[], Any],
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], Any]:
    """Measure complete eager calls and retain the final result for validation."""
    result = None
    for _ in range(warmups):
        result = function()
    samples = []
    for repetition in range(repetitions):
        gc.collect()
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        print(
            f"operation={name} repetition={repetition} seconds={elapsed:.9f}",
            file=sys.stderr,
            flush=True,
        )
    median = float(np.median(samples))
    return {
        "seconds": samples,
        "median_seconds": median,
        "series_per_second": None,
    }, result


def coefficient_arrays(coefficients: Any, field: str, batch: bool) -> dict[str, np.ndarray]:
    """Normalize loop and batch coefficient fields to `(series, constituent)`."""
    fields = ["A", "g"] if field == "scalar" else ["Lsmaj", "Lsmin", "theta", "g"]
    output = {}
    for name in fields:
        if batch:
            output[name] = np.asarray(getattr(coefficients, name))
        else:
            output[name] = np.stack([np.asarray(getattr(value, name)) for value in coefficients])
    first = coefficients if batch else coefficients[0]
    confidence_field = "A_ci" if field == "scalar" else "Lsmaj_ci"
    if hasattr(first, confidence_field) and getattr(first, confidence_field) is not None:
        confidence_fields = (
            ["A_ci", "g_ci", "SNR"]
            if field == "scalar"
            else [
                "Lsmaj_ci",
                "Lsmin_ci",
                "theta_ci",
                "g_ci",
                "SNR",
            ]
        )
        for name in confidence_fields:
            if batch:
                output[name] = np.asarray(getattr(coefficients, name))
            else:
                output[name] = np.stack(
                    [np.asarray(getattr(value, name)) for value in coefficients]
                )
    return output


def validate_results(
    field: str,
    oracle_coefficients: list[Any],
    loop_coefficients: list[Any],
    batch_coefficients: rutide.CoefficientBatch,
    oracle_tide: tuple[np.ndarray, ...],
    loop_tide: tuple[np.ndarray, ...],
    batch_tide: tuple[np.ndarray, ...],
) -> dict[str, float]:
    """Fail a benchmark that does not preserve the already-validated numerics."""
    oracle = coefficient_arrays(oracle_coefficients, field, False)
    loop = coefficient_arrays(loop_coefficients, field, False)
    batch = coefficient_arrays(batch_coefficients, field, True)
    errors: dict[str, float] = {}
    angle_fields = {"g", "theta"}
    for name in oracle:
        oracle_delta = (
            angular_delta(oracle[name], loop[name])
            if name in angle_fields
            else (oracle[name] - loop[name])
        )
        batch_delta = (
            angular_delta(loop[name], batch[name])
            if name in angle_fields
            else (loop[name] - batch[name])
        )
        errors[f"utide_vs_rutide_{name}_max_abs"] = finite_max_abs(oracle_delta)
        errors[f"rutide_loop_vs_batch_{name}_max_abs"] = finite_max_abs(batch_delta)
        errors[f"utide_vs_rutide_{name}_max_relative"] = finite_max_relative(
            oracle_delta, oracle[name]
        )
        errors[f"rutide_loop_vs_batch_{name}_max_relative"] = finite_max_relative(
            batch_delta, loop[name]
        )
        if name in angle_fields:
            if finite_max_abs(oracle_delta) > 2e-4 or finite_max_abs(batch_delta) > 2e-8:
                raise RuntimeError(f"coefficient angle mismatch in {name}")
        else:
            np.testing.assert_allclose(oracle[name], loop[name], rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(loop[name], batch[name], rtol=2e-10, atol=2e-10)
    for component, (oracle_values, loop_values, batch_values) in enumerate(
        zip(oracle_tide, loop_tide, batch_tide, strict=True)
    ):
        errors[f"utide_vs_rutide_reconstruction_{component}_max_abs"] = finite_max_abs(
            oracle_values - loop_values
        )
        errors[f"rutide_loop_vs_batch_reconstruction_{component}_max_abs"] = finite_max_abs(
            loop_values - batch_values
        )
        errors[f"utide_vs_rutide_reconstruction_{component}_max_relative"] = finite_max_relative(
            oracle_values - loop_values, oracle_values
        )
        errors[f"rutide_loop_vs_batch_reconstruction_{component}_max_relative"] = (
            finite_max_relative(loop_values - batch_values, loop_values)
        )
        np.testing.assert_allclose(oracle_values, loop_values, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(loop_values, batch_values, rtol=2e-10, atol=2e-10)
    return errors


def angular_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return signed shortest differences in degrees."""
    return (left - right + 180.0) % 360.0 - 180.0


def finite_max_abs(values: np.ndarray) -> float:
    """Return a JSON-safe maximum over finite entries."""
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    return float(np.max(finite)) if finite.size else 0.0


def finite_max_relative(delta: np.ndarray, reference: np.ndarray) -> float:
    """Return maximum relative error where both input arrays are finite."""
    delta = np.asarray(delta)
    reference = np.asarray(reference)
    finite = np.isfinite(delta) & np.isfinite(reference)
    if not finite.any():
        return 0.0
    scale = np.maximum(np.abs(reference[finite]), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(delta[finite]) / scale))


def result_digest(arrays: dict[str, np.ndarray] | tuple[np.ndarray, ...]) -> str:
    """Create a stable checksum proving that timed calls returned materialized data."""
    values = arrays.values() if isinstance(arrays, dict) else arrays
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.uint64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def git_output(arguments: list[str]) -> str:
    """Read repository identity without mutating benchmark state."""
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("scalar", "vector"), default="scalar")
    parser.add_argument("--sampling", choices=("regular", "irregular"), default="regular")
    parser.add_argument(
        "--profile",
        choices=("ols", "linear-colored", "robust-colored"),
        default="ols",
    )
    parser.add_argument("--samples", type=int, default=745)
    parser.add_argument("--series-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--memory-limit-mb", type=float, default=512.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--utide-root", type=Path, default=DEFAULT_UTIDE_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.samples < 32
        or args.series_count < 1
        or args.workers < 1
        or args.memory_limit_mb <= 0
        or args.warmups < 0
        or args.repetitions < 1
    ):
        parser.error(
            "samples must be at least 32; series-count, workers, memory-limit-mb, and "
            "repetitions must be positive; warmups must be non-negative"
        )

    oracle_module = load_oracle(args.utide_root)
    fixture = make_fixture(args.field, args.sampling, args.samples, args.series_count)
    options = solve_options(args.profile)
    with threadpool_limits(limits=1):
        limited_threadpools = threadpool_info()
        solve_utide, oracle_coefficients = measure(
            "solve_utide_loop",
            lambda: solve_loop(oracle_module, fixture, options),
            args.warmups,
            args.repetitions,
        )
        solve_rutide, loop_coefficients = measure(
            "solve_rutide_loop",
            lambda: solve_loop(rutide, fixture, options),
            args.warmups,
            args.repetitions,
        )
        solve_batch_result, batch_coefficients = measure(
            "solve_rutide_batch",
            lambda: solve_batch(fixture, options, args.workers, args.memory_limit_mb),
            args.warmups,
            args.repetitions,
        )
        reconstruct_utide, oracle_tide = measure(
            "reconstruct_utide_loop",
            lambda: reconstruct_loop(oracle_module, fixture, oracle_coefficients),
            args.warmups,
            args.repetitions,
        )
        reconstruct_rutide, loop_tide = measure(
            "reconstruct_rutide_loop",
            lambda: reconstruct_loop(rutide, fixture, loop_coefficients),
            args.warmups,
            args.repetitions,
        )
        reconstruct_batch_result, batch_tide = measure(
            "reconstruct_rutide_batch",
            lambda: reconstruct_batch(fixture, batch_coefficients),
            args.warmups,
            args.repetitions,
        )

    timings = {
        "solve": {
            "utide_loop": solve_utide,
            "rutide_loop": solve_rutide,
            "rutide_batch": solve_batch_result,
        },
        "reconstruct": {
            "utide_loop": reconstruct_utide,
            "rutide_loop": reconstruct_rutide,
            "rutide_batch": reconstruct_batch_result,
        },
    }
    for operation in timings.values():
        for timing in operation.values():
            timing["series_per_second"] = args.series_count / timing["median_seconds"]
    correctness = validate_results(
        args.field,
        oracle_coefficients,
        loop_coefficients,
        batch_coefficients,
        oracle_tide,
        loop_tide,
        batch_tide,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "field": args.field,
            "sampling": args.sampling,
            "profile": args.profile,
            "samples": args.samples,
            "series_count": args.series_count,
            "workers": args.workers,
            "memory_limit_mb": args.memory_limit_mb,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "blas_threads": 1,
            "constituents": _CONSTITUENTS,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "rutide": rutide.__version__,
            "utide_revision": EXPECTED_UTIDE_REVISION,
            "repository_revision": git_output(["rev-parse", "HEAD"]),
            "repository_dirty": bool(git_output(["status", "--porcelain"])),
            "threadpools": limited_threadpools,
        },
        "timings": timings,
        "speedups": {
            operation: {
                "rutide_loop_vs_utide": values["utide_loop"]["median_seconds"]
                / values["rutide_loop"]["median_seconds"],
                "rutide_batch_vs_utide": values["utide_loop"]["median_seconds"]
                / values["rutide_batch"]["median_seconds"],
                "rutide_batch_vs_rutide_loop": values["rutide_loop"]["median_seconds"]
                / values["rutide_batch"]["median_seconds"],
            }
            for operation, values in timings.items()
        },
        "correctness": correctness,
        "digests": {
            "utide_coefficients": result_digest(
                coefficient_arrays(oracle_coefficients, args.field, False)
            ),
            "rutide_loop_coefficients": result_digest(
                coefficient_arrays(loop_coefficients, args.field, False)
            ),
            "rutide_batch_coefficients": result_digest(
                coefficient_arrays(batch_coefficients, args.field, True)
            ),
            "utide_reconstruction": result_digest(oracle_tide),
            "rutide_loop_reconstruction": result_digest(loop_tide),
            "rutide_batch_reconstruction": result_digest(batch_tide),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
