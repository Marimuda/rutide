"""Compare a RUTide coefficient file with the pinned Python UTide oracle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from .constants import DEFAULT_FIXTURE, DEFAULT_UTIDE_ROOT
from .fixture import reconstruct_mjd
from .runner import _profile_options, load_oracle

DEFAULT_TOLERANCES = {
    "amplitude": 3e-12,
    "phase_degrees": 3e-9,
    "mean": 3e-12,
    "slope_per_day": 3e-12,
    "frequency_cph": 1e-15,
}


def circular_phase_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return the shortest absolute separation between angles in degrees."""
    return np.abs((actual - expected + 180.0) % 360.0 - 180.0)


def _finite(values: Any, description: str) -> np.ndarray:
    array = np.ma.asarray(values, dtype=np.float64)
    if np.ma.is_masked(array) and np.ma.count_masked(array):
        raise ValueError(f"{description} contains masked values")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{description} contains non-finite values")
    return result


def _maximum_error(
    values: list[tuple[float, int, str | None]],
    tolerance: float,
) -> dict[str, Any]:
    error, node_index, constituent = max(values, key=lambda item: item[0])
    result: dict[str, Any] = {
        "maximum_absolute_error": float(error),
        "tolerance": tolerance,
        "within_tolerance": bool(error <= tolerance),
        "worst_node_index": int(node_index),
    }
    if constituent is not None:
        result["worst_constituent"] = constituent
    return result


def compare_with_oracle(
    rust_output: Path,
    fixture: Path,
    utide_root: Path,
) -> dict[str, Any]:
    """Compare every series in one RUTide output against Python UTide."""
    oracle = load_oracle(utide_root)
    rust_path = rust_output.resolve(strict=True)
    fixture_path = fixture.resolve(strict=True)

    with netCDF4.Dataset(rust_path, mode="r") as result_dataset:
        if result_dataset.getncattr("profile") != "fixed-constituents-greenwich-nodal-ols":
            raise ValueError("RUTide output uses an unsupported analysis profile")
        names = str(result_dataset.getncattr("constituent_names")).split(",")
        if not names or len(set(names)) != len(names):
            raise ValueError(f"constituent names must be non-empty and unique: {names}")
        indices = np.asarray(result_dataset.variables["node_index"][:], dtype=np.int64)
        latitudes = _finite(result_dataset.variables["latitude"][:], "output latitude")
        amplitudes = _finite(result_dataset.variables["amplitude"][:], "output amplitude")
        phases = _finite(result_dataset.variables["phase"][:], "output phase")
        means = _finite(result_dataset.variables["mean"][:], "output mean")
        slopes = _finite(result_dataset.variables["slope"][:], "output slope")
        frequencies = _finite(result_dataset.variables["frequency"][:], "output frequency")
        result_digest = str(result_dataset.getncattr("result_sha256"))

    series_count = len(indices)
    constituent_count = len(names)
    if series_count == 0 or len(set(int(index) for index in indices)) != series_count:
        raise ValueError("output node indices must be non-empty and unique")
    if amplitudes.shape != (series_count, constituent_count):
        raise ValueError(f"unexpected amplitude shape: {amplitudes.shape}")
    if phases.shape != amplitudes.shape:
        raise ValueError(f"unexpected phase shape: {phases.shape}")
    if means.shape != (series_count,) or slopes.shape != (series_count,):
        raise ValueError("unexpected mean or slope shape")
    if latitudes.shape != (series_count,) or frequencies.shape != (constituent_count,):
        raise ValueError("unexpected latitude or frequency shape")

    errors: dict[str, list[tuple[float, int, str | None]]] = {
        name: [] for name in DEFAULT_TOLERANCES
    }
    options = _profile_options("fixed-constituents")
    options["constit"] = names
    with netCDF4.Dataset(fixture_path, mode="r") as source_dataset:
        time = reconstruct_mjd(
            source_dataset.variables["Itime"][:],
            source_dataset.variables["Itime2"][:],
        )
        source_node_count = len(source_dataset.dimensions["node"])
        for position, raw_node_index in enumerate(indices):
            node_index = int(raw_node_index)
            if not 0 <= node_index < source_node_count:
                raise ValueError(f"output node index is out of range: {node_index}")
            source_latitude = float(source_dataset.variables["lat"][node_index])
            if latitudes[position] != source_latitude:
                raise ValueError(f"output latitude differs from source at node {node_index}")
            observations = _finite(
                source_dataset.variables["zeta"][:, node_index],
                f"source zeta at node {node_index}",
            )
            coefficient = oracle.solve(
                time,
                observations,
                lat=source_latitude,
                **options,
            )
            oracle_names = [str(value) for value in coefficient.name.tolist()]
            order = [oracle_names.index(name) for name in names]

            expected_amplitude = np.asarray(coefficient.A, dtype=np.float64)[order]
            expected_phase = np.asarray(coefficient.g, dtype=np.float64)[order]
            expected_frequency = np.asarray(coefficient.aux.frq, dtype=np.float64)[order]
            for constituent_position, constituent in enumerate(names):
                errors["amplitude"].append(
                    (
                        abs(
                            amplitudes[position, constituent_position]
                            - expected_amplitude[constituent_position]
                        ),
                        node_index,
                        constituent,
                    )
                )
                errors["phase_degrees"].append(
                    (
                        float(
                            circular_phase_error(
                                phases[position, constituent_position],
                                expected_phase[constituent_position],
                            )
                        ),
                        node_index,
                        constituent,
                    )
                )
                errors["frequency_cph"].append(
                    (
                        abs(
                            frequencies[constituent_position]
                            - expected_frequency[constituent_position]
                        ),
                        node_index,
                        constituent,
                    )
                )
            errors["mean"].append(
                (abs(means[position] - float(coefficient.mean)), node_index, None)
            )
            errors["slope_per_day"].append(
                (abs(slopes[position] - float(coefficient.slope)), node_index, None)
            )

    metrics = {
        name: _maximum_error(values, DEFAULT_TOLERANCES[name]) for name, values in errors.items()
    }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "implementation_under_test": "rutide",
        "oracle": "python-utide",
        "profile": "fixed-constituents-greenwich-nodal-ols",
        "series": series_count,
        "constituents": names,
        "fixture": str(fixture_path),
        "rust_output": str(rust_path),
        "rust_result_sha256": result_digest,
        "metrics": metrics,
        "passed": all(metric["within_tolerance"] for metric in metrics.values()),
    }


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse comparison command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--utide-root", type=Path, default=DEFAULT_UTIDE_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run the comparison and return nonzero when a tolerance is exceeded."""
    args = parse_args(arguments)
    result = compare_with_oracle(args.rust_output, args.fixture, args.utide_root)
    rendered = json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote parity result: {args.output}")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
