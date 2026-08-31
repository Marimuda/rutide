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
    "complex_coefficient": 3e-12,
    "percent_energy": 1e-10,
    "amplitude_ci": 1e-10,
    "phase_ci_degrees": 1e-7,
    "mean": 3e-12,
    "slope_per_day": 3e-12,
    "frequency_cph": 1e-15,
    "reference_time_mjd": 1e-12,
    "reconstruction": 1e-9,
}
BASE_PHASE_TOLERANCE_DEGREES = 3e-9
SIGNAL_TO_NOISE_ABSOLUTE_TOLERANCE = 1e-6
SIGNAL_TO_NOISE_RELATIVE_TOLERANCE = 1e-8
MJD_TO_PYTHON_DATENUM = 678_576.0


def circular_phase_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return the shortest absolute separation between angles in degrees."""
    return np.abs((actual - expected + 180.0) % 360.0 - 180.0)


def phase_tolerance_degrees(amplitude: float) -> float:
    """Return a phase tolerance consistent with the complex-coefficient bound."""
    coefficient_tolerance = DEFAULT_TOLERANCES["complex_coefficient"]
    if amplitude <= coefficient_tolerance:
        return 180.0
    geometric_bound = np.degrees(np.arcsin(min(1.0, coefficient_tolerance / amplitude)))
    return max(BASE_PHASE_TOLERANCE_DEGREES, float(geometric_bound))


def _finite(values: Any, description: str) -> np.ndarray:
    array = np.ma.asarray(values, dtype=np.float64)
    if np.ma.is_masked(array) and np.ma.count_masked(array):
        raise ValueError(f"{description} contains masked values")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{description} contains non-finite values")
    return result


def _observations_with_nan(values: Any, description: str) -> np.ndarray:
    array = np.ma.asarray(values, dtype=np.float64)
    result = np.asarray(array.filled(np.nan), dtype=np.float64)
    if np.isinf(result).any():
        raise ValueError(f"{description} contains infinite values")
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


def _maximum_phase_error(values: list[tuple[float, float, int, str]]) -> dict[str, Any]:
    error, tolerance, node_index, constituent = max(
        values,
        key=lambda item: item[0] / item[1],
    )
    maximum_error, _, maximum_node, maximum_constituent = max(values, key=lambda item: item[0])
    return {
        "maximum_absolute_error": float(maximum_error),
        "maximum_error_node_index": int(maximum_node),
        "maximum_error_constituent": maximum_constituent,
        "base_tolerance": BASE_PHASE_TOLERANCE_DEGREES,
        "near_zero_rule": "max(base, asin(complex_coefficient_tolerance / amplitude))",
        "worst_tolerance_ratio": float(error / tolerance),
        "tolerance_at_worst_ratio": float(tolerance),
        "within_tolerance": bool(all(item[0] <= item[1] for item in values)),
        "worst_node_index": int(node_index),
        "worst_constituent": constituent,
    }


def _maximum_snr_error(values: list[tuple[float, float, int, str]]) -> dict[str, Any]:
    error, expected, node_index, constituent = max(
        values,
        key=lambda item: (
            item[0]
            / (
                SIGNAL_TO_NOISE_ABSOLUTE_TOLERANCE
                + SIGNAL_TO_NOISE_RELATIVE_TOLERANCE * abs(item[1])
            )
        ),
    )
    tolerance = SIGNAL_TO_NOISE_ABSOLUTE_TOLERANCE + (
        SIGNAL_TO_NOISE_RELATIVE_TOLERANCE * abs(expected)
    )
    return {
        "maximum_absolute_error": float(max(item[0] for item in values)),
        "absolute_tolerance": SIGNAL_TO_NOISE_ABSOLUTE_TOLERANCE,
        "relative_tolerance": SIGNAL_TO_NOISE_RELATIVE_TOLERANCE,
        "worst_tolerance_ratio": float(error / tolerance),
        "within_tolerance": bool(
            all(
                item[0]
                <= SIGNAL_TO_NOISE_ABSOLUTE_TOLERANCE
                + SIGNAL_TO_NOISE_RELATIVE_TOLERANCE * abs(item[1])
                for item in values
            )
        ),
        "worst_node_index": int(node_index),
        "worst_constituent": constituent,
    }


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
        profile = str(result_dataset.getncattr("profile"))
        if profile not in {
            "fixed-constituents-greenwich-nodal-ols",
            "rayleigh-auto-greenwich-nodal-ols",
        }:
            raise ValueError("RUTide output uses an unsupported analysis profile")
        selection_method = (
            str(result_dataset.getncattr("constituent_selection"))
            if "constituent_selection" in result_dataset.ncattrs()
            else "explicit"
        )
        confidence_interval = (
            str(result_dataset.getncattr("confidence_interval"))
            if "confidence_interval" in result_dataset.ncattrs()
            else "none"
        )
        confidence_noise = (
            str(result_dataset.getncattr("confidence_noise"))
            if confidence_interval == "linear"
            else None
        )
        names = str(result_dataset.getncattr("constituent_names")).split(",")
        if not names or len(set(names)) != len(names):
            raise ValueError(f"constituent names must be non-empty and unique: {names}")
        indices = np.asarray(result_dataset.variables["node_index"][:], dtype=np.int64)
        latitudes = _finite(result_dataset.variables["latitude"][:], "output latitude")
        amplitudes = _finite(result_dataset.variables["amplitude"][:], "output amplitude")
        phases = _finite(result_dataset.variables["phase"][:], "output phase")
        percent_energy = _finite(
            result_dataset.variables["percent_energy"][:],
            "output percent energy",
        )
        means = _finite(result_dataset.variables["mean"][:], "output mean")
        slopes = _finite(result_dataset.variables["slope"][:], "output slope")
        frequencies = _finite(result_dataset.variables["frequency"][:], "output frequency")
        observation_counts = (
            np.asarray(result_dataset.variables["observation_count"][:], dtype=np.int64)
            if "observation_count" in result_dataset.variables
            else None
        )
        reference_times = (
            _finite(result_dataset.variables["reference_time"][:], "output reference time")
            if "reference_time" in result_dataset.variables
            else None
        )
        result_digest = str(result_dataset.getncattr("result_sha256"))
        if confidence_interval == "linear":
            amplitude_ci = _finite(
                result_dataset.variables["amplitude_ci"][:],
                "output amplitude CI",
            )
            phase_ci = _finite(
                result_dataset.variables["phase_ci"][:],
                "output phase CI",
            )
            signal_to_noise = _finite(
                result_dataset.variables["signal_to_noise"][:],
                "output signal-to-noise ratio",
            )
        else:
            amplitude_ci = phase_ci = signal_to_noise = None
        rayleigh_min = (
            float(result_dataset.getncattr("rayleigh_min"))
            if selection_method == "rayleigh"
            else None
        )
        if "reconstruction" in result_dataset.variables:
            reconstruction = _finite(
                result_dataset.variables["reconstruction"][:],
                "output reconstruction",
            )
            reconstruction_time = _finite(
                result_dataset.variables["time"][:],
                "output reconstruction time",
            )
            reconstruction_filter = str(result_dataset.getncattr("reconstruction_filter"))
            reconstruction_constituents = (
                str(result_dataset.getncattr("reconstruction_constituents")).split(",")
                if reconstruction_filter == "constituents"
                else None
            )
            reconstruction_minimum_pe = (
                float(result_dataset.getncattr("reconstruction_minimum_percent_energy"))
                if reconstruction_filter == "diagnostics"
                else None
            )
            reconstruction_minimum_snr = (
                float(result_dataset.getncattr("reconstruction_minimum_signal_to_noise"))
                if "reconstruction_minimum_signal_to_noise" in result_dataset.ncattrs()
                else None
            )
        else:
            reconstruction = reconstruction_time = None
            reconstruction_filter = None
            reconstruction_constituents = None
            reconstruction_minimum_pe = reconstruction_minimum_snr = None

    series_count = len(indices)
    constituent_count = len(names)
    if series_count == 0 or len(set(int(index) for index in indices)) != series_count:
        raise ValueError("output node indices must be non-empty and unique")
    if amplitudes.shape != (series_count, constituent_count):
        raise ValueError(f"unexpected amplitude shape: {amplitudes.shape}")
    if phases.shape != amplitudes.shape:
        raise ValueError(f"unexpected phase shape: {phases.shape}")
    if percent_energy.shape != amplitudes.shape:
        raise ValueError(f"unexpected percent-energy shape: {percent_energy.shape}")
    for description, values in (
        ("amplitude CI", amplitude_ci),
        ("phase CI", phase_ci),
        ("signal-to-noise ratio", signal_to_noise),
    ):
        if values is not None and values.shape != amplitudes.shape:
            raise ValueError(f"unexpected {description} shape: {values.shape}")
    if means.shape != (series_count,) or slopes.shape != (series_count,):
        raise ValueError("unexpected mean or slope shape")
    if latitudes.shape != (series_count,):
        raise ValueError("unexpected latitude shape")
    if frequencies.shape == (constituent_count,):
        frequencies = np.broadcast_to(frequencies, (series_count, constituent_count))
    elif frequencies.shape != (series_count, constituent_count):
        raise ValueError(f"unexpected frequency shape: {frequencies.shape}")
    if observation_counts is not None and observation_counts.shape != (series_count,):
        raise ValueError("unexpected observation-count shape")
    if reference_times is not None and reference_times.shape != (series_count,):
        raise ValueError("unexpected reference-time shape")
    if reconstruction is not None:
        if reconstruction_time.ndim != 1 or reconstruction_time.size == 0:
            raise ValueError("unexpected reconstruction time shape")
        if reconstruction.shape != (reconstruction_time.size, series_count):
            raise ValueError(f"unexpected reconstruction shape: {reconstruction.shape}")
        if reconstruction_filter not in {"all", "constituents", "diagnostics"}:
            raise ValueError(f"unsupported reconstruction filter: {reconstruction_filter}")

    metric_names = [
        "amplitude",
        "complex_coefficient",
        "percent_energy",
        "mean",
        "slope_per_day",
        "frequency_cph",
    ]
    if confidence_interval == "linear":
        metric_names.extend(["amplitude_ci", "phase_ci_degrees"])
    elif confidence_interval != "none":
        raise ValueError(f"unsupported confidence interval: {confidence_interval}")
    if reconstruction is not None:
        metric_names.append("reconstruction")
    if reference_times is not None:
        metric_names.append("reference_time_mjd")
    errors: dict[str, list[tuple[float, int, str | None]]] = {name: [] for name in metric_names}
    phase_errors: list[tuple[float, float, int, str]] = []
    snr_errors: list[tuple[float, float, int, str]] = []
    options = _profile_options("fixed-constituents")
    options["conf_int"] = confidence_interval
    options["white"] = confidence_noise == "white"
    if confidence_interval == "linear" and confidence_noise not in {"white", "colored"}:
        raise ValueError(f"unsupported confidence noise model: {confidence_noise}")
    if selection_method == "rayleigh":
        if profile != "rayleigh-auto-greenwich-nodal-ols" or rayleigh_min is None:
            raise ValueError("inconsistent Rayleigh selection metadata")
        options["constit"] = "auto"
        options["Rayleigh_min"] = rayleigh_min
    elif selection_method == "explicit":
        if profile != "fixed-constituents-greenwich-nodal-ols":
            raise ValueError("inconsistent explicit selection metadata")
        options["constit"] = names
    else:
        raise ValueError(f"unsupported constituent selection method: {selection_method}")
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
            observations = _observations_with_nan(
                source_dataset.variables["zeta"][:, node_index],
                f"source zeta at node {node_index}",
            )
            expected_observation_count = int(np.isfinite(observations).sum())
            if (
                observation_counts is not None
                and observation_counts[position] != expected_observation_count
            ):
                raise ValueError(f"output observation count differs at node {node_index}")
            coefficient = oracle.solve(
                time,
                observations,
                lat=source_latitude,
                **options,
            )
            oracle_names = [str(value) for value in coefficient.name.tolist()]
            if len(oracle_names) != len(names) or set(oracle_names) != set(names):
                raise ValueError(
                    "RUTide selection differs from Python UTide: "
                    f"rust={names}, python={oracle_names}"
                )
            order = [oracle_names.index(name) for name in names]

            expected_amplitude = np.asarray(coefficient.A, dtype=np.float64)[order]
            expected_phase = np.asarray(coefficient.g, dtype=np.float64)[order]
            expected_frequency = np.asarray(coefficient.aux.frq, dtype=np.float64)[order]
            expected_percent_energy = np.asarray(coefficient.PE, dtype=np.float64)[order]
            if confidence_interval == "linear":
                expected_amplitude_ci = np.asarray(coefficient.A_ci, dtype=np.float64)[order]
                expected_phase_ci = np.asarray(coefficient.g_ci, dtype=np.float64)[order]
                expected_snr = np.asarray(coefficient.SNR, dtype=np.float64)[order]
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
                phase_error = float(
                    circular_phase_error(
                        phases[position, constituent_position],
                        expected_phase[constituent_position],
                    )
                )
                phase_errors.append(
                    (
                        phase_error,
                        phase_tolerance_degrees(float(expected_amplitude[constituent_position])),
                        node_index,
                        constituent,
                    )
                )
                actual_complex = amplitudes[position, constituent_position] * np.exp(
                    -1j * np.deg2rad(phases[position, constituent_position])
                )
                expected_complex = expected_amplitude[constituent_position] * np.exp(
                    -1j * np.deg2rad(expected_phase[constituent_position])
                )
                errors["complex_coefficient"].append(
                    (abs(actual_complex - expected_complex), node_index, constituent)
                )
                errors["percent_energy"].append(
                    (
                        abs(
                            percent_energy[position, constituent_position]
                            - expected_percent_energy[constituent_position]
                        ),
                        node_index,
                        constituent,
                    )
                )
                if confidence_interval == "linear":
                    for metric, actual, expected in (
                        (
                            "amplitude_ci",
                            amplitude_ci[position, constituent_position],
                            expected_amplitude_ci[constituent_position],
                        ),
                        (
                            "phase_ci_degrees",
                            phase_ci[position, constituent_position],
                            expected_phase_ci[constituent_position],
                        ),
                    ):
                        errors[metric].append(
                            (abs(float(actual - expected)), node_index, constituent)
                        )
                    actual_snr = signal_to_noise[position, constituent_position]
                    expected_constituent_snr = expected_snr[constituent_position]
                    snr_errors.append(
                        (
                            abs(float(actual_snr - expected_constituent_snr)),
                            float(expected_constituent_snr),
                            node_index,
                            constituent,
                        )
                    )
                errors["frequency_cph"].append(
                    (
                        abs(
                            frequencies[position, constituent_position]
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
            if reference_times is not None:
                expected_reference_mjd = float(coefficient.aux.reftime) - MJD_TO_PYTHON_DATENUM
                errors["reference_time_mjd"].append(
                    (abs(reference_times[position] - expected_reference_mjd), node_index, None)
                )
            if reconstruction is not None:
                selected_names = _reconstruction_names(
                    reconstruction_filter,
                    reconstruction_constituents,
                    reconstruction_minimum_pe,
                    reconstruction_minimum_snr,
                    coefficient,
                )
                expected_reconstruction = np.asarray(
                    oracle.reconstruct(
                        reconstruction_time,
                        coefficient,
                        epoch="1858-11-17",
                        constit=selected_names,
                        verbose=False,
                    ).h,
                    dtype=np.float64,
                )
                for actual, expected in zip(
                    reconstruction[:, position],
                    expected_reconstruction,
                    strict=True,
                ):
                    errors["reconstruction"].append(
                        (abs(float(actual - expected)), node_index, None)
                    )

    metrics = {
        name: _maximum_error(values, DEFAULT_TOLERANCES[name]) for name, values in errors.items()
    }
    metrics["phase_degrees"] = _maximum_phase_error(phase_errors)
    if confidence_interval == "linear":
        metrics["signal_to_noise"] = _maximum_snr_error(snr_errors)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "implementation_under_test": "rutide",
        "oracle": "python-utide",
        "profile": profile,
        "constituent_selection": selection_method,
        "confidence_interval": confidence_interval,
        "confidence_noise": confidence_noise,
        "reconstruction_filter": reconstruction_filter,
        "series": series_count,
        "constituents": names,
        "fixture": str(fixture_path),
        "rust_output": str(rust_path),
        "rust_result_sha256": result_digest,
        "metrics": metrics,
        "passed": all(metric["within_tolerance"] for metric in metrics.values()),
    }


def _reconstruction_names(
    reconstruction_filter: str,
    explicit_names: list[str] | None,
    minimum_percent_energy: float | None,
    minimum_signal_to_noise: float | None,
    coefficient: Any,
) -> list[str]:
    names = [str(value) for value in coefficient.name.tolist()]
    if reconstruction_filter == "all":
        return names
    if reconstruction_filter == "constituents":
        if not explicit_names or not set(explicit_names).issubset(names):
            raise ValueError("invalid explicit reconstruction constituent metadata")
        return explicit_names
    if minimum_percent_energy is None:
        raise ValueError("diagnostic reconstruction is missing its PE threshold")
    percent_energy = np.asarray(coefficient.PE, dtype=np.float64)
    selected = percent_energy >= minimum_percent_energy
    if minimum_signal_to_noise is not None:
        if not hasattr(coefficient, "SNR"):
            raise ValueError("diagnostic reconstruction requires unavailable Python SNR")
        selected &= np.asarray(coefficient.SNR, dtype=np.float64) >= minimum_signal_to_noise
    return [name for name, include in zip(names, selected, strict=True) if include]


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
