"""Compare RUTide vector-current output with the pinned Python UTide oracle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from .compare import (
    SIGNAL_TO_NOISE_ABSOLUTE_TOLERANCE,
    SIGNAL_TO_NOISE_RELATIVE_TOLERANCE,
    _finite,
    _maximum_error,
    _maximum_snr_error,
    _observations_with_nan,
)
from .constants import DEFAULT_FIXTURE, DEFAULT_UTIDE_ROOT
from .fixture import reconstruct_mjd
from .runner import _profile_options, load_oracle

TOLERANCES = {
    "semi_major": 5e-12,
    "semi_minor": 5e-12,
    "component_coefficient": 5e-12,
    "percent_energy": 1e-9,
    "semi_major_ci": 1e-9,
    "semi_minor_ci": 1e-9,
    "inclination_ci_degrees": 1e-5,
    "phase_ci_degrees": 1e-5,
    "eastward_mean": 5e-12,
    "northward_mean": 5e-12,
    "eastward_slope_per_day": 5e-12,
    "northward_slope_per_day": 5e-12,
    "frequency_cph": 1e-15,
    "reference_time_mjd": 1e-12,
}


def ellipse_component_coefficients(
    semi_major: np.ndarray,
    semi_minor: np.ndarray,
    inclination_degrees: np.ndarray,
    phase_degrees: np.ndarray,
) -> np.ndarray:
    """Convert ellipse parameters to east/north cosine/sine coefficients."""
    theta = np.radians(inclination_degrees)
    phase = np.radians(phase_degrees)
    positive = 0.5 * (semi_major + semi_minor) * np.exp(1j * (theta - phase))
    negative = 0.5 * (semi_major - semi_minor) * np.exp(1j * (theta + phase))
    return np.stack(
        (
            np.real(positive + negative),
            -(np.imag(positive) - np.imag(negative)),
            np.imag(positive + negative),
            np.real(positive) - np.real(negative),
        ),
        axis=-1,
    )


def _read_output(path: Path) -> dict[str, Any]:
    with netCDF4.Dataset(path.resolve(strict=True), mode="r") as dataset:
        confidence = str(dataset.getncattr("confidence_interval"))
        result: dict[str, Any] = {
            "profile": str(dataset.getncattr("profile")),
            "selection": str(dataset.getncattr("constituent_selection")),
            "confidence": confidence,
            "confidence_noise": (
                str(dataset.getncattr("confidence_noise")) if confidence == "linear" else None
            ),
            "names": str(dataset.getncattr("constituent_names")).split(","),
            "indices": np.asarray(dataset.variables["element_index"][:], dtype=np.int64),
            "latitudes": _finite(dataset.variables["latitude"][:], "output latitude"),
            "observation_counts": np.asarray(
                dataset.variables["observation_count"][:], dtype=np.int64
            ),
            "reference_times": _finite(
                dataset.variables["reference_time"][:], "output reference time"
            ),
            "frequencies": _finite(dataset.variables["frequency"][:], "output frequency"),
            "semi_major": _finite(dataset.variables["semi_major"][:], "output semi-major"),
            "semi_minor": _finite(dataset.variables["semi_minor"][:], "output semi-minor"),
            "inclination": _finite(
                dataset.variables["inclination"][:], "output inclination"
            ),
            "phase": _finite(dataset.variables["phase"][:], "output phase"),
            "percent_energy": _finite(
                dataset.variables["percent_energy"][:], "output percent energy"
            ),
            "eastward_mean": _finite(
                dataset.variables["eastward_mean"][:], "output eastward mean"
            ),
            "northward_mean": _finite(
                dataset.variables["northward_mean"][:], "output northward mean"
            ),
            "eastward_slope": _finite(
                dataset.variables["eastward_slope"][:], "output eastward slope"
            ),
            "northward_slope": _finite(
                dataset.variables["northward_slope"][:], "output northward slope"
            ),
            "result_sha256": str(dataset.getncattr("result_sha256")),
        }
        if confidence == "linear":
            for name in (
                "semi_major_ci",
                "semi_minor_ci",
                "inclination_ci",
                "phase_ci",
                "signal_to_noise",
            ):
                result[name] = _finite(dataset.variables[name][:], f"output {name}")
        selection = result["selection"]
        result["rayleigh_min"] = (
            float(dataset.getncattr("rayleigh_min")) if selection == "rayleigh" else None
        )
    return result


def _validate_output_shapes(output: dict[str, Any]) -> None:
    series_count = len(output["indices"])
    constituent_count = len(output["names"])
    if series_count == 0 or len(set(int(value) for value in output["indices"])) != series_count:
        raise ValueError("element indices must be non-empty and unique")
    if constituent_count == 0 or len(set(output["names"])) != constituent_count:
        raise ValueError("constituent names must be non-empty and unique")
    matrix_shape = (series_count, constituent_count)
    for name in (
        "frequencies",
        "semi_major",
        "semi_minor",
        "inclination",
        "phase",
        "percent_energy",
    ):
        if output[name].shape != matrix_shape:
            raise ValueError(f"unexpected {name} shape: {output[name].shape}")
    for name in (
        "latitudes",
        "observation_counts",
        "reference_times",
        "eastward_mean",
        "northward_mean",
        "eastward_slope",
        "northward_slope",
    ):
        if output[name].shape != (series_count,):
            raise ValueError(f"unexpected {name} shape: {output[name].shape}")
    if output["confidence"] == "linear":
        for name in (
            "semi_major_ci",
            "semi_minor_ci",
            "inclination_ci",
            "phase_ci",
            "signal_to_noise",
        ):
            if output[name].shape != matrix_shape:
                raise ValueError(f"unexpected {name} shape: {output[name].shape}")
    elif output["confidence"] != "none":
        raise ValueError(f"unsupported confidence method: {output['confidence']}")


def compare_vector_with_oracle(
    rust_output: Path,
    fixture: Path,
    utide_root: Path,
) -> dict[str, Any]:
    """Compare every vector series in a RUTide output with Python UTide."""
    oracle = load_oracle(utide_root)
    output = _read_output(rust_output)
    _validate_output_shapes(output)
    if output["profile"] not in {
        "fixed-constituents-greenwich-nodal-vector-ols",
        "rayleigh-auto-greenwich-nodal-vector-ols",
    }:
        raise ValueError("RUTide output uses an unsupported vector profile")
    options = _profile_options("fixed-constituents")
    options["conf_int"] = output["confidence"]
    options["white"] = output["confidence_noise"] == "white"
    if output["confidence"] == "linear" and output["confidence_noise"] not in {
        "white",
        "colored",
    }:
        raise ValueError("linear confidence requires white or colored noise metadata")
    if output["selection"] == "explicit":
        options["constit"] = output["names"]
    elif output["selection"] == "rayleigh":
        options["constit"] = "auto"
        options["Rayleigh_min"] = output["rayleigh_min"]
    else:
        raise ValueError(f"unsupported constituent selection: {output['selection']}")

    metric_names = list(TOLERANCES)
    if output["confidence"] != "linear":
        metric_names = [name for name in metric_names if "_ci" not in name]
    errors: dict[str, list[tuple[float, int, str | None]]] = {
        name: [] for name in metric_names
    }
    snr_errors: list[tuple[float, float, int, str]] = []
    raw_angle_errors = {"inclination_degrees": 0.0, "phase_degrees": 0.0}
    fixture_path = fixture.resolve(strict=True)
    with netCDF4.Dataset(fixture_path, mode="r") as source:
        time = reconstruct_mjd(source.variables["Itime"][:], source.variables["Itime2"][:])
        element_count = len(source.dimensions["nele"])
        for position, raw_element_index in enumerate(output["indices"]):
            element_index = int(raw_element_index)
            if not 0 <= element_index < element_count:
                raise ValueError(f"element index is out of range: {element_index}")
            latitude = float(source.variables["latc"][element_index])
            if output["latitudes"][position] != latitude:
                raise ValueError(f"latitude differs at element {element_index}")
            eastward = _observations_with_nan(
                source.variables["ua"][:, element_index],
                f"source ua at element {element_index}",
            )
            northward = _observations_with_nan(
                source.variables["va"][:, element_index],
                f"source va at element {element_index}",
            )
            valid = np.isfinite(eastward) & np.isfinite(northward)
            if output["observation_counts"][position] != int(valid.sum()):
                raise ValueError(f"observation count differs at element {element_index}")
            coefficient = oracle.solve(
                time,
                np.where(valid, eastward, np.nan),
                np.where(valid, northward, np.nan),
                lat=latitude,
                **options,
            )
            oracle_names = [str(value) for value in coefficient.name.tolist()]
            if set(oracle_names) != set(output["names"]):
                raise ValueError(
                    f"constituent selection differs: rust={output['names']}, python={oracle_names}"
                )
            order = [oracle_names.index(name) for name in output["names"]]
            expected = {
                "semi_major": np.asarray(coefficient.Lsmaj)[order],
                "semi_minor": np.asarray(coefficient.Lsmin)[order],
                "inclination": np.asarray(coefficient.theta)[order],
                "phase": np.asarray(coefficient.g)[order],
                "percent_energy": np.asarray(coefficient.PE)[order],
                "frequencies": np.asarray(coefficient.aux.frq)[order],
            }
            names = output["names"]
            for metric, output_name in (
                ("semi_major", "semi_major"),
                ("semi_minor", "semi_minor"),
                ("percent_energy", "percent_energy"),
                ("frequency_cph", "frequencies"),
            ):
                values = np.abs(output[output_name][position] - expected[output_name])
                errors[metric].extend(
                    (float(value), element_index, name)
                    for value, name in zip(values, names, strict=True)
                )
            actual_coefficients = ellipse_component_coefficients(
                output["semi_major"][position],
                output["semi_minor"][position],
                output["inclination"][position],
                output["phase"][position],
            )
            expected_coefficients = ellipse_component_coefficients(
                expected["semi_major"],
                expected["semi_minor"],
                expected["inclination"],
                expected["phase"],
            )
            coefficient_errors = np.max(
                np.abs(actual_coefficients - expected_coefficients), axis=1
            )
            errors["component_coefficient"].extend(
                (float(value), element_index, name)
                for value, name in zip(coefficient_errors, names, strict=True)
            )
            raw_angle_errors["inclination_degrees"] = max(
                raw_angle_errors["inclination_degrees"],
                float(
                    np.max(
                        np.abs(
                            (output["inclination"][position] - expected["inclination"] + 90.0)
                            % 180.0
                            - 90.0
                        )
                    )
                ),
            )
            raw_angle_errors["phase_degrees"] = max(
                raw_angle_errors["phase_degrees"],
                float(
                    np.max(
                        np.abs(
                            (output["phase"][position] - expected["phase"] + 180.0) % 360.0
                            - 180.0
                        )
                    )
                ),
            )
            scalar_expected = {
                "eastward_mean": float(coefficient.umean),
                "northward_mean": float(coefficient.vmean),
                "eastward_slope_per_day": float(coefficient.uslope),
                "northward_slope_per_day": float(coefficient.vslope),
                "reference_time_mjd": float(np.mean(time[valid])),
            }
            for metric, output_name in (
                ("eastward_mean", "eastward_mean"),
                ("northward_mean", "northward_mean"),
                ("eastward_slope_per_day", "eastward_slope"),
                ("northward_slope_per_day", "northward_slope"),
                ("reference_time_mjd", "reference_times"),
            ):
                errors[metric].append(
                    (
                        abs(float(output[output_name][position]) - scalar_expected[metric]),
                        element_index,
                        None,
                    )
                )
            if output["confidence"] == "linear":
                for metric, output_name, coefficient_name in (
                    ("semi_major_ci", "semi_major_ci", "Lsmaj_ci"),
                    ("semi_minor_ci", "semi_minor_ci", "Lsmin_ci"),
                    ("inclination_ci_degrees", "inclination_ci", "theta_ci"),
                    ("phase_ci_degrees", "phase_ci", "g_ci"),
                ):
                    expected_values = np.asarray(coefficient[coefficient_name])[order]
                    values = np.abs(output[output_name][position] - expected_values)
                    errors[metric].extend(
                        (float(value), element_index, name)
                        for value, name in zip(values, names, strict=True)
                    )
                expected_snr = np.asarray(coefficient.SNR)[order]
                snr_errors.extend(
                    (
                        abs(float(actual - expected_value)),
                        float(expected_value),
                        element_index,
                        name,
                    )
                    for actual, expected_value, name in zip(
                        output["signal_to_noise"][position], expected_snr, names, strict=True
                    )
                )

    metrics = {
        name: _maximum_error(values, TOLERANCES[name]) for name, values in errors.items()
    }
    if snr_errors:
        metrics["signal_to_noise"] = _maximum_snr_error(snr_errors)
    passed = all(metric["within_tolerance"] for metric in metrics.values())
    return {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "implementation": "rutide-vs-python-utide-vector",
        "rust_output": str(rust_output.resolve(strict=True)),
        "fixture": str(fixture_path),
        "series": len(output["indices"]),
        "constituents": output["names"],
        "confidence_interval": output["confidence"],
        "confidence_noise": output["confidence_noise"],
        "result_sha256": output["result_sha256"],
        "raw_maximum_angle_errors": raw_angle_errors,
        "signal_to_noise_tolerances": {
            "absolute": SIGNAL_TO_NOISE_ABSOLUTE_TOLERANCE,
            "relative": SIGNAL_TO_NOISE_RELATIVE_TOLERANCE,
        },
        "metrics": metrics,
        "passed": passed,
    }


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse vector-comparison arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--utide-root", type=Path, default=DEFAULT_UTIDE_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run vector comparison and return nonzero on a tolerance failure."""
    args = parse_args(arguments)
    report = compare_vector_with_oracle(args.rust_output, args.fixture, args.utide_root)
    text = json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote comparison result: {args.output}")
    else:
        print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
