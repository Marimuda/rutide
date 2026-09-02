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
        vertical_mode = (
            str(dataset.getncattr("vertical_mode"))
            if "vertical_mode" in dataset.ncattrs()
            else "depth-averaged"
        )
        element_indices = np.asarray(dataset.variables["element_index"][:], dtype=np.int64)
        latitudes = _finite(dataset.variables["latitude"][:], "output latitude")
        if vertical_mode == "sigma-layer":
            selected_layers = np.asarray(dataset.variables["siglay_index"][:], dtype=np.int64)
            selected_depths: np.ndarray | None = None
            indices = np.tile(element_indices, len(selected_layers))
            series_layers: np.ndarray | None = np.repeat(selected_layers, len(element_indices))
            series_depths: np.ndarray | None = None
            latitudes = np.tile(latitudes, len(selected_layers))
        elif vertical_mode == "fixed-depth":
            selected_layers = None
            selected_depths = _finite(dataset.variables["depth"][:], "output depth")
            indices = np.tile(element_indices, len(selected_depths))
            series_layers = None
            series_depths = np.repeat(selected_depths, len(element_indices))
            latitudes = np.tile(latitudes, len(selected_depths))
        elif vertical_mode == "depth-averaged":
            selected_layers = None
            selected_depths = None
            indices = element_indices
            series_layers = None
            series_depths = None
        else:
            raise ValueError(f"unsupported vertical mode: {vertical_mode}")

        def series_values(name: str, description: str) -> np.ndarray:
            values = _finite(dataset.variables[name][:], description)
            if vertical_mode in {"sigma-layer", "fixed-depth"}:
                return values.reshape((-1, *values.shape[2:]))
            return values

        result: dict[str, Any] = {
            "profile": str(dataset.getncattr("profile")),
            "selection": str(dataset.getncattr("constituent_selection")),
            "confidence": confidence,
            "confidence_noise": (
                str(dataset.getncattr("confidence_noise")) if confidence == "linear" else None
            ),
            "names": str(dataset.getncattr("constituent_names")).split(","),
            "vertical_mode": vertical_mode,
            "indices": indices,
            "layer_indices": series_layers,
            "depths_meters": series_depths,
            "latitudes": latitudes,
            "observation_counts": series_values(
                "observation_count", "output observation count"
            ).astype(np.int64),
            "reference_times": series_values("reference_time", "output reference time"),
            "frequencies": series_values("frequency", "output frequency"),
            "semi_major": series_values("semi_major", "output semi-major"),
            "semi_minor": series_values("semi_minor", "output semi-minor"),
            "inclination": series_values("inclination", "output inclination"),
            "phase": series_values("phase", "output phase"),
            "percent_energy": series_values("percent_energy", "output percent energy"),
            "eastward_mean": series_values("eastward_mean", "output eastward mean"),
            "northward_mean": series_values("northward_mean", "output northward mean"),
            "eastward_slope": series_values("eastward_slope", "output eastward slope"),
            "northward_slope": series_values("northward_slope", "output northward slope"),
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
                result[name] = series_values(name, f"output {name}")
        selection = result["selection"]
        result["rayleigh_min"] = (
            float(dataset.getncattr("rayleigh_min")) if selection == "rayleigh" else None
        )
    return result


def _validate_output_shapes(output: dict[str, Any]) -> None:
    series_count = len(output["indices"])
    constituent_count = len(output["names"])
    if series_count == 0:
        raise ValueError("vector series must be non-empty")
    vertical_coordinates = (
        output["layer_indices"]
        if output["layer_indices"] is not None
        else output["depths_meters"]
        if output["depths_meters"] is not None
        else np.full(series_count, -1, dtype=np.int64)
    )
    coordinates = list(zip(vertical_coordinates, output["indices"], strict=True))
    if (
        len(set((float(vertical), int(element)) for vertical, element in coordinates))
        != series_count
    ):
        raise ValueError("vertical-element coordinates must be unique")
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
        "fixed-constituents-greenwich-nodal-sigma-layer-vector-ols",
        "rayleigh-auto-greenwich-nodal-sigma-layer-vector-ols",
        "fixed-constituents-greenwich-nodal-fixed-depth-vector-ols",
        "rayleigh-auto-greenwich-nodal-fixed-depth-vector-ols",
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
    errors: dict[str, list[tuple[float, int, str | None]]] = {name: [] for name in metric_names}
    snr_errors: list[tuple[float, float, int, str]] = []
    raw_angle_errors = {"inclination_degrees": 0.0, "phase_degrees": 0.0}
    fixture_path = fixture.resolve(strict=True)
    with netCDF4.Dataset(fixture_path, mode="r") as source:
        time = reconstruct_mjd(source.variables["Itime"][:], source.variables["Itime2"][:])
        element_count = len(source.dimensions["nele"])
        for position, raw_element_index in enumerate(output["indices"]):
            element_index = int(raw_element_index)
            layer_index = (
                int(output["layer_indices"][position])
                if output["layer_indices"] is not None
                else None
            )
            depth_meters = (
                float(output["depths_meters"][position])
                if output["depths_meters"] is not None
                else None
            )
            if not 0 <= element_index < element_count:
                raise ValueError(f"element index is out of range: {element_index}")
            latitude = float(source.variables["latc"][element_index])
            if output["latitudes"][position] != latitude:
                raise ValueError(f"latitude differs at element {element_index}")
            if depth_meters is not None:
                eastward_source, northward_source = _fixed_depth_current(
                    source, element_index, depth_meters
                )
                source_label = f"depth {depth_meters} m, element {element_index}"
            elif layer_index is None:
                eastward_source = source.variables["ua"][:, element_index]
                northward_source = source.variables["va"][:, element_index]
                source_label = f"element {element_index}"
            else:
                layer_count = len(source.dimensions["siglay"])
                if not 0 <= layer_index < layer_count:
                    raise ValueError(f"sigma-layer index is out of range: {layer_index}")
                eastward_source = source.variables["u"][:, layer_index, element_index]
                northward_source = source.variables["v"][:, layer_index, element_index]
                source_label = f"layer {layer_index}, element {element_index}"
            eastward = _observations_with_nan(
                eastward_source,
                f"source eastward current at {source_label}",
            )
            northward = _observations_with_nan(
                northward_source,
                f"source northward current at {source_label}",
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
            coefficient_errors = np.max(np.abs(actual_coefficients - expected_coefficients), axis=1)
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
                            (output["phase"][position] - expected["phase"] + 180.0) % 360.0 - 180.0
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

    metrics = {name: _maximum_error(values, TOLERANCES[name]) for name, values in errors.items()}
    if snr_errors:
        metrics["signal_to_noise"] = _maximum_snr_error(snr_errors)
    passed = all(metric["within_tolerance"] for metric in metrics.values())
    return {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "implementation": "rutide-vs-python-utide-vector",
        "rust_output": str(rust_output.resolve(strict=True)),
        "fixture": str(fixture_path),
        "series": len(output["indices"]),
        "vertical_mode": output["vertical_mode"],
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


def _fixed_depth_current(
    source: netCDF4.Dataset,
    element_index: int,
    target_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen RUTide FVCOM centroid/layer-centre interpolation."""
    nodes = np.asarray(source.variables["nv"][:, element_index], dtype=np.int64) - 1
    if nodes.shape != (3,) or np.any(nodes < 0) or np.any(nodes >= len(source.dimensions["node"])):
        raise ValueError(f"invalid FVCOM connectivity at element {element_index}")
    sigma = _observations_with_nan(
        source.variables["siglay"][:, nodes], f"siglay at element {element_index}"
    )
    bathymetry = _observations_with_nan(
        source.variables["h"][nodes], f"bathymetry at element {element_index}"
    )
    surface = _observations_with_nan(
        source.variables["zeta"][:, nodes], f"surface at element {element_index}"
    )
    wet = _observations_with_nan(
        source.variables["wet_cells"][:, element_index], f"wet mask at element {element_index}"
    )
    eastward_layers = _observations_with_nan(
        source.variables["u"][:, :, element_index], f"eastward layers at element {element_index}"
    )
    northward_layers = _observations_with_nan(
        source.variables["v"][:, :, element_index], f"northward layers at element {element_index}"
    )
    layer_depths = np.mean(
        -sigma[None, :, :] * (bathymetry[None, None, :] + surface[:, None, :]), axis=2
    )
    eastward = np.full(layer_depths.shape[0], np.nan, dtype=np.float64)
    northward = np.full(layer_depths.shape[0], np.nan, dtype=np.float64)
    geometry_valid = np.isfinite(layer_depths).all(axis=1) & np.all(
        np.diff(layer_depths, axis=1) > 0.0, axis=1
    )
    for layer in range(layer_depths.shape[1] - 1):
        lower = layer_depths[:, layer]
        upper = layer_depths[:, layer + 1]
        selected = (
            geometry_valid
            & (wet == 1.0)
            & (target_depth >= lower)
            & (target_depth <= upper)
            & ~np.isfinite(eastward)
        )
        weight = np.divide(
            target_depth - lower,
            upper - lower,
            out=np.zeros_like(lower),
            where=selected,
        )
        u0 = eastward_layers[:, layer]
        u1 = eastward_layers[:, layer + 1]
        v0 = northward_layers[:, layer]
        v1 = northward_layers[:, layer + 1]
        exact_lower = selected & (target_depth == lower) & np.isfinite(u0) & np.isfinite(v0)
        exact_upper = selected & (target_depth == upper) & np.isfinite(u1) & np.isfinite(v1)
        interpolated = (
            selected
            & ~exact_lower
            & ~exact_upper
            & np.isfinite(u0)
            & np.isfinite(u1)
            & np.isfinite(v0)
            & np.isfinite(v1)
        )
        eastward[exact_lower] = u0[exact_lower]
        northward[exact_lower] = v0[exact_lower]
        eastward[exact_upper] = u1[exact_upper]
        northward[exact_upper] = v1[exact_upper]
        eastward[interpolated] = u0[interpolated] + weight[interpolated] * (
            u1[interpolated] - u0[interpolated]
        )
        northward[interpolated] = v0[interpolated] + weight[interpolated] * (
            v1[interpolated] - v0[interpolated]
        )
    return eastward, northward


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
