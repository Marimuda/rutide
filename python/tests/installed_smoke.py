"""Smoke-test an installed RUTide distribution, including its native extension."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import rutide
import rutide._native as native

M2_CPH = 1.932_273_6 / 24.0
S2_CPH = 2.0 / 24.0


def harmonic(
    time_mjd: np.ndarray, frequency_cph: float, amplitude: float, phase: float
) -> np.ndarray:
    hours = (time_mjd - np.mean(time_mjd)) * 24.0
    return amplitude * np.cos(2.0 * np.pi * frequency_cph * hours + phase)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: installed_smoke.py EXPECTED_VERSION")
    expected_version = sys.argv[1]
    if rutide.__version__ != expected_version:
        raise RuntimeError(
            f"installed RUTide version {rutide.__version__!r} != {expected_version!r}"
        )

    time_mjd = 60_000.0 + np.arange(1_000, dtype=np.float64) / 24.0
    height = harmonic(time_mjd, M2_CPH, 0.8, 0.2)
    scalar = rutide.solve(
        time_mjd,
        height,
        lat=60.0,
        constit=["M2"],
        conf_int="none",
        trend=False,
        phase="raw",
        nodal=False,
        verbose=False,
    )
    restored_height = rutide.reconstruct(time_mjd, scalar, min_SNR=None, verbose=False).h
    np.testing.assert_allclose(restored_height, height, rtol=5e-6, atol=5e-6)

    eastward = np.column_stack(
        [
            harmonic(time_mjd, M2_CPH, 0.7, 0.1),
            harmonic(time_mjd, M2_CPH, 0.9, -0.2) + harmonic(time_mjd, S2_CPH, 0.2, 0.4),
        ]
    )
    northward = np.column_stack(
        [
            harmonic(time_mjd, M2_CPH, 0.3, -0.4),
            harmonic(time_mjd, M2_CPH, 0.25, 0.3) + harmonic(time_mjd, S2_CPH, 0.1, -0.1),
        ]
    )
    coefficients = rutide.solve_many(
        time_mjd,
        eastward,
        northward,
        lat=[59.0, 61.0],
        constit=["M2", "S2"],
        conf_int="linear",
        white=True,
        trend=False,
        phase="raw",
        nodal=False,
        workers=2,
        verbose=False,
    )
    currents = rutide.reconstruct_many(time_mjd, coefficients, min_SNR=None, verbose=False)
    np.testing.assert_allclose(currents.u, eastward, rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(currents.v, northward, rtol=5e-6, atol=5e-6)

    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "installed-smoke.rutide.npz"
        coefficients.save(archive)
        loaded = rutide.load(archive, workers=1)
        loaded_currents = rutide.reconstruct_many(time_mjd, loaded, min_SNR=None, verbose=False)
    np.testing.assert_array_equal(loaded_currents.u, currents.u)
    np.testing.assert_array_equal(loaded_currents.v, currents.v)

    print(
        json.dumps(
            {
                "rutide_version": rutide.__version__,
                "native_extension": str(Path(native.__file__).resolve()),
                "scalar_observations": len(time_mjd),
                "vector_series": int(coefficients.Lsmaj.shape[0]),
                "coefficient_archive_round_trip": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
