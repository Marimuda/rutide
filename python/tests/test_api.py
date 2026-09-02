from __future__ import annotations

import importlib.metadata
import unittest
from datetime import datetime, timedelta

import numpy as np
import rutide

M2_CPH = 0.0805114007
S2_CPH = 1.0 / 12.0


def harmonic(
    time_mjd: np.ndarray, frequency_cph: float, amplitude: float, phase: float
) -> np.ndarray:
    centered_days = time_mjd - np.mean(time_mjd)
    return amplitude * np.cos(2 * np.pi * frequency_cph * 24 * centered_days - phase)


class BunchTests(unittest.TestCase):
    def test_attribute_and_mapping_access_share_state(self) -> None:
        bunch = rutide.Bunch(answer=42)
        self.assertEqual(bunch.answer, 42)
        bunch.units = "m"
        self.assertEqual(bunch["units"], "m")
        with self.assertRaises(AttributeError):
            _ = bunch.missing


class ScalarApiTests(unittest.TestCase):
    def test_scalar_fit_missing_values_aliases_and_reconstruction(self) -> None:
        time = 60_000.0 + np.arange(24 * 70, dtype=np.float64) / 24.0
        observations = 0.35 + harmonic(time, M2_CPH, 1.2, 0.4) + harmonic(time, S2_CPH, 0.3, -0.2)
        time[15] = np.nan
        observations[71] = np.nan

        coefficients = rutide.solve(
            time,
            observations,
            lat=62.0,
            constit=["M2", "S2"],
            conf_int="none",
            trend=False,
            phase="raw",
            nodal=False,
            order_constit="frequency",
            verbose=False,
        )

        np.testing.assert_array_equal(coefficients.name, ["M2", "S2"])
        np.testing.assert_array_equal(coefficients.A, coefficients.amplitude)
        np.testing.assert_array_equal(coefficients.g, coefficients.phase_degrees)
        self.assertFalse(coefficients.A.flags.writeable)
        self.assertEqual(coefficients.nobs_original, time.size)
        self.assertEqual(coefficients.nobs, time.size - 2)
        self.assertEqual(coefficients.aux.lat, 62.0)
        self.assertIsNone(coefficients.robust)
        self.assertEqual(rutide.__version__, importlib.metadata.version("rutide"))

        target = time.copy()
        target[15] = 60_000.0 + 15 / 24.0
        reconstructed = rutide.reconstruct(
            target,
            coefficients,
            min_SNR=None,
            verbose=False,
        )
        expected = 0.35 + harmonic(target, M2_CPH, 1.2, 0.4) + harmonic(target, S2_CPH, 0.3, -0.2)
        self.assertLess(float(np.max(np.abs(reconstructed.h - expected))), 5e-5)

    def test_irregular_colored_confidence_uses_lomb_scargle(self) -> None:
        steps = np.where(np.arange(1_400) % 11 == 0, 1.35, 1.0)
        time = 60_100.0 + np.cumsum(steps) / 24.0
        observations = harmonic(time, M2_CPH, 0.8, 0.2)
        observations += 0.015 * np.sin(np.arange(time.size) * 0.37)

        coefficients = rutide.solve(
            time,
            observations,
            lat=58.0,
            constit=["M2", "S2"],
            trend=False,
            conf_int="linear",
            white=False,
            verbose=False,
        )

        self.assertTrue(np.all(np.isfinite(coefficients.A_ci)))
        self.assertTrue(np.all(np.isfinite(coefficients.SNR)))

    def test_masked_arrays_are_normalized_as_missing_rows(self) -> None:
        time = 60_000.0 + np.arange(900) / 24.0
        observations = harmonic(time, M2_CPH, 1.0, 0.0)
        masked_time = np.ma.array(time, mask=np.arange(time.size) == 10)
        masked_observations = np.ma.array(observations, mask=np.arange(observations.size) == 11)
        coefficients = rutide.solve(
            masked_time,
            masked_observations,
            lat=60.0,
            constit=["M2"],
            conf_int="none",
            trend=False,
            verbose=False,
        )
        self.assertEqual(coefficients.nobs, time.size - 2)

    def test_snr_reconstruction_requires_confidence(self) -> None:
        time = 60_000.0 + np.arange(800) / 24.0
        coefficients = rutide.solve(
            time,
            harmonic(time, M2_CPH, 1.0, 0.0),
            lat=60.0,
            constit=["M2"],
            conf_int="none",
            trend=False,
            verbose=False,
        )
        with self.assertRaisesRegex(ValueError, "SNR"):
            rutide.reconstruct(time, coefficients, verbose=False)

    def test_datetime_input_and_nat_output(self) -> None:
        start = datetime(2024, 1, 1)
        datetimes = np.array([start + timedelta(hours=index) for index in range(900)])
        mjd = 60_310.0 + np.arange(datetimes.size) / 24.0
        observations = harmonic(mjd, M2_CPH, 0.7, -0.3)
        coefficients = rutide.solve(
            datetimes,
            observations,
            lat=61.0,
            constit=["M2"],
            conf_int="none",
            trend=False,
            phase="raw",
            nodal=False,
            verbose=False,
        )
        target = datetimes.astype("datetime64[ms]")
        target[10] = np.datetime64("NaT", "ms")
        reconstructed = rutide.reconstruct(
            target,
            coefficients,
            min_SNR=None,
            verbose=False,
        )
        self.assertTrue(np.isnan(reconstructed.h[10]))
        self.assertTrue(np.all(np.isfinite(np.delete(reconstructed.h, 10))))

    def test_numeric_epoch_conversions_reach_the_same_mjd(self) -> None:
        mjd = 60_310.0 + np.arange(900) / 24.0
        observations = harmonic(mjd, M2_CPH, 0.7, -0.3)
        profiles = [
            (mjd, "mjd"),
            (mjd + 678_576.0, "python"),
            (mjd + 678_942.0, "matlab"),
            ((mjd - 40_587.0) * 86_400.0, "unix"),
            (mjd - 60_310.0, "2024-01-01"),
        ]
        references = []
        for time, epoch in profiles:
            coefficients = rutide.solve(
                time,
                observations,
                lat=61.0,
                constit=["M2"],
                conf_int="none",
                trend=False,
                phase="raw",
                nodal=False,
                epoch=epoch,
                verbose=False,
            )
            references.append(coefficients.reference_time_mjd)
        np.testing.assert_allclose(references, references[0], rtol=0.0, atol=2e-10)

    def test_robust_diagnostics_are_numpy_arrays(self) -> None:
        time = 60_000.0 + np.arange(1_200) / 24.0
        observations = harmonic(time, M2_CPH, 1.0, 0.5)
        observations[300] += 20.0
        coefficients = rutide.solve(
            time,
            observations,
            lat=60.0,
            constit=["M2"],
            conf_int="none",
            method="robust",
            trend=False,
            robust_kw={"weight": "cauchy"},
            verbose=False,
        )
        self.assertIsInstance(coefficients.weights, np.ndarray)
        self.assertEqual(coefficients.weights.shape, time.shape)
        self.assertLess(coefficients.weights[300], 0.1)
        self.assertGreaterEqual(coefficients.robust.iterations, 1)

    def test_conflicting_robust_aliases_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "weight_function.*weight"):
            rutide.solve(
                [60_000.0, 60_001.0],
                [1.0, 2.0],
                lat=60.0,
                constit=["M2"],
                method="robust",
                robust_kw={"weight_function": "cauchy", "weight": "fair"},
                verbose=False,
            )

    def test_monte_carlo_confidence_with_exact_inference(self) -> None:
        time = 60_000.0 + np.arange(1_600) / 24.0
        observations = harmonic(time, M2_CPH, 1.0, 0.2)
        observations += harmonic(time, S2_CPH, 0.2, 0.45)
        observations += 0.01 * np.sin(np.arange(time.size) * 0.29)
        coefficients = rutide.solve(
            time,
            observations,
            lat=60.0,
            constit=["M2", "S2"],
            conf_int="MC",
            MC_n=64,
            MC_seed=7,
            trend=False,
            white=True,
            infer={
                "inferred_names": ["S2"],
                "reference_names": ["M2"],
                "amp_ratios": [0.2],
                "phase_offsets": [15.0],
                "approximate": False,
            },
            verbose=False,
        )
        np.testing.assert_array_equal(np.sort(coefficients.name), ["M2", "S2"])
        self.assertTrue(np.all(np.isfinite(coefficients.A_ci)))


class VectorApiTests(unittest.TestCase):
    def test_vector_ellipse_fit_and_reconstruction(self) -> None:
        time = 60_000.0 + np.arange(24 * 70) / 24.0
        eastward = 0.1 + harmonic(time, M2_CPH, 0.8, 0.15)
        northward = -0.2 + harmonic(time, M2_CPH, 0.45, -0.7)
        coefficients = rutide.solve(
            time,
            eastward,
            northward,
            lat=62.0,
            constit=["M2"],
            conf_int="linear",
            white=True,
            trend=False,
            phase="raw",
            nodal=False,
            verbose=False,
        )
        self.assertIn("Lsmaj", coefficients)
        self.assertIn("theta", coefficients)
        self.assertTrue(np.isfinite(coefficients.Lsmaj_ci[0]))
        reconstructed = rutide.reconstruct(time, coefficients, min_SNR=None, verbose=False)
        self.assertLess(float(np.max(np.abs(reconstructed.u - eastward))), 5e-5)
        self.assertLess(float(np.max(np.abs(reconstructed.v - northward))), 5e-5)

    def test_vector_exact_inference_and_joint_missing_mask(self) -> None:
        time = 60_000.0 + np.arange(1_600) / 24.0
        eastward = harmonic(time, M2_CPH, 0.8, 0.2)
        northward = harmonic(time, M2_CPH, 0.4, -0.5)
        eastward += harmonic(time, S2_CPH, 0.1, 0.4)
        northward += harmonic(time, S2_CPH, 0.05, -0.1)
        eastward[20] = np.nan
        northward[21] = np.nan
        coefficients = rutide.solve(
            time,
            eastward,
            northward,
            lat=62.0,
            constit=["M2", "S2"],
            conf_int="linear",
            white=True,
            trend=False,
            infer={
                "inferred_names": ["S2"],
                "reference_names": ["M2"],
                "amp_ratios": [0.15, 0.1],
                "phase_offsets": [10.0, -20.0],
                "approximate": False,
            },
            verbose=False,
        )
        self.assertEqual(coefficients.nobs, time.size - 2)
        np.testing.assert_array_equal(np.sort(coefficients.name), ["M2", "S2"])
        self.assertTrue(np.all(np.isfinite(coefficients.Lsmaj_ci)))


if __name__ == "__main__":
    unittest.main()
