from __future__ import annotations

import importlib.metadata
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

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


class BatchApiTests(unittest.TestCase):
    def test_scalar_batch_matches_individual_irregular_colored_fits(self) -> None:
        count = 1_200
        series_count = 4
        steps = np.where(np.arange(count) % 17 == 0, 1.2, 1.0)
        time = 60_100.0 + np.cumsum(steps) / 24.0
        observations = np.column_stack(
            [
                0.1 * series
                + harmonic(time, M2_CPH, 0.8 + 0.05 * series, 0.1 * series)
                + harmonic(time, S2_CPH, 0.2, -0.05 * series)
                + 0.01 * np.sin(np.arange(count) * (0.2 + 0.01 * series))
                for series in range(series_count)
            ]
        )
        observations[30, 1] = np.nan
        observations[31:34, 3] = np.nan
        time[7] = np.nan
        latitudes = np.linspace(57.0, 63.0, series_count)

        batch = rutide.solve_many(
            time,
            observations,
            lat=latitudes,
            constit=["M2", "S2"],
            conf_int="linear",
            white=False,
            trend=False,
            phase="raw",
            nodal=False,
            order_constit="frequency",
            workers=2,
            memory_limit_mb=0.02,
            verbose=False,
        )

        self.assertIsInstance(batch, rutide.CoefficientBatch)
        self.assertEqual(batch.A.shape, (series_count, 2))
        self.assertEqual(batch.frequency_cph.shape, (series_count, 2))
        self.assertEqual(batch.rank_index.shape, (series_count, 2))
        self.assertEqual(batch.worker_count, 2)
        self.assertLess(batch.chunk_series, series_count)
        np.testing.assert_array_equal(batch.aux.time_position, np.delete(np.arange(count), 7))
        self.assertFalse(batch.A.flags.writeable)

        for series in range(series_count):
            coefficient = rutide.solve(
                time,
                observations[:, series],
                lat=float(latitudes[series]),
                constit=["M2", "S2"],
                conf_int="linear",
                white=False,
                trend=False,
                phase="raw",
                nodal=False,
                order_constit="frequency",
                verbose=False,
            )
            for batch_name, single_name in [
                ("A", "A"),
                ("g", "g"),
                ("A_ci", "A_ci"),
                ("g_ci", "g_ci"),
                ("PE", "PE"),
                ("SNR", "SNR"),
            ]:
                with self.subTest(series=series, field=batch_name):
                    np.testing.assert_allclose(
                        batch[batch_name][series],
                        coefficient[single_name],
                        rtol=1e-9 if batch_name == "SNR" else 1e-10,
                        atol=2e-11,
                    )
            self.assertAlmostEqual(batch.mean[series], coefficient.mean, places=12)
            self.assertAlmostEqual(
                batch.reference_time_mjd[series], coefficient.reference_time_mjd, places=12
            )

        target = time.copy()
        target[7] = 60_100.0 + 7 / 24.0
        reconstructed = rutide.reconstruct_many(target, batch, min_SNR=None, verbose=False)
        self.assertEqual(reconstructed.h.shape, observations.shape)
        for series in range(series_count):
            coefficient = rutide.solve(
                time,
                observations[:, series],
                lat=float(latitudes[series]),
                constit=["M2", "S2"],
                conf_int="linear",
                trend=False,
                phase="raw",
                nodal=False,
                order_constit="frequency",
                verbose=False,
            )
            expected = rutide.reconstruct(target, coefficient, min_SNR=None, verbose=False)
            np.testing.assert_allclose(
                reconstructed.h[:, series], expected.h, rtol=2e-11, atol=2e-11
            )

    def test_vector_batch_inference_joint_masks_and_reconstruction(self) -> None:
        count = 1_400
        series_count = 3
        time = 60_200.0 + np.arange(count) / 24.0
        eastward = np.column_stack(
            [
                harmonic(time, M2_CPH, 0.7 + 0.1 * series, 0.2)
                + harmonic(time, S2_CPH, 0.14 + 0.02 * series, 0.4)
                for series in range(series_count)
            ]
        )
        northward = np.column_stack(
            [
                harmonic(time, M2_CPH, 0.3, -0.25 + 0.1 * series)
                + harmonic(time, S2_CPH, 0.06, -0.05 + 0.1 * series)
                for series in range(series_count)
            ]
        )
        eastward[20, 0] = np.nan
        northward[21, 0] = np.nan
        northward[30, 2] = np.nan
        inference = {
            "inferred_names": ["S2"],
            "reference_names": ["M2"],
            "amp_ratios": [0.2, 0.2],
            "phase_offsets": [0.0, 0.0],
            "approximate": False,
        }

        batch = rutide.solve_many(
            time,
            eastward,
            northward,
            lat=60.0,
            constit=["M2", "S2"],
            infer=inference,
            conf_int="none",
            trend=False,
            phase="raw",
            nodal=False,
            order_constit="frequency",
            workers=2,
            verbose=False,
        )

        np.testing.assert_array_equal(batch.nobs, [count - 2, count, count - 1])
        self.assertEqual(batch.Lsmaj.shape, (series_count, 2))
        currents = rutide.reconstruct_many(time, batch, min_SNR=None, verbose=False)
        self.assertEqual(currents.u.shape, eastward.shape)
        self.assertEqual(currents.v.shape, northward.shape)
        for series in range(series_count):
            coefficient = rutide.solve(
                time,
                eastward[:, series],
                northward[:, series],
                lat=60.0,
                constit=["M2", "S2"],
                infer=inference,
                conf_int="none",
                trend=False,
                phase="raw",
                nodal=False,
                order_constit="frequency",
                verbose=False,
            )
            np.testing.assert_allclose(
                batch.Lsmaj[series], coefficient.Lsmaj, rtol=2e-11, atol=2e-11
            )
            expected = rutide.reconstruct(time, coefficient, min_SNR=None, verbose=False)
            np.testing.assert_allclose(currents.u[:, series], expected.u, rtol=2e-11, atol=2e-11)
            np.testing.assert_allclose(currents.v[:, series], expected.v, rtol=2e-11, atol=2e-11)

    def test_batch_monte_carlo_is_worker_and_chunk_invariant(self) -> None:
        count = 1_000
        series_count = 5
        time = 60_000.0 + np.arange(count) / 24.0
        observations = np.column_stack(
            [
                harmonic(time, M2_CPH, 1.0 + 0.03 * series, 0.1 * series)
                + 0.02 * np.sin(np.arange(count) * (0.31 + 0.01 * series))
                for series in range(series_count)
            ]
        )
        observations[100, :] = np.nan
        common = dict(
            lat=np.linspace(58.0, 62.0, series_count),
            constit=["M2"],
            conf_int="MC",
            MC_n=48,
            MC_seed=99,
            trend=False,
            phase="raw",
            nodal=False,
            verbose=False,
        )
        serial = rutide.solve_many(time, observations, workers=1, memory_limit_mb=None, **common)
        parallel = rutide.solve_many(time, observations, workers=3, memory_limit_mb=0.01, **common)
        for field in ["A", "g", "A_ci", "g_ci", "PE", "SNR"]:
            np.testing.assert_array_equal(serial[field], parallel[field])

    def test_robust_batch_exposes_dense_mask_aligned_diagnostics(self) -> None:
        count = 1_000
        time = 60_000.0 + np.arange(count) / 24.0
        observations = np.column_stack(
            [harmonic(time, M2_CPH, 1.0, 0.1), harmonic(time, M2_CPH, 0.8, -0.2)]
        )
        observations[100, 0] = np.nan
        observations[400, 1] += 15.0
        coefficients = rutide.solve_many(
            time,
            observations,
            lat=60.0,
            constit=["M2"],
            conf_int="none",
            method="robust",
            trend=False,
            robust_kw={"weight": "cauchy"},
            workers=2,
            verbose=False,
        )
        self.assertEqual(coefficients.weights.shape, observations.shape)
        self.assertTrue(np.isnan(coefficients.weights[100, 0]))
        self.assertLess(coefficients.weights[400, 1], 0.1)
        np.testing.assert_array_equal(coefficients.weights, coefficients.robust.weights)


class PersistenceTests(unittest.TestCase):
    def test_scalar_robust_inference_round_trip_preserves_results(self) -> None:
        count = 1_300
        time = 60_300.0 + np.arange(count) / 24.0
        observations = harmonic(time, M2_CPH, 1.0, 0.2)
        observations += harmonic(time, S2_CPH, 0.2, 0.45)
        observations += 0.01 * np.sin(np.arange(count) * 0.27)
        observations[100] = np.nan
        observations[500] += 8.0
        coefficients = rutide.solve(
            time,
            observations,
            lat=60.0,
            constit=["M2", "S2"],
            conf_int="MC",
            MC_n=48,
            MC_seed=11,
            method="robust",
            trend=False,
            phase="raw",
            nodal=False,
            infer={
                "inferred_names": ["S2"],
                "reference_names": ["M2"],
                "amp_ratios": [0.2],
                "phase_offsets": [15.0],
            },
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "scalar.rutide.npz"
            self.assertEqual(coefficients.save(path), path)
            restored = rutide.load(path)

        self.assertIsInstance(restored, rutide.Coefficient)
        for field in ["name", "A", "g", "A_ci", "g_ci", "PE", "SNR", "weights"]:
            np.testing.assert_array_equal(coefficients[field], restored[field])
        self.assertEqual(coefficients.robust.iterations, restored.robust.iterations)
        target = 60_295.0 + np.arange(1_500) / 36.0
        expected = rutide.reconstruct(target, coefficients, min_SNR=None, verbose=False)
        actual = rutide.reconstruct(target, restored, min_SNR=None, verbose=False)
        np.testing.assert_array_equal(expected.h, actual.h)

    def test_vector_batch_round_trip_and_worker_override(self) -> None:
        count = 1_100
        series_count = 3
        time = 60_400.0 + np.arange(count) / 24.0
        eastward = np.column_stack(
            [
                harmonic(time, M2_CPH, 0.8 + 0.05 * series, 0.2)
                + 0.02 * np.sin(np.arange(count) * 0.31)
                for series in range(series_count)
            ]
        )
        northward = np.column_stack(
            [
                harmonic(time, M2_CPH, 0.35, -0.4 + 0.1 * series)
                + 0.01 * np.cos(np.arange(count) * 0.23)
                for series in range(series_count)
            ]
        )
        eastward[40, 1] = np.nan
        northward[41, 1] = np.nan
        eastward[500, 2] += 12.0
        coefficients = rutide.solve_many(
            time,
            eastward,
            northward,
            lat=np.linspace(58.0, 62.0, series_count),
            constit=["M2"],
            conf_int="linear",
            method="robust",
            trend=False,
            phase="raw",
            nodal=False,
            workers=2,
            memory_limit_mb=0.02,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vector-batch.rutide.npz"
            rutide.save(coefficients, path, compressed=False)
            with np.load(path, allow_pickle=False) as archive:
                self.assertLessEqual(len(archive.files), 3)
                self.assertTrue(
                    all(
                        name == "__rutide_metadata__" or name.startswith("array_blob_")
                        for name in archive.files
                    )
                )
            restored = rutide.load(path, workers=1)

        self.assertIsInstance(restored, rutide.CoefficientBatch)
        self.assertEqual(restored.worker_count, 1)
        for field in [
            "name",
            "frequency_cph",
            "rank_index",
            "Lsmaj",
            "Lsmin",
            "theta",
            "g",
            "Lsmaj_ci",
            "Lsmin_ci",
            "theta_ci",
            "g_ci",
            "PE",
            "SNR",
            "weights",
            "nobs",
        ]:
            np.testing.assert_array_equal(coefficients[field], restored[field])
        target = 60_395.0 + np.arange(900) / 18.0
        expected = rutide.reconstruct_many(target, coefficients, min_SNR=None, verbose=False)
        actual = rutide.reconstruct_many(target, restored, min_SNR=None, verbose=False)
        np.testing.assert_array_equal(expected.u, actual.u)
        np.testing.assert_array_equal(expected.v, actual.v)

    def test_archive_validation_rejects_missing_and_unknown_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            missing = directory / "missing.npz"
            np.savez(missing, values=np.arange(3))
            with self.assertRaisesRegex(ValueError, "missing coefficient metadata"):
                rutide.load(missing)

            time = 60_000.0 + np.arange(800) / 24.0
            coefficients = rutide.solve(
                time,
                harmonic(time, M2_CPH, 1.0, 0.1),
                lat=60.0,
                constit=["M2"],
                conf_int="none",
                trend=False,
                verbose=False,
            )
            valid = directory / "valid.npz"
            coefficients.save(valid)
            with np.load(valid, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            metadata_name = "__rutide_metadata__"
            document = json.loads(arrays[metadata_name].tobytes().decode("utf-8"))
            document["schema_version"] = 999
            arrays[metadata_name] = np.frombuffer(json.dumps(document).encode(), dtype=np.uint8)
            invalid = directory / "future.npz"
            np.savez(invalid, **arrays)
            with self.assertRaisesRegex(ValueError, "unsupported coefficient archive schema"):
                rutide.load(invalid)

            with np.load(valid, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            document = json.loads(arrays[metadata_name].tobytes().decode("utf-8"))
            document["snapshot"]["time_mjd"]["__rutide_array__"]["offset"] = 10**12
            arrays[metadata_name] = np.frombuffer(json.dumps(document).encode(), dtype=np.uint8)
            invalid = directory / "invalid-bounds.npz"
            np.savez(invalid, **arrays)
            with self.assertRaisesRegex(ValueError, "exceeds its blob"):
                rutide.load(invalid)

    def test_single_fit_rejects_worker_override(self) -> None:
        time = 60_000.0 + np.arange(800) / 24.0
        coefficients = rutide.solve(
            time,
            harmonic(time, M2_CPH, 1.0, 0.1),
            lat=60.0,
            constit=["M2"],
            conf_int="none",
            trend=False,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single.npz"
            coefficients.save(path)
            with self.assertRaisesRegex(ValueError, "coefficient batch"):
                rutide.load(path, workers=2)


if __name__ == "__main__":
    unittest.main()
