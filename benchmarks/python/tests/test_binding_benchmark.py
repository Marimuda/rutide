"""Unit tests for the matched public-binding benchmark fixture."""

from __future__ import annotations

import unittest

import numpy as np

from rutide_baseline.binding_benchmark import make_fixture, result_digest, solve_options


class BindingBenchmarkTests(unittest.TestCase):
    def test_fixture_is_time_major_distinct_and_deterministic(self) -> None:
        first = make_fixture("vector", "irregular", 128, 6)
        second = make_fixture("vector", "irregular", 128, 6)
        self.assertEqual(first.eastward.shape, (128, 6))
        self.assertEqual(first.northward.shape, (128, 6))
        self.assertEqual(first.latitudes.shape, (6,))
        self.assertTrue(np.all(np.diff(first.time) > 0))
        self.assertGreater(np.count_nonzero(~np.isfinite(first.eastward)), 0)
        self.assertGreater(np.count_nonzero(~np.isfinite(first.northward)), 0)
        self.assertFalse(np.array_equal(first.eastward[:, 0], first.eastward[:, 1]))
        np.testing.assert_array_equal(first.time, second.time)
        np.testing.assert_array_equal(first.eastward, second.eastward)
        np.testing.assert_array_equal(first.northward, second.northward)

    def test_profiles_change_only_the_intended_analysis_axes(self) -> None:
        ols = solve_options("ols")
        linear = solve_options("linear-colored")
        robust = solve_options("robust-colored")
        self.assertEqual(ols["conf_int"], "none")
        self.assertEqual(linear["conf_int"], "linear")
        self.assertEqual(linear["method"], "ols")
        self.assertEqual(robust["conf_int"], "linear")
        self.assertEqual(robust["method"], "robust")
        self.assertFalse(robust["white"])

    def test_digest_includes_shape_and_data(self) -> None:
        values = np.arange(6, dtype=np.float64)
        self.assertNotEqual(result_digest((values,)), result_digest((values.reshape(2, 3),)))
        self.assertEqual(result_digest((values,)), result_digest((values.copy(),)))


if __name__ == "__main__":
    unittest.main()
