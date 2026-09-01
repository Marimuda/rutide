"""Tests for the matched inferred-constituent throughput fixtures."""

from __future__ import annotations

import unittest

import numpy as np

from rutide_baseline.inference_benchmark import (
    _inference,
    _scalar_observations,
    _times,
    _vector_observations,
)


class InferenceBenchmarkTests(unittest.TestCase):
    def test_scalar_and_vector_relationship_order_matches_python_convention(self) -> None:
        scalar = _inference("scalar", "exact")
        self.assertEqual(scalar["inferred_names"], ["S2", "O1"])
        self.assertEqual(scalar["reference_names"], ["M2", "K1"])
        self.assertEqual(scalar["amp_ratios"], [0.35, 0.5])
        self.assertEqual(scalar["phase_offsets"], [20.0, 45.0])
        self.assertFalse(scalar["approximate"])

        vector = _inference("vector", "approximate")
        self.assertEqual(vector["amp_ratios"], [0.35, 0.5, 0.25, 0.4])
        self.assertEqual(vector["phase_offsets"], [20.0, 45.0, -10.0, 30.0])
        self.assertTrue(vector["approximate"])

    def test_sampling_profiles_have_expected_lengths_and_missing_masks(self) -> None:
        regular_times = _times("regular")
        irregular_times = _times("irregular")
        self.assertEqual(len(regular_times), 745)
        np.testing.assert_allclose(np.diff(regular_times), 1 / 24)
        self.assertGreater(np.max(np.abs(irregular_times - regular_times)), 0.001)
        self.assertTrue(np.all(np.diff(irregular_times) > 0))

        self.assertEqual(np.count_nonzero(~np.isfinite(_scalar_observations("regular"))), 0)
        self.assertEqual(np.count_nonzero(~np.isfinite(_scalar_observations("irregular"))), 3)
        eastward, northward = _vector_observations(irregular_times, "irregular")
        self.assertEqual(np.count_nonzero(~(np.isfinite(eastward) & np.isfinite(northward))), 4)


if __name__ == "__main__":
    unittest.main()
