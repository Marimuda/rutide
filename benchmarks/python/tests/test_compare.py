"""Unit tests for cross-language comparison utilities."""

import unittest

import numpy as np

from rutide_baseline.compare import (
    BASE_PHASE_TOLERANCE_DEGREES,
    _maximum_error,
    circular_phase_error,
    phase_tolerance_degrees,
)


class ComparisonUtilitiesTests(unittest.TestCase):
    def test_circular_phase_error_wraps_at_360_degrees(self) -> None:
        actual = np.array([359.0, 1.0, 180.0])
        expected = np.array([1.0, 359.0, 0.0])
        np.testing.assert_allclose(circular_phase_error(actual, expected), [2.0, 2.0, 180.0])

    def test_maximum_error_retains_location_and_tolerance_result(self) -> None:
        result = _maximum_error([(1e-14, 2, "M2"), (2e-14, 7, "S2")], 3e-14)
        self.assertEqual(result["maximum_absolute_error"], 2e-14)
        self.assertEqual(result["worst_node_index"], 7)
        self.assertEqual(result["worst_constituent"], "S2")
        self.assertTrue(result["within_tolerance"])

    def test_phase_tolerance_only_relaxes_for_near_zero_amplitude(self) -> None:
        self.assertEqual(phase_tolerance_degrees(1.0), BASE_PHASE_TOLERANCE_DEGREES)
        self.assertGreater(phase_tolerance_degrees(1e-4), BASE_PHASE_TOLERANCE_DEGREES)
        self.assertEqual(phase_tolerance_degrees(0.0), 180.0)


if __name__ == "__main__":
    unittest.main()
