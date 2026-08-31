"""Unit tests for cross-language comparison utilities."""

import unittest

import numpy as np

from rutide_baseline.compare import _maximum_error, circular_phase_error


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


if __name__ == "__main__":
    unittest.main()
