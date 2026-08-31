"""Unit tests for deterministic fixture utilities."""

import unittest

import numpy as np

from rutide_baseline.fixture import array_digest, deterministic_indices, reconstruct_mjd


class FixtureUtilitiesTests(unittest.TestCase):
    def test_reconstruct_mjd_uses_integer_milliseconds(self) -> None:
        actual = reconstruct_mjd(
            np.array([58_113, 58_113]),
            np.array([0, 3_600_000]),
        )
        np.testing.assert_allclose(actual, [58_113.0, 58_113.0 + 1.0 / 24.0])

    def test_deterministic_indices_are_unique_sorted_and_bounded(self) -> None:
        actual = deterministic_indices(1_003, 32, [1, 17, 999])
        self.assertEqual(actual, sorted(actual))
        self.assertEqual(len(actual), 32)
        self.assertEqual(len(set(actual)), 32)
        self.assertGreaterEqual(min(actual), 0)
        self.assertLess(max(actual), 1_003)
        self.assertEqual(actual, deterministic_indices(1_003, 32, [1, 17, 999]))

    def test_deterministic_indices_reject_too_many_anchors(self) -> None:
        with self.assertRaises(ValueError):
            deterministic_indices(10, 2, [1, 2, 3])

    def test_array_digest_includes_mask(self) -> None:
        values = np.ma.array([1.0, 2.0], mask=[False, False])
        masked = np.ma.array([1.0, 2.0], mask=[False, True])
        self.assertNotEqual(array_digest(values), array_digest(masked))


if __name__ == "__main__":
    unittest.main()
