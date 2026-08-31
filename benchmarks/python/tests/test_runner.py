"""Unit tests for benchmark-result canonicalization."""

import unittest

from rutide_baseline.runner import aggregate_result_digest


class ResultDigestTests(unittest.TestCase):
    def test_aggregate_digest_is_spatially_ordered(self) -> None:
        first = "00" * 32
        second = "ff" * 32
        forward = [(3, first, None), (8, second, None)]
        reverse = list(reversed(forward))
        self.assertEqual(
            aggregate_result_digest(forward),
            aggregate_result_digest(reverse),
        )

    def test_aggregate_digest_changes_with_spatial_index(self) -> None:
        series = "11" * 32
        self.assertNotEqual(
            aggregate_result_digest([(1, series, None)]),
            aggregate_result_digest([(2, series, None)]),
        )


if __name__ == "__main__":
    unittest.main()
