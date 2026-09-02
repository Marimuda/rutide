"""Unit tests for the real-data installed-package acceptance harness."""

from __future__ import annotations

import unittest

import numpy as np

from rutide_baseline.real_data_acceptance import (
    array_digest,
    evenly_spaced_indices,
    read_time_major_columns,
)


class RealDataAcceptanceTests(unittest.TestCase):
    def test_selection_spans_axis_without_duplicates(self) -> None:
        np.testing.assert_array_equal(evenly_spaced_indices(10, 1), [0])
        np.testing.assert_array_equal(evenly_spaced_indices(10, 4), [0, 3, 6, 9])
        np.testing.assert_array_equal(evenly_spaced_indices(4, 4), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            evenly_spaced_indices(4, 5)

    def test_digest_includes_dtype_shape_and_values(self) -> None:
        values = np.array([1.0, np.nan])
        self.assertEqual(array_digest(values), array_digest(values.copy()))
        self.assertNotEqual(array_digest(values), array_digest(values.reshape(1, 2)))
        self.assertNotEqual(array_digest(values), array_digest(values.astype(np.float32)))

    def test_bounded_reader_preserves_sparse_columns(self) -> None:
        source = np.arange(7 * 11, dtype=np.float32).reshape(7, 11)
        indices = np.array([0, 4, 10])
        actual = read_time_major_columns(source, indices, working_memory_mb=0.00005)
        np.testing.assert_array_equal(actual, source[:, indices])
        self.assertEqual(actual.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
