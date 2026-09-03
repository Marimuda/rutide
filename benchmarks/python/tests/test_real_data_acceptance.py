"""Unit tests for the real-data installed-package acceptance harness."""

from __future__ import annotations

import unittest

import numpy as np

from rutide_baseline.real_data_acceptance import (
    array_digest,
    evenly_spaced_indices,
    finite_range,
    read_time_major_columns,
    sampling_summary,
    shared_velocity_units,
)


class Variable:
    def __init__(self, units: object = None) -> None:
        self.units = units


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

    def test_vector_units_must_be_present_and_equal(self) -> None:
        self.assertEqual(shared_velocity_units(Variable("m s-1"), Variable("m s-1")), "m s-1")
        with self.assertRaisesRegex(ValueError, "must both declare"):
            shared_velocity_units(Variable("m s-1"), Variable())
        with self.assertRaisesRegex(ValueError, "disagree"):
            shared_velocity_units(Variable("m s-1"), Variable("cm s-1"))

    def test_sampling_summary_checks_chronology_and_reports_gaps(self) -> None:
        summary = sampling_summary(np.array([60_000.0, 60_000.5, 60_001.0, 60_002.0]))
        self.assertEqual(summary["record_span_days"], 2.0)
        self.assertEqual(summary["median_interval_hours"], 12.0)
        self.assertEqual(summary["largest_gap_hours"], 24.0)
        self.assertEqual(summary["non_median_intervals"], 1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            sampling_summary(np.array([2.0, 1.0]))

    def test_finite_range_rejects_an_empty_diagnostic(self) -> None:
        self.assertEqual(
            finite_range(np.array([1.0, np.nan, 3.0])),
            {"finite": 2, "total": 3, "minimum": 1.0, "median": 2.0, "maximum": 3.0},
        )
        with self.assertRaisesRegex(RuntimeError, "no finite"):
            finite_range(np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()
