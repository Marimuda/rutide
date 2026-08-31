//! Fixed-frequency scalar harmonic analysis.

use std::{cmp::Ordering, f64::consts::TAU};

use faer::{
    Mat,
    linalg::solvers::{ColPivQr, SolveLstsq},
};

use crate::AnalysisError;

/// A named tidal constituent with a fixed frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Constituent {
    /// Conventional constituent name, such as `M2`.
    pub name: String,
    /// Frequency in cycles per hour.
    pub frequency_cph: f64,
}

impl Constituent {
    /// Construct a named constituent measured in cycles per hour.
    #[must_use]
    pub fn new(name: impl Into<String>, frequency_cph: f64) -> Self {
        Self {
            name: name.into(),
            frequency_cph,
        }
    }
}

/// Scalar coefficients returned by a fixed raw-phase OLS fit.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarSolution {
    /// Amplitude for each prepared constituent, in input order.
    pub amplitude: Vec<f64>,
    /// Raw phase in degrees in the half-open range `[0, 360)`.
    pub phase_degrees: Vec<f64>,
    /// Percentage of total resolved harmonic energy for each constituent.
    pub percent_energy: Vec<f64>,
    /// Fitted constant offset.
    pub mean: f64,
    /// Fitted linear trend per day.
    pub slope_per_day: f64,
}

impl ScalarSolution {
    /// Return constituent indices ranked by descending percent energy.
    ///
    /// Coefficient arrays remain in prepared constituent order. This index view
    /// provides Python `UTide`-style PE presentation without losing stable
    /// constituent identity in multi-series outputs.
    #[must_use]
    pub fn constituent_indices_by_percent_energy(&self) -> Vec<usize> {
        let mut indices = (0..self.amplitude.len()).collect::<Vec<_>>();
        indices.sort_by(|left, right| {
            self.amplitude[*right]
                .total_cmp(&self.amplitude[*left])
                .then_with(|| left.cmp(right))
        });
        indices
    }
}

/// A reusable fixed-constituent, raw-phase ordinary least-squares model.
///
/// Preparation constructs and factorizes one real harmonic design matrix. The
/// factorization is then reused for every spatial series sharing the timestamps.
/// The model always includes a mean and linear trend to match the initial Python
/// `UTide` parity profile.
#[derive(Debug)]
pub struct FixedRawOls {
    constituents: Vec<Constituent>,
    time_count: usize,
    time_span_days: f64,
    decomposition: ColPivQr<f64>,
}

impl FixedRawOls {
    /// Validate timestamps and constituents, build the basis, and factorize it.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for empty, non-finite, non-increasing, duplicate,
    /// or underdetermined inputs.
    pub fn prepare(time_days: &[f64], constituents: &[Constituent]) -> Result<Self, AnalysisError> {
        validate_constituents(constituents)?;
        let (reference_time_days, time_span_days) = validate_time(time_days, constituents.len())?;
        let harmonic_columns = constituents.len() * 2;
        let column_count = harmonic_columns + 2;

        let design = Mat::from_fn(time_days.len(), column_count, |row, column| {
            match column.cmp(&harmonic_columns) {
                Ordering::Less => {
                    let constituent = &constituents[column / 2];
                    let angle = TAU
                        * 24.0
                        * (time_days[row] - reference_time_days)
                        * constituent.frequency_cph;
                    if column % 2 == 0 {
                        angle.cos()
                    } else {
                        angle.sin()
                    }
                }
                Ordering::Equal => 1.0,
                Ordering::Greater => (time_days[row] - reference_time_days) / time_span_days,
            }
        });

        Ok(Self::from_design(
            constituents.to_vec(),
            time_days.len(),
            time_span_days,
            &design,
        ))
    }

    pub(crate) fn from_design(
        constituents: Vec<Constituent>,
        time_count: usize,
        time_span_days: f64,
        design: &Mat<f64>,
    ) -> Self {
        Self {
            constituents,
            time_count,
            time_span_days,
            decomposition: design.col_piv_qr(),
        }
    }

    /// Return the prepared constituents in coefficient order.
    #[must_use]
    pub fn constituents(&self) -> &[Constituent] {
        &self.constituents
    }

    /// Return the number of observations expected in each spatial series.
    #[must_use]
    pub const fn time_count(&self) -> usize {
        self.time_count
    }

    /// Fit one complete, finite scalar observation series.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] if the series length differs from the prepared
    /// timestamps or contains a non-finite value.
    pub fn solve(&self, observations: &[f64]) -> Result<ScalarSolution, AnalysisError> {
        let mut solutions = self.solve_many_time_major(observations, 1)?;
        solutions.pop().ok_or(AnalysisError::EmptySeries)
    }

    /// Fit several complete scalar series stored in time-major order.
    ///
    /// `observations[time * series_count + series]` is the value for one time and
    /// spatial series. This matches an in-memory `zeta(time, node)` array and lets
    /// the least-squares factorization process many right-hand sides together.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] if the flattened shape is inconsistent, no
    /// series are supplied, or any observation is non-finite.
    pub fn solve_many_time_major(
        &self,
        observations: &[f64],
        series_count: usize,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        if series_count == 0 {
            return Err(AnalysisError::EmptySeries);
        }
        let expected = self.time_count.saturating_mul(series_count);
        if observations.len() != expected {
            return Err(AnalysisError::ObservationShape {
                actual: observations.len(),
                expected,
            });
        }
        for (index, value) in observations.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(AnalysisError::NonFiniteObservation {
                    series: index % series_count,
                    time: index / series_count,
                });
            }
        }

        let right_hand_sides = Mat::from_fn(self.time_count, series_count, |time, series| {
            observations[time * series_count + series]
        });
        let coefficients = self.decomposition.solve_lstsq(right_hand_sides.as_ref());
        let harmonic_columns = self.constituents.len() * 2;

        Ok((0..series_count)
            .map(|series| {
                let mut amplitude = Vec::with_capacity(self.constituents.len());
                let mut phase_degrees = Vec::with_capacity(self.constituents.len());
                for constituent in 0..self.constituents.len() {
                    let cosine = coefficients[(constituent * 2, series)];
                    let sine = coefficients[(constituent * 2 + 1, series)];
                    amplitude.push(cosine.hypot(sine));
                    phase_degrees.push(sine.atan2(cosine).to_degrees().rem_euclid(360.0));
                }
                let total_energy = amplitude.iter().map(|value| value * value).sum::<f64>();
                let percent_energy = amplitude
                    .iter()
                    .map(|value| 100.0 * (value * value) / total_energy)
                    .collect();
                ScalarSolution {
                    amplitude,
                    phase_degrees,
                    percent_energy,
                    mean: coefficients[(harmonic_columns, series)],
                    slope_per_day: coefficients[(harmonic_columns + 1, series)]
                        / self.time_span_days,
                }
            })
            .collect())
    }
}

fn validate_constituents(constituents: &[Constituent]) -> Result<(), AnalysisError> {
    if constituents.is_empty() {
        return Err(AnalysisError::EmptyConstituents);
    }
    for (index, constituent) in constituents.iter().enumerate() {
        if constituent.name.trim().is_empty() {
            return Err(AnalysisError::EmptyConstituentName { index });
        }
        if !constituent.frequency_cph.is_finite() || constituent.frequency_cph <= 0.0 {
            return Err(AnalysisError::InvalidFrequency { index });
        }
        if constituents[..index]
            .iter()
            .any(|earlier| earlier.frequency_cph.to_bits() == constituent.frequency_cph.to_bits())
        {
            return Err(AnalysisError::DuplicateFrequency { index });
        }
    }
    Ok(())
}

pub(crate) fn validate_time(
    time_days: &[f64],
    constituent_count: usize,
) -> Result<(f64, f64), AnalysisError> {
    if time_days.is_empty() {
        return Err(AnalysisError::EmptyTime);
    }
    for (index, value) in time_days.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(AnalysisError::NonFiniteTime { index });
        }
        if index > 0 && value <= time_days[index - 1] {
            return Err(AnalysisError::NonIncreasingTime { index });
        }
    }
    let required = constituent_count * 2 + 3;
    if time_days.len() < required {
        return Err(AnalysisError::InsufficientObservations {
            actual: time_days.len(),
            required,
        });
    }

    let first = time_days[0];
    let last = time_days[time_days.len() - 1];
    Ok((first.midpoint(last), last - first))
}

#[cfg(test)]
mod tests {
    use super::{Constituent, FixedRawOls};
    use crate::AnalysisError;
    use std::f64::consts::TAU;

    fn constituents() -> Vec<Constituent> {
        vec![
            Constituent::new("M2", 0.080_511_400_671_577_2),
            Constituent::new("S2", 1.0 / 12.0),
            Constituent::new("K1", 0.041_780_746_221_637_22),
        ]
    }

    fn times() -> Vec<f64> {
        (0..745)
            .map(|index| 58_113.0 + f64::from(index) / 24.0)
            .collect()
    }

    fn signal(
        times: &[f64],
        constituents: &[Constituent],
        amplitudes: &[f64],
        phases: &[f64],
        mean: f64,
        slope: f64,
    ) -> Vec<f64> {
        let reference = times[0].midpoint(times[times.len() - 1]);
        times
            .iter()
            .map(|time| {
                let harmonics = constituents
                    .iter()
                    .zip(amplitudes)
                    .zip(phases)
                    .map(|((constituent, amplitude), phase)| {
                        let angle = TAU * 24.0 * (time - reference) * constituent.frequency_cph;
                        amplitude * (angle - phase.to_radians()).cos()
                    })
                    .sum::<f64>();
                mean + slope * (time - reference) + harmonics
            })
            .collect()
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn recovers_exact_synthetic_coefficients() {
        let constituents = constituents();
        let times = times();
        let expected_amplitude = [0.7, 0.2, 0.11];
        let expected_phase = [34.0, 229.0, 321.0];
        let observations = signal(
            &times,
            &constituents,
            &expected_amplitude,
            &expected_phase,
            0.09,
            0.0017,
        );
        let model = FixedRawOls::prepare(&times, &constituents).expect("valid model");
        let solution = model.solve(&observations).expect("valid observations");

        for (actual, expected) in solution.amplitude.iter().zip(expected_amplitude) {
            assert_close(*actual, expected, 1e-12);
        }
        for (actual, expected) in solution.phase_degrees.iter().zip(expected_phase) {
            assert_close(*actual, expected, 1e-10);
        }
        assert_close(solution.percent_energy.iter().sum(), 100.0, 1e-12);
        assert_eq!(solution.constituent_indices_by_percent_energy(), [0, 1, 2]);
        assert_close(solution.mean, 0.09, 1e-12);
        assert_close(solution.slope_per_day, 0.0017, 1e-12);
    }

    #[test]
    fn solves_multiple_time_major_series() {
        let constituents = constituents();
        let times = times();
        let first = signal(
            &times,
            &constituents,
            &[0.7, 0.2, 0.11],
            &[34.0, 229.0, 321.0],
            0.09,
            0.0017,
        );
        let second = signal(
            &times,
            &constituents,
            &[0.3, 0.4, 0.05],
            &[11.0, 92.0, 270.0],
            -0.2,
            -0.003,
        );
        let mut time_major = Vec::with_capacity(times.len() * 2);
        for time in 0..times.len() {
            time_major.push(first[time]);
            time_major.push(second[time]);
        }

        let model = FixedRawOls::prepare(&times, &constituents).expect("valid model");
        let solutions = model
            .solve_many_time_major(&time_major, 2)
            .expect("valid observations");
        assert_close(solutions[0].amplitude[0], 0.7, 1e-12);
        assert_close(solutions[1].amplitude[0], 0.3, 1e-12);
        assert_close(solutions[1].phase_degrees[1], 92.0, 1e-10);
        assert_close(solutions[1].mean, -0.2, 1e-12);
        assert_close(solutions[1].slope_per_day, -0.003, 1e-12);
    }

    #[test]
    fn rejects_non_finite_observation_with_coordinates() {
        let constituents = constituents();
        let times = times();
        let model = FixedRawOls::prepare(&times, &constituents).expect("valid model");
        let mut observations = vec![0.0; times.len() * 2];
        observations[17] = f64::NAN;
        assert_eq!(
            model.solve_many_time_major(&observations, 2),
            Err(AnalysisError::NonFiniteObservation { series: 1, time: 8 })
        );
    }

    #[test]
    fn rejects_duplicate_frequency() {
        let times = times();
        let duplicate = vec![
            Constituent::new("first", 0.1),
            Constituent::new("second", 0.1),
        ];
        assert!(matches!(
            FixedRawOls::prepare(&times, &duplicate),
            Err(AnalysisError::DuplicateFrequency { index: 1 })
        ));
    }
}
