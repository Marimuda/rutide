//! Fixed-frequency scalar harmonic analysis.

use std::{
    cmp::Ordering,
    collections::HashMap,
    f64::consts::{PI, TAU},
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use faer::{
    Mat,
    linalg::solvers::{ColPivQr, DenseSolveCore, SolveLstsq},
};
use rustfft::{Fft, FftPlanner, num_complex::Complex};

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

/// Residual-noise model for a linearized 95% confidence interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearConfidence {
    /// One variance estimate shared by all frequencies.
    White,
    /// `UTide`'s band-averaged residual spectrum for equidistant timestamps.
    Colored,
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
    /// Linearized 95% amplitude confidence-interval half-widths, when requested.
    pub amplitude_ci: Option<Vec<f64>>,
    /// Linearized 95% phase confidence-interval half-widths in degrees.
    pub phase_ci_degrees: Option<Vec<f64>>,
    /// Signal-to-noise ratio derived from amplitude and amplitude CI.
    pub signal_to_noise: Option<Vec<f64>>,
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

    /// Return constituent indices ranked by descending signal-to-noise ratio.
    ///
    /// Returns `None` when the solution was produced without confidence
    /// intervals, because SNR is not an independent amplitude-only diagnostic.
    #[must_use]
    pub fn constituent_indices_by_signal_to_noise(&self) -> Option<Vec<usize>> {
        self.signal_to_noise.as_ref().map(|signal_to_noise| {
            let mut indices = (0..signal_to_noise.len()).collect::<Vec<_>>();
            indices.sort_by(|left, right| {
                signal_to_noise[*right]
                    .total_cmp(&signal_to_noise[*left])
                    .then_with(|| left.cmp(right))
            });
            indices
        })
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
    effective_record_length_days: f64,
    sample_interval_hours: Option<f64>,
    design: Mat<f64>,
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
            equidistant_sample_interval_hours(time_days),
            design,
        ))
    }

    pub(crate) fn from_design(
        constituents: Vec<Constituent>,
        time_count: usize,
        time_span_days: f64,
        sample_interval_hours: Option<f64>,
        design: Mat<f64>,
    ) -> Self {
        let decomposition = design.col_piv_qr();
        let time_count_f64 = usize_to_f64(time_count);
        Self {
            constituents,
            time_count,
            time_span_days,
            effective_record_length_days: time_span_days * time_count_f64 / (time_count_f64 - 1.0),
            sample_interval_hours,
            design,
            decomposition,
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

    /// Fit one series and calculate linearized 95% confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for an invalid observation series or when the
    /// colored-noise model is requested for non-equidistant timestamps.
    pub fn solve_with_linear_confidence(
        &self,
        observations: &[f64],
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        let mut solutions =
            self.solve_many_time_major_with_linear_confidence(observations, 1, noise)?;
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
        self.solve_many_time_major_impl(observations, series_count, None)
    }

    /// Fit multiple series and calculate linearized 95% confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid observation data or when colored
    /// noise is requested for non-equidistant timestamps.
    pub fn solve_many_time_major_with_linear_confidence(
        &self,
        observations: &[f64],
        series_count: usize,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_many_time_major_impl(observations, series_count, Some(noise))
    }

    fn solve_many_time_major_impl(
        &self,
        observations: &[f64],
        series_count: usize,
        confidence: Option<LinearConfidence>,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        if series_count == 0 {
            return Err(AnalysisError::EmptySeries);
        }
        if confidence == Some(LinearConfidence::Colored) && self.sample_interval_hours.is_none() {
            return Err(AnalysisError::UnevenTimeForColoredConfidence);
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
        let variance_weights = confidence.map(|_| self.linear_variance_weights());

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
                let (amplitude_ci, phase_ci_degrees, signal_to_noise) =
                    if let (Some(noise), Some(variance_weights)) =
                        (confidence, variance_weights.as_ref())
                    {
                        let intervals = self.linear_confidence_intervals(
                            observations,
                            series_count,
                            series,
                            coefficients.as_ref(),
                            variance_weights,
                            noise,
                        );
                        let signal_to_noise = amplitude
                            .iter()
                            .zip(&intervals.amplitude)
                            .map(|(amplitude, interval)| {
                                amplitude.powi(2) / (interval / 1.96).powi(2)
                            })
                            .collect();
                        (
                            Some(intervals.amplitude),
                            Some(intervals.phase_degrees),
                            Some(signal_to_noise),
                        )
                    } else {
                        (None, None, None)
                    };
                ScalarSolution {
                    amplitude,
                    phase_degrees,
                    percent_energy,
                    amplitude_ci,
                    phase_ci_degrees,
                    signal_to_noise,
                    mean: coefficients[(harmonic_columns, series)],
                    slope_per_day: coefficients[(harmonic_columns + 1, series)]
                        / self.time_span_days,
                }
            })
            .collect())
    }

    fn linear_variance_weights(&self) -> Vec<f64> {
        let column_count = self.design.ncols();
        let normal = Mat::from_fn(column_count, column_count, |row, column| {
            (0..self.time_count)
                .map(|time| self.design[(time, row)] * self.design[(time, column)])
                .sum::<f64>()
        });
        let covariance = normal.partial_piv_lu().inverse();
        (0..column_count)
            .map(|index| covariance[(index, index)])
            .collect()
    }

    fn linear_confidence_intervals(
        &self,
        observations: &[f64],
        series_count: usize,
        series: usize,
        coefficients: faer::MatRef<'_, f64>,
        variance_weights: &[f64],
        noise: LinearConfidence,
    ) -> LinearIntervals {
        let mut residual = Vec::with_capacity(self.time_count);
        let mut observation_energy = 0.0;
        let mut model_observation_product = 0.0;
        for time in 0..self.time_count {
            let fitted = (0..self.design.ncols())
                .map(|column| self.design[(time, column)] * coefficients[(column, series)])
                .sum::<f64>();
            let observation = observations[time * series_count + series];
            residual.push(observation - fitted);
            observation_energy += observation * observation;
            model_observation_product += fitted * observation;
        }

        let white_variance = (observation_energy - model_observation_product)
            / usize_to_f64(self.time_count - self.design.ncols());
        let colored_power = (noise == LinearConfidence::Colored).then(|| {
            band_averaged_residual_power(
                &residual,
                self.sample_interval_hours
                    .expect("colored confidence validates equidistant timestamps"),
                self.effective_record_length_days,
                &self.constituents,
            )
        });
        let mut amplitude = Vec::with_capacity(self.constituents.len());
        let mut phase_degrees = Vec::with_capacity(self.constituents.len());
        for constituent in 0..self.constituents.len() {
            let cosine = coefficients[(constituent * 2, series)];
            let sine = coefficients[(constituent * 2 + 1, series)];
            let cosine_weight = variance_weights[constituent * 2];
            let sine_weight = variance_weights[constituent * 2 + 1];
            let (cosine_variance, sine_variance) = match &colored_power {
                Some(power) => {
                    let denominator = cosine_weight + sine_weight;
                    (
                        power[constituent] * cosine_weight / denominator,
                        power[constituent] * sine_weight / denominator,
                    )
                }
                None => (white_variance * cosine_weight, white_variance * sine_weight),
            };
            let magnitude_squared = cosine * cosine + sine * sine;
            let amplitude_sigma = ((cosine * cosine * cosine_variance
                + sine * sine * sine_variance)
                / magnitude_squared)
                .sqrt();
            let phase_sigma_radians = ((sine * sine * cosine_variance
                + cosine * cosine * sine_variance)
                / magnitude_squared.powi(2))
            .sqrt();
            amplitude.push(1.96 * amplitude_sigma);
            phase_degrees.push(1.96 * phase_sigma_radians * 180.0 / PI);
        }
        LinearIntervals {
            amplitude,
            phase_degrees,
        }
    }
}

struct LinearIntervals {
    amplitude: Vec<f64>,
    phase_degrees: Vec<f64>,
}

const FREQUENCY_BANDS_CPH: [[f64; 2]; 9] = [
    [0.000_10, 0.004_17],
    [0.031_92, 0.048_59],
    [0.072_18, 0.088_84],
    [0.112_43, 0.129_10],
    [0.152_69, 0.169_36],
    [0.192_95, 0.209_61],
    [0.233_20, 0.251_00],
    [0.260_00, 0.290_00],
    [0.300_00, 0.500_00],
];

fn band_averaged_residual_power(
    residual: &[f64],
    sample_interval_hours: f64,
    effective_record_length_days: f64,
    constituents: &[Constituent],
) -> Vec<f64> {
    let sample_count = residual.len() - residual.len() % 2;
    let mean = residual[..sample_count].iter().sum::<f64>() / usize_to_f64(sample_count);
    let mut window_energy = 0.0;
    let mut spectrum = residual[..sample_count]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            let window = 0.5 - 0.5 * (TAU * usize_to_f64(index) / usize_to_f64(sample_count)).cos();
            window_energy += window * window;
            Complex::new((value - mean) * window, 0.0)
        })
        .collect::<Vec<_>>();
    cached_forward_fft(sample_count).process(&mut spectrum);

    let frequency_count = sample_count / 2 + 1;
    let frequency_spacing = 1.0 / (usize_to_f64(sample_count) * sample_interval_hours);
    let normalization = sample_interval_hours / window_energy;
    let mut power = spectrum[..frequency_count]
        .iter()
        .map(|value| value.norm_sqr() * normalization)
        .collect::<Vec<_>>();
    for value in &mut power[1..frequency_count - 1] {
        *value *= 2.0;
    }

    let mut excluded = vec![false; frequency_count];
    for constituent in constituents {
        let nearest = (constituent.frequency_cph / frequency_spacing)
            .round_ties_even()
            .clamp(0.0, usize_to_f64(frequency_count - 1));
        excluded[bounded_frequency_index(nearest)] = true;
    }

    let band_power = FREQUENCY_BANDS_CPH.map(|[lower, upper]| {
        let start = (0..frequency_count)
            .find(|index| usize_to_f64(*index) * frequency_spacing >= lower)
            .unwrap_or(frequency_count);
        let upper_insertion = (0..frequency_count)
            .find(|index| usize_to_f64(*index) * frequency_spacing >= upper)
            .unwrap_or(frequency_count);
        let stop = (upper_insertion + 1).min(frequency_count);
        let mut sum = 0.0;
        let mut count = 0_usize;
        for index in start..stop {
            if !excluded[index] {
                sum += power[index];
                count += 1;
            }
        }
        sum / usize_to_f64(count)
    });
    let density_to_power = 1.0 / (effective_record_length_days * 24.0);
    constituents
        .iter()
        .map(|constituent| {
            FREQUENCY_BANDS_CPH
                .iter()
                .zip(band_power)
                .find(|([lower, upper], _)| {
                    constituent.frequency_cph >= *lower && constituent.frequency_cph <= *upper
                })
                .map_or(0.0, |(_, power)| power * density_to_power)
        })
        .collect()
}

type FftPlan = Arc<dyn Fft<f64>>;
type FftPlanCache = Mutex<HashMap<usize, FftPlan>>;

fn cached_forward_fft(length: usize) -> FftPlan {
    static PLANS: OnceLock<FftPlanCache> = OnceLock::new();
    let plans = PLANS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut plans = plans.lock().unwrap_or_else(PoisonError::into_inner);
    plans.entry(length).or_insert_with(|| {
        let mut planner = FftPlanner::new();
        planner.plan_fft_forward(length)
    });
    Arc::clone(&plans[&length])
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rounded FFT index is finite, integral, and clamped to the output range"
)]
fn bounded_frequency_index(value: f64) -> usize {
    value as usize
}

pub(crate) fn equidistant_sample_interval_hours(time_days: &[f64]) -> Option<f64> {
    let mut unique_deltas = time_days
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    unique_deltas.sort_by(f64::total_cmp);
    unique_deltas.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let mean = unique_deltas.iter().sum::<f64>() / usize_to_f64(unique_deltas.len());
    let variance = unique_deltas
        .iter()
        .map(|delta| (delta - mean).powi(2))
        .sum::<f64>()
        / usize_to_f64(unique_deltas.len());
    (variance < f64::EPSILON).then(|| 24.0 * (time_days[1] - time_days[0]))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "allocated slice and matrix lengths are exactly representable in practical f64 analyses"
)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
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
    use super::{Constituent, FixedRawOls, LinearConfidence};
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
        assert!(solution.amplitude_ci.is_none());
        assert!(solution.phase_ci_degrees.is_none());
        assert!(solution.signal_to_noise.is_none());
        assert!(solution.constituent_indices_by_signal_to_noise().is_none());
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
    fn supports_white_confidence_but_rejects_colored_noise_for_irregular_time() {
        let constituents = constituents();
        let mut time = times();
        time[20] += 0.001;
        let observations = signal(
            &time,
            &constituents,
            &[0.7, 0.2, 0.11],
            &[34.0, 229.0, 321.0],
            0.09,
            0.0017,
        );
        let model = FixedRawOls::prepare(&time, &constituents).expect("valid irregular model");
        assert!(
            model
                .solve_with_linear_confidence(&observations, LinearConfidence::White)
                .is_ok()
        );
        assert_eq!(
            model.solve_with_linear_confidence(&observations, LinearConfidence::Colored),
            Err(AnalysisError::UnevenTimeForColoredConfidence)
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
