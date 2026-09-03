//! Fixed-frequency scalar harmonic analysis.

use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::HashMap,
    f64::consts::{PI, TAU},
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use faer::{
    Mat, c64,
    linalg::solvers::{ColPivQr, DenseSolveCore, SolveLstsq},
};
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::{
    AnalysisError, MonteCarloOptions, RobustDiagnostics, RobustOptions, VectorSolution,
    monte_carlo::{scalar_intervals as scalar_monte_carlo_intervals, vector_intervals},
    robust::{RobustFit, fit_with_initial as robust_fit_with_initial},
    sampling::{
        COLORED_NOISE_FREQUENCY_BANDS_CPH as FREQUENCY_BANDS_CPH,
        equidistant_sample_interval_hours, lomb_frequencies,
    },
    vector::from_component_solutions,
};

// Applying a precomputed least-squares projection turns sufficiently wide
// batches into one cache-friendly matrix product. Below this threshold the
// projection's one-time construction costs more than applying the retained QR.
const MIN_PROJECTED_BATCH_SERIES: usize = 16;

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
    /// `UTide`'s band-averaged FFT or Lomb–Scargle residual spectrum.
    Colored,
}

/// Options controlling the non-harmonic terms in a fitted model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FitOptions {
    /// Fit a linear trend in addition to the always-present mean.
    pub trend: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self { trend: true }
    }
}

#[derive(Clone, Copy)]
enum ConfidenceSpec<'noise> {
    None,
    Shared(LinearConfidence),
    BySeries(&'noise [LinearConfidence]),
}

impl ConfidenceSpec<'_> {
    fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    fn for_series(self, series: usize) -> Option<LinearConfidence> {
        match self {
            Self::None => None,
            Self::Shared(noise) => Some(noise),
            Self::BySeries(noise) => noise.get(series).copied(),
        }
    }
}

pub(crate) type ScalarCoefficientCovariance = [[f64; 2]; 2];
pub(crate) type ScalarConfidenceSolution = (ScalarSolution, Vec<ScalarCoefficientCovariance>);

/// Scalar coefficients returned by a fixed raw-phase OLS fit.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarSolution {
    /// Cosine coefficient for each prepared constituent.
    pub cosine_coefficient: Vec<f64>,
    /// Sine coefficient for each prepared constituent.
    pub sine_coefficient: Vec<f64>,
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
    /// Estimated cosine-coefficient variance used by linear confidence intervals.
    pub cosine_coefficient_variance: Option<Vec<f64>>,
    /// Estimated sine-coefficient variance used by linear confidence intervals.
    pub sine_coefficient_variance: Option<Vec<f64>>,
    /// Fitted constant offset.
    pub mean: f64,
    /// Fitted linear trend per day.
    pub slope_per_day: f64,
    /// Epoch at which `mean` is defined, in the same day coordinate as the fit.
    pub reference_time_days: f64,
    /// Iteration, weight, leverage, scale, and residual diagnostics for robust fits.
    pub robust: Option<RobustDiagnostics>,
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
/// The model always includes a mean. A linear trend is enabled by default to
/// preserve the initial Python `UTide` parity profile.
#[derive(Debug)]
pub struct FixedRawOls {
    constituents: Vec<Constituent>,
    confidence_constituents: Vec<Constituent>,
    python_inference_non_reference_count: Option<usize>,
    time_count: usize,
    time_span_days: f64,
    effective_record_length_days: f64,
    sample_interval_hours: Option<f64>,
    spectrum_time_count: usize,
    spectrum_observation_positions: Option<Vec<usize>>,
    irregular_spectrum: Option<IrregularSpectrumSampling>,
    reference_time_days: f64,
    fit_options: FitOptions,
    design: Mat<f64>,
    decomposition: ColPivQr<f64>,
    batch_projection: OnceLock<Mat<f64>>,
}

impl FixedRawOls {
    /// Validate timestamps and constituents, build the basis, and factorize it.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for empty, non-finite, non-increasing, duplicate,
    /// or underdetermined inputs.
    pub fn prepare(time_days: &[f64], constituents: &[Constituent]) -> Result<Self, AnalysisError> {
        Self::prepare_with_options(time_days, constituents, FitOptions::default())
    }

    /// Validate inputs and prepare a model with explicit non-harmonic terms.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for empty, non-finite, non-increasing, duplicate,
    /// or underdetermined inputs.
    pub fn prepare_with_options(
        time_days: &[f64],
        constituents: &[Constituent],
        fit_options: FitOptions,
    ) -> Result<Self, AnalysisError> {
        validate_constituents(constituents)?;
        let (reference_time_days, time_span_days) =
            validate_time_with_options(time_days, constituents.len(), fit_options)?;
        let harmonic_columns = constituents.len() * 2;
        let column_count = harmonic_columns + 1 + usize::from(fit_options.trend);

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
            reference_time_days,
            ConfidenceSampling::complete(time_days, time_span_days),
            fit_options,
            design,
        ))
    }

    pub(crate) fn from_design(
        constituents: Vec<Constituent>,
        time_count: usize,
        time_span_days: f64,
        reference_time_days: f64,
        confidence_sampling: ConfidenceSampling,
        fit_options: FitOptions,
        design: Mat<f64>,
    ) -> Self {
        let confidence_constituents = constituents.clone();
        Self::from_design_with_confidence_constituents(
            constituents,
            confidence_constituents,
            None,
            time_count,
            time_span_days,
            reference_time_days,
            confidence_sampling,
            fit_options,
            design,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "prepared inference models carry distinct fit/report constituents and sampling metadata"
    )]
    pub(crate) fn from_design_with_confidence_constituents(
        constituents: Vec<Constituent>,
        confidence_constituents: Vec<Constituent>,
        python_inference_non_reference_count: Option<usize>,
        time_count: usize,
        time_span_days: f64,
        reference_time_days: f64,
        confidence_sampling: ConfidenceSampling,
        fit_options: FitOptions,
        design: Mat<f64>,
    ) -> Self {
        let decomposition = design.col_piv_qr();
        Self {
            constituents,
            confidence_constituents,
            python_inference_non_reference_count,
            time_count,
            time_span_days,
            effective_record_length_days: confidence_sampling.effective_record_length_days,
            sample_interval_hours: confidence_sampling.sample_interval_hours,
            spectrum_time_count: confidence_sampling.spectrum_time_count,
            spectrum_observation_positions: confidence_sampling.observation_positions,
            irregular_spectrum: confidence_sampling.irregular_spectrum,
            reference_time_days,
            fit_options,
            design,
            decomposition,
            batch_projection: OnceLock::new(),
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

    /// Return the configured non-harmonic fit terms.
    #[must_use]
    pub const fn fit_options(&self) -> FitOptions {
        self.fit_options
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
    /// Returns [`AnalysisError`] for an invalid observation series.
    pub fn solve_with_linear_confidence(
        &self,
        observations: &[f64],
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        let mut solutions =
            self.solve_many_time_major_with_linear_confidence(observations, 1, noise)?;
        solutions.pop().ok_or(AnalysisError::EmptySeries)
    }

    /// Fit one series with reproducible nonlinear Monte Carlo confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid observations, options, or covariance.
    pub fn solve_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        let mut solutions = self.solve_many_time_major_with_monte_carlo_confidence(
            observations,
            1,
            options,
            noise,
        )?;
        solutions.pop().ok_or(AnalysisError::EmptySeries)
    }

    /// Fit one series with configured iteratively reweighted least squares.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid observations or robust options,
    /// degenerate residual scale, invalid leverage, or non-convergence.
    pub fn solve_robust(
        &self,
        observations: &[f64],
        options: RobustOptions,
    ) -> Result<ScalarSolution, AnalysisError> {
        let observations = self.observation_matrix(observations, 1)?;
        let fit = self.robust_fit(&observations, options)?;
        Ok(self.robust_component_solution(&fit, observations.as_ref(), 0, None))
    }

    /// Robustly fit one series with linearized confidence intervals and SNR.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input or robust fitting failure.
    pub fn solve_robust_with_linear_confidence(
        &self,
        observations: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        let observations = self.observation_matrix(observations, 1)?;
        let fit = self.robust_fit(&observations, options)?;
        Ok(self.robust_component_solution(&fit, observations.as_ref(), 0, Some(noise)))
    }

    /// Robustly fit one series with nonlinear Monte Carlo confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid input, robust fitting failure,
    /// invalid Monte Carlo options, or an unsampleable covariance.
    pub fn solve_robust_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarSolution, AnalysisError> {
        self.solve_robust_with_monte_carlo_confidence_from_stream(
            observations,
            robust_options,
            monte_carlo_options,
            noise,
            0,
        )
    }

    pub(crate) fn solve_robust_with_monte_carlo_confidence_from_stream(
        &self,
        observations: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
        stream: u64,
    ) -> Result<ScalarSolution, AnalysisError> {
        monte_carlo_options.validate()?;
        let (mut solution, covariances) = self.solve_robust_with_scalar_coefficient_covariances(
            observations,
            robust_options,
            noise,
        )?;
        self.apply_scalar_monte_carlo_intervals(
            &mut solution,
            &covariances,
            monte_carlo_options,
            stream,
        )?;
        Ok(solution)
    }

    pub(crate) fn solve_with_scalar_coefficient_covariances(
        &self,
        observations: &[f64],
        noise: LinearConfidence,
    ) -> Result<ScalarConfidenceSolution, AnalysisError> {
        let observation_matrix = self.observation_matrix(observations, 1)?;
        let coefficients = self.decomposition.solve_lstsq(observation_matrix.as_ref());
        let solution = self.component_solution(coefficients.as_ref(), 0, None);
        let normal_inverse = self.coefficient_normal_inverse(None);
        let covariances = self.scalar_coefficient_covariances(
            observations,
            1,
            0,
            coefficients.as_ref(),
            &normal_inverse,
            noise,
            None,
        );
        Ok((solution, covariances))
    }

    pub(crate) fn solve_robust_with_scalar_coefficient_covariances(
        &self,
        observations: &[f64],
        robust_options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<ScalarConfidenceSolution, AnalysisError> {
        let observation_matrix = self.observation_matrix(observations, 1)?;
        let fit = self.robust_fit(&observation_matrix, robust_options)?;
        let solution =
            self.component_solution(fit.coefficients.as_ref(), 0, Some(fit.diagnostics.clone()));
        let normal_inverse = self.coefficient_normal_inverse(Some(&fit.diagnostics.weights));
        let covariances = self.scalar_coefficient_covariances(
            observations,
            1,
            0,
            fit.coefficients.as_ref(),
            &normal_inverse,
            noise,
            Some(&fit.diagnostics.weights),
        );
        Ok((solution, covariances))
    }

    pub(crate) fn solve_two_component_robust(
        &self,
        observations: &[f64],
        options: RobustOptions,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        let observations = self.observation_matrix(observations, 2)?;
        let fit = self.robust_fit(&observations, options)?;
        Ok((0..2)
            .map(|component| {
                self.robust_component_solution(&fit, observations.as_ref(), component, None)
            })
            .collect())
    }

    pub(crate) fn solve_two_component_robust_with_linear_confidence(
        &self,
        observations: &[f64],
        options: RobustOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        let observations = self.observation_matrix(observations, 2)?;
        let fit = self.robust_fit(&observations, options)?;
        Ok([LinearConfidence::White, noise]
            .into_iter()
            .enumerate()
            .map(|(component, noise)| {
                self.robust_component_solution(&fit, observations.as_ref(), component, Some(noise))
            })
            .collect())
    }

    pub(crate) fn solve_vector_robust_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        robust_options: RobustOptions,
        monte_carlo_options: MonteCarloOptions,
        noise: LinearConfidence,
        stream: u64,
    ) -> Result<VectorSolution, AnalysisError> {
        monte_carlo_options.validate()?;
        let observations = self.observation_matrix(observations, 2)?;
        let fit = self.robust_fit(&observations, robust_options)?;
        let eastward =
            self.component_solution(fit.coefficients.as_ref(), 0, Some(fit.diagnostics.clone()));
        let northward =
            self.component_solution(fit.coefficients.as_ref(), 1, Some(fit.diagnostics.clone()));
        let mut solution = from_component_solutions(&eastward, &northward)?;
        let normal_inverse = self.coefficient_normal_inverse(Some(&fit.diagnostics.weights));
        let covariances = self.vector_coefficient_covariances(
            observations.as_ref(),
            fit.coefficients.as_ref(),
            &normal_inverse,
            noise,
            Some(&fit.diagnostics.weights),
        );
        self.apply_vector_monte_carlo_intervals(
            &mut solution,
            &covariances,
            monte_carlo_options,
            stream,
        )?;
        Ok(solution)
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
        self.solve_many_time_major_impl(observations, series_count, ConfidenceSpec::None)
    }

    /// Fit multiple series and calculate linearized 95% confidence intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid observation data.
    pub fn solve_many_time_major_with_linear_confidence(
        &self,
        observations: &[f64],
        series_count: usize,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_many_time_major_impl(observations, series_count, ConfidenceSpec::Shared(noise))
    }

    /// Fit multiple series with reproducible nonlinear Monte Carlo intervals.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] for invalid shapes, values, options, or covariance.
    pub fn solve_many_time_major_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        series_count: usize,
        options: MonteCarloOptions,
        noise: LinearConfidence,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_many_time_major_with_monte_carlo_confidence_from_stream(
            observations,
            series_count,
            options,
            noise,
            0,
        )
    }

    pub(crate) fn solve_many_time_major_with_monte_carlo_confidence_from_stream(
        &self,
        observations: &[f64],
        series_count: usize,
        options: MonteCarloOptions,
        noise: LinearConfidence,
        first_stream: u64,
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        options.validate()?;
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
        let coefficients = self.solve_batch_coefficients(right_hand_sides.as_ref(), series_count);
        let normal_inverse = self.coefficient_normal_inverse(None);
        (0..series_count)
            .map(|series| {
                let mut solution = self.component_solution(coefficients.as_ref(), series, None);
                let covariances = self.scalar_coefficient_covariances(
                    observations,
                    series_count,
                    series,
                    coefficients.as_ref(),
                    &normal_inverse,
                    noise,
                    None,
                );
                let stream = first_stream.wrapping_add(
                    u64::try_from(series).expect("series index is representable as u64"),
                );
                self.apply_scalar_monte_carlo_intervals(
                    &mut solution,
                    &covariances,
                    options,
                    stream,
                )?;
                Ok(solution)
            })
            .collect()
    }

    pub(crate) fn solve_vector_with_monte_carlo_confidence(
        &self,
        observations: &[f64],
        options: MonteCarloOptions,
        noise: LinearConfidence,
        stream: u64,
    ) -> Result<VectorSolution, AnalysisError> {
        options.validate()?;
        let observations = self.observation_matrix(observations, 2)?;
        let coefficients = self.decomposition.solve_lstsq(observations.as_ref());
        let eastward = self.component_solution(coefficients.as_ref(), 0, None);
        let northward = self.component_solution(coefficients.as_ref(), 1, None);
        let mut solution = from_component_solutions(&eastward, &northward)?;
        let normal_inverse = self.coefficient_normal_inverse(None);
        let covariances = self.vector_coefficient_covariances(
            observations.as_ref(),
            coefficients.as_ref(),
            &normal_inverse,
            noise,
            None,
        );
        self.apply_vector_monte_carlo_intervals(&mut solution, &covariances, options, stream)?;
        Ok(solution)
    }

    pub(crate) fn solve_many_time_major_with_linear_confidence_by_series(
        &self,
        observations: &[f64],
        noise_by_series: &[LinearConfidence],
    ) -> Result<Vec<ScalarSolution>, AnalysisError> {
        self.solve_many_time_major_impl(
            observations,
            noise_by_series.len(),
            ConfidenceSpec::BySeries(noise_by_series),
        )
    }

    fn solve_many_time_major_impl(
        &self,
        observations: &[f64],
        series_count: usize,
        confidence: ConfidenceSpec<'_>,
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
        let coefficients = self.solve_batch_coefficients(right_hand_sides.as_ref(), series_count);
        let variance_weights = confidence
            .is_enabled()
            .then(|| self.linear_variance_weights(None));

        Ok((0..series_count)
            .map(|series| {
                let mut solution = self.component_solution(coefficients.as_ref(), series, None);
                if let (Some(noise), Some(variance_weights)) =
                    (confidence.for_series(series), variance_weights.as_ref())
                {
                    let intervals = self.linear_confidence_intervals(
                        observations,
                        series_count,
                        series,
                        coefficients.as_ref(),
                        variance_weights,
                        noise,
                        None,
                    );
                    solution.signal_to_noise = Some(
                        solution
                            .amplitude
                            .iter()
                            .zip(&intervals.amplitude)
                            .map(|(amplitude, interval)| {
                                amplitude.powi(2) / (interval / 1.96).powi(2)
                            })
                            .collect(),
                    );
                    solution.amplitude_ci = Some(intervals.amplitude);
                    solution.phase_ci_degrees = Some(intervals.phase_degrees);
                    solution.cosine_coefficient_variance = Some(intervals.cosine_variance);
                    solution.sine_coefficient_variance = Some(intervals.sine_variance);
                }
                solution
            })
            .collect())
    }

    fn component_solution(
        &self,
        coefficients: faer::MatRef<'_, f64>,
        component: usize,
        robust: Option<RobustDiagnostics>,
    ) -> ScalarSolution {
        let mut cosine_coefficient = Vec::with_capacity(self.constituents.len());
        let mut sine_coefficient = Vec::with_capacity(self.constituents.len());
        let mut amplitude = Vec::with_capacity(self.constituents.len());
        let mut phase_degrees = Vec::with_capacity(self.constituents.len());
        for constituent in 0..self.constituents.len() {
            let cosine = coefficients[(constituent * 2, component)];
            let sine = coefficients[(constituent * 2 + 1, component)];
            cosine_coefficient.push(cosine);
            sine_coefficient.push(sine);
            amplitude.push(cosine.hypot(sine));
            phase_degrees.push(sine.atan2(cosine).to_degrees().rem_euclid(360.0));
        }
        let total_energy = amplitude.iter().map(|value| value * value).sum::<f64>();
        let percent_energy = amplitude
            .iter()
            .map(|value| 100.0 * value * value / total_energy)
            .collect();
        let harmonic_columns = self.constituents.len() * 2;
        ScalarSolution {
            cosine_coefficient,
            sine_coefficient,
            amplitude,
            phase_degrees,
            percent_energy,
            amplitude_ci: None,
            phase_ci_degrees: None,
            signal_to_noise: None,
            cosine_coefficient_variance: None,
            sine_coefficient_variance: None,
            mean: coefficients[(harmonic_columns, component)],
            slope_per_day: if self.fit_options.trend {
                coefficients[(harmonic_columns + 1, component)] / self.time_span_days
            } else {
                0.0
            },
            reference_time_days: self.reference_time_days,
            robust,
        }
    }

    fn solve_batch_coefficients(
        &self,
        right_hand_sides: faer::MatRef<'_, f64>,
        series_count: usize,
    ) -> Mat<f64> {
        if series_count < MIN_PROJECTED_BATCH_SERIES {
            return self.decomposition.solve_lstsq(right_hand_sides);
        }
        let projection = self.batch_projection.get_or_init(|| {
            let identity = Mat::identity(self.time_count, self.time_count);
            self.decomposition.solve_lstsq(identity.as_ref())
        });
        projection.as_ref() * right_hand_sides
    }

    fn observation_matrix(
        &self,
        observations: &[f64],
        component_count: usize,
    ) -> Result<Mat<f64>, AnalysisError> {
        let expected = self.time_count.saturating_mul(component_count);
        if observations.len() != expected {
            return Err(AnalysisError::ObservationShape {
                actual: observations.len(),
                expected,
            });
        }
        for (index, value) in observations.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(AnalysisError::NonFiniteObservation {
                    series: index % component_count,
                    time: index / component_count,
                });
            }
        }
        Ok(Mat::from_fn(
            self.time_count,
            component_count,
            |time, component| observations[time * component_count + component],
        ))
    }

    fn robust_fit(
        &self,
        observations: &Mat<f64>,
        options: RobustOptions,
    ) -> Result<RobustFit, AnalysisError> {
        let initial_coefficients = self.decomposition.solve_lstsq(observations.as_ref());
        robust_fit_with_initial(&self.design, observations, initial_coefficients, options)
    }

    fn robust_component_solution(
        &self,
        fit: &RobustFit,
        observations: faer::MatRef<'_, f64>,
        component: usize,
        confidence: Option<LinearConfidence>,
    ) -> ScalarSolution {
        let mut cosine_coefficient = Vec::with_capacity(self.constituents.len());
        let mut sine_coefficient = Vec::with_capacity(self.constituents.len());
        let mut amplitude = Vec::with_capacity(self.constituents.len());
        let mut phase_degrees = Vec::with_capacity(self.constituents.len());
        for constituent in 0..self.constituents.len() {
            let cosine = fit.coefficients[(constituent * 2, component)];
            let sine = fit.coefficients[(constituent * 2 + 1, component)];
            cosine_coefficient.push(cosine);
            sine_coefficient.push(sine);
            amplitude.push(cosine.hypot(sine));
            phase_degrees.push(sine.atan2(cosine).to_degrees().rem_euclid(360.0));
        }
        let total_energy = amplitude.iter().map(|value| value * value).sum::<f64>();
        let percent_energy = amplitude
            .iter()
            .map(|value| 100.0 * value * value / total_energy)
            .collect();
        let harmonic_columns = self.constituents.len() * 2;
        let (
            amplitude_ci,
            phase_ci_degrees,
            signal_to_noise,
            cosine_coefficient_variance,
            sine_coefficient_variance,
        ) = if let Some(noise) = confidence {
            let variance_weights = self.linear_variance_weights(Some(&fit.diagnostics.weights));
            let component_observations = (0..self.time_count)
                .map(|time| observations[(time, component)])
                .collect::<Vec<_>>();
            let intervals = self.linear_confidence_intervals(
                &component_observations,
                1,
                0,
                fit.coefficients.as_ref().subcols(component, 1),
                &variance_weights,
                noise,
                Some(&fit.diagnostics.weights),
            );
            let signal_to_noise = amplitude
                .iter()
                .zip(&intervals.amplitude)
                .map(|(amplitude, interval)| amplitude.powi(2) / (interval / 1.96).powi(2))
                .collect();
            (
                Some(intervals.amplitude),
                Some(intervals.phase_degrees),
                Some(signal_to_noise),
                Some(intervals.cosine_variance),
                Some(intervals.sine_variance),
            )
        } else {
            (None, None, None, None, None)
        };
        ScalarSolution {
            cosine_coefficient,
            sine_coefficient,
            amplitude,
            phase_degrees,
            percent_energy,
            amplitude_ci,
            phase_ci_degrees,
            signal_to_noise,
            cosine_coefficient_variance,
            sine_coefficient_variance,
            mean: fit.coefficients[(harmonic_columns, component)],
            slope_per_day: if self.fit_options.trend {
                fit.coefficients[(harmonic_columns + 1, component)] / self.time_span_days
            } else {
                0.0
            },
            reference_time_days: self.reference_time_days,
            robust: Some(fit.diagnostics.clone()),
        }
    }

    fn coefficient_normal_inverse(&self, weights: Option<&[f64]>) -> Mat<f64> {
        let column_count = self.design.ncols();
        let normal = Mat::from_fn(column_count, column_count, |row, column| {
            (0..self.time_count)
                .map(|time| {
                    weights.map_or(1.0, |weights| weights[time])
                        * self.design[(time, row)]
                        * self.design[(time, column)]
                })
                .sum::<f64>()
        });
        normal.partial_piv_lu().inverse()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "observation layout, fitted coefficients, covariance, noise, and robust weights are explicit"
    )]
    fn scalar_coefficient_covariances(
        &self,
        observations: &[f64],
        series_count: usize,
        series: usize,
        coefficients: faer::MatRef<'_, f64>,
        normal_inverse: &Mat<f64>,
        noise: LinearConfidence,
        weights: Option<&[f64]>,
    ) -> Vec<[[f64; 2]; 2]> {
        let mut residual = Vec::with_capacity(self.time_count);
        let mut misfit = 0.0;
        for time in 0..self.time_count {
            let fitted = (0..self.design.ncols())
                .map(|column| self.design[(time, column)] * coefficients[(column, series)])
                .sum::<f64>();
            let observation = observations[time * series_count + series];
            let weight = weights.map_or(1.0, |weights| weights[time]);
            residual.push(weight * (observation - fitted));
            misfit += weight * observation * (observation - fitted);
        }
        let variance = misfit / usize_to_f64(self.time_count - self.design.ncols());
        let colored_power =
            (noise == LinearConfidence::Colored).then(|| self.colored_residual_power(&residual));
        (0..self.constituents.len())
            .map(|constituent| {
                let cosine = constituent * 2;
                let sine = cosine + 1;
                let mut covariance = [
                    [
                        variance * normal_inverse[(cosine, cosine)],
                        variance * normal_inverse[(cosine, sine)],
                    ],
                    [
                        variance * normal_inverse[(sine, cosine)],
                        variance * normal_inverse[(sine, sine)],
                    ],
                ];
                if let Some(power) = &colored_power {
                    let scale = power[constituent] / (covariance[0][0] + covariance[1][1]);
                    for row in &mut covariance {
                        for value in row {
                            *value *= scale;
                        }
                    }
                }
                covariance
            })
            .collect()
    }

    fn vector_coefficient_covariances(
        &self,
        observations: faer::MatRef<'_, f64>,
        coefficients: faer::MatRef<'_, f64>,
        normal_inverse: &Mat<f64>,
        noise: LinearConfidence,
        weights: Option<&[f64]>,
    ) -> Vec<[[f64; 4]; 4]> {
        let mut eastward_residual = Vec::with_capacity(self.time_count);
        let mut northward_residual = Vec::with_capacity(self.time_count);
        let mut eastward_misfit = 0.0;
        let mut northward_misfit = 0.0;
        let mut cross_misfit = 0.0;
        for time in 0..self.time_count {
            let fitted = |component| {
                (0..self.design.ncols())
                    .map(|column| self.design[(time, column)] * coefficients[(column, component)])
                    .sum::<f64>()
            };
            let fitted_eastward = fitted(0);
            let fitted_northward = fitted(1);
            let eastward = observations[(time, 0)];
            let northward = observations[(time, 1)];
            let weight = weights.map_or(1.0, |weights| weights[time]);
            let eastward_error = eastward - fitted_eastward;
            let northward_error = northward - fitted_northward;
            eastward_residual.push(weight * eastward_error);
            northward_residual.push(weight * northward_error);
            eastward_misfit += weight * eastward * eastward_error;
            northward_misfit += weight * northward * northward_error;
            cross_misfit +=
                weight * (eastward * northward_error + northward * eastward_error) / 2.0;
        }
        let degrees_of_freedom = usize_to_f64(self.time_count - self.design.ncols());
        let residual_covariance = [
            eastward_misfit / degrees_of_freedom,
            northward_misfit / degrees_of_freedom,
            cross_misfit / degrees_of_freedom,
        ];
        let colored_power = (noise == LinearConfidence::Colored)
            .then(|| self.vector_colored_residual_power(&eastward_residual, &northward_residual));
        (0..self.constituents.len())
            .map(|constituent| {
                let indices = [constituent * 2, constituent * 2 + 1];
                let mut eastward = [[0.0; 2]; 2];
                let mut northward = [[0.0; 2]; 2];
                let mut cross = [[0.0; 2]; 2];
                for row in 0..2 {
                    for column in 0..2 {
                        let design_covariance = normal_inverse[(indices[row], indices[column])];
                        eastward[row][column] = residual_covariance[0] * design_covariance;
                        northward[row][column] = residual_covariance[1] * design_covariance;
                        cross[row][column] = residual_covariance[2] * design_covariance;
                    }
                }
                if let Some(power) = &colored_power {
                    let eastward_trace = matrix_trace(&eastward);
                    let northward_trace = matrix_trace(&northward);
                    let cross_absolute_sum = matrix_absolute_sum(&cross);
                    scale_matrix(&mut eastward, power.eastward[constituent], eastward_trace);
                    scale_matrix(
                        &mut northward,
                        power.northward[constituent],
                        northward_trace,
                    );
                    scale_matrix(&mut cross, power.cross[constituent], cross_absolute_sum);
                }
                [
                    [eastward[0][0], eastward[0][1], cross[0][0], cross[0][1]],
                    [eastward[1][0], eastward[1][1], cross[1][0], cross[1][1]],
                    [cross[0][0], cross[1][0], northward[0][0], northward[0][1]],
                    [cross[0][1], cross[1][1], northward[1][0], northward[1][1]],
                ]
            })
            .collect()
    }

    fn apply_scalar_monte_carlo_intervals(
        &self,
        solution: &mut ScalarSolution,
        covariances: &[[[f64; 2]; 2]],
        options: MonteCarloOptions,
        stream: u64,
    ) -> Result<(), AnalysisError> {
        let mut amplitude = Vec::with_capacity(self.constituents.len());
        let mut phase = Vec::with_capacity(self.constituents.len());
        let mut cosine_variance = Vec::with_capacity(self.constituents.len());
        let mut sine_variance = Vec::with_capacity(self.constituents.len());
        for (constituent, covariance) in covariances.iter().copied().enumerate() {
            let constituent_stream = constituent_stream(stream, constituent);
            let intervals = scalar_monte_carlo_intervals(
                [
                    solution.cosine_coefficient[constituent],
                    solution.sine_coefficient[constituent],
                ],
                covariance,
                options,
                constituent_stream,
            )
            .ok_or(AnalysisError::InvalidConfidenceCovariance { constituent })?;
            amplitude.push(intervals.amplitude);
            phase.push(intervals.phase_degrees);
            cosine_variance.push(covariance[0][0]);
            sine_variance.push(covariance[1][1]);
        }
        solution.signal_to_noise = Some(
            solution
                .amplitude
                .iter()
                .zip(&amplitude)
                .map(|(value, interval)| value.powi(2) / (interval / 1.96).powi(2))
                .collect(),
        );
        solution.amplitude_ci = Some(amplitude);
        solution.phase_ci_degrees = Some(phase);
        solution.cosine_coefficient_variance = Some(cosine_variance);
        solution.sine_coefficient_variance = Some(sine_variance);
        Ok(())
    }

    fn apply_vector_monte_carlo_intervals(
        &self,
        solution: &mut VectorSolution,
        covariances: &[[[f64; 4]; 4]],
        options: MonteCarloOptions,
        stream: u64,
    ) -> Result<(), AnalysisError> {
        let mut semi_major = Vec::with_capacity(self.constituents.len());
        let mut semi_minor = Vec::with_capacity(self.constituents.len());
        let mut inclination = Vec::with_capacity(self.constituents.len());
        let mut phase = Vec::with_capacity(self.constituents.len());
        let (eastward, northward) = solution.component_solutions();
        for (constituent, covariance) in covariances.iter().copied().enumerate() {
            let intervals = vector_intervals(
                [
                    eastward.cosine_coefficient[constituent],
                    eastward.sine_coefficient[constituent],
                    northward.cosine_coefficient[constituent],
                    northward.sine_coefficient[constituent],
                ],
                covariance,
                options,
                constituent_stream(stream, constituent),
            )
            .ok_or(AnalysisError::InvalidConfidenceCovariance { constituent })?;
            semi_major.push(intervals.semi_major);
            semi_minor.push(intervals.semi_minor);
            inclination.push(intervals.inclination_degrees);
            phase.push(intervals.phase_degrees);
        }
        solution.signal_to_noise = Some(
            solution
                .semi_major
                .iter()
                .zip(&solution.semi_minor)
                .zip(&semi_major)
                .zip(&semi_minor)
                .map(|(((major, minor), major_ci), minor_ci)| {
                    (major.powi(2) + minor.powi(2))
                        / ((major_ci / 1.96).powi(2) + (minor_ci / 1.96).powi(2))
                })
                .collect(),
        );
        solution.semi_major_ci = Some(semi_major);
        solution.semi_minor_ci = Some(semi_minor);
        solution.inclination_ci_degrees = Some(inclination);
        solution.phase_ci_degrees = Some(phase);
        Ok(())
    }

    fn colored_residual_power(&self, residual: &[f64]) -> Vec<f64> {
        if let Some(sample_interval_hours) = self.sample_interval_hours {
            let residual = self.residual_on_regular_spectrum_grid(residual);
            band_averaged_fft_residual_power(
                &residual,
                sample_interval_hours,
                self.effective_record_length_days,
                &self.confidence_constituents,
            )
        } else {
            self.irregular_spectrum
                .as_ref()
                .expect("irregular confidence sampling retains timestamps")
                .band_averaged_residual_power(
                    residual,
                    self.effective_record_length_days,
                    &self.confidence_constituents,
                )
        }
    }

    fn vector_colored_residual_power(
        &self,
        eastward: &[f64],
        northward: &[f64],
    ) -> VectorResidualPower {
        if let Some(sample_interval_hours) = self.sample_interval_hours {
            let eastward = self.residual_on_regular_spectrum_grid(eastward);
            let northward = self.residual_on_regular_spectrum_grid(northward);
            band_averaged_fft_vector_residual_power(
                &eastward,
                &northward,
                sample_interval_hours,
                self.effective_record_length_days,
                &self.confidence_constituents,
            )
        } else {
            self.irregular_spectrum
                .as_ref()
                .expect("irregular confidence sampling retains timestamps")
                .band_averaged_vector_residual_power(
                    eastward,
                    northward,
                    self.effective_record_length_days,
                    &self.confidence_constituents,
                )
        }
    }

    fn linear_variance_weights(&self, weights: Option<&[f64]>) -> Vec<f64> {
        if let Some(non_reference_count) = self.python_inference_non_reference_count {
            return self.python_complex_variance_weights(weights, non_reference_count);
        }
        let covariance = self.coefficient_normal_inverse(weights);
        (0..self.design.ncols())
            .map(|index| covariance[(index, index)])
            .collect()
    }

    fn python_complex_variance_weights(
        &self,
        weights: Option<&[f64]>,
        non_reference_count: usize,
    ) -> Vec<f64> {
        let constituent_count = self.constituents.len();
        let reference_count = constituent_count - non_reference_count;
        let column_count = self.design.ncols();
        let basis = Mat::from_fn(self.time_count, column_count, |time, column| {
            if column < non_reference_count {
                c64::new(
                    self.design[(time, column * 2)],
                    self.design[(time, column * 2 + 1)],
                )
            } else if column < non_reference_count * 2 {
                let constituent = column - non_reference_count;
                c64::new(
                    self.design[(time, constituent * 2)],
                    -self.design[(time, constituent * 2 + 1)],
                )
            } else if column < non_reference_count * 2 + reference_count {
                let constituent = non_reference_count + column - non_reference_count * 2;
                c64::new(
                    self.design[(time, constituent * 2)],
                    self.design[(time, constituent * 2 + 1)],
                )
            } else if column < constituent_count * 2 {
                let constituent =
                    non_reference_count + column - (non_reference_count * 2 + reference_count);
                c64::new(
                    self.design[(time, constituent * 2)],
                    -self.design[(time, constituent * 2 + 1)],
                )
            } else {
                c64::new(self.design[(time, column)], 0.0)
            }
        });
        let covariance_normal = Mat::from_fn(column_count, column_count, |row, column| {
            (0..self.time_count)
                .map(|time| {
                    basis[(time, row)].conj()
                        * weights.map_or(1.0, |weights| weights[time])
                        * basis[(time, column)]
                })
                .sum::<c64>()
        });
        let pseudo_normal = Mat::from_fn(column_count, column_count, |row, column| {
            (0..self.time_count)
                .map(|time| {
                    basis[(time, row)]
                        * weights.map_or(1.0, |weights| weights[time])
                        * basis[(time, column)]
                })
                .sum::<c64>()
        });
        let covariance = covariance_normal.partial_piv_lu().inverse();
        let pseudo_covariance = pseudo_normal.partial_piv_lu().inverse();
        let mut variance_weights = vec![0.0; self.design.ncols()];
        for constituent in 0..constituent_count {
            let positive = constituent;
            let negative = constituent + constituent_count;
            let gall_positive_positive =
                covariance[(positive, positive)] + pseudo_covariance[(positive, positive)];
            let gall_negative_negative =
                covariance[(negative, negative)] + pseudo_covariance[(negative, negative)];
            let gall_positive_negative =
                covariance[(positive, negative)] + pseudo_covariance[(positive, negative)];
            let hall_positive_positive =
                covariance[(positive, positive)] - pseudo_covariance[(positive, positive)];
            let hall_negative_negative =
                covariance[(negative, negative)] - pseudo_covariance[(negative, negative)];
            let hall_positive_negative =
                covariance[(positive, negative)] - pseudo_covariance[(positive, negative)];
            variance_weights[constituent * 2] =
                (gall_positive_positive + gall_negative_negative + gall_positive_negative * 2.0).re
                    / 2.0;
            variance_weights[constituent * 2 + 1] =
                (hall_positive_positive + hall_negative_negative - hall_positive_negative * 2.0).re
                    / 2.0;
        }
        variance_weights
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "keeps observation layout, coefficients, noise, and optional robust weights explicit"
    )]
    fn linear_confidence_intervals(
        &self,
        observations: &[f64],
        series_count: usize,
        series: usize,
        coefficients: faer::MatRef<'_, f64>,
        variance_weights: &[f64],
        noise: LinearConfidence,
        weights: Option<&[f64]>,
    ) -> LinearIntervals {
        let mut residual = Vec::with_capacity(self.time_count);
        let mut observation_energy = 0.0;
        let mut model_observation_product = 0.0;
        for time in 0..self.time_count {
            let fitted = (0..self.design.ncols())
                .map(|column| self.design[(time, column)] * coefficients[(column, series)])
                .sum::<f64>();
            let observation = observations[time * series_count + series];
            let weight = weights.map_or(1.0, |weights| weights[time]);
            residual.push(weight * (observation - fitted));
            observation_energy += weight * observation * observation;
            model_observation_product += weight * fitted * observation;
        }

        let white_variance = (observation_energy - model_observation_product)
            / usize_to_f64(self.time_count - self.design.ncols());
        let colored_power = (noise == LinearConfidence::Colored).then(|| {
            if let Some(sample_interval_hours) = self.sample_interval_hours {
                let residual = self.residual_on_regular_spectrum_grid(&residual);
                band_averaged_fft_residual_power(
                    &residual,
                    sample_interval_hours,
                    self.effective_record_length_days,
                    &self.confidence_constituents,
                )
            } else {
                self.irregular_spectrum
                    .as_ref()
                    .expect("irregular confidence sampling retains timestamps")
                    .band_averaged_residual_power(
                        &residual,
                        self.effective_record_length_days,
                        &self.confidence_constituents,
                    )
            }
        });
        let mut amplitude = Vec::with_capacity(self.constituents.len());
        let mut phase_degrees = Vec::with_capacity(self.constituents.len());
        let mut cosine_variance_values = Vec::with_capacity(self.constituents.len());
        let mut sine_variance_values = Vec::with_capacity(self.constituents.len());
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
            cosine_variance_values.push(cosine_variance);
            sine_variance_values.push(sine_variance);
        }
        LinearIntervals {
            amplitude,
            phase_degrees,
            cosine_variance: cosine_variance_values,
            sine_variance: sine_variance_values,
        }
    }

    fn residual_on_regular_spectrum_grid<'residual>(
        &self,
        residual: &'residual [f64],
    ) -> Cow<'residual, [f64]> {
        let Some(positions) = &self.spectrum_observation_positions else {
            return Cow::Borrowed(residual);
        };
        debug_assert_eq!(positions.len(), residual.len());
        let mut interpolated = vec![residual[0]; self.spectrum_time_count];
        let mut valid = 0;
        for (grid_index, value) in interpolated.iter_mut().enumerate() {
            while valid + 1 < positions.len() && positions[valid + 1] <= grid_index {
                valid += 1;
            }
            *value = if positions[valid] == grid_index || valid + 1 == positions.len() {
                residual[valid]
            } else if grid_index < positions[0] {
                residual[0]
            } else {
                let left_position = positions[valid];
                let right_position = positions[valid + 1];
                let fraction = usize_to_f64(grid_index - left_position)
                    / usize_to_f64(right_position - left_position);
                residual[valid] + fraction * (residual[valid + 1] - residual[valid])
            };
        }
        Cow::Owned(interpolated)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConfidenceSampling {
    pub(crate) sample_interval_hours: Option<f64>,
    pub(crate) effective_record_length_days: f64,
    pub(crate) spectrum_time_count: usize,
    pub(crate) observation_positions: Option<Vec<usize>>,
    irregular_spectrum: Option<IrregularSpectrumSampling>,
}

impl ConfidenceSampling {
    pub(crate) fn complete(time_days: &[f64], time_span_days: f64) -> Self {
        let time_count = time_days.len();
        let time_count_f64 = usize_to_f64(time_count);
        let sample_interval_hours = equidistant_sample_interval_hours(time_days);
        Self {
            sample_interval_hours,
            effective_record_length_days: time_span_days * time_count_f64 / (time_count_f64 - 1.0),
            spectrum_time_count: time_count,
            observation_positions: None,
            irregular_spectrum: sample_interval_hours
                .is_none()
                .then(|| IrregularSpectrumSampling::new(time_days, true)),
        }
    }

    pub(crate) fn regular_gappy(
        sample_interval_hours: f64,
        time_span_days: f64,
        spectrum_time_count: usize,
        observation_positions: Vec<usize>,
    ) -> Self {
        let time_count_f64 = usize_to_f64(spectrum_time_count);
        Self {
            sample_interval_hours: Some(sample_interval_hours),
            effective_record_length_days: time_span_days * time_count_f64 / (time_count_f64 - 1.0),
            spectrum_time_count,
            observation_positions: (observation_positions.len() != spectrum_time_count)
                .then_some(observation_positions),
            irregular_spectrum: None,
        }
    }

    pub(crate) fn irregular(time_days: &[f64], time_span_days: f64, share_plan: bool) -> Self {
        let time_count = time_days.len();
        let time_count_f64 = usize_to_f64(time_count);
        Self {
            sample_interval_hours: None,
            effective_record_length_days: time_span_days * time_count_f64 / (time_count_f64 - 1.0),
            spectrum_time_count: time_count,
            observation_positions: None,
            irregular_spectrum: Some(IrregularSpectrumSampling::new(time_days, share_plan)),
        }
    }

    pub(crate) fn precompute_shared_irregular_plan(&self) {
        if let Some(sampling) = &self.irregular_spectrum {
            sampling.precompute_shared_plan();
        }
    }

    pub(crate) fn band_averaged_residual_power(
        &self,
        residual: &[f64],
        constituents: &[Constituent],
    ) -> Vec<f64> {
        if let Some(sample_interval_hours) = self.sample_interval_hours {
            let interpolated = self.residual_on_spectrum_grid(residual);
            band_averaged_fft_residual_power(
                &interpolated,
                sample_interval_hours,
                self.effective_record_length_days,
                constituents,
            )
        } else {
            self.irregular_spectrum
                .as_ref()
                .expect("irregular confidence sampling retains timestamps")
                .band_averaged_residual_power(
                    residual,
                    self.effective_record_length_days,
                    constituents,
                )
        }
    }

    pub(crate) fn band_averaged_vector_residual_power(
        &self,
        eastward: &[f64],
        northward: &[f64],
        constituents: &[Constituent],
    ) -> VectorResidualPower {
        if let Some(sample_interval_hours) = self.sample_interval_hours {
            let eastward = self.residual_on_spectrum_grid(eastward);
            let northward = self.residual_on_spectrum_grid(northward);
            band_averaged_fft_vector_residual_power(
                &eastward,
                &northward,
                sample_interval_hours,
                self.effective_record_length_days,
                constituents,
            )
        } else {
            self.irregular_spectrum
                .as_ref()
                .expect("irregular confidence sampling retains timestamps")
                .band_averaged_vector_residual_power(
                    eastward,
                    northward,
                    self.effective_record_length_days,
                    constituents,
                )
        }
    }

    fn residual_on_spectrum_grid<'residual>(
        &self,
        residual: &'residual [f64],
    ) -> Cow<'residual, [f64]> {
        let Some(positions) = &self.observation_positions else {
            return Cow::Borrowed(residual);
        };
        debug_assert_eq!(positions.len(), residual.len());
        let mut interpolated = vec![residual[0]; self.spectrum_time_count];
        let mut valid = 0;
        for (grid_index, value) in interpolated.iter_mut().enumerate() {
            while valid + 1 < positions.len() && positions[valid + 1] <= grid_index {
                valid += 1;
            }
            *value = if positions[valid] == grid_index || valid + 1 == positions.len() {
                residual[valid]
            } else if grid_index < positions[0] {
                residual[0]
            } else {
                let left = positions[valid];
                let right = positions[valid + 1];
                let fraction = usize_to_f64(grid_index - left) / usize_to_f64(right - left);
                residual[valid] + fraction * (residual[valid + 1] - residual[valid])
            };
        }
        Cow::Owned(interpolated)
    }
}

#[derive(Clone, Debug)]
struct IrregularSpectrumSampling {
    time_hours: Arc<[f64]>,
    shared_plan: Option<Arc<OnceLock<LombScarglePlan>>>,
}

impl IrregularSpectrumSampling {
    fn new(time_days: &[f64], share_plan: bool) -> Self {
        let time_hours = time_days
            .iter()
            .map(|time| time * 24.0)
            .collect::<Arc<[f64]>>();
        let cache_fits_memory_budget =
            estimated_lomb_basis_bytes(&time_hours) <= MAX_CACHED_LOMB_BASIS_BYTES;
        Self {
            time_hours,
            shared_plan: (share_plan && cache_fits_memory_budget)
                .then(|| Arc::new(OnceLock::new())),
        }
    }

    fn band_averaged_residual_power(
        &self,
        residual: &[f64],
        effective_record_length_days: f64,
        constituents: &[Constituent],
    ) -> Vec<f64> {
        self.shared_plan.as_ref().map_or_else(
            || {
                band_averaged_lomb_residual_power(
                    residual,
                    &self.time_hours,
                    effective_record_length_days,
                    constituents,
                )
            },
            |shared_plan| {
                shared_plan
                    .get_or_init(|| LombScarglePlan::new(&self.time_hours))
                    .band_averaged_residual_power(
                        residual,
                        effective_record_length_days,
                        constituents,
                    )
            },
        )
    }

    fn band_averaged_vector_residual_power(
        &self,
        eastward: &[f64],
        northward: &[f64],
        effective_record_length_days: f64,
        constituents: &[Constituent],
    ) -> VectorResidualPower {
        self.shared_plan.as_ref().map_or_else(
            || {
                band_averaged_lomb_vector_residual_power(
                    eastward,
                    northward,
                    &self.time_hours,
                    effective_record_length_days,
                    constituents,
                )
            },
            |shared_plan| {
                shared_plan
                    .get_or_init(|| LombScarglePlan::new(&self.time_hours))
                    .band_averaged_vector_residual_power(
                        eastward,
                        northward,
                        effective_record_length_days,
                        constituents,
                    )
            },
        )
    }

    fn precompute_shared_plan(&self) {
        if let Some(shared_plan) = &self.shared_plan {
            shared_plan.get_or_init(|| LombScarglePlan::new_parallel(&self.time_hours));
        }
    }
}

struct LinearIntervals {
    amplitude: Vec<f64>,
    phase_degrees: Vec<f64>,
    cosine_variance: Vec<f64>,
    sine_variance: Vec<f64>,
}

pub(crate) struct VectorResidualPower {
    pub(crate) eastward: Vec<f64>,
    pub(crate) northward: Vec<f64>,
    pub(crate) cross: Vec<f64>,
}

fn matrix_trace(matrix: &[[f64; 2]; 2]) -> f64 {
    matrix[0][0] + matrix[1][1]
}

fn matrix_absolute_sum(matrix: &[[f64; 2]; 2]) -> f64 {
    matrix.iter().flatten().map(|value| value.abs()).sum()
}

fn scale_matrix(matrix: &mut [[f64; 2]; 2], numerator: f64, denominator: f64) {
    if denominator == 0.0 {
        for row in matrix {
            row.fill(0.0);
        }
        return;
    }
    let scale = numerator / denominator;
    for row in matrix {
        for value in row {
            *value *= scale;
        }
    }
}

pub(crate) fn constituent_stream(series_stream: u64, constituent: usize) -> u64 {
    series_stream
        .wrapping_mul(0xD134_2543_DE82_EF95)
        .wrapping_add(
            u64::try_from(constituent).expect("constituent index is representable as u64"),
        )
}

// Each plan stores both phase-shifted bases. Long records fall back to the
// direct kernel instead of silently allocating hundreds of megabytes per mask.
const MAX_CACHED_LOMB_BASIS_BYTES: usize = 16 * 1024 * 1024;

fn band_averaged_fft_residual_power(
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

    regular_band_power_by_constituent(
        frequency_spacing,
        &power,
        effective_record_length_days,
        constituents,
    )
}

fn band_averaged_fft_vector_residual_power(
    eastward: &[f64],
    northward: &[f64],
    sample_interval_hours: f64,
    effective_record_length_days: f64,
    constituents: &[Constituent],
) -> VectorResidualPower {
    debug_assert_eq!(eastward.len(), northward.len());
    let sample_count = eastward.len() - eastward.len() % 2;
    let denominator = usize_to_f64(sample_count);
    let eastward_mean = eastward[..sample_count].iter().sum::<f64>() / denominator;
    let northward_mean = northward[..sample_count].iter().sum::<f64>() / denominator;
    let mut window_energy = 0.0;
    let mut eastward_spectrum = Vec::with_capacity(sample_count);
    let mut northward_spectrum = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let window = periodic_hann(index, sample_count);
        window_energy += window * window;
        eastward_spectrum.push(Complex::new(
            (eastward[index] - eastward_mean) * window,
            0.0,
        ));
        northward_spectrum.push(Complex::new(
            (northward[index] - northward_mean) * window,
            0.0,
        ));
    }
    let transform = cached_forward_fft(sample_count);
    transform.process(&mut eastward_spectrum);
    transform.process(&mut northward_spectrum);

    let frequency_count = sample_count / 2 + 1;
    let frequency_spacing = 1.0 / (denominator * sample_interval_hours);
    let normalization = sample_interval_hours / window_energy;
    let mut eastward_power = Vec::with_capacity(frequency_count);
    let mut northward_power = Vec::with_capacity(frequency_count);
    let mut cross_power = Vec::with_capacity(frequency_count);
    for (index, (eastward, northward)) in eastward_spectrum[..frequency_count]
        .iter()
        .zip(&northward_spectrum[..frequency_count])
        .enumerate()
    {
        let one_sided = if index == 0 || index + 1 == frequency_count {
            1.0
        } else {
            2.0
        };
        eastward_power.push(eastward.norm_sqr() * normalization * one_sided);
        northward_power.push(northward.norm_sqr() * normalization * one_sided);
        cross_power.push((eastward.conj() * northward).re * normalization * one_sided);
    }

    VectorResidualPower {
        eastward: regular_band_power_by_constituent(
            frequency_spacing,
            &eastward_power,
            effective_record_length_days,
            constituents,
        ),
        northward: regular_band_power_by_constituent(
            frequency_spacing,
            &northward_power,
            effective_record_length_days,
            constituents,
        ),
        cross: regular_band_power_by_constituent(
            frequency_spacing,
            &cross_power,
            effective_record_length_days,
            constituents,
        ),
    }
}

fn regular_band_power_by_constituent(
    frequency_spacing: f64,
    spectrum: &[f64],
    effective_record_length_days: f64,
    constituents: &[Constituent],
) -> Vec<f64> {
    let frequency_count = spectrum.len();

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
                sum += spectrum[index];
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

fn band_averaged_lomb_residual_power(
    residual: &[f64],
    time_hours: &[f64],
    effective_record_length_days: f64,
    constituents: &[Constituent],
) -> Vec<f64> {
    let sample_count = residual.len() - residual.len() % 2;
    let residual = &residual[..sample_count];
    let time_hours = &time_hours[..sample_count];
    let mean = residual.iter().sum::<f64>() / usize_to_f64(sample_count);
    let first_time = time_hours[0];
    let time_span = time_hours[sample_count - 1] - first_time;
    let uniform_step = time_span / usize_to_f64(sample_count - 1);
    let mut window_energy = 0.0;
    let windowed = residual
        .iter()
        .zip(time_hours)
        .map(|(value, time)| {
            let uniform_position =
                ((*time - first_time) / uniform_step).clamp(0.0, usize_to_f64(sample_count - 1));
            let left = bounded_frequency_index(uniform_position.floor());
            let right = (left + 1).min(sample_count - 1);
            let fraction = uniform_position - usize_to_f64(left);
            let left_window = periodic_hann(left, sample_count);
            let right_window = periodic_hann(right, sample_count);
            let window = left_window + fraction * (right_window - left_window);
            window_energy += window * window;
            (*value - mean) * window
        })
        .collect::<Vec<_>>();

    let frequencies = lomb_frequencies(time_hours);
    let delta_time = time_span / usize_to_f64(sample_count - 1);
    let normalization = 2.0 * delta_time * usize_to_f64(sample_count) / window_energy;
    let spectrum = frequencies
        .iter()
        .copied()
        .map(|frequency| {
            normalization * lomb_scargle_unnormalized(time_hours, &windowed, frequency)
        })
        .collect::<Vec<_>>();
    band_power_by_constituent(
        &frequencies,
        &spectrum,
        effective_record_length_days,
        constituents,
    )
}

fn band_averaged_lomb_vector_residual_power(
    eastward: &[f64],
    northward: &[f64],
    time_hours: &[f64],
    effective_record_length_days: f64,
    constituents: &[Constituent],
) -> VectorResidualPower {
    debug_assert_eq!(eastward.len(), northward.len());
    let sample_count = eastward.len() - eastward.len() % 2;
    let eastward = &eastward[..sample_count];
    let northward = &northward[..sample_count];
    let time_hours = &time_hours[..sample_count];
    let denominator = usize_to_f64(sample_count);
    let eastward_mean = eastward.iter().sum::<f64>() / denominator;
    let northward_mean = northward.iter().sum::<f64>() / denominator;
    let first_time = time_hours[0];
    let time_span = time_hours[sample_count - 1] - first_time;
    let uniform_step = time_span / usize_to_f64(sample_count - 1);
    let mut window_energy = 0.0;
    let mut eastward_windowed = Vec::with_capacity(sample_count);
    let mut northward_windowed = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let uniform_position = ((time_hours[index] - first_time) / uniform_step)
            .clamp(0.0, usize_to_f64(sample_count - 1));
        let left = bounded_frequency_index(uniform_position.floor());
        let right = (left + 1).min(sample_count - 1);
        let fraction = uniform_position - usize_to_f64(left);
        let window = periodic_hann(left, sample_count)
            + fraction * (periodic_hann(right, sample_count) - periodic_hann(left, sample_count));
        window_energy += window * window;
        eastward_windowed.push((eastward[index] - eastward_mean) * window);
        northward_windowed.push((northward[index] - northward_mean) * window);
    }

    let frequencies = lomb_frequencies(time_hours);
    let delta_time = time_span / usize_to_f64(sample_count - 1);
    let normalization = 2.0 * delta_time * denominator / window_energy;
    let mut eastward_spectrum = Vec::with_capacity(frequencies.len());
    let mut northward_spectrum = Vec::with_capacity(frequencies.len());
    let mut cross_spectrum = Vec::with_capacity(frequencies.len());
    for frequency in &frequencies {
        let (eastward_power, northward_power, cross_power) = lomb_scargle_unnormalized_vector(
            time_hours,
            &eastward_windowed,
            &northward_windowed,
            *frequency,
        );
        eastward_spectrum.push(normalization * eastward_power);
        northward_spectrum.push(normalization * northward_power);
        cross_spectrum.push(normalization * cross_power);
    }
    VectorResidualPower {
        eastward: band_power_by_constituent(
            &frequencies,
            &eastward_spectrum,
            effective_record_length_days,
            constituents,
        ),
        northward: band_power_by_constituent(
            &frequencies,
            &northward_spectrum,
            effective_record_length_days,
            constituents,
        ),
        cross: band_power_by_constituent(
            &frequencies,
            &cross_spectrum,
            effective_record_length_days,
            constituents,
        ),
    }
}

/// Timestamp-only work shared by every colored-confidence solve on one record.
///
/// The frequency-major layout keeps each projection contiguous. A plan is
/// created lazily and only for records known to be reusable; unique missing-data
/// masks retain the direct implementation above to avoid an `O(N * F)` memory
/// allocation that cannot be amortized.
#[derive(Debug)]
struct LombScarglePlan {
    sample_count: usize,
    window: Vec<f64>,
    frequencies: Vec<f64>,
    frequency_bases: Vec<LombFrequencyBasis>,
    normalization: f64,
}

impl LombScarglePlan {
    fn new(time_hours: &[f64]) -> Self {
        Self::new_with_parallelism(time_hours, false)
    }

    fn new_parallel(time_hours: &[f64]) -> Self {
        Self::new_with_parallelism(time_hours, true)
    }

    fn new_with_parallelism(time_hours: &[f64], parallel: bool) -> Self {
        let sample_count = time_hours.len() - time_hours.len() % 2;
        let time_hours = &time_hours[..sample_count];
        let first_time = time_hours[0];
        let time_span = time_hours[sample_count - 1] - first_time;
        let uniform_step = time_span / usize_to_f64(sample_count - 1);
        let mut window_energy = 0.0;
        let window = time_hours
            .iter()
            .map(|time| {
                let uniform_position = ((*time - first_time) / uniform_step)
                    .clamp(0.0, usize_to_f64(sample_count - 1));
                let left = bounded_frequency_index(uniform_position.floor());
                let right = (left + 1).min(sample_count - 1);
                let fraction = uniform_position - usize_to_f64(left);
                let left_window = periodic_hann(left, sample_count);
                let right_window = periodic_hann(right, sample_count);
                let value = left_window + fraction * (right_window - left_window);
                window_energy += value * value;
                value
            })
            .collect::<Vec<_>>();

        let frequencies = lomb_frequencies(time_hours);
        let build_basis =
            |frequency: &f64| LombFrequencyBasis::new(time_hours, first_time, *frequency);
        let frequency_bases = if parallel {
            frequencies.par_iter().map(build_basis).collect()
        } else {
            frequencies.iter().map(build_basis).collect()
        };

        let delta_time = time_span / usize_to_f64(sample_count - 1);
        let normalization = 2.0 * delta_time * usize_to_f64(sample_count) / window_energy;
        Self {
            sample_count,
            window,
            frequencies,
            frequency_bases,
            normalization,
        }
    }

    fn band_averaged_residual_power(
        &self,
        residual: &[f64],
        effective_record_length_days: f64,
        constituents: &[Constituent],
    ) -> Vec<f64> {
        let residual = &residual[..self.sample_count];
        let mean = residual.iter().sum::<f64>() / usize_to_f64(self.sample_count);
        let windowed = residual
            .iter()
            .zip(&self.window)
            .map(|(value, window)| (*value - mean) * window)
            .collect::<Vec<_>>();
        let spectrum = self
            .frequency_bases
            .iter()
            .map(|basis| {
                let (cosine_projection, sine_projection) =
                    basis.cosine.iter().zip(&basis.sine).zip(&windowed).fold(
                        (0.0, 0.0),
                        |acc, ((cosine, sine), value)| {
                            (acc.0 + value * cosine, acc.1 + value * sine)
                        },
                    );
                self.normalization
                    * (cosine_projection.powi(2) / basis.cosine_energy)
                        .midpoint(sine_projection.powi(2) / basis.sine_energy)
            })
            .collect::<Vec<_>>();
        band_power_by_constituent(
            &self.frequencies,
            &spectrum,
            effective_record_length_days,
            constituents,
        )
    }

    fn band_averaged_vector_residual_power(
        &self,
        eastward: &[f64],
        northward: &[f64],
        effective_record_length_days: f64,
        constituents: &[Constituent],
    ) -> VectorResidualPower {
        debug_assert_eq!(eastward.len(), northward.len());
        let eastward = &eastward[..self.sample_count];
        let northward = &northward[..self.sample_count];
        let denominator = usize_to_f64(self.sample_count);
        let eastward_mean = eastward.iter().sum::<f64>() / denominator;
        let northward_mean = northward.iter().sum::<f64>() / denominator;
        let eastward_windowed = eastward
            .iter()
            .zip(&self.window)
            .map(|(value, window)| (*value - eastward_mean) * window)
            .collect::<Vec<_>>();
        let northward_windowed = northward
            .iter()
            .zip(&self.window)
            .map(|(value, window)| (*value - northward_mean) * window)
            .collect::<Vec<_>>();
        let mut eastward_spectrum = Vec::with_capacity(self.frequencies.len());
        let mut northward_spectrum = Vec::with_capacity(self.frequencies.len());
        let mut cross_spectrum = Vec::with_capacity(self.frequencies.len());
        for basis in &self.frequency_bases {
            let projections = basis
                .cosine
                .iter()
                .zip(&basis.sine)
                .zip(&eastward_windowed)
                .zip(&northward_windowed)
                .fold(
                    [0.0; 4],
                    |mut sums, (((cosine, sine), eastward), northward)| {
                        sums[0] += eastward * cosine;
                        sums[1] += eastward * sine;
                        sums[2] += northward * cosine;
                        sums[3] += northward * sine;
                        sums
                    },
                );
            eastward_spectrum.push(
                self.normalization
                    * (projections[0].powi(2) / basis.cosine_energy)
                        .midpoint(projections[1].powi(2) / basis.sine_energy),
            );
            northward_spectrum.push(
                self.normalization
                    * (projections[2].powi(2) / basis.cosine_energy)
                        .midpoint(projections[3].powi(2) / basis.sine_energy),
            );
            cross_spectrum.push(
                self.normalization
                    * (projections[0] * projections[2] / basis.cosine_energy)
                        .midpoint(projections[1] * projections[3] / basis.sine_energy),
            );
        }
        VectorResidualPower {
            eastward: band_power_by_constituent(
                &self.frequencies,
                &eastward_spectrum,
                effective_record_length_days,
                constituents,
            ),
            northward: band_power_by_constituent(
                &self.frequencies,
                &northward_spectrum,
                effective_record_length_days,
                constituents,
            ),
            cross: band_power_by_constituent(
                &self.frequencies,
                &cross_spectrum,
                effective_record_length_days,
                constituents,
            ),
        }
    }
}

#[derive(Debug)]
struct LombFrequencyBasis {
    cosine: Vec<f64>,
    sine: Vec<f64>,
    cosine_energy: f64,
    sine_energy: f64,
}

impl LombFrequencyBasis {
    fn new(time_hours: &[f64], first_time: f64, frequency: f64) -> Self {
        let angular_frequency = TAU * frequency;
        let mut sine = Vec::with_capacity(time_hours.len());
        let mut cosine = Vec::with_capacity(time_hours.len());
        let mut double_sine = 0.0;
        let mut double_cosine = 0.0;
        for time in time_hours {
            let (basis_sine, basis_cosine) = (angular_frequency * (*time - first_time)).sin_cos();
            sine.push(basis_sine);
            cosine.push(basis_cosine);
            double_sine += 2.0 * basis_sine * basis_cosine;
            double_cosine += basis_cosine * basis_cosine - basis_sine * basis_sine;
        }
        let phase_shift = 0.5 * double_sine.atan2(double_cosine);
        let (phase_sine, phase_cosine) = phase_shift.sin_cos();
        let mut cosine_energy = 0.0;
        let mut sine_energy = 0.0;
        for index in 0..time_hours.len() {
            let raw_sine = sine[index];
            let raw_cosine = cosine[index];
            let basis_sine = raw_sine * phase_cosine - raw_cosine * phase_sine;
            let basis_cosine = raw_cosine * phase_cosine + raw_sine * phase_sine;
            sine[index] = basis_sine;
            cosine[index] = basis_cosine;
            cosine_energy += basis_cosine * basis_cosine;
            sine_energy += basis_sine * basis_sine;
        }
        Self {
            cosine,
            sine,
            cosine_energy,
            sine_energy,
        }
    }
}

fn periodic_hann(index: usize, length: usize) -> f64 {
    0.5 - 0.5 * (TAU * usize_to_f64(index) / usize_to_f64(length)).cos()
}

fn estimated_lomb_basis_bytes(time_hours: &[f64]) -> usize {
    let sample_count = time_hours.len() - time_hours.len() % 2;
    let frequency_count = lomb_frequencies(&time_hours[..sample_count]).len();
    sample_count
        .saturating_mul(frequency_count)
        .saturating_mul(2 * std::mem::size_of::<f64>())
}

fn lomb_scargle_unnormalized(time_hours: &[f64], values: &[f64], frequency_cph: f64) -> f64 {
    let angular_frequency = TAU * frequency_cph;
    let (double_sine, double_cosine) = time_hours.iter().fold((0.0, 0.0), |acc, time| {
        let angle = 2.0 * angular_frequency * time;
        (acc.0 + angle.sin(), acc.1 + angle.cos())
    });
    let phase_shift = 0.5 * double_sine.atan2(double_cosine);
    let (cosine_projection, cosine_energy, sine_projection, sine_energy) = time_hours
        .iter()
        .zip(values)
        .fold((0.0, 0.0, 0.0, 0.0), |acc, (time, value)| {
            let (sine, cosine) = (angular_frequency * time - phase_shift).sin_cos();
            (
                acc.0 + value * cosine,
                acc.1 + cosine * cosine,
                acc.2 + value * sine,
                acc.3 + sine * sine,
            )
        });
    (cosine_projection.powi(2) / cosine_energy).midpoint(sine_projection.powi(2) / sine_energy)
}

fn lomb_scargle_unnormalized_vector(
    time_hours: &[f64],
    eastward: &[f64],
    northward: &[f64],
    frequency_cph: f64,
) -> (f64, f64, f64) {
    let angular_frequency = TAU * frequency_cph;
    let (double_sine, double_cosine) = time_hours.iter().fold((0.0, 0.0), |acc, time| {
        let angle = 2.0 * angular_frequency * time;
        (acc.0 + angle.sin(), acc.1 + angle.cos())
    });
    let phase_shift = 0.5 * double_sine.atan2(double_cosine);
    let projections = time_hours.iter().zip(eastward).zip(northward).fold(
        [0.0; 6],
        |mut sums, ((time, eastward), northward)| {
            let (sine, cosine) = (angular_frequency * time - phase_shift).sin_cos();
            sums[0] += eastward * cosine;
            sums[1] += eastward * sine;
            sums[2] += northward * cosine;
            sums[3] += northward * sine;
            sums[4] += cosine * cosine;
            sums[5] += sine * sine;
            sums
        },
    );
    (
        (projections[0].powi(2) / projections[4]).midpoint(projections[1].powi(2) / projections[5]),
        (projections[2].powi(2) / projections[4]).midpoint(projections[3].powi(2) / projections[5]),
        (projections[0] * projections[2] / projections[4])
            .midpoint(projections[1] * projections[3] / projections[5]),
    )
}

fn band_power_by_constituent(
    frequencies: &[f64],
    spectrum: &[f64],
    effective_record_length_days: f64,
    constituents: &[Constituent],
) -> Vec<f64> {
    let mut excluded = vec![false; frequencies.len()];
    for constituent in constituents {
        let interpolated = interpolated_frequency_index(frequencies, constituent.frequency_cph);
        let nearest = interpolated
            .round_ties_even()
            .clamp(0.0, usize_to_f64(frequencies.len() - 1));
        excluded[bounded_frequency_index(nearest)] = true;
    }
    let band_power = FREQUENCY_BANDS_CPH.map(|[lower, upper]| {
        let start = frequencies.partition_point(|frequency| *frequency < lower);
        let upper_insertion = frequencies.partition_point(|frequency| *frequency < upper);
        let stop = (upper_insertion + 1).min(frequencies.len());
        let mut sum = 0.0;
        let mut count = 0_usize;
        for index in start..stop {
            if !excluded[index] {
                sum += spectrum[index];
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

fn interpolated_frequency_index(frequencies: &[f64], frequency: f64) -> f64 {
    if frequency <= frequencies[0] {
        return 0.0;
    }
    let last = frequencies.len() - 1;
    if frequency >= frequencies[last] {
        return usize_to_f64(last);
    }
    let right = frequencies.partition_point(|candidate| *candidate < frequency);
    let left = right - 1;
    usize_to_f64(left) + (frequency - frequencies[left]) / (frequencies[right] - frequencies[left])
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

#[allow(
    clippy::cast_precision_loss,
    reason = "allocated slice and matrix lengths are exactly representable in practical f64 analyses"
)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

pub(crate) fn validate_constituents(constituents: &[Constituent]) -> Result<(), AnalysisError> {
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
    validate_time_with_options(time_days, constituent_count, FitOptions::default())
}

pub(crate) fn validate_time_with_options(
    time_days: &[f64],
    constituent_count: usize,
    fit_options: FitOptions,
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
    let parameter_count = constituent_count * 2 + 1 + usize::from(fit_options.trend);
    let required = parameter_count + 1;
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
    use super::{
        Constituent, FitOptions, FixedRawOls, IrregularSpectrumSampling, LinearConfidence,
        LombScarglePlan, MIN_PROJECTED_BATCH_SERIES, band_averaged_fft_vector_residual_power,
        band_averaged_lomb_residual_power, band_averaged_lomb_vector_residual_power,
        lomb_frequencies, usize_to_f64,
    };
    use crate::AnalysisError;
    use std::f64::consts::TAU;

    fn constituents() -> Vec<Constituent> {
        vec![
            Constituent::new("M2", 0.080_511_400_671_577_2),
            Constituent::new("S2", 1.0 / 12.0),
            Constituent::new("K1", 0.041_780_746_221_637_22),
        ]
    }

    #[test]
    fn trend_disabled_model_uses_one_fewer_column_and_reports_zero_slope() {
        let times = [0.0, 0.2, 0.55, 1.0];
        let constituent = Constituent::new("test", 1.0 / 24.0);
        let observations = times
            .iter()
            .map(|time| {
                let angle = TAU * time;
                0.4 + 1.2 * angle.cos() - 0.3 * angle.sin()
            })
            .collect::<Vec<_>>();
        assert!(FixedRawOls::prepare(&times, std::slice::from_ref(&constituent)).is_err());

        let model =
            FixedRawOls::prepare_with_options(&times, &[constituent], FitOptions { trend: false })
                .expect("four samples overdetermine cosine, sine, and mean");
        let solution = model.solve(&observations).expect("valid observations");
        assert_eq!(model.fit_options(), FitOptions { trend: false });
        // The returned raw coefficients are defined at the record midpoint.
        assert_close(solution.cosine_coefficient[0], -1.2, 1e-12);
        assert_close(solution.sine_coefficient[0], 0.3, 1e-12);
        assert_close(solution.mean, 0.4, 1e-12);
        assert_close(solution.slope_per_day, 0.0, f64::EPSILON);
    }

    fn times() -> Vec<f64> {
        (0..745)
            .map(|index| 58_113.0 + f64::from(index) / 24.0)
            .collect()
    }

    #[test]
    fn irregular_basis_cache_is_opt_in_and_memory_bounded() {
        let short = (0_u32..745)
            .map(|index| 58_113.0 + f64::from(index) / 24.0)
            .collect::<Vec<_>>();
        assert!(
            IrregularSpectrumSampling::new(&short, true)
                .shared_plan
                .is_some()
        );
        assert!(
            IrregularSpectrumSampling::new(&short, false)
                .shared_plan
                .is_none()
        );

        let long = (0_u32..5_000)
            .map(|index| 58_113.0 + f64::from(index) / 24.0)
            .collect::<Vec<_>>();
        assert!(
            IrregularSpectrumSampling::new(&long, true)
                .shared_plan
                .is_none()
        );
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
    fn projected_batch_matches_individual_qr_solves() {
        let constituents = constituents();
        let times = times();
        let series_count = MIN_PROJECTED_BATCH_SERIES;
        let mut time_major = Vec::with_capacity(times.len() * series_count);
        for (time_index, time) in times.iter().copied().enumerate() {
            for series in 0..series_count {
                let series = f64::from(u32::try_from(series).expect("small fixture"));
                let time_index = f64::from(u32::try_from(time_index).expect("small fixture"));
                time_major.push(
                    0.03 * series
                        + 0.4 * (time * 1.7 + series * 0.1).cos()
                        + 0.07 * (time_index / (5.0 + series * 0.02)).sin(),
                );
            }
        }

        let model = FixedRawOls::prepare(&times, &constituents).expect("valid model");
        let batch = model
            .solve_many_time_major(&time_major, series_count)
            .expect("valid projected batch");
        for series in 0..series_count {
            let observations = time_major
                .chunks_exact(series_count)
                .map(|row| row[series])
                .collect::<Vec<_>>();
            let individual = model.solve(&observations).expect("valid QR solve");
            for (actual, expected) in batch[series]
                .cosine_coefficient
                .iter()
                .zip(&individual.cosine_coefficient)
                .chain(
                    batch[series]
                        .sine_coefficient
                        .iter()
                        .zip(&individual.sine_coefficient),
                )
            {
                assert_close(*actual, *expected, 2e-12);
            }
            assert_close(batch[series].mean, individual.mean, 2e-12);
            assert_close(batch[series].slope_per_day, individual.slope_per_day, 2e-12);
        }
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
    fn supports_white_and_colored_confidence_for_irregular_time() {
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
        assert!(
            model
                .solve_with_linear_confidence(&observations, LinearConfidence::Colored)
                .is_ok()
        );
    }

    #[test]
    fn lomb_scargle_band_power_matches_python_utide() {
        let sample_count = 744_usize;
        let mut time_hours = Vec::with_capacity(sample_count);
        time_hours.push(58_113.0 * 24.0);
        for index in 1..sample_count {
            let index = usize_to_f64(index);
            let step = 1.0 + 0.08 * (index * 0.37).sin() + 0.03 * (index * 0.11).cos();
            time_hours.push(time_hours[time_hours.len() - 1] + step);
        }
        let residual = (0..sample_count)
            .map(|index| {
                let index = usize_to_f64(index);
                0.3 * (index * 0.071).sin()
                    + 0.2 * (index * 0.173).cos()
                    + 0.04 * (index * index * 0.003).sin()
            })
            .collect::<Vec<_>>();
        let constituents = constituents();
        let effective_record_length_days = (time_hours[sample_count - 1] - time_hours[0]) / 24.0
            * usize_to_f64(sample_count)
            / usize_to_f64(sample_count - 1);

        let frequencies = lomb_frequencies(&time_hours);
        assert_eq!(frequencies.len(), 258);
        assert_close(frequencies[0], 0.001_343_755_858_561_583_2, 5e-15);
        assert_close(
            frequencies[frequencies.len() - 1],
            0.498_533_423_526_347_4,
            2e-12,
        );
        let power = band_averaged_lomb_residual_power(
            &residual,
            &time_hours,
            effective_record_length_days,
            &constituents,
        );
        for (actual, expected) in power.iter().zip([
            5.806_375_784_746_85e-6,
            5.806_375_784_746_85e-6,
            7.338_218_018_124_799_5e-6,
        ]) {
            assert_close(*actual, expected, 2e-14);
        }
    }

    #[test]
    fn vector_cross_spectra_match_python_utide_for_regular_and_irregular_time() {
        let sample_count = 744_usize;
        let eastward = (0..sample_count)
            .map(|index| {
                let index = usize_to_f64(index);
                0.3 * (index * 0.071).sin()
                    + 0.2 * (index * 0.173).cos()
                    + 0.04 * (index * index * 0.003).sin()
            })
            .collect::<Vec<_>>();
        let northward = (0..sample_count)
            .map(|index| {
                let index = usize_to_f64(index);
                -0.17 * (index * 0.053).cos()
                    + 0.11 * (index * 0.131).sin()
                    + 0.025 * (index * index * 0.002).cos()
            })
            .collect::<Vec<_>>();
        let constituents = constituents();
        let regular = band_averaged_fft_vector_residual_power(
            &eastward,
            &northward,
            1.0,
            31.0,
            &constituents,
        );
        for (actual, expected) in regular.eastward.iter().zip([
            6.320_636_150_930_253e-8,
            6.320_636_150_930_253e-8,
            9.910_877_830_793_654e-8,
        ]) {
            assert_close(*actual, expected, 1e-17);
        }
        for (actual, expected) in regular.northward.iter().zip([
            1.673_595_936_245_554_2e-7,
            1.673_595_936_245_554_2e-7,
            1.383_637_853_460_331_8e-8,
        ]) {
            assert_close(*actual, expected, 1e-17);
        }
        for (actual, expected) in regular.cross.iter().zip([
            -1.124_815_607_960_934_4e-8,
            -1.124_815_607_960_934_4e-8,
            7.508_596_259_390_926e-9,
        ]) {
            assert_close(*actual, expected, 1e-17);
        }

        let mut time_hours = Vec::with_capacity(sample_count);
        time_hours.push(58_113.0 * 24.0);
        for index in 1..sample_count {
            let index = usize_to_f64(index);
            let step = 1.0 + 0.08 * (index * 0.37).sin() + 0.03 * (index * 0.11).cos();
            time_hours.push(time_hours[time_hours.len() - 1] + step);
        }
        let effective_record_length_days = (time_hours[sample_count - 1] - time_hours[0]) / 24.0
            * usize_to_f64(sample_count)
            / usize_to_f64(sample_count - 1);
        let irregular = band_averaged_lomb_vector_residual_power(
            &eastward,
            &northward,
            &time_hours,
            effective_record_length_days,
            &constituents,
        );
        let planned = LombScarglePlan::new(&time_hours).band_averaged_vector_residual_power(
            &eastward,
            &northward,
            effective_record_length_days,
            &constituents,
        );
        for (actual, expected) in irregular.eastward.iter().zip([
            5.806_375_783_887_969e-6,
            5.806_375_783_887_969e-6,
            7.338_218_018_561_512e-6,
        ]) {
            assert_close(*actual, expected, 2e-15);
        }
        for (actual, expected) in irregular.northward.iter().zip([
            1.245_734_923_566_412_4e-6,
            1.245_734_923_566_412_4e-6,
            1.933_804_951_157_388_8e-6,
        ]) {
            assert_close(*actual, expected, 2e-15);
        }
        for (actual, expected) in irregular.cross.iter().zip([
            -1.011_981_330_836_897_9e-7,
            -1.011_981_330_836_897_9e-7,
            -1.231_584_575_732_415e-7,
        ]) {
            assert_close(*actual, expected, 2e-16);
        }
        for (direct, cached) in [
            (&irregular.eastward, &planned.eastward),
            (&irregular.northward, &planned.northward),
            (&irregular.cross, &planned.cross),
        ] {
            for (direct, cached) in direct.iter().zip(cached) {
                assert_close(*direct, *cached, 2e-15);
            }
        }
    }

    #[test]
    fn lomb_scargle_matches_random_jitter_clustered_gap_and_band_edges() {
        let sample_count = 744_usize;
        let mut state = 0x1234_5678_u32;
        let mut time_hours = Vec::with_capacity(sample_count);
        time_hours.push(58_113.0 * 24.0);
        for _ in 1..sample_count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let step = 0.85 + 0.3 * f64::from(state) / 4_294_967_296.0;
            time_hours.push(time_hours[time_hours.len() - 1] + step);
        }
        let residual = (0..sample_count)
            .map(|index| {
                let index = usize_to_f64(index);
                0.3 * (index * 0.071).sin()
                    + 0.2 * (index * 0.173).cos()
                    + 0.04 * (index * index * 0.003).sin()
            })
            .collect::<Vec<_>>();
        let constituents = [
            Constituent::new("M2", 0.080_511_400_671_577_2),
            Constituent::new("S2", 1.0 / 12.0),
            Constituent::new("K1", 0.041_780_746_221_637_22),
            Constituent::new("lower band upper edge", 0.004_17),
            Constituent::new("diurnal lower edge", 0.031_92),
        ];

        let calculate = |times: &[f64], values: &[f64]| {
            let effective_days = (times[times.len() - 1] - times[0]) / 24.0
                * usize_to_f64(times.len())
                / usize_to_f64(times.len() - 1);
            band_averaged_lomb_residual_power(values, times, effective_days, &constituents)
        };
        let random_jitter = calculate(&time_hours, &residual);
        let planned_random_jitter = LombScarglePlan::new(&time_hours).band_averaged_residual_power(
            &residual,
            (time_hours[sample_count - 1] - time_hours[0]) / 24.0 * usize_to_f64(sample_count)
                / usize_to_f64(sample_count - 1),
            &constituents,
        );
        for (planned, direct) in planned_random_jitter.iter().zip(&random_jitter) {
            assert_close(*planned, *direct, 5e-16);
        }
        for (actual, expected) in random_jitter.iter().zip([
            3.070_558_088_690_660_4e-6,
            3.070_558_088_690_660_4e-6,
            6.552_421_375_679_652e-6,
            1.397_353_459_283_417_8e-5,
            6.552_421_375_679_652e-6,
        ]) {
            assert_close(*actual, expected, 2e-13);
        }

        let retained = (0..sample_count)
            .filter(|index| !(250..310).contains(index))
            .collect::<Vec<_>>();
        let clustered_time = retained
            .iter()
            .map(|index| time_hours[*index])
            .collect::<Vec<_>>();
        let clustered_residual = retained
            .iter()
            .map(|index| residual[*index])
            .collect::<Vec<_>>();
        let clustered_gap = calculate(&clustered_time, &clustered_residual);
        for (actual, expected) in clustered_gap.iter().zip([
            5.830_275_141_122_778_5e-6,
            5.830_275_141_122_778_5e-6,
            5.425_865_304_226_831e-5,
            4.865_532_439_775_149e-4,
            5.425_865_304_226_831e-5,
        ]) {
            assert_close(*actual, expected, 2e-12);
        }
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
