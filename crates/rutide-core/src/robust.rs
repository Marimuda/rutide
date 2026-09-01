//! Iteratively reweighted least squares for outlier-resistant harmonic fits.

use faer::{
    Mat, c64,
    linalg::solvers::{DenseSolveCore, SolveLstsq},
};

use crate::AnalysisError;

const MAD_NORMALIZATION: f64 = 0.6745;

/// Configuration for Python-UTide-compatible Cauchy robust fitting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustOptions {
    /// Cauchy tuning constant. Larger values reduce outlier sensitivity.
    pub tuning_constant: f64,
    /// Stop when fractional weighted-mean-square improvement falls below this.
    pub tolerance: f64,
    /// Maximum weighted least-squares iterations.
    pub max_iterations: usize,
}

impl Default for RobustOptions {
    fn default() -> Self {
        Self {
            tuning_constant: 2.385,
            tolerance: 0.001,
            max_iterations: 50,
        }
    }
}

impl RobustOptions {
    pub(crate) fn validate(self) -> Result<(), AnalysisError> {
        if !self.tuning_constant.is_finite() || self.tuning_constant <= 0.0 {
            return Err(AnalysisError::InvalidRobustTuningConstant);
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(AnalysisError::InvalidRobustTolerance);
        }
        if self.max_iterations == 0 {
            return Err(AnalysisError::InvalidRobustIterationLimit);
        }
        Ok(())
    }
}

/// Why a robust fit stopped successfully.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RobustTermination {
    /// The weighted mean-square improvement fell below the configured tolerance.
    Tolerance,
    /// The new objective was worse, so the preceding iterate was retained.
    ObjectiveIncrease,
    /// The OLS residual was numerically zero, so reweighting was unnecessary.
    ExactFit,
}

/// Auditable diagnostics from a converged robust fit.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustDiagnostics {
    /// Final weight for each retained time row, in timestamp order.
    pub weights: Vec<f64>,
    /// OLS leverage, or diagonal of the unweighted model hat matrix.
    pub leverage: Vec<f64>,
    /// One-based number of weighted least-squares iterations performed.
    pub iterations: usize,
    /// Successful stopping condition.
    pub termination: RobustTermination,
    /// Median-absolute-deviation residual scale that produced the final weights.
    pub residual_scale: f64,
    /// Root-mean-square residual of the initial OLS fit.
    pub ols_rms_residual: f64,
    /// Root-mean-square unweighted residual of the retained robust fit.
    pub rms_residual: f64,
}

pub(crate) struct RobustFit {
    pub(crate) coefficients: Mat<f64>,
    pub(crate) diagnostics: RobustDiagnostics,
}

pub(crate) struct ComplexRobustFit {
    pub(crate) coefficients: Mat<c64>,
    pub(crate) diagnostics: RobustDiagnostics,
}

struct Iteration {
    coefficients: Mat<f64>,
    weights: Vec<f64>,
    residuals: Vec<[f64; 2]>,
    residual_sum_squares: f64,
    weighted_mean_square: f64,
    weight_scale: Option<f64>,
}

struct ComplexIteration {
    coefficients: Mat<c64>,
    weights: Vec<f64>,
    residuals: Vec<c64>,
    residual_sum_squares: f64,
    weighted_mean_square: f64,
    weight_scale: Option<f64>,
}

#[cfg(test)]
fn fit(
    design: &Mat<f64>,
    observations: &Mat<f64>,
    options: RobustOptions,
) -> Result<RobustFit, AnalysisError> {
    let initial_coefficients = design.col_piv_qr().solve_lstsq(observations.as_ref());
    fit_with_initial(design, observations, initial_coefficients, options)
}

pub(crate) fn fit_with_initial(
    design: &Mat<f64>,
    observations: &Mat<f64>,
    initial_coefficients: Mat<f64>,
    options: RobustOptions,
) -> Result<RobustFit, AnalysisError> {
    options.validate()?;
    debug_assert_eq!(design.nrows(), observations.nrows());
    debug_assert!(matches!(observations.ncols(), 1 | 2));
    debug_assert_eq!(initial_coefficients.nrows(), design.ncols());
    debug_assert_eq!(initial_coefficients.ncols(), observations.ncols());

    let leverage = leverage(design)?;
    let residual_factor = leverage
        .iter()
        .map(|value| 1.0 / (options.tuning_constant * (1.0 - value).sqrt()))
        .collect::<Vec<_>>();
    let mut weights = vec![1.0; design.nrows()];
    let mut weight_scale = None;
    let mut previous: Option<Iteration> = None;
    let mut ols_rms_residual = 0.0;

    let mut initial_coefficients = Some(initial_coefficients);
    for iteration_index in 0..options.max_iterations {
        let current = if let Some(coefficients) = initial_coefficients.take() {
            iteration_from_coefficients(design, observations, coefficients, weights, weight_scale)
        } else {
            solve_iteration(design, observations, weights, weight_scale)
        };
        if iteration_index == 0 {
            ols_rms_residual = (current.residual_sum_squares / usize_to_f64(design.nrows())).sqrt();
        }

        if let Some(previous_iteration) = previous.take() {
            let improvement = (previous_iteration.weighted_mean_square
                - current.weighted_mean_square)
                / previous_iteration.weighted_mean_square;
            if improvement < 0.0 {
                return Ok(completed_fit(
                    previous_iteration,
                    leverage,
                    iteration_index,
                    RobustTermination::ObjectiveIncrease,
                    ols_rms_residual,
                ));
            }
            if improvement < options.tolerance {
                return Ok(completed_fit(
                    current,
                    leverage,
                    iteration_index + 1,
                    RobustTermination::Tolerance,
                    ols_rms_residual,
                ));
            }
        }

        let scale = residual_scale(&current.residuals, observations.ncols());
        if scale == 0.0 {
            if is_numerically_exact(&current.residuals, observations) {
                return Ok(completed_fit(
                    current,
                    leverage,
                    iteration_index + 1,
                    RobustTermination::ExactFit,
                    ols_rms_residual,
                ));
            }
            return Err(AnalysisError::DegenerateRobustScale);
        }
        weights = current
            .residuals
            .iter()
            .zip(&residual_factor)
            .map(|(residual, factor)| {
                let normalized = factor * residual[0].hypot(residual[1]) / scale;
                1.0 / normalized.mul_add(normalized, 1.0)
            })
            .collect();
        weight_scale = Some(scale);
        previous = Some(current);
    }

    Err(AnalysisError::RobustDidNotConverge {
        iterations: options.max_iterations,
    })
}

pub(crate) fn fit_complex_with_initial(
    design: &Mat<c64>,
    observations: &Mat<c64>,
    initial_coefficients: Mat<c64>,
    options: RobustOptions,
) -> Result<ComplexRobustFit, AnalysisError> {
    options.validate()?;
    debug_assert_eq!(design.nrows(), observations.nrows());
    debug_assert_eq!(observations.ncols(), 1);
    debug_assert_eq!(initial_coefficients.nrows(), design.ncols());
    debug_assert_eq!(initial_coefficients.ncols(), 1);

    let leverage = complex_leverage(design)?;
    let residual_factor = leverage
        .iter()
        .map(|value| 1.0 / (options.tuning_constant * (1.0 - value).sqrt()))
        .collect::<Vec<_>>();
    let mut weights = vec![1.0; design.nrows()];
    let mut weight_scale = None;
    let mut previous: Option<ComplexIteration> = None;
    let mut ols_rms_residual = 0.0;

    let mut initial_coefficients = Some(initial_coefficients);
    for iteration_index in 0..options.max_iterations {
        let current = if let Some(coefficients) = initial_coefficients.take() {
            complex_iteration_from_coefficients(
                design,
                observations,
                coefficients,
                weights,
                weight_scale,
            )
        } else {
            solve_complex_iteration(design, observations, weights, weight_scale)
        };
        if iteration_index == 0 {
            ols_rms_residual = (current.residual_sum_squares / usize_to_f64(design.nrows())).sqrt();
        }

        if let Some(previous_iteration) = previous.take() {
            let improvement = (previous_iteration.weighted_mean_square
                - current.weighted_mean_square)
                / previous_iteration.weighted_mean_square;
            if improvement < 0.0 {
                return Ok(completed_complex_fit(
                    previous_iteration,
                    leverage,
                    iteration_index,
                    RobustTermination::ObjectiveIncrease,
                    ols_rms_residual,
                ));
            }
            if improvement < options.tolerance {
                return Ok(completed_complex_fit(
                    current,
                    leverage,
                    iteration_index + 1,
                    RobustTermination::Tolerance,
                    ols_rms_residual,
                ));
            }
        }

        let residual_pairs = current
            .residuals
            .iter()
            .map(|residual| [residual.re, residual.im])
            .collect::<Vec<_>>();
        let scale = residual_scale(&residual_pairs, 2);
        if scale == 0.0 {
            if complex_is_numerically_exact(&current.residuals, observations) {
                return Ok(completed_complex_fit(
                    current,
                    leverage,
                    iteration_index + 1,
                    RobustTermination::ExactFit,
                    ols_rms_residual,
                ));
            }
            return Err(AnalysisError::DegenerateRobustScale);
        }
        weights = current
            .residuals
            .iter()
            .zip(&residual_factor)
            .map(|(residual, factor)| {
                let normalized = factor * residual.re.hypot(residual.im) / scale;
                1.0 / normalized.mul_add(normalized, 1.0)
            })
            .collect();
        weight_scale = Some(scale);
        previous = Some(current);
    }

    Err(AnalysisError::RobustDidNotConverge {
        iterations: options.max_iterations,
    })
}

fn solve_iteration(
    design: &Mat<f64>,
    observations: &Mat<f64>,
    weights: Vec<f64>,
    weight_scale: Option<f64>,
) -> Iteration {
    let weighted_design = Mat::from_fn(design.nrows(), design.ncols(), |row, column| {
        weights[row] * design[(row, column)]
    });
    let weighted_observations =
        Mat::from_fn(observations.nrows(), observations.ncols(), |row, column| {
            weights[row] * observations[(row, column)]
        });
    let coefficients = weighted_design
        .col_piv_qr()
        .solve_lstsq(weighted_observations.as_ref());
    iteration_from_coefficients(design, observations, coefficients, weights, weight_scale)
}

fn solve_complex_iteration(
    design: &Mat<c64>,
    observations: &Mat<c64>,
    weights: Vec<f64>,
    weight_scale: Option<f64>,
) -> ComplexIteration {
    let weighted_design = Mat::from_fn(design.nrows(), design.ncols(), |row, column| {
        weights[row] * design[(row, column)]
    });
    let weighted_observations = Mat::from_fn(observations.nrows(), 1, |row, _| {
        weights[row] * observations[(row, 0)]
    });
    let coefficients = weighted_design
        .col_piv_qr()
        .solve_lstsq(weighted_observations.as_ref());
    complex_iteration_from_coefficients(design, observations, coefficients, weights, weight_scale)
}

fn iteration_from_coefficients(
    design: &Mat<f64>,
    observations: &Mat<f64>,
    coefficients: Mat<f64>,
    weights: Vec<f64>,
    weight_scale: Option<f64>,
) -> Iteration {
    let residuals = residuals(design, observations, &coefficients);
    let residual_sum_squares = residuals
        .iter()
        .zip(&weights)
        .map(|(residual, weight)| {
            let weighted_real = weight * residual[0];
            let weighted_imaginary = weight * residual[1];
            weighted_real.mul_add(weighted_real, weighted_imaginary * weighted_imaginary)
        })
        .sum::<f64>();
    let weighted_mean_square = residual_sum_squares / weights.iter().sum::<f64>();
    Iteration {
        coefficients,
        weights,
        residuals,
        residual_sum_squares,
        weighted_mean_square,
        weight_scale,
    }
}

fn complex_iteration_from_coefficients(
    design: &Mat<c64>,
    observations: &Mat<c64>,
    coefficients: Mat<c64>,
    weights: Vec<f64>,
    weight_scale: Option<f64>,
) -> ComplexIteration {
    let residuals = complex_residuals(design, observations, &coefficients);
    let residual_sum_squares = residuals
        .iter()
        .zip(&weights)
        .map(|(residual, weight)| {
            let real = weight * residual.re;
            let imaginary = weight * residual.im;
            real.mul_add(real, imaginary * imaginary)
        })
        .sum::<f64>();
    let weighted_mean_square = residual_sum_squares / weights.iter().sum::<f64>();
    ComplexIteration {
        coefficients,
        weights,
        residuals,
        residual_sum_squares,
        weighted_mean_square,
        weight_scale,
    }
}

fn completed_fit(
    iteration: Iteration,
    leverage: Vec<f64>,
    iterations: usize,
    termination: RobustTermination,
    ols_rms_residual: f64,
) -> RobustFit {
    let rms_residual = (iteration
        .residuals
        .iter()
        .map(|residual| residual[0].mul_add(residual[0], residual[1] * residual[1]))
        .sum::<f64>()
        / usize_to_f64(iteration.residuals.len()))
    .sqrt();
    RobustFit {
        coefficients: iteration.coefficients,
        diagnostics: RobustDiagnostics {
            weights: iteration.weights,
            leverage,
            iterations,
            termination,
            residual_scale: iteration.weight_scale.unwrap_or(0.0),
            ols_rms_residual,
            rms_residual,
        },
    }
}

fn completed_complex_fit(
    iteration: ComplexIteration,
    leverage: Vec<f64>,
    iterations: usize,
    termination: RobustTermination,
    ols_rms_residual: f64,
) -> ComplexRobustFit {
    let rms_residual = (iteration
        .residuals
        .iter()
        .map(|residual| residual.re.mul_add(residual.re, residual.im * residual.im))
        .sum::<f64>()
        / usize_to_f64(iteration.residuals.len()))
    .sqrt();
    ComplexRobustFit {
        coefficients: iteration.coefficients,
        diagnostics: RobustDiagnostics {
            weights: iteration.weights,
            leverage,
            iterations,
            termination,
            residual_scale: iteration.weight_scale.unwrap_or(0.0),
            ols_rms_residual,
            rms_residual,
        },
    }
}

fn leverage(design: &Mat<f64>) -> Result<Vec<f64>, AnalysisError> {
    let normal = Mat::from_fn(design.ncols(), design.ncols(), |row, column| {
        (0..design.nrows())
            .map(|time| design[(time, row)] * design[(time, column)])
            .sum::<f64>()
    });
    let covariance = normal.partial_piv_lu().inverse();
    (0..design.nrows())
        .map(|time| {
            let value = (0..design.ncols())
                .map(|row| {
                    let projected = (0..design.ncols())
                        .map(|column| covariance[(row, column)] * design[(time, column)])
                        .sum::<f64>();
                    design[(time, row)] * projected
                })
                .sum::<f64>()
                .abs();
            if value.is_finite() && value < 1.0 {
                Ok(value)
            } else {
                Err(AnalysisError::InvalidRobustLeverage { time })
            }
        })
        .collect()
}

fn complex_leverage(design: &Mat<c64>) -> Result<Vec<f64>, AnalysisError> {
    let normal = Mat::from_fn(design.ncols(), design.ncols(), |row, column| {
        (0..design.nrows())
            .map(|time| design[(time, row)].conj() * design[(time, column)])
            .sum::<c64>()
    });
    let covariance = normal.partial_piv_lu().inverse();
    (0..design.nrows())
        .map(|time| {
            let value = (0..design.ncols())
                .map(|row| {
                    let projected = (0..design.ncols())
                        .map(|column| covariance[(row, column)] * design[(time, column)].conj())
                        .sum::<c64>();
                    design[(time, row)] * projected
                })
                .sum::<c64>();
            let absolute = value.re.hypot(value.im);
            if absolute.is_finite() && absolute < 1.0 {
                Ok(absolute)
            } else {
                Err(AnalysisError::InvalidRobustLeverage { time })
            }
        })
        .collect()
}

fn residuals(design: &Mat<f64>, observations: &Mat<f64>, coefficients: &Mat<f64>) -> Vec<[f64; 2]> {
    (0..design.nrows())
        .map(|time| {
            let fitted = |component| {
                (0..design.ncols())
                    .map(|column| design[(time, column)] * coefficients[(column, component)])
                    .sum::<f64>()
            };
            [
                observations[(time, 0)] - fitted(0),
                if observations.ncols() == 2 {
                    observations[(time, 1)] - fitted(1)
                } else {
                    0.0
                },
            ]
        })
        .collect()
}

fn complex_residuals(
    design: &Mat<c64>,
    observations: &Mat<c64>,
    coefficients: &Mat<c64>,
) -> Vec<c64> {
    (0..design.nrows())
        .map(|time| {
            let fitted = (0..design.ncols())
                .map(|column| design[(time, column)] * coefficients[(column, 0)])
                .sum::<c64>();
            observations[(time, 0)] - fitted
        })
        .collect()
}

fn residual_scale(residuals: &[[f64; 2]], component_count: usize) -> f64 {
    let center = if component_count == 1 {
        [
            median(residuals.iter().map(|value| value[0]).collect()),
            0.0,
        ]
    } else {
        median_complex(residuals)
    };
    median(
        residuals
            .iter()
            .map(|value| (value[0] - center[0]).hypot(value[1] - center[1]))
            .collect(),
    ) / MAD_NORMALIZATION
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[middle - 1].midpoint(values[middle])
    } else {
        values[middle]
    }
}

fn median_complex(values: &[[f64; 2]]) -> [f64; 2] {
    let mut values = values.to_vec();
    values.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
    });
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        [
            values[middle - 1][0].midpoint(values[middle][0]),
            values[middle - 1][1].midpoint(values[middle][1]),
        ]
    } else {
        values[middle]
    }
}

fn is_numerically_exact(residuals: &[[f64; 2]], observations: &Mat<f64>) -> bool {
    let observation_scale = observations
        .col_iter()
        .flat_map(|column| column.iter().copied())
        .map(f64::abs)
        .fold(1.0, f64::max);
    let tolerance = 64.0 * f64::EPSILON * observation_scale * usize_to_f64(observations.nrows());
    residuals
        .iter()
        .all(|residual| residual[0].hypot(residual[1]) <= tolerance)
}

fn complex_is_numerically_exact(residuals: &[c64], observations: &Mat<c64>) -> bool {
    let observation_scale = observations
        .col_iter()
        .flat_map(|column| column.iter().copied())
        .map(|value| value.re.hypot(value.im))
        .fold(1.0, f64::max);
    let tolerance = 64.0 * f64::EPSILON * observation_scale * usize_to_f64(observations.nrows());
    residuals
        .iter()
        .all(|residual| residual.re.hypot(residual.im) <= tolerance)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "practical record lengths are exactly representable as f64"
)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::{RobustOptions, RobustTermination, fit, usize_to_f64};
    use crate::AnalysisError;
    use faer::Mat;

    #[test]
    fn exact_fit_returns_ols_with_uniform_weights() {
        let design = Mat::from_fn(
            9,
            2,
            |row, column| {
                if column == 0 { 1.0 } else { usize_to_f64(row) }
            },
        );
        let observations = Mat::from_fn(9, 1, |row, _| 2.0 + 0.25 * usize_to_f64(row));
        let result = fit(&design, &observations, RobustOptions::default()).expect("exact fit");
        assert_eq!(result.diagnostics.termination, RobustTermination::ExactFit);
        assert_eq!(result.diagnostics.iterations, 1);
        assert!(
            result
                .diagnostics
                .weights
                .iter()
                .all(|weight| (*weight - 1.0).abs() <= f64::EPSILON)
        );
        assert!((result.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((result.coefficients[(1, 0)] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_options_and_iteration_exhaustion() {
        let design = Mat::from_fn(
            9,
            2,
            |row, column| {
                if column == 0 { 1.0 } else { usize_to_f64(row) }
            },
        );
        let observations = Mat::from_fn(9, 1, |row, _| {
            2.0 + 0.25 * usize_to_f64(row) + usize_to_f64(row).sin() * 0.1
        });
        assert!(matches!(
            fit(
                &design,
                &observations,
                RobustOptions {
                    tuning_constant: 0.0,
                    ..RobustOptions::default()
                }
            ),
            Err(AnalysisError::InvalidRobustTuningConstant)
        ));
        assert!(matches!(
            fit(
                &design,
                &observations,
                RobustOptions {
                    tolerance: f64::NAN,
                    ..RobustOptions::default()
                }
            ),
            Err(AnalysisError::InvalidRobustTolerance)
        ));
        assert!(matches!(
            fit(
                &design,
                &observations,
                RobustOptions {
                    max_iterations: 1,
                    ..RobustOptions::default()
                }
            ),
            Err(AnalysisError::RobustDidNotConverge { iterations: 1 })
        ));
    }

    #[test]
    fn rejects_zero_mad_for_a_nonexact_majority_cluster() {
        let design = Mat::from_fn(7, 1, |_, _| 1.0);
        let values = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let observations = Mat::from_fn(7, 1, |row, _| values[row]);
        assert!(matches!(
            fit(&design, &observations, RobustOptions::default()),
            Err(AnalysisError::DegenerateRobustScale)
        ));
    }
}
