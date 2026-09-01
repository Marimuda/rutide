//! Deterministic small-matrix sampling for nonlinear confidence intervals.

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::{AnalysisError, vector::ellipse_parameters};

const MAD_NORMALIZATION: f64 = 0.6745;
const CONFIDENCE_SCALE: f64 = 1.96;

/// Configuration for reproducible Monte Carlo confidence intervals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MonteCarloOptions {
    /// Number of coefficient realizations drawn for every constituent.
    pub realizations: usize,
    /// Root seed. Series and constituents receive deterministic derived streams.
    pub seed: u64,
}

impl Default for MonteCarloOptions {
    fn default() -> Self {
        Self {
            realizations: 200,
            seed: 0,
        }
    }
}

impl MonteCarloOptions {
    pub(crate) fn validate(self) -> Result<(), AnalysisError> {
        if self.realizations < 2 {
            return Err(AnalysisError::InvalidMonteCarloRealizationCount);
        }
        Ok(())
    }
}

pub(crate) struct ScalarMonteCarloIntervals {
    pub(crate) amplitude: f64,
    pub(crate) phase_degrees: f64,
}

pub(crate) struct VectorMonteCarloIntervals {
    pub(crate) semi_major: f64,
    pub(crate) semi_minor: f64,
    pub(crate) inclination_degrees: f64,
    pub(crate) phase_degrees: f64,
}

pub(crate) fn scalar_intervals(
    coefficients: [f64; 2],
    covariance: [[f64; 2]; 2],
    options: MonteCarloOptions,
    stream: u64,
) -> Option<ScalarMonteCarloIntervals> {
    let samples = multivariate_samples(coefficients, covariance, options, stream)?;
    let mut amplitude = Vec::with_capacity(options.realizations);
    let mut phase = Vec::with_capacity(options.realizations);
    for [cosine, sine] in samples {
        amplitude.push(cosine.hypot(sine));
        phase.push(sine.atan2(cosine).to_degrees().rem_euclid(360.0));
    }
    phase[0] = coefficients[1]
        .atan2(coefficients[0])
        .to_degrees()
        .rem_euclid(360.0);
    cluster_degrees(&mut phase, 360.0);
    Some(ScalarMonteCarloIntervals {
        amplitude: confidence_mad(&amplitude),
        phase_degrees: confidence_mad(&phase),
    })
}

pub(crate) fn vector_intervals(
    coefficients: [f64; 4],
    covariance: [[f64; 4]; 4],
    options: MonteCarloOptions,
    stream: u64,
) -> Option<VectorMonteCarloIntervals> {
    let samples = multivariate_samples(coefficients, covariance, options, stream)?;
    let mut semi_major = Vec::with_capacity(options.realizations);
    let mut semi_minor = Vec::with_capacity(options.realizations);
    let mut inclination = Vec::with_capacity(options.realizations);
    let mut phase = Vec::with_capacity(options.realizations);
    for [
        eastward_cosine,
        eastward_sine,
        northward_cosine,
        northward_sine,
    ] in samples
    {
        let ellipse = ellipse_parameters(
            eastward_cosine,
            eastward_sine,
            northward_cosine,
            northward_sine,
        );
        semi_major.push(ellipse.0);
        semi_minor.push(ellipse.1);
        inclination.push(ellipse.2);
        phase.push(ellipse.3);
    }
    let fitted = ellipse_parameters(
        coefficients[0],
        coefficients[1],
        coefficients[2],
        coefficients[3],
    );
    inclination[0] = fitted.2;
    phase[0] = fitted.3;
    cluster_degrees(&mut inclination, 360.0);
    cluster_degrees(&mut phase, 360.0);
    Some(VectorMonteCarloIntervals {
        semi_major: confidence_mad(&semi_major),
        semi_minor: confidence_mad(&semi_minor),
        inclination_degrees: confidence_mad(&inclination),
        phase_degrees: confidence_mad(&phase),
    })
}

fn multivariate_samples<const DIMENSION: usize>(
    mean: [f64; DIMENSION],
    covariance: [[f64; DIMENSION]; DIMENSION],
    options: MonteCarloOptions,
    stream: u64,
) -> Option<Vec<[f64; DIMENSION]>> {
    let covariance = nearest_positive_definite(covariance)?;
    let cholesky = cholesky(covariance)?;
    let mut random = ChaCha12Rng::seed_from_u64(derived_seed(options.seed, stream));
    let normal = StandardNormal;
    let mut output = Vec::with_capacity(options.realizations);
    for _ in 0..options.realizations {
        let independent: [f64; DIMENSION] = std::array::from_fn(|_| normal.sample(&mut random));
        output.push(std::array::from_fn(|row| {
            mean[row]
                + (0..=row)
                    .map(|column| cholesky[row][column] * independent[column])
                    .sum::<f64>()
        }));
    }
    Some(output)
}

fn nearest_positive_definite<const DIMENSION: usize>(
    covariance: [[f64; DIMENSION]; DIMENSION],
) -> Option<[[f64; DIMENSION]; DIMENSION]> {
    if covariance.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    let symmetric = std::array::from_fn(|row| {
        std::array::from_fn(|column| covariance[row][column].midpoint(covariance[column][row]))
    });
    if cholesky(symmetric).is_some() {
        return Some(symmetric);
    }

    let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(symmetric);
    let mut repaired = [[0.0; DIMENSION]; DIMENSION];
    for row in 0..DIMENSION {
        for column in 0..DIMENSION {
            repaired[row][column] = (0..DIMENSION)
                .map(|index| {
                    eigenvectors[row][index]
                        * eigenvalues[index].max(0.0)
                        * eigenvectors[column][index]
                })
                .sum::<f64>();
        }
    }
    repaired = std::array::from_fn(|row| {
        std::array::from_fn(|column| repaired[row][column].midpoint(repaired[column][row]))
    });
    let maximum = eigenvalues.into_iter().map(f64::abs).fold(0.0, f64::max);
    let increment = (maximum.next_up() - maximum).max(f64::MIN_POSITIVE);
    for multiplier in 1_u32..=100 {
        let mut candidate = repaired;
        for (index, row) in candidate.iter_mut().enumerate() {
            row[index] += increment * f64::from(multiplier);
        }
        if cholesky(candidate).is_some() {
            return Some(candidate);
        }
    }
    None
}

fn symmetric_eigendecomposition<const DIMENSION: usize>(
    mut matrix: [[f64; DIMENSION]; DIMENSION],
) -> ([f64; DIMENSION], [[f64; DIMENSION]; DIMENSION]) {
    let mut vectors = [[0.0; DIMENSION]; DIMENSION];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..(64 * DIMENSION * DIMENSION) {
        let mut pivot = (0, 0);
        let mut largest = 0.0;
        for (row, values) in matrix.iter().enumerate() {
            for (column, value) in values.iter().enumerate().skip(row + 1) {
                if value.abs() > largest {
                    largest = value.abs();
                    pivot = (row, column);
                }
            }
        }
        let scale = matrix
            .iter()
            .enumerate()
            .map(|(index, row)| row[index].abs())
            .fold(1.0, f64::max);
        if largest <= 32.0 * f64::EPSILON * scale {
            break;
        }
        let (left, right) = pivot;
        let angle =
            0.5 * (2.0 * matrix[left][right]).atan2(matrix[right][right] - matrix[left][left]);
        let (sine, cosine) = angle.sin_cos();
        let left_diagonal = matrix[left][left];
        let right_diagonal = matrix[right][right];
        let off_diagonal = matrix[left][right];
        matrix[left][left] = cosine * cosine * left_diagonal - 2.0 * sine * cosine * off_diagonal
            + sine * sine * right_diagonal;
        matrix[right][right] = sine * sine * left_diagonal
            + 2.0 * sine * cosine * off_diagonal
            + cosine * cosine * right_diagonal;
        matrix[left][right] = 0.0;
        matrix[right][left] = 0.0;
        for index in 0..DIMENSION {
            if index != left && index != right {
                let old_left = matrix[index][left];
                let old_right = matrix[index][right];
                matrix[index][left] = cosine * old_left - sine * old_right;
                matrix[left][index] = matrix[index][left];
                matrix[index][right] = sine * old_left + cosine * old_right;
                matrix[right][index] = matrix[index][right];
            }
            let vector_left = vectors[index][left];
            let vector_right = vectors[index][right];
            vectors[index][left] = cosine * vector_left - sine * vector_right;
            vectors[index][right] = sine * vector_left + cosine * vector_right;
        }
    }
    (std::array::from_fn(|index| matrix[index][index]), vectors)
}

fn cholesky<const DIMENSION: usize>(
    matrix: [[f64; DIMENSION]; DIMENSION],
) -> Option<[[f64; DIMENSION]; DIMENSION]> {
    let mut output = [[0.0; DIMENSION]; DIMENSION];
    for row in 0..DIMENSION {
        for column in 0..=row {
            let residual = matrix[row][column]
                - (0..column)
                    .map(|index| output[row][index] * output[column][index])
                    .sum::<f64>();
            if row == column {
                if !residual.is_finite() || residual <= 0.0 {
                    return None;
                }
                output[row][column] = residual.sqrt();
            } else {
                output[row][column] = residual / output[column][column];
            }
        }
    }
    Some(output)
}

fn cluster_degrees(values: &mut [f64], period: f64) {
    let center = values[0];
    for value in values {
        *value = center + (*value - center + period / 2.0).rem_euclid(period) - period / 2.0;
    }
}

fn confidence_mad(values: &[f64]) -> f64 {
    let center = median(values.to_vec());
    CONFIDENCE_SCALE * median(values.iter().map(|value| (*value - center).abs()).collect())
        / MAD_NORMALIZATION
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

fn derived_seed(seed: u64, stream: u64) -> u64 {
    let mut value = seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{
        MonteCarloOptions, cholesky, multivariate_samples, nearest_positive_definite,
        scalar_intervals, vector_intervals,
    };

    #[test]
    fn repairs_a_positive_semidefinite_covariance() {
        let repaired = nearest_positive_definite([[1.0, 1.0], [1.0, 1.0]])
            .expect("semidefinite covariance is repairable");
        assert!(cholesky(repaired).is_some());
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the seeded generator must be bit-for-bit reproducible for an identical stream"
    )]
    fn sampling_is_seeded_and_stream_specific() {
        let options = MonteCarloOptions {
            realizations: 200,
            seed: 42,
        };
        let covariance = [[0.04, 0.01], [0.01, 0.09]];
        let first = scalar_intervals([1.0, 0.5], covariance, options, 7).expect("valid covariance");
        let repeated =
            scalar_intervals([1.0, 0.5], covariance, options, 7).expect("valid covariance");
        let other = scalar_intervals([1.0, 0.5], covariance, options, 8).expect("valid covariance");
        assert_eq!(first.amplitude, repeated.amplitude);
        assert_eq!(first.phase_degrees, repeated.phase_degrees);
        assert_ne!(first.amplitude, other.amplitude);
    }

    #[test]
    fn vector_sampling_preserves_cross_component_covariance() {
        let covariance = [
            [0.16, 0.02, 0.08, -0.01],
            [0.02, 0.09, 0.01, 0.03],
            [0.08, 0.01, 0.25, -0.04],
            [-0.01, 0.03, -0.04, 0.12],
        ];
        let samples = multivariate_samples(
            [0.0; 4],
            covariance,
            MonteCarloOptions {
                realizations: 100_000,
                seed: 11,
            },
            5,
        )
        .expect("valid covariance");
        let sample_covariance = |left: usize, right: usize| {
            samples
                .iter()
                .map(|sample| sample[left] * sample[right])
                .sum::<f64>()
                / f64::from(u32::try_from(samples.len()).expect("fixture count fits u32"))
        };
        for (left, right) in [(0, 0), (0, 2), (1, 3), (2, 3)] {
            assert!(
                (sample_covariance(left, right) - covariance[left][right]).abs() < 0.003,
                "sample covariance ({left}, {right}) differs from its target"
            );
        }
    }

    #[test]
    fn near_degenerate_ellipse_and_indefinite_covariance_remain_finite() {
        let repaired = nearest_positive_definite([[1.0, 2.0], [2.0, 1.0]])
            .expect("indefinite covariance is repairable");
        assert!(cholesky(repaired).is_some());

        let intervals = vector_intervals(
            [1.0, 0.0, 0.0, 1.0 - 1e-12],
            [
                [0.02, 0.0, 0.01, 0.0],
                [0.0, 0.02, 0.0, 0.01],
                [0.01, 0.0, 0.02, 0.0],
                [0.0, 0.01, 0.0, 0.02],
            ],
            MonteCarloOptions {
                realizations: 2_000,
                seed: 17,
            },
            0,
        )
        .expect("near-degenerate ellipse covariance is sampleable");
        assert!(intervals.semi_major.is_finite());
        assert!(intervals.semi_minor.is_finite());
        assert!(intervals.inclination_degrees.is_finite());
        assert!(intervals.phase_degrees.is_finite());
    }
}
