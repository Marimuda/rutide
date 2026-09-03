//! Constituent-identifiability and reconstructed-fit diagnostics.
//!
//! The definitions in this module follow Codiga (2011), section II.D. Pairwise
//! diagnostics use the nearest directly modeled constituents in frequency;
//! inferred constituents are reported separately and do not participate in the
//! neighbor graph.

use crate::{
    error::AnalysisError,
    scalar::{Constituent, validate_constituents},
};
use faer::Mat;

/// How a fitted constituent participates in independence diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticConstituentRole {
    /// A non-reference or reference constituent represented by model parameters.
    Direct,
    /// A constituent constrained from a directly fitted reference constituent.
    Inferred,
}

/// Diagnostics comparing one constituent with an adjacent-frequency neighbor.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighboringConstituentDiagnostics {
    /// Position of the neighboring constituent in the prepared output order.
    pub index: usize,
    /// Name of the neighboring constituent.
    pub name: String,
    /// Frequency of the neighboring constituent in cycles per hour.
    pub frequency_cph: f64,
    /// Conventional Rayleigh criterion from Codiga (2011), equation 81.
    pub rayleigh_criterion: f64,
    /// Noise-modified Rayleigh criterion, populated when SNR is available.
    pub noise_modified_rayleigh_criterion: Option<f64>,
    /// Maximum coefficient correlation, populated when covariance is available.
    pub maximum_correlation: Option<f64>,
}

/// Lower- and higher-frequency independence diagnostics for one constituent.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstituentIndependenceDiagnostics {
    /// Nearest directly modeled constituent at a lower frequency, if one exists.
    pub lower: Option<NeighboringConstituentDiagnostics>,
    /// Nearest directly modeled constituent at a higher frequency, if one exists.
    pub higher: Option<NeighboringConstituentDiagnostics>,
}

/// Mean-square tidal variance captured by complete and significant-constituent fits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TidalVarianceDiagnostics {
    /// Mean-square raw signal after removal of the fitted mean and trend.
    pub raw_tidal_variance: f64,
    /// Mean-square reconstruction containing every fitted constituent.
    pub all_constituent_tidal_variance: f64,
    /// Mean-square reconstruction containing only significant constituents.
    pub significant_constituent_tidal_variance: Option<f64>,
    /// Percentage of raw tidal variance captured by all constituents.
    ///
    /// This is `None` when the detrended raw input has zero variance.
    pub all_constituent_percent_tidal_variance: Option<f64>,
    /// Percentage of raw tidal variance captured by significant constituents.
    ///
    /// This is `None` when no significant subset was supplied or the detrended
    /// raw input has zero variance.
    pub significant_constituent_percent_tidal_variance: Option<f64>,
}

/// Whole-model independence diagnostics from Codiga (2011), equations 84–86.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WholeModelIndependenceDiagnostics {
    /// Two-norm condition number of the complete fitted basis, conventionally `K`.
    pub basis_condition_number: f64,
    /// Energy SNR of the complete model, conventionally `SNRallc`.
    ///
    /// This is `None` when both model energy and residual mean square are zero.
    pub all_constituent_signal_to_noise: Option<f64>,
    /// `SNRallc / K`, the report's whole-model error-bound diagnostic.
    ///
    /// This is `None` for an indeterminate infinite-over-infinite ratio or when
    /// `all_constituent_signal_to_noise` is undefined.
    pub condition_adjusted_signal_to_noise: Option<f64>,
}

/// Configuration for the extended constituent-selection diagnostic suite.
///
/// Defaults reproduce MATLAB `UTide`: `Rmin = 1` and an SNR significance
/// threshold of 2. Fields are private so the suite can grow compatibly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstituentDiagnosticsOptions {
    rayleigh_minimum: f64,
    minimum_signal_to_noise: f64,
}

impl Default for ConstituentDiagnosticsOptions {
    fn default() -> Self {
        Self {
            rayleigh_minimum: 1.0,
            minimum_signal_to_noise: 2.0,
        }
    }
}

impl ConstituentDiagnosticsOptions {
    /// Set the denominator of the conventional Rayleigh criterion.
    #[must_use]
    pub const fn with_rayleigh_minimum(mut self, rayleigh_minimum: f64) -> Self {
        self.rayleigh_minimum = rayleigh_minimum;
        self
    }

    /// Set the inclusive SNR threshold for significant-subset tidal variance.
    #[must_use]
    pub const fn with_minimum_signal_to_noise(mut self, minimum_signal_to_noise: f64) -> Self {
        self.minimum_signal_to_noise = minimum_signal_to_noise;
        self
    }

    /// Return the Rayleigh criterion denominator.
    #[must_use]
    pub const fn rayleigh_minimum(self) -> f64 {
        self.rayleigh_minimum
    }

    /// Return the inclusive significant-constituent SNR threshold.
    #[must_use]
    pub const fn minimum_signal_to_noise(self) -> f64 {
        self.minimum_signal_to_noise
    }

    pub(crate) fn validate(self) -> Result<(), AnalysisError> {
        if !self.rayleigh_minimum.is_finite() || self.rayleigh_minimum <= 0.0 {
            return Err(AnalysisError::InvalidRayleighMinimum);
        }
        if !self.minimum_signal_to_noise.is_finite() || self.minimum_signal_to_noise < 0.0 {
            return Err(AnalysisError::InvalidDiagnosticThreshold {
                diagnostic: "signal_to_noise",
            });
        }
        Ok(())
    }
}

/// Extended evidence for constituent identifiability and captured tidal variance.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstituentSelectionDiagnostics {
    /// Lower/higher directly modeled neighbor diagnostics in solution order.
    pub constituents: Vec<ConstituentIndependenceDiagnostics>,
    /// Condition-number and complete-model SNR diagnostics.
    pub whole_model: WholeModelIndependenceDiagnostics,
    /// Raw, complete-fit, and SNR-filtered reconstructed tidal variance.
    pub tidal_variance: TidalVarianceDiagnostics,
    /// Rayleigh denominator used for this calculation.
    pub rayleigh_minimum: f64,
    /// Inclusive SNR threshold used for the significant reconstruction.
    pub minimum_signal_to_noise: f64,
}

/// Calculate conventional Rayleigh diagnostics for adjacent fitted frequencies.
///
/// Results remain aligned with `constituents`. Directly modeled constituents are
/// sorted by frequency only to construct their neighbor relationships. Inferred
/// constituents receive no neighbors, matching the original MATLAB `UTide` table.
/// The criterion is
/// `24 * effective_record_length_days * frequency_separation_cph / rayleigh_min`.
///
/// # Errors
///
/// Returns [`AnalysisError`] for invalid constituents, a role-count mismatch, or
/// a non-finite/non-positive effective record length or Rayleigh minimum.
pub fn adjacent_constituent_diagnostics(
    constituents: &[Constituent],
    roles: &[DiagnosticConstituentRole],
    effective_record_length_days: f64,
    rayleigh_min: f64,
) -> Result<Vec<ConstituentIndependenceDiagnostics>, AnalysisError> {
    validate_constituents(constituents)?;
    if roles.len() != constituents.len() {
        return Err(AnalysisError::InvalidSolutionShape {
            field: "diagnostic_constituent_roles",
            actual: roles.len(),
            expected: constituents.len(),
        });
    }
    if !effective_record_length_days.is_finite() || effective_record_length_days <= 0.0 {
        return Err(AnalysisError::InvalidDiagnosticEffectiveRecordLength);
    }
    if !rayleigh_min.is_finite() || rayleigh_min <= 0.0 {
        return Err(AnalysisError::InvalidRayleighMinimum);
    }

    let mut direct_indices = roles
        .iter()
        .enumerate()
        .filter_map(|(index, role)| (*role == DiagnosticConstituentRole::Direct).then_some(index))
        .collect::<Vec<_>>();
    direct_indices.sort_by(|left, right| {
        constituents[*left]
            .frequency_cph
            .total_cmp(&constituents[*right].frequency_cph)
            .then_with(|| left.cmp(right))
    });

    let mut output = vec![
        ConstituentIndependenceDiagnostics {
            lower: None,
            higher: None,
        };
        constituents.len()
    ];
    for pair in direct_indices.windows(2) {
        let lower = pair[0];
        let higher = pair[1];
        let separation = constituents[higher].frequency_cph - constituents[lower].frequency_cph;
        let rayleigh_criterion = 24.0 * effective_record_length_days * separation / rayleigh_min;
        output[lower].higher = Some(neighbor(higher, &constituents[higher], rayleigh_criterion));
        output[higher].lower = Some(neighbor(lower, &constituents[lower], rayleigh_criterion));
    }
    Ok(output)
}

/// Calculate Codiga scalar tidal-variance diagnostics from detrended signals.
///
/// Each input must contain only its tidal component: the fitted mean and trend
/// must already have been removed. `significant_constituents_tidal` is normally
/// reconstructed from constituents meeting a configured SNR threshold.
///
/// # Errors
///
/// Returns [`AnalysisError`] for empty input, inconsistent lengths, or non-finite
/// values.
pub fn scalar_tidal_variance_diagnostics(
    raw_tidal: &[f64],
    all_constituents_tidal: &[f64],
    significant_constituents_tidal: Option<&[f64]>,
) -> Result<TidalVarianceDiagnostics, AnalysisError> {
    validate_tidal_values("raw_tidal", raw_tidal, raw_tidal.len())?;
    if raw_tidal.is_empty() {
        return Err(AnalysisError::EmptySeries);
    }
    validate_tidal_values(
        "all_constituents_tidal",
        all_constituents_tidal,
        raw_tidal.len(),
    )?;
    if let Some(values) = significant_constituents_tidal {
        validate_tidal_values("significant_constituents_tidal", values, raw_tidal.len())?;
    }

    let raw = mean_square_scalar(raw_tidal);
    let all = mean_square_scalar(all_constituents_tidal);
    let significant = significant_constituents_tidal.map(mean_square_scalar);
    Ok(tidal_variance_diagnostics(raw, all, significant))
}

/// Calculate Codiga vector tidal-variance diagnostics from detrended components.
///
/// Tidal variance is the time mean of `eastward.powi(2) + northward.powi(2)`.
/// Every component must have its fitted mean and trend removed before this call.
///
/// # Errors
///
/// Returns [`AnalysisError`] for empty input, inconsistent lengths, an incomplete
/// significant-component pair, or non-finite values.
#[allow(
    clippy::too_many_arguments,
    reason = "raw, complete-fit, and optional significant-fit vector components are explicit"
)]
pub fn vector_tidal_variance_diagnostics(
    raw_eastward_tidal: &[f64],
    raw_northward_tidal: &[f64],
    all_constituents_eastward_tidal: &[f64],
    all_constituents_northward_tidal: &[f64],
    significant_constituents_eastward_tidal: Option<&[f64]>,
    significant_constituents_northward_tidal: Option<&[f64]>,
) -> Result<TidalVarianceDiagnostics, AnalysisError> {
    let time_count = raw_eastward_tidal.len();
    if time_count == 0 {
        return Err(AnalysisError::EmptySeries);
    }
    for (field, values) in [
        ("raw_eastward_tidal", raw_eastward_tidal),
        ("raw_northward_tidal", raw_northward_tidal),
        (
            "all_constituents_eastward_tidal",
            all_constituents_eastward_tidal,
        ),
        (
            "all_constituents_northward_tidal",
            all_constituents_northward_tidal,
        ),
    ] {
        validate_tidal_values(field, values, time_count)?;
    }
    let significant = match (
        significant_constituents_eastward_tidal,
        significant_constituents_northward_tidal,
    ) {
        (Some(eastward), Some(northward)) => {
            validate_tidal_values(
                "significant_constituents_eastward_tidal",
                eastward,
                time_count,
            )?;
            validate_tidal_values(
                "significant_constituents_northward_tidal",
                northward,
                time_count,
            )?;
            Some(mean_square_vector(eastward, northward))
        }
        (None, None) => None,
        (Some(_), None) => {
            return Err(AnalysisError::InvalidSolutionShape {
                field: "significant_constituents_northward_tidal",
                actual: 0,
                expected: time_count,
            });
        }
        (None, Some(_)) => {
            return Err(AnalysisError::InvalidSolutionShape {
                field: "significant_constituents_eastward_tidal",
                actual: 0,
                expected: time_count,
            });
        }
    };

    Ok(tidal_variance_diagnostics(
        mean_square_vector(raw_eastward_tidal, raw_northward_tidal),
        mean_square_vector(
            all_constituents_eastward_tidal,
            all_constituents_northward_tidal,
        ),
        significant,
    ))
}

pub(crate) fn populate_noise_modified_rayleigh(
    diagnostics: &mut [ConstituentIndependenceDiagnostics],
    signal_to_noise: &[f64],
) -> Result<(), AnalysisError> {
    if signal_to_noise.len() != diagnostics.len() {
        return Err(AnalysisError::InvalidSolutionShape {
            field: "signal_to_noise",
            actual: signal_to_noise.len(),
            expected: diagnostics.len(),
        });
    }
    if let Some((index, _)) = signal_to_noise
        .iter()
        .enumerate()
        .find(|(_, value)| value.is_nan() || **value < 0.0)
    {
        return Err(AnalysisError::InvalidDiagnosticSignalToNoise { index });
    }

    for (constituent, independence) in diagnostics.iter_mut().enumerate() {
        for neighbor in [&mut independence.lower, &mut independence.higher]
            .into_iter()
            .flatten()
        {
            let mean_signal_to_noise =
                signal_to_noise[constituent].midpoint(signal_to_noise[neighbor.index]);
            neighbor.noise_modified_rayleigh_criterion =
                Some(neighbor.rayleigh_criterion * mean_signal_to_noise.sqrt());
        }
    }
    Ok(())
}

pub(crate) fn populate_maximum_correlation(
    diagnostics: &mut [ConstituentIndependenceDiagnostics],
    coefficient_covariance: &Mat<f64>,
) -> Result<(), AnalysisError> {
    let harmonic_parameter_count = diagnostics.len().saturating_mul(2);
    if coefficient_covariance.nrows() < harmonic_parameter_count {
        return Err(AnalysisError::InvalidSolutionShape {
            field: "diagnostic_coefficient_covariance_rows",
            actual: coefficient_covariance.nrows(),
            expected: harmonic_parameter_count,
        });
    }
    if coefficient_covariance.ncols() < harmonic_parameter_count {
        return Err(AnalysisError::InvalidSolutionShape {
            field: "diagnostic_coefficient_covariance_columns",
            actual: coefficient_covariance.ncols(),
            expected: harmonic_parameter_count,
        });
    }
    for parameter in 0..harmonic_parameter_count {
        let variance = coefficient_covariance[(parameter, parameter)];
        if !variance.is_finite() || variance <= 0.0 {
            return Err(AnalysisError::InvalidDiagnosticCoefficientVariance { parameter });
        }
    }

    for (constituent, independence) in diagnostics.iter_mut().enumerate() {
        for neighbor in [&mut independence.lower, &mut independence.higher]
            .into_iter()
            .flatten()
        {
            let mut maximum: f64 = 0.0;
            for left_offset in 0..2 {
                let left = constituent * 2 + left_offset;
                for right_offset in 0..2 {
                    let right = neighbor.index * 2 + right_offset;
                    let denominator = (coefficient_covariance[(left, left)]
                        * coefficient_covariance[(right, right)])
                        .sqrt();
                    let correlation = coefficient_covariance[(left, right)] / denominator;
                    if !correlation.is_finite() {
                        return Err(AnalysisError::InvalidDiagnosticCoefficientCorrelation {
                            left,
                            right,
                        });
                    }
                    maximum = maximum.max(correlation.abs());
                }
            }
            neighbor.maximum_correlation = Some(maximum);
        }
    }
    Ok(())
}

pub(crate) fn whole_model_independence_diagnostics(
    basis_condition_number: f64,
    model_energy: f64,
    residual_mean_square: f64,
) -> Result<WholeModelIndependenceDiagnostics, AnalysisError> {
    if basis_condition_number.is_nan() || basis_condition_number <= 0.0 {
        return Err(AnalysisError::InvalidDiagnosticBasisConditionNumber);
    }
    for (field, value) in [
        ("diagnostic_model_energy", model_energy),
        ("diagnostic_residual_mean_square", residual_mean_square),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(AnalysisError::InvalidDiagnosticEnergy { field });
        }
    }

    let all_constituent_signal_to_noise = match (model_energy, residual_mean_square) {
        (0.0, 0.0) => None,
        (_, 0.0) => Some(f64::INFINITY),
        _ => Some(model_energy / residual_mean_square),
    };
    let condition_adjusted_signal_to_noise = all_constituent_signal_to_noise.and_then(|snr| {
        if snr.is_infinite() && basis_condition_number.is_infinite() {
            None
        } else {
            Some(snr / basis_condition_number)
        }
    });
    Ok(WholeModelIndependenceDiagnostics {
        basis_condition_number,
        all_constituent_signal_to_noise,
        condition_adjusted_signal_to_noise,
    })
}

fn neighbor(
    index: usize,
    constituent: &Constituent,
    rayleigh_criterion: f64,
) -> NeighboringConstituentDiagnostics {
    NeighboringConstituentDiagnostics {
        index,
        name: constituent.name.clone(),
        frequency_cph: constituent.frequency_cph,
        rayleigh_criterion,
        noise_modified_rayleigh_criterion: None,
        maximum_correlation: None,
    }
}

fn validate_tidal_values(
    field: &'static str,
    values: &[f64],
    expected: usize,
) -> Result<(), AnalysisError> {
    if values.len() != expected {
        return Err(AnalysisError::InvalidSolutionShape {
            field,
            actual: values.len(),
            expected,
        });
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(AnalysisError::NonFiniteDiagnosticValue { field, index });
    }
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "practical time-series lengths are represented exactly as f64"
)]
fn mean_square_scalar(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64
}

#[allow(
    clippy::cast_precision_loss,
    reason = "practical time-series lengths are represented exactly as f64"
)]
fn mean_square_vector(eastward: &[f64], northward: &[f64]) -> f64 {
    eastward
        .iter()
        .zip(northward)
        .map(|(eastward, northward)| eastward * eastward + northward * northward)
        .sum::<f64>()
        / eastward.len() as f64
}

pub(crate) fn tidal_variance_diagnostics(
    raw: f64,
    all: f64,
    significant: Option<f64>,
) -> TidalVarianceDiagnostics {
    let percent = |value| (raw > 0.0).then(|| 100.0 * value / raw);
    TidalVarianceDiagnostics {
        raw_tidal_variance: raw,
        all_constituent_tidal_variance: all,
        significant_constituent_tidal_variance: significant,
        all_constituent_percent_tidal_variance: percent(all),
        significant_constituent_percent_tidal_variance: significant.and_then(percent),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DiagnosticConstituentRole, adjacent_constituent_diagnostics, populate_maximum_correlation,
        populate_noise_modified_rayleigh, scalar_tidal_variance_diagnostics,
        vector_tidal_variance_diagnostics, whole_model_independence_diagnostics,
    };
    use crate::{AnalysisError, Constituent};
    use faer::Mat;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "actual {actual:?} differs from expected {expected:?}"
        );
    }

    #[test]
    fn rayleigh_neighbors_use_frequency_order_and_exclude_inference() {
        let constituents = [
            Constituent::new("high", 0.10),
            Constituent::new("low", 0.04),
            Constituent::new("middle", 0.05),
            Constituent::new("inferred", 0.06),
        ];
        let roles = [
            DiagnosticConstituentRole::Direct,
            DiagnosticConstituentRole::Direct,
            DiagnosticConstituentRole::Direct,
            DiagnosticConstituentRole::Inferred,
        ];
        let diagnostics = adjacent_constituent_diagnostics(&constituents, &roles, 10.0, 2.0)
            .expect("valid diagnostics");

        assert!(diagnostics[0].higher.is_none());
        let high_lower = diagnostics[0].lower.as_ref().expect("high lower neighbor");
        assert_eq!(high_lower.index, 2);
        assert_eq!(high_lower.name, "middle");
        assert!((high_lower.rayleigh_criterion - 6.0).abs() < 1e-12);

        assert!(diagnostics[1].lower.is_none());
        let low_higher = diagnostics[1].higher.as_ref().expect("low higher neighbor");
        assert_eq!(low_higher.index, 2);
        assert!((low_higher.rayleigh_criterion - 1.2).abs() < 1e-12);

        assert_eq!(
            diagnostics[2].lower.as_ref().map(|value| value.index),
            Some(1)
        );
        assert_eq!(
            diagnostics[2].higher.as_ref().map(|value| value.index),
            Some(0)
        );
        assert!(diagnostics[3].lower.is_none());
        assert!(diagnostics[3].higher.is_none());
        assert!(
            diagnostics
                .iter()
                .flat_map(|value| [value.lower.as_ref(), value.higher.as_ref()])
                .flatten()
                .all(
                    |neighbor| neighbor.noise_modified_rayleigh_criterion.is_none()
                        && neighbor.maximum_correlation.is_none()
                )
        );
    }

    #[test]
    fn rayleigh_diagnostics_reject_invalid_metadata() {
        let constituent = [Constituent::new("M2", 0.080_511_400_7)];
        assert_eq!(
            adjacent_constituent_diagnostics(&constituent, &[], 10.0, 1.0),
            Err(AnalysisError::InvalidSolutionShape {
                field: "diagnostic_constituent_roles",
                actual: 0,
                expected: 1,
            })
        );
        assert_eq!(
            adjacent_constituent_diagnostics(
                &constituent,
                &[DiagnosticConstituentRole::Direct],
                0.0,
                1.0,
            ),
            Err(AnalysisError::InvalidDiagnosticEffectiveRecordLength)
        );
    }

    #[test]
    fn scalar_tidal_variance_matches_codiga_mean_square_definition() {
        let diagnostics =
            scalar_tidal_variance_diagnostics(&[1.0, -1.0], &[0.5, -0.5], Some(&[0.25, -0.25]))
                .expect("valid scalar variance");
        assert_close(diagnostics.raw_tidal_variance, 1.0);
        assert_close(diagnostics.all_constituent_tidal_variance, 0.25);
        assert_eq!(
            diagnostics.significant_constituent_tidal_variance,
            Some(0.0625)
        );
        assert_eq!(
            diagnostics.all_constituent_percent_tidal_variance,
            Some(25.0)
        );
        assert_eq!(
            diagnostics.significant_constituent_percent_tidal_variance,
            Some(6.25)
        );
    }

    #[test]
    fn vector_tidal_variance_sums_component_energy() {
        let diagnostics = vector_tidal_variance_diagnostics(
            &[1.0, -1.0],
            &[2.0, -2.0],
            &[0.5, -0.5],
            &[1.0, -1.0],
            Some(&[0.25, -0.25]),
            Some(&[0.5, -0.5]),
        )
        .expect("valid vector variance");
        assert_close(diagnostics.raw_tidal_variance, 5.0);
        assert_close(diagnostics.all_constituent_tidal_variance, 1.25);
        assert_eq!(
            diagnostics.significant_constituent_tidal_variance,
            Some(0.3125)
        );
        assert_eq!(
            diagnostics.all_constituent_percent_tidal_variance,
            Some(25.0)
        );
        assert_eq!(
            diagnostics.significant_constituent_percent_tidal_variance,
            Some(6.25)
        );
    }

    #[test]
    fn zero_raw_variance_has_explicitly_undefined_percentages() {
        let diagnostics = scalar_tidal_variance_diagnostics(&[0.0, 0.0], &[0.0, 0.0], None)
            .expect("valid zero variance");
        assert_close(diagnostics.raw_tidal_variance, 0.0);
        assert_eq!(diagnostics.all_constituent_percent_tidal_variance, None);
        assert_eq!(
            diagnostics.significant_constituent_percent_tidal_variance,
            None
        );
    }

    #[test]
    fn tidal_variance_rejects_invalid_shapes_and_values() {
        assert_eq!(
            scalar_tidal_variance_diagnostics(&[1.0], &[], None),
            Err(AnalysisError::InvalidSolutionShape {
                field: "all_constituents_tidal",
                actual: 0,
                expected: 1,
            })
        );
        assert_eq!(
            scalar_tidal_variance_diagnostics(&[f64::NAN], &[0.0], None),
            Err(AnalysisError::NonFiniteDiagnosticValue {
                field: "raw_tidal",
                index: 0,
            })
        );
    }

    #[test]
    fn noise_modified_rayleigh_uses_adjacent_mean_snr() {
        let constituents = [
            Constituent::new("left", 0.04),
            Constituent::new("right", 0.05),
        ];
        let roles = [
            DiagnosticConstituentRole::Direct,
            DiagnosticConstituentRole::Direct,
        ];
        let mut diagnostics = adjacent_constituent_diagnostics(&constituents, &roles, 10.0, 1.0)
            .expect("valid neighbors");
        populate_noise_modified_rayleigh(&mut diagnostics, &[8.0, 10.0]).expect("valid SNR");

        assert_close(
            diagnostics[0]
                .higher
                .as_ref()
                .and_then(|neighbor| neighbor.noise_modified_rayleigh_criterion)
                .expect("higher RNM"),
            7.2,
        );
        assert_eq!(
            diagnostics[1]
                .lower
                .as_ref()
                .and_then(|neighbor| neighbor.noise_modified_rayleigh_criterion),
            diagnostics[0]
                .higher
                .as_ref()
                .and_then(|neighbor| neighbor.noise_modified_rayleigh_criterion)
        );
        assert_eq!(
            populate_noise_modified_rayleigh(&mut diagnostics, &[-1.0, 2.0]),
            Err(AnalysisError::InvalidDiagnosticSignalToNoise { index: 0 })
        );
    }

    #[test]
    fn maximum_correlation_scans_each_adjacent_parameter_block() {
        let constituents = [
            Constituent::new("left", 0.04),
            Constituent::new("right", 0.05),
        ];
        let roles = [
            DiagnosticConstituentRole::Direct,
            DiagnosticConstituentRole::Direct,
        ];
        let mut diagnostics = adjacent_constituent_diagnostics(&constituents, &roles, 10.0, 1.0)
            .expect("valid neighbors");
        let covariance = Mat::from_fn(4, 4, |row, column| match (row, column) {
            (0, 0) => 4.0,
            (1, 1) => 9.0,
            (2, 2) => 16.0,
            (3, 3) => 25.0,
            (0, 2) | (2, 0) => 6.0,
            (1, 3) | (3, 1) => -3.0,
            _ => 0.0,
        });
        populate_maximum_correlation(&mut diagnostics, &covariance)
            .expect("positive covariance diagonal");

        assert_eq!(
            diagnostics[0]
                .higher
                .as_ref()
                .and_then(|neighbor| neighbor.maximum_correlation),
            Some(0.75)
        );
        assert_eq!(
            diagnostics[1]
                .lower
                .as_ref()
                .and_then(|neighbor| neighbor.maximum_correlation),
            Some(0.75)
        );
    }

    #[test]
    fn whole_model_metrics_preserve_exact_and_undefined_limits() {
        let finite = whole_model_independence_diagnostics(12.0, 240.0, 2.0)
            .expect("valid whole-model diagnostics");
        assert_eq!(finite.all_constituent_signal_to_noise, Some(120.0));
        assert_eq!(finite.condition_adjusted_signal_to_noise, Some(10.0));

        let exact = whole_model_independence_diagnostics(2.0, 1.0, 0.0).expect("valid exact fit");
        assert_eq!(exact.all_constituent_signal_to_noise, Some(f64::INFINITY));
        assert_eq!(
            exact.condition_adjusted_signal_to_noise,
            Some(f64::INFINITY)
        );

        let undefined = whole_model_independence_diagnostics(f64::INFINITY, 0.0, 0.0)
            .expect("valid zero-energy fit");
        assert_eq!(undefined.all_constituent_signal_to_noise, None);
        assert_eq!(undefined.condition_adjusted_signal_to_noise, None);
    }
}
