//! Two-component tidal-current ellipse solutions.

use std::f64::consts::PI;

use crate::{AnalysisError, RobustDiagnostics, ScalarSolution};

/// Reconstruction-ready Cartesian coefficients for a two-component current.
///
/// Unlike ellipse phase and inclination, these coefficients are continuous
/// across angular wrap boundaries and can therefore form the numerical payload
/// of a spatial harmonic-current atlas.  Each harmonic component follows the
/// same convention as [`ScalarSolution`]:
///
/// `component = cosine_coefficient * cos(basis) + sine_coefficient * sin(basis)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CartesianVectorSolution {
    /// Eastward cosine coefficient for each constituent.
    pub eastward_cosine_coefficient: Vec<f64>,
    /// Eastward sine coefficient for each constituent.
    pub eastward_sine_coefficient: Vec<f64>,
    /// Northward cosine coefficient for each constituent.
    pub northward_cosine_coefficient: Vec<f64>,
    /// Northward sine coefficient for each constituent.
    pub northward_sine_coefficient: Vec<f64>,
    /// Percent of total resolved ellipse energy for each constituent.
    pub percent_energy: Vec<f64>,
    /// Ellipse-energy signal-to-noise ratio, when confidence was requested.
    pub signal_to_noise: Option<Vec<f64>>,
    /// Fitted eastward constant offset.
    pub eastward_mean: f64,
    /// Fitted northward constant offset.
    pub northward_mean: f64,
    /// Fitted eastward trend per day.
    pub eastward_slope_per_day: f64,
    /// Fitted northward trend per day.
    pub northward_slope_per_day: f64,
    /// Epoch at which both means are defined, in fit-day coordinates.
    pub reference_time_days: f64,
}

/// Harmonic current ellipses derived from joint eastward/northward OLS fits.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorSolution {
    /// Semi-major axis for each constituent.
    pub semi_major: Vec<f64>,
    /// Signed semi-minor axis for each constituent.
    pub semi_minor: Vec<f64>,
    /// Counter-clockwise major-axis orientation in `[0, 180)` degrees.
    pub inclination_degrees: Vec<f64>,
    /// Greenwich phase in `[0, 360)` degrees.
    pub phase_degrees: Vec<f64>,
    /// Percent of total resolved ellipse energy for each constituent.
    pub percent_energy: Vec<f64>,
    /// Linearized 95% semi-major confidence half-widths.
    pub semi_major_ci: Option<Vec<f64>>,
    /// Linearized 95% semi-minor confidence half-widths.
    pub semi_minor_ci: Option<Vec<f64>>,
    /// Linearized 95% inclination confidence half-widths in degrees.
    pub inclination_ci_degrees: Option<Vec<f64>>,
    /// Linearized 95% phase confidence half-widths in degrees.
    pub phase_ci_degrees: Option<Vec<f64>>,
    /// Ellipse-energy signal-to-noise ratio derived from axis confidence intervals.
    pub signal_to_noise: Option<Vec<f64>>,
    /// Fitted eastward constant offset.
    pub eastward_mean: f64,
    /// Fitted northward constant offset.
    pub northward_mean: f64,
    /// Fitted eastward trend per day.
    pub eastward_slope_per_day: f64,
    /// Fitted northward trend per day.
    pub northward_slope_per_day: f64,
    /// Epoch at which both means are defined, in fit-day coordinates.
    pub reference_time_days: f64,
    /// Shared iteration and weight diagnostics for a joint robust vector fit.
    pub robust: Option<RobustDiagnostics>,
}

/// Eastward and northward current reconstructed at a shared set of timestamps.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorReconstruction {
    /// Reconstructed eastward velocity in target-time order.
    pub eastward: Vec<f64>,
    /// Reconstructed northward velocity in target-time order.
    pub northward: Vec<f64>,
}

/// One instantaneous two-component current.
///
/// Batches of this compact type preserve caller order and avoid allocating two
/// one-value vectors per spatial query location.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorCurrent {
    /// Reconstructed eastward velocity.
    pub eastward: f64,
    /// Reconstructed northward velocity.
    pub northward: f64,
}

impl VectorSolution {
    /// Return constituent indices ranked by descending percent energy.
    #[must_use]
    pub fn constituent_indices_by_percent_energy(&self) -> Vec<usize> {
        descending_indices(&self.percent_energy)
    }

    /// Return constituent indices ranked by descending SNR, when available.
    #[must_use]
    pub fn constituent_indices_by_signal_to_noise(&self) -> Option<Vec<usize>> {
        self.signal_to_noise.as_deref().map(descending_indices)
    }

    /// Convert ellipse parameters into reconstruction-ready Cartesian coefficients.
    ///
    /// This representation should be preferred for spatial interpolation because
    /// it has no phase or inclination wrap discontinuity.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] if an ellipse or diagnostic array does not have
    /// the same constituent count as `semi_major`.
    pub fn cartesian(&self) -> Result<CartesianVectorSolution, AnalysisError> {
        let constituent_count = self.semi_major.len();
        for (field, actual) in [
            ("semi_minor", self.semi_minor.len()),
            ("inclination_degrees", self.inclination_degrees.len()),
            ("phase_degrees", self.phase_degrees.len()),
            ("percent_energy", self.percent_energy.len()),
        ] {
            if actual != constituent_count {
                return Err(AnalysisError::InvalidSolutionShape {
                    field,
                    actual,
                    expected: constituent_count,
                });
            }
        }
        if let Some(signal_to_noise) = &self.signal_to_noise
            && signal_to_noise.len() != constituent_count
        {
            return Err(AnalysisError::InvalidSolutionShape {
                field: "signal_to_noise",
                actual: signal_to_noise.len(),
                expected: constituent_count,
            });
        }

        let mut eastward_cosine = Vec::with_capacity(self.semi_major.len());
        let mut eastward_sine = Vec::with_capacity(self.semi_major.len());
        let mut northward_cosine = Vec::with_capacity(self.semi_major.len());
        let mut northward_sine = Vec::with_capacity(self.semi_major.len());
        for constituent in 0..self.semi_major.len() {
            let theta = self.inclination_degrees[constituent].to_radians();
            let phase = self.phase_degrees[constituent].to_radians();
            let positive_radius =
                self.semi_major[constituent].midpoint(self.semi_minor[constituent]);
            let negative_radius =
                0.5 * (self.semi_major[constituent] - self.semi_minor[constituent]);
            let positive_real = positive_radius * (theta - phase).cos();
            let positive_imaginary = positive_radius * (theta - phase).sin();
            let negative_real = negative_radius * (theta + phase).cos();
            let negative_imaginary = negative_radius * (theta + phase).sin();
            eastward_cosine.push(positive_real + negative_real);
            eastward_sine.push(-(positive_imaginary - negative_imaginary));
            northward_cosine.push(positive_imaginary + negative_imaginary);
            northward_sine.push(positive_real - negative_real);
        }
        Ok(CartesianVectorSolution {
            eastward_cosine_coefficient: eastward_cosine,
            eastward_sine_coefficient: eastward_sine,
            northward_cosine_coefficient: northward_cosine,
            northward_sine_coefficient: northward_sine,
            percent_energy: self.percent_energy.clone(),
            signal_to_noise: self.signal_to_noise.clone(),
            eastward_mean: self.eastward_mean,
            northward_mean: self.northward_mean,
            eastward_slope_per_day: self.eastward_slope_per_day,
            northward_slope_per_day: self.northward_slope_per_day,
            reference_time_days: self.reference_time_days,
        })
    }

    pub(crate) fn component_solutions(&self) -> (ScalarSolution, ScalarSolution) {
        let cartesian = self
            .cartesian()
            .expect("internally produced vector solutions have consistent shapes");
        (
            component_solution(
                cartesian.eastward_cosine_coefficient,
                cartesian.eastward_sine_coefficient,
                &cartesian.percent_energy,
                cartesian.signal_to_noise.as_deref(),
                cartesian.eastward_mean,
                cartesian.eastward_slope_per_day,
                cartesian.reference_time_days,
                self.robust.clone(),
            ),
            component_solution(
                cartesian.northward_cosine_coefficient,
                cartesian.northward_sine_coefficient,
                &cartesian.percent_energy,
                cartesian.signal_to_noise.as_deref(),
                cartesian.northward_mean,
                cartesian.northward_slope_per_day,
                cartesian.reference_time_days,
                self.robust.clone(),
            ),
        )
    }
}

pub(crate) fn from_component_solutions(
    eastward: &ScalarSolution,
    northward: &ScalarSolution,
) -> Result<VectorSolution, AnalysisError> {
    let constituent_count = eastward.cosine_coefficient.len();
    for (field, actual) in [
        (
            "northward_cosine_coefficient",
            northward.cosine_coefficient.len(),
        ),
        ("eastward_sine_coefficient", eastward.sine_coefficient.len()),
        (
            "northward_sine_coefficient",
            northward.sine_coefficient.len(),
        ),
    ] {
        if actual != constituent_count {
            return Err(AnalysisError::InvalidSolutionShape {
                field,
                actual,
                expected: constituent_count,
            });
        }
    }

    let mut semi_major = Vec::with_capacity(constituent_count);
    let mut semi_minor = Vec::with_capacity(constituent_count);
    let mut inclination_degrees = Vec::with_capacity(constituent_count);
    let mut phase_degrees = Vec::with_capacity(constituent_count);
    for constituent in 0..constituent_count {
        let (major, minor, inclination, phase) = ellipse_parameters(
            eastward.cosine_coefficient[constituent],
            eastward.sine_coefficient[constituent],
            northward.cosine_coefficient[constituent],
            northward.sine_coefficient[constituent],
        );
        semi_major.push(major);
        semi_minor.push(minor);
        inclination_degrees.push(inclination);
        phase_degrees.push(phase);
    }
    let energy = semi_major
        .iter()
        .zip(&semi_minor)
        .map(|(major, minor)| major * major + minor * minor)
        .collect::<Vec<_>>();
    let total_energy = energy.iter().sum::<f64>();
    let percent_energy = energy
        .iter()
        .map(|value| 100.0 * value / total_energy)
        .collect::<Vec<_>>();

    let intervals = vector_intervals(eastward, northward)?;
    let (semi_major_ci, semi_minor_ci, inclination_ci_degrees, phase_ci_degrees, signal_to_noise) =
        match intervals {
            Some(intervals) => {
                let signal_to_noise = energy
                    .iter()
                    .zip(&intervals.semi_major)
                    .zip(&intervals.semi_minor)
                    .map(|((energy, major_ci), minor_ci)| {
                        energy / ((major_ci / 1.96).powi(2) + (minor_ci / 1.96).powi(2))
                    })
                    .collect();
                (
                    Some(intervals.semi_major),
                    Some(intervals.semi_minor),
                    Some(intervals.inclination_degrees),
                    Some(intervals.phase_degrees),
                    Some(signal_to_noise),
                )
            }
            None => (None, None, None, None, None),
        };
    let robust = match (&eastward.robust, &northward.robust) {
        (None, None) => None,
        (Some(eastward), Some(northward)) if eastward == northward => Some(eastward.clone()),
        _ => {
            return Err(AnalysisError::InvalidSolutionShape {
                field: "vector robust diagnostics",
                actual: 1,
                expected: 2,
            });
        }
    };

    Ok(VectorSolution {
        semi_major,
        semi_minor,
        inclination_degrees,
        phase_degrees,
        percent_energy,
        semi_major_ci,
        semi_minor_ci,
        inclination_ci_degrees,
        phase_ci_degrees,
        signal_to_noise,
        eastward_mean: eastward.mean,
        northward_mean: northward.mean,
        eastward_slope_per_day: eastward.slope_per_day,
        northward_slope_per_day: northward.slope_per_day,
        reference_time_days: eastward.reference_time_days,
        robust,
    })
}

pub(crate) fn ellipse_parameters(
    eastward_cosine: f64,
    eastward_sine: f64,
    northward_cosine: f64,
    northward_sine: f64,
) -> (f64, f64, f64, f64) {
    let positive_real = eastward_cosine.midpoint(northward_sine);
    let positive_imaginary = 0.5 * (northward_cosine - eastward_sine);
    let negative_real = 0.5 * (eastward_cosine - northward_sine);
    let negative_imaginary = northward_cosine.midpoint(eastward_sine);
    let positive_radius = positive_real.hypot(positive_imaginary);
    let negative_radius = negative_real.hypot(negative_imaginary);
    let positive_angle = positive_imaginary.atan2(positive_real).to_degrees();
    let negative_angle = negative_imaginary.atan2(negative_real).to_degrees();
    let inclination = positive_angle.midpoint(negative_angle).rem_euclid(180.0);
    (
        positive_radius + negative_radius,
        positive_radius - negative_radius,
        inclination,
        (-positive_angle + inclination).rem_euclid(360.0),
    )
}

struct VectorIntervals {
    semi_major: Vec<f64>,
    semi_minor: Vec<f64>,
    inclination_degrees: Vec<f64>,
    phase_degrees: Vec<f64>,
}

fn vector_intervals(
    eastward: &ScalarSolution,
    northward: &ScalarSolution,
) -> Result<Option<VectorIntervals>, AnalysisError> {
    let variances = match (
        eastward.cosine_coefficient_variance.as_deref(),
        eastward.sine_coefficient_variance.as_deref(),
        northward.cosine_coefficient_variance.as_deref(),
        northward.sine_coefficient_variance.as_deref(),
    ) {
        (None, None, None, None) => return Ok(None),
        (Some(east_cos), Some(east_sin), Some(north_cos), Some(north_sin)) => {
            (east_cos, east_sin, north_cos, north_sin)
        }
        _ => {
            return Err(AnalysisError::InvalidSolutionShape {
                field: "vector coefficient variances",
                actual: 0,
                expected: eastward.cosine_coefficient.len(),
            });
        }
    };
    let mut intervals = VectorIntervals {
        semi_major: Vec::new(),
        semi_minor: Vec::new(),
        inclination_degrees: Vec::new(),
        phase_degrees: Vec::new(),
    };
    for constituent in 0..eastward.cosine_coefficient.len() {
        let values = linearized_ellipse_sigmas(
            eastward.cosine_coefficient[constituent],
            eastward.sine_coefficient[constituent],
            northward.cosine_coefficient[constituent],
            northward.sine_coefficient[constituent],
            variances.0[constituent].sqrt(),
            variances.1[constituent].sqrt(),
            variances.2[constituent].sqrt(),
            variances.3[constituent].sqrt(),
        );
        intervals.semi_major.push(1.96 * values.0);
        intervals.semi_minor.push(1.96 * values.1);
        intervals.phase_degrees.push(1.96 * values.2);
        intervals.inclination_degrees.push(1.96 * values.3);
    }
    Ok(Some(intervals))
}

#[allow(
    clippy::too_many_arguments,
    reason = "matches UTide's four coefficients and four sigmas"
)]
pub(crate) fn linearized_ellipse_sigmas(
    eastward_cosine: f64,
    eastward_sine: f64,
    northward_cosine: f64,
    northward_sine: f64,
    sigma_eastward_cosine: f64,
    sigma_eastward_sine: f64,
    sigma_northward_cosine: f64,
    sigma_northward_sine: f64,
) -> (f64, f64, f64, f64) {
    let positive_radius =
        0.5 * (eastward_cosine + northward_sine).hypot(northward_cosine - eastward_sine);
    let negative_radius =
        0.5 * (eastward_cosine - northward_sine).hypot(northward_cosine + eastward_sine);
    let ex = (eastward_cosine + northward_sine) / positive_radius;
    let fx = (eastward_cosine - northward_sine) / negative_radius;
    let gx = (eastward_sine - northward_cosine) / positive_radius;
    let hx = (eastward_sine + northward_cosine) / negative_radius;
    let variances = [
        sigma_eastward_cosine.powi(2),
        sigma_eastward_sine.powi(2),
        sigma_northward_cosine.powi(2),
        sigma_northward_sine.powi(2),
    ];
    let major = weighted_sigma(
        [
            0.25 * (ex + fx),
            0.25 * (gx + hx),
            0.25 * (hx - gx),
            0.25 * (ex - fx),
        ],
        variances,
    );
    let minor = weighted_sigma(
        [
            0.25 * (ex - fx),
            0.25 * (gx - hx),
            0.25 * (hx + gx),
            0.25 * (ex + fx),
        ],
        variances,
    );
    let phase = angular_sigma(
        eastward_cosine,
        eastward_sine,
        northward_cosine,
        northward_sine,
        variances,
        true,
    );
    let inclination = angular_sigma(
        eastward_cosine,
        eastward_sine,
        northward_cosine,
        northward_sine,
        variances,
        false,
    );
    (major, minor, phase, inclination)
}

fn weighted_sigma(derivatives: [f64; 4], variances: [f64; 4]) -> f64 {
    derivatives
        .into_iter()
        .zip(variances)
        .map(|(derivative, variance)| derivative.powi(2) * variance)
        .sum::<f64>()
        .sqrt()
}

fn angular_sigma(
    eastward_cosine: f64,
    eastward_sine: f64,
    northward_cosine: f64,
    northward_sine: f64,
    variances: [f64; 4],
    phase: bool,
) -> f64 {
    let (numerator, denominator, derivatives) = if phase {
        let numerator = 2.0 * (eastward_cosine * eastward_sine + northward_cosine * northward_sine);
        let denominator = eastward_cosine.powi(2) - eastward_sine.powi(2)
            + northward_cosine.powi(2)
            - northward_sine.powi(2);
        (
            numerator,
            denominator,
            [
                denominator * eastward_sine - numerator * eastward_cosine,
                denominator * eastward_cosine + numerator * eastward_sine,
                denominator * northward_sine - numerator * northward_cosine,
                denominator * northward_cosine + numerator * northward_sine,
            ],
        )
    } else {
        let numerator = 2.0 * (eastward_cosine * northward_cosine + eastward_sine * northward_sine);
        let denominator = eastward_cosine.powi(2) + eastward_sine.powi(2)
            - northward_cosine.powi(2)
            - northward_sine.powi(2);
        (
            numerator,
            denominator,
            [
                denominator * northward_cosine - numerator * eastward_cosine,
                denominator * northward_sine - numerator * eastward_sine,
                denominator * eastward_cosine + numerator * northward_cosine,
                denominator * eastward_sine + numerator * northward_sine,
            ],
        )
    };
    let scale = (numerator.powi(2) + denominator.powi(2)).recip();
    weighted_sigma(derivatives.map(|value| value * scale), variances) * 180.0 / PI
}

#[allow(
    clippy::too_many_arguments,
    reason = "component reconstruction carries coefficients, diagnostics, mean, trend, and epoch"
)]
fn component_solution(
    cosine_coefficient: Vec<f64>,
    sine_coefficient: Vec<f64>,
    percent_energy: &[f64],
    signal_to_noise: Option<&[f64]>,
    mean: f64,
    slope_per_day: f64,
    reference_time_days: f64,
    robust: Option<RobustDiagnostics>,
) -> ScalarSolution {
    let amplitude = cosine_coefficient
        .iter()
        .zip(&sine_coefficient)
        .map(|(cosine, sine)| cosine.hypot(*sine))
        .collect();
    let phase_degrees = cosine_coefficient
        .iter()
        .zip(&sine_coefficient)
        .map(|(cosine, sine)| sine.atan2(*cosine).to_degrees().rem_euclid(360.0))
        .collect();
    ScalarSolution {
        cosine_coefficient,
        sine_coefficient,
        amplitude,
        phase_degrees,
        percent_energy: percent_energy.to_vec(),
        amplitude_ci: None,
        phase_ci_degrees: None,
        signal_to_noise: signal_to_noise.map(<[f64]>::to_vec),
        cosine_coefficient_variance: None,
        sine_coefficient_variance: None,
        mean,
        slope_per_day,
        reference_time_days,
        robust,
    }
}

fn descending_indices(values: &[f64]) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indices
}

#[cfg(test)]
mod tests {
    use super::{VectorSolution, ellipse_parameters};
    use crate::AnalysisError;

    fn solution(phase_degrees: f64) -> VectorSolution {
        VectorSolution {
            semi_major: vec![1.2],
            semi_minor: vec![-0.3],
            inclination_degrees: vec![42.0],
            phase_degrees: vec![phase_degrees],
            percent_energy: vec![100.0],
            semi_major_ci: None,
            semi_minor_ci: None,
            inclination_ci_degrees: None,
            phase_ci_degrees: None,
            signal_to_noise: Some(vec![25.0]),
            eastward_mean: 0.2,
            northward_mean: -0.1,
            eastward_slope_per_day: 0.01,
            northward_slope_per_day: -0.02,
            reference_time_days: 60_000.0,
            robust: None,
        }
    }

    #[test]
    fn cartesian_coefficients_round_trip_through_the_ellipse_representation() {
        let expected = solution(359.0);
        let cartesian = expected.cartesian().expect("consistent ellipse solution");
        let actual = ellipse_parameters(
            cartesian.eastward_cosine_coefficient[0],
            cartesian.eastward_sine_coefficient[0],
            cartesian.northward_cosine_coefficient[0],
            cartesian.northward_sine_coefficient[0],
        );
        for (actual, expected) in [
            (actual.0, expected.semi_major[0]),
            (actual.1, expected.semi_minor[0]),
            (actual.2, expected.inclination_degrees[0]),
            (actual.3, expected.phase_degrees[0]),
        ] {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn cartesian_coefficients_remain_continuous_across_phase_wrap() {
        let below = solution(359.0).cartesian().expect("valid ellipse");
        let above = solution(1.0).cartesian().expect("valid ellipse");
        for (below, above) in [
            (
                below.eastward_cosine_coefficient[0],
                above.eastward_cosine_coefficient[0],
            ),
            (
                below.eastward_sine_coefficient[0],
                above.eastward_sine_coefficient[0],
            ),
            (
                below.northward_cosine_coefficient[0],
                above.northward_cosine_coefficient[0],
            ),
            (
                below.northward_sine_coefficient[0],
                above.northward_sine_coefficient[0],
            ),
        ] {
            assert!((below - above).abs() < 0.04);
        }
    }

    #[test]
    fn cartesian_conversion_rejects_mismatched_ellipse_arrays() {
        let mut invalid = solution(30.0);
        invalid.phase_degrees.clear();
        assert!(matches!(
            invalid.cartesian(),
            Err(AnalysisError::InvalidSolutionShape {
                field: "phase_degrees",
                actual: 0,
                expected: 1,
            })
        ));
    }
}
