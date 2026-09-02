//! Errors produced while preparing or solving a harmonic analysis.

use std::{error::Error, fmt};

/// An invalid input to a harmonic analysis.
#[derive(Clone, Debug, PartialEq)]
pub enum AnalysisError {
    /// No timestamps were supplied.
    EmptyTime,
    /// There are too few observations for the requested model columns.
    InsufficientObservations {
        /// Number of observations received.
        actual: usize,
        /// Minimum number needed to overdetermine the model.
        required: usize,
    },
    /// A timestamp is NaN or infinite.
    NonFiniteTime {
        /// Position of the invalid timestamp.
        index: usize,
    },
    /// A reconstruction fit-reference epoch is NaN or infinite.
    NonFiniteReferenceTime,
    /// Timestamps are not strictly increasing.
    NonIncreasingTime {
        /// Position of the latter timestamp in the invalid pair.
        index: usize,
    },
    /// Latitude is non-finite or outside the physical range.
    InvalidLatitude,
    /// Python `UTide`'s exact nodal formula is singular at precisely zero latitude.
    EquatorialLatitude,
    /// No constituents were requested.
    EmptyConstituents,
    /// The Rayleigh criterion is not finite and strictly positive.
    InvalidRayleighMinimum,
    /// A reconstruction threshold is negative or non-finite.
    InvalidReconstructionThreshold {
        /// Diagnostic whose threshold is invalid.
        diagnostic: &'static str,
    },
    /// A coefficient array does not match the prepared constituent count.
    InvalidSolutionShape {
        /// Coefficient or diagnostic field with the invalid length.
        field: &'static str,
        /// Number of values received.
        actual: usize,
        /// Number of values required by the reconstruction basis.
        expected: usize,
    },
    /// SNR filtering was requested for coefficients without confidence intervals.
    MissingSignalToNoise,
    /// An explicit reconstruction constituent was not part of the fitted model.
    UnpreparedReconstructionConstituent {
        /// Conventional catalog name.
        name: &'static str,
    },
    /// An explicit reconstruction constituent occurs more than once.
    DuplicateReconstructionConstituent {
        /// Position of the latter duplicate.
        index: usize,
    },
    /// A constituent name is empty.
    EmptyConstituentName {
        /// Position of the invalid constituent.
        index: usize,
    },
    /// A constituent frequency is not finite and positive.
    InvalidFrequency {
        /// Position of the invalid constituent.
        index: usize,
    },
    /// Two requested constituents have the same frequency.
    DuplicateFrequency {
        /// Position of the latter duplicate constituent.
        index: usize,
    },
    /// No inferred/reference relationships were supplied.
    EmptyInference,
    /// An inference amplitude ratio is negative, NaN, or infinite.
    InvalidInferenceAmplitudeRatio {
        /// Position of the invalid relationship.
        index: usize,
    },
    /// An inference phase offset is NaN or infinite.
    InvalidInferencePhaseOffset {
        /// Position of the invalid relationship.
        index: usize,
    },
    /// The same constituent is inferred more than once.
    DuplicateInferredConstituent {
        /// Position of the latter relationship.
        index: usize,
    },
    /// A relationship attempts to infer a constituent from itself.
    SelfInference {
        /// Position of the invalid relationship.
        index: usize,
    },
    /// An inferred constituent is also used as a reference, forming a chain or cycle.
    InferenceReferenceIsInferred {
        /// Conventional catalog name of the invalid reference.
        name: &'static str,
    },
    /// No observation series were supplied.
    EmptySeries,
    /// The flattened time-major observation shape does not match the model.
    ObservationShape {
        /// Number of values received.
        actual: usize,
        /// Number of values required by `time_count * series_count`.
        expected: usize,
    },
    /// An observation is NaN or infinite.
    NonFiniteObservation {
        /// Spatial-series position of the invalid value.
        series: usize,
        /// Time position of the invalid value.
        time: usize,
    },
    /// The robust Cauchy tuning constant is not finite and positive.
    InvalidRobustTuningConstant,
    /// The robust convergence tolerance is not finite and positive.
    InvalidRobustTolerance,
    /// The robust iteration limit is zero.
    InvalidRobustIterationLimit,
    /// A model row has invalid leverage for robust residual normalization.
    InvalidRobustLeverage {
        /// Time position of the invalid leverage.
        time: usize,
    },
    /// The residual MAD collapsed to zero even though the fit is not exact.
    DegenerateRobustScale,
    /// Robust fitting exhausted its configured iteration limit.
    RobustDidNotConverge {
        /// Number of completed iterations.
        iterations: usize,
    },
    /// A Monte Carlo confidence calculation requested fewer than two realizations.
    InvalidMonteCarloRealizationCount,
    /// A coefficient covariance matrix was not finite or could not be repaired.
    InvalidConfidenceCovariance {
        /// Constituent position whose covariance could not be sampled.
        constituent: usize,
    },
}

impl fmt::Display for AnalysisError {
    #[allow(
        clippy::too_many_lines,
        reason = "keeps every public error variant's user-facing message exhaustive in one match"
    )]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTime => formatter.write_str("at least one timestamp is required"),
            Self::InsufficientObservations { actual, required } => write!(
                formatter,
                "model requires at least {required} observations, received {actual}"
            ),
            Self::NonFiniteTime { index } => {
                write!(formatter, "timestamp at index {index} is not finite")
            }
            Self::NonFiniteReferenceTime => {
                formatter.write_str("reconstruction reference time must be finite")
            }
            Self::NonIncreasingTime { index } => write!(
                formatter,
                "timestamps must be strictly increasing; violation at index {index}"
            ),
            Self::InvalidLatitude => formatter
                .write_str("latitude must be finite and between -90 and 90 degrees inclusive"),
            Self::EquatorialLatitude => formatter.write_str(
                "exact UTide nodal corrections are undefined at precisely zero latitude",
            ),
            Self::EmptyConstituents => formatter.write_str("at least one constituent is required"),
            Self::InvalidRayleighMinimum => {
                formatter.write_str("Rayleigh minimum must be finite and greater than zero")
            }
            Self::InvalidReconstructionThreshold { diagnostic } => write!(
                formatter,
                "{diagnostic} reconstruction threshold must be finite and non-negative"
            ),
            Self::InvalidSolutionShape {
                field,
                actual,
                expected,
            } => write!(
                formatter,
                "solution field {field:?} contains {actual} values; expected {expected}"
            ),
            Self::MissingSignalToNoise => formatter.write_str(
                "SNR reconstruction filtering requires a solution with confidence intervals",
            ),
            Self::UnpreparedReconstructionConstituent { name } => write!(
                formatter,
                "constituent {name} was not included in the fitted model"
            ),
            Self::DuplicateReconstructionConstituent { index } => write!(
                formatter,
                "reconstruction constituent at index {index} duplicates an earlier entry"
            ),
            Self::EmptyConstituentName { index } => {
                write!(formatter, "constituent at index {index} has an empty name")
            }
            Self::InvalidFrequency { index } => write!(
                formatter,
                "constituent at index {index} must have a finite positive frequency"
            ),
            Self::DuplicateFrequency { index } => write!(
                formatter,
                "constituent at index {index} duplicates an earlier frequency"
            ),
            Self::EmptyInference => {
                formatter.write_str("at least one inferred/reference relationship is required")
            }
            Self::InvalidInferenceAmplitudeRatio { index } => write!(
                formatter,
                "inference amplitude ratio at index {index} must be finite and non-negative"
            ),
            Self::InvalidInferencePhaseOffset { index } => write!(
                formatter,
                "inference phase offset at index {index} must be finite"
            ),
            Self::DuplicateInferredConstituent { index } => write!(
                formatter,
                "inferred constituent at index {index} duplicates an earlier relationship"
            ),
            Self::SelfInference { index } => write!(
                formatter,
                "inference relationship at index {index} uses the same constituent as reference and inferred"
            ),
            Self::InferenceReferenceIsInferred { name } => write!(
                formatter,
                "inference reference {name} is itself inferred; chains and cycles are unsupported"
            ),
            Self::EmptySeries => formatter.write_str("at least one observation series is required"),
            Self::ObservationShape { actual, expected } => write!(
                formatter,
                "flattened observations contain {actual} values; expected {expected}"
            ),
            Self::NonFiniteObservation { series, time } => write!(
                formatter,
                "observation for series {series} at time index {time} is not finite"
            ),
            Self::InvalidRobustTuningConstant => {
                formatter.write_str("robust tuning constant must be finite and greater than zero")
            }
            Self::InvalidRobustTolerance => formatter
                .write_str("robust convergence tolerance must be finite and greater than zero"),
            Self::InvalidRobustIterationLimit => {
                formatter.write_str("robust iteration limit must be greater than zero")
            }
            Self::InvalidRobustLeverage { time } => write!(
                formatter,
                "robust leverage at time index {time} must be finite and less than one"
            ),
            Self::DegenerateRobustScale => formatter.write_str(
                "robust residual scale is zero for a non-exact fit; weights are undefined",
            ),
            Self::RobustDidNotConverge { iterations } => write!(
                formatter,
                "robust fit did not converge within {iterations} iterations"
            ),
            Self::InvalidMonteCarloRealizationCount => {
                formatter.write_str("Monte Carlo confidence requires at least two realizations")
            }
            Self::InvalidConfidenceCovariance { constituent } => write!(
                formatter,
                "coefficient covariance for constituent {constituent} is not finite or could not be repaired for sampling"
            ),
        }
    }
}

impl Error for AnalysisError {}
