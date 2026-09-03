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
    /// A requested prepared reconstruction timestamp does not exist.
    ReconstructionTimeIndexOutOfBounds {
        /// Requested target-time position.
        index: usize,
        /// Number of timestamps retained by the reconstructor.
        time_count: usize,
    },
    /// Timestamps are not strictly increasing.
    NonIncreasingTime {
        /// Position of the latter timestamp in the invalid pair.
        index: usize,
    },
    /// A proleptic-Gregorian civil timestamp has an out-of-range component.
    InvalidGregorianDateTime {
        /// Name of the invalid component.
        component: &'static str,
    },
    /// A finite-observation mask does not match its source time axis.
    SamplingMaskShape {
        /// Number of mask values received.
        actual: usize,
        /// Number required by the source time axis.
        expected: usize,
    },
    /// Fewer than two observations remain for sampling diagnostics.
    InsufficientSamplingObservations {
        /// Number of finite observations retained.
        actual: usize,
    },
    /// A fitted frequency supplied to sampling diagnostics is invalid.
    InvalidSamplingFrequency {
        /// Position of the invalid frequency.
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
    /// The effective record length supplied to selection diagnostics is invalid.
    InvalidDiagnosticEffectiveRecordLength,
    /// A detrended value supplied to reconstructed-fit diagnostics is not finite.
    NonFiniteDiagnosticValue {
        /// Diagnostic input field containing the invalid value.
        field: &'static str,
        /// Position of the invalid value.
        index: usize,
    },
    /// A constituent SNR supplied to independence diagnostics is NaN or negative.
    InvalidDiagnosticSignalToNoise {
        /// Position of the invalid constituent SNR.
        index: usize,
    },
    /// A coefficient variance is non-finite or not strictly positive.
    InvalidDiagnosticCoefficientVariance {
        /// Position of the invalid harmonic parameter.
        parameter: usize,
    },
    /// A coefficient covariance could not be normalized into a correlation.
    InvalidDiagnosticCoefficientCorrelation {
        /// Row parameter of the invalid covariance.
        left: usize,
        /// Column parameter of the invalid covariance.
        right: usize,
    },
    /// A diagnostic basis condition number is NaN or not strictly positive.
    InvalidDiagnosticBasisConditionNumber,
    /// The small SVD used for a basis condition number did not converge.
    DiagnosticDecompositionFailed,
    /// A whole-model diagnostic energy is non-finite or negative.
    InvalidDiagnosticEnergy {
        /// Diagnostic energy field containing the invalid value.
        field: &'static str,
    },
    /// A constituent-diagnostic threshold is negative or non-finite.
    InvalidDiagnosticThreshold {
        /// Diagnostic whose threshold is invalid.
        diagnostic: &'static str,
    },
    /// A solution was not produced by the prepared model used for diagnostics.
    DiagnosticSolutionModelMismatch,
    /// A robust weight supplied to model diagnostics is negative or non-finite.
    InvalidDiagnosticWeight {
        /// Time position of the invalid weight.
        time: usize,
    },
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
    /// An SNR-dependent operation was requested without confidence intervals.
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
    /// The robust weight-function tuning constant is not finite and positive.
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
            Self::ReconstructionTimeIndexOutOfBounds { index, time_count } => write!(
                formatter,
                "reconstruction time index {index} is outside the {time_count} prepared timestamps"
            ),
            Self::NonIncreasingTime { index } => write!(
                formatter,
                "timestamps must be strictly increasing; violation at index {index}"
            ),
            Self::InvalidGregorianDateTime { component } => {
                write!(formatter, "Gregorian datetime has an invalid {component}")
            }
            Self::SamplingMaskShape { actual, expected } => write!(
                formatter,
                "sampling mask contains {actual} values; expected {expected}"
            ),
            Self::InsufficientSamplingObservations { actual } => write!(
                formatter,
                "sampling diagnostics require at least two finite observations; received {actual}"
            ),
            Self::InvalidSamplingFrequency { index } => write!(
                formatter,
                "sampling diagnostic frequency at index {index} must be finite and non-negative"
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
            Self::InvalidDiagnosticEffectiveRecordLength => formatter.write_str(
                "diagnostic effective record length must be finite and greater than zero",
            ),
            Self::NonFiniteDiagnosticValue { field, index } => write!(
                formatter,
                "diagnostic field {field:?} contains a non-finite value at index {index}"
            ),
            Self::InvalidDiagnosticSignalToNoise { index } => write!(
                formatter,
                "diagnostic signal-to-noise ratio at index {index} must be non-negative and not NaN"
            ),
            Self::InvalidDiagnosticCoefficientVariance { parameter } => write!(
                formatter,
                "diagnostic coefficient variance for parameter {parameter} must be finite and greater than zero"
            ),
            Self::InvalidDiagnosticCoefficientCorrelation { left, right } => write!(
                formatter,
                "diagnostic coefficient correlation for parameters {left} and {right} is not finite"
            ),
            Self::InvalidDiagnosticBasisConditionNumber => formatter
                .write_str("diagnostic basis condition number must be positive and not NaN"),
            Self::DiagnosticDecompositionFailed => formatter
                .write_str("diagnostic basis singular-value decomposition did not converge"),
            Self::InvalidDiagnosticEnergy { field } => write!(
                formatter,
                "diagnostic energy field {field:?} must be finite and non-negative"
            ),
            Self::InvalidDiagnosticThreshold { diagnostic } => write!(
                formatter,
                "{diagnostic} diagnostic threshold must be finite and non-negative"
            ),
            Self::DiagnosticSolutionModelMismatch => formatter
                .write_str("diagnostic solution reference time does not match the prepared model"),
            Self::InvalidDiagnosticWeight { time } => write!(
                formatter,
                "diagnostic robust weight at time {time} must be finite and non-negative"
            ),
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
                "the requested SNR-dependent operation requires a solution with confidence intervals",
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
