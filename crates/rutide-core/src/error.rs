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
    /// Timestamps are not strictly increasing.
    NonIncreasingTime {
        /// Position of the latter timestamp in the invalid pair.
        index: usize,
    },
    /// No constituents were requested.
    EmptyConstituents,
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
}

impl fmt::Display for AnalysisError {
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
            Self::NonIncreasingTime { index } => write!(
                formatter,
                "timestamps must be strictly increasing; violation at index {index}"
            ),
            Self::EmptyConstituents => formatter.write_str("at least one constituent is required"),
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
            Self::EmptySeries => formatter.write_str("at least one observation series is required"),
            Self::ObservationShape { actual, expected } => write!(
                formatter,
                "flattened observations contain {actual} values; expected {expected}"
            ),
            Self::NonFiniteObservation { series, time } => write!(
                formatter,
                "observation for series {series} at time index {time} is not finite"
            ),
        }
    }
}

impl Error for AnalysisError {}
