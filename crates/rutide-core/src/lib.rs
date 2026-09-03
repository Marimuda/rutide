//! Numerical building blocks for `RUTide` harmonic analysis.
//!
//! This crate will contain the dependency-light scientific kernel. `NetCDF` input,
//! command-line behavior, Python bindings, and benchmark orchestration belong in
//! separate workspace crates.

mod astronomy;
mod catalog;
mod corrected;
mod diagnostics;
mod error;
mod monte_carlo;
mod robust;
mod sampling;
mod scalar;
mod selection;
mod time;
mod vector;

pub use catalog::{
    CATALOG_ORACLE_REVISION, CATALOG_SOURCE_SHA256, CONSTITUENT_COUNT, TidalConstituent,
    UnknownTidalConstituent,
};
pub use corrected::{
    GreenwichNodalBatch, GreenwichNodalOls, GreenwichNodalReconstructor, InferenceMode,
    NodalCorrections, NonHarmonicTerms, PhaseReference, ReconstructionFilter, ScalarInferenceBatch,
    ScalarInferenceOls, ScalarInferenceRelation, SolverOptions, VectorInferenceBatch,
    VectorInferenceOls, VectorInferenceRelation,
};
pub use diagnostics::{
    ConstituentDiagnosticsOptions, ConstituentIndependenceDiagnostics,
    ConstituentSelectionDiagnostics, DiagnosticConstituentRole, NeighboringConstituentDiagnostics,
    TidalVarianceDiagnostics, WholeModelIndependenceDiagnostics, adjacent_constituent_diagnostics,
    scalar_tidal_variance_diagnostics, vector_tidal_variance_diagnostics,
};
pub use error::AnalysisError;
pub use monte_carlo::MonteCarloOptions;
pub use robust::{RobustDiagnostics, RobustOptions, RobustTermination, RobustWeightFunction};
pub use sampling::{
    COLORED_NOISE_FREQUENCY_BANDS_CPH, ResidualSpectrumMethod, SamplingDiagnostics,
    SamplingDiagnosticsPlan,
};
pub use scalar::{Constituent, FitOptions, FixedRawOls, LinearConfidence, ScalarSolution};
pub use selection::{RayleighSelection, select_constituents_by_rayleigh};
pub use time::{
    GregorianDateTime, NormalizedTimeAxis, TimeEpoch, normalize_numeric_time,
    system_time_to_modified_julian_day,
};
pub use vector::{CartesianVectorSolution, VectorCurrent, VectorReconstruction, VectorSolution};

/// The `RUTide` core crate version used to produce a result.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
