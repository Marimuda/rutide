//! Numerical building blocks for `RUTide` harmonic analysis.
//!
//! This crate will contain the dependency-light scientific kernel. `NetCDF` input,
//! command-line behavior, Python bindings, and benchmark orchestration belong in
//! separate workspace crates.

mod astronomy;
mod catalog;
mod corrected;
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
    NodalCorrections, PhaseReference, ReconstructionFilter, ScalarInferenceBatch,
    ScalarInferenceOls, ScalarInferenceRelation, SolverOptions, VectorInferenceBatch,
    VectorInferenceOls, VectorInferenceRelation,
};
pub use error::AnalysisError;
pub use monte_carlo::MonteCarloOptions;
pub use robust::{RobustDiagnostics, RobustOptions, RobustTermination};
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
pub use vector::{VectorReconstruction, VectorSolution};

/// The `RUTide` core crate version used to produce a result.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_workspace_package() {
        assert_eq!(VERSION, "0.1.0");
    }
}
