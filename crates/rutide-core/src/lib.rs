//! Numerical building blocks for `RUTide` harmonic analysis.
//!
//! This crate will contain the dependency-light scientific kernel. `NetCDF` input,
//! command-line behavior, Python bindings, and benchmark orchestration belong in
//! separate workspace crates.

mod astronomy;
mod catalog;
mod corrected;
mod error;
mod scalar;

pub use catalog::{
    CATALOG_ORACLE_REVISION, CATALOG_SOURCE_SHA256, CONSTITUENT_COUNT, TidalConstituent,
    UnknownTidalConstituent,
};
pub use corrected::{GreenwichNodalBatch, GreenwichNodalOls};
pub use error::AnalysisError;
pub use scalar::{Constituent, FixedRawOls, ScalarSolution};

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
