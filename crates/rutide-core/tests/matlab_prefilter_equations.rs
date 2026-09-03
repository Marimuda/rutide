//! Frozen equation-level fixture for MATLAB `UTide`'s `PreFilt` path.
//!
//! The expected values independently implement `ut_E` lines 1881--1884 from
//! revision `4a6354f`: linear interpolation at constituent frequency followed
//! by a unity substitution outside the configured acceptable gain range.

use rutide_core::{Constituent, PreFilterCorrection, PreFilterFallback};

#[test]
fn matlab_prefilter_interpolation_and_unity_substitution_are_frozen() {
    let correction =
        PreFilterCorrection::new(vec![0.0, 0.05, 0.1], vec![1.0, 0.4, 0.005], 0.01, 100.0)
            .expect("valid MATLAB-style transfer function")
            .with_fallback(PreFilterFallback::Unity);
    let constituents = [
        Constituent::new("interpolated", 0.025),
        Constituent::new("unacceptable", 0.1),
        Constituent::new("outside", 0.12),
    ];

    let actual = correction
        .resolve_constituent_gains(&constituents)
        .expect("MATLAB unity fallback");
    let interpolation_fraction = (0.025 - 0.0) / (0.05 - 0.0);
    let interpolated = 1.0 + interpolation_fraction * (0.4 - 1.0);
    assert_eq!(actual, vec![interpolated, 1.0, 1.0]);
}
