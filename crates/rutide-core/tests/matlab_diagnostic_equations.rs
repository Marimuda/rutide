//! Frozen equation-level oracle from Codiga's MATLAB `UTide` `ut_solv.m`.
//!
//! Source: `OceanMetSEPA/utide_toolbox`, `ut_solv.m` lines 2545-2581 on the
//! repository's single public commit. These deliberately simple inputs keep
//! every expected value independently auditable without requiring MATLAB at
//! test time.

use rutide_core::{
    Constituent, DiagnosticConstituentRole, adjacent_constituent_diagnostics,
    scalar_tidal_variance_diagnostics, vector_tidal_variance_diagnostics,
};

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= 32.0 * f64::EPSILON * scale,
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}

#[test]
fn matlab_rayleigh_and_tidal_variance_equations_are_frozen() {
    // MATLAB: RR = 24 * elor * (frq(c2) - frq(c1)) / rmin.
    let constituents = [
        Constituent::new("A", 0.04),
        Constituent::new("B", 0.05),
        Constituent::new("C", 0.08),
    ];
    let diagnostics = adjacent_constituent_diagnostics(
        &constituents,
        &[DiagnosticConstituentRole::Direct; 3],
        10.0,
        1.0,
    )
    .expect("valid MATLAB Rayleigh oracle inputs");
    assert_close(
        diagnostics[0]
            .higher
            .as_ref()
            .expect("A/B neighbor")
            .rayleigh_criterion,
        2.4,
    );
    assert_close(
        diagnostics[1]
            .higher
            .as_ref()
            .expect("B/C neighbor")
            .rayleigh_criterion,
        7.2,
    );

    // MATLAB scalar: mean(urawtid.^2), with PTV = 100 * TV / TVraw.
    let scalar = scalar_tidal_variance_diagnostics(
        &[1.0, -1.0, 2.0, -2.0],
        &[0.8, -0.8, 1.6, -1.6],
        Some(&[0.6, -0.6, 1.2, -1.2]),
    )
    .expect("valid MATLAB scalar-variance oracle inputs");
    assert_close(scalar.raw_tidal_variance, 2.5);
    assert_close(scalar.all_constituent_tidal_variance, 1.6);
    assert_close(
        scalar
            .significant_constituent_tidal_variance
            .expect("significant scalar variance"),
        0.9,
    );
    assert_close(
        scalar
            .all_constituent_percent_tidal_variance
            .expect("all scalar percent variance"),
        64.0,
    );
    assert_close(
        scalar
            .significant_constituent_percent_tidal_variance
            .expect("significant scalar percent variance"),
        36.0,
    );

    // MATLAB vector: mean(urawtid.^2 + vrawtid.^2).
    let vector = vector_tidal_variance_diagnostics(
        &[1.0, -1.0],
        &[2.0, -2.0],
        &[0.8, -0.8],
        &[1.6, -1.6],
        Some(&[0.6, -0.6]),
        Some(&[1.2, -1.2]),
    )
    .expect("valid MATLAB vector-variance oracle inputs");
    assert_close(vector.raw_tidal_variance, 5.0);
    assert_close(vector.all_constituent_tidal_variance, 3.2);
    assert_close(
        vector
            .significant_constituent_tidal_variance
            .expect("significant vector variance"),
        1.8,
    );
    assert_close(
        vector
            .all_constituent_percent_tidal_variance
            .expect("all vector percent variance"),
        64.0,
    );
    assert_close(
        vector
            .significant_constituent_percent_tidal_variance
            .expect("significant vector percent variance"),
        36.0,
    );
}
