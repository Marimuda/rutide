//! Exact corrected-basis parity against the pinned Python `UTide` oracle.

use rutide_core::{GreenwichNodalBatch, GreenwichNodalOls, TidalConstituent};

const CONSTITUENTS: [TidalConstituent; 5] = [
    TidalConstituent::M2,
    TidalConstituent::S2,
    TidalConstituent::N2,
    TidalConstituent::K1,
    TidalConstituent::O1,
];
const LATITUDE_DEGREES_NORTH: f64 = 60.957_717_895_507_81;
const EXPECTED_AMPLITUDE: [f64; 5] = [
    0.653_826_519_379_141_7,
    0.226_316_745_310_547_24,
    0.158_004_284_372_840_4,
    0.112_468_705_272_284_42,
    0.068_797_125_518_099_88,
];
const EXPECTED_PHASE_DEGREES: [f64; 5] = [
    189.408_331_235_106_77,
    228.999_253_389_370_64,
    163.408_535_069_689_07,
    154.311_435_223_673_04,
    20.062_605_139_407_12,
];
const EXPECTED_MEAN: f64 = 0.091_040_690_255_747_43;
const EXPECTED_SLOPE_PER_DAY: f64 = 0.001_734_852_911_719_784_6;

fn oracle_times() -> Vec<f64> {
    (0_u32..745)
        .map(|index| {
            let day = 58_113 + index / 24;
            let milliseconds = (index % 24) * 3_600_000;
            f64::from(day) + f64::from(milliseconds) / 86_400_000.0
        })
        .collect()
}

fn oracle_observations() -> Vec<f64> {
    include_str!("data/fvcom_node_0_zeta_f32.hex")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let bits = u32::from_str_radix(line, 16).expect("fixture contains valid hexadecimal");
            f64::from(f32::from_bits(bits))
        })
        .collect()
}

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn matches_python_utide_for_real_fvcom_elevation() {
    let time = oracle_times();
    let observations = oracle_observations();
    assert_eq!(time.len(), observations.len());

    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid corrected oracle model");
    let solution = model
        .solve(&observations)
        .expect("valid oracle observations");
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)
        .expect("valid corrected oracle batch");
    let batch_solution = batch
        .solve_time_major(&observations, &[LATITUDE_DEGREES_NORTH])
        .expect("valid batched oracle observations");
    assert_eq!(batch_solution.as_slice(), std::slice::from_ref(&solution));

    for (index, (actual, expected)) in solution
        .amplitude
        .iter()
        .zip(EXPECTED_AMPLITUDE)
        .enumerate()
    {
        assert_close(&format!("amplitude[{index}]"), *actual, expected, 3e-12);
    }
    for (index, (actual, expected)) in solution
        .phase_degrees
        .iter()
        .zip(EXPECTED_PHASE_DEGREES)
        .enumerate()
    {
        assert_close(&format!("phase[{index}]"), *actual, expected, 3e-9);
    }
    assert_close("mean", solution.mean, EXPECTED_MEAN, 3e-12);
    assert_close(
        "slope_per_day",
        solution.slope_per_day,
        EXPECTED_SLOPE_PER_DAY,
        3e-12,
    );
}
