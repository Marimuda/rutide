//! Cross-language parity against the pinned Python `UTide` fixed-raw profile.

use rutide_core::{Constituent, FixedRawOls};

const FREQUENCIES_CPH: [(&str, f64); 5] = [
    ("M2", 0.080_511_400_671_577_2),
    ("S2", 0.083_333_333_333_333_33),
    ("N2", 0.078_999_248_699_109_28),
    ("K1", 0.041_780_746_221_637_22),
    ("O1", 0.038_730_654_449_939_97),
];
const EXPECTED_AMPLITUDE: [f64; 5] = [
    0.672_212_080_983_501_3,
    0.225_933_490_105_240_9,
    0.162_148_198_300_141_32,
    0.103_698_132_731_651_86,
    0.060_258_545_157_276_57,
];
const EXPECTED_PHASE_DEGREES: [f64; 5] = [
    34.370_624_971_387_79,
    228.905_303_520_041_7,
    123.434_087_495_025_96,
    321.201_305_972_653_2,
    54.581_120_861_150_35,
];
const EXPECTED_MEAN: f64 = 0.091_040_439_975_918_36;
const EXPECTED_SLOPE_PER_DAY: f64 = 0.001_734_844_350_544_766_5;

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
    let constituents: Vec<_> = FREQUENCIES_CPH
        .into_iter()
        .map(|(name, frequency)| Constituent::new(name, frequency))
        .collect();

    let model = FixedRawOls::prepare(&time, &constituents).expect("valid oracle model");
    let solution = model
        .solve(&observations)
        .expect("valid oracle observations");

    for (index, (actual, expected)) in solution
        .amplitude
        .iter()
        .zip(EXPECTED_AMPLITUDE)
        .enumerate()
    {
        assert_close(&format!("amplitude[{index}]"), *actual, expected, 2e-12);
    }
    for (index, (actual, expected)) in solution
        .phase_degrees
        .iter()
        .zip(EXPECTED_PHASE_DEGREES)
        .enumerate()
    {
        assert_close(&format!("phase[{index}]"), *actual, expected, 2e-9);
    }
    assert_close("mean", solution.mean, EXPECTED_MEAN, 2e-12);
    assert_close(
        "slope_per_day",
        solution.slope_per_day,
        EXPECTED_SLOPE_PER_DAY,
        2e-12,
    );
}
