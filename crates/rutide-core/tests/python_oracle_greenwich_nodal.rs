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
const EXPANDED_NAMES: [&str; 10] = ["Q1", "O1", "P1", "K1", "N2", "M2", "S2", "K2", "MK3", "M4"];
const EXPANDED_FREQUENCY_CPH: [f64; 10] = [
    0.037_218_502_477_472_055,
    0.038_730_654_449_939_97,
    0.041_552_587_111_696_106,
    0.041_780_746_221_637_22,
    0.078_999_248_699_109_28,
    0.080_511_400_671_577_2,
    0.083_333_333_333_333_33,
    0.083_561_492_443_274_45,
    0.122_292_146_893_214_41,
    0.161_022_801_343_154_4,
];
const EXPANDED_AMPLITUDE: [f64; 10] = [
    0.035_408_505_012_405_75,
    0.072_258_524_679_194_01,
    0.026_258_892_235_930_73,
    0.087_464_009_386_541_1,
    0.156_822_497_713_486_1,
    0.649_924_641_080_296_1,
    0.282_598_348_134_895_54,
    0.117_313_123_756_242_74,
    0.002_780_611_410_494_215_5,
    0.005_178_796_242_056_03,
];
const EXPANDED_PHASE_DEGREES: [f64; 10] = [
    317.416_502_976_933_9,
    20.895_690_476_510_218,
    149.823_972_269_302_46,
    146.794_774_856_902_93,
    162.836_312_961_680_96,
    189.720_455_779_160_8,
    212.940_800_268_409_96,
    195.883_783_281_502_93,
    20.194_826_642_368_177,
    63.294_262_613_617_256,
];

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

#[test]
fn matches_python_utide_for_expanded_base_and_shallow_constituents() {
    let time = oracle_times();
    let observations = oracle_observations();
    let constituents =
        EXPANDED_NAMES.map(|name| name.parse::<TidalConstituent>().expect("catalog name"));
    assert!(constituents[8].is_shallow());
    assert!(constituents[9].is_shallow());

    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &constituents,
    )
    .expect("valid expanded oracle model");
    let solution = model.solve(&observations).expect("valid observations");

    for (index, (constituent, expected)) in model
        .constituents()
        .iter()
        .zip(EXPANDED_FREQUENCY_CPH)
        .enumerate()
    {
        assert_close(
            &format!("frequency_cph[{index}]"),
            constituent.frequency_cph,
            expected,
            1e-15,
        );
    }
    for (index, (actual, expected)) in solution
        .amplitude
        .iter()
        .zip(EXPANDED_AMPLITUDE)
        .enumerate()
    {
        assert_close(&format!("amplitude[{index}]"), *actual, expected, 5e-12);
    }
    for (index, (actual, expected)) in solution
        .phase_degrees
        .iter()
        .zip(EXPANDED_PHASE_DEGREES)
        .enumerate()
    {
        assert_close(&format!("phase[{index}]"), *actual, expected, 5e-9);
    }
    assert_close(
        "expanded mean",
        solution.mean,
        0.090_589_928_587_125_18,
        5e-12,
    );
    assert_close(
        "expanded slope_per_day",
        solution.slope_per_day,
        0.001_777_072_665_086_449_2,
        5e-12,
    );
}
