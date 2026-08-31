//! Exact corrected-basis parity against the pinned Python `UTide` oracle.

use rutide_core::{GreenwichNodalBatch, GreenwichNodalOls, LinearConfidence, TidalConstituent};

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
const EXPECTED_PERCENT_ENERGY: [f64; 5] = [
    82.042_836_434_238_57,
    9.829_897_310_697_666,
    4.791_299_617_550_922,
    2.427_610_428_601_086_7,
    0.908_356_208_911_755_3,
];
const EXPECTED_COLORED_AMPLITUDE_CI: [f64; 5] = [
    0.015_535_798_974_675_425,
    0.015_537_235_572_573_34,
    0.015_537_821_673_906_426,
    0.010_076_433_270_712_83,
    0.010_093_351_511_400_065,
];
const EXPECTED_COLORED_PHASE_CI: [f64; 5] = [
    1.361_407_136_642_137_4,
    3.932_726_094_588_971,
    5.632_810_429_209_694,
    5.144_722_957_722_21,
    8.396_448_378_163_738,
];
const EXPECTED_COLORED_SNR: [f64; 5] = [
    6_804.089_537_467_822,
    815.075_838_563_201_3,
    397.255_211_785_300_16,
    478.588_068_898_858_9,
    178.476_866_307_692_97,
];
const EXPECTED_WHITE_AMPLITUDE_CI: [f64; 5] = [
    0.014_416_494_272_349_716,
    0.014_786_379_430_783_9,
    0.014_462_354_328_718_091,
    0.016_040_024_362_452_69,
    0.016_925_748_523_369_234,
];
const EXPECTED_WHITE_PHASE_CI: [f64; 5] = [
    1.263_322_100_120_534_1,
    3.742_672_237_948_587_3,
    5.242_929_285_933_047,
    8.189_552_727_926_335,
    14.080_176_795_362_373,
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
    for (index, (actual, expected)) in solution
        .percent_energy
        .iter()
        .zip(EXPECTED_PERCENT_ENERGY)
        .enumerate()
    {
        assert_close(
            &format!("percent_energy[{index}]"),
            *actual,
            expected,
            1e-11,
        );
    }
    assert_eq!(
        solution.constituent_indices_by_percent_energy(),
        [0, 1, 2, 3, 4]
    );
    assert_close("mean", solution.mean, EXPECTED_MEAN, 3e-12);
    assert_close(
        "slope_per_day",
        solution.slope_per_day,
        EXPECTED_SLOPE_PER_DAY,
        3e-12,
    );
}

#[test]
fn matches_python_utide_linear_confidence_and_derived_snr() {
    let time = oracle_times();
    let observations = oracle_observations();
    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid corrected oracle model");

    let colored = model
        .solve_with_linear_confidence(&observations, LinearConfidence::Colored)
        .expect("valid colored confidence model");
    for (label, actual, expected, tolerance) in [
        (
            "colored amplitude CI",
            colored.amplitude_ci.as_ref().expect("amplitude CI"),
            &EXPECTED_COLORED_AMPLITUDE_CI,
            2e-11,
        ),
        (
            "colored phase CI",
            colored.phase_ci_degrees.as_ref().expect("phase CI"),
            &EXPECTED_COLORED_PHASE_CI,
            1e-8,
        ),
        (
            "colored SNR",
            colored.signal_to_noise.as_ref().expect("SNR"),
            &EXPECTED_COLORED_SNR,
            2e-5,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }

    let white = model
        .solve_with_linear_confidence(&observations, LinearConfidence::White)
        .expect("valid white confidence model");
    for (label, actual, expected, tolerance) in [
        (
            "white amplitude CI",
            white.amplitude_ci.as_ref().expect("amplitude CI"),
            &EXPECTED_WHITE_AMPLITUDE_CI,
            2e-11,
        ),
        (
            "white phase CI",
            white.phase_ci_degrees.as_ref().expect("phase CI"),
            &EXPECTED_WHITE_PHASE_CI,
            1e-8,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    assert_eq!(
        colored.constituent_indices_by_signal_to_noise(),
        Some(vec![0, 1, 3, 2, 4])
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
