//! Exact corrected-basis parity against the pinned Python `UTide` oracle.

use rutide_core::{
    AnalysisError, GreenwichNodalBatch, GreenwichNodalOls, GreenwichNodalReconstructor,
    LinearConfidence, ReconstructionFilter, TidalConstituent,
};

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
const RECONSTRUCTION_TIMES: [f64; 6] = [
    58_113.0,
    58_113.0 + 0.5 / 24.0,
    58_120.25,
    58_144.0,
    58_144.5,
    58_150.125,
];
const EXPECTED_RECONSTRUCTION_ALL: [f64; 6] = [
    0.390_104_863_158_723_7,
    0.343_246_939_718_997_2,
    1.119_821_622_320_454_7,
    0.472_351_041_373_717,
    0.470_951_167_466_406_86,
    -0.294_077_952_460_847_2,
];
const EXPECTED_RECONSTRUCTION_M2_K1: [f64; 6] = [
    0.614_214_411_844_286_9,
    0.647_585_514_494_066_5,
    0.784_133_854_208_182_3,
    0.467_730_422_346_794_96,
    0.468_482_583_980_166_84,
    -0.092_424_856_881_404_75,
];
const EXPECTED_RECONSTRUCTION_DIAGNOSTIC: [f64; 6] = [
    0.560_571_894_280_522_9,
    0.548_679_775_063_807,
    0.879_754_392_912_107,
    0.380_275_915_277_449_97,
    0.259_596_331_751_810_85,
    -0.289_606_411_866_790_26,
];
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

fn synthetic_vector_observations(time: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let reference = time[0].midpoint(time[time.len() - 1]);
    let mut eastward = Vec::with_capacity(time.len());
    let mut northward = Vec::with_capacity(time.len());
    for (index, time) in time.iter().copied().enumerate() {
        let index = f64::from(u32::try_from(index).expect("fixture index fits u32"));
        eastward.push(
            0.15 + 0.0008 * (time - reference)
                + 0.42 * (index / 11.0).sin()
                + 0.17 * (index / 37.0).cos()
                + 0.05 * (index / 3.7).sin(),
        );
        northward.push(
            -0.08 - 0.0003 * (time - reference) + 0.31 * (index / 13.0).cos()
                - 0.12 * (index / 29.0).sin()
                + 0.04 * (index / 4.1).cos(),
        );
    }
    (eastward, northward)
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
fn matches_python_utide_reconstruction_and_filters_at_original_and_held_out_times() {
    let time = oracle_times();
    let observations = oracle_observations();
    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid corrected oracle model");
    let solution = model
        .solve_with_linear_confidence(&observations, LinearConfidence::Colored)
        .expect("valid colored solution");

    for (label, filter, expected) in [
        (
            "all",
            ReconstructionFilter::All,
            &EXPECTED_RECONSTRUCTION_ALL,
        ),
        (
            "M2,K1",
            ReconstructionFilter::Constituents(vec![TidalConstituent::M2, TidalConstituent::K1]),
            &EXPECTED_RECONSTRUCTION_M2_K1,
        ),
        (
            "PE and SNR",
            ReconstructionFilter::Diagnostics {
                minimum_percent_energy: 5.0,
                minimum_signal_to_noise: Some(500.0),
            },
            &EXPECTED_RECONSTRUCTION_DIAGNOSTIC,
        ),
    ] {
        let actual = model
            .reconstruct_modified_julian_days(&RECONSTRUCTION_TIMES, &solution, &filter)
            .expect("valid reconstruction");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("{label} reconstruction[{index}]"),
                *actual,
                *expected,
                5e-10,
            );
        }
    }

    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)
        .expect("valid corrected oracle batch");
    let reconstructor = batch
        .reconstructor_modified_julian_days(&RECONSTRUCTION_TIMES)
        .expect("valid batch reconstruction basis");
    let batch_values = reconstructor
        .reconstruct_many_series_major(
            std::slice::from_ref(&solution),
            &[LATITUDE_DEGREES_NORTH],
            &ReconstructionFilter::All,
        )
        .expect("valid batch reconstruction");
    let standalone = GreenwichNodalReconstructor::prepare_modified_julian_days(
        &RECONSTRUCTION_TIMES,
        model.reference_time_modified_julian_day(),
        &CONSTITUENTS,
    )
    .expect("valid standalone reconstruction basis")
    .reconstruct_at_latitude(
        &solution,
        LATITUDE_DEGREES_NORTH,
        &ReconstructionFilter::All,
    )
    .expect("valid standalone reconstruction");
    assert_eq!(batch_values, [standalone]);

    let no_confidence = model.solve(&observations).expect("valid solution");
    assert_eq!(
        model.reconstruct_modified_julian_days(
            &RECONSTRUCTION_TIMES,
            &no_confidence,
            &ReconstructionFilter::Diagnostics {
                minimum_percent_energy: 0.0,
                minimum_signal_to_noise: Some(2.0),
            },
        ),
        Err(AnalysisError::MissingSignalToNoise)
    );
}

#[test]
fn matches_python_utide_for_gappy_equidistant_scalar_observations() {
    let time = oracle_times();
    let mut observations = oracle_observations();
    for index in [0, 137, 411] {
        observations[index] = f64::NAN;
    }
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)
        .expect("valid corrected oracle batch");
    let solution = batch
        .solve_time_major_with_missing_and_linear_confidence(
            &observations,
            &[LATITUDE_DEGREES_NORTH],
            LinearConfidence::Colored,
        )
        .expect("valid gappy colored solution")
        .pop()
        .expect("one solution");

    for (label, actual, expected, tolerance) in [
        (
            "gappy amplitude",
            &solution.amplitude,
            &[
                0.654_472_399_747_378_1,
                0.225_234_032_280_502_12,
                0.157_231_287_210_143_68,
                0.111_716_028_794_704_59,
                0.067_578_131_538_407_5,
            ],
            3e-12,
        ),
        (
            "gappy colored amplitude CI",
            solution.amplitude_ci.as_ref().expect("amplitude CI"),
            &[
                0.015_585_131_683_005_718,
                0.015_561_465_937_189_782,
                0.015_555_229_620_349_435,
                0.010_021_657_250_895_571,
                0.010_042_170_306_974_657,
            ],
            2e-11,
        ),
        (
            "gappy colored phase CI",
            solution.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                1.360_097_443_107_087_4,
                3.958_124_784_909_186_3,
                5.672_291_016_064_416,
                5.154_105_365_276_658,
                8.503_072_256_489_247,
            ],
            1e-8,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    assert_close("gappy mean", solution.mean, 0.091_441_800_298_683_73, 3e-12);
    assert_close(
        "gappy slope",
        solution.slope_per_day,
        0.001_680_389_880_195_352_6,
        3e-12,
    );
    assert_close(
        "gappy reference time",
        solution.reference_time_days,
        58_128.5,
        0.0,
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one oracle test keeps the complete Python vector reference together"
)]
fn matches_python_utide_for_vector_ellipse_and_linear_confidence() {
    let time = oracle_times();
    let (eastward, northward) = synthetic_vector_observations(&time);
    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid vector oracle model");
    let solution = model
        .solve_vector_with_linear_confidence(&eastward, &northward, LinearConfidence::Colored)
        .expect("valid vector solution");
    for (label, actual, expected, tolerance) in [
        (
            "semi-major",
            &solution.semi_major,
            &[
                0.002_486_233_839_933_877_7,
                0.002_666_631_814_593_371,
                0.001_976_887_831_240_454,
                0.003_275_660_317_339_441,
                0.043_164_565_030_656_69,
            ],
            3e-12,
        ),
        (
            "semi-minor",
            &solution.semi_minor,
            &[
                0.000_092_720_326_357_076_47,
                -0.000_777_074_217_725_045,
                0.001_283_217_988_927_234_3,
                -0.001_555_730_689_804_322_1,
                -0.007_662_224_838_614_503,
            ],
            3e-12,
        ),
        (
            "inclination",
            &solution.inclination_degrees,
            &[
                22.873_737_196_375_4,
                15.895_822_755_234_192,
                45.248_612_515_338_9,
                160.750_488_449_543_4,
                84.464_012_966_697_37,
            ],
            3e-9,
        ),
        (
            "vector phase",
            &solution.phase_degrees,
            &[
                253.895_020_678_863_92,
                101.169_510_998_694_63,
                333.136_727_463_567_07,
                4.911_679_784_761_191,
                173.409_132_482_763_14,
            ],
            3e-9,
        ),
        (
            "semi-major CI",
            solution.semi_major_ci.as_ref().expect("major CI"),
            &[
                0.030_245_151_900_170_797,
                0.032_399_581_824_420_73,
                0.023_184_999_745_889_23,
                0.034_460_336_365_175_95,
                0.003_814_729_197_693_577_4,
            ],
            1e-9,
        ),
        (
            "semi-minor CI",
            solution.semi_minor_ci.as_ref().expect("minor CI"),
            &[
                0.012_764_231_471_522_686,
                0.009_221_318_938_658_544,
                0.023_382_510_755_167_354,
                0.012_102_614_259_441_032,
                0.038_311_096_549_943_98,
            ],
            1e-9,
        ),
        (
            "inclination CI",
            solution
                .inclination_ci_degrees
                .as_ref()
                .expect("inclination CI"),
            &[
                295.708_730_458_798_1,
                309.785_271_593_489_14,
                1_392.701_460_726_081_7,
                460.086_677_051_758_7,
                52.592_221_479_106_08,
            ],
            1e-6,
        ),
        (
            "vector phase CI",
            solution.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                698.071_275_281_994_2,
                762.332_233_985_891_9,
                1_387.680_485_030_409,
                791.616_825_599_554_2,
                10.696_586_935_146_975,
            ],
            1e-6,
        ),
        (
            "vector SNR",
            solution.signal_to_noise.as_ref().expect("SNR"),
            &[
                0.022_064_998_050_471_282,
                0.026_117_339_241_458_662,
                0.019_680_292_215_999_23,
                0.037_869_900_296_619_72,
                4.980_886_894_475_968,
            ],
            1e-9,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    assert_close(
        "eastward mean",
        solution.eastward_mean,
        0.163_543_485_854_886_54,
        3e-12,
    );
    assert_close(
        "northward mean",
        solution.northward_mean,
        -0.076_833_155_211_310_57,
        3e-12,
    );
    assert_close(
        "eastward slope",
        solution.eastward_slope_per_day,
        0.000_755_094_997_466_368_3,
        3e-12,
    );
    assert_close(
        "northward slope",
        solution.northward_slope_per_day,
        0.001_988_433_323_868_402_5,
        3e-12,
    );
    let reconstruction = model
        .reconstruct_vector_modified_julian_days(
            &[58_113.0, 58_120.25, 58_144.5],
            &solution,
            &ReconstructionFilter::All,
        )
        .expect("valid vector reconstruction");
    for (label, actual, expected) in [
        (
            "eastward reconstruction",
            reconstruction.eastward,
            [
                0.154_198_324_752_660_13,
                0.151_328_812_754_194_66,
                0.179_997_581_693_468_05,
            ],
        ),
        (
            "northward reconstruction",
            reconstruction.northward,
            [
                -0.069_338_712_966_28,
                -0.092_497_626_762_27,
                -0.059_352_174_782_471_005,
            ],
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, expected, 3e-12);
        }
    }
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
