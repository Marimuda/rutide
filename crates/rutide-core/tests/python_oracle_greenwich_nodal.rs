//! Exact corrected-basis parity against the pinned Python `UTide` oracle.

use rayon::ThreadPoolBuilder;
use rutide_core::{
    AnalysisError, FitOptions, GreenwichNodalBatch, GreenwichNodalOls, GreenwichNodalReconstructor,
    LinearConfidence, MonteCarloOptions, ReconstructionFilter, RobustOptions, TidalConstituent,
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

fn irregular_oracle_times() -> Vec<f64> {
    let mut times = oracle_times();
    let final_index = times.len() - 1;
    for (index, time) in times.iter_mut().enumerate().take(final_index).skip(1) {
        let index = f64::from(u32::try_from(index).expect("fixture index fits u32"));
        *time += 0.002 * (index * 0.37).sin() + 0.0007 * (index * 0.11).cos();
    }
    times
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

fn assert_relative_close(label: &str, actual: f64, expected: f64, relative_tolerance: f64) {
    let tolerance = relative_tolerance * expected.abs().max(f64::MIN_POSITIVE);
    assert_close(label, actual, expected, tolerance);
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
#[allow(
    clippy::too_many_lines,
    reason = "freezes scalar and vector coefficients plus colored intervals for Python trend=False"
)]
fn matches_python_utide_with_trend_disabled() {
    let time = oracle_times();
    let observations = oracle_observations();
    let options = FitOptions { trend: false };
    let model = GreenwichNodalOls::prepare_modified_julian_days_with_options(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
        options,
    )
    .expect("valid trend-disabled model");
    let scalar = model
        .solve_with_linear_confidence(&observations, LinearConfidence::Colored)
        .expect("valid trend-disabled scalar solution");
    assert_eq!(model.fit_options(), options);
    for (field, actual, expected, tolerance) in [
        (
            "amplitude",
            &scalar.amplitude,
            &[
                0.653_688_008_517_015_4,
                0.226_499_047_317_034_44,
                0.158_185_721_266_017_32,
                0.112_154_840_796_228_5,
                0.069_248_097_197_129_27,
            ],
            5e-11,
        ),
        (
            "phase",
            &scalar.phase_degrees,
            &[
                189.390_582_953_384_47,
                229.039_448_491_749_17,
                163.365_202_479_752_87,
                154.510_882_931_514_06,
                20.326_520_499_173_974,
            ],
            5e-8,
        ),
        (
            "amplitude CI",
            scalar.amplitude_ci.as_ref().expect("amplitude CI"),
            &[
                0.015_573_715_781_056_981,
                0.015_574_797_226_646_77,
                0.015_575_490_201_899_739,
                0.010_021_669_866_443_34,
                0.010_038_253_524_617_6,
            ],
            5e-9,
        ),
        (
            "phase CI",
            scalar.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                1.364_964_903_876_680_2,
                3.939_085_971_636_990_4,
                5.639_949_666_784_921,
                5.130_549_625_688_577,
                8.295_769_367_692_161,
            ],
            5e-7,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("no-trend {field}[{index}]"),
                *actual,
                *expected,
                tolerance,
            );
        }
    }
    assert_close(
        "no-trend mean",
        scalar.mean,
        0.091_040_665_648_788_57,
        5e-11,
    );
    assert_close("no-trend slope", scalar.slope_per_day, 0.0, f64::EPSILON);

    let batch = GreenwichNodalBatch::prepare_modified_julian_days_with_options(
        &time,
        &CONSTITUENTS,
        options,
    )
    .expect("valid trend-disabled batch");
    assert_eq!(batch.fit_options(), options);
    let batch_solution = batch
        .solve_time_major_with_linear_confidence(
            &observations,
            &[LATITUDE_DEGREES_NORTH],
            LinearConfidence::Colored,
        )
        .expect("valid batch solution");
    assert_eq!(batch_solution.as_slice(), std::slice::from_ref(&scalar));

    let (eastward, northward) = synthetic_vector_observations(&time);
    let vector = model
        .solve_vector_with_linear_confidence(&eastward, &northward, LinearConfidence::Colored)
        .expect("valid trend-disabled vector solution");
    for (field, actual, expected, tolerance) in [
        (
            "semi-major",
            &vector.semi_major,
            &[
                0.002_291_214_337_322_088_4,
                0.002_505_978_497_015_262,
                0.001_756_108_956_550_487,
                0.003_274_763_239_083_977_4,
                0.042_862_136_703_990_93,
            ],
            5e-10,
        ),
        (
            "semi-minor",
            &vector.semi_minor,
            &[
                0.000_128_699_131_225_115_54,
                -0.000_724_357_429_841_571_2,
                0.001_328_799_778_751_671_5,
                -0.000_946_774_273_530_124_6,
                -0.007_503_205_767_192_839,
            ],
            5e-10,
        ),
        (
            "semi-major CI",
            vector.semi_major_ci.as_ref().expect("major CI"),
            &[
                0.031_287_706_568_072_65,
                0.033_176_453_074_374_154,
                0.024_002_310_153_427_48,
                0.034_054_810_054_763_54,
                0.003_630_607_578_576_585_6,
            ],
            5e-8,
        ),
        (
            "semi-minor CI",
            vector.semi_minor_ci.as_ref().expect("minor CI"),
            &[
                0.009_880_436_014_417_356,
                0.005_739_563_754_980_892,
                0.022_518_491_594_073_757,
                0.013_152_688_247_846_097,
                0.038_303_286_451_683_224,
            ],
            5e-8,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("no-trend {field}[{index}]"),
                *actual,
                *expected,
                tolerance,
            );
        }
    }
    assert_close(
        "no-trend eastward mean",
        vector.eastward_mean,
        0.163_543_475_144_704,
        5e-11,
    );
    assert_close(
        "no-trend northward mean",
        vector.northward_mean,
        -0.076_833_183_415_025_49,
        5e-11,
    );
    assert_close(
        "no-trend eastward slope",
        vector.eastward_slope_per_day,
        0.0,
        f64::EPSILON,
    );
    assert_close(
        "no-trend northward slope",
        vector.northward_slope_per_day,
        0.0,
        f64::EPSILON,
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "freezes coefficients, weights, and confidence oracle values"
)]
fn matches_python_utide_cauchy_robust_scalar_fit() {
    let time = oracle_times();
    let mut observations = oracle_observations();
    for (index, offset) in [(71, 5.0), (218, -4.0), (503, 6.0)] {
        observations[index] += offset;
    }
    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid corrected robust model");
    let solution = model
        .solve_robust(&observations, RobustOptions::default())
        .expect("converged robust solution");

    for (index, (actual, expected)) in solution
        .amplitude
        .iter()
        .zip([
            0.658_145_852_459_635_5,
            0.238_805_188_209_338_16,
            0.156_177_076_824_299_6,
            0.106_458_942_117_917_18,
            0.061_386_581_707_001_765,
        ])
        .enumerate()
    {
        assert_close(
            &format!("robust amplitude[{index}]"),
            *actual,
            expected,
            3e-11,
        );
    }
    for (index, (actual, expected)) in solution
        .phase_degrees
        .iter()
        .zip([
            189.139_324_737_893_02,
            231.567_611_148_729_24,
            161.293_601_526_333_96,
            153.739_401_353_766_28,
            22.440_374_521_922_72,
        ])
        .enumerate()
    {
        assert_close(&format!("robust phase[{index}]"), *actual, expected, 3e-9);
    }
    assert_close(
        "robust mean",
        solution.mean,
        0.089_398_869_808_239_88,
        3e-11,
    );
    assert_close(
        "robust slope",
        solution.slope_per_day,
        0.001_264_652_294_300_096_9,
        3e-12,
    );

    let diagnostics = solution.robust.as_ref().expect("robust diagnostics");
    assert_eq!(diagnostics.iterations, 5);
    let indices = [0, 1, 70, 71, 72, 217, 218, 219, 502, 503, 504, 744];
    let expected_weights = [
        0.374_888_711_962_230_84,
        0.612_494_258_822_018_7,
        0.962_171_726_691_640_4,
        0.004_055_824_836_021_76,
        0.987_574_556_280_989_9,
        0.923_011_380_567_195_2,
        0.006_517_669_413_056_472,
        0.930_571_095_455_172_4,
        0.637_865_641_244_842_8,
        0.002_562_676_748_444_911,
        0.610_158_416_922_485_6,
        0.574_118_461_172_164_1,
    ];
    for (index, expected) in indices.into_iter().zip(expected_weights) {
        assert_close(
            &format!("robust weight[{index}]"),
            diagnostics.weights[index],
            expected,
            3e-10,
        );
    }
    assert_close(
        "robust weight sum",
        diagnostics.weights.iter().sum(),
        640.463_424_939_137,
        2e-8,
    );
    assert_close(
        "robust OLS RMS",
        diagnostics.ols_rms_residual,
        0.353_713_248_377_981_35,
        3e-11,
    );

    let colored = model
        .solve_robust_with_linear_confidence(
            &observations,
            RobustOptions::default(),
            LinearConfidence::Colored,
        )
        .expect("converged robust colored solution");
    for (label, actual, expected, tolerance) in [
        (
            "robust colored amplitude CI",
            colored.amplitude_ci.as_ref().expect("amplitude CI"),
            &[
                0.010_385_114_891_016_55,
                0.010_373_818_074_631_45,
                0.010_387_623_062_905_044,
                0.006_423_739_604_029_797,
                0.006_450_801_786_618_674,
            ],
            3e-10,
        ),
        (
            "robust colored phase CI",
            colored.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                0.902_627_245_321_272_8,
                2.490_347_825_733_839,
                3.802_839_792_387_647_6,
                3.478_522_677_579_728_7,
                6.007_376_471_310_384,
            ],
            3e-8,
        ),
        (
            "robust colored SNR",
            colored.signal_to_noise.as_ref().expect("SNR"),
            &[
                15_428.859_677_577_424,
                2_035.740_437_742_849_5,
                868.389_089_955_865_3,
                1_055.116_900_744_206,
                347.881_751_992_609_4,
            ],
            3e-4,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
}

#[test]
fn matches_python_utide_for_a_sustained_scalar_outlier() {
    let time = oracle_times();
    let mut observations = oracle_observations();
    for value in &mut observations[300..331] {
        *value += 2.5;
    }
    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid sustained-outlier model");
    let solution = model
        .solve_robust_with_linear_confidence(
            &observations,
            RobustOptions::default(),
            LinearConfidence::Colored,
        )
        .expect("converged sustained-outlier solution");

    for (label, actual, expected, tolerance) in [
        (
            "sustained robust amplitude",
            &solution.amplitude,
            &[
                0.658_342_939_774_640_9,
                0.237_586_228_525_747_96,
                0.154_849_024_020_135_53,
                0.106_145_206_841_438_44,
                0.062_397_163_868_370_305,
            ],
            3e-11,
        ),
        (
            "sustained robust phase",
            &solution.phase_degrees,
            &[
                188.968_537_730_831_88,
                231.314_689_016_924_43,
                161.007_478_621_623_1,
                154.146_371_902_974_8,
                19.648_934_651_665_137,
            ],
            3e-8,
        ),
        (
            "sustained robust amplitude CI",
            solution.amplitude_ci.as_ref().expect("amplitude CI"),
            &[
                0.009_980_564_103_119_892,
                0.009_974_265_253_914_148,
                0.009_985_259_389_109_383,
                0.006_986_736_584_833_989,
                0.007_007_945_389_750_689,
            ],
            3e-10,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    assert_close(
        "sustained robust mean",
        solution.mean,
        0.097_490_406_999_837_45,
        3e-11,
    );
    assert_close(
        "sustained robust slope",
        solution.slope_per_day,
        0.001_091_945_399_845_701_5,
        3e-12,
    );
    let diagnostics = solution.robust.as_ref().expect("robust diagnostics");
    assert_eq!(diagnostics.iterations, 6);
    assert_close(
        "sustained robust weight sum",
        diagnostics.weights.iter().sum(),
        621.260_918_730_234_7,
        2e-8,
    );
    for (index, expected) in [
        (299, 0.707_734_756_258_129_8),
        (300, 0.018_265_131_761_312_274),
        (315, 0.017_999_228_385_467_99),
        (330, 0.017_047_553_732_599_69),
        (331, 0.886_312_389_678_020_5),
    ] {
        assert_close(
            &format!("sustained robust weight[{index}]"),
            diagnostics.weights[index],
            expected,
            3e-10,
        );
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "freezes ellipse, shared-weight, and confidence oracle values"
)]
fn matches_python_utide_cauchy_robust_vector_fit() {
    let time = oracle_times();
    let (mut eastward, mut northward) = synthetic_vector_observations(&time);
    eastward[71] += 5.0;
    northward[218] -= 4.0;
    eastward[503] += 4.0;
    northward[503] += 3.0;
    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid corrected robust vector model");
    let solution = model
        .solve_vector_robust(&eastward, &northward, RobustOptions::default())
        .expect("converged robust vector solution");

    for (label, actual, expected, tolerance) in [
        (
            "robust major",
            &solution.semi_major,
            &[
                0.001_713_330_656_679_418_2,
                0.001_581_357_412_332_718_6,
                0.001_917_337_390_693_039_4,
                0.009_257_122_602_617_771,
                0.041_361_184_676_057_525,
            ],
            3e-11,
        ),
        (
            "robust minor",
            &solution.semi_minor,
            &[
                0.000_315_415_181_843_113_04,
                0.000_344_199_099_888_868_23,
                -0.000_762_919_912_225_096_1,
                -0.001_007_737_463_980_055_6,
                0.001_563_596_907_072_053_3,
            ],
            3e-11,
        ),
        (
            "robust inclination",
            &solution.inclination_degrees,
            &[
                54.734_344_768_150_784,
                62.422_775_315_423_41,
                140.474_675_685_810_9,
                78.088_619_298_543_46,
                83.703_604_289_762_1,
            ],
            3e-8,
        ),
        (
            "robust vector phase",
            &solution.phase_degrees,
            &[
                272.396_149_942_525_1,
                40.766_579_729_213_575,
                358.529_955_598_924_74,
                67.998_280_349_531_38,
                170.709_998_564_574_33,
            ],
            3e-8,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    for (label, actual, expected) in [
        (
            "robust eastward mean",
            solution.eastward_mean,
            0.165_074_288_673_386_08,
        ),
        (
            "robust northward mean",
            solution.northward_mean,
            -0.075_146_691_017_529_74,
        ),
        (
            "robust eastward slope",
            solution.eastward_slope_per_day,
            0.000_739_585_526_115_897_6,
        ),
        (
            "robust northward slope",
            solution.northward_slope_per_day,
            0.002_161_909_613_034_094_5,
        ),
    ] {
        assert_close(label, actual, expected, 3e-11);
    }
    let diagnostics = solution.robust.as_ref().expect("robust diagnostics");
    assert_eq!(diagnostics.iterations, 2);
    let indices = [0, 1, 70, 71, 72, 217, 218, 219, 502, 503, 504, 744];
    let expected_weights = [
        0.938_389_168_459_219_7,
        0.927_385_799_668_620_8,
        0.985_308_419_455_378_7,
        0.080_626_786_320_981_97,
        0.985_778_464_704_937,
        0.887_028_441_025_758_5,
        0.105_719_095_502_464_94,
        0.879_609_883_903_025_9,
        0.905_775_564_509_730_2,
        0.067_068_919_213_801_43,
        0.922_123_625_774_698,
        0.915_197_903_015_222_5,
    ];
    for (index, expected) in indices.into_iter().zip(expected_weights) {
        assert_close(
            &format!("robust vector weight[{index}]"),
            diagnostics.weights[index],
            expected,
            3e-10,
        );
    }
    assert_close(
        "robust vector weight sum",
        diagnostics.weights.iter().sum(),
        690.821_363_451_973_6,
        2e-8,
    );

    let colored = model
        .solve_vector_robust_with_linear_confidence(
            &eastward,
            &northward,
            RobustOptions::default(),
            LinearConfidence::Colored,
        )
        .expect("converged robust vector colored solution");
    for (label, actual, expected, tolerance) in [
        (
            "robust major CI",
            colored.semi_major_ci.as_ref().expect("major CI"),
            &[
                0.019_021_861_841_394_26,
                0.015_663_060_754_516_75,
                0.025_512_587_765_766_263,
                0.007_767_036_245_261_25,
                0.004_587_458_651_368_575,
            ],
            3e-9,
        ),
        (
            "robust minor CI",
            colored.semi_minor_ci.as_ref().expect("minor CI"),
            &[
                0.026_922_542_442_636_678,
                0.029_953_928_792_999_622,
                0.021_023_072_531_914_273,
                0.035_905_748_209_963_73,
                0.038_382_077_594_627_93,
            ],
            3e-9,
        ),
        (
            "robust inclination CI",
            colored
                .inclination_ci_degrees
                .as_ref()
                .expect("inclination CI"),
            &[
                938.445_433_969_345_5,
                1_146.524_273_781_227,
                829.665_047_623_809,
                225.058_360_856_494_52,
                53.380_941_043_150_585,
            ],
            3e-5,
        ),
        (
            "robust vector phase CI",
            colored.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                681.239_538_040_506,
                645.224_183_570_930_8,
                952.023_907_436_457_8,
                54.457_715_882_511_83,
                6.659_258_879_053_22,
            ],
            3e-5,
        ),
        (
            "robust vector SNR",
            colored.signal_to_noise.as_ref().expect("SNR"),
            &[
                0.010_729_456_592_531_265,
                0.008_806_276_629_376_933,
                0.014_968_421_077_119_544,
                0.246_826_476_229_682_7,
                4.404_546_799_827_36,
            ],
            3e-7,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "keeps the scalar/vector robust Monte Carlo distribution fixture together"
)]
fn matches_python_utide_robust_monte_carlo_distributions() {
    let time = oracle_times();
    let mut scalar = oracle_observations();
    for value in &mut scalar[300..331] {
        *value += 2.5;
    }
    let (mut eastward, mut northward) = synthetic_vector_observations(&time);
    eastward[71] += 5.0;
    northward[218] -= 4.0;
    eastward[503] += 4.0;
    northward[503] += 3.0;
    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid robust Monte Carlo model");
    let monte_carlo = MonteCarloOptions {
        realizations: 200,
        seed: 20_260_901,
    };
    let scalar_solution = model
        .solve_robust_with_monte_carlo_confidence(
            &scalar,
            RobustOptions::default(),
            monte_carlo,
            LinearConfidence::Colored,
        )
        .expect("valid robust scalar Monte Carlo solution");
    let repeated_scalar = model
        .solve_robust_with_monte_carlo_confidence(
            &scalar,
            RobustOptions::default(),
            monte_carlo,
            LinearConfidence::Colored,
        )
        .expect("repeated robust scalar Monte Carlo solution");
    assert_eq!(scalar_solution, repeated_scalar);
    for (label, actual, expected) in [
        (
            "robust scalar amplitude",
            scalar_solution.amplitude_ci.as_ref().expect("amplitude CI"),
            [
                0.011_431_456_519_778_508,
                0.009_684_383_682_427_149,
                0.009_439_902_243_920_658,
                0.007_469_864_766_697_188,
                0.006_041_246_282_303_7,
            ],
        ),
        (
            "robust scalar phase",
            scalar_solution.phase_ci_degrees.as_ref().expect("phase CI"),
            [
                0.777_964_488_039_244_8,
                2.392_433_116_570_795_3,
                3.729_879_411_361_814_3,
                3.741_124_458_639_11,
                6.696_069_382_811_561,
            ],
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_relative_close(
                &format!("Python {label} distribution[{index}]"),
                *actual,
                expected,
                0.4,
            );
        }
    }

    let vector_solution = model
        .solve_vector_robust_with_monte_carlo_confidence(
            &eastward,
            &northward,
            RobustOptions::default(),
            monte_carlo,
            LinearConfidence::Colored,
        )
        .expect("valid robust vector Monte Carlo solution");
    let repeated_vector = model
        .solve_vector_robust_with_monte_carlo_confidence(
            &eastward,
            &northward,
            RobustOptions::default(),
            monte_carlo,
            LinearConfidence::Colored,
        )
        .expect("repeated robust vector Monte Carlo solution");
    assert_eq!(vector_solution, repeated_vector);
    for (label, actual, expected) in [
        (
            "robust semi-major",
            vector_solution.semi_major_ci.as_ref().expect("major CI"),
            [
                0.001_277_301_854_155_115_8,
                0.001_312_428_842_497_684_7,
                0.000_744_371_441_394_875,
                0.005_388_622_301_025_978_5,
                0.002_225_152_684_396_533,
            ],
        ),
        (
            "robust semi-minor",
            vector_solution.semi_minor_ci.as_ref().expect("minor CI"),
            [
                0.000_799_296_664_879_357_7,
                0.000_843_586_778_021_803_7,
                0.001_365_395_275_044_186_8,
                0.009_971_925_857_103_667,
                0.011_992_608_950_899_849,
            ],
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_relative_close(
                &format!("Python {label} distribution[{index}]"),
                *actual,
                expected,
                0.5,
            );
        }
    }
    for (label, actual, expected) in [
        (
            "robust inclination",
            vector_solution
                .inclination_ci_degrees
                .as_ref()
                .expect("inclination CI"),
            [
                25.804_778_818_913_476,
                29.111_393_916_986_2,
                55.116_758_138_846_265,
                81.886_804_436_116_32,
                17.385_958_019_492_02,
            ],
        ),
        (
            "robust phase",
            vector_solution.phase_ci_degrees.as_ref().expect("phase CI"),
            [
                61.364_398_346_163_63,
                56.614_228_443_396_534,
                36.135_518_507_707_41,
                31.273_255_490_590_09,
                3.578_451_831_924_899_5,
            ],
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("Python {label} distribution[{index}]"),
                *actual,
                expected,
                65.0,
            );
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "freezes the irregular missing-data robust oracle surface"
)]
fn matches_python_utide_irregular_gappy_robust_scalar_confidence() {
    let time = irregular_oracle_times();
    let mut observations = oracle_observations();
    for index in [0, 137, 411] {
        observations[index] = f64::NAN;
    }
    for (index, offset) in [(71, 5.0), (218, -4.0), (503, 6.0)] {
        observations[index] += offset;
    }
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)
        .expect("valid irregular robust batch");
    let solution = batch
        .solve_time_major_with_missing_robust_and_linear_confidence(
            &observations,
            &[LATITUDE_DEGREES_NORTH],
            RobustOptions::default(),
            LinearConfidence::Colored,
        )
        .expect("valid irregular gappy robust solution")
        .pop()
        .expect("one solution");

    for (label, actual, expected, tolerance) in [
        (
            "irregular robust amplitude",
            &solution.amplitude,
            &[
                0.657_927_742_975_213_3,
                0.238_701_910_993_591_12,
                0.155_956_854_288_977_46,
                0.106_736_908_837_608_04,
                0.060_364_270_942_384_876,
            ],
            3e-11,
        ),
        (
            "irregular robust phase",
            &solution.phase_degrees,
            &[
                189.122_416_169_234,
                231.677_635_699_404_2,
                161.153_429_767_961_37,
                153.132_226_705_290_66,
                23.340_449_062_890_297,
            ],
            3e-8,
        ),
        (
            "irregular robust amplitude CI",
            solution.amplitude_ci.as_ref().expect("amplitude CI"),
            &[
                0.010_414_147_275_322_905,
                0.010_391_256_399_231_398,
                0.010_402_205_691_361_75,
                0.006_490_397_571_268_867,
                0.006_517_219_634_566_622,
            ],
            3e-10,
        ),
        (
            "irregular robust phase CI",
            solution.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                0.903_503_456_428_789_4,
                2.495_805_738_183_243,
                3.815_965_584_190_676,
                3.506_987_579_226_474_4,
                6.175_706_127_678_937,
            ],
            3e-8,
        ),
        (
            "irregular robust SNR",
            solution.signal_to_noise.as_ref().expect("SNR"),
            &[
                15_332.787_333_378_024,
                2_027.158_993_881_378,
                863.515_627_774_444_5,
                1_038.959_896_509_877_4,
                329.569_753_913_328_04,
            ],
            3e-4,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    assert_close(
        "irregular robust mean",
        solution.mean,
        0.089_125_477_176_249_97,
        3e-11,
    );
    assert_close(
        "irregular robust slope",
        solution.slope_per_day,
        0.001_333_053_598_877_835_8,
        3e-12,
    );
    let diagnostics = solution.robust.as_ref().expect("robust diagnostics");
    assert_eq!(diagnostics.iterations, 5);
    assert_close(
        "irregular robust weight sum",
        diagnostics.weights.iter().sum(),
        636.801_555_578_963_6,
        2e-8,
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "freezes the irregular vector robust oracle surface"
)]
fn matches_python_utide_irregular_gappy_robust_vector_confidence() {
    let time = irregular_oracle_times();
    let (mut eastward, mut northward) = synthetic_vector_observations(&time);
    for index in [0, 137] {
        eastward[index] = f64::NAN;
    }
    for index in [2, 411] {
        northward[index] = f64::NAN;
    }
    eastward[71] += 5.0;
    northward[218] -= 4.0;
    eastward[503] += 4.0;
    northward[503] += 3.0;
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)
        .expect("valid irregular robust vector batch");
    let solution = batch
        .solve_vector_time_major_with_missing_robust_and_linear_confidence(
            &eastward,
            &northward,
            &[LATITUDE_DEGREES_NORTH],
            RobustOptions::default(),
            LinearConfidence::Colored,
        )
        .expect("valid irregular gappy robust vector solution")
        .pop()
        .expect("one solution");

    for (label, actual, expected, tolerance) in [
        (
            "irregular robust major",
            &solution.semi_major,
            &[
                0.004_115_156_844_539_959,
                0.005_218_003_164_742_43,
                0.002_326_072_643_694_429_6,
                0.009_857_054_281_994_973,
                0.039_498_350_563_153_73,
            ],
            3e-11,
        ),
        (
            "irregular robust minor",
            &solution.semi_minor,
            &[
                0.000_916_731_988_710_150_5,
                0.000_282_629_326_177_253_56,
                -0.000_064_581_611_139_997_3,
                -0.005_026_497_127_687_541,
                0.001_331_153_217_211_611_4,
            ],
            3e-11,
        ),
        (
            "irregular robust major CI",
            solution.semi_major_ci.as_ref().expect("major CI"),
            &[
                0.032_268_877_105_344_485,
                0.032_279_726_103_875_54,
                0.006_672_687_737_287_073,
                0.009_943_006_289_697_47,
                0.005_836_982_751_398_109,
            ],
            3e-9,
        ),
        (
            "irregular robust minor CI",
            solution.semi_minor_ci.as_ref().expect("minor CI"),
            &[
                0.006_813_687_190_281_185_5,
                0.009_993_422_011_621_261,
                0.032_353_448_821_291_76,
                0.035_365_207_874_963_035,
                0.038_206_098_592_043_87,
            ],
            3e-9,
        ),
        (
            "irregular robust inclination CI",
            solution
                .inclination_ci_degrees
                .as_ref()
                .expect("inclination CI"),
            &[
                144.814_112_360_895_54,
                111.724_914_567_179_78,
                798.004_037_913_649_1,
                280.939_164_739_871_8,
                55.653_325_452_181_85,
            ],
            3e-5,
        ),
        (
            "irregular robust phase CI",
            solution.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                471.984_783_023_444_9,
                355.545_710_314_022_2,
                165.884_275_265_114_33,
                161.864_038_243_268_65,
                8.654_641_100_289_224,
            ],
            3e-5,
        ),
        (
            "irregular robust vector SNR",
            solution.signal_to_noise.as_ref().expect("SNR"),
            &[
                0.062_778_047_938_033_27,
                0.091_872_243_248_639_93,
                0.019_061_670_487_992_278,
                0.348_495_694_077_026_4,
                4.016_774_717_067_462,
            ],
            3e-7,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    let diagnostics = solution.robust.as_ref().expect("robust diagnostics");
    assert_eq!(diagnostics.iterations, 3);
    assert_close(
        "irregular robust vector weight sum",
        diagnostics.weights.iter().sum(),
        681.384_349_164_300_4,
        2e-8,
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
#[allow(
    clippy::too_many_lines,
    reason = "the complete seeded scalar/vector Python distribution fixture is kept together"
)]
fn matches_python_utide_monte_carlo_distributions_and_is_seeded() {
    let time = oracle_times();
    let observations = oracle_observations();
    let (eastward, northward) = synthetic_vector_observations(&time);
    let model = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE_DEGREES_NORTH,
        &CONSTITUENTS,
    )
    .expect("valid corrected oracle model");
    let options = MonteCarloOptions {
        realizations: 200,
        seed: 20_260_901,
    };
    let python_scalar = [
        (
            [
                0.015_390_187_735_185_013,
                0.014_836_832_580_709_936,
                0.013_180_176_073_152_965,
                0.017_751_814_120_816_43,
                0.017_032_818_560_040_974,
            ],
            [
                1.077_601_599_482_886,
                3.494_841_552_403_972,
                5.276_068_182_594_5,
                7.793_996_513_296_821,
                12.539_001_774_507_195,
            ],
        ),
        (
            [
                0.016_600_237_688_198_165,
                0.015_590_733_709_989_545,
                0.014_168_073_155_724_216,
                0.011_308_226_576_140_666,
                0.010_074_011_256_927_688,
            ],
            [
                1.161_783_655_626_603_5,
                3.673_015_853_736_915_3,
                5.660_455_518_062_66,
                4.908_830_759_160_057,
                7.579_216_505_486_345,
            ],
        ),
    ];
    let python_vector = [
        (
            [
                0.019_116_448_084_726_47,
                0.019_828_077_985_657_63,
                0.021_674_660_608_658_535,
                0.023_583_065_303_905_526,
                0.026_280_447_483_522_176,
            ],
            [
                0.017_363_586_620_039_867,
                0.018_998_717_659_692_067,
                0.016_339_548_572_974_375,
                0.018_369_063_144_899_625,
                0.031_536_679_923_300_025,
            ],
            [
                127.801_151_729_750_15,
                147.205_186_135_319_93,
                171.578_067_238_242_14,
                147.743_267_306_170_74,
                61.443_804_943_222_21,
            ],
            [
                257.763_199_814_040_04,
                281.405_982_861_164_8,
                249.320_177_967_104_54,
                267.156_504_526_476_97,
                45.186_434_076_137_985,
            ],
        ),
        (
            [
                0.000_682_426_272_567_639_9,
                0.000_766_865_388_369_797_1,
                0.000_755_744_378_294_567,
                0.009_403_599_432_942_817,
                0.001_400_852_483_438_941_2,
            ],
            [
                0.000_250_856_434_448_247_1,
                0.000_251_238_099_642_961_2,
                0.000_342_506_221_146_407_5,
                0.002_587_006_800_046_218_3,
                0.011_678_164_704_629_158,
            ],
            [
                6.088_866_589_127_396,
                8.392_193_596_345_162,
                30.339_031_245_947_03,
                114.526_858_502_183_1,
                18.071_897_363_741_993,
            ],
            [
                20.947_486_290_729_515,
                16.929_445_892_401_93,
                38.401_645_027_718_12,
                129.185_762_626_979_3,
                3.468_237_859_626_629,
            ],
        ),
    ];
    let rust_seeded_scalar = [
        [
            0.014_818_281_824_986_958,
            0.014_529_575_594_899_912,
            0.013_507_954_506_873_678,
            0.015_281_429_463_492_627,
            0.018_867_370_059_998_73,
        ],
        [
            0.015_962_825_514_650_22,
            0.015_273_346_047_457_018,
            0.014_531_561_933_941_829,
            0.009_556_599_593_868_124,
            0.011_133_272_612_067_546,
        ],
    ];

    for (noise_index, noise) in [LinearConfidence::White, LinearConfidence::Colored]
        .into_iter()
        .enumerate()
    {
        let scalar = model
            .solve_with_monte_carlo_confidence(&observations, options, noise)
            .expect("scalar MC");
        let repeated_scalar = model
            .solve_with_monte_carlo_confidence(&observations, options, noise)
            .expect("repeated scalar MC");
        assert_eq!(scalar, repeated_scalar);
        let scalar_amplitude = scalar.amplitude_ci.as_ref().expect("scalar amplitude CI");
        let scalar_phase = scalar.phase_ci_degrees.as_ref().expect("scalar phase CI");
        for (index, ((actual, seeded), oracle)) in scalar_amplitude
            .iter()
            .zip(rust_seeded_scalar[noise_index])
            .zip(python_scalar[noise_index].0)
            .enumerate()
        {
            assert_close(
                &format!("seeded scalar amplitude[{index}]"),
                *actual,
                seeded,
                1e-14,
            );
            assert_relative_close(
                &format!("Python scalar amplitude distribution[{index}]"),
                *actual,
                oracle,
                0.3,
            );
        }
        for (index, (actual, oracle)) in scalar_phase
            .iter()
            .zip(python_scalar[noise_index].1)
            .enumerate()
        {
            assert_relative_close(
                &format!("Python scalar phase distribution[{index}]"),
                *actual,
                oracle,
                0.3,
            );
        }

        let vector = model
            .solve_vector_with_monte_carlo_confidence(&eastward, &northward, options, noise)
            .expect("vector MC");
        let repeated_vector = model
            .solve_vector_with_monte_carlo_confidence(&eastward, &northward, options, noise)
            .expect("repeated vector MC");
        assert_eq!(vector, repeated_vector);
        for (label, actual, oracle) in [
            (
                "semi-major",
                vector.semi_major_ci.as_ref().expect("semi-major CI"),
                python_vector[noise_index].0,
            ),
            (
                "semi-minor",
                vector.semi_minor_ci.as_ref().expect("semi-minor CI"),
                python_vector[noise_index].1,
            ),
        ] {
            for (index, (actual, oracle)) in actual.iter().zip(oracle).enumerate() {
                assert_relative_close(
                    &format!("Python vector {label} distribution[{index}]"),
                    *actual,
                    oracle,
                    0.4,
                );
            }
        }
        for (label, actual, oracle) in [
            (
                "inclination",
                vector
                    .inclination_ci_degrees
                    .as_ref()
                    .expect("inclination CI"),
                python_vector[noise_index].2,
            ),
            (
                "phase",
                vector.phase_ci_degrees.as_ref().expect("phase CI"),
                python_vector[noise_index].3,
            ),
        ] {
            for (index, (actual, oracle)) in actual.iter().zip(oracle).enumerate() {
                assert_close(
                    &format!("Python vector {label} distribution[{index}]"),
                    *actual,
                    oracle,
                    65.0,
                );
            }
        }
    }
}

#[test]
fn gappy_irregular_monte_carlo_is_reproducible_across_worker_counts() {
    faer::set_global_parallelism(faer::Par::Seq);
    let time = irregular_oracle_times();
    let scalar = oracle_observations();
    let (eastward, northward) = synthetic_vector_observations(&time);
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)
        .expect("valid irregular batch");
    let mut scalar_time_major = Vec::with_capacity(time.len() * 2);
    let mut eastward_time_major = Vec::with_capacity(time.len() * 2);
    let mut northward_time_major = Vec::with_capacity(time.len() * 2);
    for time_index in 0..time.len() {
        scalar_time_major.extend([scalar[time_index]; 2]);
        eastward_time_major.extend([eastward[time_index]; 2]);
        northward_time_major.extend([northward[time_index]; 2]);
    }
    for time_index in [137, 411] {
        scalar_time_major[time_index * 2] = f64::NAN;
        scalar_time_major[time_index * 2 + 1] = f64::NAN;
        eastward_time_major[time_index * 2] = f64::NAN;
        eastward_time_major[time_index * 2 + 1] = f64::NAN;
    }
    let options = MonteCarloOptions {
        realizations: 200,
        seed: 71,
    };
    let solve = |workers| {
        ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("valid worker pool")
            .install(|| {
                let scalar = batch
                    .solve_time_major_with_missing_and_monte_carlo_confidence(
                        &scalar_time_major,
                        &[LATITUDE_DEGREES_NORTH; 2],
                        options,
                        LinearConfidence::Colored,
                    )
                    .expect("valid gappy scalar Monte Carlo batch");
                let vector = batch
                    .solve_vector_time_major_with_missing_and_monte_carlo_confidence(
                        &eastward_time_major,
                        &northward_time_major,
                        &[LATITUDE_DEGREES_NORTH; 2],
                        options,
                        LinearConfidence::Colored,
                    )
                    .expect("valid gappy vector Monte Carlo batch");
                (scalar, vector)
            })
    };
    let single_worker = solve(1);
    let four_workers = solve(4);
    assert_eq!(single_worker, four_workers);
    assert_ne!(
        single_worker.0[0].amplitude_ci, single_worker.0[1].amplitude_ci,
        "identical series receive independent deterministic streams"
    );
    assert_ne!(
        single_worker.1[0].semi_major_ci, single_worker.1[1].semi_major_ci,
        "identical vectors receive independent deterministic streams"
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
    reason = "one oracle test keeps the complete irregular scalar reference together"
)]
fn matches_python_utide_lomb_scargle_confidence_for_irregular_scalar_observations() {
    let time = irregular_oracle_times();
    let mut observations = oracle_observations();
    for index in [0, 137, 411] {
        observations[index] = f64::NAN;
    }
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)
        .expect("valid irregular corrected oracle batch");
    let solution = batch
        .solve_time_major_with_missing_and_linear_confidence(
            &observations,
            &[LATITUDE_DEGREES_NORTH],
            LinearConfidence::Colored,
        )
        .expect("valid irregular colored solution")
        .pop()
        .expect("one solution");

    for (label, actual, expected, tolerance) in [
        (
            "irregular amplitude",
            &solution.amplitude,
            &[
                0.654_244_444_119_638_3,
                0.225_093_924_932_780_27,
                0.157_258_414_492_770_03,
                0.111_820_151_047_458_36,
                0.067_598_797_087_460_1,
            ],
            3e-12,
        ),
        (
            "irregular phase",
            &solution.phase_degrees,
            &[
                189.385_326_025_101,
                229.311_114_833_716_77,
                163.441_755_838_302_3,
                153.491_412_732_848_03,
                21.234_738_319_774_65,
            ],
            3e-9,
        ),
        (
            "irregular colored amplitude CI",
            solution.amplitude_ci.as_ref().expect("amplitude CI"),
            &[
                0.015_522_659_450_016_2,
                0.015_499_371_970_084_946,
                0.015_492_939_103_060_327,
                0.010_278_475_329_965_786,
                0.010_298_920_863_638_402,
            ],
            2e-10,
        ),
        (
            "irregular colored phase CI",
            solution.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                1.355_181_069_898_776,
                3.944_824_843_568_596,
                5.648_820_583_983_031,
                5.280_968_612_410_327,
                8.718_322_709_819_404,
            ],
            2e-8,
        ),
        (
            "irregular colored SNR",
            solution.signal_to_noise.as_ref().expect("SNR"),
            &[
                6_824.329_220_307_737,
                810.235_912_193_672_8,
                395.796_819_916_912_97,
                454.668_530_767_094_5,
                165.503_291_805_900_1,
            ],
            2e-4,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    for (label, actual, expected) in [
        (
            "irregular cosine variance",
            solution
                .cosine_coefficient_variance
                .as_ref()
                .expect("cosine variance"),
            &[
                6.273_296_097_747_199e-5,
                6.248_402_438_138_547e-5,
                6.247_327_888_939_909e-5,
                2.745_107_443_210_023_5e-5,
                2.762_256_173_493_316_7e-5,
            ],
        ),
        (
            "irregular sine variance",
            solution
                .sine_coefficient_variance
                .as_ref()
                .expect("sine variance"),
            &[
                6.232_197_048_334_572e-5,
                6.257_090_707_943_217e-5,
                6.258_165_257_141_862e-5,
                2.770_066_174_747_745e-5,
                2.752_917_444_464_449_4e-5,
            ],
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, 2e-12);
        }
    }
    assert_close(
        "irregular mean",
        solution.mean,
        0.091_510_994_992_299_66,
        3e-12,
    );
    assert_close(
        "irregular slope",
        solution.slope_per_day,
        0.001_690_339_784_294_292_8,
        3e-12,
    );
    assert_close(
        "irregular reference time",
        solution.reference_time_days,
        58_128.521_542_833_4,
        1e-12,
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
#[allow(
    clippy::too_many_lines,
    reason = "one oracle test keeps the complete irregular vector reference together"
)]
fn matches_python_utide_lomb_scargle_confidence_for_irregular_vector_observations() {
    let time = irregular_oracle_times();
    let (mut eastward, mut northward) = synthetic_vector_observations(&time);
    for index in [0, 137] {
        eastward[index] = f64::NAN;
    }
    for index in [2, 411] {
        northward[index] = f64::NAN;
    }
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)
        .expect("valid irregular vector oracle batch");
    let solution = batch
        .solve_vector_time_major_with_missing_and_linear_confidence(
            &eastward,
            &northward,
            &[LATITUDE_DEGREES_NORTH],
            LinearConfidence::Colored,
        )
        .expect("valid irregular colored vector solution")
        .pop()
        .expect("one vector solution");

    for (label, actual, expected, tolerance) in [
        (
            "irregular semi-major",
            &solution.semi_major,
            &[
                0.002_386_378_307_631_415_3,
                0.003_318_271_025_053_142_5,
                0.001_376_241_525_120_659_7,
                0.004_225_774_688_046_809,
                0.040_869_601_922_192_59,
            ],
            3e-12,
        ),
        (
            "irregular semi-minor",
            &solution.semi_minor,
            &[
                0.001_427_609_021_093_964,
                0.000_791_733_465_406_382_5,
                0.000_073_174_719_220_492_69,
                -0.001_119_063_664_119_936_1,
                -0.007_840_851_008_634_082,
            ],
            3e-12,
        ),
        (
            "irregular inclination",
            &solution.inclination_degrees,
            &[
                155.010_955_107_475_55,
                2.937_837_451_292_324_4,
                111.026_671_474_869_69,
                13.834_456_402_777_79,
                85.363_321_834_123_72,
            ],
            3e-9,
        ),
        (
            "irregular vector phase",
            &solution.phase_degrees,
            &[
                42.643_567_016_682_44,
                125.447_870_539_818_84,
                44.227_553_019_517_12,
                180.776_420_577_477_86,
                172.609_588_506_942_83,
            ],
            3e-9,
        ),
        (
            "irregular semi-major CI",
            solution.semi_major_ci.as_ref().expect("major CI"),
            &[
                0.029_880_753_967_559_567,
                0.033_769_790_010_141_32,
                0.011_893_351_268_993_392,
                0.035_586_826_997_351_46,
                0.003_403_928_484_171_693_7,
            ],
            2e-9,
        ),
        (
            "irregular semi-minor CI",
            solution.semi_minor_ci.as_ref().expect("minor CI"),
            &[
                0.013_945_981_894_905_556,
                0.001_953_023_361_151_898,
                0.030_864_569_621_632_09,
                0.008_892_053_668_296_35,
                0.038_506_104_015_462_43,
            ],
            2e-9,
        ),
        (
            "irregular inclination CI",
            solution
                .inclination_ci_degrees
                .as_ref()
                .expect("inclination CI"),
            &[
                846.402_724_591_051_1,
                151.777_641_587_505_34,
                1_288.864_148_354_187_6,
                188.979_848_465_717_62,
                56.176_163_351_497_046,
            ],
            2e-5,
        ),
        (
            "irregular vector phase CI",
            solution.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                1_158.814_086_751_269_3,
                618.119_967_113_884,
                501.301_513_125_509_5,
                521.948_325_050_910_5,
                11.858_031_940_865_116,
            ],
            2e-5,
        ),
        (
            "irregular vector SNR",
            solution.signal_to_noise.as_ref().expect("SNR"),
            &[
                0.027_320_175_586_342_813,
                0.039_072_825_884_677_115,
                0.006_669_311_650_112_340_5,
                0.054_560_726_743_910_805,
                4.452_161_824_047_751,
            ],
            2e-7,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{label}[{index}]"), *actual, *expected, tolerance);
        }
    }
    for (label, actual, expected) in [
        (
            "irregular eastward mean",
            solution.eastward_mean,
            0.163_587_537_768_660_45,
        ),
        (
            "irregular northward mean",
            solution.northward_mean,
            -0.077_977_441_474_055_65,
        ),
        (
            "irregular eastward slope",
            solution.eastward_slope_per_day,
            0.000_823_089_465_449_228_9,
        ),
        (
            "irregular northward slope",
            solution.northward_slope_per_day,
            0.002_158_481_741_248_824,
        ),
        (
            "irregular vector reference time",
            solution.reference_time_days,
            58_128.521_542_833_4,
        ),
    ] {
        assert_close(label, actual, expected, 3e-12);
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
