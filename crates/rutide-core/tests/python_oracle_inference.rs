//! Scalar inferred-constituent parity against the pinned Python `UTide` oracle.

use rayon::ThreadPoolBuilder;
use rutide_core::{
    AnalysisError, FitOptions, GreenwichNodalOls, InferenceMode, LinearConfidence,
    MonteCarloOptions, NodalCorrections, PhaseReference, ReconstructionFilter, RobustOptions,
    ScalarInferenceBatch, ScalarInferenceOls, ScalarInferenceRelation, SolverOptions,
    TidalConstituent, VectorInferenceBatch, VectorInferenceOls, VectorInferenceRelation,
};

const LATITUDE: f64 = 60.957_717_895_507_81;
const RELATIONSHIPS: [ScalarInferenceRelation; 2] = [
    ScalarInferenceRelation::new(TidalConstituent::S2, TidalConstituent::M2, 0.35, 20.0),
    ScalarInferenceRelation::new(TidalConstituent::O1, TidalConstituent::K1, 0.5, 45.0),
];
const RECONSTRUCTION_TIMES: [f64; 3] = [58_113.0, 58_120.25, 58_144.5];
const VECTOR_RELATIONSHIPS: [VectorInferenceRelation; 2] = [
    VectorInferenceRelation::new(
        TidalConstituent::S2,
        TidalConstituent::M2,
        0.35,
        20.0,
        0.25,
        -10.0,
    ),
    VectorInferenceRelation::new(
        TidalConstituent::O1,
        TidalConstituent::K1,
        0.5,
        45.0,
        0.4,
        30.0,
    ),
];

struct Expected {
    amplitude: [f64; 5],
    phase: [f64; 5],
    amplitude_ci: [f64; 5],
    phase_ci: [f64; 5],
    percent_energy: [f64; 5],
    signal_to_noise: [f64; 5],
    mean: f64,
    slope: f64,
    reconstruction: [f64; 3],
}

const UNRESOLVED_EXACT: Expected = Expected {
    amplitude: [
        0.007_064_024_350_640_951,
        0.533_240_864_327_627_4,
        0.070_335_163_955_440_22,
        0.186_634_302_514_669_56,
        0.035_167_581_977_720_117,
    ],
    phase: [
        129.523_126_958_217_03,
        176.916_724_887_305_58,
        166.206_453_179_064_97,
        156.916_724_887_305_58,
        121.206_453_179_064_97,
    ],
    amplitude_ci: [
        0.028_022_469_027_663_22,
        0.023_322_914_801_892_844,
        0.026_806_863_940_406_933,
        0.005_778_287_568_140_492,
        0.009_619_339_470_931_425,
    ],
    phase_ci: [
        226.182_479_771_688_7,
        2.528_779_184_859_080_6,
        22.485_309_142_959_732,
        0.625_184_235_038_083_3,
        7.836_017_183_910_285,
    ],
    percent_energy: [
        0.015_334_547_870_495_272,
        87.380_283_157_480_14,
        1.520_238_086_286_436_9,
        10.704_084_686_791_314,
        0.380_059_521_571_609_3,
    ],
    signal_to_noise: [
        0.244_120_202_957_309_07,
        2_008.137_001_687_871_8,
        26.446_312_833_861_21,
        4_007.714_497_910_050_3,
        51.346_004_767_210_53,
    ],
    mean: 0.121_046_186_450_023_95,
    slope: 0.039_694_528_768_266_92,
    reconstruction: [
        0.305_040_343_880_944_75,
        1.029_613_965_756_220_9,
        1.413_865_535_274_13,
    ],
};

const UNRESOLVED_APPROXIMATE: Expected = Expected {
    amplitude: [
        0.006_952_494_495_653_259,
        0.593_848_352_071_065_8,
        0.105_122_428_293_499_35,
        0.207_846_923_224_873,
        0.052_561_214_146_749_67,
    ],
    phase: [
        139.900_740_452_254_75,
        184.152_348_251_748_4,
        165.980_870_226_092_83,
        164.152_348_251_748_44,
        120.980_870_226_092_88,
    ],
    amplitude_ci: [
        0.028_730_687_181_718_256,
        0.027_092_191_158_387_024,
        0.029_176_037_769_708_74,
        0.006_710_452_640_652_063,
        0.010_365_296_669_458_842,
    ],
    phase_ci: [
        237.475_110_474_626_18,
        2.632_111_916_930_068_4,
        16.055_885_912_810_75,
        0.650_889_220_098_713_7,
        5.649_486_624_327_965,
    ],
    percent_energy: [
        0.011_797_671_902_103_441,
        86.072_837_046_802_38,
        2.697_154_194_449_771,
        10.543_922_538_233_29,
        0.674_288_548_612_442_6,
    ],
    signal_to_noise: [
        0.224_957_990_001_938_25,
        1_845.758_986_135_453_6,
        49.871_253_756_007_576,
        3_685.494_024_496_563,
        98.782_371_160_922_98,
    ],
    mean: 0.117_336_319_382_299_77,
    slope: 0.038_925_808_184_239_3,
    reconstruction: [
        0.303_985_385_740_650_15,
        1.130_035_721_862_445_7,
        1.345_557_787_855_611_8,
    ],
};

const RESOLVED_EXACT: Expected = Expected {
    amplitude: [
        0.005_521_074_447_027_312,
        0.615_418_608_291_606_2,
        0.100_230_855_525_271_11,
        0.215_396_512_902_062_2,
        0.050_115_427_762_635_535,
    ],
    phase: [
        63.989_864_653_787_045,
        192.777_576_937_923_56,
        141.244_621_757_990_4,
        172.777_576_937_923_56,
        96.244_621_757_990_42,
    ],
    amplitude_ci: [
        0.017_785_003_838_142_85,
        0.016_947_359_179_399_3,
        0.017_626_591_114_083_073,
        0.004_194_222_923_414_985,
        0.006_231_879_105_189_176,
    ],
    phase_ci: [
        184.535_730_108_191_22,
        1.577_696_690_079_040_8,
        10.075_831_384_529_58,
        0.390_463_220_757_254_4,
        3.562_379_761_121_35,
    ],
    percent_energy: [
        0.006_963_809_196_279_752,
        86.524_859_798_998_86,
        2.295_104_853_141_985_8,
        10.599_295_325_377_362,
        0.573_776_213_285_496,
    ],
    signal_to_noise: [
        0.370_212_793_595_641,
        5_065.814_963_144_129,
        124.216_233_631_653_57,
        10_131.796_537_869_106,
        248.437_406_357_944_16,
    ],
    mean: 0.091_108_756_452_680_53,
    slope: 0.002_110_171_206_181_667_3,
    reconstruction: [
        0.373_598_579_049_894_35,
        1.002_148_247_063_583_9,
        0.135_517_991_525_632_2,
    ],
};

const RESOLVED_APPROXIMATE: Expected = Expected {
    amplitude: [
        0.005_229_454_597_124_373,
        0.643_285_317_861_861_3,
        0.114_127_814_702_362_26,
        0.225_149_861_251_651_4,
        0.057_063_907_351_181_13,
    ],
    phase: [
        61.066_331_489_894_935,
        187.628_354_749_857_6,
        157.342_834_734_313_7,
        167.628_354_749_857_64,
        112.342_834_734_313_68,
    ],
    amplitude_ci: [
        0.018_242_391_645_777_973,
        0.017_193_196_825_776_774,
        0.018_424_672_950_364_11,
        0.004_255_239_967_442_13,
        0.006_515_455_080_576_328_5,
    ],
    phase_ci: [
        199.743_532_578_908_57,
        1.531_787_367_894_221_8,
        9.253_601_712_659_762,
        0.379_085_490_791_590_4,
        3.270_964_915_061_733,
    ],
    percent_energy: [
        0.005_687_648_498_206_425,
        86.065_130_955_563_52,
        2.708_962_283_105_401,
        10.542_978_542_056_526,
        0.677_240_570_776_350_2,
    ],
    signal_to_noise: [
        0.315_690_407_333_796_3,
        5_377.818_631_462_754,
        147.399_350_765_120_43,
        10_754.925_271_866_41,
        294.676_596_097_923_83,
    ],
    mean: 0.090_814_391_567_065_04,
    slope: 0.002_095_191_027_092_266_5,
    reconstruction: [
        0.402_009_506_391_814_04,
        1.036_404_688_787_713_6,
        0.219_234_153_519_145_21,
    ],
};

fn oracle_times(count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| {
            58_113.0 + f64::from(u32::try_from(index).expect("oracle index fits u32")) / 24.0
        })
        .collect()
}

fn requested() -> [TidalConstituent; 5] {
    [
        TidalConstituent::from_name("M4").expect("catalog contains M4"),
        TidalConstituent::M2,
        TidalConstituent::S2,
        TidalConstituent::K1,
        TidalConstituent::O1,
    ]
}

fn oracle_observations(count: usize) -> Vec<f64> {
    include_str!("data/fvcom_node_0_zeta_f32.hex")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .take(count)
        .map(|line| {
            f64::from(f32::from_bits(
                u32::from_str_radix(line, 16).expect("fixture contains hexadecimal f32 bits"),
            ))
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

fn check_oracle(count: usize, mode: InferenceMode, expected: &Expected) {
    let time = oracle_times(count);
    let observations = oracle_observations(count);
    let model = ScalarInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &RELATIONSHIPS,
        mode,
    )
    .expect("valid inferred model");
    assert_eq!(
        model
            .tidal_constituents()
            .iter()
            .map(|constituent| constituent.name())
            .collect::<Vec<_>>(),
        ["M4", "M2", "K1", "S2", "O1"]
    );
    let solution = model
        .solve_with_linear_confidence(&observations, LinearConfidence::White)
        .expect("valid inferred solution");
    for (field, actual, expected, tolerance) in [
        ("amplitude", &solution.amplitude, &expected.amplitude, 5e-11),
        ("phase", &solution.phase_degrees, &expected.phase, 5e-8),
        (
            "amplitude_ci",
            solution
                .amplitude_ci
                .as_ref()
                .expect("confidence solution contains amplitude intervals"),
            &expected.amplitude_ci,
            5e-10,
        ),
        (
            "phase_ci",
            solution
                .phase_ci_degrees
                .as_ref()
                .expect("confidence solution contains phase intervals"),
            &expected.phase_ci,
            5e-7,
        ),
        (
            "percent_energy",
            &solution.percent_energy,
            &expected.percent_energy,
            5e-9,
        ),
        (
            "signal_to_noise",
            solution
                .signal_to_noise
                .as_ref()
                .expect("confidence solution contains SNR"),
            &expected.signal_to_noise,
            5e-5,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, tolerance);
        }
    }
    assert_close("mean", solution.mean, expected.mean, 5e-11);
    assert_close("slope", solution.slope_per_day, expected.slope, 5e-11);
    let reconstruction = model
        .reconstruct_modified_julian_days(
            &RECONSTRUCTION_TIMES,
            &solution,
            &ReconstructionFilter::All,
        )
        .expect("valid inferred reconstruction");
    for (index, (actual, expected)) in reconstruction
        .iter()
        .zip(expected.reconstruction)
        .enumerate()
    {
        assert_close(
            &format!("reconstruction[{index}]"),
            *actual,
            expected,
            5e-10,
        );
    }
}

#[test]
fn matches_resolved_and_unresolved_exact_and_approximate_oracles() {
    check_oracle(169, InferenceMode::Exact, &UNRESOLVED_EXACT);
    check_oracle(169, InferenceMode::Approximate, &UNRESOLVED_APPROXIMATE);
    check_oracle(745, InferenceMode::Exact, &RESOLVED_EXACT);
    check_oracle(745, InferenceMode::Approximate, &RESOLVED_APPROXIMATE);
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "freezes raw-phase scalar and rotary inference for both inference bases"
)]
fn matches_python_utide_raw_phase_inference() {
    let time = oracle_times(745);
    let observations = oracle_observations(745);
    let (eastward, northward) = synthetic_vector_observations(&time);
    let solver_options = SolverOptions::new(FitOptions::default(), PhaseReference::Raw);
    let cases = [
        (
            InferenceMode::Exact,
            [
                0.005_028_435_291_965_646,
                0.520_730_863_552_237_2,
                0.072_060_131_614_432_98,
                0.182_255_802_243_283_03,
                0.036_030_065_807_216_49,
            ],
            [
                105.526_653_483_661_26,
                26.759_103_653_931_25,
                330.955_162_374_618_96,
                6.759_103_653_931_246,
                285.955_162_374_618_96,
            ],
            [
                0.023_755_802_182_448_618,
                0.023_082_978_863_443_085,
                0.023_227_973_669_740_765,
                0.005_712_339_749_791_541,
                0.008_213_443_400_514_409,
            ],
            [
                0.001_436_137_553_619_472_4,
                0.003_321_938_416_592_665,
                0.017_710_989_158_913_27,
                0.000_991_493_526_142_924_3,
                0.007_725_093_906_552_807,
            ],
            [
                90.966_028_641_381_09,
                97.469_789_977_797_26,
                244.326_021_062_437_35,
                92.469_789_977_797_26,
                206.826_021_062_437_38,
            ],
        ),
        (
            InferenceMode::Approximate,
            [
                0.005_229_454_597_209_795,
                0.643_285_317_861_481_1,
                0.114_127_814_702_408_49,
                0.225_149_861_251_518_37,
                0.057_063_907_351_204_24,
            ],
            [
                108.035_852_537_566_64,
                31.113_115_275_527_484,
                317.374_753_679_574_2,
                11.113_115_275_527_482,
                272.374_753_679_574_27,
            ],
            [
                0.018_276_605_363_953_42,
                0.017_201_516_538_898_797,
                0.018_418_399_867_047_27,
                0.004_257_003_743_669_934,
                0.006_512_075_460_551_206_5,
            ],
            [
                0.001_419_564_960_052_018_2,
                0.002_892_924_116_988_444_4,
                0.006_303_401_752_531_963,
                0.000_873_617_305_017_977_8,
                0.002_687_291_860_869_615,
            ],
            [
                90.921_709_702_780_5,
                96.539_030_785_258_35,
                196.284_745_298_951_2,
                91.539_030_785_258_33,
                158.784_745_298_951_2,
            ],
        ),
    ];

    for (
        mode,
        expected_amplitude,
        expected_phase,
        expected_ci,
        expected_major,
        expected_vector_phase,
    ) in cases
    {
        let scalar_model = ScalarInferenceOls::prepare_modified_julian_days_with_solver_options(
            &time,
            LATITUDE,
            &requested(),
            &RELATIONSHIPS,
            mode,
            solver_options,
        )
        .expect("valid raw-phase scalar inference model");
        assert_eq!(scalar_model.phase_reference(), PhaseReference::Raw);
        let scalar = scalar_model
            .solve_with_linear_confidence(&observations, LinearConfidence::White)
            .expect("valid raw-phase scalar inference solution");
        for (field, actual, expected, tolerance) in [
            ("amplitude", &scalar.amplitude, &expected_amplitude, 5e-10),
            ("phase", &scalar.phase_degrees, &expected_phase, 5e-7),
            (
                "amplitude CI",
                scalar.amplitude_ci.as_ref().expect("amplitude CI"),
                &expected_ci,
                5e-8,
            ),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_close(
                    &format!("raw {mode:?} scalar {field}[{index}]"),
                    *actual,
                    *expected,
                    tolerance,
                );
            }
        }

        let vector_model = VectorInferenceOls::prepare_modified_julian_days_with_solver_options(
            &time,
            LATITUDE,
            &requested(),
            &VECTOR_RELATIONSHIPS,
            mode,
            solver_options,
        )
        .expect("valid raw-phase vector inference model");
        assert_eq!(vector_model.phase_reference(), PhaseReference::Raw);
        let vector = vector_model
            .solve_vector_with_linear_confidence(&eastward, &northward, LinearConfidence::White)
            .expect("valid raw-phase vector inference solution");
        for (field, actual, expected, tolerance) in [
            ("semi-major", &vector.semi_major, &expected_major, 5e-10),
            ("phase", &vector.phase_degrees, &expected_vector_phase, 5e-7),
        ] {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_close(
                    &format!("raw {mode:?} vector {field}[{index}]"),
                    *actual,
                    *expected,
                    tolerance,
                );
            }
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "freezes scalar and rotary inference across both alternative nodal modes"
)]
fn alternative_nodal_modes_match_python_utide_inference() {
    let time = oracle_times(745);
    let observations = oracle_observations(745);
    let (eastward, northward) = synthetic_vector_observations(&time);

    let linear_options =
        SolverOptions::default().with_nodal_corrections(NodalCorrections::LinearTime);
    let scalar_model = ScalarInferenceOls::prepare_modified_julian_days_with_solver_options(
        &time,
        LATITUDE,
        &requested(),
        &RELATIONSHIPS,
        InferenceMode::Exact,
        linear_options,
    )
    .expect("valid midpoint-nodal scalar inference model");
    assert_eq!(
        scalar_model.nodal_corrections(),
        NodalCorrections::LinearTime
    );
    let scalar = scalar_model
        .solve(&observations)
        .expect("valid midpoint-nodal scalar inference solution");
    for (field, actual, expected, tolerance) in [
        (
            "amplitude",
            &scalar.amplitude,
            &[
                0.005_525_084_799_315_93,
                0.615_422_169_499_873_4,
                0.100_246_939_462_444_58,
                0.215_397_759_324_955_58,
                0.050_123_469_731_222_28,
            ],
            5e-10,
        ),
        (
            "phase",
            &scalar.phase_degrees,
            &[
                63.992_268_196_591_48,
                192.776_378_556_749_36,
                141.236_175_489_729_76,
                172.776_378_556_749_36,
                96.236_175_489_729_77,
            ],
            5e-7,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("linear nodal inferred scalar {field}[{index}]"),
                *actual,
                *expected,
                tolerance,
            );
        }
    }
    let scalar_reconstruction = scalar_model
        .reconstruct_modified_julian_days(
            &RECONSTRUCTION_TIMES,
            &scalar,
            &ReconstructionFilter::All,
        )
        .expect("valid midpoint-nodal inferred reconstruction");
    for (index, (actual, expected)) in scalar_reconstruction
        .iter()
        .zip([
            0.373_027_598_279_288_66,
            1.002_104_748_337_354,
            0.135_731_879_810_277_62,
        ])
        .enumerate()
    {
        assert_close(
            &format!("linear nodal scalar reconstruction[{index}]"),
            *actual,
            expected,
            2e-9,
        );
    }

    let disabled_options =
        SolverOptions::default().with_nodal_corrections(NodalCorrections::Disabled);
    let vector_model = VectorInferenceOls::prepare_modified_julian_days_with_solver_options(
        &time,
        LATITUDE,
        &requested(),
        &VECTOR_RELATIONSHIPS,
        InferenceMode::Approximate,
        disabled_options,
    )
    .expect("valid disabled-nodal vector inference model");
    assert_eq!(vector_model.nodal_corrections(), NodalCorrections::Disabled);
    let vector = vector_model
        .solve_vector(&eastward, &northward)
        .expect("valid disabled-nodal vector inference solution");
    for (field, actual, expected, tolerance) in [
        (
            "semi-major",
            &vector.semi_major,
            &[
                0.001_500_512_164_471_63,
                0.002_974_227_166_087_83,
                0.005_825_808_775_455_12,
                0.000_898_125_884_127_56,
                0.002_485_866_125_374_48,
            ],
            5e-10,
        ),
        (
            "semi-minor",
            &vector.semi_minor,
            &[
                0.000_181_162_837_348_11,
                0.000_117_154_686_024_23,
                -0.002_714_956_471_606_51,
                0.000_183_857_764_111_66,
                -0.000_930_439_973_450_17,
            ],
            5e-10,
        ),
        (
            "phase",
            &vector.phase_degrees,
            &[
                46.889_729_136_548_82,
                254.521_147_412_288_14,
                43.461_496_435_649_835,
                249.521_147_412_288_14,
                5.961_496_435_649_835,
            ],
            5e-7,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("disabled nodal inferred vector {field}[{index}]"),
                *actual,
                *expected,
                tolerance,
            );
        }
    }
}

#[test]
fn scalar_monte_carlo_propagates_shared_reference_draws() {
    let time = oracle_times(745);
    let observations = oracle_observations(745);
    let options = MonteCarloOptions {
        realizations: 257,
        seed: 20_260_902,
    };
    for mode in [InferenceMode::Exact, InferenceMode::Approximate] {
        let model = ScalarInferenceOls::prepare_modified_julian_days(
            &time,
            LATITUDE,
            &requested(),
            &RELATIONSHIPS,
            mode,
        )
        .expect("valid inferred model");
        for noise in [LinearConfidence::White, LinearConfidence::Colored] {
            let solution = model
                .solve_with_monte_carlo_confidence(&observations, options, noise)
                .expect("valid inferred Monte Carlo solution");
            let repeated = model
                .solve_with_monte_carlo_confidence(&observations, options, noise)
                .expect("repeated inferred Monte Carlo solution");
            assert_eq!(solution, repeated);

            let amplitude_ci = solution.amplitude_ci.as_ref().expect("amplitude CI");
            let phase_ci = solution.phase_ci_degrees.as_ref().expect("phase CI");
            assert_close(
                "S2 amplitude CI ratio",
                amplitude_ci[3],
                RELATIONSHIPS[0].amplitude_ratio * amplitude_ci[1],
                2e-14,
            );
            assert_close("S2 phase CI rotation", phase_ci[3], phase_ci[1], 2e-12);
            assert_close(
                "O1 amplitude CI ratio",
                amplitude_ci[4],
                RELATIONSHIPS[1].amplitude_ratio * amplitude_ci[2],
                2e-14,
            );
            assert_close("O1 phase CI rotation", phase_ci[4], phase_ci[2], 2e-12);
            assert!(
                solution
                    .signal_to_noise
                    .as_ref()
                    .expect("Monte Carlo SNR")
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }

    let model = ScalarInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &RELATIONSHIPS,
        InferenceMode::Exact,
    )
    .expect("valid robust inferred model");
    let mut contaminated = observations;
    for (index, offset) in [(71, 5.0), (218, -4.0), (503, 6.0)] {
        contaminated[index] += offset;
    }
    let solution = model
        .solve_robust_with_monte_carlo_confidence(
            &contaminated,
            RobustOptions::default(),
            options,
            LinearConfidence::Colored,
        )
        .expect("valid robust inferred Monte Carlo solution");
    assert!(solution.robust.is_some());
    let amplitude_ci = solution.amplitude_ci.as_ref().expect("robust amplitude CI");
    let phase_ci = solution.phase_ci_degrees.as_ref().expect("robust phase CI");
    assert_close(
        "robust S2 amplitude CI ratio",
        amplitude_ci[3],
        RELATIONSHIPS[0].amplitude_ratio * amplitude_ci[1],
        2e-14,
    );
    assert_close(
        "robust S2 phase CI rotation",
        phase_ci[3],
        phase_ci[1],
        2e-12,
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "keeps shared-draw invariants, independent covariance parity, and robust coverage on one fixture"
)]
fn vector_monte_carlo_propagates_shared_rotary_reference_draws() {
    let time = oracle_times(745);
    let (eastward, northward) = synthetic_vector_observations(&time);
    let relationship = [VectorInferenceRelation::new(
        TidalConstituent::S2,
        TidalConstituent::M2,
        0.35,
        0.0,
        0.35,
        0.0,
    )];
    let options = MonteCarloOptions {
        realizations: 257,
        seed: 20_260_903,
    };
    for mode in [InferenceMode::Exact, InferenceMode::Approximate] {
        let model = VectorInferenceOls::prepare_modified_julian_days(
            &time,
            LATITUDE,
            &requested(),
            &relationship,
            mode,
        )
        .expect("valid vector inference model");
        assert_eq!(
            model
                .tidal_constituents()
                .iter()
                .map(|constituent| constituent.name())
                .collect::<Vec<_>>(),
            ["M4", "K1", "O1", "M2", "S2"]
        );
        for noise in [LinearConfidence::White, LinearConfidence::Colored] {
            let solution = model
                .solve_vector_with_monte_carlo_confidence(&eastward, &northward, options, noise)
                .expect("valid vector inferred Monte Carlo solution");
            let repeated = model
                .solve_vector_with_monte_carlo_confidence(&eastward, &northward, options, noise)
                .expect("repeated vector inferred Monte Carlo solution");
            assert_eq!(solution, repeated);
            let major_ci = solution.semi_major_ci.as_ref().expect("major CI");
            let minor_ci = solution.semi_minor_ci.as_ref().expect("minor CI");
            let inclination_ci = solution
                .inclination_ci_degrees
                .as_ref()
                .expect("inclination CI");
            let phase_ci = solution.phase_ci_degrees.as_ref().expect("phase CI");
            assert_close("S2 major CI ratio", major_ci[4], 0.35 * major_ci[3], 2e-14);
            assert_close("S2 minor CI ratio", minor_ci[4], 0.35 * minor_ci[3], 2e-14);
            assert_close(
                "S2 inclination CI scaling invariance",
                inclination_ci[4],
                inclination_ci[3],
                2e-12,
            );
            assert_close(
                "S2 phase CI scaling invariance",
                phase_ci[4],
                phase_ci[3],
                2e-12,
            );
        }
    }

    let approximate = VectorInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &relationship,
        InferenceMode::Approximate,
    )
    .expect("valid approximate vector inference model");
    let independent = GreenwichNodalOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &[
            TidalConstituent::from_name("M4").expect("catalog contains M4"),
            TidalConstituent::K1,
            TidalConstituent::O1,
            TidalConstituent::M2,
        ],
    )
    .expect("valid independent comparison model");
    for noise in [LinearConfidence::White, LinearConfidence::Colored] {
        let inferred = approximate
            .solve_vector_with_monte_carlo_confidence(&eastward, &northward, options, noise)
            .expect("valid approximate inference solution");
        let independent = independent
            .solve_vector_with_monte_carlo_confidence(&eastward, &northward, options, noise)
            .expect("valid independent solution");
        for (label, inferred, independent) in [
            (
                "major CI",
                inferred.semi_major_ci.as_ref().expect("inferred major CI"),
                independent
                    .semi_major_ci
                    .as_ref()
                    .expect("independent major CI"),
            ),
            (
                "minor CI",
                inferred.semi_minor_ci.as_ref().expect("inferred minor CI"),
                independent
                    .semi_minor_ci
                    .as_ref()
                    .expect("independent minor CI"),
            ),
            (
                "inclination CI",
                inferred
                    .inclination_ci_degrees
                    .as_ref()
                    .expect("inferred inclination CI"),
                independent
                    .inclination_ci_degrees
                    .as_ref()
                    .expect("independent inclination CI"),
            ),
            (
                "phase CI",
                inferred
                    .phase_ci_degrees
                    .as_ref()
                    .expect("inferred phase CI"),
                independent
                    .phase_ci_degrees
                    .as_ref()
                    .expect("independent phase CI"),
            ),
        ] {
            // Colored inference also excludes the inferred S2 frequency from
            // its residual band. The independent four-constituent model does
            // not, so only its ordinary M4/K1/O1 bands are equivalent. White
            // covariance is equivalent for all four independently fitted
            // constituents, including the M2 reference.
            let equivalent_count = if noise == LinearConfidence::White {
                4
            } else {
                3
            };
            for constituent in 0..equivalent_count {
                assert_close(
                    &format!("approximate {label}[{constituent}]"),
                    inferred[constituent],
                    independent[constituent],
                    2e-9,
                );
            }
        }
    }

    let model = VectorInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &relationship,
        InferenceMode::Exact,
    )
    .expect("valid robust vector inference model");
    let mut contaminated_eastward = eastward;
    let mut contaminated_northward = northward;
    contaminated_eastward[71] += 5.0;
    contaminated_northward[218] -= 4.0;
    contaminated_eastward[503] += 4.0;
    contaminated_northward[503] += 3.0;
    let solution = model
        .solve_vector_robust_with_monte_carlo_confidence(
            &contaminated_eastward,
            &contaminated_northward,
            RobustOptions::default(),
            options,
            LinearConfidence::Colored,
        )
        .expect("valid robust vector inferred Monte Carlo solution");
    assert!(solution.robust.is_some());
    let major_ci = solution.semi_major_ci.as_ref().expect("robust major CI");
    assert_close(
        "robust S2 major CI ratio",
        major_ci[4],
        0.35 * major_ci[3],
        2e-14,
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "covers scalar/vector irregular masks, worker invariance, and robust batch dispatch together"
)]
fn irregular_gappy_trend_disabled_inference_monte_carlo_batches_are_reproducible() {
    faer::set_global_parallelism(faer::Par::Seq);
    let mut time = oracle_times(745);
    let final_index = time.len() - 1;
    for (index, value) in time.iter_mut().enumerate().take(final_index).skip(1) {
        let index = f64::from(u32::try_from(index).expect("fixture index fits u32"));
        *value += 0.002 * (index * 0.37).sin() + 0.0007 * (index * 0.11).cos();
    }
    let series_count = 4;
    let latitudes = (0..series_count)
        .map(|series| {
            LATITUDE + f64::from(u32::try_from(series).expect("series count fits u32")) * 1e-5
        })
        .collect::<Vec<_>>();
    let scalar_source = oracle_observations(745);
    let mut scalar = Vec::with_capacity(time.len() * series_count);
    for value in scalar_source {
        for series in 0..series_count {
            scalar.push(
                value + f64::from(u32::try_from(series).expect("series count fits u32")) * 1e-4,
            );
        }
    }
    for series in 0..series_count {
        scalar[(137 + series * 11) * series_count + series] = f64::NAN;
        scalar[(411 + series * 7) * series_count + series] = f64::NAN;
    }
    let (eastward_source, northward_source) = synthetic_vector_observations(&time);
    let mut eastward = Vec::with_capacity(time.len() * series_count);
    let mut northward = Vec::with_capacity(time.len() * series_count);
    for (east, north) in eastward_source.into_iter().zip(northward_source) {
        for series in 0..series_count {
            let offset = f64::from(u32::try_from(series).expect("series count fits u32")) * 1e-4;
            eastward.push(east + offset);
            northward.push(north - offset);
        }
    }
    for series in 0..series_count {
        eastward[(101 + series * 13) * series_count + series] = f64::NAN;
        northward[(379 + series * 9) * series_count + series] = f64::NAN;
    }
    let fit_options = FitOptions { trend: false };
    let scalar_batch = ScalarInferenceBatch::prepare_modified_julian_days_with_options(
        &time,
        &requested(),
        &RELATIONSHIPS,
        InferenceMode::Exact,
        fit_options,
    )
    .expect("valid scalar inference batch");
    let vector_batch = VectorInferenceBatch::prepare_modified_julian_days_with_options(
        &time,
        &requested(),
        &VECTOR_RELATIONSHIPS,
        InferenceMode::Exact,
        fit_options,
    )
    .expect("valid vector inference batch");
    assert_eq!(scalar_batch.fit_options(), fit_options);
    assert_eq!(vector_batch.fit_options(), fit_options);
    let options = MonteCarloOptions {
        realizations: 97,
        seed: 20_260_904,
    };
    let solve = |workers| {
        ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("valid thread pool")
            .install(|| {
                let scalar = scalar_batch
                    .solve_time_major_with_missing_and_monte_carlo_confidence(
                        &scalar,
                        &latitudes,
                        options,
                        LinearConfidence::Colored,
                    )
                    .expect("valid scalar batch solution");
                let vector = vector_batch
                    .solve_vector_time_major_with_missing_and_monte_carlo_confidence(
                        &eastward,
                        &northward,
                        &latitudes,
                        options,
                        LinearConfidence::Colored,
                    )
                    .expect("valid vector batch solution");
                (scalar, vector)
            })
    };
    let sequential = solve(1);
    let parallel = solve(4);
    assert_eq!(sequential, parallel);
    assert!(
        sequential
            .0
            .iter()
            .all(|solution| solution.slope_per_day.to_bits() == 0.0_f64.to_bits())
    );
    assert!(sequential.1.iter().all(|solution| {
        solution.eastward_slope_per_day.to_bits() == 0.0_f64.to_bits()
            && solution.northward_slope_per_day.to_bits() == 0.0_f64.to_bits()
    }));
    assert_ne!(
        sequential.0[0].amplitude_ci, sequential.0[1].amplitude_ci,
        "series must receive distinct derived random streams"
    );

    let robust_scalar = scalar_batch
        .solve_time_major_with_missing_robust_and_monte_carlo_confidence(
            &scalar,
            &latitudes,
            RobustOptions::default(),
            options,
            LinearConfidence::Colored,
        )
        .expect("valid robust scalar batch");
    let robust_vector = vector_batch
        .solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence(
            &eastward,
            &northward,
            &latitudes,
            RobustOptions::default(),
            options,
            LinearConfidence::Colored,
        )
        .expect("valid robust vector batch");
    assert!(
        robust_scalar
            .iter()
            .all(|solution| solution.robust.is_some())
    );
    assert!(
        robust_vector
            .iter()
            .all(|solution| solution.robust.is_some())
    );
    assert!(
        robust_scalar
            .iter()
            .all(|solution| solution.slope_per_day.to_bits() == 0.0_f64.to_bits())
    );
    assert!(robust_vector.iter().all(|solution| {
        solution.eastward_slope_per_day.to_bits() == 0.0_f64.to_bits()
            && solution.northward_slope_per_day.to_bits() == 0.0_f64.to_bits()
    }));
}

#[test]
fn rejects_invalid_scalar_inference_graphs_and_values() {
    let time = oracle_times(24);
    let invalid = [
        (
            ScalarInferenceRelation::new(TidalConstituent::S2, TidalConstituent::M2, f64::NAN, 0.0),
            AnalysisError::InvalidInferenceAmplitudeRatio { index: 0 },
        ),
        (
            ScalarInferenceRelation::new(
                TidalConstituent::S2,
                TidalConstituent::M2,
                1.0,
                f64::INFINITY,
            ),
            AnalysisError::InvalidInferencePhaseOffset { index: 0 },
        ),
        (
            ScalarInferenceRelation::new(TidalConstituent::M2, TidalConstituent::M2, 1.0, 0.0),
            AnalysisError::SelfInference { index: 0 },
        ),
    ];
    for (relationship, expected) in invalid {
        assert_eq!(
            ScalarInferenceOls::prepare_modified_julian_days(
                &time,
                LATITUDE,
                &requested(),
                &[relationship],
                InferenceMode::Exact,
            )
            .expect_err("invalid inference relationship must be rejected"),
            expected
        );
    }

    let chain = [
        ScalarInferenceRelation::new(TidalConstituent::S2, TidalConstituent::M2, 1.0, 0.0),
        ScalarInferenceRelation::new(TidalConstituent::O1, TidalConstituent::S2, 1.0, 0.0),
    ];
    assert_eq!(
        ScalarInferenceOls::prepare_modified_julian_days(
            &time,
            LATITUDE,
            &requested(),
            &chain,
            InferenceMode::Exact,
        )
        .expect_err("inference chains must be rejected"),
        AnalysisError::InferenceReferenceIsInferred { name: "S2" }
    );

    let invalid_vector = VectorInferenceRelation::new(
        TidalConstituent::S2,
        TidalConstituent::M2,
        0.5,
        0.0,
        -0.5,
        0.0,
    );
    assert_eq!(
        VectorInferenceOls::prepare_modified_julian_days(
            &time,
            LATITUDE,
            &requested(),
            &[invalid_vector],
            InferenceMode::Exact,
        )
        .expect_err("negative vector rotary ratios must be rejected"),
        AnalysisError::InvalidInferenceAmplitudeRatio { index: 0 }
    );
}

#[test]
fn matches_resolved_exact_colored_confidence_oracle() {
    let time = oracle_times(745);
    let observations = oracle_observations(745);
    let model = ScalarInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &RELATIONSHIPS,
        InferenceMode::Exact,
    )
    .expect("valid exact inferred model");
    let solution = model
        .solve_with_linear_confidence(&observations, LinearConfidence::Colored)
        .expect("valid colored inferred solution");
    for (field, actual, expected, tolerance) in [
        (
            "amplitude_ci",
            solution
                .amplitude_ci
                .as_ref()
                .expect("colored solution contains amplitude intervals"),
            &[
                0.000_130_029_606_331_219_4,
                0.052_297_539_558_245_74,
                0.016_044_499_687_224_198,
                0.012_942_874_280_969_307,
                0.005_672_530_877_177_953,
            ],
            5e-9,
        ),
        (
            "phase_ci",
            solution
                .phase_ci_degrees
                .as_ref()
                .expect("colored solution contains phase intervals"),
            &[
                1.349_176_449_911_740_5,
                4.868_584_785_800_625,
                9.171_465_568_770_48,
                1.204_923_169_293_231_8,
                3.242_634_982_178_661_3,
            ],
            5e-7,
        ),
        (
            "signal_to_noise",
            solution
                .signal_to_noise
                .as_ref()
                .expect("colored solution contains SNR"),
            &[
                6_925.878_266_313_302,
                531.974_574_765_084_3,
                149.921_065_153_114_85,
                1_063.966_645_851_198_3,
                299.848_091_478_189_36,
            ],
            5e-5,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, tolerance);
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "freezes scalar and rotary-vector inference for Python trend=False"
)]
fn matches_python_utide_inference_with_trend_disabled() {
    let time = oracle_times(745);
    let observations = oracle_observations(745);
    let fit_options = FitOptions { trend: false };
    let scalar_model = ScalarInferenceOls::prepare_modified_julian_days_with_options(
        &time,
        LATITUDE,
        &requested(),
        &RELATIONSHIPS,
        InferenceMode::Exact,
        fit_options,
    )
    .expect("valid trend-disabled scalar inference model");
    let scalar = scalar_model
        .solve_with_linear_confidence(&observations, LinearConfidence::Colored)
        .expect("valid trend-disabled inferred scalar solution");
    assert_eq!(scalar_model.fit_options(), fit_options);
    for (field, actual, expected, tolerance) in [
        (
            "amplitude",
            &scalar.amplitude,
            &[
                0.005_364_921_502_067_653,
                0.615_213_497_191_656_6,
                0.099_986_944_929_154_57,
                0.215_324_724_017_079_83,
                0.049_993_472_464_577_283,
            ],
            5e-10,
        ),
        (
            "phase",
            &scalar.phase_degrees,
            &[
                64.730_174_848_578_77,
                192.763_649_962_347_59,
                141.344_139_143_178_38,
                172.763_649_962_347_59,
                96.344_139_143_178_38,
            ],
            5e-7,
        ),
        (
            "amplitude CI",
            scalar.amplitude_ci.as_ref().expect("amplitude CI"),
            &[
                0.000_158_384_269_193_132_32,
                0.052_306_183_492_147_18,
                0.016_050_196_369_348_72,
                0.012_944_964_861_062_482,
                0.005_674_556_037_602_392,
            ],
            5e-8,
        ),
        (
            "phase CI",
            scalar.phase_ci_degrees.as_ref().expect("phase CI"),
            &[
                1.691_216_902_897_597_4,
                4.870_856_360_473_176,
                9.197_138_964_587_024,
                1.205_489_893_390_832_5,
                3.251_705_628_124_404,
            ],
            5e-6,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("no-trend inferred scalar {field}[{index}]"),
                *actual,
                *expected,
                tolerance,
            );
        }
    }
    assert_close(
        "no-trend inferred scalar mean",
        scalar.mean,
        0.091_108_518_233_299_45,
        5e-10,
    );
    assert_close(
        "no-trend inferred scalar slope",
        scalar.slope_per_day,
        0.0,
        f64::EPSILON,
    );

    let (eastward, northward) = synthetic_vector_observations(&time);
    let vector_model = VectorInferenceOls::prepare_modified_julian_days_with_options(
        &time,
        LATITUDE,
        &requested(),
        &VECTOR_RELATIONSHIPS,
        InferenceMode::Exact,
        fit_options,
    )
    .expect("valid trend-disabled vector inference model");
    let vector = vector_model
        .solve_vector_with_linear_confidence(&eastward, &northward, LinearConfidence::Colored)
        .expect("valid trend-disabled inferred vector solution");
    assert_eq!(vector_model.fit_options(), fit_options);
    for (field, actual, expected, tolerance) in [
        (
            "semi-major",
            &vector.semi_major,
            &[
                0.001_294_202_248_565_717,
                0.001_930_193_703_780_505_8,
                0.011_509_594_522_023_988,
                0.000_597_849_819_522_564_2,
                0.005_183_940_379_616_224,
            ],
            5e-10,
        ),
        (
            "semi-minor",
            &vector.semi_minor,
            &[
                0.000_194_752_769_899_466_3,
                0.000_375_834_167_768_249_1,
                0.000_092_456_894_108_540_99,
                0.000_209_259_935_519_499_94,
                0.000_617_085_328_450_042_6,
            ],
            5e-10,
        ),
        (
            "semi-major CI",
            vector.semi_major_ci.as_ref().expect("major CI"),
            &[
                0.027_136_145_362_273_455,
                0.025_593_115_614_905_22,
                0.015_258_408_962_170_858,
                0.005_957_444_066_705_783,
                0.009_489_063_409_879_714,
            ],
            5e-7,
        ),
        (
            "semi-minor CI",
            vector.semi_minor_ci.as_ref().expect("minor CI"),
            &[
                0.009_347_825_716_291_443,
                0.010_599_870_336_009_799,
                0.025_406_663_954_917_884,
                0.005_957_444_066_705_782_5,
                0.009_489_063_409_879_714,
            ],
            5e-7,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("no-trend inferred vector {field}[{index}]"),
                *actual,
                *expected,
                tolerance,
            );
        }
    }
    assert_close(
        "no-trend inferred eastward mean",
        vector.eastward_mean,
        0.163_559_643_029_948_27,
        5e-10,
    );
    assert_close(
        "no-trend inferred northward mean",
        vector.northward_mean,
        -0.076_932_838_516_875_59,
        5e-10,
    );
    assert_close(
        "no-trend inferred eastward slope",
        vector.eastward_slope_per_day,
        0.0,
        f64::EPSILON,
    );
    assert_close(
        "no-trend inferred northward slope",
        vector.northward_slope_per_day,
        0.0,
        f64::EPSILON,
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "keeps every vector coefficient, interval, and reconstruction oracle in one fixture"
)]
fn matches_resolved_exact_vector_inference_oracle() {
    let time = oracle_times(745);
    let (eastward, northward) = synthetic_vector_observations(&time);
    let model = VectorInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &VECTOR_RELATIONSHIPS,
        InferenceMode::Exact,
    )
    .expect("valid exact vector inference model");
    let solution = model
        .solve_vector_with_linear_confidence(&eastward, &northward, LinearConfidence::White)
        .expect("valid exact vector inference solution");
    for (field, actual, expected, tolerance) in [
        (
            "semi_major",
            &solution.semi_major,
            &[
                0.001_407_920_440_945_174_7,
                0.002_090_612_554_378_58,
                0.011_508_568_119_819_637,
                0.000_644_069_690_393_918_9,
                0.005_179_687_361_785_794,
            ],
            5e-11,
        ),
        (
            "semi_minor",
            &solution.semi_minor,
            &[
                0.000_185_983_948_857_171_68,
                0.000_337_718_481_606_899_1,
                0.000_016_634_157_339_116_308,
                0.000_205_846_172_200_998_7,
                0.000_582_913_776_793_584_3,
            ],
            5e-11,
        ),
        (
            "inclination",
            &solution.inclination_degrees,
            &[
                24.369_186_727_347_31,
                27.974_372_636_132_912,
                60.495_721_494_951_65,
                42.974_372_636_132_905,
                67.995_721_494_951_67,
            ],
            5e-8,
        ),
        (
            "phase",
            &solution.phase_degrees,
            &[
                44.312_896_465_342_05,
                245.624_603_252_718_5,
                201.204_818_439_787_86,
                240.624_603_252_718_46,
                163.704_818_439_787_86,
            ],
            5e-8,
        ),
        (
            "semi_major_ci",
            solution
                .semi_major_ci
                .as_ref()
                .expect("vector confidence contains major intervals"),
            &[
                0.028_690_987_341_039_025,
                0.027_691_479_783_168_37,
                0.028_913_193_825_560_728,
                0.008_422_402_294_410_412,
                0.013_081_812_736_393_179,
            ],
            5e-9,
        ),
        (
            "semi_minor_ci",
            solution
                .semi_minor_ci
                .as_ref()
                .expect("vector confidence contains minor intervals"),
            &[
                0.028_693_601_478_011_75,
                0.027_693_727_887_231_134,
                0.028_875_614_459_945_56,
                0.008_422_402_294_410_412,
                0.013_081_812_736_393_177,
            ],
            5e-9,
        ),
        (
            "percent_energy",
            &solution.percent_energy,
            &[
                1.210_763_122_342_355_8,
                2.692_307_775_054_401,
                79.512_129_532_811_61,
                0.274_469_825_634_424_57,
                16.310_329_744_157_2,
            ],
            5e-8,
        ),
        (
            "signal_to_noise",
            solution
                .signal_to_noise
                .as_ref()
                .expect("vector confidence contains SNR"),
            &[
                0.004_705_663_648_382_104,
                0.011_232_834_847_357_293,
                0.304_717_902_460_835_1,
                0.012_379_836_181_647_973,
                0.304_943_330_895_505_85,
            ],
            5e-8,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, tolerance);
        }
    }
    assert_close(
        "eastward_mean",
        solution.eastward_mean,
        0.163_559_915_423_324_51,
        5e-11,
    );
    assert_close(
        "northward_mean",
        solution.northward_mean,
        -0.076_932_476_537_805_51,
        5e-11,
    );
    assert_close(
        "eastward_slope",
        solution.eastward_slope_per_day,
        0.000_733_633_115_831_299_4,
        5e-11,
    );
    assert_close(
        "northward_slope",
        solution.northward_slope_per_day,
        0.001_951_992_501_818_680_2,
        5e-11,
    );
    let reconstruction = model
        .reconstruct_vector_modified_julian_days(
            &RECONSTRUCTION_TIMES,
            &solution,
            &ReconstructionFilter::All,
        )
        .expect("valid inferred vector reconstruction");
    for (field, actual, expected) in [
        (
            "reconstruction_eastward",
            &reconstruction.eastward,
            &[
                0.150_056_288_504_953_73,
                0.158_924_765_047_456_06,
                0.176_940_550_404_792_62,
            ],
        ),
        (
            "reconstruction_northward",
            &reconstruction.northward,
            &[
                -0.111_460_613_566_090_16,
                -0.094_613_429_164_934_2,
                -0.039_702_780_371_134_117,
            ],
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, 5e-10);
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "freezes coupled robust ellipses, diagnostics, intervals, and reconstruction together"
)]
fn matches_resolved_exact_robust_vector_inference_oracle() {
    let time = oracle_times(745);
    let (mut eastward, mut northward) = synthetic_vector_observations(&time);
    for (index, eastward_outlier, northward_outlier) in
        [(33, 3.0, -2.5), (211, -2.4, 3.2), (510, 4.0, 3.5)]
    {
        eastward[index] += eastward_outlier;
        northward[index] += northward_outlier;
    }
    let model = VectorInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &VECTOR_RELATIONSHIPS,
        InferenceMode::Exact,
    )
    .expect("valid robust vector inference model");
    let solution = model
        .solve_vector_robust_with_linear_confidence(
            &eastward,
            &northward,
            RobustOptions::default(),
            LinearConfidence::White,
        )
        .expect("valid robust vector inference solution");
    for (field, actual, expected, tolerance) in [
        (
            "semi_major",
            &solution.semi_major,
            &[
                0.003_828_149_278_447_270_8,
                0.003_685_374_929_143_065,
                0.012_816_252_494_210_874,
                0.001_022_806_688_243_341_5,
                0.005_912_024_400_890_821_5,
            ],
            5e-10,
        ),
        (
            "semi_minor",
            &solution.semi_minor,
            &[
                -0.002_375_619_626_209_732_3,
                -0.001_656_115_809_991_561,
                0.002_894_215_569_918_538,
                -0.000_312_565_996_540_314_9,
                0.001_943_209_631_173_886_2,
            ],
            5e-10,
        ),
        (
            "inclination",
            &solution.inclination_degrees,
            &[
                28.597_902_984_005_586,
                75.556_177_227_940_08,
                31.392_430_614_142_853,
                90.556_177_227_940_08,
                38.892_430_614_142_85,
            ],
            5e-7,
        ),
        (
            "phase",
            &solution.phase_degrees,
            &[
                62.676_432_014_437_836,
                205.867_843_234_095_18,
                134.505_327_740_328_14,
                200.867_843_234_095_18,
                97.005_327_740_328_14,
            ],
            5e-7,
        ),
        (
            "semi_major_ci",
            solution
                .semi_major_ci
                .as_ref()
                .expect("robust vector confidence contains major intervals"),
            &[
                0.028_758_510_970_297_146,
                0.027_763_185_276_047_066,
                0.028_974_415_164_226_296,
                0.008_446_409_204_575_064,
                0.013_122_647_163_503_8,
            ],
            5e-8,
        ),
        (
            "semi_minor_ci",
            solution
                .semi_minor_ci
                .as_ref()
                .expect("robust vector confidence contains minor intervals"),
            &[
                0.028_816_022_483_273_38,
                0.027_778_418_078_398_658,
                0.028_991_770_980_478_237,
                0.008_446_409_204_575_065,
                0.013_122_647_163_503_8,
            ],
            5e-8,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(
                &format!("robust_{field}[{index}]"),
                *actual,
                *expected,
                tolerance,
            );
        }
    }
    for (field, actual, expected) in [
        (
            "eastward_mean",
            solution.eastward_mean,
            0.164_326_356_109_881_17,
        ),
        (
            "northward_mean",
            solution.northward_mean,
            -0.073_035_711_340_204_02,
        ),
        (
            "eastward_slope",
            solution.eastward_slope_per_day,
            0.000_855_311_061_192_666_9,
        ),
        (
            "northward_slope",
            solution.northward_slope_per_day,
            0.002_041_823_754_524_812_3,
        ),
    ] {
        assert_close(field, actual, expected, 5e-10);
    }
    let diagnostics = solution.robust.as_ref().expect("robust diagnostics");
    assert_eq!(diagnostics.iterations, 2);
    for (position, expected) in [
        (0, 0.919_575_027_491_081),
        (33, 0.088_855_050_987_428_13),
        (211, 0.124_214_443_740_685_09),
        (510, 0.052_969_643_620_864_69),
        (744, 0.913_389_944_585_981_9),
    ] {
        assert_close(
            &format!("robust_weight[{position}]"),
            diagnostics.weights[position],
            expected,
            5e-8,
        );
    }
    for (position, expected) in [
        (0, 0.010_689_442_245_621_017),
        (33, 0.009_980_304_875_996_836),
        (211, 0.013_370_639_401_765_35),
        (510, 0.013_753_648_872_101_934),
        (744, 0.011_647_158_724_217_692),
    ] {
        assert_close(
            &format!("robust_leverage[{position}]"),
            diagnostics.leverage[position],
            expected,
            5e-10,
        );
    }
    assert_close(
        "robust_ols_rms",
        diagnostics.ols_rms_residual,
        0.494_881_202_636_785_37,
        5e-9,
    );
    assert_close(
        "robust_rms",
        diagnostics.rms_residual,
        0.495_619_070_132_678_6,
        5e-9,
    );

    let reconstruction = model
        .reconstruct_vector_modified_julian_days(
            &RECONSTRUCTION_TIMES,
            &solution,
            &ReconstructionFilter::All,
        )
        .expect("valid robust inferred reconstruction");
    for (field, actual, expected) in [
        (
            "robust_reconstruction_eastward",
            &reconstruction.eastward,
            &[
                0.143_905_869_205_494_8,
                0.170_038_405_967_186_9,
                0.170_396_546_423_146_25,
            ],
        ),
        (
            "robust_reconstruction_northward",
            &reconstruction.northward,
            &[
                -0.103_978_904_725_694_4,
                -0.076_612_332_306_421_62,
                -0.041_420_540_958_734_39,
            ],
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, 5e-9);
        }
    }

    let colored = model
        .solve_vector_robust_with_linear_confidence(
            &eastward,
            &northward,
            RobustOptions::default(),
            LinearConfidence::Colored,
        )
        .expect("valid colored robust vector inference solution");
    for (field, actual, expected) in [
        (
            "colored_major_ci",
            colored
                .semi_major_ci
                .as_ref()
                .expect("colored major intervals"),
            &[
                0.025_263_501_186_252_68,
                0.007_113_991_699_530_108,
                0.024_961_934_987_802_848,
                0.005_983_748_698_121_433,
                0.009_510_324_577_108_95,
            ],
        ),
        (
            "colored_minor_ci",
            colored
                .semi_minor_ci
                .as_ref()
                .expect("colored minor intervals"),
            &[
                0.013_866_696_366_408_264,
                0.026_901_392_195_889_05,
                0.016_102_732_952_262_125,
                0.005_983_748_698_121_433,
                0.009_510_324_577_108_95,
            ],
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, 5e-7);
        }
    }
}

fn check_vector_ellipse(
    count: usize,
    mode: InferenceMode,
    expected_major: &[f64; 5],
    expected_minor: &[f64; 5],
    expected_inclination: &[f64; 5],
    expected_phase: &[f64; 5],
) {
    let time = oracle_times(count);
    let (eastward, northward) = synthetic_vector_observations(&time);
    let model = VectorInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &VECTOR_RELATIONSHIPS,
        mode,
    )
    .expect("valid vector inference model");
    let solution = model
        .solve_vector(&eastward, &northward)
        .expect("valid vector inference observations");
    for (field, actual, expected, tolerance) in [
        ("semi_major", &solution.semi_major, expected_major, 5e-11),
        ("semi_minor", &solution.semi_minor, expected_minor, 5e-11),
        (
            "inclination",
            &solution.inclination_degrees,
            expected_inclination,
            5e-8,
        ),
        ("phase", &solution.phase_degrees, expected_phase, 5e-8),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, tolerance);
        }
    }
}

#[test]
fn matches_unresolved_and_approximate_vector_ellipse_oracles() {
    check_vector_ellipse(
        169,
        InferenceMode::Exact,
        &[
            0.005_691_288_851_948_564,
            0.013_444_201_325_114_27,
            0.029_830_045_346_458_53,
            0.003_734_781_606_630_359,
            0.014_180_603_390_018_889,
        ],
        &[
            0.003_219_918_445_964_943_4,
            -0.005_969_575_818_078_434,
            0.015_141_659_682_251_011,
            -0.001_118_662_679_167_816_7,
            0.008_305_249_124_335_879,
        ],
        &[
            99.090_341_402_017_32,
            53.059_162_600_699_494,
            35.883_492_147_495_25,
            68.059_162_600_699_51,
            43.383_492_147_495_25,
        ],
        &[
            34.513_530_730_812_52,
            266.705_163_984_708_5,
            64.361_959_857_307_58,
            261.705_163_984_708_46,
            26.861_959_857_307_59,
        ],
    );
    check_vector_ellipse(
        169,
        InferenceMode::Approximate,
        &[
            0.005_671_652_502_119_847,
            0.016_112_588_323_643_11,
            0.049_269_113_328_217_54,
            0.004_572_830_717_424_066,
            0.022_952_723_523_839_723,
        ],
        &[
            0.003_372_618_718_473_810_4,
            -0.005_218_915_593_377_333,
            0.015_632_450_522_836_603,
            -0.000_760_045_261_831_044_5,
            0.009_498_058_401_687_347,
        ],
        &[
            101.409_758_739_030_54,
            66.997_727_688_143_85,
            56.892_524_085_768_1,
            81.997_727_688_143_84,
            64.392_524_085_768_12,
        ],
        &[
            35.489_638_592_827_376,
            257.212_474_276_669_5,
            72.523_133_086_453_65,
            252.212_474_276_669_55,
            35.023_133_086_453_67,
        ],
    );
    check_vector_ellipse(
        745,
        InferenceMode::Approximate,
        &[
            0.001_419_564_959_975_855,
            0.002_892_924_116_977_396_4,
            0.006_303_401_752_535_92,
            0.000_873_617_305_015_497_6,
            0.002_687_291_860_871_099_7,
        ],
        &[
            0.000_172_333_198_488_661_9,
            0.000_114_801_398_445_579_14,
            -0.002_984_778_555_401_283_6,
            0.000_179_086_625_382_543_67,
            -0.001_027_980_262_303_782,
        ],
        &[
            24.714_009_739_394_456,
            24.737_573_280_915_18,
            108.793_813_042_33,
            39.737_573_280_915_17,
            116.293_813_042_330_02,
        ],
        &[
            43.952_188_651_260_13,
            253.054_270_259_882_7,
            36.252_826_353_608_85,
            248.054_270_259_882_7,
            358.752_826_353_608_84,
        ],
    );
}

#[test]
fn matches_resolved_exact_vector_colored_confidence_oracle() {
    let time = oracle_times(745);
    let (eastward, northward) = synthetic_vector_observations(&time);
    let model = VectorInferenceOls::prepare_modified_julian_days(
        &time,
        LATITUDE,
        &requested(),
        &VECTOR_RELATIONSHIPS,
        InferenceMode::Exact,
    )
    .expect("valid exact vector inference model");
    let solution = model
        .solve_vector_with_linear_confidence(&eastward, &northward, LinearConfidence::Colored)
        .expect("valid colored vector inference solution");
    for (field, actual, expected, tolerance) in [
        (
            "semi_major_ci",
            solution
                .semi_major_ci
                .as_ref()
                .expect("colored vector solution contains major intervals"),
            &[
                0.026_133_970_076_758_575,
                0.024_455_249_849_153_86,
                0.015_351_169_329_795_213,
                0.005_955_334_142_149_809,
                0.009_485_447_628_870_975,
            ],
            5e-8,
        ),
        (
            "semi_minor_ci",
            solution
                .semi_minor_ci
                .as_ref()
                .expect("colored vector solution contains minor intervals"),
            &[
                0.011_838_199_569_600_407,
                0.012_991_004_776_935_836,
                0.025_337_141_909_765_732,
                0.005_955_334_142_149_81,
                0.009_485_447_628_870_975,
            ],
            5e-8,
        ),
        (
            "signal_to_noise",
            solution
                .signal_to_noise
                .as_ref()
                .expect("colored vector solution contains SNR"),
            &[
                0.009_412_703_895_376_576,
                0.022_467_276_872_470_257,
                0.579_755_117_359_606_4,
                0.024_761_365_635_209_573,
                0.580_015_146_258_317_4,
            ],
            5e-7,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, tolerance);
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "covers scalar and joint-vector missing inference through the same pinned fixture"
)]
fn matches_gappy_scalar_and_vector_colored_inference_oracles() {
    let time = oracle_times(745);
    let observations = oracle_observations(745);
    let mut scalar_time_major = Vec::with_capacity(time.len() * 2);
    for (index, observation) in observations.iter().copied().enumerate() {
        let missing = index % 17 == 3 || (300..320).contains(&index);
        scalar_time_major.extend([observation, if missing { f64::NAN } else { observation }]);
    }
    let scalar_batch = ScalarInferenceBatch::prepare_modified_julian_days(
        &time,
        &requested(),
        &RELATIONSHIPS,
        InferenceMode::Exact,
    )
    .expect("valid scalar inference batch");
    let scalar = scalar_batch
        .solve_time_major_with_missing_and_linear_confidence(
            &scalar_time_major,
            &[LATITUDE, LATITUDE],
            LinearConfidence::Colored,
        )
        .expect("valid gappy scalar inference solutions")
        .pop()
        .expect("two scalar solutions");
    for (field, actual, expected, tolerance) in [
        (
            "scalar_amplitude",
            &scalar.amplitude,
            &[
                0.005_864_340_119_835_285,
                0.609_947_129_142_678_2,
                0.097_191_005_929_159_73,
                0.213_481_495_199_937_3,
                0.048_595_502_964_579_866,
            ],
            5e-10,
        ),
        (
            "scalar_phase",
            &scalar.phase_degrees,
            &[
                58.437_121_026_985_04,
                192.535_377_639_505_67,
                144.362_980_846_243_46,
                172.535_377_639_505_67,
                99.362_980_846_243_45,
            ],
            5e-7,
        ),
        (
            "scalar_amplitude_ci",
            scalar
                .amplitude_ci
                .as_ref()
                .expect("gappy scalar solution contains intervals"),
            &[
                0.001_300_628_633_285_616_7,
                0.049_291_115_609_988_466,
                0.015_988_184_407_163_975,
                0.012_195_481_630_753_093,
                0.005_651_202_975_666_341,
            ],
            5e-8,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, tolerance);
        }
    }

    let (eastward, northward) = synthetic_vector_observations(&time);
    let mut eastward_time_major = Vec::with_capacity(time.len() * 2);
    let mut northward_time_major = Vec::with_capacity(time.len() * 2);
    for index in 0..time.len() {
        let eastward_missing = index % 19 == 2 || (400..412).contains(&index);
        let northward_missing = index % 23 == 5;
        eastward_time_major.extend([
            eastward[index],
            if eastward_missing {
                f64::NAN
            } else {
                eastward[index]
            },
        ]);
        northward_time_major.extend([
            northward[index],
            if northward_missing {
                f64::NAN
            } else {
                northward[index]
            },
        ]);
    }
    let vector_batch = VectorInferenceBatch::prepare_modified_julian_days(
        &time,
        &requested(),
        &VECTOR_RELATIONSHIPS,
        InferenceMode::Exact,
    )
    .expect("valid vector inference batch");
    let vector = vector_batch
        .solve_vector_time_major_with_missing_and_linear_confidence(
            &eastward_time_major,
            &northward_time_major,
            &[LATITUDE, LATITUDE],
            LinearConfidence::Colored,
        )
        .expect("valid gappy vector inference solutions")
        .pop()
        .expect("two vector solutions");
    for (field, actual, expected, tolerance) in [
        (
            "vector_major",
            &vector.semi_major,
            &[
                0.010_919_083_847_383_654,
                0.003_623_338_328_987_682,
                0.016_346_488_230_777_525,
                0.001_115_459_964_675_995,
                0.007_376_100_345_387_472,
            ],
            5e-10,
        ),
        (
            "vector_minor",
            &vector.semi_minor,
            &[
                -0.002_550_902_796_222_810_6,
                0.000_569_169_319_593_808_2,
                0.000_403_612_830_751_697_7,
                0.000_351_917_712_327_526_7,
                0.000_998_950_185_377_141,
            ],
            5e-10,
        ),
        (
            "vector_major_ci",
            vector
                .semi_major_ci
                .as_ref()
                .expect("gappy vector solution contains major intervals"),
            &[
                0.011_330_009_891_136_366,
                0.029_140_192_288_313_56,
                0.012_467_124_019_200_916,
                0.006_316_103_179_087_014,
                0.010_046_050_239_825_665,
            ],
            5e-8,
        ),
        (
            "vector_minor_ci",
            vector
                .semi_minor_ci
                .as_ref()
                .expect("gappy vector solution contains minor intervals"),
            &[
                0.028_253_270_002_152_067,
                0.003_658_545_908_396_234_6,
                0.028_796_653_472_701_474,
                0.006_316_103_179_087_014,
                0.010_046_050_239_825_665,
            ],
            5e-8,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(&format!("{field}[{index}]"), *actual, *expected, tolerance);
        }
    }
}
