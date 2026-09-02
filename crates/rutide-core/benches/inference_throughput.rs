//! Standalone throughput probe for inferred-constituent colored confidence.

use std::{env, error::Error, hint::black_box, time::Instant};

use faer::{Par, set_global_parallelism};
use rayon::{ThreadPool, ThreadPoolBuilder};
use rutide_core::{
    InferenceMode, LinearConfidence, MonteCarloOptions, RobustOptions, ScalarInferenceBatch,
    ScalarInferenceRelation, TidalConstituent, VectorInferenceBatch, VectorInferenceRelation,
};

const CONSTITUENTS: [TidalConstituent; 5] = [
    TidalConstituent::N2,
    TidalConstituent::M2,
    TidalConstituent::S2,
    TidalConstituent::K1,
    TidalConstituent::O1,
];
const SCALAR_RELATIONSHIPS: [ScalarInferenceRelation; 2] = [
    ScalarInferenceRelation::new(TidalConstituent::S2, TidalConstituent::M2, 0.35, 20.0),
    ScalarInferenceRelation::new(TidalConstituent::O1, TidalConstituent::K1, 0.5, 45.0),
];
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
const FIXTURE_LATITUDE: f64 = 60.957_717_895_507_81;

fn setting(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn nonnegative_setting(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn u64_setting(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn benchmark_setting(name: &str, default: &str, allowed: &[&str]) -> Result<String, String> {
    let value = env::var(name).unwrap_or_else(|_| default.to_owned());
    if allowed.contains(&value.as_str()) {
        Ok(value)
    } else {
        Err(format!("{name} must be one of {}", allowed.join(", ")))
    }
}

fn times(sampling: &str) -> Vec<f64> {
    let mut times = (0_u32..745)
        .map(|index| 58_113.0 + f64::from(index) / 24.0)
        .collect::<Vec<_>>();
    if sampling == "irregular" {
        let final_index = times.len() - 1;
        for (index, time) in times.iter_mut().enumerate().take(final_index).skip(1) {
            let index = f64::from(u32::try_from(index).expect("fixture index fits u32"));
            *time += 0.002 * (index * 0.37).sin() + 0.0007 * (index * 0.11).cos();
        }
    }
    times
}

fn scalar_observations(sampling: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut values = include_str!("../tests/data/fvcom_node_0_zeta_f32.hex")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let bits = u32::from_str_radix(line, 16)?;
            Ok(f64::from(f32::from_bits(bits)))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    if sampling == "irregular" {
        for index in [0, 137, 411] {
            values[index] = f64::NAN;
        }
    }
    Ok(values)
}

fn vector_observations(times: &[f64], sampling: &str) -> (Vec<f64>, Vec<f64>) {
    let reference = times[0].midpoint(times[times.len() - 1]);
    let mut eastward = Vec::with_capacity(times.len());
    let mut northward = Vec::with_capacity(times.len());
    for (index, time) in times.iter().copied().enumerate() {
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
    if sampling == "irregular" {
        for index in [0, 137] {
            eastward[index] = f64::NAN;
        }
        for index in [2, 411] {
            northward[index] = f64::NAN;
        }
    }
    (eastward, northward)
}

fn time_major(source: &[f64], series_count: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(source.len() * series_count);
    for value in source.iter().copied() {
        values.extend(std::iter::repeat_n(value, series_count));
    }
    values
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure(
    mut run: impl FnMut() -> Result<f64, Box<dyn Error>>,
    warmups: usize,
    repetitions: usize,
) -> Result<(f64, f64), Box<dyn Error>> {
    for _ in 0..warmups {
        black_box(run()?);
    }
    let mut seconds = Vec::with_capacity(repetitions);
    let mut retained_checksum = 0.0;
    for repetition in 0..repetitions {
        let start = Instant::now();
        let checksum = run()?;
        let elapsed = start.elapsed().as_secs_f64();
        black_box(checksum);
        retained_checksum = checksum;
        seconds.push(elapsed);
        println!("repetition={repetition} seconds={elapsed:.9} checksum={checksum:.12e}");
    }
    Ok((median(&mut seconds), retained_checksum))
}

struct RunSettings {
    monte_carlo: bool,
    monte_carlo_options: MonteCarloOptions,
    robust: bool,
    warmups: usize,
    repetitions: usize,
}

fn measure_scalar(
    pool: &ThreadPool,
    batch: &ScalarInferenceBatch,
    observations: &[f64],
    latitudes: &[f64],
    settings: &RunSettings,
) -> Result<(f64, f64), Box<dyn Error>> {
    measure(
        || {
            let solutions = pool.install(|| {
                if settings.robust && settings.monte_carlo {
                    batch.solve_time_major_with_missing_robust_and_monte_carlo_confidence(
                        observations,
                        latitudes,
                        RobustOptions::default(),
                        settings.monte_carlo_options,
                        LinearConfidence::Colored,
                    )
                } else if settings.robust {
                    batch.solve_time_major_with_missing_robust_and_linear_confidence(
                        observations,
                        latitudes,
                        RobustOptions::default(),
                        LinearConfidence::Colored,
                    )
                } else if settings.monte_carlo {
                    batch.solve_time_major_with_missing_and_monte_carlo_confidence(
                        observations,
                        latitudes,
                        settings.monte_carlo_options,
                        LinearConfidence::Colored,
                    )
                } else {
                    batch.solve_time_major_with_missing_and_linear_confidence(
                        observations,
                        latitudes,
                        LinearConfidence::Colored,
                    )
                }
            })?;
            Ok(solutions
                .iter()
                .flat_map(|solution| solution.amplitude_ci.as_ref().expect("requested CI"))
                .sum())
        },
        settings.warmups,
        settings.repetitions,
    )
}

fn measure_vector(
    pool: &ThreadPool,
    batch: &VectorInferenceBatch,
    eastward: &[f64],
    northward: &[f64],
    latitudes: &[f64],
    settings: &RunSettings,
) -> Result<(f64, f64), Box<dyn Error>> {
    measure(
        || {
            let solutions = pool.install(|| {
                if settings.robust && settings.monte_carlo {
                    batch.solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence(
                        eastward,
                        northward,
                        latitudes,
                        RobustOptions::default(),
                        settings.monte_carlo_options,
                        LinearConfidence::Colored,
                    )
                } else if settings.robust {
                    batch.solve_vector_time_major_with_missing_robust_and_linear_confidence(
                        eastward,
                        northward,
                        latitudes,
                        RobustOptions::default(),
                        LinearConfidence::Colored,
                    )
                } else if settings.monte_carlo {
                    batch.solve_vector_time_major_with_missing_and_monte_carlo_confidence(
                        eastward,
                        northward,
                        latitudes,
                        settings.monte_carlo_options,
                        LinearConfidence::Colored,
                    )
                } else {
                    batch.solve_vector_time_major_with_missing_and_linear_confidence(
                        eastward,
                        northward,
                        latitudes,
                        LinearConfidence::Colored,
                    )
                }
            })?;
            Ok(solutions
                .iter()
                .flat_map(|solution| solution.semi_major_ci.as_ref().expect("requested CI"))
                .sum())
        },
        settings.warmups,
        settings.repetitions,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let field = benchmark_setting("RUTIDE_BENCH_FIELD", "scalar", &["scalar", "vector"])?;
    let sampling = benchmark_setting(
        "RUTIDE_BENCH_SAMPLING",
        "irregular",
        &["regular", "irregular"],
    )?;
    let mode_name = benchmark_setting(
        "RUTIDE_BENCH_INFERENCE_MODE",
        "exact",
        &["exact", "approximate"],
    )?;
    let mode = if mode_name == "exact" {
        InferenceMode::Exact
    } else {
        InferenceMode::Approximate
    };
    let confidence = benchmark_setting(
        "RUTIDE_BENCH_CONFIDENCE",
        "linear",
        &["linear", "monte-carlo"],
    )?;
    let method = benchmark_setting("RUTIDE_BENCH_METHOD", "ols", &["ols", "robust"])?;
    let monte_carlo_options = MonteCarloOptions {
        realizations: setting("RUTIDE_BENCH_MC_REALIZATIONS", 200),
        seed: u64_setting("RUTIDE_BENCH_MC_SEED", 0),
    };
    let series_count = setting("RUTIDE_BENCH_SERIES", 100);
    let repetitions = setting("RUTIDE_BENCH_REPETITIONS", 5);
    let warmups = nonnegative_setting("RUTIDE_BENCH_WARMUPS", 1);
    let workers = setting("RUTIDE_BENCH_WORKERS", 1);
    let run_settings = RunSettings {
        monte_carlo: confidence == "monte-carlo",
        monte_carlo_options,
        robust: method == "robust",
        warmups,
        repetitions,
    };
    set_global_parallelism(Par::Seq);

    let times = times(&sampling);
    let latitudes = (0..series_count)
        .map(|series| {
            FIXTURE_LATITUDE
                + f64::from(u32::try_from(series).expect("series count fits u32")) * 1e-5
        })
        .collect::<Vec<_>>();
    let mut scalar_source = scalar_observations(&sampling)?;
    let (mut eastward_source, mut northward_source) = vector_observations(&times, &sampling);
    if method == "robust" {
        scalar_source[225] += 5.0;
        eastward_source[225] += 5.0;
        northward_source[513] -= 4.0;
    }
    let scalar = time_major(&scalar_source, series_count);
    let eastward = time_major(&eastward_source, series_count);
    let northward = time_major(&northward_source, series_count);
    let pool = ThreadPoolBuilder::new().num_threads(workers).build()?;

    let prepare_start = Instant::now();
    let (median_seconds, checksum) = if field == "scalar" {
        let batch = ScalarInferenceBatch::prepare_modified_julian_days(
            &times,
            &CONSTITUENTS,
            &SCALAR_RELATIONSHIPS,
            mode,
        )?;
        let prepare_seconds = prepare_start.elapsed().as_secs_f64();
        let measured = measure_scalar(&pool, &batch, &scalar, &latitudes, &run_settings)?;
        println!("prepare_seconds={prepare_seconds:.9}");
        measured
    } else {
        let batch = VectorInferenceBatch::prepare_modified_julian_days(
            &times,
            &CONSTITUENTS,
            &VECTOR_RELATIONSHIPS,
            mode,
        )?;
        let prepare_seconds = prepare_start.elapsed().as_secs_f64();
        let measured = measure_vector(
            &pool,
            &batch,
            &eastward,
            &northward,
            &latitudes,
            &run_settings,
        )?;
        println!("prepare_seconds={prepare_seconds:.9}");
        measured
    };
    println!(
        "summary field={field} sampling={sampling} inference_mode={mode_name} method={method} \
         confidence={confidence} \
         series={series_count} workers={workers} warmups={warmups} repetitions={repetitions} \
         mc_realizations={} mc_seed={} \
         median_seconds={median_seconds:.9} median_series_per_second={:.3} \
         checksum={checksum:.12e}",
        monte_carlo_options.realizations,
        monte_carlo_options.seed,
        f64::from(u32::try_from(series_count)?) / median_seconds,
    );
    Ok(())
}
