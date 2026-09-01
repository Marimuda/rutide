//! Standalone throughput probe for robust fits with colored linear confidence.

use std::{env, error::Error, hint::black_box, time::Instant};

use faer::{Par, set_global_parallelism};
use rayon::ThreadPoolBuilder;
use rutide_core::{
    AnalysisError, GreenwichNodalBatch, LinearConfidence, MonteCarloOptions, RobustOptions,
    ScalarSolution, TidalConstituent, VectorSolution,
};

const CONSTITUENTS: [TidalConstituent; 5] = [
    TidalConstituent::M2,
    TidalConstituent::S2,
    TidalConstituent::N2,
    TidalConstituent::K1,
    TidalConstituent::O1,
];
const FIXTURE_LATITUDE: f64 = 60.957_717_895_507_81;

#[derive(Clone, Copy)]
struct Measurement {
    checksum: f64,
    iteration_sum: usize,
    minimum_iterations: usize,
    maximum_iterations: usize,
}

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

fn times() -> Vec<f64> {
    (0_u32..745)
        .map(|index| 58_113.0 + f64::from(index) / 24.0)
        .collect()
}

fn scalar_observations() -> Result<Vec<f64>, Box<dyn Error>> {
    let mut values = include_str!("../tests/data/fvcom_node_0_zeta_f32.hex")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let bits = u32::from_str_radix(line, 16)?;
            Ok(f64::from(f32::from_bits(bits)))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    for (index, offset) in [(71, 5.0), (218, -4.0), (503, 6.0)] {
        values[index] += offset;
    }
    Ok(values)
}

fn vector_observations(times: &[f64]) -> (Vec<f64>, Vec<f64>) {
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
    eastward[71] += 5.0;
    northward[218] -= 4.0;
    eastward[503] += 4.0;
    northward[503] += 3.0;
    (eastward, northward)
}

fn time_major(source: &[f64], series_count: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(source.len() * series_count);
    for value in source.iter().copied() {
        values.extend(std::iter::repeat_n(value, series_count));
    }
    values
}

fn scalar_measurement(solutions: &[ScalarSolution]) -> Measurement {
    measurement(solutions.iter().map(|solution| {
        (
            solution.amplitude_ci.as_ref().expect("requested CI")[0],
            solution
                .robust
                .as_ref()
                .expect("requested robust fit")
                .iterations,
        )
    }))
}

fn vector_measurement(solutions: &[VectorSolution]) -> Measurement {
    measurement(solutions.iter().map(|solution| {
        (
            solution.semi_major_ci.as_ref().expect("requested CI")[0],
            solution
                .robust
                .as_ref()
                .expect("requested robust fit")
                .iterations,
        )
    }))
}

fn measurement(values: impl Iterator<Item = (f64, usize)>) -> Measurement {
    let mut checksum = 0.0;
    let mut iteration_sum = 0;
    let mut minimum_iterations = usize::MAX;
    let mut maximum_iterations = 0;
    for (value, iterations) in values {
        checksum += value;
        iteration_sum += iterations;
        minimum_iterations = minimum_iterations.min(iterations);
        maximum_iterations = maximum_iterations.max(iterations);
    }
    Measurement {
        checksum,
        iteration_sum,
        minimum_iterations,
        maximum_iterations,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn solve_scalar(
    batch: &GreenwichNodalBatch,
    observations: &[f64],
    latitudes: &[f64],
    confidence: &str,
) -> Result<Vec<ScalarSolution>, AnalysisError> {
    if confidence == "linear" {
        batch.solve_time_major_with_missing_robust_and_linear_confidence(
            observations,
            latitudes,
            RobustOptions::default(),
            LinearConfidence::Colored,
        )
    } else {
        batch.solve_time_major_with_missing_robust_and_monte_carlo_confidence(
            observations,
            latitudes,
            RobustOptions::default(),
            MonteCarloOptions::default(),
            LinearConfidence::Colored,
        )
    }
}

fn solve_vector(
    batch: &GreenwichNodalBatch,
    eastward: &[f64],
    northward: &[f64],
    latitudes: &[f64],
    confidence: &str,
) -> Result<Vec<VectorSolution>, AnalysisError> {
    if confidence == "linear" {
        batch.solve_vector_time_major_with_missing_robust_and_linear_confidence(
            eastward,
            northward,
            latitudes,
            RobustOptions::default(),
            LinearConfidence::Colored,
        )
    } else {
        batch.solve_vector_time_major_with_missing_robust_and_monte_carlo_confidence(
            eastward,
            northward,
            latitudes,
            RobustOptions::default(),
            MonteCarloOptions::default(),
            LinearConfidence::Colored,
        )
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let field = env::var("RUTIDE_BENCH_FIELD").unwrap_or_else(|_| "scalar".to_owned());
    if field != "scalar" && field != "vector" {
        return Err("RUTIDE_BENCH_FIELD must be scalar or vector".into());
    }
    let confidence = env::var("RUTIDE_BENCH_CONFIDENCE").unwrap_or_else(|_| "linear".to_owned());
    if confidence != "linear" && confidence != "monte-carlo" {
        return Err("RUTIDE_BENCH_CONFIDENCE must be linear or monte-carlo".into());
    }
    let series_count = setting("RUTIDE_BENCH_SERIES", 100);
    let repetitions = setting("RUTIDE_BENCH_REPETITIONS", 5);
    let warmups = nonnegative_setting("RUTIDE_BENCH_WARMUPS", 1);
    let workers = setting("RUTIDE_BENCH_WORKERS", 1);
    let realizations = if confidence == "monte-carlo" {
        MonteCarloOptions::default().realizations
    } else {
        0
    };
    set_global_parallelism(Par::Seq);

    let times = times();
    let latitudes = (0..series_count)
        .map(|series| {
            FIXTURE_LATITUDE
                + f64::from(u32::try_from(series).expect("series count fits u32")) * 1e-5
        })
        .collect::<Vec<_>>();
    let scalar = time_major(&scalar_observations()?, series_count);
    let (eastward_source, northward_source) = vector_observations(&times);
    let eastward = time_major(&eastward_source, series_count);
    let northward = time_major(&northward_source, series_count);
    let pool = ThreadPoolBuilder::new().num_threads(workers).build()?;
    let prepare_start = Instant::now();
    let batch = GreenwichNodalBatch::prepare_modified_julian_days(&times, &CONSTITUENTS)?;
    let prepare_seconds = prepare_start.elapsed().as_secs_f64();

    let run = || -> Result<Measurement, Box<dyn Error>> {
        if field == "scalar" {
            let solutions =
                pool.install(|| solve_scalar(&batch, &scalar, &latitudes, &confidence))?;
            Ok(scalar_measurement(&solutions))
        } else {
            let solutions = pool
                .install(|| solve_vector(&batch, &eastward, &northward, &latitudes, &confidence))?;
            Ok(vector_measurement(&solutions))
        }
    };
    for _ in 0..warmups {
        black_box(run()?);
    }

    let mut seconds = Vec::with_capacity(repetitions);
    let mut retained_measurement = None;
    for repetition in 0..repetitions {
        let start = Instant::now();
        let result = run()?;
        let elapsed = start.elapsed().as_secs_f64();
        black_box(result);
        seconds.push(elapsed);
        retained_measurement = Some(result);
        println!(
            "repetition={repetition} seconds={elapsed:.9} checksum={:.12e} \
             iteration_sum={} iteration_min={} iteration_max={}",
            result.checksum,
            result.iteration_sum,
            result.minimum_iterations,
            result.maximum_iterations,
        );
    }
    let result = retained_measurement.expect("at least one repetition");
    let median_seconds = median(&mut seconds);
    println!(
        "summary field={field} confidence={confidence} realizations={} series={series_count} \
         workers={workers} warmups={warmups} repetitions={repetitions} \
         prepare_seconds={prepare_seconds:.9} \
         median_seconds={median_seconds:.9} median_series_per_second={:.3} \
         iteration_mean={:.3} iteration_min={} iteration_max={}",
        realizations,
        f64::from(u32::try_from(series_count)?) / median_seconds,
        f64::from(u32::try_from(result.iteration_sum)?) / f64::from(u32::try_from(series_count)?),
        result.minimum_iterations,
        result.maximum_iterations,
    );
    Ok(())
}
