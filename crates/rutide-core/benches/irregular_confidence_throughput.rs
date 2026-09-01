//! Standalone throughput probe for irregular colored-noise confidence intervals.

use std::{env, error::Error, hint::black_box, time::Instant};

use faer::{Par, set_global_parallelism};
use rayon::ThreadPoolBuilder;
use rutide_core::{GreenwichNodalBatch, LinearConfidence, TidalConstituent};

const CONSTITUENTS: [TidalConstituent; 5] = [
    TidalConstituent::M2,
    TidalConstituent::S2,
    TidalConstituent::N2,
    TidalConstituent::K1,
    TidalConstituent::O1,
];
const FIXTURE_LATITUDE: f64 = 60.957_717_895_507_81;

fn setting(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn irregular_times() -> Vec<f64> {
    let mut times = (0_u32..745)
        .map(|index| 58_113.0 + f64::from(index) / 24.0)
        .collect::<Vec<_>>();
    let final_index = times.len() - 1;
    for (index, time) in times.iter_mut().enumerate().take(final_index).skip(1) {
        let index = f64::from(u32::try_from(index).expect("fixture index fits u32"));
        *time += 0.002 * (index * 0.37).sin() + 0.0007 * (index * 0.11).cos();
    }
    times
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
    for index in [0, 137, 411] {
        values[index] = f64::NAN;
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
    for index in [0, 137] {
        eastward[index] = f64::NAN;
    }
    for index in [2, 411] {
        northward[index] = f64::NAN;
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

fn main() -> Result<(), Box<dyn Error>> {
    let field = env::var("RUTIDE_BENCH_FIELD").unwrap_or_else(|_| "scalar".to_owned());
    if field != "scalar" && field != "vector" {
        return Err("RUTIDE_BENCH_FIELD must be scalar or vector".into());
    }
    let series_count = setting("RUTIDE_BENCH_SERIES", 100);
    let repetitions = setting("RUTIDE_BENCH_REPETITIONS", 5);
    let warmups = setting("RUTIDE_BENCH_WARMUPS", 1);
    let workers = setting("RUTIDE_BENCH_WORKERS", 1);
    set_global_parallelism(Par::Seq);

    let times = irregular_times();
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

    let run = || -> Result<f64, Box<dyn Error>> {
        if field == "scalar" {
            let solutions = pool.install(|| {
                batch.solve_time_major_with_missing_and_linear_confidence(
                    &scalar,
                    &latitudes,
                    LinearConfidence::Colored,
                )
            })?;
            Ok(solutions
                .iter()
                .map(|solution| solution.amplitude_ci.as_ref().expect("requested CI")[0])
                .sum())
        } else {
            let solutions = pool.install(|| {
                batch.solve_vector_time_major_with_missing_and_linear_confidence(
                    &eastward,
                    &northward,
                    &latitudes,
                    LinearConfidence::Colored,
                )
            })?;
            Ok(solutions
                .iter()
                .map(|solution| solution.semi_major_ci.as_ref().expect("requested CI")[0])
                .sum())
        }
    };
    for _ in 0..warmups {
        black_box(run()?);
    }

    let mut seconds = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let start = Instant::now();
        let checksum = run()?;
        let elapsed = start.elapsed().as_secs_f64();
        black_box(checksum);
        seconds.push(elapsed);
        println!("repetition={repetition} seconds={elapsed:.9} checksum={checksum:.12e}");
    }
    let median_seconds = median(&mut seconds);
    println!(
        "summary field={field} series={series_count} workers={workers} warmups={warmups} \
         repetitions={repetitions} prepare_seconds={prepare_seconds:.9} \
         median_seconds={median_seconds:.9} median_series_per_second={:.3}",
        f64::from(u32::try_from(series_count)?) / median_seconds,
    );
    Ok(())
}
