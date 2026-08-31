//! Standalone throughput probe for exact Greenwich and nodal corrections.

use std::{env, error::Error, hint::black_box, time::Instant};

use faer::{Par, set_global_parallelism};
use rayon::ThreadPoolBuilder;
use rutide_core::{GreenwichNodalBatch, TidalConstituent};

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

fn times() -> Vec<f64> {
    (0_u32..745)
        .map(|index| {
            let day = 58_113 + index / 24;
            let milliseconds = (index % 24) * 3_600_000;
            f64::from(day) + f64::from(milliseconds) / 86_400_000.0
        })
        .collect()
}

fn observations() -> Result<Vec<f64>, Box<dyn Error>> {
    include_str!("../tests/data/fvcom_node_0_zeta_f32.hex")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let bits = u32::from_str_radix(line, 16)?;
            Ok(f64::from(f32::from_bits(bits)))
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let series_count = setting("RUTIDE_BENCH_SERIES", 1_000);
    let series_count_float = f64::from(u32::try_from(series_count)?);
    let repetitions = setting("RUTIDE_BENCH_REPETITIONS", 5);
    let warmups = setting("RUTIDE_BENCH_WARMUPS", 2);
    let workers = setting("RUTIDE_BENCH_WORKERS", 1);
    set_global_parallelism(Par::Seq);

    let time = times();
    let source = observations()?;
    let mut time_major = Vec::with_capacity(time.len() * series_count);
    for value in source.iter().copied() {
        time_major.extend(std::iter::repeat_n(value, series_count));
    }
    let latitudes = (0..series_count)
        .map(|series| {
            let series_float = f64::from(u32::try_from(series).expect("series count fits u32"));
            FIXTURE_LATITUDE + (series_float - series_count_float / 2.0) * 1e-5
        })
        .collect::<Vec<_>>();
    let pool = ThreadPoolBuilder::new().num_threads(workers).build()?;
    let prepare_start = Instant::now();
    let varying_batch = GreenwichNodalBatch::prepare_modified_julian_days(&time, &CONSTITUENTS)?;
    let prepare_seconds = prepare_start.elapsed().as_secs_f64();
    for _ in 0..warmups {
        black_box(pool.install(|| varying_batch.solve_time_major(&time_major, &latitudes))?);
    }

    let mut varying_seconds = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let varying_start = Instant::now();
        black_box(pool.install(|| varying_batch.solve_time_major(&time_major, &latitudes))?);
        let varying_elapsed = varying_start.elapsed().as_secs_f64();
        varying_seconds.push(varying_elapsed);
        println!("repetition={repetition} varying_seconds={varying_elapsed:.9}");
    }

    let varying_median = median(&mut varying_seconds);
    println!(
        "summary series={series_count} workers={workers} warmups={warmups} \
         repetitions={repetitions} prepare_seconds={prepare_seconds:.9} \
         varying_median_seconds={varying_median:.9} varying_series_per_second={:.3}",
        series_count_float / varying_median,
    );
    Ok(())
}
