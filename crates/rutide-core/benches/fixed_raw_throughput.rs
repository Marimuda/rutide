//! Standalone throughput probe for the reusable fixed-raw OLS kernel.

use std::{env, error::Error, hint::black_box, time::Instant};

use faer::{Par, set_global_parallelism};
use rutide_core::{Constituent, FixedRawOls};

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

fn constituents() -> Vec<Constituent> {
    [
        ("M2", 0.080_511_400_671_577_2),
        ("S2", 0.083_333_333_333_333_33),
        ("N2", 0.078_999_248_699_109_28),
        ("K1", 0.041_780_746_221_637_22),
        ("O1", 0.038_730_654_449_939_97),
    ]
    .into_iter()
    .map(|(name, frequency)| Constituent::new(name, frequency))
    .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let series_count = setting("RUTIDE_BENCH_SERIES", 1_000);
    let repetitions = setting("RUTIDE_BENCH_REPETITIONS", 5);
    let workers = setting("RUTIDE_BENCH_WORKERS", 1);
    let warmups = setting("RUTIDE_BENCH_WARMUPS", 5);
    let series_count_float = f64::from(u32::try_from(series_count)?);
    if workers == 1 {
        set_global_parallelism(Par::Seq);
    } else {
        set_global_parallelism(Par::rayon(workers));
    }

    let time = times();
    let source = observations()?;
    let mut time_major = Vec::with_capacity(time.len() * series_count);
    for value in source {
        time_major.extend(std::iter::repeat_n(value, series_count));
    }

    let prepare_start = Instant::now();
    let model = FixedRawOls::prepare(&time, &constituents())?;
    let prepare_seconds = prepare_start.elapsed().as_secs_f64();
    for _ in 0..warmups {
        black_box(model.solve_many_time_major(&time_major, series_count)?);
    }

    let mut seconds = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let start = Instant::now();
        let solutions = model.solve_many_time_major(&time_major, series_count)?;
        let elapsed = start.elapsed().as_secs_f64();
        black_box(solutions);
        seconds.push(elapsed);
        println!(
            "repetition={repetition} seconds={elapsed:.9} series_per_second={:.3}",
            series_count_float / elapsed
        );
    }
    seconds.sort_by(f64::total_cmp);
    let median = seconds[seconds.len() / 2];
    println!(
        "summary series={series_count} workers={workers} warmups={warmups} \
         repetitions={repetitions} \
         prepare_seconds={prepare_seconds:.9} median_seconds={median:.9} \
         median_series_per_second={:.3}",
        series_count_float / median
    );
    Ok(())
}
