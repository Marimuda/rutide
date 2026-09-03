//! Standalone current-atlas temporal-query throughput probe.

use std::{env, error::Error, hint::black_box, time::Instant};

use rayon::ThreadPoolBuilder;
use rutide_core::{
    GreenwichNodalReconstructor, NonHarmonicTerms, ReconstructionFilter, TidalConstituent,
    VectorSolution,
};

const CONSTITUENTS: [TidalConstituent; 5] = [
    TidalConstituent::M2,
    TidalConstituent::S2,
    TidalConstituent::N2,
    TidalConstituent::K1,
    TidalConstituent::O1,
];

fn setting(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn solution() -> VectorSolution {
    VectorSolution {
        semi_major: vec![0.9, 0.3, 0.2, 0.15, 0.1],
        semi_minor: vec![-0.2, 0.05, -0.03, 0.02, -0.01],
        inclination_degrees: vec![35.0, 42.0, 28.0, 95.0, 110.0],
        phase_degrees: vec![359.0, 22.0, 145.0, 270.0, 5.0],
        percent_energy: vec![70.0, 15.0, 8.0, 5.0, 2.0],
        semi_major_ci: None,
        semi_minor_ci: None,
        inclination_ci_degrees: None,
        phase_ci_degrees: None,
        signal_to_noise: None,
        eastward_mean: 0.08,
        northward_mean: -0.03,
        eastward_slope_per_day: 0.0,
        northward_slope_per_day: 0.0,
        reference_time_days: 60_000.0,
        robust: None,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let series_count = setting("RUTIDE_BENCH_SERIES", 10_000);
    let repetitions = setting("RUTIDE_BENCH_REPETITIONS", 7);
    let warmups = setting("RUTIDE_BENCH_WARMUPS", 2);
    let workers = setting("RUTIDE_BENCH_WORKERS", 1);
    let series_count_float = f64::from(u32::try_from(series_count)?);
    let latitudes = (0..series_count)
        .map(|series| {
            u32::try_from(series % 10_000)
                .map(|position| 58.0 + 6.0 * f64::from(position) / 10_000.0)
        })
        .collect::<Result<Vec<_>, std::num::TryFromIntError>>()?;
    let ellipse_solutions = vec![solution(); series_count];
    let cartesian_solutions = ellipse_solutions
        .iter()
        .map(VectorSolution::cartesian)
        .collect::<Result<Vec<_>, _>>()?;
    let reconstructor = GreenwichNodalReconstructor::prepare_modified_julian_days(
        &[61_234.25],
        60_000.0,
        &CONSTITUENTS,
    )?;
    let pool = ThreadPoolBuilder::new().num_threads(workers).build()?;
    let filter = ReconstructionFilter::All;

    let legacy = pool.install(|| {
        reconstructor.reconstruct_many_vectors_series_major(&ellipse_solutions, &latitudes, &filter)
    })?;
    let query = pool.install(|| {
        reconstructor.reconstruct_cartesian_vectors_at_time_index(
            0,
            &cartesian_solutions,
            &latitudes,
            &filter,
            NonHarmonicTerms::MeanAndTrend,
        )
    })?;
    for (legacy, query) in legacy.iter().zip(&query) {
        assert_eq!(legacy.eastward, [query.eastward]);
        assert_eq!(legacy.northward, [query.northward]);
    }

    for _ in 0..warmups {
        black_box(pool.install(|| {
            reconstructor.reconstruct_many_vectors_series_major(
                &ellipse_solutions,
                &latitudes,
                &filter,
            )
        })?);
        black_box(pool.install(|| {
            reconstructor.reconstruct_cartesian_vectors_at_time_index(
                0,
                &cartesian_solutions,
                &latitudes,
                &filter,
                NonHarmonicTerms::MeanAndTrend,
            )
        })?);
    }

    let mut legacy_seconds = Vec::with_capacity(repetitions);
    let mut query_seconds = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let start = Instant::now();
        black_box(pool.install(|| {
            reconstructor.reconstruct_many_vectors_series_major(
                &ellipse_solutions,
                &latitudes,
                &filter,
            )
        })?);
        let legacy = start.elapsed().as_secs_f64();

        let start = Instant::now();
        black_box(pool.install(|| {
            reconstructor.reconstruct_cartesian_vectors_at_time_index(
                0,
                &cartesian_solutions,
                &latitudes,
                &filter,
                NonHarmonicTerms::MeanAndTrend,
            )
        })?);
        let query = start.elapsed().as_secs_f64();
        legacy_seconds.push(legacy);
        query_seconds.push(query);
        println!("repetition={repetition} legacy_seconds={legacy:.9} query_seconds={query:.9}");
    }

    let legacy = median(&mut legacy_seconds);
    let query = median(&mut query_seconds);
    println!(
        "summary series={series_count} constituents={} workers={workers} warmups={warmups} \
         repetitions={repetitions} legacy_median_seconds={legacy:.9} \
         query_median_seconds={query:.9} speedup={:.3} query_currents_per_second={:.3}",
        CONSTITUENTS.len(),
        legacy / query,
        series_count_float / query,
    );
    Ok(())
}
