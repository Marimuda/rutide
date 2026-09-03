# Temporal current-query throughput: 2026-09-03

RUTide's current-atlas temporal path evaluates cached Cartesian current
coefficients directly at one prepared timestep. Compared with the existing
ellipse-oriented compatibility reconstruction, it is 5.76--5.79x faster on one
worker and reaches 16.74 million complete `(u, v)` current evaluations per
second for 100,000 locations with 16 workers.

This is an isolated temporal benchmark. It does not include mesh search,
spatial interpolation, Python/OpenDrift calls, or coefficient I/O, and must not
be presented as end-to-end particle-trajectory throughput.

## Environment and method

- host: AMD EPYC 7713, 128 physical CPUs, one visible NUMA node, Linux 6.8;
- compiler: Rust 1.98.0, release benchmark profile;
- workload: M2/S2/N2/K1/O1 at varying latitudes, exact Greenwich phase and
  exact nodal corrections, one target timestep, mean included, no trend;
- validation: every direct Cartesian result is compared exactly with the
  existing vector reconstruction before timing;
- measurement: two warmups followed by the median of seven repetitions for
  10,000 locations and five repetitions for 100,000 locations; and
- execution: a dedicated Rayon pool with the stated worker count, without CPU
  affinity on an otherwise non-isolated host.

The retained command shapes were:

```console
RUTIDE_BENCH_SERIES=10000 RUTIDE_BENCH_WORKERS=1 \
  cargo bench -p rutide-core --bench temporal_query_throughput

RUTIDE_BENCH_SERIES=100000 RUTIDE_BENCH_REPETITIONS=5 \
  RUTIDE_BENCH_WORKERS=16 \
  cargo bench -p rutide-core --bench temporal_query_throughput
```

## Measurements

| Locations | Workers | Compatibility path (s) | Cartesian query (s) | Speedup | Currents/s |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 1 | 0.006565893 | 0.001134790 | 5.786x | 8,812,203 |
| 10,000 | 16 | 0.005558816 | 0.003060880 | 1.816x | 3,267,034 |
| 100,000 | 1 | 0.073905203 | 0.012840150 | 5.756x | 7,788,071 |
| 100,000 | 16 | 0.019724938 | 0.005972555 | 3.303x | 16,743,253 |

At 10,000 locations, thread scheduling dominates the compact inner loop and one
worker is 2.70x faster than 16. At 100,000 locations, 16 workers are 2.15x
faster than one. A future end-to-end atlas reader should therefore choose its
parallelism at the batch level and retain a serial threshold instead of
unconditionally consuming the machine's full worker count.

## What changed

The compatibility path converts each stored ellipse into two temporary scalar
solutions, builds two output vectors, and evaluates eastward and northward
components separately. The query path instead:

1. converts each fitted ellipse once into four phase-wrap-safe Cartesian
   coefficients;
2. prepares target-time astronomy once;
3. evaluates both components in the same constituent loop; and
4. returns one compact `VectorCurrent` per location.

The direct interface also makes mean/trend inclusion explicit, preventing a
long prediction from silently extrapolating a fitted trend.

## Remaining performance boundary

This measurement establishes that temporal harmonic evaluation is unlikely to
be the dominant cost in the planned reader. The next benchmark must add the
FVCOM mesh locator, boundary-safe interpolation of Cartesian coefficients, and
binding overhead. Those measurements will determine whether element lookup,
memory layout, or Python/OpenDrift integration is the actual end-to-end
bottleneck.
