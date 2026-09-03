# Compute-kernel optimization: 2026-09-03

This round optimized only costs isolated by matched measurements. It removes
per-timestamp corrected-basis allocations, keeps nodal corrections in Cartesian
form, reuses latitude-independent astronomical bases across sufficiently large
shared-mask batches, and applies a cached least-squares projection to wide
fixed-design batches. The largest isolated gain is 3.92x; installed-binding
reconstruction improves by 1.29–1.90x, and the real FVCOM reconstruction stage
improves by 1.87x.

These results do not imply that Rust is automatically fast. They show that
layout, allocation, repeated transcendental work, and the chosen linear-algebra
operation mattered more than the source language.

## Environment and method

- host: two AMD EPYC 7713 sockets exposed as 128 physical CPUs, one visible
  NUMA node, AVX2/FMA, Xen virtual machine;
- compiler: Rust 1.98.0, thin LTO, one codegen unit, and
  `-C target-cpu=native`;
- workload: 745 samples, M2/S2/N2/K1/O1, mean and trend unless stated otherwise;
- timing: steady-state medians from the existing black-box benchmark probes;
- affinity: not pinned, so small regressions in highly parallel/noisy profiles
  are not treated as actionable; and
- hardware counters: unavailable because the host sets
  `perf_event_paranoid=4`. Diagnosis therefore used mode isolation, worker
  scaling, source inspection, and before/after controlled timings.

No result below includes Python environment creation or wheel compilation. The
real-data acceptance timings use the installed public Python API and include its
object materialization within each timed stage.

## Bottlenecks demonstrated by measurement

The exact Greenwich/nodal solver originally took 1.65 seconds for 5,000 series
on one worker, versus 0.96–1.02 seconds with nodal corrections disabled or raw
phase. Inspection of that isolated difference found a fresh nodal-correction
vector for every timestamp of every series and a Cartesian-to-polar-to-Cartesian
round trip in the innermost basis loop.

After eliminating that work, exact, linear-time, and disabled nodal modes all
fell into the same approximately 0.75-second range before astronomical sharing.
This confirmed that the removed allocation and conversion work—not the required
astronomical model—caused the exact-nodal tax. Precomputing the astronomical
complex basis for the reused complete mask then reduced the 5,000-series exact
case to approximately 0.42 seconds.

The fixed raw batch showed the opposite scaling signature: 10,000 series were
fastest on one worker and became slower with 16–128 workers. The design is fixed
for every right-hand side, so a lazily cached QR-derived least-squares projection
turns the wide solve into one dense matrix multiplication. Small batches retain
the direct QR path because building the projection would cost more than it saves.

## Corrected solve throughput

Ten thousand independent series, exact Greenwich phase and exact nodal
corrections:

| Workers | Before (s) | After (s) | Speedup |
|---:|---:|---:|---:|
| 1 | 3.035460 | 0.773626 | 3.92x |
| 16 | 0.239768 | 0.105353 | 2.28x |
| 64 | 0.087484 | 0.048006 | 1.82x |
| 128 | 0.075295 | 0.049087 | 1.53x |

The new optimum is approximately 64 workers. The 128-worker result is flat or
slower, so using every logical CPU is not the fastest policy for this workload.

The same corrected-basis improvements also benefit irregular colored-confidence
and robust paths. At 1,000 series, scalar/vector irregular runs improve by
1.18–1.57x across the stable 1/16/64-worker comparisons. Scalar robust runs
improve by 1.05–1.38x; vector robust runs improve by 1.12–1.36x at 1, 16, and 128
workers. A single noisy vector robust 64-worker result regressed from 0.0385 to
0.0412 seconds and is not used to claim a gain.

## Fixed raw batch throughput

Ten thousand series with one reusable fixed design and no confidence intervals:

| Workers | Before (s) | After (s) | Speedup |
|---:|---:|---:|---:|
| 1 | 0.149169 | 0.067539 | 2.21x |
| 16 | 0.179070 | 0.085139 | 2.10x |
| 64 | 0.185757 | 0.099431 | 1.87x |
| 128 | 0.188472 | 0.097732 | 1.93x |

One worker remains fastest because the post-optimization operation is small and
bandwidth/output dominated. Even a cold first projected solve measured 0.0984
seconds, below the old 0.1492-second steady-state result.

For this profile the cached projection is 71,520 bytes (69.8 KiB). Constructing
it temporarily solves against an approximately 4.24 MiB identity matrix. The
projection is created only on the first batch containing at least 16 series and
is then reused safely by subsequent calls.

## Public Python reconstruction

The installed-wheel probe used 100 series, 745 samples, five constituents, and
16 requested workers:

| Field/API | Before (s) | After (s) | Speedup |
|---|---:|---:|---:|
| scalar batch reconstruction | 0.007563 | 0.004610 | 1.64x |
| scalar one-series loop | 0.201735 | 0.109417 | 1.84x |
| vector batch reconstruction | 0.011355 | 0.005978 | 1.90x |
| vector one-series loop | 0.149442 | 0.115756 | 1.29x |

The new reconstructor precomputes astronomy phasors for target times, converts
the selected solution coefficients once, and performs complex multiplication in
the inner loop. It no longer recomputes phases and polar conversions for every
time/constituent pair.

## Real-data acceptance

The installed public API was re-run on the complete NOAA/NDBC ADCP profile and
the deterministic 4,096-series selection from the largest local FVCOM fixture:

| Workload/stage | Previous (s) | Optimized (s) | Speedup |
|---|---:|---:|---:|
| ADCP solve | 2.7881 | 2.7297 | 1.02x |
| ADCP reconstruction | 0.0959 | 0.0612 | 1.57x |
| FVCOM solve | 0.2908 | 0.2716 | 1.07x |
| FVCOM reconstruction | 0.1991 | 0.1065 | 1.87x |

Coefficient save/load and restored reconstruction remained bitwise equal in
both workloads. The corrected solve gain is smaller here because the public
workflow includes robust/confidence/result-construction work outside the
optimized basis kernel and the workloads are smaller than the isolated 10,000-
series throughput probe.

## Memory bounds and correctness

Astronomical bases are cached only for the existing bounded shared-mask groups:
at least 16 series must reuse a mask and at most four groups are cached. A
745-by-146 complex-`f64` basis occupies 1,740,320 bytes (1.66 MiB), so the
worst-case cache added by this policy is approximately 6.64 MiB per batch.
Unique and lightly reused masks take the direct low-memory path.

The corrected-basis, inference, fixed-raw, and reconstruction oracle suites pass.
Batch/individual equivalence has dedicated coverage for both the shared
astronomical cache and projected raw solve. Installed Python loop and batch
outputs remain mutually bitwise equal, persistence round trips remain bitwise
equal, and the pinned-Python errors remain approximately `1e-13` for fitted
coefficients, `1e-11` degrees for phase, and `7e-12` for reconstruction.

The Cartesian formulation can change final floating-point bits compared with an
older RUTide build because it removes inverse polar transformations. That is an
algebraically equivalent evaluation-order change, not a scientific-contract
change; deterministic outputs and the documented cross-language tolerances are
retained.

## Remaining measured ceiling

The corrected kernel stops scaling materially near 64 workers and the fixed raw
kernel is fastest on one worker. The next candidates are therefore not more
threads or generic unsafe/SIMD rewrites. They need separate measurements:

- batch-native structure-of-arrays results to reduce thousands of small result
  vectors and Python object materialization;
- reusable workspaces and specialized small weighted solves for robust IRLS;
- matrix projection for sufficiently reused shared-mask Lomb–Scargle records;
  and
- profiling the binding conversion path independently from the native solve.

Those are follow-up hypotheses, not optimizations justified by this snapshot.
