# Full-field memory optimization: 2026-08-31

The memory target is resolved for the current application profile. At 64
workers, median peak resident memory fell from 5,659,356 KiB (5.40 GiB) to
723,352 KiB (0.690 GiB), an 87.2% reduction. Median whole-process wall time also
fell from 3.15 to 1.51 seconds. The 32-node Python parity errors and both Rust
result digests are bit-for-bit unchanged.

## Scope and build

- optimized implementation: `f628317beab9d3e51a2174351e822d1307e92c79`
- parent application snapshot: `df7bec5426b442b52d50685dbb2f3a592e236618`
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`
- workload: all 75,160 distinct `zeta` series with 745 timestamps
- profile: M2, S2, N2, K1, and O1; exact Greenwich/nodal OLS with mean and trend
- Rust build: 1.98.0, thin LTO, `-C target-cpu=native`
- allocator: [`mimalloc` 0.1.52](https://docs.rs/mimalloc/0.1.52/mimalloc/)
  (`libmimalloc-sys` 0.1.49), application binary only
- source conversion: NetCDF-C promotes source `float32` values directly into
  the final Rust `f64` observation allocation

The allocator remains an application choice: `rutide-core` does not install or
require a global allocator. The CLI owns the policy because its full-field
parallel allocation pattern produced the measured problem.

## Diagnosis

At 10,000 series, changing Rayon's ordered `Result<Vec<_>>` collection to a
preallocated output vector did not change peak RSS (775,020 versus 770,796 KiB).
Grouping several series inside each Rayon task likewise measured 756,660 KiB.
This ruled out retained result-collection state and task granularity as the main
cause.

Forcing glibc to trim and map aggressively reduced the 10,000-series RSS to
101,376 KiB, but increased wall time from 0.60 to 4.55 seconds. This confirmed
that released short-lived QR storage was responsible for the resident high-water
mark, while also showing that eager OS reclamation was the wrong tradeoff.

Using mimalloc allowed those parallel allocations to be reused efficiently:

| 10,000-series configuration | Wall (s) | User CPU (s) | System CPU (s) | Peak RSS (KiB) |
|---|---:|---:|---:|---:|
| Original glibc allocator | 0.60 | 6.13 | 2.12 | 775,020 |
| mimalloc | 0.46 | 5.42 | 0.38 | 342,492 |

The second improvement removed an avoidable input-copy peak. The original read
allocated the 224 MB source `float32` field and then a second 448 MB `f64` field.
Requesting `f64` directly from NetCDF-C preserves every source `float32` value
exactly while allocating only the final buffer. On the full field this reduced a
mimalloc trial from 1,381,716 to 726,344 KiB and shortened the input stage from
0.732 to 0.532 seconds.

## Final full-field scaling

Every parallel configuration below has three separate warm-cache process
invocations. Stage and RSS values are medians. All repetitions produced full
result digest
`7ff5c55373bd3d071f1365f19c2cb57b8b6a4a6d0b66c5f455e9de80af9087f1`.

| Workers | Input (s) | Solve (s) | Hash (s) | Output (s) | Internal total (s) | Process wall (s) | Peak RSS (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.509 | 2.402 | 0.013 | 0.042 | 2.947 | 2.98 | 0.532 |
| 32 | 0.540 | 1.247 | 0.012 | 0.041 | 1.852 | 1.89 | 0.585 |
| 64 | 0.533 | 0.834 | 0.012 | 0.042 | 1.441 | 1.51 | 0.690 |
| 128 | 0.532 | 0.805 | 0.016 | 0.049 | 1.442 | 1.54 | 0.899 |

The retained 64-worker samples were:

| Repetition | Input (s) | Solve (s) | Internal total (s) | Process wall (s) | Peak RSS (KiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.509167 | 0.836373 | 1.392121 | 1.47 | 724,552 |
| 2 | 0.541664 | 0.833984 | 1.442126 | 1.51 | 723,352 |
| 3 | 0.532997 | 0.831200 | 1.440800 | 1.51 | 721,396 |

One 1-worker control used 497,700 KiB and 32.73 seconds whole-process wall time.
The 128-worker solve median is only 0.029 seconds lower than 64 workers, while
using about 31% more resident memory. Sixty-four workers therefore remains the
recommended setting on this machine.

Relative to the previous 64-worker medians, the final path reduces:

- peak RSS by 87.2%;
- process wall time by 52.1%;
- internal application time by 37.7%;
- NetCDF input time by 48.6%; and
- system CPU time by about 91.5%.

Against the retained practical Python whole-process median of 64.69 seconds, the
updated Rust application is 42.8x faster. This is still conservative because the
Rust process writes all coefficients to NetCDF while the Python baseline writes
only its digest and small JSON report.

## Correctness

The optimized application re-ran the frozen 32-node comparator. The result
digest remained
`d8782e9e4261101284da40d2301dcef8d83f67c5377dcb3fa5d13253863960f2`,
and the maximum differences from Python UTide remained:

| Quantity | Maximum absolute error | Tolerance |
|---|---:|---:|
| amplitude | 4.5582e-14 | 3e-12 |
| circular phase (degrees) | 8.6743e-11 | 3e-9 |
| mean | 7.4663e-15 | 3e-12 |
| slope per day | 8.4237e-16 | 3e-12 |
| frequency (cycles/hour) | 0 | 1e-15 |

No numerical solver or correction formula changed. A self-contained Rust test
also verifies that noncontiguous `float32` NetCDF columns are promoted and
reordered into `f64` without changing their values.
