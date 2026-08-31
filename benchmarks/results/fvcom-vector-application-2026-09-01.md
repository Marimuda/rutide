# FVCOM vector-current application snapshot: 2026-09-01

The five-constituent depth-averaged current application passes the provisional
coverage and performance gates. On all 144,860 FVCOM elements, RUTide's median
whole-process wall time is 5.39 seconds versus 126.97 seconds for the practical
32-process Python UTide baseline: a 23.6x user-visible speedup. RUTide also writes
a 44,053,960-byte NetCDF file containing every ellipse result, while the Python
harness hashes results and writes only a small JSON manifest.

This snapshot measures the fixed M2, S2, N2, K1, and O1 profile without confidence
intervals or reconstruction. Colored vector confidence intervals were separately
validated on all 32 frozen correctness elements, but are not included in the
full-field timing.

## Revisions, workload, and environment

- RUTide vector implementation: `6b1b80df`;
- benchmark/oracle harness: `edc66348`;
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`;
- source: `frs2f_0001.nc`, 25,778,391,080-byte CDF-2 container;
- selected data: `Itime`, `Itime2`, `latc`, all `ua(745, 144860)`, and all
  `va(745, 144860)` values;
- logical source payload: 863,951,000 bytes (about 0.805 GiB), not the complete
  25.78 GB container;
- missing observations: zero in this fixture; joint-component missing handling is
  covered by synthetic NetCDF and core tests;
- profile: five fixed constituents, OLS, mean and trend, exact Greenwich phase and
  nodal corrections, no confidence intervals, and no reconstruction;
- machine: 2 x AMD EPYC 7713, 128 physical cores, 251 GiB RAM, one exposed NUMA
  node, Xen VM, Linux 6.8.0-111-generic;
- Rust: 1.98.0, `--release`, thin LTO, one codegen unit,
  `RUSTFLAGS="-C target-cpu=native"`, NetCDF-C 4.8.1, mimalloc;
- Python: CPython 3.10.12, NumPy 2.2.6, SciPy 1.13.1, netCDF4 1.7.2,
  NetCDF-C 4.9.3, and one BLAS thread per process.

The file was warm in the page cache: GNU `time` reported no major faults and zero
filesystem-input blocks. CPU affinity was not fixed and the host was not otherwise
isolated, so differences on the order of a few percent are noise; the large
cross-language margin is the decision-relevant result.

Rust used one discarded 64-worker whole-field warm-up, a worker sweep, and five
retained repetitions at both 64 and 128 workers. Python used three independent
process invocations. Each loaded the full vector field, performed one unreported
one-series warm-up, then ran one complete 32-process solve.

## Correctness

The RUTide command selected the 32 deterministic fixture elements and serialized
their ellipses. The independent comparator matched constituents by name and reran
the pinned Python oracle for every element. The fixed no-CI native build and the
colored-CI native build both passed.

| Quantity | Maximum absolute error | Tolerance |
|---|---:|---:|
| reconstructed east/north harmonic coefficient | 1.0049e-12 | 5e-12 |
| semi-major axis | 1.4078e-13 | 5e-12 |
| signed semi-minor axis | 4.5394e-14 | 5e-12 |
| inclination (raw circular error, degrees) | 1.1006e-10 | coefficient backstop |
| Greenwich phase (raw circular error, degrees) | 1.5581e-10 | coefficient backstop |
| percent energy | 2.1924e-11 | 1e-9 |
| eastward/northward mean | 2.1511e-14 | 5e-12 |
| eastward/northward slope per day | 3.5838e-15 | 5e-12 |
| frequency (cycles/hour) | 0 | 1e-15 |
| reference time (MJD) | 0 | 1e-12 |

The colored vector interval check additionally produced:

| Quantity | Maximum absolute error | Tolerance |
|---|---:|---:|
| semi-major CI | 2.9228e-11 | 1e-9 |
| semi-minor CI | 1.4941e-11 | 1e-9 |
| inclination CI (degrees) | 1.2554e-7 | 1e-5 |
| phase CI (degrees) | 1.2581e-7 | 1e-5 |
| SNR | 2.6918e-6 | 1e-6 absolute + 1e-8 relative |

The full Rust field produced the stable digest
`cf6b69d869a9d841cd1b3461ecaf0193b4a2288e021dfa79d20abbb69b321a2a`
for every worker count and repetition. Python's independent canonical schema
produced
`eb3d2fae96dd9df173013593356bbdb84d72b1a02053187f071e6fe81a3701d9`
in all three runs. Digest equality is not expected across different schemas;
numeric parity is established by the 32-element comparison above.

## Rust full-field scaling

Rows for 1, 16, and 32 workers are one post-warm-up sample. Rows for 64 and 128
workers are medians of five complete processes. Process wall and peak RSS are
from GNU `time`; internal stages use the application's monotonic clock.

| Workers | Input (s) | Solve (s) | Internal total (s) | Process wall (s) | Peak RSS (GiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.944 | 58.931 | 61.076 | 61.13 | 1.809 |
| 16 | 2.238 | 5.528 | 7.953 | 8.01 | 1.873 |
| 32 | 2.341 | 3.702 | 6.232 | 6.32 | 1.972 |
| 64 | 2.239 | 2.789 | 5.270 | 5.39 | 2.099 |
| 128 | 2.358 | 2.706 | 5.243 | 5.37 | 2.370 |

The 64-to-128 difference is below the uncontrolled-host noise floor. Sixty-four
workers are the practical choice: 128 workers improve median process wall by only
0.02 seconds while retaining about 0.27 GiB more memory. Relative to one worker,
the 64-worker solve is 21.1x faster. CPU utilization reaches only about 14 cores
on average at 64 workers, showing that short per-element factorizations and memory
traffic limit further scaling.

The five retained 64-worker samples were:

| Repetition | Input (s) | Solve (s) | Hash (s) | Output (s) | Internal total (s) | Process wall (s) | Peak RSS (KiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.393161 | 2.733801 | 0.099271 | 0.082119 | 5.323969 | 5.44 | 2,223,128 |
| 2 | 2.262772 | 2.858962 | 0.102588 | 0.084067 | 5.325608 | 5.45 | 2,191,676 |
| 3 | 2.157523 | 2.657727 | 0.106947 | 0.082185 | 5.018899 | 5.13 | 2,185,604 |
| 4 | 2.197956 | 2.789024 | 0.089166 | 0.078600 | 5.170601 | 5.28 | 2,214,572 |
| 5 | 2.238955 | 2.842723 | 0.094795 | 0.072421 | 5.269959 | 5.39 | 2,201,352 |
| **median** | **2.238955** | **2.789024** | **0.099271** | **0.082119** | **5.269959** | **5.39** | **2,201,352** |

## Practical Python full field

Python's solve timer includes pool creation, every two-component `utide.solve`,
summary construction, and digesting, but excludes the separately recorded NetCDF
load. It does not write a full coefficient file, so the application comparison
slightly favors Python.

| Repetition | Input (s) | Solve and hash (s) | Measured sum (s) | Process wall (s) | Reported peak RSS (KiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.232100 | 115.875946 | 120.108046 | 126.97 | 3,264,116 |
| 2 | 4.699000 | 115.129859 | 119.828859 | 126.47 | 3,264,112 |
| 3 | 4.631815 | 115.892742 | 120.524558 | 127.02 | 3,264,920 |
| **median** | **4.631815** | **115.875946** | **120.108046** | **126.97** | **3,264,116** |

The reported Python high-water mark belongs to the parent/process measurement and
does not sum simultaneous resident memory across all 32 forked workers, so it is
not a directly comparable aggregate pool-memory figure.

## Decision

The primary process-boundary speedup is `126.97 / 5.39 = 23.6x`. The sum of
explicitly instrumented stages gives `120.108046 / 5.269959 = 22.8x`. At the
solve layer, Rust processes about 51,939 elements/s and Python about 1,250
elements/s, a 41.5x throughput ratio. All exceed the 3x application threshold by
a wide margin.

The result supports continuing the optimized Rust path for vector currents. The
next resource opportunity is input memory: two complete f64 component arrays
dominate the 2.10 GiB Rust high-water mark. A bounded series-chunk pipeline could
reduce this without changing the scientific kernel. Functionality still outside
the current scope includes robust fitting, inference, Monte Carlo intervals, and
Lomb–Scargle spectra for truly irregular colored-noise records.
