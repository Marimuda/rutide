# FVCOM scalar application snapshot: 2026-08-31

The initial fixed-constituent application gate passes. On the complete 75,160-node
FVCOM elevation field, RUTide's median whole-process wall time is 3.15 seconds
versus 64.69 seconds for the best previously observed Python worker count: a
20.5x user-visible speedup. RUTide also writes all coefficients to NetCDF, while
the Python harness only hashes every result and writes a small JSON manifest.

This is evidence for continuing the narrow optimized Rust path, not evidence that
the implemented subset replaces all of UTide. Automatic constituent selection,
confidence intervals, gappy series, vector currents, and reconstruction remain
outside this measurement.

Follow-up: the [memory optimization snapshot](memory-optimization-2026-08-31.md)
reduces the 64-worker peak from 5.40 GiB to 0.690 GiB and process wall time from
3.15 to 1.51 seconds without changing results. The measurements below remain the
frozen pre-optimization application baseline.

## Revisions, workload, and environment

- RUTide: `febdefcebc8346b45b4deed8f64d9fd3452c9627`
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`
- source: `frs2f_0001.nc`, 25,778,391,080-byte CDF-2 container
- selected data: `Itime`, `Itime2`, `lat`, and all `zeta(745, 75160)` values
- logical source payload: 224,283,400 bytes
- profile: M2, S2, N2, K1, and O1; OLS; mean and trend; exact Greenwich
  phase and nodal corrections; no confidence intervals
- machine: 128 physical AMD EPYC 7713 cores, 251 GiB RAM, Xen VM, Linux
  6.8.0-111-generic
- Rust: 1.98.0, native CPU build, thin LTO, NetCDF-C 4.8.1
- Python: CPython 3.10.12, NumPy 2.2.6, SciPy 1.13.1, netCDF4 1.7.2,
  NetCDF-C 4.9.3, and one BLAS thread per process

The file was already cached: GNU `time` reported zero filesystem-input blocks
and zero major page faults for every retained run. These are therefore labeled
warm-cache measurements, not cold-cache measurements. CPU affinity was not
fixed and the host was not otherwise isolated, so the snapshot is suitable for
the current large-margin decision but not a controlled hardware publication.

Each retained row is a separate process invocation. Rust had one discarded
full-field warm-up before three repetitions at each worker count. Each Python
invocation loaded the field, performed its configured one-series warm-up, and
then executed one complete 32-process measurement. The earlier Python worker
sweep identified 32 processes as the best practical configuration.

## Correctness gate

The application selected the 32 frozen nodes, serialized them to NetCDF, and the
comparison command independently reran Python UTide for every node. All 32
series passed the frozen tolerances.

| Quantity | Maximum absolute error | Tolerance | Worst node |
|---|---:|---:|---:|
| amplitude | 4.5582e-14 | 3e-12 | 74949 |
| circular phase (degrees) | 8.6743e-11 | 3e-9 | 45095 |
| mean | 7.4663e-15 | 3e-12 | 30123 |
| slope per day | 8.4237e-16 | 3e-12 | 62197 |
| frequency (cycles/hour) | 0 | 1e-15 | 0 |

The 32-node Rust result digest was
`d8782e9e4261101284da40d2301dcef8d83f67c5377dcb3fa5d13253863960f2`.
The complete Rust field produced
`7ff5c55373bd3d071f1365f19c2cb57b8b6a4a6d0b66c5f455e9de80af9087f1`
in every worker configuration and repetition.

## Rust full-field scaling

Stage columns are medians from the application's internal monotonic clock.
Process wall and maximum resident set size come from GNU `time`. Internal total
ends after the NetCDF coefficient file is installed; process wall additionally
includes dynamic startup, argument handling, console JSON, and teardown.

| Workers | Input (s) | Solve (s) | Result hash (s) | NetCDF output (s) | Internal total (s) | Process wall (s) | Peak RSS (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.974 | 29.868 | 0.017 | 0.040 | 30.981 | 31.07 | 0.64 |
| 16 | 0.993 | 2.541 | 0.039 | 0.058 | 3.598 | 4.40 | 5.05 |
| 32 | 0.995 | 1.421 | 0.040 | 0.056 | 2.539 | 3.26 | 5.49 |
| 64 | 1.036 | 1.071 | 0.051 | 0.062 | 2.311 | 3.15 | 5.40 |
| 128 | 1.037 | 1.202 | 0.040 | 0.052 | 2.346 | 3.04 | 5.51 |

The 64-worker configuration has the best internal total and solve medians. The
small whole-process difference between 64 and 128 workers is below the
uncontrolled-host noise floor and does not justify using twice as many workers.

The retained 64-worker samples were:

| Repetition | Input (s) | Solve (s) | Hash (s) | Output (s) | Internal total (s) | Process wall (s) | Peak RSS (KiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.035977 | 1.248446 | 0.049686 | 0.038194 | 2.388144 | 3.27 | 5,758,056 |
| 2 | 1.111549 | 1.071086 | 0.050988 | 0.062139 | 2.311020 | 3.15 | 5,655,264 |
| 3 | 1.001912 | 1.047097 | 0.060644 | 0.063385 | 2.186312 | 3.02 | 5,659,356 |

The coefficient output is 8,432,699 bytes and contains node indices, latitudes,
frequencies, amplitudes, Greenwich phases, means, and slopes for every series.

## Practical Python full field

The Python solve timer includes process-pool creation, every `utide.solve` call,
summary construction, and digesting, but not the separately recorded NetCDF load.
It does not write a full coefficient array. Its stable aggregate digest was
`b8874da183e41c0cf8bb56b437eae960caa9b189dd3ed1e2e6b96c3273da9fc7`.
Python and Rust digest schemas differ, so equality is established by the numeric
32-node comparison rather than by comparing these two aggregate strings.

| Repetition | Input (s) | Solve and hash (s) | Measured sum (s) | Process wall (s) | Reported peak RSS (KiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.049164 | 59.657625 | 60.706790 | 63.16 | 1,046,588 |
| 2 | 1.246220 | 60.793947 | 62.040167 | 64.69 | 1,045,704 |
| 3 | 1.081960 | 61.925391 | 63.007351 | 65.83 | 1,047,452 |
| **median** | **1.081960** | **60.793947** | **62.040167** | **64.69** | **1,046,588** |

GNU `time` and `getrusage` do not sum simultaneously resident memory across the
32 forked Python workers, so the Python RSS column must not be interpreted as
aggregate pool memory or directly compared with the single Rust process.

## Decision

Using the full process boundary, the measured speedup is `64.69 / 3.15 = 20.5x`.
Using the sum of explicitly instrumented input and compute/output stages, it is
`62.040167 / 2.311020 = 26.8x`. The process-boundary figure is the primary,
more conservative result. Both greatly exceed the provisional 3x application
threshold, and the 64-worker Rust process uses only about 2.2% of host RAM.

The application gate therefore passes for this five-constituent, complete-series
scalar profile. The next optimization target is memory: the high-water mark
scales with the number of short-lived QR factorizations and is consistent with
the system allocator retaining their buffers. A one-worker run stays near 0.64
GiB, and chunking Rayon tasks did not materially reduce the high-water mark.
There were no major faults, and the allocation remained far below available
memory, but a reusable per-worker factorization workspace or a more suitable
allocation strategy should be tested before broadening the solver catalog.
