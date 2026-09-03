# Bounded FVCOM input pipeline: 2026-09-03

Automatic scalar and regular-vector FVCOM analyses now overlap one dedicated
NetCDF read/conversion with the worker pool's solve of the preceding chunk. On
the complete warm-cache fixture, scalar process wall falls from 1.05 to 0.91
seconds and depth-averaged vector wall falls from 3.93 to 2.79 seconds at the
matched 64-worker setting. That is a 1.15x and 1.41x process-wall improvement,
respectively, with unchanged result digests and slightly lower peak RSS.

A vector worker sweep moves the practical optimum to the 48–56-worker range.
The retained 48-worker median is 2.50 seconds, making the current native process
approximately 50.8x faster than the retained 126.97-second practical Python
UTide baseline. The 64-worker scalar result is approximately 71.1x faster than
its retained 64.69-second Python baseline.

## Scope and environment

- implementation parent: `4daa849`;
- source: `frs2f_0001.nc`, a 25,778,391,080-byte CDF-2 container;
- scalar input: all 75,160 `zeta` nodes and 745 timestamps;
- vector input: all 144,860 depth-averaged `ua`/`va` elements and 745 timestamps;
- profile: M2/S2/N2/K1/O1, OLS, mean and trend, exact Greenwich phase and nodal
  corrections, no confidence interval or reconstruction;
- build: Rust 1.98.0, release profile, thin LTO, one codegen unit,
  `-C target-cpu=native`, NetCDF-C 4.8.1, and mimalloc;
- execution: warm page cache, no fixed CPU affinity, and an otherwise
  non-isolated Xen host with 128 physical AMD EPYC 7713 CPUs; and
- measurement: three separate GNU `time` process invocations per retained row,
  with application stage durations from its monotonic clock.

Hardware performance counters remain unavailable because the host sets
`perf_event_paranoid=4`. Large relative changes are retained; worker-count
differences of a few percent remain host-noise sensitive.

## Design and memory bound

The reader owns the only source NetCDF handle. A zero-capacity rendezvous channel
allows it to construct chunk N+1 while the caller solves chunk N, but prevents it
from starting chunk N+2 until the caller receives N+1. At most two input chunks
can therefore exist.

Automatic planning counts both buffers inside the existing 512 MiB logical
observation budget. The vector chunk width changes from 44,992 series in four
sequential chunks to approximately 22,500 series in seven overlapped chunks. The
scalar field changes from one 75,160-series chunk to two chunks, the first of
which contains 44,992 series. The reported `maximum_observation_buffer_bytes`
now covers all concurrently resident promoted input arrays.

Explicit `--chunk-series` values remain sequential, preserving their exact
memory and reproducibility meaning. Small inputs that fit inside the double-
buffer budget also remain sequential. Fixed-physical-depth input remains
sequential because its read stage includes worker-pool interpolation and several
geometry/current buffers; competing with the solve pool there requires a
separate measurement and memory model.

Dropping the pipeline closes its receiver before joining the reader. An analysis
or output error therefore releases a reader blocked in the rendezvous send
instead of leaking or deadlocking a thread.

## Matched 64-worker result

The sequential controls use explicit chunk sizes equal to the previous
automatic plan. Active input time measures work performed inside the reader. It
overlaps solve/result work in the pipelined rows, so stage columns are not
additive there; `Internal total` and process wall remain elapsed durations.

| Field/path | Input (s) | Solve (s) | Result (s) | Output (s) | Internal total (s) | Process wall (s) | Peak RSS (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scalar sequential | 0.3890 | 0.4023 | 0.1009 | 0.0579 | 0.9579 | 1.05 | 875,404 |
| Scalar pipelined | 0.4086 | 0.4073 | 0.1043 | 0.0510 | 0.8282 | 0.91 | 849,528 |
| Vector sequential | 1.7631 | 1.6536 | 0.2325 | 0.1308 | 3.8154 | 3.93 | 1,004,216 |
| Vector pipelined | 1.9733 | 1.7292 | 0.2898 | 0.1262 | 2.6782 | 2.79 | 966,124 |

Input and solve each become modestly slower while running together because they
share memory bandwidth and CPU resources. The elapsed reduction demonstrates
that the overlap remains profitable despite that contention. Compared with the
matched controls, scalar wall falls 13.3%, vector wall falls 29.0%, scalar RSS
falls 3.0%, and vector RSS falls 3.8%.

## Vector worker sweep

All rows use automatic seven-chunk overlap. The 48-worker result is the practical
recommendation: it is within 0.01 seconds of the lowest median, has much lower
variance than the 56-worker samples, and uses fewer resources.

| Workers | Process wall samples (s) | Median (s) | Median peak RSS (KiB) |
|---:|---|---:|---:|
| 16 | 2.63, 2.52, 2.55 | 2.55 | 791,248 |
| 24 | 2.57, 2.56, 2.59 | 2.57 | 832,792 |
| 32 | 2.58, 2.74, 2.58 | 2.58 | 872,712 |
| 40 | 2.68, 2.60, 2.63 | 2.63 | 917,856 |
| 48 | 2.57, 2.50, 2.47 | 2.50 | 924,180 |
| 56 | 2.95, 2.45, 2.49 | 2.49 | 978,104 |
| 64 | 2.79, 3.00, 2.65 | 2.79 | 966,124 |
| 128 | 2.79, 3.11, 2.96 | 2.96 | 1,250,956 |

Using every CPU is counterproductive. At 128 workers the process consumes much
more aggregate CPU time and memory without improving wall time.

The scalar 16/32/48/64-worker medians are 1.08, 0.93, 0.92, and 0.91 seconds.
Sixty-four workers remains the scalar choice on this host, although 32–48 workers
offer almost the same elapsed time with lower RSS.

## Correctness and output

All scalar runs produced digest
`01e318eb246b8583fdce747d50f30022a6fb31386c61df675cc469ed9e97cbeb`.
All vector runs produced digest
`276f174b0fd124fd36c6146e145972146d685944f9376dae8406c2825da6dafe`.
These match their sequential controls across every worker and chunk width.

The scalar output is 32,492,753 bytes and the vector output is 77,672,816 bytes.
Output takes only about 3–6% of elapsed time and remains a poor optimization
target. Focused tests cover the two-buffer bound, exact chunk contents and joint
missing masks, explicit sequential overrides, cancellation while the producer is
blocked, and existing chunk/worker/Monte-Carlo determinism.

## Remaining opportunity

The active reader still takes approximately 1.9 seconds for the vector input and
contends with solving. Further improvement requires isolating NetCDF-C traversal,
source conversion, and memory-copy costs. A specialized CDF-2 reader or reusable
contiguous sidecar should be considered only after a matched decoder benchmark;
neither is justified by this pipeline result alone.
