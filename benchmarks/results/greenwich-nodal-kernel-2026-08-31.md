# Exact Greenwich/nodal kernel snapshot: 2026-08-31

This snapshot evaluates the second scientifically comparable Rust kernel. It is
a solve-only microbenchmark, not the application-level NetCDF benchmark.

## Revisions and profile

- RUTide implementation: `70706b0b58d93dd41f9203b14354c2ccf9653368`
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`
- timestamps: 745 exact hourly Modified Julian Days
- constituents: M2, S2, N2, K1, and O1
- model: OLS with mean and trend, exact Greenwich phase, exact nodal/satellite
  corrections, and no confidence intervals
- Rust build: benchmark profile with `-C target-cpu=native`
- machine: 128 physical AMD EPYC 7713 cores

## Compatibility result

For real FVCOM `zeta[:, 0]` at latitude `60.95771789550781`, the direct and bulk
Rust APIs produce the same result and match the pinned Python oracle within:

- `3e-12` for amplitude, mean, and slope; and
- `3e-9` degrees for Greenwich phase.

The test uses the exact versioned `float32` observation bits. Astronomy at the
fixture reference time and reference-time constituent frequencies have separate
unit tests.

## Python baseline

Python processed the first 1,000 distinct FVCOM nodes. The solve timer includes
result canonicalization and hashing. BLAS was restricted to one thread in each
process. All worker configurations produced digest
`9704eee18720a685370be3e8e94f1c43504198d81da6eeddb0265f477f62b31c`.

| Mode | Workers | Median solve (s) | Throughput (series/s) | Repetitions |
|---|---:|---:|---:|---:|
| canonical | 1 | 20.8022 | 48.1 | 3 |
| process pool | 16 | 1.9663 | 508.6 | 3 |
| process pool | 32 | 1.5754 | 634.8 | 3 |
| process pool | 64 | 2.0679 | 483.6 | 3 |
| process pool | 128 | 3.2588 | 306.9 | 3 |

The 32-process entry is the best observed median. A clean-tree repeat measured
1.6636 seconds, so the faster 1.5754-second result is retained as the more
conservative Rust comparison.

## Rust scaling

Rust used 10,000 varying-latitude fits to obtain stable sustained timings. It
repeated the real node-zero observations and varied latitude near the fixture
latitude. Observation values do not change the QR workload, but this synthetic
spatial input must be replaced by the actual field in the application benchmark.
The one-time shared astronomical preparation took 0.0011 to 0.0018 seconds and is
reported separately from the medians below.

| Workers | Median solve (s) | Throughput (series/s) | Repetitions |
|---:|---:|---:|---:|
| 1 | 4.3743 | 2,286 | 7 |
| 8 | 0.5569 | 17,956 | 7 |
| 16 | 0.3160 | 31,642 | 7 |
| 32 | 0.1741 | 57,441 | 7 |
| 64 | 0.1420 | 70,445 | 7 |
| 128 | 0.1585 | 63,074 | 7 |

The corrected Rust kernel is about 48 times the canonical Python throughput with
one worker and about 111 times the best observed Python process-pool throughput
at each implementation's best worker count. A separate full-size synthetic run
of 75,160 series with 64 workers had a 0.7419-second median, or 101,301 series/s.

## Interpretation and limitations

This result passes the planned bulk gate for the narrow five-constituent profile
and strongly supports proceeding to application integration. It does not yet
establish the project's end-to-end speedup because:

- Python uses 1,000 actual observation and latitude columns while Rust currently
  uses repeated observations and nearby synthetic latitudes;
- Python's timer includes result hashing while Rust's does not;
- NetCDF reading, missing-value grouping, and output serialization are excluded;
- only five non-shallow constituents are implemented; and
- automatic selection, confidence intervals, vectors, and reconstruction remain
  outside the Rust path.

The next decision-quality measurement must read the actual `zeta(time, node)`
field and retain distinct latitude-specific outputs through serialization.
