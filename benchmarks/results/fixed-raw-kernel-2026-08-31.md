# Fixed-raw Rust kernel snapshot: 2026-08-31

This snapshot validates the first scientifically comparable Rust kernel. It is a
microbenchmark and parity gate, not the final application benchmark.

## Compatibility result

The Rust fixed-raw OLS result for FVCOM `zeta[:, 0]` matches the pinned Python
UTide oracle with acceptance tolerances of:

- `2e-12` for amplitude, mean, and slope; and
- `2e-9` degrees for raw phase.

The real-FVCOM parity test and exact `float32` input bit patterns are versioned
under `crates/rutide-core/tests/`.

## Throughput configuration

Python used the first 1,000 real `zeta` nodes. Rust used 1,000 right-hand sides
containing repetitions of the real node-zero series. Observation values do not
change the dense QR solve workload, but this distinction must be removed in the
eventual end-to-end benchmark.

Both used 745 timestamps, the same five frequencies, mean, trend, OLS, raw phase,
no nodal corrections, and no confidence intervals. Rust was built with the bench
profile and `-C target-cpu=native`; its factorization was prepared once and reused.

| Implementation | Workers | Median solve (s) | Throughput (series/s) | Repetitions |
|---|---:|---:|---:|---:|
| Python canonical | 1 | 2.6632 | 375.5 | 3 |
| Python process pool | 32 | 0.7868 | 1,270.9 | 3 |
| Rust/faer shared QR | 1 | 0.00343 | 291,202 | 10 |
| Rust/faer shared QR | 8 | 0.01608 | 62,198 | 10 |
| Rust/faer shared QR | 16 | 0.01690 | 59,181 | 10 |
| Rust/faer shared QR | 32 | 0.02111 | 47,366 | 10 |

The small 745-by-12 design favors sequential execution of the shared
factorization; parallel solver overhead exceeds useful work at 1,000 right-hand
sides. Spatial chunk-level parallelism may still help larger workloads and must be
measured separately.

The sequential Rust solve is about 229 times faster than the tuned 32-process
Python result in this narrow test. Cold Rust preparation took about 0.079 seconds;
including it reduces the advantage to about 9.5 times for only 1,000 series.
Preparation is amortized over 75,160 series in the primary workload.

## Limitations

- The Python solve timer currently includes canonical result hashing; the Rust
  probe constructs results but does not hash them.
- NetCDF I/O and result serialization are excluded.
- Rust does not yet implement Greenwich phase, nodal corrections, confidence
  intervals, automatic constituent selection, or missing observations.
- The Rust microbenchmark repeats one real series rather than reading 1,000
  distinct series.

The magnitude is therefore evidence that the shared-factorization design is worth
continuing, not a final speedup claim.
