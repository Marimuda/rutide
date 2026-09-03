# Constituent-identifiability diagnostics: 2026-09-03

RUTide's extended Codiga diagnostics remain an opt-in analysis product. On the
complete retained FVCOM field, enabling them changed median scalar process wall
from 1.07 s to 2.01 s and vector process wall from 3.03 s to 4.92 s. Median peak
RSS increased by approximately 82 MiB and 67 MiB, respectively. Runs without
the flag preserve the pre-diagnostic result content and execute no extended
diagnostic calculation.

Python UTide exposes only PE and SNR, so there is no matched Python runtime for
the restored RR/RNM/Corrmax/K/tidal-variance suite. This benchmark therefore
isolates RUTide diagnostics off versus on under otherwise identical options.

## Scope and environment

- source: `frs2f_0001.nc`, a 25,778,391,080-byte CDF-2 container;
- measured source tree: the diagnostics integration based on parent `30ce750`;
- scalar input: all 75,160 `zeta` nodes and 745 timestamps;
- vector input: all 144,860 depth-averaged `ua`/`va` elements and 745 timestamps;
- profile: M2/S2/N2/K1/O1, OLS, mean and trend, exact Greenwich phase and nodal
  corrections, linear white confidence, no reconstruction, and an inclusive
  diagnostic SNR threshold of 2 when enabled;
- build: Rust 1.98.0, release profile, thin LTO, one codegen unit,
  `RUSTFLAGS="-C target-cpu=native"`, NetCDF-C 4.8.1, and mimalloc;
- execution: warm page cache, 64 scalar workers, 48 vector workers, no fixed CPU
  affinity, and an otherwise non-isolated Xen host with 128 physical AMD EPYC
  7713 CPUs, 251 GiB RAM, AVX2/FMA, and one NUMA node; and
- measurement: three separate GNU `time` process invocations per row. Medians
  below use process wall and peak RSS; stage details are retained from the final
  application report in each row.

Each enabled command differs from its control only by
`--constituent-diagnostics`. Both use `--confidence linear`; this is required to
define SNR, RNM, `SNRallc`, and the significant-constituent variance subset.
The four retained command shapes were:

```console
/usr/bin/time -f '%e %M' target/release/rutide analyze-scalar \
  --input ../projects/fvcom/claude_scratchpad/baroclinic_vikc1701/run/frs2f_0001.nc \
  --output /tmp/rutide-diag-baseline.nc --report /tmp/rutide-diag-baseline.json \
  --confidence linear --white-noise --workers 64 --overwrite

/usr/bin/time -f '%e %M' target/release/rutide analyze-scalar \
  --input ../projects/fvcom/claude_scratchpad/baroclinic_vikc1701/run/frs2f_0001.nc \
  --output /tmp/rutide-diag-enabled.nc --report /tmp/rutide-diag-enabled.json \
  --confidence linear --white-noise --constituent-diagnostics \
  --workers 64 --overwrite

/usr/bin/time -f '%e %M' target/release/rutide analyze-vector \
  --input ../projects/fvcom/claude_scratchpad/baroclinic_vikc1701/run/frs2f_0001.nc \
  --output /tmp/rutide-vector-diag-baseline.nc \
  --report /tmp/rutide-vector-diag-baseline.json \
  --confidence linear --white-noise --workers 48 --overwrite

/usr/bin/time -f '%e %M' target/release/rutide analyze-vector \
  --input ../projects/fvcom/claude_scratchpad/baroclinic_vikc1701/run/frs2f_0001.nc \
  --output /tmp/rutide-vector-diag-enabled.nc \
  --report /tmp/rutide-vector-diag-enabled.json \
  --confidence linear --white-noise --constituent-diagnostics \
  --workers 48 --overwrite
```

The release binary was built once with
`RUSTFLAGS="-C target-cpu=native" cargo build --release --locked` before timing.

## Whole-field measurements

| Field | Diagnostics | Wall samples (s) | Median wall (s) | Median RSS (KiB) | Output bytes |
|---|---|---:|---:|---:|---:|
| Scalar | off | 0.93, 1.07, 1.17 | 1.07 | 856,280 | 41,513,303 |
| Scalar | on | 1.83, 2.01, 2.01 | 2.01 | 940,256 | 70,385,016 |
| Vector | off | 2.70, 3.03, 3.26 | 3.03 | 953,316 | 106,646,465 |
| Vector | on | 4.56, 5.04, 4.92 | 4.92 | 1,022,356 | 162,286,539 |

The scalar enabled profile is 1.88x the control wall time, an increase of 0.94 s
or 87.9%; median peak RSS rises 83,976 KiB, or 9.8%. The vector enabled profile
is 1.62x the control wall time, an increase of 1.89 s or 62.4%; median peak RSS
rises 69,040 KiB, or 7.2%. Output grows because every record gains whole-model
metrics and both neighbor directions gain RR, RNM, Corrmax, and stable neighbor
indices for each constituent.

## Stage evidence

Input work can overlap solve/result processing, so stage values are diagnostic
and are not summed to reproduce elapsed time. `Total` is authoritative.

| Field | Diagnostics | Input (s) | Solve (s) | Result (s) | Output (s) | Total (s) |
|---|---|---:|---:|---:|---:|---:|
| Scalar | off | 0.4171 | 0.5059 | 0.1075 | 0.2087 | 1.0686 |
| Scalar | on | 0.3444 | 0.4288 | 0.8806 | 0.3368 | 1.8802 |
| Vector | off | 2.0425 | 1.7837 | 0.3215 | 0.4863 | 3.1267 |
| Vector | on | 1.8649 | 1.4845 | 2.2694 | 0.6875 | 4.7577 |

The added time is overwhelmingly result processing: approximately 0.77 s for
scalar and 1.95 s for vector in the retained reports, compared with only about
0.13 s and 0.20 s of additional output work. The feature is therefore not I/O-
bound. The demonstrated next optimization target is an integrated solve-and-
diagnose path that reuses solve-time bases, fitted reconstructions, and small
factorization products rather than rebuilding them through the public post-fit
diagnostic interface. That work should retain the independent post-fit API and
must be benchmarked before it changes the application path.

Buffered full-field output accounts for the measured RSS increase. Vertically
resolved vector output writes diagnostics per chunk to the existing incremental
NetCDF path and does not retain a whole-run diagnostic vector.

## Correctness

The controls produced scalar digest
`d8fd9e2fabd3e8cea72a016be3da41dc122dd5939b53db86cf66b71103878723`
and vector digest
`565a993378b1aa5034d31bb6cf40b33cbda03cb1e4a4b83cfab1b3e025c1159c`.
Enabled runs produced diagnostic-aware scalar digest
`b6543a83f7b99c5d3fae62aa631efdbb2e0934759ece435a2cba17d74238d75a`
and vector digest
`8b6efbe7da5f2254d0f3ef8685024e56447811ddcf0a51fa1700e1aeea034d63`.
Repeated enabled runs and chunked application tests establish stable digests;
focused NetCDF tests cover complete field names, configuration attributes, `-1`
missing-neighbor indices, `NaN` unavailable values, and fixed-depth incremental
output.

Equation-level fixtures independently implement Codiga equation 81 and equations
99–102 from the original MATLAB `ut_solv.m` and compare them with the public Rust
kernels. The broader scalar/vector, inference, robust, missing-value, batch, and
Python paths remain covered by their retained suites.
