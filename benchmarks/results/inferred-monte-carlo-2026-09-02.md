# Inferred Monte Carlo confidence check: 2026-09-02

Inferred Monte Carlo confidence retains the throughput characteristics of the
existing inference batch. On the focused 100-series irregular workload, 200
realizations add 0.05–2.24% to scalar elapsed time and 27.73–33.74% to vector
elapsed time relative to linear confidence measured in the same session. The
larger vector cost is expected: it samples a complete 4×4 covariance and converts
every realization to current-ellipse parameters.

This is a targeted Rust regression and scaling check, not a cross-language
speedup claim. Python UTide raises `NotImplementedError` when Monte Carlo
confidence and inference are combined, so no scientifically equivalent Python
timing exists.

## Revisions, environment, and workload

- scalar shared-draw propagation: `465b5cf`;
- vector rotary shared-draw propagation: `fe1c5bf`;
- deterministic missing/irregular batches: `666508f`;
- CLI and NetCDF integration: `4fa0f3e`;
- benchmark mode: `e2ff0d2` (the same source was measured immediately before
  this checkpoint commit);
- machine: 2 x AMD EPYC 7713, 128 physical cores, Xen VM, Linux
  6.8.0-111-generic;
- Rust: 1.98.0, benchmark profile, thin LTO,
  `RUSTFLAGS="-Ctarget-cpu=native"`;
- 100 series, 745 nominal samples over approximately 31 days, deterministic
  timestamp jitter, scalar gaps, and a joint vector missing-value mask;
- exact inference, OLS, mean and trend, exact Greenwich/nodal corrections, and
  colored Lomb–Scargle residual spectra;
- five requested constituents, with S2 constrained from M2 and O1 from K1;
- 200 Monte Carlo realizations, root seed zero;
- one unreported warm-up and five measured repetitions per cell.

The host was not isolated and CPU affinity was not fixed. Linear and Monte Carlo
cells were measured back-to-back with identical inputs, but these sub-130 ms
absolute timings should still be treated as a focused regression check rather
than publication-grade performance evidence.

## Results

| Field | Workers | Linear median (s) | Monte Carlo median (s) | MC overhead | MC series/s |
|---|---:|---:|---:|---:|---:|
| Scalar | 1 | 0.085406219 | 0.085453031 | 0.05% | 1,170.23 |
| Scalar | 16 | 0.009990334 | 0.010214315 | 2.24% | 9,790.18 |
| Vector | 1 | 0.095696642 | 0.127981033 | 33.74% | 781.37 |
| Vector | 16 | 0.011059488 | 0.014126344 | 27.73% | 7,078.97 |

### Monte Carlo samples

| Field | Workers | Seconds |
|---|---:|---|
| Scalar | 1 | 0.086472060, 0.085453031, 0.087107862, 0.085242712, 0.082691719 |
| Scalar | 16 | 0.011478971, 0.010126700, 0.009381271, 0.010214315, 0.010786970 |
| Vector | 1 | 0.129335271, 0.127569182, 0.127260625, 0.127981033, 0.130673239 |
| Vector | 16 | 0.014214039, 0.014126344, 0.014595324, 0.013917138, 0.013212313 |

The scalar Monte Carlo checksum was `1.064635297054e1` with both worker counts;
the vector checksum was `1.384502058829e0` with both worker counts. This confirms
that derived per-series random streams make the results bitwise independent of
Rayon scheduling for the benchmarked path.

## Decision

Correlated inference propagation does not introduce a new optimization target.
Scalar sampling is effectively hidden by the irregular colored-spectrum work.
Vector sampling has a visible but bounded cost and still processes more than
7,000 series/s at 16 workers. The next development increment can therefore
return to solver-option parity and sampling diagnostics rather than optimizing
this path prematurely.
