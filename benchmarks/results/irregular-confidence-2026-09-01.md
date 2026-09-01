# Irregular colored-confidence benchmark: 2026-09-01

The new Lomb–Scargle path passes its scalar and vector correctness gates and
retains a material throughput advantage over pinned Python UTide. On the shared
100-series workload, one Rust worker is 3.57x faster for scalar analysis and 6.62x
faster for vector analysis than canonical single-process Python. With 32 workers
or processes, the corresponding advantages are 3.29x and 6.24x.

This is a focused irregular-confidence benchmark, not a whole-FVCOM application
claim. The production FVCOM fixture is hourly and therefore remains on the FFT
colored-spectrum path.

## Revisions, environment, and workload

- RUTide scalar Lomb–Scargle implementation: `b4121a9`;
- RUTide irregular vector support: `dac8ba97`;
- benchmark harness: `b45fafd`;
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`;
- machine: 2 x AMD EPYC 7713, 128 physical cores, Xen VM, Linux
  6.8.0-111-generic;
- Rust: 1.98.0, benchmark profile, thin LTO,
  `RUSTFLAGS="-C target-cpu=native"`;
- Python: CPython 3.10.12, NumPy 2.2.6, SciPy 1.13.1, one BLAS thread per
  process;
- series per repetition: 100, with latitude increasing by `1e-5` degrees per
  series;
- samples per source record: 745 over approximately 31 days;
- constituents: M2, S2, N2, K1, and O1;
- model: OLS, mean and trend, exact Greenwich phase and nodal corrections, and
  linear colored-noise confidence intervals;
- timestamps: nominally hourly with deterministic sinusoidal jitter;
- scalar validity: 742 retained observations, 742 used by the spectrum;
- vector joint validity: 741 retained observations, 740 used after UTide's
  odd-length spectral truncation;
- retained record span: 30.9569143332 days;
- largest retained-time gap: 2.0294369305 hours;
- evaluated Lomb–Scargle frequencies: 256 scalar and 255 vector;
- scalar frequency counts by UTide band: 4, 14, 14, 14, 13, 13, 14, 23, 147;
- vector frequency counts by UTide band: 4, 14, 14, 13, 13, 13, 14, 23, 147.

Each configuration used one unreported warm-up and five measured repetitions.
Rust's Rayon pool and Python's `fork` process pool were created before the warm-up
and retained across repetitions. Excluding pool creation favors Python relative
to a one-shot user invocation. Input generation and result formatting were
outside both measured regions. Rust's approximately 1.2–1.5 ms shared astronomy
preparation was also excluded.

The host was not otherwise isolated and CPU affinity was not fixed. Small timing
differences should therefore be treated as noise; the multi-fold language and
implementation differences are the decision-relevant result.

## Scalar results

| Implementation | Workers | Measured seconds | Median (s) | Series/s | Python-relative speedup |
|---|---:|---|---:|---:|---:|
| Python UTide | 1 | 3.366911, 3.212216, 3.187103, 3.567398, 3.460468 | 3.366911 | 29.70 | 1.00x |
| RUTide | 1 | 0.928967, 0.943860, 0.942249, 0.954633, 0.989054 | 0.943860 | 105.95 | 3.57x |
| Python UTide | 16 | 0.271267, 0.269504, 0.258240, 0.258195, 0.261866 | 0.261866 | 381.87 | 1.00x |
| RUTide | 16 | 0.074091, 0.071814, 0.071359, 0.070104, 0.069940 | 0.071359 | 1,401.37 | 3.67x |
| Python UTide | 32 | 0.161721, 0.169178, 0.163119, 0.157098, 0.158523 | 0.161721 | 618.35 | 1.00x |
| RUTide | 32 | 0.050631, 0.049199, 0.049471, 0.048654, 0.046449 | 0.049199 | 2,032.58 | 3.29x |

The stable 100-series first-amplitude-CI checksum was
`1.552265945565e0` in Rust and `1.552265945245e0` in Python. The small aggregate
difference is consistent with the much tighter per-quantity oracle tolerances.

## Vector results

| Implementation | Workers | Measured seconds | Median (s) | Series/s | Python-relative speedup |
|---|---:|---|---:|---:|---:|
| Python UTide | 1 | 6.041851, 6.580931, 6.561769, 6.598419, 6.250325 | 6.561769 | 15.24 | 1.00x |
| RUTide | 1 | 0.990994, 0.976813, 0.978001, 0.992903, 1.076228 | 0.990994 | 100.91 | 6.62x |
| Python UTide | 16 | 0.504444, 0.461499, 0.461581, 0.459128, 0.457922 | 0.461499 | 216.69 | 1.00x |
| RUTide | 16 | 0.076082, 0.074748, 0.073879, 0.072516, 0.072630 | 0.073879 | 1,353.57 | 6.25x |
| Python UTide | 32 | 0.302468, 0.313729, 0.304065, 0.284398, 0.274476 | 0.302468 | 330.61 | 1.00x |
| RUTide | 32 | 0.051031, 0.047188, 0.048483, 0.048594, 0.047243 | 0.048483 | 2,062.56 | 6.24x |

The stable 100-series first-major-axis-CI checksum was `2.988075381125e0` in
Rust and `2.988075380934e0` in Python.

Python's irregular complex periodogram calculates eastward and northward auto
spectra plus a cross-spectrum. Its current linearized vector confidence code only
uses the northward colored band and leaves the eastward coefficient pair white.
RUTide reproduces that output behavior while avoiding the two unused spectra.
The cross-spectrum remains required for the planned Monte Carlo covariance path
and will be implemented with that consumer.

## Correctness scope

The versioned oracle tests compare direct band power and end-to-end results. They
cover deterministic and pseudo-random jitter, isolated and clustered gaps,
frequency-band boundaries, coefficient variances, scalar amplitude/phase
intervals and SNR, and all four vector ellipse intervals plus SNR. The real-FVCOM
scalar fixture and deterministic vector fixture both use exact Greenwich/nodal
corrections.

## Decision

The Lomb–Scargle implementation clears the project's 3x performance threshold at
one, 16, and 32 workers for both scalar and vector profiles. No immediate spectral
kernel optimization is required for viability. Precomputing or caching
phase-shifted trigonometric bases remains a possible optimization for workloads
with many series sharing an identical irregular valid-time mask, but it would
trade several megabytes per mask for speed and should be justified by a real
observational workload before implementation.

The next scientific coverage increment is robust fitting. Sampling-quality
diagnostics—retained count, span, largest gap, and spectral-band coverage—should
also be exposed in application reports before presenting Lomb–Scargle intervals
as sufficient evidence that an arbitrary record is well sampled.
