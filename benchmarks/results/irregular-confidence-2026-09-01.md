# Irregular colored-confidence benchmark: 2026-09-01

The bounded reusable Lomb–Scargle plan restores the performance profile that was
missing from the first irregular-confidence implementation. On the same
100-series workload, optimized RUTide is now 38.59x faster than pinned Python
UTide for scalar analysis and 70.06x faster for vector analysis on one worker.
At 16 workers the advantages are 26.12x and 41.59x; at 32 workers they are 20.05x
and 34.58x.

Against the original direct Rust Lomb–Scargle kernel, the optimization is
5.54–10.82x faster depending on field and worker count. The earlier 3–6x Python
comparison was therefore not an inherent cost of irregular sampling: it exposed
repeated timestamp-only work in the first implementation.

This is a focused irregular-confidence benchmark, not a whole-FVCOM application
claim. The production FVCOM fixture is hourly and therefore remains on the FFT
colored-spectrum path.

## Revisions, environment, and workload

- bounded reusable Lomb–Scargle plans: `97f675c`;
- initial scalar Lomb–Scargle implementation: `b4121a9`;
- irregular vector support: `dac8ba97`;
- benchmark harness: `b45fafd`, extended by `97f675c` to permit a zero-warm-up
  cold measurement;
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`;
- machine: 2 x AMD EPYC 7713, 128 physical cores, Xen VM, Linux
  6.8.0-111-generic;
- Rust: 1.98.0, benchmark profile, thin LTO,
  `RUSTFLAGS="-C target-cpu=native"`;
- Python: CPython 3.10.12, NumPy 2.2.6, SciPy 1.13.1, one BLAS thread per
  process;
- primary comparison: 100 series per repetition, with latitude increasing by
  `1e-5` degrees per series;
- scaling comparison: 1,000 series per repetition;
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

Rust and Python pools were created before the warm-up and retained across
repetitions. Input generation and result formatting were outside the measured
regions. Rust's approximately 1.1–1.6 ms shared astronomy preparation was also
excluded. The host was not otherwise isolated and CPU affinity was not fixed.

## Optimization and memory policy

For a repeated valid-time mask, RUTide now calculates the irregular Hann window,
frequency grid, phase-shifted sine/cosine bases, and their energies once. Each
series then needs only residual windowing, contiguous dot products, normalization,
and band averaging. Basis construction is parallelized across frequencies and
uses time relative to the record start to avoid unnecessarily large trigonometric
arguments.

The speed path is deliberately bounded:

- a batch mask must be shared by at least 16 series before it is planned;
- only the four most-used eligible masks in a batch may receive plans;
- the two trigonometric bases are capped at 16 MiB per mask;
- unique, lightly reused, or long-record masks retain the direct `O(N * F)`
  kernel and do not allocate a basis cache.

The primary scalar plan stores 3,039,232 basis bytes and the vector plan stores
3,019,200 basis bytes, plus small window, frequency, vector-header, and lock
overheads. Thus the normal one-mask workload pays about 3 MiB—not one copy per
series—and the batch policy caps cached basis storage at 64 MiB.

## Primary 100-series results

Each configuration used one unreported warm-up and five measured repetitions.
The warm-up initializes the reusable Rust plan, so these are steady-state numbers
for a prepared batch or repeated analysis. Cold behavior is reported separately.

### Scalar

| Implementation | Workers | Measured seconds | Median (s) | Series/s | Python-relative speedup |
|---|---:|---|---:|---:|---:|
| Python UTide | 1 | 3.366911, 3.212216, 3.187103, 3.567398, 3.460468 | 3.366911 | 29.70 | 1.00x |
| RUTide | 1 | 0.095352, 0.089142, 0.085169, 0.087249, 0.083900 | 0.087249 | 1,146.15 | 38.59x |
| Python UTide | 16 | 0.271267, 0.269504, 0.258240, 0.258195, 0.261866 | 0.261866 | 381.87 | 1.00x |
| RUTide | 16 | 0.011465, 0.010762, 0.009333, 0.009404, 0.010027 | 0.010027 | 9,973.39 | 26.12x |
| Python UTide | 32 | 0.161721, 0.169178, 0.163119, 0.157098, 0.158523 | 0.161721 | 618.35 | 1.00x |
| RUTide | 32 | 0.010061, 0.008216, 0.007598, 0.007589, 0.008066 | 0.008066 | 12,397.36 | 20.05x |

The stable 100-series first-amplitude-CI checksum is `1.552265945565e0` in
Rust and `1.552265945245e0` in Python.

### Vector

| Implementation | Workers | Measured seconds | Median (s) | Series/s | Python-relative speedup |
|---|---:|---|---:|---:|---:|
| Python UTide | 1 | 6.041851, 6.580931, 6.561769, 6.598419, 6.250325 | 6.561769 | 15.24 | 1.00x |
| RUTide | 1 | 0.098724, 0.088201, 0.087189, 0.093656, 0.096731 | 0.093656 | 1,067.74 | 70.06x |
| Python UTide | 16 | 0.504444, 0.461499, 0.461581, 0.459128, 0.457922 | 0.461499 | 216.69 | 1.00x |
| RUTide | 16 | 0.012109, 0.011881, 0.011061, 0.011095, 0.010294 | 0.011095 | 9,012.83 | 41.59x |
| Python UTide | 32 | 0.302468, 0.313729, 0.304065, 0.284398, 0.274476 | 0.302468 | 330.61 | 1.00x |
| RUTide | 32 | 0.009751, 0.009499, 0.008424, 0.008747, 0.008407 | 0.008747 | 11,432.44 | 34.58x |

The stable 100-series first-major-axis-CI checksum is `2.988075381125e0` in
Rust and `2.988075380934e0` in Python.

### Improvement over the direct Rust kernel

| Field | Workers | Direct Rust median (s) | Planned Rust median (s) | Internal speedup |
|---|---:|---:|---:|---:|
| Scalar | 1 | 0.943860 | 0.087249 | 10.82x |
| Scalar | 16 | 0.071359 | 0.010027 | 7.12x |
| Scalar | 32 | 0.049199 | 0.008066 | 6.10x |
| Vector | 1 | 0.990994 | 0.093656 | 10.58x |
| Vector | 16 | 0.073879 | 0.011095 | 6.66x |
| Vector | 32 | 0.048483 | 0.008747 | 5.54x |

## 1,000-series scaling results

The larger comparison used five Rust repetitions and three Python repetitions.
Python used process chunks of four series. Single-process Python was not repeated
at this size because the 100-series result already measures the same one-series
API without process scheduling; the 1,000-series run is intended to test parallel
scaling and amortization.

| Field | Implementation | Workers | Measured seconds | Median (s) | Series/s | Python-relative speedup |
|---|---|---:|---|---:|---:|---:|
| Scalar | RUTide | 1 | 0.797716, 0.815393, 0.835884, 0.845689, 0.799369 | 0.815393 | 1,226.40 | — |
| Scalar | Python UTide | 16 | 2.556607, 2.359162, 2.362616 | 2.362616 | 423.26 | 1.00x |
| Scalar | RUTide | 16 | 0.071282, 0.068025, 0.063031, 0.062528, 0.064764 | 0.064764 | 15,440.68 | 36.48x |
| Scalar | Python UTide | 32 | 1.288092, 1.378953, 1.409823 | 1.378953 | 725.19 | 1.00x |
| Scalar | RUTide | 32 | 0.050133, 0.045139, 0.043623, 0.042129, 0.041186 | 0.043623 | 22,923.73 | 31.61x |
| Vector | RUTide | 1 | 0.765920, 0.772026, 0.822656, 0.818341, 0.783530 | 0.783530 | 1,276.28 | — |
| Vector | Python UTide | 16 | 4.696752, 4.502779, 4.563187 | 4.563187 | 219.15 | 1.00x |
| Vector | RUTide | 16 | 0.074056, 0.069007, 0.077143, 0.074034, 0.070998 | 0.074034 | 13,507.30 | 61.64x |
| Vector | Python UTide | 32 | 2.532288, 2.369742, 2.580877 | 2.532288 | 394.90 | 1.00x |
| Vector | RUTide | 32 | 0.048967, 0.052358, 0.049684, 0.049763, 0.043351 | 0.049684 | 20,127.12 | 50.97x |

## Cold-plan context

Zero-warm-up, one-sample measurements intentionally include lazy plan creation
and other first-use runtime costs. They are context rather than headline results:

| Series | Field | Workers | Cold first solve (s) |
|---:|---|---:|---:|
| 100 | Scalar | 1 | 0.204056 |
| 100 | Scalar | 32 | 0.093048 |
| 100 | Vector | 1 | 0.215133 |
| 100 | Vector | 32 | 0.123361 |
| 1,000 | Scalar | 32 | 0.140547 |
| 1,000 | Vector | 32 | 0.188521 |

This distinction matters: the plan is deliberately an amortized optimization.
Small or lightly reused mask groups stay on the direct path, while production-size
shared-mask fields recover the setup cost within the same analysis.

## Correctness scope

The versioned oracle tests compare direct band power and end-to-end results. They
cover deterministic and pseudo-random jitter, isolated and clustered gaps,
frequency-band boundaries, coefficient variances, scalar amplitude/phase
intervals and SNR, and all four vector ellipse intervals plus SNR. The planned
kernel is checked directly against the Python-compatible direct kernel. The
real-FVCOM scalar fixture and deterministic vector fixture both use exact
Greenwich/nodal corrections.

Python's irregular complex periodogram calculates eastward and northward auto
spectra plus a cross-spectrum. Its current linearized vector confidence code only
uses the northward colored band and leaves the eastward coefficient pair white.
RUTide reproduces that output behavior while avoiding the two unused spectra. The
cross-spectrum remains required for the planned Monte Carlo covariance path and
will be implemented with that consumer.

## Decision

The Lomb–Scargle path now exceeds the project's 20x Python-performance target on
the primary 100-series workload at one, 16, and 32 workers for both scalar and
vector profiles. The 1,000-series process-pool comparison widens the advantage to
31.61–61.64x at 16 and 32 workers. Further spectral work, such as projecting many
residuals as one matrix operation, is optional rather than a viability blocker.

The next scientific coverage increment remains robust fitting. Sampling-quality
diagnostics—retained count, span, largest gap, and spectral-band coverage—should
also be exposed before Lomb–Scargle intervals are presented as sufficient
evidence that an arbitrary record is well sampled.
