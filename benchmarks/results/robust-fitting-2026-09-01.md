# Robust-fitting benchmark: 2026-09-01

RUTide reproduces pinned Python UTide's Cauchy IRLS results and is 19.00x faster
for scalar analysis and 19.46x faster for vector analysis on the canonical
one-worker, 100-series comparison. Across the complete 100-series worker matrix,
the advantage is 7.66–19.46x. On 1,000 series, RUTide is 13.39–17.66x faster at
16 and 32 workers.

Every compared scalar series performs five IRLS iterations and every vector
series performs two. The speed difference therefore does not come from a looser
tolerance or early exit in one implementation.

This is a focused solve benchmark for the robust scientific profile, not a
whole-NetCDF or whole-FVCOM timing. Input generation, pool creation, and shared
astronomy preparation are outside the measured regions; latitude-specific exact
model construction, robust fitting, colored confidence, and result construction
are inside them.

## Revisions, environment, and workload

- Cauchy robust core: `a3c9cbb`;
- FVCOM CLI and NetCDF integration: `10fa12a`;
- benchmark harness: `8b6552a`;
- prepared-OLS reuse optimization: `3a3744e`;
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`;
- machine: 2 x AMD EPYC 7713, 128 physical cores, Xen VM, Linux
  6.8.0-111-generic;
- Rust: 1.98.0, benchmark profile, thin LTO,
  `RUSTFLAGS="-C target-cpu=native"`;
- Python: CPython 3.10.12, NumPy 2.2.6, SciPy 1.13.1, one BLAS thread per
  process;
- samples per record: 745 exact hourly Modified Julian Days over 31 days;
- constituents: M2, S2, N2, K1, and O1;
- corrections: exact Greenwich phase and nodal/satellite terms;
- model: fitted mean and trend, Cauchy robust IRLS, and linear colored-noise
  confidence intervals on the regular FFT path;
- robust options: tuning constant 2.385, fractional tolerance 0.001, and maximum
  50 iterations;
- latitude: `60.95771789550781` plus `1e-5` degrees per series;
- scalar input: versioned real FVCOM node-zero elevation with offsets +5, -4,
  and +6 at sample indices 71, 218, and 503;
- vector input: deterministic eastward/northward signal with one eastward +5
  spike, one northward -4 spike, and a joint (+4, +3) spike at the same three
  indices.

The 100-series runs used one unreported warm-up and five measured repetitions.
The 1,000-series scaling runs used one warm-up and three repetitions. Python used
process chunks of one series for the 100-series matrix and four for the
1,000-series matrix. The host was not otherwise isolated and CPU affinity was not
fixed.

## Scientific parity and work equivalence

The 100-series scalar first-amplitude-CI checksum is `1.038511490236e0` in Rust
and `1.038511489782e0` in Python. The vector first-major-axis-CI checksum is
`1.902186175256e0` in Rust and `1.902186175223e0` in Python. These small aggregate
differences are consistent with the tighter versioned coefficient/CI oracle
tests, which pass for regular and irregular gappy scalar/vector records.

| Field | Series | Iterations/series | Iteration range | Rust/Python agreement |
|---|---:|---:|---:|---|
| Scalar | 100 and 1,000 | 5.000 | 5–5 | identical |
| Vector | 100 and 1,000 | 2.000 | 2–2 | identical |

The implementation also has focused tests for exact clean data returning OLS, a
31-sample sustained outlier block, invalid options and leverage, a non-exact
zero-MAD scale, and iteration-limit exhaustion. Vector residual magnitudes
produce one shared weight per eastward/northward pair, matching Python's complex
fit.

## Primary 100-series results

### Scalar

| Implementation | Workers | Measured seconds | Median (s) | Series/s | Python-relative speedup |
|---|---:|---|---:|---:|---:|
| Python UTide | 1 | 3.294738, 3.141929, 3.540762, 3.741912, 3.243412 | 3.294738 | 30.35 | 1.00x |
| RUTide | 1 | 0.219626, 0.173373, 0.170884, 0.170000, 0.178969 | 0.173373 | 576.79 | 19.00x |
| Python UTide | 16 | 0.275014, 0.262917, 0.271529, 0.256515, 0.265015 | 0.265015 | 377.34 | 1.00x |
| RUTide | 16 | 0.028533, 0.023200, 0.022269, 0.022075, 0.021716 | 0.022269 | 4,490.47 | 11.90x |
| Python UTide | 32 | 0.170928, 0.154322, 0.150547, 0.153988, 0.144941 | 0.153988 | 649.40 | 1.00x |
| RUTide | 32 | 0.031730, 0.020102, 0.023547, 0.018182, 0.016876 | 0.020102 | 4,974.62 | 7.66x |

### Vector

| Implementation | Workers | Measured seconds | Median (s) | Series/s | Python-relative speedup |
|---|---:|---|---:|---:|---:|
| Python UTide | 1 | 3.538942, 3.250935, 3.363788, 3.286089, 3.524158 | 3.363788 | 29.73 | 1.00x |
| RUTide | 1 | 0.167004, 0.175705, 0.172838, 0.172038, 0.175304 | 0.172838 | 578.58 | 19.46x |
| Python UTide | 16 | 0.268316, 0.244851, 0.241263, 0.248432, 0.257043 | 0.248432 | 402.53 | 1.00x |
| RUTide | 16 | 0.022390, 0.021349, 0.018342, 0.020255, 0.020062 | 0.020255 | 4,937.00 | 12.27x |
| Python UTide | 32 | 0.177873, 0.159948, 0.147980, 0.146463, 0.151091 | 0.151091 | 661.85 | 1.00x |
| RUTide | 32 | 0.018325, 0.020930, 0.016545, 0.018997, 0.016049 | 0.018325 | 5,457.00 | 8.25x |

The smaller relative gains at 16 and 32 workers are a workload-granularity
effect: 100 series provide only 3.1–6.25 fits per worker. Both implementations
finish in tens to hundreds of milliseconds, so process/task scheduling is a
material fraction of elapsed time. The sustained comparison below reduces that
effect.

## 1,000-series scaling results

| Field | Implementation | Workers | Measured seconds | Median (s) | Series/s | Python-relative speedup |
|---|---|---:|---|---:|---:|---:|
| Scalar | Python UTide | 16 | 2.442584, 2.495048, 2.502832 | 2.495048 | 400.79 | 1.00x |
| Scalar | RUTide | 16 | 0.173790, 0.177805, 0.164169 | 0.173790 | 5,754.08 | 14.36x |
| Scalar | Python UTide | 32 | 1.492873, 1.329128, 1.381715 | 1.381715 | 723.74 | 1.00x |
| Scalar | RUTide | 32 | 0.103175, 0.108746, 0.101229 | 0.103175 | 9,692.24 | 13.39x |
| Vector | Python UTide | 16 | 2.370961, 2.358155, 2.427690 | 2.370961 | 421.77 | 1.00x |
| Vector | RUTide | 16 | 0.133495, 0.134240, 0.138306 | 0.134240 | 7,449.37 | 17.66x |
| Vector | Python UTide | 32 | 1.314490, 1.320447, 1.281718 | 1.314490 | 760.75 | 1.00x |
| Vector | RUTide | 32 | 0.080078, 0.079348, 0.108795 | 0.080078 | 12,487.88 | 16.42x |

## Optimization audit and interpretation

The prepared model's existing unweighted QR is now reused for the first robust
iteration instead of factorizing the same design again. A trial replacement of
the subsequent weighted QR solves with normal equations passed this fixture but
did not improve sustained throughput and would square the design condition
number; it was therefore rejected. The retained implementation keeps
column-pivoted QR for weighted iterations.

Robust analysis is inherently a different workload from the earlier OLS and
Lomb–Scargle profiles. It performs repeated weighted factorizations—five
iterations for this scalar fixture—and then computes confidence from the weighted
residuals and weighted design. The measured 13–19x advantage is consequently the
appropriate claim for this profile, not the 20–70x irregular-spectrum result.

The result remains a strong rewrite signal: Rust keeps double-digit sustained
speedups while exposing richer convergence diagnostics than the public Python
coefficient surface. Further gains should be sought through batched small-matrix
work or reducing per-latitude preparation, and must retain the current oracle and
conditioning guarantees.

## 2026-09-02 robust-weight extension check

The implementation based on parent `96dceeb` adds Andrews, bisquare, Fair,
Huber, logistic, OLS, Talwar, and Welsch without changing the Cauchy default.
Bisquare, Cauchy, Fair, OLS, Talwar, and Welsch match pinned Python scalar and
vector solutions, weight sums, and iteration counts. Welsch additionally passes
the exact inferred-vector oracle and the application integration path with
Monte Carlo confidence and reconstruction. Pinned Python UTide raises an
ambiguous-array truth-value exception for Andrews, Huber, and logistic; their
standard scalar formulas are therefore covered as explicit Rust extensions.

The original one-worker, 100-series Cauchy probe was repeated after adding the
enum dispatch. Thin LTO and `-C target-cpu=native` were retained, as were the
fixture, five repetitions, one warm-up, checksums, and iteration counts.

| Field | Historical median (s) | Extension median (s) | Extension series/s | Checksum | Iterations/series |
|---|---:|---:|---:|---:|---:|
| Scalar | 0.173373 | 0.138518 | 721.93 | 1.038511490236e0 | 5 |
| Vector | 0.172838 | 0.120023 | 833.17 | 1.902186175256e0 | 2 |

The extension does not regress the default profile. These unisolated runs are a
regression check rather than evidence that the enum caused the apparent
historical improvement; compiler, host, and background-state differences can
easily move sub-second measurements. The unchanged checksums and iteration
counts are the relevant work-equivalence gate.
