# Monte Carlo confidence benchmark: 2026-09-01

RUTide's deterministic 200-realization Monte Carlo confidence is 17.24–61.09x
faster than pinned Python UTide on the canonical one-worker, 100-series
comparison. At 16 workers the advantage is 10.80–35.08x. These timings include
complete 2×2 scalar or 4×4 vector covariance sampling and UTide-compatible
median-absolute-deviation confidence intervals; they are not linear-CI timings.

## Revisions, environment, and workload

- Monte Carlo core and robust integration: `9fdad81`, `4651a2b`;
- FVCOM batch, CLI, NetCDF, and reproducibility integration: `2a95950`;
- benchmark harness: `cd8397c`;
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`;
- machine: 2 x AMD EPYC 7713, 128 physical cores, Xen VM, Linux
  6.8.0-111-generic;
- Rust: 1.98.0, benchmark profile, thin LTO,
  `RUSTFLAGS="-C target-cpu=native"`;
- Python: CPython 3.10.12, NumPy 2.2.6, SciPy 1.13.1, one BLAS thread per
  process;
- 100 series, five constituents (M2, S2, N2, K1, O1), one unreported warm-up,
  and five measured repetitions;
- 200 realizations. Python ignores configurable `MC_n` and always uses 200;
  RUTide uses its effective default of 200 with root seed zero;
- exact Greenwich phase and nodal corrections, fitted mean and trend, and
  colored residual noise.

Input generation, process/thread-pool construction, and shared astronomy
preparation are outside the measured region. The host was not otherwise
isolated and CPU affinity was not fixed. Python's random confidence checksums
vary across repetitions; RUTide's seeded checksums are stable by design.

## Results

| Profile | Field | Workers | Python median (s) | Rust median (s) | Rust series/s | Speedup |
|---|---|---:|---:|---:|---:|---:|
| Irregular/gappy OLS | Scalar | 1 | 4.100097 | 0.119589 | 836.19 | 34.28x |
| Irregular/gappy OLS | Vector | 1 | 7.566957 | 0.123862 | 807.35 | 61.09x |
| Irregular/gappy OLS | Scalar | 16 | 0.307877 | 0.013396 | 7,464.92 | 22.98x |
| Irregular/gappy OLS | Vector | 16 | 0.548879 | 0.015647 | 6,391.05 | 35.08x |
| Regular robust | Scalar | 1 | 3.760928 | 0.218091 | 458.52 | 17.24x |
| Regular robust | Vector | 1 | 3.820148 | 0.185055 | 540.38 | 20.64x |
| Regular robust | Scalar | 16 | 0.286853 | 0.026558 | 3,765.37 | 10.80x |
| Regular robust | Vector | 16 | 0.290423 | 0.022585 | 4,427.67 | 12.86x |

The irregular profile has 745 nominally hourly samples with deterministic time
jitter and missing values. Its colored scalar path uses Lomb–Scargle band power;
the vector path additionally calculates the real eastward/northward
cross-spectrum needed for the full 4×4 covariance. The regular robust profile
uses the established outlier fixtures and FFT colored spectra. Every scalar
series performs five Cauchy IRLS iterations in both implementations, and every
vector series performs two.

### Measured samples

| Profile | Field | Workers | Python seconds | Rust seconds |
|---|---|---:|---|---|
| Irregular OLS | Scalar | 1 | 4.100097, 3.535192, 4.211146, 4.124597, 3.795415 | 0.118865, 0.124296, 0.123899, 0.109382, 0.119589 |
| Irregular OLS | Vector | 1 | 6.948339, 7.407146, 7.566957, 7.581146, 7.582168 | 0.145667, 0.143479, 0.122633, 0.118264, 0.123862 |
| Irregular OLS | Scalar | 16 | 0.325993, 0.299679, 0.309194, 0.296976, 0.307877 | 0.012928, 0.013461, 0.012888, 0.014100, 0.013396 |
| Irregular OLS | Vector | 16 | 0.545787, 0.548879, 0.538133, 0.569172, 0.551922 | 0.017774, 0.015647, 0.015817, 0.015098, 0.015331 |
| Regular robust | Scalar | 1 | 3.816736, 3.855034, 3.225562, 3.287319, 3.760928 | 0.226762, 0.223064, 0.218091, 0.200974, 0.196571 |
| Regular robust | Vector | 1 | 3.743411, 3.851968, 3.837926, 3.820148, 3.758630 | 0.185055, 0.184277, 0.186965, 0.193506, 0.181269 |
| Regular robust | Scalar | 16 | 0.290480, 0.292492, 0.286853, 0.286486, 0.278327 | 0.029683, 0.027407, 0.026558, 0.023967, 0.022979 |
| Regular robust | Vector | 16 | 0.290423, 0.289648, 0.293151, 0.278449, 0.294042 | 0.024731, 0.021282, 0.022915, 0.022585, 0.021374 |

## Interpretation

The nonlinear sampling and ellipse conversion do not erase the rewrite's
performance advantage. Irregular vector Monte Carlo is the richest measured
profile—missing observations, Lomb–Scargle auto- and cross-spectra, complete
4×4 covariance repair/sampling, and nonlinear ellipse intervals—and remains
61.09x faster on one worker and 35.08x faster on 16.

The smaller robust speedups are expected because repeated weighted QR
factorizations dominate that profile; the Monte Carlo work is only the final
confidence stage. The 16-worker, 100-series cases also expose scheduling
overhead because each worker receives only about six series. Even there RUTide
retains a 10.80–35.08x advantage.

This closes the performance acceptance task for non-inferred Monte Carlo
confidence. Monte Carlo propagation through inferred-constituent relationships
remains an explicitly rejected, scientifically feasible extension requiring a
separate correlated-reference sampling contract and oracle-independent tests.
