# Public Python binding benchmark: 2026-09-02

The native `solve_many` endpoint is the performance product: it is
111.37–186.12x faster than a matched Python UTide loop for these 100-series
fits, while the one-series `rutide.solve` loop ranges from 0.93x to 1.68x. The
difference is not a regression in the batch kernel. It quantifies the value of
sharing astronomical preparation and repeated missing masks/Lomb plans, keeping
iteration and scheduling in Rust, and crossing the Python boundary once.

Native batch reconstruction is 124.68–262.74x faster than the Python UTide loop
and 12.87–18.06x faster than a loop over `rutide.reconstruct`. These timings
include construction of the public Python result objects and output NumPy
arrays, but exclude fixture generation and the preceding fit.

## Revisions, environment, and protocol

- benchmark implementation: `300051d`;
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`, verified
  clean and imported directly from `UTide/`;
- machine: 2 x AMD EPYC 7713, 128 physical cores, one thread per core, Xen VM,
  Linux 6.8.0-111-generic;
- Python 3.10.12, NumPy 2.2.6, SciPy 1.13.1, RUTide 0.2.0;
- NumPy and SciPy OpenBLAS pools limited to one thread;
- 100 distinct scalar or vector series, 745 samples, five fixed constituents,
  mean and trend, exact Greenwich phase and nodal corrections;
- regular OLS profiles have no confidence calculation; irregular profiles have
  deterministic timestamp jitter, four repeated missing-mask groups, and
  colored linear confidence through Lomb–Scargle spectra;
- RUTide batch uses 16 Rayon workers and a 512 MiB temporary-component budget;
- one unreported warmup followed by five retained repetitions; medians below;
- the source tree and oracle were clean, but CPU affinity was not fixed and the
  shared host was not isolated.

## Solve results

| Field/profile | UTide loop (s) | RUTide loop (s) | RUTide batch (s) | Rust loop / UTide | Batch / UTide | Batch / Rust loop |
|---|---:|---:|---:|---:|---:|---:|
| Scalar, regular OLS | 2.051354 | 1.812878 | 0.014217 | 1.13x | 144.29x | 127.52x |
| Scalar, irregular colored | 2.726596 | 2.928104 | 0.024483 | 0.93x | 111.37x | 119.60x |
| Vector, regular OLS | 2.014024 | 1.894562 | 0.013011 | 1.06x | 154.79x | 145.61x |
| Vector, irregular colored | 5.164698 | 3.071224 | 0.027749 | 1.68x | 186.12x | 110.68x |

The one-series Rust loop is intentionally included as an attribution control.
It repeatedly rebuilds each model and crosses the extension boundary once per
series, just as Python UTide does. On scalar irregular confidence that overhead
makes it 7% slower than the mature SciPy implementation. This does not affect
the intended station/FVCOM path, where `solve_many` amortizes that work and
processes 3,604–7,686 series/s in these profiles.

## Reconstruction results

| Field/profile | UTide loop (s) | RUTide loop (s) | RUTide batch (s) | Rust loop / UTide | Batch / UTide | Batch / Rust loop |
|---|---:|---:|---:|---:|---:|---:|
| Scalar, regular OLS | 1.925050 | 0.132287 | 0.007327 | 14.55x | 262.74x | 18.06x |
| Scalar, irregular colored | 1.539126 | 0.128831 | 0.007857 | 11.95x | 195.88x | 16.40x |
| Vector, regular OLS | 1.786867 | 0.151286 | 0.011315 | 11.81x | 157.93x | 13.37x |
| Vector, irregular colored | 1.527345 | 0.157635 | 0.012250 | 9.69x | 124.68x | 12.87x |

## Numerical checks

Every timed run performed coefficient and reconstruction checks before writing
its report. RUTide loop and batch values were bitwise identical for all four
retained profiles. Against Python UTide, the largest non-SNR discrepancies were:

- amplitude or ellipse-axis coefficient: `1.31e-13`;
- phase or inclination: `1.05e-10` degrees;
- amplitude/axis confidence interval: `2.36e-13`;
- angular confidence interval: `6.78e-11` degrees; and
- reconstructed elevation/current: `7.56e-12`.

The maximum absolute SNR differences were 23.31 scalar and 4.62 vector because
the deterministic low-noise fixture produces SNR values as large as 4.06e9 and
1.87e9. Their maximum relative errors were `1.01e-8` and `3.04e-9`, respectively.
The harness rejects coefficient or reconstruction drift outside its frozen
absolute/relative tolerances and records output digests to prove eager result
materialization.

## Retained repetitions

| Field/profile | UTide solve (s) | RUTide-loop solve (s) | RUTide-batch solve (s) |
|---|---|---|---|
| Scalar OLS | 2.103613, 2.051354, 2.060280, 2.049113, 2.015045 | 1.605641, 1.855271, 1.812878, 1.857898, 1.667748 | 0.014312, 0.014217, 0.014971, 0.012927, 0.012191 |
| Scalar Lomb | 2.722747, 2.721058, 2.726596, 2.741540, 2.846810 | 3.168825, 2.928104, 2.741725, 2.768500, 3.175628 | 0.032344, 0.020397, 0.025057, 0.024483, 0.021779 |
| Vector OLS | 2.061151, 2.006863, 1.994096, 2.014024, 2.024622 | 1.982037, 1.894562, 1.534980, 1.190893, 2.020052 | 0.011722, 0.013011, 0.011381, 0.014067, 0.014231 |
| Vector Lomb | 5.078704, 5.783439, 5.606014, 5.072524, 5.164698 | 3.283986, 2.929575, 3.071224, 3.210586, 2.848344 | 0.031850, 0.027749, 0.024444, 0.029929, 0.025127 |

The vector one-series OLS timings are visibly noisy (1.19–2.02 s), reinforcing
that these are engineering measurements on a shared machine, not isolated
publication results. The conclusion is nevertheless insensitive to that cell:
the slowest retained batch/UTide solve ratio is still greater than 84x, and the
five-sample median is 154.79x.

## Decision

Keep `solve` for convenient one-record analysis and compatibility, but direct
ADCP station collections and FVCOM arrays to `solve_many`. Optimizing the scalar
wrapper alone would at best target a roughly UTide-equivalent 1.8–3.1 second
loop, while shared batch execution completes the same 100 fits in 13–28 ms. The
next performance work should therefore preserve and extend batch reuse rather
than complicate the one-series path with a process-global cache.
