# Robust coupled-vector inference benchmark: 2026-09-03

RUTide is 84.12x faster than pinned Python UTide on one worker and 58.15x faster
on 16 workers for the retained irregular, exact-inference, robust coupled-vector
profile. Robust IRLS takes approximately 2.04x as long as the corresponding
RUTide OLS path; it does not introduce a separate performance cliff.

## Workload and controls

- machine: 2 x AMD EPYC 7713, 128 physical cores, Xen VM, Linux
  6.8.0-111-generic;
- Rust: 1.98.0, benchmark profile, thin LTO, `-Ctarget-cpu=native`;
- Python: CPython 3.10.12, NumPy 2.2.6, SciPy 1.13.1, one BLAS thread per
  process;
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`;
- 100 distinct latitude-tagged repetitions of a 745-row vector record;
- deterministic irregular timestamps and four jointly missing component rows;
- exact positive/negative rotary inference for S2/M2 and O1/K1;
- Cauchy coupled-complex IRLS with isolated +5 m/s eastward and -4 m/s
  northward outliers;
- exact Greenwich phase and nodal corrections with Lomb–Scargle colored linear
  confidence;
- one unreported warm-up and five measured repetitions;
- one-worker runs pinned to CPU 0; 16-worker runs pinned to the same frozen CPU
  set used by the earlier inference benchmark.

Preparation and pool construction remain outside the timed solve region. Rust
keeps inner `faer` parallelism sequential; only the explicit batch worker pool
is parallel. Python calls the pinned one-series public API through a fork pool.

## Results

| Method | Workers | Python median (s) | Rust median (s) | Rust series/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Robust | 1 | 12.590454 | 0.149673 | 668.12 | 84.12x |
| Robust | 16 | 0.936701 | 0.016107 | 6,208.44 | 58.15x |
| OLS | 1 | — | 0.073158 | 1,366.90 | — |
| OLS | 16 | — | 0.007935 | 12,602.60 | — |

The robust semi-major confidence checksum is `8.007988014488` in Python and
`8.007988013114` in Rust, an absolute difference of about `1.37e-9`. Every
repetition within each implementation produced the same checksum.

These measurements are solve-only and profile-specific. They establish that the
implemented robust inference composition remains fast; they are not a whole-file
or whole-FVCOM timing claim.
