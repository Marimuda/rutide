# Inferred-constituent benchmark: 2026-09-01

RUTide's exact inferred-constituent path is 48.97–228.45x faster than pinned
Python UTide across the controlled 100-series scalar/vector, FFT/Lomb, and
1/16/32-worker matrix. On the less granular 1,000-series workload, the measured
advantage is 71.93–140.44x. Python-compatible approximate inference is
69.30–86.55x faster at 16 workers.

These are focused solve-only measurements, not whole-NetCDF or whole-FVCOM
claims. They include latitude-specific model construction, solving, diagnostics,
and colored linear confidence intervals. Shared astronomy preparation, input
generation, pool construction, and result printing are outside the timed region.

## Revisions, environment, and workload

- scalar and coupled-vector inference kernels: `ab5bdb1` and `892e720`;
- varying-latitude/missing-value inference batches: `40410ae`;
- CLI and versioned NetCDF integration: `ad11291`;
- matched Rust/Python benchmark harness: `e29e784`;
- Python UTide oracle: `8fabe121752bc317931472a10a42e306715106de`;
- machine: 2 x AMD EPYC 7713, 128 physical cores, Xen VM, Linux
  6.8.0-111-generic;
- Rust: 1.98.0, thin-LTO benchmark profile,
  `RUSTFLAGS="-Ctarget-cpu=native"`;
- Python: CPython 3.10.12, NumPy 2.2.6, SciPy 1.13.1, one BLAS thread per
  process;
- samples per source record: 745 over approximately 31 days;
- requested constituents: N2, M2, S2, K1, and O1;
- reported order: N2, M2, K1, S2, and O1;
- scalar relationships: S2/M2 amplitude 0.35 and phase offset 20 degrees;
  O1/K1 amplitude 0.50 and phase offset 45 degrees;
- vector relationships: the scalar positive-rotary ratios plus negative-rotary
  S2/M2 amplitude 0.25 and phase offset -10 degrees, and O1/K1 amplitude 0.40
  and phase offset 30 degrees;
- model: OLS, mean and trend, exact Greenwich phase and nodal/satellite terms,
  and colored linear confidence intervals;
- latitude: `60.95771789550781` plus `1e-5` degrees per series.

The regular workload uses exact hourly timestamps and complete records, so
colored confidence takes the FFT route. The irregular workload applies the same
deterministic sinusoidal timestamp jitter as the dedicated Lomb benchmark. Its
scalar record retains 742 observations; the vector current uses a joint mask
with 741 retained component pairs. Repeated masks exercise the bounded shared
Lomb–Scargle plan used by the application batch path.

The 100-series matrix used one unreported warm-up and five measured repetitions;
Python process chunks contain one series. The 1,000-series matrix used one
warm-up, three measured repetitions, and Python chunks of four.

## Host-load control and excluded measurements

During the initial unpinned matrix, host load rose above 70 and `mpstat` showed
an external nice-priority workload on many CPUs plus virtual-machine steal time.
Two early one-process Python sessions also overlapped. Those measurements are
excluded completely.

For the retained matrix, Rust and Python were run sequentially and pinned with
`taskset` to the same observed-idle CPU sets. One-worker runs used CPU 0; the
16/32-worker runs used prefixes of this frozen set:

```text
0,1,3,5,8,11,13,17,22,24,27,28,29,30,32,37,
40,41,42,43,44,45,47,50,54,55,56,58,60,64,67,68
```

The VM was still not isolated and retained CPUs showed steal time, so absolute
times are machine-state-specific. Identical affinity and sequential pairing make
the relative results substantially more defensible, but they should still be
repeated on a quiet host before publication outside this feasibility project.

## Exact inference: 100 series

| Sampling | Field | Workers | Python median (s) | Rust median (s) | Speedup |
|---|---|---:|---:|---:|---:|
| Regular FFT | Scalar | 1 | 14.509127 | 0.063510 | 228.45x |
| Regular FFT | Scalar | 16 | 1.117560 | 0.011203 | 99.76x |
| Regular FFT | Scalar | 32 | 0.714667 | 0.007234 | 98.79x |
| Regular FFT | Vector | 1 | 13.974152 | 0.064758 | 215.79x |
| Regular FFT | Vector | 16 | 1.164276 | 0.012827 | 90.77x |
| Regular FFT | Vector | 32 | 0.731773 | 0.012133 | 60.31x |
| Irregular Lomb | Scalar | 1 | 14.626889 | 0.091957 | 159.06x |
| Irregular Lomb | Scalar | 16 | 1.165147 | 0.013300 | 87.60x |
| Irregular Lomb | Scalar | 32 | 0.761107 | 0.015544 | 48.97x |
| Irregular Lomb | Vector | 1 | 17.664233 | 0.104981 | 168.26x |
| Irregular Lomb | Vector | 16 | 1.457022 | 0.019152 | 76.08x |
| Irregular Lomb | Vector | 32 | 0.837884 | 0.014566 | 57.52x |

The 32-worker entries contain only 3.125 fits per worker and Rust completes in
7–16 ms, so task scheduling is a substantial fraction of the Rust time. The
sustained matrix below is the better parallel-throughput comparison.

## Exact inference: 1,000 series

| Sampling | Field | Workers | Python median (s) | Rust median (s) | Speedup |
|---|---|---:|---:|---:|---:|
| Regular FFT | Scalar | 16 | 9.387898 | 0.066844 | 140.44x |
| Regular FFT | Scalar | 32 | 5.429463 | 0.057525 | 94.38x |
| Regular FFT | Vector | 16 | 10.633818 | 0.097214 | 109.39x |
| Regular FFT | Vector | 32 | 6.345211 | 0.065746 | 96.51x |
| Irregular Lomb | Scalar | 16 | 9.937842 | 0.138157 | 71.93x |
| Irregular Lomb | Scalar | 32 | 5.904617 | 0.064733 | 91.22x |
| Irregular Lomb | Vector | 16 | 12.193244 | 0.122276 | 99.72x |

The interrupted 32-worker irregular-vector pair was not needed to establish the
sustained range and was not substituted with an earlier unpinned result.

## Approximate-mode sensitivity

Approximate inference was measured at 100 series and 16 workers to verify that
Python compatibility mode does not introduce a separate performance cliff.

| Sampling | Field | Python median (s) | Rust median (s) | Speedup |
|---|---|---:|---:|---:|
| Regular FFT | Scalar | 0.818116 | 0.009453 | 86.55x |
| Regular FFT | Vector | 0.862795 | 0.011330 | 76.15x |
| Irregular Lomb | Scalar | 0.881145 | 0.012715 | 69.30x |
| Irregular Lomb | Vector | 1.147275 | 0.013983 | 82.05x |

## Work and correctness equivalence

Both probes solve the same observations and latitude sequence with the same
relationships, inference mode, corrections, confidence model, validity mask,
and result ordering. Python invokes its public one-series API in each worker;
Rust invokes the production varying-latitude batch API. Rust keeps `faer`'s
internal parallelism sequential so only the explicitly recorded outer worker
pool supplies parallelism.

The probes sum all five scalar amplitude-CI values or all five vector
semi-major-CI values per series. Representative 100-series exact checksums are:

| Sampling | Field | Python | Rust | Absolute difference |
|---|---|---:|---:|---:|
| Regular FFT | Scalar | 10.203040828330 | 10.203040832780 | 4.45e-9 |
| Regular FFT | Vector | 7.760759993851 | 7.760759993919 | 6.8e-11 |
| Irregular Lomb | Scalar | 10.039191574672 | 10.039191575120 | 4.48e-10 |
| Irregular Lomb | Vector | 6.182629517640 | 6.182629512787 | 4.85e-9 |

The differences scale linearly when the identical fixture is repeated 1,000
times and are consistent with the stricter per-coefficient pinned-oracle tests.
Those tests cover resolved/unresolved exact and approximate fits, positive and
negative rotary vector constraints, white/colored confidence, PE/SNR,
reconstruction, invalid graphs, and gappy joint masks.

## Decision

Inference does not recreate the earlier 3–6x performance concern. Exact and
approximate modes both retain large advantages through scalar and coupled-vector
fits, complete and missing records, FFT and Lomb colored spectra, and realistic
parallel batches. No inference-specific optimization is needed before continuing
functional coverage.

The remaining inference gap is robust coupled-vector fitting. The CLI currently
rejects that combination explicitly; implementing it with pinned coefficient,
weight, interval, and iteration-count oracles is the next increment.
