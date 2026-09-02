# Fixed-physical-depth FVCOM currents — 2026-09-02

This snapshot validates and measures schema-v13 interpolation of FVCOM native
sigma-layer currents to physical depths below the instantaneous free surface. It
uses the 25,778,391,080-byte `frs2f_0001.nc` fixture with 745 hourly records, 10
native layers, 144,860 elements, and 75,160 nodes on the repository's dual AMD
EPYC 7713 host. Rust was built in release mode with thin LTO, one codegen unit,
`-C target-cpu=native`, NetCDF-C 4.8.1, and mimalloc. Runs used 64 workers and
five explicit constituents (`M2,S2,N2,K1,O1`), a mean and trend, Greenwich
phases, exact nodal corrections, ordinary least squares, and no confidence
intervals or reconstruction.

These are single unisolated, warm-cache observations. They characterize the
implementation and memory/runtime tradeoff on this machine; differences of a
few percent should not be treated as a distributional claim.

The implementation parent was `d465b36`; this document and the fixed-depth
changes are committed together. The host exposed 128 physical cores in one NUMA
node and 251 GiB RAM. CPU affinity was not fixed.

The measured throughput command was:

```console
/usr/bin/time -v target/release/rutide analyze-vector \
  --input /path/to/frs2f_0001.nc \
  --output /tmp/rutide-fixed-depth-10m.nc \
  --report /tmp/rutide-fixed-depth-10m.json \
  --depths 10 --constituents M2,S2,N2,K1,O1 \
  --workers 64 --chunk-series 16384 --overwrite
```

Omitting `--chunk-series 16384` produced the automatic-memory row below. The
real-file correctness output used `--element-count 32 --depths 100,500` and was
checked with `python -m rutide_baseline.compare_vector` from the locked
`benchmarks/python` environment.

## Scientific correctness

The real-file check crossed the first 32 elements with requested depths of 100 m
and 500 m, producing 64 valid series. The comparator independently read
`siglay`, `h`, `zeta`, `nv`, `wet_cells`, `u`, and `v`, reproduced the frozen
triangle-centroid and moving-free-surface interpolation in Python, and then
called pinned Python UTide revision
`8fabe121752bc317931472a10a42e306715106de` for every interpolated series.

| Quantity | Maximum absolute error | Tolerance |
|---|---:|---:|
| reconstructed east/north harmonic coefficient | 7.5953e-14 | 5e-12 |
| semi-major axis | 3.0129e-14 | 5e-12 |
| signed semi-minor axis | 2.6722e-14 | 5e-12 |
| inclination (raw circular error, degrees) | 1.8767e-10 | coefficient backstop |
| Greenwich phase (raw circular error, degrees) | 4.1615e-10 | coefficient backstop |
| percent energy | 3.1719e-11 | 1e-9 |
| eastward/northward mean | 2.0678e-15 | 5e-12 |
| eastward/northward slope per day | 2.7213e-16 | 5e-12 |
| frequency (cycles/hour) | 0 | 1e-15 |
| reference time (MJD) | 0 | 1e-12 |

Every tolerance passed. The Rust result digest was
`86ef545772509958fe3c3761afe873c8cd2150b3d11418cdd9944f4c2a153be0`.
The synthetic integration fixture separately covers time-varying surface
elevation, horizontally varying sigma coordinates, exact layer-centre matches,
missing bracket currents and geometry, dry cells, unavailable output rows,
out-of-order ragged robust diagnostics, reconstruction, and chunk-invariant
digests.

## Full-field 10 m result

At 10 m, 131,557 of the rectangular 144,860 element coordinates had enough
physical samples for the five-constituent model. The other 13,303 coordinates
(9.18%) were retained with `analysis_status=unavailable`; 13,993 coordinates
had at least one jointly missing observation. Both chunk profiles produced the
same canonical result digest,
`f5c5c865f2c781722dbe78212e6084a154b17e13772a1022391a3af010d73cb0`.

| Chunk profile | Elements/chunk | Chunks | Input (s) | Solve (s) | Result (s) | Output (s) | Internal total (s) | Process wall (s) | Peak RSS (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| automatic 512 MiB budget | 3,456 | 42 | 68.4366 | 4.8091 | 0.4300 | 0.2215 | 73.9446 | 74.00 | 835,684 |
| machine-tuned `--chunk-series 16384` | 16,384 | 9 | 59.0790 | 3.1068 | 0.3098 | 0.1976 | 62.7356 | 62.84 | 4,062,200 |

The selected semantic input payload was 9,294,946,360 bytes (8.66 GiB), and the
NetCDF coefficient file was 77,675,646 bytes. The automatic profile bounded its
promoted chunk arrays to 535,541,760 bytes and peaked at 0.80 GiB RSS. Using the
ample host memory for larger element blocks reduced wall time by 15.1%, at a
4.86x peak-RSS cost. This makes the automatic profile the portable default and
the explicit 16,384-element profile the better throughput choice on this
251 GiB machine.

Compacting fitted rows before each batch solve was important because the fixed
depth product contains unavailable shallow cells. On the same 16,384-element
profile, that optimization reduced solve time from 10.7089 s to 3.1068 s
(3.45x) and internal total time from 70.0247 s to 62.7356 s (10.4%). The final
solver-only throughput is about 42,345 fitted series/s. Input and interpolation
now consume 94.2% of the optimized internal total, so future acceleration belongs
in classic-NetCDF access and interpolation layout rather than the harmonic
kernel.

## Python baseline interpretation

Python UTide does not provide FVCOM fixed-physical-depth preprocessing, so there
is no honest like-for-like Python application timer to divide into the 62.84 s
Rust result. Correctness is instead established by independent Python
interpolation followed by the pinned solver, as described above.

For scale only, the retained depth-averaged five-constituent Python run processes
the same 745-point vector fits at about 1,250 elements/s. Applying that already
measured solver rate to the 131,557 fitted 10 m series implies roughly 105 s for
Python UTide fitting alone, before any fixed-depth interpolation or full NetCDF
output. RUTide's matched harmonic stage is 33.9x faster by this throughput
comparison. This is deliberately not labeled an end-to-end fixed-depth speedup;
the 62.84 s Rust application is currently limited by work that Python UTide does
not implement.
