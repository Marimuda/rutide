# Depth-resolved FVCOM currents — 2026-09-02

This snapshot validates and measures native FVCOM sigma-layer current analysis.
It uses the 25,778,391,080-byte `frs2f_0001.nc` fixture with 745 hourly records,
10 sigma layers, and 144,860 elements on the repository's dual AMD EPYC 7713
host. Rust was built in release mode with `-C target-cpu=native`; runs used 64
workers, five explicit constituents (`M2,S2,N2,K1,O1`), Greenwich phases, exact
nodal corrections, a mean and trend, ordinary least squares, and no confidence
intervals or reconstruction. Timings are single unisolated observations and are
therefore performance snapshots rather than distributional claims.

## Correctness

The schema-v12 path was compared with the pinned Python UTide oracle for 96
series: three requested layers (`0,5,9`) crossed with the 32 frozen sparse
element anchors. The Rust selection was layer-major, used the source
`u(time,siglay,nele)` and `v(time,siglay,nele)` variables, and passed every
existing vector tolerance. The largest component-coefficient error was
`1.0797e-12` against a `5e-12` tolerance; the largest semi-major and semi-minor
errors were `1.5354e-13` and `6.3449e-14`. Its canonical result digest was
`80566f6facd6f186729594d363ff2765beadda5bed2c8d517f720becf0650c18`.

The Rust integration fixture separately covers out-of-order layers and elements,
a chunk boundary crossing between layers, joint missing-component masks, robust
fitting, Monte Carlo confidence, and reconstruction. Whole and chunked results
are bit-identical.

## Full spatial results

| Selected field | Series | Chunks | Report total | Input | Solve | Result processing | Output | Peak RSS | Output size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| layer 0 | 144,860 | 4 | 7.67 s | 4.92 s | 2.36 s | 0.25 s | 0.12 s | 1,191,940 KiB | 76,515,400 B |
| all 10 layers | 1,448,600 | 33 | 75.28 s | 49.37 s | 21.93 s | 2.82 s | 1.13 s | 2,460,120 KiB | 744,030,280 B |

External `/usr/bin/time -v` wall times were 7.77 s and 75.75 s. Ten times as
many fits cost 9.75x wall time but only 2.06x peak RSS. End-to-end throughput was
about 18,600 series/s for one layer and 19,100 series/s for all layers; solver-only
throughput was about 61,500 and 66,000 series/s. The automatic planner used
44,992 series per chunk and bounded the logical promoted `u`/`v` observation
buffer to 536,304,640 bytes in both runs.

The all-layer digest was
`94940a8ab5b77529775dad46259dff24983bdce49b7b4ba71b1df49f958fb505`.
Classic-NetCDF record-layout input accounted for 65.6% of reported all-layer
time, while solving accounted for 29.1%. Contiguous layer-element chunks retain
the single-hyperslab path. Sparse selections instead traverse records in storage
order and coalesce nearby elements, avoiding repeated whole-file seeks while
preserving the exact requested output order.

Base all-layer OLS is practical without an incremental result sink on this host.
Such a sink remains a targeted future optimization for result-rich combinations
such as Monte Carlo confidence, robust ragged diagnostics, or complete
three-dimensional reconstruction, not a prerequisite for native-layer analysis.
