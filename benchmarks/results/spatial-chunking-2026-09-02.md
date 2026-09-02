# FVCOM spatial-chunk snapshot: 2026-09-02

Bounded spatial input removes the complete promoted current field from the
RUTide process without changing its scientific result. On all 144,860 FVCOM
elements, the automatic four-chunk path reduced median 64-worker peak RSS from
2,342,608 KiB (2.23 GiB) to 984,952 KiB (0.94 GiB), a 58.0% reduction. Median
whole-process wall time changed from 4.93 to 5.40 seconds, a 9.5% cost. Both
profiles produced the same canonical digest.

Relative to the previously retained 126.97-second practical 32-process Python
UTide baseline, the bounded default remains 23.5x faster. That Python number is
from the matched five-constituent vector snapshot; Python hashes results without
writing the complete 44 MB coefficient file, so the comparison does not favor
RUTide.

## Revisions, workload, and environment

- implementation parent: `eceaf6d`;
- source: `frs2f_0001.nc`, a 25,778,391,080-byte CDF-2 container;
- selected input: 745 timestamps, 144,860 elements, `latc`, `ua`, and `va`;
- profile: M2, S2, N2, K1, and O1; OLS; mean and trend; exact Greenwich phase
  and nodal corrections; no confidence interval or reconstruction;
- machine: 2 x AMD EPYC 7713, 128 physical cores, 251 GiB RAM, one exposed NUMA
  node, Xen VM, Linux;
- build: Rust 1.98.0 release profile, thin LTO, one codegen unit,
  `RUSTFLAGS="-C target-cpu=native"`, NetCDF-C 4.8.1, and mimalloc;
- execution: 64 Rayon workers, warm page cache, one discarded automatic-chunk
  warm-up, then three interleaved retained runs per profile;
- measurement: GNU `time` process wall and maximum RSS plus application stage
  timers. CPU affinity was not fixed and the host was not isolated.

The automatic planner targeted 512 MiB of promoted component storage. It chose
44,992 elements per chunk, four chunks, and a maximum logical observation buffer
of 536,304,640 bytes. The comparison passed `--chunk-series 200000`, which was
capped to all 144,860 elements and allocated 1,726,731,200 logical observation
bytes in one chunk.

## Retained vector measurements

| Profile | Repetition | Input (s) | Solve (s) | Result processing (s) | Output (s) | Internal total (s) | Process wall (s) | Peak RSS (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| automatic | 1 | 1.8156 | 2.7559 | 0.2944 | 0.1382 | 5.0206 | 5.13 | 973,884 |
| automatic | 2 | 1.9744 | 2.8131 | 0.3146 | 0.1531 | 5.2768 | 5.40 | 984,952 |
| automatic | 3 | 1.9791 | 3.0996 | 0.2939 | 0.1246 | 5.5146 | 5.62 | 986,592 |
| one chunk | 1 | 1.7958 | 2.6440 | 0.2201 | 0.1304 | 4.8074 | 4.93 | 2,310,640 |
| one chunk | 2 | 2.1390 | 2.6122 | 0.2156 | 0.1297 | 5.1168 | 5.23 | 2,342,956 |
| one chunk | 3 | 1.8897 | 2.5118 | 0.2351 | 0.1214 | 4.7739 | 4.89 | 2,342,608 |
| **automatic median** |  | **1.9744** | **2.8131** | **0.2944** | **0.1382** | **5.2768** | **5.40** | **984,952** |
| **one-chunk median** |  | **1.8897** | **2.6122** | **0.2201** | **0.1297** | **4.8074** | **4.93** | **2,342,608** |

The canonical digest was
`aa4dc61710c4344118a1f01de142232f0b88ac2a25343462f7c9414bdf4e2188`
for every retained run. Synthetic application coverage additionally compares
whole-field and one-series chunks through joint missing masks, robust fitting,
Monte Carlo confidence, reconstruction, NetCDF serialization, and inferred
currents. Global Monte Carlo stream offsets preserve bitwise results when a
chunk begins after series zero.

The complete 75,160-node scalar field fits below the automatic budget, so the
final planner correctly keeps it in one 447,953,600-byte observation chunk. A
retained run completed in 1.60 seconds at 863,880 KiB peak RSS. An explicit
32,768-node bound completed in 1.63 seconds at 548,576 KiB. Both produced digest
`fcd4377aa86c295bdbdc14c9f9b326595c9543fd7728cc319ef79ffcc48e8676`,
showing that users can trade memory for a small runtime cost even when automatic
chunking is unnecessary.

## Chunk-size sweep and decision

Single retained probes at 16,384, 32,768, and 65,536 elements peaked at 656,992,
828,352, and 1,249,700 KiB respectively. Their wall times were 10.06, 6.30, and
5.98 seconds. The automatically selected 44,992-element chunks performed better
than those probes at 5.13–5.62 seconds while staying below 1 GiB RSS, so the
portable 512 MiB observation budget is retained as the default.

The output layer deliberately retains compact coefficient and diagnostic arrays
and writes each NetCDF variable in bulk. This preserves output throughput and a
simple atomic temporary-file transaction. A streaming result sink would only be
valuable once those compact arrays, rather than time-by-space source fields,
dominate memory.
